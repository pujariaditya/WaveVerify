#!/usr/bin/env python3
"""Audio Watermarking Training Script.

This module implements the training pipeline for an audio watermarking system that combines
generator, detector, locator, and discriminator models. It includes comprehensive training
loops, validation procedures, and checkpoint management.

Key Features:
    - Multi-model training with generator and discriminator
    - Watermark detection and localization
    - Comprehensive loss functions (STFT, MEL, waveform, adversarial)
    - Distributed training support via accelerator
    - WandB integration for experiment tracking
    - Automatic checkpointing with best model selection
"""

# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
import gc
import logging
import os
import sys
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, Callable, TypeAlias

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
import argbind
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from audiotools import ml
from audiotools.core import util
from audiotools.data import transforms
from audiotools.data.datasets import AudioDataset, AudioLoader, ConcatDataset
from audiotools.ml.decorators import Tracker, timer, when
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# =============================================================================
# LOCAL IMPORTS
# =============================================================================
# Add project root to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from model import AudioWatermarking, Detector, Discriminator, Generator, Locator
import evaluate
import loss

# =============================================================================
# TYPE DEFINITIONS
# =============================================================================
# Type aliases for improved code clarity and type safety
DeviceType: TypeAlias = Union[str, torch.device]
TensorDict: TypeAlias = Dict[str, torch.Tensor]
MetricsDict: TypeAlias = Dict[str, float]
ConfigDict: TypeAlias = Dict[str, Any]
PathType: TypeAlias = Union[str, Path]
LossWeights: TypeAlias = Dict[str, float]
TransformFn: TypeAlias = Callable[[Any], bool]
BatchData: TypeAlias = Dict[str, Union[torch.Tensor, Dict[str, Any]]]
StateDict: TypeAlias = Dict[str, Any]
OptimizerType: TypeAlias = Union[optim.AdamW, optim.Optimizer]
SchedulerType: TypeAlias = optim.lr_scheduler.ExponentialLR

# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================
class AudioWatermarkingError(Exception):
    """Base exception for audio watermarking operations."""
    pass


class ModelInitializationError(AudioWatermarkingError):
    """Raised when model initialization fails."""
    pass


class CheckpointError(AudioWatermarkingError):
    """Raised when checkpoint operations fail."""
    pass


class DatasetError(AudioWatermarkingError):
    """Raised when dataset loading or processing fails."""
    pass


class TrainingError(AudioWatermarkingError):
    """Raised when training loop encounters an error."""
    pass


class ValidationError(AudioWatermarkingError):
    """Raised when validation loop encounters an error."""
    pass


# =============================================================================
# CONFIGURATION
# =============================================================================
# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set up structured logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('train.log')
    ]
)

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
# Enable cudnn autotuner to speed up training
# This can be altered by the seed function for reproducibility
DEFAULT_CUDNN_BENCHMARK: int = 1
torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", DEFAULT_CUDNN_BENCHMARK)))

# Configuration constants
DEFAULT_MESSAGE_THRESHOLD: float = 0.5
DEFAULT_BATCH_SIZE: int = 12
DEFAULT_VAL_BATCH_SIZE: int = 10
DEFAULT_NUM_WORKERS: int = 8
DEFAULT_NUM_ITERS: int = 250000
MAX_GRADIENT_NORM: float = 10.0
DEFAULT_MESSAGE_LENGTH: int = 16
DEFAULT_SAMPLE_FREQ: int = 10000
DEFAULT_VALID_FREQ: int = 1000
DEFAULT_SAVE_PATH: str = "ckpt"
DEFAULT_SEED: int = 0

# Message generation constants
MESSAGE_MIN_VALUE: int = 0
MESSAGE_MAX_VALUE: int = 2

# Training checkpoint iterations
DEFAULT_SAVE_ITERS: List[int] = [10000, 50000, 100000, 200000]

# Default loss weight coefficients
DEFAULT_LOSS_WEIGHTS: Dict[str, float] = {
    "waveform/loss": 1000.0,
    "mel/loss": 20.0,
    "stft/loss": 10.0,
    "adv/gen_loss": 40.0,
    "loc/loss": 100.0,
    "dec/loss": 10000.0,
}

# System configuration constants
THREAD_RATIO: float = 0.5  # Use half of available CPU cores
WORKER_TRACEBACK_LIMIT: int = 0  # Limit traceback on non-main processes
BYTES_TO_MB: int = 1024 ** 2  # Bytes to megabytes conversion factor

# Model key prefixes for state dict compatibility
DISCRIMINATOR_KEY_PREFIX: str = "discriminators.0."

# Audio sample phase names
AUDIO_SAMPLE_PHASE: str = "audio_sample"
AUDIO_SAMPLE_TYPES: List[str] = ["original_signal", "recons", "watermarked_signal"]

# =============================================================================
# OPTIMIZER AND ACCELERATOR BINDINGS
# =============================================================================
# Bind optimizers with argbind for configuration management
AdamW = argbind.bind(torch.optim.AdamW, "generator", "detector", "locator", "discriminator")
Accelerator = argbind.bind(ml.Accelerator, without_prefix=True)


@argbind.bind("generator", "discriminator")
def ExponentialLR(
    optimizer: optim.Optimizer, 
    gamma: float = 1.0
) -> optim.lr_scheduler.ExponentialLR:
    """Create an exponential learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        gamma: Multiplicative factor of learning rate decay
        
    Returns:
        ExponentialLR scheduler instance
    """
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)


# =============================================================================
# MODEL BINDINGS
# =============================================================================
# Bind model classes for configuration
Generator = argbind.bind(Generator)
Detector = argbind.bind(Detector)
Locator = argbind.bind(Locator)
Discriminator = argbind.bind(Discriminator)
AudioWatermarking = argbind.bind(AudioWatermarking)

# =============================================================================
# DATA BINDINGS
# =============================================================================
# Bind dataset classes for train and validation
AudioDataset = argbind.bind(AudioDataset, "train", "val")
AudioLoader = argbind.bind(AudioLoader, "train", "val")

# =============================================================================
# TRANSFORM BINDINGS
# =============================================================================
# Define filter function for transform module binding
transform_filter_fn: TransformFn = lambda fn: (
    hasattr(fn, "transform") and 
    fn.__qualname__ not in ["BaseTransform", "Compose", "Choose"]
)
transform_module = argbind.bind_module(transforms, "train", "val", filter_fn=transform_filter_fn)

# =============================================================================
# LOSS FUNCTION BINDINGS
# =============================================================================
# Bind loss functions from loss module
loss_filter_fn: TransformFn = lambda fn: hasattr(fn, "forward") and "Loss" in fn.__name__
losses = argbind.bind_module(loss, filter_fn=loss_filter_fn)

# =============================================================================
# EVALUATION METRIC BINDINGS
# =============================================================================
# Bind evaluation metrics from evaluate module
eval_filter_fn: TransformFn = lambda fn: hasattr(fn, "forward")
evaluation = argbind.bind_module(evaluate, filter_fn=eval_filter_fn)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap model from DistributedDataParallel wrapper if necessary.
    
    Args:
        model: Model that may be wrapped in DDP
        
    Returns:
        Unwrapped model instance
    """
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def generate_random_message(
    batch_size: int, 
    message_length: int, 
    device: DeviceType
) -> torch.Tensor:
    """Generate random binary messages for watermarking.
    
    Args:
        batch_size: Number of messages to generate
        message_length: Length of each message in bits
        device: Device to place the tensor on
        
    Returns:
        Tensor of random binary messages with shape (batch_size, message_length)
    """
    return torch.randint(
        MESSAGE_MIN_VALUE, 
        MESSAGE_MAX_VALUE, 
        (batch_size, message_length), 
        dtype=torch.long
    ).to(device)


def safe_wandb_log(metrics: Dict[str, Any], prefix: str = "") -> None:
    """Safely log metrics to WandB with error handling.
    
    Args:
        metrics: Dictionary of metrics to log
        prefix: Optional prefix to add to metric names
    """
    try:
        if prefix:
            metrics = {f"{prefix}/{key}": value for key, value in metrics.items()}
        wandb.log(metrics)
    except Exception as e:
        logger.warning(f"Failed to log metrics to wandb: {str(e)}")


def create_checkpoint_directory(base_path: PathType, tag: str) -> Dict[str, Path]:
    """Create directory structure for checkpoint saving.
    
    Args:
        base_path: Base directory for checkpoints
        tag: Checkpoint tag/version
        
    Returns:
        Dictionary mapping component names to their paths
        
    Raises:
        IOError: If directory creation fails
    """
    try:
        paths = {
            "generator": Path(base_path) / tag / "generator",
            "detector": Path(base_path) / tag / "detector",
            "locator": Path(base_path) / tag / "locator",
            "discriminator": Path(base_path) / tag / "discriminator"
        }
        
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
            
        return paths
    except Exception as e:
        logger.error(f"Failed to create checkpoint directories: {str(e)}", exc_info=True)
        raise CheckpointError("Checkpoint directory creation failed") from e


def log_gpu_memory(gpu_id: Optional[int] = None) -> None:
    """Log GPU memory usage statistics.
    
    Args:
        gpu_id: Specific GPU ID to log, or None for all GPUs
    """
    if not torch.cuda.is_available():
        return
        
    if gpu_id is not None:
        logger.info(f"GPU {gpu_id} memory:")
        logger.info(f"  Allocated: {torch.cuda.memory_allocated(gpu_id) / BYTES_TO_MB:.2f} MB")
        logger.info(f"  Cached: {torch.cuda.memory_reserved(gpu_id) / BYTES_TO_MB:.2f} MB")
    else:
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i} memory:")
            logger.info(f"  Allocated: {torch.cuda.memory_allocated(i) / BYTES_TO_MB:.2f} MB")
            logger.info(f"  Cached: {torch.cuda.memory_reserved(i) / BYTES_TO_MB:.2f} MB")


def clear_gpu_memory(gpu_id: Optional[int] = None) -> None:
    """Clear memory for specific GPU or all GPUs.
    
    Args:
        gpu_id: Specific GPU ID to clear, or None to clear all GPUs
    """
    try:
        # First check if CUDA is available
        if not torch.cuda.is_available():
            logger.info("CUDA is not available. Running on CPU.")
            return
        
        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
        
        # Handle specific GPU or all GPUs
        if gpu_id is not None:
            # Verify the requested GPU is valid
            if gpu_id >= num_gpus:
                logger.warning(f"GPU {gpu_id} not available. Only {num_gpus} GPUs found.")
                return
            
            # Clear specific GPU
            device = torch.device(f'cuda:{gpu_id}')
            torch.cuda.set_device(device)
            
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                
                # Clear all PyTorch tensors from this specific GPU
                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj) and obj.device.index == gpu_id:
                            del obj
                    except Exception:
                        # Ignore errors from accessing object attributes
                        pass
                
                # Force garbage collection
                gc.collect()
                
                # Report memory status
                logger.info(f"Cleared GPU {gpu_id} memory:")
                log_gpu_memory(gpu_id)
        else:
            # Clear all available GPUs
            torch.cuda.empty_cache()
            gc.collect()
            
            # Report memory status for all GPUs
            logger.info("GPU memory after clearing:")
            log_gpu_memory()
    except Exception as e:
        logger.error(f"Error clearing GPU memory: {str(e)}", exc_info=True)
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            logger.info(f"Current device: {torch.cuda.current_device()}")


def get_infinite_loader(dataloader: DataLoader) -> Iterator[BatchData]:
    """Create an infinite iterator from a dataloader.
    
    Args:
        dataloader: PyTorch DataLoader to iterate over infinitely
        
    Returns:
        Iterator that yields batches infinitely
        
    Yields:
        BatchData: Dictionary containing batch data
    """
    while True:
        for batch in dataloader:
            yield batch


@argbind.bind("train", "val")
def build_transform(
    augment_prob: float = 1.0,
    preprocess: List[str] = ["Identity"],
    augment: List[str] = ["Identity"],
    postprocess: List[str] = ["Identity"],
) -> transforms.Compose:
    """Build a transform pipeline for audio data.
    
    Creates a composed transform with preprocessing, augmentation, and postprocessing stages.
    
    Args:
        augment_prob: Probability of applying augmentation transforms
        preprocess: List of preprocessing transform names
        augment: List of augmentation transform names
        postprocess: List of postprocessing transform names
        
    Returns:
        Composed transform pipeline
        
    Raises:
        AttributeError: If transform name not found in transform module
    """
    # Convert transform names to transform objects
    to_transform = lambda transform_list: [
        getattr(transform_module, transform_name)() 
        for transform_name in transform_list
    ]
    
    # Build transform pipeline stages
    preprocess_transforms = transforms.Compose(*to_transform(preprocess), name="preprocess")
    augment_transforms = transforms.Compose(*to_transform(augment), name="augment", prob=augment_prob)
    postprocess_transforms = transforms.Compose(*to_transform(postprocess), name="postprocess")
    
    # Compose all stages
    transform = transforms.Compose(preprocess_transforms, augment_transforms, postprocess_transforms)
    return transform


@argbind.bind("train", "val", "test")
def build_dataset(
    sample_rate: int,
    folders: Optional[Dict[str, List[str]]] = None,
) -> ConcatDataset:
    """Build a concatenated dataset from multiple audio folders.
    
    Creates individual datasets for each folder group and concatenates them.
    The ConcatDataset cycles through each dataset during iteration.
    
    Args:
        sample_rate: Target sample rate for audio
        folders: Dictionary mapping dataset names to lists of folder paths
        
    Returns:
        Concatenated dataset containing all audio sources
        
    Raises:
        ValueError: If folders is None or empty
    """
    if not folders:
        raise DatasetError("folders parameter must be provided and non-empty")
    
    datasets = []
    
    # Create a dataset for each folder group
    for dataset_name, folder_paths in folders.items():
        logger.info(f"Loading dataset '{dataset_name}' from {len(folder_paths)} folders")
        
        loader = AudioLoader(sources=folder_paths)
        transform = build_transform()
        dataset = AudioDataset(loader, sample_rate, transform=transform)
        datasets.append(dataset)
    
    # Concatenate all datasets
    concatenated_dataset = ConcatDataset(datasets)
    concatenated_dataset.transform = transform  # Store transform for reference
    
    return concatenated_dataset


@dataclass
class State:
    """Training state container for audio watermarking models.
    
    This dataclass holds all components needed during training including models,
    optimizers, loss functions, evaluation metrics, and datasets. It serves as a
    centralized container that gets passed between training functions to maintain
    state throughout the training process.
    
    Attributes:
        audiowatermarking_model: Combined watermarking model containing generator,
            detector, and locator components. The generator embeds watermarks,
            the detector identifies watermark presence, and the locator determines
            watermark position in the audio signal.
        optimizer: Optimizer for the watermarking model parameters
        scheduler: Learning rate scheduler for adaptive learning rate decay
        discriminator: Discriminator model for adversarial training to improve
            perceptual quality of watermarked audio
        optimizer_d: Optimizer specifically for discriminator parameters
        scheduler_d: Learning rate scheduler for discriminator training
        message_threshold: Binary threshold (0-1) for watermark detection decisions
        stft_loss: Multi-scale STFT loss for frequency domain reconstruction
        mel_loss: Mel-spectrogram loss for perceptual frequency matching
        gan_loss: GAN loss for adversarial training between generator/discriminator
        waveform_loss: L1 loss on raw waveforms for time-domain reconstruction
        loc_loss: Localization loss for accurate watermark position detection
        dec_loss: Decoding loss for accurate message extraction from watermarks
        pesq_eval: PESQ (Perceptual Evaluation of Speech Quality) metric
        stoi_eval: STOI (Short-Time Objective Intelligibility) metric
        sisnr_eval: SI-SNR (Scale-Invariant Signal-to-Noise Ratio) metric
        ber_eval: BER (Bit Error Rate) metric for watermark accuracy
        train_data: Training dataset with augmentation transforms
        val_data: Validation dataset for model evaluation
        tracker: Progress tracker for logging and checkpoint management
    """
    audiowatermarking_model: AudioWatermarking
    optimizer: OptimizerType
    scheduler: SchedulerType
    
    discriminator: Discriminator
    optimizer_d: OptimizerType
    scheduler_d: SchedulerType
    
    message_threshold: float
    
    stft_loss: losses.MultiScaleSTFTLoss
    mel_loss: losses.MelSpectrogramLoss
    gan_loss: losses.GANLoss
    waveform_loss: losses.L1Loss
    
    loc_loss: losses.LocalizationLoss
    dec_loss: losses.DecodingLoss
    
    pesq_eval: evaluation.PESQ
    stoi_eval: evaluation.STOI
    sisnr_eval: evaluation.SISNR
    ber_eval: evaluation.BER
    
    train_data: AudioDataset
    val_data: AudioDataset
    
    tracker: Tracker

def load_state_dict_from_path(
    path: Path, 
    map_location: DeviceType = 'cpu'
) -> Optional[StateDict]:
    """Safely load a state dictionary from a file path.
    
    Args:
        path: Path to the state dict file
        map_location: Device to map tensors to during loading
        
    Returns:
        State dictionary if file exists, None otherwise
        
    Raises:
        RuntimeError: If file exists but cannot be loaded
    """
    if not path.exists():
        logger.warning(f"State dict path does not exist: {path}")
        return None
        
    try:
        state_dict = torch.load(path, map_location=map_location)
        logger.info(f"Successfully loaded state dict from {path}")
        return state_dict
    except Exception as e:
        logger.error(f"Failed to load state dict from {path}: {str(e)}", exc_info=True)
        raise CheckpointError(f"Cannot load state dict from {path}") from e

# =============================================================================
# MODEL LOADING AND INITIALIZATION
# =============================================================================
def _initialize_models() -> Tuple[AudioWatermarking, Discriminator]:
    """Initialize all model components.
    
    Creates fresh instances of all models required for training:
    - Generator: Embeds watermarks into audio signals
    - Detector: Identifies presence of watermarks
    - Locator: Determines position of watermarks
    - Discriminator: Distinguishes real from watermarked audio
    
    The generator, detector, and locator are combined into a single
    AudioWatermarking model for coordinated training.
    
    Returns:
        Tuple of (AudioWatermarking model, Discriminator model)
        
    Raises:
        ModelInitializationError: If any model component fails to initialize
            due to configuration errors or resource constraints
    """
    try:
        generator = Generator()
        detector = Detector()
        locator = Locator()
        discriminator = Discriminator()
        audiowatermarking_model = AudioWatermarking(generator, detector, locator)
        
        logger.info("Successfully initialized all models")
        return audiowatermarking_model, discriminator
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {str(e)}", exc_info=True)
        raise ModelInitializationError("Model initialization failed") from e


def _load_checkpoint_states(
    audiowatermarking_model: AudioWatermarking,
    discriminator: Discriminator,
    save_path: PathType,
    tag: str,
    map_location: DeviceType
) -> Tuple[Optional[StateDict], Optional[StateDict], Optional[StateDict], Optional[StateDict], Optional[StateDict]]:
    """Load checkpoint states for all components.
    
    Args:
        audiowatermarking_model: AudioWatermarking model to load states into
        discriminator: Discriminator model to load states into
        save_path: Base checkpoint directory
        tag: Checkpoint tag to load
        map_location: Device to map tensors to
        
    Returns:
        Tuple of (optimizer_state, scheduler_state, optimizer_d_state, scheduler_d_state, tracker_state)
        
    Raises:
        FileNotFoundError: If checkpoint tag not found
        RuntimeError: If checkpoint loading fails
    """
    checkpoint_base = Path(save_path) / tag
    
    if not checkpoint_base.exists():
        raise FileNotFoundError(f"Checkpoint tag '{tag}' not found at {save_path}")
    
    # Define component paths
    generator_folder = checkpoint_base / "generator"
    detector_folder = checkpoint_base / "detector"
    locator_folder = checkpoint_base / "locator"
    discriminator_folder = checkpoint_base / "discriminator"
    
    # Load generator state
    generator_state = load_state_dict_from_path(generator_folder / "model.pth", map_location)
    if generator_state:
        audiowatermarking_model.generator.load_state_dict(generator_state)
    
    # Load optimizer and scheduler states
    optimizer_state = load_state_dict_from_path(generator_folder / "optimizer.pth", map_location)
    scheduler_state = load_state_dict_from_path(generator_folder / "scheduler.pth", map_location)
    tracker_state = load_state_dict_from_path(generator_folder / "tracker.pth", map_location)
    
    # Load detector state
    detector_state = load_state_dict_from_path(detector_folder / "model.pth", map_location)
    if detector_state:
        audiowatermarking_model.detector.load_state_dict(detector_state)
    
    # Load locator state
    locator_state = load_state_dict_from_path(locator_folder / "model.pth", map_location)
    if locator_state:
        audiowatermarking_model.locator.load_state_dict(locator_state)
    
    # Load discriminator with key remapping
    try:
        discriminator_state = load_state_dict_from_path(discriminator_folder / "model.pth", map_location)
        if discriminator_state:
            # Remap keys for backward compatibility
            remapped_state = {
                k.replace("discriminators.", DISCRIMINATOR_KEY_PREFIX): v 
                for k, v in discriminator_state.items()
            }
            discriminator.load_state_dict(remapped_state, strict=False)
            logger.info("Loaded discriminator state dict with prefix remapping")
    except Exception as e:
        logger.warning(f"Could not load discriminator state: {str(e)}")
    
    # Load discriminator optimizer states
    optimizer_d_state = load_state_dict_from_path(discriminator_folder / "optimizer.pth", map_location)
    scheduler_d_state = load_state_dict_from_path(discriminator_folder / "scheduler.pth", map_location)
    
    logger.info(f"Successfully resumed training from tag: {tag}")
    
    return optimizer_state, scheduler_state, optimizer_d_state, scheduler_d_state, tracker_state


def _initialize_optimizers(
    audiowatermarking_model: AudioWatermarking,
    discriminator: Discriminator,
    args: ConfigDict,
    accel: ml.Accelerator,
    optimizer_state: Optional[StateDict],
    scheduler_state: Optional[StateDict],
    optimizer_d_state: Optional[StateDict],
    scheduler_d_state: Optional[StateDict]
) -> Tuple[OptimizerType, SchedulerType, OptimizerType, SchedulerType]:
    """Initialize optimizers and schedulers with optional state loading.
    
    Args:
        audiowatermarking_model: Model for generator optimizer
        discriminator: Model for discriminator optimizer
        args: Configuration arguments
        accel: Accelerator for distributed training
        optimizer_state: Optional state for generator optimizer
        scheduler_state: Optional state for generator scheduler
        optimizer_d_state: Optional state for discriminator optimizer
        scheduler_d_state: Optional state for discriminator scheduler
        
    Returns:
        Tuple of (optimizer, scheduler, optimizer_d, scheduler_d)
    """
    # Initialize generator optimizer and scheduler
    with argbind.scope(args, "audiowatermarking"):
        optimizer = AdamW(audiowatermarking_model.parameters(), use_zero=accel.use_ddp)
        scheduler = ExponentialLR(optimizer)
    
    # Initialize discriminator optimizer and scheduler
    with argbind.scope(args, "discriminator"):
        optimizer_d = AdamW(discriminator.parameters(), use_zero=accel.use_ddp)
        scheduler_d = ExponentialLR(optimizer_d)
    
    # Load states if available
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
    if scheduler_state:
        scheduler.load_state_dict(scheduler_state)
    if optimizer_d_state:
        optimizer_d.load_state_dict(optimizer_d_state)
    if scheduler_d_state:
        scheduler_d.load_state_dict(scheduler_d_state)
    
    return optimizer, scheduler, optimizer_d, scheduler_d


def _initialize_losses() -> Tuple[losses.L1Loss, losses.MultiScaleSTFTLoss, losses.MelSpectrogramLoss, 
                                  losses.GANLoss, losses.LocalizationLoss, losses.DecodingLoss]:
    """Initialize all loss functions.
    
    Returns:
        Tuple of loss functions
    """
    waveform_loss = losses.L1Loss()  # L1 loss on raw waveforms
    stft_loss = losses.MultiScaleSTFTLoss()  # Multi-scale STFT loss
    mel_loss = losses.MelSpectrogramLoss()  # Mel-spectrogram loss
    gan_loss = losses.GANLoss()  # Placeholder - discriminator added in load()
    loc_loss = losses.LocalizationLoss()  # Localization loss for watermark position
    dec_loss = losses.DecodingLoss()  # Decoding loss for message extraction
    
    return waveform_loss, stft_loss, mel_loss, gan_loss, loc_loss, dec_loss


def _initialize_evaluation_metrics(sample_rate: int) -> Tuple[evaluation.PESQ, evaluation.STOI, 
                                                              evaluation.SISNR, evaluation.BER]:
    """Initialize evaluation metrics.
    
    Args:
        sample_rate: Audio sample rate for PESQ metric
        
    Returns:
        Tuple of evaluation metrics
    """
    pesq_eval = evaluation.PESQ(target_sr=sample_rate)  # Perceptual quality metric
    stoi_eval = evaluation.STOI()  # Short-time objective intelligibility
    sisnr_eval = evaluation.SISNR()  # Scale-invariant signal-to-noise ratio
    ber_eval = evaluation.BER()  # Bit error rate for watermark detection
    
    return pesq_eval, stoi_eval, sisnr_eval, ber_eval


@argbind.bind(without_prefix=True)
def load(
    args: ConfigDict,
    accel: ml.Accelerator,
    tracker: Tracker,
    save_path: PathType,
    resume: bool = False,
    tag: str = "latest",
    map_location: Optional[DeviceType] = None,
) -> State:
    """Load models and create training state.
    
    Initializes all models, optimizers, schedulers, loss functions, and datasets.
    Optionally resumes from a checkpoint if resume=True.
    
    Args:
        args: Configuration arguments
        accel: Accelerator for distributed training
        tracker: Progress tracker for logging
        save_path: Base path for saving checkpoints
        resume: Whether to resume from checkpoint
        tag: Checkpoint tag to resume from
        map_location: Device to map tensors to when loading
        
    Returns:
        State object containing all training components
        
    Raises:
        RuntimeError: If checkpoint loading fails
        ValueError: If required configurations are missing
    """
    if map_location is None:
        map_location = 'cpu'
    
    # Initialize models
    audiowatermarking_model, discriminator = _initialize_models()
    
    # Load checkpoint states if resuming
    if resume:
        try:
            optimizer_state, scheduler_state, optimizer_d_state, scheduler_d_state, tracker_state = \
                _load_checkpoint_states(
                    audiowatermarking_model, discriminator, save_path, tag, map_location
                )
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {str(e)}", exc_info=True)
            raise CheckpointError(f"Cannot resume from checkpoint tag '{tag}'") from e
    else:
        # Initialize fresh training state
        optimizer_state = None
        scheduler_state = None
        optimizer_d_state = None
        scheduler_d_state = None
        tracker_state = None
        logger.info("Initializing training from scratch")
    
    # Log model architectures if debug logging is enabled
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Generator architecture:\n{audiowatermarking_model.generator}")
        logger.debug(f"Detector architecture:\n{audiowatermarking_model.detector}")
        logger.debug(f"Locator architecture:\n{audiowatermarking_model.locator}")
        logger.debug(f"Discriminator architecture:\n{discriminator}")
    
    # Prepare models for distributed training
    audiowatermarking_model = accel.prepare_model(audiowatermarking_model)
    discriminator = accel.prepare_model(discriminator)
    
    # Initialize optimizers and schedulers
    optimizer, scheduler, optimizer_d, scheduler_d = _initialize_optimizers(
        audiowatermarking_model, discriminator, args, accel,
        optimizer_state, scheduler_state, optimizer_d_state, scheduler_d_state
    )
    
    # Load tracker state if available
    if tracker_state:
        tracker.load_state_dict(tracker_state)
    
    # Extract sample rate from unwrapped generator model
    sample_rate = unwrap_model(audiowatermarking_model).generator.sample_rate
    
    # Set default message detection threshold
    message_threshold = DEFAULT_MESSAGE_THRESHOLD
    
    # Build datasets
    with argbind.scope(args, "train"):
        train_data = build_dataset(sample_rate)
    with argbind.scope(args, "val"):
        val_data = build_dataset(sample_rate)
    
    # Initialize loss functions
    waveform_loss, stft_loss, mel_loss, gan_loss_base, loc_loss, dec_loss = _initialize_losses()
    
    # Create GAN loss with discriminator
    gan_loss = losses.GANLoss(discriminator)
    
    # Initialize evaluation metrics
    pesq_eval, stoi_eval, sisnr_eval, ber_eval = _initialize_evaluation_metrics(sample_rate)

    return State(
        audiowatermarking_model= audiowatermarking_model,
        optimizer=optimizer,
        scheduler=scheduler,
        message_threshold=message_threshold,
        discriminator=discriminator,
        optimizer_d=optimizer_d,
        scheduler_d=scheduler_d,
        waveform_loss=waveform_loss,
        stft_loss=stft_loss,
        mel_loss=mel_loss,
        gan_loss=gan_loss,
        loc_loss=loc_loss,
        dec_loss=dec_loss,
        pesq_eval = pesq_eval,
        stoi_eval = stoi_eval,
        sisnr_eval = sisnr_eval,
        ber_eval = ber_eval,
        tracker=tracker,
        train_data=train_data,
        val_data=val_data,
    )


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================
def _compute_reconstruction_losses(
    state: State,
    watermarked_signal: torch.Tensor,
    signal: torch.Tensor,
    recons: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Compute reconstruction losses for validation.
    
    Args:
        state: Current training state
        watermarked_signal: Watermarked audio signal
        signal: Original signal
        recons: Reconstructed signal
        
    Returns:
        Dictionary of reconstruction losses
    """
    losses = {
        "stft/loss": state.stft_loss(watermarked_signal, signal),
        "mel/loss": state.mel_loss(watermarked_signal, signal),
        "waveform/loss": state.waveform_loss(watermarked_signal, signal)
    }
    return losses


def _compute_adversarial_losses(
    state: State,
    watermarked_signal: torch.Tensor,
    signal: torch.Tensor,
    recons: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Compute adversarial losses for validation.
    
    Args:
        state: Current training state
        watermarked_signal: Watermarked audio signal
        signal: Original signal
        recons: Reconstructed signal
        
    Returns:
        Dictionary of adversarial losses
    """
    losses = {}
    
    # Discriminator loss (no gradient penalty in validation)
    losses["adv/disc_loss"] = state.gan_loss.discriminator_loss(
        recons, signal, use_gradient_penalty=False
    )
    
    # Generator and feature losses
    losses["adv/gen_loss"], losses["adv/feat_loss"] = state.gan_loss.generator_loss(
        watermarked_signal, signal
    )
    
    return losses


def _compute_detection_losses(
    state: State,
    results_dict: Dict[str, Any],
    message: torch.Tensor
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    """Compute detection and localization losses.
    
    Args:
        state: Current training state
        results_dict: Dictionary of results from different effects
        message: Binary message
        
    Returns:
        Tuple of (losses dict, per-effect metrics dict)
    """
    total_detector_loss = 0.0
    total_locator_loss = 0.0
    per_effect_metrics = {}
    
    # Compute losses for each effect
    for effect_name, result in results_dict.items():
        detector_output = result['detector_output']
        locator_output = result['locator_output']
        mask = result['mask']
        
        total_detector_loss += state.dec_loss(detector_output, mask, message)
        total_locator_loss += state.loc_loss(locator_output, mask)
        
        # Extract per-effect metrics
        per_effect_metrics[f"{effect_name}/ber"] = result['ber']
        per_effect_metrics[f"{effect_name}/miou"] = result['miou']
    
    # Average losses across all effects
    losses = {
        "dec/loss": total_detector_loss / len(results_dict),
        "loc/loss": total_locator_loss / len(results_dict)
    }
    
    return losses, per_effect_metrics


def _compute_quality_metrics(
    state: State,
    watermarked_signal: torch.Tensor,
    signal: torch.Tensor,
    output: Dict[str, torch.Tensor]
) -> MetricsDict:
    """Compute perceptual quality metrics and prepare final metrics dict.
    
    Args:
        state: Current training state
        watermarked_signal: Watermarked audio signal
        signal: Original signal
        output: Dictionary of all computed losses
        
    Returns:
        Dictionary of all validation metrics
    """
    metrics = {
        "Total Loss": output["loss"].item(),
        "MEL Loss": output["mel/loss"].item(),
        "STFT Loss": output["stft/loss"].item(),
        "Waveform Loss": output["waveform/loss"].item(),
        "Generator Loss": output["adv/gen_loss"].item(),
        "Discriminator Loss": output["adv/disc_loss"].item(),
        "Feature Loss": output["adv/feat_loss"].item(),
        "Decoding Loss": output["dec/loss"].item(),
        "Localization Loss": output["loc/loss"].item(),
        "STOI": state.stoi_eval(watermarked_signal, signal).item(),
        "PESQ": state.pesq_eval(watermarked_signal, signal).item(),
        "SISNR": state.sisnr_eval(watermarked_signal, signal).item(),
    }
    
    # Add per-effect metrics
    for key, value in output.items():
        if key.endswith(("ber", "miou")):
            metrics[key] = value
    
    return metrics


@timer()
@torch.no_grad()
def val_loop(
    batch: BatchData, 
    state: State, 
    accel: ml.Accelerator, 
    lambdas: LossWeights
) -> MetricsDict:
    """Execute single validation loop iteration.
    
    Performs forward pass through models in evaluation mode and computes
    all validation metrics without gradient computation.
    
    Args:
        batch: Batch of validation data
        state: Current training state
        accel: Accelerator for distributed validation
        lambdas: Loss weight coefficients
        
    Returns:
        Dictionary of validation metrics
        
    Raises:
        RuntimeError: If validation forward pass fails
    """
    try:
        # Set model to evaluation mode to disable dropout and batch norm updates
        state.audiowatermarking_model.eval()
        
        # Prepare batch data and move to appropriate device
        batch = util.prepare_batch(batch, accel.device)
        phase = "valid"
        
        # Apply validation transforms (typically less aggressive than training)
        signal = state.val_data.transform(
            batch["signal"].clone(), **batch["transform_args"]
        )
        
        # Generate random binary messages for watermark validation
        message = generate_random_message(
            signal.shape[0], DEFAULT_MESSAGE_LENGTH, accel.device
        )
        
        # Forward pass through model
        recons, watermarked_signal, results_dict = state.audiowatermarking_model(
            signal, message, phase
        )
    
        # Compute all losses with automatic mixed precision
        with accel.autocast():
            output = {}
            
            # Compute reconstruction losses
            reconstruction_losses = _compute_reconstruction_losses(
                state, watermarked_signal, signal, recons
            )
            output.update(reconstruction_losses)
            
            # Compute adversarial losses
            adversarial_losses = _compute_adversarial_losses(
                state, watermarked_signal, signal, recons
            )
            output.update(adversarial_losses)
            
            # Compute detection and localization losses
            detection_losses, per_effect_metrics = _compute_detection_losses(
                state, results_dict, message
            )
            output.update(detection_losses)
            output.update(per_effect_metrics)
            
            # Compute weighted total loss
            output["loss"] = sum([
                value * lambdas[key] 
                for key, value in output.items() 
                if key in lambdas
            ])
            
            logger.info(f"Validation Loss: {output['loss'].item():.4f}")
        
        # Compute quality metrics and prepare final output
        metrics = _compute_quality_metrics(state, watermarked_signal, signal, output)
        
        # Log validation metrics to wandb
        safe_wandb_log(metrics, prefix="val")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Validation loop failed: {str(e)}", exc_info=True)
        raise ValidationError("Validation forward pass failed") from e

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================
def _prepare_training_batch(
    state: State,
    batch: BatchData,
    accel: ml.Accelerator
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare batch data for training.
    
    This function handles data preparation including device placement,
    augmentation transforms, and random message generation. All operations
    are performed without gradient computation for efficiency.
    
    Args:
        state: Current training state containing data transforms
        batch: Raw batch data from dataloader
        accel: Accelerator for proper device placement
        
    Returns:
        Tuple of (signal, message) tensors where:
            - signal: Augmented audio signal tensor
            - message: Random binary message for watermarking
    """
    # Prepare batch and move to device for GPU acceleration
    batch = util.prepare_batch(batch, accel.device)
    
    with torch.no_grad():
        # Apply data augmentation transforms to increase robustness
        signal = state.train_data.transform(
            batch["signal"].clone(), **batch["transform_args"]
        )
        
        # Generate random binary messages for watermark embedding
        message = generate_random_message(
            signal.shape[0], DEFAULT_MESSAGE_LENGTH, accel.device
        )
    
    return signal, message


def _update_discriminator(
    state: State,
    accel: ml.Accelerator,
    recons: torch.Tensor,
    signal: torch.Tensor
) -> Tuple[torch.Tensor, float]:
    """Update discriminator with adversarial loss.
    
    Performs a complete discriminator update step including loss computation,
    backpropagation, gradient clipping, and optimizer step. The discriminator
    learns to distinguish between real and watermarked audio.
    
    Args:
        state: Current training state with discriminator and optimizer
        accel: Accelerator for mixed precision and distributed training
        recons: Reconstructed signal from generator (fake samples)
        signal: Original clean signal (real samples)
        
    Returns:
        Tuple of (discriminator loss, gradient norm) where:
            - discriminator loss: Computed adversarial loss value
            - gradient norm: L2 norm of gradients after clipping
    """
    # Compute discriminator loss
    with accel.autocast():
        disc_loss = state.gan_loss.discriminator_loss(recons, signal)
    
    # Backward pass
    state.optimizer_d.zero_grad()
    accel.backward(disc_loss)
    
    # Gradient clipping
    accel.scaler.unscale_(state.optimizer_d)
    grad_norm_d = torch.nn.utils.clip_grad_norm_(
        state.discriminator.parameters(), MAX_GRADIENT_NORM
    )
    
    # Update weights
    accel.step(state.optimizer_d)
    state.scheduler_d.step()
    
    return disc_loss, grad_norm_d


def _update_generator(
    state: State,
    accel: ml.Accelerator,
    model: nn.Module,
    watermarked_signal: torch.Tensor,
    signal: torch.Tensor,
    detector_out: torch.Tensor,
    gt_presence: torch.Tensor,
    locator_out: torch.Tensor,
    message: torch.Tensor,
    lambdas: LossWeights
) -> Tuple[Dict[str, torch.Tensor], float]:
    """Update generator with combined losses.
    
    Args:
        state: Current training state
        accel: Accelerator for distributed training
        model: Unwrapped model
        watermarked_signal: Watermarked audio signal
        signal: Original signal
        detector_out: Detector output
        gt_presence: Ground truth watermark presence
        locator_out: Locator output
        message: Binary message
        lambdas: Loss weight coefficients
        
    Returns:
        Tuple of (loss dictionary, gradient norm)
    """
    output = {}
    
    with accel.autocast():
        # Reconstruction losses
        output["stft/loss"] = state.stft_loss(watermarked_signal, signal)
        output["mel/loss"] = state.mel_loss(watermarked_signal, signal)
        output["waveform/loss"] = state.waveform_loss(watermarked_signal, signal)
        
        # Adversarial losses
        output["adv/gen_loss"], output["adv/feat_loss"] = state.gan_loss.generator_loss(
            watermarked_signal, signal
        )
        
        # Detection and localization losses
        output["dec/loss"] = state.dec_loss(detector_out, gt_presence, message)
        output["loc/loss"] = state.loc_loss(locator_out, gt_presence)
        
        # Compute weighted total loss
        output["loss"] = sum([
            value * lambdas[key] 
            for key, value in output.items() 
            if key in lambdas
        ])
    
    # Backward pass
    state.audiowatermarking_model.zero_grad()
    accel.backward(output["loss"])
    
    # Gradient clipping
    accel.scaler.unscale_(state.optimizer)
    grad_norm_gen = torch.nn.utils.clip_grad_norm_(
        model.generator.parameters(), MAX_GRADIENT_NORM
    )
    
    # Update weights
    accel.step(state.optimizer)
    state.scheduler.step()
    accel.update()
    
    return output, grad_norm_gen


def _log_training_metrics(
    output: Dict[str, torch.Tensor],
    grad_norm_gen: float,
    grad_norm_d: float,
    state: State,
    signal: torch.Tensor,
    accel: ml.Accelerator
) -> MetricsDict:
    """Log training metrics and prepare output dictionary.
    
    Args:
        output: Dictionary of losses
        grad_norm_gen: Generator gradient norm
        grad_norm_d: Discriminator gradient norm
        state: Current training state
        signal: Input signal for batch size calculation
        accel: Accelerator for world size
        
    Returns:
        Dictionary of scalar metrics
    """
    # Log main training metrics
    training_metrics = {
        "Generator Loss": output["adv/gen_loss"].item(),
        "Discriminator Loss": output["adv/disc_loss"].item(),
        "Feature Loss": output["adv/feat_loss"].item(),
        "MEL Loss": output["mel/loss"].item(),
        "STFT Loss": output["stft/loss"].item(),
        "Waveform Loss": output["waveform/loss"].item(),
        "Decoding Loss": output["dec/loss"].item(),
        "Localization Loss": output["loc/loss"].item(),
        "Total Loss": output["loss"].item()
    }
    safe_wandb_log(training_metrics)
    
    # Record training statistics
    output["other/gen_learning_rate"] = state.optimizer.param_groups[0]["lr"]
    output["other/batch_size"] = signal.batch_size * accel.world_size
    output["other/grad_norm_gen"] = grad_norm_gen
    output["other/grad_norm_d"] = grad_norm_d
    
    # Log learning metrics
    learning_metrics = {
        "Learning Rate": output["other/gen_learning_rate"],
        "Batch Size": output["other/batch_size"],
        "Generator Gradient Norm": grad_norm_gen,
        "Discriminator Gradient Norm": grad_norm_d
    }
    safe_wandb_log(learning_metrics)
    
    # Convert tensors to scalars and sort output
    return {
        key: value.item() if hasattr(value, 'item') else value 
        for key, value in sorted(output.items())
    }


@timer()
def train_loop(
    state: State, 
    batch: BatchData, 
    accel: ml.Accelerator, 
    lambdas: LossWeights
) -> MetricsDict:
    """Execute single training loop iteration.
    
    Performs forward and backward passes for both generator and discriminator,
    updates model parameters, and computes training metrics.
    
    Args:
        state: Current training state
        batch: Batch of training data  
        accel: Accelerator for distributed training
        lambdas: Loss weight coefficients
        
    Returns:
        Dictionary of training metrics
        
    Raises:
        RuntimeError: If training step fails
    """
    try:
        # Set models to training mode
        state.audiowatermarking_model.train()
        state.discriminator.train()
        
        phase = 'train'
        
        # Unwrap model from DDP if necessary
        model = unwrap_model(state.audiowatermarking_model)
        
        # Prepare batch data
        signal, message = _prepare_training_batch(state, batch, accel)
        
        # Forward pass with automatic mixed precision
        with accel.autocast():
            recons, watermarked_signal, detector_out, gt_presence, update_original, stats, locator_out = model(
                signal, message, phase
            )
        
        # Update discriminator
        disc_loss, grad_norm_d = _update_discriminator(state, accel, recons, signal)
        
        # Update generator
        output, grad_norm_gen = _update_generator(
            state, accel, model, watermarked_signal, signal,
            detector_out, gt_presence, locator_out, message, lambdas
        )
        
        # Add discriminator loss to output
        output["adv/disc_loss"] = disc_loss
        
        # Log metrics and return formatted output
        return _log_training_metrics(output, grad_norm_gen, grad_norm_d, state, signal, accel)
    
    except Exception as e:
        logger.error(f"An error occurred in training loop: {str(e)}", exc_info=True) 
        raise TrainingError("Training loop failed") from e

def checkpoint(
    state: State, 
    save_iters: List[int], 
    save_path: PathType
) -> None:
    """Save model checkpoints with multiple tags.
    
    Saves model, optimizer, and scheduler states for all components.
    Automatically tags checkpoints as 'latest', 'best' (if applicable),
    and iteration-specific tags.
    
    Args:
        state: Current training state
        save_iters: List of iterations to create named checkpoints
        save_path: Base directory for saving checkpoints
        
    Raises:
        IOError: If checkpoint saving fails
    """
    try:
        tags = ["latest"]
        logger.info(f"Saving checkpoint to {Path(save_path).absolute()}")
        
        # Unwrap model from DDP if necessary
        model = unwrap_model(state.audiowatermarking_model)
    
        # If we have validation history, check if this is the best model so far
        if (
            state.tracker.history.get("val")
            and state.tracker.history["val"].get("Total Loss")
            and len(state.tracker.history["val"]["Total Loss"]) > 0
        ):
            if state.tracker.is_best("val", "Total Loss"):
                logger.info("New best model found - saving checkpoint")
                tags.append("best")

        if state.tracker.step in save_iters:
            tags.append(f"{state.tracker.step // 1000}k")

        for tag in tags:
            # Create checkpoint directories
            paths = create_checkpoint_directory(save_path, tag)
            gen_path = paths["generator"]
            det_path = paths["detector"]
            loc_path = paths["locator"]
            disc_path = paths["discriminator"]


            # Saving generator components
            torch.save(model.generator.state_dict(), gen_path / "model.pth")
            torch.save(model.detector.state_dict(), det_path / "model.pth")
            torch.save(model.locator.state_dict(), loc_path / "model.pth")
            torch.save(state.optimizer.state_dict(), gen_path / "optimizer.pth")
            torch.save(state.scheduler.state_dict(), gen_path / "scheduler.pth")
            torch.save(state.tracker.state_dict(), gen_path / "tracker.pth")

            # Saving discriminator components
            discriminator_state_dict = state.discriminator.state_dict()
            discriminator_state_dict = {f"{DISCRIMINATOR_KEY_PREFIX}{k}": v for k, v in discriminator_state_dict.items()}
            torch.save(discriminator_state_dict, disc_path / "model.pth")

            torch.save(state.optimizer_d.state_dict(), disc_path / "optimizer.pth")
            torch.save(state.scheduler_d.state_dict(), disc_path / "scheduler.pth")

            logger.info(f"Checkpoint saved successfully under tag: {tag}")
        
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {str(e)}", exc_info=True)
        raise CheckpointError("Checkpoint saving failed") from e

@torch.no_grad()
def save_samples(
    state: State, 
    val_idx: List[int], 
    writer: SummaryWriter
) -> None:
    """Save audio samples to TensorBoard and WandB.
    
    Args:
        state: Current training state
        val_idx: Indices of validation samples to save
        writer: TensorBoard summary writer
        
    Raises:
        RuntimeError: If sample generation fails
    """
    logger.info("Saving audio samples to TensorBoard and WandB")
    
    try:
        phase = AUDIO_SAMPLE_PHASE
        state.audiowatermarking_model.eval()
        
        # Collect validation samples
        samples = [state.val_data[idx] for idx in val_idx]
        batch = state.val_data.collate(samples)
        batch = util.prepare_batch(batch, accel.device)
        
        # Apply transforms
        signal = state.train_data.transform(
            batch["signal"].clone(), **batch["transform_args"]
        )
        
        # Generate random messages
        message = generate_random_message(
            signal.shape[0], DEFAULT_MESSAGE_LENGTH, accel.device
        )
        
        # Generate watermarked samples
        recons, watermarked_signal = state.audiowatermarking_model(signal, message, phase)

        # Prepare audio dictionary
        audio_dict = {
            "recons": recons,
            "watermarked_signal": watermarked_signal
        }
        
        # Include original signal at first iteration
        if state.tracker.step == 0:
            audio_dict["original_signal"] = signal
        
        # Save audio samples
        for audio_name, audio_tensor in audio_dict.items():
            if audio_name in AUDIO_SAMPLE_TYPES:
                for sample_idx in range(audio_tensor.batch_size):
                    # Save to TensorBoard
                    audio_tensor[sample_idx].cpu().write_audio_to_tb(
                        f"{audio_name}/sample_{sample_idx}.wav", 
                        writer, 
                        state.tracker.step
                    )
                    
                    # Save to WandB
                    audio_metrics = {
                        f"{audio_name}/sample_{sample_idx}": wandb.Audio(
                            audio_tensor[sample_idx].cpu().numpy().squeeze(),
                            sample_rate=audio_tensor.sample_rate,
                            caption=f"Sample {sample_idx}"
                        )
                    }
                    safe_wandb_log(audio_metrics)
                        
    except Exception as e:
        logger.error(f"Failed to save audio samples: {str(e)}", exc_info=True)
        raise ValidationError("Audio sample generation failed") from e 

def validate(
    state: State, 
    val_dataloader: DataLoader, 
    accel: ml.Accelerator, 
    lambdas: LossWeights
) -> MetricsDict:
    """Run validation on entire validation dataset.
    
    Args:
        state: Current training state
        val_dataloader: Validation data loader
        accel: Accelerator for distributed validation
        lambdas: Loss weight coefficients
        
    Returns:
        Dictionary of aggregated validation metrics
    """
    for batch in val_dataloader:
        output = val_loop(batch, state, accel, lambdas)

    # Consolidate state dicts if using ZeroRedundancyOptimizer
    if hasattr(state.optimizer, "consolidate_state_dict"):
        state.optimizer.consolidate_state_dict()
        state.optimizer_d.consolidate_state_dict()

    return output

# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================
@argbind.bind(without_prefix=True)
def train(
    args: ConfigDict,
    accel: ml.Accelerator,
    seed: int = DEFAULT_SEED,
    save_path: PathType = DEFAULT_SAVE_PATH,
    num_iters: int = DEFAULT_NUM_ITERS,
    save_iters: List[int] = DEFAULT_SAVE_ITERS,
    sample_freq: int = DEFAULT_SAMPLE_FREQ,
    valid_freq: int = DEFAULT_VALID_FREQ,
    batch_size: int = DEFAULT_BATCH_SIZE,
    val_batch_size: int = DEFAULT_VAL_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    val_idx: List[int] = [0, 1, 2, 3, 4, 5],
    lambdas: LossWeights = DEFAULT_LOSS_WEIGHTS,
) -> None:
    """Main training loop for audio watermarking.
    
    Args:
        args: Configuration arguments
        accel: Accelerator for distributed training
        seed: Random seed for reproducibility
        save_path: Directory for checkpoints
        num_iters: Total training iterations
        save_iters: Iterations at which to save checkpoints
        sample_freq: Frequency of saving audio samples
        valid_freq: Frequency of validation
        batch_size: Training batch size
        val_batch_size: Validation batch size
        num_workers: Number of data loading workers
        val_idx: Indices of validation samples to save
        lambdas: Loss weight coefficients
    """
    # Set random seed for reproducibility
    util.seed(seed)
    
    # Create checkpoint directory
    Path(save_path).mkdir(exist_ok=True, parents=True)
    
    # Initialize TensorBoard writer (only on main process)
    writer = (
        SummaryWriter(log_dir=f"{save_path}/logs") 
        if accel.local_rank == 0 else None
    )
    
    # Initialize progress tracker
    tracker = Tracker(
        writer=writer, 
        log_file=f"{save_path}/log.txt", 
        rank=accel.local_rank
    )

    # Load models and create training state
    state = load(args, accel, tracker, save_path)
    
    # Create training dataloader with resumption support
    train_dataloader = accel.prepare_dataloader(
        state.train_data,
        start_idx=state.tracker.step * batch_size,  # Resume from correct position
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=state.train_data.collate,
    )
    train_dataloader = get_infinite_loader(train_dataloader)
    
    # Create validation dataloader
    val_dataloader = accel.prepare_dataloader(
        state.val_data,
        start_idx=0,
        num_workers=num_workers,
        batch_size=val_batch_size,
        collate_fn=state.val_data.collate,
        persistent_workers=True if num_workers > 0 else False,
    )

    # Wrap functions with tracking and logging decorators
    global train_loop, val_loop, validate, save_samples, checkpoint
    
    # Wrap training loop with progress tracking
    train_loop = tracker.log("train", "value", history=False)(
        tracker.track("train", num_iters, completed=state.tracker.step)(train_loop)
    )
    
    # Wrap validation functions with tracking
    val_loop = tracker.track("val", len(val_dataloader))(val_loop)
    validate = tracker.log("val", "mean")(validate)
    
    # Ensure these functions only run on main process
    save_samples = when(lambda: accel.local_rank == 0)(save_samples)
    checkpoint = when(lambda: accel.local_rank == 0)(checkpoint)
    
    # Main training loop
    with tracker.live:
        for tracker.step, batch in enumerate(train_dataloader, start=tracker.step):
            # Execute training step
            train_loop(state, batch, accel, lambdas)
            
            # Check if this is the last iteration
            is_last_iteration = (
                tracker.step == num_iters - 1 if num_iters is not None else False
            )
            
            # Save audio samples at specified frequency
            if tracker.step % sample_freq == 0 or is_last_iteration:
                save_samples(state, val_idx, writer)
            
            # Run validation at specified frequency
            if tracker.step % valid_freq == 0 or is_last_iteration:
                validate(state, val_dataloader, accel, lambdas)
                checkpoint(state, save_iters, save_path)
                
                # Reset validation progress bar
                tracker.done("val", f"Iteration {tracker.step}")
            
            # Exit if reached target iterations
            if is_last_iteration:
                break
            

@argbind.bind(without_prefix=True)
def setup_run(args: ConfigDict) -> wandb.Run:
    """Initialize WandB run for experiment tracking.
    
    Args:
        args: Configuration arguments including project and run name
        
    Returns:
        WandB run object
        
    Raises:
        wandb.Error: If WandB initialization fails
    """
    try:
        run = wandb.init(
            project=args['project'],
            name=f"experiment_{args['run_name']}"
        )
        wandb.config.update(args)
        return run
    except Exception as e:
        logger.error(f"Failed to initialize WandB: {str(e)}", exc_info=True)
        raise


# =============================================================================
# MAIN EXECUTION HELPERS
# =============================================================================
def _setup_environment() -> None:
    """Set up the training environment.
    
    Configures warning filters and other environment settings.
    """
    # Suppress specific deprecation warnings
    warnings.filterwarnings(
        "ignore",
        message=".*`torch.distributed.reduce_op` is deprecated.*",
        category=FutureWarning
    )


def _configure_threads() -> None:
    """Configure OpenMP threads for optimal performance."""
    num_cores = os.cpu_count()
    max_threads = max(1, int(num_cores * THREAD_RATIO))  # Use configured ratio of available cores
    os.environ['OMP_NUM_THREADS'] = str(max_threads)
    logger.info(f"Configured OpenMP with {max_threads} threads (out of {num_cores} cores)")


def _log_gpu_info() -> None:
    """Log available GPU information."""
    logger.info(f"Number of available GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for gpu_index in range(torch.cuda.device_count()):
            logger.info(f"GPU {gpu_index}: {torch.cuda.get_device_name(gpu_index)}")


def _run_training(args: ConfigDict) -> None:
    """Run the training process with proper error handling.
    
    Args:
        args: Configuration arguments
        
    Raises:
        RuntimeError: If training fails
    """
    try:
        with argbind.scope(args):
            with Accelerator() as accel:
                # Limit traceback on non-main processes for cleaner output
                if accel.local_rank != 0:
                    sys.tracebacklimit = WORKER_TRACEBACK_LIMIT
                
                # Initialize experiment tracking
                logger.info("Initializing experiment tracking...")
                run = setup_run(args)
                
                # Start training
                logger.info("Starting training...")
                train(args, accel)
                
                # Successful completion
                logger.info("Training completed successfully!")
                
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        raise
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise TrainingError("Training execution failed") from e
    finally:
        # Ensure proper cleanup
        if 'run' in locals() and run is not None:
            try:
                wandb.finish()
                logger.info("WandB run finished successfully")
            except Exception as e:
                logger.warning(f"Failed to finish WandB run: {str(e)}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main() -> None:
    """Main entry point for the training script.
    
    Handles environment setup, GPU memory clearing, argument parsing,
    and launches training with proper error handling.
    
    Raises:
        RuntimeError: If training setup or execution fails
    """
    # Set up environment
    _setup_environment()
    
    # Configure thread settings
    _configure_threads()
    
    # Parse command-line arguments
    args = argbind.parse_args()
    
    # Enable debug mode only on main process
    args["args.debug"] = int(os.getenv("LOCAL_RANK", 0)) == 0
    
    # Log configuration summary
    logger.info("Configuration loaded successfully")
    logger.debug(f"Debug mode: {args['args.debug']}")
    
    # Log GPU information
    _log_gpu_info()
    
    # Clear GPU memory before starting
    gpu_id: Optional[int] = 0  # Change to specific GPU ID or None for all GPUs
    clear_gpu_memory(gpu_id)
    
    # Run training with error handling
    try:
        _run_training(args)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        sys.exit(1)


if __name__ == "__main__":
    main()