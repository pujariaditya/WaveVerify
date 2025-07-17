"""
Audio watermarking module for embedding and detecting watermarks in audio signals.

This module implements an adaptive watermarking system that can embed imperceptible
watermarks into audio signals and detect them even after various audio transformations.
The system uses neural networks for generation, detection, and localization of watermarks.
"""

# =============================================================================
# IMPORTS
# =============================================================================
# Standard library imports
import logging
import os
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import torch
import yaml
from audiotools import AudioSignal

# Local imports
from utils import apply_effect, LocalizationAugmentation, SequenceAugmentation
from utils.effect_scheduler import EffectScheduler
from scripts.evaluate import BER, MIOU

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handler if logger doesn't have one
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# =============================================================================
# CONSTANTS
# =============================================================================
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_WINDOW_DURATION = 0.1
DEFAULT_K_VALUE = 5
BER_THRESHOLD = 0.5
EFFECT_UPDATE_LOG_INTERVAL = 100

# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================
def load_effects_config() -> Tuple[
    List[Tuple[str, Dict[str, Any]]],
    List[Tuple[str, Dict[str, Any]]],
    Dict[str, Dict[str, Any]],
    Dict[str, float]
]:
    """
    Load effect configuration from YAML file with fallback to defaults.
    
    Attempts to load configuration from 'conf/effects_config.yml'.
    
    Returns:
        Tuple containing:
            - train_effects: List of (effect_name, params) tuples for training
            - eval_effects: List of (effect_name, params) tuples for evaluation  
            - effect_param_grid: Dictionary mapping effect names to parameter grids
            - scheduler_config: Configuration for the effect scheduler
            
    Raises:
        None - Falls back to default configuration on any error
    """
    # Construct configuration file path
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        'conf', 
        'effects_config.yml'
    )
    
    try:
        # Attempt to load configuration from YAML file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Successfully loaded effects config from {config_path}")
        
        # Convert train and eval effects to expected format
        # Each effect is stored as (name, params) tuple
        train_effects = [
            (effect['name'], effect['params']) 
            for effect in config['train_effects']
        ]
        
        eval_effects = [
            (effect['name'], effect['params'])
            for effect in config['eval_effects']
        ]
        
        # Convert list parameters to tuples for consistency
        # Many audio effects expect tuple parameters (e.g., frequency ranges)
        for effects_list in [train_effects, eval_effects]:
            for _, params in effects_list:
                for key, value in params.items():
                    if isinstance(value, list) and len(value) == 2:
                        params[key] = tuple(value)
        
        return (
            train_effects,
            eval_effects,
            config['effect_param_grid'],
            config['scheduler_config']
        )
        
    except Exception as e:
        logger.warning(f"Failed to load effects config: {str(e)}", exc_info=True)
        logger.info("Using default configuration")
        
        # Fallback to default configuration
        # These effects are commonly used for audio watermark robustness testing
        train_effects = [
            ('identity', {}),  # No transformation
            ('highpass_filter', {'cutoff_freq': 500}),
            ('highpass_filter', {'cutoff_freq': 3500}),
            ('lowpass_filter', {'cutoff_freq': 1000}),
            ('lowpass_filter', {'cutoff_freq': 2000}),
            ('bandpass_filter', {'cutoff_freq_low': 300, 'cutoff_freq_high': 4000}),
            ('speed', {'speed': 0.8}),  # 20% slowdown
            ('resample', {'new_sample_rate': 32000}),
            ('random_noise', {'noise_std': 0.001}),
        ]
        
        eval_effects = [
            ('identity', {}),
            ('resample', {'new_sample_rate': 32000}),
            ('speed', {'speed': 0.8}),
            ('random_noise', {'noise_std': 0.001}),
            ('lowpass_filter', {'cutoff_freq': 2000}),
            ('highpass_filter', {'cutoff_freq': 3500}),
            ('bandpass_filter', {'cutoff_freq_low': 300, 'cutoff_freq_high': 4000}),
        ]
        
        # Parameter grid for adaptive effect selection
        effect_param_grid = {
            'identity': {},
            'highpass_filter': {
                'cutoff_freq': {'choices': [500, 3500]}
            },
            'lowpass_filter': {
                'cutoff_freq': {'choices': [1000, 2000]}
            },
            'bandpass_filter': {
                'cutoff_freq_low': {'choices': [300]},
                'cutoff_freq_high': {'choices': [4000]}
            },
            'speed': {
                'speed': {'choices': [0.8]}
            },
            'resample': {
                'new_sample_rate': {'choices': [32000]}
            },
            'random_noise': {
                'noise_std': {'choices': [0.001]}
            }
        }
        
        # Scheduler configuration for adaptive effect selection
        scheduler_config = {
            'beta': 0.9,  # Exponential moving average decay
            'ber_threshold': 0.001,  # Target bit error rate
            'miou_threshold': 0.95  # Target mean IoU
        }
        
        return train_effects, eval_effects, effect_param_grid, scheduler_config

# =============================================================================
# LOAD CONFIGURATION AT MODULE LEVEL
# =============================================================================
TRAIN_EFFECTS, EVAL_EFFECTS, EFFECT_PARAM_GRID, SCHEDULER_CONFIG = load_effects_config()

# =============================================================================
# MAIN WATERMARKING CLASS
# =============================================================================
class AudioWatermarking(torch.nn.Module):
    """
    Neural audio watermarking system with adaptive robustness.
    
    This class implements a complete watermarking pipeline including:
    - Watermark generation using neural networks
    - Watermark detection from audio signals
    - Watermark localization within audio
    - Adaptive effect selection for robustness training
    
    Attributes:
        generator: Neural network for generating watermark signals
        detector: Neural network for detecting watermark bits
        locator: Neural network for localizing watermark presence
        sample_rate: Audio sample rate in Hz
        window_duration: Duration of processing windows in seconds
        k: Number of windows for watermark embedding
        ber_calculator: Bit error rate calculator
        miou_calculator: Mean IoU calculator for localization
        localization_augmenter: Augmentation for watermark localization
        seq_augmenter: Sequential augmentation for robustness
        effect_scheduler: Adaptive scheduler for effect selection
        effect_update_count: Counter for effect updates
    """
    
    def __init__(
        self,
        generator: torch.nn.Module,
        detector: torch.nn.Module,
        locator: torch.nn.Module,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        window_duration: float = DEFAULT_WINDOW_DURATION,
        k: int = DEFAULT_K_VALUE
    ) -> None:
        """
        Initialize the audio watermarking system.
        
        Args:
            generator: Neural network module for watermark generation
            detector: Neural network module for watermark detection
            locator: Neural network module for watermark localization
            sample_rate: Audio sample rate in Hz (default: 16000)
            window_duration: Duration of processing windows in seconds (default: 0.1)
            k: Number of windows for watermark embedding (default: 5)
            
        Raises:
            ValueError: If sample_rate <= 0 or window_duration <= 0 or k <= 0
        """
        super().__init__()
        
        # Validate inputs
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if window_duration <= 0:
            raise ValueError(f"window_duration must be positive, got {window_duration}")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        
        # Store neural network components
        self.generator = generator
        self.detector = detector
        self.locator = locator
        
        # Store audio processing parameters
        self.sample_rate = sample_rate
        self.window_duration = window_duration
        self.k = k
        
        # Initialize evaluation metrics
        self.ber_calculator = BER(threshold=BER_THRESHOLD)
        self.miou_calculator = MIOU()
        
        # Initialize augmentation modules
        self.localization_augmenter = LocalizationAugmentation(
            self.sample_rate, 
            self.window_duration
        )
        self.seq_augmenter = SequenceAugmentation(self.sample_rate)
        
        # Initialize adaptive effect scheduler
        self.effect_scheduler = EffectScheduler(
            effect_params=EFFECT_PARAM_GRID,
            beta=SCHEDULER_CONFIG.get('beta', 0.9),
            ber_threshold=SCHEDULER_CONFIG.get('ber_threshold', 0.001),
            miou_threshold=SCHEDULER_CONFIG.get('miou_threshold', 0.95)
        )
        
        # Track number of effect updates for logging
        self.effect_update_count = 0
        
        logger.info(
            f"Initialized AudioWatermarking with sample_rate={sample_rate}, "
            f"window_duration={window_duration}, k={k}"
        )

    def forward(
        self,
        signal: AudioSignal,
        msg: torch.Tensor,
        phase: str = 'train'
    ) -> Union[
        Tuple[AudioSignal, AudioSignal],  # For audio_sample phase
        Tuple[AudioSignal, AudioSignal, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],  # For train phase
        Tuple[AudioSignal, AudioSignal, Dict[str, Dict[str, torch.Tensor]]]  # For valid phase
    ]:
        """
        Forward pass of the watermarking system.
        
        Processes audio signals through the watermarking pipeline based on the
        specified phase (train, valid, or audio_sample).
        
        Args:
            signal: Input audio signal to watermark
            msg: Binary message tensor to embed (shape: [batch, msg_length])
            phase: Processing phase - 'train', 'valid', or 'audio_sample'
            
        Returns:
            Different outputs based on phase:
            - audio_sample: (reconstructed_signal, watermarked_signal)
            - train: (reconstructed_signal, watermarked_signal, detector_out, 
                     mask, updated_original, stats, locator_out)
            - valid: (reconstructed_signal, watermarked_signal, results_dict)
            
        Raises:
            RuntimeError: If forward pass fails during processing
            ValueError: If phase is not recognized
        """
        if phase == 'train':
            return self._forward_train(signal, msg)
        elif phase == 'audio_sample':
            return self._forward_audio_sample(signal, msg)
        elif phase == 'valid':
            return self._forward_valid(signal, msg)
        else:
            raise ValueError(f"Unknown phase: {phase}. Expected 'train', 'valid', or 'audio_sample'")

    def _forward_train(
        self,
        signal: AudioSignal,
        msg: torch.Tensor
    ) -> Tuple[AudioSignal, AudioSignal, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any], torch.Tensor]:
        """
        Training forward pass with augmentations and adaptive effects.
        
        Args:
            signal: Input audio signal
            msg: Binary message tensor
            
        Returns:
            Tuple containing multiple outputs for training
            
        Raises:
            RuntimeError: If processing fails
        """
        try:
            # Generate watermark and embed in signal
            reconstructed_signal = self.generator(signal, msg)
            watermarked_signal = reconstructed_signal + signal.audio_data

            # Apply augmentations
            watermarked_augmented, mask, updated_original, stats = self._apply_augmentations(
                signal.audio_data, 
                watermarked_signal.audio_data
            )

            # Apply adaptive effects
            watermarked_effect, mask, effects_applied = self._apply_adaptive_effects(
                watermarked_augmented, 
                mask, 
                msg.shape[0]
            )

            # Store applied effects in stats
            stats['selected_effects'] = effects_applied

            # Run detection and localization
            watermarked_signal_effect = AudioSignal(
                watermarked_effect.to(msg.device), 
                watermarked_augmented.sample_rate
            )
            
            locator_out = self.locator(watermarked_signal_effect)
            detector_out = self.detector(watermarked_signal_effect)

            # Ensure mask is on correct device
            mask = mask.to(msg.device)
            
            # Update effect scheduler with metrics
            self._update_effect_metrics(
                detector_out, 
                locator_out, 
                msg, 
                mask, 
                effects_applied
            )

            return (
                reconstructed_signal, 
                watermarked_signal, 
                detector_out, 
                mask, 
                updated_original, 
                stats, 
                locator_out
            )

        except Exception as e:
            logger.error(
                f"Error in watermarking forward pass (train phase): {str(e)}", 
                exc_info=True
            )
            logger.error(f"Batch info - Signal shape: {signal.audio_data.shape}")
            logger.error(f"Message shape: {msg.shape}")
            logger.error(f"Device: {msg.device}")
            
            raise RuntimeError(
                f"Watermarking forward pass failed in train phase: {str(e)}"
            ) from e

    def _forward_audio_sample(
        self,
        signal: AudioSignal,
        msg: torch.Tensor
    ) -> Tuple[AudioSignal, AudioSignal]:
        """
        Generate audio sample without augmentations.
        
        Args:
            signal: Input audio signal
            msg: Binary message tensor
            
        Returns:
            Tuple of (reconstructed_signal, watermarked_signal)
        """
        with torch.no_grad():
            reconstructed_signal = self.generator(signal, msg)
            watermarked_signal = reconstructed_signal + signal.audio_data
            return reconstructed_signal, watermarked_signal

    def _forward_valid(
        self,
        signal: AudioSignal,
        msg: torch.Tensor
    ) -> Tuple[AudioSignal, AudioSignal, Dict[str, Dict[str, Any]]]:
        """
        Validation forward pass with fixed evaluation effects.
        
        Args:
            signal: Input audio signal
            msg: Binary message tensor
            
        Returns:
            Tuple of (reconstructed_signal, watermarked_signal, results_dict)
        """
        with torch.no_grad():
            # Generate watermark
            reconstructed_signal = self.generator(signal, msg)
            watermarked_signal = reconstructed_signal + signal.audio_data

            # Apply augmentations
            watermarked_augmented, mask, updated_original, stats = self._apply_augmentations(
                signal.audio_data,
                watermarked_signal.audio_data
            )

            # Evaluate on each effect type
            results_dict = {}
            
            for effect_name, effect_params in EVAL_EFFECTS:
                # Apply effect and evaluate
                result = self._evaluate_single_effect(
                    watermarked_augmented,
                    mask,
                    msg,
                    effect_name,
                    effect_params
                )
                results_dict[effect_name] = result

            return reconstructed_signal, watermarked_signal, results_dict

    def _apply_augmentations(
        self,
        original: torch.Tensor,
        watermarked: torch.Tensor
    ) -> Tuple[AudioSignal, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Apply localization and sequence augmentations.
        
        Args:
            original: Original audio tensor
            watermarked: Watermarked audio tensor
            
        Returns:
            Tuple of (augmented_signal, mask, updated_original, stats)
        """
        # Apply localization augmentation
        watermarked_augmented, mask, updated_original, stats_loc = (
            self.localization_augmenter(original, watermarked)
        )

        # Apply sequence augmentation
        watermarked_augmented, updated_original, mask, stats_seq, _ = (
            self.seq_augmenter(
                updated_original, 
                watermarked_augmented.audio_data, 
                mask
            )
        )

        # Merge statistics
        stats = {**stats_loc, **stats_seq}

        return watermarked_augmented, mask, updated_original, stats

    def _apply_adaptive_effects(
        self,
        watermarked_signal: AudioSignal,
        mask: torch.Tensor,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[str, Dict[str, Any]]]]:
        """
        Apply adaptively selected effects to watermarked audio.
        
        Args:
            watermarked_signal: Watermarked audio signal
            mask: Binary mask tensor
            batch_size: Number of samples in batch
            
        Returns:
            Tuple of (processed_audio, updated_mask, effects_applied)
        """
        # Select effects using adaptive scheduler
        selected_effects = self.effect_scheduler.select_effects(batch_size)
        
        # Move to CPU for effect processing
        watermarked_signal_effect = watermarked_signal.audio_data.cpu()
        current_sample_rate = watermarked_signal.sample_rate
        
        if mask is not None:
            mask = mask.cpu()
        
        effects_applied = []
        
        # Group samples by effect type for efficient batch processing
        effect_groups = self._group_effects_by_type(selected_effects)
        
        # Process each effect group
        for effect_name, group_data in effect_groups.items():
            indices = group_data['indices']
            params_list = group_data['params']
            
            # Check if batch processing is possible
            can_batch = all(p == params_list[0] for p in params_list)
            
            if can_batch and len(indices) > 1:
                # Batch process samples with identical parameters
                watermarked_signal_effect, mask = self._process_effect_batch(
                    watermarked_signal_effect,
                    mask,
                    effect_name,
                    params_list[0],
                    indices,
                    current_sample_rate
                )
            else:
                # Process individually if parameters differ
                watermarked_signal_effect, mask = self._process_effect_individual(
                    watermarked_signal_effect,
                    mask,
                    effect_name,
                    params_list,
                    indices,
                    current_sample_rate
                )
            
            # Track applied effects
            for params in params_list:
                effects_applied.append((effect_name, params))

        return watermarked_signal_effect, mask, effects_applied

    def _group_effects_by_type(
        self,
        selected_effects: List[Tuple[str, Dict[str, Any]]]
    ) -> Dict[str, Dict[str, List[Any]]]:
        """
        Group effects by type for batch processing.
        
        Args:
            selected_effects: List of (effect_name, params) tuples
            
        Returns:
            Dictionary mapping effect names to indices and parameters
        """
        effect_groups = {}
        
        for i, (effect_name, effect_params) in enumerate(selected_effects):
            if effect_name not in effect_groups:
                effect_groups[effect_name] = {'indices': [], 'params': []}
            effect_groups[effect_name]['indices'].append(i)
            effect_groups[effect_name]['params'].append(effect_params)
            
        return effect_groups

    def _process_effect_batch(
        self,
        audio: torch.Tensor,
        mask: Optional[torch.Tensor],
        effect_name: str,
        params: Dict[str, Any],
        indices: List[int],
        sample_rate: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process multiple samples with the same effect in batch.
        
        Args:
            audio: Audio tensor
            mask: Optional mask tensor
            effect_name: Name of effect to apply
            params: Effect parameters
            indices: Sample indices to process
            sample_rate: Audio sample rate
            
        Returns:
            Tuple of (updated_audio, updated_mask)
        """
        # Extract batch for processing
        batch_audio = audio[indices]
        batch_mask = mask[indices] if mask is not None else None
        
        # Apply effect to batch
        processed_audio, new_mask = apply_effect(
            audio=batch_audio,
            effect_type=effect_name,
            sample_rate=sample_rate,
            mask=batch_mask,
            **params
        )
        
        # Update results in original tensors
        for idx, orig_idx in enumerate(indices):
            audio[orig_idx] = processed_audio[idx]
            if new_mask is not None and mask is not None:
                mask[orig_idx] = new_mask[idx]
                
        return audio, mask

    def _process_effect_individual(
        self,
        audio: torch.Tensor,
        mask: Optional[torch.Tensor],
        effect_name: str,
        params_list: List[Dict[str, Any]],
        indices: List[int],
        sample_rate: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process samples individually with different parameters.
        
        Args:
            audio: Audio tensor
            mask: Optional mask tensor
            effect_name: Name of effect to apply
            params_list: List of effect parameters
            indices: Sample indices to process
            sample_rate: Audio sample rate
            
        Returns:
            Tuple of (updated_audio, updated_mask)
        """
        for idx, params in zip(indices, params_list):
            # Extract single sample
            sample_audio = audio[idx:idx+1]
            sample_mask = mask[idx:idx+1] if mask is not None else None
            
            # Apply effect
            processed_audio, new_mask = apply_effect(
                audio=sample_audio,
                effect_type=effect_name,
                sample_rate=sample_rate,
                mask=sample_mask,
                **params
            )
            
            # Update results
            audio[idx] = processed_audio[0]
            if new_mask is not None and mask is not None:
                mask[idx] = new_mask[0]
                
        return audio, mask

    def _update_effect_metrics(
        self,
        detector_out: torch.Tensor,
        locator_out: torch.Tensor,
        msg: torch.Tensor,
        mask: torch.Tensor,
        effects_applied: List[Tuple[str, Dict[str, Any]]]
    ) -> None:
        """
        Update effect scheduler with performance metrics.
        
        Args:
            detector_out: Detector network output
            locator_out: Locator network output
            msg: Original message tensor
            mask: Binary mask tensor
            effects_applied: List of applied effects
        """
        with torch.no_grad():
            # Binarize locator output
            locator_out_binary = (locator_out > 0.5).float()
            
            # Compute per-sample metrics and update scheduler
            for i, (effect_name, effect_params) in enumerate(effects_applied):
                # Extract sample-specific outputs
                sample_detector = detector_out[i:i+1]
                sample_mask = mask[i:i+1]
                sample_locator_binary = locator_out_binary[i:i+1]
                sample_msg = msg[i:i+1]
                
                # Compute BER for this sample
                sample_ber = self.ber_calculator(
                    sample_detector, 
                    sample_msg, 
                    sample_mask
                )
                
                # Compute mIoU for this sample
                sample_miou = self.miou_calculator(
                    sample_locator_binary, 
                    sample_mask
                )
                
                # Update effect scheduler with metrics
                self.effect_scheduler.update_effect_metrics(
                    effect_name, 
                    effect_params, 
                    sample_ber.item(), 
                    sample_miou.item()
                )
                
                self.effect_update_count += 1
                
                # Log adaptive behavior periodically
                if self.effect_update_count % EFFECT_UPDATE_LOG_INTERVAL == 0:
                    self.effect_scheduler.log_adaptive_behavior()
                    logger.info(f"Total effect updates: {self.effect_update_count}")

    def _evaluate_single_effect(
        self,
        watermarked_signal: AudioSignal,
        mask: torch.Tensor,
        msg: torch.Tensor,
        effect_name: str,
        effect_params: Dict[str, Any]
    ) -> Dict[str, Union[torch.Tensor, float]]:
        """
        Evaluate watermark performance under a single effect.
        
        Args:
            watermarked_signal: Watermarked audio signal
            mask: Binary mask tensor
            msg: Original message tensor
            effect_name: Name of effect to apply
            effect_params: Effect parameters
            
        Returns:
            Dictionary containing evaluation results
        """
        # Apply effect to entire batch
        mask_cpu = mask.cpu() if mask is not None else None
        
        watermarked_signal_effect, new_mask = apply_effect(
            audio=watermarked_signal.audio_data.cpu(),
            effect_type=effect_name,
            sample_rate=watermarked_signal.sample_rate,
            mask=mask_cpu,
            **effect_params
        )
        
        # Create AudioSignal and run detection
        watermarked_signal_effect = AudioSignal(
            watermarked_signal_effect.to(msg.device),
            watermarked_signal.sample_rate
        )
        
        locator_out = self.locator(watermarked_signal_effect)
        detector_out = self.detector(watermarked_signal_effect)
        
        # Compute metrics
        locator_out_binary = (locator_out > 0.5).float()
        new_mask = new_mask.to(msg.device)
        
        ber = self.ber_calculator(detector_out, msg, new_mask)
        miou = self.miou_calculator(locator_out_binary, new_mask).item()
        
        return {
            'detector_output': detector_out,
            'locator_output': locator_out,
            'mask': new_mask,
            'ber': ber,
            'miou': miou
        }