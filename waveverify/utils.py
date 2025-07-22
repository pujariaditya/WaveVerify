"""
Utility functions for WaveVerify package.

Handles checkpoint downloading, audio loading, and other helper functions.
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import logging
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, Union

# Third-party imports
import torch
import torchaudio
from tqdm import tqdm

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Configure logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler if none exists
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

# Checkpoint metadata
CHECKPOINTS: Final[Dict[str, Dict[str, Any]]] = {
    "base": {
        # TODO: add checkpoint url      
        "url": "",
        "size": "150MB",
        "sha256": None  # Will be updated when checkpoint is released
    }
}

# Default audio parameters
DEFAULT_SAMPLE_RATE: Final[int] = 16000
DEFAULT_BITS: Final[int] = 16
AUDIO_CLAMP_MIN: Final[float] = -1.0
AUDIO_CLAMP_MAX: Final[float] = 1.0
DECISION_THRESHOLD: Final[float] = 0.5

# =============================================================================
# HELPER CLASSES
# =============================================================================

class DownloadProgressBar(tqdm):
    """Progress bar for file downloads.
    
    Attributes:
        total: Total size of the download
        n: Current progress
    """
    
    def update_to(self, b: int = 1, bsize: int = 1, tsize: Optional[int] = None) -> None:
        """Update progress bar with download status.
        
        Args:
            b: Number of blocks transferred so far
            bsize: Size of each block in bytes
            tsize: Total size of the file in bytes
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

# =============================================================================
# PUBLIC FUNCTIONS
# =============================================================================

def download_checkpoint(checkpoint_name: str = "base", 
                       force_download: bool = False) -> Path:
    """
    Download and extract model checkpoint.
    
    Args:
        checkpoint_name: Name of checkpoint to download (currently only "base")
        force_download: Force re-download even if exists
        
    Returns:
        Path to extracted checkpoint directory
        
    Raises:
        ValueError: If checkpoint_name is not recognized
        URLError: If download fails
        IOError: If extraction fails
    """
    logger.info(f"Attempting to download checkpoint: {checkpoint_name}")
    
    # Validate checkpoint name
    if checkpoint_name not in CHECKPOINTS:
        logger.error(f"Unknown checkpoint requested: {checkpoint_name}")
        raise ValueError(f"Unknown checkpoint: {checkpoint_name}")
    
    # Setup paths
    try:
        checkpoint_dir = Path(__file__).parent.parent / "checkpoint"
        checkpoint_dir.mkdir(exist_ok=True)
        logger.debug(f"Checkpoint directory: {checkpoint_dir}")
    except Exception as e:
        logger.error(f"Failed to create checkpoint directory: {str(e)}", exc_info=True)
        raise IOError(f"Cannot create checkpoint directory: {str(e)}")
    
    checkpoint_path = checkpoint_dir / f"waveverify_{checkpoint_name}"
    tar_path = checkpoint_dir / f"waveverify_{checkpoint_name}.tar.gz"
    
    # Check if already exists
    if checkpoint_path.exists() and not force_download:
        logger.info(f"Checkpoint already exists at {checkpoint_path}")
        return checkpoint_path
    
    # Download checkpoint
    checkpoint_info = CHECKPOINTS[checkpoint_name]
    logger.info(f"Downloading WaveVerify {checkpoint_name} checkpoint ({checkpoint_info['size']})...")
    
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=tar_path.name) as t:
            urllib.request.urlretrieve(checkpoint_info['url'], tar_path, reporthook=t.update_to)
        logger.info("Download completed successfully")
    except Exception as e:
        logger.error(f"Error downloading checkpoint: {str(e)}", exc_info=True)
        logger.error(f"Please download manually from: {checkpoint_info['url']}")
        logger.error(f"And place it at: {tar_path}")
        raise
    
    # Extract tar file
    logger.info("Extracting checkpoint...")
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            # Validate tar members for security
            for member in tar.getmembers():
                if member.name.startswith('/') or '..' in member.name:
                    logger.error(f"Unsafe tar member detected: {member.name}")
                    raise ValueError(f"Unsafe tar member: {member.name}")
            tar.extractall(checkpoint_dir)
        logger.info("Extraction completed successfully")
    except Exception as e:
        logger.error(f"Error extracting checkpoint: {str(e)}", exc_info=True)
        raise IOError(f"Cannot extract checkpoint: {str(e)}")
    
    # Clean up tar file
    try:
        tar_path.unlink()
        logger.debug("Cleaned up tar file")
    except Exception as e:
        logger.warning(f"Could not remove tar file: {str(e)}")
    
    logger.info(f"Checkpoint extracted to {checkpoint_path}")
    return checkpoint_path


def load_audio(audio_path: Union[str, Path], 
               target_sr: int = DEFAULT_SAMPLE_RATE) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (default: 16000)
        
    Returns:
        Tuple of (audio_tensor, sample_rate)
        
    Raises:
        FileNotFoundError: If audio file does not exist
        ValueError: If audio file is corrupt or invalid
        RuntimeError: If audio loading fails
    """
    logger.info(f"Loading audio from: {audio_path}")
    
    # Validate input path
    audio_path = Path(audio_path)
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    if not audio_path.is_file():
        logger.error(f"Path is not a file: {audio_path}")
        raise ValueError(f"Path is not a file: {audio_path}")
    
    # Load audio
    try:
        waveform, sample_rate = torchaudio.load(str(audio_path))
        logger.debug(f"Loaded audio with shape: {waveform.shape}, sample rate: {sample_rate}")
    except Exception as e:
        logger.error(f"Failed to load audio: {str(e)}", exc_info=True)
        raise RuntimeError(f"Cannot load audio file: {str(e)}")
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        logger.debug("Converting stereo to mono")
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if sample_rate != target_sr:
        logger.debug(f"Resampling from {sample_rate}Hz to {target_sr}Hz")
        try:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
            sample_rate = target_sr
        except Exception as e:
            logger.error(f"Resampling failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Cannot resample audio: {str(e)}")
    
    logger.info(f"Audio loaded successfully: shape={waveform.shape}, sr={sample_rate}")
    return waveform, sample_rate


def save_audio(audio: torch.Tensor, 
               path: Union[str, Path], 
               sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
    """
    Save audio tensor to file.
    
    Args:
        audio: Audio tensor to save
        path: Output file path
        sample_rate: Sample rate of audio
        
    Raises:
        ValueError: If audio tensor has invalid shape
        IOError: If file cannot be written
        RuntimeError: If audio saving fails
    """
    logger.info(f"Saving audio to: {path}")
    
    # Validate inputs
    if not isinstance(audio, torch.Tensor):
        logger.error(f"Invalid audio type: {type(audio)}")
        raise ValueError(f"Audio must be torch.Tensor, got {type(audio)}")
    
    if sample_rate <= 0:
        logger.error(f"Invalid sample rate: {sample_rate}")
        raise ValueError(f"Sample rate must be positive, got {sample_rate}")
    
    path = Path(path)
    
    # Create parent directory if needed
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {path.parent}")
    except Exception as e:
        logger.error(f"Cannot create directory: {str(e)}", exc_info=True)
        raise IOError(f"Cannot create directory: {str(e)}")
    
    # Ensure audio is 2D (channels, samples)
    original_shape = audio.shape
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
        logger.debug("Converted 1D audio to 2D")
    elif audio.dim() == 3:
        # Assume shape is (batch, channels, samples) - take first batch
        audio = audio.squeeze(0)
        logger.debug("Converted 3D audio to 2D")
    elif audio.dim() != 2:
        logger.error(f"Invalid audio shape: {original_shape}")
        raise ValueError(f"Audio must be 1D, 2D, or 3D tensor, got shape {original_shape}")
    
    # Clamp to valid range
    audio = torch.clamp(audio, AUDIO_CLAMP_MIN, AUDIO_CLAMP_MAX)
    logger.debug(f"Clamped audio to range [{AUDIO_CLAMP_MIN}, {AUDIO_CLAMP_MAX}]")
    
    # Save
    try:
        torchaudio.save(str(path), audio.cpu(), sample_rate)
        logger.info(f"Audio saved successfully: shape={audio.shape}, sr={sample_rate}")
    except Exception as e:
        logger.error(f"Failed to save audio: {str(e)}", exc_info=True)
        raise RuntimeError(f"Cannot save audio: {str(e)}")


def message_to_tensor(message: Union[str, List[int]], bits: int = DEFAULT_BITS) -> torch.Tensor:
    """
    Convert message to tensor format expected by model.
    
    Args:
        message: Binary string or list of ints (0 or 1)
        bits: Expected number of bits
        
    Returns:
        Tensor of shape (1, bits)
        
    Raises:
        ValueError: If message has wrong length or invalid format
        TypeError: If message type is not supported
    """
    logger.debug(f"Converting message to tensor: type={type(message)}, bits={bits}")
    
    # Validate bits parameter
    if bits <= 0:
        logger.error(f"Invalid bits value: {bits}")
        raise ValueError(f"Bits must be positive, got {bits}")
    
    # Convert message to list of ints
    try:
        if isinstance(message, str):
            # Validate binary string
            if not all(c in '01' for c in message):
                logger.error(f"Invalid binary string: contains non-binary characters")
                raise ValueError("Message string must contain only '0' and '1'")
            
            if len(message) != bits:
                logger.error(f"Message length mismatch: expected {bits}, got {len(message)}")
                raise ValueError(f"Message must be {bits} bits, got {len(message)}")
            
            message_list = [int(b) for b in message]
            
        elif isinstance(message, list):
            # Validate list of ints
            if not all(isinstance(x, int) and x in [0, 1] for x in message):
                logger.error("Invalid message list: contains non-binary values")
                raise ValueError("Message list must contain only 0 and 1")
            
            if len(message) != bits:
                logger.error(f"Message length mismatch: expected {bits}, got {len(message)}")
                raise ValueError(f"Message must be {bits} elements, got {len(message)}")
            
            message_list = message
            
        else:
            logger.error(f"Unsupported message type: {type(message)}")
            raise TypeError(f"Message must be str or list, got {type(message)}")
            
    except Exception as e:
        logger.error(f"Failed to convert message: {str(e)}", exc_info=True)
        raise
    
    # Convert to tensor
    try:
        tensor = torch.tensor(message_list, dtype=torch.float32).unsqueeze(0)
        logger.debug(f"Created tensor with shape: {tensor.shape}")
        return tensor
    except Exception as e:
        logger.error(f"Failed to create tensor: {str(e)}", exc_info=True)
        raise RuntimeError(f"Cannot create tensor: {str(e)}")


def tensor_to_message(tensor: torch.Tensor, threshold: float = DECISION_THRESHOLD) -> str:
    """
    Convert tensor back to binary message string.
    
    Args:
        tensor: Model output tensor
        threshold: Decision threshold for binary conversion
        
    Returns:
        Binary string message
        
    Raises:
        ValueError: If tensor has invalid shape or threshold is invalid
        TypeError: If inputs have wrong type
    """
    logger.debug(f"Converting tensor to message: shape={tensor.shape}, threshold={threshold}")
    
    # Validate inputs
    if not isinstance(tensor, torch.Tensor):
        logger.error(f"Invalid tensor type: {type(tensor)}")
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    
    if not 0 <= threshold <= 1:
        logger.error(f"Invalid threshold: {threshold}")
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
    
    # Handle different tensor shapes
    original_shape = tensor.shape
    
    if tensor.dim() == 3:  # (batch, bits, time)
        # Average over time dimension to get consensus
        logger.debug("Averaging 3D tensor over time dimension")
        tensor = tensor.mean(dim=2)
    
    if tensor.dim() == 2:  # (batch, bits)
        # Take first batch element
        logger.debug("Extracting first batch element from 2D tensor")
        tensor = tensor[0]
    
    if tensor.dim() != 1:
        logger.error(f"Invalid tensor shape after processing: {tensor.shape}")
        raise ValueError(f"Cannot process tensor with shape {original_shape}")
    
    # Convert to binary decisions
    try:
        binary = (tensor >= threshold).int()
        logger.debug(f"Applied threshold {threshold} to get binary values")
        
        # Convert to string
        message = ''.join(str(b.item()) for b in binary)
        logger.debug(f"Converted to binary string of length: {len(message)}")
        
        return message
        
    except Exception as e:
        logger.error(f"Failed to convert tensor to message: {str(e)}", exc_info=True)
        raise RuntimeError(f"Cannot convert tensor to message: {str(e)}")