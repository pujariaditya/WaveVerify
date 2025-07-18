"""
Core WaveVerify class for audio watermarking.

Provides high-level API for embedding, detecting, and locating watermarks in audio.
"""

# =============================================================================
# Standard Library Imports
# =============================================================================
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# =============================================================================
# Third-Party Imports
# =============================================================================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import argbind
from audiotools import AudioSignal

# =============================================================================
# Local Imports
# =============================================================================
# Add project root to path for local imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model import AudioWatermarking, Generator, Detector, Locator
from .utils import (
    download_checkpoint, 
    load_audio, 
    save_audio,
    message_to_tensor,
    tensor_to_message
)
from .config import initialize_models_with_config, get_model_config
from .watermark_id import WatermarkID

# =============================================================================
# Logger Configuration
# =============================================================================
logger = logging.getLogger(__name__)

# =============================================================================
# WaveVerify Main Class
# =============================================================================
class WaveVerify:
    """
    Main interface for WaveVerify audio watermarking.
    
    Provides methods for embedding, detecting, and locating watermarks in audio files.
    Supports both CPU and GPU operations with automatic device detection.
    """
    
    # Model configuration constants
    DEFAULT_SAMPLE_RATE: int = 16000
    DEFAULT_WATERMARK_BITS: int = 16
    
    def __init__(self, checkpoint: str = "base", device: str = "auto") -> None:
        """
        Initialize WaveVerify with pre-trained models.
        
        Args:
            checkpoint: Checkpoint name or path. If "base", downloads pre-trained model.
                       Can be a path to a custom checkpoint directory.
            device: Device to run on ("cuda", "cpu", or "auto" to detect).
                   Auto detection prefers CUDA if available.
                   
        Raises:
            FileNotFoundError: If checkpoint path does not exist.
            RuntimeError: If model loading fails.
        """
        try:
            # Setup device with automatic detection
            self.device = self._setup_device(device)
            logger.info(f"Using device: {self.device}")
            
            # Handle checkpoint retrieval
            checkpoint_path = self._resolve_checkpoint_path(checkpoint)
            
            # Load model with error handling
            self.model = self._load_model(checkpoint_path)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Model configuration parameters
            self.sample_rate = self.DEFAULT_SAMPLE_RATE
            self.watermark_bits = self.DEFAULT_WATERMARK_BITS
            
            logger.info("WaveVerify initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize WaveVerify: {str(e)}", exc_info=True)
            raise RuntimeError(f"WaveVerify initialization failed: {str(e)}") from e
    
    def _setup_device(self, device: str) -> torch.device:
        """
        Setup computation device based on user preference.
        
        Args:
            device: Device string ("cuda", "cpu", or "auto")
            
        Returns:
            torch.device: Configured device for computation
        """
        if device == "auto":
            # Automatic detection prefers CUDA if available
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(device)
    
    def _remove_parametrizations_from_model(self, model: nn.Module) -> None:
        """Remove parametrizations from a model for loading clean checkpoints.
        
        This function removes parametrizations from model modules to make them
        compatible with checkpoints that were saved with parametrizations removed.
        
        Args:
            model: Model to remove parametrizations from
        """
        removed_count = 0
        
        # Find all parametrized modules
        for name, module in model.named_modules():
            if hasattr(module, 'parametrizations') and hasattr(module.parametrizations, 'weight'):
                try:
                    # Remove parametrization
                    parametrize.remove_parametrizations(module, 'weight')
                    removed_count += 1
                    logger.debug(f"Removed parametrization from {name}")
                except Exception as e:
                    logger.warning(f"Failed to remove parametrization from {name}: {e}")
        
        if removed_count > 0:
            logger.info(f"Removed parametrizations from {removed_count} modules for checkpoint compatibility")
    
    def _find_atomic_checkpoint_file(self, checkpoint_path: Path) -> Path:
        """Find the atomic checkpoint file in the given path.
        
        Args:
            checkpoint_path: Path to checkpoint file or directory
            
        Returns:
            Path to the atomic checkpoint file
            
        Raises:
            FileNotFoundError: If no atomic checkpoint file is found
        """
        if checkpoint_path.is_file() and checkpoint_path.suffix == '.pth':
            return checkpoint_path
        else:
            # Look for atomic checkpoint files in directory
            atomic_files = list(checkpoint_path.glob('*.pth'))
            if not atomic_files:
                raise FileNotFoundError(f"No atomic checkpoint files found in {checkpoint_path}")
            
            # Try best.pth first, then latest.pth, then any other
            for preferred in ['best.pth', 'latest.pth']:
                for pth_file in atomic_files:
                    if pth_file.name == preferred:
                        return pth_file
            
            # Return first available
            return atomic_files[0]
    
    def _convert_spec_keys(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Pass through state dict without conversion.
        
        Since we're now saving with the correct configuration during training,
        spec parameters should match between checkpoint and model.
        
        Args:
            state_dict: State dict from checkpoint
            
        Returns:
            Same state dict without modification
        """
        return state_dict
    
    def _resolve_checkpoint_path(self, checkpoint: str) -> Path:
        """
        Resolve checkpoint path from name or path string.
        
        Args:
            checkpoint: Checkpoint identifier or path
            
        Returns:
            Path: Resolved checkpoint path
            
        Raises:
            FileNotFoundError: If checkpoint path does not exist
        """
        if checkpoint == "base":
            # Download base model if needed
            logger.info("Downloading base checkpoint if not present")
            checkpoint_path = download_checkpoint("base")
        else:
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        return checkpoint_path
    
    def _load_model(self, checkpoint_path: Path) -> AudioWatermarking:
        """
        Load AudioWatermarking model from checkpoint with automatic format detection.
        
        Supports both atomic checkpoint format (single .pth file) and legacy 
        multi-file format for backward compatibility.
        
        Args:
            checkpoint_path: Path to checkpoint directory or atomic checkpoint file
            
        Returns:
            AudioWatermarking: Loaded and configured model
            
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            # Detect checkpoint format first
            if self._is_atomic_checkpoint(checkpoint_path):
                logger.info("Detected atomic checkpoint format")
                # Try to load configuration from checkpoint
                atomic_file = self._find_atomic_checkpoint_file(checkpoint_path)
                checkpoint = torch.load(atomic_file, map_location='cpu')
                checkpoint_config = checkpoint.get('config', None)
                
                if checkpoint_config is not None:
                    logger.info(f"Using configuration from checkpoint ({len(checkpoint_config)} parameters)")
                    # Use the checkpoint configuration directly
                    config = checkpoint_config
                elif checkpoint_path.name == "base":
                    # Fallback to base configuration
                    config_path = Path(__file__).parent.parent / "conf" / "base.yml"
                    if config_path.exists():
                        logger.info(f"Loading configuration from: {config_path}")
                        config = initialize_models_with_config(config_path)
                    else:
                        logger.warning("Base configuration not found, using default")
                        config = initialize_models_with_config()
                else:
                    # Use default configuration
                    config = initialize_models_with_config()
            else:
                # Legacy checkpoint - use default or inferred configuration
                if checkpoint_path.name == "base":
                    config_path = Path(__file__).parent.parent / "conf" / "base.yml"
                    if config_path.exists():
                        logger.info(f"Loading configuration from: {config_path}")
                        config = initialize_models_with_config(config_path)
                    else:
                        logger.warning("Base configuration not found, using default")
                        config = initialize_models_with_config()
                else:
                    config = initialize_models_with_config()
            
            # Import model classes directly from model module  
            from model.generator import Generator as GeneratorBase
            from model.detector import Detector as DetectorBase
            from model.locator import Locator as LocatorBase
            
            # Bind model classes with argbind for configuration
            Generator = argbind.bind(GeneratorBase)
            Detector = argbind.bind(DetectorBase)
            Locator = argbind.bind(LocatorBase)
            
            # Initialize models with configuration scope
            with argbind.scope(config):
                generator = Generator()
                detector = Detector()
                locator = Locator()
            
            # Create AudioWatermarking composite model
            model = AudioWatermarking(generator, detector, locator)
            
            # Detect checkpoint format and load accordingly
            if self._is_atomic_checkpoint(checkpoint_path):
                logger.info("Detected atomic checkpoint format")
                self._load_atomic_checkpoint(model, checkpoint_path)
            else:
                logger.info("Detected legacy multi-file checkpoint format")
                self._load_legacy_checkpoint(model, checkpoint_path)
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise RuntimeError(f"Model loading failed: {str(e)}") from e
    
    def _is_atomic_checkpoint(self, checkpoint_path: Path) -> bool:
        """
        Detect if checkpoint is in atomic format (single .pth file) or legacy format.
        
        Args:
            checkpoint_path: Path to checkpoint directory or file
            
        Returns:
            True if atomic format, False if legacy format
        """
        # Check for atomic checkpoint patterns
        if checkpoint_path.is_file() and checkpoint_path.suffix == '.pth':
            return True
        
        # Check for potential atomic checkpoint files in directory
        if checkpoint_path.is_dir():
            atomic_files = list(checkpoint_path.glob('*.pth'))
            if atomic_files:
                # Check if any .pth file contains atomic structure
                for pth_file in atomic_files:
                    try:
                        checkpoint = torch.load(pth_file, map_location='cpu')
                        if isinstance(checkpoint, dict) and 'models' in checkpoint:
                            return True
                    except Exception:
                        continue
        
        return False
    
    def _load_atomic_checkpoint(self, model: AudioWatermarking, checkpoint_path: Path) -> None:
        """
        Load model from atomic checkpoint format (single .pth file).
        
        Args:
            model: AudioWatermarking model to load weights into
            checkpoint_path: Path to checkpoint file or directory containing atomic checkpoint
            
        Raises:
            RuntimeError: If atomic checkpoint loading fails
        """
        try:
            # Find the atomic checkpoint file
            if checkpoint_path.is_file() and checkpoint_path.suffix == '.pth':
                atomic_file = checkpoint_path
            else:
                # Look for atomic checkpoint files in directory
                atomic_files = list(checkpoint_path.glob('*.pth'))
                if not atomic_files:
                    raise FileNotFoundError(f"No atomic checkpoint files found in {checkpoint_path}")
                
                # Try to find a valid atomic checkpoint
                atomic_file = None
                for pth_file in atomic_files:
                    try:
                        checkpoint = torch.load(pth_file, map_location='cpu')
                        if isinstance(checkpoint, dict) and 'models' in checkpoint:
                            atomic_file = pth_file
                            break
                    except Exception:
                        continue
                
                if atomic_file is None:
                    raise FileNotFoundError(f"No valid atomic checkpoint found in {checkpoint_path}")
            
            logger.info(f"Loading atomic checkpoint from: {atomic_file}")
            
            # Load checkpoint data
            checkpoint = torch.load(atomic_file, map_location='cpu')
            
            # Validate checkpoint structure
            if not isinstance(checkpoint, dict) or 'models' not in checkpoint:
                raise ValueError("Invalid atomic checkpoint format - missing 'models' key")
            
            models = checkpoint['models']
            
            # Remove parametrizations from fresh models to match checkpoint format
            logger.info("Removing parametrizations from fresh models for checkpoint compatibility")
            self._remove_parametrizations_from_model(model.generator)
            self._remove_parametrizations_from_model(model.detector)
            self._remove_parametrizations_from_model(model.locator)
            
            # Load model components - use strict=False to handle configuration mismatches
            if 'generator' in models:
                generator_state = self._convert_spec_keys(models['generator'])
                missing_keys, unexpected_keys = model.generator.load_state_dict(generator_state, strict=False)
                logger.info("Loaded generator from atomic checkpoint")
                if missing_keys:
                    logger.warning(f"Generator missing keys: {len(missing_keys)} keys")
                if unexpected_keys:
                    logger.warning(f"Generator unexpected keys: {len(unexpected_keys)} keys")
            else:
                logger.warning("Generator not found in atomic checkpoint")
            
            if 'detector' in models:
                detector_state = self._convert_spec_keys(models['detector'])
                missing_keys, unexpected_keys = model.detector.load_state_dict(detector_state, strict=False)
                logger.info("Loaded detector from atomic checkpoint")
                if missing_keys:
                    logger.warning(f"Detector missing keys: {len(missing_keys)} keys")
                if unexpected_keys:
                    logger.warning(f"Detector unexpected keys: {len(unexpected_keys)} keys")
            else:
                logger.warning("Detector not found in atomic checkpoint")
            
            if 'locator' in models:
                locator_state = self._convert_spec_keys(models['locator'])
                missing_keys, unexpected_keys = model.locator.load_state_dict(locator_state, strict=False)
                logger.info("Loaded locator from atomic checkpoint")
                if missing_keys:
                    logger.warning(f"Locator missing keys: {len(missing_keys)} keys")
                if unexpected_keys:
                    logger.warning(f"Locator unexpected keys: {len(unexpected_keys)} keys")
            else:
                logger.warning("Locator not found in atomic checkpoint")
            
            # Log checkpoint metadata
            if 'step' in checkpoint:
                logger.info(f"Loaded checkpoint from training step: {checkpoint['step']}")
            
            # Check if checkpoint contains configuration
            config = checkpoint.get('config', None)
            if config is not None:
                logger.info(f"Found configuration in checkpoint with {len(config)} parameters")
            
            logger.info("Successfully loaded atomic checkpoint")
            
            # Return the configuration if present
            return config
            
        except Exception as e:
            logger.error(f"Failed to load atomic checkpoint: {str(e)}", exc_info=True)
            raise RuntimeError(f"Atomic checkpoint loading failed: {str(e)}") from e
    
    def _load_legacy_checkpoint(self, model: AudioWatermarking, checkpoint_path: Path) -> None:
        """
        Load model from legacy multi-file checkpoint format.
        
        Args:
            model: AudioWatermarking model to load weights into
            checkpoint_path: Path to checkpoint directory
            
        Raises:
            RuntimeError: If legacy checkpoint loading fails
        """
        try:
            # Component paths mapping
            components = {
                "generator": (model.generator, checkpoint_path / "generator" / "model.pth"),
                "detector": (model.detector, checkpoint_path / "detector" / "model.pth"),
                "locator": (model.locator, checkpoint_path / "locator" / "model.pth")
            }
            
            for component_name, (component_model, weight_path) in components.items():
                if weight_path.exists():
                    logger.info(f"Loading {component_name} from {weight_path}")
                    state_dict = torch.load(weight_path, map_location="cpu")
                    
                    # Load with strict=True to catch any issues
                    try:
                        component_model.load_state_dict(state_dict, strict=True)
                        logger.info(f"Successfully loaded {component_name} with strict loading")
                    except RuntimeError as e:
                        logger.error(f"Failed to load {component_name} with strict loading: {str(e)}")
                        raise RuntimeError(
                            f"Failed to load {component_name} due to checkpoint incompatibility. "
                            f"This indicates the checkpoint was trained with a different model architecture."
                        ) from e
                else:
                    logger.warning(f"{component_name.capitalize()} checkpoint not found at {weight_path}")
            
            logger.info("Successfully loaded legacy checkpoint")
            
        except Exception as e:
            logger.error(f"Failed to load legacy checkpoint: {str(e)}", exc_info=True)
            raise RuntimeError(f"Legacy checkpoint loading failed: {str(e)}") from e
    
    
    # =============================================================================
    # Public API Methods
    # =============================================================================
    
    def embed(self, 
              audio_path: Union[str, Path], 
              watermark_id: Union[WatermarkID, str, int],
              output_path: Optional[Union[str, Path]] = None) -> Tuple[np.ndarray, int, WatermarkID]:
        """
        Embed watermark identity into audio file.
        
        Args:
            audio_path: Path to input audio file. Supports common audio formats
                       (wav, mp3, flac, etc.)
            watermark_id: WatermarkID object or value that can be converted to one.
                         Can be a string, int (0-65535), or WatermarkID instance.
            output_path: Optional path to save watermarked audio. If not provided,
                        only returns the watermarked audio array.
            
        Returns:
            Tuple containing:
                - watermarked_audio (np.ndarray): Watermarked audio samples
                - sample_rate (int): Audio sample rate
                - watermark_id (WatermarkID): The embedded watermark identity
            
        Raises:
            ValueError: If watermark_id cannot be converted to valid WatermarkID
            FileNotFoundError: If audio file not found
            RuntimeError: If embedding process fails
        """
        try:
            # Validate and convert watermark ID
            watermark_id = self._validate_watermark_id(watermark_id)
            logger.info(f"Embedding watermark: {watermark_id}")
            
            # Load and prepare audio
            audio_tensor, sample_rate = load_audio(audio_path, self.sample_rate)
            audio_tensor = audio_tensor.to(self.device)
            
            # Get binary representation from WatermarkID
            message_bits = watermark_id.to_bits()
            
            # Convert message to tensor format expected by model
            message_tensor = message_to_tensor(message_bits, self.watermark_bits).to(self.device)
            
            # Create AudioSignal object for model input
            audio_signal = AudioSignal(audio_tensor.unsqueeze(0), self.sample_rate)
            audio_signal = audio_signal.to(self.device)
            
            # Embed watermark using model inference
            with torch.no_grad():
                # Use audio_sample phase for inference mode
                # Returns tuple: (reconstructed_signal, watermarked_signal)
                outputs = self.model(audio_signal, message_tensor, phase='audio_sample')
                _, watermarked_signal = outputs  # Get the watermarked signal
            
            # Extract audio data from signal wrapper
            watermarked_tensor = watermarked_signal.audio_data.squeeze(0)
            
            # Save output if path provided
            if output_path:
                save_audio(watermarked_tensor, output_path, self.sample_rate)
                logger.info(f"Watermarked audio saved to: {output_path}")
            
            # Convert to numpy for return value
            watermarked_np = watermarked_tensor.cpu().numpy().squeeze()
            
            return watermarked_np, self.sample_rate, watermark_id
            
        except Exception as e:
            logger.error(f"Embedding failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to embed watermark: {str(e)}") from e
    
    def detect(self, audio_path: Union[str, Path]) -> Tuple[WatermarkID, float]:
        """
        Detect watermark identity in audio file.
        
        Args:
            audio_path: Path to audio file to analyze. Supports common audio
                       formats (wav, mp3, flac, etc.)
            
        Returns:
            Tuple containing:
                - watermark_id (WatermarkID): Detected watermark identity
                - confidence (float): Detection confidence score (0.0 to 1.0)
        
        Raises:
            FileNotFoundError: If audio file not found
            RuntimeError: If detection process fails
        """
        try:
            # Load and prepare audio
            audio_tensor, sample_rate = load_audio(audio_path, self.sample_rate)
            audio_tensor = audio_tensor.to(self.device)
            
            # Create AudioSignal object for model input
            audio_signal = AudioSignal(audio_tensor.unsqueeze(0), self.sample_rate)
            audio_signal = audio_signal.to(self.device)
            
            # Detect watermark using model
            with torch.no_grad():
                # Run detector network
                detector_output = self.model.detector(audio_signal)
                
                # Convert logits to probabilities using sigmoid
                bit_probabilities = torch.sigmoid(detector_output)
                
                # Average confidence across time dimension for stability
                avg_probabilities = bit_probabilities.mean(dim=2)  # Shape: (1, bits)
                
                # Overall confidence is mean of individual bit confidences
                confidence_score = avg_probabilities.mean().item()
                
                # Convert probabilities to message bits
                detected_message = tensor_to_message(bit_probabilities)
            
            # Create WatermarkID object from detected bits
            detected_id = WatermarkID.custom(detected_message)
            
            logger.info(f"Detected watermark: {detected_id} (confidence: {confidence_score:.2%})")
            
            return detected_id, confidence_score
            
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to detect watermark: {str(e)}") from e
    
    def locate(self, audio_path: Union[str, Path]) -> np.ndarray:
        """
        Locate watermark regions in audio file.
        
        Produces a frame-level mask indicating which parts of the audio
        contain watermark information.
        
        Args:
            audio_path: Path to audio file to analyze
            
        Returns:
            np.ndarray: Localization mask as 1D array matching audio length.
                       Values range from 0.0 (no watermark) to 1.0 (watermark present).
        
        Raises:
            FileNotFoundError: If audio file not found
            RuntimeError: If localization process fails
        """
        try:
            # Load and prepare audio
            audio_tensor, sample_rate = load_audio(audio_path, self.sample_rate)
            audio_tensor = audio_tensor.to(self.device)
            original_length = audio_tensor.shape[-1]
            
            # Create AudioSignal object for model input
            audio_signal = AudioSignal(audio_tensor.unsqueeze(0), self.sample_rate)
            audio_signal = audio_signal.to(self.device)
            
            # Run locator network
            with torch.no_grad():
                locator_output = self.model.locator(audio_signal)
                
                # Convert logits to probabilities
                localization_probs = torch.sigmoid(locator_output)
                
                # Remove batch and channel dimensions
                localization_mask = localization_probs.squeeze()
                
                # Interpolate mask to match original audio length if needed
                if localization_mask.shape[0] != original_length:
                    localization_mask = nn.functional.interpolate(
                        localization_mask.unsqueeze(0).unsqueeze(0),
                        size=original_length,
                        mode='linear',
                        align_corners=False
                    ).squeeze()
            
            # Convert to numpy array
            mask_np = localization_mask.cpu().numpy()
            
            logger.info(f"Localization complete. Average watermark presence: {mask_np.mean():.2%}")
            
            return mask_np
            
        except Exception as e:
            logger.error(f"Localization failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to locate watermark: {str(e)}") from e
    
    def verify(self, 
               audio_path: Union[str, Path], 
               expected_watermark: Union[WatermarkID, str, int]) -> bool:
        """
        Verify if audio contains a specific watermark.
        
        Convenience method that combines detection and comparison.
        
        Args:
            audio_path: Path to audio file to verify
            expected_watermark: Expected WatermarkID or value that can be 
                              converted to one
            
        Returns:
            bool: True if detected watermark matches expected watermark,
                 False otherwise
        
        Raises:
            ValueError: If expected_watermark is invalid
            FileNotFoundError: If audio file not found
            RuntimeError: If verification process fails
        """
        try:
            # Validate expected watermark
            expected_watermark = self._validate_watermark_id(expected_watermark)
            
            # Detect watermark in audio
            detected_watermark, confidence = self.detect(audio_path)
            
            # Compare watermarks
            matches = detected_watermark == expected_watermark
            
            # Log verification result
            if matches:
                logger.info(f"✓ Watermark verified (confidence: {confidence:.2%})")
            else:
                logger.warning(f"✗ Watermark mismatch (confidence: {confidence:.2%})")
                logger.warning(f"  Expected: {expected_watermark}")
                logger.warning(f"  Detected: {detected_watermark}")
            
            return matches
            
        except Exception as e:
            logger.error(f"Verification failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to verify watermark: {str(e)}") from e
    
    # =============================================================================
    # Helper Methods
    # =============================================================================
    
    def _validate_watermark_id(self, watermark_id: Union[WatermarkID, str, int]) -> WatermarkID:
        """
        Validate and convert input to WatermarkID object.
        
        Args:
            watermark_id: Input watermark identifier
            
        Returns:
            WatermarkID: Validated watermark ID object
            
        Raises:
            ValueError: If watermark_id cannot be converted
        """
        if not isinstance(watermark_id, WatermarkID):
            try:
                watermark_id = WatermarkID.custom(watermark_id)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Invalid watermark_id: {e}. "
                    f"Use WatermarkID.for_creator(), .for_timestamp(), etc. "
                    f"or provide a 16-bit binary string, int (0-65535), or 2 bytes."
                )
        return watermark_id