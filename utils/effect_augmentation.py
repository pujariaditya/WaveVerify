# =============================================================================
# Audio Effect Augmentation Module
# =============================================================================
"""
Audio effect augmentation module with gradient support through Straight-Through Estimators.

This module provides various audio effects that can be applied during training
with proper gradient flow, including compression, filtering, noise addition,
and other transformations commonly used in audio watermarking research.

Architecture Overview:
    - AudioProcessor: Utility class for audio/mask length adjustment
    - STE Classes: Implement Straight-Through Estimators for non-differentiable operations
    - AudioEffects: Collection of audio transformation methods
    - apply_effect: Main entry point for applying effects with shape normalization

Key Features:
    - Gradient-aware audio transformations using STEs
    - Batch processing support with proper dimension handling
    - Mask-aware processing for watermark preservation
    - Comprehensive error handling and recovery
    - Device-agnostic operations

Usage Example:
    >>> import torch
    >>> from effect_augmentation import apply_effect
    >>> 
    >>> # Create sample audio and mask
    >>> audio = torch.randn(4, 1, 16000)  # [batch, channels, samples]
    >>> mask = torch.ones_like(audio)
    >>> 
    >>> # Apply echo effect
    >>> processed_audio, processed_mask = apply_effect(
    ...     audio, 'echo', sample_rate=16000, mask=mask,
    ...     volume_range=(0.1, 0.3), duration_range=(0.1, 0.2)
    ... )
    >>> 
    >>> # Apply compression
    >>> compressed_audio, _ = apply_effect(
    ...     audio, 'mp3_lossy_compression', sample_rate=16000,
    ...     bitrate_choice=['128k']
    ... )

Performance Considerations:
    - SoX and FFmpeg operations run on CPU (automatic device transfer)
    - Encodec models are cached for repeated use
    - Batch processing is parallelized where possible
    - Memory usage scales with batch size and audio length

Common Pitfalls:
    - Ensure audio and mask tensors are on the same device
    - Sample rate must match the audio content
    - Some effects may change audio length (handled automatically)
    - Binary masks should contain only 0.0 and 1.0 values
"""

# =============================================================================
# Standard Library Imports
# =============================================================================
import itertools
import logging
import random
import subprocess
import tempfile
from collections import defaultdict
from inspect import signature, isfunction
from typing import Any, Dict, List, Optional, Tuple, Union

# =============================================================================
# Third-Party Imports
# =============================================================================
import julius
import numpy as np
import torch
import torchaudio
from torch.autograd import Function

# =============================================================================
# Module Configuration
# =============================================================================
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================
# Default audio processing parameters
DEFAULT_SAMPLE_RATE: int = 16000  # Standard sample rate for speech processing (Hz)
DEFAULT_CHANNELS: int = 1  # Mono audio default (stereo would be 2)
MIN_AUDIO_LENGTH: int = 2  # Minimum samples to prevent division/processing errors
EPSILON: float = 1e-5  # Small value for numerical stability in division/normalization

# Audio format constants
# These delays are inherent to the encoding process and must be compensated
MP3_ENCODER_DELAY: int = 1152  # LAME MP3 encoder introduces 1152 sample delay (MPEG standard)
AAC_ENCODER_DELAY: int = 1024  # AAC encoder introduces 1024 sample delay (AAC-LC profile)

# Effect parameter ranges for randomization during training
# These ranges are empirically chosen to provide realistic augmentation
NOISE_STD_RANGE: Tuple[float, float] = (0.0001, 0.1)  # Min/max noise standard deviation
VOLUME_RANGE: Tuple[float, float] = (0.1, 0.5)  # Echo volume range (10%-50% of original)
DURATION_RANGE: Tuple[float, float] = (0.1, 0.5)  # Echo delay range in seconds
WINDOW_SIZE_RANGE: Tuple[int, int] = (2, 10)  # Smoothing filter window size range

# =============================================================================
# AudioProcessor Class
# =============================================================================
class AudioProcessor:
    """
    Utility class for handling audio and mask trimming and padding after applying audio effects.
    
    This class provides static methods for adjusting audio tensor and mask lengths
    to ensure consistency after audio transformations.
    """

    @staticmethod
    def adjust_audio_length(
        tensor: torch.Tensor,
        target_length: int,
        mode: str = 'pad_truncate'
    ) -> torch.Tensor:
        """
        Adjust audio tensor length to match the target length.
        
        This method handles length mismatches that occur after audio processing,
        ensuring consistent tensor dimensions for batch processing.
        
        Args:
            tensor (torch.Tensor): Input audio tensor of shape (..., time).
                Can be any dimensionality as long as time is the last dimension.
            target_length (int): Desired length of the output tensor.
                Must be a positive integer.
            mode (str, optional): Adjustment mode. Defaults to 'pad_truncate'.
                - 'pad_truncate': Pad with zeros or truncate to match target length
                - 'stretch': Stretch or compress using linear interpolation
                - 'nearest': Use nearest neighbor interpolation
        
        Returns:
            torch.Tensor: Adjusted audio tensor with shape (..., target_length).
                Preserves all dimensions except the last (time) dimension.
            
        Raises:
            ValueError: If target_length is not positive or mode is not recognized.
            RuntimeError: If tensor operations fail due to memory or device issues.
            
        Examples:
            >>> audio = torch.randn(4, 1, 1000)  # [batch, channels, time]
            >>> adjusted = AudioProcessor.adjust_audio_length(audio, 1500, 'pad_truncate')
            >>> assert adjusted.shape == (4, 1, 1500)
        """
        # Input validation with descriptive error messages
        if not isinstance(target_length, int) or target_length <= 0:
            raise ValueError(
                f"Target length must be a positive integer, got {target_length} "
                f"of type {type(target_length).__name__}"
            )
        
        if mode not in ['pad_truncate', 'stretch', 'nearest']:
            raise ValueError(
                f"Unknown mode: '{mode}'. Valid modes are: 'pad_truncate', 'stretch', 'nearest'"
            )
        
        current_length = tensor.shape[-1]
        
        # Return unchanged if already at target length
        if current_length == target_length:
            return tensor
            
        try:
            if mode == 'pad_truncate':
                if current_length > target_length:
                    # Truncate from the end to preserve the beginning of the signal
                    logger.debug(f"Truncating audio from {current_length} to {target_length} samples")
                    return tensor[..., :target_length]
                else:
                    # Pad with zeros at the end to maintain causality
                    pad_length = target_length - current_length
                    logger.debug(f"Padding audio with {pad_length} zeros")
                    padding = torch.zeros(
                        *tensor.shape[:-1], pad_length, 
                        device=tensor.device, 
                        dtype=tensor.dtype
                    )
                    return torch.cat([tensor, padding], dim=-1)
                    
            elif mode in ['stretch', 'nearest']:
                # Use interpolation to adjust length
                # Linear interpolation for smooth transitions, nearest for discrete values
                interpolation_mode = 'linear' if mode == 'stretch' else 'nearest'
                
                # Ensure tensor has at least 3 dimensions for interpolate
                original_dim = tensor.dim()
                tensor_reshaped = tensor.unsqueeze(0) if tensor.dim() == 1 else tensor
                
                logger.debug(
                    f"Interpolating audio from {current_length} to {target_length} "
                    f"samples using {interpolation_mode} mode"
                )
                
                interpolated = torch.nn.functional.interpolate(
                    tensor_reshaped,
                    size=target_length,
                    mode=interpolation_mode,
                    align_corners=False if interpolation_mode == 'linear' else None
                )
                
                # Restore original dimensionality
                return interpolated.squeeze(0) if original_dim == 1 else interpolated
                
        except ValueError:
            # Re-raise ValueError with original message
            raise
        except RuntimeError as e:
            # Handle PyTorch runtime errors (e.g., memory, device issues)
            logger.error(
                f"Runtime error adjusting audio length from {current_length} to {target_length}: {str(e)}",
                exc_info=True
            )
            raise RuntimeError(
                f"Failed to adjust audio length due to runtime error: {str(e)}"
            ) from e
        except Exception as e:
            # Catch any other unexpected errors
            logger.error(
                f"Unexpected error adjusting audio length: {type(e).__name__}: {str(e)}",
                exc_info=True
            )
            raise RuntimeError(
                f"Unexpected error in adjust_audio_length: {type(e).__name__}: {str(e)}"
            ) from e

    @staticmethod
    def adjust_mask_length(
        mask: torch.Tensor,
        target_length: int,
        mode: str = 'pad_truncate'
    ) -> torch.Tensor:
        """
        Adjust mask length to match the target length while preserving binary values.
        
        Masks typically indicate presence (1.0) or absence (0.0) of watermarks
        and must remain binary after adjustment.
        
        Args:
            mask (torch.Tensor): Input mask tensor of shape (..., time).
                Expected to contain binary values (0.0 or 1.0).
            target_length (int): Desired length of the output mask.
                Must be a positive integer.
            mode (str, optional): Adjustment mode. Defaults to 'pad_truncate'.
                - 'pad_truncate': Pad with zeros or truncate
                - 'stretch': Stretch using linear interpolation with thresholding
                - 'nearest-exact': Use nearest neighbor interpolation
        
        Returns:
            torch.Tensor: Adjusted mask tensor with shape (..., target_length).
                Maintains binary values (0.0 or 1.0) after adjustment.
            
        Raises:
            ValueError: If target_length is not positive or mode is not recognized.
            RuntimeError: If tensor operations fail.
            
        Examples:
            >>> mask = torch.ones(4, 1, 1000)  # [batch, channels, time]
            >>> adjusted = AudioProcessor.adjust_mask_length(mask, 1500)
            >>> assert adjusted.shape == (4, 1, 1500)
            >>> assert torch.all((adjusted == 0) | (adjusted == 1))  # Binary check
        """
        # Input validation with descriptive error messages
        if not isinstance(target_length, int) or target_length <= 0:
            raise ValueError(
                f"Target length must be a positive integer, got {target_length} "
                f"of type {type(target_length).__name__}"
            )
            
        if mode not in ['pad_truncate', 'stretch', 'nearest-exact']:
            raise ValueError(
                f"Unknown mode: '{mode}'. Valid modes are: 'pad_truncate', 'stretch', 'nearest-exact'"
            )
            
        current_length = mask.shape[-1]
        
        # Return unchanged if already at target length
        if current_length == target_length:
            return mask
            
        try:
            if mode == 'pad_truncate':
                if current_length > target_length:
                    # Truncate from the end to preserve watermark at the beginning
                    logger.debug(f"Truncating mask from {current_length} to {target_length} samples")
                    return mask[..., :target_length]
                else:
                    # Pad with zeros (no watermark in padded region)
                    pad_length = target_length - current_length
                    logger.debug(f"Padding mask with {pad_length} zeros")
                    padding = torch.zeros(
                        *mask.shape[:-1], pad_length, 
                        device=mask.device, 
                        dtype=mask.dtype
                    )
                    return torch.cat([mask, padding], dim=-1)
                    
            elif mode == 'stretch':
                # Linear interpolation with thresholding to maintain binary mask
                mask_float = mask.float()
                original_dim = mask.dim()
                mask_reshaped = mask_float.unsqueeze(0) if mask_float.dim() == 1 else mask_float
                
                logger.debug(
                    f"Stretching mask from {current_length} to {target_length} samples "
                    f"with threshold at 0.5"
                )
                
                interpolated = torch.nn.functional.interpolate(
                    mask_reshaped,
                    size=target_length,
                    mode='linear',
                    align_corners=False
                )
                
                # Threshold at 0.5 to maintain binary values
                # Values > 0.5 become 1.0, others become 0.0
                thresholded = (interpolated > 0.5).float()
                return thresholded.squeeze(0) if original_dim == 1 else thresholded
                
            elif mode == 'nearest-exact':
                # Nearest neighbor interpolation preserves exact values
                mask_float = mask.float()
                original_dim = mask.dim()
                mask_reshaped = mask_float.unsqueeze(0) if mask_float.dim() == 1 else mask_float
                
                logger.debug(
                    f"Resizing mask from {current_length} to {target_length} samples "
                    f"using nearest neighbor"
                )
                
                interpolated = torch.nn.functional.interpolate(
                    mask_reshaped,
                    size=target_length,
                    mode='nearest-exact'
                )
                
                # Nearest neighbor preserves original values, no thresholding needed
                return interpolated.squeeze(0) if original_dim == 1 else interpolated
                
        except ValueError:
            # Re-raise ValueError with original message
            raise
        except RuntimeError as e:
            # Handle PyTorch runtime errors
            logger.error(
                f"Runtime error adjusting mask length from {current_length} to {target_length}: {str(e)}",
                exc_info=True
            )
            raise RuntimeError(
                f"Failed to adjust mask length due to runtime error: {str(e)}"
            ) from e
        except Exception as e:
            # Catch any other unexpected errors
            logger.error(
                f"Unexpected error adjusting mask length: {type(e).__name__}: {str(e)}",
                exc_info=True
            )
            raise RuntimeError(
                f"Unexpected error in adjust_mask_length: {type(e).__name__}: {str(e)}"
            ) from e

    @staticmethod
    def adjust_lengths(
        tensor: torch.Tensor,
        mask: Optional[torch.Tensor],
        target_length: int,
        audio_mode: str = 'pad_truncate',
        mask_mode: str = 'nearest-exact'
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Adjust lengths of audio tensor and mask to match the target length.
        
        Convenience method that handles both audio and mask adjustment in a single call,
        ensuring consistent lengths for batch processing.
        
        Args:
            tensor (torch.Tensor): Input audio tensor of shape (..., time).
            mask (Optional[torch.Tensor]): Optional mask tensor with same shape as audio.
                If None, only audio is adjusted.
            target_length (int): Desired output length for both tensors.
                Must be a positive integer.
            audio_mode (str, optional): Mode for adjusting audio length.
                Defaults to 'pad_truncate'. See adjust_audio_length for options.
            mask_mode (str, optional): Mode for adjusting mask length.
                Defaults to 'nearest-exact'. See adjust_mask_length for options.
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: 
                - Adjusted audio tensor with shape (..., target_length)
                - Adjusted mask tensor with same shape, or None if no mask provided
                
        Raises:
            ValueError: If target_length is invalid or modes are not recognized.
            RuntimeError: If adjustment operations fail.
            
        Examples:
            >>> audio = torch.randn(4, 1, 1000)
            >>> mask = torch.ones_like(audio)
            >>> adj_audio, adj_mask = AudioProcessor.adjust_lengths(
            ...     audio, mask, 1500, audio_mode='stretch'
            ... )
            >>> assert adj_audio.shape == adj_mask.shape == (4, 1, 1500)
        """
        try:
            # Validate inputs
            if not isinstance(target_length, int) or target_length <= 0:
                raise ValueError(
                    f"Target length must be a positive integer, got {target_length}"
                )
                
            # Log adjustment operation
            logger.debug(
                f"Adjusting lengths to {target_length} - "
                f"Audio mode: {audio_mode}, Mask mode: {mask_mode}"
            )
            
            # Adjust audio length
            adjusted_tensor = AudioProcessor.adjust_audio_length(
                tensor, target_length, mode=audio_mode
            )
            
            # Adjust mask length if provided
            adjusted_mask = None
            if mask is not None:
                # Ensure mask has same device as audio
                if mask.device != tensor.device:
                    logger.warning(
                        f"Mask device ({mask.device}) differs from audio device ({tensor.device}). "
                        f"Moving mask to audio device."
                    )
                    mask = mask.to(tensor.device)
                    
                adjusted_mask = AudioProcessor.adjust_mask_length(
                    mask, target_length, mode=mask_mode
                )
                
            return adjusted_tensor, adjusted_mask
            
        except (ValueError, RuntimeError):
            # Re-raise known exceptions
            raise
        except Exception as e:
            # Handle unexpected errors
            logger.error(
                f"Unexpected error adjusting lengths: {type(e).__name__}: {str(e)}",
                exc_info=True
            )
            raise RuntimeError(
                f"Failed to adjust lengths: {type(e).__name__}: {str(e)}"
            ) from e

# =============================================================================
# Straight-Through Estimator (STE) Base Classes
# =============================================================================
class _BaseEffectSTE(Function):
    """
    Base class for audio effect STEs with common utilities.
    
    This class provides the foundation for implementing differentiable audio effects
    using the straight-through estimator pattern. STEs allow gradient flow through
    non-differentiable operations by using the identity function in the backward pass.
    
    The STE pattern is crucial for training neural networks with non-differentiable
    audio effects like compression, quantization, or external processing tools.
    
    Reference:
        Bengio et al., "Estimating or Propagating Gradients Through Stochastic
        Neurons for Conditional Computation" (2013)
    """

    @staticmethod
    def _adjust_tensor_length(tensor: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Adjust tensor length through padding or trimming.
        
        Utility method for maintaining consistent tensor dimensions after
        effects that may change audio length.
        
        Args:
            tensor (torch.Tensor): Input tensor to adjust.
            target_length (int): Target length for the tensor.
            
        Returns:
            torch.Tensor: Adjusted tensor with shape (..., target_length).
            
        Note:
            Uses pad_truncate mode for simplicity and efficiency.
        """
        return AudioProcessor.adjust_audio_length(tensor, target_length, mode='pad_truncate')

# =============================================================================
# SoX Effects STE
# =============================================================================
class _SoxEffectSTE(_BaseEffectSTE):
    """
    STE for SoX-based audio effects.
    
    This class implements a straight-through estimator for SoX audio effects,
    allowing gradient flow through non-differentiable SoX operations.
    
    SoX (Sound eXchange) is a command-line audio processing tool that provides
    many professional audio effects. This STE wrapper makes these effects
    available for use in differentiable pipelines.
    
    Supported effects include:
        - Speed changes
        - Equalization
        - Various filters and transformations
    
    Note:
        SoX operations run on CPU, so tensors are automatically transferred
        and returned to the original device.
    """

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, effects: List[List[str]], sample_rate: int, mode: str) -> torch.Tensor:
        """
        Forward pass for SoX effects.
        
        Args:
            ctx (Any): PyTorch context for saving tensors for backward pass.
            tensor (torch.Tensor): Input audio tensor of shape (batch, channels, time).
                Will be automatically moved to CPU for processing.
            effects (List[List[str]]): List of SoX effect specifications.
                Each effect is a list of strings, e.g., [['speed', '0.9'], ['rate', '16000']].
            sample_rate (int): Audio sample rate in Hz.
            mode (str): Length adjustment mode after effect application.
                Options: 'pad_truncate', 'stretch', 'nearest'.
            
        Returns:
            torch.Tensor: Processed audio tensor with original shape preserved.
                Returned on the same device as the input tensor.
                
        Raises:
            RuntimeError: If SoX processing fails for any batch item.
            
        Note:
            Failed effects on individual batch items will log a warning and
            return the original audio for that item (fail-safe behavior).
        """
        # Save context for backward pass
        ctx.save_for_backward(tensor)
        ctx.original_length = tensor.shape[-1]
        ctx.mode = mode
        original_device = tensor.device

        batch_size = tensor.size(0)
        augmented_tensors = []

        # Move tensor to CPU for SoX processing (required by torchaudio)
        tensor_cpu = tensor.cpu()
        
        logger.debug(f"Applying SoX effects to batch of size {batch_size}: {effects}")

        for i in range(batch_size):
            try:
                # Apply SoX effects to individual batch item
                augmented_tensor, output_sr = torchaudio.sox_effects.apply_effects_tensor(
                    tensor_cpu[i], sample_rate, effects
                )
                
                # Verify sample rate hasn't changed unexpectedly
                if output_sr != sample_rate:
                    logger.warning(
                        f"SoX changed sample rate from {sample_rate} to {output_sr} Hz "
                        f"for batch item {i}"
                    )
                
                # Ensure proper shape (SoX may return 1D tensor for mono)
                if augmented_tensor.dim() == 1:
                    augmented_tensor = augmented_tensor.unsqueeze(0)
                
                # Adjust length to match original
                # Add batch dimension for adjustment function
                length_adjusted = AudioProcessor.adjust_audio_length(
                    augmented_tensor.unsqueeze(0),  # [1, channels, samples]
                    ctx.original_length,
                    mode=ctx.mode
                )
                augmented_tensors.append(length_adjusted.squeeze(0))
                
            except torchaudio.sox_effects.sox_effects.SoxEffectsError as e:
                # Specific SoX error - log details
                logger.warning(
                    f"SoX effect failed for batch {i}: {str(e)}. "
                    f"Effects: {effects}. Using original audio."
                )
                augmented_tensors.append(tensor_cpu[i])
            except Exception as e:
                # Other errors - log with type
                logger.warning(
                    f"Unexpected error applying SoX effect to batch {i}: "
                    f"{type(e).__name__}: {str(e)}. Using original audio."
                )
                augmented_tensors.append(tensor_cpu[i])

        # Stack results and move back to original device
        try:
            output = torch.stack(augmented_tensors, dim=0).to(original_device)
            return output
        except RuntimeError as e:
            logger.error(
                f"Failed to stack SoX results: {str(e)}. "
                f"Returning original tensor."
            )
            return tensor

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        """
        Backward pass using straight-through estimator.
        
        The STE pattern passes gradients through unchanged, treating the
        non-differentiable SoX operation as an identity function for backprop.
        
        Args:
            ctx (Any): Context with saved tensors (not used in STE).
            grad_output (torch.Tensor): Gradient from subsequent layers.
            
        Returns:
            Tuple[torch.Tensor, None, None, None]: 
                - grad_output: Unchanged gradient for the input tensor
                - None: No gradient for effects parameter
                - None: No gradient for sample_rate parameter
                - None: No gradient for mode parameter
        """
        # STE: pass gradient through unchanged
        return grad_output, None, None, None

# =============================================================================
# Compression Effects STE
# =============================================================================
class _CompressionSTE(_BaseEffectSTE):
    """
    STE for lossy compression effects (MP3/AAC).
    
    Implements differentiable lossy audio compression using FFmpeg for
    realistic compression artifacts during training. This is crucial for
    watermarking systems that need to be robust against compression.
    
    Compression introduces:
        - Quantization noise
        - Frequency masking
        - Pre-echo artifacts
        - Encoder/decoder delays
    
    Supported formats:
        - MP3: Using LAME encoder (libmp3lame)
        - AAC: Using native AAC encoder
    
    Note:
        Requires FFmpeg to be installed and accessible in PATH.
        Compression runs on CPU with temporary file I/O.
    """

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, compression_type: str, bitrate: str, sample_rate: int) -> torch.Tensor:
        """
        Forward pass for compression effects.
        
        Args:
            ctx (Any): PyTorch context for saving tensors for backward pass.
            tensor (torch.Tensor): Input audio tensor of shape (batch, channels, time).
                Will be processed on CPU due to FFmpeg requirements.
            compression_type (str): Type of compression ('mp3' or 'aac').
                Determines codec and encoder delay compensation.
            bitrate (str): Target bitrate (e.g., '128k', '192k', '256k').
                Lower bitrates introduce more artifacts.
            sample_rate (int): Audio sample rate in Hz.
                Must match the actual audio content.
            
        Returns:
            torch.Tensor: Compressed audio tensor with original shape preserved.
                Includes automatic encoder delay compensation.
            
        Raises:
            ValueError: If compression type is not supported.
            RuntimeError: If FFmpeg is not available or compression fails.
            
        Note:
            Failed compressions on individual batch items will log an error
            and return the original audio for that item (fail-safe behavior).
        """
        # Save context for backward pass
        ctx.save_for_backward(tensor)
        ctx.original_length = tensor.shape[-1]
        ctx.device = tensor.device
        batch_size = tensor.size(0)

        # Validate compression type
        if compression_type not in ['mp3', 'aac']:
            raise ValueError(
                f"Unsupported compression type: '{compression_type}'. "
                f"Supported types are: 'mp3', 'aac'"
            )
            
        logger.debug(
            f"Applying {compression_type.upper()} compression at {bitrate} "
            f"to batch of size {batch_size}"
        )

        processed_tensors = []
        
        for i in range(batch_size):
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in, \
                     tempfile.NamedTemporaryFile(suffix=f".{compression_type}", delete=False) as f_out:

                    # Save audio to temporary WAV file
                    # Ensure audio is on CPU for file I/O
                    torchaudio.save(f_in.name, tensor[i].cpu(), sample_rate)

                    # Determine codec based on compression type
                    codec = 'aac' if compression_type == 'aac' else 'libmp3lame'
                    
                    # Build FFmpeg command with appropriate settings
                    # -y: overwrite output file
                    # -i: input file
                    # -ar: audio sample rate
                    # -b:a: audio bitrate
                    # -c:a: audio codec
                    command = [
                        "ffmpeg", "-y", "-i", f_in.name,
                        "-ar", str(sample_rate),
                        "-b:a", bitrate,
                        "-c:a", codec,
                        f_out.name
                    ]

                    # Execute compression
                    subprocess.run(
                        command, 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL, 
                        check=True
                    )
                    
                    # Load compressed audio
                    compressed_tensor, _ = torchaudio.load(f_out.name)

                    # Handle encoder delay and padding
                    # Compression often adds samples due to encoder/decoder delays
                    compressed_length = compressed_tensor.shape[-1]
                    original_length = ctx.original_length

                    # Get encoder delay based on format
                    # These are standard delays for each codec
                    encoder_delay = AAC_ENCODER_DELAY if compression_type == "aac" else MP3_ENCODER_DELAY

                    # Calculate trimming amounts to preserve alignment
                    total_extra = compressed_length - original_length
                    if total_extra > 0:
                        # Compressed audio is longer - trim excess
                        # Ensure encoder_delay does not exceed total_extra
                        encoder_delay = min(encoder_delay, total_extra)
                        # Trim encoder delay from start, remaining from end
                        start_trim = encoder_delay
                        end_trim = total_extra - start_trim
                        compressed_tensor = compressed_tensor[..., start_trim:compressed_length - end_trim]
                        logger.debug(
                            f"Trimmed {start_trim} samples from start, {end_trim} from end "
                            f"(batch {i})"
                        )
                    elif total_extra < 0:
                        # Compressed audio is shorter - pad with zeros
                        pad_length = original_length - compressed_length
                        padding = torch.zeros(
                            *compressed_tensor.shape[:-1], pad_length, 
                            device=compressed_tensor.device,
                            dtype=compressed_tensor.dtype
                        )
                        compressed_tensor = torch.cat([compressed_tensor, padding], dim=-1)
                        logger.debug(f"Padded {pad_length} zeros at end (batch {i})")

                    # Final safety check - ensure exact length match
                    compressed_tensor = compressed_tensor[..., :original_length]

                    # Send back to the original device
                    compressed_tensor = compressed_tensor.to(ctx.device)
                    processed_tensors.append(compressed_tensor)

            except subprocess.CalledProcessError as e:
                # FFmpeg execution failed
                logger.error(
                    f"FFmpeg {compression_type} compression failed for batch {i}: {str(e)}. "
                    f"Command: {' '.join(command)}. Using original audio.",
                    exc_info=True
                )
                processed_tensors.append(tensor[i])
            except FileNotFoundError as e:
                # FFmpeg not found
                logger.error(
                    f"FFmpeg not found in PATH: {str(e)}. "
                    f"Please install FFmpeg to use compression effects."
                )
                processed_tensors.append(tensor[i])
            except Exception as e:
                # Other errors
                logger.warning(
                    f"Compression failed for batch {i}: {type(e).__name__}: {str(e)}. "
                    f"Using original audio."
                )
                processed_tensors.append(tensor[i])
            finally:
                # Clean up temporary files
                try:
                    import os
                    # Safe cleanup with existence check
                    if 'f_in' in locals() and hasattr(f_in, 'name') and os.path.exists(f_in.name):
                        os.unlink(f_in.name)
                    if 'f_out' in locals() and hasattr(f_out, 'name') and os.path.exists(f_out.name):
                        os.unlink(f_out.name)
                except Exception as cleanup_error:
                    logger.debug(f"Failed to clean up temp files: {cleanup_error}")

        output = torch.stack(processed_tensors, dim=0)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        """
        Backward pass using straight-through estimator.
        
        The STE pattern passes gradients through unchanged, allowing the model
        to learn features that are robust to compression artifacts.
        
        Args:
            ctx (Any): Context with saved tensors (not used in STE).
            grad_output (torch.Tensor): Gradient from subsequent layers.
            
        Returns:
            Tuple[torch.Tensor, None, None, None]:
                - grad_output: Unchanged gradient for the input tensor
                - None: No gradient for compression_type parameter
                - None: No gradient for bitrate parameter  
                - None: No gradient for sample_rate parameter
                
        Note:
            The clone() ensures gradient isolation and prevents in-place
            modifications that could affect other parts of the graph.
        """
        # STE: pass gradient through unchanged
        # Clone to ensure gradient isolation
        grad_input = grad_output.clone()
        return grad_input, None, None, None

# =============================================================================
# Encodec Compression STE
# =============================================================================
class _EncodecSTE(Function):
    """
    STE for Encodec-based compression with model caching.
    
    This class implements neural audio compression using Facebook's Encodec model
    with efficient model caching for repeated use. Encodec provides high-quality
    neural audio compression at various bitrates.
    
    Features:
        - Neural codec with learned representations
        - Variable bitrate control (1.5 - 24 kb/s)
        - Automatic sample rate handling
        - Model caching for efficiency
    
    Available models:
        - facebook/encodec_24khz: 24kHz model
        - facebook/encodec_32khz: 32kHz model (music)
    
    Reference:
        Défossez et al., "High Fidelity Neural Audio Compression" (2022)
    """

    # Class-level model cache to avoid repeated loading
    _model_cache: Dict[str, Any] = defaultdict(lambda: None)

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, model_id: str, bandwidth: float, 
                sample_rate: int, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for Encodec compression.
        
        Args:
            ctx (Any): PyTorch context for saving tensors for backward pass.
            tensor (torch.Tensor): Input audio tensor of shape (batch, channels, time).
                Will be processed on CPU due to model requirements.
            model_id (str): Hugging Face model identifier.
                E.g., 'facebook/encodec_24khz' or 'facebook/encodec_32khz'.
            bandwidth (float): Target bandwidth for compression in kb/s.
                Typical values: 1.5, 3.0, 6.0, 12.0, 24.0.
            sample_rate (int): Input audio sample rate in Hz.
                Will be resampled if necessary to match model requirements.
            mask (Optional[torch.Tensor]): Optional mask tensor.
                Currently not used but maintained for API consistency.
            
        Returns:
            torch.Tensor: Compressed audio tensor with original shape and device.
                Automatic resampling ensures compatibility.
                
        Raises:
            ImportError: If transformers library is not installed.
            ValueError: If model_id format is invalid.
            RuntimeError: If model loading or processing fails.
            
        Note:
            The first call with a new model_id will download and cache the model.
            Subsequent calls reuse the cached model for efficiency.
        """
        # Save context for backward pass
        ctx.save_for_backward(tensor)
        ctx.original_length = tensor.shape[-1]
        original_device = tensor.device

        # Move input tensor to CPU for processing
        # Encodec models typically run on CPU for inference
        tensor = tensor.detach().cpu()
        model_key = model_id
        
        logger.debug(
            f"Applying Encodec compression with model '{model_id}' "
            f"at {bandwidth} kb/s bandwidth"
        )

        # Load and cache the model if not already cached
        if _EncodecSTE._model_cache[model_key] is None:
            try:
                from transformers import EncodecModel
                logger.info(f"Loading Encodec model: {model_id} (first time, will be cached)")
                
                # Load model from Hugging Face hub
                model = EncodecModel.from_pretrained(model_id)
                model = model.cpu()  # Ensure CPU processing
                model.eval()  # Set to evaluation mode

                # Cache the model for future use
                _EncodecSTE._model_cache[model_key] = model
                logger.info(f"Successfully cached Encodec model: {model_id}")
                
            except ImportError as e:
                error_msg = (
                    "transformers library not installed. "
                    "Install with: pip install transformers>=4.31.0"
                )
                logger.error(error_msg)
                raise ImportError(error_msg) from e
            except ValueError as e:
                # Invalid model ID format
                logger.error(
                    f"Invalid Encodec model ID '{model_id}': {str(e)}. "
                    f"Valid IDs: facebook/encodec_24khz, facebook/encodec_32khz"
                )
                return tensor.to(original_device)
            except Exception as e:
                # Model loading failed
                logger.error(
                    f"Failed to load EncodecModel '{model_id}': {type(e).__name__}: {str(e)}",
                    exc_info=True
                )
                return tensor.to(original_device)

        # Retrieve cached model
        model = _EncodecSTE._model_cache.get(model_key)
        if model is None:
            logger.error(
                f"Encodec model '{model_id}' not found in cache after loading attempt. "
                f"Using identity transformation."
            )
            return tensor.to(original_device)

        try:
            # Infer target sample rate from model_id
            # Encodec models are trained for specific sample rates
            if "24khz" in model_id.lower():
                target_sr = 24000
            elif "32khz" in model_id.lower():
                target_sr = 32000
            else:
                # Attempt to extract from model config if available
                if hasattr(model, 'config') and hasattr(model.config, 'sampling_rate'):
                    target_sr = model.config.sampling_rate
                    logger.info(f"Using sampling rate {target_sr} from model config")
                else:
                    raise ValueError(
                        f"Cannot infer sampling rate from model_id: '{model_id}'. "
                        f"Expected '24khz' or '32khz' in model name."
                    )

            # Resample if necessary to match model requirements
            if sample_rate != target_sr:
                logger.debug(
                    f"Resampling audio from {sample_rate}Hz to {target_sr}Hz "
                    f"for Encodec model compatibility"
                )
                tensor = torchaudio.functional.resample(tensor, sample_rate, target_sr)

            # Apply Encodec compression
            # Use no_grad for inference efficiency
            with torch.no_grad():
                # Process through Encodec model
                outputs = model(tensor, bandwidth=bandwidth)
                output_tensor = outputs.audio_values
                
                # Log compression stats if available
                if hasattr(outputs, 'audio_codes'):
                    logger.debug(
                        f"Encodec compressed to {outputs.audio_codes.shape[-1]} codes "
                        f"at {bandwidth} kb/s"
                    )

            # Resample back to original sample rate if needed
            if sample_rate != target_sr:
                logger.debug(f"Resampling back from {target_sr}Hz to {sample_rate}Hz")
                output_tensor = torchaudio.functional.resample(output_tensor, target_sr, sample_rate)
            
            # Ensure output matches input length
            if output_tensor.shape[-1] != ctx.original_length:
                output_tensor = AudioProcessor.adjust_audio_length(
                    output_tensor, ctx.original_length, mode='pad_truncate'
                )
                
            # Return to original device
            output_tensor = output_tensor.to(original_device)
            return output_tensor

        except ValueError as e:
            # Configuration or parameter errors
            logger.error(
                f"Encodec parameter error: {str(e)}. "
                f"Check bandwidth ({bandwidth}) and model compatibility.",
                exc_info=True
            )
            return tensor.to(original_device)
        except RuntimeError as e:
            # Model execution errors
            logger.error(
                f"Encodec runtime error: {str(e)}. "
                f"This may be due to memory constraints or tensor shape issues.",
                exc_info=True
            )
            return tensor.to(original_device)
        except Exception as e:
            # Other unexpected errors
            logger.error(
                f"Unexpected Encodec error: {type(e).__name__}: {str(e)}",
                exc_info=True
            )
            return tensor.to(original_device)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None, None]:
        """
        Backward pass using straight-through estimator.
        
        The STE allows gradients to flow through the non-differentiable
        neural compression, enabling end-to-end training of systems that
        include Encodec compression.
        
        Args:
            ctx (Any): Context with saved tensors (not used in STE).
            grad_output (torch.Tensor): Gradient from subsequent layers.
            
        Returns:
            Tuple[torch.Tensor, None, None, None, None]:
                - grad_output: Unchanged gradient for the input tensor
                - None: No gradient for model_id parameter
                - None: No gradient for bandwidth parameter
                - None: No gradient for sample_rate parameter
                - None: No gradient for mask parameter
        """
        # STE: pass gradient through unchanged
        return grad_output, None, None, None, None

# =============================================================================
# Quantization STE
# =============================================================================
class _QuantizationSTE(Function):
    """
    STE for quantization effect.
    
    Implements differentiable bit-depth reduction for simulating
    low-quality audio codecs.
    """

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, bit_depth: int) -> torch.Tensor:
        """
        Forward pass for quantization.
        
        Args:
            ctx: Context for saving tensors
            tensor: Input audio tensor
            bit_depth: Target bit depth (e.g., 8, 16)
            
        Returns:
            torch.Tensor: Quantized audio tensor
        """
        ctx.save_for_backward(tensor)
        
        # Calculate quantization levels
        max_val = 2 ** (bit_depth - 1) - 1
        
        # Quantize: scale, round, then scale back
        tensor_int = (tensor * max_val).round()
        quantized = tensor_int / max_val
        
        return quantized

    @staticmethod
    def backward(_ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Backward pass using straight-through estimator.
        
        Args:
            ctx: Context with saved tensors
            grad_output: Gradient from subsequent layers
            
        Returns:
            Tuple of gradients for each input
        """
        return grad_output, None

# =============================================================================
# Shush Effect STE
# =============================================================================
class _ShushSTE(_BaseEffectSTE):
    """
    STE for the shush effect (zeroing out quiet samples).
    
    This effect simulates aggressive noise gating by removing
    the quietest samples in the audio. Unlike traditional noise gates
    that use thresholds, this effect removes a fixed percentage of
    the quietest samples regardless of their absolute level.
    
    Use cases:
        - Simulating aggressive noise reduction
        - Testing robustness to sample dropout
        - Removing low-energy components
    """

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, fraction: float) -> torch.Tensor:
        """
        Forward pass for shush effect.
        
        Args:
            ctx (Any): PyTorch context for saving tensors for backward pass.
            tensor (torch.Tensor): Input audio tensor of shape (batch, channels, time).
            fraction (float): Fraction of quietest samples to zero out (0.0 to 1.0).
                E.g., 0.1 removes the quietest 10% of samples.
            
        Returns:
            torch.Tensor: Audio with quiet samples zeroed out.
                Shape remains unchanged, but quiet samples become 0.
                
        Note:
            The effect is applied independently to each channel in each
            batch item, preserving channel-specific characteristics.
        """
        with torch.no_grad():
            batch_size, channels, num_samples = tensor.size()
            k = int(num_samples * fraction)
            
            # Ensure we don't try to zero out all samples
            # Keep at least one sample to avoid complete silence
            k = min(k, num_samples - 1)
            
            logger.debug(f"Shush effect: zeroing {k} quietest samples per channel ({fraction:.1%})")

            # Create mask for samples to keep (1) or remove (0)
            shush_mask = torch.ones_like(tensor)
            
            # Reshape for easier processing
            tensor_flat = tensor.view(batch_size * channels, num_samples)
            
            # Find k quietest samples per channel based on absolute value
            # This removes both positive and negative low-amplitude samples
            _, indices = torch.topk(tensor_flat.abs(), k, dim=1, largest=False)
            
            # Zero out quiet samples in the mask
            shush_mask_flat = shush_mask.view(batch_size * channels, num_samples)
            shush_mask_flat.scatter_(1, indices, 0)
            
            # Restore original shape
            shush_mask = shush_mask_flat.view(batch_size, channels, num_samples)

        # Save mask for backward pass
        ctx.save_for_backward(shush_mask)
        
        # Apply mask to zero out quiet samples
        return tensor * shush_mask

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Backward pass preserving gradients for non-zeroed samples.
        
        Unlike pure STE, this effect modifies gradients to respect the
        zeroing operation. Gradients for zeroed samples are blocked,
        while gradients for preserved samples pass through.
        
        Args:
            ctx (Any): Context with saved shush mask.
            grad_output (torch.Tensor): Gradient from subsequent layers.
            
        Returns:
            Tuple[torch.Tensor, None]:
                - Masked gradient (zero for removed samples)
                - None: No gradient for fraction parameter
                
        Note:
            This selective gradient allows the model to learn features
            in the non-quiet parts of the signal.
        """
        shush_mask, = ctx.saved_tensors
        # Apply mask to gradients - zeroed samples get zero gradient
        return grad_output * shush_mask, None

# =============================================================================
# Median Filter STE
# =============================================================================
class _MedianFilterSTE(Function):
    """
    STE for median filtering.
    
    Implements differentiable median filtering using scipy's implementation
    with straight-through gradient estimation. Median filtering is effective
    for removing impulse noise while preserving edges.
    
    Properties:
        - Non-linear filter
        - Preserves edges better than linear filters
        - Effective against salt-and-pepper noise
        - Computational cost increases with kernel size
    
    Note:
        Requires scipy to be installed. Processing is done on CPU
        using numpy arrays for compatibility with scipy.
    """

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """
        Forward pass for median filtering.
        
        Args:
            ctx (Any): PyTorch context for saving tensors for backward pass.
            tensor (torch.Tensor): Input audio tensor of shape (batch, channels, time).
            kernel_size (int): Size of the median filter window.
                Must be odd for symmetric filtering. If even, will be incremented.
            
        Returns:
            torch.Tensor: Median filtered audio with same shape as input.
                Edge handling uses 'reflect' mode internally.
            
        Raises:
            ImportError: If scipy is not installed.
            ValueError: If kernel_size is invalid (< 1).
            RuntimeError: If filtering fails on any channel.
            
        Note:
            The median filter is applied independently to each channel
            in each batch item. Failed channels use identity (original audio).
        """
        try:
            from scipy.signal import medfilt
        except ImportError:
            error_msg = "scipy is required for median filtering. Install with: pip install scipy"
            logger.error(error_msg)
            raise ImportError(error_msg)

        # Validate kernel size
        if kernel_size < 1:
            raise ValueError(f"Kernel size must be positive, got {kernel_size}")

        # Save input for backward pass
        ctx.save_for_backward(tensor)
        
        # Ensure kernel size is odd for symmetric window
        if kernel_size % 2 == 0:
            kernel_size += 1
            logger.warning(f"Kernel size must be odd for symmetric filtering, adjusted to {kernel_size}")

        # Log filtering operation
        logger.debug(f"Applying median filter with kernel size {kernel_size}")
        
        # Process on CPU with numpy for scipy compatibility
        original_device = tensor.device
        tensor_np = tensor.detach().cpu().numpy()
        output_np = np.zeros_like(tensor_np)
        
        # Apply median filter to each batch and channel independently
        for b in range(tensor_np.shape[0]):
            for c in range(tensor_np.shape[1]):
                try:
                    # medfilt uses 'reflect' mode at boundaries by default
                    output_np[b, c] = medfilt(tensor_np[b, c], kernel_size=kernel_size)
                except Exception as e:
                    # Log error and use original audio for failed channel
                    logger.warning(
                        f"Median filter failed for batch {b}, channel {c}: "
                        f"{type(e).__name__}: {str(e)}. Using original audio."
                    )
                    output_np[b, c] = tensor_np[b, c]

        # Convert back to tensor on original device
        output_tensor = torch.from_numpy(output_np).to(tensor.device)
        return output_tensor

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Backward pass using straight-through estimator.
        
        The STE allows gradients to flow through the non-differentiable
        median filter operation, treating it as identity for backprop.
        
        Args:
            ctx (Any): Context with saved tensors (not used in STE).
            grad_output (torch.Tensor): Gradient from subsequent layers.
            
        Returns:
            Tuple[torch.Tensor, None]:
                - grad_output: Unchanged gradient for the input tensor
                - None: No gradient for kernel_size parameter
        """
        # STE: pass gradient through unchanged
        return grad_output, None

# =============================================================================
# AudioEffects Class
# =============================================================================
class AudioEffects:
    """
    Collection of audio effects with proper gradient handling through STEs.
    
    This class provides a comprehensive set of audio transformations commonly used
    in audio processing and watermarking research. All effects support gradient
    flow for use in differentiable pipelines.
    
    Effect Categories:
        - Filtering: lowpass, highpass, bandpass, median
        - Compression: mp3, aac, encodec
        - Noise: white noise, pink noise, random noise
        - Time-domain: echo, speed, smooth
        - Amplitude: scaling, quantization, sample suppression
        - Equalization: random parametric EQ
    
    All methods follow the pattern:
        - Input: (tensor, mask, **kwargs) 
        - Output: (processed_tensor, processed_mask)
        - Mask indicates watermark presence and is updated appropriately
    
    Note:
        Effects are designed to be robust and fail gracefully,
        returning the original audio if processing fails.
    """

    @staticmethod
    def identity(
        tensor: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Identity transform (no-op).
        
        Args:
            tensor: Input audio tensor
            mask: Optional mask tensor
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Unchanged audio and mask
        """
        return tensor, mask

    @staticmethod
    def speed(
        tensor: torch.Tensor,
        speed: Union[float, Tuple[float, float]] = 1.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        mask: Optional[torch.Tensor] = None,
        **kwargs  # Accepts but ignores additional arguments for API consistency
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Speed up or slow down audio without changing pitch.
        
        Uses SoX's speed effect which maintains pitch while changing tempo.
        This is different from resampling which would change both speed and pitch.
        
        Args:
            tensor (torch.Tensor): Input audio tensor.
            speed (Union[float, Tuple[float, float]], optional): Speed factor.
                - float: Fixed speed (e.g., 0.5 = half speed, 2.0 = double speed)
                - tuple: Random speed chosen from range (min, max)
                Defaults to 1.0 (no change).
            sample_rate (int, optional): Audio sample rate in Hz.
                Defaults to DEFAULT_SAMPLE_RATE.
            mask (Optional[torch.Tensor], optional): Mask tensor.
                Will be time-stretched to match output length.
            **kwargs: Additional arguments (ignored for compatibility).
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: 
                - Speed-adjusted audio (length changes by 1/speed factor)
                - Time-adjusted mask if provided
                
        Raises:
            ValueError: If speed is not positive.
            
        Examples:
            >>> # Double speed (half duration)
            >>> fast_audio, fast_mask = AudioEffects.speed(audio, speed=2.0)
            >>> # Random speed between 0.8x and 1.2x
            >>> var_audio, var_mask = AudioEffects.speed(audio, speed=(0.8, 1.2))
        """
        try:
            # Handle random speed selection
            if isinstance(speed, tuple):
                speed = random.uniform(*speed)
                
            # Validate speed
            if speed <= 0:
                raise ValueError(f"Speed must be positive, got {speed}")
                
            # Apply speed effect using SoX
            effects = [
                ['speed', f'{speed}'], 
                ['rate', f'{sample_rate}']
            ]
            
            output_tensor = _SoxEffectSTE.apply(tensor, effects, sample_rate, 'stretch')

            # Adjust mask if provided
            if mask is not None:
                new_length = output_tensor.shape[-1]
                mask = AudioProcessor.adjust_mask_length(
                    mask, new_length, mode='nearest-exact'
                )

            return output_tensor, mask
            
        except Exception as e:
            logger.error(f"Speed effect failed: {str(e)}", exc_info=True)
            return tensor, mask

    @staticmethod
    def resample(
        tensor: torch.Tensor,
        new_sample_rate: int,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Resample audio to a new sample rate and back.
        
        This simulates the artifacts introduced by sample rate conversion.
        
        Args:
            tensor: Input audio tensor
            new_sample_rate: Target sample rate for resampling
            sample_rate: Original sample rate
            mask: Optional mask tensor
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Resampled audio and mask
            
        Raises:
            ValueError: If new_sample_rate is not a positive integer
        """
        # Validate inputs
        if not isinstance(new_sample_rate, int) or new_sample_rate <= 0:
            raise ValueError(f"new_sample_rate must be positive int, got {new_sample_rate}")
            
        try:
            device = tensor.device
            
            # Resample to new sample rate
            resampler_down = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=new_sample_rate
            ).to(device)
            intermediate_tensor = resampler_down(tensor)

            # Resample back to original sample rate
            resampler_up = torchaudio.transforms.Resample(
                orig_freq=new_sample_rate, 
                new_freq=sample_rate
            ).to(device)
            output_tensor = resampler_up(intermediate_tensor)

            return output_tensor, mask
            
        except Exception as e:
            logger.error(f"Resample effect failed: {str(e)}", exc_info=True)
            return tensor, mask

    @staticmethod
    def echo(
        tensor: torch.Tensor,
        volume_range: Tuple[float, float] = (0.1, 0.5),
        duration_range: Tuple[float, float] = (0.1, 0.5),
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply echo effect to audio.
        
        Creates a delayed copy of the signal with variable volume to simulate
        acoustic reflections. The echo is created using convolution with an
        impulse response containing the direct sound and delayed reflection.
        
        Args:
            tensor (torch.Tensor): Input audio tensor of shape [batch, channels, time].
            volume_range (Tuple[float, float], optional): Range for echo volume.
                Defaults to (0.1, 0.5), meaning echo is 10-50% of original.
            duration_range (Tuple[float, float], optional): Range for echo delay
                in seconds. Defaults to (0.1, 0.5) seconds.
            sample_rate (int, optional): Audio sample rate in Hz.
                Defaults to DEFAULT_SAMPLE_RATE.
            mask (Optional[torch.Tensor], optional): Mask tensor with same shape.
                Echo doesn't modify mask values.
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: 
                - Audio with echo applied (same shape as input)
                - Unchanged mask (echo preserves watermark)
            
        Raises:
            RuntimeError: If tensor and mask are on different devices.
            
        Note:
            Echo is applied using FFT convolution for efficiency.
            The output is normalized to prevent clipping.
        """
        # Device consistency check
        if mask is not None and mask.device != tensor.device:
            raise RuntimeError(
                f"Device mismatch in echo effect: tensor on {tensor.device}, mask on {mask.device}"
            )

        try:
            # Get audio length and validate
            current_length = tensor.shape[-1]
            max_duration = current_length / sample_rate
            
            # Check if audio is too short for echo
            if max_duration <= 0 or current_length < MIN_AUDIO_LENGTH:
                logger.warning("Audio too short for echo effect, returning unchanged")
                return tensor, mask
            
            # Generate random echo parameters
            duration = torch.FloatTensor(1).uniform_(*duration_range).item()
            # Cap duration at 50% of audio length to ensure echo doesn't dominate
            safe_max_duration = max_duration * 0.5
            duration = min(duration, safe_max_duration)
            
            volume = torch.FloatTensor(1).uniform_(*volume_range).item()
            n_samples = int(sample_rate * duration)
            
            # Ensure minimum samples for valid convolution
            n_samples = max(n_samples, MIN_AUDIO_LENGTH)
                
            # Create impulse response for echo
            # The impulse response models: direct sound + single reflection
            impulse_response = torch.zeros(n_samples).type(tensor.type()).to(tensor.device)
            impulse_response[0] = 1.0  # Direct sound at t=0
            impulse_response[-1] = volume  # Delayed echo at t=duration
            
            # Add batch and channel dimensions
            impulse_response = impulse_response.unsqueeze(0).unsqueeze(0)

            # Apply convolution for echo effect
            # FFT convolution is more efficient than direct convolution for longer IRs
            reverbed_signal = julius.fft_conv1d(tensor, impulse_response)
            reverbed_signal = reverbed_signal.to(tensor.device)

            # Normalize to preserve original amplitude range and prevent clipping
            max_reverbed = torch.max(torch.abs(reverbed_signal))
            max_original = torch.max(torch.abs(tensor))
            if max_reverbed > 0 and max_original > 0:
                # Scale reverbed signal to match original peak amplitude
                reverbed_signal = reverbed_signal / max_reverbed * max_original

            # Ensure output matches input length (convolution may extend signal)
            reverbed_length = reverbed_signal.shape[-1]
            tmp = torch.zeros_like(tensor)
            # Copy only the valid portion, truncating any convolution tail
            tmp[..., :min(reverbed_length, current_length)] = reverbed_signal[..., :current_length]
            reverbed_signal = tmp

            # Echo doesn't affect watermark presence in mask
            return reverbed_signal, mask
            
        except Exception as e:
            logger.error(f"Echo effect failed: {str(e)}", exc_info=True)
            return tensor, mask

    @staticmethod
    def pink_noise(
        tensor: torch.Tensor,
        noise_std: float = 0.01,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Add pink noise (1/f noise) to audio.
        
        Pink noise has equal energy per octave, making it more natural
        sounding than white noise. It's commonly found in nature and
        music, making it useful for realistic augmentation.
        
        Args:
            tensor (torch.Tensor): Input audio tensor of any shape.
            noise_std (float, optional): Standard deviation of noise. Defaults to 0.01.
                Controls the noise amplitude relative to the signal.
            mask (Optional[torch.Tensor], optional): Mask tensor. Defaults to None.
                Pink noise doesn't affect mask values.
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: 
                - Audio with pink noise added
                - Unchanged mask (noise doesn't affect watermark presence)
                
        Note:
            Uses the Voss-McCartney algorithm for efficient pink noise generation.
            The algorithm uses multiple white noise generators updated at different
            rates to approximate 1/f spectrum.
        """
        def generate_pink_noise(size: int, depth: int = 16) -> torch.Tensor:
            """
            Generate pink noise using the Voss-McCartney algorithm.
            
            Args:
                size: Number of samples to generate
                depth: Number of random generators (higher = better approximation)
                
            Returns:
                Pink noise samples normalized to [-1, 1]
            """
            array = np.zeros(size)
            nums = np.zeros(depth)  # Array of random number generators
            
            for i in range(size):
                # Always update the first generator (highest frequency)
                nums[0] = np.random.randn()
                # Sum all generators for current sample
                array[i] = nums.sum()
                
                # Randomly update one generator (creates 1/f characteristic)
                # This is the key to the Voss-McCartney algorithm
                idx = np.random.randint(0, depth)
                nums[idx] = np.random.randn()
                
            # Normalize to [-1, 1] range
            max_val = np.max(np.abs(array))
            if max_val > 0:
                array = array / max_val
            return torch.from_numpy(array).float()
            
        try:
            # Generate pink noise matching tensor shape
            noise_shape = tensor.shape
            noise = generate_pink_noise(tensor.numel(), depth=16)
            noise = noise.reshape(noise_shape).to(tensor.device)
            
            # Scale noise by standard deviation
            noise = noise * noise_std
            
            # Add noise to signal
            output_tensor = tensor + noise
            
            return output_tensor, mask
            
        except Exception as e:
            logger.error(f"Pink noise effect failed: {str(e)}", exc_info=True)
            return tensor, mask

    @staticmethod
    def highpass_filter(
        tensor: torch.Tensor,
        cutoff_freq: float = 500,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply highpass filter to remove low frequencies.
        
        Args:
            tensor: Input audio tensor
            cutoff_freq: Cutoff frequency in Hz
            sample_rate: Audio sample rate
            mask: Optional mask tensor
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Filtered audio and mask
        """
        try:
            # Calculate Nyquist frequency
            nyquist = sample_rate / 2
            
            # Ensure cutoff is valid
            adjusted_cutoff_freq = max(0.0, min(cutoff_freq, nyquist - EPSILON))
            if adjusted_cutoff_freq != cutoff_freq:
                logger.warning(
                    f"Adjusted highpass cutoff from {cutoff_freq} Hz to {adjusted_cutoff_freq} Hz"
                )
                
            # Normalize cutoff frequency
            cutoff = adjusted_cutoff_freq / nyquist
            
            # Apply filter
            output_tensor = julius.highpass_filter(tensor, cutoff)
            
            return output_tensor, mask
            
        except Exception as e:
            logger.error(f"Highpass filter failed: {str(e)}", exc_info=True)
            return tensor, mask

    @staticmethod
    def lowpass_filter(
        tensor: torch.Tensor,
        cutoff_freq: float = 3000,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply lowpass filter to remove high frequencies.
        
        Args:
            tensor: Input audio tensor
            cutoff_freq: Cutoff frequency in Hz
            sample_rate: Audio sample rate
            mask: Optional mask tensor
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Filtered audio and mask
        """
        try:
            # Calculate Nyquist frequency
            nyquist = sample_rate / 2
            
            # Ensure cutoff is valid
            adjusted_cutoff_freq = max(0.0, min(cutoff_freq, nyquist - EPSILON))
            if adjusted_cutoff_freq != cutoff_freq:
                logger.warning(
                    f"Adjusted lowpass cutoff from {cutoff_freq} Hz to {adjusted_cutoff_freq} Hz"
                )
                
            # Normalize cutoff frequency
            cutoff = adjusted_cutoff_freq / nyquist
            
            # Apply filter
            output_tensor = julius.lowpass_filter(tensor, cutoff)
            
            return output_tensor, mask
            
        except Exception as e:
            logger.error(f"Lowpass filter failed: {str(e)}", exc_info=True)
            return tensor, mask

    @staticmethod
    def bandpass_filter(
        tensor: torch.Tensor,
        cutoff_freq_low: float,
        cutoff_freq_high: float,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply bandpass filter to keep only a frequency range.
        
        Removes frequencies outside the specified band, useful for
        isolating specific frequency components or simulating
        band-limited transmission channels.
        
        Args:
            tensor (torch.Tensor): Input audio tensor.
            cutoff_freq_low (float): Low cutoff frequency in Hz.
                Frequencies below this are attenuated.
            cutoff_freq_high (float): High cutoff frequency in Hz.
                Frequencies above this are attenuated.
            sample_rate (int, optional): Audio sample rate in Hz.
                Defaults to DEFAULT_SAMPLE_RATE.
            mask (Optional[torch.Tensor], optional): Mask tensor.
                Bandpass filtering preserves mask values.
            **kwargs: Additional arguments (ignored).
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: 
                - Bandpass filtered audio
                - Unchanged mask
            
        Raises:
            ValueError: If cutoff frequencies are invalid or out of order.
            
        Note:
            Cutoff frequencies are automatically adjusted to valid range
            [0, nyquist) where nyquist = sample_rate / 2.
        """
        try:
            # Validate input parameters
            if cutoff_freq_low < 0:
                raise ValueError(f"Low cutoff frequency must be non-negative, got {cutoff_freq_low} Hz")
            if cutoff_freq_high < 0:
                raise ValueError(f"High cutoff frequency must be non-negative, got {cutoff_freq_high} Hz")
                
            nyquist = sample_rate / 2.0

            # Adjust cutoff frequencies to valid range [0, nyquist)
            adjusted_low = max(0.0, min(cutoff_freq_low, nyquist - EPSILON))
            adjusted_high = max(0.0, min(cutoff_freq_high, nyquist - EPSILON))
            
            # Log adjustments if made
            if adjusted_low != cutoff_freq_low or adjusted_high != cutoff_freq_high:
                logger.warning(
                    f"Adjusted bandpass cutoffs: {cutoff_freq_low}-{cutoff_freq_high} Hz "
                    f"to {adjusted_low}-{adjusted_high} Hz (nyquist={nyquist} Hz)"
                )

            # Ensure low < high
            if adjusted_low >= adjusted_high:
                raise ValueError(
                    f"Low cutoff {adjusted_low} Hz must be less than high cutoff {adjusted_high} Hz"
                )

            # Normalize cutoff frequencies to [0, 1] for julius
            normalized_low = adjusted_low / nyquist
            normalized_high = adjusted_high / nyquist

            # Validate normalized values
            if not (0.0 < normalized_low < 1.0) or not (0.0 < normalized_high < 1.0):
                raise ValueError(
                    f"Normalized cutoffs must be between 0 and 1. "
                    f"Got low: {normalized_low}, high: {normalized_high}"
                )
                
            logger.debug(
                f"Applying bandpass filter: {adjusted_low:.1f}-{adjusted_high:.1f} Hz "
                f"(normalized: {normalized_low:.3f}-{normalized_high:.3f})"
            )

            # Apply bandpass filter using julius
            output_tensor = julius.bandpass_filter(tensor, normalized_low, normalized_high)
            
            # Bandpass filtering doesn't affect mask
            return output_tensor, mask
            
        except ValueError:
            # Re-raise ValueError with original message
            raise
        except Exception as e:
            # Log unexpected errors and return original
            logger.error(
                f"Bandpass filter failed: {type(e).__name__}: {str(e)}. "
                f"Cutoffs: {cutoff_freq_low}-{cutoff_freq_high} Hz, "
                f"Sample rate: {sample_rate} Hz",
                exc_info=True
            )
            return tensor, mask

    @staticmethod
    def median_filter(
        tensor: torch.Tensor,
        kernel_size: int = 3,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply median filtering for impulse noise reduction.
        
        Args:
            tensor: Input audio tensor
            kernel_size: Size of median filter window (must be odd)
            mask: Optional mask tensor
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Filtered audio and mask
        """
        try:
            # Validate kernel size
            if kernel_size < 1:
                raise ValueError(f"Kernel size must be positive, got {kernel_size}")
                
            output_tensor = _MedianFilterSTE.apply(tensor, kernel_size)
            return output_tensor, mask
            
        except Exception as e:
            logger.error(f"Median filter failed: {str(e)}", exc_info=True)
            return tensor, mask
    
    @staticmethod
    def smooth(
        tensor: torch.Tensor,
        window_size_range: Tuple[int, int] = (2, 10),
        mask: Optional[torch.Tensor] = None,
        valid_threshold: float = 0.5  
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply smoothing filter with moving average.
        
        This effect applies a uniform moving average filter to reduce
        high-frequency content and create a smoother signal. Unlike
        traditional lowpass filters, this uses a simple averaging window.
        
        Args:
            tensor (torch.Tensor): Input audio tensor of any shape.
            window_size_range (Tuple[int, int], optional): Range for random
                window size selection. Defaults to (2, 10) samples.
            mask (Optional[torch.Tensor], optional): Mask tensor indicating
                watermark presence. Will be updated based on valid samples.
            valid_threshold (float, optional): Minimum fraction of valid (non-zero)
                samples in window to preserve mask value. Defaults to 0.5.
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: 
                - Smoothed audio with reduced high frequencies
                - Updated mask based on valid sample ratio
            
        Raises:
            RuntimeError: If tensor and mask are on different devices.
            
        Note:
            The mask is updated to reflect whether enough valid samples
            contributed to each output position. This prevents watermark
            detection in heavily smoothed regions.
        """
        # Device consistency check
        if mask is not None and mask.device != tensor.device:
            raise RuntimeError(
                f"Device mismatch in smooth effect: tensor on {tensor.device}, mask on {mask.device}"
            )
        
        try:
            # Select random window size
            window_size = int(torch.FloatTensor(1).uniform_(*window_size_range))
            
            # Create uniform smoothing kernel (moving average)
            # All weights are equal = 1/window_size
            kernel = torch.ones(1, 1, window_size).type(tensor.type()) / window_size
            kernel = kernel.to(tensor.device)

            # Calculate padding for length preservation
            # Use symmetric padding to maintain signal length after convolution
            pad_size = window_size - 1
            pad_left = pad_size // 2
            pad_right = pad_size - pad_left

            # Pad input with reflection to avoid edge artifacts
            # Reflection padding mirrors the signal at boundaries
            padded = torch.nn.functional.pad(tensor, (pad_left, pad_right), mode='reflect')
            
            # Apply convolution using FFT for efficiency
            smoothed = julius.fft_conv1d(padded, kernel)
            smoothed = smoothed.to(tensor.device)

            # Ensure output length matches input
            if smoothed.shape[-1] != tensor.shape[-1]:
                smoothed = AudioProcessor.adjust_audio_length(smoothed, tensor.shape[-1])

            # Update mask based on valid sample ratio
            if mask is not None:
                mask = mask.clone()
                
                # Calculate valid sample ratio for each position
                # This determines how many non-zero mask values contribute to each output
                kernel_ones = torch.ones(1, 1, window_size).to(mask.device)
                # Pad mask with zeros (no watermark at boundaries)
                padded_mask = torch.nn.functional.pad(mask, (pad_left, pad_right), mode='constant', value=0)
                # Convolve to get sum of mask values in each window
                valid_ratio = julius.fft_conv1d(padded_mask.float(), kernel_ones) / window_size
                valid_ratio = valid_ratio.to(mask.device)
                
                # Update mask: keep watermark only where enough valid samples contributed
                # This prevents false watermark detection in heavily smoothed regions
                mask = (valid_ratio >= valid_threshold).float()
                
                # Ensure mask length matches audio length
                if mask.shape[-1] != tensor.shape[-1]:
                    mask = AudioProcessor.adjust_mask_length(mask, tensor.shape[-1], mode='pad_truncate')

            return smoothed, mask
            
        except Exception as e:
            logger.error(f"Smooth effect failed: {str(e)}", exc_info=True)
            return tensor, mask

    @staticmethod
    def amplitude_scaling(
        tensor: torch.Tensor,
        scale: float = 1.0,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Scale audio amplitude by a constant factor.
        
        Args:
            tensor: Input audio tensor
            scale: Amplitude scaling factor
            mask: Optional mask tensor
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Scaled audio and mask
        """
        try:
            if scale < 0:
                logger.warning(f"Negative scale {scale} will invert audio phase")
                
            output_tensor = tensor * scale
            return output_tensor, mask
            
        except Exception as e:
            logger.error(f"Amplitude scaling failed: {str(e)}", exc_info=True)
            return tensor, mask

    @staticmethod
    def quantization(
        tensor: torch.Tensor,
        bit_depth: int = 16,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply bit-depth quantization to simulate low-quality audio.
        
        Args:
            tensor: Input audio tensor
            bit_depth: Target bit depth (1-32)
            mask: Optional mask tensor
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Quantized audio and mask
        """
        try:
            # Validate bit depth
            if not 1 <= bit_depth <= 32:
                raise ValueError(f"Bit depth must be between 1 and 32, got {bit_depth}")
                
            output_tensor = _QuantizationSTE.apply(tensor, bit_depth)
            return output_tensor, mask
            
        except Exception as e:
            logger.error(f"Quantization failed: {str(e)}", exc_info=True)
            return tensor, mask

    @staticmethod
    def sample_suppression(
        tensor: torch.Tensor,
        suppression_percentage: float = 0.1,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Randomly zero out a percentage of samples.
        
        Args:
            tensor: Input audio tensor
            suppression_percentage: Fraction of samples to suppress (0-1)
            mask: Optional mask tensor
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Audio with suppressed samples and updated mask
        """
        try:
            # Validate percentage
            if not 0 <= suppression_percentage <= 1:
                raise ValueError(f"Suppression percentage must be between 0 and 1, got {suppression_percentage}")
                
            num_samples = int(tensor.shape[-1] * suppression_percentage)
            output_tensor = tensor.clone()
            
            # Suppress samples for each batch and channel
            for b in range(tensor.shape[0]):
                for c in range(tensor.shape[1]):
                    # Select random indices to suppress
                    indices = torch.randperm(tensor.shape[-1])[:num_samples]
                    output_tensor[b, c, indices] = 0

                    # Update mask if provided
                    if mask is not None:
                        mask[b, c, indices] = 0
                        
            return output_tensor, mask
            
        except Exception as e:
            logger.error(f"Sample suppression failed: {str(e)}", exc_info=True)
            return tensor, mask

    @staticmethod
    def random_noise(
        tensor: torch.Tensor,
        noise_std: float = 0.001,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Add Gaussian white noise to audio.
        
        Args:
            tensor: Input audio tensor
            noise_std: Standard deviation of noise
            mask: Optional mask tensor
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Noisy audio and mask
        """
        try:
            if noise_std < 0:
                raise ValueError(f"Noise std must be non-negative, got {noise_std}")
                
            noise = torch.randn_like(tensor) * noise_std
            noisy_tensor = tensor + noise
            
            return noisy_tensor, mask
            
        except Exception as e:
            logger.error(f"Random noise failed: {str(e)}", exc_info=True)
            return tensor, mask

    @staticmethod
    def mp3_lossy_compression(
        tensor: torch.Tensor,
        bitrate_choice: List[str] = ["128k"],
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply MP3 lossy compression.
        
        Args:
            tensor: Input audio tensor
            bitrate_choice: List of bitrates to choose from
            sample_rate: Audio sample rate
            mask: Optional mask tensor
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Compressed audio and adjusted mask
        """
        try:
            bitrate = bitrate_choice[0] if bitrate_choice else "128k"
            output_tensor = _CompressionSTE.apply(tensor, "mp3", bitrate, sample_rate)

            # Adjust mask length if needed
            if mask is not None:
                new_length = output_tensor.shape[-1]
                mask = mask.clone().detach()
                mask = AudioProcessor.adjust_mask_length(
                    mask, new_length, mode='nearest-exact'
                )

            return output_tensor, mask
            
        except Exception as e:
            logger.error(f"MP3 compression failed: {str(e)}", exc_info=True)
            return tensor, mask

    @staticmethod
    def aac_lossy_compression(
        tensor: torch.Tensor,
        bitrate_choice: List[str] = ["128k"],
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply AAC lossy compression.
        
        Args:
            tensor: Input audio tensor
            bitrate_choice: List of bitrates to choose from
            sample_rate: Audio sample rate
            mask: Optional mask tensor
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Compressed audio and adjusted mask
        """
        try:
            bitrate = bitrate_choice[0] if bitrate_choice else "128k"
            output_tensor = _CompressionSTE.apply(tensor, "aac", bitrate, sample_rate)

            # Adjust mask length if needed
            if mask is not None:
                new_length = output_tensor.shape[-1]
                mask = mask.clone().detach()
                mask = AudioProcessor.adjust_mask_length(
                    mask, new_length, mode='nearest-exact'
                )

            return output_tensor, mask
            
        except Exception as e:
            logger.error(f"AAC compression failed: {str(e)}", exc_info=True)
            return tensor, mask

    @staticmethod
    def encodec(
        tensor: torch.Tensor,
        model_id: str = "facebook/encodec_24khz",
        bandwidth: float = 12.0,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply Encodec neural compression.
        
        Args:
            tensor: Input audio tensor
            model_id: Hugging Face model identifier
            bandwidth: Target bandwidth in kb/s
            sample_rate: Audio sample rate
            mask: Optional mask tensor
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Compressed audio and adjusted mask
        """
        try:
            output_tensor = _EncodecSTE.apply(
                tensor, model_id, bandwidth, sample_rate, mask
            )

            # Adjust mask length if needed
            if mask is not None:
                new_length = output_tensor.shape[-1]
                mask = AudioProcessor.adjust_mask_length(
                    mask, new_length, mode='nearest-exact'
                )
                
            return output_tensor, mask
            
        except Exception as e:
            logger.error(f"Encodec compression failed: {str(e)}", exc_info=True)
            return tensor, mask

    @staticmethod
    def random_equalization(
        tensor: torch.Tensor,
        freq: float,
        q: float,
        gain: float,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply parametric equalization at specified frequency.
        
        Implements a bell-shaped EQ curve that boosts or cuts frequencies
        around a center frequency. Useful for simulating frequency-specific
        enhancements or attenuations in audio systems.
        
        Args:
            tensor (torch.Tensor): Input audio tensor.
            freq (float): Center frequency in Hz for the EQ band.
                Must be positive and less than nyquist frequency.
            q (float): Q factor controlling bandwidth.
                Higher Q = narrower band. Typical range: 0.1 to 10.
            gain (float): Gain/attenuation in dB.
                Positive = boost, negative = cut. Typical range: -24 to +24 dB.
            sample_rate (int, optional): Audio sample rate in Hz.
                Defaults to DEFAULT_SAMPLE_RATE.
            mask (Optional[torch.Tensor], optional): Mask tensor.
                EQ doesn't affect mask values.
            **kwargs: Additional arguments (ignored).
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: 
                - Equalized audio with frequency-specific adjustment
                - Adjusted mask if length changed
                
        Raises:
            ValueError: If frequency or Q factor are not positive.
            
        Note:
            The Q factor relates to bandwidth as: bandwidth = freq / Q.
            A Q of 1.0 gives a bandwidth equal to the center frequency.
        """
        try:
            # Validate parameters
            if freq <= 0:
                raise ValueError(f"Frequency must be positive, got {freq} Hz")
            if q <= 0:
                raise ValueError(f"Q factor must be positive, got {q}")
                
            # Ensure frequency is below nyquist
            nyquist = sample_rate / 2.0
            if freq >= nyquist:
                logger.warning(
                    f"EQ frequency {freq} Hz exceeds nyquist {nyquist} Hz, "
                    f"clamping to {nyquist - 1} Hz"
                )
                freq = nyquist - 1
                
            # Log EQ parameters
            bandwidth = freq / q
            logger.debug(
                f"Applying parametric EQ: {freq:.1f} Hz, Q={q:.2f} "
                f"(bandwidth={bandwidth:.1f} Hz), gain={gain:.1f} dB"
            )
                
            effects = [['equalizer', f'{freq}', f'{q}', f'{gain}']]
            
            output_tensor = _SoxEffectSTE.apply(tensor, effects, sample_rate, 'stretch')

            # Adjust mask if needed
            if mask is not None:
                new_length = output_tensor.shape[-1]
                if new_length != mask.shape[-1]:
                    mask = AudioProcessor.adjust_mask_length(
                        mask, new_length, mode='nearest-exact'
                    )
                    
            return output_tensor, mask
            
        except Exception as e:
            logger.error(f"Equalization failed: {str(e)}", exc_info=True)
            return tensor, mask

    @staticmethod
    def white_noise(
        tensor: torch.Tensor,
        noise_std: float = 0.01,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Add white noise to audio.
        
        Args:
            tensor: Input audio tensor
            noise_std: Standard deviation of noise
            mask: Optional mask tensor
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Noisy audio and mask
        """
        try:
            if noise_std < 0:
                raise ValueError(f"Noise std must be non-negative, got {noise_std}")
                
            noise = torch.randn_like(tensor) * noise_std
            output_tensor = tensor + noise
            
            return output_tensor, mask
            
        except Exception as e:
            logger.error(f"White noise failed: {str(e)}", exc_info=True)
            return tensor, mask

    @staticmethod
    def shush(
        tensor: torch.Tensor,
        fraction: float = 0.1,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Zero out the quietest samples (aggressive noise gating).
        
        Args:
            tensor: Input audio tensor
            fraction: Fraction of quietest samples to remove
            mask: Optional mask tensor
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Gated audio and updated mask
        """
        try:
            # Validate fraction
            if not 0 <= fraction <= 1:
                raise ValueError(f"Fraction must be between 0 and 1, got {fraction}")
                
            output_tensor = _ShushSTE.apply(tensor, fraction)

            # Update mask for zeroed samples
            if mask is not None:
                zero_mask = (output_tensor == 0).float()
                mask = mask * (1 - zero_mask)
                
            return output_tensor, mask
            
        except Exception as e:
            logger.error(f"Shush effect failed: {str(e)}", exc_info=True)
            return tensor, mask

# =============================================================================
# Main Effect Application Function
# =============================================================================
def apply_effect(
    audio: torch.Tensor,
    effect_type: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    mask: Optional[torch.Tensor] = None,
    **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Apply audio effect with proper tensor shape handling.
    
    This is the main entry point for applying effects to audio tensors.
    It handles input validation, shape normalization, and error recovery.
    Automatically normalizes input shapes to 3D for consistent processing.
    
    Args:
        audio (torch.Tensor): Input audio tensor. Can be:
            - 1D: [samples] - treated as single channel
            - 2D: [batch/channels, samples] - auto-detected
            - 3D: [batch, channels, samples] - standard format
        effect_type (str): Name of the effect to apply. Must match a method
            name in the AudioEffects class.
        sample_rate (int, optional): Audio sample rate in Hz.
            Defaults to DEFAULT_SAMPLE_RATE (16000).
        mask (Optional[torch.Tensor], optional): Binary mask tensor indicating
            watermark presence. Should have same shape as audio or be broadcastable.
        **kwargs: Effect-specific parameters. Unknown parameters are filtered out.
        
    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: 
            - Processed audio tensor on original device
            - Processed mask tensor (if provided) or None
            Both maintain the original input shapes.
        
    Raises:
        ValueError: If audio shape is unsupported (> 3D) or effect type is invalid.
        RuntimeError: If audio and mask are on different devices.
        
    Examples:
        >>> # Apply echo to mono audio
        >>> audio = torch.randn(16000)  # 1 second at 16kHz
        >>> processed, _ = apply_effect(audio, 'echo', sample_rate=16000)
        >>> 
        >>> # Apply compression with mask
        >>> audio = torch.randn(4, 1, 16000)  # Batch of 4
        >>> mask = torch.ones_like(audio)
        >>> compressed, mask_out = apply_effect(
        ...     audio, 'mp3_lossy_compression',
        ...     bitrate_choice=['128k'],
        ...     mask=mask
        ... )
        
    Note:
        Effects that fail will log an error and return the original audio/mask.
        This fail-safe behavior ensures robustness in production pipelines.
    """
    device = audio.device
    audio = audio.clone()
    
    if mask is not None:
        mask = mask.clone()
        # Device consistency check
        if mask.device != audio.device:
            raise RuntimeError(
                f"Device mismatch: audio is on {audio.device} but mask is on {mask.device}. "
                f"Please ensure both tensors are on the same device before applying effects."
            )

    # Log effect application
    logger.debug(
        f"Applying effect '{effect_type}' to audio shape {audio.shape} "
        f"with sample rate {sample_rate} Hz"
    )
    
    # Store original shape for restoration
    original_audio_shape = audio.shape
    original_mask_shape = mask.shape if mask is not None else None
    
    # Normalize audio to 3D: [batch, channel, time]
    if audio.dim() == 1:
        # 1D: Assume single channel, single batch
        audio = audio.unsqueeze(0).unsqueeze(0)  # [time] -> [1, 1, time]
    elif audio.dim() == 2:
        # 2D: Need to determine if [batch, time] or [channels, time]
        # Heuristic: more than 16 in first dimension likely means it's time
        if audio.shape[0] > 16:
            # Likely [time, channels] or just [time, 1]
            audio = audio.unsqueeze(0)  # -> [1, channels, time] or [1, 1, time]
        else:
            # Likely [batch, time] or [channels, time]
            audio = audio.unsqueeze(1)  # -> [batch, 1, time]
    elif audio.dim() == 3:
        pass  # Already in correct format [batch, channels, time]
    else:
        raise ValueError(
            f'Unsupported audio tensor shape: {audio.shape}. '
            f'Expected 1D, 2D, or 3D tensor.'
        )

    # Normalize mask shape to match audio
    if mask is not None:
        if mask.dim() == 1:
            # 1D mask: assume same structure as audio
            mask = mask.unsqueeze(0).unsqueeze(0)  # [time] -> [1, 1, time]
        elif mask.dim() == 2:
            # 2D mask: need to match audio's batch dimension
            if mask.shape[0] != audio.shape[0]:
                # Mask has different batch size, try to broadcast
                if mask.shape[0] == 1:
                    # Broadcast single mask to all batches
                    mask = mask.expand(audio.shape[0], -1, -1)
                else:
                    mask = mask.unsqueeze(0)  # Add batch dimension
            mask = mask.unsqueeze(1) if mask.dim() == 2 else mask  # Add channel dim
        elif mask.dim() == 3:
            # 3D mask: ensure channel dimension compatibility
            if mask.shape[1] != audio.shape[1]:
                if mask.shape[1] == 1:
                    # Broadcast single channel mask to all channels
                    mask = mask.expand(-1, audio.shape[1], -1)
                else:
                    # Use only first channel of mask
                    logger.warning(
                        f"Mask has {mask.shape[1]} channels but audio has {audio.shape[1]}. "
                        f"Using first mask channel only."
                    )
                    mask = mask[:, :1]  # Take first channel
        else:
            raise ValueError(
                f'Unsupported mask tensor shape: {mask.shape}. '
                f'Expected 1D, 2D, or 3D tensor matching audio shape.'
            )

    # Get and validate effect handler
    if not hasattr(AudioEffects, effect_type):
        # List available effects for helpful error message
        available_effects = sorted([
            attr for attr in dir(AudioEffects) 
            if not attr.startswith('_') and 
            callable(getattr(AudioEffects, attr))
        ])
        raise ValueError(
            f"Invalid effect type: '{effect_type}'. "
            f"Available effects: {', '.join(available_effects)}"
        )
    
    effect_handler = getattr(AudioEffects, effect_type)
    if not isfunction(effect_handler):
        raise ValueError(
            f"Invalid effect type: '{effect_type}' is not a callable function"
        )

    # Filter kwargs based on effect handler signature
    # This prevents passing unknown parameters that would cause errors
    sig = signature(effect_handler)
    param_names = [param.name for param in sig.parameters.values()]
    
    # Filter out unknown parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in param_names}
    unknown_params = set(kwargs.keys()) - set(filtered_kwargs.keys())
    if unknown_params:
        logger.debug(
            f"Ignoring unknown parameters for effect '{effect_type}': "
            f"{', '.join(unknown_params)}"
        )

    # Add sample rate if the effect accepts it
    if 'sample_rate' in param_names:
        filtered_kwargs['sample_rate'] = sample_rate

    try:
        # Apply the effect
        result = effect_handler(audio, mask=mask, **filtered_kwargs)
        
        # Handle return value
        if isinstance(result, tuple):
            output_tensor, output_mask = result
        else:
            output_tensor = result
            output_mask = mask

        # Log dimension changes if any
        if output_tensor.shape != audio.shape:
            logger.debug(
                f"Effect '{effect_type}' changed tensor shape from "
                f"{audio.shape} to {output_tensor.shape}"
            )
            
        # Restore original shapes if needed
        # This ensures the output matches the input format
        if len(original_audio_shape) < len(output_tensor.shape):
            # Remove added dimensions
            while output_tensor.dim() > len(original_audio_shape):
                output_tensor = output_tensor.squeeze(0)
                
        if output_mask is not None and original_mask_shape is not None:
            if len(original_mask_shape) < len(output_mask.shape):
                while output_mask.dim() > len(original_mask_shape):
                    output_mask = output_mask.squeeze(0)

        # Ensure output is on original device
        return output_tensor.to(device), output_mask.to(device) if output_mask is not None else None
        
    except Exception as e:
        # Enhanced error logging with full context
        error_details = (
            f"Error applying effect '{effect_type}': {type(e).__name__}: {e}\n"
            f"Audio Shape: {audio.shape}, Device: {device}, Sample Rate: {sample_rate}\n"
            f"Mask Shape: {mask.shape if mask is not None else 'None'}, "
            f"Mask Device: {mask.device if mask is not None else 'N/A'}\n"
            f"Parameters: {filtered_kwargs}"
        )
        
        # Log full traceback for debugging
        logger.error(error_details, exc_info=True)
        
        # Fail gracefully: return original audio/mask on correct device
        # Restore original shapes before returning
        while audio.dim() > len(original_audio_shape):
            audio = audio.squeeze(0)
            
        if mask is not None and original_mask_shape is not None:
            while mask.dim() > len(original_mask_shape):
                mask = mask.squeeze(0)
                
        return audio.to(device), mask.to(device) if mask is not None else None

# =============================================================================
# Test Functions
# =============================================================================
def test_all_effects():
    """
    Test all effects in the AudioEffects class with batch processing.
    
    This function validates that all effects:
    - Preserve batch dimensions
    - Maintain binary mask values
    - Handle errors gracefully
    - Process batches correctly
    
    Returns:
        Tuple[List[str], List[Tuple[str, str]]]: Lists of passed and failed effects
    """

    # Test configuration
    batch_size = 4
    sample_rate = DEFAULT_SAMPLE_RATE
    duration = 1  # seconds
    num_samples = int(sample_rate * duration)
    num_channels = 1

    # Define test parameters for each effect
    effect_params = {
        'identity': {},
        'speed': {'speed': [0.8]},
        'resample': {'new_sample_rate': [8000]},
        'echo': {'volume_range': [(0.1, 0.3)], 'duration_range': [(0.1, 0.3)]},
        'pink_noise': {'noise_std': [0.001]},
        'lowpass_filter': {'cutoff_freq': [1000]},
        'highpass_filter': {'cutoff_freq': [1000]},
        'median_filter': {'kernel_size': [3]},
        'bandpass_filter': {
            'cutoff_freq_low': [1000],
            'cutoff_freq_high': [3000]
        },
        'smooth': {'window_size_range': [(5, 8)]},
        'amplitude_scaling': {'scale': [0.5]},
        'quantization': {'bit_depth': [8]},
        'sample_suppression': {'suppression_percentage': [0.001]},
        'random_noise': {'noise_std': [0.001]},
        'mp3_lossy_compression': {'bitrate_choice': [["64k"]]},
        'aac_lossy_compression': {'bitrate_choice': [["64k"]]},
        'encodec': {
            'model_id': ['facebook/encodec_24khz'],
            'bandwidth': [12.0]
        },
        'random_equalization': {
            'freq': [100],
            'q': [0.3],
            'gain': [-24]
        },
        'white_noise': {'noise_std': [0.001]},
        'shush': {'fraction': [0.0001]}
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Running tests on device: {device}")

    # Generate test data
    input_signal = torch.randn(batch_size, num_channels, num_samples).to(device)
    mask = torch.randint(0, 2, (batch_size, num_channels, num_samples), dtype=torch.float32).to(device)

    # Get all effect methods
    effect_methods = [
        attr for attr in dir(AudioEffects)
        if callable(getattr(AudioEffects, attr)) and not attr.startswith('_')
    ]

    # Store results
    batch_processing_results = []
    shape_mismatch_effects = []

    for effect_name in effect_methods:
        if effect_name not in effect_params:
            logger.info(f"Skipping effect {effect_name}, no parameters defined.")
            continue

        logger.info(f"Testing effect: {effect_name}")
        
        # Get parameters for the effect
        params_dict = effect_params.get(effect_name, {})
        if params_dict:
            keys, values = zip(*params_dict.items())
            params_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
        else:
            params_list = [{}]

        for params in params_list:
            try:
                # Apply effect with batch input
                effect_kwargs = params.copy()
                effect_kwargs['sample_rate'] = sample_rate

                # Process batch
                processed_tensor, processed_mask = apply_effect(
                    input_signal, effect_name, mask=mask.clone(), **effect_kwargs
                )

                # Validate batch dimension
                if processed_tensor.shape[0] != batch_size:
                    logger.error(
                        f"{effect_name}: Batch dimension changed from {batch_size} to {processed_tensor.shape[0]}"
                    )
                    shape_mismatch_effects.append((effect_name, "batch_dim_changed"))
                    continue

                # Validate channel dimension
                if processed_tensor.shape[1] != num_channels:
                    logger.error(
                        f"{effect_name}: Channel dimension changed from {num_channels} to {processed_tensor.shape[1]}"
                    )
                    shape_mismatch_effects.append((effect_name, "channel_dim_changed"))
                    continue

                # Check batch variation
                batch_variation = torch.std(processed_tensor, dim=0).mean().item()
                if batch_variation < 1e-6:
                    logger.warning(f"{effect_name}: Low batch variation detected ({batch_variation:.2e})")

                # Validate mask if present
                if processed_mask is not None:
                    if processed_mask.shape != processed_tensor.shape:
                        logger.error(
                            f"{effect_name}: Mask shape mismatch - Audio: {processed_tensor.shape}, "
                            f"Mask: {processed_mask.shape}"
                        )
                        shape_mismatch_effects.append((effect_name, "mask_shape_mismatch"))
                        continue

                    # Verify mask remains binary
                    unique_values = torch.unique(processed_mask)
                    is_binary = all(v in [0.0, 1.0] for v in unique_values.tolist())
                    if not is_binary:
                        logger.error(f"{effect_name}: Non-binary mask values detected: {unique_values.tolist()}")
                        continue

                logger.info(f"✓ {effect_name}: Passed batch processing test")
                batch_processing_results.append(effect_name)

            except Exception as e:
                logger.error(f"{effect_name}: Error during batch processing - {str(e)}", exc_info=True)
                continue

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Batch Processing Test Summary")
    logger.info("="*60)
    logger.info(f"\nPassed Effects ({len(batch_processing_results)}):")
    for effect in batch_processing_results:
        logger.info(f"  ✓ {effect}")

    if shape_mismatch_effects:
        logger.info(f"\nFailed Effects ({len(shape_mismatch_effects)}):")
        for effect, reason in shape_mismatch_effects:
            logger.info(f"  ✗ {effect}: {reason}")

    return batch_processing_results, shape_mismatch_effects

def test_echo_effect():
    """
    Test the echo effect with visualization.
    
    This function demonstrates the echo effect on a sample audio file
    and can be extended to include visualization of the results.
    """
    import torchaudio
    import numpy as np
    
    logger.info("Testing echo effect with audio file")
    
    try:
        # Load test audio
        audio_path = '../audio_samples/birds_chirping.wav'
        audio, sr = torchaudio.load(audio_path)
        audio = audio.unsqueeze(0)  # Add batch dimension: [1, channels, samples]
        original_mask = torch.ones_like(audio)

        # Apply echo effect
        echo_params = {
            'volume_range': (0.1, 0.5),
            'duration_range': (0.1, 0.5),
            'sample_rate': sr
        }
        
        output_signal, _ = AudioEffects.echo(
            audio,
            mask=original_mask.clone(),
            **echo_params
        )

        logger.info(f"Input shape: {audio.shape}")
        logger.info(f"Output shape: {output_signal.shape}")
        logger.info(f"Max input amplitude: {audio.abs().max():.4f}")
        logger.info(f"Max output amplitude: {output_signal.abs().max():.4f}")
        
        # Save output for verification
        output_path = 'echo_test_output.wav'
        torchaudio.save(output_path, output_signal.squeeze(0), sr)
        logger.info(f"Saved echo output to: {output_path}")
        
    except FileNotFoundError:
        logger.warning("Test audio file not found, using synthetic signal")
        
        # Create synthetic test signal
        sample_rate = 16000
        duration = 1.0
        freq = 440  # A4 note
        t = torch.linspace(0, duration, int(sample_rate * duration))
        audio = torch.sin(2 * np.pi * freq * t).unsqueeze(0).unsqueeze(0)
        
        # Apply echo
        output_signal, _ = AudioEffects.echo(
            audio,
            volume_range=(0.3, 0.3),
            duration_range=(0.08, 0.08),
            sample_rate=sample_rate
        )
        
        logger.info("Echo effect applied to synthetic signal")
        
    except Exception as e:
        logger.error(f"Echo test failed: {str(e)}", exc_info=True)

# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting audio effects module tests")
    
    # Run echo effect test
    test_echo_effect()
    
    # Optionally run full test suite
    # Uncomment the following line to run all effects tests
    # test_all_effects()