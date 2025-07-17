"""
Audio watermark detector module for real-time watermarking system.

This module provides the Detector class which implements a neural network-based
approach for detecting watermarks embedded in audio signals.
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import logging
import math
import os
import sys
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.ml import BaseModel
from torch import nn

# Local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import modules as m

# =============================================================================
# MODULE CONFIGURATION
# =============================================================================

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler if logger has no handlers
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Type aliases
Array = Union[np.ndarray, list]

# Constants
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MESSAGE_THRESHOLD = 0.5
MIN_AUDIO_LENGTH = 1  # Minimum audio length in samples

# =============================================================================
# MAIN CLASS
# =============================================================================

class Detector(BaseModel):
    """
    Neural network-based watermark detector for audio signals.
    
    This class implements a convolutional neural network that detects watermarks
    embedded in audio signals. It uses an encoder architecture followed by
    transposed convolution and classification layers to extract watermark bits.
    
    Attributes:
        nbits (int): Number of watermark bits to detect
        ratios (List[int]): Stride ratios for the encoder layers
        dimension (int): Dimensionality of the encoded representation
        output_dim (int): Output dimension before final classification
        sample_rate (int): Expected sample rate of input audio
        hop_length (int): Hop length for processing audio frames
        encoder (m.SEANetEncoder): Encoder network for feature extraction
        reverse_convolution (nn.ConvTranspose1d): Transposed convolution layer
        last_layer (nn.Conv1d): Final classification layer
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels_audio: int = 1,
        dimension: int = 128,
        channels_enc: int = 64,
        n_fft_base: int = 64,
        n_residual_enc: int = 2,
        res_scale_enc: float = 0.5773502691896258,
        strides: List[int] = [8, 5, 4, 2],
        activation: str = 'ELU',
        activation_kwargs: Dict[str, Any] = {'alpha': 1.0},
        norm: str = 'weight_norm',
        norm_kwargs: Dict[str, Any] = {},
        kernel_size: int = 5,
        last_kernel_size: int = 5,
        residual_kernel_size: int = 5,
        dilation_base: int = 1,
        skip: str = 'identity',
        act_all: bool = False,
        expansion: int = 1,
        groups: int = -1,
        encoder_l2norm: bool = True,
        bias: bool = True,
        spec: str = 'stft',
        spec_compression: str = 'log',
        pad_mode: str = 'constant',
        causal: bool = True,
        zero_init: bool = True,
        inout_norm: bool = True,
        output_dim: int = 32,
        nbits: int = 16
    ) -> None:
        """
        Initialize the Detector model.
        
        Args:
            sample_rate: Audio sample rate in Hz
            channels_audio: Number of audio channels (1 for mono, 2 for stereo)
            dimension: Dimension of the latent representation
            channels_enc: Number of channels in encoder layers
            n_fft_base: Base FFT size for spectral processing
            n_residual_enc: Number of residual layers in encoder
            res_scale_enc: Scaling factor for residual connections
            strides: List of stride values for each encoder layer
            activation: Activation function name
            activation_kwargs: Arguments for activation function
            norm: Normalization method ('weight_norm', 'batch_norm', etc.)
            norm_kwargs: Arguments for normalization
            kernel_size: Kernel size for convolutional layers
            last_kernel_size: Kernel size for the last encoder layer
            residual_kernel_size: Kernel size for residual blocks
            dilation_base: Base dilation rate
            skip: Skip connection type ('identity', 'residual', etc.)
            act_all: Whether to apply activation to all layers
            expansion: Channel expansion factor
            groups: Number of groups for grouped convolution (-1 for depthwise)
            encoder_l2norm: Whether to apply L2 normalization in encoder
            bias: Whether to use bias in convolutional layers
            spec: Spectral transform type ('stft', 'mel', etc.)
            spec_compression: Compression method for spectral features
            pad_mode: Padding mode for convolutions
            causal: Whether to use causal convolutions
            zero_init: Whether to initialize some weights to zero
            inout_norm: Whether to normalize input/output
            output_dim: Dimension before final classification layer
            nbits: Number of watermark bits to detect
            
        Raises:
            ValueError: If invalid parameters are provided
        """
        try:
            super().__init__()
            
            # Validate inputs
            if sample_rate <= 0:
                raise ValueError(f"Invalid sample_rate: {sample_rate}. Must be positive.")
            if nbits <= 0:
                raise ValueError(f"Invalid nbits: {nbits}. Must be positive.")
            if not strides or any(s <= 0 for s in strides):
                raise ValueError(f"Invalid strides: {strides}. All values must be positive.")
            
            # Store configuration
            self.nbits = nbits
            self.ratios = strides
            self.dimension = dimension
            self.output_dim = output_dim
            self.sample_rate = sample_rate
            
            # Calculate derived parameters
            self.hop_length = np.prod(self.ratios)
            self.stride = self.kernel_size = np.prod(self.ratios)
            
            logger.info(f"Initializing Detector with {nbits} bits, sample_rate={sample_rate}Hz")
            
            # Initialize encoder network
            self.encoder = m.SEANetEncoder(
                channels=channels_audio,
                dimension=dimension,
                n_filters=channels_enc,
                n_fft_base=n_fft_base,
                n_residual_layers=n_residual_enc,
                ratios=strides,
                activation=activation,
                activation_params=activation_kwargs,
                norm=norm,
                norm_params=norm_kwargs,
                kernel_size=kernel_size,
                last_kernel_size=last_kernel_size,
                residual_kernel_size=residual_kernel_size,
                dilation_base=dilation_base,
                skip=skip,
                causal=causal,
                act_all=act_all,
                expansion=expansion,
                groups=groups,
                l2norm=encoder_l2norm,
                bias=bias,
                spec=spec,
                spec_compression=spec_compression,
                res_scale=res_scale_enc,
                pad_mode=pad_mode,
                zero_init=zero_init,
                inout_norm=inout_norm
            )
            
            # Initialize reverse convolution layer
            self.reverse_convolution = torch.nn.ConvTranspose1d(
                in_channels=self.dimension,
                out_channels=self.output_dim,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=0
            )
            
            # Initialize final classification layer
            self.last_layer = nn.Conv1d(output_dim, nbits, 1)
            
            logger.info("Detector initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize Detector: {str(e)}", exc_info=True)
            raise

    def preprocess(
        self, 
        audio_data: torch.Tensor, 
        sample_rate: Optional[int] = None
    ) -> Tuple[int, torch.Tensor]:
        """
        Preprocess audio data for detection.
        
        This method validates the sample rate and pads the audio to ensure
        it has a length that is a multiple of hop_length for proper processing.
        
        Args:
            audio_data: Input audio tensor of shape (batch, channels, length)
            sample_rate: Sample rate of the audio. If None, uses model's sample_rate
            
        Returns:
            Tuple containing:
                - Original length of the audio
                - Padded audio tensor
                
        Raises:
            AssertionError: If sample rate doesn't match model's expected rate
            ValueError: If audio_data has invalid shape
        """
        try:
            # Validate sample rate
            if sample_rate is None:
                sample_rate = self.sample_rate
            assert sample_rate == self.sample_rate, \
                f"Sample rate mismatch: expected {self.sample_rate}, got {sample_rate}"
            
            # Validate audio shape
            if audio_data.dim() != 3:
                raise ValueError(f"Expected 3D tensor, got {audio_data.dim()}D")
            
            # Get original length
            length = audio_data.shape[-1]
            
            # Calculate padding needed to make length divisible by hop_length
            right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
            
            # Apply padding if necessary
            if right_pad > 0:
                audio_data = nn.functional.pad(audio_data, (0, right_pad))
                logger.debug(f"Padded audio from {length} to {audio_data.shape[-1]} samples")
            
            return length, audio_data
            
        except Exception as e:
            logger.error(f"Error in preprocess: {str(e)}", exc_info=True)
            raise

    def decode(
        self, 
        audio_data: torch.Tensor, 
        orig_nframes: int
    ) -> torch.Tensor:
        """
        Decode watermark from preprocessed audio data.
        
        This method passes the audio through the encoder, applies reverse
        convolution, and produces watermark bit predictions.
        
        Args:
            audio_data: Preprocessed audio tensor of shape (batch, channels, length)
            orig_nframes: Original number of frames before padding
            
        Returns:
            Tensor of shape (batch, nbits, time) containing watermark predictions
            
        Raises:
            RuntimeError: If tensor operations fail
        """
        try:
            # Pass through encoder to get latent representation
            encoded_features = self.encoder(audio_data, None)
            
            # Apply reverse convolution to expand temporal dimension
            upsampled_features = self.reverse_convolution(encoded_features)
            
            # Trim to original length
            upsampled_features = upsampled_features[:, :, :orig_nframes]
            
            # Apply final classification layer to get bit predictions
            bit_predictions = self.last_layer(upsampled_features)
            
            logger.debug(f"Decoded shape: {bit_predictions.shape}")
            
            return bit_predictions
            
        except Exception as e:
            logger.error(f"Error in decode: {str(e)}", exc_info=True)
            raise

    def postprocess(
        self, 
        result: torch.Tensor, 
        message_threshold: float = DEFAULT_MESSAGE_THRESHOLD
    ) -> torch.Tensor:
        """
        Post-process decoder output to extract binary watermark bits.
        
        This method applies softmax normalization, temporal averaging,
        and thresholding to convert raw predictions to binary bits.
        
        Args:
            result: Raw decoder output of shape (batch, nbits, time)
            message_threshold: Threshold for converting probabilities to binary bits
            
        Returns:
            Binary tensor of shape (batch, nbits) containing detected watermark bits
            
        Raises:
            ValueError: If message_threshold is not in range [0, 1]
        """
        try:
            # Validate threshold
            if not 0 <= message_threshold <= 1:
                raise ValueError(f"message_threshold must be in [0, 1], got {message_threshold}")
            
            # Apply softmax normalization across bit dimension
            result = torch.softmax(result, dim=1)
            
            # Average predictions across time dimension
            decoded_message = result.mean(dim=-1)
            
            # Apply sigmoid to get probabilities
            message_probabilities = torch.sigmoid(decoded_message)
            
            # Threshold to get binary bits
            message_bits = torch.gt(message_probabilities, message_threshold).int()
            
            logger.debug(f"Detected {message_bits.sum().item()} positive bits out of {self.nbits}")
            
            return message_bits
            
        except Exception as e:
            logger.error(f"Error in postprocess: {str(e)}", exc_info=True)
            raise

    def forward(self, audio_signal: AudioSignal) -> torch.Tensor:
        """
        Forward pass for watermark detection.
        
        Args:
            audio_signal: AudioSignal object containing audio data
            
        Returns:
            Raw logits tensor of shape (batch, nbits, time)
            
        Raises:
            RuntimeError: If forward pass fails
        """
        try:
            # Extract audio data and length
            length = audio_signal.audio_data.shape[-1]
            audio_data = audio_signal.audio_data
            
            # Decode watermark
            result = self.decode(audio_data, length)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}", exc_info=True)
            raise

    def detect(
        self, 
        audio_signal: AudioSignal, 
        verbose: bool = False
    ) -> torch.Tensor:
        """
        High-level method to detect watermark bits from audio.
        
        This method runs the full detection pipeline in evaluation mode
        (no gradients) and returns binary watermark bits.
        
        Args:
            audio_signal: AudioSignal object containing audio to analyze
            verbose: Whether to print detection results
            
        Returns:
            Binary tensor of shape (batch, nbits) containing detected bits
            
        Raises:
            RuntimeError: If detection fails
        """
        try:
            with torch.no_grad():
                # Run forward pass to get raw predictions
                raw_logits = self(audio_signal)
                
                # Post-process to get binary bits
                bits = self.postprocess(raw_logits)
                
                if verbose:
                    batch_size = bits.shape[0]
                    for i in range(batch_size):
                        detected_bits = bits[i].cpu().numpy()
                        bit_string = ''.join(map(str, detected_bits))
                        logger.info(f"Batch {i}: Detected bits: {bit_string}")
                        print(f"Detection complete for batch {i}: {bit_string}")
                
            return bits
            
        except Exception as e:
            logger.error(f"Error in detect: {str(e)}", exc_info=True)
            raise


# =============================================================================
# MAIN BLOCK - Testing and Demonstration
# =============================================================================

if __name__ == "__main__":
    import numpy as np
    from functools import partial
    
    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)
    
    try:
        # Initialize model
        device = "cpu"
        model = Detector().to(device)
        logger.info(f"Model initialized on {device}")
        
        # Add parameter count information to model representation
        for module_name, module in model.named_modules():
            original_repr = module.extra_repr()
            param_count = sum([np.prod(p.size()) for p in module.parameters()])
            
            # Create enhanced representation function
            enhanced_repr_fn = lambda orig, count: orig + f" {count/1e6:<.3f}M params."
            setattr(module, "extra_repr", partial(enhanced_repr_fn, orig=original_repr, count=param_count))
        
        # Calculate and display total parameters
        total_params = sum([np.prod(p.size()) for p in model.parameters()])
        print(f"Total # of params: {total_params:,}")
        logger.info(f"Total model parameters: {total_params:,}")
        
        # Test with sample audio
        test_duration_seconds = 1
        test_length = DEFAULT_SAMPLE_RATE * test_duration_seconds
        test_audio = AudioSignal(
            torch.randn(1, 1, test_length), 
            sample_rate=DEFAULT_SAMPLE_RATE
        ).to(model.device)
        test_audio.audio_data.requires_grad_(True)
        test_audio.audio_data.retain_grad()
        
        # Forward pass
        output = model(test_audio)
        print(f"Input shape: {test_audio.audio_data.shape}")
        print(f"Output shape: {output.shape}")
        
        # Calculate receptive field using gradient backpropagation
        gradient_tensor = torch.zeros_like(output)
        gradient_tensor[:, gradient_tensor.shape[1] // 2] = 1  # Set gradient at middle channel
        output.backward(gradient_tensor)
        
        # Analyze gradient map to determine receptive field
        gradient_map = test_audio.audio_data.grad.squeeze(0)
        non_zero_gradient = (gradient_map != 0).sum(0)  # Sum across feature dimension
        receptive_field_size = (non_zero_gradient != 0).sum()
        
        print(f"Receptive field: {receptive_field_size.item()} samples")
        logger.info(f"Calculated receptive field: {receptive_field_size.item()} samples")
        
        # Test detection on longer audio
        test_minutes = 1
        long_test_audio = AudioSignal(
            torch.randn(1, 1, DEFAULT_SAMPLE_RATE * 60 * test_minutes), 
            DEFAULT_SAMPLE_RATE
        )
        
        # Run detection
        print(f"\nTesting detection on {test_minutes} minute(s) of audio...")
        detected_bits = model.detect(long_test_audio, verbose=True)
        
    except Exception as e:
        logger.error(f"Error in main block: {str(e)}", exc_info=True)
        raise