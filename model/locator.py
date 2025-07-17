# =============================================================================
# IMPORTS
# =============================================================================
# Standard library imports
from __future__ import annotations
import logging
import math
import os
import sys
import time
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from audiotools import AudioSignal
from audiotools.ml import BaseModel

# Local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
import modules as m

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# =============================================================================
# CONSTANTS AND TYPE ALIASES
# =============================================================================
Array = Union[np.ndarray, List[float]]
TensorType = torch.Tensor
AudioTensor = torch.Tensor  # Shape: [batch_size, channels, time]
LatentTensor = torch.Tensor  # Shape: [batch_size, dimension, time]

# =============================================================================
# MAIN CLASS
# =============================================================================
class Locator(BaseModel):
    """Neural network model for audio watermark localization.
    
    This model encodes audio signals into a latent representation and processes
    them through transposed convolutions to localize watermarks in the audio stream.
    
    Args:
        sample_rate: Audio sample rate in Hz.
        channels_audio: Number of audio channels (1 for mono, 2 for stereo).
        dimension: Dimension of the latent representation.
        channels_enc: Number of channels in the encoder.
        n_fft_base: Base FFT size for spectral processing.
        n_residual_enc: Number of residual layers in encoder.
        res_scale_enc: Residual scaling factor for stability.
        strides: List of stride values for downsampling.
        activation: Activation function name.
        activation_kwargs: Additional arguments for activation function.
        norm: Normalization method to use.
        norm_kwargs: Additional arguments for normalization.
        kernel_size: Kernel size for convolutional layers.
        last_kernel_size: Kernel size for the last layer.
        residual_kernel_size: Kernel size for residual connections.
        dilation_base: Base dilation rate.
        skip: Type of skip connection.
        act_all: Whether to apply activation to all layers.
        expansion: Channel expansion factor.
        groups: Number of groups for grouped convolution.
        encoder_l2norm: Whether to apply L2 normalization in encoder.
        bias: Whether to use bias in convolutional layers.
        spec: Spectral transformation type.
        spec_compression: Type of spectral compression.
        pad_mode: Padding mode for convolutions.
        causal: Whether to use causal convolutions.
        zero_init: Whether to initialize with zeros.
        inout_norm: Whether to normalize input/output.
        output_dim: Output dimension after transposed convolution.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels_audio: int = 1,
        dimension: int = 64,
        channels_enc: int = 32,
        n_fft_base: int = 64,
        n_residual_enc: int = 1,
        res_scale_enc: float = 0.5773502691896258,
        strides: List[int] = [8, 4],  
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
        output_dim: int = 32
    ) -> None:

        super().__init__()
        
        try:
            # Store configuration parameters
            self.ratios = strides
            self.dimension = dimension
            self.output_dim = output_dim
            self.sample_rate = sample_rate
            
            # Calculate derived parameters
            self.hop_length = int(np.prod(self.ratios))  # Total downsampling factor
            self.stride = self.kernel_size = int(np.prod(self.ratios))  # Ensure kernel matches stride
            
            logger.info(f"Initializing Locator with sample_rate={sample_rate}Hz, "
                       f"dimension={dimension}, hop_length={self.hop_length}")
        except Exception as e:
            logger.error(f"Error initializing Locator parameters: {str(e)}", exc_info=True)
            raise

        # Initialize encoder network
        try:
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
            logger.debug("SEANetEncoder initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SEANetEncoder: {str(e)}", exc_info=True)
            raise

        # Initialize transposed convolution for upsampling
        try:
            self.reverse_convolution = nn.ConvTranspose1d(
                in_channels=self.dimension,
                out_channels=self.output_dim,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=0
            )
            
            # Final layer to produce single-channel output
            self.last_layer = nn.Conv1d(self.output_dim, 1, 1)
            
            logger.debug(f"Initialized transposed convolution and final layer")
        except Exception as e:
            logger.error(f"Failed to initialize convolution layers: {str(e)}", exc_info=True)
            raise

    def preprocess(self, audio_data: AudioTensor, sample_rate: Optional[int] = None) -> Tuple[int, AudioTensor]:
        """Preprocess audio data for model input.
        
        Args:
            audio_data: Input audio tensor of shape [B, C, T].
            sample_rate: Sample rate in Hz. If None, uses model's sample rate.
            
        Returns:
            Tuple of (original_length, padded_audio_data)
            
        Raises:
            ValueError: If sample rate doesn't match model's expected rate.
        """
        try:
            # Validate sample rate
            if sample_rate is None:
                sample_rate = self.sample_rate
            
            if sample_rate != self.sample_rate:
                raise ValueError(f"Sample rate mismatch: expected {self.sample_rate}, got {sample_rate}")
            
            # Calculate padding to make length divisible by hop_length
            original_length = audio_data.shape[-1]
            frames_needed = math.ceil(original_length / self.hop_length)
            target_length = frames_needed * self.hop_length
            right_pad = target_length - original_length
            
            # Apply padding if necessary
            if right_pad > 0:
                audio_data = F.pad(audio_data, (0, right_pad), mode='constant', value=0)
                logger.debug(f"Padded audio from {original_length} to {target_length} samples")
            
            return original_length, audio_data
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}", exc_info=True)
            raise

    def decode(
        self,
        audio_data: AudioTensor,
        original_frame_count: int
    ) -> TensorType:
        """Process audio through encoder and decoder to localize watermarks.
        
        Args:
            audio_data: Input audio tensor of shape [B, 1, T].
            original_frame_count: Original number of frames before padding.
            
        Returns:
            Tensor of shape [B, 1, T] representing watermark localization.
            
        Raises:
            RuntimeError: If tensor operations fail.
        """
        try:
            # Encode audio to latent representation
            latent_representation = self.encoder(audio_data, None)
            logger.debug(f"Encoded to latent shape: {latent_representation.shape}")
            
            # Upsample using transposed convolution
            upsampled = self.reverse_convolution(latent_representation)
            logger.debug(f"Upsampled to shape: {upsampled.shape}")
            
            # Trim to original length
            trimmed = upsampled[:, :, :original_frame_count]
            
            # Apply final layer to get single-channel output
            result = self.last_layer(trimmed)
            logger.debug(f"Final output shape: {result.shape}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in decode: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to decode audio: {str(e)}")


    def forward(self, audio_signal: AudioSignal) -> TensorType:
        """Forward pass for watermark localization.
        
        Args:
            audio_signal: Input audio signal object containing audio data.
            
        Returns:
            Tensor of shape [B, 1, T] representing watermark localization scores.
            
        Raises:
            ValueError: If input validation fails.
            RuntimeError: If model execution fails.
        """
        try:
            # Validate input
            if audio_signal is None or audio_signal.audio_data is None:
                raise ValueError("Invalid audio signal input")
            
            # Get original length and audio data
            original_length = audio_signal.audio_data.shape[-1]
            audio_data = audio_signal.audio_data
            
            logger.debug(f"Processing audio with shape: {audio_data.shape}")
            
            # Process through the model
            result = self.decode(audio_data, original_length)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}", exc_info=True)
            raise

# =============================================================================
# TEST AND DEMO CODE
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    # Configure logging for test
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test Locator model')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu/cuda)')
    parser.add_argument('--num-runs', type=int, default=100, help='Number of timing runs')
    parser.add_argument('--audio-length', type=int, default=16000, help='Test audio length in samples')
    args = parser.parse_args()
    
    try:
        # Enable anomaly detection for debugging
        torch.autograd.set_detect_anomaly(True)
        
        # Initialize model
        logger.info(f"Initializing Locator model on {args.device}")
        model = Locator().to(args.device)
        
        # Add parameter count to module representations
        for name, module in model.named_modules():
            original_repr = module.extra_repr()
            param_count = sum([np.prod(p.size()) for p in module.parameters()])
            enhance_repr = lambda o, p: o + f" {p/1e6:<.3f}M params."
            setattr(module, "extra_repr", partial(enhance_repr, o=original_repr, p=param_count))
        
        # Calculate total parameters
        total_params = sum([np.prod(p.size()) for p in model.parameters()])
        logger.info(f"Total model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        
        # Create test audio signal
        test_audio = AudioSignal(
            torch.randn(1, 1, args.audio_length), 
            sample_rate=16000
        ).to(model.device)
        test_audio.audio_data.requires_grad_(True)
        test_audio.audio_data.retain_grad()
        
        # Warm-up run
        logger.info("Performing warm-up run...")
        _ = model(test_audio)
        
        # Timing runs
        logger.info(f"Running {args.num_runs} timing iterations...")
        inference_times = []
        
        for i in range(args.num_runs):
            start_time = time.time()
            _ = model(test_audio)
            
            # Synchronize for accurate GPU timing
            if model.device.type == 'cuda':
                torch.cuda.synchronize()
                
            inference_times.append(time.time() - start_time)
        
        # Calculate and display statistics
        mean_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        
        logger.info(f"\nInference time statistics for {args.audio_length/16000:.1f}s audio:")
        logger.info(f"  Mean: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")
        logger.info(f"  Min:  {min_time*1000:.2f} ms")
        logger.info(f"  Max:  {max_time*1000:.2f} ms")
        logger.info(f"  Real-time factor: {(args.audio_length/16000)/mean_time:.2f}x")
        
        # Analyze receptive field
        if test_audio.audio_data.grad is not None:
            gradient_map = test_audio.audio_data.grad.squeeze(0)
            # Count non-zero gradient positions
            non_zero_gradient = (gradient_map != 0).sum(0)
            receptive_field_size = (non_zero_gradient != 0).sum()
            logger.info(f"\nReceptive field size: {receptive_field_size.item()} samples")
        
        # Test with longer audio
        logger.info("\nTesting with 60-second audio...")
        long_audio = AudioSignal(
            torch.randn(1, 1, 16000 * 60), 
            sample_rate=16000
        ).to(model.device)
        
        start_time = time.time()
        _ = model(long_audio)
        if model.device.type == 'cuda':
            torch.cuda.synchronize()
        long_inference_time = time.time() - start_time
        
        logger.info(f"60-second audio inference time: {long_inference_time:.3f}s")
        logger.info(f"Real-time factor: {60.0/long_inference_time:.2f}x")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        raise