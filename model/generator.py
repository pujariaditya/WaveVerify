"""
Audio watermarking generator module.

This module implements a neural network-based audio watermarking generator that embeds
imperceptible watermarks into audio signals using an encoder-decoder architecture.
"""

# =============================================================================
# Standard Library Imports
# =============================================================================
import logging
import math
import os
import sys
from functools import partial
from typing import Dict, List, Optional, Tuple, Union, Any

# =============================================================================
# Third-Party Imports
# =============================================================================
import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.ml import BaseModel
from torch import nn

# =============================================================================
# Local Imports
# =============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import modules as m

# =============================================================================
# Module Configuration
# =============================================================================
logger = logging.getLogger(__name__)

# Type aliases for clarity
Array = Union[np.ndarray, List[float]]
TensorDict = Dict[str, torch.Tensor]

# =============================================================================
# Main Classes
# =============================================================================
class Generator(BaseModel):
    """
    Neural audio watermarking generator using SEANet architecture.
    
    This generator embeds imperceptible watermarks into audio signals through
    an encoder-decoder pipeline with residual connections and multi-scale processing.
    
    Attributes:
        sample_rate (int): Audio sample rate in Hz
        dimension (int): Latent space dimension
        hop_length (int): Total stride of the encoder/decoder
        encoder (SEANetEncoder): Encoder network for embedding watermarks
        decoder (SEANetDecoder): Decoder network for reconstructing audio
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels_audio: int = 1,
        dimension: int = 128,
        msg_dimension: int = 16,
        channels_enc: int = 64,
        channels_dec: int = 96,
        n_fft_base: int = 64,
        n_residual_enc: int = 2,
        n_residual_dec: int = 3,
        res_scale_enc: float = 0.5773502691896258,
        res_scale_dec: float = 0.5773502691896258,
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
        final_activation: str = 'Tanh',
        act_all: bool = False,
        expansion: int = 1,
        groups: int = -1,
        encoder_l2norm: bool = True,
        bias: bool = True,
        spec: str = 'stft',
        spec_layer: str = '1x1_zero',
        spec_compression: str = 'log',
        spec_learnable: bool = False,
        pad_mode: str = 'constant',
        causal: bool = True,
        zero_init: bool = True,
        inout_norm: bool = True,
        nbits: int = 16,
        embedding_dim: int = 64,
        embedding_layers: int = 2,
        freq_bands: int = 4,
    ) -> None:
        """
        Initialize the audio watermarking generator.
        
        Args:
            sample_rate: Audio sample rate in Hz
            channels_audio: Number of audio channels (1 for mono, 2 for stereo)
            dimension: Latent representation dimension
            msg_dimension: Watermark message dimension in bits
            channels_enc: Number of channels in encoder
            channels_dec: Number of channels in decoder
            n_fft_base: Base FFT size for spectral processing
            n_residual_enc: Number of residual blocks in encoder
            n_residual_dec: Number of residual blocks in decoder
            res_scale_enc: Residual scaling factor for encoder
            res_scale_dec: Residual scaling factor for decoder
            strides: Downsampling/upsampling factors for each layer
            activation: Activation function name
            activation_kwargs: Additional arguments for activation function
            norm: Normalization method ('weight_norm', 'batch_norm', etc.)
            norm_kwargs: Additional arguments for normalization
            kernel_size: Convolution kernel size
            last_kernel_size: Kernel size for final layer
            residual_kernel_size: Kernel size in residual blocks
            dilation_base: Base dilation factor
            skip: Skip connection type ('identity', 'none')
            final_activation: Final layer activation function
            act_all: Whether to apply activation to all layers
            expansion: Channel expansion factor
            groups: Number of groups for grouped convolution (-1 for default)
            encoder_l2norm: Whether to apply L2 normalization in encoder
            bias: Whether to use bias in convolutions
            spec: Spectral transform type ('stft', 'melspec')
            spec_layer: Spectral layer configuration
            spec_compression: Compression method for spectral features
            spec_learnable: Whether spectral transform is learnable
            pad_mode: Padding mode for convolutions
            causal: Whether to use causal convolutions
            zero_init: Whether to initialize some layers with zeros
            inout_norm: Whether to normalize input/output
            nbits: Number of bits for watermark message
            embedding_dim: Dimension of watermark embedding
            embedding_layers: Number of layers in embedding MLP
            freq_bands: Number of frequency bands for multi-band processing
            
        Raises:
            ValueError: If invalid parameters are provided
        """
        super().__init__()
        
        try:
            # Validate input parameters
            if sample_rate <= 0:
                raise ValueError(f"Sample rate must be positive, got {sample_rate}")
            if dimension <= 0:
                raise ValueError(f"Dimension must be positive, got {dimension}")
            if msg_dimension <= 0:
                raise ValueError(f"Message dimension must be positive, got {msg_dimension}")
            
            # Store configuration
            self.nbits = nbits
            self.ratios = strides
            self.dimension = dimension
            self.sample_rate = sample_rate
            self.hop_length = int(np.prod(self.ratios))
            
            logger.info(f"Initializing Generator with sample_rate={sample_rate}, "
                       f"dimension={dimension}, msg_dimension={msg_dimension}, "
                       f"hop_length={self.hop_length}")
            
            # Initialize encoder with watermark embedding capability
            # Note: embedding_dim and embedding_layers might not be in config
            # but are required for msg_embedding functionality
            self.encoder = m.SEANetEncoder(
                channels=channels_audio,
                dimension=dimension,
                msg_dimension=msg_dimension,
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
                inout_norm=inout_norm,
                embedding_dim=embedding_dim,
                embedding_layers=embedding_layers,
                freq_bands=freq_bands
            )
            
            # Initialize decoder for audio reconstruction
            self.decoder = m.SEANetDecoder(
                channels=channels_audio,
                dimension=dimension,
                n_filters=channels_dec,
                n_residual_layers=n_residual_dec,
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
                final_activation=final_activation,
                act_all=act_all,
                expansion=expansion,
                groups=groups,
                bias=bias,
                res_scale=res_scale_dec,
                pad_mode=pad_mode,
                zero_init=zero_init,
                inout_norm=inout_norm
            )
            
            logger.info("Generator initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Generator: {str(e)}", exc_info=True)
            raise

    def preprocess(
        self,
        audio_data: torch.Tensor,
        sample_rate: Optional[int] = None
    ) -> torch.Tensor:
        """
        Preprocess audio data for encoding.
        
        Ensures audio is properly padded to match the model's hop length requirements.
        
        Args:
            audio_data: Input audio tensor of shape [B, C, T]
            sample_rate: Sample rate of the audio (must match model's sample rate)
            
        Returns:
            Preprocessed audio tensor with proper padding
            
        Raises:
            AssertionError: If sample rate doesn't match model's expected rate
        """
        try:
            # Use model's sample rate if not provided
            if sample_rate is None:
                sample_rate = self.sample_rate
            
            # Validate sample rate matches expected
            assert sample_rate == self.sample_rate, \
                f"Sample rate mismatch: expected {self.sample_rate}, got {sample_rate}"
            
            # Calculate required padding for hop length alignment
            audio_length = audio_data.shape[-1]
            required_length = math.ceil(audio_length / self.hop_length) * self.hop_length
            right_pad = required_length - audio_length
            
            # Apply padding if necessary
            if right_pad > 0:
                audio_data = nn.functional.pad(audio_data, (0, right_pad))
                logger.debug(f"Applied padding of {right_pad} samples to align with hop_length")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}", exc_info=True)
            raise

    def encode(
        self,
        audio_data: torch.Tensor,
        msg: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode audio data with watermark message.
        
        Args:
            audio_data: Audio tensor of shape [B, C, T] where B is batch size,
                       C is channels, and T is time samples
            msg: Watermark message tensor of shape [B, msg_dimension]
            
        Returns:
            Latent representation tensor of shape [B, dimension, T'] where
            T' is the compressed time dimension
            
        Raises:
            RuntimeError: If encoding fails
        """
        try:
            # ------------------------------------------------------------------
            # Use input tensor's device for all operations
            # ------------------------------------------------------------------
            device = audio_data.device

            if msg.device != device:
                msg = msg.to(device)

            logger.debug(
                f"Encoding audio with shape {audio_data.shape} on {audio_data.device} "
                f"and message shape {msg.shape} on {msg.device}"
            )
            
            # Encode audio with watermark message
            latent_codes = self.encoder(audio_data, msg)
            
            logger.debug(f"Generated latent codes with shape {latent_codes.shape}")
            return latent_codes
            
        except Exception as e:
            logger.error(f"Error during encoding: {str(e)}", exc_info=True)
            raise RuntimeError(f"Encoding failed: {str(e)}") from e

    def decode(self, latent_codes: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation back to audio.
        
        Args:
            latent_codes: Latent tensor of shape [B, dimension, T']
            
        Returns:
            Reconstructed audio tensor of shape [B, C, T]
            
        Raises:
            RuntimeError: If decoding fails
        """
        try:
            logger.debug(f"Decoding latent codes with shape {latent_codes.shape}")
            
            # Decode latent representation
            reconstructed_audio = self.decoder(latent_codes)
            
            logger.debug(f"Reconstructed audio with shape {reconstructed_audio.shape}")
            return reconstructed_audio
            
        except Exception as e:
            logger.error(f"Error during decoding: {str(e)}", exc_info=True)
            raise RuntimeError(f"Decoding failed: {str(e)}") from e

    def forward(
        self,
        audio_signal: AudioSignal,
        msg: torch.Tensor,
        sample_rate: Optional[int] = None
    ) -> AudioSignal:
        """
        Forward pass: embed watermark into audio signal.
        
        Args:
            audio_signal: Input audio signal object containing audio data
            msg: Watermark message tensor of shape [B, msg_dimension]
            sample_rate: Sample rate in Hz (must match model's sample rate if provided)
            
        Returns:
            Watermarked audio signal with same duration as input
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If forward pass encounters errors
        """
        try:
            # Validate inputs
            if not isinstance(audio_signal, AudioSignal):
                raise ValueError("Input must be an AudioSignal object")

            # Use input tensor's device for all operations
            device = audio_signal.audio_data.device

            # Move AudioSignal to ensure consistency
            if audio_signal.device != device:
                audio_signal = audio_signal.to(device)

            # Ensure message tensor is on the same device
            if msg.device != device:
                msg = msg.to(device)

            # Extract audio data and original length
            original_length = audio_signal.audio_data.shape[-1]
            audio_data = audio_signal.audio_data
            
            logger.debug(f"Processing audio signal with shape {audio_data.shape}")
            
            # Encode with watermark
            latent_representation = self.encode(audio_data, msg)
            
            # Decode to reconstruct watermarked audio
            watermarked_audio = self.decode(latent_representation)
            
            # Trim to original length (remove padding)
            watermarked_audio = watermarked_audio[..., :original_length]
            
            # Create output audio signal
            watermarked_signal = AudioSignal(
                watermarked_audio,
                sample_rate=audio_signal.sample_rate
            )
            
            logger.debug(f"Generated watermarked audio with shape {watermarked_audio.shape}")
            return watermarked_signal
            
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}", exc_info=True)
            raise RuntimeError(f"Forward pass failed: {str(e)}") from e


# =============================================================================
# Test Functions
# =============================================================================
def print_model_summary(model: Generator) -> None:
    """
    Print a summary of the model architecture and parameters.
    
    Args:
        model: Generator model instance
    """
    try:
        # Add parameter count to module representations
        for name, module in model.named_modules():
            original_repr = module.extra_repr()
            param_count = sum(np.prod(p.size()) for p in module.parameters())
            
            # Create enhanced representation with parameter count
            def enhanced_repr(original: str, params: float) -> str:
                return f"{original} {params/1e6:<.3f}M params."
            
            setattr(module, "extra_repr", partial(enhanced_repr, original=original_repr, params=param_count))
        
        # Calculate total parameters
        total_params = sum(np.prod(p.size()) for p in model.parameters())
        logger.info(f"Total model parameters: {total_params:,}")
        print(f"Total # of params: {total_params:,}")
        
    except Exception as e:
        logger.error(f"Error printing model summary: {str(e)}", exc_info=True)


def test_receptive_field(model: Generator, device: str = "cpu") -> int:
    """
    Test the receptive field of the model using gradient analysis.
    
    Args:
        model: Generator model instance
        device: Device to run the test on
        
    Returns:
        Receptive field size in samples
    """
    try:
        # Create test input
        test_length = 16000  # 1 second at 16kHz
        test_audio = AudioSignal(
            torch.randn(1, 1, test_length),
            sample_rate=16000
        ).to(device)
        
        # Enable gradient computation
        test_audio.audio_data.requires_grad_(True)
        test_audio.audio_data.retain_grad()
        
        # Create test message
        test_message = torch.randint(0, 2, (1, 16), dtype=torch.long).to(device)
        
        # Forward pass
        output = model(test_audio, test_message)
        logger.info(f"Input shape: {test_audio.audio_data.shape}")
        logger.info(f"Output shape: {output.audio_data.shape}")
        
        # Create gradient at center position
        grad_output = torch.zeros_like(output.audio_data)
        center_position = grad_output.shape[-1] // 2
        grad_output[:, :, center_position] = 1
        
        # Backward pass
        output.audio_data.backward(grad_output)
        
        # Analyze gradient to determine receptive field
        gradient_map = test_audio.audio_data.grad.squeeze(0)
        # Sum across channels to get overall influence
        gradient_influence = (gradient_map != 0).sum(0)
        # Count non-zero positions
        receptive_field_size = (gradient_influence != 0).sum().item()
        
        logger.info(f"Receptive field: {receptive_field_size} samples")
        return receptive_field_size
        
    except Exception as e:
        logger.error(f"Error testing receptive field: {str(e)}", exc_info=True)
        return -1


def test_watermark_generation(model: Generator, device: str = "cpu") -> None:
    """
    Test watermark generation with a sample audio.
    
    Args:
        model: Generator model instance
        device: Device to run the test on
    """
    try:
        # Create 60-second test audio
        test_duration_seconds = 60
        test_sample_rate = 16000
        test_length = test_sample_rate * test_duration_seconds
        
        # Generate random audio signal
        test_audio_signal = AudioSignal(
            torch.randn(1, 1, test_length),
            sample_rate=test_sample_rate,
            device=device
        )
        
        # Generate random 16-bit watermark message
        watermark_message = torch.randint(0, 2, (1, 16), dtype=torch.long).to(device)
        
        logger.info(f"Testing watermark generation with {test_duration_seconds}s audio")
        logger.info(f"Watermark message: {watermark_message.squeeze().tolist()}")
        
        # Generate watermarked audio
        if hasattr(model, 'generate'):
            model.generate(test_audio_signal, watermark_message, verbose=True)
        else:
            # If generate method doesn't exist, use forward pass
            watermarked = model(test_audio_signal, watermark_message)
            logger.info(f"Watermarked audio shape: {watermarked.audio_data.shape}")
        
    except Exception as e:
        logger.error(f"Error testing watermark generation: {str(e)}", exc_info=True)


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Enable anomaly detection for debugging
        torch.autograd.set_detect_anomaly(True)
        logger.info("Anomaly detection enabled")
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Initialize model
        model = Generator().to(device)
        logger.info("Model initialized successfully")
        
        # Print model summary
        print_model_summary(model)
        
        # Test receptive field
        receptive_field = test_receptive_field(model, device)
        print(f"Receptive field: {receptive_field}")
        
        # Test watermark generation
        test_watermark_generation(model, device)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise