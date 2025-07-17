# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Encodec SEANet-based encoder and decoder implementation."""

# =============================================================================
# IMPORTS
# =============================================================================
# Standard library imports
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# Local imports
from . import (
    SConv1d,
    SConvTranspose1d,
    CausalSTFT
)
from .audio_modules import STFT

# Configure logger
logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def dws_conv_block(
    act: nn.Module,
    activation_params: Dict[str, Any],
    in_chs: int,
    out_chs: int,
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    norm: str = "weight_norm",
    norm_params: Dict[str, Any] = {},
    causal: bool = False,
    pad_mode: str = 'constant',
    act_all: bool = False,
    transposed: bool = False,
    expansion: int = 1,
    groups: int = -1,
    bias: bool = True,
) -> List[nn.Module]:
    """Creates a depth-wise separable convolution block.
    
    Args:
        act: Activation module class.
        activation_params: Parameters for activation function.
        in_chs: Number of input channels.
        out_chs: Number of output channels.
        kernel_size: Size of convolutional kernel.
        stride: Stride for convolution.
        dilation: Dilation factor for convolution.
        norm: Normalization method name.
        norm_params: Parameters for normalization.
        causal: Whether to use causal convolution.
        pad_mode: Padding mode for convolution.
        act_all: Whether to add activation after all layers.
        transposed: Whether to use transposed convolution.
        expansion: Channel expansion factor.
        groups: Number of groups for grouped convolution.
        bias: Whether to use bias in convolutions.
        
    Returns:
        List of neural network modules comprising the block.
        
    Raises:
        ValueError: If invalid parameters are provided.
    """
    try:
        # Build depth-wise separable convolution block
        block = [
            act(**activation_params),
            SConv1d(in_chs, out_chs, kernel_size=1, norm=norm, norm_kwargs=norm_params,
                    bias=bias if act_all else False,
                    nonlinearity='relu'),
        ]
        if act_all:
            block.append(act(**activation_params))
        
        # Select appropriate convolution type
        Conv = SConvTranspose1d if transposed else SConv1d
        
        # Calculate groups if not specified
        if groups == -1:
            groups = out_chs // expansion
            
        # Add grouped convolution layer
        block.append(
            Conv(
                out_chs, out_chs, kernel_size=kernel_size, stride=stride, dilation=dilation,
                groups=groups, norm=norm, norm_kwargs=norm_params, causal=causal,
                pad_mode=pad_mode, bias=bias,
                nonlinearity='relu' if act_all else 'linear'
            )
        )
        
        logger.debug(f"Created DWS conv block: in_chs={in_chs}, out_chs={out_chs}, kernel_size={kernel_size}")
        return block
        
    except Exception as e:
        logger.error(f"Error creating DWS conv block: {str(e)}", exc_info=True)
        raise


# =============================================================================
# RESIDUAL BLOCKS
# =============================================================================

class SEANetResnetBlock(nn.Module):
    """Residual block from SEANet model with skip connections.
    
    This block implements a residual connection with optional depth-wise separable
    convolutions and various skip connection strategies.
    
    Args:
        dim: Dimension of the input/output channels.
        kernel_size: Size of convolutional kernels.
        dilations: List of dilation factors for temporal modeling.
        activation: Name of activation function to use.
        activation_params: Parameters for the activation function.
        norm: Normalization method name.
        norm_params: Parameters for normalization layers.
        causal: Whether to use causal (unidirectional) convolution.
        pad_mode: Padding mode for convolutions.
        skip: Skip connection type ('identity', '1x1', 'scale', 'exp_scale', 'channelwise_scale').
        act_all: Whether to add activation after all layers.
        expansion: Channel expansion factor for depth-wise separable convs.
        groups: Number of groups for grouped convolution.
        bias: Whether to use bias in convolutions.
        res_scale: Residual scaling factor.
        idx: Block index for progressive scaling.
        zero_init: Whether to initialize residual scale to zero.
        
    Raises:
        ValueError: If invalid skip connection type is specified.
    """
    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        dilations: List[int] = [1, 1],
        activation: str = 'ELU',
        activation_params: Dict[str, Any] = {'alpha': 1.0},
        norm: str = 'weight_norm',
        norm_params: Dict[str, Any] = {},
        causal: bool = True,
        pad_mode: str = 'constant',
        skip: str = '1x1',
        act_all: bool = False,
        expansion: int = 1,
        groups: int = -1,
        bias: bool = True,
        res_scale: Optional[float] = None,
        idx: int = 0,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        
        try:
            # Get activation class
            act = getattr(nn, activation)
            
            # Build residual block layers
            block = []
            inplace_act_params = activation_params.copy()
            inplace_act_params["inplace"] = True
            
            # Calculate pre-scaling factor for progressive residual scaling
            self.pre_scale = (1 + idx * res_scale**2)**-0.5 if res_scale is not None else None
            
            # Build dilated convolution layers
            for i, dilation in enumerate(dilations):
                _activation_params = activation_params if i == 0 else inplace_act_params
                block += dws_conv_block(
                    act,
                    inplace_act_params,  # Using inplace activation for efficiency
                    dim,
                    dim,
                    kernel_size,
                    dilation=dilation,
                    norm=norm,
                    norm_params=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                    act_all=act_all,
                    expansion=expansion,
                    groups=groups,
                    bias=bias,
                )
                
            self.block = nn.Sequential(*block)
            self.shortcut: nn.Module
            
            # Configure skip connection based on type
            self.scale = None
            self.exp_scale = False
            
            if skip == "identity":
                self.shortcut = nn.Identity()
            elif skip == "1x1":
                # 1x1 convolution for skip connection
                self.shortcut = SConv1d(dim, dim, kernel_size=1, norm=norm, norm_kwargs=norm_params,
                                        bias=bias)
            elif skip == "scale":
                # Learnable scalar scaling
                self.scale = nn.Parameter(torch.ones(1, 1, 1))
            elif skip == "exp_scale":
                # Exponential scaling (initialized to 0 -> exp(0) = 1)
                self.scale = nn.Parameter(torch.zeros(1, 1, 1))
                self.exp_scale = True
            elif skip == "channelwise_scale":
                # Channel-wise scaling factors
                self.scale = nn.Parameter(torch.ones(1, dim, 1))
            else:
                raise ValueError(f"Unknown skip connection type: {skip}")
            
            # Residual scaling configuration
            self.res_scale = res_scale
            if zero_init:
                # Zero initialization for gradual feature introduction
                self.res_scale_param = nn.Parameter(torch.zeros(1))
            else:
                self.res_scale_param = None
                
            logger.debug(f"Created SEANetResnetBlock: dim={dim}, skip={skip}, dilations={dilations}")
            
        except Exception as e:
            logger.error(f"Error initializing SEANetResnetBlock: {str(e)}", exc_info=True)
            raise

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through residual block.
        
        Args:
            x: Input tensor of shape (batch, channels, time).
            
        Returns:
            Output tensor with same shape as input.
        """
        try:
            # Compute skip connection
            if self.scale is not None:
                scale = self.scale
                if self.exp_scale:
                    scale = scale.exp()
                shortcut = scale * x
            else:
                shortcut = self.shortcut(x)
            
            # Apply pre-scaling for progressive residual scaling
            if self.pre_scale is not None:
                x = x * self.pre_scale
                
            # Forward through residual block
            y: Tensor = self.block(x)
            
            # Apply residual scaling and add skip connection
            scale = 1.0 if self.res_scale is None else self.res_scale
            if self.res_scale_param is not None:
                scale = scale * self.res_scale_param
                
            # In-place operations for memory efficiency
            return y.mul_(scale).add_(shortcut)
            
        except Exception as e:
            logger.error(f"Error in SEANetResnetBlock forward pass: {str(e)}", exc_info=True)
            raise


# =============================================================================
# NORMALIZATION LAYERS
# =============================================================================

class L2Norm(nn.Module):
    """L2 normalization layer with optional scaling.
    
    Args:
        channels: Number of channels for scaling factor calculation.
        eps: Small epsilon value for numerical stability.
        inout_norm: Whether to apply input/output normalization scaling.
    """
    
    def __init__(self, channels: int, eps: float = 1e-12, inout_norm: bool = True) -> None:
        super().__init__()
        self.scale = channels ** 0.5
        self.eps = eps
        self.inout_norm = inout_norm
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply L2 normalization to input tensor.
        
        Args:
            x: Input tensor of shape (batch, channels, time).
            
        Returns:
            L2-normalized tensor with optional scaling.
        """
        # Normalize along channel dimension
        y = nn.functional.normalize(x, p=2.0, dim=1, eps=self.eps)
        
        # Apply scaling to maintain magnitude
        if self.inout_norm:
            return y.mul_(self.scale)
        return y


class Scale(nn.Module):
    """Scaling layer for feature normalization.
    
    Args:
        dim: Number of channels to scale.
        value: Initial scaling value.
        learnable: Whether scaling is learnable or fixed.
        inplace: Whether to perform in-place scaling.
    """
    
    def __init__(
        self,
        dim: int,
        value: float = 1.0,
        learnable: bool = True,
        inplace: bool = False
    ) -> None:
        super().__init__()
        
        if learnable:
            # Learnable channel-wise scaling
            self.scale = nn.Parameter(torch.ones(1, dim, 1) * value)
        else:
            # Fixed scaling value
            self.scale = value
        self.inplace = inplace
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply scaling to input tensor.
        
        Args:
            x: Input tensor.
            
        Returns:
            Scaled tensor.
        """
        if self.inplace:
            return x.mul_(self.scale)
        return self.scale * x

# =============================================================================
# SPECTROGRAM PROCESSING
# =============================================================================

class SpecBlock(nn.Module):
    """Spectrogram processing block with compression and normalization.
    
    This block computes spectrograms from raw audio and applies various
    transformations including compression and normalization.
    
    Args:
        spec: Spectrogram type ('stft' or '').
        spec_compression: Compression type ('log', numeric power, or '').
        n_fft: FFT size for spectrogram computation.
        channels: Number of output channels.
        stride: Hop size for spectrogram.
        norm: Normalization method.
        norm_params: Normalization parameters.
        bias: Whether to use bias in convolution.
        pad_mode: Padding mode.
        learnable: Whether STFT parameters are learnable.
        causal: Whether to use causal processing.
        mean: Mean for normalization.
        std: Standard deviation for normalization.
        res_scale: Residual scaling factor.
        zero_init: Whether to zero-initialize scaling parameter.
        inout_norm: Whether to apply input/output normalization.
        
    Raises:
        ValueError: If unknown spectrogram type is specified.
    """
    
    def __init__(
        self,
        spec: str,
        spec_compression: str,
        n_fft: int,
        channels: int,
        stride: int,
        norm: str,
        norm_params: Dict[str, Any],
        bias: bool,
        pad_mode: str,
        learnable: bool,
        causal: bool = True,
        mean: float = 0.0,
        std: float = 1.0,
        res_scale: Optional[float] = 1.0,
        zero_init: bool = True,
        inout_norm: bool = True,
    ) -> None:
        super().__init__()
        self.learnable = learnable
        
        try:
            # Configure spectrogram computation
            if spec == "stft":
                if causal:
                    self.spec = CausalSTFT(n_fft=n_fft, hop_size=stride, pad_mode=pad_mode, learnable=learnable)
                else:
                    self.spec = STFT(n_fft=n_fft, hop_size=stride, center=False, magnitude=True)
            elif spec == "":
                # No spectrogram computation
                self.spec = None
                return
            else:
                raise ValueError(f"Unknown spec type: {spec}")
            
            # Configure compression type
            if spec_compression == "log":
                self.compression = "log"
            elif spec_compression == "":
                self.compression = ""
            else:
                # Power compression with specified exponent
                self.compression = float(spec_compression)
            
            # Store normalization parameters
            self.inout_norm = inout_norm
            self.mean, self.std = mean, std
            self.scale = res_scale
            self.scale_param = None
            
            # Create convolution layer for spectrogram processing
            # Input channels = n_fft//2 + 1 (number of frequency bins)
            self.layer = SConv1d(
                n_fft//2+1, channels, 1,
                norm=norm, norm_kwargs=norm_params,
                bias=bias, pad_mode=pad_mode
            )
            
            if zero_init:
                # Zero initialization for gradual feature introduction
                self.scale_param = nn.Parameter(torch.zeros(1))
                
            logger.debug(f"Created SpecBlock: spec={spec}, compression={spec_compression}, n_fft={n_fft}")
            
        except Exception as e:
            logger.error(f"Error initializing SpecBlock: {str(e)}", exc_info=True)
            raise
     

    def forward(self, x: Tensor, wav: Tensor) -> Tensor:
        """Process spectrogram and add to input features.
        
        Args:
            x: Input feature tensor.
            wav: Raw waveform tensor.
            
        Returns:
            Input tensor with added spectrogram features.
        """
        if self.spec is None:
            # No spectrogram processing
            return x
        
        try:
            # Compute spectrogram from waveform
            y: Tensor = self.spec(wav)
            
            # Apply compression
            if self.compression == "log":
                # Logarithmic compression with numerical stability
                y = y.clamp_min(1e-5).log_()
            elif self.compression == "":
                # No compression
                pass
            else:
                # Power compression: sign(x) * |x|^compression
                y = y.sign() * y.abs().pow(self.compression)
            
            # Normalize spectrogram features
            if self.inout_norm:
                y.sub_(self.mean).div_(self.std)
            
            # Apply convolution layer to process spectrogram
            y = self.layer(y)
            
            # Compute scaling factor
            scale = 1.0 if self.scale is None else self.scale
            if self.scale_param is not None:
                scale = self.scale_param * scale
            
            # Add scaled spectrogram features to input (residual connection)
            x.add_(y.mul_(scale))
            
            return x
            
        except Exception as e:
            logger.error(f"Error in SpecBlock forward pass: {str(e)}", exc_info=True)
            raise


# =============================================================================
# MODULATION LAYERS
# =============================================================================

class FiLM(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer.
    
    FiLM applies feature-wise affine transformations based on conditioning
    information, enabling adaptive feature modulation.
    
    Args:
        condition_dim: Dimension of conditioning vector.
    """
    
    def __init__(self, condition_dim: int) -> None:
        super().__init__()
        
        # Linear layers for computing modulation parameters
        self.gamma_layer = nn.Linear(condition_dim, 1)  # Multiplicative modulation
        self.beta_layer = nn.Linear(condition_dim, 1)   # Additive modulation

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """Apply FiLM modulation to input features.
        
        Args:
            x: Input tensor of shape (batch, channels, time).
            condition: Conditioning tensor of shape (batch, condition_dim).
            
        Returns:
            Modulated tensor with same shape as input.
        """
        # Compute modulation parameters
        gamma = self.gamma_layer(condition).unsqueeze(-1)  # Shape: (batch, 1, 1)
        beta = self.beta_layer(condition).unsqueeze(-1)    # Shape: (batch, 1, 1)
        
        # Apply affine transformation: y = gamma * x + beta
        return (x * gamma) + beta
# =============================================================================
# ENCODER AND DECODER NETWORKS
# =============================================================================

class SEANetEncoder(nn.Module):
    """SEANet encoder with hierarchical feature extraction and FiLM modulation.
    
    This encoder processes audio through multiple scales with residual blocks,
    spectrogram features, and adaptive modulation based on message embedding.
    
    Args:
        channels: Number of input audio channels.
        dimension: Output dimension of encoded features.
        msg_dimension: Dimension of message to embed.
        n_filters: Base number of filters.
        n_fft_base: Base FFT size for spectrograms.
        n_residual_layers: Number of residual blocks per scale.
        ratios: Downsampling ratios for each scale.
        activation: Activation function name.
        activation_params: Parameters for activation function.
        norm: Normalization method.
        norm_params: Normalization parameters.
        kernel_size: Kernel size for initial convolution.
        last_kernel_size: Kernel size for final convolution.
        residual_kernel_size: Kernel size for residual blocks.
        dilation_base: Base for exponential dilation.
        skip: Skip connection type for residual blocks.
        causal: Whether to use causal convolutions.
        pad_mode: Padding mode for convolutions.
        act_all: Whether to add activation after all layers.
        expansion: Channel expansion factor.
        groups: Number of groups for grouped convolutions.
        l2norm: Whether to apply L2 normalization to output.
        bias: Whether to use bias in convolutions.
        spec: Spectrogram type.
        spec_compression: Spectrogram compression type.
        spec_learnable: Whether spectrogram parameters are learnable.
        res_scale: Residual scaling factor.
        wav_std: Standard deviation for waveform normalization.
        spec_means: Per-scale spectrogram normalization means.
        spec_stds: Per-scale spectrogram normalization stds.
        zero_init: Whether to zero-initialize scaling parameters.
        inout_norm: Whether to apply input/output normalization.
        embedding_dim: Dimension of message embedding.
        embedding_layers: Number of layers in embedding MLP.
        freq_bands: Number of frequency bands for FiLM modulation.
        
    Raises:
        ValueError: If invalid configuration is provided.
    """
    
    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        msg_dimension: int = 16,
        n_filters: int = 32,
        n_fft_base: int = 64,
        n_residual_layers: int = 1,
        ratios: List[int] = [8, 5, 4, 2],
        activation: str = 'ELU',
        activation_params: Dict[str, Any] = {'alpha': 1.0},
        norm: str = 'weight_norm',
        norm_params: Dict[str, Any] = {},
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        skip: str = '1x1',
        causal: bool = False,
        pad_mode: str = 'constant',
        act_all: bool = False,
        expansion: int = 1,
        groups: int = -1,
        l2norm: bool = False,
        bias: bool = True,
        spec: str = "stft",
        spec_compression: str = "",
        spec_learnable: bool = False,
        res_scale: Optional[float] = None,
        wav_std: float = 0.1122080159,
        spec_means: List[float] = [-4.554, -4.315, -4.021, -3.726, -3.477],
        spec_stds: List[float] = [2.830, 2.837, 2.817, 2.796, 2.871],
        zero_init: bool = True,
        inout_norm: bool = True,
        embedding_dim: int = 64,
        embedding_layers: int = 2,
        freq_bands: int = 4,
    ) -> None:
        super().__init__()
        
        try:
            # Store configuration
            self.dimension = dimension
            self.n_filters = n_filters
            self.ratios = list(reversed(ratios))  # Process from coarse to fine
            del ratios
            self.n_residual_layers = n_residual_layers
            self.hop_length = int(np.prod(self.ratios))  # Total downsampling factor
            self.freq_bands = freq_bands
            
            # Get activation class
            act = getattr(nn, activation)
            mult = 1  # Channel multiplier
            
            # Initial convolution with optional waveform normalization
            self.conv_pre = nn.Sequential(
                Scale(1, value=1/wav_std, learnable=False, inplace=False) if inout_norm else nn.Identity(),
                SConv1d(
                    channels, mult * n_filters, kernel_size,
                    norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode, bias=bias
                )
            )
            
            logger.info(f"Initializing SEANetEncoder: dimension={dimension}, ratios={self.ratios}")
            
        except Exception as e:
            logger.error(f"Error in SEANetEncoder initialization: {str(e)}", exc_info=True)
            raise
            # Build encoder blocks for each scale
            self.blocks = nn.ModuleList()
            self.spec_blocks = nn.ModuleList()
            self.downsample = nn.ModuleList()
            self.film_layers = nn.ModuleList()  # FiLM modulation layers
            
            stride = 1  # Cumulative stride for spectrogram blocks
            
            for block_idx, ratio in enumerate(self.ratios):
                # Build residual blocks for current scale
                block = []
                for j in range(1, n_residual_layers + 1):
                    # Index for progressive scaling
                    idx = j - 1 if spec == "" else j
                    
                    # Create residual block with exponential dilation
                    block += [
                        SEANetResnetBlock(
                            mult * n_filters,
                            kernel_size=residual_kernel_size,
                            dilations=[dilation_base ** j, 1],  # Exponential dilation pattern
                            norm=norm,
                            norm_params=norm_params,
                            activation=activation,
                            activation_params=activation_params,
                            causal=causal,
                            pad_mode=pad_mode,
                            skip=skip,
                            act_all=act_all,
                            expansion=expansion,
                            groups=groups,
                            bias=bias,
                            res_scale=res_scale,
                            idx=idx,
                            zero_init=zero_init,
                        )
                    ]
                self.blocks.append(nn.Sequential(*block))
                
                # Add spectrogram processing block
                spec_block = SpecBlock(
                    spec,
                    spec_compression,
                    mult * n_fft_base,      # Scale FFT size with depth
                    mult * n_filters,       # Output channels
                    stride,                 # Current cumulative stride
                    norm,
                    norm_params,
                    bias=False,
                    pad_mode=pad_mode,
                    learnable=spec_learnable,
                    causal=causal,
                    mean=spec_means[block_idx],  # Scale-specific normalization
                    std=spec_stds[block_idx],
                    res_scale=res_scale,
                    zero_init=zero_init,
                    inout_norm=inout_norm,
                )
                self.spec_blocks.append(spec_block)
                stride *= ratio  # Update cumulative stride

                # Build downsampling layers
                if res_scale is None:
                    scale_layer = nn.Identity()
                else:
                    # Progressive residual scaling
                    scale_layer = Scale(
                        1,
                        value=(1 + n_residual_layers * res_scale**2)**-0.5,
                        learnable=False,
                        inplace=True
                    )
                    
                # Downsampling with channel expansion
                downsample = nn.Sequential(
                    scale_layer,
                    act(inplace=True, **activation_params),
                    # 1x1 conv for channel expansion
                    SConv1d(
                        mult * n_filters,
                        mult * n_filters * 2,
                        1,
                        norm=norm,
                        norm_kwargs=norm_params,
                        bias=False,
                        nonlinearity='relu'
                    ),
                    # Strided convolution for downsampling
                    SConv1d(
                        mult * n_filters * 2,
                        mult * n_filters * 2,
                        kernel_size=ratio * 2,
                        stride=ratio,
                        groups=mult * n_filters * 2,  # Depth-wise convolution
                        norm=norm,
                        norm_kwargs=norm_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        bias=bias
                    ),
                )
                self.downsample.append(downsample)
                
                # Add FiLM modulation layer (not used in init, defined later)
                self.film_layers.append(FiLM(msg_dimension))
                
                mult *= 2  # Double channels after each scale

            # Final spectrogram block
            self.spec_post = SpecBlock(
                spec,
                spec_compression,
                mult * n_fft_base,
                mult * n_filters,
                stride,
                norm,
                norm_params,
                bias=False,
                pad_mode=pad_mode,
                learnable=spec_learnable,
                causal=causal,
                mean=spec_means[-1],
                std=spec_stds[-1],
                res_scale=res_scale,
                zero_init=zero_init,
                inout_norm=inout_norm,
            )
            
            # Final convolution and projection layers
            self.conv_post = nn.Sequential(
                act(inplace=False, **activation_params),
                # Depth-wise convolution
                SConv1d(
                    mult * n_filters,
                    mult * n_filters,
                    last_kernel_size,
                    groups=mult * n_filters,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                    bias=False,
                    nonlinearity='relu'
                ),
                # Project to output dimension
                SConv1d(
                    mult * n_filters,
                    dimension,
                    1,
                    norm=norm,
                    norm_kwargs=norm_params,
                    bias=bias
                ),
                # Optional L2 normalization
                L2Norm(dimension, inout_norm=inout_norm) if l2norm else nn.Identity(),
            )
            
            if l2norm:
                # Special initialization for L2 norm case
                # Prevents gradient explosion with silent audio
                self.conv_post[-2].conv.conv.bias.data.normal_()

            # Build message embedding network
            embedding_mlp = []
            for _ in range(embedding_layers):
                embedding_mlp.append(nn.Linear(embedding_dim, embedding_dim))
                embedding_mlp.append(nn.ReLU())
            
            self.msg_embedding = nn.Sequential(
                nn.Linear(msg_dimension, embedding_dim),
                *embedding_mlp
            )
            
            # Create frequency-specific FiLM layers for adaptive modulation
            # One set of FiLM layers per scale and frequency band
            self.film_layers = nn.ModuleList([
                nn.ModuleList([FiLM(embedding_dim) for _ in range(freq_bands)])
                for mult in [1, 2, 4, 8]  # Corresponding to each scale
            ])
            
            logger.info("SEANetEncoder initialization complete")

    def forward(self, x: Tensor, msg: Tensor) -> Tensor:
        """Encode audio with embedded message.
        
        Args:
            x: Input audio tensor of shape (batch, channels, time).
            msg: Message tensor to embed of shape (batch, msg_dimension).
            
        Returns:
            Encoded features of shape (batch, dimension, time//hop_length).
        """
        try:
            # Keep reference to original waveform for spectrogram computation
            wav = x
            
            # Initial convolution and normalization
            x = self.conv_pre(x)
            
            # Process message embedding if provided
            msg_embedded = None
            if msg is not None:
                msg = msg.float()
                # Transform message to embedding space
                msg_embedded = self.msg_embedding(msg)

            # Process through encoder scales
            for block_idx, (block, spec_block, downsample, film_layer) in enumerate(zip(
                self.blocks, self.spec_blocks, self.downsample, self.film_layers
            )):
                # Residual blocks
                x = block(x)
                
                # Add spectrogram features
                x = spec_block(x, wav)
                
                # Downsample
                x = downsample(x)
                
                # Apply FiLM modulation if message is provided
                if msg_embedded is not None:
                    # Split features into frequency bands for band-specific modulation
                    band_width = x.shape[1] // self.freq_bands
                    x_bands = []  # Collect modulated bands
                    
                    for band_idx in range(self.freq_bands):
                        # Extract frequency band
                        start_channel = band_idx * band_width
                        end_channel = (band_idx + 1) * band_width
                        x_band = x[:, start_channel:end_channel]
                        
                        # Apply band-specific FiLM modulation
                        x_band = film_layer[band_idx](x_band, msg_embedded)
                        x_bands.append(x_band)
                    
                    # Recombine modulated frequency bands
                    x = torch.cat(x_bands, dim=1)

            # Final processing
            x = self.spec_post(x, wav)
            x = self.conv_post(x)
            
            return x
            
        except Exception as e:
            logger.error(f"Error in SEANetEncoder forward pass: {str(e)}", exc_info=True)
            raise


class SEANetDecoder(nn.Module):
    """SEANet decoder for audio synthesis from encoded features.
    
    This decoder mirrors the encoder architecture, progressively upsampling
    features back to audio resolution.
    
    Args:
        channels: Number of output audio channels.
        dimension: Input dimension of encoded features.
        n_filters: Base number of filters.
        n_residual_layers: Number of residual blocks per scale.
        ratios: Upsampling ratios for each scale.
        activation: Activation function name.
        activation_params: Parameters for activation function.
        norm: Normalization method.
        norm_params: Normalization parameters.
        kernel_size: Kernel size for initial convolution.
        last_kernel_size: Kernel size for final convolution.
        residual_kernel_size: Kernel size for residual blocks.
        dilation_base: Base for exponential dilation.
        skip: Skip connection type for residual blocks.
        causal: Whether to use causal convolutions.
        pad_mode: Padding mode for convolutions.
        trim_right_ratio: Ratio for trimming transposed conv artifacts.
        final_activation: Optional final activation function.
        final_activation_params: Parameters for final activation.
        act_all: Whether to add activation after all layers.
        expansion: Channel expansion factor.
        groups: Number of groups for grouped convolutions.
        bias: Whether to use bias in convolutions.
        res_scale: Residual scaling factor.
        wav_std: Standard deviation for waveform denormalization.
        zero_init: Whether to zero-initialize scaling parameters.
        inout_norm: Whether to apply input/output normalization.
        
    Raises:
        ValueError: If invalid configuration is provided.
    """
    
    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 1,
        ratios: List[int] = [8, 5, 4, 2],
        activation: str = 'ELU',
        activation_params: Dict[str, Any] = {'alpha': 1.0},
        norm: str = 'weight_norm',
        norm_params: Dict[str, Any] = {},
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        skip: str = '1x1',
        causal: bool = False,
        pad_mode: str = 'constant',
        trim_right_ratio: float = 1.0,
        final_activation: Optional[str] = None,
        final_activation_params: Optional[Dict[str, Any]] = None,
        act_all: bool = False,
        expansion: int = 1,
        groups: int = -1,
        bias: bool = True,
        res_scale: Optional[float] = None,
        wav_std: float = 0.1122080159,
        zero_init: bool = True,
        inout_norm: bool = True,
    ) -> None:
        super().__init__()
        
        try:
            # Store configuration
            self.dimension = dimension
            self.channels = channels
            self.n_filters = n_filters
            self.ratios = ratios
            del ratios
            self.n_residual_layers = n_residual_layers
            self.hop_length = int(np.prod(self.ratios))  # Total upsampling factor
            
            # Get activation class
            act = getattr(nn, activation)
            
            # Calculate initial channel multiplier (inverse of encoder)
            mult = int(2 ** len(self.ratios))
            
            # Build decoder layers
            model: List[nn.Module] = []
            
            # Initial projection from latent dimension
            model += [
                SConv1d(
                    dimension,
                    mult * n_filters,
                    1,
                    norm=norm,
                    norm_kwargs=norm_params,
                    bias=False
                ),
                # Depth-wise convolution
                SConv1d(
                    mult * n_filters,
                    mult * n_filters,
                    kernel_size,
                    groups=mult * n_filters,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                    bias=bias
                )
            ]
            
            logger.info(f"Initializing SEANetDecoder: dimension={dimension}, ratios={self.ratios}")
            
        except Exception as e:
            logger.error(f"Error in SEANetDecoder initialization: {str(e)}", exc_info=True)
            raise

            # Build upsampling blocks for each scale
            for i, ratio in enumerate(self.ratios):
                # Add progressive residual scaling
                if i > 0:
                    if res_scale is None:
                        scale_layer = nn.Identity()
                    else:
                        scale_layer = Scale(
                            1,
                            value=(1 + n_residual_layers * res_scale**2)**-0.5,
                            learnable=False,
                            inplace=True
                        )
                else:
                    scale_layer = nn.Identity()
                    
                # Upsampling layers
                model += [
                    scale_layer,
                    act(inplace=True, **activation_params),
                    # Transposed convolution for upsampling
                    SConvTranspose1d(
                        mult * n_filters,
                        mult * n_filters,
                        kernel_size=ratio * 2,
                        stride=ratio,
                        groups=mult * n_filters,  # Depth-wise
                        norm=norm,
                        norm_kwargs=norm_params,
                        causal=causal,
                        trim_right_ratio=trim_right_ratio,
                        bias=False,
                        nonlinearity='relu'
                    ),
                    # Channel reduction
                    SConv1d(
                        mult * n_filters,
                        mult * n_filters // 2,
                        1,
                        norm=norm,
                        norm_kwargs=norm_params,
                        bias=bias
                    )
                ]
                
                # Add residual blocks for current scale
                for j in range(n_residual_layers):
                    model += [
                        SEANetResnetBlock(
                            mult * n_filters // 2,
                            kernel_size=residual_kernel_size,
                            dilations=[dilation_base ** j, 1],  # Exponential dilation
                            activation=activation,
                            activation_params=activation_params,
                            norm=norm,
                            norm_params=norm_params,
                            causal=causal,
                            pad_mode=pad_mode,
                            skip=skip,
                            act_all=act_all,
                            expansion=expansion,
                            groups=groups,
                            bias=bias,
                            res_scale=res_scale,
                            idx=j,
                            zero_init=zero_init,
                        )
                    ]
                    
                mult //= 2  # Halve channels after each scale

            # Final output layers
            if res_scale is None:
                scale_layer = nn.Identity()
            else:
                scale_layer = Scale(
                    1,
                    value=(1 + n_residual_layers * res_scale**2)**-0.5,
                    learnable=False,
                    inplace=True
                )
                
            model += [
                scale_layer,
                act(inplace=True, **activation_params),
                # Final convolution to output channels
                SConv1d(
                    n_filters,
                    channels,
                    last_kernel_size,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                    bias=bias,
                    nonlinearity='relu'
                ),
                # Denormalize waveform
                Scale(1, value=wav_std, learnable=False, inplace=True) if inout_norm else nn.Identity(),
            ]
            
            # Add optional final activation (e.g., tanh for bounded output)
            if final_activation is not None:
                final_act = getattr(nn, final_activation)
                final_activation_params = final_activation_params or {}
                model += [
                    final_act(**final_activation_params)
                ]
                
            self.model = nn.Sequential(*model)
            
            logger.info("SEANetDecoder initialization complete")

    def forward(self, z: Tensor) -> Tensor:
        """Decode features to audio.
        
        Args:
            z: Encoded features of shape (batch, dimension, time).
            
        Returns:
            Decoded audio of shape (batch, channels, time * hop_length).
        """
        try:
            y = self.model(z)
            return y
            
        except Exception as e:
            logger.error(f"Error in SEANetDecoder forward pass: {str(e)}", exc_info=True)
            raise