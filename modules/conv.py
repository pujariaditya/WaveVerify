# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Convolutional layers wrappers and utilities."""

# =============================================================================
# IMPORTS
# =============================================================================

import logging
import math
import warnings
from typing import Any, Dict, Optional, Tuple, Union, FrozenSet

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, parametrizations

from .norm import ConvLayerNorm
from .weight_standardization import weight_standardization

# =============================================================================
# LOGGING SETUP
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

CONV_NORMALIZATIONS: FrozenSet[str] = frozenset([
    'none', 'weight_norm', 'spectral_norm', 'weight_standardization',
    'time_layer_norm', 'layer_norm', 'time_group_norm'
])


# =============================================================================
# NORMALIZATION UTILITY FUNCTIONS
# =============================================================================

def apply_parametrization_norm(
    module: nn.Module,
    norm: str = 'none',
    **norm_kwargs: Any
) -> nn.Module:
    """Apply parametrization-based normalization to a module.
    
    Args:
        module: The neural network module to apply normalization to.
        norm: Type of normalization. Must be one of CONV_NORMALIZATIONS.
        **norm_kwargs: Additional keyword arguments for the normalization method.
        
    Returns:
        The module with normalization applied.
        
    Raises:
        ValueError: If norm is not in CONV_NORMALIZATIONS.
        RuntimeError: If normalization application fails.
    """
    try:
        if norm not in CONV_NORMALIZATIONS:
            raise ValueError(
                f"Invalid normalization type: {norm}. "
                f"Must be one of {CONV_NORMALIZATIONS}"
            )
        
        if norm == 'weight_norm':
            return parametrizations.weight_norm(module, **norm_kwargs)
        elif norm == 'spectral_norm':
            return spectral_norm(module, **norm_kwargs)
        elif norm == 'weight_standardization':
            return weight_standardization(module, **norm_kwargs)
        else:
            # No parametrization needed for other normalization types
            return module
            
    except Exception as e:
        logger.error(
            f"Failed to apply {norm} normalization: {str(e)}",
            exc_info=True
        )
        raise RuntimeError(f"Failed to apply normalization: {str(e)}") from e


def get_norm_module(
    module: nn.Module,
    causal: bool = False,
    norm: str = 'none',
    **norm_kwargs: Any
) -> nn.Module:
    """Return the proper normalization module.
    
    If causal is True, this will ensure the returned module is causal,
    or raise an error if the normalization doesn't support causal evaluation.
    
    Args:
        module: The base module to apply normalization to.
        causal: Whether to enforce causal evaluation.
        norm: Type of normalization. Must be one of CONV_NORMALIZATIONS.
        **norm_kwargs: Additional keyword arguments for the normalization.
        
    Returns:
        The appropriate normalization module.
        
    Raises:
        ValueError: If norm is not supported or incompatible with causal mode.
        TypeError: If module type is incompatible with the normalization.
    """
    try:
        if norm not in CONV_NORMALIZATIONS:
            raise ValueError(
                f"Invalid normalization type: {norm}. "
                f"Must be one of {CONV_NORMALIZATIONS}"
            )
        
        if norm == 'layer_norm':
            if not isinstance(module, nn.modules.conv._ConvNd):
                raise TypeError(
                    f"layer_norm requires a convolutional module, "
                    f"got {type(module).__name__}"
                )
            return ConvLayerNorm(module.out_channels, **norm_kwargs)
            
        elif norm == 'time_group_norm':
            if causal:
                raise ValueError(
                    "GroupNorm doesn't support causal evaluation. "
                    "Please use a different normalization method."
                )
            if not isinstance(module, nn.modules.conv._ConvNd):
                raise TypeError(
                    f"time_group_norm requires a convolutional module, "
                    f"got {type(module).__name__}"
                )
            return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
            
        else:
            # Return identity for 'none' and other normalizations
            # that don't require a separate module
            return nn.Identity()
            
    except Exception as e:
        logger.error(
            f"Failed to create norm module for {norm}: {str(e)}",
            exc_info=True
        )
        raise


# =============================================================================
# PADDING UTILITY FUNCTIONS
# =============================================================================

def get_extra_padding_for_conv1d(
    input_tensor: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding_total: int = 0
) -> int:
    """Calculate extra padding needed for conv1d to ensure full windows.
    
    Args:
        input_tensor: Input tensor with shape [..., length].
        kernel_size: Size of the convolutional kernel.
        stride: Stride of the convolution.
        padding_total: Total padding already applied.
        
    Returns:
        Extra padding needed at the end.
        
    Raises:
        ValueError: If input parameters are invalid.
    """
    try:
        if kernel_size <= 0 or stride <= 0:
            raise ValueError(
                f"kernel_size and stride must be positive, "
                f"got kernel_size={kernel_size}, stride={stride}"
            )
        
        # Get the length of the last dimension (time dimension)
        length = input_tensor.shape[-1]
        
        # Calculate number of frames that will be produced
        n_frames = (length - kernel_size + padding_total) / stride + 1
        
        # Calculate ideal length to have complete frames
        ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
        
        return max(0, ideal_length - length)
        
    except Exception as e:
        logger.error(
            f"Error calculating extra padding: {str(e)}",
            exc_info=True
        )
        raise


def pad_for_conv1d(
    input_tensor: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding_total: int = 0
) -> torch.Tensor:
    """Pad input for conv1d to ensure the last window is complete.
    
    Extra padding is added at the end. This is required to ensure that we can
    rebuild an output of the same length, as otherwise, even with padding,
    some time steps might get removed.
    
    Example:
        With total_padding=4, kernel_size=4, stride=2:
        Input:        [1, 2, 3, 4, 5]
        Padded:       [0, 0, 1, 2, 3, 4, 5, 0, 0]  # 0s are padding
        Conv frames:  [   1,    2,    3        ]    # Last 0 never used
        Transposed:   [0, 0, 1, 2, 3, 4, 5, 0  ]   # Position 5 removed
        After unpad:  [      1, 2, 3, 4        ]    # Missing time step!
    
    Args:
        input_tensor: Input tensor to pad.
        kernel_size: Size of the convolutional kernel.
        stride: Stride of the convolution.
        padding_total: Total padding already applied.
        
    Returns:
        Padded tensor.
        
    Raises:
        RuntimeError: If padding calculation fails.
    """
    try:
        extra_padding = get_extra_padding_for_conv1d(
            input_tensor, kernel_size, stride, padding_total
        )
        return F.pad(input_tensor, (0, extra_padding))
        
    except Exception as e:
        logger.error(
            f"Failed to pad tensor for conv1d: {str(e)}",
            exc_info=True
        )
        raise RuntimeError(f"Padding failed: {str(e)}") from e


def pad1d(
    input_tensor: torch.Tensor,
    paddings: Tuple[int, int],
    mode: str = 'constant',
    value: float = 0.
) -> torch.Tensor:
    """Apply 1D padding with support for reflect padding on small inputs.
    
    This wrapper handles the case where reflect padding is requested on inputs
    smaller than the padding size by adding temporary padding.
    
    Args:
        input_tensor: Input tensor to pad.
        paddings: Tuple of (left_padding, right_padding).
        mode: Padding mode ('constant', 'reflect', 'replicate', 'circular').
        value: Fill value for 'constant' padding mode.
        
    Returns:
        Padded tensor.
        
    Raises:
        ValueError: If padding values are negative.
        RuntimeError: If padding operation fails.
    """
    try:
        padding_left, padding_right = paddings
        
        if padding_left < 0 or padding_right < 0:
            raise ValueError(
                f"Padding values must be non-negative, "
                f"got left={padding_left}, right={padding_right}"
            )
        
        # Handle reflection padding for small inputs
        if mode == 'reflect':
            length = input_tensor.shape[-1]
            max_pad = max(padding_left, padding_right)
            extra_pad = 0
            
            # If input is too small for reflect padding, add temporary padding
            if length <= max_pad:
                extra_pad = max_pad - length + 1
                input_tensor = F.pad(input_tensor, (0, extra_pad))
            
            # Apply the actual padding
            padded = F.pad(input_tensor, paddings, mode)
            
            # Remove the temporary padding
            if extra_pad > 0:
                end = padded.shape[-1] - extra_pad
                return padded[..., :end]
            return padded
        else:
            # For other padding modes, apply directly
            return F.pad(input_tensor, paddings, mode, value)
            
    except Exception as e:
        logger.error(
            f"Padding operation failed: {str(e)}",
            exc_info=True
        )
        raise RuntimeError(f"Failed to apply padding: {str(e)}") from e


def unpad1d(
    input_tensor: torch.Tensor,
    paddings: Tuple[int, int]
) -> torch.Tensor:
    """Remove padding from a 1D tensor.
    
    Args:
        input_tensor: Padded input tensor.
        paddings: Tuple of (left_padding, right_padding) to remove.
        
    Returns:
        Tensor with padding removed.
        
    Raises:
        ValueError: If padding values are invalid.
    """
    try:
        padding_left, padding_right = paddings
        
        if padding_left < 0 or padding_right < 0:
            raise ValueError(
                f"Padding values must be non-negative, "
                f"got left={padding_left}, right={padding_right}"
            )
        
        total_padding = padding_left + padding_right
        if total_padding > input_tensor.shape[-1]:
            raise ValueError(
                f"Total padding ({total_padding}) exceeds tensor length "
                f"({input_tensor.shape[-1]})"
            )
        
        # Calculate the end index
        end = input_tensor.shape[-1] - padding_right
        
        # Remove padding from both sides
        return input_tensor[..., padding_left:end]
        
    except Exception as e:
        logger.error(
            f"Failed to remove padding: {str(e)}",
            exc_info=True
        )
        raise


# =============================================================================
# NORMALIZED CONVOLUTION LAYERS
# =============================================================================

class NormConv1d(nn.Module):
    """Conv1d with integrated normalization.
    
    Provides a uniform interface for different normalization approaches
    applied to 1D convolutions.
    
    Args:
        *args: Positional arguments for nn.Conv1d.
        causal: Whether to use causal convolution.
        norm: Type of normalization to apply.
        nonlinearity: Type of nonlinearity for weight initialization.
        norm_kwargs: Additional arguments for normalization.
        **kwargs: Additional arguments for nn.Conv1d.
    """
    
    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = 'none',
        nonlinearity: str = 'linear',
        norm_kwargs: Dict[str, Any] = {},
        **kwargs
    ):
        super().__init__()
        
        try:
            # Create the base convolution layer
            conv = nn.Conv1d(*args, **kwargs)
            
            # Initialize weights using Kaiming initialization
            nn.init.kaiming_normal_(conv.weight, nonlinearity=nonlinearity)
            if conv.bias is not None:
                conv.bias.data.zero_()
            
            # Apply parametrization-based normalization
            self.conv = apply_parametrization_norm(conv, norm, **norm_kwargs)
            
            # Get the normalization module
            self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
            self.norm_type = norm
            
        except Exception as e:
            logger.error(
                f"Failed to initialize NormConv1d: {str(e)}",
                exc_info=True
            )
            raise
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolution and normalization.
        
        Args:
            input_tensor: Input tensor of shape [batch, channels, length].
            
        Returns:
            Output tensor after convolution and normalization.
        """
        try:
            output = self.conv(input_tensor)
            output = self.norm(output)
            return output
            
        except Exception as e:
            logger.error(
                f"Forward pass failed in NormConv1d: {str(e)}",
                exc_info=True
            )
            raise


class NormConv2d(nn.Module):
    """Conv2d with integrated normalization.
    
    Provides a uniform interface for different normalization approaches
    applied to 2D convolutions.
    
    Args:
        *args: Positional arguments for nn.Conv2d.
        norm: Type of normalization to apply.
        nonlinearity: Type of nonlinearity for weight initialization.
        norm_kwargs: Additional arguments for normalization.
        **kwargs: Additional arguments for nn.Conv2d.
    """
    
    def __init__(
        self,
        *args,
        norm: str = 'none',
        nonlinearity: str = 'linear',
        norm_kwargs: Dict[str, Any] = {},
        **kwargs
    ):
        super().__init__()
        
        try:
            # Create the base convolution layer
            conv = nn.Conv2d(*args, **kwargs)
            
            # Note: Weight initialization is commented out in original
            # This might be intentional for specific use cases
            
            # Apply parametrization-based normalization
            self.conv = apply_parametrization_norm(conv, norm, **norm_kwargs)
            
            # Get the normalization module (2D conv is never causal)
            self.norm = get_norm_module(
                self.conv, causal=False, norm=norm, **norm_kwargs
            )
            self.norm_type = norm
            
        except Exception as e:
            logger.error(
                f"Failed to initialize NormConv2d: {str(e)}",
                exc_info=True
            )
            raise
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolution and normalization.
        
        Args:
            input_tensor: Input tensor of shape [batch, channels, height, width].
            
        Returns:
            Output tensor after convolution and normalization.
        """
        try:
            output = self.conv(input_tensor)
            output = self.norm(output)
            return output
            
        except Exception as e:
            logger.error(
                f"Forward pass failed in NormConv2d: {str(e)}",
                exc_info=True
            )
            raise


class NormConvTranspose1d(nn.Module):
    """ConvTranspose1d with integrated normalization.
    
    Provides a uniform interface for different normalization approaches
    applied to 1D transposed convolutions.
    
    Args:
        *args: Positional arguments for nn.ConvTranspose1d.
        causal: Whether to use causal convolution.
        norm: Type of normalization to apply.
        nonlinearity: Type of nonlinearity for weight initialization.
        norm_kwargs: Additional arguments for normalization.
        **kwargs: Additional arguments for nn.ConvTranspose1d.
    """
    
    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = 'none',
        nonlinearity: str = 'linear',
        norm_kwargs: Dict[str, Any] = {},
        **kwargs
    ):
        super().__init__()
        
        try:
            # Create the base transposed convolution layer
            convtr = nn.ConvTranspose1d(*args, **kwargs)
            
            # Initialize weights using Kaiming initialization
            nn.init.kaiming_normal_(convtr.weight, nonlinearity=nonlinearity)
            # Note: Stride-based scaling is commented out in original
            
            if convtr.bias is not None:
                convtr.bias.data.zero_()
            
            # Apply parametrization-based normalization
            self.convtr = apply_parametrization_norm(convtr, norm, **norm_kwargs)
            
            # Get the normalization module
            self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)
            self.norm_type = norm
            
        except Exception as e:
            logger.error(
                f"Failed to initialize NormConvTranspose1d: {str(e)}",
                exc_info=True
            )
            raise
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through transposed convolution and normalization.
        
        Args:
            input_tensor: Input tensor of shape [batch, channels, length].
            
        Returns:
            Output tensor after transposed convolution and normalization.
        """
        try:
            output = self.convtr(input_tensor)
            output = self.norm(output)
            return output
            
        except Exception as e:
            logger.error(
                f"Forward pass failed in NormConvTranspose1d: {str(e)}",
                exc_info=True
            )
            raise


class NormConvTranspose2d(nn.Module):
    """ConvTranspose2d with integrated normalization.
    
    Provides a uniform interface for different normalization approaches
    applied to 2D transposed convolutions.
    
    Args:
        *args: Positional arguments for nn.ConvTranspose2d.
        norm: Type of normalization to apply.
        nonlinearity: Type of nonlinearity for weight initialization.
        norm_kwargs: Additional arguments for normalization.
        **kwargs: Additional arguments for nn.ConvTranspose2d.
    """
    
    def __init__(
        self,
        *args,
        norm: str = 'none',
        nonlinearity: str = 'linear',
        norm_kwargs: Dict[str, Any] = {},
        **kwargs
    ):
        super().__init__()
        
        try:
            # Create the base transposed convolution layer
            convtr = nn.ConvTranspose2d(*args, **kwargs)
            
            # Note: Weight initialization is commented out in original
            # This might be intentional for specific use cases
            
            # Apply parametrization-based normalization
            self.convtr = apply_parametrization_norm(convtr, norm, **norm_kwargs)
            
            # Get the normalization module (2D conv is never causal)
            self.norm = get_norm_module(
                self.convtr, causal=False, norm=norm, **norm_kwargs
            )
            
        except Exception as e:
            logger.error(
                f"Failed to initialize NormConvTranspose2d: {str(e)}",
                exc_info=True
            )
            raise
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through transposed convolution and normalization.
        
        Args:
            input_tensor: Input tensor of shape [batch, channels, height, width].
            
        Returns:
            Output tensor after transposed convolution and normalization.
        """
        try:
            output = self.convtr(input_tensor)
            output = self.norm(output)
            return output
            
        except Exception as e:
            logger.error(
                f"Forward pass failed in NormConvTranspose2d: {str(e)}",
                exc_info=True
            )
            raise


# =============================================================================
# SMART CONVOLUTION LAYERS WITH PADDING HANDLING
# =============================================================================

class SConv1d(nn.Module):
    """Smart Conv1d with automatic padding and normalization handling.
    
    This layer automatically handles asymmetric or causal padding based on
    the convolution parameters, ensuring proper output dimensions.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        dilation: Spacing between kernel elements.
        groups: Number of blocked connections from input to output.
        bias: Whether to add a learnable bias.
        causal: Whether to use causal (left-only) padding.
        norm: Type of normalization to apply.
        norm_kwargs: Additional arguments for normalization.
        pad_mode: Padding mode ('constant', 'reflect', 'replicate', 'circular').
        nonlinearity: Type of nonlinearity for weight initialization.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: str = 'none',
        norm_kwargs: Dict[str, Any] = {},
        pad_mode: str = 'constant',
        nonlinearity: str = 'linear'
    ):
        super().__init__()
        
        try:
            # Warn about potentially problematic configurations
            if stride > 1 and dilation > 1:
                warnings.warn(
                    f"SConv1d initialized with stride > 1 and dilation > 1 "
                    f"(kernel_size={kernel_size}, stride={stride}, "
                    f"dilation={dilation}). This may lead to unexpected behavior."
                )
            
            # Create the normalized convolution layer
            self.conv = NormConv1d(
                in_channels, out_channels, kernel_size, stride,
                dilation=dilation, groups=groups, bias=bias, causal=causal,
                norm=norm, norm_kwargs=norm_kwargs, nonlinearity=nonlinearity
            )
            
            self.causal = causal
            self.pad_mode = pad_mode
            
        except Exception as e:
            logger.error(
                f"Failed to initialize SConv1d: {str(e)}",
                exc_info=True
            )
            raise
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass with automatic padding.
        
        Args:
            input_tensor: Input tensor of shape [batch, channels, length].
            
        Returns:
            Output tensor after padded convolution.
        """
        try:
            # Extract convolution parameters
            kernel_size = self.conv.conv.kernel_size[0]
            stride = self.conv.conv.stride[0]
            dilation = self.conv.conv.dilation[0]
            
            # Calculate total padding needed
            # Formula ensures proper dimension preservation
            padding_total = (kernel_size - 1) * dilation - (stride - 1)
            
            # Calculate extra padding for complete windows
            extra_padding = get_extra_padding_for_conv1d(
                input_tensor, kernel_size, stride, padding_total
            )
            
            if self.causal:
                # For causal convolution, pad only on the left
                input_tensor = pad1d(
                    input_tensor,
                    (padding_total, extra_padding),
                    mode=self.pad_mode
                )
            else:
                # For non-causal, distribute padding asymmetrically if needed
                padding_right = padding_total // 2
                padding_left = padding_total - padding_right
                input_tensor = pad1d(
                    input_tensor,
                    (padding_left, padding_right + extra_padding),
                    mode=self.pad_mode
                )
            
            return self.conv(input_tensor)
            
        except Exception as e:
            logger.error(
                f"Forward pass failed in SConv1d: {str(e)}",
                exc_info=True
            )
            raise


class SConvTranspose1d(nn.Module):
    """Smart ConvTranspose1d with automatic padding removal and normalization.
    
    This layer automatically handles asymmetric or causal padding removal
    based on the transposed convolution parameters.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        dilation: Spacing between kernel elements.
        groups: Number of blocked connections from input to output.
        causal: Whether to use causal padding removal.
        norm: Type of normalization to apply.
        trim_right_ratio: Ratio of right padding to trim (causal only).
        norm_kwargs: Additional arguments for normalization.
        pad_mode: Padding mode (currently unused but kept for API consistency).
        bias: Whether to add a learnable bias.
        nonlinearity: Type of nonlinearity for weight initialization.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        causal: bool = False,
        norm: str = 'none',
        trim_right_ratio: float = 1.,
        norm_kwargs: Dict[str, Any] = {},
        pad_mode: str = "constant",
        bias: bool = True,
        nonlinearity: str = 'linear'
    ):
        super().__init__()
        
        try:
            # Validate trim_right_ratio
            if not causal and trim_right_ratio != 1.0:
                raise ValueError(
                    "`trim_right_ratio` != 1.0 only makes sense for causal "
                    "convolutions"
                )
            
            if not (0.0 <= trim_right_ratio <= 1.0):
                raise ValueError(
                    f"trim_right_ratio must be between 0.0 and 1.0, "
                    f"got {trim_right_ratio}"
                )
            
            # Create the normalized transposed convolution layer
            self.convtr = NormConvTranspose1d(
                in_channels, out_channels, kernel_size, stride,
                dilation=dilation, groups=groups, bias=bias,
                nonlinearity=nonlinearity, causal=causal,
                norm=norm, norm_kwargs=norm_kwargs
            )
            
            self.causal = causal
            self.trim_right_ratio = trim_right_ratio
            
        except Exception as e:
            logger.error(
                f"Failed to initialize SConvTranspose1d: {str(e)}",
                exc_info=True
            )
            raise
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass with automatic padding removal.
        
        Args:
            input_tensor: Input tensor of shape [batch, channels, length].
            
        Returns:
            Output tensor after transposed convolution and padding removal.
        """
        try:
            # Extract convolution parameters
            kernel_size = self.convtr.convtr.kernel_size[0]
            stride = self.convtr.convtr.stride[0]
            
            # Calculate total padding added by transposed convolution
            padding_total = kernel_size - stride
            
            # Apply transposed convolution
            output = self.convtr(input_tensor)
            
            # Remove padding based on causal/non-causal mode
            # Note: We only trim fixed padding here. Extra padding from
            # pad_for_conv1d is handled separately to maintain encoder-decoder
            # symmetry
            if self.causal:
                # For causal mode, trim padding according to ratio
                # trim_right_ratio=1.0 means trim all padding from right
                padding_right = math.ceil(padding_total * self.trim_right_ratio)
                padding_left = padding_total - padding_right
                output = unpad1d(output, (padding_left, padding_right))
            else:
                # For non-causal, distribute padding removal asymmetrically
                padding_right = padding_total // 2
                padding_left = padding_total - padding_right
                output = unpad1d(output, (padding_left, padding_right))
            
            return output
            
        except Exception as e:
            logger.error(
                f"Forward pass failed in SConvTranspose1d: {str(e)}",
                exc_info=True
            )
            raise


# =============================================================================
# STFT IMPLEMENTATION
# =============================================================================

class CausalSTFT(nn.Module):
    """Short-Time Fourier Transform implemented with Conv1d.
    
    This implementation uses Conv1d instead of torch.fft.rfft for ONNX
    compatibility. The convolution approach allows for efficient computation
    while maintaining export capabilities.
    
    The STFT decomposes a signal into its frequency components over time,
    producing a time-frequency representation.
    
    Args:
        n_fft: FFT window size (number of frequency bins = n_fft//2 + 1).
        hop_size: Number of samples between successive frames.
        win_size: Window size. If None, uses n_fft.
        win_type: Type of window function ('hann', 'hamming', etc.).
        window: Custom window tensor. If provided, overrides win_type.
        norm: FFT normalization mode ('forward', 'backward', 'ortho').
        pad_mode: Padding mode for signal edges.
        learnable: Whether the STFT weights should be learnable.
        eps: Small epsilon for numerical stability in magnitude calculation.
        device: Device to create tensors on.
        dtype: Data type for tensors.
        
    Input shape:
        - 2D: [batch, length]
        - 3D: [batch, 1, length]
        
    Output shape:
        [batch, n_fft//2+1, n_frames]
        
    Example:
        >>> stft = CausalSTFT(n_fft=1024, hop_size=256)
        >>> signal = torch.randn(32, 16000)  # 32 batch, 1 second at 16kHz
        >>> spectrum = stft(signal)  # [32, 513, 63]
    """
    
    __constants__ = ["n_fft", "hop_size", "cache_len", "norm", "eps", "pad_mode"]
    
    def __init__(
        self,
        n_fft: int,
        hop_size: int,
        win_size: Optional[int] = None,
        win_type: Optional[str] = "hann",
        window: Optional[Tensor] = None,
        norm: Optional[str] = "backward",
        pad_mode: str = "constant",
        learnable: bool = False,
        eps: float = 1e-12,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        
        try:
            # Validate parameters
            if n_fft <= 0 or hop_size <= 0:
                raise ValueError(
                    f"n_fft and hop_size must be positive, got n_fft={n_fft}, "
                    f"hop_size={hop_size}"
                )
            
            if norm not in ["forward", "backward", "ortho", None]:
                raise ValueError(
                    f"Invalid norm mode: {norm}. Must be 'forward', "
                    f"'backward', 'ortho', or None"
                )
            
            # Store parameters
            self.n_fft = n_fft
            self.hop_size = hop_size
            self.cache_len = n_fft - 1  # Padding for causal STFT
            self.norm = norm
            self.pad_mode = pad_mode
            self.eps = eps
            
            # Set default dtype if not provided
            if dtype is None:
                dtype = torch.float32
            factory_kwargs = {'device': device, 'dtype': dtype}
            
            # Create or validate window
            if win_size is None:
                win_size = n_fft
            
            if window is not None:
                # Use provided window
                win_size = window.size(-1)
                if win_size < n_fft:
                    # Pad window to n_fft size
                    padding = n_fft - win_size
                    window = F.pad(window, (padding//2, padding - padding//2))
            elif win_type is None:
                # No window (rectangular)
                window = torch.ones(n_fft, **factory_kwargs)
            else:
                # Create window using torch window functions
                try:
                    window_fn = getattr(torch, f"{win_type}_window")
                    window = window_fn(win_size, device=device)
                except AttributeError:
                    raise ValueError(f"Unknown window type: {win_type}")
                
                if win_size < n_fft:
                    # Pad window to n_fft size
                    padding = n_fft - win_size
                    window = F.pad(window, (padding//2, padding - padding//2))
            
            if n_fft < win_size:
                raise ValueError(
                    f"n_fft ({n_fft}) must be >= win_size ({win_size})"
                )
            
            # Create DFT matrix for convolution-based STFT
            # n: time indices, k: frequency indices
            time_indices = torch.arange(n_fft, **factory_kwargs).view(1, 1, n_fft)
            freq_indices = torch.arange(n_fft//2+1, **factory_kwargs).view(-1, 1, 1)
            
            # DFT basis functions: exp(-2πi*k*n/N) = cos - i*sin
            angular_freq = -2 * math.pi / n_fft * freq_indices * time_indices
            cos_basis = torch.cos(angular_freq)
            sin_basis = torch.sin(angular_freq)
            
            # Stack real and imaginary parts and apply window
            weight = torch.cat([cos_basis, sin_basis], dim=0) * window
            
            # Apply normalization
            if norm == "forward":
                weight /= n_fft
            elif norm == "backward":
                pass  # No scaling
            elif norm == "ortho":
                weight /= math.sqrt(n_fft)
            
            # Register as parameter or buffer
            if learnable:
                self.weight = nn.Parameter(weight)
            else:
                self.register_buffer("weight", weight)
                self.weight: Tensor  # Type annotation for clarity
                
        except Exception as e:
            logger.error(
                f"Failed to initialize CausalSTFT: {str(e)}",
                exc_info=True
            )
            raise
    
    def forward(self, input_signal: Tensor) -> Tensor:
        """Compute STFT of input signal.
        
        Args:
            input_signal: Input signal tensor of shape [batch, length] or
                         [batch, 1, length].
                         
        Returns:
            Magnitude spectrum of shape [batch, n_fft//2+1, n_frames].
            
        Raises:
            RuntimeError: If STFT computation fails.
        """
        try:
            # Ensure 3D input: [batch, channels, length]
            if input_signal.dim() == 2:
                input_signal = input_signal.unsqueeze(1)
            
            # Apply causal padding (pad left side)
            input_signal = F.pad(
                input_signal,
                (self.cache_len, 0),
                mode=self.pad_mode
            )
            
            # Compute STFT using convolution
            # Output shape: [batch, 2*(n_fft//2+1), n_frames]
            stft_complex = F.conv1d(
                input_signal,
                self.weight,
                bias=None,
                stride=self.hop_size
            )
            
            # Reshape to separate real and imaginary parts
            batch_size, channels, n_frames = stft_complex.shape
            stft_complex = stft_complex.view(
                batch_size, 2, channels//2, n_frames
            )
            
            # Compute magnitude: sqrt(real^2 + imag^2)
            # Using clamp_min for numerical stability
            magnitude = stft_complex.square().sum(dim=1).clamp_min(self.eps).sqrt()
            
            return magnitude
            
        except Exception as e:
            logger.error(
                f"STFT forward pass failed: {str(e)}",
                exc_info=True
            )
            raise RuntimeError(f"STFT computation failed: {str(e)}") from e