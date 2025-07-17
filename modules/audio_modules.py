"""Audio processing modules for real-time watermarking.

This module provides implementations of various audio transforms including:
- STDCT: Short-Time Discrete Cosine Transform
- MDCT: Modified Discrete Cosine Transform  
- STFT: Short-Time Fourier Transform
- PQMF: Pseudo-Quadrature Mirror Filter

These transforms are optimized for real-time audio processing and watermarking applications.
"""

# =============================================================================
# IMPORTS
# =============================================================================
# Standard library imports
import logging
import math
import warnings
from typing import Any, List, Optional, Tuple, Union

# Third-party imports
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from scipy.signal.windows import kaiser
from torch import Tensor, nn

# =============================================================================
# LOGGING SETUP
# =============================================================================
logger = logging.getLogger(__name__)

# =============================================================================
# MODULE CONSTANTS
# =============================================================================
# Mathematical constants
PI = math.pi
SQRT_2 = math.sqrt(2)

# Default parameters
DEFAULT_WINDOW_TYPE = "hann"
DEFAULT_PAD_MODE = "reflect"
DEFAULT_BETA = 9.0
DEFAULT_CUTOFF_RATIO = 0.142
DEFAULT_TAPS = 62
DEFAULT_SUBBANDS = 4


# =============================================================================
# TRANSFORM CLASSES
# =============================================================================

class STDCT(nn.Module):
    """Short-Time Discrete Cosine Transform Type-II.
    
    Implements the DCT-II transform with overlapping windows for time-frequency analysis.
    This is commonly used in audio coding and watermarking applications.
    
    The DCT-II formula: X[k] = sum(x[n] * cos(π/N * k * (n + 0.5)))
    
    Args:
        N: DCT size (number of frequency bins)
        hop_size: Number of samples between successive frames
        win_size: Window size (defaults to N if not specified)
        win_type: Type of window function ("hann", "hamming", etc.)
        center: If True, centers the signal by padding
        window: Custom window tensor (overrides win_type if provided)
        device: Device to place tensors on
        dtype: Data type for tensors
        
    Raises:
        ValueError: If N < win_size
        RuntimeError: If NOLA constraint is violated during inverse transform
    """

    __constants__ = ["N", "hop_size", "padding"]

    def __init__(self, N: int, hop_size: int, win_size: Optional[int] = None,
                 win_type: Optional[str] = DEFAULT_WINDOW_TYPE, center: bool = False,
                 window: Optional[Tensor] = None, device: Optional[torch.device] = None, 
                 dtype: Optional[torch.dtype] = None) -> None:
        """Initialize STDCT transform.
        
        Args:
            N: DCT size (number of frequency bins)
            hop_size: Number of samples between successive frames
            win_size: Window size (defaults to N if not specified)
            win_type: Type of window function ("hann", "hamming", etc.)
            center: If True, centers the signal by padding
            window: Custom window tensor (overrides win_type if provided)
            device: Device to place tensors on
            dtype: Data type for tensors
            
        Raises:
            ValueError: If N < win_size or invalid window type
        """
        super().__init__()
        
        # Validate inputs
        if N <= 0 or hop_size <= 0:
            raise ValueError(f"N ({N}) and hop_size ({hop_size}) must be positive")
            
        self.N = N
        self.hop_size = hop_size
        
        # Calculate padding based on center mode
        if center:
            self.padding = (N + 1) // 2  # Equivalent to ceil(N / 2)
            self.output_padding = N % 2
        else:
            self.padding = (N - hop_size + 1) // 2  # Equivalent to ceil((N - hop_size) / 2)
            self.output_padding = (N - hop_size) % 2
            self.clip = (hop_size % 2 == 1)

        factory_kwargs = {'device': device, 'dtype': dtype}

        # Initialize window
        if win_size is None:
            win_size = N
        
        try:
            if window is not None:
                win_size = window.size(-1)
                if win_size < N:
                    # Pad window symmetrically to match DCT size
                    padding = N - win_size
                    window = F.pad(window, (padding//2, padding - padding//2))
            elif win_type is None:
                window = torch.ones(N, dtype=torch.float32, device=device)
            else:
                # Get window function from torch
                window_fn = getattr(torch, f"{win_type}_window", None)
                if window_fn is None:
                    raise ValueError(f"Unknown window type: {win_type}")
                window = window_fn(win_size, device=device)
                if win_size < N:
                    padding = N - win_size
                    window = F.pad(window, (padding//2, padding - padding//2))
        except Exception as e:
            logger.error(f"Error creating window: {str(e)}", exc_info=True)
            raise
            
        if N < win_size:
            raise ValueError(f"N ({N}) must be >= win_size ({win_size})")
        
        # Create DCT-II filter bank
        time_indices = torch.arange(N, dtype=torch.float32, device=device).view(1, 1, N)
        freq_indices = time_indices.view(N, 1, 1)
        
        # DCT-II basis functions: cos(π/N * k * (n + 0.5))
        dct_basis = torch.cos(PI/N * freq_indices * (time_indices + 0.5)) * math.sqrt(2/N)
        dct_basis[0, 0, :] /= SQRT_2  # Normalize DC component
        
        # Apply window to filter
        dct_filter = (dct_basis * window.view(1, 1, N)).to(**factory_kwargs)
        window_square = window.square().view(1, -1, 1).to(**factory_kwargs)
        
        self.register_buffer('filter', dct_filter)
        self.register_buffer('window_square', window_square)
        self.filter: Tensor
        self.window_square: Tensor
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply STDCT transform to input signal.
        
        Args:
            x: Input tensor of shape [B, 1, hop_size*T] or [B, hop_size*T]
            
        Returns:
            Tensor: DCT coefficients of shape [B, N, T+1] (center=True) or [B, N, T] (center=False)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Apply DCT filter bank using 1D convolution
        spectrum = F.conv1d(x, self.filter, bias=None, stride=self.hop_size,
                           padding=self.padding)
        
        # Remove extra frame if needed (for odd hop sizes when center=False)
        if hasattr(self, 'clip') and self.clip:
            spectrum = spectrum[:, :, :-1]
            
        return spectrum
    
    def inverse(self, spec: Tensor) -> Tensor:
        """Apply inverse STDCT transform.
        
        Args:
            spec: DCT coefficients of shape [B, N, T+1] (center=True) or [B, N, T] (center=False)
            
        Returns:
            Tensor: Reconstructed signal of shape [B, 1, hop_size*T]
            
        Raises:
            RuntimeError: If NOLA constraint is violated (division by zero in overlap-add)
        """
        # Apply inverse DCT using transposed convolution
        wav = F.conv_transpose1d(spec, self.filter, bias=None, stride=self.hop_size,
                                padding=self.padding, output_padding=self.output_padding)
        
        batch_size, num_frames = spec.size(0), spec.size(-1)
        window_square = self.window_square.expand(batch_size, -1, num_frames)
        
        # Calculate output length
        output_length = (self.hop_size * num_frames + (self.N - self.hop_size) 
                        - 2 * self.padding + self.output_padding)
        
        # Compute overlap-add normalization using fold operation
        window_square_inverse = F.fold(
            window_square,
            output_size=(1, output_length),
            kernel_size=(1, self.N),
            stride=(1, self.hop_size),
            padding=(0, self.padding)
        ).squeeze(2)

        # Check NOLA (Nonzero Overlap-Add) constraint
        if not torch.all(torch.ne(window_square_inverse, 0.0)):
            logger.error("NOLA constraint violated: division by zero in overlap-add")
            raise RuntimeError("NOLA constraint violated: window configuration results in "
                             "zero values in overlap-add normalization")
            
        return wav / window_square_inverse


class MDCT(nn.Module):
    """Modified Discrete Cosine Transform.
    
    Implements the MDCT/IMDCT pair for perfect reconstruction filterbanks.
    The MDCT is widely used in audio coding (MP3, AAC) due to its 
    critical sampling property and energy compaction.
    
    MDCT formula: X[k] = sum(x[n] * cos(π/N * (n + 0.5 + N/2) * (k + 0.5)))
    
    Args:
        N: Transform size (frame size = 2*N)
        normalize: If True, applies energy normalization
        device: Device to place tensors on
        dtype: Data type for tensors
    """

    __constants__ = ["N", "filter", "normalize"]

    def __init__(self, N: int, normalize: bool = True, 
                 device: Optional[torch.device] = None, 
                 dtype: Optional[torch.dtype] = None) -> None:
        """Initialize MDCT transform.
        
        Args:
            N: Transform size (frame size = 2*N)
            normalize: If True, applies energy normalization
            device: Device to place tensors on
            dtype: Data type for tensors
            
        Raises:
            ValueError: If N is not positive
        """
        super().__init__()
        
        if N <= 0:
            raise ValueError(f"N must be positive, got {N}")
            
        self.N = N
        self.normalize = normalize

        try:
            # Create frequency indices k: [0, 1, ..., N-1]
            freq_indices = torch.arange(N, dtype=torch.float32, device=device).view(N, 1, 1)
            # Create time indices n: [0, 1, ..., 2N-1]
            time_indices = torch.arange(2*N, dtype=torch.float32, device=device).view(1, 1, 2*N)
            
            # MDCT basis: cos(π/N * (n + 0.5 + N/2) * (k + 0.5))
            mdct_filter = torch.cos(PI/N * (time_indices + 0.5 + N/2) * (freq_indices + 0.5))
            
            if normalize:
                mdct_filter /= math.sqrt(N)
                
            mdct_filter = mdct_filter.to(device=device, dtype=dtype)
            self.register_buffer("filter", mdct_filter)
            self.filter: Tensor
            
        except Exception as e:
            logger.error(f"Error creating MDCT filter: {str(e)}", exc_info=True)
            raise

    def forward(self, x: Tensor) -> Tensor:
        """Apply MDCT transform.
        
        Args:
            x: Input signal of shape [B, 1, N*T]
            
        Returns:
            Tensor: MDCT coefficients of shape [B, N, T+1]
        """
        # Apply MDCT using strided convolution with N-sample padding
        return F.conv1d(x, self.filter, bias=None, stride=self.N, padding=self.N)
    
    def inverse(self, x: Tensor) -> Tensor:
        """Apply inverse MDCT transform.
        
        Args:
            x: MDCT coefficients of shape [B, N, T+1]
            
        Returns:
            Tensor: Reconstructed signal of shape [B, 1, N*T]
        """
        # Apply scaling for non-normalized mode
        if self.normalize:
            mdct_filter = self.filter
        else:
            # Scale by 1/N for perfect reconstruction in non-normalized mode
            mdct_filter = self.filter / self.N
            
        # Apply inverse MDCT using transposed convolution
        return F.conv_transpose1d(x, mdct_filter, bias=None, stride=self.N, padding=self.N)


class STFT(nn.Module):
    """Short-Time Fourier Transform.
    
    Provides a PyTorch-compatible STFT/ISTFT implementation with various
    windowing options and parameterization for audio processing.
    
    Args:
        n_fft: FFT size
        hop_size: Number of samples between successive frames
        win_size: Window size (defaults to n_fft)
        center: If True, centers the signal by reflection padding
        magnitude: If True, returns magnitude instead of complex values
        win_type: Type of window function
        window: Custom window tensor
        normalized: Apply normalization in ISTFT
        pad_mode: Padding mode for center=True
        backend: Implementation backend ("torch" or "custom")
        device: Device to place tensors on
        dtype: Data type for tensors
        
    Raises:
        ValueError: If n_fft < win_size or invalid backend
        NotImplementedError: If center=False for inverse transform
    """

    __constants__ = ["n_fft", "hop_size", "normalize", "pad_mode"]

    def __init__(self, n_fft: int, hop_size: int, win_size: Optional[int] = None,
                 center: bool = True, magnitude: bool = True,
                 win_type: Optional[str] = DEFAULT_WINDOW_TYPE,
                 window: Optional[Tensor] = None, normalized: bool = False,
                 pad_mode: str = DEFAULT_PAD_MODE,
                 backend: str = "torch", device: Optional[torch.device] = None, 
                 dtype: Optional[torch.dtype] = None) -> None:
        """Initialize STFT transform.
        
        Args:
            n_fft: FFT size
            hop_size: Number of samples between successive frames
            win_size: Window size (defaults to n_fft)
            center: If True, centers the signal by reflection padding
            magnitude: If True, returns magnitude instead of complex values
            win_type: Type of window function
            window: Custom window tensor
            normalized: Apply normalization in ISTFT
            pad_mode: Padding mode for center=True
            backend: Implementation backend ("torch" or "custom")
            device: Device to place tensors on
            dtype: Data type for tensors
            
        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__()
        
        # Validate inputs
        if backend not in ["torch", "custom"]:
            raise ValueError(f"Invalid backend: {backend}. Must be 'torch' or 'custom'")
            
        if n_fft <= 0 or hop_size <= 0:
            raise ValueError(f"n_fft ({n_fft}) and hop_size ({hop_size}) must be positive")
            
        self.backend = backend
        self.normalized = normalized
        self.center = center
        self.magnitude = magnitude
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.padding = 0 if center else (n_fft + 1 - hop_size) // 2
        self.clip = (hop_size % 2 == 1) and not center
        self.pad_mode = pad_mode
        
        if win_size is None:
            win_size = n_fft
        
        try:
            # Initialize window
            if window is not None:
                win_size = window.size(-1)
            elif win_type is None:
                window = torch.ones(win_size, device=device, dtype=dtype)
            else:
                # Get window function from torch
                window_fn = getattr(torch, f"{win_type}_window", None)
                if window_fn is None:
                    raise ValueError(f"Unknown window type: {win_type}")
                window = window_fn(win_size, device=device, dtype=dtype)
                
            self.register_buffer("window", window)
            self.window: Optional[Tensor] = self.window  # Type hint for registered buffer
            self.win_size = win_size
            
        except Exception as e:
            logger.error(f"Error creating window: {str(e)}", exc_info=True)
            raise
            
        if n_fft < win_size:
            raise ValueError(f"n_fft ({n_fft}) must be >= win_size ({win_size})")
        
        if backend == "custom":
            raise NotImplementedError("Custom backend is not implemented yet")

    def forward(self, x: Tensor) -> Tensor:
        """Apply STFT transform.
        
        Args:
            x: Input signal of shape [B, T_wav] or [B, 1, T_wav]
            
        Returns:
            Tensor: Spectrogram of shape [B, n_fft//2+1, T_spec(, 2)]
                    Last dimension is present only if magnitude=False
        """
        # Handle 3D input
        if x.dim() == 3:  # [B, 1, T] -> [B, T]
            x = x.squeeze(1)
            
        # Apply padding for non-centered mode
        if self.padding > 0:
            x = F.pad(x.unsqueeze(0), (self.padding, self.padding), 
                     mode=self.pad_mode).squeeze(0)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Ignore torch.stft deprecation warnings
                spec = torch.stft(x, self.n_fft, hop_length=self.hop_size, 
                                 win_length=self.win_size, window=self.window, 
                                 center=self.center, pad_mode=self.pad_mode,
                                 normalized=False, onesided=True, return_complex=False)
            
            # Convert to magnitude if requested
            if self.magnitude:
                spec = torch.linalg.norm(spec, dim=-1)
            
            # Remove extra frame for odd hop sizes in non-centered mode
            if self.clip:
                spec = spec[:, :, :-1]

            return spec
            
        except Exception as e:
            logger.error(f"Error in STFT forward: {str(e)}", exc_info=True)
            raise

    def inverse(self, spec: Tensor) -> Tensor:
        """Apply inverse STFT transform.
        
        Args:
            spec: Complex spectrogram of shape [B, n_fft//2+1, T_spec, 2]
            
        Returns:
            Tensor: Reconstructed signal of shape [B, T_wav]
            
        Raises:
            NotImplementedError: If center=False (not supported for inverse)
        """
        if not self.center:
            raise NotImplementedError(
                "Inverse STFT with center=False is not implemented. "
                "Please use center=True for inverse transforms."
            )
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Ignore torch.istft deprecation warnings
                wav = torch.istft(spec, self.n_fft, hop_length=self.hop_size,
                                 win_length=self.win_size, center=self.center, 
                                 normalized=self.normalized, window=self.window, 
                                 onesided=True, return_complex=False)

            return wav
            
        except Exception as e:
            logger.error(f"Error in STFT inverse: {str(e)}", exc_info=True)
            raise


# =============================================================================
# FILTER DESIGN FUNCTIONS
# =============================================================================

def design_prototype_filter(taps: int = DEFAULT_TAPS, 
                          cutoff_ratio: float = DEFAULT_CUTOFF_RATIO, 
                          beta: float = DEFAULT_BETA) -> np.ndarray:
    """Design prototype filter for PQMF using Kaiser window method.
    
    This method implements the approach described in "A Kaiser window approach 
    for the design of prototype filters of cosine modulated filterbanks" 
    (https://ieeexplore.ieee.org/abstract/document/681427).
    
    The filter is designed using the windowed sinc method with a Kaiser window
    for optimal stopband attenuation.
    
    Args:
        taps: Number of filter taps (must be even)
        cutoff_ratio: Normalized cutoff frequency (0 < ratio < 1)
        beta: Kaiser window shape parameter (higher = more stopband attenuation)
        
    Returns:
        np.ndarray: Impulse response of prototype filter with shape (taps + 1,)
        
    Raises:
        ValueError: If taps is odd or cutoff_ratio is out of range
    """
    # Validate arguments
    if taps % 2 != 0:
        raise ValueError(f"Number of taps must be even, got {taps}")
    if not 0.0 < cutoff_ratio < 1.0:
        raise ValueError(f"Cutoff ratio must be between 0 and 1, got {cutoff_ratio}")

    try:
        # Calculate normalized cutoff frequency
        omega_c = PI * cutoff_ratio
        
        # Create time indices centered at zero
        n = np.arange(taps + 1) - 0.5 * taps
        
        # Design ideal lowpass filter using sinc function
        # h[n] = sin(ωc * n) / (π * n)
        with np.errstate(invalid="ignore"):  # Handle division by zero at n=0
            h_ideal = np.sin(omega_c * n) / (PI * n)
            
        # Fix the center tap (n=0) using L'Hôpital's rule
        h_ideal[taps // 2] = cutoff_ratio

        # Apply Kaiser window for sidelobe control
        kaiser_window = kaiser(taps + 1, beta)
        h_prototype = h_ideal * kaiser_window

        return h_prototype
        
    except Exception as e:
        logger.error(f"Error designing prototype filter: {str(e)}", exc_info=True)
        raise


class PQMF(nn.Module):
    """Pseudo-Quadrature Mirror Filter Bank.
    
    Implements a critically sampled cosine-modulated filterbank for
    multiresolution audio analysis and synthesis. PQMF provides perfect
    reconstruction with proper filter design.
    
    The filters are cosine-modulated versions of a prototype lowpass filter,
    designed using the Kaiser window method for optimal frequency selectivity.
    
    Args:
        subbands: Number of frequency subbands
        taps: Number of filter taps
        cutoff_freq: Normalized cutoff frequency for prototype filter
        beta: Kaiser window shape parameter
        
    Raises:
        ValueError: If parameters are invalid
    """
    
    def __init__(self, subbands: int = DEFAULT_SUBBANDS, taps: int = DEFAULT_TAPS, 
                 cutoff_freq: float = DEFAULT_CUTOFF_RATIO, 
                 beta: float = DEFAULT_BETA) -> None:
        """Initialize PQMF filterbank.
        
        Args:
            subbands: Number of frequency subbands (typically 4, 8, or 16)
            taps: Number of filter taps (must be even)
            cutoff_freq: Normalized cutoff frequency for prototype filter
            beta: Kaiser window shape parameter
        """
        super().__init__()
        
        # Validate inputs
        if subbands <= 0:
            raise ValueError(f"Number of subbands must be positive, got {subbands}")
            
        try:
            # Design prototype lowpass filter
            h_proto = torch.from_numpy(
                design_prototype_filter(taps, cutoff_freq, beta)
            ).to(dtype=torch.float32).unsqueeze(0)
            
            # Create modulation indices
            # k: subband index [0, 1, ..., subbands-1]
            # n: time index [0, 1, ..., taps]
            subband_idx = torch.arange(subbands, dtype=torch.float32).unsqueeze(1)
            time_idx = torch.arange(taps + 1, dtype=torch.float32).unsqueeze(0)
            
            # Generate cosine-modulated filter bank
            # Each subband filter is a cosine-modulated version of the prototype
            # Phase term includes both linear phase and alternating π/4 shift
            modulation = torch.cos(
                (2 * subband_idx + 1) * PI / (2 * subbands) * (time_idx - taps / 2) 
                + (-1)**subband_idx * PI / 4
            )
            
            # Apply modulation to prototype filter with proper scaling
            # Factor of 2 for real-valued transform, sqrt(subbands) for energy normalization
            pqmf_filter = 2 * h_proto * modulation.unsqueeze(1) * math.sqrt(subbands)
            
            self.taps = taps
            self.subbands = subbands
            self.register_buffer("pqmf_filter", pqmf_filter)
            self.pqmf_filter: Tensor
            
        except Exception as e:
            logger.error(f"Error initializing PQMF: {str(e)}", exc_info=True)
            raise
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass defaults to analysis transform.
        
        Args:
            x: Input signal
            
        Returns:
            Tensor: Subband signals
        """
        return self.analysis(x)
    
    def analysis(self, x: Tensor) -> Tensor:
        """Apply PQMF analysis (decomposition) transform.
        
        Decomposes the input signal into multiple frequency subbands
        using the cosine-modulated filterbank.
        
        Args:
            x: Input signal of shape [B, T] or [B, 1, T]
            
        Returns:
            Tensor: Subband signals of shape [B, subbands, T//subbands]
        """
        # Ensure 3D input shape
        if x.dim() == 2:  # [B, T] -> [B, 1, T]
            x = x.unsqueeze(1)
            
        # Apply filterbank using strided convolution
        # Stride = subbands for critical sampling
        subband_signals = F.conv1d(
            x, self.pqmf_filter, None, 
            stride=self.subbands, 
            padding=self.taps//2  # Maintain signal length
        )
        
        return subband_signals
    
    def synthesis(self, x: Tensor) -> Tensor:
        """Apply PQMF synthesis (reconstruction) transform.
        
        Reconstructs the full-band signal from subband components
        using the transposed filterbank operation.
        
        Args:
            x: Subband signals of shape [B, subbands, T//subbands]
            
        Returns:
            Tensor: Reconstructed signal of shape [B, 1, T]
        """
        padding = self.taps // 2
        
        # Apply synthesis filterbank using transposed convolution
        # Output padding ensures correct output length for critical sampling
        reconstructed = F.conv_transpose1d(
            x, self.pqmf_filter, None, 
            stride=self.subbands, 
            padding=padding,
            output_padding=self.subbands - 1
        )
        
        return reconstructed