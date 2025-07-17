"""
Audio evaluation metrics module for watermarking system performance assessment.

This module provides various audio quality and watermark detection metrics including:
- STOI (Short-Time Objective Intelligibility)
- SI-SNR (Scale-Invariant Source-to-Noise Ratio)
- PESQ (Perceptual Evaluation of Speech Quality)
- BER (Bit Error Rate) for watermark detection
- MIOU (Mean Intersection over Union) for localization

These metrics are used to evaluate both the perceptual quality of watermarked audio
and the accuracy of watermark detection and localization.
"""

# =============================================================================
# IMPORTS
# =============================================================================
# Standard library imports
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pystoi
import torch
from audiotools import AudioSignal
from pesq import NoUtterancesError, pesq as pesq_fn
from torch import nn

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_PESQ_MODE = "wb"  # Wide-band mode
DEFAULT_BER_THRESHOLD = 0.5
EPSILON = 1e-8  # Small value to avoid division by zero

# Plot styling constants
PLOT_STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'axes.titlepad': 27,
    'axes.labelpad': 25,
    'font.size': 36
}

COLORS = {
    'background': '#FFFFFF',
    'text': '#2E3440',
    'primary': '#5E81AC',
}

# =============================================================================
# AUDIO QUALITY METRICS
# =============================================================================

class STOI(nn.Module):
    """
    Short-Time Objective Intelligibility (STOI) metric for audio quality assessment.
    
    STOI is a metric that predicts the intelligibility of noisy/processed speech
    relative to clean speech. Values range from 0 to 1, with higher values
    indicating better intelligibility.
    """
    
    def __init__(self, extended: bool = False) -> None:
        """
        Initializes the STOI computation utility.

        Args:
            extended: Whether to use the extended STOI algorithm which is more
                     accurate for severely degraded speech. Default is False.
        """
        super(STOI, self).__init__()
        self.extended = extended
        logger.debug(f"Initialized STOI metric with extended={extended}")

    def forward(self, estimates: AudioSignal, references: AudioSignal) -> torch.Tensor:
        """
        Computes the STOI score for batches of audio signals.

        Args:
            estimates: The processed or denoised audio signal batch.
            references: The reference or original audio signal batch.

        Returns:
            Mean STOI score across the batch as a scalar tensor.
            
        Raises:
            ValueError: If batch sizes don't match or sample rates differ.
            RuntimeError: If STOI computation fails for any sample.
        """
        try:
            # Validate inputs
            if estimates.batch_size != references.batch_size:
                raise ValueError(f"Batch size mismatch: estimates={estimates.batch_size}, "
                               f"references={references.batch_size}")
            
            # Clone and convert to mono for STOI computation
            estimates = estimates.clone().to_mono()
            references = references.clone().to_mono()
            
            stoi_scores: List[float] = []
            
            for i in range(estimates.batch_size):
                try:
                    # Extract numpy arrays from audio tensors
                    est_np = estimates.audio_data[i, 0].detach().cpu().numpy()
                    ref_np = references.audio_data[i, 0].detach().cpu().numpy()
                    
                    # Compute STOI score for current sample pair
                    stoi_score = pystoi.stoi(
                        ref_np,
                        est_np,
                        references.sample_rate,
                        extended=self.extended,
                    )
                    stoi_scores.append(stoi_score)
                    
                except Exception as e:
                    logger.error(f"Failed to compute STOI for sample {i}: {str(e)}", exc_info=True)
                    raise RuntimeError(f"STOI computation failed for sample {i}") from e
            
            # Convert to tensor and compute mean
            stois_tensor = torch.tensor(stoi_scores, dtype=torch.float32, 
                                       device=estimates.audio_data.device)
            mean_stoi = stois_tensor.mean()
            
            logger.debug(f"Computed STOI scores: mean={mean_stoi:.4f}, "
                        f"std={stois_tensor.std():.4f}")
            
            return mean_stoi
            
        except Exception as e:
            logger.error(f"Error in STOI forward pass: {str(e)}", exc_info=True)
            raise

class SISNR(nn.Module):
    """
    Scale-Invariant Source-to-Noise Ratio (SI-SNR) metric for audio quality assessment.
    
    SI-SNR is invariant to the scaling of the target and estimated signals, making it
    suitable for evaluating source separation and speech enhancement systems. Higher
    values indicate better quality, typically ranging from -20 to 20 dB.
    """
    
    def __init__(self, eps: float = EPSILON) -> None:
        """
        Initializes the SISNR module.

        Args:
            eps: Small value to avoid division by zero and numerical instability.
                Defaults to EPSILON (1e-8).
        """
        super(SISNR, self).__init__()
        self.eps = eps
        logger.debug(f"Initialized SI-SNR metric with eps={eps}")

    def forward(self, estimates: AudioSignal, references: AudioSignal) -> torch.Tensor:
        """
        Calculate the mean SI-SNR for a batch of audio signal pairs.

        Args:
            estimates: Output/estimated signal batch with shape [B, C, T].
            references: Reference/target signal batch with shape [B, C, T].

        Returns:
            Mean SI-SNR across the batch in dB as a scalar tensor.
            
        Raises:
            ValueError: If batch sizes don't match or signals have different shapes.
            RuntimeError: If SI-SNR computation fails.
        """
        try:
            # Ensure signals are on the same device
            estimates = estimates.to(references.device)
            
            # Extract audio tensors and remove channel dimension
            out_sig = estimates.audio_data.squeeze(1)  # Shape: [batch_size, time]
            ref_sig = references.audio_data.squeeze(1)  # Shape: [batch_size, time]
            
            # Validate batch dimensions
            if ref_sig.size(0) != out_sig.size(0):
                raise ValueError(f"Batch size mismatch: references={ref_sig.size(0)}, "
                               f"estimates={out_sig.size(0)}")
            
            if ref_sig.size(1) != out_sig.size(1):
                raise ValueError(f"Time dimension mismatch: references={ref_sig.size(1)}, "
                               f"estimates={out_sig.size(1)}")
            
            # Zero-mean normalization
            ref_sig = ref_sig - torch.mean(ref_sig, dim=1, keepdim=True)
            out_sig = out_sig - torch.mean(out_sig, dim=1, keepdim=True)
            
            # Calculate reference signal energy
            ref_energy = torch.sum(ref_sig ** 2, dim=1, keepdim=True) + self.eps
            
            # Project estimated signal onto reference signal
            # proj = <s_target, s_estimate> / ||s_target||^2 * s_target
            dot_product = torch.sum(ref_sig * out_sig, dim=1, keepdim=True)
            proj = dot_product * ref_sig / ref_energy
            
            # Calculate noise component
            noise = out_sig - proj
            
            # Calculate SI-SNR = 10 * log10(||proj||^2 / ||noise||^2)
            proj_power = torch.sum(proj ** 2, dim=1)
            noise_power = torch.sum(noise ** 2, dim=1) + self.eps
            ratio = proj_power / noise_power
            sisnr = 10 * torch.log10(ratio + self.eps)
            
            mean_sisnr = sisnr.mean()
            
            logger.debug(f"Computed SI-SNR: mean={mean_sisnr:.2f} dB, "
                        f"std={sisnr.std():.2f} dB")
            
            return mean_sisnr
            
        except Exception as e:
            logger.error(f"Error in SI-SNR computation: {str(e)}", exc_info=True)
            raise RuntimeError("SI-SNR computation failed") from e

class PESQ(nn.Module):
    """
    Perceptual Evaluation of Speech Quality (PESQ) metric implementation.
    
    PESQ is an ITU-T standard (P.862) for automated assessment of speech quality.
    It provides MOS-LQO (Mean Opinion Score - Listening Quality Objective) scores
    ranging from -0.5 to 4.5, with higher scores indicating better quality.
    """
    
    def __init__(self, mode: str = DEFAULT_PESQ_MODE, target_sr: int = DEFAULT_SAMPLE_RATE) -> None:
        """
        Initializes the PESQ computation utility.

        Args:
            mode: PESQ mode - 'wb' for wide-band (16kHz) or 'nb' for narrow-band (8kHz).
                 Default is 'wb'.
            target_sr: Target sample rate for resampling audio before PESQ computation.
                      Should be 16000 for 'wb' mode or 8000 for 'nb' mode.
                      
        Raises:
            ValueError: If invalid mode or mismatched mode/sample rate combination.
        """
        super(PESQ, self).__init__()
        
        # Validate mode
        if mode not in ['wb', 'nb']:
            raise ValueError(f"Invalid PESQ mode: {mode}. Must be 'wb' or 'nb'")
        
        # Validate mode and sample rate combination
        if mode == 'wb' and target_sr != 16000:
            logger.warning(f"Wide-band mode typically uses 16kHz, got {target_sr}Hz")
        elif mode == 'nb' and target_sr != 8000:
            logger.warning(f"Narrow-band mode typically uses 8kHz, got {target_sr}Hz")
        
        self.mode = mode
        self.target_sr = target_sr
        logger.debug(f"Initialized PESQ metric with mode={mode}, target_sr={target_sr}")

    def forward(self, estimates: AudioSignal, references: AudioSignal) -> torch.Tensor:
        """
        Compute PESQ scores for a batch of audio signals.

        Args:
            estimates: Degraded/processed speech signals batch.
            references: Reference/original speech signals batch.

        Returns:
            Mean PESQ score (MOS-LQO) across valid samples in the batch.
            Returns 0.0 if no valid samples could be processed.
            
        Raises:
            ValueError: If batch sizes don't match.
            RuntimeError: If all samples fail PESQ computation.
        """
        try:
            # Validate inputs
            if estimates.batch_size != references.batch_size:
                raise ValueError(f"Batch size mismatch: estimates={estimates.batch_size}, "
                               f"references={references.batch_size}")
            
            # Prepare signals: convert to mono and resample
            estimates = estimates.clone().to_mono().resample(self.target_sr)
            references = references.clone().to_mono().resample(self.target_sr)
            
            pesq_scores: List[float] = []
            failed_samples = 0
            
            for i in range(estimates.batch_size):
                try:
                    # Convert to numpy for PESQ computation
                    ref_np = references.audio_data[i, 0].detach().cpu().numpy()
                    est_np = estimates.audio_data[i, 0].detach().cpu().numpy()
                    
                    # Compute PESQ score
                    pesq_score = pesq_fn(
                        self.target_sr,
                        ref_np,
                        est_np,
                        self.mode,
                    )
                    pesq_scores.append(pesq_score)
                    
                except NoUtterancesError:
                    # This occurs when the signal is too quiet or corrupted
                    logger.warning(f"No utterances detected in sample {i}, skipping")
                    failed_samples += 1
                    continue
                except Exception as e:
                    logger.error(f"PESQ computation failed for sample {i}: {str(e)}", 
                               exc_info=True)
                    failed_samples += 1
                    continue
            
            # Check if we have any valid scores
            if not pesq_scores:
                logger.error(f"All {estimates.batch_size} samples failed PESQ computation")
                return torch.tensor(0.0, device=estimates.device)
            
            # Convert to tensor and compute mean
            pesqs_tensor = torch.tensor(pesq_scores, dtype=torch.float32, 
                                       device=estimates.device)
            mean_pesq = pesqs_tensor.mean()
            
            if failed_samples > 0:
                logger.warning(f"PESQ computation failed for {failed_samples}/{estimates.batch_size} "
                             f"samples. Mean computed from {len(pesq_scores)} valid samples.")
            
            logger.debug(f"Computed PESQ scores: mean={mean_pesq:.3f}, "
                        f"std={pesqs_tensor.std():.3f}, "
                        f"valid_samples={len(pesq_scores)}")
            
            return mean_pesq
            
        except Exception as e:
            logger.error(f"Error in PESQ forward pass: {str(e)}", exc_info=True)
            raise


    
# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_synthetic_audio(db_reduction: float = -20, 
                           num_samples: int = 6,
                           duration_samples: int = 16000,
                           sample_rate: int = DEFAULT_SAMPLE_RATE) -> Tuple[AudioSignal, AudioSignal]:
    """
    Generate synthetic audio signals with specified loudness reduction for testing.
    
    This function creates pairs of audio signals where the second signal is a
    volume-reduced version of the first, useful for testing audio quality metrics.
    
    Args:
        db_reduction: Reduction in decibels (negative value means reduction).
                     e.g., -20 means reduce by 20dB.
        num_samples: Number of audio samples to generate in the batch.
        duration_samples: Duration of each audio sample in samples.
        sample_rate: Sample rate of the generated audio.
    
    Returns:
        Tuple containing:
            - Original audio signal batch
            - Volume-reduced audio signal batch
            
    Raises:
        ValueError: If invalid parameters are provided.
    """
    try:
        # Validate parameters
        if db_reduction > 0:
            logger.warning(f"Positive dB reduction ({db_reduction}) will increase volume")
        
        if num_samples <= 0:
            raise ValueError(f"Number of samples must be positive, got {num_samples}")
        
        logger.debug(f"Generating synthetic audio: {num_samples} samples, "
                    f"{duration_samples} samples duration, {db_reduction}dB reduction")
        
        # Generate random audio signals
        audio_signal_1_list = [
            AudioSignal(torch.randn(duration_samples), sample_rate) 
            for _ in range(num_samples)
        ]
        audio_signal_1 = AudioSignal.batch(audio_signal_1_list)

        # Convert dB reduction to linear amplitude scaling factor
        # amplitude_scale = 10^(dB/20)
        amplitude_scale = 10 ** (db_reduction / 20.0)
        logger.debug(f"Amplitude scaling factor: {amplitude_scale:.4f}")
        
        # Create volume-reduced versions
        audio_signal_2_list = [
            AudioSignal(signal.audio_data * amplitude_scale, signal.sample_rate) 
            for signal in audio_signal_1_list
        ]
        audio_signal_2 = AudioSignal.batch(audio_signal_2_list)
        
        return audio_signal_1, audio_signal_2
        
    except Exception as e:
        logger.error(f"Error generating synthetic audio: {str(e)}", exc_info=True)
        raise

# =============================================================================
# WATERMARKING METRICS
# =============================================================================

class BER(nn.Module):
    """
    Bit Error Rate (BER) metric for watermark detection accuracy.
    
    BER measures the fraction of incorrectly decoded bits in the watermark.
    Lower values indicate better detection accuracy, with 0 being perfect
    and 0.5 being random chance.
    """
    
    def __init__(self, threshold: float = DEFAULT_BER_THRESHOLD, eps: float = EPSILON) -> None:
        """
        Initialize BER metric calculator.
        
        Args:
            threshold: Decision threshold for converting probabilities to binary.
                      Values >= threshold are decoded as 1, otherwise 0.
            eps: Small value for numerical stability in division.
        """
        super().__init__()
        self.threshold = threshold
        self.eps = eps
        logger.debug(f"Initialized BER metric with threshold={threshold}, eps={eps}")

    def forward(self, 
                decoded_logits: torch.Tensor, 
                original_bits: torch.Tensor, 
                presence_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate bit error rate between decoded and original watermark bits.
        
        Args:
            decoded_logits: Model outputs before sigmoid activation. Shape: [B, W, T]
                          where B=batch, W=watermark_bits, T=time_steps.
            original_bits: Ground truth watermark bits (0 or 1). Shape: [B, W].
            presence_mask: Optional mask indicating valid time steps. Shape: [B, 1, T].
                         If provided, BER is computed only over valid regions.
        
        Returns:
            Bit error rate as a scalar tensor (0 to 1).
            
        Raises:
            ValueError: If tensor shapes are incompatible.
            RuntimeError: If BER computation fails.
        """
        try:
            # Validate input shapes
            B, W, T = decoded_logits.shape
            if original_bits.shape != (B, W):
                raise ValueError(f"Shape mismatch: decoded_logits has batch={B}, bits={W}, "
                               f"but original_bits has shape {original_bits.shape}")
            
            # Convert logits to probabilities using sigmoid
            probs = torch.sigmoid(decoded_logits)  # [B, W, T]
            
            if presence_mask is not None:
                # Validate presence mask shape
                if presence_mask.shape[0] != B or presence_mask.shape[2] != T:
                    raise ValueError(f"Presence mask shape {presence_mask.shape} incompatible "
                                   f"with batch size {B} and time steps {T}")
                
                # Expand mask to match watermark bits dimension
                mask = presence_mask.expand(-1, W, -1)  # [B, W, T]
                
                # Identify bits with at least one valid time step
                valid_bits = mask.sum(dim=2) > 0  # [B, W]
                
                # Average probabilities over valid time steps only
                mask_sum = mask.sum(dim=2) + self.eps  # Avoid division by zero
                avg_probs = (probs * mask).sum(dim=2) / mask_sum  # [B, W]
            else:
                # No mask provided - use all time steps
                avg_probs = probs.mean(dim=2)  # [B, W]
                valid_bits = torch.ones((B, W), dtype=torch.bool, device=decoded_logits.device)
            
            # Make binary decisions based on threshold
            decoded_bits = (avg_probs >= self.threshold).float()  # [B, W]
            
            # Calculate bit errors only for valid bits
            original_bits_float = original_bits.float()
            bit_errors = (decoded_bits != original_bits_float) * valid_bits
            
            # Compute BER
            total_errors = bit_errors.sum()
            total_valid_bits = valid_bits.sum()
            
            if total_valid_bits > 0:
                ber = total_errors / total_valid_bits
                logger.debug(f"BER computation: {total_errors:.0f} errors / "
                           f"{total_valid_bits:.0f} bits = {ber:.4f}")
            else:
                logger.warning("No valid bits found for BER computation, returning 0")
                ber = torch.tensor(0.0, device=decoded_logits.device)
            
            return ber
            
        except Exception as e:
            logger.error(f"Error in BER computation: {str(e)}", exc_info=True)
            raise RuntimeError("BER computation failed") from e

class Evaluate_BER(nn.Module):
    """
    Simplified BER calculator for direct probability comparison.
    
    This class provides a simpler BER calculation when you have probability
    values rather than logits, useful for evaluation scenarios.
    """
    
    def __init__(self, threshold: float = DEFAULT_BER_THRESHOLD) -> None:
        """
        Initialize simplified BER evaluator.
        
        Args:
            threshold: Decision threshold for binarizing probabilities.
        """
        super(Evaluate_BER, self).__init__()
        self.threshold = threshold
        logger.debug(f"Initialized Evaluate_BER with threshold={threshold}")

    def forward(self, 
                watermark_decoded_tensor: torch.Tensor, 
                watermark_tensor: torch.Tensor) -> torch.Tensor:
        """
        Calculate BER between decoded and original watermark probabilities.
        
        Args:
            watermark_decoded_tensor: Decoded watermark probabilities.
            watermark_tensor: Original watermark values.
            
        Returns:
            Bit error rate as a scalar tensor.
            
        Raises:
            ValueError: If tensor shapes don't match.
        """
        try:
            # Validate shapes
            if watermark_decoded_tensor.shape != watermark_tensor.shape:
                raise ValueError(f"Shape mismatch: decoded={watermark_decoded_tensor.shape}, "
                               f"original={watermark_tensor.shape}")
            
            # Binarize both tensors using threshold
            watermark_decoded_binary = (watermark_decoded_tensor >= self.threshold).float()
            watermark_binary = (watermark_tensor >= self.threshold).float()
            
            # Calculate accuracy and convert to error rate
            accuracy = (watermark_decoded_binary == watermark_binary).float().mean()
            ber = 1 - accuracy
            
            logger.debug(f"Evaluate_BER: accuracy={accuracy:.4f}, BER={ber:.4f}")
            
            return ber
            
        except Exception as e:
            logger.error(f"Error in Evaluate_BER: {str(e)}", exc_info=True)
            raise
    
class MIOU(nn.Module):
    """
    Mean Intersection over Union (mIOU) metric for watermark localization.
    
    This metric evaluates how well the predicted watermark regions match the
    ground truth regions. It computes IoU for both watermarked (foreground)
    and non-watermarked (background) regions, then averages them.
    """
    
    def __init__(self) -> None:
        """
        Initialize MIOU calculator.
        """
        super(MIOU, self).__init__()
        logger.debug("Initialized MIOU metric")

    def forward(self, 
                predicted_mask: Union[torch.Tensor, np.ndarray], 
                ground_truth_mask: Union[torch.Tensor, np.ndarray]) -> float:
        """
        Calculate mean Intersection over Union for binary masks.
        
        Args:
            predicted_mask: Predicted binary mask (0 or 1 values).
            ground_truth_mask: Ground truth binary mask (0 or 1 values).
            
        Returns:
            Mean IoU score averaged over background and foreground classes.
            Range is [0, 1] with 1 being perfect overlap.
            
        Raises:
            ValueError: If masks have different shapes or invalid values.
            RuntimeError: If IoU computation fails.
        """
        try:
            # Convert tensors to numpy arrays for computation
            if torch.is_tensor(predicted_mask):
                predicted_np = predicted_mask.detach().cpu().numpy()
            else:
                predicted_np = predicted_mask
                
            if torch.is_tensor(ground_truth_mask):
                ground_truth_np = ground_truth_mask.detach().cpu().numpy()
            else:
                ground_truth_np = ground_truth_mask
            
            # Validate shapes
            if predicted_np.shape != ground_truth_np.shape:
                raise ValueError(f"Shape mismatch: predicted={predicted_np.shape}, "
                               f"ground_truth={ground_truth_np.shape}")
            
            # Validate binary masks
            unique_pred = np.unique(predicted_np)
            unique_gt = np.unique(ground_truth_np)
            
            if not np.all(np.isin(unique_pred, [0, 1])):
                raise ValueError(f"Predicted mask contains non-binary values: {unique_pred}")
            if not np.all(np.isin(unique_gt, [0, 1])):
                raise ValueError(f"Ground truth mask contains non-binary values: {unique_gt}")
            
            # Calculate IoU for foreground (watermarked regions, class 1)
            intersection_fg = np.logical_and(predicted_np == 1, ground_truth_np == 1)
            union_fg = np.logical_or(predicted_np == 1, ground_truth_np == 1)
            
            # Handle edge case where there's no foreground
            if np.sum(union_fg) == 0:
                iou_fg = 1.0 if np.sum(intersection_fg) == 0 else 0.0
            else:
                iou_fg = np.sum(intersection_fg) / np.sum(union_fg)
            
            # Calculate IoU for background (non-watermarked regions, class 0)
            intersection_bg = np.logical_and(predicted_np == 0, ground_truth_np == 0)
            union_bg = np.logical_or(predicted_np == 0, ground_truth_np == 0)
            
            # Handle edge case where there's no background
            if np.sum(union_bg) == 0:
                iou_bg = 1.0 if np.sum(intersection_bg) == 0 else 0.0
            else:
                iou_bg = np.sum(intersection_bg) / np.sum(union_bg)
            
            # Calculate mean IoU
            miou = (iou_fg + iou_bg) / 2
            
            logger.debug(f"MIOU computation: foreground_IoU={iou_fg:.4f}, "
                        f"background_IoU={iou_bg:.4f}, mean={miou:.4f}")
            
            return miou
            
        except Exception as e:
            logger.error(f"Error in MIOU computation: {str(e)}", exc_info=True)
            raise RuntimeError("MIOU computation failed") from e
    

# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_ber(output_dir: str = ".", 
             batch_size: int = 1,
             watermark_bits: int = 16,
             time_steps: int = 16000,
             noise_scale: float = 0.5) -> None:
    """
    Test BER metric implementation with visualization.
    
    This function tests the BER calculator with synthetic data and generates
    a visualization of the decoded probabilities.
    
    Args:
        output_dir: Directory to save visualization outputs.
        batch_size: Number of samples in test batch.
        watermark_bits: Number of watermark bits.
        time_steps: Number of time steps in audio signal.
        noise_scale: Standard deviation of noise added to logits.
        
    Raises:
        IOError: If unable to save visualization.
        RuntimeError: If BER computation fails.
    """
    try:
        logger.info("Starting BER metric test")
        
        # Initialize BER calculator
        ber = BER()
        
        # Generate test data
        logger.debug(f"Generating test data: B={batch_size}, W={watermark_bits}, T={time_steps}")
        original_bits = torch.randint(0, 2, (batch_size, watermark_bits))  # Binary ground truth
        
        # Create realistic logits with controlled noise
        noise = torch.randn(batch_size, watermark_bits, time_steps) * noise_scale
        perfect_logits = torch.where(
            original_bits.unsqueeze(-1) == 1,
            2.0 * torch.ones(batch_size, watermark_bits, time_steps),  # Positive logits for 1s
            -2.0 * torch.ones(batch_size, watermark_bits, time_steps)  # Negative logits for 0s
        ) + noise
        
        # Create partial presence mask (first half of signal is valid)
        partial_presence = torch.zeros(batch_size, 1, time_steps)
        partial_presence[:, :, :time_steps//2] = 1
        
        # Configure matplotlib styling
        plt.style.use('default')
        plt.rcParams.update(PLOT_STYLE)
        
        # Create visualization
        logger.debug("Creating probability visualization")
        fig = plt.figure(figsize=(12, 8), facecolor=COLORS['background'])
        ax = fig.add_subplot(111)
        
        # Convert logits to probabilities and visualize
        probs = torch.sigmoid(perfect_logits)
        
        # Show first 100 time steps for clarity
        display_steps = min(100, time_steps)
        im = ax.imshow(
            probs[0, :, :display_steps].detach().cpu().numpy(),
            aspect='auto',
            cmap='RdYlBu_r',  # Blue=high probability, Red=low probability
            vmin=0,
            vmax=1,
            interpolation='nearest'
        )
        
        # Add title and labels
        ax.set_title(f'Decoded Probabilities (First {display_steps} Time Steps)', 
                    fontsize=22, pad=20, color=COLORS['text'])
        ax.set_xlabel('Time Step', fontsize=20, color=COLORS['text'])
        ax.set_ylabel('Bit Index', fontsize=20, color=COLORS['text'])
        ax.tick_params(axis='both', labelsize=18, colors=COLORS['text'])
        ax.set_facecolor(COLORS['background'])
        
        # Add colorbar with threshold indicator
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Probability', fontsize=20, color=COLORS['text'])
        cbar.ax.tick_params(labelsize=18, colors=COLORS['text'])
        
        # Mark decision threshold on colorbar
        cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['0.0', '0.25', '0.5\n(threshold)', '0.75', '1.0'])
        
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(output_dir, 'decoded_probabilities.pdf')
        plt.savefig(output_path, bbox_inches='tight', transparent=True, dpi=300)
        plt.close()
        logger.info(f"Saved probability visualization to {output_path}")
        
        # Run BER tests
        logger.debug("Computing BER scores")
        full_presence = torch.ones(batch_size, 1, time_steps)
        
        # Test with full presence mask
        ber_full = ber(perfect_logits, original_bits, full_presence)
        
        # Test with partial presence mask
        ber_partial = ber(perfect_logits, original_bits, partial_presence)
        
        # Log results
        logger.info("BER Test Results:")
        logger.info(f"  BER with full presence:    {ber_full:.4f}")
        logger.info(f"  BER with partial presence: {ber_partial:.4f}")
        
        # Validate results
        if ber_full > 0.1:  # Should be near 0 for perfect decoding with low noise
            logger.warning(f"Unexpectedly high BER with full presence: {ber_full:.4f}")
        
        logger.info("BER test completed successfully")
        
    except Exception as e:
        logger.error(f"BER test failed: {str(e)}", exc_info=True)
        raise

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """
    Main entry point for evaluation module testing.
    
    Sets up logging and runs test suite for evaluation metrics.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger.info("Starting evaluation metrics test suite")
    
    try:
        # Run BER test with visualization
        test_ber()
        
        # Additional tests can be added here
        logger.info("All tests completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Test suite interrupted by user")
    except Exception as e:
        logger.error(f"Test suite failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()