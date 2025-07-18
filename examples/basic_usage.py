#!/usr/bin/env python3
"""
Basic usage example for WaveVerify package.

This example demonstrates:
1. Loading a pretrained model
2. Embedding a watermark
3. Detecting the watermark
4. Verifying the watermark matches
"""

# =============================================================================
# IMPORTS
# =============================================================================
# Standard library imports
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple

# Third-party imports
# (None in this module)

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from waveverify import WaveVerify, WatermarkID

# =============================================================================
# CONSTANTS
# =============================================================================
# File paths
DEFAULT_INPUT_AUDIO: str = "example_audio.wav"
DEFAULT_OUTPUT_AUDIO: str = "watermarked_example.wav"

# Confidence thresholds
WATERMARK_DETECTION_THRESHOLD: float = 0.5  # Minimum confidence for positive detection
WATERMARK_MASK_THRESHOLD: float = 0.5      # Threshold for binary mask in location detection

# Demo constants
DEMO_CREATOR_ID: str = "demo_user_2024"

# Exit codes
EXIT_SUCCESS: int = 0
EXIT_FILE_NOT_FOUND: int = 1
EXIT_PROCESSING_ERROR: int = 2

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
# Configure logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main() -> int:
    """
    Execute the WaveVerify basic usage demonstration.
    
    This function demonstrates the complete workflow of watermarking audio files,
    including embedding, detection, verification, and location identification.
    
    Args:
        None
    
    Returns:
        int: Exit code (0 for success, non-zero for errors)
            - 0: Successful execution
            - 1: Input file not found
            - 2: Processing error occurred
    
    Raises:
        FileNotFoundError: If the input audio file doesn't exist
        ValueError: If watermark operations fail
        Exception: For any unexpected errors during processing
    """
    # Track execution time for performance monitoring
    start_time: float = time.time()
    
    # Initialize file paths
    input_audio_path: str = DEFAULT_INPUT_AUDIO
    watermarked_audio_path: str = DEFAULT_OUTPUT_AUDIO
    
    try:
        # =============================================================================
        # INPUT VALIDATION
        # =============================================================================
        # Check if input file exists before processing
        input_path_obj: Path = Path(input_audio_path)
        if not input_path_obj.exists():
            logger.error(f"Input file not found: {input_audio_path}")
            print(f"Error: Input file '{input_audio_path}' not found.")
            print("Please provide an audio file for testing.")
            return EXIT_FILE_NOT_FOUND
        
        logger.info(f"Processing audio file: {input_audio_path}")
        
        # =============================================================================
        # WAVEVERIFY INITIALIZATION
        # =============================================================================
        print("Initializing WaveVerify...")
        try:
            wave_verifier: WaveVerify = WaveVerify()
            logger.info("WaveVerify initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize WaveVerify: {str(e)}", exc_info=True)
            print(f"Error: Failed to initialize WaveVerify - {str(e)}")
            return EXIT_PROCESSING_ERROR
        
        # =============================================================================
        # WATERMARK EMBEDDING
        # =============================================================================
        print("\n1. Embedding watermark...")
        try:
            # Create a watermark ID for a content creator
            watermark: WatermarkID = WatermarkID.for_creator(DEMO_CREATOR_ID)
            logger.info(f"Created watermark ID for creator: {DEMO_CREATOR_ID}")
            
            # Embed the watermark into the audio file
            audio_data: Any
            sample_rate: int
            embedded_watermark: WatermarkID
            audio_data, sample_rate, embedded_watermark = wave_verifier.embed(
                input_audio_path, 
                watermark, 
                output_path=watermarked_audio_path
            )
            
            print(f"   ✓ Watermark embedded")
            print(f"   Watermark: {embedded_watermark}")
            print(f"   Saved to: {watermarked_audio_path}")
            logger.info(f"Watermark embedded successfully: {embedded_watermark}")
            
        except Exception as e:
            logger.error(f"Failed to embed watermark: {str(e)}", exc_info=True)
            print(f"   ✗ Error embedding watermark: {str(e)}")
            return EXIT_PROCESSING_ERROR
        
        # =============================================================================
        # WATERMARK DETECTION
        # =============================================================================
        print("\n2. Detecting watermark...")
        try:
            # Detect watermark from the watermarked audio
            detected_watermark: Optional[WatermarkID]
            confidence: float
            detected_watermark, confidence = wave_verifier.detect(watermarked_audio_path)
            
            print(f"   ✓ Watermark detected")
            print(f"   Watermark: {detected_watermark}")
            print(f"   Confidence: {confidence:.2%}")
            logger.info(f"Watermark detected with confidence: {confidence:.2%}")
            
        except Exception as e:
            logger.error(f"Failed to detect watermark: {str(e)}", exc_info=True)
            print(f"   ✗ Error detecting watermark: {str(e)}")
            return EXIT_PROCESSING_ERROR
        
        # =============================================================================
        # WATERMARK VERIFICATION
        # =============================================================================
        print("\n3. Verifying watermark...")
        try:
            # Verify that the detected watermark matches the original
            is_verified: bool = wave_verifier.verify(watermarked_audio_path, watermark)
            
            if is_verified:
                print("   ✓ Watermark verification successful!")
                logger.info("Watermark verification successful")
            else:
                print("   ✗ Watermark verification failed!")
                logger.warning("Watermark verification failed")
                
        except Exception as e:
            logger.error(f"Failed to verify watermark: {str(e)}", exc_info=True)
            print(f"   ✗ Error verifying watermark: {str(e)}")
            return EXIT_PROCESSING_ERROR
        
        # =============================================================================
        # WATERMARK LOCATION
        # =============================================================================
        print("\n4. Locating watermark regions...")
        try:
            # Get a mask indicating where the watermark is present in the audio
            watermark_mask: Any = wave_verifier.locate(watermarked_audio_path)
            
            # Calculate percentage of audio that contains watermark
            # Using threshold to create binary mask for percentage calculation
            watermarked_percentage: float = (watermark_mask > WATERMARK_MASK_THRESHOLD).mean() * 100
            
            print(f"   ✓ Watermark covers {watermarked_percentage:.1f}% of the audio")
            logger.info(f"Watermark coverage: {watermarked_percentage:.1f}%")
            
        except Exception as e:
            logger.error(f"Failed to locate watermark: {str(e)}", exc_info=True)
            print(f"   ✗ Error locating watermark: {str(e)}")
            return EXIT_PROCESSING_ERROR
        
        # =============================================================================
        # FALSE POSITIVE TEST
        # =============================================================================
        print("\n5. Testing on original (non-watermarked) audio...")
        try:
            # Test detection on original audio to check for false positives
            detected_watermark_original: Optional[WatermarkID]
            confidence_original: float
            detected_watermark_original, confidence_original = wave_verifier.detect(input_audio_path)
            
            print(f"   Confidence on original: {confidence_original:.2%}")
            
            # Check if confidence is below threshold (expected for non-watermarked audio)
            if confidence_original < WATERMARK_DETECTION_THRESHOLD:
                print("   ✓ Correctly identified as non-watermarked")
                logger.info("Original audio correctly identified as non-watermarked")
            else:
                print("   ✗ False positive detection")
                logger.warning(f"False positive detection with confidence: {confidence_original:.2%}")
                
        except Exception as e:
            logger.error(f"Failed to test original audio: {str(e)}", exc_info=True)
            print(f"   ✗ Error testing original audio: {str(e)}")
            return EXIT_PROCESSING_ERROR
        
        # Calculate and log total execution time
        execution_time: float = time.time() - start_time
        logger.info(f"Total execution time: {execution_time:.2f} seconds")
        print(f"\n✓ All tests completed successfully in {execution_time:.2f} seconds")
        
        return EXIT_SUCCESS
        
    except Exception as e:
        # Catch-all for any unexpected errors
        logger.error(f"Unexpected error in main execution: {str(e)}", exc_info=True)
        print(f"\n✗ Unexpected error: {str(e)}")
        return EXIT_PROCESSING_ERROR


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # Execute main function and exit with appropriate code
    exit_code: int = main()
    sys.exit(exit_code)