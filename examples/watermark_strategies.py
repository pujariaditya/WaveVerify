#!/usr/bin/env python3
"""
Watermark strategies example for WaveVerify.

This module demonstrates different approaches to creating watermark IDs
for various use cases including content creators, timestamps, licensing,
tracking, and custom implementations.
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple, Union

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Local imports
from waveverify import WaveVerify, WatermarkID

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('watermark_strategies.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# Supported license types
SUPPORTED_LICENSES = {
    "CC-BY": "Creative Commons Attribution",
    "CC-BY-SA": "Creative Commons Attribution-ShareAlike",
    "CC-BY-NC": "Creative Commons Attribution-NonCommercial",
    "CC-BY-NC-SA": "Creative Commons Attribution-NonCommercial-ShareAlike",
    "CC-BY-ND": "Creative Commons Attribution-NoDerivatives",
    "CC-BY-NC-ND": "Creative Commons Attribution-NonCommercial-NoDerivatives",
    "CC0": "Creative Commons Zero - Public Domain",
    "ALL-RIGHTS": "All Rights Reserved",
    "MIT": "MIT License",
    "GPL": "GNU General Public License",
    "APACHE": "Apache License"
}

# Output formatting
SECTION_SEPARATOR = "=" * 60
SUBSECTION_SEPARATOR = "-" * 40

# Default values
DEFAULT_AUDIO_FILE = "sample_audio.wav"
DEFAULT_OUTPUT_FILE = "watermarked_output.wav"

# =============================================================================
# WATERMARK DEMONSTRATION FUNCTIONS
# =============================================================================

def demonstrate_creator_watermark() -> None:
    """
    Demonstrate how content creators can watermark their work.
    
    This function shows various use cases for creator-based watermarking
    including musicians, podcast creators, and news organizations. Each
    watermark is generated using a creator-specific identifier.
    
    Args:
        None
        
    Returns:
        None
        
    Raises:
        Exception: If watermark creation fails for any reason
    """
    logger.info("Starting creator watermark demonstration")
    print(f"\n{SUBSECTION_SEPARATOR}")
    print("CREATOR WATERMARKING")
    print(SUBSECTION_SEPARATOR)
    
    try:
        # Musicians/Artists watermarking
        artist_id = "taylor_swift_2024"
        artist_watermark = WatermarkID.for_creator(artist_id)
        logger.info(f"Created artist watermark for: {artist_id}")
        
        print(f"Artist watermark: {artist_watermark}")
        print(f"  Binary representation: {artist_watermark.to_bits()}")
        print(f"  Hexadecimal: {artist_watermark.to_hex()}")
        print()
        
        # Podcast creators watermarking
        podcast_id = "mypodcast_episode_42"
        podcast_watermark = WatermarkID.for_creator(podcast_id)
        logger.info(f"Created podcast watermark for: {podcast_id}")
        
        print(f"Podcast watermark: {podcast_watermark}")
        print(f"  Use case: Tracking podcast episode distribution")
        print()
        
        # News organizations watermarking
        news_id = "CNN_interview_20240717"
        news_watermark = WatermarkID.for_creator(news_id)
        logger.info(f"Created news watermark for: {news_id}")
        
        print(f"News watermark: {news_watermark}")
        print(f"  Use case: Protecting exclusive interviews and content")
        
    except Exception as e:
        error_msg = f"Error creating creator watermark: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"ERROR: {error_msg}")


def demonstrate_timestamp_watermark() -> None:
    """
    Demonstrate how to use timestamps for tracking when content was created.
    
    This function illustrates timestamp-based watermarking for both current
    time and specific date/time values. Useful for temporal tracking and
    content versioning.
    
    Args:
        None
        
    Returns:
        None
        
    Raises:
        ValueError: If an invalid timestamp is provided
        Exception: If watermark creation fails
    """
    logger.info("Starting timestamp watermark demonstration")
    print(f"\n{SUBSECTION_SEPARATOR}")
    print("TIMESTAMP WATERMARKING")
    print(SUBSECTION_SEPARATOR)
    
    try:
        # Current time watermarking
        current_watermark = WatermarkID.for_timestamp()
        current_time = datetime.now()
        logger.info(f"Created current timestamp watermark at {current_time}")
        
        print(f"Current time watermark: {current_watermark}")
        print(f"  Generated at: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Specific date/time watermarking
        specific_time = datetime(2024, 7, 17, 14, 30, 0)
        
        # Validate the datetime object
        if specific_time > datetime.now():
            logger.warning(f"Future timestamp specified: {specific_time}")
            
        dated_watermark = WatermarkID.for_timestamp(specific_time)
        logger.info(f"Created specific timestamp watermark for {specific_time}")
        
        print(f"Specific time watermark: {dated_watermark}")
        print(f"  Timestamp details:")
        print(f"    - Year: {specific_time.year}")
        print(f"    - Month: {specific_time.month}")
        print(f"    - Day: {specific_time.day}")
        print(f"    - Quarter of day: {specific_time.hour // 6}")
        print(f"  Use case: Version control and release tracking")
        
    except ValueError as e:
        error_msg = f"Invalid timestamp provided: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"ERROR: {error_msg}")
    except Exception as e:
        error_msg = f"Error creating timestamp watermark: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"ERROR: {error_msg}")


def demonstrate_license_watermark() -> None:
    """
    Demonstrate how to embed license information in watermarks.
    
    This function shows how to create watermarks that encode licensing
    information, supporting standard licenses like Creative Commons
    as well as custom license types.
    
    Args:
        None
        
    Returns:
        None
        
    Raises:
        Exception: If watermark creation fails
    """
    logger.info("Starting license watermark demonstration")
    print(f"\n{SUBSECTION_SEPARATOR}")
    print("LICENSE WATERMARKING")
    print(SUBSECTION_SEPARATOR)
    
    try:
        # Creative Commons licenses
        cc_licenses = ["CC-BY", "CC-BY-SA", "CC-BY-NC"]
        
        print("Standard Creative Commons Licenses:")
        for license_type in cc_licenses:
            license_watermark = WatermarkID.for_license(license_type)
            logger.info(f"Created license watermark for: {license_type}")
            
            license_desc = SUPPORTED_LICENSES.get(license_type, "Unknown license")
            print(f"  {license_type}: {license_watermark}")
            print(f"    Description: {license_desc}")
        
        print()
        
        # All rights reserved
        all_rights_watermark = WatermarkID.for_license("ALL-RIGHTS")
        logger.info("Created all rights reserved watermark")
        
        print(f"All Rights Reserved: {all_rights_watermark}")
        print(f"  Use case: Commercial content protection")
        print()
        
        # Custom license example
        custom_license = "MyCompany-Internal-Use-Only"
        custom_watermark = WatermarkID.for_license(custom_license)
        logger.info(f"Created custom license watermark: {custom_license}")
        
        print(f"Custom License: {custom_watermark}")
        print(f"  License ID: {custom_license}")
        print(f"  Use case: Corporate internal distribution")
        
    except Exception as e:
        error_msg = f"Error creating license watermark: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"ERROR: {error_msg}")


def demonstrate_tracking_watermark() -> None:
    """
    Demonstrate how to use watermarks for tracking distribution.
    
    This function illustrates tracking watermarks for various use cases
    including order tracking, legal document tracking, and media episode
    tracking.
    
    Args:
        None
        
    Returns:
        None
        
    Raises:
        Exception: If watermark creation fails
    """
    logger.info("Starting tracking watermark demonstration")
    print(f"\n{SUBSECTION_SEPARATOR}")
    print("TRACKING WATERMARKING")
    print(SUBSECTION_SEPARATOR)
    
    try:
        # Simple numeric tracking
        order_id = "12345"
        order_watermark = WatermarkID.for_tracking(order_id)
        logger.info(f"Created order tracking watermark: {order_id}")
        
        print(f"Order tracking: {order_watermark}")
        print(f"  Order ID: {order_id}")
        print(f"  Use case: E-commerce audio product distribution")
        print()
        
        # Complex legal tracking IDs
        case_id = "LEGAL-2024-CASE-001-DEPO-15"
        case_watermark = WatermarkID.for_tracking(case_id)
        logger.info(f"Created legal case tracking watermark: {case_id}")
        
        print(f"Legal case tracking: {case_watermark}")
        print(f"  Case ID: {case_id}")
        print(f"  Use case: Legal audio evidence tracking")
        print()
        
        # Media episode tracking
        episode_id = "S03E15"
        episode_watermark = WatermarkID.for_tracking(episode_id)
        logger.info(f"Created episode tracking watermark: {episode_id}")
        
        print(f"Episode tracking: {episode_watermark}")
        print(f"  Episode: {episode_id}")
        print(f"  Use case: TV/Podcast series distribution tracking")
        
    except Exception as e:
        error_msg = f"Error creating tracking watermark: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"ERROR: {error_msg}")


def demonstrate_custom_watermark() -> None:
    """
    Demonstrate how to create custom watermarks from various data types.
    
    This function shows how to create watermarks from binary strings,
    integers, and byte sequences for specialized use cases that don't
    fit standard categories.
    
    Args:
        None
        
    Returns:
        None
        
    Raises:
        ValueError: If invalid input is provided for custom watermark
        Exception: If watermark creation fails
    """
    logger.info("Starting custom watermark demonstration")
    print(f"\n{SUBSECTION_SEPARATOR}")
    print("CUSTOM WATERMARKING")
    print(SUBSECTION_SEPARATOR)
    
    try:
        # Binary string watermark
        binary_data = "1111000011110000"
        
        # Validate binary string
        if not all(bit in '01' for bit in binary_data):
            raise ValueError(f"Invalid binary string: {binary_data}")
            
        binary_watermark = WatermarkID.custom(binary_data)
        logger.info(f"Created binary custom watermark: {binary_data}")
        
        print(f"Binary custom watermark: {binary_watermark}")
        print(f"  Input: {binary_data}")
        print(f"  Use case: Direct binary pattern encoding")
        print()
        
        # Integer watermark
        int_value = 42
        
        # Validate integer range (example: 16-bit range)
        if not 0 <= int_value <= 65535:
            logger.warning(f"Integer value {int_value} may be out of expected range")
            
        int_watermark = WatermarkID.custom(int_value)
        logger.info(f"Created integer custom watermark: {int_value}")
        
        print(f"Integer custom watermark: {int_watermark}")
        print(f"  Integer value: {int_value}")
        print(f"  Hexadecimal: {int_watermark.to_hex()}")
        print(f"  Use case: Numeric ID encoding")
        print()
        
        # Bytes watermark
        bytes_data = b'\xAB\xCD'
        bytes_watermark = WatermarkID.custom(bytes_data)
        logger.info(f"Created bytes custom watermark: {bytes_data.hex()}")
        
        print(f"Bytes custom watermark: {bytes_watermark}")
        print(f"  Byte sequence: {bytes_data.hex().upper()}")
        print(f"  Hexadecimal: {bytes_watermark.to_hex()}")
        print(f"  Use case: Raw data pattern encoding")
        
    except ValueError as e:
        error_msg = f"Invalid input for custom watermark: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"ERROR: {error_msg}")
    except Exception as e:
        error_msg = f"Error creating custom watermark: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"ERROR: {error_msg}")


def practical_example() -> None:
    """
    Complete practical example demonstrating real-world watermarking scenario.
    
    This function shows a comprehensive example of watermarking a music
    release with different strategies for creator identification, release
    tracking, and distribution platform differentiation.
    
    Args:
        None
        
    Returns:
        None
        
    Raises:
        FileNotFoundError: If the audio file doesn't exist
        Exception: If watermark embedding or detection fails
    """
    logger.info("Starting practical example demonstration")
    print(f"\n{SUBSECTION_SEPARATOR}")
    print("PRACTICAL EXAMPLE: MUSIC RELEASE")
    print(SUBSECTION_SEPARATOR)
    
    try:
        # Artist and release information
        artist_name = "BeyonceKnowles"
        release_date = datetime(2024, 8, 1)
        
        logger.info(f"Demonstrating music release watermarking for {artist_name}")
        
        print(f"Artist: {artist_name}")
        print(f"Release Date: {release_date.strftime('%Y-%m-%d')}")
        print()
        
        # Option 1: Creator-based watermark
        creator_id = f"{artist_name}_SingleRelease"
        creator_watermark = WatermarkID.for_creator(creator_id)
        logger.info(f"Created creator watermark: {creator_id}")
        
        print("Option 1 - Creator-based watermark:")
        print(f"  Watermark: {creator_watermark}")
        print(f"  Purpose: Artist identification")
        print()
        
        # Option 2: Timestamp-based watermark
        timestamp_watermark = WatermarkID.for_timestamp(release_date)
        logger.info(f"Created timestamp watermark for release date")
        
        print("Option 2 - Timestamp-based watermark:")
        print(f"  Watermark: {timestamp_watermark}")
        print(f"  Purpose: Release date tracking")
        print()
        
        # Option 3: Platform-specific tracking
        platforms = {
            "SPOTIFY_EXCLUSIVE_2024": "Spotify",
            "APPLE_MUSIC_2024": "Apple Music",
            "AMAZON_MUSIC_2024": "Amazon Music"
        }
        
        print("Option 3 - Platform-specific tracking:")
        for platform_id, platform_name in platforms.items():
            platform_watermark = WatermarkID.for_tracking(platform_id)
            logger.info(f"Created platform watermark for {platform_name}")
            print(f"  {platform_name}: {platform_watermark}")
        
        print()
        
        # Actual embedding demonstration (if audio file exists)
        audio_file = Path("unreleased_single.wav")
        output_file = Path("single_watermarked.wav")
        
        if audio_file.exists():
            logger.info(f"Audio file found: {audio_file}")
            print(f"Audio file found: {audio_file}")
            print("Performing actual watermark embedding...")
            
            try:
                # Initialize WaveVerify
                wv = WaveVerify()
                logger.info("WaveVerify initialized successfully")
                
                # Embed creator watermark
                wv.embed(str(audio_file), creator_watermark, str(output_file))
                logger.info(f"Watermark embedded successfully to {output_file}")
                print(f"✓ Watermark embedded successfully to: {output_file}")
                
                # Verify the embedded watermark
                detected_watermark, confidence = wv.detect(str(output_file))
                logger.info(f"Watermark detected: {detected_watermark}, confidence: {confidence:.2%}")
                
                print(f"✓ Verification successful:")
                print(f"  Detected: {detected_watermark}")
                print(f"  Confidence: {confidence:.2%}")
                
                # Validate detection
                if detected_watermark == creator_watermark:
                    print("✓ Watermark matches original!")
                else:
                    logger.warning("Detected watermark doesn't match original")
                    print("⚠ Warning: Detected watermark doesn't match original")
                    
            except Exception as e:
                error_msg = f"Error during embedding/detection: {str(e)}"
                logger.error(error_msg, exc_info=True)
                print(f"ERROR: {error_msg}")
        else:
            logger.info(f"Audio file not found: {audio_file}")
            print(f"Note: Audio file '{audio_file}' not found.")
            print("      In production, replace with actual audio file path.")
            
    except Exception as e:
        error_msg = f"Error in practical example: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"ERROR: {error_msg}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_demonstrations() -> None:
    """
    Run all watermark strategy demonstrations.
    
    Args:
        None
        
    Returns:
        None
        
    Raises:
        Exception: If any demonstration fails critically
    """
    try:
        demonstrate_creator_watermark()
        demonstrate_timestamp_watermark()
        demonstrate_license_watermark()
        demonstrate_tracking_watermark()
        demonstrate_custom_watermark()
        practical_example()
        
    except Exception as e:
        logger.error(f"Critical error in demonstrations: {str(e)}", exc_info=True)
        raise


def print_summary() -> None:
    """
    Print summary of watermark strategies and best practices.
    
    Args:
        None
        
    Returns:
        None
    """
    print(f"\n{SECTION_SEPARATOR}")
    print("KEY INSIGHTS AND BEST PRACTICES")
    print(SECTION_SEPARATOR)
    print()
    print("1. Creator Watermarks (for_creator)")
    print("   - Use for artist/creator identification")
    print("   - Ideal for content attribution")
    print()
    print("2. Timestamp Watermarks (for_timestamp)")
    print("   - Use for temporal tracking and versioning")
    print("   - Helpful for release management")
    print()
    print("3. License Watermarks (for_license)")
    print("   - Use for rights management")
    print("   - Supports standard and custom licenses")
    print()
    print("4. Tracking Watermarks (for_tracking)")
    print("   - Use for distribution tracking")
    print("   - Platform-specific identification")
    print()
    print("5. Custom Watermarks (custom)")
    print("   - Use only when other methods don't fit")
    print("   - Supports binary, integer, and byte inputs")
    print()
    print("Remember: Choose the appropriate watermark type based on your")
    print("specific use case for optimal results and maintainability.")


def main() -> None:
    """
    Main entry point for the watermark strategies demonstration.
    
    Args:
        None
        
    Returns:
        None
        
    Raises:
        SystemExit: On critical errors
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="WaveVerify Watermark Strategies Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run all demonstrations
  %(prog)s --demo creator     # Run only creator watermark demo
  %(prog)s --demo timestamp   # Run only timestamp watermark demo
  %(prog)s --list            # List available demonstrations
        """
    )
    
    parser.add_argument(
        "--demo",
        type=str,
        choices=["creator", "timestamp", "license", "tracking", "custom", "practical"],
        help="Run specific demonstration only"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available demonstrations"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Update logging level if specified
    if args.log_level:
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        logger.setLevel(getattr(logging, args.log_level))
    
    try:
        logger.info("Starting WaveVerify Watermark Strategies Demo")
        
        print(SECTION_SEPARATOR)
        print("WAVEVERIFY WATERMARK STRATEGIES DEMO")
        print(SECTION_SEPARATOR)
        
        if args.list:
            print("\nAvailable demonstrations:")
            print("  - creator:    Creator watermarking strategies")
            print("  - timestamp:  Timestamp-based watermarking")
            print("  - license:    License information embedding")
            print("  - tracking:   Distribution tracking watermarks")
            print("  - custom:     Custom watermark creation")
            print("  - practical:  Real-world example")
            return
        
        # Run specific demo or all demos
        if args.demo:
            logger.info(f"Running specific demo: {args.demo}")
            demo_map = {
                "creator": demonstrate_creator_watermark,
                "timestamp": demonstrate_timestamp_watermark,
                "license": demonstrate_license_watermark,
                "tracking": demonstrate_tracking_watermark,
                "custom": demonstrate_custom_watermark,
                "practical": practical_example
            }
            demo_map[args.demo]()
        else:
            logger.info("Running all demonstrations")
            run_all_demonstrations()
        
        # Print summary
        print_summary()
        
        logger.info("WaveVerify Watermark Strategies Demo completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
        print("\n\nDemo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        print(f"\nFATAL ERROR: {str(e)}")
        print("Check watermark_strategies.log for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()