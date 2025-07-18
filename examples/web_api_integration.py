#!/usr/bin/env python3
"""
Web API integration example for WaveVerify.

This example shows how to integrate WaveVerify into a Flask web application
for audio authentication services with enterprise-grade error handling and logging.
"""

# =============================================================================
# Standard Library Imports
# =============================================================================
import os
import sys
import logging
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, Set

# =============================================================================
# Third-Party Imports
# =============================================================================
from flask import Flask, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# =============================================================================
# Local Imports
# =============================================================================
sys.path.insert(0, str(Path(__file__).parent.parent))
from waveverify import WaveVerify, WatermarkID

# =============================================================================
# Logging Configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('waveverify_api.log')
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# Application Configuration
# =============================================================================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Initialize WaveVerify (shared across requests for performance)
wv = WaveVerify()
logger.info("WaveVerify engine initialized successfully")

# =============================================================================
# Constants
# =============================================================================
ALLOWED_EXTENSIONS: Set[str] = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
CONFIDENCE_THRESHOLD_AUTHENTIC: float = 0.8
CONFIDENCE_THRESHOLD_SUSPICIOUS: float = 0.5
DEFAULT_WATERMARK_TYPE: str = 'timestamp'

# =============================================================================
# Custom Exceptions
# =============================================================================
class WaveVerifyAPIError(Exception):
    """Base exception for WaveVerify API errors."""
    pass

class InvalidFileError(WaveVerifyAPIError):
    """Raised when an invalid file is provided."""
    pass

class WatermarkError(WaveVerifyAPIError):
    """Raised when watermark operations fail."""
    pass

# =============================================================================
# Helper Functions
# =============================================================================
def generate_request_id() -> str:
    """
    Generate a unique request ID for tracking.
    
    Returns:
        str: UUID string for request tracking
    """
    return str(uuid.uuid4())


def allowed_file(filename: str) -> bool:
    """
    Check if file has allowed audio extension.
    
    Args:
        filename: Name of the file to check
        
    Returns:
        bool: True if file extension is allowed, False otherwise
    """
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def validate_audio_upload(request_id: str) -> Tuple[Any, str]:
    """
    Validate audio file upload from request.
    
    Args:
        request_id: Unique request ID for logging
        
    Returns:
        Tuple[FileStorage, str]: Uploaded file object and secure filename
        
    Raises:
        InvalidFileError: If no file provided or invalid file type
    """
    # Check if audio file was uploaded
    if 'audio' not in request.files:
        logger.warning(f"Request {request_id}: No audio file provided")
        raise InvalidFileError('No audio file provided')
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        logger.warning(f"Request {request_id}: Empty filename")
        raise InvalidFileError('No file selected')
    
    # Validate file extension
    if not allowed_file(audio_file.filename):
        logger.warning(f"Request {request_id}: Invalid file type: {audio_file.filename}")
        raise InvalidFileError(f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}')
    
    # Secure the filename to prevent directory traversal attacks
    safe_filename = secure_filename(audio_file.filename)
    logger.info(f"Request {request_id}: Valid audio file received: {safe_filename}")
    
    return audio_file, safe_filename


def create_watermark(watermark_type: str, watermark_value: str, request_id: str) -> WatermarkID:
    """
    Create WatermarkID object based on type and value.
    
    Args:
        watermark_type: Type of watermark ('creator', 'timestamp', 'license', 'tracking', 'custom')
        watermark_value: Value for the watermark (type-dependent)
        request_id: Request ID for logging
        
    Returns:
        WatermarkID: Created watermark object
        
    Raises:
        WatermarkError: If watermark creation fails
    """
    try:
        if watermark_type == 'creator':
            if not watermark_value:
                raise WatermarkError('Creator ID required for creator watermark')
            return WatermarkID.for_creator(watermark_value)
            
        elif watermark_type == 'timestamp':
            # Default to current time if no value provided
            logger.info(f"Request {request_id}: Creating timestamp watermark")
            return WatermarkID.for_timestamp()
            
        elif watermark_type == 'license':
            if not watermark_value:
                raise WatermarkError('License type required for license watermark')
            return WatermarkID.for_license(watermark_value)
            
        elif watermark_type == 'tracking':
            if not watermark_value:
                raise WatermarkError('Tracking ID required for tracking watermark')
            return WatermarkID.for_tracking(watermark_value)
            
        elif watermark_type == 'custom':
            if not watermark_value:
                raise WatermarkError('Value required for custom watermark')
            
            # Try to parse as integer first, then as binary string
            if watermark_value.isdigit():
                value = int(watermark_value)
                if not 0 <= value <= 65535:  # 16-bit range check
                    raise WatermarkError('Custom watermark integer must be 0-65535')
                return WatermarkID.custom(value)
                
            elif len(watermark_value) == 16 and all(c in '01' for c in watermark_value):
                return WatermarkID.custom(watermark_value)
                
            else:
                raise WatermarkError('Custom watermark must be 16-bit binary string or integer 0-65535')
                
        else:
            raise WatermarkError(f'Invalid watermark type: {watermark_type}')
            
    except ValueError as e:
        logger.error(f"Request {request_id}: Watermark creation failed: {str(e)}", exc_info=True)
        raise WatermarkError(f'Invalid watermark value: {str(e)}')


def create_error_response(error: Union[str, Exception], status_code: int, request_id: str) -> Response:
    """
    Create standardized error response.
    
    Args:
        error: Error message or exception
        status_code: HTTP status code
        request_id: Request ID for tracking
        
    Returns:
        Response: Flask JSON response with error details
    """
    error_message = str(error)
    response_data = {
        'error': error_message,
        'request_id': request_id,
        'timestamp': datetime.now().isoformat()
    }
    
    logger.error(f"Request {request_id}: Error response {status_code}: {error_message}")
    return jsonify(response_data), status_code

# =============================================================================
# API Endpoints
# =============================================================================
@app.route('/health', methods=['GET'])
def health_check() -> Response:
    """
    Health check endpoint for service monitoring.
    
    Returns:
        Response: JSON response with service status and metadata
    """
    request_id = generate_request_id()
    
    try:
        # Check if WaveVerify instance is healthy
        health_status = {
            'status': 'healthy',
            'service': 'WaveVerify API',
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat(),
            'request_id': request_id,
            'capabilities': {
                'embed': True,
                'detect': True,
                'max_file_size_mb': app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
            }
        }
        
        logger.info(f"Request {request_id}: Health check successful")
        return jsonify(health_status)
        
    except Exception as e:
        logger.error(f"Request {request_id}: Health check failed: {str(e)}", exc_info=True)
        return create_error_response(e, 500, request_id)


@app.route('/watermark/embed', methods=['POST'])
def embed_watermark() -> Union[Response, Tuple[Response, int]]:
    """
    Embed watermark into uploaded audio file.
    
    Request form data:
        audio: Audio file upload (required)
        watermark_type: Type of watermark - 'creator', 'timestamp', 'license', 'tracking', or 'custom'
        watermark_value: Value appropriate for the watermark type (optional for timestamp)
    
    Response:
        Success: Watermarked audio file with metadata in headers
        Error: JSON error response
        
    Response Headers:
        X-Request-Id: Unique request identifier
        X-Watermark-Type: Type of embedded watermark
        X-Watermark-ID: String representation of watermark
        X-Watermark-Hex: Hexadecimal representation
        X-Watermark-Bits: Binary representation
    """
    request_id = generate_request_id()
    logger.info(f"Request {request_id}: Embed watermark request received")
    
    temp_input = None
    temp_output = None
    
    try:
        # Validate and get uploaded file
        audio_file, safe_filename = validate_audio_upload(request_id)
        
        # Get watermark parameters with defaults
        watermark_type = request.form.get('watermark_type', DEFAULT_WATERMARK_TYPE)
        watermark_value = request.form.get('watermark_value', '')
        
        logger.info(f"Request {request_id}: Watermark type={watermark_type}, has_value={bool(watermark_value)}")
        
        # Create watermark based on type
        watermark = create_watermark(watermark_type, watermark_value, request_id)
        
        # Save uploaded file temporarily with secure handling
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(safe_filename).suffix) as tmp_file:
            audio_file.save(tmp_file.name)
            temp_input = tmp_file.name
            logger.info(f"Request {request_id}: Saved temp input file: {temp_input}")
        
        # Create temporary output file
        temp_output = tempfile.mktemp(suffix='.wav')
        
        # Embed watermark with error handling
        logger.info(f"Request {request_id}: Embedding watermark into audio")
        _, _, embedded_watermark = wv.embed(temp_input, watermark, output_path=temp_output)
        
        logger.info(f"Request {request_id}: Watermark embedded successfully: {embedded_watermark}")
        
        # Prepare response with watermarked file
        response = send_file(
            temp_output,
            as_attachment=True,
            download_name=f"watermarked_{safe_filename}",
            mimetype='audio/wav'
        )
        
        # Add comprehensive metadata to response headers
        response.headers['X-Request-Id'] = request_id
        response.headers['X-Watermark-Type'] = watermark_type
        response.headers['X-Watermark-ID'] = str(embedded_watermark)
        response.headers['X-Watermark-Hex'] = embedded_watermark.to_hex()
        response.headers['X-Watermark-Bits'] = embedded_watermark.to_bits()
        response.headers['X-Processing-Time'] = datetime.now().isoformat()
        
        return response
        
    except InvalidFileError as e:
        return create_error_response(e, 400, request_id)
        
    except WatermarkError as e:
        return create_error_response(e, 400, request_id)
        
    except RequestEntityTooLarge:
        logger.warning(f"Request {request_id}: File too large")
        return create_error_response('File size exceeds maximum allowed', 413, request_id)
        
    except Exception as e:
        logger.error(f"Request {request_id}: Unexpected error: {str(e)}", exc_info=True)
        return create_error_response(f'Internal server error: {str(e)}', 500, request_id)
        
    finally:
        # Clean up temporary files in all cases
        if temp_input and os.path.exists(temp_input):
            try:
                os.unlink(temp_input)
                logger.debug(f"Request {request_id}: Cleaned up temp input file")
            except Exception as e:
                logger.warning(f"Request {request_id}: Failed to clean temp input: {str(e)}")
        # Note: temp_output cleanup is handled by Flask after file sending


@app.route('/watermark/detect', methods=['POST'])
def detect_watermark() -> Response:
    """
    Detect watermark in uploaded audio file.
    
    Request form data:
        audio: Audio file upload (required)
    
    Returns:
        Response: JSON response with detection results
        
    Response JSON:
        filename: Name of analyzed file
        watermark_detected: Boolean indicating if watermark found
        confidence: Float confidence score (0-1)
        authentication_status: 'authentic', 'suspicious', or 'not_watermarked'
        watermark: Optional dict with watermark details if detected
        request_id: Unique request identifier
        timestamp: Processing timestamp
    """
    request_id = generate_request_id()
    logger.info(f"Request {request_id}: Detect watermark request received")
    
    temp_input = None
    
    try:
        # Validate and get uploaded file
        audio_file, safe_filename = validate_audio_upload(request_id)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(safe_filename).suffix) as tmp_file:
            audio_file.save(tmp_file.name)
            temp_input = tmp_file.name
            logger.info(f"Request {request_id}: Saved temp file for detection: {temp_input}")
        
        # Detect watermark
        logger.info(f"Request {request_id}: Starting watermark detection")
        detected_watermark, confidence = wv.detect(temp_input)
        
        # Determine authentication status based on confidence thresholds
        is_authentic = confidence > CONFIDENCE_THRESHOLD_AUTHENTIC
        is_suspicious = confidence > CONFIDENCE_THRESHOLD_SUSPICIOUS
        
        if is_authentic:
            status = 'authentic'
        elif is_suspicious:
            status = 'suspicious'
        else:
            status = 'not_watermarked'
            
        logger.info(f"Request {request_id}: Detection complete - confidence={confidence:.3f}, status={status}")
        
        # Build response
        result: Dict[str, Any] = {
            'filename': safe_filename,
            'watermark_detected': confidence > CONFIDENCE_THRESHOLD_SUSPICIOUS,
            'confidence': float(confidence),
            'authentication_status': status,
            'request_id': request_id,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add watermark details if detected with sufficient confidence
        if confidence > CONFIDENCE_THRESHOLD_SUSPICIOUS and detected_watermark:
            result['watermark'] = {
                'id': str(detected_watermark),
                'hex': detected_watermark.to_hex(),
                'bits': detected_watermark.to_bits(),
                'metadata': detected_watermark.metadata
            }
            logger.info(f"Request {request_id}: Watermark details included in response")
        
        return jsonify(result)
        
    except InvalidFileError as e:
        return create_error_response(e, 400, request_id)
        
    except RequestEntityTooLarge:
        logger.warning(f"Request {request_id}: File too large")
        return create_error_response('File size exceeds maximum allowed', 413, request_id)
        
    except Exception as e:
        logger.error(f"Request {request_id}: Unexpected error: {str(e)}", exc_info=True)
        return create_error_response(f'Internal server error: {str(e)}', 500, request_id)
        
    finally:
        # Clean up temporary file
        if temp_input and os.path.exists(temp_input):
            try:
                os.unlink(temp_input)
                logger.debug(f"Request {request_id}: Cleaned up temp file")
            except Exception as e:
                logger.warning(f"Request {request_id}: Failed to clean temp file: {str(e)}")

# =============================================================================
# Application Entry Point
# =============================================================================
if __name__ == '__main__':
    # Application startup logging
    logger.info("="*80)
    logger.info("Starting WaveVerify Web API...")
    logger.info("="*80)
    
    # Log API endpoints
    logger.info("API Endpoints:")
    logger.info("  POST /watermark/embed    - Embed watermark into audio file")
    logger.info("  POST /watermark/detect   - Detect watermark in audio file")
    logger.info("  GET  /health            - Health check endpoint")
    
    # Log example usage
    logger.info("\nExample Usage:")
    logger.info("  # Embed creator watermark:")
    logger.info("  curl -X POST -F 'audio=@input.wav' \\")
    logger.info("       -F 'watermark_type=creator' \\")
    logger.info("       -F 'watermark_value=artist_name_2024' \\")
    logger.info("       http://localhost:5000/watermark/embed -o watermarked.wav")
    
    logger.info("\n  # Embed timestamp watermark (current time):")
    logger.info("  curl -X POST -F 'audio=@input.wav' \\")
    logger.info("       -F 'watermark_type=timestamp' \\")
    logger.info("       http://localhost:5000/watermark/embed -o watermarked.wav")
    
    logger.info("\n  # Detect watermark:")
    logger.info("  curl -X POST -F 'audio=@watermarked.wav' \\")
    logger.info("       http://localhost:5000/watermark/detect")
    
    logger.info("="*80)
    
    # Start Flask application
    app.run(host='0.0.0.0', port=5000, debug=True)