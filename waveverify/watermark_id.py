"""
WatermarkID - Core identity system for audio watermarking.

This module provides a high-level abstraction for watermark messages,
focusing on real-world use cases rather than raw binary strings.
"""

import hashlib
import logging
from datetime import datetime
from typing import Union, Optional, Dict, Any

logger = logging.getLogger(__name__)


class WatermarkID:
    """
    Represents a watermark identity - the core of audio authentication.
    
    A WatermarkID encapsulates a 16-bit watermark message along with
    metadata about its purpose and meaning. This design ensures every
    watermark has a clear purpose and can be traced back to its origin.
    
    IMPORTANT: The watermarking model can only embed exactly 16 bits.
    All factory methods and custom inputs are constrained to produce
    exactly 16-bit binary strings (65,536 possible values).
    """
    
    def __init__(self, bits: str):
        """
        Private constructor - use factory methods instead.
        
        Args:
            bits: 16-bit binary string (exactly 16 characters of '0' or '1')
            
        Raises:
            ValueError: If bits is not a valid 16-bit binary string
        """
        self._validate_bits(bits)
        self.bits = bits
        self.metadata: Dict[str, Any] = {}
        
        # Extra safety assertion to guarantee 16-bit constraint
        assert len(self.bits) == 16, f"Internal error: bits must be 16 chars, got {len(self.bits)}"
    
    def _validate_bits(self, bits: str) -> None:
        """Validate that bits is a proper 16-bit binary string."""
        if not isinstance(bits, str):
            raise TypeError(f"Bits must be string, got {type(bits)}")
        if len(bits) != 16:
            raise ValueError(f"Bits must be exactly 16 characters, got {len(bits)}")
        if not all(c in '01' for c in bits):
            raise ValueError(f"Bits must contain only 0 and 1, got: {bits}")
    
    @classmethod
    def for_creator(cls, creator_id: str) -> 'WatermarkID':
        """
        Create watermark for content creator identification.
        
        This method generates a consistent watermark for a given creator ID,
        useful for artists, musicians, and content creators to mark their work.
        
        Args:
            creator_id: Unique identifier for the creator (e.g., "john_doe_music")
            
        Returns:
            WatermarkID configured for creator identification
            
        Example:
            >>> wid = WatermarkID.for_creator("beyonce_2024")
            >>> print(wid)  # WatermarkID(creator='beyonce_2024')
        """
        if not creator_id or not isinstance(creator_id, str):
            raise ValueError("Creator ID must be a non-empty string")
        
        # Hash creator_id to 16 bits deterministically
        hash_bytes = hashlib.md5(creator_id.encode('utf-8')).digest()
        bits = ''.join(format(b, '08b') for b in hash_bytes[:2])
        
        instance = cls(bits)
        instance.metadata = {
            'type': 'creator',
            'id': creator_id,
            'hash_method': 'md5_first_2_bytes'
        }
        
        logger.debug(f"Created WatermarkID for creator: {creator_id}")
        return instance
    
    @classmethod
    def for_timestamp(cls, timestamp: Optional[datetime] = None) -> 'WatermarkID':
        """
        Create time-based watermark for tracking when audio was created.
        
        Encodes timestamp into 16 bits using:
        - 5 bits: year (0-31, offset from 2024)
        - 4 bits: month (1-12)
        - 5 bits: day (1-31)
        - 2 bits: quarter of day (0-3, each = 6 hours)
        
        Args:
            timestamp: Datetime to encode (default: current time)
            
        Returns:
            WatermarkID configured for timestamp tracking
            
        Example:
            >>> wid = WatermarkID.for_timestamp()
            >>> print(wid)  # WatermarkID(time='2024-07-17T12:34:56')
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Validate year range (2024-2055)
        year_offset = timestamp.year - 2024
        if year_offset < 0 or year_offset > 31:
            raise ValueError(f"Year must be between 2024 and 2055, got {timestamp.year}")
        
        # Calculate quarter of day (0-3)
        quarter = timestamp.hour // 6
        
        # Pack into 16 bits
        bits = (f"{year_offset:05b}"     # 5 bits: year offset
                f"{timestamp.month:04b}"   # 4 bits: month
                f"{timestamp.day:05b}"     # 5 bits: day
                f"{quarter:02b}")          # 2 bits: quarter of day
        
        instance = cls(bits)
        instance.metadata = {
            'type': 'timestamp',
            'time': timestamp.isoformat(),
            'year': timestamp.year,
            'month': timestamp.month,
            'day': timestamp.day,
            'quarter': quarter
        }
        
        logger.debug(f"Created WatermarkID for timestamp: {timestamp.isoformat()}")
        return instance
    
    @classmethod
    def for_license(cls, license_type: str) -> 'WatermarkID':
        """
        Create watermark for license/rights management.
        
        Supports common Creative Commons licenses and custom licensing.
        
        Args:
            license_type: License identifier (e.g., "CC-BY", "ALL-RIGHTS")
            
        Returns:
            WatermarkID configured for license tracking
            
        Example:
            >>> wid = WatermarkID.for_license("CC-BY-4.0")
            >>> print(wid)  # WatermarkID(license='CC-BY-4.0')
        """
        # Predefined license codes (16-bit values)
        licenses = {
            'CC0': 0x0000,           # Public domain
            'CC-BY': 0x0001,         # Attribution
            'CC-BY-SA': 0x0002,      # Attribution-ShareAlike
            'CC-BY-NC': 0x0003,      # Attribution-NonCommercial
            'CC-BY-NC-SA': 0x0004,   # Attribution-NonCommercial-ShareAlike
            'CC-BY-ND': 0x0005,      # Attribution-NoDerivatives
            'CC-BY-NC-ND': 0x0006,   # Attribution-NonCommercial-NoDerivatives
            'ALL-RIGHTS': 0xFFFF,    # All rights reserved
            'CUSTOM': 0x8000,        # Custom license
        }
        
        # Normalize license type
        normalized = license_type.upper().replace('_', '-')
        
        # Try exact match first
        if normalized in licenses:
            code = licenses[normalized]
        else:
            # Remove version numbers for matching
            base_license = normalized.split('-')[0] if '-' in normalized else normalized
            if base_license == 'CC' and '-' in normalized:
                # Handle CC licenses - take up to 3 parts (e.g., CC-BY-SA)
                parts = normalized.split('-')
                base_license = '-'.join(parts[:min(3, len(parts))])
            
            # Get license code
            code = licenses.get(base_license, licenses['CUSTOM'])
        
        # If custom, incorporate license string into code
        if code == licenses['CUSTOM']:
            # Hash the license type and use lower bits
            hash_val = hashlib.md5(license_type.encode()).digest()
            code = 0x8000 | (int.from_bytes(hash_val[:2], 'big') & 0x7FFF)
        
        bits = format(code, '016b')
        
        instance = cls(bits)
        instance.metadata = {
            'type': 'license',
            'license': license_type,
            'code': f"0x{code:04X}",
            'is_custom': code >= 0x8000
        }
        
        logger.debug(f"Created WatermarkID for license: {license_type}")
        return instance
    
    @classmethod
    def for_tracking(cls, tracking_id: str) -> 'WatermarkID':
        """
        Create watermark for distribution tracking.
        
        Useful for tracking specific copies, episodes, or distributions.
        
        Args:
            tracking_id: Unique tracking identifier (e.g., "podcast-ep-123")
            
        Returns:
            WatermarkID configured for tracking
            
        Example:
            >>> wid = WatermarkID.for_tracking("CASE-2024-001")
            >>> print(wid)  # WatermarkID(tracking='CASE-2024-001')
        """
        if not tracking_id or not isinstance(tracking_id, str):
            raise ValueError("Tracking ID must be a non-empty string")
        
        # Handle different tracking ID formats
        if tracking_id.isdigit() and len(tracking_id) <= 5:
            # Try direct numeric ID
            tracking_num = int(tracking_id)
            if tracking_num <= 65535:
                # Can encode directly
                bits = format(tracking_num, '016b')
                id_type = 'numeric'
            else:
                # Too large, use hash instead
                hash_bytes = hashlib.md5(tracking_id.encode('utf-8')).digest()
                bits = ''.join(format(b, '08b') for b in hash_bytes[:2])
                id_type = 'hashed'
        else:
            # Hash longer/complex IDs
            hash_bytes = hashlib.md5(tracking_id.encode('utf-8')).digest()
            bits = ''.join(format(b, '08b') for b in hash_bytes[:2])
            id_type = 'hashed'
        
        instance = cls(bits)
        instance.metadata = {
            'type': 'tracking',
            'id': tracking_id,
            'id_type': id_type
        }
        
        logger.debug(f"Created WatermarkID for tracking: {tracking_id}")
        return instance
    
    @classmethod
    def custom(cls, value: Union[str, int, bytes]) -> 'WatermarkID':
        """
        Create custom watermark with validation.
        
        Supports multiple input formats for flexibility.
        
        Args:
            value: One of:
                - 16-bit binary string: "1010101010101010"
                - Integer (0-65535): 42
                - 2 bytes: b'\\xAB\\xCD'
                
        Returns:
            WatermarkID with custom configuration
            
        Raises:
            ValueError: If value cannot be converted to 16 bits
            TypeError: If value type is not supported
        """
        if isinstance(value, str):
            if len(value) == 16 and all(c in '01' for c in value):
                bits = value
            else:
                raise ValueError(
                    f"String must be 16-bit binary (got {len(value)} chars). "
                    f"Example: '1010101010101010'"
                )
        elif isinstance(value, int):
            if 0 <= value <= 65535:
                bits = format(value, '016b')
            else:
                raise ValueError(f"Integer must be 0-65535, got {value}")
        elif isinstance(value, bytes):
            if len(value) == 2:
                bits = ''.join(format(b, '08b') for b in value)
            else:
                raise ValueError(f"Bytes must be exactly 2 bytes, got {len(value)}")
        else:
            raise TypeError(
                f"Unsupported type {type(value)}. "
                f"Use string, int, or bytes."
            )
        
        instance = cls(bits)
        instance.metadata = {
            'type': 'custom',
            'value': str(value),
            'value_type': type(value).__name__
        }
        
        logger.debug(f"Created custom WatermarkID from {type(value).__name__}")
        return instance
    
    def to_bits(self) -> str:
        """
        Get 16-bit binary representation.
        
        Returns:
            16-character string of 0s and 1s
        """
        return self.bits
    
    def to_hex(self) -> str:
        """
        Get hexadecimal representation.
        
        Returns:
            4-character hex string (e.g., "ABCD")
        """
        return format(int(self.bits, 2), '04X')
    
    def to_int(self) -> int:
        """
        Get integer representation.
        
        Returns:
            Integer value (0-65535)
        """
        return int(self.bits, 2)
    
    def to_bytes(self) -> bytes:
        """
        Get bytes representation.
        
        Returns:
            2-byte sequence
        """
        val = self.to_int()
        return bytes([(val >> 8) & 0xFF, val & 0xFF])
    
    def __str__(self) -> str:
        """Human-readable representation."""
        meta_type = self.metadata.get('type', 'unknown')
        
        if meta_type == 'creator':
            return f"WatermarkID(creator='{self.metadata['id']}')"
        elif meta_type == 'timestamp':
            return f"WatermarkID(time='{self.metadata['time']}')"
        elif meta_type == 'license':
            return f"WatermarkID(license='{self.metadata['license']}')"
        elif meta_type == 'tracking':
            return f"WatermarkID(tracking='{self.metadata['id']}')"
        elif meta_type == 'custom':
            return f"WatermarkID(custom={self.to_hex()})"
        else:
            return f"WatermarkID(bits='{self.bits}')"
    
    def __repr__(self) -> str:
        """Developer representation."""
        return f"WatermarkID(bits='{self.bits}', metadata={self.metadata})"
    
    def __eq__(self, other) -> bool:
        """Check equality based on bits."""
        if isinstance(other, WatermarkID):
            return self.bits == other.bits
        return False
    
    def __hash__(self) -> int:
        """Make WatermarkID hashable."""
        return hash(self.bits)