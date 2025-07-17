"""
Superposition Serializer - AGENT 1 MISSION COMPLETE

This module provides comprehensive serialization and persistence capabilities
for superposition states, enabling storage, retrieval, and transmission of
superposition data across different systems and formats.

Key Features:
- Multiple serialization formats (JSON, pickle, HDF5, custom binary)
- Compression and optimization for large superposition states
- Versioning and backward compatibility
- Validation and integrity checking
- High-performance serialization/deserialization
- Integration with existing GrandModel patterns

Supported Formats:
- JSON: Human-readable, cross-platform
- Pickle: Python native, fast for Python systems
- HDF5: Efficient for large numerical data
- Binary: Custom optimized format
- MessagePack: Compact binary format

Author: Agent 1 - Universal Superposition Core Architect
Version: 1.0 - Complete Serialization System
"""

import numpy as np
import torch
import json
import pickle
import gzip
import bz2
import lzma
from typing import Dict, List, Optional, Any, Union, Tuple, BinaryIO, TextIO
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import time
import logging
from pathlib import Path
import threading
from functools import wraps
import base64
import hashlib
import struct
from datetime import datetime
import warnings

# Optional imports for advanced formats
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    h5py = None

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False
    msgpack = None

from .universal_superposition import SuperpositionState, ActionSpaceType, SuperpositionError
from .superposition_validator import SuperpositionValidator, ValidationLevel

logger = logging.getLogger(__name__)


class SerializationFormat(Enum):
    """Supported serialization formats"""
    JSON = "json"
    PICKLE = "pickle"
    HDF5 = "hdf5"
    BINARY = "binary"
    MSGPACK = "msgpack"


class CompressionType(Enum):
    """Supported compression types"""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZMA = "lzma"


@dataclass
class SerializationMetadata:
    """Metadata for serialized superposition states"""
    format_version: str = "1.0"
    serialization_format: SerializationFormat = SerializationFormat.JSON
    compression_type: CompressionType = CompressionType.NONE
    timestamp: datetime = field(default_factory=datetime.now)
    checksum: str = ""
    original_size_bytes: int = 0
    compressed_size_bytes: int = 0
    serialization_time_ms: float = 0.0
    validation_passed: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "format_version": self.format_version,
            "serialization_format": self.serialization_format.value,
            "compression_type": self.compression_type.value,
            "timestamp": self.timestamp.isoformat(),
            "checksum": self.checksum,
            "original_size_bytes": self.original_size_bytes,
            "compressed_size_bytes": self.compressed_size_bytes,
            "serialization_time_ms": self.serialization_time_ms,
            "validation_passed": self.validation_passed
        }


@dataclass
class SerializedSuperposition:
    """Container for serialized superposition data"""
    data: bytes
    metadata: SerializationMetadata
    
    def get_compression_ratio(self) -> float:
        """Calculate compression ratio"""
        if self.metadata.original_size_bytes == 0:
            return 1.0
        return self.metadata.compressed_size_bytes / self.metadata.original_size_bytes


class SerializationError(SuperpositionError):
    """Exception for serialization errors"""
    pass


class DeserializationError(SuperpositionError):
    """Exception for deserialization errors"""
    pass


class SuperpositionSerializer(ABC):
    """Abstract base class for superposition serializers"""
    
    @abstractmethod
    def serialize(self, state: SuperpositionState) -> bytes:
        """Serialize superposition state to bytes"""
        pass
    
    @abstractmethod
    def deserialize(self, data: bytes) -> SuperpositionState:
        """Deserialize bytes to superposition state"""
        pass
    
    @abstractmethod
    def get_format(self) -> SerializationFormat:
        """Get serialization format"""
        pass


class JSONSerializer(SuperpositionSerializer):
    """JSON serializer for superposition states"""
    
    def __init__(self, indent: Optional[int] = None):
        self.indent = indent
    
    def serialize(self, state: SuperpositionState) -> bytes:
        """Serialize to JSON"""
        try:
            # Convert to serializable format
            def make_serializable(obj):
                """Convert numpy arrays and other non-serializable objects to serializable format"""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, torch.Tensor):
                    return obj.numpy().tolist()
                elif isinstance(obj, list):
                    return [make_serializable(item) for item in obj]
                elif isinstance(obj, dict):
                    return {key: make_serializable(value) for key, value in obj.items()}
                else:
                    return obj
            
            data = {
                "amplitudes": {
                    "real": state.amplitudes.real.numpy().tolist(),
                    "imag": state.amplitudes.imag.numpy().tolist()
                },
                "basis_actions": make_serializable(state.basis_actions),
                "action_space_type": state.action_space_type.value,
                "original_format": state.original_format,
                "timestamp": state.timestamp,
                "conversion_time_ms": state.conversion_time_ms,
                "validation_passed": state.validation_passed,
                "coherence_measure": state.coherence_measure,
                "entanglement_info": make_serializable(state.entanglement_info)
            }
            
            json_str = json.dumps(data, indent=self.indent)
            return json_str.encode('utf-8')
            
        except Exception as e:
            raise SerializationError(f"JSON serialization failed: {str(e)}") from e
    
    def deserialize(self, data: bytes) -> SuperpositionState:
        """Deserialize from JSON"""
        try:
            json_str = data.decode('utf-8')
            data_dict = json.loads(json_str)
            
            # Reconstruct complex amplitudes
            real_part = np.array(data_dict["amplitudes"]["real"])
            imag_part = np.array(data_dict["amplitudes"]["imag"])
            amplitudes = torch.tensor(real_part + 1j * imag_part, dtype=torch.complex64)
            
            # Reconstruct superposition state
            state = SuperpositionState(
                amplitudes=amplitudes,
                basis_actions=data_dict["basis_actions"],
                action_space_type=ActionSpaceType(data_dict["action_space_type"]),
                original_format=data_dict["original_format"],
                timestamp=data_dict["timestamp"],
                conversion_time_ms=data_dict["conversion_time_ms"],
                validation_passed=data_dict["validation_passed"],
                coherence_measure=data_dict["coherence_measure"],
                entanglement_info=data_dict["entanglement_info"]
            )
            
            return state
            
        except Exception as e:
            raise DeserializationError(f"JSON deserialization failed: {str(e)}") from e
    
    def get_format(self) -> SerializationFormat:
        return SerializationFormat.JSON


class PickleSerializer(SuperpositionSerializer):
    """Pickle serializer for superposition states"""
    
    def __init__(self, protocol: int = pickle.HIGHEST_PROTOCOL):
        self.protocol = protocol
    
    def serialize(self, state: SuperpositionState) -> bytes:
        """Serialize to pickle"""
        try:
            return pickle.dumps(state, protocol=self.protocol)
        except Exception as e:
            raise SerializationError(f"Pickle serialization failed: {str(e)}") from e
    
    def deserialize(self, data: bytes) -> SuperpositionState:
        """Deserialize from pickle"""
        try:
            return pickle.loads(data)
        except Exception as e:
            raise DeserializationError(f"Pickle deserialization failed: {str(e)}") from e
    
    def get_format(self) -> SerializationFormat:
        return SerializationFormat.PICKLE


class HDF5Serializer(SuperpositionSerializer):
    """HDF5 serializer for superposition states"""
    
    def __init__(self):
        if not HDF5_AVAILABLE:
            raise ImportError("HDF5 serialization requires h5py package")
    
    def serialize(self, state: SuperpositionState) -> bytes:
        """Serialize to HDF5"""
        try:
            import io
            
            # Create in-memory file
            buffer = io.BytesIO()
            
            with h5py.File(buffer, 'w') as f:
                # Store amplitudes
                f.create_dataset('amplitudes_real', data=state.amplitudes.real.numpy())
                f.create_dataset('amplitudes_imag', data=state.amplitudes.imag.numpy())
                
                # Store basis actions (as strings)
                basis_str = json.dumps(state.basis_actions)
                f.create_dataset('basis_actions', data=basis_str)
                
                # Store metadata
                f.attrs['action_space_type'] = state.action_space_type.value
                f.attrs['original_format'] = state.original_format
                f.attrs['timestamp'] = state.timestamp
                f.attrs['conversion_time_ms'] = state.conversion_time_ms
                f.attrs['validation_passed'] = state.validation_passed
                f.attrs['coherence_measure'] = state.coherence_measure
                
                # Store entanglement info if present
                if state.entanglement_info:
                    f.attrs['entanglement_info'] = json.dumps(state.entanglement_info)
            
            return buffer.getvalue()
            
        except Exception as e:
            raise SerializationError(f"HDF5 serialization failed: {str(e)}") from e
    
    def deserialize(self, data: bytes) -> SuperpositionState:
        """Deserialize from HDF5"""
        try:
            import io
            
            buffer = io.BytesIO(data)
            
            with h5py.File(buffer, 'r') as f:
                # Load amplitudes
                real_part = f['amplitudes_real'][:]
                imag_part = f['amplitudes_imag'][:]
                amplitudes = torch.tensor(real_part + 1j * imag_part, dtype=torch.complex64)
                
                # Load basis actions
                basis_str = f['basis_actions'][()].decode('utf-8')
                basis_actions = json.loads(basis_str)
                
                # Load metadata
                action_space_type = ActionSpaceType(f.attrs['action_space_type'])
                original_format = f.attrs['original_format']
                timestamp = f.attrs['timestamp']
                conversion_time_ms = f.attrs['conversion_time_ms']
                validation_passed = f.attrs['validation_passed']
                coherence_measure = f.attrs['coherence_measure']
                
                # Load entanglement info if present
                entanglement_info = None
                if 'entanglement_info' in f.attrs:
                    entanglement_info = json.loads(f.attrs['entanglement_info'])
                
                state = SuperpositionState(
                    amplitudes=amplitudes,
                    basis_actions=basis_actions,
                    action_space_type=action_space_type,
                    original_format=original_format,
                    timestamp=timestamp,
                    conversion_time_ms=conversion_time_ms,
                    validation_passed=validation_passed,
                    coherence_measure=coherence_measure,
                    entanglement_info=entanglement_info
                )
                
                return state
                
        except Exception as e:
            raise DeserializationError(f"HDF5 deserialization failed: {str(e)}") from e
    
    def get_format(self) -> SerializationFormat:
        return SerializationFormat.HDF5


class BinarySerializer(SuperpositionSerializer):
    """Custom binary serializer for superposition states"""
    
    def __init__(self):
        self.magic_bytes = b'SUPERPOS'
        self.version = 1
    
    def serialize(self, state: SuperpositionState) -> bytes:
        """Serialize to custom binary format"""
        try:
            buffer = bytearray()
            
            # Magic bytes and version
            buffer.extend(self.magic_bytes)
            buffer.extend(struct.pack('<I', self.version))
            
            # Amplitudes
            amp_real = state.amplitudes.real.numpy().astype(np.float32)
            amp_imag = state.amplitudes.imag.numpy().astype(np.float32)
            
            buffer.extend(struct.pack('<I', len(amp_real)))  # Length
            buffer.extend(amp_real.tobytes())
            buffer.extend(amp_imag.tobytes())
            
            # Basis actions (JSON encoded)
            def make_serializable(obj):
                """Convert numpy arrays and other non-serializable objects to serializable format"""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, torch.Tensor):
                    return obj.numpy().tolist()
                elif isinstance(obj, list):
                    return [make_serializable(item) for item in obj]
                elif isinstance(obj, dict):
                    return {key: make_serializable(value) for key, value in obj.items()}
                else:
                    return obj
            
            basis_json = json.dumps(make_serializable(state.basis_actions)).encode('utf-8')
            buffer.extend(struct.pack('<I', len(basis_json)))
            buffer.extend(basis_json)
            
            # Metadata
            metadata = {
                'action_space_type': state.action_space_type.value,
                'original_format': state.original_format,
                'timestamp': state.timestamp,
                'conversion_time_ms': state.conversion_time_ms,
                'validation_passed': state.validation_passed,
                'coherence_measure': state.coherence_measure,
                'entanglement_info': state.entanglement_info
            }
            
            metadata_json = json.dumps(metadata).encode('utf-8')
            buffer.extend(struct.pack('<I', len(metadata_json)))
            buffer.extend(metadata_json)
            
            return bytes(buffer)
            
        except Exception as e:
            raise SerializationError(f"Binary serialization failed: {str(e)}") from e
    
    def deserialize(self, data: bytes) -> SuperpositionState:
        """Deserialize from custom binary format"""
        try:
            offset = 0
            
            # Check magic bytes
            if data[offset:offset+8] != self.magic_bytes:
                raise DeserializationError("Invalid magic bytes")
            offset += 8
            
            # Check version
            version = struct.unpack('<I', data[offset:offset+4])[0]
            if version != self.version:
                raise DeserializationError(f"Unsupported version: {version}")
            offset += 4
            
            # Read amplitudes
            amp_len = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4
            
            real_bytes = data[offset:offset+amp_len*4]
            offset += amp_len * 4
            imag_bytes = data[offset:offset+amp_len*4]
            offset += amp_len * 4
            
            amp_real = np.frombuffer(real_bytes, dtype=np.float32)
            amp_imag = np.frombuffer(imag_bytes, dtype=np.float32)
            amplitudes = torch.tensor(amp_real + 1j * amp_imag, dtype=torch.complex64)
            
            # Read basis actions
            basis_len = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4
            basis_json = data[offset:offset+basis_len].decode('utf-8')
            offset += basis_len
            basis_actions = json.loads(basis_json)
            
            # Read metadata
            metadata_len = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4
            metadata_json = data[offset:offset+metadata_len].decode('utf-8')
            metadata = json.loads(metadata_json)
            
            # Reconstruct state
            state = SuperpositionState(
                amplitudes=amplitudes,
                basis_actions=basis_actions,
                action_space_type=ActionSpaceType(metadata['action_space_type']),
                original_format=metadata['original_format'],
                timestamp=metadata['timestamp'],
                conversion_time_ms=metadata['conversion_time_ms'],
                validation_passed=metadata['validation_passed'],
                coherence_measure=metadata['coherence_measure'],
                entanglement_info=metadata['entanglement_info']
            )
            
            return state
            
        except Exception as e:
            raise DeserializationError(f"Binary deserialization failed: {str(e)}") from e
    
    def get_format(self) -> SerializationFormat:
        return SerializationFormat.BINARY


class MessagePackSerializer(SuperpositionSerializer):
    """MessagePack serializer for superposition states"""
    
    def __init__(self):
        if not MSGPACK_AVAILABLE:
            raise ImportError("MessagePack serialization requires msgpack package")
    
    def serialize(self, state: SuperpositionState) -> bytes:
        """Serialize to MessagePack"""
        try:
            def make_serializable(obj):
                """Convert numpy arrays and other non-serializable objects to serializable format"""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, torch.Tensor):
                    return obj.numpy().tolist()
                elif isinstance(obj, list):
                    return [make_serializable(item) for item in obj]
                elif isinstance(obj, dict):
                    return {key: make_serializable(value) for key, value in obj.items()}
                else:
                    return obj
            
            data = {
                "amplitudes": {
                    "real": state.amplitudes.real.numpy().tolist(),
                    "imag": state.amplitudes.imag.numpy().tolist()
                },
                "basis_actions": make_serializable(state.basis_actions),
                "action_space_type": state.action_space_type.value,
                "original_format": state.original_format,
                "timestamp": state.timestamp,
                "conversion_time_ms": state.conversion_time_ms,
                "validation_passed": state.validation_passed,
                "coherence_measure": state.coherence_measure,
                "entanglement_info": make_serializable(state.entanglement_info)
            }
            
            return msgpack.packb(data, use_bin_type=True)
            
        except Exception as e:
            raise SerializationError(f"MessagePack serialization failed: {str(e)}") from e
    
    def deserialize(self, data: bytes) -> SuperpositionState:
        """Deserialize from MessagePack"""
        try:
            data_dict = msgpack.unpackb(data, raw=False)
            
            # Reconstruct complex amplitudes
            real_part = np.array(data_dict["amplitudes"]["real"])
            imag_part = np.array(data_dict["amplitudes"]["imag"])
            amplitudes = torch.tensor(real_part + 1j * imag_part, dtype=torch.complex64)
            
            # Reconstruct superposition state
            state = SuperpositionState(
                amplitudes=amplitudes,
                basis_actions=data_dict["basis_actions"],
                action_space_type=ActionSpaceType(data_dict["action_space_type"]),
                original_format=data_dict["original_format"],
                timestamp=data_dict["timestamp"],
                conversion_time_ms=data_dict["conversion_time_ms"],
                validation_passed=data_dict["validation_passed"],
                coherence_measure=data_dict["coherence_measure"],
                entanglement_info=data_dict["entanglement_info"]
            )
            
            return state
            
        except Exception as e:
            raise DeserializationError(f"MessagePack deserialization failed: {str(e)}") from e
    
    def get_format(self) -> SerializationFormat:
        return SerializationFormat.MSGPACK


class SuperpositionPersistence:
    """
    Main persistence manager for superposition states
    
    Provides high-level interface for saving/loading superposition states
    with support for multiple formats, compression, and validation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize persistence manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.default_format = SerializationFormat(self.config.get('default_format', 'json'))
        self.default_compression = CompressionType(self.config.get('default_compression', 'none'))
        self.validate_on_save = self.config.get('validate_on_save', True)
        self.validate_on_load = self.config.get('validate_on_load', True)
        
        # Initialize serializers
        self.serializers = {
            SerializationFormat.JSON: JSONSerializer(),
            SerializationFormat.PICKLE: PickleSerializer(),
            SerializationFormat.BINARY: BinarySerializer()
        }
        
        # Add optional serializers
        if HDF5_AVAILABLE:
            self.serializers[SerializationFormat.HDF5] = HDF5Serializer()
        
        if MSGPACK_AVAILABLE:
            self.serializers[SerializationFormat.MSGPACK] = MessagePackSerializer()
        
        # Initialize validator
        self.validator = SuperpositionValidator()
        
        # Statistics
        self.save_count = 0
        self.load_count = 0
        self.total_save_time = 0.0
        self.total_load_time = 0.0
        self.lock = threading.Lock()
        
        logger.info(f"Initialized SuperpositionPersistence with {len(self.serializers)} serializers")
    
    def save(self, 
             state: SuperpositionState, 
             path: Union[str, Path],
             format: Optional[SerializationFormat] = None,
             compression: Optional[CompressionType] = None,
             metadata: Optional[Dict[str, Any]] = None) -> SerializationMetadata:
        """
        Save superposition state to file
        
        Args:
            state: Superposition state to save
            path: File path
            format: Serialization format
            compression: Compression type
            metadata: Additional metadata
            
        Returns:
            SerializationMetadata
        """
        start_time = time.time()
        
        # Use defaults if not specified
        format = format or self.default_format
        compression = compression or self.default_compression
        
        try:
            # Validate state if enabled
            if self.validate_on_save:
                report = self.validator.validate(state, ValidationLevel.STANDARD)
                if report.critical_errors > 0:
                    raise SerializationError(f"State validation failed: {report.critical_errors} critical errors")
            
            # Serialize
            if format not in self.serializers:
                raise SerializationError(f"Unsupported format: {format}")
            
            serializer = self.serializers[format]
            data = serializer.serialize(state)
            
            # Compress if requested
            compressed_data = self._compress_data(data, compression)
            
            # Calculate checksum
            checksum = hashlib.sha256(compressed_data).hexdigest()
            
            # Create metadata
            serialization_metadata = SerializationMetadata(
                serialization_format=format,
                compression_type=compression,
                checksum=checksum,
                original_size_bytes=len(data),
                compressed_size_bytes=len(compressed_data),
                serialization_time_ms=(time.time() - start_time) * 1000,
                validation_passed=True
            )
            
            if metadata:
                serialization_metadata.to_dict().update(metadata)
            
            # Save to file
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'wb') as f:
                # Write metadata header
                metadata_json = json.dumps(serialization_metadata.to_dict()).encode('utf-8')
                f.write(struct.pack('<I', len(metadata_json)))
                f.write(metadata_json)
                
                # Write data
                f.write(compressed_data)
            
            # Update statistics
            with self.lock:
                self.save_count += 1
                self.total_save_time += (time.time() - start_time)
            
            logger.info(f"Saved superposition to {path} ({format.value}, {len(compressed_data)} bytes)")
            
            return serialization_metadata
            
        except Exception as e:
            raise SerializationError(f"Failed to save superposition: {str(e)}") from e
    
    def load(self, 
             path: Union[str, Path],
             validate: Optional[bool] = None) -> Tuple[SuperpositionState, SerializationMetadata]:
        """
        Load superposition state from file
        
        Args:
            path: File path
            validate: Whether to validate loaded state
            
        Returns:
            Tuple of (SuperpositionState, SerializationMetadata)
        """
        start_time = time.time()
        
        validate = validate if validate is not None else self.validate_on_load
        
        try:
            path = Path(path)
            if not path.exists():
                raise DeserializationError(f"File not found: {path}")
            
            with open(path, 'rb') as f:
                # Read metadata header
                metadata_len = struct.unpack('<I', f.read(4))[0]
                metadata_json = f.read(metadata_len).decode('utf-8')
                metadata_dict = json.loads(metadata_json)
                
                # Read data
                compressed_data = f.read()
            
            # Reconstruct metadata
            metadata = SerializationMetadata(**metadata_dict)
            
            # Verify checksum
            calculated_checksum = hashlib.sha256(compressed_data).hexdigest()
            if calculated_checksum != metadata.checksum:
                raise DeserializationError("Checksum mismatch - file may be corrupted")
            
            # Decompress
            data = self._decompress_data(compressed_data, metadata.compression_type)
            
            # Deserialize
            if metadata.serialization_format not in self.serializers:
                raise DeserializationError(f"Unsupported format: {metadata.serialization_format}")
            
            serializer = self.serializers[metadata.serialization_format]
            state = serializer.deserialize(data)
            
            # Validate if requested
            if validate:
                report = self.validator.validate(state, ValidationLevel.STANDARD)
                if report.critical_errors > 0:
                    logger.warning(f"Loaded state has validation issues: {report.critical_errors} critical errors")
            
            # Update statistics
            with self.lock:
                self.load_count += 1
                self.total_load_time += (time.time() - start_time)
            
            logger.info(f"Loaded superposition from {path} ({metadata.serialization_format.value})")
            
            return state, metadata
            
        except Exception as e:
            raise DeserializationError(f"Failed to load superposition: {str(e)}") from e
    
    def save_batch(self, 
                   states: List[SuperpositionState], 
                   directory: Union[str, Path],
                   format: Optional[SerializationFormat] = None,
                   compression: Optional[CompressionType] = None) -> List[SerializationMetadata]:
        """
        Save multiple superposition states
        
        Args:
            states: List of superposition states
            directory: Directory to save to
            format: Serialization format
            compression: Compression type
            
        Returns:
            List of SerializationMetadata
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        metadata_list = []
        
        for i, state in enumerate(states):
            filename = f"superposition_{i:04d}.{format.value if format else self.default_format.value}"
            path = directory / filename
            
            try:
                metadata = self.save(state, path, format, compression)
                metadata_list.append(metadata)
            except Exception as e:
                logger.error(f"Failed to save state {i}: {str(e)}")
                # Create error metadata
                error_metadata = SerializationMetadata(
                    serialization_format=format or self.default_format,
                    compression_type=compression or self.default_compression,
                    validation_passed=False
                )
                metadata_list.append(error_metadata)
        
        return metadata_list
    
    def load_batch(self, 
                   directory: Union[str, Path],
                   pattern: str = "superposition_*.json") -> List[Tuple[SuperpositionState, SerializationMetadata]]:
        """
        Load multiple superposition states
        
        Args:
            directory: Directory to load from
            pattern: File pattern to match
            
        Returns:
            List of (SuperpositionState, SerializationMetadata) tuples
        """
        directory = Path(directory)
        files = list(directory.glob(pattern))
        
        results = []
        
        for file_path in sorted(files):
            try:
                state, metadata = self.load(file_path)
                results.append((state, metadata))
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {str(e)}")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get persistence statistics"""
        with self.lock:
            return {
                "save_count": self.save_count,
                "load_count": self.load_count,
                "total_save_time_ms": self.total_save_time * 1000,
                "total_load_time_ms": self.total_load_time * 1000,
                "average_save_time_ms": (self.total_save_time / self.save_count * 1000) if self.save_count > 0 else 0,
                "average_load_time_ms": (self.total_load_time / self.load_count * 1000) if self.load_count > 0 else 0,
                "supported_formats": list(self.serializers.keys())
            }
    
    def _compress_data(self, data: bytes, compression: CompressionType) -> bytes:
        """Compress data using specified compression type"""
        if compression == CompressionType.NONE:
            return data
        elif compression == CompressionType.GZIP:
            return gzip.compress(data)
        elif compression == CompressionType.BZIP2:
            return bz2.compress(data)
        elif compression == CompressionType.LZMA:
            return lzma.compress(data)
        else:
            raise SerializationError(f"Unsupported compression type: {compression}")
    
    def _decompress_data(self, data: bytes, compression: CompressionType) -> bytes:
        """Decompress data using specified compression type"""
        if compression == CompressionType.NONE:
            return data
        elif compression == CompressionType.GZIP:
            return gzip.decompress(data)
        elif compression == CompressionType.BZIP2:
            return bz2.decompress(data)
        elif compression == CompressionType.LZMA:
            return lzma.decompress(data)
        else:
            raise DeserializationError(f"Unsupported compression type: {compression}")


# Factory function
def create_persistence_manager(config: Optional[Dict[str, Any]] = None) -> SuperpositionPersistence:
    """Create SuperpositionPersistence instance"""
    return SuperpositionPersistence(config)


# Convenience functions
def save_superposition(state: SuperpositionState, 
                      path: Union[str, Path],
                      format: SerializationFormat = SerializationFormat.JSON) -> SerializationMetadata:
    """Convenience function to save superposition"""
    manager = create_persistence_manager()
    return manager.save(state, path, format)


def load_superposition(path: Union[str, Path]) -> SuperpositionState:
    """Convenience function to load superposition"""
    manager = create_persistence_manager()
    state, _ = manager.load(path)
    return state


# Test function
def test_serialization():
    """Test serialization functionality"""
    print("🧪 Testing Superposition Serialization")
    
    # Import required modules
    from .universal_superposition import create_uniform_superposition
    import tempfile
    
    # Create test state
    test_state = create_uniform_superposition(["action_a", "action_b", "action_c"])
    
    # Test different formats
    manager = create_persistence_manager()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test each available format
        for format in manager.serializers.keys():
            try:
                # Save
                save_path = temp_path / f"test_state.{format.value}"
                metadata = manager.save(test_state, save_path, format)
                print(f"✅ {format.value}: Saved ({metadata.compressed_size_bytes} bytes)")
                
                # Load
                loaded_state, loaded_metadata = manager.load(save_path)
                print(f"✅ {format.value}: Loaded successfully")
                
                # Verify
                if len(loaded_state.basis_actions) == len(test_state.basis_actions):
                    print(f"✅ {format.value}: Verification passed")
                else:
                    print(f"❌ {format.value}: Verification failed")
                    
            except Exception as e:
                print(f"❌ {format.value}: {str(e)}")
    
    # Performance statistics
    stats = manager.get_statistics()
    print(f"\n📊 Performance Statistics:")
    print(f"   Save operations: {stats['save_count']}")
    print(f"   Load operations: {stats['load_count']}")
    print(f"   Average save time: {stats['average_save_time_ms']:.2f}ms")
    print(f"   Average load time: {stats['average_load_time_ms']:.2f}ms")
    
    print("\n✅ Serialization testing complete!")


if __name__ == "__main__":
    test_serialization()