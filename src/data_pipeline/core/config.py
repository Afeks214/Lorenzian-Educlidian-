"""Configuration management for data pipeline"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

@dataclass
class DataPipelineConfig:
    """Configuration for the data pipeline system"""
    
    # Data Loading Configuration
    chunk_size: int = 10000  # Rows to process at once
    max_workers: int = os.cpu_count() or 4  # Parallel processing workers
    memory_limit_mb: int = 1024  # Memory limit in MB
    
    # File Processing Configuration
    supported_formats: List[str] = field(default_factory=lambda: ['.csv', '.parquet', '.h5', '.feather'])
    compression_types: List[str] = field(default_factory=lambda: ['gzip', 'bz2', 'xz', 'snappy'])
    
    # Streaming Configuration
    stream_buffer_size: int = 1000  # Buffer size for streaming
    stream_timeout: float = 30.0  # Timeout in seconds
    
    # Caching Configuration
    cache_dir: str = "/tmp/data_pipeline_cache"
    cache_max_size_gb: float = 10.0  # Maximum cache size in GB
    cache_ttl_hours: int = 24  # Cache time-to-live
    
    # Validation Configuration
    validation_sample_size: int = 1000  # Sample size for validation
    validation_threshold: float = 0.95  # Validation pass threshold
    
    # Performance Configuration
    performance_monitoring: bool = True
    performance_log_interval: int = 1000  # Log every N operations
    memory_check_interval: int = 100  # Check memory every N operations
    
    # Database Configuration
    db_connection_pool_size: int = 10
    db_timeout: float = 30.0
    
    # Advanced Configuration
    enable_compression: bool = True
    enable_parallel_processing: bool = True
    enable_memory_mapping: bool = True
    enable_adaptive_chunking: bool = True
    
    def __post_init__(self):
        """Post-initialization validation"""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if self.memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb must be positive")
        
        # Create cache directory if it doesn't exist
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def get_optimal_chunk_size(self, file_size_mb: float) -> int:
        """Calculate optimal chunk size based on file size and memory limits"""
        if not self.enable_adaptive_chunking:
            return self.chunk_size
        
        # Adaptive chunking based on file size and available memory
        target_memory_usage = min(self.memory_limit_mb * 0.8, file_size_mb * 0.1)
        optimal_chunk = int(target_memory_usage * 1024 * 1024 / 8)  # Assuming 8 bytes per value
        
        # Ensure minimum and maximum bounds
        return max(1000, min(optimal_chunk, self.chunk_size * 10))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'chunk_size': self.chunk_size,
            'max_workers': self.max_workers,
            'memory_limit_mb': self.memory_limit_mb,
            'supported_formats': self.supported_formats,
            'compression_types': self.compression_types,
            'stream_buffer_size': self.stream_buffer_size,
            'stream_timeout': self.stream_timeout,
            'cache_dir': self.cache_dir,
            'cache_max_size_gb': self.cache_max_size_gb,
            'cache_ttl_hours': self.cache_ttl_hours,
            'validation_sample_size': self.validation_sample_size,
            'validation_threshold': self.validation_threshold,
            'performance_monitoring': self.performance_monitoring,
            'performance_log_interval': self.performance_log_interval,
            'memory_check_interval': self.memory_check_interval,
            'db_connection_pool_size': self.db_connection_pool_size,
            'db_timeout': self.db_timeout,
            'enable_compression': self.enable_compression,
            'enable_parallel_processing': self.enable_parallel_processing,
            'enable_memory_mapping': self.enable_memory_mapping,
            'enable_adaptive_chunking': self.enable_adaptive_chunking
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataPipelineConfig':
        """Create configuration from dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def for_production(cls) -> 'DataPipelineConfig':
        """Create production-optimized configuration"""
        return cls(
            chunk_size=50000,
            max_workers=min(32, os.cpu_count() or 4),
            memory_limit_mb=8192,
            stream_buffer_size=5000,
            cache_max_size_gb=50.0,
            cache_ttl_hours=48,
            validation_sample_size=10000,
            enable_compression=True,
            enable_parallel_processing=True,
            enable_memory_mapping=True,
            enable_adaptive_chunking=True
        )
    
    @classmethod
    def for_development(cls) -> 'DataPipelineConfig':
        """Create development-friendly configuration"""
        return cls(
            chunk_size=1000,
            max_workers=2,
            memory_limit_mb=512,
            stream_buffer_size=100,
            cache_max_size_gb=1.0,
            cache_ttl_hours=2,
            validation_sample_size=100,
            enable_compression=False,
            enable_parallel_processing=False,
            enable_memory_mapping=False,
            enable_adaptive_chunking=False
        )