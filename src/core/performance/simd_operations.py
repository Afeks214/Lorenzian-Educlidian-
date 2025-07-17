"""
SIMD-Optimized Mathematical Operations - Vectorized computations for maximum throughput.
Implements SIMD (Single Instruction, Multiple Data) optimizations for mathematical operations.
"""

import numpy as np
import torch
import threading
import time
from typing import Union, Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import structlog

try:
    import numba
    from numba import jit, vectorize, cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger = structlog.get_logger(__name__)
    logger.warning("Numba not available, SIMD optimizations will be limited")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

logger = structlog.get_logger(__name__)


class SIMDBackend(Enum):
    """SIMD backends."""
    NUMPY = "numpy"
    NUMBA = "numba"
    TORCH = "torch"
    CUPY = "cupy"
    CUDA = "cuda"


@dataclass
class SIMDMetrics:
    """Metrics for SIMD operations."""
    operations_count: int = 0
    vectorized_operations: int = 0
    total_elements_processed: int = 0
    total_time_ms: float = 0.0
    average_throughput_ops_per_sec: float = 0.0
    backend_usage: Dict[str, int] = None
    
    def __post_init__(self):
        if self.backend_usage is None:
            self.backend_usage = {}


class SIMDConfig:
    """Configuration for SIMD operations."""
    
    def __init__(self):
        self.enabled = True
        self.preferred_backend = SIMDBackend.NUMPY
        self.auto_select_backend = True
        self.min_elements_for_vectorization = 1024
        self.use_gpu_acceleration = torch.cuda.is_available()
        self.numba_parallel = True
        self.numba_cache = True
        
        # Backend availability
        self.backends_available = {
            SIMDBackend.NUMPY: True,
            SIMDBackend.NUMBA: NUMBA_AVAILABLE,
            SIMDBackend.TORCH: True,
            SIMDBackend.CUPY: CUPY_AVAILABLE,
            SIMDBackend.CUDA: torch.cuda.is_available()
        }
        
        logger.info("SIMD configuration initialized", 
                   backends_available=self.backends_available,
                   preferred_backend=self.preferred_backend.value)


class SIMDMath:
    """
    SIMD-optimized mathematical operations.
    Provides vectorized implementations of common mathematical functions.
    """
    
    def __init__(self, config: Optional[SIMDConfig] = None):
        self.config = config or SIMDConfig()
        self._metrics = SIMDMetrics()
        self._lock = threading.RLock()
        
        # Compiled functions cache
        self._compiled_functions = {}
        
        # Initialize SIMD functions
        self._init_simd_functions()
        
        logger.info("SIMD math initialized", config=self.config.__dict__)
    
    def _init_simd_functions(self):
        """Initialize SIMD-optimized functions."""
        if NUMBA_AVAILABLE:
            self._init_numba_functions()
        
        logger.debug("SIMD functions initialized")
    
    def _init_numba_functions(self):
        """Initialize Numba-compiled functions."""
        if not NUMBA_AVAILABLE:
            return
        
        # Vectorized addition
        @vectorize(['float32(float32, float32)', 'float64(float64, float64)'], 
                  target='parallel' if self.config.numba_parallel else 'cpu',
                  cache=self.config.numba_cache)
        def numba_add(a, b):
            return a + b
        
        # Vectorized multiplication
        @vectorize(['float32(float32, float32)', 'float64(float64, float64)'], 
                  target='parallel' if self.config.numba_parallel else 'cpu',
                  cache=self.config.numba_cache)
        def numba_mul(a, b):
            return a * b
        
        # Vectorized division
        @vectorize(['float32(float32, float32)', 'float64(float64, float64)'], 
                  target='parallel' if self.config.numba_parallel else 'cpu',
                  cache=self.config.numba_cache)
        def numba_div(a, b):
            return a / b
        
        # Vectorized square root
        @vectorize(['float32(float32)', 'float64(float64)'], 
                  target='parallel' if self.config.numba_parallel else 'cpu',
                  cache=self.config.numba_cache)
        def numba_sqrt(a):
            return np.sqrt(a)
        
        # Vectorized exponential
        @vectorize(['float32(float32)', 'float64(float64)'], 
                  target='parallel' if self.config.numba_parallel else 'cpu',
                  cache=self.config.numba_cache)
        def numba_exp(a):
            return np.exp(a)
        
        # Vectorized logarithm
        @vectorize(['float32(float32)', 'float64(float64)'], 
                  target='parallel' if self.config.numba_parallel else 'cpu',
                  cache=self.config.numba_cache)
        def numba_log(a):
            return np.log(a)
        
        # Vectorized tanh
        @vectorize(['float32(float32)', 'float64(float64)'], 
                  target='parallel' if self.config.numba_parallel else 'cpu',
                  cache=self.config.numba_cache)
        def numba_tanh(a):
            return np.tanh(a)
        
        # Vectorized sigmoid
        @vectorize(['float32(float32)', 'float64(float64)'], 
                  target='parallel' if self.config.numba_parallel else 'cpu',
                  cache=self.config.numba_cache)
        def numba_sigmoid(a):
            return 1.0 / (1.0 + np.exp(-a))
        
        # Vectorized ReLU
        @vectorize(['float32(float32)', 'float64(float64)'], 
                  target='parallel' if self.config.numba_parallel else 'cpu',
                  cache=self.config.numba_cache)
        def numba_relu(a):
            return max(0.0, a)
        
        # Store compiled functions
        self._compiled_functions.update({
            'numba_add': numba_add,
            'numba_mul': numba_mul,
            'numba_div': numba_div,
            'numba_sqrt': numba_sqrt,
            'numba_exp': numba_exp,
            'numba_log': numba_log,
            'numba_tanh': numba_tanh,
            'numba_sigmoid': numba_sigmoid,
            'numba_relu': numba_relu
        })
    
    def _select_backend(self, data: Union[np.ndarray, torch.Tensor]) -> SIMDBackend:
        """Select optimal backend for data."""
        if not self.config.auto_select_backend:
            return self.config.preferred_backend
        
        # Check data type and size
        if isinstance(data, torch.Tensor):
            if data.is_cuda and self.config.backends_available[SIMDBackend.CUDA]:
                return SIMDBackend.CUDA
            elif self.config.backends_available[SIMDBackend.TORCH]:
                return SIMDBackend.TORCH
        
        if isinstance(data, np.ndarray):
            if data.size >= self.config.min_elements_for_vectorization:
                if CUPY_AVAILABLE and self.config.use_gpu_acceleration:
                    return SIMDBackend.CUPY
                elif NUMBA_AVAILABLE:
                    return SIMDBackend.NUMBA
            
            return SIMDBackend.NUMPY
        
        return self.config.preferred_backend
    
    def _to_backend_array(self, data: Union[np.ndarray, torch.Tensor], backend: SIMDBackend) -> Any:
        """Convert data to backend-specific format."""
        if backend == SIMDBackend.NUMPY:
            if isinstance(data, torch.Tensor):
                return data.detach().cpu().numpy()
            return data
        
        elif backend == SIMDBackend.TORCH:
            if isinstance(data, np.ndarray):
                return torch.from_numpy(data)
            return data
        
        elif backend == SIMDBackend.CUPY and CUPY_AVAILABLE:
            if isinstance(data, torch.Tensor):
                return cp.asarray(data.detach().cpu().numpy())
            elif isinstance(data, np.ndarray):
                return cp.asarray(data)
            return data
        
        elif backend == SIMDBackend.CUDA:
            if isinstance(data, torch.Tensor):
                return data.cuda() if not data.is_cuda else data
            elif isinstance(data, np.ndarray):
                return torch.from_numpy(data).cuda()
        
        return data
    
    def _from_backend_array(self, data: Any, backend: SIMDBackend, target_type: type) -> Union[np.ndarray, torch.Tensor]:
        """Convert data from backend-specific format."""
        if target_type == torch.Tensor:
            if backend == SIMDBackend.NUMPY:
                return torch.from_numpy(data)
            elif backend == SIMDBackend.CUPY and CUPY_AVAILABLE:
                return torch.from_numpy(cp.asnumpy(data))
            return data
        
        elif target_type == np.ndarray:
            if backend == SIMDBackend.TORCH:
                return data.detach().cpu().numpy()
            elif backend == SIMDBackend.CUDA:
                return data.detach().cpu().numpy()
            elif backend == SIMDBackend.CUPY and CUPY_AVAILABLE:
                return cp.asnumpy(data)
            return data
        
        return data
    
    def _track_operation(self, operation_name: str, backend: SIMDBackend, elements: int, time_ms: float):
        """Track operation metrics."""
        with self._lock:
            self._metrics.operations_count += 1
            self._metrics.vectorized_operations += 1
            self._metrics.total_elements_processed += elements
            self._metrics.total_time_ms += time_ms
            
            backend_name = backend.value
            if backend_name not in self._metrics.backend_usage:
                self._metrics.backend_usage[backend_name] = 0
            self._metrics.backend_usage[backend_name] += 1
            
            # Update average throughput
            if self._metrics.total_time_ms > 0:
                self._metrics.average_throughput_ops_per_sec = (
                    self._metrics.operations_count / (self._metrics.total_time_ms / 1000.0)
                )
    
    def add(self, a: Union[np.ndarray, torch.Tensor], b: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """SIMD-optimized addition."""
        if not self.config.enabled:
            return a + b
        
        start_time = time.time()
        target_type = type(a)
        backend = self._select_backend(a)
        
        # Convert to backend format
        a_backend = self._to_backend_array(a, backend)
        b_backend = self._to_backend_array(b, backend)
        
        # Perform operation
        if backend == SIMDBackend.NUMBA and 'numba_add' in self._compiled_functions:
            result = self._compiled_functions['numba_add'](a_backend, b_backend)
        elif backend == SIMDBackend.CUPY and CUPY_AVAILABLE:
            result = a_backend + b_backend
        elif backend == SIMDBackend.TORCH:
            result = torch.add(a_backend, b_backend)
        elif backend == SIMDBackend.CUDA:
            result = torch.add(a_backend, b_backend)
        else:
            result = np.add(a_backend, b_backend)
        
        # Convert back to target type
        result = self._from_backend_array(result, backend, target_type)
        
        # Track metrics
        end_time = time.time()
        elements = a.numel() if isinstance(a, torch.Tensor) else a.size
        self._track_operation('add', backend, elements, (end_time - start_time) * 1000)
        
        return result
    
    def multiply(self, a: Union[np.ndarray, torch.Tensor], b: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """SIMD-optimized multiplication."""
        if not self.config.enabled:
            return a * b
        
        start_time = time.time()
        target_type = type(a)
        backend = self._select_backend(a)
        
        # Convert to backend format
        a_backend = self._to_backend_array(a, backend)
        b_backend = self._to_backend_array(b, backend)
        
        # Perform operation
        if backend == SIMDBackend.NUMBA and 'numba_mul' in self._compiled_functions:
            result = self._compiled_functions['numba_mul'](a_backend, b_backend)
        elif backend == SIMDBackend.CUPY and CUPY_AVAILABLE:
            result = a_backend * b_backend
        elif backend == SIMDBackend.TORCH:
            result = torch.mul(a_backend, b_backend)
        elif backend == SIMDBackend.CUDA:
            result = torch.mul(a_backend, b_backend)
        else:
            result = np.multiply(a_backend, b_backend)
        
        # Convert back to target type
        result = self._from_backend_array(result, backend, target_type)
        
        # Track metrics
        end_time = time.time()
        elements = a.numel() if isinstance(a, torch.Tensor) else a.size
        self._track_operation('multiply', backend, elements, (end_time - start_time) * 1000)
        
        return result
    
    def divide(self, a: Union[np.ndarray, torch.Tensor], b: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """SIMD-optimized division."""
        if not self.config.enabled:
            return a / b
        
        start_time = time.time()
        target_type = type(a)
        backend = self._select_backend(a)
        
        # Convert to backend format
        a_backend = self._to_backend_array(a, backend)
        b_backend = self._to_backend_array(b, backend)
        
        # Perform operation
        if backend == SIMDBackend.NUMBA and 'numba_div' in self._compiled_functions:
            result = self._compiled_functions['numba_div'](a_backend, b_backend)
        elif backend == SIMDBackend.CUPY and CUPY_AVAILABLE:
            result = a_backend / b_backend
        elif backend == SIMDBackend.TORCH:
            result = torch.div(a_backend, b_backend)
        elif backend == SIMDBackend.CUDA:
            result = torch.div(a_backend, b_backend)
        else:
            result = np.divide(a_backend, b_backend)
        
        # Convert back to target type
        result = self._from_backend_array(result, backend, target_type)
        
        # Track metrics
        end_time = time.time()
        elements = a.numel() if isinstance(a, torch.Tensor) else a.size
        self._track_operation('divide', backend, elements, (end_time - start_time) * 1000)
        
        return result
    
    def sqrt(self, a: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """SIMD-optimized square root."""
        if not self.config.enabled:
            return np.sqrt(a) if isinstance(a, np.ndarray) else torch.sqrt(a)
        
        start_time = time.time()
        target_type = type(a)
        backend = self._select_backend(a)
        
        # Convert to backend format
        a_backend = self._to_backend_array(a, backend)
        
        # Perform operation
        if backend == SIMDBackend.NUMBA and 'numba_sqrt' in self._compiled_functions:
            result = self._compiled_functions['numba_sqrt'](a_backend)
        elif backend == SIMDBackend.CUPY and CUPY_AVAILABLE:
            result = cp.sqrt(a_backend)
        elif backend == SIMDBackend.TORCH:
            result = torch.sqrt(a_backend)
        elif backend == SIMDBackend.CUDA:
            result = torch.sqrt(a_backend)
        else:
            result = np.sqrt(a_backend)
        
        # Convert back to target type
        result = self._from_backend_array(result, backend, target_type)
        
        # Track metrics
        end_time = time.time()
        elements = a.numel() if isinstance(a, torch.Tensor) else a.size
        self._track_operation('sqrt', backend, elements, (end_time - start_time) * 1000)
        
        return result
    
    def exp(self, a: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """SIMD-optimized exponential."""
        if not self.config.enabled:
            return np.exp(a) if isinstance(a, np.ndarray) else torch.exp(a)
        
        start_time = time.time()
        target_type = type(a)
        backend = self._select_backend(a)
        
        # Convert to backend format
        a_backend = self._to_backend_array(a, backend)
        
        # Perform operation
        if backend == SIMDBackend.NUMBA and 'numba_exp' in self._compiled_functions:
            result = self._compiled_functions['numba_exp'](a_backend)
        elif backend == SIMDBackend.CUPY and CUPY_AVAILABLE:
            result = cp.exp(a_backend)
        elif backend == SIMDBackend.TORCH:
            result = torch.exp(a_backend)
        elif backend == SIMDBackend.CUDA:
            result = torch.exp(a_backend)
        else:
            result = np.exp(a_backend)
        
        # Convert back to target type
        result = self._from_backend_array(result, backend, target_type)
        
        # Track metrics
        end_time = time.time()
        elements = a.numel() if isinstance(a, torch.Tensor) else a.size
        self._track_operation('exp', backend, elements, (end_time - start_time) * 1000)
        
        return result
    
    def log(self, a: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """SIMD-optimized logarithm."""
        if not self.config.enabled:
            return np.log(a) if isinstance(a, np.ndarray) else torch.log(a)
        
        start_time = time.time()
        target_type = type(a)
        backend = self._select_backend(a)
        
        # Convert to backend format
        a_backend = self._to_backend_array(a, backend)
        
        # Perform operation
        if backend == SIMDBackend.NUMBA and 'numba_log' in self._compiled_functions:
            result = self._compiled_functions['numba_log'](a_backend)
        elif backend == SIMDBackend.CUPY and CUPY_AVAILABLE:
            result = cp.log(a_backend)
        elif backend == SIMDBackend.TORCH:
            result = torch.log(a_backend)
        elif backend == SIMDBackend.CUDA:
            result = torch.log(a_backend)
        else:
            result = np.log(a_backend)
        
        # Convert back to target type
        result = self._from_backend_array(result, backend, target_type)
        
        # Track metrics
        end_time = time.time()
        elements = a.numel() if isinstance(a, torch.Tensor) else a.size
        self._track_operation('log', backend, elements, (end_time - start_time) * 1000)
        
        return result
    
    def tanh(self, a: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """SIMD-optimized hyperbolic tangent."""
        if not self.config.enabled:
            return np.tanh(a) if isinstance(a, np.ndarray) else torch.tanh(a)
        
        start_time = time.time()
        target_type = type(a)
        backend = self._select_backend(a)
        
        # Convert to backend format
        a_backend = self._to_backend_array(a, backend)
        
        # Perform operation
        if backend == SIMDBackend.NUMBA and 'numba_tanh' in self._compiled_functions:
            result = self._compiled_functions['numba_tanh'](a_backend)
        elif backend == SIMDBackend.CUPY and CUPY_AVAILABLE:
            result = cp.tanh(a_backend)
        elif backend == SIMDBackend.TORCH:
            result = torch.tanh(a_backend)
        elif backend == SIMDBackend.CUDA:
            result = torch.tanh(a_backend)
        else:
            result = np.tanh(a_backend)
        
        # Convert back to target type
        result = self._from_backend_array(result, backend, target_type)
        
        # Track metrics
        end_time = time.time()
        elements = a.numel() if isinstance(a, torch.Tensor) else a.size
        self._track_operation('tanh', backend, elements, (end_time - start_time) * 1000)
        
        return result
    
    def sigmoid(self, a: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """SIMD-optimized sigmoid."""
        if not self.config.enabled:
            if isinstance(a, np.ndarray):
                return 1.0 / (1.0 + np.exp(-a))
            else:
                return torch.sigmoid(a)
        
        start_time = time.time()
        target_type = type(a)
        backend = self._select_backend(a)
        
        # Convert to backend format
        a_backend = self._to_backend_array(a, backend)
        
        # Perform operation
        if backend == SIMDBackend.NUMBA and 'numba_sigmoid' in self._compiled_functions:
            result = self._compiled_functions['numba_sigmoid'](a_backend)
        elif backend == SIMDBackend.CUPY and CUPY_AVAILABLE:
            result = 1.0 / (1.0 + cp.exp(-a_backend))
        elif backend == SIMDBackend.TORCH:
            result = torch.sigmoid(a_backend)
        elif backend == SIMDBackend.CUDA:
            result = torch.sigmoid(a_backend)
        else:
            result = 1.0 / (1.0 + np.exp(-a_backend))
        
        # Convert back to target type
        result = self._from_backend_array(result, backend, target_type)
        
        # Track metrics
        end_time = time.time()
        elements = a.numel() if isinstance(a, torch.Tensor) else a.size
        self._track_operation('sigmoid', backend, elements, (end_time - start_time) * 1000)
        
        return result
    
    def relu(self, a: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """SIMD-optimized ReLU."""
        if not self.config.enabled:
            if isinstance(a, np.ndarray):
                return np.maximum(0, a)
            else:
                return torch.relu(a)
        
        start_time = time.time()
        target_type = type(a)
        backend = self._select_backend(a)
        
        # Convert to backend format
        a_backend = self._to_backend_array(a, backend)
        
        # Perform operation
        if backend == SIMDBackend.NUMBA and 'numba_relu' in self._compiled_functions:
            result = self._compiled_functions['numba_relu'](a_backend)
        elif backend == SIMDBackend.CUPY and CUPY_AVAILABLE:
            result = cp.maximum(0, a_backend)
        elif backend == SIMDBackend.TORCH:
            result = torch.relu(a_backend)
        elif backend == SIMDBackend.CUDA:
            result = torch.relu(a_backend)
        else:
            result = np.maximum(0, a_backend)
        
        # Convert back to target type
        result = self._from_backend_array(result, backend, target_type)
        
        # Track metrics
        end_time = time.time()
        elements = a.numel() if isinstance(a, torch.Tensor) else a.size
        self._track_operation('relu', backend, elements, (end_time - start_time) * 1000)
        
        return result
    
    def sum(self, a: Union[np.ndarray, torch.Tensor], axis: Optional[int] = None) -> Union[np.ndarray, torch.Tensor]:
        """SIMD-optimized sum."""
        if not self.config.enabled:
            return np.sum(a, axis=axis) if isinstance(a, np.ndarray) else torch.sum(a, dim=axis)
        
        start_time = time.time()
        target_type = type(a)
        backend = self._select_backend(a)
        
        # Convert to backend format
        a_backend = self._to_backend_array(a, backend)
        
        # Perform operation
        if backend == SIMDBackend.CUPY and CUPY_AVAILABLE:
            result = cp.sum(a_backend, axis=axis)
        elif backend == SIMDBackend.TORCH:
            result = torch.sum(a_backend, dim=axis)
        elif backend == SIMDBackend.CUDA:
            result = torch.sum(a_backend, dim=axis)
        else:
            result = np.sum(a_backend, axis=axis)
        
        # Convert back to target type
        result = self._from_backend_array(result, backend, target_type)
        
        # Track metrics
        end_time = time.time()
        elements = a.numel() if isinstance(a, torch.Tensor) else a.size
        self._track_operation('sum', backend, elements, (end_time - start_time) * 1000)
        
        return result
    
    def mean(self, a: Union[np.ndarray, torch.Tensor], axis: Optional[int] = None) -> Union[np.ndarray, torch.Tensor]:
        """SIMD-optimized mean."""
        if not self.config.enabled:
            return np.mean(a, axis=axis) if isinstance(a, np.ndarray) else torch.mean(a, dim=axis)
        
        start_time = time.time()
        target_type = type(a)
        backend = self._select_backend(a)
        
        # Convert to backend format
        a_backend = self._to_backend_array(a, backend)
        
        # Perform operation
        if backend == SIMDBackend.CUPY and CUPY_AVAILABLE:
            result = cp.mean(a_backend, axis=axis)
        elif backend == SIMDBackend.TORCH:
            result = torch.mean(a_backend, dim=axis)
        elif backend == SIMDBackend.CUDA:
            result = torch.mean(a_backend, dim=axis)
        else:
            result = np.mean(a_backend, axis=axis)
        
        # Convert back to target type
        result = self._from_backend_array(result, backend, target_type)
        
        # Track metrics
        end_time = time.time()
        elements = a.numel() if isinstance(a, torch.Tensor) else a.size
        self._track_operation('mean', backend, elements, (end_time - start_time) * 1000)
        
        return result
    
    def std(self, a: Union[np.ndarray, torch.Tensor], axis: Optional[int] = None) -> Union[np.ndarray, torch.Tensor]:
        """SIMD-optimized standard deviation."""
        if not self.config.enabled:
            return np.std(a, axis=axis) if isinstance(a, np.ndarray) else torch.std(a, dim=axis)
        
        start_time = time.time()
        target_type = type(a)
        backend = self._select_backend(a)
        
        # Convert to backend format
        a_backend = self._to_backend_array(a, backend)
        
        # Perform operation
        if backend == SIMDBackend.CUPY and CUPY_AVAILABLE:
            result = cp.std(a_backend, axis=axis)
        elif backend == SIMDBackend.TORCH:
            result = torch.std(a_backend, dim=axis)
        elif backend == SIMDBackend.CUDA:
            result = torch.std(a_backend, dim=axis)
        else:
            result = np.std(a_backend, axis=axis)
        
        # Convert back to target type
        result = self._from_backend_array(result, backend, target_type)
        
        # Track metrics
        end_time = time.time()
        elements = a.numel() if isinstance(a, torch.Tensor) else a.size
        self._track_operation('std', backend, elements, (end_time - start_time) * 1000)
        
        return result
    
    def max(self, a: Union[np.ndarray, torch.Tensor], axis: Optional[int] = None) -> Union[np.ndarray, torch.Tensor]:
        """SIMD-optimized max."""
        if not self.config.enabled:
            return np.max(a, axis=axis) if isinstance(a, np.ndarray) else torch.max(a, dim=axis)
        
        start_time = time.time()
        target_type = type(a)
        backend = self._select_backend(a)
        
        # Convert to backend format
        a_backend = self._to_backend_array(a, backend)
        
        # Perform operation
        if backend == SIMDBackend.CUPY and CUPY_AVAILABLE:
            result = cp.max(a_backend, axis=axis)
        elif backend == SIMDBackend.TORCH:
            if axis is None:
                result = torch.max(a_backend)
            else:
                result = torch.max(a_backend, dim=axis)[0]  # Return values only
        elif backend == SIMDBackend.CUDA:
            if axis is None:
                result = torch.max(a_backend)
            else:
                result = torch.max(a_backend, dim=axis)[0]  # Return values only
        else:
            result = np.max(a_backend, axis=axis)
        
        # Convert back to target type
        result = self._from_backend_array(result, backend, target_type)
        
        # Track metrics
        end_time = time.time()
        elements = a.numel() if isinstance(a, torch.Tensor) else a.size
        self._track_operation('max', backend, elements, (end_time - start_time) * 1000)
        
        return result
    
    def min(self, a: Union[np.ndarray, torch.Tensor], axis: Optional[int] = None) -> Union[np.ndarray, torch.Tensor]:
        """SIMD-optimized min."""
        if not self.config.enabled:
            return np.min(a, axis=axis) if isinstance(a, np.ndarray) else torch.min(a, dim=axis)
        
        start_time = time.time()
        target_type = type(a)
        backend = self._select_backend(a)
        
        # Convert to backend format
        a_backend = self._to_backend_array(a, backend)
        
        # Perform operation
        if backend == SIMDBackend.CUPY and CUPY_AVAILABLE:
            result = cp.min(a_backend, axis=axis)
        elif backend == SIMDBackend.TORCH:
            if axis is None:
                result = torch.min(a_backend)
            else:
                result = torch.min(a_backend, dim=axis)[0]  # Return values only
        elif backend == SIMDBackend.CUDA:
            if axis is None:
                result = torch.min(a_backend)
            else:
                result = torch.min(a_backend, dim=axis)[0]  # Return values only
        else:
            result = np.min(a_backend, axis=axis)
        
        # Convert back to target type
        result = self._from_backend_array(result, backend, target_type)
        
        # Track metrics
        end_time = time.time()
        elements = a.numel() if isinstance(a, torch.Tensor) else a.size
        self._track_operation('min', backend, elements, (end_time - start_time) * 1000)
        
        return result
    
    def softmax(self, a: Union[np.ndarray, torch.Tensor], axis: int = -1) -> Union[np.ndarray, torch.Tensor]:
        """SIMD-optimized softmax."""
        if not self.config.enabled:
            if isinstance(a, np.ndarray):
                exp_a = np.exp(a - np.max(a, axis=axis, keepdims=True))
                return exp_a / np.sum(exp_a, axis=axis, keepdims=True)
            else:
                return torch.softmax(a, dim=axis)
        
        start_time = time.time()
        target_type = type(a)
        backend = self._select_backend(a)
        
        # Convert to backend format
        a_backend = self._to_backend_array(a, backend)
        
        # Perform operation
        if backend == SIMDBackend.CUPY and CUPY_AVAILABLE:
            exp_a = cp.exp(a_backend - cp.max(a_backend, axis=axis, keepdims=True))
            result = exp_a / cp.sum(exp_a, axis=axis, keepdims=True)
        elif backend == SIMDBackend.TORCH:
            result = torch.softmax(a_backend, dim=axis)
        elif backend == SIMDBackend.CUDA:
            result = torch.softmax(a_backend, dim=axis)
        else:
            exp_a = np.exp(a_backend - np.max(a_backend, axis=axis, keepdims=True))
            result = exp_a / np.sum(exp_a, axis=axis, keepdims=True)
        
        # Convert back to target type
        result = self._from_backend_array(result, backend, target_type)
        
        # Track metrics
        end_time = time.time()
        elements = a.numel() if isinstance(a, torch.Tensor) else a.size
        self._track_operation('softmax', backend, elements, (end_time - start_time) * 1000)
        
        return result
    
    def matmul(self, a: Union[np.ndarray, torch.Tensor], b: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """SIMD-optimized matrix multiplication."""
        if not self.config.enabled:
            return np.matmul(a, b) if isinstance(a, np.ndarray) else torch.matmul(a, b)
        
        start_time = time.time()
        target_type = type(a)
        backend = self._select_backend(a)
        
        # Convert to backend format
        a_backend = self._to_backend_array(a, backend)
        b_backend = self._to_backend_array(b, backend)
        
        # Perform operation
        if backend == SIMDBackend.CUPY and CUPY_AVAILABLE:
            result = cp.matmul(a_backend, b_backend)
        elif backend == SIMDBackend.TORCH:
            result = torch.matmul(a_backend, b_backend)
        elif backend == SIMDBackend.CUDA:
            result = torch.matmul(a_backend, b_backend)
        else:
            result = np.matmul(a_backend, b_backend)
        
        # Convert back to target type
        result = self._from_backend_array(result, backend, target_type)
        
        # Track metrics
        end_time = time.time()
        elements = a.numel() if isinstance(a, torch.Tensor) else a.size
        self._track_operation('matmul', backend, elements, (end_time - start_time) * 1000)
        
        return result
    
    def get_metrics(self) -> SIMDMetrics:
        """Get SIMD metrics."""
        with self._lock:
            return self._metrics
    
    def reset_metrics(self):
        """Reset SIMD metrics."""
        with self._lock:
            self._metrics = SIMDMetrics()
    
    def get_config(self) -> SIMDConfig:
        """Get SIMD configuration."""
        return self.config
    
    def set_config(self, config: SIMDConfig):
        """Set SIMD configuration."""
        self.config = config
        # Re-initialize functions if needed
        if NUMBA_AVAILABLE:
            self._init_numba_functions()


# Global SIMD math instance
_global_simd_math: Optional[SIMDMath] = None


def get_simd_math() -> SIMDMath:
    """Get the global SIMD math instance."""
    global _global_simd_math
    if _global_simd_math is None:
        _global_simd_math = SIMDMath()
    return _global_simd_math


def set_simd_math(simd_math: SIMDMath):
    """Set the global SIMD math instance."""
    global _global_simd_math
    _global_simd_math = simd_math


# Convenience functions
def vectorized_add(a: Union[np.ndarray, torch.Tensor], b: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Vectorized addition."""
    return get_simd_math().add(a, b)


def vectorized_mul(a: Union[np.ndarray, torch.Tensor], b: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Vectorized multiplication."""
    return get_simd_math().multiply(a, b)


def vectorized_sum(a: Union[np.ndarray, torch.Tensor], axis: Optional[int] = None) -> Union[np.ndarray, torch.Tensor]:
    """Vectorized sum."""
    return get_simd_math().sum(a, axis)


def vectorized_mean(a: Union[np.ndarray, torch.Tensor], axis: Optional[int] = None) -> Union[np.ndarray, torch.Tensor]:
    """Vectorized mean."""
    return get_simd_math().mean(a, axis)


def vectorized_std(a: Union[np.ndarray, torch.Tensor], axis: Optional[int] = None) -> Union[np.ndarray, torch.Tensor]:
    """Vectorized standard deviation."""
    return get_simd_math().std(a, axis)


def vectorized_max(a: Union[np.ndarray, torch.Tensor], axis: Optional[int] = None) -> Union[np.ndarray, torch.Tensor]:
    """Vectorized max."""
    return get_simd_math().max(a, axis)


def vectorized_min(a: Union[np.ndarray, torch.Tensor], axis: Optional[int] = None) -> Union[np.ndarray, torch.Tensor]:
    """Vectorized min."""
    return get_simd_math().min(a, axis)


def vectorized_softmax(a: Union[np.ndarray, torch.Tensor], axis: int = -1) -> Union[np.ndarray, torch.Tensor]:
    """Vectorized softmax."""
    return get_simd_math().softmax(a, axis)


def vectorized_relu(a: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Vectorized ReLU."""
    return get_simd_math().relu(a)


def vectorized_tanh(a: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Vectorized tanh."""
    return get_simd_math().tanh(a)


def vectorized_sigmoid(a: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Vectorized sigmoid."""
    return get_simd_math().sigmoid(a)


def vectorized_matmul(a: Union[np.ndarray, torch.Tensor], b: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Vectorized matrix multiplication."""
    return get_simd_math().matmul(a, b)


def enable_simd_optimizations():
    """Enable SIMD optimizations globally."""
    simd_math = get_simd_math()
    simd_math.config.enabled = True
    logger.info("SIMD optimizations enabled")


def disable_simd_optimizations():
    """Disable SIMD optimizations globally."""
    simd_math = get_simd_math()
    simd_math.config.enabled = False
    logger.info("SIMD optimizations disabled")


def get_simd_metrics() -> SIMDMetrics:
    """Get global SIMD metrics."""
    return get_simd_math().get_metrics()


def reset_simd_metrics():
    """Reset global SIMD metrics."""
    get_simd_math().reset_metrics()


def benchmark_simd_operation(operation_name: str, operation_func, *args, **kwargs) -> Dict[str, Any]:
    """Benchmark a SIMD operation."""
    import time
    
    # Run with SIMD enabled
    enable_simd_optimizations()
    start_time = time.time()
    simd_result = operation_func(*args, **kwargs)
    simd_time = time.time() - start_time
    
    # Run with SIMD disabled
    disable_simd_optimizations()
    start_time = time.time()
    standard_result = operation_func(*args, **kwargs)
    standard_time = time.time() - start_time
    
    # Re-enable SIMD
    enable_simd_optimizations()
    
    # Calculate speedup
    speedup = standard_time / simd_time if simd_time > 0 else 0
    
    return {
        'operation': operation_name,
        'simd_time_ms': simd_time * 1000,
        'standard_time_ms': standard_time * 1000,
        'speedup': speedup,
        'simd_enabled': True
    }