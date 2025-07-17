"""
Data Flow Coordinator for NQ Data Pipeline

Manages data synchronization, concurrent processing, and consistency
between execution engine and risk management notebooks.
"""

import threading
import time
import queue
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import numpy as np
from enum import Enum
import json
import pickle
import hashlib
from datetime import datetime, timedelta

class DataStreamType(Enum):
    """Types of data streams"""
    MARKET_DATA = "market_data"
    FEATURES = "features"
    PREDICTIONS = "predictions"
    RISK_METRICS = "risk_metrics"
    PERFORMANCE = "performance"

class DataStreamStatus(Enum):
    """Status of data streams"""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class DataMessage:
    """Data message for inter-notebook communication"""
    stream_id: str
    stream_type: DataStreamType
    data: Any
    timestamp: float
    sequence_number: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'stream_id': self.stream_id,
            'stream_type': self.stream_type.value,
            'data': self.data,
            'timestamp': self.timestamp,
            'sequence_number': self.sequence_number,
            'metadata': self.metadata
        }

class DataStream:
    """Represents a data stream between notebooks"""
    
    def __init__(self, 
                 stream_id: str,
                 stream_type: DataStreamType,
                 buffer_size: int = 1000):
        self.stream_id = stream_id
        self.stream_type = stream_type
        self.buffer_size = buffer_size
        
        # Stream state
        self.status = DataStreamStatus.ACTIVE
        self.sequence_number = 0
        
        # Data buffer
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.subscribers = []
        
        # Statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.last_message_time = None
        self.start_time = time.time()
        
        # Synchronization
        self.lock = threading.RLock()
        
        self.logger = logging.getLogger(f"DataStream.{stream_id}")
    
    def publish(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Publish data to stream"""
        with self.lock:
            if self.status != DataStreamStatus.ACTIVE:
                return False
            
            # Create message
            message = DataMessage(
                stream_id=self.stream_id,
                stream_type=self.stream_type,
                data=data,
                timestamp=time.time(),
                sequence_number=self.sequence_number,
                metadata=metadata or {}
            )
            
            try:
                # Add to buffer
                self.buffer.put(message, block=False)
                self.sequence_number += 1
                self.messages_sent += 1
                self.last_message_time = time.time()
                
                # Notify subscribers
                self._notify_subscribers(message)
                
                return True
                
            except queue.Full:
                self.logger.warning(f"Buffer full for stream {self.stream_id}")
                return False
    
    def subscribe(self, callback: Callable[[DataMessage], None]):
        """Subscribe to stream updates"""
        with self.lock:
            self.subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[DataMessage], None]):
        """Unsubscribe from stream updates"""
        with self.lock:
            if callback in self.subscribers:
                self.subscribers.remove(callback)
    
    def _notify_subscribers(self, message: DataMessage):
        """Notify all subscribers of new message"""
        for callback in self.subscribers:
            try:
                callback(message)
            except Exception as e:
                self.logger.error(f"Subscriber callback error: {e}")
    
    def get_messages(self, max_messages: int = 10) -> List[DataMessage]:
        """Get messages from stream"""
        messages = []
        
        try:
            while len(messages) < max_messages:
                message = self.buffer.get(block=False)
                messages.append(message)
                self.messages_received += 1
        except queue.Empty:
            pass
        
        return messages
    
    def pause(self):
        """Pause stream"""
        with self.lock:
            self.status = DataStreamStatus.PAUSED
    
    def resume(self):
        """Resume stream"""
        with self.lock:
            self.status = DataStreamStatus.ACTIVE
    
    def stop(self):
        """Stop stream"""
        with self.lock:
            self.status = DataStreamStatus.STOPPED
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stream statistics"""
        with self.lock:
            uptime = time.time() - self.start_time
            
            return {
                'stream_id': self.stream_id,
                'stream_type': self.stream_type.value,
                'status': self.status.value,
                'messages_sent': self.messages_sent,
                'messages_received': self.messages_received,
                'buffer_size': self.buffer.qsize(),
                'max_buffer_size': self.buffer_size,
                'subscribers': len(self.subscribers),
                'uptime_seconds': uptime,
                'last_message_time': self.last_message_time,
                'message_rate': self.messages_sent / uptime if uptime > 0 else 0
            }

class DataConsistencyChecker:
    """Ensure data consistency across notebooks"""
    
    def __init__(self):
        self.checksums = {}
        self.validation_rules = {}
        self.logger = logging.getLogger(__name__)
    
    def add_validation_rule(self, rule_name: str, rule_func: Callable[[Any], bool]):
        """Add custom validation rule"""
        self.validation_rules[rule_name] = rule_func
    
    def calculate_checksum(self, data: Any) -> str:
        """Calculate checksum for data"""
        if isinstance(data, pd.DataFrame):
            # Use DataFrame hash
            return hashlib.md5(str(data.values.data.tobytes()).encode()).hexdigest()
        elif isinstance(data, np.ndarray):
            # Use numpy array hash
            return hashlib.md5(data.tobytes()).hexdigest()
        else:
            # Use pickle for other types
            return hashlib.md5(pickle.dumps(data)).hexdigest()
    
    def validate_data(self, data: Any, data_id: str) -> Dict[str, Any]:
        """Validate data consistency"""
        result = {
            'data_id': data_id,
            'is_valid': True,
            'checksum': self.calculate_checksum(data),
            'validation_errors': [],
            'validation_warnings': []
        }
        
        # Check against stored checksum
        if data_id in self.checksums:
            if result['checksum'] != self.checksums[data_id]:
                result['is_valid'] = False
                result['validation_errors'].append(f"Checksum mismatch for {data_id}")
        else:
            # Store new checksum
            self.checksums[data_id] = result['checksum']
        
        # Run custom validation rules
        for rule_name, rule_func in self.validation_rules.items():
            try:
                if not rule_func(data):
                    result['is_valid'] = False
                    result['validation_errors'].append(f"Validation rule '{rule_name}' failed")
            except Exception as e:
                result['validation_warnings'].append(f"Validation rule '{rule_name}' error: {e}")
        
        return result
    
    def sync_checksums(self, other_checker: 'DataConsistencyChecker'):
        """Synchronize checksums with another checker"""
        for data_id, checksum in other_checker.checksums.items():
            if data_id not in self.checksums:
                self.checksums[data_id] = checksum
            elif self.checksums[data_id] != checksum:
                self.logger.warning(f"Checksum conflict for {data_id}")

class ConcurrentDataProcessor:
    """Process data concurrently across multiple workers"""
    
    def __init__(self, 
                 max_workers: int = 4,
                 use_processes: bool = False):
        self.max_workers = max_workers
        self.use_processes = use_processes
        
        # Create executor
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.processing_times = []
        self.completed_tasks = 0
        self.failed_tasks = 0
    
    def process_data_parallel(self, 
                            data_chunks: List[Any],
                            processing_func: Callable[[Any], Any],
                            callback: Optional[Callable[[Any], None]] = None) -> List[Any]:
        """Process data chunks in parallel"""
        start_time = time.time()
        
        # Submit tasks
        futures = []
        for i, chunk in enumerate(data_chunks):
            future = self.executor.submit(processing_func, chunk)
            futures.append((i, future))
        
        # Collect results
        results = [None] * len(data_chunks)
        
        for i, future in futures:
            try:
                result = future.result()
                results[i] = result
                self.completed_tasks += 1
                
                if callback:
                    callback(result)
                    
            except Exception as e:
                self.logger.error(f"Task {i} failed: {e}")
                self.failed_tasks += 1
                results[i] = None
        
        # Record performance
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        self.logger.info(f"Processed {len(data_chunks)} chunks in {processing_time:.2f}s")
        
        return results
    
    def process_data_streaming(self, 
                             data_stream: DataStream,
                             processing_func: Callable[[Any], Any],
                             output_stream: DataStream,
                             batch_size: int = 10):
        """Process data from stream continuously"""
        def process_batch():
            while data_stream.status == DataStreamStatus.ACTIVE:
                # Get batch of messages
                messages = data_stream.get_messages(batch_size)
                
                if messages:
                    # Process batch
                    processed_data = []
                    for message in messages:
                        try:
                            result = processing_func(message.data)
                            processed_data.append(result)
                        except Exception as e:
                            self.logger.error(f"Processing error: {e}")
                    
                    # Send to output stream
                    for data in processed_data:
                        output_stream.publish(data)
                
                time.sleep(0.01)  # Small delay to prevent busy waiting
        
        # Start processing thread
        processing_thread = threading.Thread(target=process_batch)
        processing_thread.daemon = True
        processing_thread.start()
        
        return processing_thread
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get processing performance statistics"""
        if not self.processing_times:
            return {'status': 'No processing completed yet'}
        
        return {
            'max_workers': self.max_workers,
            'use_processes': self.use_processes,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'avg_processing_time': np.mean(self.processing_times),
            'total_processing_time': sum(self.processing_times),
            'success_rate': self.completed_tasks / (self.completed_tasks + self.failed_tasks) if (self.completed_tasks + self.failed_tasks) > 0 else 0
        }
    
    def shutdown(self):
        """Shutdown executor"""
        self.executor.shutdown(wait=True)

class DataFlowCoordinator:
    """Central coordinator for data flow between notebooks"""
    
    def __init__(self, 
                 coordination_dir: str = "/tmp/data_flow_coordination",
                 enable_persistence: bool = True):
        """
        Initialize data flow coordinator
        
        Args:
            coordination_dir: Directory for coordination files
            enable_persistence: Enable state persistence
        """
        self.coordination_dir = Path(coordination_dir)
        self.coordination_dir.mkdir(parents=True, exist_ok=True)
        self.enable_persistence = enable_persistence
        
        # Data streams
        self.streams: Dict[str, DataStream] = {}
        
        # Coordination components
        self.consistency_checker = DataConsistencyChecker()
        self.processor = ConcurrentDataProcessor()
        
        # Synchronization
        self.lock = threading.RLock()
        
        # Coordination state
        self.notebook_registry = {}
        self.active_sessions = set()
        
        self.logger = logging.getLogger(__name__)
        
        # Load persisted state
        if self.enable_persistence:
            self._load_state()
        
        self.logger.info("Data flow coordinator initialized")
    
    def register_notebook(self, notebook_id: str, notebook_type: str, capabilities: List[str]):
        """Register a notebook with the coordinator"""
        with self.lock:
            self.notebook_registry[notebook_id] = {
                'type': notebook_type,
                'capabilities': capabilities,
                'registered_at': time.time(),
                'last_activity': time.time(),
                'streams': []
            }
            
            self.active_sessions.add(notebook_id)
            
            if self.enable_persistence:
                self._save_state()
            
            self.logger.info(f"Registered notebook: {notebook_id} ({notebook_type})")
    
    def unregister_notebook(self, notebook_id: str):
        """Unregister a notebook"""
        with self.lock:
            if notebook_id in self.notebook_registry:
                # Clean up streams
                notebook_info = self.notebook_registry[notebook_id]
                for stream_id in notebook_info['streams']:
                    self.remove_stream(stream_id)
                
                del self.notebook_registry[notebook_id]
                self.active_sessions.discard(notebook_id)
                
                if self.enable_persistence:
                    self._save_state()
                
                self.logger.info(f"Unregistered notebook: {notebook_id}")
    
    def create_stream(self, 
                     stream_id: str,
                     stream_type: DataStreamType,
                     producer_notebook: str,
                     consumer_notebooks: List[str],
                     buffer_size: int = 1000) -> DataStream:
        """Create a new data stream"""
        with self.lock:
            if stream_id in self.streams:
                raise ValueError(f"Stream {stream_id} already exists")
            
            # Create stream
            stream = DataStream(stream_id, stream_type, buffer_size)
            self.streams[stream_id] = stream
            
            # Update notebook registry
            if producer_notebook in self.notebook_registry:
                self.notebook_registry[producer_notebook]['streams'].append(stream_id)
            
            for consumer in consumer_notebooks:
                if consumer in self.notebook_registry:
                    self.notebook_registry[consumer]['streams'].append(stream_id)
            
            if self.enable_persistence:
                self._save_state()
            
            self.logger.info(f"Created stream: {stream_id} ({stream_type.value})")
            return stream
    
    def get_stream(self, stream_id: str) -> Optional[DataStream]:
        """Get existing stream"""
        return self.streams.get(stream_id)
    
    def remove_stream(self, stream_id: str):
        """Remove a stream"""
        with self.lock:
            if stream_id in self.streams:
                stream = self.streams[stream_id]
                stream.stop()
                del self.streams[stream_id]
                
                # Update notebook registry
                for notebook_info in self.notebook_registry.values():
                    if stream_id in notebook_info['streams']:
                        notebook_info['streams'].remove(stream_id)
                
                if self.enable_persistence:
                    self._save_state()
                
                self.logger.info(f"Removed stream: {stream_id}")
    
    def synchronize_data(self, 
                        source_notebook: str,
                        target_notebook: str,
                        data_id: str,
                        data: Any) -> bool:
        """Synchronize data between notebooks"""
        # Validate data consistency
        validation_result = self.consistency_checker.validate_data(data, data_id)
        
        if not validation_result['is_valid']:
            self.logger.error(f"Data validation failed for {data_id}: {validation_result['validation_errors']}")
            return False
        
        # Create synchronization stream if it doesn't exist
        stream_id = f"sync_{source_notebook}_{target_notebook}"
        
        if stream_id not in self.streams:
            self.create_stream(
                stream_id=stream_id,
                stream_type=DataStreamType.MARKET_DATA,
                producer_notebook=source_notebook,
                consumer_notebooks=[target_notebook]
            )
        
        # Send data
        stream = self.streams[stream_id]
        success = stream.publish(data, {
            'data_id': data_id,
            'source_notebook': source_notebook,
            'target_notebook': target_notebook,
            'validation_result': validation_result
        })
        
        if success:
            self.logger.info(f"Synchronized {data_id} from {source_notebook} to {target_notebook}")
        
        return success
    
    def coordinate_concurrent_processing(self, 
                                       data_chunks: List[Any],
                                       processing_func: Callable[[Any], Any],
                                       notebook_id: str) -> List[Any]:
        """Coordinate concurrent processing for a notebook"""
        # Update activity timestamp
        if notebook_id in self.notebook_registry:
            self.notebook_registry[notebook_id]['last_activity'] = time.time()
        
        # Process data
        results = self.processor.process_data_parallel(data_chunks, processing_func)
        
        return results
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get comprehensive coordination status"""
        with self.lock:
            stream_stats = {}
            for stream_id, stream in self.streams.items():
                stream_stats[stream_id] = stream.get_stats()
            
            return {
                'active_notebooks': len(self.active_sessions),
                'registered_notebooks': len(self.notebook_registry),
                'active_streams': len(self.streams),
                'notebook_registry': self.notebook_registry,
                'stream_statistics': stream_stats,
                'processor_stats': self.processor.get_performance_stats(),
                'coordination_uptime': time.time() - self._start_time if hasattr(self, '_start_time') else 0
            }
    
    def _save_state(self):
        """Save coordination state to disk"""
        if not self.enable_persistence:
            return
        
        state_file = self.coordination_dir / "coordination_state.json"
        
        state = {
            'notebook_registry': self.notebook_registry,
            'active_sessions': list(self.active_sessions),
            'stream_ids': list(self.streams.keys()),
            'save_timestamp': time.time()
        }
        
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def _load_state(self):
        """Load coordination state from disk"""
        state_file = self.coordination_dir / "coordination_state.json"
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                self.notebook_registry = state.get('notebook_registry', {})
                self.active_sessions = set(state.get('active_sessions', []))
                
                self.logger.info("Loaded coordination state from disk")
            except Exception as e:
                self.logger.error(f"Failed to load state: {e}")
    
    def cleanup(self):
        """Cleanup coordinator resources"""
        # Stop all streams
        for stream in self.streams.values():
            stream.stop()
        
        # Shutdown processor
        self.processor.shutdown()
        
        # Save final state
        if self.enable_persistence:
            self._save_state()
        
        self.logger.info("Data flow coordinator cleaned up")
    
    def __del__(self):
        """Destructor"""
        self.cleanup()

# Utility functions for notebook integration
def create_notebook_client(notebook_id: str, notebook_type: str, coordinator: DataFlowCoordinator):
    """Create a notebook client for easier integration"""
    
    class NotebookClient:
        def __init__(self, notebook_id: str, notebook_type: str, coordinator: DataFlowCoordinator):
            self.notebook_id = notebook_id
            self.notebook_type = notebook_type
            self.coordinator = coordinator
            self.capabilities = []
            
            # Register with coordinator
            self.coordinator.register_notebook(notebook_id, notebook_type, self.capabilities)
        
        def add_capability(self, capability: str):
            """Add a capability to this notebook"""
            self.capabilities.append(capability)
        
        def create_data_stream(self, stream_id: str, stream_type: DataStreamType, consumers: List[str]):
            """Create a data stream from this notebook"""
            return self.coordinator.create_stream(
                stream_id=stream_id,
                stream_type=stream_type,
                producer_notebook=self.notebook_id,
                consumer_notebooks=consumers
            )
        
        def get_data_stream(self, stream_id: str):
            """Get a data stream"""
            return self.coordinator.get_stream(stream_id)
        
        def sync_data(self, target_notebook: str, data_id: str, data: Any):
            """Synchronize data with another notebook"""
            return self.coordinator.synchronize_data(
                source_notebook=self.notebook_id,
                target_notebook=target_notebook,
                data_id=data_id,
                data=data
            )
        
        def process_concurrent(self, data_chunks: List[Any], processing_func: Callable[[Any], Any]):
            """Process data concurrently"""
            return self.coordinator.coordinate_concurrent_processing(
                data_chunks=data_chunks,
                processing_func=processing_func,
                notebook_id=self.notebook_id
            )
        
        def cleanup(self):
            """Cleanup notebook client"""
            self.coordinator.unregister_notebook(self.notebook_id)
    
    return NotebookClient(notebook_id, notebook_type, coordinator)