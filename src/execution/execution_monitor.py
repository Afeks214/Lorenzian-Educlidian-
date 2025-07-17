"""
Execution Monitor - Tracks execution pipeline performance without changing strategy logic
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque

try:
    import redis.asyncio as redis
    from redis.asyncio.client import Redis
except ImportError:
    redis = None
    Redis = None

logger = logging.getLogger(__name__)


@dataclass
class ExecutionStats:
    """Execution statistics"""
    total_signals: int = 0
    signals_processed: int = 0
    trades_executed: int = 0
    trades_successful: int = 0
    trades_failed: int = 0
    avg_processing_time_ms: float = 0.0
    conversion_rate: float = 0.0
    success_rate: float = 0.0
    
    def calculate_rates(self):
        """Calculate conversion and success rates"""
        if self.signals_processed > 0:
            self.conversion_rate = self.trades_executed / self.signals_processed
        if self.trades_executed > 0:
            self.success_rate = self.trades_successful / self.trades_executed


class ExecutionMonitor:
    """
    Monitors execution pipeline performance and health.
    
    Tracks the complete signal-to-trade conversion process without
    modifying any strategy logic.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/2"):
        self.redis_url = redis_url
        self.redis_client: Optional[Redis] = None
        
        # Statistics tracking
        self.stats = ExecutionStats()
        self.hourly_stats = defaultdict(lambda: ExecutionStats())
        self.recent_events = deque(maxlen=1000)
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)
        self.conversion_history = deque(maxlen=24)  # 24 hours
        
        # Event tracking
        self.signal_events = {}  # Track signal->execution pipeline
        self.running = False
        
        logger.info("ExecutionMonitor initialized")
    
    async def initialize(self):
        """Initialize Redis connection and start monitoring."""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("ExecutionMonitor connected to Redis")
            
        except Exception as e:
            logger.error(f"Failed to initialize ExecutionMonitor: {e}")
            raise
    
    async def start_monitoring(self):
        """Start monitoring execution pipeline."""
        if self.running:
            logger.warning("Monitor already running")
            return
        
        self.running = True
        logger.info("🔍 Starting execution monitoring")
        
        # Start monitoring tasks
        tasks = [
            self._monitor_synergy_signals(),
            self._monitor_execution_events(),
            self._monitor_order_events(),
            self._calculate_hourly_stats()
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in monitoring: {e}")
        finally:
            self.running = False
            logger.info("🛑 Execution monitoring stopped")
    
    async def _monitor_synergy_signals(self):
        """Monitor synergy signals from tactical controller."""
        while self.running:
            try:
                # Listen to synergy events stream
                messages = await self.redis_client.xread(
                    {'synergy_events': '$'},
                    count=10,
                    block=1000
                )
                
                for stream, stream_messages in messages:
                    for message_id, fields in stream_messages:
                        await self._process_synergy_signal(message_id, fields)
                        
            except Exception as e:
                logger.error(f"Error monitoring synergy signals: {e}")
                await asyncio.sleep(1)
    
    async def _monitor_execution_events(self):
        """Monitor execution events from execution engine."""
        while self.running:
            try:
                # Listen to execution events stream
                messages = await self.redis_client.xread(
                    {'execution_stream': '$'},
                    count=10,
                    block=1000
                )
                
                for stream, stream_messages in messages:
                    for message_id, fields in stream_messages:
                        await self._process_execution_event(message_id, fields)
                        
            except Exception as e:
                logger.error(f"Error monitoring execution events: {e}")
                await asyncio.sleep(1)
    
    async def _monitor_order_events(self):
        """Monitor order events from order management."""
        while self.running:
            try:
                # Subscribe to order events
                pubsub = self.redis_client.pubsub()
                await pubsub.subscribe('order_events')
                
                async for message in pubsub.listen():
                    if message['type'] == 'message':
                        try:
                            data = json.loads(message['data'])
                            await self._process_order_event(data)
                        except Exception as e:
                            logger.error(f"Error processing order event: {e}")
                            
            except Exception as e:
                logger.error(f"Error monitoring order events: {e}")
                await asyncio.sleep(1)
    
    async def _process_synergy_signal(self, message_id: str, fields: Dict[bytes, bytes]):
        """Process synergy signal event."""
        try:
            # Parse signal data
            signal_data = {}
            for key, value in fields.items():
                try:
                    signal_data[key.decode()] = json.loads(value.decode())
                except json.JSONDecodeError:
                    signal_data[key.decode()] = value.decode()
            
            correlation_id = signal_data.get('correlation_id', 'unknown')
            
            # Track signal
            self.stats.total_signals += 1
            
            # Store for pipeline tracking
            self.signal_events[correlation_id] = {
                'signal_time': time.time(),
                'signal_data': signal_data,
                'execution_time': None,
                'order_time': None,
                'status': 'signal_received'
            }
            
            # Add to recent events
            self.recent_events.append({
                'type': 'signal',
                'correlation_id': correlation_id,
                'timestamp': time.time(),
                'data': signal_data
            })
            
            logger.debug(f"📊 Signal tracked: {correlation_id}")
            
        except Exception as e:
            logger.error(f"Error processing synergy signal: {e}")
    
    async def _process_execution_event(self, message_id: str, fields: Dict[bytes, bytes]):
        """Process execution event."""
        try:
            # Parse execution data
            execution_data = {}
            for key, value in fields.items():
                try:
                    execution_data[key.decode()] = json.loads(value.decode())
                except json.JSONDecodeError:
                    execution_data[key.decode()] = value.decode()
            
            correlation_id = execution_data.get('correlation_id', 'unknown')
            action = execution_data.get('action', 'unknown')
            
            # Track execution
            self.stats.signals_processed += 1
            
            if action != 'hold':
                self.stats.trades_executed += 1
            
            # Update pipeline tracking
            if correlation_id in self.signal_events:
                self.signal_events[correlation_id]['execution_time'] = time.time()
                self.signal_events[correlation_id]['status'] = 'execution_attempted'
                
                # Calculate processing time
                signal_time = self.signal_events[correlation_id]['signal_time']
                processing_time = (time.time() - signal_time) * 1000
                self.processing_times.append(processing_time)
                
                # Update average processing time
                if self.processing_times:
                    self.stats.avg_processing_time_ms = sum(self.processing_times) / len(self.processing_times)
            
            # Add to recent events
            self.recent_events.append({
                'type': 'execution',
                'correlation_id': correlation_id,
                'timestamp': time.time(),
                'data': execution_data
            })
            
            logger.debug(f"📊 Execution tracked: {correlation_id} -> {action}")
            
        except Exception as e:
            logger.error(f"Error processing execution event: {e}")
    
    async def _process_order_event(self, data: Dict[str, Any]):
        """Process order event."""
        try:
            correlation_id = data.get('correlation_id', 'unknown')
            status = data.get('status', 'unknown')
            
            # Track order success/failure
            if status == 'filled':
                self.stats.trades_successful += 1
            elif status == 'failed':
                self.stats.trades_failed += 1
            
            # Update pipeline tracking
            if correlation_id in self.signal_events:
                self.signal_events[correlation_id]['order_time'] = time.time()
                self.signal_events[correlation_id]['status'] = f'order_{status}'
            
            # Add to recent events
            self.recent_events.append({
                'type': 'order',
                'correlation_id': correlation_id,
                'timestamp': time.time(),
                'data': data
            })
            
            logger.debug(f"📊 Order tracked: {correlation_id} -> {status}")
            
        except Exception as e:
            logger.error(f"Error processing order event: {e}")
    
    async def _calculate_hourly_stats(self):
        """Calculate hourly statistics."""
        while self.running:
            try:
                # Wait for next hour
                await asyncio.sleep(3600)  # 1 hour
                
                # Calculate current stats
                current_hour = datetime.now().hour
                self.hourly_stats[current_hour] = ExecutionStats(
                    total_signals=self.stats.total_signals,
                    signals_processed=self.stats.signals_processed,
                    trades_executed=self.stats.trades_executed,
                    trades_successful=self.stats.trades_successful,
                    trades_failed=self.stats.trades_failed,
                    avg_processing_time_ms=self.stats.avg_processing_time_ms
                )
                
                # Calculate rates
                self.hourly_stats[current_hour].calculate_rates()
                
                # Add to conversion history
                self.conversion_history.append({
                    'hour': current_hour,
                    'conversion_rate': self.hourly_stats[current_hour].conversion_rate,
                    'success_rate': self.hourly_stats[current_hour].success_rate
                })
                
                # Log hourly summary
                logger.info(
                    f"📈 Hourly stats (hour {current_hour}): "
                    f"Conversion: {self.hourly_stats[current_hour].conversion_rate:.2%}, "
                    f"Success: {self.hourly_stats[current_hour].success_rate:.2%}"
                )
                
            except Exception as e:
                logger.error(f"Error calculating hourly stats: {e}")
    
    async def stop_monitoring(self):
        """Stop monitoring."""
        self.running = False
        logger.info("Stopping execution monitoring")
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.stop_monitoring()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("ExecutionMonitor cleanup complete")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current execution statistics."""
        # Calculate current rates
        self.stats.calculate_rates()
        
        return {
            'total_signals': self.stats.total_signals,
            'signals_processed': self.stats.signals_processed,
            'trades_executed': self.stats.trades_executed,
            'trades_successful': self.stats.trades_successful,
            'trades_failed': self.stats.trades_failed,
            'conversion_rate': self.stats.conversion_rate,
            'success_rate': self.stats.success_rate,
            'avg_processing_time_ms': self.stats.avg_processing_time_ms,
            'pipeline_health': self._get_pipeline_health()
        }
    
    def _get_pipeline_health(self) -> Dict[str, Any]:
        """Get pipeline health metrics."""
        # Calculate pipeline efficiency
        total_events = len(self.signal_events)
        completed_pipelines = sum(1 for event in self.signal_events.values() 
                                 if event['status'].startswith('order_'))
        
        pipeline_completion_rate = completed_pipelines / total_events if total_events > 0 else 0
        
        # Recent performance
        recent_conversion_rate = self.conversion_history[-1]['conversion_rate'] if self.conversion_history else 0
        
        return {
            'pipeline_completion_rate': pipeline_completion_rate,
            'recent_conversion_rate': recent_conversion_rate,
            'active_pipelines': len([e for e in self.signal_events.values() 
                                   if e['status'] in ['signal_received', 'execution_attempted']]),
            'avg_pipeline_time_ms': self.stats.avg_processing_time_ms
        }
    
    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent pipeline events."""
        return list(self.recent_events)[-limit:]
    
    def get_conversion_history(self) -> List[Dict[str, Any]]:
        """Get conversion rate history."""
        return list(self.conversion_history)
    
    def get_pipeline_details(self, correlation_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed pipeline information for a specific correlation ID."""
        return self.signal_events.get(correlation_id)
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Get comprehensive summary report."""
        return {
            'current_stats': self.get_current_stats(),
            'pipeline_health': self._get_pipeline_health(),
            'conversion_history': self.get_conversion_history(),
            'recent_events_count': len(self.recent_events),
            'active_pipelines': len(self.signal_events),
            'monitoring_status': 'running' if self.running else 'stopped'
        }