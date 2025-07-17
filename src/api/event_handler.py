"""
Event handler for processing SYNERGY_DETECTED and other system events.
Implements Redis Pub/Sub with correlation ID tracking.
"""

import json
import asyncio
from typing import Dict, Any, Callable, List, Optional
from datetime import datetime
from src.utils.redis_compat import redis_client, create_redis_pool_async

from src.monitoring.logger_config import get_logger, set_correlation_id, get_correlation_id
from src.monitoring.metrics_exporter import metrics_exporter

logger = get_logger(__name__)

class EventHandler:
    """
    Handles event processing with Redis Pub/Sub.
    Supports event replay and dead letter queue for reliability.
    """
    
    def __init__(self, redis_url: str):
        """Initialize event handler with Redis connection."""
        self.redis_url = redis_url
        self.redis = None
        self.pubsub = None
        self.subscribers: Dict[str, List[Callable]] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
    async def start(self):
        """Start the event handler and connect to Redis."""
        logger.info("Starting event handler")
        
        try:
            # Create Redis connection using our compatibility layer
            self.redis = await create_redis_pool_async(
                self.redis_url,
                encoding='utf-8',
                decode_responses=True
            )
            
            # Create pubsub instance
            self.pubsub = self.redis.pubsub()
            
            self._running = True
            
            # Start event processing task
            task = asyncio.create_task(self._process_events())
            self._tasks.append(task)
            
            logger.info("Event handler started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start event handler: {e}")
            raise
            
    async def stop(self):
        """Stop the event handler and cleanup resources."""
        logger.info("Stopping event handler")
        
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            
        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Close Redis connections
        if self.pubsub:
            await self.pubsub.close()
            
        if self.redis:
            self.redis.close()
            await self.redis.wait_closed()
            
        logger.info("Event handler stopped")
        
    async def publish_event(self, event_type: str, event_data: Dict[str, Any], 
                          correlation_id: Optional[str] = None):
        """
        Publish an event to Redis.
        
        Args:
            event_type: Type of event (e.g., SYNERGY_DETECTED)
            event_data: Event payload
            correlation_id: Optional correlation ID
        """
        if not self.redis:
            raise RuntimeError("Event handler not started")
            
        if not correlation_id:
            correlation_id = get_correlation_id() or set_correlation_id()
            
        # Prepare event message
        message = {
            "type": event_type,
            "data": event_data,
            "correlation_id": correlation_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Publish to Redis channel
        channel = f"events:{event_type}"
        await self.redis.publish(channel, json.dumps(message))
        
        # Store in event history for replay capability
        history_key = f"event_history:{event_type}:{correlation_id}"
        await self.redis.setex(history_key, 3600, json.dumps(message))  # 1 hour TTL
        
        # Update metrics
        metrics_exporter.event_bus_throughput.labels(
            event_type=event_type,
            status="published"
        ).inc()
        
        logger.info(
            f"Published event",
            event_type=event_type,
            correlation_id=correlation_id
        )
        
    async def subscribe(self, event_type: str, callback: Callable):
        """
        Subscribe to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Async function to call when event is received
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
            
            # Subscribe to Redis channel if first subscriber
            if self.pubsub:
                channel = f"events:{event_type}"
                await self.pubsub.subscribe(channel)
                logger.info(f"Subscribed to channel: {channel}")
                
        self.subscribers[event_type].append(callback)
        
    async def unsubscribe(self, event_type: str, callback: Callable):
        """
        Unsubscribe from an event type.
        
        Args:
            event_type: Type of event to unsubscribe from
            callback: Callback function to remove
        """
        if event_type in self.subscribers:
            self.subscribers[event_type].remove(callback)
            
            # Unsubscribe from Redis channel if no more subscribers
            if not self.subscribers[event_type]:
                del self.subscribers[event_type]
                if self.pubsub:
                    channel = f"events:{event_type}"
                    await self.pubsub.unsubscribe(channel)
                    logger.info(f"Unsubscribed from channel: {channel}")
                    
    async def _process_events(self):
        """Process incoming events from Redis."""
        if not self.pubsub:
            return
            
        logger.info("Starting event processing loop")
        
        try:
            while self._running:
                try:
                    # Get message with timeout
                    message = await asyncio.wait_for(
                        self.pubsub.get_message(ignore_subscribe_messages=True),
                        timeout=1.0
                    )
                    
                    if message and message['type'] == 'message':
                        await self._handle_message(message)
                        
                except asyncio.TimeoutError:
                    # Normal timeout, continue loop
                    continue
                except Exception as e:
                    logger.error(f"Error processing event: {e}")
                    metrics_exporter.record_error("event_processing_error", "event_handler")
                    
        except asyncio.CancelledError:
            logger.info("Event processing loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Fatal error in event processing: {e}")
            raise
            
    async def _handle_message(self, message: Dict[str, Any]):
        """Handle a single message from Redis."""
        try:
            # Parse message
            channel = message['channel']
            data = json.loads(message['data'])
            
            event_type = data['type']
            correlation_id = data.get('correlation_id')
            
            # Set correlation ID for this context
            if correlation_id:
                set_correlation_id(correlation_id)
                
            logger.info(
                f"Received event",
                event_type=event_type,
                correlation_id=correlation_id
            )
            
            # Update metrics
            metrics_exporter.event_bus_throughput.labels(
                event_type=event_type,
                status="received"
            ).inc()
            
            # Call subscribers
            if event_type in self.subscribers:
                tasks = []
                for callback in self.subscribers[event_type]:
                    task = asyncio.create_task(self._safe_callback(callback, data))
                    tasks.append(task)
                    
                # Wait for all callbacks to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check for errors
                errors = [r for r in results if isinstance(r, Exception)]
                if errors:
                    logger.error(
                        f"Errors in event callbacks",
                        event_type=event_type,
                        error_count=len(errors)
                    )
                    # Send to dead letter queue
                    await self._send_to_dlq(data, errors)
                    
        except Exception as e:
            logger.error(
                f"Failed to handle message",
                error=str(e),
                exc_info=True
            )
            metrics_exporter.record_error("message_handling_error", "event_handler")
            
    async def _safe_callback(self, callback: Callable, event_data: Dict[str, Any]):
        """Safely execute a callback with error handling."""
        try:
            await callback(event_data)
        except Exception as e:
            logger.error(
                f"Callback error",
                callback=callback.__name__,
                error=str(e),
                exc_info=True
            )
            raise
            
    async def _send_to_dlq(self, event_data: Dict[str, Any], errors: List[Exception]):
        """Send failed event to dead letter queue."""
        if not self.redis:
            return
            
        dlq_entry = {
            "event": event_data,
            "errors": [str(e) for e in errors],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        dlq_key = f"dlq:{event_data['type']}"
        await self.redis.lpush(dlq_key, json.dumps(dlq_entry))
        
        # Trim to keep only last 1000 entries
        await self.redis.ltrim(dlq_key, 0, 999)
        
        logger.warning(
            f"Event sent to DLQ",
            event_type=event_data['type'],
            correlation_id=event_data.get('correlation_id')
        )
        
    async def replay_events(self, event_type: str, start_time: datetime, 
                          end_time: datetime) -> List[Dict[str, Any]]:
        """
        Replay events within a time range.
        
        Args:
            event_type: Type of events to replay
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            List of events in the time range
        """
        if not self.redis:
            raise RuntimeError("Event handler not started")
            
        pattern = f"event_history:{event_type}:*"
        events = []
        
        # Scan for matching keys
        async for key in self.redis.scan_iter(match=pattern):
            event_json = await self.redis.get(key)
            if event_json:
                event = json.loads(event_json)
                event_time = datetime.fromisoformat(event['timestamp'])
                
                if start_time <= event_time <= end_time:
                    events.append(event)
                    
        # Sort by timestamp
        events.sort(key=lambda e: e['timestamp'])
        
        logger.info(
            f"Replayed events",
            event_type=event_type,
            count=len(events)
        )
        
        return events
        
    async def get_dlq_entries(self, event_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get entries from dead letter queue.
        
        Args:
            event_type: Type of events
            limit: Maximum number of entries to return
            
        Returns:
            List of DLQ entries
        """
        if not self.redis:
            raise RuntimeError("Event handler not started")
            
        dlq_key = f"dlq:{event_type}"
        entries = []
        
        # Get entries from Redis list
        raw_entries = await self.redis.lrange(dlq_key, 0, limit - 1)
        
        for entry_json in raw_entries:
            entry = json.loads(entry_json)
            entries.append(entry)
            
        return entries