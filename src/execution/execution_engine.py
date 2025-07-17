"""
Execution Engine - Handles trade execution from tactical decisions
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime

try:
    import redis.asyncio as redis
    from redis.asyncio.client import Redis
except ImportError:
    redis = None
    Redis = None

from src.core.events import EventBus, Event, EventType

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Execution Engine for processing tactical decisions and executing trades.
    
    Bridges the gap between tactical decisions and actual order execution.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/2"):
        self.redis_url = redis_url
        self.redis_client: Optional[Redis] = None
        
        
        # We'll create a mock event bus for now
        self.event_bus = EventBus()
        
        # Initialize order manager with mock router for now
        self.order_manager = None  # We'll create a simpler version
        
        # Execution tracking
        self.executions_processed = 0
        self.executions_success = 0
        self.executions_failed = 0
        self.running = False
        
        logger.info("ExecutionEngine initialized")
    
    async def initialize(self):
        """Initialize Redis connection and start listening."""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("ExecutionEngine connected to Redis")
            
        except Exception as e:
            logger.error(f"Failed to initialize ExecutionEngine: {e}")
            raise
    
    async def start_execution_listener(self):
        """Start listening for execution events."""
        if self.running:
            logger.warning("Execution listener already running")
            return
        
        self.running = True
        logger.info("🚀 Starting execution listener")
        
        try:
            # Subscribe to execution events
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe('execution_events')
            
            # Also listen to execution stream
            stream_task = asyncio.create_task(self._listen_execution_stream())
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        await self._process_execution_event(data)
                    except Exception as e:
                        logger.error(f"Error processing execution event: {e}")
                        
        except Exception as e:
            logger.error(f"Error in execution listener: {e}")
        finally:
            self.running = False
            logger.info("🛑 Execution listener stopped")
    
    async def _listen_execution_stream(self):
        """Listen to execution stream for persistence."""
        while self.running:
            try:
                # Read from execution stream
                messages = await self.redis_client.xread(
                    {'execution_stream': '$'},
                    count=10,
                    block=1000
                )
                
                for stream, stream_messages in messages:
                    for message_id, fields in stream_messages:
                        try:
                            # Convert bytes to strings
                            data = {}
                            for key, value in fields.items():
                                try:
                                    data[key.decode()] = json.loads(value.decode())
                                except json.JSONDecodeError:
                                    data[key.decode()] = value.decode()
                            
                            await self._process_execution_event(data)
                            
                        except Exception as e:
                            logger.error(f"Error processing stream message: {e}")
                
            except Exception as e:
                logger.error(f"Error in stream listener: {e}")
                await asyncio.sleep(1)
    
    async def _process_execution_event(self, data: Dict[str, Any]):
        """Process execution event and execute trade."""
        start_time = time.perf_counter()
        
        try:
            self.executions_processed += 1
            
            correlation_id = data.get('correlation_id', 'unknown')
            action = data.get('action', 'hold')
            confidence = data.get('confidence', 0.0)
            execution_command = data.get('execution_command', {})
            
            logger.info(
                f"📥 Processing execution event: {action} (confidence: {confidence:.3f}) "
                f"[correlation_id: {correlation_id}]"
            )
            
            # Skip if action is hold
            if action == 'hold':
                logger.info(f"Skipping hold action [correlation_id: {correlation_id}]")
                return
            
            # Skip if no execution command
            if not execution_command or execution_command.get('action') != 'execute_trade':
                logger.info(f"No execution command or not execute_trade [correlation_id: {correlation_id}]")
                return
            
            # Simulate order execution (in production, this would use real order manager)
            try:
                # Generate order ID
                order_id = f"ord_{int(time.time() * 1000)}_{correlation_id[:8]}"
                
                # Simulate order processing
                await asyncio.sleep(0.001)  # 1ms simulation
                
                # Extract order details for logging
                symbol = execution_command.get('symbol', 'UNKNOWN')
                side = execution_command.get('side', 'BUY')
                quantity = execution_command.get('quantity', 1)
                
                self.executions_success += 1
                processing_time = (time.perf_counter() - start_time) * 1000
                
                logger.info(
                    f"✅ Order executed successfully: {order_id} "
                    f"[correlation_id: {correlation_id}] "
                    f"[processing_time: {processing_time:.2f}ms] "
                    f"[symbol: {symbol}] [side: {side}] [quantity: {quantity}]"
                )
                
                # Emit ORDER_FILLED event (simulated)
                await self._emit_order_event(order_id, correlation_id, 'filled')
                
            except Exception as e:
                self.executions_failed += 1
                logger.error(f"Order execution failed: {e} [correlation_id: {correlation_id}]")
                
        except Exception as e:
            self.executions_failed += 1
            logger.error(f"Error processing execution event: {e}")
    
    
    async def _emit_order_event(self, order_id: str, correlation_id: str, status: str):
        """Emit order event back to system."""
        try:
            event_data = {
                'order_id': order_id,
                'correlation_id': correlation_id,
                'status': status,
                'timestamp': time.time()
            }
            
            await self.redis_client.publish(
                'order_events',
                json.dumps(event_data)
            )
            
        except Exception as e:
            logger.error(f"Error emitting order event: {e}")
    
    async def stop_execution_listener(self):
        """Stop the execution listener."""
        self.running = False
        logger.info("Stopping execution listener")
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.stop_execution_listener()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("ExecutionEngine cleanup complete")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        total = max(1, self.executions_processed)
        success_rate = self.executions_success / total
        
        return {
            'executions_processed': self.executions_processed,
            'executions_success': self.executions_success,
            'executions_failed': self.executions_failed,
            'success_rate': success_rate,
            'running': self.running,
            'conversion_rate': success_rate  # This is our key metric
        }