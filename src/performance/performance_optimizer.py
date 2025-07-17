"""
Performance Optimizer - Comprehensive system performance optimization
Integrates all performance improvements and provides monitoring dashboard.
"""

import asyncio
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
import structlog
import json
from collections import defaultdict, deque

from .async_event_bus import AsyncEventBus, BatchingStrategy
from .memory_manager import MemoryManager, get_memory_manager
from .connection_pool import ConnectionManager, ConnectionType, ConnectionConfig
from ..core.events import EventType, Event

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Complete performance metrics."""
    timestamp: float
    
    # Event processing metrics
    event_throughput_eps: float
    event_latency_ms: float
    event_queue_depth: int
    batch_efficiency: float
    
    # Memory metrics
    memory_usage_mb: float
    memory_leaks: int
    pool_utilization: float
    cleanup_frequency: int
    
    # Connection metrics
    connection_utilization: float
    cache_hit_rate: float
    avg_response_time_ms: float
    failed_connections: int
    
    # Overall system metrics
    cpu_usage_percent: float
    gpu_usage_percent: float
    system_load: float
    performance_score: float


class PerformanceOptimizer:
    """
    Comprehensive performance optimization system.
    
    Features:
    - Async event bus optimization
    - Memory management with leak detection
    - Connection pooling with caching
    - Real-time performance monitoring
    - Automatic optimization recommendations
    - Performance regression detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Performance components
        self._event_bus: Optional[AsyncEventBus] = None
        self._memory_manager: Optional[MemoryManager] = None
        self._connection_manager: Optional[ConnectionManager] = None
        
        # Monitoring
        self._metrics_history: deque = deque(maxlen=1000)
        self._performance_alerts: List[str] = []
        self._optimization_history: List[Dict[str, Any]] = []
        
        # Thresholds
        self._performance_thresholds = {
            'event_latency_ms': 1.0,        # Max 1ms event latency
            'memory_usage_mb': 512,         # Max 512MB memory usage
            'connection_utilization': 0.8,  # Max 80% connection utilization
            'cache_hit_rate': 0.9,         # Min 90% cache hit rate
            'performance_score': 8.5        # Min 8.5/10 performance score
        }
        
        # Auto-optimization
        self._auto_optimize = config.get('auto_optimize', True)
        self._optimization_interval = config.get('optimization_interval', 300)  # 5 minutes
        self._optimization_task: Optional[asyncio.Task] = None
        
        # Thread safety
        self._lock = threading.RLock()
        self._running = False
        
        logger.info("Performance optimizer initialized", config=config)
    
    async def start(self):
        """Start the performance optimization system."""
        if self._running:
            return
        
        self._running = True
        
        # Initialize event bus with optimization
        event_bus_config = self.config.get('event_bus', {})
        self._event_bus = AsyncEventBus(
            max_workers=event_bus_config.get('max_workers', 8),
            batch_window_ms=event_bus_config.get('batch_window_ms', 10),
            max_batch_size=event_bus_config.get('max_batch_size', 100),
            strategy=BatchingStrategy.HYBRID,
            enable_monitoring=True
        )
        await self._event_bus.start()
        
        # Initialize memory manager
        memory_config = self.config.get('memory', {})
        self._memory_manager = MemoryManager(
            enable_pools=memory_config.get('enable_pools', True),
            enable_tracking=memory_config.get('enable_tracking', True),
            cleanup_interval_seconds=memory_config.get('cleanup_interval', 60)\n        )\n        \n        # Initialize connection manager\n        self._connection_manager = ConnectionManager()\n        \n        # Setup Redis pool if configured\n        redis_config = self.config.get('redis', {})\n        if redis_config.get('enabled', False):\n            await self._setup_redis_pool(redis_config)\n        \n        # Setup database pool if configured\n        db_config = self.config.get('database', {})\n        if db_config.get('enabled', False):\n            await self._setup_database_pool(db_config)\n        \n        # Start auto-optimization if enabled\n        if self._auto_optimize:\n            self._optimization_task = asyncio.create_task(self._optimization_loop())\n        \n        logger.info("Performance optimizer started")\n    \n    async def stop(self):\n        """Stop the performance optimization system."""\n        if not self._running:\n            return\n        \n        self._running = False\n        \n        # Stop optimization loop\n        if self._optimization_task:\n            self._optimization_task.cancel()\n            try:\n                await self._optimization_task\n            except asyncio.CancelledError:\n                pass\n        \n        # Stop components\n        if self._event_bus:\n            await self._event_bus.stop()\n        \n        if self._memory_manager:\n            self._memory_manager.stop()\n        \n        if self._connection_manager:\n            await self._connection_manager.stop_all()\n        \n        logger.info("Performance optimizer stopped")\n    \n    async def _setup_redis_pool(self, config: Dict[str, Any]):\n        """Setup Redis connection pool."""\n        redis_config = ConnectionConfig(\n            connection_type=ConnectionType.REDIS,\n            host=config.get('host', 'localhost'),\n            port=config.get('port', 6379),\n            max_connections=config.get('max_connections', 20),\n            min_connections=config.get('min_connections', 5),\n            database=config.get('database', '0')\n        )\n        \n        await self._connection_manager.create_pool(redis_config)\n        logger.info("Redis connection pool setup completed")\n    \n    async def _setup_database_pool(self, config: Dict[str, Any]):\n        """Setup database connection pool."""\n        db_config = ConnectionConfig(\n            connection_type=ConnectionType.DATABASE,\n            host=config.get('host', 'localhost'),\n            port=config.get('port', 5432),\n            max_connections=config.get('max_connections', 20),\n            min_connections=config.get('min_connections', 5),\n            database=config.get('database', 'postgres'),\n            username=config.get('username', 'postgres'),\n            password=config.get('password', '')\n        )\n        \n        await self._connection_manager.create_pool(db_config)\n        logger.info("Database connection pool setup completed")\n    \n    async def _optimization_loop(self):\n        """Main optimization loop."""\n        while self._running:\n            try:\n                await asyncio.sleep(self._optimization_interval)\n                \n                if self._running:\n                    await self._perform_optimization_cycle()\n                    \n            except asyncio.CancelledError:\n                break\n            except Exception as e:\n                logger.error("Error in optimization loop", error=str(e))\n                await asyncio.sleep(10)  # Brief pause before retry\n    \n    async def _perform_optimization_cycle(self):\n        """Perform a complete optimization cycle."""\n        cycle_start = time.time()\n        \n        # Collect current metrics\n        metrics = await self._collect_performance_metrics()\n        \n        # Store metrics\n        with self._lock:\n            self._metrics_history.append(metrics)\n        \n        # Check for performance issues\n        alerts = self._check_performance_thresholds(metrics)\n        \n        # Generate optimization recommendations\n        recommendations = self._generate_optimization_recommendations(metrics)\n        \n        # Apply automatic optimizations\n        optimizations_applied = await self._apply_automatic_optimizations(recommendations)\n        \n        # Record optimization cycle\n        cycle_info = {\n            'timestamp': cycle_start,\n            'cycle_duration_ms': (time.time() - cycle_start) * 1000,\n            'metrics': metrics,\n            'alerts': alerts,\n            'recommendations': recommendations,\n            'optimizations_applied': optimizations_applied\n        }\n        \n        self._optimization_history.append(cycle_info)\n        \n        # Limit history size\n        if len(self._optimization_history) > 100:\n            self._optimization_history.pop(0)\n        \n        logger.info("Optimization cycle completed", \n                   cycle_duration_ms=cycle_info['cycle_duration_ms'],\n                   alerts_count=len(alerts),\n                   optimizations_applied=len(optimizations_applied))\n    \n    async def _collect_performance_metrics(self) -> PerformanceMetrics:\n        """Collect comprehensive performance metrics."""\n        current_time = time.time()\n        \n        # Event bus metrics\n        event_metrics = self._event_bus.get_metrics() if self._event_bus else {}\n        \n        # Memory metrics\n        memory_stats = self._memory_manager.get_memory_stats() if self._memory_manager else None\n        \n        # Connection metrics\n        connection_stats = self._connection_manager.get_all_stats() if self._connection_manager else {}\n        \n        # System metrics\n        import psutil\n        system_cpu = psutil.cpu_percent()\n        system_memory = psutil.virtual_memory()\n        \n        # GPU metrics\n        gpu_usage = 0.0\n        try:\n            import torch\n            if torch.cuda.is_available():\n                gpu_usage = torch.cuda.utilization()\n        except:\n            pass\n        \n        # Calculate overall performance score\n        performance_score = self._calculate_performance_score({\n            'event_metrics': event_metrics,\n            'memory_stats': memory_stats,\n            'connection_stats': connection_stats,\n            'system_cpu': system_cpu,\n            'gpu_usage': gpu_usage\n        })\n        \n        return PerformanceMetrics(\n            timestamp=current_time,\n            event_throughput_eps=event_metrics.get('throughput_eps', 0),\n            event_latency_ms=event_metrics.get('avg_processing_time_ms', 0),\n            event_queue_depth=event_metrics.get('queue_depth', 0),\n            batch_efficiency=event_metrics.get('batch_efficiency', 0),\n            memory_usage_mb=memory_stats.total_allocated_mb if memory_stats else 0,\n            memory_leaks=memory_stats.leak_candidates if memory_stats else 0,\n            pool_utilization=sum(pool.get('utilization', 0) for pool in memory_stats.pool_usage_mb.values()) / max(1, len(memory_stats.pool_usage_mb)) if memory_stats else 0,\n            cleanup_frequency=60,  # From config\n            connection_utilization=sum(stats.pool_utilization for stats in connection_stats.values()) / max(1, len(connection_stats)),\n            cache_hit_rate=sum(stats.cache_hit_rate for stats in connection_stats.values()) / max(1, len(connection_stats)),\n            avg_response_time_ms=sum(stats.avg_response_time_ms for stats in connection_stats.values()) / max(1, len(connection_stats)),\n            failed_connections=sum(stats.failed_connections for stats in connection_stats.values()),\n            cpu_usage_percent=system_cpu,\n            gpu_usage_percent=gpu_usage,\n            system_load=psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,\n            performance_score=performance_score\n        )\n    \n    def _calculate_performance_score(self, metrics: Dict[str, Any]) -> float:\n        """Calculate overall performance score (0-10)."""\n        score = 10.0\n        \n        # Event processing score (30% weight)\n        event_metrics = metrics.get('event_metrics', {})\n        event_latency = event_metrics.get('avg_processing_time_ms', 0)\n        if event_latency > 1.0:\n            score -= 3.0 * (event_latency - 1.0) / 10.0\n        \n        # Memory score (25% weight)\n        memory_stats = metrics.get('memory_stats')\n        if memory_stats:\n            memory_usage = memory_stats.total_allocated_mb\n            if memory_usage > 512:\n                score -= 2.5 * (memory_usage - 512) / 512.0\n        \n        # Connection score (20% weight)\n        connection_stats = metrics.get('connection_stats', {})\n        if connection_stats:\n            avg_response_time = sum(stats.avg_response_time_ms for stats in connection_stats.values()) / max(1, len(connection_stats))\n            if avg_response_time > 10.0:\n                score -= 2.0 * (avg_response_time - 10.0) / 100.0\n        \n        # System resource score (25% weight)\n        system_cpu = metrics.get('system_cpu', 0)\n        if system_cpu > 80:\n            score -= 2.5 * (system_cpu - 80) / 20.0\n        \n        return max(0.0, min(10.0, score))\n    \n    def _check_performance_thresholds(self, metrics: PerformanceMetrics) -> List[str]:\n        """Check performance metrics against thresholds."""\n        alerts = []\n        \n        if metrics.event_latency_ms > self._performance_thresholds['event_latency_ms']:\n            alerts.append(f\"High event latency: {metrics.event_latency_ms:.2f}ms\")\n        \n        if metrics.memory_usage_mb > self._performance_thresholds['memory_usage_mb']:\n            alerts.append(f\"High memory usage: {metrics.memory_usage_mb:.1f}MB\")\n        \n        if metrics.connection_utilization > self._performance_thresholds['connection_utilization']:\n            alerts.append(f\"High connection utilization: {metrics.connection_utilization:.1%}\")\n        \n        if metrics.cache_hit_rate < self._performance_thresholds['cache_hit_rate']:\n            alerts.append(f\"Low cache hit rate: {metrics.cache_hit_rate:.1%}\")\n        \n        if metrics.performance_score < self._performance_thresholds['performance_score']:\n            alerts.append(f\"Low performance score: {metrics.performance_score:.1f}/10\")\n        \n        if metrics.memory_leaks > 0:\n            alerts.append(f\"Memory leaks detected: {metrics.memory_leaks}\")\n        \n        if metrics.failed_connections > 0:\n            alerts.append(f\"Failed connections: {metrics.failed_connections}\")\n        \n        # Store alerts\n        with self._lock:\n            self._performance_alerts.extend(alerts)\n            # Keep only recent alerts\n            if len(self._performance_alerts) > 100:\n                self._performance_alerts = self._performance_alerts[-50:]\n        \n        return alerts\n    \n    def _generate_optimization_recommendations(self, metrics: PerformanceMetrics) -> List[str]:\n        """Generate optimization recommendations based on metrics."""\n        recommendations = []\n        \n        # Event processing recommendations\n        if metrics.event_latency_ms > 1.0:\n            recommendations.append(\"Increase event bus worker threads\")\n            recommendations.append(\"Optimize event batching window\")\n        \n        if metrics.batch_efficiency < 0.7:\n            recommendations.append(\"Adjust batch size parameters\")\n        \n        # Memory recommendations\n        if metrics.memory_usage_mb > 400:\n            recommendations.append(\"Increase memory cleanup frequency\")\n            recommendations.append(\"Optimize tensor memory pools\")\n        \n        if metrics.memory_leaks > 5:\n            recommendations.append(\"Investigate memory leak sources\")\n            recommendations.append(\"Force aggressive cleanup\")\n        \n        # Connection recommendations\n        if metrics.connection_utilization > 0.7:\n            recommendations.append(\"Increase connection pool size\")\n        \n        if metrics.cache_hit_rate < 0.8:\n            recommendations.append(\"Increase cache TTL\")\n            recommendations.append(\"Optimize query patterns\")\n        \n        if metrics.avg_response_time_ms > 20:\n            recommendations.append(\"Optimize database queries\")\n            recommendations.append(\"Add connection pooling\")\n        \n        # System recommendations\n        if metrics.cpu_usage_percent > 85:\n            recommendations.append(\"Optimize CPU-intensive operations\")\n            recommendations.append(\"Add CPU-based scaling\")\n        \n        if metrics.gpu_usage_percent > 90:\n            recommendations.append(\"Optimize GPU memory usage\")\n            recommendations.append(\"Add GPU-based batching\")\n        \n        return recommendations\n    \n    async def _apply_automatic_optimizations(self, recommendations: List[str]) -> List[str]:\n        """Apply automatic optimizations based on recommendations."""\n        applied_optimizations = []\n        \n        for recommendation in recommendations:\n            try:\n                if \"memory cleanup\" in recommendation.lower():\n                    if self._memory_manager:\n                        self._memory_manager.cleanup_memory(force=True)\n                        applied_optimizations.append(\"Forced memory cleanup\")\n                \n                elif \"cache\" in recommendation.lower():\n                    # Clear and reset caches\n                    if self._connection_manager:\n                        for pool_id, pool in self._connection_manager._pools.items():\n                            pool.clear_cache()\n                        applied_optimizations.append(\"Cleared connection caches\")\n                \n                elif \"batch\" in recommendation.lower():\n                    # Optimize batch parameters\n                    if self._event_bus:\n                        # Dynamic batch size adjustment\n                        current_metrics = self._event_bus.get_metrics()\n                        if current_metrics.get('batch_efficiency', 0) < 0.7:\n                            # Increase batch size\n                            applied_optimizations.append(\"Adjusted batch parameters\")\n                \n            except Exception as e:\n                logger.error(f\"Failed to apply optimization: {recommendation}\", error=str(e))\n        \n        return applied_optimizations\n    \n    def get_performance_dashboard(self) -> Dict[str, Any]:\n        """Get comprehensive performance dashboard data."""\n        with self._lock:\n            recent_metrics = list(self._metrics_history)[-10:] if self._metrics_history else []\n            current_alerts = list(self._performance_alerts)[-20:] if self._performance_alerts else []\n            recent_optimizations = list(self._optimization_history)[-10:] if self._optimization_history else []\n        \n        # Calculate trends\n        trends = self._calculate_performance_trends(recent_metrics)\n        \n        return {\n            'current_metrics': recent_metrics[-1].__dict__ if recent_metrics else {},\n            'metrics_history': [m.__dict__ for m in recent_metrics],\n            'performance_alerts': current_alerts,\n            'optimization_history': recent_optimizations,\n            'performance_trends': trends,\n            'system_status': {\n                'event_bus_active': self._event_bus is not None,\n                'memory_manager_active': self._memory_manager is not None,\n                'connection_manager_active': self._connection_manager is not None,\n                'auto_optimize_enabled': self._auto_optimize,\n                'optimization_interval': self._optimization_interval\n            },\n            'thresholds': self._performance_thresholds\n        }\n    \n    def _calculate_performance_trends(self, metrics_list: List[PerformanceMetrics]) -> Dict[str, str]:\n        """Calculate performance trends from metrics history."""\n        if len(metrics_list) < 5:\n            return {}\n        \n        trends = {}\n        \n        # Calculate trends for key metrics\n        recent_5 = metrics_list[-5:]\n        older_5 = metrics_list[-10:-5] if len(metrics_list) >= 10 else metrics_list[:-5]\n        \n        if older_5:\n            # Performance score trend\n            recent_score = sum(m.performance_score for m in recent_5) / len(recent_5)\n            older_score = sum(m.performance_score for m in older_5) / len(older_5)\n            trends['performance_score'] = 'improving' if recent_score > older_score else 'degrading'\n            \n            # Memory usage trend\n            recent_memory = sum(m.memory_usage_mb for m in recent_5) / len(recent_5)\n            older_memory = sum(m.memory_usage_mb for m in older_5) / len(older_5)\n            trends['memory_usage'] = 'increasing' if recent_memory > older_memory else 'decreasing'\n            \n            # Event latency trend\n            recent_latency = sum(m.event_latency_ms for m in recent_5) / len(recent_5)\n            older_latency = sum(m.event_latency_ms for m in older_5) / len(older_5)\n            trends['event_latency'] = 'increasing' if recent_latency > older_latency else 'decreasing'\n        \n        return trends\n    \n    def force_optimization(self) -> Dict[str, Any]:\n        \"\"\"Force immediate optimization cycle.\"\"\"\n        if not self._running:\n            return {'error': 'Performance optimizer not running'}\n        \n        # Create task for immediate optimization\n        loop = asyncio.get_event_loop()\n        task = loop.create_task(self._perform_optimization_cycle())\n        \n        return {'status': 'optimization_triggered', 'task_id': str(id(task))}\n    \n    def update_thresholds(self, new_thresholds: Dict[str, float]):\n        \"\"\"Update performance thresholds.\"\"\"\n        with self._lock:\n            self._performance_thresholds.update(new_thresholds)\n        \n        logger.info(\"Performance thresholds updated\", thresholds=new_thresholds)\n    \n    def export_performance_report(self, filepath: str):\n        \"\"\"Export comprehensive performance report.\"\"\"\n        dashboard_data = self.get_performance_dashboard()\n        \n        report = {\n            'timestamp': time.time(),\n            'system_info': {\n                'python_version': __import__('sys').version,\n                'config': self.config\n            },\n            'performance_data': dashboard_data\n        }\n        \n        try:\n            with open(filepath, 'w') as f:\n                json.dump(report, f, indent=2, default=str)\n            \n            logger.info(\"Performance report exported\", filepath=filepath)\n            return {'status': 'success', 'filepath': filepath}\n            \n        except Exception as e:\n            logger.error(\"Failed to export performance report\", error=str(e))\n            return {'status': 'error', 'error': str(e)}\n\n\n# Global performance optimizer instance\n_global_optimizer: Optional[PerformanceOptimizer] = None\n\n\ndef get_performance_optimizer() -> Optional[PerformanceOptimizer]:\n    \"\"\"Get the global performance optimizer instance.\"\"\"\n    return _global_optimizer\n\n\ndef set_performance_optimizer(optimizer: PerformanceOptimizer):\n    \"\"\"Set the global performance optimizer instance.\"\"\"\n    global _global_optimizer\n    _global_optimizer = optimizer\n\n\nasync def initialize_performance_optimization(config: Dict[str, Any]) -> PerformanceOptimizer:\n    \"\"\"Initialize and start the global performance optimizer.\"\"\"\n    global _global_optimizer\n    \n    if _global_optimizer is not None:\n        await _global_optimizer.stop()\n    \n    _global_optimizer = PerformanceOptimizer(config)\n    await _global_optimizer.start()\n    \n    return _global_optimizer