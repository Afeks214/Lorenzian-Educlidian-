"""
Execution Tracker

Real-time execution monitoring and performance tracking for ultra-low latency requirements.
Tracks all execution metrics including latency, fill rates, and implementation shortfall.
"""

import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import structlog

from .order_types import Order, OrderStatus, OrderExecution
from ..analytics.performance_metrics import PerformanceCalculator

logger = structlog.get_logger()


@dataclass
class ExecutionMetrics:
    """Comprehensive execution performance metrics"""
    
    # Latency metrics (in microseconds)
    order_placement_latency: float = 0.0
    acknowledgement_latency: float = 0.0
    fill_latency: float = 0.0
    total_execution_time: float = 0.0
    
    # Fill metrics
    fill_ratio: float = 0.0
    partial_fill_count: int = 0
    complete_fill_count: int = 0
    
    # Cost metrics
    implementation_shortfall: float = 0.0
    market_impact: float = 0.0
    timing_cost: float = 0.0
    commission_rate: float = 0.0
    
    # Venue metrics
    venue_fill_rates: Dict[str, float] = field(default_factory=dict)
    venue_latencies: Dict[str, float] = field(default_factory=dict)
    
    # Quality scores
    execution_quality_score: float = 0.0
    slippage: float = 0.0
    
    @property
    def meets_latency_target(self) -> bool:
        """Check if execution meets <500μs target"""
        return self.total_execution_time <= 500.0
    
    @property
    def meets_fill_target(self) -> bool:
        """Check if execution meets >99.8% fill rate target"""
        return self.fill_ratio >= 0.998


@dataclass
class RealTimeStats:
    """Real-time execution statistics"""
    
    # Current window stats (last N orders)
    window_size: int = 1000
    orders_processed: int = 0
    avg_latency: float = 0.0
    fill_rate: float = 0.0
    success_rate: float = 0.0
    
    # Running totals
    total_orders: int = 0
    total_filled: int = 0
    total_rejected: int = 0
    
    # Performance targets
    latency_target_met: float = 0.0  # % meeting <500μs
    fill_target_met: float = 0.0    # % meeting >99.8%
    
    # Alert thresholds
    latency_violations: int = 0
    fill_rate_violations: int = 0


class ExecutionTracker:
    """
    Real-time execution tracking and performance monitoring.
    
    Provides millisecond-level tracking of order execution performance
    with comprehensive metrics and alerting.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.performance_calc = PerformanceCalculator()
        
        # Execution tracking
        self.active_orders: Dict[str, Order] = {}
        self.completed_orders: Dict[str, Order] = {}
        self.execution_history: deque = deque(maxlen=10000)
        
        # Performance metrics
        self.real_time_stats = RealTimeStats()
        self.venue_stats: Dict[str, ExecutionMetrics] = defaultdict(ExecutionMetrics)
        self.symbol_stats: Dict[str, ExecutionMetrics] = defaultdict(ExecutionMetrics)
        
        # Latency tracking
        self.latency_history: deque = deque(maxlen=5000)
        self.latency_percentiles: Dict[str, float] = {}
        
        # Threading for real-time updates
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="execution_tracker")
        self.lock = threading.RLock()
        
        # Performance targets
        self.latency_target_us = self.config.get('latency_target_us', 500)
        self.fill_rate_target = self.config.get('fill_rate_target', 0.998)
        
        # Alerting
        self.alert_thresholds = {
            'max_latency_us': self.config.get('max_latency_us', 1000),
            'min_fill_rate': self.config.get('min_fill_rate', 0.95),
            'latency_violation_threshold': self.config.get('latency_violation_threshold', 10),
            'fill_violation_threshold': self.config.get('fill_violation_threshold', 5)
        }
        
        logger.info(
            "ExecutionTracker initialized",
            latency_target_us=self.latency_target_us,
            fill_rate_target=self.fill_rate_target
        )
    
    def track_order_submission(self, order: Order, submission_timestamp: datetime = None) -> None:
        """Track order submission with precise timing"""
        with self.lock:
            submission_time = submission_timestamp or datetime.now()
            
            # Calculate submission latency
            if order.created_timestamp:
                submission_latency = (submission_time - order.created_timestamp).total_seconds() * 1_000_000
                order.order_placement_latency = submission_latency
                
                # Update latency tracking
                self.latency_history.append(submission_latency)
                self._update_latency_percentiles()
            
            # Add to active orders
            self.active_orders[order.order_id] = order
            order.submitted_timestamp = submission_time
            
            # Update real-time stats
            self.real_time_stats.orders_processed += 1
            self.real_time_stats.total_orders += 1
            
            logger.debug(
                "Order submission tracked",
                order_id=order.order_id,
                symbol=order.symbol,
                submission_latency_us=order.order_placement_latency
            )
    
    def track_order_acknowledgement(self, order_id: str, ack_timestamp: datetime = None) -> None:
        """Track order acknowledgement"""
        with self.lock:
            order = self.active_orders.get(order_id)
            if not order:
                logger.warning("Order not found for acknowledgement", order_id=order_id)
                return
            
            ack_time = ack_timestamp or datetime.now()
            
            # Calculate acknowledgement latency
            if order.submitted_timestamp:
                ack_latency = (ack_time - order.submitted_timestamp).total_seconds() * 1_000_000
                order.acknowledgement_latency = ack_latency
                
                logger.debug(
                    "Order acknowledgement tracked",
                    order_id=order_id,
                    ack_latency_us=ack_latency
                )
    
    def track_order_execution(self, order_id: str, execution: OrderExecution) -> None:
        """Track individual order execution"""
        with self.lock:
            order = self.active_orders.get(order_id)
            if not order:
                logger.warning("Order not found for execution", order_id=order_id)
                return
            
            # Add execution to order
            order.add_execution(execution)
            
            # Calculate fill latency
            if order.submitted_timestamp:
                fill_latency = (execution.timestamp - order.submitted_timestamp).total_seconds() * 1_000_000
                order.fill_latency = fill_latency
            
            # Update venue stats
            venue_stats = self.venue_stats[execution.venue]
            self._update_venue_metrics(venue_stats, order, execution)
            
            # Update symbol stats
            symbol_stats = self.symbol_stats[order.symbol]
            self._update_symbol_metrics(symbol_stats, order, execution)
            
            # Move to completed if fully filled
            if order.is_complete:
                self._complete_order(order)
            
            logger.debug(
                "Order execution tracked",
                order_id=order_id,
                execution_id=execution.execution_id,
                fill_latency_us=order.fill_latency,
                venue=execution.venue
            )
    
    def track_order_completion(self, order_id: str, final_status: OrderStatus) -> None:
        """Track order completion (filled, cancelled, rejected)"""
        with self.lock:
            order = self.active_orders.get(order_id)
            if not order:
                logger.warning("Order not found for completion", order_id=order_id)
                return
            
            order.update_status(final_status)
            self._complete_order(order)
    
    def _complete_order(self, order: Order) -> None:
        """Complete order tracking and update statistics"""
        # Move from active to completed
        if order.order_id in self.active_orders:
            del self.active_orders[order.order_id]
        
        self.completed_orders[order.order_id] = order
        
        # Calculate final metrics
        execution_metrics = self._calculate_execution_metrics(order)
        
        # Add to execution history
        self.execution_history.append({
            'order_id': order.order_id,
            'symbol': order.symbol,
            'timestamp': order.last_updated,
            'metrics': execution_metrics,
            'performance': order.get_performance_metrics()
        })
        
        # Update real-time statistics
        self._update_real_time_stats(order, execution_metrics)
        
        # Check for performance violations
        self._check_performance_violations(order, execution_metrics)
        
        logger.info(
            "Order completed",
            order_id=order.order_id,
            symbol=order.symbol,
            status=order.status.value,
            fill_ratio=order.fill_ratio,
            execution_quality=execution_metrics.execution_quality_score
        )
    
    def _calculate_execution_metrics(self, order: Order) -> ExecutionMetrics:
        """Calculate comprehensive execution metrics for order"""
        metrics = ExecutionMetrics()
        
        # Latency metrics
        metrics.order_placement_latency = order.order_placement_latency or 0.0
        metrics.acknowledgement_latency = order.acknowledgement_latency or 0.0
        metrics.fill_latency = order.fill_latency or 0.0
        
        if order.submitted_timestamp and order.last_updated:
            metrics.total_execution_time = (
                order.last_updated - order.submitted_timestamp
            ).total_seconds() * 1_000_000
        
        # Fill metrics
        metrics.fill_ratio = order.fill_ratio
        if order.status == OrderStatus.FILLED:
            metrics.complete_fill_count = 1
        elif order.status == OrderStatus.PARTIALLY_FILLED:
            metrics.partial_fill_count = 1
        
        # Use performance calculator for advanced metrics
        if order.executions:
            advanced_metrics = self.performance_calc.calculate_execution_quality(order)
            metrics.implementation_shortfall = advanced_metrics.get('implementation_shortfall', 0.0)
            metrics.market_impact = advanced_metrics.get('market_impact', 0.0)
            metrics.timing_cost = advanced_metrics.get('timing_cost', 0.0)
            metrics.execution_quality_score = advanced_metrics.get('quality_score', 0.0)
            metrics.slippage = advanced_metrics.get('slippage', 0.0)
        
        # Commission rate
        if order.filled_quantity > 0:
            metrics.commission_rate = order.total_commission / order.filled_quantity
        
        return metrics
    
    def _update_venue_metrics(self, venue_stats: ExecutionMetrics, order: Order, execution: OrderExecution) -> None:
        """Update venue-specific metrics"""
        # Update latency
        if order.fill_latency:
            if execution.venue not in venue_stats.venue_latencies:
                venue_stats.venue_latencies[execution.venue] = order.fill_latency
            else:
                # Running average
                current_avg = venue_stats.venue_latencies[execution.venue]
                venue_stats.venue_latencies[execution.venue] = (current_avg + order.fill_latency) / 2
        
        # Update fill rates (simplified - would be more sophisticated in production)
        venue_stats.fill_ratio = order.fill_ratio
    
    def _update_symbol_metrics(self, symbol_stats: ExecutionMetrics, order: Order, execution: OrderExecution) -> None:
        """Update symbol-specific metrics"""
        symbol_stats.fill_ratio = order.fill_ratio
        
        if order.fill_latency:
            symbol_stats.fill_latency = order.fill_latency
    
    def _update_real_time_stats(self, order: Order, metrics: ExecutionMetrics) -> None:
        """Update real-time performance statistics"""
        stats = self.real_time_stats
        
        # Update fill statistics
        if order.status == OrderStatus.FILLED:
            stats.total_filled += 1
        elif order.status == OrderStatus.REJECTED:
            stats.total_rejected += 1
        
        # Calculate rates
        if stats.total_orders > 0:
            stats.fill_rate = stats.total_filled / stats.total_orders
            stats.success_rate = (stats.total_orders - stats.total_rejected) / stats.total_orders
        
        # Update latency averages
        if metrics.total_execution_time > 0:
            if stats.avg_latency == 0:
                stats.avg_latency = metrics.total_execution_time
            else:
                # Exponentially weighted moving average
                alpha = 0.1
                stats.avg_latency = alpha * metrics.total_execution_time + (1 - alpha) * stats.avg_latency
        
        # Update target achievement rates
        if metrics.meets_latency_target:
            stats.latency_target_met = (stats.latency_target_met * (stats.orders_processed - 1) + 1) / stats.orders_processed
        else:
            stats.latency_target_met = (stats.latency_target_met * (stats.orders_processed - 1)) / stats.orders_processed
        
        if metrics.meets_fill_target:
            stats.fill_target_met = (stats.fill_target_met * (stats.orders_processed - 1) + 1) / stats.orders_processed
        else:
            stats.fill_target_met = (stats.fill_target_met * (stats.orders_processed - 1)) / stats.orders_processed
    
    def _update_latency_percentiles(self) -> None:
        """Update latency percentile calculations"""
        if len(self.latency_history) < 10:
            return
        
        sorted_latencies = sorted(self.latency_history)
        n = len(sorted_latencies)
        
        self.latency_percentiles = {
            'p50': sorted_latencies[int(0.50 * n)],
            'p90': sorted_latencies[int(0.90 * n)],
            'p95': sorted_latencies[int(0.95 * n)],
            'p99': sorted_latencies[int(0.99 * n)],
            'p99.9': sorted_latencies[int(0.999 * n)] if n >= 1000 else sorted_latencies[-1]
        }
    
    def _check_performance_violations(self, order: Order, metrics: ExecutionMetrics) -> None:
        """Check for performance violations and generate alerts"""
        
        # Latency violation
        if metrics.total_execution_time > self.alert_thresholds['max_latency_us']:
            self.real_time_stats.latency_violations += 1
            logger.warning(
                "Latency violation",
                order_id=order.order_id,
                latency_us=metrics.total_execution_time,
                threshold_us=self.alert_thresholds['max_latency_us']
            )
        
        # Fill rate violation
        if metrics.fill_ratio < self.alert_thresholds['min_fill_rate']:
            self.real_time_stats.fill_rate_violations += 1
            logger.warning(
                "Fill rate violation",
                order_id=order.order_id,
                fill_ratio=metrics.fill_ratio,
                threshold=self.alert_thresholds['min_fill_rate']
            )
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time performance metrics"""
        with self.lock:
            return {
                'orders': {
                    'active': len(self.active_orders),
                    'completed': len(self.completed_orders),
                    'total_processed': self.real_time_stats.total_orders,
                    'success_rate': self.real_time_stats.success_rate
                },
                'latency': {
                    'current_avg_us': self.real_time_stats.avg_latency,
                    'target_achievement_rate': self.real_time_stats.latency_target_met,
                    'violations': self.real_time_stats.latency_violations,
                    'percentiles': self.latency_percentiles.copy()
                },
                'fill_rates': {
                    'current_rate': self.real_time_stats.fill_rate,
                    'target_achievement_rate': self.real_time_stats.fill_target_met,
                    'violations': self.real_time_stats.fill_rate_violations,
                    'total_filled': self.real_time_stats.total_filled,
                    'total_rejected': self.real_time_stats.total_rejected
                },
                'performance_targets': {
                    'latency_target_us': self.latency_target_us,
                    'fill_rate_target': self.fill_rate_target,
                    'meeting_latency_target': self.real_time_stats.latency_target_met >= 0.95,
                    'meeting_fill_target': self.real_time_stats.fill_target_met >= 0.95
                }
            }
    
    def get_venue_performance(self) -> Dict[str, Any]:
        """Get venue-specific performance metrics"""
        with self.lock:
            venue_data = {}
            
            for venue, stats in self.venue_stats.items():
                venue_data[venue] = {
                    'fill_rate': stats.fill_ratio,
                    'avg_latency_us': stats.venue_latencies.get(venue, 0.0),
                    'execution_quality': stats.execution_quality_score,
                    'implementation_shortfall': stats.implementation_shortfall
                }
            
            return venue_data
    
    def get_symbol_performance(self) -> Dict[str, Any]:
        """Get symbol-specific performance metrics"""
        with self.lock:
            symbol_data = {}
            
            for symbol, stats in self.symbol_stats.items():
                symbol_data[symbol] = {
                    'fill_rate': stats.fill_ratio,
                    'avg_latency_us': stats.fill_latency,
                    'execution_quality': stats.execution_quality_score,
                    'slippage': stats.slippage
                }
            
            return symbol_data
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent execution history"""
        with self.lock:
            # Return most recent executions
            recent_executions = list(self.execution_history)[-limit:]
            return recent_executions
    
    def generate_performance_report(self, time_period: timedelta = None) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        time_period = time_period or timedelta(hours=1)
        cutoff_time = datetime.now() - time_period
        
        with self.lock:
            # Filter recent orders
            recent_orders = [
                order for order in self.completed_orders.values()
                if order.last_updated >= cutoff_time
            ]
            
            if not recent_orders:
                return {'error': 'No orders in specified time period'}
            
            # Calculate aggregate metrics
            total_orders = len(recent_orders)
            filled_orders = [o for o in recent_orders if o.status == OrderStatus.FILLED]
            rejected_orders = [o for o in recent_orders if o.status == OrderStatus.REJECTED]
            
            # Latency statistics
            latencies = [o.fill_latency for o in recent_orders if o.fill_latency]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
            
            # Fill statistics
            fill_ratios = [o.fill_ratio for o in recent_orders]
            avg_fill_ratio = sum(fill_ratios) / len(fill_ratios) if fill_ratios else 0.0
            
            return {
                'time_period': str(time_period),
                'summary': {
                    'total_orders': total_orders,
                    'filled_orders': len(filled_orders),
                    'rejected_orders': len(rejected_orders),
                    'success_rate': (total_orders - len(rejected_orders)) / total_orders,
                    'avg_fill_ratio': avg_fill_ratio,
                    'avg_latency_us': avg_latency
                },
                'performance_targets': {
                    'latency_target_met': sum(1 for l in latencies if l <= self.latency_target_us) / len(latencies) if latencies else 0.0,
                    'fill_target_met': sum(1 for r in fill_ratios if r >= self.fill_rate_target) / len(fill_ratios) if fill_ratios else 0.0
                },
                'venue_breakdown': self.get_venue_performance(),
                'symbol_breakdown': self.get_symbol_performance()
            }
    
    def reset_statistics(self) -> None:
        """Reset all tracking statistics"""
        with self.lock:
            self.real_time_stats = RealTimeStats()
            self.venue_stats.clear()
            self.symbol_stats.clear()
            self.latency_history.clear()
            self.latency_percentiles.clear()
            
            logger.info("Execution tracker statistics reset")
    
    def shutdown(self) -> None:
        """Shutdown execution tracker"""
        self.executor.shutdown(wait=True)
        logger.info("ExecutionTracker shutdown complete")