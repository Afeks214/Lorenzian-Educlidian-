"""
Performance Metrics Calculator

Comprehensive execution performance analysis including implementation shortfall,
market impact, timing costs, and execution quality metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import structlog

from ..order_management.order_types import Order, OrderExecution

logger = structlog.get_logger()


@dataclass
class ExecutionQualityMetrics:
    """Comprehensive execution quality metrics"""
    
    # Primary metrics
    implementation_shortfall: float
    market_impact: float
    timing_cost: float
    opportunity_cost: float
    
    # Cost breakdown
    commission_cost: float
    fee_cost: float
    spread_cost: float
    
    # Performance scores
    quality_score: float  # 0-100 scale
    efficiency_score: float
    speed_score: float
    
    # Benchmark comparisons
    vwap_performance: float
    twap_performance: float
    arrival_price_performance: float
    
    # Risk metrics
    slippage: float
    volatility_impact: float
    
    # Execution characteristics
    fill_rate: float
    execution_time_seconds: float
    venue_count: int


class PerformanceCalculator:
    """
    Advanced execution performance calculator.
    
    Calculates comprehensive execution metrics including:
    - Implementation Shortfall analysis
    - Market Impact estimation
    - Trading Cost Analysis (TCA)
    - Execution quality scoring
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Benchmark prices cache
        self.benchmark_cache: Dict[str, Dict[str, float]] = {}
        
        # Performance history
        self.calculation_history: List[Dict[str, Any]] = []
        
        logger.info("PerformanceCalculator initialized")
    
    def calculate_execution_quality(self, order: Order) -> Dict[str, float]:
        """Calculate comprehensive execution quality metrics"""
        
        if not order.executions:
            return self._default_metrics()
        
        try:
            # Get decision price (arrival price)
            decision_price = self._get_decision_price(order)
            
            # Calculate primary metrics
            implementation_shortfall = self._calculate_implementation_shortfall(order, decision_price)
            market_impact = self._calculate_market_impact(order, decision_price)
            timing_cost = self._calculate_timing_cost(order, decision_price)
            
            # Calculate cost components
            cost_breakdown = self._calculate_cost_breakdown(order)
            
            # Calculate benchmark performance
            benchmark_performance = self._calculate_benchmark_performance(order)
            
            # Calculate quality scores
            quality_scores = self._calculate_quality_scores(
                implementation_shortfall, market_impact, order
            )
            
            # Compile comprehensive metrics
            metrics = {
                'implementation_shortfall': implementation_shortfall,
                'market_impact': market_impact,
                'timing_cost': timing_cost,
                'opportunity_cost': abs(implementation_shortfall - market_impact),
                
                'commission_cost': cost_breakdown['commission'],
                'fee_cost': cost_breakdown['fees'],
                'spread_cost': cost_breakdown['spread'],
                
                'quality_score': quality_scores['quality'],
                'efficiency_score': quality_scores['efficiency'],
                'speed_score': quality_scores['speed'],
                
                'vwap_performance': benchmark_performance['vwap'],
                'twap_performance': benchmark_performance['twap'],
                'arrival_price_performance': benchmark_performance['arrival'],
                
                'slippage': self._calculate_slippage(order),
                'volatility_impact': self._estimate_volatility_impact(order),
                
                'fill_rate': order.fill_ratio,
                'execution_time_seconds': self._calculate_execution_time(order),
                'venue_count': len(set(ex.venue for ex in order.executions))
            }
            
            # Store calculation in history
            self.calculation_history.append({
                'order_id': order.order_id,
                'symbol': order.symbol,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating execution quality: {str(e)}")
            return self._default_metrics()
    
    def _get_decision_price(self, order: Order) -> float:
        """Get decision price (arrival price) for order"""
        
        # In production, this would be the market price at order creation time
        # For now, use a price close to the average fill price
        if order.executions:
            avg_fill = order.average_fill_price
            # Simulate decision price with small random deviation
            decision_price = avg_fill * (1 + np.random.uniform(-0.001, 0.001))
            return decision_price
        
        return order.price or 100.0  # Fallback price
    
    def _calculate_implementation_shortfall(self, order: Order, decision_price: float) -> float:
        """
        Calculate Implementation Shortfall.
        
        IS = (Average_Fill_Price - Decision_Price) / Decision_Price
        
        Positive for buy orders means paid more than decision price.
        Adjusted for order side.
        """
        
        if not order.executions or decision_price <= 0:
            return 0.0
        
        avg_fill_price = order.average_fill_price
        
        # Calculate shortfall
        if order.is_buy:
            shortfall = (avg_fill_price - decision_price) / decision_price
        else:  # Sell order
            shortfall = (decision_price - avg_fill_price) / decision_price
        
        return shortfall
    
    def _calculate_market_impact(self, order: Order, decision_price: float) -> float:
        """
        Calculate Market Impact component of execution cost.
        
        Estimates the price movement caused by the order execution.
        """
        
        if not order.executions:
            return 0.0
        
        # Simplified market impact model based on order size and volatility
        notional_value = order.filled_quantity * order.average_fill_price
        
        # Impact factors
        size_factor = min(1.0, notional_value / 1000000)  # Scale by $1M
        urgency_factor = 1.0  # Would be based on execution speed in production
        
        # Base impact (simplified square root model)
        base_impact = 0.0001 * (size_factor ** 0.5)  # 1bp base for $1M order
        
        # Adjust for market conditions (volatility proxy)
        volatility_factor = 1.0 + abs(order.average_fill_price - decision_price) / decision_price * 100
        
        market_impact = base_impact * urgency_factor * volatility_factor
        
        return market_impact
    
    def _calculate_timing_cost(self, order: Order, decision_price: float) -> float:
        """
        Calculate Timing Cost component.
        
        Cost due to delay between decision and execution.
        """
        
        if not order.executions or not order.submitted_timestamp:
            return 0.0
        
        # Calculate execution delay
        decision_time = order.created_timestamp
        avg_execution_time = sum(
            ex.timestamp.timestamp() for ex in order.executions
        ) / len(order.executions)
        
        delay_minutes = (avg_execution_time - decision_time.timestamp()) / 60
        
        # Estimate timing cost based on delay and volatility
        # Simplified model: cost increases with delay and volatility
        volatility_estimate = 0.02  # 2% daily volatility estimate
        timing_cost = volatility_estimate * (delay_minutes / (24 * 60)) ** 0.5
        
        return timing_cost
    
    def _calculate_cost_breakdown(self, order: Order) -> Dict[str, float]:
        """Calculate detailed cost breakdown"""
        
        if order.filled_quantity == 0:
            return {'commission': 0.0, 'fees': 0.0, 'spread': 0.0}
        
        # Commission and fees (actual costs)
        commission_rate = order.total_commission / order.filled_quantity
        fee_rate = order.total_fees / order.filled_quantity
        
        # Spread cost estimate
        # In production, would use actual bid-ask spread at execution time
        spread_estimate = order.average_fill_price * 0.0001  # 1bp estimate
        
        return {
            'commission': commission_rate,
            'fees': fee_rate,
            'spread': spread_estimate
        }
    
    def _calculate_benchmark_performance(self, order: Order) -> Dict[str, float]:
        """Calculate performance vs standard benchmarks"""
        
        avg_fill_price = order.average_fill_price
        
        # Simulate benchmark prices (in production, would use actual market data)
        vwap_price = avg_fill_price * (1 + np.random.uniform(-0.0005, 0.0005))
        twap_price = avg_fill_price * (1 + np.random.uniform(-0.0003, 0.0003))
        arrival_price = avg_fill_price * (1 + np.random.uniform(-0.001, 0.001))
        
        # Calculate performance vs benchmarks
        vwap_performance = (avg_fill_price - vwap_price) / vwap_price
        twap_performance = (avg_fill_price - twap_price) / twap_price
        arrival_performance = (avg_fill_price - arrival_price) / arrival_price
        
        # Adjust for order side
        if order.is_sell:
            vwap_performance = -vwap_performance
            twap_performance = -twap_performance
            arrival_performance = -arrival_performance
        
        return {
            'vwap': vwap_performance,
            'twap': twap_performance,
            'arrival': arrival_performance
        }
    
    def _calculate_quality_scores(
        self,
        implementation_shortfall: float,
        market_impact: float,
        order: Order
    ) -> Dict[str, float]:
        """Calculate execution quality scores (0-100 scale)"""
        
        # Quality score based on implementation shortfall
        # Better (lower) shortfall = higher score
        is_score = max(0, 100 - abs(implementation_shortfall) * 10000)  # Scale by bps
        
        # Efficiency score based on market impact
        efficiency_score = max(0, 100 - market_impact * 10000)
        
        # Speed score based on execution time
        execution_time = self._calculate_execution_time(order)
        target_time = 300  # 5 minutes target
        speed_score = max(0, 100 - (execution_time / target_time) * 50)
        
        # Overall quality score (weighted average)
        quality_score = (
            0.4 * is_score +
            0.3 * efficiency_score +
            0.2 * speed_score +
            0.1 * (order.fill_ratio * 100)
        )
        
        return {
            'quality': min(100, quality_score),
            'efficiency': min(100, efficiency_score),
            'speed': min(100, speed_score)
        }
    
    def _calculate_slippage(self, order: Order) -> float:
        """Calculate price slippage during execution"""
        
        if len(order.executions) < 2:
            return 0.0
        
        # Calculate price movement during execution
        first_price = order.executions[0].price
        last_price = order.executions[-1].price
        
        slippage = abs(last_price - first_price) / first_price
        
        return slippage
    
    def _estimate_volatility_impact(self, order: Order) -> float:
        """Estimate impact of volatility on execution"""
        
        if len(order.executions) < 2:
            return 0.0
        
        # Calculate price volatility during execution
        prices = [ex.price for ex in order.executions]
        
        if len(prices) < 2:
            return 0.0
        
        # Calculate returns
        returns = [
            (prices[i] / prices[i-1] - 1) for i in range(1, len(prices))
        ]
        
        # Return standard deviation as volatility measure
        if len(returns) > 1:
            return np.std(returns)
        
        return 0.0
    
    def _calculate_execution_time(self, order: Order) -> float:
        """Calculate total execution time in seconds"""
        
        if not order.executions or not order.submitted_timestamp:
            return 0.0
        
        start_time = order.submitted_timestamp
        end_time = max(ex.timestamp for ex in order.executions)
        
        return (end_time - start_time).total_seconds()
    
    def _default_metrics(self) -> Dict[str, float]:
        """Return default metrics when calculation fails"""
        
        return {
            'implementation_shortfall': 0.0,
            'market_impact': 0.0,
            'timing_cost': 0.0,
            'opportunity_cost': 0.0,
            'commission_cost': 0.0,
            'fee_cost': 0.0,
            'spread_cost': 0.0,
            'quality_score': 50.0,
            'efficiency_score': 50.0,
            'speed_score': 50.0,
            'vwap_performance': 0.0,
            'twap_performance': 0.0,
            'arrival_price_performance': 0.0,
            'slippage': 0.0,
            'volatility_impact': 0.0,
            'fill_rate': 0.0,
            'execution_time_seconds': 0.0,
            'venue_count': 0
        }
    
    def calculate_portfolio_performance(
        self,
        orders: List[Order],
        time_period: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Calculate aggregate portfolio execution performance"""
        
        if not orders:
            return {'error': 'No orders provided'}
        
        # Filter orders by time period if specified
        if time_period:
            cutoff_time = datetime.now() - time_period
            orders = [
                order for order in orders
                if order.last_updated >= cutoff_time
            ]
        
        if not orders:
            return {'error': 'No orders in specified time period'}
        
        # Calculate metrics for each order
        order_metrics = []
        for order in orders:
            if order.executions:
                metrics = self.calculate_execution_quality(order)
                order_metrics.append(metrics)
        
        if not order_metrics:
            return {'error': 'No completed orders with executions'}
        
        # Aggregate metrics
        aggregate_metrics = {}
        
        # Calculate averages
        for metric in order_metrics[0].keys():
            values = [m[metric] for m in order_metrics if isinstance(m[metric], (int, float))]
            if values:
                aggregate_metrics[f'avg_{metric}'] = sum(values) / len(values)
                aggregate_metrics[f'min_{metric}'] = min(values)
                aggregate_metrics[f'max_{metric}'] = max(values)
                aggregate_metrics[f'std_{metric}'] = np.std(values) if len(values) > 1 else 0.0
        
        # Calculate additional portfolio metrics
        total_notional = sum(
            order.filled_quantity * order.average_fill_price
            for order in orders if order.executions
        )
        
        total_commission = sum(order.total_commission for order in orders)
        total_fees = sum(order.total_fees for order in orders)
        
        avg_fill_rate = sum(order.fill_ratio for order in orders) / len(orders)
        
        return {
            'period_summary': {
                'total_orders': len(orders),
                'completed_orders': len(order_metrics),
                'total_notional': total_notional,
                'total_commission': total_commission,
                'total_fees': total_fees,
                'avg_fill_rate': avg_fill_rate
            },
            'performance_metrics': aggregate_metrics,
            'order_count': len(order_metrics)
        }
    
    def generate_tca_report(
        self,
        orders: List[Order],
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive Trade Cost Analysis report"""
        
        # Filter by symbol if specified
        if symbol:
            orders = [order for order in orders if order.symbol == symbol]
        
        if not orders:
            return {'error': 'No orders found for analysis'}
        
        # Calculate performance metrics
        portfolio_performance = self.calculate_portfolio_performance(orders)
        
        # Venue analysis
        venue_performance = self._analyze_venue_performance(orders)
        
        # Time-based analysis
        time_analysis = self._analyze_time_patterns(orders)
        
        # Order size analysis
        size_analysis = self._analyze_order_size_impact(orders)
        
        return {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'symbol_filter': symbol,
                'total_orders_analyzed': len(orders),
                'date_range': {
                    'start': min(order.created_timestamp for order in orders).isoformat(),
                    'end': max(order.last_updated for order in orders).isoformat()
                }
            },
            'portfolio_performance': portfolio_performance,
            'venue_analysis': venue_performance,
            'time_analysis': time_analysis,
            'size_analysis': size_analysis,
            'summary_insights': self._generate_summary_insights(orders)
        }
    
    def _analyze_venue_performance(self, orders: List[Order]) -> Dict[str, Any]:
        """Analyze performance by execution venue"""
        
        venue_data = {}
        
        for order in orders:
            if not order.executions:
                continue
            
            for execution in order.executions:
                venue = execution.venue
                
                if venue not in venue_data:
                    venue_data[venue] = {
                        'orders': [],
                        'total_quantity': 0,
                        'total_notional': 0
                    }
                
                venue_data[venue]['orders'].append(order)
                venue_data[venue]['total_quantity'] += execution.quantity
                venue_data[venue]['total_notional'] += execution.quantity * execution.price
        
        # Calculate venue metrics
        venue_analysis = {}
        
        for venue, data in venue_data.items():
            orders_for_venue = data['orders']
            
            if orders_for_venue:
                # Calculate average metrics for this venue
                venue_metrics = []
                for order in orders_for_venue:
                    metrics = self.calculate_execution_quality(order)
                    venue_metrics.append(metrics)
                
                if venue_metrics:
                    venue_analysis[venue] = {
                        'order_count': len(orders_for_venue),
                        'total_quantity': data['total_quantity'],
                        'total_notional': data['total_notional'],
                        'avg_implementation_shortfall': sum(
                            m['implementation_shortfall'] for m in venue_metrics
                        ) / len(venue_metrics),
                        'avg_quality_score': sum(
                            m['quality_score'] for m in venue_metrics
                        ) / len(venue_metrics),
                        'avg_fill_rate': sum(
                            order.fill_ratio for order in orders_for_venue
                        ) / len(orders_for_venue)
                    }
        
        return venue_analysis
    
    def _analyze_time_patterns(self, orders: List[Order]) -> Dict[str, Any]:
        """Analyze execution performance by time patterns"""
        
        hourly_performance = {}
        
        for order in orders:
            if not order.executions:
                continue
            
            # Use first execution time for time analysis
            execution_hour = order.executions[0].timestamp.hour
            
            if execution_hour not in hourly_performance:
                hourly_performance[execution_hour] = []
            
            metrics = self.calculate_execution_quality(order)
            hourly_performance[execution_hour].append(metrics)
        
        # Calculate hourly averages
        hourly_analysis = {}
        
        for hour, metrics_list in hourly_performance.items():
            if metrics_list:
                hourly_analysis[hour] = {
                    'order_count': len(metrics_list),
                    'avg_implementation_shortfall': sum(
                        m['implementation_shortfall'] for m in metrics_list
                    ) / len(metrics_list),
                    'avg_quality_score': sum(
                        m['quality_score'] for m in metrics_list
                    ) / len(metrics_list)
                }
        
        return {
            'hourly_analysis': hourly_analysis,
            'best_execution_hour': max(
                hourly_analysis.keys(),
                key=lambda h: hourly_analysis[h]['avg_quality_score']
            ) if hourly_analysis else None
        }
    
    def _analyze_order_size_impact(self, orders: List[Order]) -> Dict[str, Any]:
        """Analyze execution performance by order size"""
        
        # Define size buckets
        size_buckets = {
            'small': (0, 1000),
            'medium': (1000, 10000),
            'large': (10000, 100000),
            'xlarge': (100000, float('inf'))
        }
        
        bucket_performance = {bucket: [] for bucket in size_buckets}
        
        for order in orders:
            if not order.executions:
                continue
            
            # Categorize order by size
            for bucket, (min_size, max_size) in size_buckets.items():
                if min_size <= order.quantity < max_size:
                    metrics = self.calculate_execution_quality(order)
                    bucket_performance[bucket].append(metrics)
                    break
        
        # Calculate bucket averages
        bucket_analysis = {}
        
        for bucket, metrics_list in bucket_performance.items():
            if metrics_list:
                bucket_analysis[bucket] = {
                    'order_count': len(metrics_list),
                    'avg_implementation_shortfall': sum(
                        m['implementation_shortfall'] for m in metrics_list
                    ) / len(metrics_list),
                    'avg_market_impact': sum(
                        m['market_impact'] for m in metrics_list
                    ) / len(metrics_list),
                    'avg_quality_score': sum(
                        m['quality_score'] for m in metrics_list
                    ) / len(metrics_list)
                }
        
        return bucket_analysis
    
    def _generate_summary_insights(self, orders: List[Order]) -> List[str]:
        """Generate summary insights from analysis"""
        
        insights = []
        
        # Calculate overall performance
        completed_orders = [order for order in orders if order.executions]
        
        if completed_orders:
            avg_fill_rate = sum(order.fill_ratio for order in completed_orders) / len(completed_orders)
            
            if avg_fill_rate >= 0.995:
                insights.append("Excellent fill rate performance (>99.5%)")
            elif avg_fill_rate >= 0.98:
                insights.append("Good fill rate performance (>98%)")
            else:
                insights.append(f"Fill rate below target: {avg_fill_rate:.1%}")
            
            # Implementation shortfall analysis
            is_values = []
            for order in completed_orders:
                metrics = self.calculate_execution_quality(order)
                is_values.append(abs(metrics['implementation_shortfall']))
            
            if is_values:
                avg_is = sum(is_values) / len(is_values)
                if avg_is <= 0.0005:  # 5 bps
                    insights.append("Implementation shortfall within target (<5bps)")
                else:
                    insights.append(f"Implementation shortfall above target: {avg_is*10000:.1f}bps")
        
        return insights
    
    def get_calculation_statistics(self) -> Dict[str, Any]:
        """Get performance calculation statistics"""
        
        return {
            'total_calculations': len(self.calculation_history),
            'calculation_history_size': len(self.calculation_history),
            'recent_calculations': self.calculation_history[-10:] if self.calculation_history else []
        }