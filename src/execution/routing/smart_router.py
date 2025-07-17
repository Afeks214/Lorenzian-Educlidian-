"""
Smart Order Router

Intelligent order routing system that optimizes execution across multiple venues.
Integrates venue management, algorithm execution, and real-time decision making.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import structlog

from ..order_management.order_types import Order, OrderType, OrderSide
from .venue_manager import VenueManager, VenueConfig, VenueType
from .algorithm_engine import AlgorithmEngine, AlgorithmType, AlgorithmConfig
from .routing_optimizer import RoutingOptimizer

logger = structlog.get_logger()


class RoutingStrategy(Enum):
    """Order routing strategies"""
    SMART = "SMART"                     # Intelligent routing
    COST_MINIMIZATION = "COST_MIN"      # Minimize execution costs
    SPEED_OPTIMIZATION = "SPEED_OPT"    # Minimize latency
    IMPACT_MINIMIZATION = "IMPACT_MIN"  # Minimize market impact
    DARK_POOL_PREFERENCE = "DARK_PREF"  # Prefer dark pools
    VENUE_SPECIFIC = "VENUE_SPEC"       # Route to specific venue
    ALGORITHM_DRIVEN = "ALGO_DRIVEN"    # Use execution algorithms


@dataclass
class RoutingResult:
    """Result of order routing decision"""
    
    # Routing decision
    venue_id: str
    strategy: RoutingStrategy
    algorithm_type: Optional[AlgorithmType] = None
    
    # Execution parameters
    expected_latency_ms: float = 0.0
    expected_cost_bps: float = 0.0
    expected_market_impact_bps: float = 0.0
    expected_fill_rate: float = 0.0
    
    # Venue details
    venue_name: str = ""
    venue_type: str = ""
    
    # Routing metadata
    routing_time_us: float = 0.0
    considered_venues: List[str] = None
    routing_reason: str = ""
    confidence_score: float = 0.0
    
    # Risk assessment
    risk_score: float = 0.0
    risk_factors: List[str] = None
    
    # Alternative options
    backup_venues: List[str] = None
    
    def __post_init__(self):
        if self.considered_venues is None:
            self.considered_venues = []
        if self.risk_factors is None:
            self.risk_factors = []
        if self.backup_venues is None:
            self.backup_venues = []


@dataclass
class SmartRouterConfig:
    """Configuration for smart order router"""
    
    # Default routing preferences
    default_strategy: RoutingStrategy = RoutingStrategy.SMART
    enable_algorithms: bool = True
    enable_dark_pools: bool = True
    
    # Performance targets
    max_routing_latency_us: float = 100.0  # 100μs routing decision
    target_fill_rate: float = 0.998        # >99.8% fill rate
    max_market_impact_bps: float = 5.0     # <5bps market impact
    
    # Cost optimization
    cost_weight: float = 0.3               # Weight for cost minimization
    speed_weight: float = 0.4              # Weight for speed optimization
    impact_weight: float = 0.3             # Weight for impact minimization
    
    # Algorithm thresholds
    algo_threshold_quantity: int = 10000   # Use algos for orders >10k shares
    algo_threshold_value: float = 1000000  # Use algos for orders >$1M
    
    # Dark pool settings
    dark_pool_min_size: int = 500          # Minimum size for dark pools
    dark_pool_preference: float = 0.3      # Preference factor for dark pools
    
    # Risk controls
    max_venue_concentration: float = 0.5   # Max 50% of volume to single venue
    enable_venue_failover: bool = True     # Enable automatic failover
    
    # Performance monitoring
    enable_routing_analytics: bool = True
    routing_decision_logging: bool = True


class SmartOrderRouter:
    """
    Intelligent order routing system for optimal execution.
    
    Provides real-time routing decisions based on:
    - Current market conditions
    - Venue performance metrics
    - Order characteristics
    - Cost optimization
    - Execution algorithms
    """
    
    def __init__(
        self,
        config: SmartRouterConfig,
        venue_manager: VenueManager,
        algorithm_engine: Optional[AlgorithmEngine] = None,
        routing_optimizer: Optional[RoutingOptimizer] = None
    ):
        self.config = config
        self.venue_manager = venue_manager
        self.algorithm_engine = algorithm_engine or AlgorithmEngine()
        self.routing_optimizer = routing_optimizer or RoutingOptimizer()
        
        # Routing statistics
        self.routing_stats = {
            'total_routed': 0,
            'by_strategy': {strategy: 0 for strategy in RoutingStrategy},
            'by_venue': {},
            'avg_routing_latency_us': 0.0,
            'routing_errors': 0
        }
        
        # Performance tracking
        self.venue_performance_cache = {}
        self.last_performance_update = datetime.now()
        
        logger.info(
            "SmartOrderRouter initialized",
            default_strategy=config.default_strategy.value,
            enable_algorithms=config.enable_algorithms
        )
    
    async def route_order(self, order: Order) -> RoutingResult:
        """
        Route order to optimal venue with comprehensive analysis.
        Target: <100μs routing decision time.
        """
        start_time = time.perf_counter()
        
        try:
            # Determine routing strategy
            strategy = self._determine_routing_strategy(order)
            
            # Get available venues
            venue_requirements = self._build_venue_requirements(order)
            available_venues = self.venue_manager.get_available_venues(venue_requirements)
            
            if not available_venues:
                raise RuntimeError("No available venues for order")
            
            # Route based on strategy
            routing_result = await self._execute_routing_strategy(
                order, strategy, available_venues
            )
            
            # Calculate routing time
            routing_time = (time.perf_counter() - start_time) * 1_000_000
            routing_result.routing_time_us = routing_time
            
            # Update statistics
            self._update_routing_stats(routing_result)
            
            # Log routing decision
            if self.config.routing_decision_logging:
                logger.info(
                    "Order routed",
                    order_id=order.order_id,
                    symbol=order.symbol,
                    venue=routing_result.venue_id,
                    strategy=strategy.value,
                    routing_time_us=routing_time,
                    confidence=routing_result.confidence_score
                )
            
            return routing_result
            
        except Exception as e:
            routing_time = (time.perf_counter() - start_time) * 1_000_000
            self.routing_stats['routing_errors'] += 1
            
            logger.error(
                "Order routing failed",
                order_id=order.order_id,
                error=str(e),
                routing_time_us=routing_time
            )
            raise
    
    async def route_order_fast(self, order: Order) -> RoutingResult:
        """
        Ultra-fast routing path for high-priority orders.
        Target: <50μs routing decision time.
        """
        start_time = time.perf_counter()
        
        try:
            # Simplified fast path routing
            available_venues = self.venue_manager.get_available_venues()
            
            if not available_venues:
                raise RuntimeError("No available venues")
            
            # Select first available venue with best recent performance
            best_venue = self._select_fast_venue(available_venues)
            
            # Create minimal routing result
            routing_result = RoutingResult(
                venue_id=best_venue,
                strategy=RoutingStrategy.SPEED_OPTIMIZATION,
                expected_latency_ms=10.0,  # Estimate
                expected_fill_rate=0.95,   # Conservative estimate
                routing_reason="Fast path routing",
                confidence_score=0.8
            )
            
            routing_time = (time.perf_counter() - start_time) * 1_000_000
            routing_result.routing_time_us = routing_time
            
            logger.debug(
                "Fast path routing",
                order_id=order.order_id,
                venue=best_venue,
                routing_time_us=routing_time
            )
            
            return routing_result
            
        except Exception as e:
            logger.error(f"Fast path routing failed: {str(e)}")
            raise
    
    def _determine_routing_strategy(self, order: Order) -> RoutingStrategy:
        """Determine optimal routing strategy for order"""
        
        # Algorithm-driven for large orders
        if (self.config.enable_algorithms and 
            (order.quantity >= self.config.algo_threshold_quantity or 
             order.notional_value >= self.config.algo_threshold_value)):
            return RoutingStrategy.ALGORITHM_DRIVEN
        
        # Speed optimization for urgent orders
        if hasattr(order, 'priority') and order.priority.value >= 4:
            return RoutingStrategy.SPEED_OPTIMIZATION
        
        # Market orders -> speed optimization
        if order.order_type == OrderType.MARKET:
            return RoutingStrategy.SPEED_OPTIMIZATION
        
        # Large quantities -> impact minimization
        if order.quantity > 50000:
            return RoutingStrategy.IMPACT_MINIMIZATION
        
        # Default to smart routing
        return self.config.default_strategy
    
    def _build_venue_requirements(self, order: Order) -> Dict[str, Any]:
        """Build venue requirements based on order characteristics"""
        
        return {
            'quantity': order.quantity,
            'notional_value': order.notional_value,
            'order_type': order.order_type.value,
            'side': order.side.value,
            'urgency': getattr(order.priority, 'value', 2)
        }
    
    async def _execute_routing_strategy(
        self,
        order: Order,
        strategy: RoutingStrategy,
        available_venues: List[str]
    ) -> RoutingResult:
        """Execute specific routing strategy"""
        
        if strategy == RoutingStrategy.SMART:
            return await self._smart_routing(order, available_venues)
        elif strategy == RoutingStrategy.COST_MINIMIZATION:
            return await self._cost_minimization_routing(order, available_venues)
        elif strategy == RoutingStrategy.SPEED_OPTIMIZATION:
            return await self._speed_optimization_routing(order, available_venues)
        elif strategy == RoutingStrategy.IMPACT_MINIMIZATION:
            return await self._impact_minimization_routing(order, available_venues)
        elif strategy == RoutingStrategy.DARK_POOL_PREFERENCE:
            return await self._dark_pool_routing(order, available_venues)
        elif strategy == RoutingStrategy.ALGORITHM_DRIVEN:
            return await self._algorithm_driven_routing(order, available_venues)
        else:
            # Fallback to smart routing
            return await self._smart_routing(order, available_venues)
    
    async def _smart_routing(self, order: Order, available_venues: List[str]) -> RoutingResult:
        """Intelligent routing using multiple optimization criteria"""
        
        # Get current venue performance
        venue_performance = self.venue_manager.get_venue_performance()
        
        # Score venues using weighted criteria
        venue_scores = {}
        
        for venue_id in available_venues:
            config = self.venue_manager.venues[venue_id]
            performance = venue_performance.get(venue_id)
            
            # Cost score
            total_cost = config.cost_per_share - config.rebate_per_share
            cost_score = 1.0 / (1.0 + total_cost * 1000)
            
            # Speed score
            if performance and performance.avg_latency > 0:
                speed_score = 1.0 / (1.0 + performance.avg_latency)
            else:
                speed_score = 1.0 / (1.0 + config.expected_latency_ms)
            
            # Impact score
            if performance and performance.market_impact_bps > 0:
                impact_score = 1.0 / (1.0 + performance.market_impact_bps)
            else:
                impact_score = 1.0 / (1.0 + config.avg_market_impact_bps)
            
            # Fill rate score
            if performance:
                fill_score = performance.fill_rate
            else:
                fill_score = config.typical_fill_rate
            
            # Weighted composite score
            composite_score = (
                self.config.cost_weight * cost_score +
                self.config.speed_weight * speed_score +
                self.config.impact_weight * impact_score +
                0.2 * fill_score  # Fill rate always important
            )
            
            venue_scores[venue_id] = composite_score
        
        # Select best venue
        best_venue = max(venue_scores, key=venue_scores.get)
        best_config = self.venue_manager.venues[best_venue]
        best_performance = venue_performance.get(best_venue)
        
        return RoutingResult(
            venue_id=best_venue,
            strategy=RoutingStrategy.SMART,
            expected_latency_ms=best_performance.avg_latency if best_performance else best_config.expected_latency_ms,
            expected_cost_bps=(best_config.cost_per_share - best_config.rebate_per_share) * 100,
            expected_market_impact_bps=best_performance.market_impact_bps if best_performance else best_config.avg_market_impact_bps,
            expected_fill_rate=best_performance.fill_rate if best_performance else best_config.typical_fill_rate,
            venue_name=best_config.name,
            venue_type=best_config.venue_type.value,
            considered_venues=list(venue_scores.keys()),
            routing_reason=f"Smart routing - composite score: {venue_scores[best_venue]:.3f}",
            confidence_score=min(venue_scores[best_venue], 1.0),
            backup_venues=sorted(venue_scores.keys(), key=venue_scores.get, reverse=True)[1:3]
        )
    
    async def _cost_minimization_routing(self, order: Order, available_venues: List[str]) -> RoutingResult:
        """Route to venue with lowest execution costs"""
        
        venue_costs = {}
        
        for venue_id in available_venues:
            config = self.venue_manager.venues[venue_id]
            
            # Calculate total cost including commissions and fees
            base_cost = config.cost_per_share - config.rebate_per_share
            
            # Add market impact estimate
            impact_cost = config.avg_market_impact_bps / 10000  # Convert bps to price factor
            
            total_cost = base_cost + impact_cost
            venue_costs[venue_id] = total_cost
        
        # Select venue with lowest cost
        best_venue = min(venue_costs, key=venue_costs.get)
        best_config = self.venue_manager.venues[best_venue]
        
        return RoutingResult(
            venue_id=best_venue,
            strategy=RoutingStrategy.COST_MINIMIZATION,
            expected_cost_bps=venue_costs[best_venue] * 100,
            expected_fill_rate=best_config.typical_fill_rate,
            routing_reason=f"Cost minimization - total cost: {venue_costs[best_venue]:.4f}",
            confidence_score=0.9
        )
    
    async def _speed_optimization_routing(self, order: Order, available_venues: List[str]) -> RoutingResult:
        """Route to venue with lowest latency"""
        
        venue_latencies = {}
        venue_performance = self.venue_manager.get_venue_performance()
        
        for venue_id in available_venues:
            config = self.venue_manager.venues[venue_id]
            performance = venue_performance.get(venue_id)
            
            # Use actual performance if available, otherwise config estimate
            latency = performance.avg_latency if performance else config.expected_latency_ms
            venue_latencies[venue_id] = latency
        
        # Select venue with lowest latency
        best_venue = min(venue_latencies, key=venue_latencies.get)
        best_config = self.venue_manager.venues[best_venue]
        
        return RoutingResult(
            venue_id=best_venue,
            strategy=RoutingStrategy.SPEED_OPTIMIZATION,
            expected_latency_ms=venue_latencies[best_venue],
            expected_fill_rate=best_config.typical_fill_rate,
            routing_reason=f"Speed optimization - latency: {venue_latencies[best_venue]:.1f}ms",
            confidence_score=0.85
        )
    
    async def _impact_minimization_routing(self, order: Order, available_venues: List[str]) -> RoutingResult:
        """Route to venue with minimal market impact"""
        
        venue_impacts = {}
        venue_performance = self.venue_manager.get_venue_performance()
        
        for venue_id in available_venues:
            config = self.venue_manager.venues[venue_id]
            performance = venue_performance.get(venue_id)
            
            # Use actual market impact if available
            impact = performance.market_impact_bps if performance else config.avg_market_impact_bps
            
            # Adjust for venue type (dark pools typically have lower impact)
            if config.venue_type == VenueType.DARK_POOL:
                impact *= 0.7  # 30% lower impact for dark pools
            
            venue_impacts[venue_id] = impact
        
        # Select venue with lowest impact
        best_venue = min(venue_impacts, key=venue_impacts.get)
        best_config = self.venue_manager.venues[best_venue]
        
        return RoutingResult(
            venue_id=best_venue,
            strategy=RoutingStrategy.IMPACT_MINIMIZATION,
            expected_market_impact_bps=venue_impacts[best_venue],
            expected_fill_rate=best_config.typical_fill_rate,
            routing_reason=f"Impact minimization - impact: {venue_impacts[best_venue]:.1f}bps",
            confidence_score=0.8
        )
    
    async def _dark_pool_routing(self, order: Order, available_venues: List[str]) -> RoutingResult:
        """Route to dark pools when appropriate"""
        
        # Filter for dark pools
        dark_pools = [
            venue_id for venue_id in available_venues
            if self.venue_manager.venues[venue_id].venue_type == VenueType.DARK_POOL
        ]
        
        if not dark_pools:
            # Fallback to regular venues if no dark pools available
            return await self._smart_routing(order, available_venues)
        
        # Check if order meets dark pool minimum size
        suitable_dark_pools = []
        for venue_id in dark_pools:
            config = self.venue_manager.venues[venue_id]
            if order.quantity >= self.config.dark_pool_min_size:
                suitable_dark_pools.append(venue_id)
        
        if not suitable_dark_pools:
            return await self._smart_routing(order, available_venues)
        
        # Select best dark pool based on performance
        best_dark_pool = self.venue_manager.select_best_venue(
            order.symbol,
            self._build_venue_requirements(order),
            "cost"
        )
        
        if best_dark_pool in suitable_dark_pools:
            config = self.venue_manager.venues[best_dark_pool]
            
            return RoutingResult(
                venue_id=best_dark_pool,
                strategy=RoutingStrategy.DARK_POOL_PREFERENCE,
                expected_market_impact_bps=config.avg_market_impact_bps * 0.7,  # Lower impact
                expected_fill_rate=config.typical_fill_rate * 0.8,  # Lower fill rate
                routing_reason="Dark pool routing for impact minimization",
                confidence_score=0.75
            )
        
        return await self._smart_routing(order, available_venues)
    
    async def _algorithm_driven_routing(self, order: Order, available_venues: List[str]) -> RoutingResult:
        """Route using execution algorithms"""
        
        if not self.config.enable_algorithms:
            return await self._smart_routing(order, available_venues)
        
        # Determine best algorithm based on order characteristics
        algorithm_type = self._select_algorithm(order)
        
        # Select venue suitable for algorithmic execution
        algo_suitable_venues = [
            venue_id for venue_id in available_venues
            if self.venue_manager.venues[venue_id].supports_algo_orders
        ]
        
        if not algo_suitable_venues:
            # Fallback if no venues support algorithms
            return await self._smart_routing(order, available_venues)
        
        # Select best venue for algorithmic execution
        best_venue = self.venue_manager.select_best_venue(
            order.symbol,
            self._build_venue_requirements(order),
            "overall"
        )
        
        if best_venue not in algo_suitable_venues:
            best_venue = algo_suitable_venues[0]
        
        config = self.venue_manager.venues[best_venue]
        
        return RoutingResult(
            venue_id=best_venue,
            strategy=RoutingStrategy.ALGORITHM_DRIVEN,
            algorithm_type=algorithm_type,
            expected_latency_ms=config.expected_latency_ms * 2,  # Algorithms take longer
            expected_market_impact_bps=config.avg_market_impact_bps * 0.6,  # Lower impact
            expected_fill_rate=config.typical_fill_rate * 0.95,
            routing_reason=f"Algorithm routing - {algorithm_type.value}",
            confidence_score=0.9
        )
    
    def _select_algorithm(self, order: Order) -> AlgorithmType:
        """Select appropriate algorithm for order"""
        
        # Large orders -> VWAP for volume matching
        if order.quantity > 100000:
            return AlgorithmType.VWAP
        
        # Medium orders -> TWAP for steady execution
        elif order.quantity > 10000:
            return AlgorithmType.TWAP
        
        # Smaller orders -> Implementation Shortfall for cost optimization
        else:
            return AlgorithmType.IMPLEMENTATION_SHORTFALL
    
    def _select_fast_venue(self, available_venues: List[str]) -> str:
        """Fast venue selection for high-priority orders"""
        
        # Use cached performance data for speed
        if not hasattr(self, '_fast_venue_cache'):
            self._fast_venue_cache = {}
            self._fast_venue_cache_time = datetime.now()
        
        # Refresh cache every 30 seconds
        if (datetime.now() - self._fast_venue_cache_time).total_seconds() > 30:
            venue_rankings = self.venue_manager.get_venue_rankings("latency")
            self._fast_venue_cache = {venue: score for venue, score in venue_rankings}
            self._fast_venue_cache_time = datetime.now()
        
        # Select venue with best latency score from available venues
        best_venue = None
        best_score = 0.0
        
        for venue_id in available_venues:
            score = self._fast_venue_cache.get(venue_id, 0.0)
            if score > best_score:
                best_score = score
                best_venue = venue_id
        
        return best_venue or available_venues[0]  # Fallback to first available
    
    def _update_routing_stats(self, routing_result: RoutingResult) -> None:
        """Update routing statistics"""
        
        self.routing_stats['total_routed'] += 1
        self.routing_stats['by_strategy'][routing_result.strategy] += 1
        
        if routing_result.venue_id in self.routing_stats['by_venue']:
            self.routing_stats['by_venue'][routing_result.venue_id] += 1
        else:
            self.routing_stats['by_venue'][routing_result.venue_id] = 1
        
        # Update average routing latency
        current_avg = self.routing_stats['avg_routing_latency_us']
        total_routed = self.routing_stats['total_routed']
        
        self.routing_stats['avg_routing_latency_us'] = (
            (current_avg * (total_routed - 1) + routing_result.routing_time_us) / total_routed
        )
    
    async def execute_order(self, order: Order) -> Dict[str, Any]:
        """Execute order through selected routing"""
        
        # Route order
        routing_result = await self.route_order(order)
        
        # Execute based on routing decision
        if routing_result.algorithm_type:
            # Use algorithm execution
            algorithm_config = AlgorithmConfig(
                algorithm_type=routing_result.algorithm_type,
                duration_minutes=60,  # Default duration
                max_participation_rate=0.20
            )
            
            algorithm_id = await self.algorithm_engine.start_algorithm(order, algorithm_config)
            
            return {
                'execution_type': 'algorithm',
                'algorithm_id': algorithm_id,
                'routing_result': routing_result,
                'venue_id': routing_result.venue_id
            }
        
        else:
            # Direct venue execution
            order_data = {
                'order_id': order.order_id,
                'symbol': order.symbol,
                'side': order.side.value,
                'quantity': order.quantity,
                'order_type': order.order_type.value,
                'price': order.price
            }
            
            execution_result = await self.venue_manager.submit_order_to_venue(
                routing_result.venue_id,
                order_data
            )
            
            return {
                'execution_type': 'direct',
                'execution_result': execution_result,
                'routing_result': routing_result,
                'venue_id': routing_result.venue_id
            }
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics"""
        
        return {
            'routing_stats': self.routing_stats.copy(),
            'venue_performance': self.venue_manager.get_venue_performance(),
            'algorithm_performance': self.algorithm_engine.get_performance_summary() if self.algorithm_engine else None,
            'config': {
                'default_strategy': self.config.default_strategy.value,
                'enable_algorithms': self.config.enable_algorithms,
                'enable_dark_pools': self.config.enable_dark_pools,
                'max_routing_latency_us': self.config.max_routing_latency_us
            }
        }
    
    def get_venue_recommendations(self, order_characteristics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get venue recommendations for given order characteristics"""
        
        # Build mock order for analysis
        from ..order_management.order_types import OrderRequest, OrderSide, OrderType
        
        mock_request = OrderRequest(
            symbol=order_characteristics.get('symbol', 'TEST'),
            side=OrderSide.BUY,
            quantity=order_characteristics.get('quantity', 100),
            order_type=OrderType.MARKET
        )
        mock_order = mock_request.to_order()
        
        # Get venue requirements
        requirements = self._build_venue_requirements(mock_order)
        available_venues = self.venue_manager.get_available_venues(requirements)
        
        # Score all available venues
        recommendations = []
        venue_performance = self.venue_manager.get_venue_performance()
        
        for venue_id in available_venues:
            config = self.venue_manager.venues[venue_id]
            performance = venue_performance.get(venue_id)
            
            recommendation = {
                'venue_id': venue_id,
                'venue_name': config.name,
                'venue_type': config.venue_type.value,
                'expected_cost_bps': (config.cost_per_share - config.rebate_per_share) * 100,
                'expected_latency_ms': performance.avg_latency if performance else config.expected_latency_ms,
                'expected_fill_rate': performance.fill_rate if performance else config.typical_fill_rate,
                'supports_algorithms': config.supports_algo_orders,
                'is_dark_pool': config.venue_type == VenueType.DARK_POOL
            }
            
            recommendations.append(recommendation)
        
        # Sort by overall quality score
        def quality_score(rec):
            cost_score = 1.0 / (1.0 + rec['expected_cost_bps'])
            speed_score = 1.0 / (1.0 + rec['expected_latency_ms'])
            fill_score = rec['expected_fill_rate']
            return (cost_score + speed_score + fill_score) / 3.0
        
        recommendations.sort(key=quality_score, reverse=True)
        
        return recommendations
    
    async def shutdown(self) -> None:
        """Shutdown smart order router"""
        logger.info("Shutting down smart order router")
        
        if self.algorithm_engine:
            await self.algorithm_engine.shutdown()
        
        await self.venue_manager.shutdown()
        
        logger.info("Smart order router shutdown complete")