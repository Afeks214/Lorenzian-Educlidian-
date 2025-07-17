"""
Venue Manager

Manages connections and performance tracking for multiple execution venues.
Provides real-time venue selection based on latency, fill rates, and costs.
"""

import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import structlog

logger = structlog.get_logger()


class VenueType(Enum):
    """Types of execution venues"""
    EXCHANGE = "EXCHANGE"           # Primary exchanges (NYSE, NASDAQ)
    ECN = "ECN"                    # Electronic Communication Networks
    DARK_POOL = "DARK_POOL"        # Dark pools
    RETAIL_MAKER = "RETAIL_MAKER"  # Retail market makers
    BROKER = "BROKER"              # Broker networks


@dataclass
class VenueConfig:
    """Configuration for a trading venue"""
    
    # Basic information
    venue_id: str
    name: str
    venue_type: VenueType
    
    # Connection details
    api_endpoint: str
    backup_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    
    # Performance characteristics
    expected_latency_ms: float = 10.0
    max_order_size: int = 1000000
    min_order_size: int = 1
    cost_per_share: float = 0.002
    rebate_per_share: float = 0.0
    
    # Market hours
    open_time: str = "09:30"
    close_time: str = "16:00"
    timezone: str = "US/Eastern"
    
    # Capabilities
    supports_market_orders: bool = True
    supports_limit_orders: bool = True
    supports_stop_orders: bool = True
    supports_iceberg: bool = False
    supports_algo_orders: bool = False
    
    # Risk settings
    max_daily_volume: int = 10000000
    max_order_value: float = 50000000
    position_limit: int = 5000000
    
    # Quality metrics
    typical_fill_rate: float = 0.98
    avg_market_impact_bps: float = 2.0
    
    # Status
    enabled: bool = True
    priority: int = 1  # Lower number = higher priority


@dataclass
class VenuePerformance:
    """Real-time venue performance metrics"""
    
    # Latency metrics (in ms)
    avg_latency: float = 0.0
    p95_latency: float = 0.0
    connection_latency: float = 0.0
    
    # Fill metrics
    fill_rate: float = 0.0
    rejection_rate: float = 0.0
    partial_fill_rate: float = 0.0
    
    # Cost metrics
    avg_effective_spread: float = 0.0
    market_impact_bps: float = 0.0
    total_cost_bps: float = 0.0
    
    # Volume metrics
    total_volume: int = 0
    total_orders: int = 0
    avg_order_size: int = 0
    
    # Status
    is_connected: bool = False
    last_heartbeat: Optional[datetime] = None
    error_count: int = 0
    
    # Time windows
    last_updated: datetime = field(default_factory=datetime.now)
    measurement_window: timedelta = field(default=timedelta(minutes=15))


class VenueConnection:
    """Connection wrapper for a trading venue"""
    
    def __init__(self, config: VenueConfig):
        self.config = config
        self.is_connected = False
        self.last_heartbeat = None
        self.connection_time = None
        self.error_count = 0
        
    async def connect(self) -> bool:
        """Connect to venue"""
        try:
            start_time = time.perf_counter()
            
            # Simulate connection (replace with actual venue API connection)
            await asyncio.sleep(0.001)  # Simulate connection latency
            
            self.is_connected = True
            self.connection_time = time.perf_counter() - start_time
            self.last_heartbeat = datetime.now()
            self.error_count = 0
            
            logger.info(
                "Connected to venue",
                venue_id=self.config.venue_id,
                connection_time_ms=self.connection_time * 1000
            )
            
            return True
            
        except Exception as e:
            self.error_count += 1
            logger.error(
                "Failed to connect to venue",
                venue_id=self.config.venue_id,
                error=str(e),
                error_count=self.error_count
            )
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from venue"""
        self.is_connected = False
        self.last_heartbeat = None
        logger.info("Disconnected from venue", venue_id=self.config.venue_id)
    
    async def submit_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit order to venue"""
        if not self.is_connected:
            raise ConnectionError(f"Not connected to venue {self.config.venue_id}")
        
        start_time = time.perf_counter()
        
        try:
            # Simulate order submission (replace with actual venue API call)
            await asyncio.sleep(0.002)  # Simulate submission latency
            
            submission_latency = (time.perf_counter() - start_time) * 1000
            
            # Mock response
            response = {
                'venue_order_id': f"{self.config.venue_id}_{int(time.time() * 1000000)}",
                'status': 'SUBMITTED',
                'timestamp': datetime.now().isoformat(),
                'latency_ms': submission_latency
            }
            
            self.last_heartbeat = datetime.now()
            
            return response
            
        except Exception as e:
            self.error_count += 1
            logger.error(
                "Order submission failed",
                venue_id=self.config.venue_id,
                error=str(e)
            )
            raise
    
    async def cancel_order(self, venue_order_id: str) -> Dict[str, Any]:
        """Cancel order at venue"""
        if not self.is_connected:
            raise ConnectionError(f"Not connected to venue {self.config.venue_id}")
        
        try:
            # Simulate cancellation
            await asyncio.sleep(0.001)
            
            response = {
                'venue_order_id': venue_order_id,
                'status': 'CANCELLED',
                'timestamp': datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            self.error_count += 1
            logger.error("Order cancellation failed", venue_id=self.config.venue_id, error=str(e))
            raise
    
    async def get_order_status(self, venue_order_id: str) -> Dict[str, Any]:
        """Get order status from venue"""
        if not self.is_connected:
            raise ConnectionError(f"Not connected to venue {self.config.venue_id}")
        
        # Simulate status check
        await asyncio.sleep(0.0005)
        
        return {
            'venue_order_id': venue_order_id,
            'status': 'FILLED',  # Mock status
            'filled_quantity': 100,
            'avg_price': 150.25,
            'timestamp': datetime.now().isoformat()
        }


class VenueManager:
    """
    Manages multiple trading venues with real-time performance tracking.
    
    Provides intelligent venue selection based on:
    - Current latency
    - Fill rates
    - Market impact
    - Cost structure
    - Available liquidity
    """
    
    def __init__(self, venue_configs: List[VenueConfig]):
        self.venues: Dict[str, VenueConfig] = {config.venue_id: config for config in venue_configs}
        self.connections: Dict[str, VenueConnection] = {}
        self.performance: Dict[str, VenuePerformance] = {}
        
        # Performance tracking
        self.latency_history: Dict[str, List[float]] = {venue_id: [] for venue_id in self.venues}
        self.fill_history: Dict[str, List[bool]] = {venue_id: [] for venue_id in self.venues}
        
        # Initialize connections and performance tracking
        for venue_id in self.venues:
            self.connections[venue_id] = VenueConnection(self.venues[venue_id])
            self.performance[venue_id] = VenuePerformance()
        
        logger.info(f"VenueManager initialized with {len(self.venues)} venues")
    
    async def connect_all_venues(self) -> Dict[str, bool]:
        """Connect to all enabled venues"""
        connection_results = {}
        
        tasks = []
        for venue_id, config in self.venues.items():
            if config.enabled:
                tasks.append(self._connect_venue(venue_id))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (venue_id, config) in enumerate(self.venues.items()):
            if config.enabled:
                connection_results[venue_id] = isinstance(results[i], bool) and results[i]
        
        connected_count = sum(connection_results.values())
        logger.info(f"Connected to {connected_count}/{len(connection_results)} venues")
        
        return connection_results
    
    async def _connect_venue(self, venue_id: str) -> bool:
        """Connect to a specific venue"""
        try:
            connection = self.connections[venue_id]
            success = await connection.connect()
            
            if success:
                self.performance[venue_id].is_connected = True
                self.performance[venue_id].last_heartbeat = datetime.now()
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to connect to venue {venue_id}: {str(e)}")
            return False
    
    def get_available_venues(self, order_requirements: Dict[str, Any] = None) -> List[str]:
        """Get list of available venues that meet order requirements"""
        available_venues = []
        
        for venue_id, config in self.venues.items():
            if not config.enabled or not self.performance[venue_id].is_connected:
                continue
            
            # Check order requirements
            if order_requirements:
                # Check order size limits
                order_size = order_requirements.get('quantity', 0)
                if order_size < config.min_order_size or order_size > config.max_order_size:
                    continue
                
                # Check order value limits
                order_value = order_requirements.get('notional_value', 0)
                if order_value > config.max_order_value:
                    continue
                
                # Check order type support
                order_type = order_requirements.get('order_type', 'MARKET')
                if order_type == 'MARKET' and not config.supports_market_orders:
                    continue
                if order_type == 'LIMIT' and not config.supports_limit_orders:
                    continue
                if order_type == 'STOP' and not config.supports_stop_orders:
                    continue
            
            available_venues.append(venue_id)
        
        # Sort by priority
        available_venues.sort(key=lambda v: self.venues[v].priority)
        
        return available_venues
    
    def select_best_venue(
        self, 
        symbol: str,
        order_requirements: Dict[str, Any],
        optimization_criteria: str = "cost"
    ) -> Optional[str]:
        """
        Select the best venue based on optimization criteria.
        
        Criteria options:
        - "cost": Minimize total execution cost
        - "speed": Minimize latency
        - "fill_rate": Maximize fill probability  
        - "market_impact": Minimize market impact
        """
        available_venues = self.get_available_venues(order_requirements)
        
        if not available_venues:
            return None
        
        if len(available_venues) == 1:
            return available_venues[0]
        
        # Score venues based on optimization criteria
        venue_scores = {}
        
        for venue_id in available_venues:
            config = self.venues[venue_id]
            performance = self.performance[venue_id]
            
            if optimization_criteria == "cost":
                # Lower cost = higher score
                total_cost = config.cost_per_share - config.rebate_per_share
                score = 1.0 / (1.0 + total_cost * 1000)  # Scale for scoring
                
            elif optimization_criteria == "speed":
                # Lower latency = higher score
                latency = performance.avg_latency or config.expected_latency_ms
                score = 1.0 / (1.0 + latency)
                
            elif optimization_criteria == "fill_rate":
                # Higher fill rate = higher score
                fill_rate = performance.fill_rate or config.typical_fill_rate
                score = fill_rate
                
            elif optimization_criteria == "market_impact":
                # Lower impact = higher score
                impact = performance.market_impact_bps or config.avg_market_impact_bps
                score = 1.0 / (1.0 + impact)
                
            else:
                # Default: balanced score
                cost_score = 1.0 / (1.0 + (config.cost_per_share * 1000))
                latency_score = 1.0 / (1.0 + (performance.avg_latency or config.expected_latency_ms))
                fill_score = performance.fill_rate or config.typical_fill_rate
                score = (cost_score + latency_score + fill_score) / 3.0
            
            venue_scores[venue_id] = score
        
        # Return venue with highest score
        best_venue = max(venue_scores, key=venue_scores.get)
        
        logger.debug(
            "Selected best venue",
            venue_id=best_venue,
            optimization_criteria=optimization_criteria,
            score=venue_scores[best_venue],
            all_scores=venue_scores
        )
        
        return best_venue
    
    async def submit_order_to_venue(
        self, 
        venue_id: str, 
        order_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Submit order to specific venue"""
        if venue_id not in self.connections:
            raise ValueError(f"Unknown venue: {venue_id}")
        
        connection = self.connections[venue_id]
        
        start_time = time.perf_counter()
        
        try:
            result = await connection.submit_order(order_data)
            
            # Track performance
            latency = (time.perf_counter() - start_time) * 1000
            self._update_venue_performance(venue_id, 'order_submitted', {
                'latency_ms': latency,
                'order_size': order_data.get('quantity', 0)
            })
            
            return result
            
        except Exception as e:
            self._update_venue_performance(venue_id, 'order_failed', {'error': str(e)})
            raise
    
    async def cancel_order_at_venue(
        self, 
        venue_id: str, 
        venue_order_id: str
    ) -> Dict[str, Any]:
        """Cancel order at specific venue"""
        if venue_id not in self.connections:
            raise ValueError(f"Unknown venue: {venue_id}")
        
        connection = self.connections[venue_id]
        return await connection.cancel_order(venue_order_id)
    
    def _update_venue_performance(
        self, 
        venue_id: str, 
        event_type: str, 
        event_data: Dict[str, Any]
    ) -> None:
        """Update venue performance metrics"""
        performance = self.performance[venue_id]
        
        if event_type == 'order_submitted':
            latency = event_data.get('latency_ms', 0)
            
            # Update latency metrics
            self.latency_history[venue_id].append(latency)
            if len(self.latency_history[venue_id]) > 1000:  # Keep last 1000 measurements
                self.latency_history[venue_id].pop(0)
            
            if self.latency_history[venue_id]:
                performance.avg_latency = sum(self.latency_history[venue_id]) / len(self.latency_history[venue_id])
                sorted_latencies = sorted(self.latency_history[venue_id])
                performance.p95_latency = sorted_latencies[int(0.95 * len(sorted_latencies))]
            
            # Update order metrics
            performance.total_orders += 1
            performance.total_volume += event_data.get('order_size', 0)
            
        elif event_type == 'order_filled':
            self.fill_history[venue_id].append(True)
            
        elif event_type == 'order_rejected':
            self.fill_history[venue_id].append(False)
            
        elif event_type == 'order_failed':
            performance.error_count += 1
        
        # Update fill rate
        if self.fill_history[venue_id]:
            if len(self.fill_history[venue_id]) > 1000:  # Keep last 1000 measurements
                self.fill_history[venue_id].pop(0)
            
            fills = sum(self.fill_history[venue_id])
            total = len(self.fill_history[venue_id])
            performance.fill_rate = fills / total if total > 0 else 0.0
        
        performance.last_updated = datetime.now()
    
    def get_venue_performance(self, venue_id: str = None) -> Dict[str, VenuePerformance]:
        """Get performance metrics for venues"""
        if venue_id:
            return {venue_id: self.performance[venue_id]} if venue_id in self.performance else {}
        return self.performance.copy()
    
    def get_venue_rankings(self, criteria: str = "overall") -> List[Tuple[str, float]]:
        """Get venues ranked by performance criteria"""
        rankings = []
        
        for venue_id, performance in self.performance.items():
            if not performance.is_connected:
                continue
            
            if criteria == "latency":
                score = 1.0 / (1.0 + performance.avg_latency) if performance.avg_latency > 0 else 0.0
            elif criteria == "fill_rate":
                score = performance.fill_rate
            elif criteria == "cost":
                config = self.venues[venue_id]
                total_cost = config.cost_per_share - config.rebate_per_share
                score = 1.0 / (1.0 + total_cost * 1000)
            else:  # overall
                latency_score = 1.0 / (1.0 + performance.avg_latency) if performance.avg_latency > 0 else 0.0
                fill_score = performance.fill_rate
                config = self.venues[venue_id]
                cost_score = 1.0 / (1.0 + (config.cost_per_share * 1000))
                score = (latency_score + fill_score + cost_score) / 3.0
            
            rankings.append((venue_id, score))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    async def health_check(self) -> Dict[str, Dict[str, Any]]:
        """Perform health check on all venues"""
        health_status = {}
        
        for venue_id, connection in self.connections.items():
            try:
                if connection.is_connected:
                    # Simple heartbeat check
                    start_time = time.perf_counter()
                    await asyncio.sleep(0.001)  # Simulate heartbeat
                    latency = (time.perf_counter() - start_time) * 1000
                    
                    health_status[venue_id] = {
                        'status': 'healthy',
                        'latency_ms': latency,
                        'last_heartbeat': connection.last_heartbeat.isoformat() if connection.last_heartbeat else None,
                        'error_count': connection.error_count
                    }
                else:
                    health_status[venue_id] = {
                        'status': 'disconnected',
                        'error_count': connection.error_count
                    }
                    
            except Exception as e:
                health_status[venue_id] = {
                    'status': 'error',
                    'error': str(e),
                    'error_count': connection.error_count
                }
        
        return health_status
    
    async def shutdown(self) -> None:
        """Shutdown venue manager and disconnect from all venues"""
        logger.info("Shutting down venue manager")
        
        disconnect_tasks = []
        for connection in self.connections.values():
            if connection.is_connected:
                disconnect_tasks.append(connection.disconnect())
        
        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks)
        
        logger.info("Venue manager shutdown complete")