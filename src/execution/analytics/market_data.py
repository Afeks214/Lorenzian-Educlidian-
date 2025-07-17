"""
Market Data Provider

Real-time market data integration for execution analytics and routing decisions.
"""

import logging


import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import random
import structlog

logger = structlog.get_logger()


@dataclass
class MarketSnapshot:
    """Real-time market data snapshot"""
    
    symbol: str
    timestamp: datetime
    bid_price: float
    ask_price: float
    mid_price: float
    bid_size: int
    ask_size: int
    last_price: float
    last_size: int
    volume: int
    high: float
    low: float
    open_price: float
    vwap: float
    
    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price
    
    @property
    def spread_bps(self) -> float:
        return (self.spread / self.mid_price) * 10000 if self.mid_price > 0 else 0


@dataclass
class HistoricalData:
    """Historical market data"""
    
    symbol: str
    start_date: datetime
    end_date: datetime
    prices: List[float]
    volumes: List[int]
    timestamps: List[datetime]
    vwap_data: List[float]
    
    @property
    def avg_volume(self) -> float:
        return sum(self.volumes) / len(self.volumes) if self.volumes else 0
    
    @property
    def volatility(self) -> float:
        if len(self.prices) < 2:
            return 0.0
        
        returns = [
            (self.prices[i] / self.prices[i-1] - 1) 
            for i in range(1, len(self.prices))
        ]
        
        if not returns:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        
        return (variance ** 0.5) * (252 ** 0.5)  # Annualized volatility


class MarketDataProvider:
    """
    Real-time market data provider for execution analytics.
    
    Provides market data feeds, historical data, and derived analytics
    for routing optimization and execution monitoring.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Market data cache
        self.market_snapshots: Dict[str, MarketSnapshot] = {}
        self.historical_cache: Dict[str, HistoricalData] = {}
        
        # Data feeds
        self.active_subscriptions: set = set()
        self.data_feed_task = None
        
        # Performance tracking
        self.update_latencies: List[float] = []
        self.last_update_time: Dict[str, datetime] = {}
        
        # Market state
        self.market_hours = {
            'open': '09:30',
            'close': '16:00',
            'timezone': 'US/Eastern'
        }
        
        logger.info("MarketDataProvider initialized")
    
    async def start_data_feeds(self) -> None:
        """Start real-time data feeds"""
        if self.data_feed_task and not self.data_feed_task.done():
            return
        
        self.data_feed_task = asyncio.create_task(self._data_feed_loop())
        logger.info("Market data feeds started")
    
    async def stop_data_feeds(self) -> None:
        """Stop real-time data feeds"""
        if self.data_feed_task:
            self.data_feed_task.cancel()
            try:
                await self.data_feed_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Market data feeds stopped")
    
    async def subscribe_symbol(self, symbol: str) -> None:
        """Subscribe to real-time data for symbol"""
        self.active_subscriptions.add(symbol)
        logger.debug(f"Subscribed to market data for {symbol}")
    
    async def unsubscribe_symbol(self, symbol: str) -> None:
        """Unsubscribe from symbol data"""
        self.active_subscriptions.discard(symbol)
        if symbol in self.market_snapshots:
            del self.market_snapshots[symbol]
        logger.debug(f"Unsubscribed from market data for {symbol}")
    
    async def get_current_data(self, symbol: str) -> Dict[str, Any]:
        """Get current market data for symbol"""
        if symbol not in self.active_subscriptions:
            await self.subscribe_symbol(symbol)
        
        snapshot = self.market_snapshots.get(symbol)
        if not snapshot:
            # Generate initial snapshot
            snapshot = await self._generate_market_snapshot(symbol)
            self.market_snapshots[symbol] = snapshot
        
        return {
            'symbol': snapshot.symbol,
            'price': snapshot.mid_price,
            'bid': snapshot.bid_price,
            'ask': snapshot.ask_price,
            'spread': snapshot.spread,
            'spread_bps': snapshot.spread_bps,
            'volume': snapshot.volume,
            'volatility': await self._calculate_current_volatility(symbol),
            'volume_rate': await self._calculate_volume_rate(symbol),
            'timestamp': snapshot.timestamp.isoformat()
        }
    
    def get_current_price(self, symbol: str) -> float:
        """Get current mid price for symbol"""
        snapshot = self.market_snapshots.get(symbol)
        if snapshot:
            return snapshot.mid_price
        
        # Return simulated price if no data
        return 150.0 + random.uniform(-5, 5)
    
    def get_spread(self, symbol: str) -> float:
        """Get current bid-ask spread"""
        snapshot = self.market_snapshots.get(symbol)
        if snapshot:
            return snapshot.spread
        
        # Return simulated spread
        return 0.01
    
    def get_volume(self, symbol: str) -> int:
        """Get current volume"""
        snapshot = self.market_snapshots.get(symbol)
        if snapshot:
            return snapshot.volume
        
        # Return simulated volume
        return random.randint(100000, 1000000)
    
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = '1min'
    ) -> HistoricalData:
        """Get historical market data"""
        
        cache_key = f"{symbol}_{start_date.date()}_{end_date.date()}_{interval}"
        
        if cache_key in self.historical_cache:
            return self.historical_cache[cache_key]
        
        # Generate simulated historical data
        historical_data = await self._generate_historical_data(
            symbol, start_date, end_date, interval
        )
        
        self.historical_cache[cache_key] = historical_data
        return historical_data
    
    async def get_historical_volume(
        self,
        symbol: str,
        days: int = 20
    ) -> Dict[str, Any]:
        """Get historical volume data for VWAP calculations"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        historical_data = await self.get_historical_data(symbol, start_date, end_date)
        
        return {
            'symbol': symbol,
            'volumes': historical_data.volumes,
            'avg_volume': historical_data.avg_volume,
            'volume_profile': await self._calculate_volume_profile(historical_data)
        }
    
    async def get_intraday_volume_pattern(self, symbol: str) -> Dict[str, Any]:
        """Get intraday volume pattern for algorithm optimization"""
        
        # Generate typical U-shaped intraday volume curve
        pattern = []
        for hour in range(7):  # 9:30 AM to 4:00 PM (6.5 hours)
            for minute in range(0, 60, 5):  # 5-minute intervals
                time_factor = (hour * 60 + minute) / (6.5 * 60)  # Normalize to [0, 1]
                
                # U-shaped curve: high at open/close, low at midday
                volume_factor = 2.0 * (time_factor**2 - time_factor + 0.5)
                volume_factor = max(0.3, min(2.0, volume_factor))
                
                pattern.append(volume_factor)
        
        return {
            'symbol': symbol,
            'curve': pattern,
            'pattern_type': 'U_shaped',
            'intervals': '5min'
        }
    
    async def _data_feed_loop(self) -> None:
        """Main data feed loop for real-time updates"""
        
        while True:
            try:
                start_time = time.perf_counter()
                
                # Update all subscribed symbols
                for symbol in list(self.active_subscriptions):
                    await self._update_symbol_data(symbol)
                
                # Track update latency
                update_latency = (time.perf_counter() - start_time) * 1000
                self.update_latencies.append(update_latency)
                
                # Keep only recent latency measurements
                if len(self.update_latencies) > 1000:
                    self.update_latencies = self.update_latencies[-1000:]
                
                # Sleep until next update (simulate 100ms feed)
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in data feed loop: {str(e)}")
                await asyncio.sleep(1.0)
    
    async def _update_symbol_data(self, symbol: str) -> None:
        """Update market data for a symbol"""
        
        current_snapshot = self.market_snapshots.get(symbol)
        
        if current_snapshot:
            # Update existing snapshot with new data
            new_snapshot = await self._simulate_market_update(current_snapshot)
        else:
            # Generate new snapshot
            new_snapshot = await self._generate_market_snapshot(symbol)
        
        self.market_snapshots[symbol] = new_snapshot
        self.last_update_time[symbol] = datetime.now()
    
    async def _generate_market_snapshot(self, symbol: str) -> MarketSnapshot:
        """Generate initial market snapshot"""
        
        # Simulate realistic market data
        base_price = 150.0 + random.uniform(-50, 50)
        spread = random.uniform(0.01, 0.05)
        
        bid_price = base_price - spread / 2
        ask_price = base_price + spread / 2
        mid_price = (bid_price + ask_price) / 2
        
        return MarketSnapshot(
            symbol=symbol,
            timestamp=datetime.now(),
            bid_price=bid_price,
            ask_price=ask_price,
            mid_price=mid_price,
            bid_size=random.randint(100, 1000),
            ask_size=random.randint(100, 1000),
            last_price=mid_price + random.uniform(-0.02, 0.02),
            last_size=random.randint(100, 500),
            volume=random.randint(100000, 1000000),
            high=base_price + random.uniform(0, 5),
            low=base_price - random.uniform(0, 5),
            open_price=base_price + random.uniform(-2, 2),
            vwap=base_price + random.uniform(-1, 1)
        )
    
    async def _simulate_market_update(self, current: MarketSnapshot) -> MarketSnapshot:
        """Simulate market data update"""
        
        # Small price movement
        price_change = random.uniform(-0.1, 0.1)
        new_mid = current.mid_price + price_change
        
        # Adjust spread slightly
        spread_change = random.uniform(-0.002, 0.002)
        new_spread = max(0.01, current.spread + spread_change)
        
        new_bid = new_mid - new_spread / 2
        new_ask = new_mid + new_spread / 2
        
        # Update volume
        volume_change = random.randint(-10000, 10000)
        new_volume = max(0, current.volume + volume_change)
        
        return MarketSnapshot(
            symbol=current.symbol,
            timestamp=datetime.now(),
            bid_price=new_bid,
            ask_price=new_ask,
            mid_price=new_mid,
            bid_size=random.randint(100, 1000),
            ask_size=random.randint(100, 1000),
            last_price=new_mid + random.uniform(-0.01, 0.01),
            last_size=random.randint(100, 500),
            volume=new_volume,
            high=max(current.high, new_mid),
            low=min(current.low, new_mid),
            open_price=current.open_price,
            vwap=current.vwap * 0.99 + new_mid * 0.01  # VWAP update
        )
    
    async def _generate_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str
    ) -> HistoricalData:
        """Generate simulated historical data"""
        
        # Generate time series
        if interval == '1min':
            delta = timedelta(minutes=1)
        elif interval == '5min':
            delta = timedelta(minutes=5)
        else:
            delta = timedelta(hours=1)
        
        timestamps = []
        prices = []
        volumes = []
        
        current_time = start_date
        current_price = 150.0
        
        while current_time <= end_date:
            # Random walk price
            price_change = random.uniform(-0.5, 0.5)
            current_price = max(1.0, current_price + price_change)
            
            prices.append(current_price)
            volumes.append(random.randint(1000, 50000))
            timestamps.append(current_time)
            
            current_time += delta
        
        # Calculate VWAP data
        vwap_data = []
        cumulative_volume = 0
        cumulative_pv = 0
        
        for price, volume in zip(prices, volumes):
            cumulative_pv += price * volume
            cumulative_volume += volume
            vwap = cumulative_pv / cumulative_volume if cumulative_volume > 0 else price
            vwap_data.append(vwap)
        
        return HistoricalData(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            prices=prices,
            volumes=volumes,
            timestamps=timestamps,
            vwap_data=vwap_data
        )
    
    async def _calculate_current_volatility(self, symbol: str) -> float:
        """Calculate current volatility estimate"""
        
        # Get recent historical data for volatility calculation
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        try:
            historical_data = await self.get_historical_data(symbol, start_date, end_date)
            return historical_data.volatility
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            # Return default volatility if calculation fails
            return 0.20  # 20% annualized volatility
    
    async def _calculate_volume_rate(self, symbol: str) -> float:
        """Calculate current volume rate (shares per minute)"""
        
        snapshot = self.market_snapshots.get(symbol)
        if not snapshot:
            return 1000.0  # Default volume rate
        
        # Simulate volume rate based on time of day
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        
        # Higher volume at market open and close
        if 9 <= hour <= 10 or 15 <= hour <= 16:
            base_rate = 5000
        else:
            base_rate = 2000
        
        # Add some randomness
        return base_rate * random.uniform(0.5, 2.0)
    
    async def _calculate_volume_profile(self, historical_data: HistoricalData) -> List[float]:
        """Calculate intraday volume profile"""
        
        # Group volumes by time of day
        hourly_volumes = [0.0] * 24
        hourly_counts = [0] * 24
        
        for timestamp, volume in zip(historical_data.timestamps, historical_data.volumes):
            hour = timestamp.hour
            hourly_volumes[hour] += volume
            hourly_counts[hour] += 1
        
        # Calculate average volume per hour
        profile = []
        for hour in range(24):
            if hourly_counts[hour] > 0:
                avg_volume = hourly_volumes[hour] / hourly_counts[hour]
            else:
                avg_volume = 0.0
            profile.append(avg_volume)
        
        return profile
    
    def get_market_status(self) -> Dict[str, Any]:
        """Get current market status"""
        
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        
        # Simplified market hours check
        market_open = 9 <= hour < 16 or (hour == 9 and minute >= 30)
        
        return {
            'is_open': market_open,
            'session': 'regular' if market_open else 'closed',
            'next_open': '09:30' if not market_open else None,
            'next_close': '16:00' if market_open else None
        }
    
    def get_feed_statistics(self) -> Dict[str, Any]:
        """Get data feed performance statistics"""
        
        if not self.update_latencies:
            return {'status': 'no_data'}
        
        avg_latency = sum(self.update_latencies) / len(self.update_latencies)
        max_latency = max(self.update_latencies)
        min_latency = min(self.update_latencies)
        
        return {
            'active_subscriptions': len(self.active_subscriptions),
            'avg_update_latency_ms': avg_latency,
            'max_update_latency_ms': max_latency,
            'min_update_latency_ms': min_latency,
            'total_updates': len(self.update_latencies),
            'symbols_tracked': list(self.active_subscriptions)
        }