"""
Microstructure Engine

Advanced market microstructure analysis engine for execution optimization.
Provides comprehensive market condition assessment and execution timing recommendations.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import structlog

logger = structlog.get_logger()


class MarketRegime(Enum):
    """Market regime classifications"""
    NORMAL = "NORMAL"
    VOLATILE = "VOLATILE"
    TRENDING = "TRENDING"
    STRESSED = "STRESSED"
    ILLIQUID = "ILLIQUID"


class ExecutionTiming(Enum):
    """Execution timing recommendations"""
    IMMEDIATE = "IMMEDIATE"
    OPTIMAL = "OPTIMAL"
    PATIENT = "PATIENT"
    AVOID = "AVOID"


@dataclass
class MarketConditions:
    """Comprehensive market conditions assessment"""
    
    symbol: str
    timestamp: datetime
    
    # Basic market data
    bid_price: float
    ask_price: float
    mid_price: float
    spread_bps: float
    
    # Liquidity metrics
    bid_size: int
    ask_size: int
    total_depth: int
    effective_spread_bps: float
    
    # Volatility and momentum
    volatility: float
    momentum: float
    price_trend: str  # UP/DOWN/FLAT
    
    # Volume and flow
    volume_rate: float
    volume_imbalance: float
    flow_toxicity: float
    
    # Market structure
    market_regime: MarketRegime
    liquidity_score: float  # 0-1 scale
    impact_cost_bps: float
    
    # Execution recommendations
    optimal_timing: ExecutionTiming
    recommended_strategy: str
    confidence_score: float
    
    # Risk metrics
    adverse_selection_risk: float
    timing_risk: float
    information_content: float


@dataclass
class MicrostructureConfig:
    """Configuration for microstructure analysis"""
    
    # Analysis windows
    short_window_seconds: int = 30
    medium_window_seconds: int = 300
    long_window_seconds: int = 1800
    
    # Volatility thresholds
    normal_volatility_threshold: float = 0.02
    high_volatility_threshold: float = 0.05
    
    # Liquidity thresholds
    min_liquidity_score: float = 0.3
    good_liquidity_threshold: float = 0.7
    
    # Spread thresholds
    tight_spread_bps: float = 5.0
    wide_spread_bps: float = 20.0
    
    # Volume thresholds
    low_volume_threshold: float = 0.5  # Relative to average
    high_volume_threshold: float = 2.0
    
    # Risk thresholds
    max_adverse_selection: float = 0.3
    max_timing_risk: float = 0.2


class MicrostructureEngine:
    """
    Advanced market microstructure analysis engine.
    
    Provides real-time analysis of market conditions including:
    - Order book dynamics
    - Liquidity assessment
    - Market impact estimation
    - Optimal execution timing
    - Flow toxicity detection
    """
    
    def __init__(self, config: MicrostructureConfig = None):
        self.config = config or MicrostructureConfig()
        
        # Market data storage
        self.market_data_history: Dict[str, List[Dict[str, Any]]] = {}
        self.order_book_history: Dict[str, List[Dict[str, Any]]] = {}
        self.trade_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Analysis cache
        self.analysis_cache: Dict[str, MarketConditions] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Background analysis
        self.analysis_tasks: Dict[str, asyncio.Task] = {}
        
        # Performance tracking
        self.analysis_latencies: List[float] = []
        
        logger.info("MicrostructureEngine initialized")
    
    async def analyze_market_conditions(self, symbol: str) -> MarketConditions:
        """
        Analyze current market conditions for symbol.
        
        Provides comprehensive microstructure analysis including liquidity,
        volatility, market impact, and execution timing recommendations.
        """
        start_time = time.perf_counter()
        
        try:
            # Check cache first
            if self._is_analysis_cached(symbol):
                return self.analysis_cache[symbol]
            
            # Gather market data
            market_data = await self._gather_market_data(symbol)
            order_book = await self._get_order_book(symbol)
            recent_trades = await self._get_recent_trades(symbol)
            
            # Perform comprehensive analysis
            conditions = await self._perform_comprehensive_analysis(
                symbol, market_data, order_book, recent_trades
            )
            
            # Cache results
            self.analysis_cache[symbol] = conditions
            self.cache_timestamps[symbol] = datetime.now()
            
            # Track analysis latency
            analysis_latency = (time.perf_counter() - start_time) * 1000
            self.analysis_latencies.append(analysis_latency)
            
            # Keep recent latencies only
            if len(self.analysis_latencies) > 1000:
                self.analysis_latencies = self.analysis_latencies[-1000:]
            
            logger.debug(
                "Market conditions analyzed",
                symbol=symbol,
                analysis_latency_ms=analysis_latency,
                regime=conditions.market_regime.value,
                timing=conditions.optimal_timing.value
            )
            
            return conditions
            
        except Exception as e:
            logger.error(f"Market analysis failed for {symbol}: {str(e)}")
            
            # Return conservative conditions on error
            return self._get_conservative_conditions(symbol)
    
    async def _gather_market_data(self, symbol: str) -> Dict[str, Any]:
        """Gather current market data"""
        
        # Simulate market data gathering
        # In production, would fetch from market data provider
        current_price = 150.0 + np.random.uniform(-5, 5)
        spread = max(0.01, np.random.uniform(0.01, 0.10))
        
        market_data = {
            'timestamp': datetime.now(),
            'bid_price': current_price - spread/2,
            'ask_price': current_price + spread/2,
            'mid_price': current_price,
            'bid_size': np.random.randint(100, 1000),
            'ask_size': np.random.randint(100, 1000),
            'last_trade_price': current_price + np.random.uniform(-0.02, 0.02),
            'last_trade_size': np.random.randint(100, 500),
            'volume': np.random.randint(10000, 100000),
            'high': current_price + np.random.uniform(0, 2),
            'low': current_price - np.random.uniform(0, 2)
        }
        
        # Store in history
        if symbol not in self.market_data_history:
            self.market_data_history[symbol] = []
        
        self.market_data_history[symbol].append(market_data)
        
        # Keep limited history
        if len(self.market_data_history[symbol]) > 10000:
            self.market_data_history[symbol] = self.market_data_history[symbol][-5000:]
        
        return market_data
    
    async def _get_order_book(self, symbol: str) -> Dict[str, Any]:
        """Get current order book snapshot"""
        
        # Simulate order book
        base_price = 150.0
        spread = 0.05
        
        # Generate bid levels
        bids = []
        current_bid = base_price - spread/2
        for i in range(10):  # 10 levels
            bids.append({
                'price': round(current_bid - i * 0.01, 2),
                'size': np.random.randint(100, 2000),
                'orders': np.random.randint(1, 10)
            })
        
        # Generate ask levels  
        asks = []
        current_ask = base_price + spread/2
        for i in range(10):  # 10 levels
            asks.append({
                'price': round(current_ask + i * 0.01, 2),
                'size': np.random.randint(100, 2000),
                'orders': np.random.randint(1, 10)
            })
        
        order_book = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'bids': bids,
            'asks': asks,
            'sequence': int(time.time() * 1000000)
        }
        
        # Store in history
        if symbol not in self.order_book_history:
            self.order_book_history[symbol] = []
        
        self.order_book_history[symbol].append(order_book)
        
        # Keep limited history
        if len(self.order_book_history[symbol]) > 1000:
            self.order_book_history[symbol] = self.order_book_history[symbol][-500:]
        
        return order_book
    
    async def _get_recent_trades(self, symbol: str) -> List[Dict[str, Any]]:
        """Get recent trade data"""
        
        # Simulate recent trades
        trades = []
        base_price = 150.0
        
        for i in range(20):  # Last 20 trades
            trade = {
                'timestamp': datetime.now() - timedelta(seconds=i*10),
                'price': base_price + np.random.uniform(-0.1, 0.1),
                'size': np.random.randint(100, 1000),
                'side': np.random.choice(['BUY', 'SELL']),
                'trade_id': f"T{int(time.time())}{i}"
            }
            trades.append(trade)
        
        # Store in history
        if symbol not in self.trade_history:
            self.trade_history[symbol] = []
        
        self.trade_history[symbol].extend(trades)
        
        # Keep limited history
        if len(self.trade_history[symbol]) > 10000:
            self.trade_history[symbol] = self.trade_history[symbol][-5000:]
        
        return trades
    
    async def _perform_comprehensive_analysis(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        order_book: Dict[str, Any],
        recent_trades: List[Dict[str, Any]]
    ) -> MarketConditions:
        """Perform comprehensive microstructure analysis"""
        
        # Basic market metrics
        bid_price = market_data['bid_price']
        ask_price = market_data['ask_price']
        mid_price = market_data['mid_price']
        spread_bps = ((ask_price - bid_price) / mid_price) * 10000
        
        # Liquidity analysis
        liquidity_metrics = self._analyze_liquidity(order_book, market_data)
        
        # Volatility analysis
        volatility_metrics = self._analyze_volatility(symbol, market_data, recent_trades)
        
        # Volume analysis
        volume_metrics = self._analyze_volume(symbol, market_data, recent_trades)
        
        # Market impact analysis
        impact_metrics = self._analyze_market_impact(order_book, recent_trades)
        
        # Flow analysis
        flow_metrics = self._analyze_order_flow(recent_trades, order_book)
        
        # Risk analysis
        risk_metrics = self._analyze_execution_risks(
            symbol, market_data, order_book, recent_trades
        )
        
        # Determine market regime
        market_regime = self._determine_market_regime(
            volatility_metrics, liquidity_metrics, volume_metrics
        )
        
        # Generate execution recommendations
        timing_rec, strategy_rec, confidence = self._generate_execution_recommendations(
            market_regime, liquidity_metrics, volatility_metrics, risk_metrics
        )
        
        # Compile comprehensive conditions
        conditions = MarketConditions(
            symbol=symbol,
            timestamp=datetime.now(),
            
            # Basic market data
            bid_price=bid_price,
            ask_price=ask_price,
            mid_price=mid_price,
            spread_bps=spread_bps,
            
            # Liquidity metrics
            bid_size=market_data['bid_size'],
            ask_size=market_data['ask_size'],
            total_depth=liquidity_metrics['total_depth'],
            effective_spread_bps=liquidity_metrics['effective_spread_bps'],
            
            # Volatility and momentum
            volatility=volatility_metrics['volatility'],
            momentum=volatility_metrics['momentum'],
            price_trend=volatility_metrics['trend'],
            
            # Volume and flow
            volume_rate=volume_metrics['volume_rate'],
            volume_imbalance=volume_metrics['imbalance'],
            flow_toxicity=flow_metrics['toxicity'],
            
            # Market structure
            market_regime=market_regime,
            liquidity_score=liquidity_metrics['liquidity_score'],
            impact_cost_bps=impact_metrics['impact_cost_bps'],
            
            # Execution recommendations
            optimal_timing=timing_rec,
            recommended_strategy=strategy_rec,
            confidence_score=confidence,
            
            # Risk metrics
            adverse_selection_risk=risk_metrics['adverse_selection'],
            timing_risk=risk_metrics['timing_risk'],
            information_content=flow_metrics['information_content']
        )
        
        return conditions
    
    def _analyze_liquidity(self, order_book: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze order book liquidity"""
        
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])
        
        # Calculate total depth
        total_bid_size = sum(level['size'] for level in bids[:5])  # Top 5 levels
        total_ask_size = sum(level['size'] for level in asks[:5])
        total_depth = total_bid_size + total_ask_size
        
        # Calculate effective spread
        if bids and asks:
            best_bid = bids[0]['price']
            best_ask = asks[0]['price']
            mid_price = (best_bid + best_ask) / 2
            effective_spread_bps = ((best_ask - best_bid) / mid_price) * 10000
        else:
            effective_spread_bps = 100.0  # Wide spread if no book
        
        # Calculate liquidity score (0-1)
        depth_score = min(1.0, total_depth / 10000)  # Normalize by 10k shares
        spread_score = max(0.0, 1.0 - effective_spread_bps / 50.0)  # Penalty for wide spreads
        liquidity_score = (depth_score + spread_score) / 2
        
        return {
            'total_depth': total_depth,
            'effective_spread_bps': effective_spread_bps,
            'liquidity_score': liquidity_score,
            'bid_ask_imbalance': (total_bid_size - total_ask_size) / max(1, total_bid_size + total_ask_size)
        }
    
    def _analyze_volatility(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        recent_trades: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Analyze price volatility and momentum"""
        
        # Get historical prices
        history = self.market_data_history.get(symbol, [])
        if len(history) < 10:
            # Insufficient data, use defaults
            return {
                'volatility': 0.02,
                'momentum': 0.0,
                'trend': 'FLAT'
            }
        
        # Calculate returns
        recent_prices = [h['mid_price'] for h in history[-30:]]  # Last 30 observations
        returns = [
            (recent_prices[i] / recent_prices[i-1] - 1)
            for i in range(1, len(recent_prices))
        ]
        
        # Calculate volatility (annualized)
        if len(returns) > 1:
            volatility = np.std(returns) * np.sqrt(252 * 24 * 12)  # Assuming 5-min intervals
        else:
            volatility = 0.02
        
        # Calculate momentum
        if len(recent_prices) >= 10:
            short_avg = np.mean(recent_prices[-5:])
            long_avg = np.mean(recent_prices[-10:])
            momentum = (short_avg / long_avg - 1)
        else:
            momentum = 0.0
        
        # Determine trend
        if momentum > 0.002:  # 20 bps
            trend = 'UP'
        elif momentum < -0.002:
            trend = 'DOWN'
        else:
            trend = 'FLAT'
        
        return {
            'volatility': volatility,
            'momentum': momentum,
            'trend': trend
        }
    
    def _analyze_volume(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        recent_trades: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Analyze volume patterns and imbalances"""
        
        current_volume = market_data.get('volume', 0)
        
        # Calculate volume rate (shares per minute)
        trade_volume = sum(trade['size'] for trade in recent_trades[-10:])  # Last 10 trades
        volume_rate = trade_volume / 10  # Approximate per-minute rate
        
        # Calculate buy/sell imbalance
        buy_volume = sum(
            trade['size'] for trade in recent_trades
            if trade['side'] == 'BUY'
        )
        sell_volume = sum(
            trade['size'] for trade in recent_trades
            if trade['side'] == 'SELL'
        )
        
        total_volume = buy_volume + sell_volume
        if total_volume > 0:
            volume_imbalance = (buy_volume - sell_volume) / total_volume
        else:
            volume_imbalance = 0.0
        
        return {
            'volume_rate': volume_rate,
            'imbalance': volume_imbalance,
            'relative_volume': current_volume / max(50000, current_volume)  # Relative to typical
        }
    
    def _analyze_market_impact(
        self,
        order_book: Dict[str, Any],
        recent_trades: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Analyze potential market impact"""
        
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])
        
        if not bids or not asks:
            return {'impact_cost_bps': 50.0}  # High impact if no book
        
        # Calculate impact for standard order sizes
        sizes_to_test = [1000, 5000, 10000]  # 1k, 5k, 10k shares
        impact_estimates = []
        
        for size in sizes_to_test:
            # Estimate impact for buy order
            remaining_size = size
            weighted_price = 0.0
            total_filled = 0
            
            for level in asks:
                if remaining_size <= 0:
                    break
                
                fill_size = min(remaining_size, level['size'])
                weighted_price += level['price'] * fill_size
                total_filled += fill_size
                remaining_size -= fill_size
            
            if total_filled > 0:
                avg_fill_price = weighted_price / total_filled
                best_ask = asks[0]['price']
                impact_bps = ((avg_fill_price - best_ask) / best_ask) * 10000
                impact_estimates.append(max(0, impact_bps))
            else:
                impact_estimates.append(100.0)  # Very high impact if can't fill
        
        # Use impact for 5k share order as representative
        impact_cost_bps = impact_estimates[1] if len(impact_estimates) > 1 else 20.0
        
        return {
            'impact_cost_bps': impact_cost_bps,
            'impact_1k': impact_estimates[0] if impact_estimates else 10.0,
            'impact_5k': impact_estimates[1] if len(impact_estimates) > 1 else 20.0,
            'impact_10k': impact_estimates[2] if len(impact_estimates) > 2 else 40.0
        }
    
    def _analyze_order_flow(
        self,
        recent_trades: List[Dict[str, Any]],
        order_book: Dict[str, Any]
    ) -> Dict[str, float]:
        """Analyze order flow and toxicity"""
        
        if not recent_trades:
            return {
                'toxicity': 0.5,
                'information_content': 0.5,
                'flow_imbalance': 0.0
            }
        
        # Calculate trade-based metrics
        prices = [trade['price'] for trade in recent_trades[-20:]]
        sizes = [trade['size'] for trade in recent_trades[-20:]]
        
        # Flow toxicity (simplified measure based on price impact)
        if len(prices) > 1:
            price_changes = [
                abs(prices[i] - prices[i-1]) / prices[i-1]
                for i in range(1, len(prices))
            ]
            avg_price_change = np.mean(price_changes)
            
            # Higher price changes = higher toxicity
            toxicity = min(1.0, avg_price_change * 1000)  # Scale to 0-1
        else:
            toxicity = 0.5
        
        # Information content (based on size-weighted price moves)
        if len(sizes) > 1 and len(prices) > 1:
            weighted_moves = [
                abs(prices[i] - prices[i-1]) * sizes[i]
                for i in range(1, len(prices))
            ]
            total_size = sum(sizes[1:])
            
            if total_size > 0:
                avg_weighted_move = sum(weighted_moves) / total_size
                information_content = min(1.0, avg_weighted_move * 10000)  # Scale
            else:
                information_content = 0.5
        else:
            information_content = 0.5
        
        # Flow imbalance
        buy_trades = [t for t in recent_trades if t['side'] == 'BUY']
        sell_trades = [t for t in recent_trades if t['side'] == 'SELL']
        
        if recent_trades:
            flow_imbalance = (len(buy_trades) - len(sell_trades)) / len(recent_trades)
        else:
            flow_imbalance = 0.0
        
        return {
            'toxicity': toxicity,
            'information_content': information_content,
            'flow_imbalance': flow_imbalance
        }
    
    def _analyze_execution_risks(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        order_book: Dict[str, Any],
        recent_trades: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Analyze execution-related risks"""
        
        # Adverse selection risk (risk of trading against informed flow)
        # Based on recent volatility and trade patterns
        history = self.market_data_history.get(symbol, [])
        if len(history) > 5:
            recent_volatility = np.std([h['mid_price'] for h in history[-10:]])
            normalized_volatility = recent_volatility / market_data['mid_price']
            adverse_selection = min(1.0, normalized_volatility * 100)
        else:
            adverse_selection = 0.3
        
        # Timing risk (risk of price movement during execution)
        # Based on volatility and market regime
        volatility = 0.02  # Default
        if len(history) > 1:
            prices = [h['mid_price'] for h in history[-20:]]
            returns = [(prices[i]/prices[i-1]-1) for i in range(1, len(prices))]
            if returns:
                volatility = np.std(returns)
        
        timing_risk = min(1.0, volatility * 50)  # Scale volatility to risk
        
        return {
            'adverse_selection': adverse_selection,
            'timing_risk': timing_risk,
            'volatility_risk': volatility
        }
    
    def _determine_market_regime(
        self,
        volatility_metrics: Dict[str, float],
        liquidity_metrics: Dict[str, float],
        volume_metrics: Dict[str, float]
    ) -> MarketRegime:
        """Determine current market regime"""
        
        volatility = volatility_metrics['volatility']
        liquidity_score = liquidity_metrics['liquidity_score']
        volume_rate = volume_metrics['volume_rate']
        
        # Classify regime based on conditions
        if volatility > self.config.high_volatility_threshold:
            if liquidity_score < self.config.min_liquidity_score:
                return MarketRegime.STRESSED
            else:
                return MarketRegime.VOLATILE
        
        elif liquidity_score < self.config.min_liquidity_score:
            return MarketRegime.ILLIQUID
        
        elif abs(volatility_metrics['momentum']) > 0.005:  # Strong trend
            return MarketRegime.TRENDING
        
        else:
            return MarketRegime.NORMAL
    
    def _generate_execution_recommendations(
        self,
        market_regime: MarketRegime,
        liquidity_metrics: Dict[str, float],
        volatility_metrics: Dict[str, float],
        risk_metrics: Dict[str, float]
    ) -> Tuple[ExecutionTiming, str, float]:
        """Generate execution timing and strategy recommendations"""
        
        # Determine timing based on regime and conditions
        if market_regime == MarketRegime.STRESSED:
            timing = ExecutionTiming.AVOID
            strategy = "WAIT_FOR_IMPROVEMENT"
            confidence = 0.9
        
        elif market_regime == MarketRegime.ILLIQUID:
            timing = ExecutionTiming.PATIENT
            strategy = "ICEBERG_ORDER"
            confidence = 0.8
        
        elif market_regime == MarketRegime.VOLATILE:
            if liquidity_metrics['liquidity_score'] > 0.7:
                timing = ExecutionTiming.IMMEDIATE
                strategy = "MARKET_ORDER"
                confidence = 0.7
            else:
                timing = ExecutionTiming.PATIENT
                strategy = "LIMIT_ORDER"
                confidence = 0.6
        
        elif market_regime == MarketRegime.TRENDING:
            if volatility_metrics['momentum'] > 0:  # Uptrend
                timing = ExecutionTiming.IMMEDIATE
                strategy = "AGGRESSIVE_LIMIT"
            else:  # Downtrend
                timing = ExecutionTiming.OPTIMAL
                strategy = "PATIENT_LIMIT"
            confidence = 0.75
        
        else:  # NORMAL regime
            timing = ExecutionTiming.OPTIMAL
            strategy = "SMART_ROUTER"
            confidence = 0.8
        
        # Adjust confidence based on risk metrics
        if risk_metrics['adverse_selection'] > 0.7:
            confidence *= 0.8
        if risk_metrics['timing_risk'] > 0.5:
            confidence *= 0.9
        
        return timing, strategy, confidence
    
    def _is_analysis_cached(self, symbol: str) -> bool:
        """Check if analysis is cached and still valid"""
        if symbol not in self.analysis_cache:
            return False
        
        cache_time = self.cache_timestamps.get(symbol)
        if not cache_time:
            return False
        
        # Cache valid for 30 seconds
        return (datetime.now() - cache_time).total_seconds() < 30
    
    def _get_conservative_conditions(self, symbol: str) -> MarketConditions:
        """Get conservative market conditions for error cases"""
        return MarketConditions(
            symbol=symbol,
            timestamp=datetime.now(),
            bid_price=149.95,
            ask_price=150.05,
            mid_price=150.0,
            spread_bps=6.67,
            bid_size=500,
            ask_size=500,
            total_depth=1000,
            effective_spread_bps=6.67,
            volatility=0.02,
            momentum=0.0,
            price_trend='FLAT',
            volume_rate=1000.0,
            volume_imbalance=0.0,
            flow_toxicity=0.5,
            market_regime=MarketRegime.NORMAL,
            liquidity_score=0.6,
            impact_cost_bps=10.0,
            optimal_timing=ExecutionTiming.PATIENT,
            recommended_strategy="LIMIT_ORDER",
            confidence_score=0.5,
            adverse_selection_risk=0.3,
            timing_risk=0.2,
            information_content=0.5
        )
    
    async def get_historical_analysis(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[MarketConditions]:
        """Get historical microstructure analysis"""
        
        # This would query historical data in production
        # For now, generate sample historical conditions
        
        historical_conditions = []
        current_time = start_time
        
        while current_time <= end_time:
            # Generate simulated historical condition
            condition = await self.analyze_market_conditions(symbol)
            condition.timestamp = current_time
            
            historical_conditions.append(condition)
            current_time += timedelta(minutes=5)
        
        return historical_conditions
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get microstructure analysis performance statistics"""
        
        if not self.analysis_latencies:
            return {'status': 'no_data'}
        
        return {
            'analysis_count': len(self.analysis_latencies),
            'avg_latency_ms': np.mean(self.analysis_latencies),
            'max_latency_ms': np.max(self.analysis_latencies),
            'min_latency_ms': np.min(self.analysis_latencies),
            'p95_latency_ms': np.percentile(self.analysis_latencies, 95),
            'cache_hit_rate': len(self.analysis_cache) / max(1, len(self.analysis_latencies)),
            'symbols_tracked': len(self.market_data_history),
            'total_market_data_points': sum(
                len(history) for history in self.market_data_history.values()
            )
        }
    
    def clear_cache(self, symbol: str = None) -> None:
        """Clear analysis cache"""
        if symbol:
            self.analysis_cache.pop(symbol, None)
            self.cache_timestamps.pop(symbol, None)
        else:
            self.analysis_cache.clear()
            self.cache_timestamps.clear()
        
        logger.info(f"Analysis cache cleared for {'all symbols' if not symbol else symbol}")
    
    async def shutdown(self) -> None:
        """Shutdown microstructure engine"""
        # Cancel any running analysis tasks
        for task in self.analysis_tasks.values():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.analysis_tasks.clear()
        logger.info("MicrostructureEngine shutdown complete")