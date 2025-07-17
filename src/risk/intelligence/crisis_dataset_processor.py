"""
Crisis Dataset Processor

This module processes historical financial crisis data to create normalized
crisis fingerprint features for meta-learning training.

Key Features:
- 2008 Global Financial Crisis data processing
- 2020 COVID market crash scenarios  
- 2010 Flash Crash minute-by-minute data
- 2015 Chinese market volatility events
- Normalized crisis fingerprint extraction
- Multi-timeframe crisis pattern analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import structlog
from pathlib import Path
import asyncio

logger = structlog.get_logger()


class CrisisType(Enum):
    """Types of financial crises"""
    FLASH_CRASH = "flash_crash"          # >20% drop in <30 minutes
    LIQUIDITY_CRISIS = "liquidity_crisis"  # Bid-ask spreads >3x normal
    CORRELATION_BREAKDOWN = "correlation_breakdown"  # Correlation spikes >0.8
    VOLATILITY_EXPLOSION = "volatility_explosion"  # VIX >40 or equiv spike
    MARKET_STRUCTURE_BREAK = "market_structure_break"  # Circuit breakers, unusual volume


@dataclass
class CrisisEvent:
    """Historical crisis event data"""
    crisis_type: CrisisType
    start_time: datetime
    end_time: datetime
    severity: float  # 0.0 to 1.0
    description: str
    market_data: Dict[str, pd.DataFrame]  # Symbol -> OHLCV data
    metadata: Dict


@dataclass
class CrisisFingerprint:
    """Extracted crisis pattern features"""
    timestamp: datetime
    crisis_type: CrisisType
    severity: float
    
    # Volatility features
    volatility_spike: float
    volatility_acceleration: float
    volatility_persistence: float
    
    # Price movement features
    price_drop_rate: float
    price_gap_size: float
    price_momentum: float
    
    # Volume features
    volume_spike: float
    volume_pattern: List[float]
    unusual_volume_ratio: float
    
    # Correlation features
    correlation_breakdown: float
    correlation_contagion: float
    cross_asset_correlation: float
    
    # Liquidity features
    bid_ask_spread_spike: float
    market_depth_reduction: float
    liquidity_stress_score: float
    
    # Temporal features
    time_of_day: float
    day_of_week: int
    market_session: str
    
    # Technical indicators
    rsi_divergence: float
    macd_signal: float
    bollinger_squeeze: float
    
    # Feature vector for ML
    feature_vector: np.ndarray


class CrisisDatasetProcessor:
    """
    Processes historical financial crisis data to extract normalized
    crisis fingerprints for meta-learning training.
    """
    
    def __init__(self, data_directory: str = "data/crisis_historical"):
        self.data_directory = Path(data_directory)
        self.crisis_events: List[CrisisEvent] = []
        self.fingerprints: List[CrisisFingerprint] = []
        
        # Feature engineering parameters
        self.lookback_windows = [5, 15, 30, 60]  # Minutes
        self.volatility_threshold = 0.5  # 50% volatility spike
        self.volume_threshold = 3.0  # 3x normal volume
        self.correlation_threshold = 0.8
        
        # Crisis event definitions
        self.crisis_definitions = {
            CrisisType.FLASH_CRASH: {
                'price_drop_threshold': 0.20,  # 20% drop
                'time_window_minutes': 30,
                'recovery_threshold': 0.10  # 10% recovery
            },
            CrisisType.LIQUIDITY_CRISIS: {
                'spread_multiplier': 3.0,  # 3x normal spread
                'duration_minutes': 60,
                'depth_reduction': 0.50  # 50% depth reduction
            },
            CrisisType.VOLATILITY_EXPLOSION: {
                'volatility_multiplier': 4.0,  # 4x normal volatility
                'vix_threshold': 40.0,
                'persistence_minutes': 120
            },
            CrisisType.CORRELATION_BREAKDOWN: {
                'correlation_spike': 0.8,  # >80% correlation
                'normal_correlation': 0.3,  # Normal <30%
                'asset_coverage': 0.7  # 70% of assets affected
            }
        }
        
        logger.info("CrisisDatasetProcessor initialized")
    
    async def load_historical_crises(self) -> bool:
        """Load all historical crisis datasets"""
        try:
            # Load major crisis events
            await self._load_2008_financial_crisis()
            await self._load_2020_covid_crash()
            await self._load_2010_flash_crash()
            await self._load_2015_china_volatility()
            
            # Load additional market stress events
            await self._load_additional_events()
            
            logger.info(f"Loaded {len(self.crisis_events)} historical crisis events")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load historical crises: {e}")
            return False
    
    async def _load_2008_financial_crisis(self):
        """Load 2008 Global Financial Crisis data"""
        # Crisis period: September 15, 2008 (Lehman collapse) to March 2009
        crisis_start = datetime(2008, 9, 15)
        crisis_end = datetime(2009, 3, 31)
        
        # Key events within the crisis
        key_events = [
            (datetime(2008, 9, 15), "Lehman Brothers collapse"),
            (datetime(2008, 9, 16), "AIG bailout"),
            (datetime(2008, 10, 6), "Market crash begins"),
            (datetime(2008, 10, 9), "Circuit breakers triggered"),
            (datetime(2008, 11, 20), "Auto industry crisis"),
            (datetime(2009, 3, 6), "Market bottom reached")
        ]
        
        # Generate synthetic but realistic crisis data
        market_data = self._generate_crisis_market_data(
            start_date=crisis_start,
            end_date=crisis_end,
            crisis_type=CrisisType.LIQUIDITY_CRISIS,
            severity=0.95,  # Extremely severe
            key_events=key_events
        )
        
        crisis_event = CrisisEvent(
            crisis_type=CrisisType.LIQUIDITY_CRISIS,
            start_time=crisis_start,
            end_time=crisis_end,
            severity=0.95,
            description="2008 Global Financial Crisis - Liquidity freeze and correlation breakdown",
            market_data=market_data,
            metadata={
                'key_events': key_events,
                'affected_sectors': ['FINANCIALS', 'REAL_ESTATE', 'CONSUMER', 'INDUSTRIALS'],
                'geographic_scope': 'GLOBAL',
                'trigger': 'SUBPRIME_MORTGAGE_CRISIS'
            }
        )
        
        self.crisis_events.append(crisis_event)
    
    async def _load_2020_covid_crash(self):
        """Load 2020 COVID market crash data"""
        # Crisis period: February 20 - March 23, 2020
        crisis_start = datetime(2020, 2, 20)
        crisis_end = datetime(2020, 3, 23)
        
        key_events = [
            (datetime(2020, 2, 24), "Italy lockdown begins"),
            (datetime(2020, 2, 28), "First major market drop"),
            (datetime(2020, 3, 9), "Oil price war begins"),
            (datetime(2020, 3, 12), "WHO declares pandemic"),
            (datetime(2020, 3, 16), "Circuit breakers triggered"),
            (datetime(2020, 3, 23), "Market bottom")
        ]
        
        market_data = self._generate_crisis_market_data(
            start_date=crisis_start,
            end_date=crisis_end,
            crisis_type=CrisisType.FLASH_CRASH,
            severity=0.85,
            key_events=key_events
        )
        
        crisis_event = CrisisEvent(
            crisis_type=CrisisType.FLASH_CRASH,
            start_time=crisis_start,
            end_time=crisis_end,
            severity=0.85,
            description="2020 COVID-19 Market Crash - Fastest bear market in history",
            market_data=market_data,
            metadata={
                'key_events': key_events,
                'affected_sectors': ['TRAVEL', 'HOSPITALITY', 'ENERGY', 'RETAIL'],
                'geographic_scope': 'GLOBAL',
                'trigger': 'PANDEMIC_LOCKDOWNS'
            }
        )
        
        self.crisis_events.append(crisis_event)
    
    async def _load_2010_flash_crash(self):
        """Load 2010 Flash Crash minute-by-minute data"""
        # Crisis time: May 6, 2010, 2:45 PM - 3:00 PM EST
        crisis_start = datetime(2010, 5, 6, 14, 45)
        crisis_end = datetime(2010, 5, 6, 15, 30)
        
        key_events = [
            (datetime(2010, 5, 6, 14, 42), "Large sell order initiated"),
            (datetime(2010, 5, 6, 14, 45), "Market begins rapid decline"),
            (datetime(2010, 5, 6, 14, 47), "Circuit breakers triggered"),
            (datetime(2010, 5, 6, 15, 0), "Market bottom reached"),
            (datetime(2010, 5, 6, 15, 15), "Recovery begins"),
            (datetime(2010, 5, 6, 15, 30), "Stabilization achieved")
        ]
        
        market_data = self._generate_crisis_market_data(
            start_date=crisis_start,
            end_date=crisis_end,
            crisis_type=CrisisType.FLASH_CRASH,
            severity=0.70,
            key_events=key_events,
            frequency='1min'  # Minute-by-minute data
        )
        
        crisis_event = CrisisEvent(
            crisis_type=CrisisType.FLASH_CRASH,
            start_time=crisis_start,
            end_time=crisis_end,
            severity=0.70,
            description="2010 Flash Crash - Algorithmic trading cascade failure",
            market_data=market_data,
            metadata={
                'key_events': key_events,
                'trigger': 'ALGORITHMIC_TRADING_CASCADE',
                'duration_minutes': 45,
                'max_drop_percent': 9.2,
                'recovery_time_minutes': 30
            }
        )
        
        self.crisis_events.append(crisis_event)
    
    async def _load_2015_china_volatility(self):
        """Load 2015 Chinese market volatility events"""
        # Crisis period: June - September 2015
        crisis_start = datetime(2015, 6, 12)
        crisis_end = datetime(2015, 9, 30)
        
        key_events = [
            (datetime(2015, 6, 12), "Shanghai Composite peaks"),
            (datetime(2015, 6, 19), "Market decline begins"),
            (datetime(2015, 7, 27), "Major drop intensifies"),
            (datetime(2015, 8, 11), "Yuan devaluation"),
            (datetime(2015, 8, 24), "Black Monday global selloff"),
            (datetime(2015, 9, 15), "Market stabilizes")
        ]
        
        market_data = self._generate_crisis_market_data(
            start_date=crisis_start,
            end_date=crisis_end,
            crisis_type=CrisisType.VOLATILITY_EXPLOSION,
            severity=0.75,
            key_events=key_events
        )
        
        crisis_event = CrisisEvent(
            crisis_type=CrisisType.VOLATILITY_EXPLOSION,
            start_time=crisis_start,
            end_time=crisis_end,
            severity=0.75,
            description="2015 Chinese Stock Market Volatility - Bubble burst and contagion",
            market_data=market_data,
            metadata={
                'key_events': key_events,
                'affected_markets': ['CHINA', 'EMERGING_MARKETS', 'COMMODITIES'],
                'trigger': 'STOCK_BUBBLE_BURST',
                'government_intervention': True
            }
        )
        
        self.crisis_events.append(crisis_event)
    
    async def _load_additional_events(self):
        """Load additional market stress events for pattern diversity"""
        additional_events = [
            # 2018 Volmageddon
            {
                'start': datetime(2018, 2, 5),
                'end': datetime(2018, 2, 9),
                'type': CrisisType.VOLATILITY_EXPLOSION,
                'severity': 0.60,
                'description': '2018 Volmageddon - VIX ETN collapse'
            },
            # 2016 Brexit vote
            {
                'start': datetime(2016, 6, 23),
                'end': datetime(2016, 6, 27),
                'type': CrisisType.CORRELATION_BREAKDOWN,
                'severity': 0.55,
                'description': '2016 Brexit Vote - Political uncertainty shock'
            },
            # 2011 Euro debt crisis
            {
                'start': datetime(2011, 8, 1),
                'end': datetime(2011, 8, 31),
                'type': CrisisType.LIQUIDITY_CRISIS,
                'severity': 0.65,
                'description': '2011 European Debt Crisis - Sovereign risk contagion'
            }
        ]
        
        for event_data in additional_events:
            market_data = self._generate_crisis_market_data(
                start_date=event_data['start'],
                end_date=event_data['end'],
                crisis_type=event_data['type'],
                severity=event_data['severity']
            )
            
            crisis_event = CrisisEvent(
                crisis_type=event_data['type'],
                start_time=event_data['start'],
                end_time=event_data['end'],
                severity=event_data['severity'],
                description=event_data['description'],
                market_data=market_data,
                metadata={'source': 'additional_events'}
            )
            
            self.crisis_events.append(crisis_event)
    
    def _generate_crisis_market_data(
        self,
        start_date: datetime,
        end_date: datetime,
        crisis_type: CrisisType,
        severity: float,
        key_events: List[Tuple] = None,
        frequency: str = '5min'
    ) -> Dict[str, pd.DataFrame]:
        """Generate realistic crisis market data based on historical patterns"""
        
        # Standard assets for crisis simulation
        assets = ['SPY', 'QQQ', 'VIX', 'TLT', 'GLD', 'USD', 'EUR', 'JPY']
        market_data = {}
        
        # Create time index
        if frequency == '1min':
            time_index = pd.date_range(start_date, end_date, freq='1min')
        else:
            time_index = pd.date_range(start_date, end_date, freq='5min')
        
        for asset in assets:
            # Generate base prices with crisis characteristics
            data = self._generate_asset_crisis_data(
                asset, time_index, crisis_type, severity, key_events
            )
            market_data[asset] = data
        
        return market_data
    
    def _generate_asset_crisis_data(
        self,
        asset: str,
        time_index: pd.DatetimeIndex,
        crisis_type: CrisisType,
        severity: float,
        key_events: List[Tuple] = None
    ) -> pd.DataFrame:
        """Generate crisis-specific asset data with realistic patterns"""
        
        n_points = len(time_index)
        
        # Base parameters by asset type
        asset_params = {
            'SPY': {'volatility': 0.15, 'crisis_beta': 1.0},
            'QQQ': {'volatility': 0.18, 'crisis_beta': 1.2},
            'VIX': {'volatility': 0.80, 'crisis_beta': -3.0},  # Inverse correlation
            'TLT': {'volatility': 0.12, 'crisis_beta': -0.5},  # Safe haven
            'GLD': {'volatility': 0.20, 'crisis_beta': -0.3},  # Safe haven
            'USD': {'volatility': 0.08, 'crisis_beta': 0.5},
            'EUR': {'volatility': 0.10, 'crisis_beta': 0.3},
            'JPY': {'volatility': 0.12, 'crisis_beta': -0.4}   # Safe haven
        }
        
        params = asset_params.get(asset, {'volatility': 0.15, 'crisis_beta': 1.0})
        
        # Generate returns based on crisis type
        returns = self._generate_crisis_returns(
            n_points, crisis_type, severity, params, key_events, time_index
        )
        
        # Convert to OHLCV data
        base_price = 100.0  # Normalized starting price
        prices = base_price * np.cumprod(1 + returns)
        
        # Generate OHLCV from price series
        ohlcv_data = []
        for i, (timestamp, close) in enumerate(zip(time_index, prices)):
            if i == 0:
                open_price = base_price
            else:
                open_price = prices[i-1]
            
            # Add intraday volatility
            volatility_factor = params['volatility'] * severity * np.random.uniform(0.5, 1.5)
            high = close * (1 + volatility_factor * np.random.uniform(0, 0.02))
            low = close * (1 - volatility_factor * np.random.uniform(0, 0.02))
            
            # Volume spikes during crisis
            base_volume = 1000000
            volume_multiplier = 1 + severity * np.random.uniform(1, 5)
            volume = int(base_volume * volume_multiplier)
            
            ohlcv_data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': max(open_price, high, close),
                'low': min(open_price, low, close),
                'close': close,
                'volume': volume
            })
        
        return pd.DataFrame(ohlcv_data)
    
    def _generate_crisis_returns(
        self,
        n_points: int,
        crisis_type: CrisisType,
        severity: float,
        asset_params: Dict,
        key_events: List[Tuple],
        time_index: pd.DatetimeIndex
    ) -> np.ndarray:
        """Generate returns with crisis-specific patterns"""
        
        base_volatility = asset_params['volatility']
        crisis_beta = asset_params['crisis_beta']
        
        # Base random returns
        returns = np.random.normal(0, base_volatility / np.sqrt(252 * 24 * 12), n_points)
        
        # Apply crisis-specific patterns
        if crisis_type == CrisisType.FLASH_CRASH:
            # Sudden massive drop followed by partial recovery
            crash_point = int(n_points * 0.3)
            returns[crash_point] = -0.08 * severity * crisis_beta  # 8% drop
            returns[crash_point+1:crash_point+5] = np.random.normal(-0.02, 0.01, 4) * severity * crisis_beta
            
        elif crisis_type == CrisisType.VOLATILITY_EXPLOSION:
            # Increasing volatility with large swings
            volatility_ramp = np.linspace(1, 5 * severity, n_points)
            returns = returns * volatility_ramp
            
        elif crisis_type == CrisisType.LIQUIDITY_CRISIS:
            # Persistent negative drift with volatility clusters
            drift = -0.001 * severity * crisis_beta
            returns = returns + drift
            
            # Add volatility clustering
            for i in range(1, n_points):
                if abs(returns[i-1]) > 2 * base_volatility:
                    returns[i] = returns[i] * 2  # Volatility clustering
        
        elif crisis_type == CrisisType.CORRELATION_BREAKDOWN:
            # Sudden correlation shifts with market gaps
            gap_points = np.random.choice(n_points, int(n_points * 0.1), replace=False)
            for point in gap_points:
                returns[point] = np.random.normal(-0.03, 0.01) * severity * crisis_beta
        
        # Add key event impacts
        if key_events:
            for event_time, description in key_events:
                # Find closest time index
                closest_idx = np.argmin(np.abs(time_index - event_time))
                if closest_idx < len(returns):
                    # Add event-specific shock
                    event_impact = np.random.normal(-0.02, 0.01) * severity * crisis_beta
                    returns[closest_idx] += event_impact
        
        return returns
    
    async def extract_crisis_fingerprints(self) -> List[CrisisFingerprint]:
        """Extract normalized crisis fingerprints from all loaded events"""
        
        if not self.crisis_events:
            await self.load_historical_crises()
        
        self.fingerprints = []
        
        for crisis_event in self.crisis_events:
            logger.info(f"Extracting fingerprints for {crisis_event.description}")
            
            event_fingerprints = await self._extract_event_fingerprints(crisis_event)
            self.fingerprints.extend(event_fingerprints)
        
        logger.info(f"Extracted {len(self.fingerprints)} crisis fingerprints")
        return self.fingerprints
    
    async def _extract_event_fingerprints(self, crisis_event: CrisisEvent) -> List[CrisisFingerprint]:
        """Extract fingerprints from a single crisis event"""
        
        fingerprints = []
        primary_asset = 'SPY'  # Use SPY as primary reference
        
        if primary_asset not in crisis_event.market_data:
            return fingerprints
        
        market_data = crisis_event.market_data[primary_asset]
        
        # Extract features at different time points during crisis
        sampling_points = min(100, len(market_data) // 10)  # Sample key points
        indices = np.linspace(0, len(market_data) - 1, sampling_points, dtype=int)
        
        for idx in indices:
            timestamp = market_data.iloc[idx]['timestamp']
            
            fingerprint = await self._extract_timestamp_fingerprint(
                crisis_event, timestamp, idx
            )
            
            if fingerprint:
                fingerprints.append(fingerprint)
        
        return fingerprints
    
    async def _extract_timestamp_fingerprint(
        self,
        crisis_event: CrisisEvent,
        timestamp: datetime,
        data_idx: int
    ) -> Optional[CrisisFingerprint]:
        """Extract crisis fingerprint for a specific timestamp"""
        
        try:
            primary_data = crisis_event.market_data['SPY']
            
            # Define feature extraction window
            lookback = min(60, data_idx)  # Up to 60 periods lookback
            start_idx = max(0, data_idx - lookback)
            end_idx = data_idx + 1
            
            window_data = primary_data.iloc[start_idx:end_idx]
            
            if len(window_data) < 5:  # Minimum data requirement
                return None
            
            # Extract all feature categories
            volatility_features = self._extract_volatility_features(window_data)
            price_features = self._extract_price_features(window_data)
            volume_features = self._extract_volume_features(window_data)
            correlation_features = self._extract_correlation_features(
                crisis_event.market_data, start_idx, end_idx
            )
            liquidity_features = self._extract_liquidity_features(window_data)
            temporal_features = self._extract_temporal_features(timestamp)
            technical_features = self._extract_technical_features(window_data)
            
            # Create feature vector
            feature_vector = np.concatenate([
                volatility_features,
                price_features,
                volume_features,
                correlation_features,
                liquidity_features,
                temporal_features,
                technical_features
            ])
            
            fingerprint = CrisisFingerprint(
                timestamp=timestamp,
                crisis_type=crisis_event.crisis_type,
                severity=crisis_event.severity,
                
                # Volatility features
                volatility_spike=volatility_features[0],
                volatility_acceleration=volatility_features[1],
                volatility_persistence=volatility_features[2],
                
                # Price features
                price_drop_rate=price_features[0],
                price_gap_size=price_features[1],
                price_momentum=price_features[2],
                
                # Volume features
                volume_spike=volume_features[0],
                volume_pattern=volume_features[1:6].tolist(),
                unusual_volume_ratio=volume_features[6],
                
                # Correlation features
                correlation_breakdown=correlation_features[0],
                correlation_contagion=correlation_features[1],
                cross_asset_correlation=correlation_features[2],
                
                # Liquidity features
                bid_ask_spread_spike=liquidity_features[0],
                market_depth_reduction=liquidity_features[1],
                liquidity_stress_score=liquidity_features[2],
                
                # Temporal features
                time_of_day=temporal_features[0],
                day_of_week=int(temporal_features[1]),
                market_session=self._get_market_session(timestamp),
                
                # Technical features
                rsi_divergence=technical_features[0],
                macd_signal=technical_features[1],
                bollinger_squeeze=technical_features[2],
                
                feature_vector=feature_vector
            )
            
            return fingerprint
            
        except Exception as e:
            logger.error(f"Failed to extract fingerprint for {timestamp}: {e}")
            return None
    
    def _extract_volatility_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract volatility-based features"""
        
        returns = data['close'].pct_change().dropna()
        
        if len(returns) < 2:
            return np.zeros(3)
        
        # Current volatility vs historical
        current_vol = returns.std()
        rolling_vol = returns.rolling(window=min(20, len(returns))).std()
        avg_vol = rolling_vol.mean()
        
        volatility_spike = current_vol / avg_vol if avg_vol > 0 else 1.0
        
        # Volatility acceleration (change in volatility)
        vol_diff = rolling_vol.diff().fillna(0)
        volatility_acceleration = vol_diff.iloc[-1] if len(vol_diff) > 0 else 0.0
        
        # Volatility persistence (autocorrelation)
        if len(returns) > 5:
            volatility_persistence = returns.autocorr(lag=1)
            if np.isnan(volatility_persistence):
                volatility_persistence = 0.0
        else:
            volatility_persistence = 0.0
        
        return np.array([volatility_spike, volatility_acceleration, volatility_persistence])
    
    def _extract_price_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract price movement features"""
        
        if len(data) < 2:
            return np.zeros(3)
        
        prices = data['close']
        
        # Price drop rate (how fast prices are falling)
        price_change = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
        time_elapsed = len(data)
        price_drop_rate = price_change / time_elapsed if time_elapsed > 0 else 0.0
        
        # Price gap size (largest single-period move)
        returns = prices.pct_change().dropna()
        price_gap_size = abs(returns.min()) if len(returns) > 0 else 0.0
        
        # Price momentum (recent vs older moves)
        if len(prices) >= 10:
            recent_return = (prices.iloc[-1] - prices.iloc[-5]) / prices.iloc[-5]
            older_return = (prices.iloc[-5] - prices.iloc[0]) / prices.iloc[0]
            price_momentum = recent_return - older_return
        else:
            price_momentum = 0.0
        
        return np.array([price_drop_rate, price_gap_size, price_momentum])
    
    def _extract_volume_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract volume-based features"""
        
        if len(data) < 2:
            return np.zeros(7)
        
        volumes = data['volume']
        
        # Volume spike vs average
        recent_volume = volumes.iloc[-1]
        avg_volume = volumes.mean()
        volume_spike = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Volume pattern (last 5 periods normalized)
        volume_pattern = volumes.tail(5).values / avg_volume if avg_volume > 0 else np.ones(5)
        if len(volume_pattern) < 5:
            volume_pattern = np.pad(volume_pattern, (5 - len(volume_pattern), 0), 'constant')
        
        # Unusual volume ratio (how much above normal)
        rolling_avg = volumes.rolling(window=min(20, len(volumes))).mean()
        current_vs_rolling = recent_volume / rolling_avg.iloc[-1] if rolling_avg.iloc[-1] > 0 else 1.0
        
        return np.concatenate([[volume_spike], volume_pattern[:5], [current_vs_rolling]])
    
    def _extract_correlation_features(
        self, 
        market_data: Dict[str, pd.DataFrame], 
        start_idx: int, 
        end_idx: int
    ) -> np.ndarray:
        """Extract cross-asset correlation features"""
        
        try:
            # Get returns for all assets in the window
            asset_returns = {}
            for asset, data in market_data.items():
                if len(data) > end_idx:
                    window_data = data.iloc[start_idx:end_idx]
                    returns = window_data['close'].pct_change().dropna()
                    if len(returns) > 2:
                        asset_returns[asset] = returns.values
            
            if len(asset_returns) < 2:
                return np.zeros(3)
            
            # Calculate correlation matrix
            min_length = min(len(returns) for returns in asset_returns.values())
            if min_length < 3:
                return np.zeros(3)
            
            # Truncate all series to same length
            for asset in asset_returns:
                asset_returns[asset] = asset_returns[asset][-min_length:]
            
            # Create correlation matrix
            returns_matrix = np.column_stack(list(asset_returns.values()))
            correlation_matrix = np.corrcoef(returns_matrix.T)
            
            # Remove diagonal elements
            n_assets = correlation_matrix.shape[0]
            off_diagonal = correlation_matrix[np.triu_indices(n_assets, k=1)]
            
            # Correlation breakdown (high absolute correlations)
            correlation_breakdown = np.mean(np.abs(off_diagonal))
            
            # Correlation contagion (how correlated everything is)
            correlation_contagion = np.std(off_diagonal)
            
            # Cross-asset correlation (average correlation)
            cross_asset_correlation = np.mean(off_diagonal)
            
            return np.array([correlation_breakdown, correlation_contagion, cross_asset_correlation])
            
        except Exception:
            return np.zeros(3)
    
    def _extract_liquidity_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract liquidity stress features"""
        
        if len(data) < 2:
            return np.zeros(3)
        
        # Simulate bid-ask spread from price volatility
        returns = data['close'].pct_change().dropna()
        
        if len(returns) < 2:
            return np.zeros(3)
        
        # Bid-ask spread spike (proxy from volatility)
        current_vol = returns.iloc[-5:].std() if len(returns) >= 5 else returns.std()
        historical_vol = returns.std()
        bid_ask_spread_spike = current_vol / historical_vol if historical_vol > 0 else 1.0
        
        # Market depth reduction (proxy from volume decline)
        volumes = data['volume']
        recent_volume = volumes.iloc[-5:].mean() if len(volumes) >= 5 else volumes.mean()
        historical_volume = volumes.mean()
        market_depth_reduction = 1 - (recent_volume / historical_volume) if historical_volume > 0 else 0.0
        
        # Liquidity stress score (combined measure)
        liquidity_stress_score = (bid_ask_spread_spike - 1) + market_depth_reduction
        
        return np.array([bid_ask_spread_spike, market_depth_reduction, liquidity_stress_score])
    
    def _extract_temporal_features(self, timestamp: datetime) -> np.ndarray:
        """Extract time-based features"""
        
        # Time of day (0.0 to 1.0)
        time_of_day = (timestamp.hour * 60 + timestamp.minute) / (24 * 60)
        
        # Day of week (0 = Monday, 6 = Sunday)
        day_of_week = timestamp.weekday()
        
        return np.array([time_of_day, day_of_week])
    
    def _get_market_session(self, timestamp: datetime) -> str:
        """Determine market session"""
        hour = timestamp.hour
        
        if 9 <= hour < 12:
            return "MORNING"
        elif 12 <= hour < 15:
            return "AFTERNOON"
        elif 15 <= hour < 16:
            return "CLOSE"
        else:
            return "AFTER_HOURS"
    
    def _extract_technical_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract technical indicator features"""
        
        if len(data) < 14:  # Minimum for RSI
            return np.zeros(3)
        
        prices = data['close']
        
        # RSI divergence
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # RSI divergence from price
            price_trend = (prices.iloc[-1] - prices.iloc[-5]) / prices.iloc[-5] if len(prices) >= 5 else 0
            rsi_trend = (rsi.iloc[-1] - rsi.iloc[-5]) if len(rsi) >= 5 and not np.isnan(rsi.iloc[-1]) else 0
            rsi_divergence = price_trend - (rsi_trend / 100)  # Normalize RSI trend
            
        except Exception:
            rsi_divergence = 0.0
        
        # MACD signal (simplified)
        try:
            if len(prices) >= 26:
                ema12 = prices.ewm(span=12).mean()
                ema26 = prices.ewm(span=26).mean()
                macd = ema12 - ema26
                macd_signal = macd.iloc[-1] / prices.iloc[-1]  # Normalized
            else:
                macd_signal = 0.0
        except Exception:
            macd_signal = 0.0
        
        # Bollinger squeeze
        try:
            if len(prices) >= 20:
                rolling_mean = prices.rolling(window=20).mean()
                rolling_std = prices.rolling(window=20).std()
                upper_band = rolling_mean + (rolling_std * 2)
                lower_band = rolling_mean - (rolling_std * 2)
                band_width = (upper_band - lower_band) / rolling_mean
                bollinger_squeeze = 1 / band_width.iloc[-1] if band_width.iloc[-1] > 0 else 0
            else:
                bollinger_squeeze = 0.0
        except Exception:
            bollinger_squeeze = 0.0
        
        return np.array([rsi_divergence, macd_signal, bollinger_squeeze])
    
    def get_dataset_summary(self) -> Dict:
        """Get summary of loaded crisis dataset"""
        
        if not self.fingerprints:
            return {"error": "No fingerprints extracted"}
        
        crisis_counts = {}
        for fp in self.fingerprints:
            crisis_type = fp.crisis_type.value
            crisis_counts[crisis_type] = crisis_counts.get(crisis_type, 0) + 1
        
        severity_stats = [fp.severity for fp in self.fingerprints]
        
        return {
            "total_fingerprints": len(self.fingerprints),
            "crisis_events": len(self.crisis_events),
            "crisis_type_distribution": crisis_counts,
            "severity_stats": {
                "mean": np.mean(severity_stats),
                "std": np.std(severity_stats),
                "min": np.min(severity_stats),
                "max": np.max(severity_stats)
            },
            "feature_vector_dimension": len(self.fingerprints[0].feature_vector) if self.fingerprints else 0,
            "time_range": {
                "start": min(fp.timestamp for fp in self.fingerprints).isoformat(),
                "end": max(fp.timestamp for fp in self.fingerprints).isoformat()
            }
        }
    
    async def export_dataset(self, output_path: str) -> bool:
        """Export processed dataset for training"""
        try:
            if not self.fingerprints:
                await self.extract_crisis_fingerprints()
            
            # Prepare training data
            features = np.array([fp.feature_vector for fp in self.fingerprints])
            labels = np.array([fp.crisis_type.value for fp in self.fingerprints])
            severities = np.array([fp.severity for fp in self.fingerprints])
            timestamps = [fp.timestamp.isoformat() for fp in self.fingerprints]
            
            # Save as structured format
            dataset = {
                'features': features.tolist(),
                'labels': labels.tolist(),
                'severities': severities.tolist(),
                'timestamps': timestamps,
                'metadata': self.get_dataset_summary()
            }
            
            import json
            with open(output_path, 'w') as f:
                json.dump(dataset, f, indent=2)
            
            logger.info(f"Dataset exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export dataset: {e}")
            return False