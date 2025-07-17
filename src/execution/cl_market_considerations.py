"""
CL Market Considerations Module
==============================

Specialized module for handling CL crude oil market-specific considerations.
Implements inventory report impacts, geopolitical risk assessment, session
liquidity analysis, and overnight gap risk management.

Key Features:
- EIA/API inventory report impact analysis
- Geopolitical risk scoring and adjustment
- Session-based liquidity assessment
- Overnight gap risk evaluation
- Currency correlation analysis
- Seasonal pattern recognition

Author: Agent 4 - Risk Management Mission
Date: 2025-07-17
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta, time
from dataclasses import dataclass, field
from enum import Enum
import json
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class InventoryReportType(Enum):
    """Types of inventory reports"""
    EIA_CRUDE = "eia_crude"
    EIA_GASOLINE = "eia_gasoline"
    EIA_DISTILLATE = "eia_distillate"
    API_CRUDE = "api_crude"
    IEA_MONTHLY = "iea_monthly"

class GeopoliticalRiskLevel(Enum):
    """Geopolitical risk levels"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class TradingSession(Enum):
    """Trading session types"""
    ASIAN = "asian"
    EUROPEAN = "european"
    US_PREMARKET = "us_premarket"
    US_REGULAR = "us_regular"
    US_AFTERHOURS = "us_afterhours"
    OVERNIGHT = "overnight"

@dataclass
class InventoryData:
    """Inventory report data structure"""
    report_type: InventoryReportType
    actual_change: float
    expected_change: float
    previous_change: float
    surprise: float
    timestamp: datetime
    significance: float = 0.0
    market_impact: float = 0.0

@dataclass
class GeopoliticalEvent:
    """Geopolitical event data structure"""
    event_id: str
    event_type: str
    region: str
    severity: GeopoliticalRiskLevel
    oil_relevance: float
    market_impact: float
    timestamp: datetime
    duration_estimate: timedelta
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SessionInfo:
    """Trading session information"""
    session: TradingSession
    start_time: time
    end_time: time
    liquidity_multiplier: float
    volatility_multiplier: float
    typical_volume: float
    major_participants: List[str]

class CLMarketAnalyzer:
    """
    Comprehensive CL market analysis system
    
    Analyzes market-specific factors affecting crude oil trading including
    inventory reports, geopolitical events, session characteristics, and
    seasonal patterns.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CL Market Analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Inventory report configuration
        self.inventory_config = config.get('inventory_reports', {})
        
        # Geopolitical risk configuration
        self.geopolitical_config = config.get('geopolitical_risk', {})
        
        # Session configuration
        self.session_config = config.get('trading_sessions', {})
        
        # Initialize sessions
        self.trading_sessions = self._initialize_trading_sessions()
        
        # Historical data storage
        self.inventory_history: List[InventoryData] = []
        self.geopolitical_events: List[GeopoliticalEvent] = []
        self.session_performance: Dict[str, Dict[str, float]] = {}
        
        # Market state tracking
        self.current_inventory_impact = 0.0
        self.current_geopolitical_risk = 0.0
        self.current_session_liquidity = 1.0
        self.overnight_gap_risk = 0.0
        
        # Currency correlations
        self.currency_correlations = {
            'USD': -0.6,   # Negative correlation with USD
            'EUR': 0.3,    # Positive correlation with EUR
            'JPY': -0.4,   # Negative correlation with JPY
            'GBP': 0.2     # Positive correlation with GBP
        }
        
        # Seasonal patterns
        self.seasonal_patterns = self._initialize_seasonal_patterns()
        
        logger.info("✅ CL Market Analyzer initialized")
        logger.info(f"   📊 Trading Sessions: {len(self.trading_sessions)}")
        logger.info(f"   📊 Inventory Reports: {len(self.inventory_config)}")
        logger.info(f"   📊 Geopolitical Factors: {len(self.geopolitical_config)}")
    
    def _initialize_trading_sessions(self) -> Dict[TradingSession, SessionInfo]:
        """Initialize trading session information"""
        sessions = {
            TradingSession.ASIAN: SessionInfo(
                session=TradingSession.ASIAN,
                start_time=time(18, 0),  # 6 PM ET
                end_time=time(2, 0),     # 2 AM ET
                liquidity_multiplier=0.6,
                volatility_multiplier=0.8,
                typical_volume=0.3,
                major_participants=["Asian hedge funds", "Commodity trading firms"]
            ),
            TradingSession.EUROPEAN: SessionInfo(
                session=TradingSession.EUROPEAN,
                start_time=time(2, 0),   # 2 AM ET
                end_time=time(9, 0),     # 9 AM ET
                liquidity_multiplier=0.8,
                volatility_multiplier=1.1,
                typical_volume=0.5,
                major_participants=["European banks", "Oil majors", "Sovereign funds"]
            ),
            TradingSession.US_PREMARKET: SessionInfo(
                session=TradingSession.US_PREMARKET,
                start_time=time(9, 0),   # 9 AM ET
                end_time=time(9, 30),    # 9:30 AM ET
                liquidity_multiplier=0.7,
                volatility_multiplier=1.3,
                typical_volume=0.4,
                major_participants=["US hedge funds", "Algorithmic traders"]
            ),
            TradingSession.US_REGULAR: SessionInfo(
                session=TradingSession.US_REGULAR,
                start_time=time(9, 30),  # 9:30 AM ET
                end_time=time(16, 0),    # 4 PM ET
                liquidity_multiplier=1.0,
                volatility_multiplier=1.0,
                typical_volume=1.0,
                major_participants=["All market participants"]
            ),
            TradingSession.US_AFTERHOURS: SessionInfo(
                session=TradingSession.US_AFTERHOURS,
                start_time=time(16, 0),  # 4 PM ET
                end_time=time(18, 0),    # 6 PM ET
                liquidity_multiplier=0.5,
                volatility_multiplier=0.9,
                typical_volume=0.2,
                major_participants=["Electronic trading systems"]
            ),
            TradingSession.OVERNIGHT: SessionInfo(
                session=TradingSession.OVERNIGHT,
                start_time=time(22, 0),  # 10 PM ET
                end_time=time(6, 0),     # 6 AM ET
                liquidity_multiplier=0.4,
                volatility_multiplier=0.7,
                typical_volume=0.1,
                major_participants=["Overnight funds", "International participants"]
            )
        }
        
        return sessions
    
    def _initialize_seasonal_patterns(self) -> Dict[str, Dict[str, float]]:
        """Initialize seasonal patterns for CL"""
        return {
            'monthly': {
                'january': 0.02,     # Winter demand
                'february': 0.01,    # Continued winter demand
                'march': -0.01,      # Spring transition
                'april': -0.02,      # Spring maintenance
                'may': 0.03,         # Driving season buildup
                'june': 0.04,        # Peak driving season
                'july': 0.05,        # Summer peak
                'august': 0.04,      # Continued summer demand
                'september': -0.01,  # Post-summer decline
                'october': -0.02,    # Autumn transition
                'november': 0.01,    # Winter preparation
                'december': 0.02     # Winter demand
            },
            'weekly': {
                'monday': 0.01,      # Week start
                'tuesday': 0.02,     # Active trading
                'wednesday': 0.03,   # Peak activity + inventory reports
                'thursday': 0.02,    # Continued activity
                'friday': -0.01,     # Position squaring
                'saturday': -0.02,   # Weekend effect
                'sunday': -0.01      # Pre-week positioning
            },
            'hourly': {
                'asian_peak': 0.8,       # Asian session peak
                'european_peak': 1.1,    # European session peak
                'us_open': 1.3,          # US market open
                'us_midday': 1.0,        # US midday
                'us_close': 0.9,         # US market close
                'overnight': 0.4         # Overnight trading
            }
        }
    
    async def analyze_inventory_impact(self, inventory_data: InventoryData) -> Dict[str, Any]:
        """
        Analyze inventory report impact on CL trading
        
        Args:
            inventory_data: Inventory report data
            
        Returns:
            Impact analysis results
        """
        try:
            # Calculate surprise factor
            surprise = (inventory_data.actual_change - inventory_data.expected_change) / 1000000  # Convert to millions
            
            # Determine significance based on surprise magnitude
            significance = min(abs(surprise) / 10.0, 1.0)  # 10M barrel surprise = 100% significant
            
            # Calculate market impact
            # Crude oil inventory: negative surprise (less build/more draw) = bullish
            if inventory_data.report_type in [InventoryReportType.EIA_CRUDE, InventoryReportType.API_CRUDE]:
                market_impact = -surprise * 0.1  # Invert for price impact
            else:
                market_impact = surprise * 0.05  # Product inventories have smaller impact
            
            # Apply time decay for older reports
            hours_since_report = (datetime.now() - inventory_data.timestamp).total_seconds() / 3600
            decay_factor = max(0.1, 1.0 - hours_since_report / 24.0)  # Decay over 24 hours
            
            adjusted_impact = market_impact * decay_factor
            
            # Store inventory data
            inventory_data.significance = significance
            inventory_data.market_impact = adjusted_impact
            self.inventory_history.append(inventory_data)
            
            # Keep only recent history
            if len(self.inventory_history) > 100:
                self.inventory_history = self.inventory_history[-100:]
            
            # Update current impact
            self.current_inventory_impact = adjusted_impact
            
            return {
                'inventory_type': inventory_data.report_type.value,
                'actual_change': inventory_data.actual_change,
                'expected_change': inventory_data.expected_change,
                'surprise': surprise,
                'significance': significance,
                'market_impact': adjusted_impact,
                'decay_factor': decay_factor,
                'trading_recommendation': self._generate_inventory_recommendation(adjusted_impact, significance),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing inventory impact: {e}")
            return {
                'error': str(e),
                'market_impact': 0.0,
                'significance': 0.0
            }
    
    def _generate_inventory_recommendation(self, impact: float, significance: float) -> Dict[str, Any]:
        """Generate trading recommendation based on inventory impact"""
        if significance < 0.2:
            return {
                'action': 'neutral',
                'confidence': 0.3,
                'reason': 'Low significance inventory report'
            }
        
        if impact > 0.02:  # Bullish impact
            return {
                'action': 'bullish',
                'confidence': min(significance * 2, 0.8),
                'reason': 'Bullish inventory surprise - consider long positions'
            }
        elif impact < -0.02:  # Bearish impact
            return {
                'action': 'bearish',
                'confidence': min(significance * 2, 0.8),
                'reason': 'Bearish inventory surprise - consider short positions'
            }
        else:
            return {
                'action': 'neutral',
                'confidence': 0.5,
                'reason': 'Neutral inventory impact'
            }
    
    async def assess_geopolitical_risk(self, 
                                     news_data: List[Dict[str, Any]],
                                     market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess geopolitical risk impact on CL trading
        
        Args:
            news_data: News/event data
            market_data: Current market data
            
        Returns:
            Geopolitical risk assessment
        """
        try:
            # Initialize risk components
            risk_components = {
                'middle_east': 0.0,
                'russia_ukraine': 0.0,
                'opec_decisions': 0.0,
                'us_sanctions': 0.0,
                'supply_disruptions': 0.0,
                'currency_instability': 0.0
            }
            
            # Analyze news sentiment and events
            for news_item in news_data:
                risk_contribution = self._analyze_news_item(news_item)
                
                # Add to appropriate risk component
                for component, value in risk_contribution.items():
                    if component in risk_components:
                        risk_components[component] += value
            
            # Calculate overall risk score
            overall_risk = np.mean(list(risk_components.values()))
            
            # Adjust based on market volatility
            current_volatility = market_data.get('volatility', 0.0)
            if current_volatility > 0.05:  # High volatility indicates elevated risk
                overall_risk *= 1.2
            
            # Classify risk level
            if overall_risk > 0.8:
                risk_level = GeopoliticalRiskLevel.CRITICAL
            elif overall_risk > 0.6:
                risk_level = GeopoliticalRiskLevel.HIGH
            elif overall_risk > 0.3:
                risk_level = GeopoliticalRiskLevel.MODERATE
            else:
                risk_level = GeopoliticalRiskLevel.LOW
            
            # Update current risk
            self.current_geopolitical_risk = overall_risk
            
            return {
                'overall_risk_score': overall_risk,
                'risk_level': risk_level.value,
                'risk_components': risk_components,
                'trading_impact': self._calculate_geopolitical_trading_impact(overall_risk),
                'recommendations': self._generate_geopolitical_recommendations(overall_risk, risk_level),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error assessing geopolitical risk: {e}")
            return {
                'overall_risk_score': 0.0,
                'risk_level': GeopoliticalRiskLevel.LOW.value,
                'error': str(e)
            }
    
    def _analyze_news_item(self, news_item: Dict[str, Any]) -> Dict[str, float]:
        """Analyze individual news item for risk contribution"""
        try:
            headline = news_item.get('headline', '').lower()
            content = news_item.get('content', '').lower()
            sentiment = news_item.get('sentiment', 0.0)
            
            risk_contribution = {}
            
            # Middle East keywords
            middle_east_keywords = ['iran', 'iraq', 'saudi', 'israel', 'yemen', 'syria', 'qatar', 'uae']
            if any(keyword in headline or keyword in content for keyword in middle_east_keywords):
                risk_contribution['middle_east'] = abs(sentiment) * 0.3
            
            # Russia/Ukraine keywords
            russia_keywords = ['russia', 'ukraine', 'putin', 'zelenskyy', 'moscow', 'kiev']
            if any(keyword in headline or keyword in content for keyword in russia_keywords):
                risk_contribution['russia_ukraine'] = abs(sentiment) * 0.25
            
            # OPEC keywords
            opec_keywords = ['opec', 'oil production', 'production cut', 'quota', 'cartel']
            if any(keyword in headline or keyword in content for keyword in opec_keywords):
                risk_contribution['opec_decisions'] = abs(sentiment) * 0.2
            
            # Sanctions keywords
            sanctions_keywords = ['sanctions', 'embargo', 'trade war', 'tariff']
            if any(keyword in headline or keyword in content for keyword in sanctions_keywords):
                risk_contribution['us_sanctions'] = abs(sentiment) * 0.15
            
            # Supply disruption keywords
            supply_keywords = ['pipeline', 'refinery', 'storm', 'hurricane', 'maintenance', 'outage']
            if any(keyword in headline or keyword in content for keyword in supply_keywords):
                risk_contribution['supply_disruptions'] = abs(sentiment) * 0.1
            
            return risk_contribution
            
        except Exception as e:
            logger.error(f"Error analyzing news item: {e}")
            return {}
    
    def _calculate_geopolitical_trading_impact(self, risk_score: float) -> Dict[str, float]:
        """Calculate trading impact of geopolitical risk"""
        return {
            'position_size_multiplier': max(0.5, 1.0 - risk_score * 0.5),
            'volatility_adjustment': 1.0 + risk_score * 0.3,
            'stop_loss_multiplier': 1.0 + risk_score * 0.2,
            'correlation_increase': risk_score * 0.1  # Increased correlation during risk events
        }
    
    def _generate_geopolitical_recommendations(self, 
                                             risk_score: float,
                                             risk_level: GeopoliticalRiskLevel) -> List[str]:
        """Generate geopolitical risk recommendations"""
        recommendations = []
        
        if risk_level == GeopoliticalRiskLevel.CRITICAL:
            recommendations.extend([
                "Reduce position sizes by 50% or more",
                "Tighten stop losses significantly",
                "Avoid new positions until risk subsides",
                "Consider hedging with options"
            ])
        elif risk_level == GeopoliticalRiskLevel.HIGH:
            recommendations.extend([
                "Reduce position sizes by 30-40%",
                "Tighten stop losses by 20%",
                "Limit new positions to high-confidence setups",
                "Monitor news closely for developments"
            ])
        elif risk_level == GeopoliticalRiskLevel.MODERATE:
            recommendations.extend([
                "Slight reduction in position sizes",
                "Normal stop loss levels",
                "Increased monitoring of risk events",
                "Consider volatility in position sizing"
            ])
        else:  # LOW
            recommendations.extend([
                "Normal position sizing",
                "Standard risk management",
                "Routine monitoring of geopolitical developments"
            ])
        
        return recommendations
    
    async def analyze_session_liquidity(self, current_time: datetime = None) -> Dict[str, Any]:
        """
        Analyze current session liquidity characteristics
        
        Args:
            current_time: Current time (defaults to now)
            
        Returns:
            Session liquidity analysis
        """
        try:
            if current_time is None:
                current_time = datetime.now()
            
            current_time_only = current_time.time()
            
            # Determine current session
            current_session = self._determine_current_session(current_time_only)
            session_info = self.trading_sessions[current_session]
            
            # Calculate liquidity factors
            liquidity_factors = {
                'base_liquidity': session_info.liquidity_multiplier,
                'volume_factor': session_info.typical_volume,
                'volatility_factor': session_info.volatility_multiplier,
                'participant_diversity': len(session_info.major_participants) / 10.0
            }
            
            # Adjust for day of week
            day_of_week = current_time.strftime('%A').lower()
            weekly_multiplier = self.seasonal_patterns['weekly'].get(day_of_week, 1.0)
            
            # Adjust for time of day
            hour = current_time.hour
            hourly_multiplier = self._get_hourly_multiplier(hour)
            
            # Calculate overall liquidity score
            overall_liquidity = (
                liquidity_factors['base_liquidity'] * 0.4 +
                liquidity_factors['volume_factor'] * 0.3 +
                liquidity_factors['volatility_factor'] * 0.2 +
                liquidity_factors['participant_diversity'] * 0.1
            ) * weekly_multiplier * hourly_multiplier
            
            # Update current session liquidity
            self.current_session_liquidity = overall_liquidity
            
            return {
                'current_session': current_session.value,
                'liquidity_score': overall_liquidity,
                'liquidity_factors': liquidity_factors,
                'weekly_multiplier': weekly_multiplier,
                'hourly_multiplier': hourly_multiplier,
                'major_participants': session_info.major_participants,
                'trading_recommendations': self._generate_liquidity_recommendations(overall_liquidity),
                'timestamp': current_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing session liquidity: {e}")
            return {
                'current_session': 'unknown',
                'liquidity_score': 0.5,
                'error': str(e)
            }
    
    def _determine_current_session(self, current_time: time) -> TradingSession:
        """Determine current trading session"""
        for session, info in self.trading_sessions.items():
            if info.start_time <= info.end_time:
                # Normal session (doesn't cross midnight)
                if info.start_time <= current_time <= info.end_time:
                    return session
            else:
                # Session crosses midnight
                if current_time >= info.start_time or current_time <= info.end_time:
                    return session
        
        # Default to overnight if no session found
        return TradingSession.OVERNIGHT
    
    def _get_hourly_multiplier(self, hour: int) -> float:
        """Get hourly liquidity multiplier"""
        if 9 <= hour <= 16:  # US trading hours
            return self.seasonal_patterns['hourly']['us_midday']
        elif 2 <= hour <= 9:  # European hours
            return self.seasonal_patterns['hourly']['european_peak']
        elif 18 <= hour <= 23 or 0 <= hour <= 2:  # Asian hours
            return self.seasonal_patterns['hourly']['asian_peak']
        else:  # Overnight
            return self.seasonal_patterns['hourly']['overnight']
    
    def _generate_liquidity_recommendations(self, liquidity_score: float) -> List[str]:
        """Generate liquidity-based recommendations"""
        recommendations = []
        
        if liquidity_score > 0.8:
            recommendations.extend([
                "High liquidity - normal position sizing",
                "Tight spreads expected",
                "Good conditions for larger trades"
            ])
        elif liquidity_score > 0.6:
            recommendations.extend([
                "Moderate liquidity - standard position sizing",
                "Monitor spreads closely",
                "Suitable for most trading strategies"
            ])
        elif liquidity_score > 0.4:
            recommendations.extend([
                "Reduced liquidity - consider smaller positions",
                "Wider spreads possible",
                "Be patient with order execution"
            ])
        else:
            recommendations.extend([
                "Low liquidity - reduce position sizes significantly",
                "Wide spreads expected",
                "Limit orders recommended over market orders",
                "Consider waiting for better liquidity"
            ])
        
        return recommendations
    
    async def calculate_overnight_gap_risk(self, 
                                         price_history: List[Dict[str, float]],
                                         current_positions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overnight gap risk for CL positions
        
        Args:
            price_history: Historical price data
            current_positions: Current position data
            
        Returns:
            Gap risk analysis
        """
        try:
            # Calculate historical gap sizes
            gaps = []
            for i in range(1, len(price_history)):
                prev_close = price_history[i-1]['close']
                current_open = price_history[i]['open']
                gap_size = abs(current_open - prev_close) / prev_close
                gaps.append(gap_size)
            
            # Calculate gap statistics
            if gaps:
                avg_gap = np.mean(gaps)
                max_gap = np.max(gaps)
                gap_95th = np.percentile(gaps, 95)
                gap_frequency = len([g for g in gaps if g > 0.01]) / len(gaps)  # Gaps > 1%
            else:
                avg_gap = max_gap = gap_95th = gap_frequency = 0.0
            
            # Assess current gap risk based on market conditions
            current_time = datetime.now()
            is_overnight = not (9 <= current_time.hour <= 16)
            
            # Risk factors
            risk_factors = {
                'overnight_session': 1.5 if is_overnight else 1.0,
                'geopolitical_risk': 1.0 + self.current_geopolitical_risk,
                'inventory_report_pending': self._check_inventory_report_pending(),
                'weekend_risk': 1.3 if current_time.weekday() >= 5 else 1.0,
                'holiday_risk': 1.2 if self._is_holiday_period(current_time) else 1.0
            }
            
            # Calculate position-specific gap risk
            position_gap_risk = {}
            total_gap_exposure = 0.0
            
            for symbol, position in current_positions.items():
                if 'CL' in symbol:
                    position_size = position.get('size', 0.0)
                    position_value = position.get('value', 0.0)
                    
                    # Estimated gap risk for this position
                    estimated_gap_risk = gap_95th * position_value * np.prod(list(risk_factors.values()))
                    
                    position_gap_risk[symbol] = {
                        'position_value': position_value,
                        'estimated_gap_risk': estimated_gap_risk,
                        'risk_percentage': estimated_gap_risk / position_value if position_value > 0 else 0
                    }
                    
                    total_gap_exposure += estimated_gap_risk
            
            # Update overnight gap risk
            self.overnight_gap_risk = total_gap_exposure
            
            return {
                'gap_statistics': {
                    'average_gap': avg_gap,
                    'maximum_gap': max_gap,
                    'gap_95th_percentile': gap_95th,
                    'gap_frequency': gap_frequency
                },
                'risk_factors': risk_factors,
                'position_gap_risk': position_gap_risk,
                'total_gap_exposure': total_gap_exposure,
                'recommendations': self._generate_gap_risk_recommendations(total_gap_exposure, risk_factors),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating overnight gap risk: {e}")
            return {
                'total_gap_exposure': 0.0,
                'error': str(e)
            }
    
    def _check_inventory_report_pending(self) -> float:
        """Check if inventory report is pending"""
        current_time = datetime.now()
        
        # EIA reports typically released Wednesday at 10:30 AM ET
        if current_time.weekday() == 2:  # Wednesday
            if current_time.hour < 10 or (current_time.hour == 10 and current_time.minute < 30):
                return 1.5  # Higher risk before report
        
        # API reports typically released Tuesday at 4:30 PM ET
        if current_time.weekday() == 1:  # Tuesday
            if current_time.hour < 16 or (current_time.hour == 16 and current_time.minute < 30):
                return 1.3  # Moderate risk before report
        
        return 1.0  # Normal risk
    
    def _is_holiday_period(self, current_time: datetime) -> bool:
        """Check if current time is near a holiday"""
        # Simplified holiday check - in production, use proper holiday calendar
        month = current_time.month
        day = current_time.day
        
        # Major holidays that affect oil markets
        holidays = [
            (1, 1),   # New Year's Day
            (7, 4),   # Independence Day
            (12, 25), # Christmas
            (11, 28), # Thanksgiving (approximate)
        ]
        
        for holiday_month, holiday_day in holidays:
            if month == holiday_month and abs(day - holiday_day) <= 1:
                return True
        
        return False
    
    def _generate_gap_risk_recommendations(self, 
                                         total_exposure: float,
                                         risk_factors: Dict[str, float]) -> List[str]:
        """Generate gap risk recommendations"""
        recommendations = []
        
        if total_exposure > 100000:  # High exposure
            recommendations.extend([
                "High overnight gap exposure - consider reducing positions",
                "Use stop losses to limit gap risk",
                "Consider hedging with options"
            ])
        elif total_exposure > 50000:  # Moderate exposure
            recommendations.extend([
                "Moderate gap exposure - monitor overnight",
                "Ensure adequate stop losses",
                "Consider position size adjustments"
            ])
        else:  # Low exposure
            recommendations.extend([
                "Low gap exposure - normal risk management",
                "Standard stop loss levels appropriate"
            ])
        
        # Add specific risk factor recommendations
        if risk_factors.get('geopolitical_risk', 1.0) > 1.3:
            recommendations.append("Elevated geopolitical risk - reduce overnight positions")
        
        if risk_factors.get('inventory_report_pending', 1.0) > 1.2:
            recommendations.append("Inventory report pending - expect higher volatility")
        
        if risk_factors.get('weekend_risk', 1.0) > 1.1:
            recommendations.append("Weekend approaching - consider closing positions")
        
        return recommendations
    
    async def analyze_currency_impact(self, 
                                    currency_data: Dict[str, float],
                                    cl_price: float) -> Dict[str, Any]:
        """
        Analyze currency impact on CL prices
        
        Args:
            currency_data: Currency exchange rates
            cl_price: Current CL price
            
        Returns:
            Currency impact analysis
        """
        try:
            currency_impacts = {}
            total_impact = 0.0
            
            for currency, rate in currency_data.items():
                if currency in self.currency_correlations:
                    correlation = self.currency_correlations[currency]
                    
                    # Calculate rate change (simplified)
                    # In production, this would use historical rate changes
                    rate_change = 0.0  # Placeholder
                    
                    # Calculate impact on CL price
                    price_impact = rate_change * correlation * cl_price
                    currency_impacts[currency] = {
                        'rate': rate,
                        'correlation': correlation,
                        'price_impact': price_impact
                    }
                    
                    total_impact += price_impact
            
            return {
                'currency_impacts': currency_impacts,
                'total_impact': total_impact,
                'impact_percentage': total_impact / cl_price if cl_price > 0 else 0,
                'dominant_currency': max(currency_impacts.keys(), 
                                       key=lambda x: abs(currency_impacts[x]['price_impact'])) if currency_impacts else None,
                'recommendations': self._generate_currency_recommendations(total_impact, currency_impacts),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing currency impact: {e}")
            return {
                'total_impact': 0.0,
                'error': str(e)
            }
    
    def _generate_currency_recommendations(self, 
                                         total_impact: float,
                                         currency_impacts: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate currency impact recommendations"""
        recommendations = []
        
        if abs(total_impact) > 1.0:  # Significant impact
            recommendations.extend([
                "Significant currency impact expected",
                "Monitor USD strength closely",
                "Consider currency hedging"
            ])
        elif abs(total_impact) > 0.5:  # Moderate impact
            recommendations.extend([
                "Moderate currency impact",
                "Factor currency moves into analysis",
                "Monitor major currency pairs"
            ])
        else:  # Low impact
            recommendations.extend([
                "Low currency impact",
                "Standard currency monitoring sufficient"
            ])
        
        # Add specific currency recommendations
        if currency_impacts.get('USD', {}).get('price_impact', 0) < -0.5:
            recommendations.append("USD strength may pressure oil prices")
        elif currency_impacts.get('USD', {}).get('price_impact', 0) > 0.5:
            recommendations.append("USD weakness may support oil prices")
        
        return recommendations
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get comprehensive market summary"""
        try:
            return {
                'timestamp': datetime.now().isoformat(),
                'inventory_impact': self.current_inventory_impact,
                'geopolitical_risk': self.current_geopolitical_risk,
                'session_liquidity': self.current_session_liquidity,
                'overnight_gap_risk': self.overnight_gap_risk,
                'recent_inventory_reports': len(self.inventory_history),
                'geopolitical_events': len(self.geopolitical_events),
                'market_conditions': {
                    'risk_level': 'high' if self.current_geopolitical_risk > 0.6 else 'moderate' if self.current_geopolitical_risk > 0.3 else 'low',
                    'liquidity_level': 'high' if self.current_session_liquidity > 0.8 else 'moderate' if self.current_session_liquidity > 0.4 else 'low',
                    'inventory_influence': 'significant' if abs(self.current_inventory_impact) > 0.02 else 'moderate' if abs(self.current_inventory_impact) > 0.01 else 'minimal'
                },
                'overall_recommendation': self._generate_overall_recommendation()
            }
        except Exception as e:
            logger.error(f"Error generating market summary: {e}")
            return {'error': str(e)}
    
    def _generate_overall_recommendation(self) -> str:
        """Generate overall market recommendation"""
        if self.current_geopolitical_risk > 0.7:
            return "High risk environment - reduce positions and tighten risk management"
        elif self.current_geopolitical_risk > 0.4 and self.current_session_liquidity < 0.5:
            return "Moderate risk with low liquidity - proceed with caution"
        elif abs(self.current_inventory_impact) > 0.03:
            return "Significant inventory impact - trade with inventory trends"
        elif self.current_session_liquidity > 0.8:
            return "Good liquidity conditions - normal trading strategies appropriate"
        else:
            return "Stable market conditions - standard risk management sufficient"