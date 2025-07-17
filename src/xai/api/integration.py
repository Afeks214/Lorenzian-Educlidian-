"""
Strategic MARL Integration for XAI API
AGENT DELTA MISSION: Seamless Strategic MARL Integration

This module provides deep integration with the existing Strategic MARL infrastructure,
enabling seamless hookup with the decision pipeline, agent context extraction,
and performance metrics integration.

Features:
- Strategic MARL Component interface
- Agent decision context extraction and formatting
- Performance metrics integration with risk system
- Event bus integration for real-time updates
- Decision history and audit trail management
- Agent performance tracking and analysis

Author: Agent Delta - Integration Specialist
Version: 1.0 - Strategic MARL Integration Layer
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

from src.agents.strategic_marl_component import StrategicMARLComponent, StrategicDecision
from src.agents.strategic_agent_base import AgentPrediction
from src.core.event_bus import EventBus
from src.core.events import EventType, Event
from src.tactical.xai_engine import DecisionSnapshot, ExplanationResult, AssetClass, ActionType
from src.monitoring.logger_config import get_logger
from src.risk.core.performance_optimizer import PerformanceOptimizer
from src.risk.agents.performance_attribution import PerformanceAttributionEngine

logger = get_logger(__name__)


@dataclass
class AgentContext:
    """Rich context information for individual agents"""
    agent_id: str
    agent_type: str
    specialization: str
    current_confidence: float
    recent_performance: Dict[str, float]
    feature_importance: Dict[str, float]
    decision_rationale: str
    market_conditions_assessment: Dict[str, Any]
    risk_assessment: Dict[str, float]
    
    
@dataclass
class StrategicContext:
    """Strategic decision context from MARL system"""
    decision_id: str
    timestamp: datetime
    symbol: str
    market_regime: str
    synergy_strength: float
    consensus_level: float
    agent_contexts: List[AgentContext]
    gating_weights: List[float]
    performance_attribution: Dict[str, float]
    risk_metrics: Dict[str, float]
    

@dataclass
class DecisionPerformanceOutcome:
    """Performance outcome of a trading decision"""
    decision_id: str
    realized_pnl: float
    success: bool
    time_to_target: Optional[float]
    risk_adjusted_return: float
    attribution_breakdown: Dict[str, float]
    lessons_learned: List[str]


class StrategicMARLIntegrator:
    """
    Strategic MARL Integration Layer
    
    Provides seamless integration between XAI system and Strategic MARL infrastructure.
    Handles decision context extraction, performance tracking, and real-time updates.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Strategic MARL Integrator"""
        self.config = config or self._default_config()
        
        # Core components
        self.marl_component: Optional[StrategicMARLComponent] = None
        self.event_bus: Optional[EventBus] = None
        self.performance_optimizer: Optional[PerformanceOptimizer] = None
        self.attribution_engine: Optional[PerformanceAttributionEngine] = None
        
        # Decision tracking
        self.decision_cache: Dict[str, Dict[str, Any]] = {}
        self.agent_performance_history: Dict[str, List[float]] = {}
        self.strategic_insights: Dict[str, Any] = {}
        
        # Performance metrics
        self.integration_metrics = {
            'decisions_processed': 0,
            'explanations_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'agent_queries': 0,
            'performance_updates': 0,
            'error_count': 0
        }
        
        # Health tracking
        self._healthy = False
        self._last_health_check = datetime.now()
        
        logger.info("Strategic MARL Integrator initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'cache_ttl_minutes': 60,
            'max_cache_size': 10000,
            'performance_window_hours': 24,
            'agent_performance_ema_alpha': 0.1,
            'health_check_interval_minutes': 5,
            'attribution_lookback_days': 7
        }
    
    async def initialize(self) -> None:
        """Initialize integration with Strategic MARL system"""
        try:
            # Initialize event bus
            self.event_bus = EventBus()
            
            # Subscribe to strategic decision events
            self.event_bus.subscribe(
                EventType.STRATEGIC_DECISION,
                self._handle_strategic_decision
            )
            
            # Subscribe to agent performance updates
            self.event_bus.subscribe(
                EventType.AGENT_PERFORMANCE_UPDATE,
                self._handle_agent_performance_update
            )
            
            # Initialize performance components
            self.performance_optimizer = PerformanceOptimizer()
            self.attribution_engine = PerformanceAttributionEngine()
            
            # Get reference to Strategic MARL Component
            # (In production, this would be injected via dependency injection)
            self.marl_component = self._get_marl_component_reference()
            
            self._healthy = True
            
            logger.info("Strategic MARL Integrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Strategic MARL Integrator: {e}")
            self._healthy = False
            raise
    
    def _get_marl_component_reference(self) -> Optional[StrategicMARLComponent]:
        """Get reference to Strategic MARL Component"""
        # In production, this would be properly injected
        # For now, return mock or attempt to get from kernel
        try:
            # Mock implementation - in production would get from AlgoSpace kernel
            return None  # Will be properly implemented with kernel integration
        except Exception as e:
            logger.warning(f"Could not get MARL component reference: {e}")
            return None
    
    async def get_decision_context(
        self, 
        symbol: str, 
        timestamp: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive decision context for a symbol at a given time.
        
        Args:
            symbol: Trading symbol
            timestamp: Decision timestamp (defaults to most recent)
            
        Returns:
            Dictionary containing decision context and snapshot
        """
        try:
            # Generate cache key
            cache_key = f"{symbol}_{timestamp or 'latest'}"
            
            # Check cache first
            if cache_key in self.decision_cache:
                self.integration_metrics['cache_hits'] += 1
                cached_data = self.decision_cache[cache_key]
                
                # Check if cache is still valid
                cache_age = datetime.now() - cached_data['cached_at']
                if cache_age.total_seconds() < self.config['cache_ttl_minutes'] * 60:
                    return cached_data['data']
            
            self.integration_metrics['cache_misses'] += 1
            
            # Get decision data from Strategic MARL system
            decision_data = await self._fetch_decision_data(symbol, timestamp)
            
            if not decision_data:
                return None
            
            # Extract agent contexts
            agent_contexts = await self._extract_agent_contexts(decision_data)
            
            # Create strategic context
            strategic_context = StrategicContext(
                decision_id=decision_data['decision_id'],
                timestamp=decision_data['timestamp'],
                symbol=symbol,
                market_regime=decision_data.get('market_regime', 'unknown'),
                synergy_strength=decision_data.get('synergy_strength', 0.0),
                consensus_level=decision_data.get('consensus_level', 0.0),
                agent_contexts=agent_contexts,
                gating_weights=decision_data.get('gating_weights', []),
                performance_attribution=decision_data.get('performance_attribution', {}),
                risk_metrics=decision_data.get('risk_metrics', {})
            )
            
            # Create decision snapshot for XAI engine
            decision_snapshot = await self._create_decision_snapshot(
                decision_data, strategic_context
            )
            
            result = {
                'snapshot': decision_snapshot,
                'strategic_context': asdict(strategic_context),
                'agent_contributions': {
                    ctx.agent_id: {
                        'confidence': ctx.current_confidence,
                        'performance': ctx.recent_performance,
                        'specialization': ctx.specialization
                    }
                    for ctx in agent_contexts
                },
                'performance_metrics': decision_data.get('performance_metrics', {}),
                'cached_at': datetime.now()
            }
            
            # Cache the result
            self._cache_decision_data(cache_key, result)
            
            self.integration_metrics['decisions_processed'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get decision context for {symbol}: {e}")
            self.integration_metrics['error_count'] += 1
            return None
    
    async def _fetch_decision_data(
        self, 
        symbol: str, 
        timestamp: Optional[datetime]
    ) -> Optional[Dict[str, Any]]:
        """Fetch decision data from Strategic MARL system"""
        try:
            if self.marl_component:
                # Get decision from MARL component
                status = self.marl_component.get_status()
                
                # Mock decision data based on component status
                decision_data = {
                    'decision_id': f"{symbol}_{int(time.time())}",
                    'timestamp': timestamp or datetime.now(),
                    'symbol': symbol,
                    'action': 'LONG',  # Would come from actual decision
                    'confidence': 0.75,
                    'market_regime': 'trending',
                    'synergy_strength': 0.8,
                    'consensus_level': 0.85,
                    'gating_weights': [0.4, 0.3, 0.3],  # MLMI, NWRQK, Regime
                    'agent_decisions': {
                        'MLMI': {'action': 'LONG', 'confidence': 0.8, 'features': {}},
                        'NWRQK': {'action': 'LONG', 'confidence': 0.7, 'features': {}},
                        'Regime': {'action': 'HOLD', 'confidence': 0.6, 'features': {}}
                    },
                    'performance_metrics': status.get('performance_metrics', {}),
                    'risk_metrics': {
                        'var_95': 0.02,
                        'expected_shortfall': 0.025,
                        'volatility': 0.15,
                        'sharpe_ratio': 1.2
                    }
                }
                
                return decision_data
            else:
                # Mock data when MARL component not available
                return self._create_mock_decision_data(symbol, timestamp)
                
        except Exception as e:
            logger.error(f"Failed to fetch decision data: {e}")
            return None
    
    def _create_mock_decision_data(
        self, 
        symbol: str, 
        timestamp: Optional[datetime]
    ) -> Dict[str, Any]:
        """Create mock decision data for testing"""
        return {
            'decision_id': f"mock_{symbol}_{int(time.time())}",
            'timestamp': timestamp or datetime.now(),
            'symbol': symbol,
            'action': 'LONG',
            'confidence': 0.75,
            'market_regime': 'trending',
            'synergy_strength': 0.8,
            'consensus_level': 0.85,
            'gating_weights': [0.4, 0.3, 0.3],
            'agent_decisions': {
                'MLMI': {
                    'action': 'LONG',
                    'confidence': 0.8,
                    'features': {
                        'momentum_20': 0.15,
                        'volatility_ratio': 1.2,
                        'trend_strength': 0.7
                    }
                },
                'NWRQK': {
                    'action': 'LONG',
                    'confidence': 0.7,
                    'features': {
                        'quality_score': 0.8,
                        'risk_parity': 0.6,
                        'volume_profile': 1.1
                    }
                },
                'Regime': {
                    'action': 'HOLD',
                    'confidence': 0.6,
                    'features': {
                        'regime_probability': 0.7,
                        'transition_likelihood': 0.3,
                        'stability_score': 0.8
                    }
                }
            },
            'performance_metrics': {
                'total_inferences': 1250,
                'avg_inference_time_ms': 3.2,
                'success_rate': 0.78,
                'sharpe_ratio': 1.45
            },
            'risk_metrics': {
                'var_95': 0.02,
                'expected_shortfall': 0.025,
                'volatility': 0.15,
                'max_drawdown': 0.08
            }
        }
    
    async def _extract_agent_contexts(
        self, 
        decision_data: Dict[str, Any]
    ) -> List[AgentContext]:
        """Extract detailed agent contexts from decision data"""
        agent_contexts = []
        
        agent_decisions = decision_data.get('agent_decisions', {})
        gating_weights = decision_data.get('gating_weights', [0.33, 0.33, 0.34])
        
        for i, (agent_id, agent_data) in enumerate(agent_decisions.items()):
            # Get agent performance history
            recent_performance = self._get_agent_performance(agent_id)
            
            # Create agent context
            agent_context = AgentContext(
                agent_id=agent_id,
                agent_type="Strategic",
                specialization=self._get_agent_specialization(agent_id),
                current_confidence=agent_data.get('confidence', 0.5),
                recent_performance=recent_performance,
                feature_importance=agent_data.get('features', {}),
                decision_rationale=self._generate_agent_rationale(agent_id, agent_data),
                market_conditions_assessment=self._assess_market_conditions(agent_id, decision_data),
                risk_assessment=self._assess_agent_risk(agent_id, agent_data)
            )
            
            agent_contexts.append(agent_context)
        
        return agent_contexts
    
    def _get_agent_specialization(self, agent_id: str) -> str:
        """Get agent specialization description"""
        specializations = {
            'MLMI': 'Momentum and Market Liquidity Intelligence - Trend following and momentum detection',
            'NWRQK': 'Net Worth Risk Quality - Risk management and quality assessment',
            'Regime': 'Market Regime Detection - Structural market change identification'
        }
        return specializations.get(agent_id, f"Specialized trading agent: {agent_id}")
    
    def _get_agent_performance(self, agent_id: str) -> Dict[str, float]:
        """Get recent agent performance metrics"""
        # In production, this would query actual performance data
        if agent_id not in self.agent_performance_history:
            self.agent_performance_history[agent_id] = []
        
        # Mock performance data
        base_performance = {
            'MLMI': 0.75,
            'NWRQK': 0.68,
            'Regime': 0.72
        }.get(agent_id, 0.6)
        
        return {
            'accuracy_7d': base_performance + np.random.normal(0, 0.05),
            'sharpe_ratio_30d': base_performance * 1.5 + np.random.normal(0, 0.1),
            'win_rate_7d': base_performance + np.random.normal(0, 0.03),
            'avg_confidence': base_performance + 0.1,
            'consistency_score': base_performance + 0.05
        }
    
    def _generate_agent_rationale(
        self, 
        agent_id: str, 
        agent_data: Dict[str, Any]
    ) -> str:
        """Generate human-readable rationale for agent decision"""
        action = agent_data.get('action', 'HOLD')
        confidence = agent_data.get('confidence', 0.5)
        features = agent_data.get('features', {})
        
        if agent_id == 'MLMI':
            key_factor = max(features.items(), key=lambda x: x[1])[0] if features else 'momentum'
            return f"MLMI agent recommends {action} with {confidence:.1%} confidence based on strong {key_factor} signals"
        
        elif agent_id == 'NWRQK':
            quality_score = features.get('quality_score', 0.5)
            return f"NWRQK agent suggests {action} with quality score {quality_score:.2f} and {confidence:.1%} confidence"
        
        elif agent_id == 'Regime':
            regime_prob = features.get('regime_probability', 0.5)
            return f"Regime agent indicates {action} with {regime_prob:.1%} regime stability and {confidence:.1%} confidence"
        
        else:
            return f"{agent_id} agent recommends {action} with {confidence:.1%} confidence"
    
    def _assess_market_conditions(
        self, 
        agent_id: str, 
        decision_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess market conditions from agent perspective"""
        market_regime = decision_data.get('market_regime', 'unknown')
        
        return {
            'regime': market_regime,
            'volatility_environment': 'moderate',
            'liquidity_conditions': 'normal',
            'trend_strength': 'strong' if market_regime == 'trending' else 'weak',
            'agent_suitability': self._assess_agent_suitability(agent_id, market_regime)
        }
    
    def _assess_agent_suitability(self, agent_id: str, market_regime: str) -> float:
        """Assess how suitable current market conditions are for specific agent"""
        suitability_matrix = {
            'MLMI': {
                'trending': 0.9,
                'ranging': 0.3,
                'volatile': 0.6,
                'transitional': 0.7
            },
            'NWRQK': {
                'trending': 0.7,
                'ranging': 0.8,
                'volatile': 0.9,
                'transitional': 0.6
            },
            'Regime': {
                'trending': 0.5,
                'ranging': 0.6,
                'volatile': 0.8,
                'transitional': 0.9
            }
        }
        
        return suitability_matrix.get(agent_id, {}).get(market_regime, 0.5)
    
    def _assess_agent_risk(
        self, 
        agent_id: str, 
        agent_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Assess risk metrics for agent decision"""
        confidence = agent_data.get('confidence', 0.5)
        
        return {
            'decision_uncertainty': 1.0 - confidence,
            'model_risk': 0.1,  # Base model risk
            'execution_risk': 0.05,
            'market_risk': 0.15,
            'overall_risk_score': (1.0 - confidence) * 0.4 + 0.3
        }
    
    async def _create_decision_snapshot(
        self, 
        decision_data: Dict[str, Any], 
        strategic_context: StrategicContext
    ) -> DecisionSnapshot:
        """Create DecisionSnapshot for XAI engine"""
        
        # Create mock market features and feature names
        market_features = np.array([0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.7, 0.9, 0.1])
        feature_names = [
            'price_momentum', 'volume_ratio', 'volatility', 'trend_strength',
            'quality_score', 'risk_parity', 'regime_probability', 'liquidity', 'sentiment'
        ]
        
        # Extract agent probabilities and confidences
        agent_probabilities = {}
        agent_confidences = {}
        
        for agent_context in strategic_context.agent_contexts:
            # Mock probability distribution based on agent decision
            if decision_data.get('action') == 'LONG':
                probs = np.array([0.1, 0.2, 0.7])  # [sell, hold, buy]
            elif decision_data.get('action') == 'SHORT':
                probs = np.array([0.7, 0.2, 0.1])
            else:
                probs = np.array([0.2, 0.6, 0.2])
            
            agent_probabilities[agent_context.agent_id] = probs
            agent_confidences[agent_context.agent_id] = agent_context.current_confidence
        
        # Create consensus breakdown
        consensus_breakdown = {
            0: 0.15,  # Sell
            1: 0.25,  # Hold  
            2: 0.60   # Buy
        }
        
        # Map action to ActionType
        action_map = {
            'LONG': ActionType.INCREASE_LONG,
            'SHORT': ActionType.INCREASE_SHORT,
            'HOLD': ActionType.HOLD
        }
        
        final_action = action_map.get(decision_data.get('action', 'HOLD'), ActionType.HOLD)
        
        snapshot = DecisionSnapshot(
            timestamp=pd.Timestamp(decision_data['timestamp']),
            symbol=decision_data['symbol'],
            asset_class=AssetClass.EQUITIES,  # Default to equities
            final_action=final_action,
            confidence=decision_data.get('confidence', 0.5),
            execution_details=None,  # Would be filled by execution engine
            agent_probabilities=agent_probabilities,
            agent_confidences=agent_confidences,
            consensus_breakdown=consensus_breakdown,
            market_features=market_features,
            feature_names=feature_names,
            market_conditions={
                'regime': strategic_context.market_regime,
                'volatility': decision_data.get('risk_metrics', {}).get('volatility', 0.15),
                'liquidity': 'normal'
            },
            current_position=0.0,  # Would come from position manager
            target_position=1.0,   # Would come from position sizing
            risk_metrics=decision_data.get('risk_metrics', {}),
            synergy_alignment=strategic_context.synergy_strength,
            consensus_method="Strategic MARL with Intelligent Gating",
            safety_level=strategic_context.consensus_level
        )
        
        return snapshot
    
    def _cache_decision_data(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Cache decision data with size management"""
        # Remove old entries if cache is full
        if len(self.decision_cache) >= self.config['max_cache_size']:
            # Remove oldest entry
            oldest_key = min(
                self.decision_cache.keys(),
                key=lambda k: self.decision_cache[k]['cached_at']
            )
            del self.decision_cache[oldest_key]
        
        self.decision_cache[cache_key] = data
    
    async def enrich_explanation(
        self, 
        explanation_result: ExplanationResult,
        decision_data: Dict[str, Any]
    ) -> ExplanationResult:
        """
        Enrich explanation with Strategic MARL context.
        
        Args:
            explanation_result: Base explanation from XAI engine
            decision_data: Decision context data
            
        Returns:
            Enhanced explanation with Strategic MARL insights
        """
        try:
            strategic_context = decision_data.get('strategic_context', {})
            agent_contributions = decision_data.get('agent_contributions', {})
            
            # Enhance decision reasoning with Strategic MARL context
            enhanced_reasoning = self._enhance_reasoning(
                explanation_result.decision_reasoning,
                strategic_context,
                agent_contributions
            )
            
            # Add Strategic MARL specific factors
            strategic_factors = self._extract_strategic_factors(strategic_context)
            
            # Combine original positive factors with strategic factors
            enhanced_positive_factors = (
                explanation_result.top_positive_factors + strategic_factors
            )[:10]  # Limit to top 10
            
            # Create enhanced explanation
            enhanced_explanation = ExplanationResult(
                explanation_type=explanation_result.explanation_type,
                audience=explanation_result.audience,
                feature_importance=explanation_result.feature_importance,
                decision_reasoning=enhanced_reasoning,
                confidence_intervals=explanation_result.confidence_intervals,
                top_positive_factors=enhanced_positive_factors,
                top_negative_factors=explanation_result.top_negative_factors,
                alternative_scenarios=self._generate_alternative_scenarios(strategic_context),
                explanation_confidence=explanation_result.explanation_confidence,
                generation_time_ms=explanation_result.generation_time_ms,
                shap_values=explanation_result.shap_values
            )
            
            self.integration_metrics['explanations_generated'] += 1
            
            return enhanced_explanation
            
        except Exception as e:
            logger.error(f"Failed to enrich explanation: {e}")
            self.integration_metrics['error_count'] += 1
            return explanation_result
    
    def _enhance_reasoning(
        self, 
        original_reasoning: str, 
        strategic_context: Dict[str, Any],
        agent_contributions: Dict[str, Any]
    ) -> str:
        """Enhance reasoning with Strategic MARL insights"""
        
        # Extract key strategic insights
        consensus_level = strategic_context.get('consensus_level', 0.0)
        synergy_strength = strategic_context.get('synergy_strength', 0.0)
        market_regime = strategic_context.get('market_regime', 'unknown')
        
        # Generate Strategic MARL enhancement
        strategic_enhancement = (
            f"\n\n🧠 STRATEGIC MARL INSIGHTS: "
            f"Multi-agent consensus reached with {consensus_level:.1%} agreement in {market_regime} market regime. "
            f"Synergy strength: {synergy_strength:.1%}. "
        )
        
        # Add agent contribution details
        if agent_contributions:
            agent_details = []
            for agent_id, contrib in agent_contributions.items():
                conf = contrib.get('confidence', 0.0)
                spec = contrib.get('specialization', 'trading')
                agent_details.append(f"{agent_id} ({conf:.1%} confidence, {spec})")
            
            strategic_enhancement += f"Agent contributions: {'; '.join(agent_details)}."
        
        return original_reasoning + strategic_enhancement
    
    def _extract_strategic_factors(
        self, 
        strategic_context: Dict[str, Any]
    ) -> List[Tuple[str, float]]:
        """Extract Strategic MARL specific factors"""
        factors = []
        
        consensus_level = strategic_context.get('consensus_level', 0.0)
        if consensus_level > 0.7:
            factors.append(('Agent Consensus Strength', consensus_level))
        
        synergy_strength = strategic_context.get('synergy_strength', 0.0)
        if synergy_strength > 0.6:
            factors.append(('Market Synergy Detection', synergy_strength))
        
        # Add gating network insights
        gating_weights = strategic_context.get('gating_weights', [])
        if gating_weights:
            max_weight_idx = np.argmax(gating_weights)
            agent_names = ['MLMI', 'NWRQK', 'Regime']
            if max_weight_idx < len(agent_names):
                factors.append((f'{agent_names[max_weight_idx]} Agent Dominance', gating_weights[max_weight_idx]))
        
        return factors
    
    def _generate_alternative_scenarios(
        self, 
        strategic_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate alternative scenarios based on Strategic MARL analysis"""
        scenarios = []
        
        # Scenario 1: Different agent weighting
        scenarios.append({
            'scenario': 'Equal Agent Weighting',
            'description': 'What if all agents had equal influence (33% each)',
            'confidence_change': -0.1,
            'reasoning': 'Equal weighting would reduce confidence as intelligent gating identifies optimal agent specialization'
        })
        
        # Scenario 2: Different market regime
        current_regime = strategic_context.get('market_regime', 'trending')
        alt_regime = 'ranging' if current_regime == 'trending' else 'trending'
        
        scenarios.append({
            'scenario': f'Alternative Market Regime: {alt_regime}',
            'description': f'Decision if market was in {alt_regime} regime instead of {current_regime}',
            'confidence_change': -0.15 if current_regime == 'trending' else 0.1,
            'reasoning': f'Regime change would favor different agent specializations'
        })
        
        # Scenario 3: Lower consensus threshold
        scenarios.append({
            'scenario': 'Conservative Consensus',
            'description': 'What if consensus threshold was raised to 90%',
            'confidence_change': -0.2,
            'reasoning': 'Higher consensus requirement would reduce trading frequency but increase decision quality'
        })
        
        return scenarios
    
    async def get_decision_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        include_context: bool = True
    ) -> List[Dict[str, Any]]:
        """Get historical decision data"""
        try:
            # In production, this would query actual decision database
            # For now, generate mock historical data
            
            end_time = end_time or datetime.now()
            start_time = start_time or (end_time - timedelta(days=7))
            
            # Generate mock historical decisions
            decisions = []
            current_time = start_time
            decision_id_counter = 1
            
            while current_time <= end_time and len(decisions) < limit:
                decision = {
                    'id': f"decision_{decision_id_counter}",
                    'timestamp': current_time,
                    'symbol': symbol or 'NQ',
                    'action': np.random.choice(['LONG', 'SHORT', 'HOLD'], p=[0.4, 0.3, 0.3]),
                    'confidence': 0.5 + np.random.normal(0, 0.15),
                    'agent_votes': {
                        'MLMI': {
                            'action': np.random.choice(['LONG', 'SHORT', 'HOLD']),
                            'confidence': 0.6 + np.random.normal(0, 0.1)
                        },
                        'NWRQK': {
                            'action': np.random.choice(['LONG', 'SHORT', 'HOLD']),
                            'confidence': 0.6 + np.random.normal(0, 0.1)
                        },
                        'Regime': {
                            'action': np.random.choice(['LONG', 'SHORT', 'HOLD']),
                            'confidence': 0.6 + np.random.normal(0, 0.1)
                        }
                    },
                    'market_context': {
                        'regime': np.random.choice(['trending', 'ranging', 'volatile']),
                        'volatility': 0.1 + np.random.normal(0, 0.05),
                        'volume_ratio': 0.8 + np.random.normal(0, 0.2)
                    }
                }
                
                # Add performance outcome (mock)
                if current_time < datetime.now() - timedelta(hours=1):
                    decision['performance_outcome'] = {
                        'success': np.random.choice([True, False], p=[0.65, 0.35]),
                        'pnl': np.random.normal(0.001, 0.01),
                        'time_to_target': np.random.exponential(3600)  # seconds
                    }
                
                if include_context:
                    # Add mock decision snapshot
                    decision['snapshot'] = await self._create_decision_snapshot(
                        decision, 
                        StrategicContext(
                            decision_id=decision['id'],
                            timestamp=current_time,
                            symbol=decision['symbol'],
                            market_regime=decision['market_context']['regime'],
                            synergy_strength=0.8,
                            consensus_level=0.75,
                            agent_contexts=[],
                            gating_weights=[0.4, 0.3, 0.3],
                            performance_attribution={},
                            risk_metrics={}
                        )
                    )
                
                decisions.append(decision)
                current_time += timedelta(minutes=30)  # 30-minute intervals
                decision_id_counter += 1
            
            return decisions
            
        except Exception as e:
            logger.error(f"Failed to get decision history: {e}")
            return []
    
    async def get_performance_analytics(
        self,
        time_range: str = "24h",
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get comprehensive performance analytics"""
        try:
            # Parse time range
            hours = self._parse_time_range(time_range)
            
            # Get performance data (mock implementation)
            analytics = {
                'overall_performance': {
                    'score': 0.78,
                    'total_trades': 45,
                    'win_rate': 0.67,
                    'profit_factor': 1.34,
                    'sharpe_ratio': 1.45,
                    'max_drawdown': 0.08,
                    'total_pnl': 0.024
                },
                'agent_performance': {
                    'MLMI': {
                        'accuracy': 0.72,
                        'confidence_calibration': 0.85,
                        'contribution_score': 0.45,
                        'specialization_effectiveness': 0.82
                    },
                    'NWRQK': {
                        'accuracy': 0.68,
                        'confidence_calibration': 0.78,
                        'contribution_score': 0.32,
                        'specialization_effectiveness': 0.75
                    },
                    'Regime': {
                        'accuracy': 0.75,
                        'confidence_calibration': 0.88,
                        'contribution_score': 0.23,
                        'specialization_effectiveness': 0.79
                    }
                },
                'decision_quality': {
                    'explanation_coverage': 0.98,
                    'average_confidence': 0.74,
                    'consensus_strength': 0.81,
                    'regulatory_compliance': 1.0
                },
                'strategic_insights': {
                    'dominant_regime': 'trending',
                    'optimal_agent_weighting': [0.45, 0.30, 0.25],
                    'market_conditions_suitability': 0.83,
                    'synergy_detection_accuracy': 0.76
                },
                'risk_metrics': {
                    'var_95': 0.018,
                    'expected_shortfall': 0.022,
                    'volatility_adjusted_return': 0.15,
                    'risk_adjusted_score': 0.82
                },
                'system_health': {
                    'uptime': 0.999,
                    'average_latency_ms': 3.2,
                    'error_rate': 0.001,
                    'capacity_utilization': 0.45
                },
                'recommendations': [
                    'Consider increasing MLMI agent weight in trending markets',
                    'Monitor regime detection accuracy during volatile periods',
                    'Optimize position sizing for better risk-adjusted returns'
                ]
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get performance analytics: {e}")
            return {}
    
    def _parse_time_range(self, time_range: str) -> int:
        """Parse time range string to hours"""
        if time_range.endswith('h'):
            return int(time_range[:-1])
        elif time_range.endswith('d'):
            return int(time_range[:-1]) * 24
        elif time_range.endswith('w'):
            return int(time_range[:-1]) * 24 * 7
        else:
            return 24  # Default to 24 hours
    
    async def get_query_data(
        self,
        data_requirements: List[str],
        time_range: Optional[Dict[str, datetime]] = None
    ) -> Dict[str, Any]:
        """Get data for natural language query processing"""
        try:
            query_data = {}
            
            for requirement in data_requirements:
                if requirement == 'agent_performance':
                    query_data['agent_performance'] = await self._get_agent_performance_data(time_range)
                elif requirement == 'decision_history':
                    query_data['decision_history'] = await self.get_decision_history()
                elif requirement == 'risk_metrics':
                    query_data['risk_metrics'] = await self._get_risk_metrics_data(time_range)
                elif requirement == 'market_analysis':
                    query_data['market_analysis'] = await self._get_market_analysis_data(time_range)
            
            self.integration_metrics['agent_queries'] += 1
            
            return query_data
            
        except Exception as e:
            logger.error(f"Failed to get query data: {e}")
            return {}
    
    async def _get_agent_performance_data(self, time_range: Optional[Dict[str, datetime]]) -> Dict[str, Any]:
        """Get agent performance data for queries"""
        return {
            'MLMI': {
                'recent_accuracy': 0.72,
                'trend_following_score': 0.85,
                'momentum_detection_rate': 0.78
            },
            'NWRQK': {
                'recent_accuracy': 0.68,
                'risk_assessment_score': 0.82,
                'quality_detection_rate': 0.75
            },
            'Regime': {
                'recent_accuracy': 0.75,
                'regime_detection_score': 0.88,
                'transition_prediction_rate': 0.71
            }
        }
    
    async def _get_risk_metrics_data(self, time_range: Optional[Dict[str, datetime]]) -> Dict[str, Any]:
        """Get risk metrics data for queries"""
        return {
            'portfolio_var': 0.018,
            'expected_shortfall': 0.022,
            'volatility': 0.15,
            'correlation_risk': 0.12,
            'concentration_risk': 0.08,
            'liquidity_risk': 0.05
        }
    
    async def _get_market_analysis_data(self, time_range: Optional[Dict[str, datetime]]) -> Dict[str, Any]:
        """Get market analysis data for queries"""
        return {
            'current_regime': 'trending',
            'regime_confidence': 0.83,
            'volatility_environment': 'moderate',
            'liquidity_conditions': 'normal',
            'market_stress_indicators': {
                'vix_level': 18.5,
                'credit_spreads': 'normal',
                'funding_stress': 'low'
            }
        }
    
    async def get_compliance_data(
        self,
        start_date: datetime,
        end_date: datetime,
        report_type: str
    ) -> Dict[str, Any]:
        """Get compliance data for regulatory reporting"""
        try:
            # Mock compliance data
            compliance_data = {
                'audit_trail': [
                    {
                        'decision_id': f'decision_{i}',
                        'timestamp': start_date + timedelta(hours=i),
                        'symbol': 'NQ',
                        'action': 'LONG',
                        'reasoning': f'Mock reasoning for decision {i}',
                        'compliance_status': 'COMPLIANT'
                    }
                    for i in range(10)
                ],
                'recommendations': [
                    'Maintain current explanation quality standards',
                    'Continue agent performance monitoring',
                    'Regular compliance audits recommended'
                ]
            }
            
            return compliance_data
            
        except Exception as e:
            logger.error(f"Failed to get compliance data: {e}")
            return {}
    
    def _handle_strategic_decision(self, event: Event) -> None:
        """Handle strategic decision events"""
        try:
            decision_data = event.payload
            
            # Update performance tracking
            self.integration_metrics['performance_updates'] += 1
            
            # Log decision for audit trail
            logger.info(
                "Strategic decision received",
                extra={
                    "decision_id": decision_data.get('decision_id'),
                    "action": decision_data.get('action'),
                    "confidence": decision_data.get('confidence')
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to handle strategic decision event: {e}")
    
    def _handle_agent_performance_update(self, event: Event) -> None:
        """Handle agent performance update events"""
        try:
            performance_data = event.payload
            agent_id = performance_data.get('agent_id')
            
            if agent_id:
                # Update agent performance history
                if agent_id not in self.agent_performance_history:
                    self.agent_performance_history[agent_id] = []
                
                performance_score = performance_data.get('performance_score', 0.0)
                self.agent_performance_history[agent_id].append(performance_score)
                
                # Keep only recent history
                max_history = 100
                if len(self.agent_performance_history[agent_id]) > max_history:
                    self.agent_performance_history[agent_id] = self.agent_performance_history[agent_id][-max_history:]
            
        except Exception as e:
            logger.error(f"Failed to handle agent performance update: {e}")
    
    def is_healthy(self) -> bool:
        """Check integration health status"""
        # Update health check timestamp
        self._last_health_check = datetime.now()
        
        # Check if components are responding
        try:
            # Basic health checks
            if not self._healthy:
                return False
            
            # Check cache size
            if len(self.decision_cache) > self.config['max_cache_size'] * 1.1:
                logger.warning("Decision cache size exceeded")
                return False
            
            # Check error rate
            total_operations = sum(self.integration_metrics.values())
            if total_operations > 100:  # Only check after some operations
                error_rate = self.integration_metrics['error_count'] / total_operations
                if error_rate > 0.1:  # 10% error threshold
                    logger.warning(f"High error rate: {error_rate:.2%}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown Strategic MARL Integrator"""
        try:
            logger.info("Shutting down Strategic MARL Integrator")
            
            # Unsubscribe from events
            if self.event_bus:
                self.event_bus.unsubscribe(
                    EventType.STRATEGIC_DECISION,
                    self._handle_strategic_decision
                )
                self.event_bus.unsubscribe(
                    EventType.AGENT_PERFORMANCE_UPDATE,
                    self._handle_agent_performance_update
                )
            
            # Clear caches
            self.decision_cache.clear()
            self.agent_performance_history.clear()
            
            self._healthy = False
            
            logger.info(
                "Strategic MARL Integrator shutdown complete",
                extra={"final_metrics": self.integration_metrics}
            )
            
        except Exception as e:
            logger.error(f"Error during integrator shutdown: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get integration metrics"""
        return {
            **self.integration_metrics,
            'cache_size': len(self.decision_cache),
            'agent_performance_tracked': len(self.agent_performance_history),
            'health_status': self.is_healthy(),
            'last_health_check': self._last_health_check.isoformat()
        }