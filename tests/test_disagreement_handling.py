"""
Test Suite for Disagreement Handling in Tactical Decision Aggregator

Comprehensive tests for production hardening enhancements:
- Disagreement penalty calculation
- Consensus filter functionality
- Jensen-Shannon divergence implementation
- Edge case handling for extreme disagreement scenarios
- Performance impact measurement

Author: Quantitative Engineer
Version: 1.0 (Production Hardening)
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List
import copy
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from components.tactical_decision_aggregator import TacticalDecisionAggregator, AgentDecision, AggregatedDecision


class TestDisagreementCalculation:
    """Test suite for disagreement calculation methods"""
    
    @pytest.fixture
    def aggregator(self):
        """Create aggregator with disagreement handling enabled"""
        config = {
            'execution_threshold': 0.65,
            'disagreement_threshold': 0.4,
            'disagreement_penalty': 0.5,
            'consensus_filter_enabled': True,
            'min_consensus_strength': 0.6,
            'max_disagreement_score': 0.8
        }
        return TacticalDecisionAggregator(config)
    
    def test_perfect_agreement_scenario(self, aggregator):
        """Test scenario with perfect agent agreement"""
        # All agents agree: 90% bullish
        agent_decisions = {
            'fvg_agent': AgentDecision(
                agent_id='fvg_agent',
                action=2,
                probabilities=np.array([0.05, 0.05, 0.90]),
                confidence=0.9,
                timestamp=1.0
            ),
            'momentum_agent': AgentDecision(
                agent_id='momentum_agent',
                action=2,
                probabilities=np.array([0.05, 0.05, 0.90]),
                confidence=0.9,
                timestamp=1.0
            ),
            'entry_opt_agent': AgentDecision(
                agent_id='entry_opt_agent',
                action=2,
                probabilities=np.array([0.05, 0.05, 0.90]),
                confidence=0.9,
                timestamp=1.0
            )
        }
        
        disagreement_score = aggregator._calculate_disagreement_score(agent_decisions)
        
        # Should be very low disagreement (near 0)
        assert disagreement_score < 0.1
        assert disagreement_score >= 0.0
    
    def test_extreme_disagreement_scenario(self, aggregator):
        """Test scenario with extreme disagreement (FVG=90% bullish, Momentum=90% bearish, Entry=neutral)"""
        agent_decisions = {
            'fvg_agent': AgentDecision(
                agent_id='fvg_agent',
                action=2,
                probabilities=np.array([0.05, 0.05, 0.90]),
                confidence=0.9,
                timestamp=1.0
            ),
            'momentum_agent': AgentDecision(
                agent_id='momentum_agent',
                action=0,
                probabilities=np.array([0.90, 0.05, 0.05]),
                confidence=0.9,
                timestamp=1.0
            ),
            'entry_opt_agent': AgentDecision(
                agent_id='entry_opt_agent',
                action=1,
                probabilities=np.array([0.33, 0.34, 0.33]),
                confidence=0.5,
                timestamp=1.0
            )
        }
        
        disagreement_score = aggregator._calculate_disagreement_score(agent_decisions)
        
        # Should be high disagreement (> 0.6)
        assert disagreement_score > 0.6
        assert disagreement_score <= 1.0
    
    def test_jensen_shannon_divergence_implementation(self, aggregator):
        """Test Jensen-Shannon divergence calculation"""
        # Test with known distributions
        dist1 = np.array([1.0, 0.0, 0.0])  # Certain bearish
        dist2 = np.array([0.0, 0.0, 1.0])  # Certain bullish
        dist3 = np.array([0.33, 0.34, 0.33])  # Neutral
        
        # Maximum divergence case
        js_div_max = aggregator._jensen_shannon_divergence([dist1, dist2])
        assert js_div_max > 0.5
        
        # Minimum divergence case (identical distributions)
        js_div_min = aggregator._jensen_shannon_divergence([dist1, dist1])
        assert js_div_min < 0.01
        
        # Mixed case
        js_div_mixed = aggregator._jensen_shannon_divergence([dist1, dist2, dist3])
        assert 0.1 < js_div_mixed < 0.9
    
    def test_consensus_strength_calculation(self, aggregator):
        """Test consensus strength calculation"""
        # Strong consensus (90% for one action)
        strong_consensus = {0: 0.05, 1: 0.05, 2: 0.90}
        consensus_strong = aggregator._calculate_consensus_strength(strong_consensus)
        assert consensus_strong > 0.8
        
        # Weak consensus (uniform distribution)
        weak_consensus = {0: 0.33, 1: 0.33, 2: 0.34}
        consensus_weak = aggregator._calculate_consensus_strength(weak_consensus)
        assert consensus_weak < 0.3
        
        # Moderate consensus (60% for one action)
        moderate_consensus = {0: 0.20, 1: 0.20, 2: 0.60}
        consensus_moderate = aggregator._calculate_consensus_strength(moderate_consensus)
        assert 0.3 < consensus_moderate < 0.8


class TestDisagreementPenalty:
    """Test suite for disagreement penalty application"""
    
    @pytest.fixture
    def aggregator(self):
        """Create aggregator with disagreement penalty enabled"""
        config = {
            'execution_threshold': 0.65,
            'disagreement_threshold': 0.4,
            'disagreement_penalty': 0.5,
            'consensus_filter_enabled': True,
            'min_consensus_strength': 0.6
        }
        return TacticalDecisionAggregator(config)
    
    def test_disagreement_penalty_application(self, aggregator):
        """Test that disagreement penalty is applied correctly"""
        # Create high disagreement scenario
        agent_outputs = {
            'fvg_agent': Mock(
                probabilities=np.array([0.05, 0.05, 0.90]),
                action=2,
                confidence=0.9,
                timestamp=1.0
            ),
            'momentum_agent': Mock(
                probabilities=np.array([0.90, 0.05, 0.05]),
                action=0,
                confidence=0.9,
                timestamp=1.0
            ),
            'entry_opt_agent': Mock(
                probabilities=np.array([0.33, 0.34, 0.33]),
                action=1,
                confidence=0.5,
                timestamp=1.0
            )
        }
        
        market_state = Mock()
        market_state.features = {'current_price': 100.0}
        
        synergy_context = {
            'type': 'TYPE_1',
            'direction': 1,
            'confidence': 0.7
        }
        
        # Execute aggregation
        decision = aggregator.aggregate_decisions(agent_outputs, market_state, synergy_context)
        
        # Should have low confidence due to penalty
        assert decision.confidence < 0.6
        
        # Should not execute due to penalty
        assert decision.execute == False
        
        # Check that penalty was recorded
        assert aggregator.performance_metrics['disagreement_penalties'] > 0
    
    def test_no_penalty_for_low_disagreement(self, aggregator):
        """Test that no penalty is applied for low disagreement"""
        # Create low disagreement scenario
        agent_outputs = {
            'fvg_agent': Mock(
                probabilities=np.array([0.10, 0.10, 0.80]),
                action=2,
                confidence=0.8,
                timestamp=1.0
            ),
            'momentum_agent': Mock(
                probabilities=np.array([0.15, 0.15, 0.70]),
                action=2,
                confidence=0.7,
                timestamp=1.0
            ),
            'entry_opt_agent': Mock(
                probabilities=np.array([0.20, 0.20, 0.60]),
                action=2,
                confidence=0.6,
                timestamp=1.0
            )
        }
        
        market_state = Mock()
        market_state.features = {'current_price': 100.0}
        
        synergy_context = {
            'type': 'TYPE_1',
            'direction': 1,
            'confidence': 0.7
        }
        
        # Execute aggregation
        decision = aggregator.aggregate_decisions(agent_outputs, market_state, synergy_context)
        
        # Should have reasonable confidence
        assert decision.confidence >= 0.6
        
        # Should execute
        assert decision.execute == True
        
        # Check that no penalty was recorded
        assert aggregator.performance_metrics['disagreement_penalties'] == 0


class TestConsensusFilter:
    """Test suite for consensus filter functionality"""
    
    @pytest.fixture
    def aggregator(self):
        """Create aggregator with consensus filter enabled"""
        config = {
            'execution_threshold': 0.65,
            'consensus_filter_enabled': True,
            'min_consensus_strength': 0.6,
            'disagreement_threshold': 0.4,
            'disagreement_penalty': 0.3
        }
        return TacticalDecisionAggregator(config)
    
    def test_consensus_filter_blocks_weak_consensus(self, aggregator):
        """Test that consensus filter blocks execution for weak consensus"""
        # Create weak consensus scenario (uniform distribution)
        agent_outputs = {
            'fvg_agent': Mock(
                probabilities=np.array([0.33, 0.34, 0.33]),
                action=1,
                confidence=0.8,
                timestamp=1.0
            ),
            'momentum_agent': Mock(
                probabilities=np.array([0.35, 0.30, 0.35]),
                action=0,
                confidence=0.8,
                timestamp=1.0
            ),
            'entry_opt_agent': Mock(
                probabilities=np.array([0.32, 0.36, 0.32]),
                action=1,
                confidence=0.8,
                timestamp=1.0
            )
        }
        
        market_state = Mock()
        market_state.features = {'current_price': 100.0}
        
        synergy_context = {
            'type': 'TYPE_1',
            'direction': 1,
            'confidence': 0.7
        }
        
        # Execute aggregation
        decision = aggregator.aggregate_decisions(agent_outputs, market_state, synergy_context)
        
        # Should not execute due to weak consensus
        assert decision.execute == False
        assert decision.confidence == 0.0
        
        # Check that consensus failure was recorded
        assert aggregator.performance_metrics['consensus_failures'] > 0
    
    def test_consensus_filter_allows_strong_consensus(self, aggregator):
        """Test that consensus filter allows execution for strong consensus"""
        # Create strong consensus scenario
        agent_outputs = {
            'fvg_agent': Mock(
                probabilities=np.array([0.05, 0.05, 0.90]),
                action=2,
                confidence=0.9,
                timestamp=1.0
            ),
            'momentum_agent': Mock(
                probabilities=np.array([0.10, 0.10, 0.80]),
                action=2,
                confidence=0.8,
                timestamp=1.0
            ),
            'entry_opt_agent': Mock(
                probabilities=np.array([0.15, 0.15, 0.70]),
                action=2,
                confidence=0.7,
                timestamp=1.0
            )
        }
        
        market_state = Mock()
        market_state.features = {'current_price': 100.0}
        
        synergy_context = {
            'type': 'TYPE_1',
            'direction': 1,
            'confidence': 0.7
        }
        
        # Execute aggregation
        decision = aggregator.aggregate_decisions(agent_outputs, market_state, synergy_context)
        
        # Should execute due to strong consensus
        assert decision.execute == True
        assert decision.confidence > 0.6
        
        # Check that no consensus failure was recorded
        assert aggregator.performance_metrics['consensus_failures'] == 0
    
    def test_consensus_filter_disabled(self, aggregator):
        """Test behavior when consensus filter is disabled"""
        # Disable consensus filter
        aggregator.consensus_filter_enabled = False
        
        # Create weak consensus scenario
        agent_outputs = {
            'fvg_agent': Mock(
                probabilities=np.array([0.33, 0.34, 0.33]),
                action=1,
                confidence=0.8,
                timestamp=1.0
            ),
            'momentum_agent': Mock(
                probabilities=np.array([0.35, 0.30, 0.35]),
                action=0,
                confidence=0.8,
                timestamp=1.0
            ),
            'entry_opt_agent': Mock(
                probabilities=np.array([0.32, 0.36, 0.32]),
                action=1,
                confidence=0.8,
                timestamp=1.0
            )
        }
        
        market_state = Mock()
        market_state.features = {'current_price': 100.0}
        
        synergy_context = {
            'type': 'TYPE_1',
            'direction': 1,
            'confidence': 0.7
        }
        
        # Execute aggregation
        decision = aggregator.aggregate_decisions(agent_outputs, market_state, synergy_context)
        
        # Should not be blocked by consensus filter
        assert decision.confidence > 0.0
        
        # Check that no consensus failure was recorded
        assert aggregator.performance_metrics['consensus_failures'] == 0


class TestDisagreementAnalysis:
    """Test suite for disagreement pattern analysis"""
    
    @pytest.fixture
    def aggregator_with_history(self):
        """Create aggregator with decision history"""
        config = {
            'execution_threshold': 0.65,
            'disagreement_threshold': 0.4,
            'disagreement_penalty': 0.5,
            'consensus_filter_enabled': True,
            'min_consensus_strength': 0.6
        }
        aggregator = TacticalDecisionAggregator(config)
        
        # Add mock decision history
        for i in range(20):
            decision = AggregatedDecision(
                execute=i % 2 == 0,
                action=i % 3,
                confidence=0.7,
                agent_votes={},
                consensus_breakdown={},
                synergy_alignment=0.5,
                execution_command=None
            )
            decision.disagreement_score = 0.3 + (i * 0.02)  # Increasing disagreement
            decision.consensus_strength = 0.8 - (i * 0.01)  # Decreasing consensus
            
            aggregator.decision_history.append({
                'decision': decision,
                'synergy_context': {},
                'timestamp': i
            })
        
        return aggregator
    
    def test_disagreement_pattern_analysis(self, aggregator_with_history):
        """Test disagreement pattern analysis"""
        analysis = aggregator_with_history.analyze_disagreement_patterns()
        
        # Check structure
        assert 'decisions_analyzed' in analysis
        assert 'disagreement_statistics' in analysis
        assert 'consensus_statistics' in analysis
        assert 'execution_statistics' in analysis
        assert 'recommendations' in analysis
        
        # Check values
        assert analysis['decisions_analyzed'] == 20
        assert analysis['disagreement_statistics']['disagreement_trend'] > 0  # Increasing
        assert analysis['consensus_statistics']['consensus_trend'] < 0  # Decreasing
        assert len(analysis['recommendations']) > 0
    
    def test_trend_calculation(self, aggregator_with_history):
        """Test trend calculation for time series"""
        # Increasing trend
        increasing_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        trend = aggregator_with_history._calculate_trend(increasing_values)
        assert trend > 0.5
        
        # Decreasing trend
        decreasing_values = [0.5, 0.4, 0.3, 0.2, 0.1]
        trend = aggregator_with_history._calculate_trend(decreasing_values)
        assert trend < -0.5
        
        # Stable trend
        stable_values = [0.3, 0.3, 0.3, 0.3, 0.3]
        trend = aggregator_with_history._calculate_trend(stable_values)
        assert abs(trend) < 0.2
    
    def test_recommendations_generation(self, aggregator_with_history):
        """Test recommendation generation"""
        # Create scenario with high disagreement
        analysis = {
            'disagreement_statistics': {'mean_disagreement': 0.8, 'disagreement_trend': 0.7},
            'consensus_statistics': {'mean_consensus_strength': 0.3, 'consensus_trend': -0.6},
            'execution_statistics': {'execution_rate': 0.2}
        }
        
        recommendations = aggregator_with_history._generate_disagreement_recommendations(analysis)
        
        # Should have multiple recommendations
        assert len(recommendations) >= 3
        
        # Check for specific recommendation types
        high_disagreement_rec = any('High average disagreement' in rec for rec in recommendations)
        low_consensus_rec = any('Low average consensus strength' in rec for rec in recommendations)
        low_execution_rec = any('Low execution rate' in rec for rec in recommendations)
        
        assert high_disagreement_rec
        assert low_consensus_rec
        assert low_execution_rec


class TestPerformanceImpact:
    """Test suite for performance impact of disagreement handling"""
    
    @pytest.fixture
    def aggregator(self):
        """Create aggregator for performance testing"""
        config = {
            'execution_threshold': 0.65,
            'disagreement_threshold': 0.4,
            'disagreement_penalty': 0.5,
            'consensus_filter_enabled': True,
            'min_consensus_strength': 0.6
        }
        return TacticalDecisionAggregator(config)
    
    def test_performance_metrics_tracking(self, aggregator):
        """Test that performance metrics are tracked correctly"""
        # Execute multiple decisions with different scenarios
        scenarios = [
            # High disagreement scenario
            {
                'fvg_agent': Mock(probabilities=np.array([0.05, 0.05, 0.90]), action=2, confidence=0.9, timestamp=1.0),
                'momentum_agent': Mock(probabilities=np.array([0.90, 0.05, 0.05]), action=0, confidence=0.9, timestamp=1.0),
                'entry_opt_agent': Mock(probabilities=np.array([0.33, 0.34, 0.33]), action=1, confidence=0.5, timestamp=1.0)
            },
            # Low disagreement scenario
            {
                'fvg_agent': Mock(probabilities=np.array([0.10, 0.10, 0.80]), action=2, confidence=0.8, timestamp=1.0),
                'momentum_agent': Mock(probabilities=np.array([0.15, 0.15, 0.70]), action=2, confidence=0.7, timestamp=1.0),
                'entry_opt_agent': Mock(probabilities=np.array([0.20, 0.20, 0.60]), action=2, confidence=0.6, timestamp=1.0)
            },
            # Weak consensus scenario
            {
                'fvg_agent': Mock(probabilities=np.array([0.33, 0.34, 0.33]), action=1, confidence=0.8, timestamp=1.0),
                'momentum_agent': Mock(probabilities=np.array([0.35, 0.30, 0.35]), action=0, confidence=0.8, timestamp=1.0),
                'entry_opt_agent': Mock(probabilities=np.array([0.32, 0.36, 0.32]), action=1, confidence=0.8, timestamp=1.0)
            }
        ]
        
        market_state = Mock()
        market_state.features = {'current_price': 100.0}
        
        synergy_context = {
            'type': 'TYPE_1',
            'direction': 1,
            'confidence': 0.7
        }
        
        for scenario in scenarios:
            aggregator.aggregate_decisions(scenario, market_state, synergy_context)
        
        # Check metrics
        metrics = aggregator.get_performance_metrics()
        
        assert metrics['total_decisions'] == 3
        assert metrics['disagreement_penalty_rate'] > 0
        assert metrics['consensus_failure_rate'] > 0
        assert 'execution_rate' in metrics
        assert 'synergy_alignment_rate' in metrics
    
    def test_latency_impact(self, aggregator):
        """Test latency impact of disagreement calculations"""
        import time
        
        # Create test scenario
        agent_outputs = {
            'fvg_agent': Mock(probabilities=np.array([0.05, 0.05, 0.90]), action=2, confidence=0.9, timestamp=1.0),
            'momentum_agent': Mock(probabilities=np.array([0.90, 0.05, 0.05]), action=0, confidence=0.9, timestamp=1.0),
            'entry_opt_agent': Mock(probabilities=np.array([0.33, 0.34, 0.33]), action=1, confidence=0.5, timestamp=1.0)
        }
        
        market_state = Mock()
        market_state.features = {'current_price': 100.0}
        
        synergy_context = {
            'type': 'TYPE_1',
            'direction': 1,
            'confidence': 0.7
        }
        
        # Measure latency
        start_time = time.time()
        for _ in range(100):
            aggregator.aggregate_decisions(agent_outputs, market_state, synergy_context)
        end_time = time.time()
        
        avg_latency = (end_time - start_time) / 100
        
        # Should be under 10ms per decision
        assert avg_latency < 0.010


if __name__ == '__main__':
    pytest.main([__file__, '-v'])