"""
Routing Optimizer

Advanced optimization engine for intelligent order routing decisions.
Uses machine learning and historical performance data to optimize routing.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import structlog

logger = structlog.get_logger()


@dataclass
class RoutingFeatures:
    """Features used for routing optimization"""
    
    # Order characteristics
    quantity: int
    notional_value: float
    order_type: str
    side: str
    urgency: float
    
    # Market conditions
    volatility: float
    spread: float
    volume_rate: float
    time_of_day: float  # 0-1 normalized
    
    # Venue characteristics
    venue_latency: float
    venue_cost: float
    venue_fill_rate: float
    venue_market_impact: float
    
    # Historical performance
    recent_performance_score: float
    symbol_venue_affinity: float


@dataclass
class OptimizationResult:
    """Result of routing optimization"""
    
    recommended_venue: str
    confidence_score: float
    expected_performance: Dict[str, float]
    feature_importance: Dict[str, float]
    alternative_venues: List[Tuple[str, float]]  # (venue_id, score)


class RoutingOptimizer:
    """
    Advanced routing optimization using machine learning and historical data.
    
    Optimizes routing decisions based on:
    - Historical execution performance
    - Real-time market conditions
    - Order characteristics
    - Venue performance patterns
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Performance history storage
        self.execution_history: List[Dict[str, Any]] = []
        self.venue_performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Feature weights (learned over time)
        self.feature_weights = {
            'cost': 0.25,
            'latency': 0.25,
            'fill_rate': 0.25,
            'market_impact': 0.25
        }
        
        # Model parameters
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.history_window_days = self.config.get('history_window_days', 30)
        self.min_samples_for_learning = self.config.get('min_samples_for_learning', 100)
        
        # Optimization state
        self.model_trained = False
        self.last_training_time = None
        
        logger.info("RoutingOptimizer initialized")
    
    def optimize_routing(
        self,
        order_features: RoutingFeatures,
        available_venues: List[str],
        venue_data: Dict[str, Dict[str, Any]]
    ) -> OptimizationResult:
        """
        Optimize routing decision based on features and historical performance.
        """
        
        # Score each available venue
        venue_scores = {}
        feature_contributions = {}
        
        for venue_id in available_venues:
            venue_info = venue_data.get(venue_id, {})
            score, contributions = self._score_venue(order_features, venue_id, venue_info)
            venue_scores[venue_id] = score
            feature_contributions[venue_id] = contributions
        
        # Select best venue
        best_venue = max(venue_scores, key=venue_scores.get)
        confidence = self._calculate_confidence(venue_scores, best_venue)
        
        # Get alternative venues
        sorted_venues = sorted(venue_scores.items(), key=lambda x: x[1], reverse=True)
        alternatives = sorted_venues[1:4]  # Top 3 alternatives
        
        # Calculate expected performance
        expected_performance = self._calculate_expected_performance(
            order_features, best_venue, venue_data.get(best_venue, {})
        )
        
        return OptimizationResult(
            recommended_venue=best_venue,
            confidence_score=confidence,
            expected_performance=expected_performance,
            feature_importance=feature_contributions.get(best_venue, {}),
            alternative_venues=alternatives
        )
    
    def _score_venue(
        self,
        features: RoutingFeatures,
        venue_id: str,
        venue_info: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        """Score a venue based on features and historical performance"""
        
        scores = {}
        
        # Cost score (lower cost = higher score)
        cost = venue_info.get('cost_per_share', features.venue_cost)
        scores['cost'] = self._normalize_score(1.0 / (1.0 + cost * 1000), 0, 1)
        
        # Latency score (lower latency = higher score)
        latency = venue_info.get('avg_latency', features.venue_latency)
        scores['latency'] = self._normalize_score(1.0 / (1.0 + latency), 0, 1)
        
        # Fill rate score
        fill_rate = venue_info.get('fill_rate', features.venue_fill_rate)
        scores['fill_rate'] = self._normalize_score(fill_rate, 0, 1)
        
        # Market impact score (lower impact = higher score)
        impact = venue_info.get('market_impact_bps', features.venue_market_impact)
        scores['market_impact'] = self._normalize_score(1.0 / (1.0 + impact), 0, 1)
        
        # Historical performance boost
        historical_score = self._get_historical_performance_score(venue_id, features)
        scores['historical'] = historical_score
        
        # Order-specific adjustments
        order_adjustment = self._calculate_order_specific_adjustment(features, venue_info)
        scores['order_specific'] = order_adjustment
        
        # Market condition adjustments
        market_adjustment = self._calculate_market_condition_adjustment(features, venue_info)
        scores['market_conditions'] = market_adjustment
        
        # Calculate weighted composite score
        composite_score = (
            self.feature_weights['cost'] * scores['cost'] +
            self.feature_weights['latency'] * scores['latency'] +
            self.feature_weights['fill_rate'] * scores['fill_rate'] +
            self.feature_weights['market_impact'] * scores['market_impact'] +
            0.1 * scores['historical'] +
            0.1 * scores['order_specific'] +
            0.1 * scores['market_conditions']
        )
        
        return composite_score, scores
    
    def _normalize_score(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize score to 0-1 range"""
        if max_val == min_val:
            return 0.5
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))
    
    def _get_historical_performance_score(
        self,
        venue_id: str,
        features: RoutingFeatures
    ) -> float:
        """Get historical performance score for venue"""
        
        venue_history = self.venue_performance_history.get(venue_id, [])
        
        if not venue_history:
            return 0.5  # Neutral score for new venues
        
        # Filter recent history
        cutoff_time = datetime.now() - timedelta(days=self.history_window_days)
        recent_history = [
            h for h in venue_history
            if datetime.fromisoformat(h.get('timestamp', '2000-01-01')) >= cutoff_time
        ]
        
        if not recent_history:
            return 0.5
        
        # Calculate average performance metrics
        avg_implementation_shortfall = np.mean([
            h.get('implementation_shortfall', 0.0) for h in recent_history
        ])
        avg_fill_rate = np.mean([
            h.get('fill_rate', 0.95) for h in recent_history
        ])
        avg_latency_performance = np.mean([
            h.get('latency_performance', 0.5) for h in recent_history
        ])
        
        # Convert to performance score (lower IS = better, higher fill rate = better)
        is_score = 1.0 / (1.0 + abs(avg_implementation_shortfall) * 1000)
        fill_score = avg_fill_rate
        latency_score = avg_latency_performance
        
        return (is_score + fill_score + latency_score) / 3.0
    
    def _calculate_order_specific_adjustment(
        self,
        features: RoutingFeatures,
        venue_info: Dict[str, Any]
    ) -> float:
        """Calculate order-specific venue suitability adjustment"""
        
        adjustment = 0.5  # Base neutral adjustment
        
        # Large order adjustments
        if features.quantity > 50000:
            # Prefer venues with good large order handling
            if venue_info.get('supports_algo_orders', False):
                adjustment += 0.2
            if venue_info.get('venue_type') == 'DARK_POOL':
                adjustment += 0.15
        
        # Urgent order adjustments
        if features.urgency > 0.8:
            # Prefer low-latency venues for urgent orders
            latency = venue_info.get('avg_latency', 10.0)
            if latency < 5.0:
                adjustment += 0.2
            elif latency > 15.0:
                adjustment -= 0.1
        
        # Market order adjustments
        if features.order_type == 'MARKET':
            # Prefer venues with good fill rates for market orders
            fill_rate = venue_info.get('fill_rate', 0.95)
            if fill_rate > 0.98:
                adjustment += 0.1
        
        # Time of day adjustments
        if features.time_of_day < 0.1 or features.time_of_day > 0.9:  # Market open/close
            # Prefer stable venues during volatile periods
            if venue_info.get('venue_type') == 'EXCHANGE':
                adjustment += 0.1
        
        return max(0.0, min(1.0, adjustment))
    
    def _calculate_market_condition_adjustment(
        self,
        features: RoutingFeatures,
        venue_info: Dict[str, Any]
    ) -> float:
        """Calculate market condition based adjustment"""
        
        adjustment = 0.5  # Base neutral adjustment
        
        # High volatility adjustments
        if features.volatility > 0.03:  # High volatility
            # Prefer venues with better market impact control
            if venue_info.get('venue_type') == 'DARK_POOL':
                adjustment += 0.15
            # Avoid venues with poor performance in volatile conditions
            if venue_info.get('volatility_performance', 0.5) < 0.3:
                adjustment -= 0.2
        
        # Wide spread adjustments
        if features.spread > 0.01:  # Wide spread (>1%)
            # Prefer venues with better spread crossing
            if venue_info.get('spread_performance', 0.5) > 0.7:
                adjustment += 0.1
        
        # High volume rate adjustments
        if features.volume_rate > 10000:  # High volume
            # Can be more aggressive with high volume
            if venue_info.get('high_volume_performance', 0.5) > 0.6:
                adjustment += 0.1
        
        return max(0.0, min(1.0, adjustment))
    
    def _calculate_confidence(
        self,
        venue_scores: Dict[str, float],
        best_venue: str
    ) -> float:
        """Calculate confidence in routing decision"""
        
        if len(venue_scores) < 2:
            return 0.8  # Reasonable confidence for single venue
        
        scores = list(venue_scores.values())
        best_score = venue_scores[best_venue]
        
        # Calculate score separation
        scores.sort(reverse=True)
        second_best = scores[1] if len(scores) > 1 else scores[0]
        
        # Higher separation = higher confidence
        separation = best_score - second_best
        
        # Scale to 0.5-1.0 range
        confidence = 0.5 + (separation * 0.5)
        
        # Boost confidence if we have historical data
        if best_venue in self.venue_performance_history:
            history_boost = min(0.1, len(self.venue_performance_history[best_venue]) / 100)
            confidence += history_boost
        
        return max(0.5, min(1.0, confidence))
    
    def _calculate_expected_performance(
        self,
        features: RoutingFeatures,
        venue_id: str,
        venue_info: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate expected performance metrics"""
        
        # Base expectations from venue configuration
        expected_latency = venue_info.get('avg_latency', 10.0)
        expected_fill_rate = venue_info.get('fill_rate', 0.95)
        expected_cost = venue_info.get('cost_per_share', 0.002)
        expected_impact = venue_info.get('market_impact_bps', 2.0)
        
        # Adjust based on order characteristics
        if features.quantity > 10000:  # Large order
            expected_latency *= 1.2  # May take longer
            expected_impact *= 1.5   # Higher impact
        
        if features.urgency > 0.8:  # Urgent order
            expected_fill_rate *= 0.98  # Slight reduction for speed
        
        # Adjust based on market conditions
        if features.volatility > 0.03:  # High volatility
            expected_impact *= 1.3
            expected_fill_rate *= 0.95
        
        return {
            'expected_latency_ms': expected_latency,
            'expected_fill_rate': expected_fill_rate,
            'expected_cost_bps': expected_cost * 100,
            'expected_market_impact_bps': expected_impact
        }
    
    def record_execution_outcome(
        self,
        venue_id: str,
        order_features: RoutingFeatures,
        actual_performance: Dict[str, float]
    ) -> None:
        """Record execution outcome for learning"""
        
        outcome = {
            'timestamp': datetime.now().isoformat(),
            'venue_id': venue_id,
            'order_features': {
                'quantity': order_features.quantity,
                'notional_value': order_features.notional_value,
                'order_type': order_features.order_type,
                'urgency': order_features.urgency
            },
            'actual_performance': actual_performance,
            'implementation_shortfall': actual_performance.get('implementation_shortfall', 0.0),
            'fill_rate': actual_performance.get('fill_rate', 0.95),
            'latency_performance': 1.0 / (1.0 + actual_performance.get('latency_ms', 10.0))
        }
        
        # Add to execution history
        self.execution_history.append(outcome)
        
        # Add to venue-specific history
        if venue_id not in self.venue_performance_history:
            self.venue_performance_history[venue_id] = []
        
        self.venue_performance_history[venue_id].append(outcome)
        
        # Limit history size
        max_history = 1000
        if len(self.execution_history) > max_history:
            self.execution_history = self.execution_history[-max_history:]
        
        for venue_history in self.venue_performance_history.values():
            if len(venue_history) > max_history:
                venue_history = venue_history[-max_history:]
        
        # Trigger learning if enough samples
        if len(self.execution_history) >= self.min_samples_for_learning:
            self._update_feature_weights()
    
    def _update_feature_weights(self) -> None:
        """Update feature weights based on historical performance"""
        
        # Simple learning approach - in production would use more sophisticated ML
        if len(self.execution_history) < self.min_samples_for_learning:
            return
        
        # Analyze correlation between features and performance
        recent_history = self.execution_history[-self.min_samples_for_learning:]
        
        # Calculate feature importance based on performance outcomes
        cost_importance = self._calculate_feature_importance(recent_history, 'cost')
        latency_importance = self._calculate_feature_importance(recent_history, 'latency')
        fill_importance = self._calculate_feature_importance(recent_history, 'fill_rate')
        impact_importance = self._calculate_feature_importance(recent_history, 'market_impact')
        
        # Update weights with learning rate
        total_importance = cost_importance + latency_importance + fill_importance + impact_importance
        
        if total_importance > 0:
            target_weights = {
                'cost': cost_importance / total_importance,
                'latency': latency_importance / total_importance,
                'fill_rate': fill_importance / total_importance,
                'market_impact': impact_importance / total_importance
            }
            
            # Blend with current weights using learning rate
            for feature in self.feature_weights:
                current = self.feature_weights[feature]
                target = target_weights[feature]
                self.feature_weights[feature] = current + self.learning_rate * (target - current)
        
        self.model_trained = True
        self.last_training_time = datetime.now()
        
        logger.debug(
            "Feature weights updated",
            weights=self.feature_weights,
            samples_used=len(recent_history)
        )
    
    def _calculate_feature_importance(
        self,
        history: List[Dict[str, Any]],
        feature: str
    ) -> float:
        """Calculate importance of feature for performance prediction"""
        
        # Simplified importance calculation
        # In production, would use proper statistical methods
        
        performance_scores = []
        for record in history:
            # Calculate overall performance score
            is_score = 1.0 / (1.0 + abs(record.get('implementation_shortfall', 0.0)) * 1000)
            fill_score = record.get('fill_rate', 0.95)
            latency_score = record.get('latency_performance', 0.5)
            
            overall_score = (is_score + fill_score + latency_score) / 3.0
            performance_scores.append(overall_score)
        
        if not performance_scores:
            return 0.25  # Default equal weight
        
        # Calculate variance in performance - higher variance = more important feature
        variance = np.var(performance_scores)
        
        # Normalize to reasonable range
        importance = min(1.0, variance * 10)
        
        return max(0.1, importance)  # Minimum importance to prevent zero weights
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics and model state"""
        
        return {
            'model_state': {
                'model_trained': self.model_trained,
                'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
                'execution_samples': len(self.execution_history),
                'venue_samples': {venue: len(history) for venue, history in self.venue_performance_history.items()}
            },
            'feature_weights': self.feature_weights.copy(),
            'config': {
                'learning_rate': self.learning_rate,
                'history_window_days': self.history_window_days,
                'min_samples_for_learning': self.min_samples_for_learning
            },
            'performance_summary': self._calculate_performance_summary()
        }
    
    def _calculate_performance_summary(self) -> Dict[str, float]:
        """Calculate summary of recent performance"""
        
        if not self.execution_history:
            return {}
        
        recent_history = self.execution_history[-100:]  # Last 100 executions
        
        avg_implementation_shortfall = np.mean([
            abs(h.get('implementation_shortfall', 0.0)) for h in recent_history
        ])
        
        avg_fill_rate = np.mean([
            h.get('fill_rate', 0.95) for h in recent_history
        ])
        
        return {
            'avg_implementation_shortfall_bps': avg_implementation_shortfall * 10000,
            'avg_fill_rate': avg_fill_rate,
            'sample_count': len(recent_history)
        }
    
    def save_model_state(self, filepath: str) -> None:
        """Save optimizer state to file"""
        
        state = {
            'feature_weights': self.feature_weights,
            'execution_history': self.execution_history[-1000:],  # Save last 1000
            'venue_performance_history': {
                venue: history[-100:] for venue, history in self.venue_performance_history.items()
            },
            'model_trained': self.model_trained,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"Model state saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model state: {str(e)}")
    
    def load_model_state(self, filepath: str) -> bool:
        """Load optimizer state from file"""
        
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.feature_weights = state.get('feature_weights', self.feature_weights)
            self.execution_history = state.get('execution_history', [])
            self.venue_performance_history = state.get('venue_performance_history', {})
            self.model_trained = state.get('model_trained', False)
            
            if state.get('last_training_time'):
                self.last_training_time = datetime.fromisoformat(state['last_training_time'])
            
            logger.info(f"Model state loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model state: {str(e)}")
            return False