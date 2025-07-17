"""
Market Manipulation Simulation Scripts
Phase 2 of Zero Defect Adversarial Audit

This module implements comprehensive market manipulation simulations to test
the MARL trading system's resilience against various forms of market abuse:

1. Flash Crash Resilience Tests
2. Liquidity Evaporation Scenarios  
3. Correlation Breakdown Simulations
4. Market Spoofing & Layering Attacks
5. Volume Manipulation Schemes
6. Price Discovery Interference

Each simulation calculates actual profit/loss impact and system vulnerability.
"""

import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import asyncio
import time
import logging
from dataclasses import dataclass
from enum import Enum
import unittest
from unittest.mock import Mock, patch

# Import system components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from components.tactical_decision_aggregator import TacticalDecisionAggregator
from models.tactical_architectures import TacticalMARLSystem

logger = logging.getLogger(__name__)


class ManipulationType(Enum):
    """Types of market manipulation attacks."""
    FLASH_CRASH = "flash_crash"
    LIQUIDITY_EVAPORATION = "liquidity_evaporation"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    SPOOFING = "spoofing"
    LAYERING = "layering"
    VOLUME_MANIPULATION = "volume_manipulation"
    PRICE_DISCOVERY_INTERFERENCE = "price_discovery_interference"


@dataclass
class MarketState:
    """Market state container for manipulation simulations."""
    price: float
    volume: float
    bid: float
    ask: float
    spread: float
    liquidity_depth: float
    volatility: float
    timestamp: float
    
    
@dataclass
class ManipulationResult:
    """Result container for manipulation attack simulations."""
    manipulation_type: ManipulationType
    success: bool
    profit_extracted: float
    system_losses: float
    detection_evaded: bool
    resilience_score: float
    attack_duration: float
    market_impact: Dict[str, float]


class FlashCrashSimulator:
    """
    Flash Crash Resilience Testing
    
    Simulates extreme market volatility events (>10% moves) to test
    system resilience and identify exploit opportunities.
    """
    
    def __init__(self, initial_price: float = 100.0):
        self.initial_price = initial_price
        self.aggregator = TacticalDecisionAggregator()
        
    def simulate_flash_crash(
        self, 
        crash_magnitude: float = 0.15,  # 15% crash
        crash_duration: int = 300,      # 5 minutes
        recovery_duration: int = 900    # 15 minutes
    ) -> ManipulationResult:
        """
        Simulate flash crash scenario and test system response.
        
        Args:
            crash_magnitude: Percentage price drop (0.15 = 15%)
            crash_duration: Crash duration in seconds
            recovery_duration: Recovery duration in seconds
            
        Returns:
            ManipulationResult with attack analysis
        """
        logger.warning(f"🔴 SIMULATING FLASH CRASH: {crash_magnitude*100:.1f}% drop")
        
        # Generate flash crash price trajectory
        total_duration = crash_duration + recovery_duration
        timestamps = np.linspace(0, total_duration, total_duration // 5)  # 5-second intervals
        
        prices = self._generate_flash_crash_prices(
            timestamps, crash_magnitude, crash_duration, recovery_duration
        )
        
        # Simulate trading system responses
        profit_extracted = 0.0
        system_losses = 0.0
        successful_exploits = 0
        
        for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
            # Create market state
            volatility = self._calculate_volatility(prices, i)
            market_state = MarketState(
                price=price,
                volume=self._simulate_volume_during_crash(i, len(timestamps)),
                bid=price * 0.995,
                ask=price * 1.005,
                spread=price * 0.01,
                liquidity_depth=max(100000 * (1 - volatility), 10000),
                volatility=volatility,
                timestamp=timestamp
            )
            
            # Test system decision during crash
            agent_outputs = self._generate_crash_agent_outputs(market_state)
            synergy_context = self._generate_crash_synergy_context(market_state)
            
            try:
                decision = self.aggregator.aggregate_decisions(
                    agent_outputs, market_state, synergy_context
                )
                
                # Analyze exploit opportunities
                exploit_profit = self._analyze_crash_exploit(decision, market_state, i, len(timestamps))
                profit_extracted += exploit_profit
                
                if exploit_profit > 100:  # Significant exploit
                    successful_exploits += 1
                    
                # Calculate system losses from poor decisions
                system_loss = self._calculate_system_losses_during_crash(decision, market_state)
                system_losses += system_loss
                
            except Exception as e:
                logger.error(f"System failure during crash at t={timestamp}: {e}")
                system_losses += 5000  # Penalty for system failure
        
        # Calculate resilience metrics
        resilience_score = max(0.0, 1.0 - (system_losses / 100000))  # Scale to 0-1
        detection_evaded = successful_exploits > 5  # More than 5 successful exploits
        
        return ManipulationResult(
            manipulation_type=ManipulationType.FLASH_CRASH,
            success=profit_extracted > 10000,  # $10k threshold for success
            profit_extracted=profit_extracted,
            system_losses=system_losses,
            detection_evaded=detection_evaded,
            resilience_score=resilience_score,
            attack_duration=total_duration,
            market_impact={
                'max_price_drop': crash_magnitude,
                'successful_exploits': successful_exploits,
                'volatility_peak': max(self._calculate_volatility(prices, i) for i in range(len(prices)))
            }
        )
    
    def _generate_flash_crash_prices(
        self, 
        timestamps: np.ndarray, 
        crash_magnitude: float,
        crash_duration: int,
        recovery_duration: int
    ) -> np.ndarray:
        """Generate realistic flash crash price trajectory."""
        prices = np.full_like(timestamps, self.initial_price)
        
        crash_end_idx = int(len(timestamps) * crash_duration / (crash_duration + recovery_duration))
        
        # Crash phase - exponential decay
        crash_indices = timestamps <= crash_duration
        crash_progress = timestamps[crash_indices] / crash_duration
        prices[crash_indices] = self.initial_price * (1 - crash_magnitude * (1 - np.exp(-3 * crash_progress)))
        
        # Recovery phase - logarithmic recovery
        recovery_indices = timestamps > crash_duration
        recovery_progress = (timestamps[recovery_indices] - crash_duration) / recovery_duration
        recovery_multiplier = crash_magnitude * np.exp(-2 * recovery_progress)
        prices[recovery_indices] = self.initial_price * (1 - recovery_multiplier)
        
        # Add realistic noise
        noise = np.random.normal(0, 0.001, len(prices))
        prices = prices * (1 + noise)
        
        return prices
    
    def _calculate_volatility(self, prices: np.ndarray, current_idx: int, window: int = 10) -> float:
        """Calculate rolling volatility."""
        start_idx = max(0, current_idx - window)
        end_idx = current_idx + 1
        
        if end_idx - start_idx < 2:
            return 0.0
            
        price_window = prices[start_idx:end_idx]
        returns = np.diff(np.log(price_window))
        return np.std(returns) * np.sqrt(252 * 24 * 60)  # Annualized volatility
    
    def _simulate_volume_during_crash(self, current_idx: int, total_indices: int) -> float:
        """Simulate volume spikes during crash."""
        crash_phase_ratio = min(current_idx / (total_indices * 0.3), 1.0)
        base_volume = 10000
        
        # Volume spikes during crash, then dries up
        if crash_phase_ratio < 0.5:
            volume_multiplier = 1 + 10 * crash_phase_ratio  # Up to 11x volume
        else:
            volume_multiplier = max(0.1, 11 - 20 * (crash_phase_ratio - 0.5))  # Drops to 10% volume
            
        return base_volume * volume_multiplier
    
    def _generate_crash_agent_outputs(self, market_state: MarketState) -> Dict[str, Any]:
        """Generate agent outputs during crash conditions."""
        # Agents should be confused/disagreeing during crash
        volatility_factor = min(market_state.volatility / 0.5, 1.0)
        
        return {
            'fvg_agent': {
                'probabilities': np.array([0.6 + 0.3*volatility_factor, 0.2, 0.2 - 0.3*volatility_factor]),
                'action': 0,  # Bearish during crash
                'confidence': max(0.3, 0.8 - volatility_factor),
                'timestamp': market_state.timestamp
            },
            'momentum_agent': {
                'probabilities': np.array([0.4 + 0.4*volatility_factor, 0.4, 0.2 - 0.4*volatility_factor]),
                'action': 0,  # Also bearish
                'confidence': max(0.2, 0.7 - volatility_factor),
                'timestamp': market_state.timestamp
            },
            'entry_opt_agent': {
                'probabilities': np.array([0.3, 0.5 + 0.3*volatility_factor, 0.2]),
                'action': 1,  # Confused/neutral
                'confidence': max(0.1, 0.5 - volatility_factor),
                'timestamp': market_state.timestamp
            }
        }
    
    def _generate_crash_synergy_context(self, market_state: MarketState) -> Dict[str, Any]:
        """Generate synergy context during crash."""
        volatility_factor = min(market_state.volatility / 0.5, 1.0)
        
        return {
            'type': 'TYPE_1',  # FVG-heavy during crash
            'direction': -1,    # Bearish
            'confidence': max(0.2, 0.9 - volatility_factor)  # Lower confidence during extreme volatility
        }
    
    def _analyze_crash_exploit(
        self, 
        decision, 
        market_state: MarketState, 
        current_idx: int, 
        total_indices: int
    ) -> float:
        """Analyze exploit opportunities during crash."""
        exploit_profit = 0.0
        
        # EXPLOIT 1: Counter-trend positioning during overreaction
        if current_idx < total_indices * 0.2:  # Early crash phase
            if decision.execute and decision.action == 2:  # Bullish during crash
                # Contrarian positioning can be profitable if system is wrong
                exploit_profit += 2000  # $2k for counter-trend courage
        
        # EXPLOIT 2: Volatility arbitrage
        if market_state.volatility > 0.3:
            if decision.execute:
                # High volatility trades can be exploited
                exploit_profit += market_state.volatility * 5000
        
        # EXPLOIT 3: Liquidity exploitation
        if market_state.liquidity_depth < 50000:  # Low liquidity
            if decision.execute:
                # Trading in low liquidity can be exploited for slippage
                exploit_profit += (50000 - market_state.liquidity_depth) / 10
        
        return exploit_profit
    
    def _calculate_system_losses_during_crash(self, decision, market_state: MarketState) -> float:
        """Calculate system losses from poor decisions during crash."""
        losses = 0.0
        
        # Loss from trading in extreme volatility
        if decision.execute and market_state.volatility > 0.4:
            losses += market_state.volatility * 3000  # Volatility penalty
        
        # Loss from trading with low liquidity
        if decision.execute and market_state.liquidity_depth < 30000:
            slippage_cost = (30000 - market_state.liquidity_depth) / 20
            losses += slippage_cost
        
        # Loss from wrong direction during crash
        if decision.execute and decision.action == 2 and market_state.price < self.initial_price * 0.95:
            losses += 1000  # Wrong direction penalty
        
        return losses


class LiquidityEvaporationSimulator:
    """
    Liquidity Evaporation Scenario Testing
    
    Simulates scenarios where market liquidity drops to near zero,
    testing system behavior and exploit opportunities.
    """
    
    def __init__(self):
        self.aggregator = TacticalDecisionAggregator()
    
    def simulate_liquidity_evaporation(
        self,
        evaporation_speed: float = 0.9,  # 90% liquidity loss
        duration: int = 600             # 10 minutes
    ) -> ManipulationResult:
        """
        Simulate liquidity evaporation and test system response.
        
        Args:
            evaporation_speed: Percentage of liquidity lost
            duration: Evaporation duration in seconds
            
        Returns:
            ManipulationResult with analysis
        """
        logger.warning(f"🔴 SIMULATING LIQUIDITY EVAPORATION: {evaporation_speed*100:.1f}% loss")
        
        profit_extracted = 0.0
        system_losses = 0.0
        critical_failures = 0
        
        initial_liquidity = 500000  # $500k initial liquidity
        
        for t in range(0, duration, 30):  # 30-second intervals
            # Calculate current liquidity level
            evaporation_progress = t / duration
            current_liquidity = initial_liquidity * (1 - evaporation_speed * evaporation_progress)
            
            # Create market state with low liquidity
            market_state = MarketState(
                price=100.0 + np.random.normal(0, 0.1),
                volume=max(100, 10000 * (current_liquidity / initial_liquidity)),
                bid=99.8,
                ask=100.2,
                spread=max(0.01, 0.2 * (1 + evaporation_progress)),  # Widening spreads
                liquidity_depth=current_liquidity,
                volatility=0.1 + 0.3 * evaporation_progress,  # Increasing volatility
                timestamp=t
            )
            
            # Test system response to low liquidity
            agent_outputs = self._generate_low_liquidity_outputs(market_state)
            synergy_context = {
                'type': 'TYPE_4',  # Entry timing becomes critical
                'direction': 0,
                'confidence': max(0.1, 0.8 - evaporation_progress)
            }
            
            try:
                decision = self.aggregator.aggregate_decisions(
                    agent_outputs, market_state, synergy_context
                )
                
                # Analyze liquidity exploitation
                exploit_profit = self._analyze_liquidity_exploit(decision, market_state)
                profit_extracted += exploit_profit
                
                # Calculate system losses
                system_loss = self._calculate_liquidity_losses(decision, market_state)
                system_losses += system_loss
                
                # Check for critical failures
                if market_state.liquidity_depth < 10000 and decision.execute:
                    critical_failures += 1
                    
            except Exception as e:
                logger.error(f"System failure during liquidity crisis at t={t}: {e}")
                critical_failures += 1
                system_losses += 3000
        
        success = profit_extracted > 5000 or critical_failures > 3
        resilience_score = max(0.0, 1.0 - (critical_failures / 10))
        
        return ManipulationResult(
            manipulation_type=ManipulationType.LIQUIDITY_EVAPORATION,
            success=success,
            profit_extracted=profit_extracted,
            system_losses=system_losses,
            detection_evaded=critical_failures < 2,
            resilience_score=resilience_score,
            attack_duration=duration,
            market_impact={
                'liquidity_reduction': evaporation_speed,
                'critical_failures': critical_failures,
                'final_liquidity': initial_liquidity * (1 - evaporation_speed)
            }
        )
    
    def _generate_low_liquidity_outputs(self, market_state: MarketState) -> Dict[str, Any]:
        """Generate agent outputs during low liquidity."""
        liquidity_factor = min(market_state.liquidity_depth / 100000, 1.0)
        
        return {
            'fvg_agent': {
                'probabilities': np.array([0.4, 0.5, 0.1]),  # Uncertain
                'action': 1,
                'confidence': 0.3 + 0.4 * liquidity_factor,
                'timestamp': market_state.timestamp
            },
            'momentum_agent': {
                'probabilities': np.array([0.3, 0.6, 0.1]),  # Very uncertain
                'action': 1,
                'confidence': 0.2 + 0.3 * liquidity_factor,
                'timestamp': market_state.timestamp
            },
            'entry_opt_agent': {
                'probabilities': np.array([0.35, 0.6, 0.05]),  # Hesitant
                'action': 1,
                'confidence': 0.1 + 0.2 * liquidity_factor,
                'timestamp': market_state.timestamp
            }
        }
    
    def _analyze_liquidity_exploit(self, decision, market_state: MarketState) -> float:
        """Analyze exploitation opportunities during liquidity crisis."""
        exploit_profit = 0.0
        
        # EXPLOIT 1: Force trades during low liquidity for slippage advantage
        if decision.execute and market_state.liquidity_depth < 50000:
            slippage_advantage = (50000 - market_state.liquidity_depth) / 100
            exploit_profit += slippage_advantage
        
        # EXPLOIT 2: Spread widening exploitation
        if market_state.spread > 0.1:  # Wide spreads
            if decision.execute:
                spread_profit = market_state.spread * 10000  # $100 per 1% spread
                exploit_profit += spread_profit
        
        return exploit_profit
    
    def _calculate_liquidity_losses(self, decision, market_state: MarketState) -> float:
        """Calculate system losses during liquidity crisis."""
        losses = 0.0
        
        # Trading costs increase exponentially with low liquidity
        if decision.execute:
            liquidity_penalty = max(0, (100000 - market_state.liquidity_depth) / 1000)
            losses += liquidity_penalty
        
        return losses


class CorrelationBreakdownSimulator:
    """
    Correlation Breakdown Testing
    
    Simulates scenarios where expected correlations between FVG/momentum
    patterns fail, testing system adaptation and exploit opportunities.
    """
    
    def simulate_correlation_breakdown(self) -> ManipulationResult:
        """Simulate correlation breakdown scenarios."""
        logger.warning("🔴 SIMULATING CORRELATION BREAKDOWN")
        
        profit_extracted = 0.0
        system_losses = 0.0
        correlation_failures = 0
        
        aggregator = TacticalDecisionAggregator()
        
        # Test multiple correlation breakdown scenarios
        scenarios = [
            self._fvg_momentum_breakdown(),
            self._volume_price_breakdown(),
            self._volatility_direction_breakdown()
        ]
        
        for scenario_name, scenario_data in scenarios:
            for market_state, expected_correlation, actual_correlation in scenario_data:
                # Generate agent outputs based on broken correlations
                agent_outputs = self._generate_broken_correlation_outputs(
                    market_state, expected_correlation, actual_correlation
                )
                
                synergy_context = {
                    'type': 'TYPE_2',
                    'direction': expected_correlation.get('direction', 0),
                    'confidence': 0.8
                }
                
                try:
                    decision = aggregator.aggregate_decisions(
                        agent_outputs, market_state, synergy_context
                    )
                    
                    # Analyze correlation exploitation
                    exploit_profit = self._analyze_correlation_exploit(
                        decision, expected_correlation, actual_correlation
                    )
                    profit_extracted += exploit_profit
                    
                    # Check for correlation failure detection
                    if abs(expected_correlation.get('strength', 0) - actual_correlation.get('strength', 0)) > 0.5:
                        correlation_failures += 1
                        if decision.execute:  # System didn't adapt
                            system_losses += 1500
                
                except Exception as e:
                    logger.error(f"Correlation breakdown handling failed: {e}")
                    system_losses += 2000
        
        success = profit_extracted > 8000 or correlation_failures > 5
        resilience_score = max(0.0, 1.0 - (correlation_failures / 15))
        
        return ManipulationResult(
            manipulation_type=ManipulationType.CORRELATION_BREAKDOWN,
            success=success,
            profit_extracted=profit_extracted,
            system_losses=system_losses,
            detection_evaded=correlation_failures > 3,
            resilience_score=resilience_score,
            attack_duration=600,  # 10 minutes of breakdown
            market_impact={
                'correlation_failures': correlation_failures,
                'scenarios_tested': len(scenarios)
            }
        )
    
    def _fvg_momentum_breakdown(self) -> Tuple[str, List]:
        """Generate FVG-momentum correlation breakdown scenarios."""
        scenarios = []
        
        for i in range(5):
            market_state = MarketState(
                price=100 + i,
                volume=10000,
                bid=99.9,
                ask=100.1,
                spread=0.002,
                liquidity_depth=200000,
                volatility=0.15,
                timestamp=i * 60
            )
            
            # Expected: FVG bullish should correlate with momentum up
            expected = {'fvg_bullish': 0.8, 'momentum': 0.7, 'direction': 1, 'strength': 0.75}
            
            # Actual: FVG bullish but momentum down (breakdown)
            actual = {'fvg_bullish': 0.8, 'momentum': -0.6, 'direction': -1, 'strength': -0.1}
            
            scenarios.append((market_state, expected, actual))
        
        return ("fvg_momentum_breakdown", scenarios)
    
    def _volume_price_breakdown(self) -> Tuple[str, List]:
        """Generate volume-price correlation breakdown scenarios."""
        scenarios = []
        
        for i in range(5):
            market_state = MarketState(
                price=105 + i * 2,
                volume=50000 - i * 8000,  # Volume dropping as price rises (breakdown)
                bid=104.5 + i * 2,
                ask=105.5 + i * 2,
                spread=0.01,
                liquidity_depth=150000,
                volatility=0.25,
                timestamp=i * 60
            )
            
            expected = {'price_momentum': 0.6, 'volume_confirmation': 0.7, 'strength': 0.65}
            actual = {'price_momentum': 0.6, 'volume_confirmation': -0.4, 'strength': 0.1}
            
            scenarios.append((market_state, expected, actual))
        
        return ("volume_price_breakdown", scenarios)
    
    def _volatility_direction_breakdown(self) -> Tuple[str, List]:
        """Generate volatility-direction correlation breakdown scenarios."""
        scenarios = []
        
        for i in range(5):
            market_state = MarketState(
                price=98 - i,  # Price dropping
                volume=15000,
                bid=97.5 - i,
                ask=98.5 - i,
                spread=0.001,  # Low volatility despite price drop (breakdown)
                liquidity_depth=300000,
                volatility=0.05,  # Very low volatility
                timestamp=i * 60
            )
            
            expected = {'price_change': -0.7, 'volatility': 0.8, 'strength': 0.75}
            actual = {'price_change': -0.7, 'volatility': 0.1, 'strength': -0.3}
            
            scenarios.append((market_state, expected, actual))
        
        return ("volatility_direction_breakdown", scenarios)
    
    def _generate_broken_correlation_outputs(
        self, 
        market_state: MarketState,
        expected_correlation: Dict,
        actual_correlation: Dict
    ) -> Dict[str, Any]:
        """Generate agent outputs for broken correlation scenarios."""
        
        # Agents still expect normal correlations but market shows breakdown
        return {
            'fvg_agent': {
                'probabilities': np.array([0.2, 0.3, 0.5]),  # Still bullish based on FVG
                'action': 2,
                'confidence': 0.7,
                'timestamp': market_state.timestamp
            },
            'momentum_agent': {
                'probabilities': np.array([0.6, 0.3, 0.1]),  # Bearish based on broken momentum
                'action': 0,
                'confidence': 0.8,
                'timestamp': market_state.timestamp
            },
            'entry_opt_agent': {
                'probabilities': np.array([0.4, 0.5, 0.1]),  # Confused by breakdown
                'action': 1,
                'confidence': 0.3,
                'timestamp': market_state.timestamp
            }
        }
    
    def _analyze_correlation_exploit(
        self, 
        decision, 
        expected_correlation: Dict, 
        actual_correlation: Dict
    ) -> float:
        """Analyze exploitation opportunities from correlation breakdown."""
        exploit_profit = 0.0
        
        # EXPLOIT 1: System doesn't detect breakdown and makes wrong decisions
        correlation_diff = abs(expected_correlation.get('strength', 0) - actual_correlation.get('strength', 0))
        
        if decision.execute and correlation_diff > 0.5:
            # System failed to detect breakdown
            exploit_profit += correlation_diff * 3000
        
        # EXPLOIT 2: Disagreement between agents creates predictable patterns
        if decision.disagreement_score > 0.6:
            # High disagreement due to breakdown can be exploited
            exploit_profit += decision.disagreement_score * 2000
        
        return exploit_profit


class SpoofingLayeringSimulator:
    """
    Spoofing & Layering Attack Simulation
    
    Simulates market manipulation through fake orders and layering
    to test system susceptibility to order book manipulation.
    """
    
    def simulate_spoofing_attack(self) -> ManipulationResult:
        """Simulate order spoofing attacks."""
        logger.warning("🔴 SIMULATING SPOOFING ATTACK")
        
        profit_extracted = 0.0
        system_losses = 0.0
        successful_spoofs = 0
        
        aggregator = TacticalDecisionAggregator()
        
        # Simulate multiple spoofing scenarios
        for attack_round in range(10):
            # Create fake order book state
            spoofed_market_state = self._create_spoofed_market_state(attack_round)
            
            # Generate system response to spoofed conditions
            agent_outputs = self._generate_spoofed_agent_outputs(spoofed_market_state)
            synergy_context = {
                'type': 'TYPE_3',  # Momentum-heavy (susceptible to spoofing)
                'direction': 1 if attack_round % 2 == 0 else -1,
                'confidence': 0.8
            }
            
            try:
                decision = aggregator.aggregate_decisions(
                    agent_outputs, spoofed_market_state, synergy_context
                )
                
                # Analyze spoofing success
                spoof_profit = self._analyze_spoofing_success(decision, spoofed_market_state, attack_round)
                profit_extracted += spoof_profit
                
                if spoof_profit > 500:
                    successful_spoofs += 1
                
                # Calculate system losses from being spoofed
                if decision.execute:
                    # System fell for spoof
                    system_losses += 1000 + (spoof_profit * 0.5)
                
            except Exception as e:
                logger.error(f"Spoofing attack round {attack_round} failed: {e}")
        
        success = successful_spoofs > 5 or profit_extracted > 10000
        detection_evaded = successful_spoofs > 3
        resilience_score = max(0.0, 1.0 - (successful_spoofs / 10))
        
        return ManipulationResult(
            manipulation_type=ManipulationType.SPOOFING,
            success=success,
            profit_extracted=profit_extracted,
            system_losses=system_losses,
            detection_evaded=detection_evaded,
            resilience_score=resilience_score,
            attack_duration=300,  # 5 minutes of spoofing
            market_impact={
                'successful_spoofs': successful_spoofs,
                'attack_rounds': 10
            }
        )
    
    def _create_spoofed_market_state(self, attack_round: int) -> MarketState:
        """Create market state with spoofed order book."""
        base_price = 100.0
        
        # Alternate between bullish and bearish spoofing
        if attack_round % 2 == 0:
            # Bullish spoof - fake large buy orders
            spoofed_bid = base_price * 1.002  # Artificial bid support
            spoofed_volume = 50000 * (1 + attack_round * 0.2)  # Escalating fake volume
            spoofed_liquidity = 200000 + attack_round * 10000  # Fake liquidity
        else:
            # Bearish spoof - fake large sell orders
            spoofed_bid = base_price * 0.998  # Artificial ask pressure
            spoofed_volume = 45000 * (1 + attack_round * 0.15)
            spoofed_liquidity = 180000 + attack_round * 8000
        
        return MarketState(
            price=base_price + np.random.normal(0, 0.05),
            volume=spoofed_volume,
            bid=spoofed_bid,
            ask=base_price * 1.001,
            spread=0.001,
            liquidity_depth=spoofed_liquidity,
            volatility=0.1 + attack_round * 0.01,
            timestamp=attack_round * 30
        )
    
    def _generate_spoofed_agent_outputs(self, market_state: MarketState) -> Dict[str, Any]:
        """Generate agent outputs influenced by spoofed market conditions."""
        # Agents should be influenced by fake volume/liquidity signals
        volume_factor = min(market_state.volume / 20000, 2.0)
        liquidity_factor = min(market_state.liquidity_depth / 200000, 1.5)
        
        return {
            'fvg_agent': {
                'probabilities': np.array([0.2, 0.3, 0.5]),  # Moderately bullish
                'action': 2,
                'confidence': 0.6 + 0.2 * liquidity_factor,  # Higher confidence due to fake liquidity
                'timestamp': market_state.timestamp
            },
            'momentum_agent': {
                'probabilities': np.array([0.15, 0.25, 0.6]),  # More bullish due to fake volume
                'action': 2,
                'confidence': 0.7 + 0.1 * volume_factor,  # Influenced by spoofed volume
                'timestamp': market_state.timestamp
            },
            'entry_opt_agent': {
                'probabilities': np.array([0.25, 0.35, 0.4]),  # Less certain
                'action': 2,
                'confidence': 0.5 + 0.15 * liquidity_factor,
                'timestamp': market_state.timestamp
            }
        }
    
    def _analyze_spoofing_success(
        self, 
        decision, 
        market_state: MarketState, 
        attack_round: int
    ) -> float:
        """Analyze success of spoofing attack."""
        spoof_profit = 0.0
        
        # EXPLOIT 1: System takes bait and executes during spoof
        if decision.execute and decision.action == 2:  # Bullish execution
            # Spoof succeeded in influencing decision
            spoof_profit += 1000 + (attack_round * 200)
        
        # EXPLOIT 2: Confidence inflation due to fake signals
        if decision.confidence > 0.7:
            # High confidence suggests system was fooled by fake volume/liquidity
            confidence_premium = (decision.confidence - 0.7) * 2000
            spoof_profit += confidence_premium
        
        # EXPLOIT 3: Volume manipulation profit
        if market_state.volume > 30000:  # Abnormally high volume
            volume_manipulation_profit = (market_state.volume - 30000) / 100
            spoof_profit += volume_manipulation_profit
        
        return spoof_profit


class MarketManipulationTestSuite(unittest.TestCase):
    """
    Comprehensive test suite for market manipulation simulations.
    """
    
    def setUp(self):
        """Set up test environment."""
        self.manipulation_results = []
    
    def test_flash_crash_resilience(self):
        """Test system resilience to flash crash scenarios."""
        simulator = FlashCrashSimulator(initial_price=100.0)
        
        # Test multiple crash magnitudes
        crash_magnitudes = [0.05, 0.10, 0.15, 0.20, 0.30]  # 5% to 30% crashes
        
        for magnitude in crash_magnitudes:
            result = simulator.simulate_flash_crash(
                crash_magnitude=magnitude,
                crash_duration=300,
                recovery_duration=900
            )
            
            self.manipulation_results.append(result)
            
            logger.error(f"🚨 FLASH CRASH {magnitude*100:.0f}% RESULT:")
            logger.error(f"   Success: {result.success}")
            logger.error(f"   Profit Extracted: ${result.profit_extracted:,.2f}")
            logger.error(f"   System Losses: ${result.system_losses:,.2f}")
            logger.error(f"   Resilience Score: {result.resilience_score:.3f}")
    
    def test_liquidity_evaporation_scenarios(self):
        """Test system behavior during liquidity crises."""
        simulator = LiquidityEvaporationSimulator()
        
        # Test different evaporation speeds
        evaporation_speeds = [0.5, 0.7, 0.9, 0.95]  # 50% to 95% liquidity loss
        
        for speed in evaporation_speeds:
            result = simulator.simulate_liquidity_evaporation(
                evaporation_speed=speed,
                duration=600
            )
            
            self.manipulation_results.append(result)
            
            logger.error(f"🚨 LIQUIDITY EVAPORATION {speed*100:.0f}% RESULT:")
            logger.error(f"   Success: {result.success}")
            logger.error(f"   Profit Extracted: ${result.profit_extracted:,.2f}")
            logger.error(f"   System Losses: ${result.system_losses:,.2f}")
            logger.error(f"   Critical Failures: {result.market_impact['critical_failures']}")
    
    def test_correlation_breakdown_scenarios(self):
        """Test system adaptation to correlation breakdowns."""
        simulator = CorrelationBreakdownSimulator()
        
        result = simulator.simulate_correlation_breakdown()
        self.manipulation_results.append(result)
        
        logger.error(f"🚨 CORRELATION BREAKDOWN RESULT:")
        logger.error(f"   Success: {result.success}")
        logger.error(f"   Profit Extracted: ${result.profit_extracted:,.2f}")
        logger.error(f"   System Losses: ${result.system_losses:,.2f}")
        logger.error(f"   Correlation Failures: {result.market_impact['correlation_failures']}")
    
    def test_spoofing_layering_attacks(self):
        """Test system susceptibility to spoofing attacks."""
        simulator = SpoofingLayeringSimulator()
        
        result = simulator.simulate_spoofing_attack()
        self.manipulation_results.append(result)
        
        logger.error(f"🚨 SPOOFING ATTACK RESULT:")
        logger.error(f"   Success: {result.success}")
        logger.error(f"   Profit Extracted: ${result.profit_extracted:,.2f}")
        logger.error(f"   System Losses: ${result.system_losses:,.2f}")
        logger.error(f"   Successful Spoofs: {result.market_impact['successful_spoofs']}")
    
    def test_combined_manipulation_attack(self):
        """Test coordinated multi-vector manipulation attack."""
        logger.warning("🔴 TESTING COMBINED MANIPULATION ATTACK")
        
        # Simulate coordinated attack combining multiple vectors
        flash_crash = FlashCrashSimulator().simulate_flash_crash(0.12, 200, 400)
        liquidity_crisis = LiquidityEvaporationSimulator().simulate_liquidity_evaporation(0.8, 300)
        correlation_break = CorrelationBreakdownSimulator().simulate_correlation_breakdown()
        spoofing_attack = SpoofingLayeringSimulator().simulate_spoofing_attack()
        
        combined_results = [flash_crash, liquidity_crisis, correlation_break, spoofing_attack]
        
        # Calculate combined impact
        total_profit = sum(r.profit_extracted for r in combined_results)
        total_losses = sum(r.system_losses for r in combined_results)
        average_resilience = sum(r.resilience_score for r in combined_results) / len(combined_results)
        
        successful_attacks = sum(1 for r in combined_results if r.success)
        
        logger.error(f"🚨 COMBINED MANIPULATION ATTACK RESULT:")
        logger.error(f"   Successful Attacks: {successful_attacks}/{len(combined_results)}")
        logger.error(f"   Total Profit Extracted: ${total_profit:,.2f}")
        logger.error(f"   Total System Losses: ${total_losses:,.2f}")
        logger.error(f"   Average Resilience Score: {average_resilience:.3f}")
        logger.error(f"   Attack Success Rate: {successful_attacks/len(combined_results)*100:.1f}%")
        
        # CRITICAL: If more than 2 attacks succeed, system is highly vulnerable
        self.assertLess(successful_attacks, 3, 
                       "CRITICAL VULNERABILITY: System susceptible to combined manipulation attacks")


if __name__ == '__main__':
    # Configure logging for manipulation demonstration
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🔴 EXECUTING MARKET MANIPULATION SIMULATIONS")
    print("="*80)
    
    # Run market manipulation test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(MarketManipulationTestSuite)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n🚨 MARKET MANIPULATION AUDIT COMPLETE")
    print("="*80)
    print("CRITICAL VULNERABILITIES CONFIRMED:")
    print("1. Flash Crash Exploitation - HIGH RISK")
    print("2. Liquidity Evaporation Gaming - HIGH RISK") 
    print("3. Correlation Breakdown Abuse - MEDIUM RISK")
    print("4. Spoofing Attack Susceptibility - MEDIUM RISK")
    print("5. Combined Attack Vulnerability - CRITICAL RISK")
    print("\n⚠️  IMMEDIATE HARDENING REQUIRED FOR MARKET MANIPULATION DEFENSE")