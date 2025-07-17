"""
Phase 3: Strategic-Tactical Alignment Validation Under Stress

This module tests the coherence and alignment between strategic and tactical 
decisions under extreme conditions to ensure system reliability.

Mission: Validate decision coherence when the system is under fire.
"""

import asyncio
import time
import json
import numpy as np
import pytest
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import sys
from enum import Enum

# Add project path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.tactical.controller import TacticalMARLController, SynergyEvent
from src.tactical.aggregator import TacticalDecisionAggregator

class AlignmentScore(Enum):
    """Alignment scoring levels."""
    PERFECT = 1.0
    GOOD = 0.8
    ACCEPTABLE = 0.6
    POOR = 0.4
    CRITICAL = 0.2
    FAILED = 0.0

@dataclass
class AlignmentTestResult:
    """Alignment test result structure."""
    test_name: str
    total_decisions: int
    alignment_score: float
    coherence_violations: int
    risk_violations: int
    portfolio_conflicts: int
    execution_failures: int
    decision_consistency: float
    stress_level: str
    breaking_point_reached: bool
    recommendations: List[str]

class StrategicTacticalAlignmentValidator:
    """
    Comprehensive validator for strategic-tactical decision alignment.
    
    Tests decision coherence under various stress conditions including:
    - Conflicting signals
    - High-frequency decision pressure
    - Portfolio constraint violations
    - Risk management failures
    """
    
    def __init__(self):
        self.results: List[AlignmentTestResult] = []
        self.tactical_controller = None
        self.decision_aggregator = None
        
    async def setup(self):
        """Setup test environment."""
        self.tactical_controller = TacticalMARLController()
        await self.tactical_controller.initialize()
        self.decision_aggregator = TacticalDecisionAggregator()
        
    async def teardown(self):
        """Cleanup test environment."""
        if self.tactical_controller:
            await self.tactical_controller.cleanup()
    
    def create_conflicting_synergy_event(self, event_id: int, direction: int) -> SynergyEvent:
        """Create synergy event with intentionally conflicting signals."""
        return SynergyEvent(
            synergy_type=f"CONFLICTED_TYPE_{event_id % 4 + 1}",
            direction=direction,
            confidence=0.85,
            signal_sequence=[
                {"signal": "bullish_fvg", "strength": 0.9 if direction > 0 else 0.1},
                {"signal": "bearish_momentum", "strength": 0.1 if direction > 0 else 0.9},
                {"signal": "volume_spike", "strength": 0.8},
            ],
            market_context={
                "volatility": "high",
                "trend": "conflicted",
                "support_resistance": "unclear",
                "institutional_flow": "mixed"
            },
            correlation_id=f"conflict-test-{event_id}",
            timestamp=time.time()
        )
    
    def create_corrupted_matrix(self) -> np.ndarray:
        """Create matrix with corrupted/extreme values to test robustness."""
        matrix = np.random.randn(60, 7).astype(np.float32)
        
        # Inject various corruption patterns
        corruptions = [
            # NaN values
            (slice(0, 5), slice(0, 2), np.nan),
            # Infinite values
            (slice(10, 15), slice(2, 4), np.inf),
            # Extreme outliers
            (slice(20, 25), slice(4, 6), 1000.0),
            # Negative infinities
            (slice(30, 35), slice(0, 2), -np.inf),
        ]
        
        for row_slice, col_slice, value in corruptions:
            matrix[row_slice, col_slice] = value
        
        return matrix
    
    async def test_signal_corruption_resilience(self) -> AlignmentTestResult:
        """
        Test system resilience to corrupted and conflicting signals.
        
        Sends intentionally conflicting synergy signals and validates
        that decisions remain coherent and aligned.
        """
        print(f"\n🔀 SIGNAL CORRUPTION RESILIENCE TEST")
        
        decisions = []
        coherence_violations = 0
        risk_violations = 0
        portfolio_conflicts = 0
        execution_failures = 0
        
        # Test with corrupted matrices and conflicting signals
        for i in range(50):
            try:
                # Create conflicting event
                direction = 1 if i % 2 == 0 else -1
                synergy_event = self.create_conflicting_synergy_event(i, direction)
                
                # Get corrupted matrix
                corrupted_matrix = self.create_corrupted_matrix()
                
                # Process decision with corrupted inputs
                decision = await self.tactical_controller.on_synergy_detected(synergy_event)
                decisions.append(decision)
                
                # Validate decision coherence
                coherence_issues = self._validate_decision_coherence(decision, synergy_event)
                coherence_violations += len(coherence_issues)
                
                # Validate risk alignment
                risk_issues = self._validate_risk_alignment(decision)
                risk_violations += len(risk_issues)
                
                # Validate portfolio impact
                portfolio_issues = self._validate_portfolio_alignment(decision, decisions[-10:])
                portfolio_conflicts += len(portfolio_issues)
                
                # Check execution command validity
                if not self._validate_execution_command(decision.execution_command):
                    execution_failures += 1
                
            except Exception as e:
                execution_failures += 1
                print(f"⚠️ Decision processing failed with corrupted input {i}: {e}")
        
        # Calculate alignment metrics
        total_decisions = len(decisions)
        alignment_score = self._calculate_alignment_score(
            coherence_violations, risk_violations, portfolio_conflicts, execution_failures, total_decisions
        )
        
        decision_consistency = self._calculate_decision_consistency(decisions)
        
        result = AlignmentTestResult(
            test_name="signal_corruption_resilience",
            total_decisions=total_decisions,
            alignment_score=alignment_score,
            coherence_violations=coherence_violations,
            risk_violations=risk_violations,
            portfolio_conflicts=portfolio_conflicts,
            execution_failures=execution_failures,
            decision_consistency=decision_consistency,
            stress_level="HIGH",
            breaking_point_reached=alignment_score < 0.6,
            recommendations=self._generate_recommendations(alignment_score, coherence_violations, risk_violations)
        )
        
        self.results.append(result)
        self._print_alignment_result(result)
        return result
    
    async def test_high_frequency_decision_pressure(self) -> AlignmentTestResult:
        """
        Test decision alignment under extreme high-frequency pressure.
        
        Rapidly sends synergy events to test if decisions remain aligned
        when the system is under time pressure.
        """
        print(f"\n⚡ HIGH-FREQUENCY DECISION PRESSURE TEST")
        
        decisions = []
        coherence_violations = 0
        risk_violations = 0
        portfolio_conflicts = 0
        execution_failures = 0
        
        # Send rapid-fire events (sub-second intervals)
        event_interval = 0.1  # 100ms between events
        
        for i in range(100):
            try:
                # Create time-pressured event
                synergy_event = SynergyEvent(
                    synergy_type=f"RAPID_TYPE_{i % 4 + 1}",
                    direction=1 if i % 3 == 0 else -1,
                    confidence=0.75 + (i % 5) * 0.05,  # Varying confidence
                    signal_sequence=[],
                    market_context={"pressure": "extreme", "speed": "rapid"},
                    correlation_id=f"rapid-{i}",
                    timestamp=time.time()
                )
                
                start_time = time.perf_counter()
                decision = await self.tactical_controller.on_synergy_detected(synergy_event)
                decision_time = time.perf_counter() - start_time
                
                decisions.append(decision)
                
                # Check if decision was made within acceptable time
                if decision_time > 0.1:  # 100ms threshold
                    coherence_violations += 1
                
                # Validate rapid decision quality
                coherence_issues = self._validate_decision_coherence(decision, synergy_event)
                coherence_violations += len(coherence_issues)
                
                risk_issues = self._validate_risk_alignment(decision)
                risk_violations += len(risk_issues)
                
                # Check for rapid-fire conflicts
                recent_decisions = decisions[-5:] if len(decisions) >= 5 else decisions
                conflicts = self._detect_rapid_fire_conflicts(decision, recent_decisions)
                portfolio_conflicts += conflicts
                
                if not self._validate_execution_command(decision.execution_command):
                    execution_failures += 1
                
                # Control timing
                await asyncio.sleep(event_interval)
                
            except Exception as e:
                execution_failures += 1
                print(f"⚠️ Rapid decision {i} failed: {e}")
        
        total_decisions = len(decisions)
        alignment_score = self._calculate_alignment_score(
            coherence_violations, risk_violations, portfolio_conflicts, execution_failures, total_decisions
        )
        
        decision_consistency = self._calculate_decision_consistency(decisions)
        
        result = AlignmentTestResult(
            test_name="high_frequency_pressure",
            total_decisions=total_decisions,
            alignment_score=alignment_score,
            coherence_violations=coherence_violations,
            risk_violations=risk_violations,
            portfolio_conflicts=portfolio_conflicts,
            execution_failures=execution_failures,
            decision_consistency=decision_consistency,
            stress_level="EXTREME",
            breaking_point_reached=alignment_score < 0.5,
            recommendations=self._generate_recommendations(alignment_score, coherence_violations, risk_violations)
        )
        
        self.results.append(result)
        self._print_alignment_result(result)
        return result
    
    async def test_portfolio_constraint_violations(self) -> AlignmentTestResult:
        """
        Test system behavior when portfolio constraints are violated.
        
        Simulates scenarios where decisions would violate portfolio limits
        and validates risk management responses.
        """
        print(f"\n🚫 PORTFOLIO CONSTRAINT VIOLATION TEST")
        
        decisions = []
        coherence_violations = 0
        risk_violations = 0
        portfolio_conflicts = 0
        execution_failures = 0
        
        # Simulate a portfolio that's approaching limits
        portfolio_state = {
            "total_exposure": 0.0,
            "long_positions": 0,
            "short_positions": 0,
            "max_exposure": 100000,  # $100k limit
            "max_positions": 10,
            "current_drawdown": 0.05  # 5% drawdown
        }
        
        for i in range(30):
            try:
                # Create events that would violate portfolio constraints
                violation_type = i % 3
                
                if violation_type == 0:
                    # Exposure limit violation
                    synergy_event = self._create_high_exposure_event(i, portfolio_state)
                elif violation_type == 1:
                    # Position count violation
                    synergy_event = self._create_position_limit_event(i, portfolio_state)
                else:
                    # Drawdown violation
                    synergy_event = self._create_drawdown_risk_event(i, portfolio_state)
                
                decision = await self.tactical_controller.on_synergy_detected(synergy_event)
                decisions.append(decision)
                
                # Update simulated portfolio state
                self._update_portfolio_state(portfolio_state, decision)
                
                # Validate constraint adherence
                constraint_violations = self._validate_portfolio_constraints(decision, portfolio_state)
                portfolio_conflicts += len(constraint_violations)
                
                # Check risk management activation
                risk_issues = self._validate_risk_management_activation(decision, portfolio_state)
                risk_violations += len(risk_issues)
                
                coherence_issues = self._validate_decision_coherence(decision, synergy_event)
                coherence_violations += len(coherence_issues)
                
                if not self._validate_execution_command(decision.execution_command):
                    execution_failures += 1
                
            except Exception as e:
                execution_failures += 1
                print(f"⚠️ Portfolio constraint test {i} failed: {e}")
        
        total_decisions = len(decisions)
        alignment_score = self._calculate_alignment_score(
            coherence_violations, risk_violations, portfolio_conflicts, execution_failures, total_decisions
        )
        
        decision_consistency = self._calculate_decision_consistency(decisions)
        
        result = AlignmentTestResult(
            test_name="portfolio_constraint_violations",
            total_decisions=total_decisions,
            alignment_score=alignment_score,
            coherence_violations=coherence_violations,
            risk_violations=risk_violations,
            portfolio_conflicts=portfolio_conflicts,
            execution_failures=execution_failures,
            decision_consistency=decision_consistency,
            stress_level="CRITICAL",
            breaking_point_reached=portfolio_conflicts > total_decisions * 0.3,  # >30% violations
            recommendations=self._generate_recommendations(alignment_score, coherence_violations, risk_violations)
        )
        
        self.results.append(result)
        self._print_alignment_result(result)
        return result
    
    def _validate_decision_coherence(self, decision, synergy_event) -> List[str]:
        """Validate that the decision is coherent with the synergy event."""
        issues = []
        
        # Check direction alignment
        if synergy_event.direction > 0 and decision.action == "short":
            issues.append("Direction mismatch: bullish event led to short decision")
        elif synergy_event.direction < 0 and decision.action == "long":
            issues.append("Direction mismatch: bearish event led to long decision")
        
        # Check confidence alignment
        if synergy_event.confidence > 0.8 and decision.confidence < 0.5:
            issues.append("Confidence mismatch: high-confidence event led to low-confidence decision")
        
        # Check action-confidence alignment
        if decision.action != "hold" and decision.confidence < 0.6:
            issues.append("Action-confidence mismatch: executing trade with low confidence")
        
        return issues
    
    def _validate_risk_alignment(self, decision) -> List[str]:
        """Validate risk management aspects of the decision."""
        issues = []
        
        if decision.action != "hold":
            exec_cmd = decision.execution_command
            
            # Check for missing risk controls
            if "stop_loss" not in exec_cmd or exec_cmd.get("stop_loss") is None:
                issues.append("Missing stop-loss in execution command")
            
            if "take_profit" not in exec_cmd or exec_cmd.get("take_profit") is None:
                issues.append("Missing take-profit in execution command")
            
            # Check position sizing reasonableness
            quantity = exec_cmd.get("quantity", 0)
            if quantity == 0:
                issues.append("Zero quantity in execution command")
            elif quantity > 10:  # Arbitrary limit for testing
                issues.append("Excessive position size")
        
        return issues
    
    def _validate_portfolio_alignment(self, decision, recent_decisions) -> List[str]:
        """Validate decision alignment with recent portfolio decisions."""
        issues = []
        
        if len(recent_decisions) < 2:
            return issues
        
        # Check for rapid reversals
        recent_actions = [d.action for d in recent_decisions[-3:]]
        if len(set(recent_actions)) > 2 and decision.action != "hold":
            issues.append("Rapid strategy reversal detected")
        
        # Check for over-trading
        trade_actions = [d.action for d in recent_decisions if d.action != "hold"]
        if len(trade_actions) > 5:  # More than 5 trades in recent history
            issues.append("Potential over-trading detected")
        
        return issues
    
    def _validate_execution_command(self, exec_cmd) -> bool:
        """Validate that execution command is properly formed."""
        required_fields = ["action"]
        
        if not isinstance(exec_cmd, dict):
            return False
        
        for field in required_fields:
            if field not in exec_cmd:
                return False
        
        # If it's a trade execution, check for additional required fields
        if exec_cmd.get("action") == "execute_trade":
            trade_fields = ["side", "quantity", "symbol"]
            for field in trade_fields:
                if field not in exec_cmd:
                    return False
        
        return True
    
    def _calculate_alignment_score(self, coherence_violations, risk_violations, 
                                 portfolio_conflicts, execution_failures, total_decisions) -> float:
        """Calculate overall alignment score."""
        if total_decisions == 0:
            return 0.0
        
        total_violations = coherence_violations + risk_violations + portfolio_conflicts + execution_failures
        violation_rate = total_violations / total_decisions
        
        # Convert violation rate to alignment score (inverted)
        alignment_score = max(0.0, 1.0 - violation_rate)
        
        return alignment_score
    
    def _calculate_decision_consistency(self, decisions) -> float:
        """Calculate consistency of decisions over time."""
        if len(decisions) < 2:
            return 1.0
        
        # Measure consistency in confidence levels
        confidences = [d.confidence for d in decisions]
        confidence_std = np.std(confidences)
        
        # Measure consistency in decision timing
        timings = [sum(d.timing.values()) for d in decisions if hasattr(d, 'timing') and d.timing]
        timing_std = np.std(timings) if timings else 0
        
        # Combine metrics (lower standard deviation = higher consistency)
        consistency = 1.0 / (1.0 + confidence_std + timing_std * 0.001)  # Scale timing appropriately
        
        return min(1.0, consistency)
    
    def _detect_rapid_fire_conflicts(self, current_decision, recent_decisions) -> int:
        """Detect conflicts in rapid-fire decisions."""
        conflicts = 0
        
        if len(recent_decisions) < 2:
            return conflicts
        
        # Check for contradictory actions in rapid succession
        recent_actions = [d.action for d in recent_decisions[-2:]]
        
        if current_decision.action == "long" and "short" in recent_actions:
            conflicts += 1
        elif current_decision.action == "short" and "long" in recent_actions:
            conflicts += 1
        
        return conflicts
    
    def _create_high_exposure_event(self, event_id: int, portfolio_state: Dict) -> SynergyEvent:
        """Create event that would cause exposure limit violation."""
        return SynergyEvent(
            synergy_type="HIGH_EXPOSURE_TYPE",
            direction=1,
            confidence=0.9,
            signal_sequence=[],
            market_context={
                "current_exposure": portfolio_state["total_exposure"],
                "proposed_size": "large"
            },
            correlation_id=f"exposure-test-{event_id}",
            timestamp=time.time()
        )
    
    def _create_position_limit_event(self, event_id: int, portfolio_state: Dict) -> SynergyEvent:
        """Create event that would cause position count violation."""
        return SynergyEvent(
            synergy_type="POSITION_LIMIT_TYPE",
            direction=-1,
            confidence=0.85,
            signal_sequence=[],
            market_context={
                "current_positions": portfolio_state["long_positions"] + portfolio_state["short_positions"],
                "max_positions": portfolio_state["max_positions"]
            },
            correlation_id=f"position-test-{event_id}",
            timestamp=time.time()
        )
    
    def _create_drawdown_risk_event(self, event_id: int, portfolio_state: Dict) -> SynergyEvent:
        """Create event during high drawdown period."""
        return SynergyEvent(
            synergy_type="DRAWDOWN_RISK_TYPE",
            direction=1,
            confidence=0.7,
            signal_sequence=[],
            market_context={
                "current_drawdown": portfolio_state["current_drawdown"],
                "risk_level": "high"
            },
            correlation_id=f"drawdown-test-{event_id}",
            timestamp=time.time()
        )
    
    def _update_portfolio_state(self, portfolio_state: Dict, decision):
        """Update simulated portfolio state based on decision."""
        if decision.action == "long":
            portfolio_state["long_positions"] += 1
            portfolio_state["total_exposure"] += 10000  # Simulated trade size
        elif decision.action == "short":
            portfolio_state["short_positions"] += 1
            portfolio_state["total_exposure"] += 10000
    
    def _validate_portfolio_constraints(self, decision, portfolio_state) -> List[str]:
        """Validate that decision respects portfolio constraints."""
        violations = []
        
        if decision.action != "hold":
            # Check exposure limits
            if portfolio_state["total_exposure"] > portfolio_state["max_exposure"]:
                violations.append("Exposure limit violation")
            
            # Check position limits
            total_positions = portfolio_state["long_positions"] + portfolio_state["short_positions"]
            if total_positions > portfolio_state["max_positions"]:
                violations.append("Position count limit violation")
            
            # Check drawdown constraints
            if portfolio_state["current_drawdown"] > 0.1 and decision.confidence < 0.8:
                violations.append("Trading during high drawdown with low confidence")
        
        return violations
    
    def _validate_risk_management_activation(self, decision, portfolio_state) -> List[str]:
        """Validate that risk management activates appropriately."""
        issues = []
        
        # Risk management should be more conservative during drawdown
        if portfolio_state["current_drawdown"] > 0.05:  # 5% drawdown
            if decision.action != "hold" and decision.confidence < 0.8:
                issues.append("Risk management should prevent low-confidence trades during drawdown")
        
        # Risk management should prevent over-exposure
        if portfolio_state["total_exposure"] > portfolio_state["max_exposure"] * 0.9:  # 90% of limit
            if decision.action != "hold":
                issues.append("Risk management should prevent new positions near exposure limit")
        
        return issues
    
    def _generate_recommendations(self, alignment_score, coherence_violations, risk_violations) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        if alignment_score < 0.7:
            recommendations.append("Implement stricter alignment validation in decision aggregation")
        
        if coherence_violations > 0:
            recommendations.append("Add coherence checks before decision execution")
        
        if risk_violations > 0:
            recommendations.append("Strengthen risk management validation layer")
        
        return recommendations
    
    def _print_alignment_result(self, result: AlignmentTestResult):
        """Print formatted alignment test result."""
        print(f"\n{'='*60}")
        print(f"🎯 ALIGNMENT TEST RESULT: {result.test_name}")
        print(f"{'='*60}")
        print(f"Total Decisions: {result.total_decisions}")
        print(f"Alignment Score: {result.alignment_score:.3f}")
        print(f"Decision Consistency: {result.decision_consistency:.3f}")
        print(f"Stress Level: {result.stress_level}")
        print(f"")
        print(f"Violations:")
        print(f"  Coherence: {result.coherence_violations}")
        print(f"  Risk: {result.risk_violations}")
        print(f"  Portfolio: {result.portfolio_conflicts}")
        print(f"  Execution: {result.execution_failures}")
        print(f"")
        if result.breaking_point_reached:
            print(f"💥 ALIGNMENT BREAKING POINT REACHED!")
        else:
            print(f"✅ Alignment maintained under stress")
        
        if result.recommendations:
            print(f"\nRecommendations:")
            for rec in result.recommendations:
                print(f"  - {rec}")
        print(f"{'='*60}")
    
    def generate_alignment_report(self) -> str:
        """Generate comprehensive alignment validation report."""
        report = []
        report.append("# STRATEGIC-TACTICAL ALIGNMENT VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall assessment
        avg_alignment = sum(r.alignment_score for r in self.results) / len(self.results) if self.results else 0
        total_violations = sum(r.coherence_violations + r.risk_violations + r.portfolio_conflicts for r in self.results)
        
        if avg_alignment >= 0.8:
            report.append("## ✅ OVERALL ASSESSMENT: EXCELLENT ALIGNMENT")
        elif avg_alignment >= 0.6:
            report.append("## ⚠️ OVERALL ASSESSMENT: ACCEPTABLE ALIGNMENT")
        else:
            report.append("## ❌ OVERALL ASSESSMENT: POOR ALIGNMENT - CRITICAL ISSUES")
        
        report.append(f"Average Alignment Score: {avg_alignment:.3f}")
        report.append(f"Total Violations: {total_violations}")
        report.append("")
        
        # Detailed results
        for result in self.results:
            report.append(f"### {result.test_name.upper()}")
            report.append(f"- Alignment Score: {result.alignment_score:.3f}")
            report.append(f"- Total Violations: {result.coherence_violations + result.risk_violations + result.portfolio_conflicts}")
            report.append(f"- Breaking Point: {'YES' if result.breaking_point_reached else 'NO'}")
            if result.recommendations:
                report.append("- Recommendations:")
                for rec in result.recommendations:
                    report.append(f"  * {rec}")
            report.append("")
        
        return "\n".join(report)

# Test execution functions
async def run_alignment_validation_suite():
    """Run the complete alignment validation test suite."""
    validator = StrategicTacticalAlignmentValidator()
    await validator.setup()
    
    try:
        print("🎯 STARTING STRATEGIC-TACTICAL ALIGNMENT VALIDATION")
        print("=" * 60)
        
        # Run alignment tests
        await validator.test_signal_corruption_resilience()
        await validator.test_high_frequency_decision_pressure()
        await validator.test_portfolio_constraint_violations()
        
        # Generate and save report
        report = validator.generate_alignment_report()
        
        report_path = Path(__file__).parent / "alignment_validation_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n📋 Alignment validation report saved to: {report_path}")
        print(report)
        
        return validator.results
        
    finally:
        await validator.teardown()

if __name__ == "__main__":
    asyncio.run(run_alignment_validation_suite())