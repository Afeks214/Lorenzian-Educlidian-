"""
Pre-Mortem Analysis Agent Integration Demo

Demonstrates the complete integration of the Pre-Mortem Analysis Agent with
existing MARL trading agents, showing the decision interception workflow,
risk analysis, and recommendation system in action.

This demo shows:
1. Integration with Position Sizing Agent (π₁)
2. Integration with Stop/Target Agent (π₂) 
3. Integration with Risk Monitor Agent (π₃)
4. Integration with Portfolio Optimizer Agent (π₄)
5. Complete decision workflow with pre-mortem analysis
6. Human review triggers and escalation procedures
7. Performance monitoring and reporting
"""

import asyncio
import time
import json
from typing import Dict, Any, List
from datetime import datetime, timedelta
import numpy as np
import structlog

from src.risk.analysis.premortem_agent import PreMortemAgent, PreMortemConfig
from src.risk.integration.decision_interceptor import (
    DecisionContext, DecisionType, DecisionPriority
)
from src.risk.agents.base_risk_agent import RiskState
from src.core.events import EventBus, EventType

logger = structlog.get_logger()


class PreMortemIntegrationDemo:
    """
    Comprehensive integration demo for Pre-Mortem Analysis Agent
    
    Demonstrates real-world integration scenarios with MARL trading agents
    and shows the complete decision analysis workflow.
    """
    
    def __init__(self):
        """Initialize integration demo"""
        self.event_bus = EventBus()
        
        # Initialize pre-mortem agent with production-like config
        premortem_config = {
            'name': 'premortem_demo_agent',
            'premortem_config': {
                'default_num_paths': 10000,
                'max_analysis_time_ms': 100.0,
                'simulation_horizon_hours': 24.0,
                'enable_gpu_acceleration': True,
                'enable_adaptive_paths': True,
                'enable_regime_detection': True
            }
        }
        
        self.premortem_agent = PreMortemAgent(premortem_config, self.event_bus)
        
        # Demo state
        self.demo_results = []
        self.portfolio_value = 1000000.0  # $1M portfolio
        self.positions = {
            'EURUSD': 100000.0,
            'GBPUSD': 75000.0,
            'USDJPY': 50000.0,
            'BTCUSD': 25000.0
        }
        
        logger.info("Pre-mortem integration demo initialized",
                   portfolio_value=self.portfolio_value,
                   positions=len(self.positions))
    
    async def run_complete_demo(self) -> Dict[str, Any]:
        """
        Run complete integration demo with all MARL agents
        
        Returns:
            Comprehensive demo results
        """
        logger.info("🚀 Starting Pre-Mortem Integration Demo")
        
        demo_start = time.perf_counter()
        
        # 1. Position Sizing Agent Integration
        position_results = await self.demo_position_sizing_integration()
        
        # 2. Stop/Target Agent Integration  
        stop_target_results = await self.demo_stop_target_integration()
        
        # 3. Risk Monitor Agent Integration
        risk_monitor_results = await self.demo_risk_monitor_integration()
        
        # 4. Portfolio Optimizer Integration
        portfolio_opt_results = await self.demo_portfolio_optimizer_integration()
        
        # 5. Emergency Scenario Testing
        emergency_results = await self.demo_emergency_scenarios()
        
        # 6. Performance Analysis
        performance_results = await self.demo_performance_analysis()
        
        demo_time = (time.perf_counter() - demo_start) * 1000
        
        # Compile results
        results = {
            'demo_summary': {
                'total_demo_time_ms': demo_time,
                'total_analyses_performed': len(self.demo_results),
                'demo_timestamp': datetime.now().isoformat()
            },
            'position_sizing_integration': position_results,
            'stop_target_integration': stop_target_results,
            'risk_monitor_integration': risk_monitor_results,
            'portfolio_optimizer_integration': portfolio_opt_results,
            'emergency_scenarios': emergency_results,
            'performance_analysis': performance_results,
            'agent_statistics': self.premortem_agent.get_analysis_stats()
        }
        
        logger.info("✅ Pre-mortem integration demo completed",
                   total_time_ms=f"{demo_time:.2f}",
                   analyses_performed=len(self.demo_results))
        
        return results
    
    async def demo_position_sizing_integration(self) -> Dict[str, Any]:
        """
        Demo integration with Position Sizing Agent (π₁)
        
        Tests various position sizing scenarios:
        - Normal position increases/decreases
        - Large position changes requiring review
        - Kelly Criterion-based sizing decisions
        """
        logger.info("🔢 Testing Position Sizing Agent Integration")
        
        scenarios = [
            # Scenario 1: Small position increase (should be GO)
            {
                'name': 'small_position_increase',
                'context': DecisionContext(
                    agent_name="position_sizing_agent",
                    decision_type=DecisionType.POSITION_SIZING,
                    current_position_size=100000.0,
                    proposed_position_size=110000.0,
                    position_change_amount=10000.0,
                    position_change_percent=10.0,
                    portfolio_impact_percent=1.0,
                    symbol="EURUSD",
                    reasoning="Kelly Criterion suggests 10% position increase",
                    confidence=0.8
                )
            },
            
            # Scenario 2: Large position increase (should be CAUTION or NO-GO)
            {
                'name': 'large_position_increase',
                'context': DecisionContext(
                    agent_name="position_sizing_agent",
                    decision_type=DecisionType.POSITION_SIZING,
                    current_position_size=100000.0,
                    proposed_position_size=200000.0,
                    position_change_amount=100000.0,
                    position_change_percent=100.0,
                    portfolio_impact_percent=10.0,
                    symbol="BTCUSD",
                    reasoning="Strong technical breakout signal",
                    confidence=0.9
                )
            },
            
            # Scenario 3: Risk reduction (should be GO)
            {
                'name': 'risk_reduction',
                'context': DecisionContext(
                    agent_name="position_sizing_agent",
                    decision_type=DecisionType.POSITION_SIZING,
                    current_position_size=150000.0,
                    proposed_position_size=100000.0,
                    position_change_amount=-50000.0,
                    position_change_percent=-33.3,
                    portfolio_impact_percent=5.0,
                    symbol="GBPUSD",
                    reasoning="VaR breach - reducing position size",
                    confidence=0.95
                )
            },
            
            # Scenario 4: Massive position change (should require human review)
            {
                'name': 'massive_position_change',
                'context': DecisionContext(
                    agent_name="position_sizing_agent",
                    decision_type=DecisionType.POSITION_SIZING,
                    current_position_size=200000.0,
                    proposed_position_size=500000.0,
                    position_change_amount=300000.0,
                    position_change_percent=150.0,
                    portfolio_impact_percent=30.0,
                    symbol="USDJPY",
                    reasoning="Major fundamental shift detected",
                    confidence=0.7
                )
            }
        ]
        
        results = {}
        
        for scenario in scenarios:
            logger.info(f"Testing scenario: {scenario['name']}")
            
            start_time = time.perf_counter()
            analysis_result = self.premortem_agent.analyze_trading_decision(scenario['context'])
            analysis_time = (time.perf_counter() - start_time) * 1000
            
            self.demo_results.append(analysis_result)
            
            results[scenario['name']] = {
                'recommendation': analysis_result.recommendation.value,
                'failure_probability': analysis_result.failure_probability,
                'confidence': analysis_result.confidence,
                'analysis_time_ms': analysis_time,
                'requires_human_review': analysis_result.requires_human_review,
                'risk_factors': analysis_result.primary_risk_factors,
                'mitigation_suggestions': analysis_result.risk_mitigation_suggestions
            }
            
            logger.info(f"Scenario result: {analysis_result.recommendation.value}",
                       failure_prob=f"{analysis_result.failure_probability:.3f}",
                       analysis_time=f"{analysis_time:.2f}ms")
        
        return {
            'total_scenarios': len(scenarios),
            'scenarios': results,
            'avg_analysis_time_ms': np.mean([r['analysis_time_ms'] for r in results.values()])
        }
    
    async def demo_stop_target_integration(self) -> Dict[str, Any]:
        """
        Demo integration with Stop/Target Agent (π₂)
        
        Tests stop-loss and take-profit adjustments:
        - Tightening stops during high volatility
        - Moving targets based on momentum
        - Emergency stop-loss triggers
        """
        logger.info("🎯 Testing Stop/Target Agent Integration")
        
        scenarios = [
            # Scenario 1: Tightening stop-loss
            {
                'name': 'tighten_stop_loss',
                'context': DecisionContext(
                    agent_name="stop_target_agent",
                    decision_type=DecisionType.STOP_TARGET_ADJUSTMENT,
                    current_position_size=100000.0,
                    symbol="EURUSD",
                    current_price=1.1000,
                    portfolio_impact_percent=8.0,
                    reasoning="Tightening stop from 1.0950 to 1.0970 due to volatility increase",
                    confidence=0.85,
                    metadata={
                        'current_stop': 1.0950,
                        'new_stop': 1.0970,
                        'current_target': 1.1150,
                        'volatility_increase': 0.15
                    }
                )
            },
            
            # Scenario 2: Aggressive target adjustment
            {
                'name': 'aggressive_target_move',
                'context': DecisionContext(
                    agent_name="stop_target_agent",
                    decision_type=DecisionType.STOP_TARGET_ADJUSTMENT,
                    current_position_size=75000.0,
                    symbol="BTCUSD",
                    current_price=50000.0,
                    portfolio_impact_percent=15.0,
                    reasoning="Moving target from 52000 to 55000 on momentum breakout",
                    confidence=0.7,
                    metadata={
                        'current_stop': 48000.0,
                        'current_target': 52000.0,
                        'new_target': 55000.0,
                        'momentum_strength': 0.9
                    }
                )
            }
        ]
        
        results = {}
        
        for scenario in scenarios:
            analysis_result = self.premortem_agent.analyze_trading_decision(scenario['context'])
            self.demo_results.append(analysis_result)
            
            results[scenario['name']] = {
                'recommendation': analysis_result.recommendation.value,
                'failure_probability': analysis_result.failure_probability,
                'requires_human_review': analysis_result.requires_human_review
            }
        
        return {
            'total_scenarios': len(scenarios),
            'scenarios': results
        }
    
    async def demo_risk_monitor_integration(self) -> Dict[str, Any]:
        """
        Demo integration with Risk Monitor Agent (π₃)
        
        Tests risk monitoring and emergency actions:
        - VaR breach responses
        - Correlation spike handling
        - Portfolio-wide risk reduction
        """
        logger.info("⚠️ Testing Risk Monitor Agent Integration")
        
        # Simulate high-risk market state
        high_risk_state = RiskState(
            account_equity_normalized=0.92,  # 8% drawdown
            open_positions_count=6,
            volatility_regime=0.85,          # High volatility
            correlation_risk=0.75,           # High correlation
            var_estimate_5pct=0.08,         # 8% VaR (high)
            current_drawdown_pct=0.08,
            margin_usage_pct=0.80,
            time_of_day_risk=0.6,
            market_stress_level=0.8,         # High stress
            liquidity_conditions=0.4         # Poor liquidity
        )
        
        scenarios = [
            # Scenario 1: VaR breach response
            {
                'name': 'var_breach_response',
                'context': DecisionContext(
                    agent_name="risk_monitor_agent",
                    decision_type=DecisionType.RISK_REDUCTION,
                    priority=DecisionPriority.HIGH,
                    portfolio_impact_percent=25.0,
                    reasoning="VaR breach detected - reducing all positions by 25%",
                    confidence=0.9,
                    current_risk_state=high_risk_state,
                    metadata={
                        'var_limit': 0.05,
                        'current_var': 0.08,
                        'breach_severity': 'HIGH'
                    }
                )
            },
            
            # Scenario 2: Emergency portfolio closure
            {
                'name': 'emergency_closure',
                'context': DecisionContext(
                    agent_name="risk_monitor_agent",
                    decision_type=DecisionType.EMERGENCY_ACTION,
                    priority=DecisionPriority.EMERGENCY,
                    portfolio_impact_percent=100.0,
                    reasoning="Critical market stress - emergency portfolio closure",
                    confidence=0.95,
                    current_risk_state=high_risk_state,
                    metadata={
                        'trigger': 'market_stress_critical',
                        'stress_level': 0.95
                    }
                )
            }
        ]
        
        results = {}
        
        for scenario in scenarios:
            # Test risk action calculation first
            action, confidence = self.premortem_agent.calculate_risk_action(
                scenario['context'].current_risk_state
            )
            
            # Then test decision analysis
            analysis_result = self.premortem_agent.analyze_trading_decision(scenario['context'])
            self.demo_results.append(analysis_result)
            
            results[scenario['name']] = {
                'risk_action': action,
                'risk_confidence': confidence,
                'recommendation': analysis_result.recommendation.value,
                'failure_probability': analysis_result.failure_probability,
                'emergency_handled': scenario['context'].priority == DecisionPriority.EMERGENCY
            }
        
        return {
            'total_scenarios': len(scenarios),
            'scenarios': results,
            'risk_constraints_validated': self.premortem_agent.validate_risk_constraints(high_risk_state)
        }
    
    async def demo_portfolio_optimizer_integration(self) -> Dict[str, Any]:
        """
        Demo integration with Portfolio Optimizer Agent (π₄)
        
        Tests portfolio rebalancing decisions:
        - Correlation-based rebalancing
        - Risk parity adjustments
        - Sector rotation decisions
        """
        logger.info("📊 Testing Portfolio Optimizer Integration")
        
        scenarios = [
            # Scenario 1: Correlation-based rebalancing
            {
                'name': 'correlation_rebalancing',
                'context': DecisionContext(
                    agent_name="portfolio_optimizer_agent",
                    decision_type=DecisionType.PORTFOLIO_REBALANCING,
                    portfolio_impact_percent=40.0,
                    reasoning="Rebalancing due to correlation spike - reducing FX exposure",
                    confidence=0.8,
                    metadata={
                        'correlation_spike': True,
                        'avg_correlation': 0.85,
                        'target_correlation': 0.60,
                        'rebalancing_type': 'correlation_driven'
                    }
                )
            },
            
            # Scenario 2: Risk parity adjustment
            {
                'name': 'risk_parity_adjustment',
                'context': DecisionContext(
                    agent_name="portfolio_optimizer_agent",
                    decision_type=DecisionType.PORTFOLIO_REBALANCING,
                    portfolio_impact_percent=30.0,
                    reasoning="Risk parity optimization - equalizing risk contributions",
                    confidence=0.75,
                    metadata={
                        'optimization_type': 'risk_parity',
                        'current_risk_concentration': 0.7,
                        'target_risk_concentration': 0.4
                    }
                )
            }
        ]
        
        results = {}
        
        for scenario in scenarios:
            analysis_result = self.premortem_agent.analyze_trading_decision(scenario['context'])
            self.demo_results.append(analysis_result)
            
            results[scenario['name']] = {
                'recommendation': analysis_result.recommendation.value,
                'failure_probability': analysis_result.failure_probability,
                'risk_factors': analysis_result.primary_risk_factors[:3],  # Top 3
                'portfolio_impact_approved': analysis_result.recommendation.value != 'NO_GO'
            }
        
        return {
            'total_scenarios': len(scenarios),
            'scenarios': results
        }
    
    async def demo_emergency_scenarios(self) -> Dict[str, Any]:
        """
        Demo emergency scenario handling
        
        Tests crisis mode and emergency decision processing:
        - Crisis mode activation/deactivation  
        - Emergency bypass functionality
        - High-priority decision routing
        """
        logger.info("🚨 Testing Emergency Scenarios")
        
        # Test crisis mode
        self.premortem_agent.enable_crisis_mode()
        
        emergency_decision = DecisionContext(
            agent_name="risk_monitor_agent",
            decision_type=DecisionType.EMERGENCY_ACTION,
            priority=DecisionPriority.EMERGENCY,
            portfolio_impact_percent=50.0,
            reasoning="Flash crash detected - emergency hedging required",
            confidence=0.95
        )
        
        start_time = time.perf_counter()
        emergency_result = self.premortem_agent.analyze_trading_decision(emergency_decision)
        emergency_time = (time.perf_counter() - start_time) * 1000
        
        self.demo_results.append(emergency_result)
        
        # Disable crisis mode
        self.premortem_agent.disable_crisis_mode()
        
        # Test normal high-priority decision
        high_priority_decision = DecisionContext(
            agent_name="position_sizing_agent",
            decision_type=DecisionType.POSITION_SIZING,
            priority=DecisionPriority.CRITICAL,
            portfolio_impact_percent=20.0,
            reasoning="Critical position adjustment required",
            confidence=0.8
        )
        
        critical_result = self.premortem_agent.analyze_trading_decision(high_priority_decision)
        self.demo_results.append(critical_result)
        
        return {
            'emergency_analysis': {
                'recommendation': emergency_result.recommendation.value,
                'analysis_time_ms': emergency_time,
                'crisis_mode_effective': emergency_time < 100  # Fast processing
            },
            'critical_priority_analysis': {
                'recommendation': critical_result.recommendation.value,
                'requires_human_review': critical_result.requires_human_review
            },
            'crisis_mode_tested': True
        }
    
    async def demo_performance_analysis(self) -> Dict[str, Any]:
        """
        Demo performance analysis and reporting
        
        Analyzes overall system performance:
        - Analysis time statistics
        - Recommendation distribution
        - Human review rates
        - Performance target achievement
        """
        logger.info("📈 Analyzing Demo Performance")
        
        # Get comprehensive stats
        agent_stats = self.premortem_agent.get_analysis_stats()
        
        # Calculate demo-specific metrics
        analysis_times = [r.total_analysis_time_ms for r in self.demo_results]
        recommendations = [r.recommendation.value for r in self.demo_results]
        human_reviews = [r.requires_human_review for r in self.demo_results]
        
        recommendation_dist = {}
        for rec in recommendations:
            recommendation_dist[rec] = recommendation_dist.get(rec, 0) + 1
        
        performance_analysis = {
            'timing_statistics': {
                'avg_analysis_time_ms': np.mean(analysis_times),
                'max_analysis_time_ms': np.max(analysis_times),
                'min_analysis_time_ms': np.min(analysis_times),
                'target_100ms_achievement_rate': np.mean([t <= 100 for t in analysis_times]),
                'target_200ms_achievement_rate': np.mean([t <= 200 for t in analysis_times])
            },
            'recommendation_distribution': recommendation_dist,
            'human_review_statistics': {
                'human_review_rate': np.mean(human_reviews),
                'total_human_reviews': sum(human_reviews),
                'automatic_approvals': len([r for r in recommendations if r == 'GO'])
            },
            'agent_statistics': agent_stats,
            'component_performance': {
                'monte_carlo_stats': self.premortem_agent.monte_carlo_engine.get_performance_stats(),
                'failure_calc_stats': self.premortem_agent.failure_calculator.get_performance_stats()
            }
        }
        
        return performance_analysis
    
    def generate_demo_report(self, results: Dict[str, Any]) -> str:
        """
        Generate comprehensive demo report
        
        Args:
            results: Demo results from run_complete_demo()
            
        Returns:
            Formatted demo report string
        """
        report = []
        report.append("=" * 80)
        report.append("PRE-MORTEM ANALYSIS AGENT - INTEGRATION DEMO REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        summary = results['demo_summary']
        report.append(f"Demo completed in {summary['total_demo_time_ms']:.2f}ms")
        report.append(f"Total analyses performed: {summary['total_analyses_performed']}")
        report.append(f"Demo timestamp: {summary['demo_timestamp']}")
        report.append("")
        
        # Performance metrics
        perf = results['performance_analysis']['timing_statistics']
        report.append("PERFORMANCE METRICS:")
        report.append(f"  Average analysis time: {perf['avg_analysis_time_ms']:.2f}ms")
        report.append(f"  100ms target achievement: {perf['target_100ms_achievement_rate']:.1%}")
        report.append(f"  200ms target achievement: {perf['target_200ms_achievement_rate']:.1%}")
        report.append("")
        
        # Recommendation distribution
        rec_dist = results['performance_analysis']['recommendation_distribution']
        report.append("RECOMMENDATION DISTRIBUTION:")
        for rec, count in rec_dist.items():
            report.append(f"  {rec}: {count}")
        report.append("")
        
        # Integration results
        report.append("INTEGRATION TEST RESULTS:")
        
        # Position sizing
        pos_results = results['position_sizing_integration']
        report.append(f"  Position Sizing Agent: {pos_results['total_scenarios']} scenarios")
        report.append(f"    Avg analysis time: {pos_results['avg_analysis_time_ms']:.2f}ms")
        
        # Risk monitor
        risk_results = results['risk_monitor_integration']
        report.append(f"  Risk Monitor Agent: {risk_results['total_scenarios']} scenarios")
        report.append(f"    Risk constraints valid: {risk_results['risk_constraints_validated']}")
        
        # Emergency handling
        emergency_results = results['emergency_scenarios']
        report.append(f"  Emergency Scenarios: Tested")
        report.append(f"    Crisis mode effective: {emergency_results['emergency_analysis']['crisis_mode_effective']}")
        
        report.append("")
        report.append("=" * 80)
        report.append("DEMO COMPLETED SUCCESSFULLY")
        report.append("=" * 80)
        
        return "\n".join(report)


async def main():
    """Run the complete pre-mortem integration demo"""
    demo = PreMortemIntegrationDemo()
    
    try:
        # Run complete demo
        results = await demo.run_complete_demo()
        
        # Generate and display report
        report = demo.generate_demo_report(results)
        print(report)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"premortem_demo_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        logger.error("Demo failed", error=str(e))
        raise


if __name__ == "__main__":
    # Run the demo
    results = asyncio.run(main())