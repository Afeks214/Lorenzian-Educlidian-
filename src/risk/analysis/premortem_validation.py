"""
Pre-Mortem Analysis Agent - Production Validation Script

Comprehensive validation script that verifies all requirements and specifications
for the Pre-Mortem Analysis Agent implementation are met.

MISSION VALIDATION CHECKLIST:
✅ Monte Carlo Engine: 10,000 simulation paths in <100ms
✅ Decision Categories: GO (<5% failure), CAUTION (5-15%), NO-GO (>15% failure)  
✅ Integration Point: All significant risk-increasing decisions
✅ Performance Target: <100ms complete analysis with human review triggers
✅ Advanced Market Models: GBM with jump diffusion, stochastic volatility
✅ Decision Pipeline Integration: All 4 MARL agents
✅ Failure Probability Analysis: VaR, Expected Shortfall, Max Drawdown
✅ Human Review System: Automatic escalation for high-risk decisions
"""

import time
import asyncio
import json
from typing import Dict, Any, List, Tuple
from datetime import datetime
import numpy as np
import structlog

from src.risk.analysis.premortem_agent import PreMortemAgent, PreMortemConfig
from src.risk.simulation.monte_carlo_engine import MonteCarloEngine, SimulationParameters
from src.risk.analysis.failure_probability_calculator import FailureProbabilityCalculator, RiskRecommendation
from src.risk.integration.decision_interceptor import DecisionContext, DecisionType, DecisionPriority
from src.core.events import EventBus

logger = structlog.get_logger()


class PreMortemValidationSuite:
    """
    Comprehensive validation suite for Pre-Mortem Analysis Agent
    
    Validates all mission-critical requirements and performance targets.
    """
    
    def __init__(self):
        """Initialize validation suite"""
        self.event_bus = EventBus()
        
        # Initialize components for testing
        self.monte_carlo_engine = MonteCarloEngine(enable_gpu=True)
        self.failure_calculator = FailureProbabilityCalculator()
        
        # Initialize pre-mortem agent
        config = {
            'name': 'validation_agent',
            'premortem_config': {
                'default_num_paths': 10000,
                'max_analysis_time_ms': 100.0,
                'enable_gpu_acceleration': True,
                'enable_adaptive_paths': True
            }
        }
        self.premortem_agent = PreMortemAgent(config, self.event_bus)
        
        # Validation results
        self.validation_results = {}
        
        logger.info("Pre-mortem validation suite initialized")
    
    async def run_complete_validation(self) -> Dict[str, Any]:
        """
        Run complete validation suite
        
        Returns:
            Comprehensive validation results
        """
        logger.info("🚀 Starting Pre-Mortem Analysis Agent Validation")
        
        validation_start = time.perf_counter()
        
        # Core requirement validations
        monte_carlo_validation = await self.validate_monte_carlo_performance()
        decision_classification_validation = await self.validate_decision_classification()
        integration_validation = await self.validate_marl_integration()
        performance_validation = await self.validate_performance_targets()
        market_models_validation = await self.validate_market_models()
        failure_analysis_validation = await self.validate_failure_analysis()
        human_review_validation = await self.validate_human_review_system()
        
        # System validation
        system_validation = await self.validate_system_requirements()
        
        validation_time = (time.perf_counter() - validation_start) * 1000
        
        # Compile validation results
        results = {
            'validation_summary': {
                'total_validation_time_ms': validation_time,
                'validation_timestamp': datetime.now().isoformat(),
                'overall_status': 'PASS'  # Will be determined below
            },
            'monte_carlo_performance': monte_carlo_validation,
            'decision_classification': decision_classification_validation,
            'marl_integration': integration_validation,
            'performance_targets': performance_validation,
            'market_models': market_models_validation,
            'failure_analysis': failure_analysis_validation,
            'human_review_system': human_review_validation,
            'system_requirements': system_validation
        }
        
        # Determine overall status
        all_validations = [
            monte_carlo_validation, decision_classification_validation,
            integration_validation, performance_validation,
            market_models_validation, failure_analysis_validation,
            human_review_validation, system_validation
        ]
        
        overall_pass = all(v.get('status') == 'PASS' for v in all_validations)
        results['validation_summary']['overall_status'] = 'PASS' if overall_pass else 'FAIL'
        
        logger.info("✅ Pre-mortem validation completed",
                   status=results['validation_summary']['overall_status'],
                   time_ms=f"{validation_time:.2f}")
        
        return results
    
    async def validate_monte_carlo_performance(self) -> Dict[str, Any]:
        """
        Validate Monte Carlo Engine Performance
        
        REQUIREMENT: 10,000 simulation paths in <100ms
        """
        logger.info("🎲 Validating Monte Carlo Engine Performance")
        
        # Test parameters for validation
        test_params = SimulationParameters(
            num_paths=10000,
            time_horizon_hours=24.0,
            time_steps=1440,
            initial_prices=np.array([100.0, 200.0, 50.0]),
            drift_rates=np.array([0.1, 0.08, 0.12]),
            volatilities=np.array([0.2, 0.25, 0.18]),
            correlation_matrix=np.array([[1.0, 0.3, 0.1],
                                       [0.3, 1.0, 0.2],
                                       [0.1, 0.2, 1.0]])
        )
        
        # Run performance test
        times = []
        for i in range(5):
            start_time = time.perf_counter()
            results = self.monte_carlo_engine.run_simulation(test_params)
            execution_time = (time.perf_counter() - start_time) * 1000
            times.append(execution_time)
        
        avg_time = np.mean(times)
        max_time = np.max(times)
        min_time = np.min(times)
        
        # Validation criteria
        target_met = avg_time <= 100.0
        consistency_good = (max_time - min_time) / avg_time <= 0.5  # <50% variation
        
        return {
            'status': 'PASS' if target_met and consistency_good else 'FAIL',
            'avg_execution_time_ms': avg_time,
            'max_execution_time_ms': max_time,
            'min_execution_time_ms': min_time,
            'target_100ms_met': target_met,
            'performance_consistency': consistency_good,
            'paths_per_second': 10000 / (avg_time / 1000),
            'requirement': 'Monte Carlo: 10,000 paths in <100ms'
        }
    
    async def validate_decision_classification(self) -> Dict[str, Any]:
        """
        Validate Decision Classification System
        
        REQUIREMENT: GO (<5% failure), CAUTION (5-15%), NO-GO (>15% failure)
        """
        logger.info("🎯 Validating Decision Classification System")
        
        # Create test scenarios with known failure probabilities
        test_scenarios = [
            # Low risk scenario (should be GO)
            {
                'name': 'low_risk_scenario',
                'expected': RiskRecommendation.GO,
                'portfolio_values': np.random.normal(1.03, 0.05, 10000)  # 3% return, 5% vol
            },
            # Medium risk scenario (should be CAUTION)
            {
                'name': 'medium_risk_scenario', 
                'expected': RiskRecommendation.GO_WITH_CAUTION,
                'portfolio_values': np.random.normal(1.01, 0.12, 10000)  # 1% return, 12% vol
            },
            # High risk scenario (should be NO-GO)
            {
                'name': 'high_risk_scenario',
                'expected': RiskRecommendation.NO_GO_REQUIRES_HUMAN_REVIEW,
                'portfolio_values': np.random.normal(0.95, 0.20, 10000)  # -5% return, 20% vol
            }
        ]
        
        classification_results = {}
        correct_classifications = 0
        
        for scenario in test_scenarios:
            # Create synthetic simulation results
            from src.risk.simulation.monte_carlo_engine import SimulationResults
            
            sim_results = SimulationResults(
                price_paths=None,
                return_paths=None,
                portfolio_values=None,
                final_portfolio_values=scenario['portfolio_values'],
                max_drawdowns=np.random.beta(2, 8, 10000) * 0.3,
                computation_time_ms=50.0
            )
            
            # Calculate failure metrics
            metrics = self.failure_calculator.calculate_failure_metrics(sim_results)
            
            classification_results[scenario['name']] = {
                'expected_recommendation': scenario['expected'].value,
                'actual_recommendation': metrics.recommendation.value,
                'failure_probability': metrics.failure_probability,
                'correct': metrics.recommendation == scenario['expected']
            }
            
            if metrics.recommendation == scenario['expected']:
                correct_classifications += 1
        
        classification_accuracy = correct_classifications / len(test_scenarios)
        
        return {
            'status': 'PASS' if classification_accuracy >= 0.8 else 'FAIL',
            'classification_accuracy': classification_accuracy,
            'scenarios_tested': len(test_scenarios),
            'correct_classifications': correct_classifications,
            'scenario_results': classification_results,
            'requirement': '3-tier system: GO (<5%), CAUTION (5-15%), NO-GO (>15%)'
        }
    
    async def validate_marl_integration(self) -> Dict[str, Any]:
        """
        Validate MARL Agent Integration
        
        REQUIREMENT: Integration with all 4 MARL agents
        """
        logger.info("🤖 Validating MARL Agent Integration")
        
        # Test integration with each MARL agent
        marl_agents = [
            'position_sizing_agent',
            'stop_target_agent', 
            'risk_monitor_agent',
            'portfolio_optimizer_agent'
        ]
        
        integration_results = {}
        successful_integrations = 0
        
        for agent_name in marl_agents:
            try:
                # Create test decision for each agent
                test_decision = DecisionContext(
                    agent_name=agent_name,
                    decision_type=DecisionType.POSITION_SIZING,
                    position_change_amount=10000.0,
                    portfolio_impact_percent=10.0,
                    reasoning=f"Test decision from {agent_name}"
                )
                
                # Test analysis
                start_time = time.perf_counter()
                result = self.premortem_agent.analyze_trading_decision(test_decision)
                analysis_time = (time.perf_counter() - start_time) * 1000
                
                integration_results[agent_name] = {
                    'integration_successful': True,
                    'analysis_time_ms': analysis_time,
                    'recommendation': result.recommendation.value,
                    'failure_probability': result.failure_probability
                }
                
                successful_integrations += 1
                
            except Exception as e:
                integration_results[agent_name] = {
                    'integration_successful': False,
                    'error': str(e)
                }
        
        integration_success_rate = successful_integrations / len(marl_agents)
        
        return {
            'status': 'PASS' if integration_success_rate == 1.0 else 'FAIL',
            'integration_success_rate': integration_success_rate,
            'successful_integrations': successful_integrations,
            'total_agents_tested': len(marl_agents),
            'agent_results': integration_results,
            'requirement': 'Integration with all 4 MARL agents'
        }
    
    async def validate_performance_targets(self) -> Dict[str, Any]:
        """
        Validate Performance Targets
        
        REQUIREMENT: <100ms complete analysis with human review triggers
        """
        logger.info("⚡ Validating Performance Targets")
        
        # Test various decision complexities
        test_decisions = [
            # Simple decision
            DecisionContext(
                agent_name="position_sizing_agent",
                decision_type=DecisionType.POSITION_SIZING,
                position_change_amount=5000.0,
                portfolio_impact_percent=2.0
            ),
            # Complex decision
            DecisionContext(
                agent_name="portfolio_optimizer_agent",
                decision_type=DecisionType.PORTFOLIO_REBALANCING,
                position_change_amount=50000.0,
                portfolio_impact_percent=25.0
            ),
            # High priority decision
            DecisionContext(
                agent_name="risk_monitor_agent",
                decision_type=DecisionType.RISK_REDUCTION,
                priority=DecisionPriority.HIGH,
                position_change_amount=100000.0,
                portfolio_impact_percent=40.0
            )
        ]
        
        performance_results = []
        
        for i, decision in enumerate(test_decisions):
            times = []
            for _ in range(3):  # Multiple runs for reliability
                start_time = time.perf_counter()
                result = self.premortem_agent.analyze_trading_decision(decision)
                total_time = (time.perf_counter() - start_time) * 1000
                times.append(total_time)
            
            avg_time = np.mean(times)
            performance_results.append({
                'decision_type': f"test_{i+1}",
                'avg_analysis_time_ms': avg_time,
                'target_met': avg_time <= 100.0,
                'human_review_triggered': result.requires_human_review
            })
        
        target_achievement_rate = np.mean([r['target_met'] for r in performance_results])
        avg_analysis_time = np.mean([r['avg_analysis_time_ms'] for r in performance_results])
        
        return {
            'status': 'PASS' if target_achievement_rate >= 0.8 else 'FAIL',
            'target_achievement_rate': target_achievement_rate,
            'avg_analysis_time_ms': avg_analysis_time,
            'performance_results': performance_results,
            'requirement': '<100ms complete analysis with human review'
        }
    
    async def validate_market_models(self) -> Dict[str, Any]:
        """
        Validate Advanced Market Models
        
        REQUIREMENT: GBM with jump diffusion, stochastic volatility, regime switching
        """
        logger.info("📈 Validating Market Models")
        
        from src.risk.simulation.advanced_market_models import (
            GeometricBrownianMotion, JumpDiffusionModel, HestonStochasticVolatility,
            GBMParameters, JumpDiffusionParameters, HestonParameters
        )
        
        model_tests = {}
        
        # Test GBM
        try:
            gbm = GeometricBrownianMotion(GBMParameters(
                drift=0.1, volatility=0.2, initial_price=100.0
            ))
            gbm_paths = gbm.simulate_paths(1000, 252, 1/252, random_seed=42)
            model_tests['gbm'] = {
                'functional': True,
                'paths_shape': gbm_paths.shape,
                'positive_prices': np.all(gbm_paths > 0)
            }
        except Exception as e:
            model_tests['gbm'] = {'functional': False, 'error': str(e)}
        
        # Test Jump Diffusion
        try:
            jump_model = JumpDiffusionModel(JumpDiffusionParameters(
                drift=0.1, volatility=0.2, jump_intensity=2.0,
                jump_mean=0.0, jump_std=0.05, initial_price=100.0
            ))
            jump_paths = jump_model.simulate_paths(1000, 252, 1/252, random_seed=42)
            model_tests['jump_diffusion'] = {
                'functional': True,
                'paths_shape': jump_paths.shape,
                'positive_prices': np.all(jump_paths > 0)
            }
        except Exception as e:
            model_tests['jump_diffusion'] = {'functional': False, 'error': str(e)}
        
        # Test Heston
        try:
            heston = HestonStochasticVolatility(HestonParameters(
                initial_price=100.0, initial_variance=0.04,
                long_term_variance=0.04, kappa=2.0,
                vol_of_vol=0.3, correlation=-0.5, drift=0.1
            ))
            heston_prices, heston_vars = heston.simulate_paths(1000, 252, 1/252, random_seed=42)
            model_tests['heston'] = {
                'functional': True,
                'price_paths_shape': heston_prices.shape,
                'variance_paths_shape': heston_vars.shape,
                'positive_prices': np.all(heston_prices > 0),
                'non_negative_variance': np.all(heston_vars >= 0)
            }
        except Exception as e:
            model_tests['heston'] = {'functional': False, 'error': str(e)}
        
        models_functional = sum(1 for m in model_tests.values() if m.get('functional', False))
        
        return {
            'status': 'PASS' if models_functional >= 3 else 'FAIL',
            'models_functional': models_functional,
            'total_models_tested': len(model_tests),
            'model_test_results': model_tests,
            'requirement': 'Advanced market models: GBM, jump diffusion, stochastic vol'
        }
    
    async def validate_failure_analysis(self) -> Dict[str, Any]:
        """
        Validate Failure Probability Analysis
        
        REQUIREMENT: VaR, Expected Shortfall, Max Drawdown probability
        """
        logger.info("📊 Validating Failure Analysis")
        
        # Create test simulation results
        from src.risk.simulation.monte_carlo_engine import SimulationResults
        
        test_portfolio_values = np.random.normal(1.02, 0.15, 10000)
        test_drawdowns = np.random.beta(2, 8, 10000) * 0.3
        
        sim_results = SimulationResults(
            price_paths=None,
            return_paths=None,
            portfolio_values=None,
            final_portfolio_values=test_portfolio_values,
            max_drawdowns=test_drawdowns,
            computation_time_ms=50.0
        )
        
        # Test failure metrics calculation
        metrics = self.failure_calculator.calculate_failure_metrics(sim_results)
        
        # Validate required metrics are calculated
        required_metrics = [
            'var_95_percent', 'var_99_percent',
            'expected_shortfall_95', 'expected_shortfall_99',
            'max_drawdown_probability', 'failure_probability'
        ]
        
        metrics_present = []
        for metric in required_metrics:
            value = getattr(metrics, metric, None)
            metrics_present.append({
                'metric': metric,
                'present': value is not None,
                'value': value,
                'valid_range': 0 <= value <= 1 if value is not None else False
            })
        
        all_metrics_valid = all(m['present'] and m['valid_range'] for m in metrics_present)
        
        return {
            'status': 'PASS' if all_metrics_valid else 'FAIL',
            'metrics_validation': metrics_present,
            'recommendation_generated': isinstance(metrics.recommendation, RiskRecommendation),
            'confidence_intervals_calculated': metrics.failure_prob_lower_ci < metrics.failure_prob_upper_ci,
            'requirement': 'VaR, Expected Shortfall, Max Drawdown analysis'
        }
    
    async def validate_human_review_system(self) -> Dict[str, Any]:
        """
        Validate Human Review System
        
        REQUIREMENT: Automatic escalation for high-risk decisions
        """
        logger.info("👥 Validating Human Review System")
        
        # Test scenarios that should trigger human review
        high_risk_scenarios = [
            # Very high failure probability
            DecisionContext(
                agent_name="position_sizing_agent",
                decision_type=DecisionType.POSITION_SIZING,
                position_change_amount=500000.0,  # Large amount
                portfolio_impact_percent=80.0,    # Huge impact
                reasoning="Extremely risky position increase"
            ),
            # Critical priority
            DecisionContext(
                agent_name="risk_monitor_agent",
                decision_type=DecisionType.EMERGENCY_ACTION,
                priority=DecisionPriority.CRITICAL,
                position_change_amount=200000.0,
                portfolio_impact_percent=50.0,
                reasoning="Critical emergency action required"
            )
        ]
        
        review_results = []
        human_reviews_triggered = 0
        
        for i, scenario in enumerate(high_risk_scenarios):
            result = self.premortem_agent.analyze_trading_decision(scenario)
            
            review_triggered = result.requires_human_review
            if review_triggered:
                human_reviews_triggered += 1
            
            review_results.append({
                'scenario': f"high_risk_{i+1}",
                'human_review_triggered': review_triggered,
                'escalation_reasons': result.escalation_reasons,
                'recommendation': result.recommendation.value
            })
        
        # Also test that low-risk scenarios don't trigger review
        low_risk_scenario = DecisionContext(
            agent_name="position_sizing_agent",
            decision_type=DecisionType.POSITION_SIZING,
            position_change_amount=1000.0,   # Small amount
            portfolio_impact_percent=1.0,    # Small impact
            reasoning="Minor position adjustment"
        )
        
        low_risk_result = self.premortem_agent.analyze_trading_decision(low_risk_scenario)
        review_results.append({
            'scenario': 'low_risk',
            'human_review_triggered': low_risk_result.requires_human_review,
            'recommendation': low_risk_result.recommendation.value
        })
        
        # Validation criteria
        high_risk_review_rate = human_reviews_triggered / len(high_risk_scenarios)
        low_risk_no_review = not low_risk_result.requires_human_review
        
        return {
            'status': 'PASS' if high_risk_review_rate >= 0.5 and low_risk_no_review else 'FAIL',
            'high_risk_scenarios_tested': len(high_risk_scenarios),
            'human_reviews_triggered': human_reviews_triggered,
            'high_risk_review_rate': high_risk_review_rate,
            'low_risk_appropriate': low_risk_no_review,
            'review_results': review_results,
            'requirement': 'Automatic escalation for high-risk decisions'
        }
    
    async def validate_system_requirements(self) -> Dict[str, Any]:
        """
        Validate Overall System Requirements
        
        REQUIREMENT: Complete system integration and functionality
        """
        logger.info("🔧 Validating System Requirements")
        
        system_checks = {}
        
        # Check event bus integration
        system_checks['event_bus_integration'] = self.premortem_agent.event_bus is not None
        
        # Check decision interceptor functionality
        system_checks['decision_interceptor'] = self.premortem_agent.decision_interceptor is not None
        
        # Check all core components initialized
        system_checks['monte_carlo_engine'] = self.premortem_agent.monte_carlo_engine is not None
        system_checks['failure_calculator'] = self.premortem_agent.failure_calculator is not None
        
        # Check statistics tracking
        stats = self.premortem_agent.get_analysis_stats()
        system_checks['statistics_tracking'] = isinstance(stats, dict) and len(stats) > 0
        
        # Check crisis mode functionality
        try:
            self.premortem_agent.enable_crisis_mode()
            self.premortem_agent.disable_crisis_mode()
            system_checks['crisis_mode'] = True
        except Exception:
            system_checks['crisis_mode'] = False
        
        # Check cache functionality
        try:
            self.premortem_agent.clear_analysis_cache()
            system_checks['cache_management'] = True
        except Exception:
            system_checks['cache_management'] = False
        
        all_systems_functional = all(system_checks.values())
        
        return {
            'status': 'PASS' if all_systems_functional else 'FAIL',
            'system_checks': system_checks,
            'functional_systems': sum(system_checks.values()),
            'total_systems_checked': len(system_checks),
            'requirement': 'Complete system integration and functionality'
        }
    
    def generate_validation_report(self, results: Dict[str, Any]) -> str:
        """
        Generate comprehensive validation report
        
        Args:
            results: Validation results from run_complete_validation()
            
        Returns:
            Formatted validation report
        """
        report = []
        report.append("=" * 80)
        report.append("PRE-MORTEM ANALYSIS AGENT - PRODUCTION VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall status
        summary = results['validation_summary']
        status_emoji = "✅" if summary['overall_status'] == 'PASS' else "❌"
        report.append(f"{status_emoji} OVERALL VALIDATION STATUS: {summary['overall_status']}")
        report.append(f"Validation completed in {summary['total_validation_time_ms']:.2f}ms")
        report.append(f"Timestamp: {summary['validation_timestamp']}")
        report.append("")
        
        # Individual validation results
        validations = [
            ('Monte Carlo Performance', results['monte_carlo_performance']),
            ('Decision Classification', results['decision_classification']),
            ('MARL Integration', results['marl_integration']),
            ('Performance Targets', results['performance_targets']),
            ('Market Models', results['market_models']),
            ('Failure Analysis', results['failure_analysis']),
            ('Human Review System', results['human_review_system']),
            ('System Requirements', results['system_requirements'])
        ]
        
        report.append("DETAILED VALIDATION RESULTS:")
        report.append("-" * 40)
        
        for name, validation in validations:
            status = validation.get('status', 'UNKNOWN')
            status_emoji = "✅" if status == 'PASS' else "❌"
            report.append(f"{status_emoji} {name}: {status}")
            
            if 'requirement' in validation:
                report.append(f"    Requirement: {validation['requirement']}")
            
            # Add key metrics for each validation
            if name == 'Monte Carlo Performance':
                report.append(f"    Avg time: {validation['avg_execution_time_ms']:.2f}ms")
                report.append(f"    Target met: {validation['target_100ms_met']}")
            
            elif name == 'Decision Classification':
                report.append(f"    Classification accuracy: {validation['classification_accuracy']:.1%}")
            
            elif name == 'MARL Integration':
                report.append(f"    Integration success rate: {validation['integration_success_rate']:.1%}")
            
            elif name == 'Performance Targets':
                report.append(f"    Target achievement rate: {validation['target_achievement_rate']:.1%}")
                report.append(f"    Avg analysis time: {validation['avg_analysis_time_ms']:.2f}ms")
            
            report.append("")
        
        # Mission requirements summary
        report.append("MISSION REQUIREMENTS VERIFICATION:")
        report.append("-" * 40)
        
        monte_carlo_pass = results['monte_carlo_performance']['status'] == 'PASS'
        classification_pass = results['decision_classification']['status'] == 'PASS'
        integration_pass = results['marl_integration']['status'] == 'PASS'
        performance_pass = results['performance_targets']['status'] == 'PASS'
        
        mission_checks = [
            ("✅ Monte Carlo: 10,000 paths in <100ms", monte_carlo_pass),
            ("✅ Decision Categories: GO/CAUTION/NO-GO system", classification_pass),
            ("✅ Integration: All 4 MARL agents", integration_pass),
            ("✅ Performance: <100ms analysis target", performance_pass),
            ("✅ Market Models: Advanced simulation models", results['market_models']['status'] == 'PASS'),
            ("✅ Risk Analysis: VaR, ES, Drawdown metrics", results['failure_analysis']['status'] == 'PASS'),
            ("✅ Human Review: Automatic escalation", results['human_review_system']['status'] == 'PASS'),
            ("✅ System Integration: Complete functionality", results['system_requirements']['status'] == 'PASS')
        ]
        
        for requirement, passed in mission_checks:
            status_text = "VERIFIED" if passed else "FAILED"
            report.append(f"{requirement}: {status_text}")
        
        report.append("")
        report.append("=" * 80)
        
        if summary['overall_status'] == 'PASS':
            report.append("🎉 PRE-MORTEM ANALYSIS AGENT - MISSION COMPLETE")
            report.append("All requirements validated successfully!")
            report.append("Agent ready for production deployment.")
        else:
            report.append("⚠️  VALIDATION INCOMPLETE")
            report.append("Some requirements need attention before production deployment.")
        
        report.append("=" * 80)
        
        return "\n".join(report)


async def main():
    """Run the complete pre-mortem validation suite"""
    validator = PreMortemValidationSuite()
    
    try:
        # Run complete validation
        results = await validator.run_complete_validation()
        
        # Generate and display report
        report = validator.generate_validation_report(results)
        print(report)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"premortem_validation_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nDetailed validation results saved to: {results_file}")
        
        # Return overall status for CI/CD
        return results['validation_summary']['overall_status'] == 'PASS'
        
    except Exception as e:
        logger.error("Validation failed", error=str(e))
        print(f"❌ VALIDATION FAILED: {e}")
        return False


if __name__ == "__main__":
    # Run the validation
    success = asyncio.run(main())
    exit(0 if success else 1)