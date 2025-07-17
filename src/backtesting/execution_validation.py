"""
Execution Validation and Backtest-Live Alignment
===============================================

AGENT 4 - REALISTIC EXECUTION ENGINE INTEGRATION

This module provides comprehensive validation to ensure that realistic
execution in backtesting aligns with live trading conditions, eliminating
backtest-live divergence.

Validation Components:
- Partial fill scenario testing
- Realistic execution timing validation
- Market condition stress testing
- Cost model accuracy validation
- Performance divergence analysis

Author: AGENT 4 - Realistic Execution Engine Integration
Date: 2025-07-17
Mission: Validate backtest-live execution alignment
"""

import pandas as pd
import numpy as np
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
import json
from pathlib import Path

# Import execution components
from backtesting.realistic_execution_integration import (
    RealisticBacktestExecutionHandler,
    BacktestExecutionConfig
)
from backtesting.dynamic_execution_costs import (
    ComprehensiveCostModel,
    create_nq_futures_cost_model,
    InstrumentType
)
from execution.realistic_execution_engine import (
    RealisticExecutionEngine,
    ExecutionOrder,
    OrderSide,
    OrderType,
    MarketConditions
)

logger = logging.getLogger(__name__)


@dataclass
class PartialFillScenario:
    """Scenario for testing partial fill conditions"""
    name: str
    description: str
    order_size: int
    market_liquidity: float  # 0.0 = illiquid, 1.0 = highly liquid
    volatility_level: float  # 0.0 = low vol, 1.0 = high vol
    time_of_day: int  # Hour of day
    expected_fill_rate: float  # Expected fill percentage
    max_fill_time_ms: float  # Maximum time to fill


@dataclass
class ExecutionTimingScenario:
    """Scenario for testing execution timing"""
    name: str
    description: str
    order_type: str
    market_conditions: str  # 'normal', 'stressed', 'illiquid'
    expected_latency_range: Tuple[float, float]  # Min, max latency in ms
    timing_tolerance_ms: float  # Acceptable timing variance


@dataclass
class ValidationResult:
    """Result of validation test"""
    test_name: str
    passed: bool
    score: float  # 0-100
    details: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class PartialFillTesting:
    """
    Testing framework for partial fill scenarios
    """
    
    def __init__(self, execution_handler: RealisticBacktestExecutionHandler):
        self.execution_handler = execution_handler
        self.test_scenarios = self._create_partial_fill_scenarios()
        self.test_results = []
    
    def _create_partial_fill_scenarios(self) -> List[PartialFillScenario]:
        """Create comprehensive partial fill test scenarios"""
        scenarios = [
            PartialFillScenario(
                name="large_order_normal_market",
                description="Large order in normal market conditions",
                order_size=50,
                market_liquidity=0.8,
                volatility_level=0.3,
                time_of_day=14,
                expected_fill_rate=0.85,
                max_fill_time_ms=200
            ),
            PartialFillScenario(
                name="large_order_illiquid_market",
                description="Large order in illiquid market conditions",
                order_size=50,
                market_liquidity=0.3,
                volatility_level=0.6,
                time_of_day=22,
                expected_fill_rate=0.60,
                max_fill_time_ms=1000
            ),
            PartialFillScenario(
                name="medium_order_high_volatility",
                description="Medium order during high volatility",
                order_size=20,
                market_liquidity=0.7,
                volatility_level=0.8,
                time_of_day=10,
                expected_fill_rate=0.75,
                max_fill_time_ms=300
            ),
            PartialFillScenario(
                name="small_order_normal_conditions",
                description="Small order in normal conditions",
                order_size=5,
                market_liquidity=0.9,
                volatility_level=0.2,
                time_of_day=14,
                expected_fill_rate=1.0,
                max_fill_time_ms=100
            ),
            PartialFillScenario(
                name="overnight_order",
                description="Order during overnight session",
                order_size=10,
                market_liquidity=0.4,
                volatility_level=0.4,
                time_of_day=2,
                expected_fill_rate=0.70,
                max_fill_time_ms=500
            )
        ]
        
        return scenarios
    
    async def run_partial_fill_tests(self) -> List[ValidationResult]:
        """Run comprehensive partial fill tests"""
        results = []
        
        for scenario in self.test_scenarios:
            try:
                result = await self._test_partial_fill_scenario(scenario)
                results.append(result)
                logger.info(f"Partial fill test completed: {scenario.name} - {'PASSED' if result.passed else 'FAILED'}")
            except Exception as e:
                logger.error(f"Partial fill test failed: {scenario.name} - {e}")
                results.append(ValidationResult(
                    test_name=scenario.name,
                    passed=False,
                    score=0,
                    details={'error': str(e)},
                    recommendations=['Fix partial fill test implementation']
                ))
        
        self.test_results = results
        return results
    
    async def _test_partial_fill_scenario(self, scenario: PartialFillScenario) -> ValidationResult:
        """Test a specific partial fill scenario"""
        # Create market conditions for scenario
        market_conditions = self._create_scenario_market_conditions(scenario)
        
        # Create test trade
        trade_data = {
            'timestamp': datetime.now().replace(hour=scenario.time_of_day),
            'symbol': 'NQ',
            'signal': 1,
            'size': scenario.order_size,
            'price': 15000.0,
            'type': 'market'
        }
        
        # Create market data
        market_data = self._create_scenario_market_data(scenario)
        
        # Execute trade with realistic conditions
        start_time = datetime.now()
        execution_result = await self.execution_handler.execute_backtest_trade(
            trade_data, market_data, {}
        )
        end_time = datetime.now()
        
        # Analyze results
        execution_time_ms = (end_time - start_time).total_seconds() * 1000
        
        if execution_result['success']:
            fill_rate = execution_result['fill_quantity'] / scenario.order_size
            timing_within_tolerance = execution_time_ms <= scenario.max_fill_time_ms
            fill_rate_acceptable = abs(fill_rate - scenario.expected_fill_rate) <= 0.2
            
            # Calculate score
            timing_score = 100 if timing_within_tolerance else 50
            fill_score = max(0, 100 - abs(fill_rate - scenario.expected_fill_rate) * 500)
            overall_score = (timing_score + fill_score) / 2
            
            test_passed = timing_within_tolerance and fill_rate_acceptable
            
            details = {
                'expected_fill_rate': scenario.expected_fill_rate,
                'actual_fill_rate': fill_rate,
                'execution_time_ms': execution_time_ms,
                'max_time_ms': scenario.max_fill_time_ms,
                'timing_acceptable': timing_within_tolerance,
                'fill_rate_acceptable': fill_rate_acceptable,
                'execution_quality': execution_result['execution_quality']
            }
            
            recommendations = []
            if not timing_within_tolerance:
                recommendations.append("Execution timing exceeds acceptable range")
            if not fill_rate_acceptable:
                recommendations.append("Fill rate deviates significantly from expected")
                
        else:
            test_passed = False
            overall_score = 0
            details = {
                'error': execution_result.get('error', 'Unknown execution error'),
                'execution_time_ms': execution_time_ms
            }
            recommendations = ['Execution failed - investigate error handling']
        
        return ValidationResult(
            test_name=scenario.name,
            passed=test_passed,
            score=overall_score,
            details=details,
            recommendations=recommendations
        )
    
    def _create_scenario_market_conditions(self, scenario: PartialFillScenario) -> Dict[str, Any]:
        """Create market conditions for test scenario"""
        return {
            'liquidity_factor': scenario.market_liquidity,
            'volatility_level': scenario.volatility_level,
            'time_of_day': scenario.time_of_day,
            'volume_ratio': scenario.market_liquidity * 1.2,
            'stress_indicator': 1.0 - scenario.market_liquidity
        }
    
    def _create_scenario_market_data(self, scenario: PartialFillScenario) -> pd.Series:
        """Create market data for test scenario"""
        base_price = 15000.0
        volatility_factor = scenario.volatility_level
        
        return pd.Series({
            'Close': base_price,
            'High': base_price * (1 + volatility_factor * 0.01),
            'Low': base_price * (1 - volatility_factor * 0.01),
            'Volume': int(1000000 * scenario.market_liquidity)
        })


class ExecutionTimingValidation:
    """
    Validation framework for execution timing
    """
    
    def __init__(self, execution_handler: RealisticBacktestExecutionHandler):
        self.execution_handler = execution_handler
        self.timing_scenarios = self._create_timing_scenarios()
        self.timing_results = []
    
    def _create_timing_scenarios(self) -> List[ExecutionTimingScenario]:
        """Create execution timing test scenarios"""
        scenarios = [
            ExecutionTimingScenario(
                name="market_order_normal",
                description="Market order in normal conditions",
                order_type="market",
                market_conditions="normal",
                expected_latency_range=(50, 200),
                timing_tolerance_ms=50
            ),
            ExecutionTimingScenario(
                name="market_order_stressed",
                description="Market order in stressed conditions",
                order_type="market",
                market_conditions="stressed",
                expected_latency_range=(100, 500),
                timing_tolerance_ms=100
            ),
            ExecutionTimingScenario(
                name="limit_order_normal",
                description="Limit order in normal conditions",
                order_type="limit",
                market_conditions="normal",
                expected_latency_range=(30, 150),
                timing_tolerance_ms=30
            ),
            ExecutionTimingScenario(
                name="market_order_illiquid",
                description="Market order in illiquid conditions",
                order_type="market",
                market_conditions="illiquid",
                expected_latency_range=(150, 800),
                timing_tolerance_ms=150
            )
        ]
        
        return scenarios
    
    async def run_timing_validation_tests(self) -> List[ValidationResult]:
        """Run comprehensive execution timing validation"""
        results = []
        
        for scenario in self.timing_scenarios:
            try:
                result = await self._test_timing_scenario(scenario)
                results.append(result)
                logger.info(f"Timing test completed: {scenario.name} - {'PASSED' if result.passed else 'FAILED'}")
            except Exception as e:
                logger.error(f"Timing test failed: {scenario.name} - {e}")
                results.append(ValidationResult(
                    test_name=scenario.name,
                    passed=False,
                    score=0,
                    details={'error': str(e)},
                    recommendations=['Fix timing validation implementation']
                ))
        
        self.timing_results = results
        return results
    
    async def _test_timing_scenario(self, scenario: ExecutionTimingScenario) -> ValidationResult:
        """Test a specific timing scenario"""
        # Create market conditions
        market_data = self._create_timing_market_data(scenario)
        
        # Create test trade
        trade_data = {
            'timestamp': datetime.now(),
            'symbol': 'NQ',
            'signal': 1,
            'size': 10,
            'price': 15000.0,
            'type': scenario.order_type
        }
        
        # Measure execution timing
        timing_measurements = []
        
        # Run multiple executions to get timing statistics
        for _ in range(5):
            start_time = datetime.now()
            execution_result = await self.execution_handler.execute_backtest_trade(
                trade_data, market_data, {}
            )
            end_time = datetime.now()
            
            execution_time_ms = (end_time - start_time).total_seconds() * 1000
            timing_measurements.append({
                'execution_time_ms': execution_time_ms,
                'success': execution_result['success'],
                'latency_ms': execution_result.get('latency_ms', 0)
            })
        
        # Analyze timing results
        successful_executions = [m for m in timing_measurements if m['success']]
        
        if successful_executions:
            avg_execution_time = np.mean([m['execution_time_ms'] for m in successful_executions])
            avg_latency = np.mean([m['latency_ms'] for m in successful_executions])
            
            # Check if timing is within expected range
            min_expected, max_expected = scenario.expected_latency_range
            timing_within_range = min_expected <= avg_latency <= max_expected
            timing_consistent = np.std([m['latency_ms'] for m in successful_executions]) <= scenario.timing_tolerance_ms
            
            # Calculate score
            range_score = 100 if timing_within_range else 50
            consistency_score = 100 if timing_consistent else 50
            overall_score = (range_score + consistency_score) / 2
            
            test_passed = timing_within_range and timing_consistent
            
            details = {
                'expected_latency_range': scenario.expected_latency_range,
                'actual_avg_latency': avg_latency,
                'avg_execution_time': avg_execution_time,
                'timing_within_range': timing_within_range,
                'timing_consistent': timing_consistent,
                'latency_std': np.std([m['latency_ms'] for m in successful_executions]),
                'successful_executions': len(successful_executions),
                'total_executions': len(timing_measurements)
            }
            
            recommendations = []
            if not timing_within_range:
                recommendations.append("Execution latency outside expected range")
            if not timing_consistent:
                recommendations.append("Execution timing inconsistent - high variance")
                
        else:
            test_passed = False
            overall_score = 0
            details = {'error': 'All executions failed'}
            recommendations = ['All timing test executions failed']
        
        return ValidationResult(
            test_name=scenario.name,
            passed=test_passed,
            score=overall_score,
            details=details,
            recommendations=recommendations
        )
    
    def _create_timing_market_data(self, scenario: ExecutionTimingScenario) -> pd.Series:
        """Create market data for timing scenario"""
        base_price = 15000.0
        
        # Adjust market data based on conditions
        if scenario.market_conditions == "stressed":
            volatility_factor = 0.02
            volume_factor = 0.5
        elif scenario.market_conditions == "illiquid":
            volatility_factor = 0.01
            volume_factor = 0.3
        else:  # normal
            volatility_factor = 0.005
            volume_factor = 1.0
        
        return pd.Series({
            'Close': base_price,
            'High': base_price * (1 + volatility_factor),
            'Low': base_price * (1 - volatility_factor),
            'Volume': int(1000000 * volume_factor)
        })


class BacktestLiveAlignmentValidator:
    """
    Comprehensive validator for backtest-live execution alignment
    """
    
    def __init__(self, 
                 execution_handler: RealisticBacktestExecutionHandler,
                 cost_model: ComprehensiveCostModel):
        self.execution_handler = execution_handler
        self.cost_model = cost_model
        
        # Validation components
        self.partial_fill_tester = PartialFillTesting(execution_handler)
        self.timing_validator = ExecutionTimingValidation(execution_handler)
        
        # Validation results
        self.validation_results = {}
        self.overall_score = 0
        self.alignment_status = "UNTESTED"
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        Run comprehensive validation of backtest-live alignment
        """
        logger.info("Starting comprehensive backtest-live alignment validation")
        
        validation_results = {}
        
        # 1. Partial Fill Validation
        logger.info("Running partial fill validation...")
        partial_fill_results = await self.partial_fill_tester.run_partial_fill_tests()
        validation_results['partial_fill_tests'] = partial_fill_results
        
        # 2. Execution Timing Validation
        logger.info("Running execution timing validation...")
        timing_results = await self.timing_validator.run_timing_validation_tests()
        validation_results['timing_tests'] = timing_results
        
        # 3. Cost Model Validation
        logger.info("Running cost model validation...")
        cost_validation_results = self._validate_cost_model()
        validation_results['cost_model_tests'] = cost_validation_results
        
        # 4. Market Condition Stress Testing
        logger.info("Running market condition stress tests...")
        stress_test_results = await self._run_market_stress_tests()
        validation_results['stress_tests'] = stress_test_results
        
        # 5. Overall Alignment Assessment
        logger.info("Calculating overall alignment score...")
        alignment_assessment = self._calculate_alignment_assessment(validation_results)
        validation_results['alignment_assessment'] = alignment_assessment
        
        # Store results
        self.validation_results = validation_results
        self.overall_score = alignment_assessment['overall_score']
        self.alignment_status = alignment_assessment['status']
        
        # Generate recommendations
        recommendations = self._generate_alignment_recommendations(validation_results)
        validation_results['recommendations'] = recommendations
        
        logger.info(f"Validation completed - Overall Score: {self.overall_score:.1f}/100")
        return validation_results
    
    def _validate_cost_model(self) -> List[ValidationResult]:
        """Validate cost model accuracy"""
        results = []
        
        # Test scenarios for cost validation
        test_scenarios = [
            {
                'name': 'small_order_normal_market',
                'order_size': 5,
                'expected_cost_range': (0.05, 0.15),  # 5-15 basis points
                'market_conditions': 'normal'
            },
            {
                'name': 'medium_order_normal_market',
                'order_size': 20,
                'expected_cost_range': (0.10, 0.25),  # 10-25 basis points
                'market_conditions': 'normal'
            },
            {
                'name': 'large_order_normal_market',
                'order_size': 50,
                'expected_cost_range': (0.20, 0.40),  # 20-40 basis points
                'market_conditions': 'normal'
            },
            {
                'name': 'medium_order_stressed_market',
                'order_size': 20,
                'expected_cost_range': (0.25, 0.60),  # 25-60 basis points
                'market_conditions': 'stressed'
            }
        ]
        
        for scenario in test_scenarios:
            try:
                # Create market data for scenario
                market_data = self._create_cost_validation_market_data(scenario)
                
                # Calculate costs
                cost_breakdown = self.cost_model.calculate_total_execution_costs(
                    order_size=scenario['order_size'],
                    order_type='market',
                    market_data=market_data,
                    timestamp=datetime.now()
                )
                
                # Validate cost is within expected range
                actual_cost_bps = cost_breakdown['cost_percentage']
                min_expected, max_expected = scenario['expected_cost_range']
                
                cost_within_range = min_expected <= actual_cost_bps <= max_expected
                
                # Calculate score
                if cost_within_range:
                    score = 100
                else:
                    # Score based on deviation from range
                    if actual_cost_bps < min_expected:
                        deviation = (min_expected - actual_cost_bps) / min_expected
                    else:
                        deviation = (actual_cost_bps - max_expected) / max_expected
                    score = max(0, 100 - deviation * 100)
                
                details = {
                    'expected_cost_range_bps': scenario['expected_cost_range'],
                    'actual_cost_bps': actual_cost_bps,
                    'cost_within_range': cost_within_range,
                    'cost_breakdown': cost_breakdown
                }
                
                recommendations = []
                if not cost_within_range:
                    if actual_cost_bps < min_expected:
                        recommendations.append("Cost model may be underestimating execution costs")
                    else:
                        recommendations.append("Cost model may be overestimating execution costs")
                
                results.append(ValidationResult(
                    test_name=scenario['name'],
                    passed=cost_within_range,
                    score=score,
                    details=details,
                    recommendations=recommendations
                ))
                
            except Exception as e:
                logger.error(f"Cost validation failed for {scenario['name']}: {e}")
                results.append(ValidationResult(
                    test_name=scenario['name'],
                    passed=False,
                    score=0,
                    details={'error': str(e)},
                    recommendations=['Fix cost model validation']
                ))
        
        return results
    
    def _create_cost_validation_market_data(self, scenario: Dict[str, Any]) -> pd.Series:
        """Create market data for cost validation"""
        base_price = 15000.0
        
        if scenario['market_conditions'] == 'stressed':
            volatility = 0.03
            volume_factor = 0.4
        else:  # normal
            volatility = 0.01
            volume_factor = 1.0
        
        return pd.Series({
            'Close': base_price,
            'High': base_price * (1 + volatility),
            'Low': base_price * (1 - volatility),
            'Volume': int(1000000 * volume_factor)
        })
    
    async def _run_market_stress_tests(self) -> List[ValidationResult]:
        """Run market condition stress tests"""
        stress_scenarios = [
            {
                'name': 'high_volatility_stress',
                'volatility_multiplier': 3.0,
                'volume_factor': 0.6,
                'expected_degradation': 0.3  # 30% degradation expected
            },
            {
                'name': 'low_liquidity_stress',
                'volatility_multiplier': 1.0,
                'volume_factor': 0.2,
                'expected_degradation': 0.4  # 40% degradation expected
            },
            {
                'name': 'combined_stress',
                'volatility_multiplier': 2.5,
                'volume_factor': 0.3,
                'expected_degradation': 0.5  # 50% degradation expected
            }
        ]
        
        results = []
        
        for scenario in stress_scenarios:
            try:
                # Create stressed market data
                market_data = self._create_stress_test_market_data(scenario)
                
                # Execute test trade
                trade_data = {
                    'timestamp': datetime.now(),
                    'symbol': 'NQ',
                    'signal': 1,
                    'size': 20,
                    'price': 15000.0,
                    'type': 'market'
                }
                
                execution_result = await self.execution_handler.execute_backtest_trade(
                    trade_data, market_data, {}
                )
                
                if execution_result['success']:
                    # Compare with baseline execution quality
                    baseline_quality = 85  # Assumed baseline quality
                    actual_quality = execution_result['execution_quality']
                    
                    quality_degradation = (baseline_quality - actual_quality) / baseline_quality
                    degradation_acceptable = quality_degradation <= scenario['expected_degradation']
                    
                    score = 100 if degradation_acceptable else max(0, 100 - abs(quality_degradation - scenario['expected_degradation']) * 200)
                    
                    details = {
                        'baseline_quality': baseline_quality,
                        'actual_quality': actual_quality,
                        'quality_degradation': quality_degradation,
                        'expected_degradation': scenario['expected_degradation'],
                        'degradation_acceptable': degradation_acceptable,
                        'execution_costs': execution_result['total_costs']
                    }
                    
                    recommendations = []
                    if not degradation_acceptable:
                        recommendations.append("Execution quality degradation exceeds expected levels under stress")
                    
                    results.append(ValidationResult(
                        test_name=scenario['name'],
                        passed=degradation_acceptable,
                        score=score,
                        details=details,
                        recommendations=recommendations
                    ))
                else:
                    results.append(ValidationResult(
                        test_name=scenario['name'],
                        passed=False,
                        score=0,
                        details={'error': execution_result.get('error', 'Execution failed')},
                        recommendations=['Execution failed under stress conditions']
                    ))
                    
            except Exception as e:
                logger.error(f"Stress test failed: {scenario['name']} - {e}")
                results.append(ValidationResult(
                    test_name=scenario['name'],
                    passed=False,
                    score=0,
                    details={'error': str(e)},
                    recommendations=['Fix stress test implementation']
                ))
        
        return results
    
    def _create_stress_test_market_data(self, scenario: Dict[str, Any]) -> pd.Series:
        """Create market data for stress testing"""
        base_price = 15000.0
        volatility = 0.01 * scenario['volatility_multiplier']
        volume = int(1000000 * scenario['volume_factor'])
        
        return pd.Series({
            'Close': base_price,
            'High': base_price * (1 + volatility),
            'Low': base_price * (1 - volatility),
            'Volume': volume
        })
    
    def _calculate_alignment_assessment(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall alignment assessment"""
        # Weight different test categories
        weights = {
            'partial_fill_tests': 0.25,
            'timing_tests': 0.25,
            'cost_model_tests': 0.30,
            'stress_tests': 0.20
        }
        
        category_scores = {}
        weighted_score = 0
        
        for category, weight in weights.items():
            if category in validation_results:
                test_results = validation_results[category]
                if test_results:
                    category_score = np.mean([r.score for r in test_results])
                    category_scores[category] = category_score
                    weighted_score += category_score * weight
                else:
                    category_scores[category] = 0
            else:
                category_scores[category] = 0
        
        # Determine alignment status
        if weighted_score >= 90:
            status = "EXCELLENT_ALIGNMENT"
        elif weighted_score >= 80:
            status = "GOOD_ALIGNMENT"
        elif weighted_score >= 70:
            status = "ACCEPTABLE_ALIGNMENT"
        elif weighted_score >= 60:
            status = "POOR_ALIGNMENT"
        else:
            status = "FAILED_ALIGNMENT"
        
        return {
            'overall_score': weighted_score,
            'status': status,
            'category_scores': category_scores,
            'weights_used': weights,
            'assessment_timestamp': datetime.now().isoformat()
        }
    
    def _generate_alignment_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Analyze each category
        for category, results in validation_results.items():
            if isinstance(results, list):
                failed_tests = [r for r in results if not r.passed]
                if failed_tests:
                    for test in failed_tests:
                        recommendations.extend(test.recommendations)
        
        # Overall recommendations based on alignment score
        alignment_score = validation_results.get('alignment_assessment', {}).get('overall_score', 0)
        
        if alignment_score < 70:
            recommendations.append("CRITICAL: Significant backtest-live divergence detected - not ready for live deployment")
        elif alignment_score < 80:
            recommendations.append("WARNING: Moderate backtest-live divergence - requires optimization before live deployment")
        elif alignment_score < 90:
            recommendations.append("GOOD: Minor backtest-live divergence - consider paper trading before live deployment")
        else:
            recommendations.append("EXCELLENT: Strong backtest-live alignment - ready for live deployment")
        
        return list(set(recommendations))  # Remove duplicates
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report"""
        if not self.validation_results:
            return "No validation results available. Run comprehensive validation first."
        
        report = []
        report.append("=" * 80)
        report.append("BACKTEST-LIVE ALIGNMENT VALIDATION REPORT")
        report.append("=" * 80)
        
        # Overall assessment
        assessment = self.validation_results.get('alignment_assessment', {})
        report.append(f"📊 OVERALL ALIGNMENT SCORE: {assessment.get('overall_score', 0):.1f}/100")
        report.append(f"📈 ALIGNMENT STATUS: {assessment.get('status', 'UNKNOWN')}")
        report.append("")
        
        # Category breakdown
        category_scores = assessment.get('category_scores', {})
        report.append("📋 CATEGORY BREAKDOWN:")
        for category, score in category_scores.items():
            status = "✅ PASS" if score >= 70 else "❌ FAIL"
            report.append(f"   {category.replace('_', ' ').title()}: {score:.1f}/100 {status}")
        report.append("")
        
        # Detailed results
        for category, results in self.validation_results.items():
            if isinstance(results, list) and results:
                report.append(f"🔍 {category.replace('_', ' ').upper()} RESULTS:")
                for result in results:
                    status = "✅ PASS" if result.passed else "❌ FAIL"
                    report.append(f"   {result.test_name}: {result.score:.1f}/100 {status}")
                report.append("")
        
        # Recommendations
        recommendations = self.validation_results.get('recommendations', [])
        if recommendations:
            report.append("🎯 RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                report.append(f"   {i}. {rec}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_validation_results(self, file_path: str = None) -> str:
        """Save validation results to file"""
        if file_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = f'/home/QuantNova/GrandModel/results/validation/backtest_live_alignment_{timestamp}.json'
        
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format
        serializable_results = self._convert_results_to_serializable(self.validation_results)
        
        with open(file_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Validation results saved: {file_path}")
        return file_path
    
    def _convert_results_to_serializable(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert validation results to JSON serializable format"""
        serializable = {}
        
        for key, value in results.items():
            if isinstance(value, list):
                serializable[key] = [
                    {
                        'test_name': r.test_name,
                        'passed': r.passed,
                        'score': r.score,
                        'details': r.details,
                        'recommendations': r.recommendations,
                        'timestamp': r.timestamp.isoformat()
                    } for r in value
                ]
            else:
                serializable[key] = value
        
        return serializable


# Main validation function
async def validate_backtest_live_alignment(
    execution_handler: RealisticBacktestExecutionHandler,
    cost_model: ComprehensiveCostModel = None
) -> Dict[str, Any]:
    """
    Run comprehensive backtest-live alignment validation
    
    Args:
        execution_handler: Realistic execution handler to validate
        cost_model: Cost model to validate (optional)
        
    Returns:
        Comprehensive validation results
    """
    if cost_model is None:
        cost_model = create_nq_futures_cost_model()
    
    validator = BacktestLiveAlignmentValidator(execution_handler, cost_model)
    
    # Run validation
    results = await validator.run_comprehensive_validation()
    
    # Generate and print report
    report = validator.generate_validation_report()
    print(report)
    
    # Save results
    validator.save_validation_results()
    
    return results


# Convenience function for testing
def create_test_execution_handler() -> RealisticBacktestExecutionHandler:
    """Create a test execution handler with realistic configuration"""
    config = BacktestExecutionConfig(
        enable_realistic_slippage=True,
        enable_market_impact=True,
        enable_execution_latency=True,
        enable_partial_fills=True,
        enable_order_book_simulation=True,
        use_dynamic_commission=True,
        include_exchange_fees=True,
        include_regulatory_fees=True
    )
    
    return RealisticBacktestExecutionHandler(
        initial_capital=100000.0,
        config=config
    )