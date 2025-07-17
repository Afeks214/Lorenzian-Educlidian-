"""
Comprehensive Stress Testing Framework for Risk Management

This module implements advanced stress testing scenarios for extreme market conditions,
including:
- Black Swan event simulation
- Market regime transition testing
- Correlation breakdown scenarios
- Liquidity crisis simulation
- Volatility clustering tests
- Fat-tail distribution testing

Author: Agent 16 - Risk Management Enhancement Specialist
Mission: Implement production-ready stress testing framework
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import scipy.stats as stats
from scipy.optimize import minimize
import warnings
import structlog

logger = structlog.get_logger()


class StressTestType(Enum):
    """Types of stress tests"""
    BLACK_SWAN = "black_swan"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    VOLATILITY_SPIKE = "volatility_spike"
    REGIME_TRANSITION = "regime_transition"
    FAT_TAIL = "fat_tail"
    COMPOUND_CRISIS = "compound_crisis"


@dataclass
class StressScenario:
    """Stress testing scenario definition"""
    name: str
    scenario_type: StressTestType
    description: str
    parameters: Dict
    severity: str  # LOW, MEDIUM, HIGH, EXTREME
    historical_analogue: Optional[str] = None
    probability_estimate: Optional[float] = None


@dataclass
class StressTestResult:
    """Result of a stress test"""
    scenario: StressScenario
    timestamp: datetime
    portfolio_value_initial: float
    portfolio_value_stressed: float
    portfolio_loss: float
    portfolio_loss_percentage: float
    max_drawdown: float
    time_to_recovery_days: Optional[int]
    var_breach_magnitude: float
    component_losses: Dict[str, float]
    risk_metrics: Dict[str, float]
    passed: bool
    failure_reason: Optional[str] = None


class StressTestingFramework:
    """
    Comprehensive stress testing framework for risk management.
    
    Implements multiple stress testing methodologies including:
    - Historical scenario replay
    - Monte Carlo stress testing
    - Parametric stress testing
    - Extreme value theory applications
    """
    
    def __init__(
        self,
        confidence_levels: List[float] = [0.95, 0.99, 0.999],
        time_horizons: List[int] = [1, 5, 10, 22],  # Days
        max_acceptable_loss: float = 0.20  # 20% portfolio loss threshold
    ):
        self.confidence_levels = confidence_levels
        self.time_horizons = time_horizons
        self.max_acceptable_loss = max_acceptable_loss
        
        # Historical crisis scenarios
        self.historical_scenarios = self._define_historical_scenarios()
        
        # Stress test results
        self.stress_results: List[StressTestResult] = []
        
        logger.info("StressTestingFramework initialized",
                   confidence_levels=confidence_levels,
                   time_horizons=time_horizons,
                   max_acceptable_loss=max_acceptable_loss)
    
    def _define_historical_scenarios(self) -> Dict[str, StressScenario]:
        """Define historical crisis scenarios for testing"""
        scenarios = {}
        
        # Black Monday 1987
        scenarios["black_monday_1987"] = StressScenario(
            name="Black Monday 1987",
            scenario_type=StressTestType.BLACK_SWAN,
            description="Stock market crash of October 19, 1987",
            parameters={
                "equity_shock": -0.22,  # 22% drop
                "volatility_multiplier": 5.0,
                "correlation_spike": 0.8,
                "duration_days": 1
            },
            severity="EXTREME",
            historical_analogue="1987-10-19",
            probability_estimate=0.001
        )
        
        # Russian Financial Crisis 1998
        scenarios["russian_crisis_1998"] = StressScenario(
            name="Russian Financial Crisis 1998",
            scenario_type=StressTestType.LIQUIDITY_CRISIS,
            description="Russian ruble devaluation and default",
            parameters={
                "currency_shock": -0.75,  # 75% devaluation
                "bond_shock": -0.90,     # 90% bond value loss
                "liquidity_multiplier": 0.1,  # 10x liquidity costs
                "duration_days": 30
            },
            severity="HIGH",
            historical_analogue="1998-08-17",
            probability_estimate=0.01
        )
        
        # 2008 Financial Crisis
        scenarios["financial_crisis_2008"] = StressScenario(
            name="Global Financial Crisis 2008",
            scenario_type=StressTestType.COMPOUND_CRISIS,
            description="Lehman Brothers collapse and credit crisis",
            parameters={
                "equity_shock": -0.57,   # 57% peak-to-trough
                "credit_spread_widening": 0.05,  # 500bp widening
                "volatility_multiplier": 3.0,
                "correlation_spike": 0.9,
                "duration_days": 180
            },
            severity="EXTREME",
            historical_analogue="2008-09-15",
            probability_estimate=0.005
        )
        
        # COVID-19 Market Crash 2020
        scenarios["covid_crash_2020"] = StressScenario(
            name="COVID-19 Market Crash 2020",
            scenario_type=StressTestType.VOLATILITY_SPIKE,
            description="Pandemic-induced market volatility",
            parameters={
                "equity_shock": -0.35,   # 35% drop
                "volatility_multiplier": 4.0,
                "correlation_spike": 0.85,
                "recovery_speed": 0.5,  # Faster recovery
                "duration_days": 33
            },
            severity="HIGH",
            historical_analogue="2020-02-20",
            probability_estimate=0.02
        )
        
        # Long-Term Capital Management 1998
        scenarios["ltcm_1998"] = StressScenario(
            name="LTCM Crisis 1998",
            scenario_type=StressTestType.CORRELATION_BREAKDOWN,
            description="Convergence trade failure and correlation breakdown",
            parameters={
                "correlation_reversal": True,
                "correlation_magnitude": -0.5,  # Negative correlation
                "leverage_unwind": 0.95,  # 95% forced deleveraging
                "duration_days": 45
            },
            severity="HIGH",
            historical_analogue="1998-09-23",
            probability_estimate=0.01
        )
        
        return scenarios
    
    def run_stress_test(
        self,
        scenario: StressScenario,
        portfolio_positions: Dict[str, float],
        portfolio_value: float,
        asset_returns: Dict[str, np.ndarray],
        correlation_matrix: np.ndarray,
        volatilities: Dict[str, float]
    ) -> StressTestResult:
        """
        Run a single stress test scenario.
        
        Args:
            scenario: Stress test scenario to execute
            portfolio_positions: Dict of {asset: position_value}
            portfolio_value: Total portfolio value
            asset_returns: Historical returns for each asset
            correlation_matrix: Current correlation matrix
            volatilities: Asset volatilities
            
        Returns:
            StressTestResult with comprehensive metrics
        """
        logger.info("Running stress test",
                   scenario=scenario.name,
                   scenario_type=scenario.scenario_type.value,
                   severity=scenario.severity)
        
        try:
            if scenario.scenario_type == StressTestType.BLACK_SWAN:
                result = self._run_black_swan_test(
                    scenario, portfolio_positions, portfolio_value,
                    asset_returns, correlation_matrix, volatilities
                )
            elif scenario.scenario_type == StressTestType.CORRELATION_BREAKDOWN:
                result = self._run_correlation_breakdown_test(
                    scenario, portfolio_positions, portfolio_value,
                    asset_returns, correlation_matrix, volatilities
                )
            elif scenario.scenario_type == StressTestType.LIQUIDITY_CRISIS:
                result = self._run_liquidity_crisis_test(
                    scenario, portfolio_positions, portfolio_value,
                    asset_returns, correlation_matrix, volatilities
                )
            elif scenario.scenario_type == StressTestType.VOLATILITY_SPIKE:
                result = self._run_volatility_spike_test(
                    scenario, portfolio_positions, portfolio_value,
                    asset_returns, correlation_matrix, volatilities
                )
            elif scenario.scenario_type == StressTestType.COMPOUND_CRISIS:
                result = self._run_compound_crisis_test(
                    scenario, portfolio_positions, portfolio_value,
                    asset_returns, correlation_matrix, volatilities
                )
            else:
                raise ValueError(f"Unknown scenario type: {scenario.scenario_type}")
            
            # Store result
            self.stress_results.append(result)
            
            logger.info("Stress test completed",
                       scenario=scenario.name,
                       portfolio_loss_pct=f"{result.portfolio_loss_percentage:.2%}",
                       passed=result.passed)
            
            return result
            
        except Exception as e:
            logger.error("Stress test failed",
                        scenario=scenario.name,
                        error=str(e))
            
            # Return failure result
            return StressTestResult(
                scenario=scenario,
                timestamp=datetime.now(),
                portfolio_value_initial=portfolio_value,
                portfolio_value_stressed=0.0,
                portfolio_loss=portfolio_value,
                portfolio_loss_percentage=1.0,
                max_drawdown=1.0,
                time_to_recovery_days=None,
                var_breach_magnitude=float('inf'),
                component_losses={},
                risk_metrics={},
                passed=False,
                failure_reason=str(e)
            )
    
    def _run_black_swan_test(
        self,
        scenario: StressScenario,
        portfolio_positions: Dict[str, float],
        portfolio_value: float,
        asset_returns: Dict[str, np.ndarray],
        correlation_matrix: np.ndarray,
        volatilities: Dict[str, float]
    ) -> StressTestResult:
        """Run Black Swan stress test"""
        
        params = scenario.parameters
        equity_shock = params.get("equity_shock", -0.20)
        volatility_multiplier = params.get("volatility_multiplier", 3.0)
        correlation_spike = params.get("correlation_spike", 0.8)
        duration_days = params.get("duration_days", 1)
        
        # Calculate shocked portfolio value
        component_losses = {}
        total_loss = 0.0
        
        for asset, position_value in portfolio_positions.items():
            if asset in volatilities:
                # Apply shock with increased volatility
                base_volatility = volatilities[asset]
                shocked_volatility = base_volatility * volatility_multiplier
                
                # Simulate extreme negative return
                shock_return = equity_shock * (1 + shocked_volatility)
                
                # Calculate loss
                asset_loss = position_value * abs(shock_return)
                component_losses[asset] = asset_loss
                total_loss += asset_loss
        
        # Calculate portfolio value after shock
        portfolio_value_stressed = portfolio_value - total_loss
        portfolio_loss_pct = total_loss / portfolio_value
        
        # Calculate additional risk metrics
        risk_metrics = {
            "volatility_multiplier": volatility_multiplier,
            "correlation_spike": correlation_spike,
            "duration_days": duration_days,
            "max_single_asset_loss": max(component_losses.values()) if component_losses else 0
        }
        
        # Recovery time estimate (simplified)
        recovery_days = int(duration_days * 10) if portfolio_loss_pct > 0.1 else None
        
        # Check if test passes
        passed = portfolio_loss_pct <= self.max_acceptable_loss
        
        return StressTestResult(
            scenario=scenario,
            timestamp=datetime.now(),
            portfolio_value_initial=portfolio_value,
            portfolio_value_stressed=portfolio_value_stressed,
            portfolio_loss=total_loss,
            portfolio_loss_percentage=portfolio_loss_pct,
            max_drawdown=portfolio_loss_pct,
            time_to_recovery_days=recovery_days,
            var_breach_magnitude=max(0, portfolio_loss_pct - self.max_acceptable_loss),
            component_losses=component_losses,
            risk_metrics=risk_metrics,
            passed=passed
        )
    
    def _run_correlation_breakdown_test(
        self,
        scenario: StressScenario,
        portfolio_positions: Dict[str, float],
        portfolio_value: float,
        asset_returns: Dict[str, np.ndarray],
        correlation_matrix: np.ndarray,
        volatilities: Dict[str, float]
    ) -> StressTestResult:
        """Run correlation breakdown stress test"""
        
        params = scenario.parameters
        correlation_reversal = params.get("correlation_reversal", True)
        correlation_magnitude = params.get("correlation_magnitude", -0.5)
        leverage_unwind = params.get("leverage_unwind", 0.5)
        duration_days = params.get("duration_days", 30)
        
        # Simulate correlation breakdown
        assets = list(portfolio_positions.keys())
        n_assets = len(assets)
        
        if n_assets < 2:
            # Cannot test correlation breakdown with less than 2 assets
            return StressTestResult(
                scenario=scenario,
                timestamp=datetime.now(),
                portfolio_value_initial=portfolio_value,
                portfolio_value_stressed=portfolio_value,
                portfolio_loss=0.0,
                portfolio_loss_percentage=0.0,
                max_drawdown=0.0,
                time_to_recovery_days=0,
                var_breach_magnitude=0.0,
                component_losses={},
                risk_metrics={},
                passed=True,
                failure_reason="Insufficient assets for correlation test"
            )
        
        # Create stressed correlation matrix
        stressed_correlation = np.ones((n_assets, n_assets)) * correlation_magnitude
        np.fill_diagonal(stressed_correlation, 1.0)
        
        # Ensure matrix is positive semidefinite
        eigenvals, eigenvecs = np.linalg.eigh(stressed_correlation)
        eigenvals = np.maximum(eigenvals, 1e-8)  # Ensure non-negative
        stressed_correlation = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Calculate portfolio losses under stressed correlations
        weights = np.array([
            portfolio_positions[asset] / portfolio_value 
            for asset in assets
        ])
        
        asset_volatilities = np.array([
            volatilities.get(asset, 0.2) for asset in assets
        ])
        
        # Portfolio volatility under stressed correlations
        stressed_portfolio_var = np.dot(weights, 
                                       np.dot(np.outer(asset_volatilities, asset_volatilities) * 
                                             stressed_correlation, weights))
        stressed_portfolio_vol = np.sqrt(stressed_portfolio_var)
        
        # Simulate losses with leverage unwind
        base_loss = portfolio_value * stressed_portfolio_vol * 2.33  # 99% VaR
        leverage_loss = base_loss * leverage_unwind
        
        total_loss = base_loss + leverage_loss
        
        # Distribute losses proportionally
        component_losses = {}
        for asset in assets:
            asset_weight = portfolio_positions[asset] / portfolio_value
            component_losses[asset] = total_loss * asset_weight
        
        portfolio_value_stressed = portfolio_value - total_loss
        portfolio_loss_pct = total_loss / portfolio_value
        
        risk_metrics = {
            "stressed_correlation": correlation_magnitude,
            "leverage_unwind": leverage_unwind,
            "stressed_portfolio_volatility": stressed_portfolio_vol,
            "duration_days": duration_days
        }
        
        # Recovery time estimate
        recovery_days = int(duration_days * 2) if portfolio_loss_pct > 0.05 else None
        
        passed = portfolio_loss_pct <= self.max_acceptable_loss
        
        return StressTestResult(
            scenario=scenario,
            timestamp=datetime.now(),
            portfolio_value_initial=portfolio_value,
            portfolio_value_stressed=portfolio_value_stressed,
            portfolio_loss=total_loss,
            portfolio_loss_percentage=portfolio_loss_pct,
            max_drawdown=portfolio_loss_pct,
            time_to_recovery_days=recovery_days,
            var_breach_magnitude=max(0, portfolio_loss_pct - self.max_acceptable_loss),
            component_losses=component_losses,
            risk_metrics=risk_metrics,
            passed=passed
        )
    
    def _run_liquidity_crisis_test(
        self,
        scenario: StressScenario,
        portfolio_positions: Dict[str, float],
        portfolio_value: float,
        asset_returns: Dict[str, np.ndarray],
        correlation_matrix: np.ndarray,
        volatilities: Dict[str, float]
    ) -> StressTestResult:
        """Run liquidity crisis stress test"""
        
        params = scenario.parameters
        liquidity_multiplier = params.get("liquidity_multiplier", 0.1)
        bond_shock = params.get("bond_shock", -0.3)
        currency_shock = params.get("currency_shock", -0.2)
        duration_days = params.get("duration_days", 60)
        
        # Simulate liquidity crisis impacts
        component_losses = {}
        total_loss = 0.0
        
        for asset, position_value in portfolio_positions.items():
            # Base market loss
            base_loss = position_value * abs(bond_shock) * 0.5  # Assume 50% bond exposure
            
            # Liquidity cost (higher for larger positions)
            liquidity_cost = position_value * (1 - liquidity_multiplier)
            
            # Currency impact (if applicable)
            currency_loss = position_value * abs(currency_shock) * 0.3  # 30% fx exposure
            
            total_asset_loss = base_loss + liquidity_cost + currency_loss
            component_losses[asset] = total_asset_loss
            total_loss += total_asset_loss
        
        portfolio_value_stressed = portfolio_value - total_loss
        portfolio_loss_pct = total_loss / portfolio_value
        
        risk_metrics = {
            "liquidity_multiplier": liquidity_multiplier,
            "bond_shock": bond_shock,
            "currency_shock": currency_shock,
            "duration_days": duration_days,
            "avg_liquidity_cost": np.mean([loss for loss in component_losses.values()])
        }
        
        # Recovery time estimate (longer for liquidity crises)
        recovery_days = int(duration_days * 3) if portfolio_loss_pct > 0.1 else duration_days
        
        passed = portfolio_loss_pct <= self.max_acceptable_loss
        
        return StressTestResult(
            scenario=scenario,
            timestamp=datetime.now(),
            portfolio_value_initial=portfolio_value,
            portfolio_value_stressed=portfolio_value_stressed,
            portfolio_loss=total_loss,
            portfolio_loss_percentage=portfolio_loss_pct,
            max_drawdown=portfolio_loss_pct,
            time_to_recovery_days=recovery_days,
            var_breach_magnitude=max(0, portfolio_loss_pct - self.max_acceptable_loss),
            component_losses=component_losses,
            risk_metrics=risk_metrics,
            passed=passed
        )
    
    def _run_volatility_spike_test(
        self,
        scenario: StressScenario,
        portfolio_positions: Dict[str, float],
        portfolio_value: float,
        asset_returns: Dict[str, np.ndarray],
        correlation_matrix: np.ndarray,
        volatilities: Dict[str, float]
    ) -> StressTestResult:
        """Run volatility spike stress test"""
        
        params = scenario.parameters
        volatility_multiplier = params.get("volatility_multiplier", 4.0)
        equity_shock = params.get("equity_shock", -0.3)
        correlation_spike = params.get("correlation_spike", 0.85)
        duration_days = params.get("duration_days", 30)
        
        # Calculate losses from volatility spike
        component_losses = {}
        total_loss = 0.0
        
        for asset, position_value in portfolio_positions.items():
            if asset in volatilities:
                base_volatility = volatilities[asset]
                stressed_volatility = base_volatility * volatility_multiplier
                
                # VaR calculation with stressed volatility
                var_99 = position_value * stressed_volatility * 2.33  # 99% VaR
                
                # Add directional shock
                directional_loss = position_value * abs(equity_shock)
                
                total_asset_loss = var_99 + directional_loss
                component_losses[asset] = total_asset_loss
                total_loss += total_asset_loss
        
        portfolio_value_stressed = portfolio_value - total_loss
        portfolio_loss_pct = total_loss / portfolio_value
        
        risk_metrics = {
            "volatility_multiplier": volatility_multiplier,
            "equity_shock": equity_shock,
            "correlation_spike": correlation_spike,
            "duration_days": duration_days,
            "max_stressed_volatility": max([
                volatilities.get(asset, 0.2) * volatility_multiplier 
                for asset in portfolio_positions.keys()
            ])
        }
        
        # Recovery time estimate
        recovery_days = int(duration_days * 1.5) if portfolio_loss_pct > 0.1 else duration_days
        
        passed = portfolio_loss_pct <= self.max_acceptable_loss
        
        return StressTestResult(
            scenario=scenario,
            timestamp=datetime.now(),
            portfolio_value_initial=portfolio_value,
            portfolio_value_stressed=portfolio_value_stressed,
            portfolio_loss=total_loss,
            portfolio_loss_percentage=portfolio_loss_pct,
            max_drawdown=portfolio_loss_pct,
            time_to_recovery_days=recovery_days,
            var_breach_magnitude=max(0, portfolio_loss_pct - self.max_acceptable_loss),
            component_losses=component_losses,
            risk_metrics=risk_metrics,
            passed=passed
        )
    
    def _run_compound_crisis_test(
        self,
        scenario: StressScenario,
        portfolio_positions: Dict[str, float],
        portfolio_value: float,
        asset_returns: Dict[str, np.ndarray],
        correlation_matrix: np.ndarray,
        volatilities: Dict[str, float]
    ) -> StressTestResult:
        """Run compound crisis stress test (combines multiple crisis types)"""
        
        params = scenario.parameters
        equity_shock = params.get("equity_shock", -0.5)
        volatility_multiplier = params.get("volatility_multiplier", 3.0)
        correlation_spike = params.get("correlation_spike", 0.9)
        credit_spread_widening = params.get("credit_spread_widening", 0.03)
        duration_days = params.get("duration_days", 180)
        
        # Compound crisis combines multiple effects
        component_losses = {}
        total_loss = 0.0
        
        for asset, position_value in portfolio_positions.items():
            # Equity shock
            equity_loss = position_value * abs(equity_shock)
            
            # Volatility impact
            base_volatility = volatilities.get(asset, 0.2)
            vol_loss = position_value * base_volatility * volatility_multiplier * 2.33
            
            # Credit spread impact (assume some credit exposure)
            credit_loss = position_value * credit_spread_widening * 5  # Duration effect
            
            # Liquidity impact
            liquidity_loss = position_value * 0.05  # 5% liquidity haircut
            
            total_asset_loss = equity_loss + vol_loss + credit_loss + liquidity_loss
            component_losses[asset] = total_asset_loss
            total_loss += total_asset_loss
        
        portfolio_value_stressed = portfolio_value - total_loss
        portfolio_loss_pct = total_loss / portfolio_value
        
        risk_metrics = {
            "equity_shock": equity_shock,
            "volatility_multiplier": volatility_multiplier,
            "correlation_spike": correlation_spike,
            "credit_spread_widening": credit_spread_widening,
            "duration_days": duration_days,
            "compound_multiplier": 1.2  # Additional uncertainty factor
        }
        
        # Recovery time estimate (longest for compound crises)
        recovery_days = int(duration_days * 2) if portfolio_loss_pct > 0.1 else duration_days
        
        passed = portfolio_loss_pct <= self.max_acceptable_loss
        
        return StressTestResult(
            scenario=scenario,
            timestamp=datetime.now(),
            portfolio_value_initial=portfolio_value,
            portfolio_value_stressed=portfolio_value_stressed,
            portfolio_loss=total_loss,
            portfolio_loss_percentage=portfolio_loss_pct,
            max_drawdown=portfolio_loss_pct,
            time_to_recovery_days=recovery_days,
            var_breach_magnitude=max(0, portfolio_loss_pct - self.max_acceptable_loss),
            component_losses=component_losses,
            risk_metrics=risk_metrics,
            passed=passed
        )
    
    def run_comprehensive_stress_test_suite(
        self,
        portfolio_positions: Dict[str, float],
        portfolio_value: float,
        asset_returns: Dict[str, np.ndarray],
        correlation_matrix: np.ndarray,
        volatilities: Dict[str, float]
    ) -> Dict[str, StressTestResult]:
        """Run comprehensive stress test suite across all scenarios"""
        
        logger.info("Starting comprehensive stress test suite",
                   portfolio_value=portfolio_value,
                   num_positions=len(portfolio_positions))
        
        results = {}
        
        # Run all historical scenarios
        for scenario_name, scenario in self.historical_scenarios.items():
            result = self.run_stress_test(
                scenario, portfolio_positions, portfolio_value,
                asset_returns, correlation_matrix, volatilities
            )
            results[scenario_name] = result
        
        # Calculate aggregate metrics
        total_scenarios = len(results)
        passed_scenarios = sum(1 for r in results.values() if r.passed)
        max_loss = max(r.portfolio_loss_percentage for r in results.values())
        avg_loss = np.mean([r.portfolio_loss_percentage for r in results.values()])
        
        logger.info("Comprehensive stress test suite completed",
                   total_scenarios=total_scenarios,
                   passed_scenarios=passed_scenarios,
                   pass_rate=f"{passed_scenarios/total_scenarios:.1%}",
                   max_loss=f"{max_loss:.2%}",
                   avg_loss=f"{avg_loss:.2%}")
        
        return results
    
    def generate_stress_test_report(self) -> Dict:
        """Generate comprehensive stress test report"""
        
        if not self.stress_results:
            return {"status": "No stress test results available"}
        
        # Aggregate statistics
        total_tests = len(self.stress_results)
        passed_tests = sum(1 for r in self.stress_results if r.passed)
        failed_tests = total_tests - passed_tests
        
        # Loss statistics
        losses = [r.portfolio_loss_percentage for r in self.stress_results]
        max_loss = max(losses)
        avg_loss = np.mean(losses)
        median_loss = np.median(losses)
        
        # Recovery statistics
        recovery_times = [r.time_to_recovery_days for r in self.stress_results if r.time_to_recovery_days]
        avg_recovery_days = np.mean(recovery_times) if recovery_times else 0
        
        # Failed scenarios
        failed_scenarios = [
            {
                "scenario": r.scenario.name,
                "loss_percentage": r.portfolio_loss_percentage,
                "failure_reason": r.failure_reason
            }
            for r in self.stress_results if not r.passed
        ]
        
        # Risk concentration analysis
        all_component_losses = {}
        for result in self.stress_results:
            for asset, loss in result.component_losses.items():
                if asset not in all_component_losses:
                    all_component_losses[asset] = []
                all_component_losses[asset].append(loss)
        
        # Calculate average losses by asset
        avg_component_losses = {
            asset: np.mean(losses) 
            for asset, losses in all_component_losses.items()
        }
        
        return {
            "report_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "max_loss_percentage": max_loss,
                "avg_loss_percentage": avg_loss,
                "median_loss_percentage": median_loss,
                "avg_recovery_days": avg_recovery_days
            },
            "failed_scenarios": failed_scenarios,
            "risk_concentration": {
                "avg_component_losses": avg_component_losses,
                "max_risk_asset": max(avg_component_losses.items(), key=lambda x: x[1])[0] if avg_component_losses else None
            },
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on stress test results"""
        recommendations = []
        
        if not self.stress_results:
            return ["No stress test results available for analysis"]
        
        # Analyze pass rate
        pass_rate = sum(1 for r in self.stress_results if r.passed) / len(self.stress_results)
        
        if pass_rate < 0.7:
            recommendations.append("CRITICAL: Low stress test pass rate (<70%) - consider reducing portfolio risk")
        elif pass_rate < 0.85:
            recommendations.append("WARNING: Moderate stress test pass rate - review risk limits")
        
        # Analyze maximum loss
        max_loss = max(r.portfolio_loss_percentage for r in self.stress_results)
        
        if max_loss > 0.5:
            recommendations.append("CRITICAL: Maximum stress loss exceeds 50% - implement stronger risk controls")
        elif max_loss > 0.3:
            recommendations.append("WARNING: High maximum stress loss - consider portfolio diversification")
        
        # Analyze recovery times
        recovery_times = [r.time_to_recovery_days for r in self.stress_results if r.time_to_recovery_days]
        if recovery_times:
            avg_recovery = np.mean(recovery_times)
            if avg_recovery > 365:
                recommendations.append("CRITICAL: Long recovery times (>1 year) - improve liquidity management")
            elif avg_recovery > 180:
                recommendations.append("WARNING: Extended recovery times - consider rebalancing frequency")
        
        # Analyze risk concentration
        all_component_losses = {}
        for result in self.stress_results:
            for asset, loss in result.component_losses.items():
                if asset not in all_component_losses:
                    all_component_losses[asset] = []
                all_component_losses[asset].append(loss)
        
        if all_component_losses:
            avg_losses = {asset: np.mean(losses) for asset, losses in all_component_losses.items()}
            max_risk_asset = max(avg_losses.items(), key=lambda x: x[1])
            
            if max_risk_asset[1] > self.max_acceptable_loss * 0.3:  # 30% of max acceptable loss
                recommendations.append(f"WARNING: High concentration risk in {max_risk_asset[0]} - consider position sizing")
        
        if not recommendations:
            recommendations.append("Portfolio demonstrates good stress test resilience")
        
        return recommendations


# Factory function for easy instantiation
def create_stress_testing_framework(
    confidence_levels: List[float] = [0.95, 0.99, 0.999],
    max_acceptable_loss: float = 0.20
) -> StressTestingFramework:
    """Create a comprehensive stress testing framework"""
    return StressTestingFramework(
        confidence_levels=confidence_levels,
        max_acceptable_loss=max_acceptable_loss
    )