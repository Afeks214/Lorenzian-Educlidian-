#!/usr/bin/env python3
"""
AGENT 7: Error Handling Validation Suite
Comprehensive validation and testing for error handling and logging systems.
"""

import asyncio
import json
import time
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from src.monitoring.trading_decision_logger import (
    TradingDecisionLogger,
    TradingDecisionType,
    TradingDecisionOutcome,
    get_trading_decision_logger
)
from src.monitoring.trading_error_integration import (
    TradingErrorIntegrator,
    get_trading_error_integrator
)
from src.core.errors.error_handler import get_error_handler
from src.core.errors.base_exceptions import (
    BaseGrandModelError,
    SystemError,
    NetworkError,
    TimeoutError,
    ValidationError,
    ErrorContext
)
from src.monitoring.structured_logging import get_logger


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    passed: bool
    duration_ms: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "duration_ms": self.duration_ms,
            "error_message": self.error_message,
            "details": self.details
        }


class ErrorHandlingValidator:
    """
    Comprehensive validator for error handling and logging systems.
    """
    
    def __init__(self):
        self.logger = get_logger("error_handling_validator")
        self.trading_logger = get_trading_decision_logger()
        self.error_handler = get_error_handler()
        self.integrator = get_trading_error_integrator()
        
        self.validation_results: List[ValidationResult] = []
        self.test_start_time: Optional[datetime] = None
        
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of error handling and logging systems."""
        
        self.test_start_time = datetime.now()
        self.logger.info("Starting comprehensive error handling validation")
        
        try:
            # Test 1: Trading Decision Logger
            await self._test_trading_decision_logger()
            
            # Test 2: Error Handler
            await self._test_error_handler()
            
            # Test 3: Silent Failure Detection
            await self._test_silent_failure_detection()
            
            # Test 4: Error Correlation
            await self._test_error_correlation()
            
            # Test 5: Integration System
            await self._test_integration_system()
            
            # Test 6: Performance Under Load
            await self._test_performance_under_load()
            
            # Test 7: Recovery Mechanisms
            await self._test_recovery_mechanisms()
            
            # Test 8: VaR Calculator Error Handling
            await self._test_var_calculator_error_handling()
            
            # Generate final report
            return self._generate_validation_report()
            
        except Exception as e:
            self.logger.error(f"Validation suite failed: {e}", exc_info=True)
            return {
                "status": "FAILED",
                "error": str(e),
                "completed_tests": len(self.validation_results)
            }
    
    async def _test_trading_decision_logger(self):
        """Test trading decision logger functionality."""
        
        test_name = "trading_decision_logger"
        start_time = time.time()
        
        try:
            # Test decision logging
            with self.trading_logger.decision_context(
                decision_type=TradingDecisionType.POSITION_SIZING,
                agent_id="test_agent",
                strategy_id="test_strategy",
                symbol="BTCUSD"
            ) as tracker:
                
                tracker.set_decision_logic("Test position sizing logic")
                tracker.set_inputs({"portfolio_value": 100000, "risk_pct": 0.02})
                tracker.update_metrics(
                    decision_latency_ms=50.0,
                    confidence_score=0.85,
                    risk_score=0.2,
                    expected_return=0.1,
                    expected_risk=0.05
                )
                tracker.set_outputs({"position_size": 2000.0})
                tracker.set_outcome(TradingDecisionOutcome.SUCCESS)
            
            # Verify decision was logged
            summary = self.trading_logger.get_decision_summary(hours=1)
            assert summary["total_decisions"] > 0, "No decisions logged"
            
            # Test performance attribution
            attribution = self.trading_logger.get_performance_attribution(hours=1)
            assert len(attribution) > 0, "No performance attribution data"
            
            # Test decision statistics
            stats = self.trading_logger.get_decision_statistics()
            assert stats["total_decisions"] > 0, "Decision statistics not updating"
            
            duration_ms = (time.time() - start_time) * 1000
            
            self.validation_results.append(ValidationResult(
                test_name=test_name,
                passed=True,
                duration_ms=duration_ms,
                details={
                    "decisions_logged": summary["total_decisions"],
                    "performance_attribution_agents": len(attribution),
                    "total_statistics": stats
                }
            ))
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.validation_results.append(ValidationResult(
                test_name=test_name,
                passed=False,
                duration_ms=duration_ms,
                error_message=str(e),
                details={"traceback": traceback.format_exc()}
            ))
    
    async def _test_error_handler(self):
        """Test error handler functionality."""
        
        test_name = "error_handler"
        start_time = time.time()
        
        try:
            # Test error statistics tracking
            initial_stats = self.error_handler.get_error_statistics()
            
            # Test different error types
            test_errors = [
                SystemError("Test system error"),
                NetworkError("Test network error"),
                TimeoutError("Test timeout error"),
                ValidationError("Test validation error")
            ]
            
            for error in test_errors:
                context = ErrorContext(
                    additional_data={"test_error": True, "error_type": type(error).__name__}
                )
                
                try:
                    self.error_handler.handle_exception(error, context, function_name="test_function")
                except:
                    pass  # Expected for non-recoverable errors
            
            # Verify error statistics updated
            final_stats = self.error_handler.get_error_statistics()
            assert final_stats["total_errors"] > initial_stats["total_errors"], "Error statistics not updating"
            
            # Test health report
            health_report = self.error_handler.get_health_report()
            assert "health_score" in health_report, "Health report missing health score"
            assert "status" in health_report, "Health report missing status"
            
            duration_ms = (time.time() - start_time) * 1000
            
            self.validation_results.append(ValidationResult(
                test_name=test_name,
                passed=True,
                duration_ms=duration_ms,
                details={
                    "errors_processed": len(test_errors),
                    "initial_error_count": initial_stats["total_errors"],
                    "final_error_count": final_stats["total_errors"],
                    "health_score": health_report["health_score"]
                }
            ))
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.validation_results.append(ValidationResult(
                test_name=test_name,
                passed=False,
                duration_ms=duration_ms,
                error_message=str(e),
                details={"traceback": traceback.format_exc()}
            ))
    
    async def _test_silent_failure_detection(self):
        """Test silent failure detection mechanisms."""
        
        test_name = "silent_failure_detection"
        start_time = time.time()
        
        try:
            # Register test function as mandatory response
            self.error_handler.register_mandatory_response_function(
                "test_silent_failure",
                validator=lambda x: x is not None and x > 0
            )
            
            # Test valid response
            valid_response = 42
            assert self.error_handler.validate_response("test_silent_failure", valid_response), "Valid response failed validation"
            
            # Test invalid responses
            invalid_responses = [None, 0, -1, ""]
            silent_failures_detected = 0
            
            for response in invalid_responses:
                if not self.error_handler.validate_response("test_silent_failure", response):
                    silent_failures_detected += 1
            
            assert silent_failures_detected > 0, "Silent failures not detected"
            
            # Check silent failure statistics
            stats = self.error_handler.get_error_statistics()
            assert stats["silent_failures"] > 0, "Silent failure statistics not updating"
            
            duration_ms = (time.time() - start_time) * 1000
            
            self.validation_results.append(ValidationResult(
                test_name=test_name,
                passed=True,
                duration_ms=duration_ms,
                details={
                    "silent_failures_detected": silent_failures_detected,
                    "total_silent_failures": stats["silent_failures"],
                    "mandatory_functions_registered": len(self.error_handler.mandatory_response_functions)
                }
            ))
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.validation_results.append(ValidationResult(
                test_name=test_name,
                passed=False,
                duration_ms=duration_ms,
                error_message=str(e),
                details={"traceback": traceback.format_exc()}
            ))
    
    async def _test_error_correlation(self):
        """Test error correlation and pattern detection."""
        
        test_name = "error_correlation"
        start_time = time.time()
        
        try:
            # Generate correlated errors
            for i in range(5):
                error = SystemError(f"Correlated error {i}")
                context = ErrorContext(
                    additional_data={"correlation_test": True, "sequence": i}
                )
                
                try:
                    self.error_handler.handle_exception(error, context, function_name="correlated_function")
                except:
                    pass
                
                # Small delay to create temporal correlation
                await asyncio.sleep(0.1)
            
            # Check if correlations were detected
            stats = self.error_handler.get_error_statistics()
            assert stats["correlated_errors"] > 0, "No correlated errors detected"
            
            duration_ms = (time.time() - start_time) * 1000
            
            self.validation_results.append(ValidationResult(
                test_name=test_name,
                passed=True,
                duration_ms=duration_ms,
                details={
                    "correlated_errors": stats["correlated_errors"],
                    "error_pattern_detection": "functional"
                }
            ))
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.validation_results.append(ValidationResult(
                test_name=test_name,
                passed=False,
                duration_ms=duration_ms,
                error_message=str(e),
                details={"traceback": traceback.format_exc()}
            ))
    
    async def _test_integration_system(self):
        """Test integration between trading logger and error handler."""
        
        test_name = "integration_system"
        start_time = time.time()
        
        try:
            # Test successful integration
            @self.integrator.trading_function(
                decision_type=TradingDecisionType.RISK_ASSESSMENT,
                agent_id="test_agent",
                strategy_id="test_strategy",
                mandatory_response=True
            )
            def successful_risk_assessment(symbol: str) -> Dict[str, float]:
                return {"risk_score": 0.3, "confidence": 0.8}
            
            # Test failing integration
            @self.integrator.trading_function(
                decision_type=TradingDecisionType.RISK_ASSESSMENT,
                agent_id="test_agent",
                strategy_id="test_strategy",
                mandatory_response=True
            )
            def failing_risk_assessment(symbol: str) -> None:
                return None  # This should trigger silent failure detection
            
            # Execute successful function
            result1 = successful_risk_assessment("BTCUSD")
            assert result1 is not None, "Successful function returned None"
            
            # Execute failing function
            result2 = failing_risk_assessment("BTCUSD")
            assert result2 is None, "Failing function should return None after error handling"
            
            # Check integration statistics
            integration_stats = self.integrator.get_integration_statistics()
            assert "trading_decisions" in integration_stats, "Missing trading decisions in integration stats"
            assert "error_handling" in integration_stats, "Missing error handling in integration stats"
            
            duration_ms = (time.time() - start_time) * 1000
            
            self.validation_results.append(ValidationResult(
                test_name=test_name,
                passed=True,
                duration_ms=duration_ms,
                details={
                    "integration_stats": integration_stats,
                    "successful_execution": result1 is not None,
                    "silent_failure_handled": result2 is None
                }
            ))
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.validation_results.append(ValidationResult(
                test_name=test_name,
                passed=False,
                duration_ms=duration_ms,
                error_message=str(e),
                details={"traceback": traceback.format_exc()}
            ))
    
    async def _test_performance_under_load(self):
        """Test performance under load."""
        
        test_name = "performance_under_load"
        start_time = time.time()
        
        try:
            # Generate load with concurrent operations
            async def generate_load():
                tasks = []
                
                for i in range(100):
                    # Create mixed workload
                    if i % 3 == 0:
                        # Trading decision
                        task = self._create_trading_decision_task(i)
                    elif i % 3 == 1:
                        # Error handling
                        task = self._create_error_handling_task(i)
                    else:
                        # Silent failure detection
                        task = self._create_silent_failure_task(i)
                    
                    tasks.append(task)
                
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Execute load test
            load_start = time.time()
            await generate_load()
            load_duration = time.time() - load_start
            
            # Check performance metrics
            stats = self.error_handler.get_error_statistics()
            decision_stats = self.trading_logger.get_decision_statistics()
            
            # Performance should be reasonable
            assert load_duration < 10.0, f"Load test took too long: {load_duration:.2f}s"
            
            duration_ms = (time.time() - start_time) * 1000
            
            self.validation_results.append(ValidationResult(
                test_name=test_name,
                passed=True,
                duration_ms=duration_ms,
                details={
                    "load_duration_seconds": load_duration,
                    "operations_per_second": 100 / load_duration,
                    "final_error_count": stats["total_errors"],
                    "final_decision_count": decision_stats["total_decisions"]
                }
            ))
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.validation_results.append(ValidationResult(
                test_name=test_name,
                passed=False,
                duration_ms=duration_ms,
                error_message=str(e),
                details={"traceback": traceback.format_exc()}
            ))
    
    async def _create_trading_decision_task(self, index: int):
        """Create a trading decision task for load testing."""
        
        with self.trading_logger.decision_context(
            decision_type=TradingDecisionType.SIGNAL_GENERATION,
            agent_id=f"load_test_agent_{index % 10}",
            strategy_id=f"load_test_strategy_{index % 5}",
            symbol="BTCUSD"
        ) as tracker:
            
            tracker.set_decision_logic(f"Load test decision {index}")
            tracker.set_inputs({"index": index})
            tracker.update_metrics(
                decision_latency_ms=10.0,
                confidence_score=0.7,
                risk_score=0.3,
                expected_return=0.05,
                expected_risk=0.02
            )
            tracker.set_outputs({"signal": "BUY", "strength": 0.8})
            tracker.set_outcome(TradingDecisionOutcome.SUCCESS)
    
    async def _create_error_handling_task(self, index: int):
        """Create an error handling task for load testing."""
        
        error = SystemError(f"Load test error {index}")
        context = ErrorContext(
            additional_data={"load_test": True, "index": index}
        )
        
        try:
            self.error_handler.handle_exception(error, context, function_name=f"load_test_function_{index}")
        except:
            pass  # Expected for some errors
    
    async def _create_silent_failure_task(self, index: int):
        """Create a silent failure detection task for load testing."""
        
        function_name = f"load_test_silent_{index}"
        
        # Register function if not already registered
        if function_name not in self.error_handler.mandatory_response_functions:
            self.error_handler.register_mandatory_response_function(
                function_name,
                validator=lambda x: x is not None
            )
        
        # Test with invalid response
        self.error_handler.validate_response(function_name, None)
    
    async def _test_recovery_mechanisms(self):
        """Test error recovery mechanisms."""
        
        test_name = "recovery_mechanisms"
        start_time = time.time()
        
        try:
            # Test fallback mechanism
            def fallback_function(error, error_occurrence):
                return {"fallback": True, "original_error": str(error)}
            
            self.error_handler.fallback_manager.register_fallback("test_fallback", fallback_function)
            
            # Test error with fallback
            test_error = SystemError("Test error for fallback")
            context = ErrorContext(additional_data={"test_fallback": True})
            
            result = self.error_handler.handle_exception(
                test_error, context, fallback_name="test_fallback", function_name="test_recovery"
            )
            
            assert result is not None, "Fallback mechanism failed"
            assert result.get("fallback") is True, "Fallback result incorrect"
            
            # Check recovery statistics
            stats = self.error_handler.get_error_statistics()
            assert stats["recovery_attempts"] > 0, "No recovery attempts recorded"
            
            duration_ms = (time.time() - start_time) * 1000
            
            self.validation_results.append(ValidationResult(
                test_name=test_name,
                passed=True,
                duration_ms=duration_ms,
                details={
                    "fallback_result": result,
                    "recovery_attempts": stats["recovery_attempts"],
                    "recovery_successes": stats["recovery_successes"]
                }
            ))
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.validation_results.append(ValidationResult(
                test_name=test_name,
                passed=False,
                duration_ms=duration_ms,
                error_message=str(e),
                details={"traceback": traceback.format_exc()}
            ))
    
    async def _test_var_calculator_error_handling(self):
        """Test VaR calculator error handling integration."""
        
        test_name = "var_calculator_error_handling"
        start_time = time.time()
        
        try:
            # Test VaR calculation with error handling
            @self.integrator.async_trading_function(
                decision_type=TradingDecisionType.RISK_ASSESSMENT,
                agent_id="var_test_agent",
                strategy_id="var_test_strategy",
                mandatory_response=True
            )
            async def mock_var_calculation(confidence_level: float) -> Dict[str, float]:
                """Mock VaR calculation that might fail."""
                if confidence_level < 0 or confidence_level > 1:
                    raise ValidationError("Invalid confidence level")
                
                return {
                    "portfolio_var": 5000.0,
                    "confidence_level": confidence_level,
                    "calculation_method": "test"
                }
            
            # Test successful VaR calculation
            result1 = await mock_var_calculation(0.95)
            assert result1 is not None, "VaR calculation failed"
            assert result1["portfolio_var"] > 0, "Invalid VaR result"
            
            # Test failing VaR calculation
            try:
                result2 = await mock_var_calculation(1.5)  # Invalid confidence level
                assert False, "Should have raised ValidationError"
            except ValidationError:
                pass  # Expected
            
            duration_ms = (time.time() - start_time) * 1000
            
            self.validation_results.append(ValidationResult(
                test_name=test_name,
                passed=True,
                duration_ms=duration_ms,
                details={
                    "successful_calculation": result1,
                    "error_handling_functional": True
                }
            ))
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.validation_results.append(ValidationResult(
                test_name=test_name,
                passed=False,
                duration_ms=duration_ms,
                error_message=str(e),
                details={"traceback": traceback.format_exc()}
            ))
    
    def _generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        
        total_tests = len(self.validation_results)
        passed_tests = sum(1 for result in self.validation_results if result.passed)
        failed_tests = total_tests - passed_tests
        
        total_duration = sum(result.duration_ms for result in self.validation_results)
        
        # Calculate success rate
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        # Get current system statistics
        error_stats = self.error_handler.get_error_statistics()
        trading_stats = self.trading_logger.get_decision_statistics()
        integration_stats = self.integrator.get_integration_statistics()
        health_report = self.error_handler.get_health_report()
        
        # Generate recommendations
        recommendations = []
        
        if success_rate < 0.8:
            recommendations.append("Address failing tests to improve system reliability")
        
        if failed_tests > 0:
            failed_test_names = [r.test_name for r in self.validation_results if not r.passed]
            recommendations.append(f"Fix failing tests: {', '.join(failed_test_names)}")
        
        if health_report["health_score"] < 80:
            recommendations.extend(health_report["recommendations"])
        
        if not recommendations:
            recommendations.append("All tests passed - system is operating correctly")
        
        return {
            "validation_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate,
                "total_duration_ms": total_duration,
                "validation_date": datetime.now().isoformat()
            },
            "test_results": [result.to_dict() for result in self.validation_results],
            "system_statistics": {
                "error_handling": error_stats,
                "trading_decisions": trading_stats,
                "integration": integration_stats,
                "health_report": health_report
            },
            "recommendations": recommendations,
            "status": "PASSED" if success_rate >= 0.8 else "FAILED"
        }


async def main():
    """Run comprehensive validation."""
    
    validator = ErrorHandlingValidator()
    
    print("🔍 Starting comprehensive error handling validation...")
    print("=" * 60)
    
    # Run validation
    start_time = time.time()
    report = await validator.run_comprehensive_validation()
    total_time = time.time() - start_time
    
    # Print summary
    summary = report["validation_summary"]
    print(f"\n📊 VALIDATION SUMMARY")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Total Duration: {total_time:.2f}s")
    print(f"Status: {report['status']}")
    
    # Print failed tests
    if summary['failed_tests'] > 0:
        print(f"\n❌ FAILED TESTS:")
        for result in report["test_results"]:
            if not result["passed"]:
                print(f"  - {result['test_name']}: {result['error_message']}")
    
    # Print recommendations
    print(f"\n📋 RECOMMENDATIONS:")
    for recommendation in report["recommendations"]:
        print(f"  - {recommendation}")
    
    # Save detailed report
    report_filename = f"error_handling_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_filename}")
    
    return report


if __name__ == "__main__":
    # Run validation
    report = asyncio.run(main())
    
    # Exit with appropriate code
    exit_code = 0 if report["status"] == "PASSED" else 1
    print(f"\n🎯 Validation {'PASSED' if exit_code == 0 else 'FAILED'}")
    exit(exit_code)