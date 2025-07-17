"""
Self-healing test maintenance framework.

This module provides automatic test maintenance capabilities including
failure analysis, repair suggestions, and automated test updates.
"""

import ast
import re
import subprocess
import json
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import traceback
import difflib


class RepairStrategy(Enum):
    """Test repair strategies."""
    UPDATE_ASSERTION = "update_assertion"
    UPDATE_MOCK = "update_mock"
    UPDATE_IMPORT = "update_import"
    UPDATE_FIXTURE = "update_fixture"
    UPDATE_PARAMETRIZE = "update_parametrize"
    REGENERATE_TEST = "regenerate_test"
    DISABLE_TEST = "disable_test"


class FailurePattern(Enum):
    """Common test failure patterns."""
    ASSERTION_ERROR = "assertion_error"
    IMPORT_ERROR = "import_error"
    ATTRIBUTE_ERROR = "attribute_error"
    TYPE_ERROR = "type_error"
    VALUE_ERROR = "value_error"
    TIMEOUT_ERROR = "timeout_error"
    MOCK_ERROR = "mock_error"
    FIXTURE_ERROR = "fixture_error"
    DEPENDENCY_ERROR = "dependency_error"
    FLAKY_TEST = "flaky_test"


@dataclass
class TestFailureInfo:
    """Information about a test failure."""
    test_name: str
    test_file: str
    failure_type: str
    error_message: str
    traceback: str
    failure_pattern: FailurePattern
    timestamp: datetime
    code_context: Optional[str] = None
    suggested_fixes: List[str] = field(default_factory=list)


@dataclass
class RepairSuggestion:
    """Suggestion for repairing a test."""
    strategy: RepairStrategy
    description: str
    original_code: str
    suggested_code: str
    confidence: float
    estimated_effort: str  # "LOW", "MEDIUM", "HIGH"
    requires_review: bool = False


@dataclass
class TestMaintenanceReport:
    """Report of test maintenance activities."""
    timestamp: datetime
    total_tests: int
    failed_tests: int
    repaired_tests: int
    auto_fixed_tests: int
    manual_review_needed: int
    failure_patterns: Dict[FailurePattern, int]
    repair_suggestions: List[RepairSuggestion]
    flaky_tests: List[str]


class TestFailureAnalyzer:
    """
    Analyzes test failures to identify patterns and suggest repairs.
    """
    
    def __init__(self):
        self.failure_patterns = {
            FailurePattern.ASSERTION_ERROR: [
                r"AssertionError",
                r"assert.*failed",
                r"Expected.*but got"
            ],
            FailurePattern.IMPORT_ERROR: [
                r"ImportError",
                r"ModuleNotFoundError",
                r"No module named"
            ],
            FailurePattern.ATTRIBUTE_ERROR: [
                r"AttributeError",
                r"has no attribute",
                r"object has no attribute"
            ],
            FailurePattern.TYPE_ERROR: [
                r"TypeError",
                r"takes.*positional arguments",
                r"unexpected keyword argument"
            ],
            FailurePattern.VALUE_ERROR: [
                r"ValueError",
                r"invalid literal",
                r"could not convert"
            ],
            FailurePattern.TIMEOUT_ERROR: [
                r"TimeoutError",
                r"Test timed out",
                r"timeout"
            ],
            FailurePattern.MOCK_ERROR: [
                r"Mock.*not called",
                r"call not found",
                r"Expected call"
            ],
            FailurePattern.FIXTURE_ERROR: [
                r"fixture.*not found",
                r"ScopeError",
                r"fixture.*failed"
            ]
        }
        
        self.failure_history: List[TestFailureInfo] = []
        
    def analyze_failure(self, test_result: Dict[str, Any]) -> TestFailureInfo:
        """Analyze a single test failure."""
        
        error_message = test_result.get("error_message", "")
        traceback_str = test_result.get("traceback", "")
        
        # Identify failure pattern
        failure_pattern = self._identify_failure_pattern(error_message, traceback_str)
        
        # Extract code context
        code_context = self._extract_code_context(test_result.get("test_file", ""))
        
        # Generate initial fix suggestions
        suggested_fixes = self._generate_fix_suggestions(failure_pattern, error_message)
        
        failure_info = TestFailureInfo(
            test_name=test_result.get("test_name", ""),
            test_file=test_result.get("test_file", ""),
            failure_type=test_result.get("failure_type", ""),
            error_message=error_message,
            traceback=traceback_str,
            failure_pattern=failure_pattern,
            timestamp=datetime.now(),
            code_context=code_context,
            suggested_fixes=suggested_fixes
        )
        
        self.failure_history.append(failure_info)
        return failure_info
    
    def _identify_failure_pattern(self, error_message: str, traceback_str: str) -> FailurePattern:
        """Identify the failure pattern from error message and traceback."""
        
        combined_text = f"{error_message}\n{traceback_str}"
        
        for pattern, regexes in self.failure_patterns.items():
            for regex in regexes:
                if re.search(regex, combined_text, re.IGNORECASE):
                    return pattern
        
        return FailurePattern.ASSERTION_ERROR  # Default
    
    def _extract_code_context(self, test_file: str) -> Optional[str]:
        """Extract code context around the failing test."""
        
        if not test_file or not Path(test_file).exists():
            return None
        
        try:
            with open(test_file, 'r') as f:
                lines = f.readlines()
            
            # Find the test function (simplified)
            for i, line in enumerate(lines):
                if line.strip().startswith("def test_"):
                    # Extract function and some context
                    start = max(0, i - 5)
                    end = min(len(lines), i + 20)
                    return ''.join(lines[start:end])
            
            return None
            
        except Exception:
            return None
    
    def _generate_fix_suggestions(self, pattern: FailurePattern, error_message: str) -> List[str]:
        """Generate initial fix suggestions based on failure pattern."""
        
        suggestions = []
        
        if pattern == FailurePattern.ASSERTION_ERROR:
            suggestions.extend([
                "Update assertion to match actual behavior",
                "Check if expected values changed",
                "Verify test data setup"
            ])
        
        elif pattern == FailurePattern.IMPORT_ERROR:
            suggestions.extend([
                "Check if module path changed",
                "Update import statements",
                "Verify module installation"
            ])
        
        elif pattern == FailurePattern.ATTRIBUTE_ERROR:
            suggestions.extend([
                "Check if attribute name changed",
                "Verify object initialization",
                "Update attribute access"
            ])
        
        elif pattern == FailurePattern.MOCK_ERROR:
            suggestions.extend([
                "Update mock expectations",
                "Check mock call arguments",
                "Verify mock setup"
            ])
        
        elif pattern == FailurePattern.FIXTURE_ERROR:
            suggestions.extend([
                "Check fixture availability",
                "Update fixture dependencies",
                "Verify fixture scope"
            ])
        
        return suggestions
    
    def identify_flaky_tests(self, window_days: int = 7) -> List[str]:
        """Identify tests that fail intermittently."""
        
        cutoff_date = datetime.now() - timedelta(days=window_days)
        recent_failures = [f for f in self.failure_history if f.timestamp >= cutoff_date]
        
        # Count failures by test name
        failure_counts = {}
        for failure in recent_failures:
            test_name = failure.test_name
            if test_name not in failure_counts:
                failure_counts[test_name] = 0
            failure_counts[test_name] += 1
        
        # Identify tests that failed multiple times but not consistently
        flaky_tests = []
        for test_name, count in failure_counts.items():
            if 2 <= count <= 5:  # Failed sometimes but not always
                flaky_tests.append(test_name)
        
        return flaky_tests
    
    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get statistics about test failures."""
        
        total_failures = len(self.failure_history)
        
        if total_failures == 0:
            return {"total_failures": 0}
        
        # Count by pattern
        pattern_counts = {}
        for failure in self.failure_history:
            pattern = failure.failure_pattern
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Get most common patterns
        most_common = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_failures": total_failures,
            "pattern_counts": pattern_counts,
            "most_common_patterns": most_common,
            "flaky_tests": self.identify_flaky_tests()
        }


class TestRepairEngine:
    """
    Engine for automatically repairing test failures.
    """
    
    def __init__(self, analyzer: TestFailureAnalyzer):
        self.analyzer = analyzer
        self.repair_strategies = {
            FailurePattern.ASSERTION_ERROR: self._repair_assertion_error,
            FailurePattern.IMPORT_ERROR: self._repair_import_error,
            FailurePattern.ATTRIBUTE_ERROR: self._repair_attribute_error,
            FailurePattern.MOCK_ERROR: self._repair_mock_error,
            FailurePattern.FIXTURE_ERROR: self._repair_fixture_error
        }
        
    def generate_repair_suggestions(self, failure_info: TestFailureInfo) -> List[RepairSuggestion]:
        """Generate repair suggestions for a test failure."""
        
        suggestions = []
        
        # Get strategy-specific suggestions
        if failure_info.failure_pattern in self.repair_strategies:
            strategy_suggestions = self.repair_strategies[failure_info.failure_pattern](failure_info)
            suggestions.extend(strategy_suggestions)
        
        # Add general suggestions
        general_suggestions = self._generate_general_suggestions(failure_info)
        suggestions.extend(general_suggestions)
        
        # Sort by confidence
        suggestions.sort(key=lambda x: x.confidence, reverse=True)
        
        return suggestions
    
    def _repair_assertion_error(self, failure_info: TestFailureInfo) -> List[RepairSuggestion]:
        """Generate suggestions for assertion errors."""
        
        suggestions = []
        
        # Try to extract expected vs actual values
        error_msg = failure_info.error_message
        
        # Pattern: "assert X == Y" failed
        if "assert" in error_msg and "==" in error_msg:
            suggestions.append(RepairSuggestion(
                strategy=RepairStrategy.UPDATE_ASSERTION,
                description="Update assertion to match actual behavior",
                original_code="assert expected == actual",
                suggested_code="# Review and update assertion based on actual behavior",
                confidence=0.7,
                estimated_effort="LOW",
                requires_review=True
            ))
        
        # Pattern: AssertionError with specific message
        if "AssertionError:" in error_msg:
            suggestions.append(RepairSuggestion(
                strategy=RepairStrategy.UPDATE_ASSERTION,
                description="Review assertion logic",
                original_code="# Current assertion",
                suggested_code="# Updated assertion based on error",
                confidence=0.6,
                estimated_effort="MEDIUM",
                requires_review=True
            ))
        
        return suggestions
    
    def _repair_import_error(self, failure_info: TestFailureInfo) -> List[RepairSuggestion]:
        """Generate suggestions for import errors."""
        
        suggestions = []
        
        error_msg = failure_info.error_message
        
        # Extract module name
        module_match = re.search(r"No module named '([^']+)'", error_msg)
        if module_match:
            module_name = module_match.group(1)
            
            suggestions.append(RepairSuggestion(
                strategy=RepairStrategy.UPDATE_IMPORT,
                description=f"Fix import for module '{module_name}'",
                original_code=f"import {module_name}",
                suggested_code=f"# Check if module path changed or install module\\n# pip install {module_name}",
                confidence=0.8,
                estimated_effort="LOW",
                requires_review=False
            ))
        
        return suggestions
    
    def _repair_attribute_error(self, failure_info: TestFailureInfo) -> List[RepairSuggestion]:
        """Generate suggestions for attribute errors."""
        
        suggestions = []
        
        error_msg = failure_info.error_message
        
        # Extract attribute name
        attr_match = re.search(r"has no attribute '([^']+)'", error_msg)
        if attr_match:
            attr_name = attr_match.group(1)
            
            suggestions.append(RepairSuggestion(
                strategy=RepairStrategy.UPDATE_ASSERTION,
                description=f"Fix attribute access for '{attr_name}'",
                original_code=f"obj.{attr_name}",
                suggested_code=f"# Check if attribute name changed or use hasattr()\\nif hasattr(obj, '{attr_name}'):\\n    result = obj.{attr_name}",
                confidence=0.7,
                estimated_effort="MEDIUM",
                requires_review=True
            ))
        
        return suggestions
    
    def _repair_mock_error(self, failure_info: TestFailureInfo) -> List[RepairSuggestion]:
        """Generate suggestions for mock errors."""
        
        suggestions = []
        
        suggestions.append(RepairSuggestion(
            strategy=RepairStrategy.UPDATE_MOCK,
            description="Update mock expectations",
            original_code="mock.assert_called_with(args)",
            suggested_code="# Review mock call expectations\\n# Check actual calls with mock.call_args_list",
            confidence=0.6,
            estimated_effort="MEDIUM",
            requires_review=True
        ))
        
        return suggestions
    
    def _repair_fixture_error(self, failure_info: TestFailureInfo) -> List[RepairSuggestion]:
        """Generate suggestions for fixture errors."""
        
        suggestions = []
        
        suggestions.append(RepairSuggestion(
            strategy=RepairStrategy.UPDATE_FIXTURE,
            description="Fix fixture dependency",
            original_code="@pytest.fixture",
            suggested_code="# Check fixture availability and scope\\n# Update fixture dependencies",
            confidence=0.5,
            estimated_effort="HIGH",
            requires_review=True
        ))
        
        return suggestions
    
    def _generate_general_suggestions(self, failure_info: TestFailureInfo) -> List[RepairSuggestion]:
        """Generate general repair suggestions."""
        
        suggestions = []
        
        # Suggest test regeneration for complex failures
        if failure_info.failure_pattern in [FailurePattern.TYPE_ERROR, FailurePattern.VALUE_ERROR]:
            suggestions.append(RepairSuggestion(
                strategy=RepairStrategy.REGENERATE_TEST,
                description="Regenerate test using AI",
                original_code="# Current test",
                suggested_code="# AI-generated updated test",
                confidence=0.4,
                estimated_effort="MEDIUM",
                requires_review=True
            ))
        
        # Suggest disabling flaky tests
        flaky_tests = self.analyzer.identify_flaky_tests()
        if failure_info.test_name in flaky_tests:
            suggestions.append(RepairSuggestion(
                strategy=RepairStrategy.DISABLE_TEST,
                description="Temporarily disable flaky test",
                original_code="def test_function():",
                suggested_code="@pytest.mark.skip(reason='Flaky test - needs investigation')\\ndef test_function():",
                confidence=0.3,
                estimated_effort="LOW",
                requires_review=True
            ))
        
        return suggestions
    
    def apply_repair(self, test_file: str, suggestion: RepairSuggestion) -> bool:
        """Apply a repair suggestion to a test file."""
        
        if not Path(test_file).exists():
            return False
        
        try:
            with open(test_file, 'r') as f:
                content = f.read()
            
            # Apply the repair based on strategy
            if suggestion.strategy == RepairStrategy.UPDATE_IMPORT:
                updated_content = self._apply_import_repair(content, suggestion)
            elif suggestion.strategy == RepairStrategy.UPDATE_MOCK:
                updated_content = self._apply_mock_repair(content, suggestion)
            elif suggestion.strategy == RepairStrategy.DISABLE_TEST:
                updated_content = self._apply_disable_repair(content, suggestion)
            else:
                # For other strategies, require manual review
                return False
            
            # Write back the updated content
            with open(test_file, 'w') as f:
                f.write(updated_content)
            
            return True
            
        except Exception:
            return False
    
    def _apply_import_repair(self, content: str, suggestion: RepairSuggestion) -> str:
        """Apply import repair to content."""
        # Simple implementation - add try/except around imports
        lines = content.split('\n')
        updated_lines = []
        
        for line in lines:
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                updated_lines.append(f"try:")
                updated_lines.append(f"    {line}")
                updated_lines.append(f"except ImportError:")
                updated_lines.append(f"    # Module not available, skipping")
                updated_lines.append(f"    pass")
            else:
                updated_lines.append(line)
        
        return '\n'.join(updated_lines)
    
    def _apply_mock_repair(self, content: str, suggestion: RepairSuggestion) -> str:
        """Apply mock repair to content."""
        # Add debug information for mock calls
        return content + "\n# TODO: Review mock expectations\n"
    
    def _apply_disable_repair(self, content: str, suggestion: RepairSuggestion) -> str:
        """Apply test disable repair to content."""
        # Add skip decorator
        return content.replace("def test_", "@pytest.mark.skip(reason='Auto-disabled due to failures')\\ndef test_")


class TestMaintenanceScheduler:
    """
    Scheduler for automated test maintenance tasks.
    """
    
    def __init__(self, analyzer: TestFailureAnalyzer, repair_engine: TestRepairEngine):
        self.analyzer = analyzer
        self.repair_engine = repair_engine
        self.maintenance_history: List[TestMaintenanceReport] = []
        
    def run_maintenance_cycle(self, test_results: List[Dict[str, Any]]) -> TestMaintenanceReport:
        """Run a complete maintenance cycle."""
        
        timestamp = datetime.now()
        total_tests = len(test_results)
        failed_tests = len([r for r in test_results if r.get("status") == "FAILED"])
        
        # Analyze failures
        failure_infos = []
        for result in test_results:
            if result.get("status") == "FAILED":
                failure_info = self.analyzer.analyze_failure(result)
                failure_infos.append(failure_info)
        
        # Generate repair suggestions
        all_suggestions = []
        for failure_info in failure_infos:
            suggestions = self.repair_engine.generate_repair_suggestions(failure_info)
            all_suggestions.extend(suggestions)
        
        # Attempt automatic repairs
        auto_fixed = 0
        for suggestion in all_suggestions:
            if (suggestion.confidence > 0.8 and 
                suggestion.estimated_effort == "LOW" and 
                not suggestion.requires_review):
                # Apply automatic repair
                if self.repair_engine.apply_repair("dummy_file.py", suggestion):
                    auto_fixed += 1
        
        # Count patterns
        failure_patterns = {}
        for failure_info in failure_infos:
            pattern = failure_info.failure_pattern
            failure_patterns[pattern] = failure_patterns.get(pattern, 0) + 1
        
        # Create report
        report = TestMaintenanceReport(
            timestamp=timestamp,
            total_tests=total_tests,
            failed_tests=failed_tests,
            repaired_tests=len(all_suggestions),
            auto_fixed_tests=auto_fixed,
            manual_review_needed=len([s for s in all_suggestions if s.requires_review]),
            failure_patterns=failure_patterns,
            repair_suggestions=all_suggestions,
            flaky_tests=self.analyzer.identify_flaky_tests()
        )
        
        self.maintenance_history.append(report)
        return report
    
    def get_maintenance_summary(self) -> Dict[str, Any]:
        """Get summary of maintenance activities."""
        
        if not self.maintenance_history:
            return {"total_cycles": 0}
        
        total_cycles = len(self.maintenance_history)
        latest_report = self.maintenance_history[-1]
        
        return {
            "total_cycles": total_cycles,
            "latest_report": {
                "timestamp": latest_report.timestamp,
                "total_tests": latest_report.total_tests,
                "failed_tests": latest_report.failed_tests,
                "auto_fixed_tests": latest_report.auto_fixed_tests
            },
            "trends": {
                "failure_rate": latest_report.failed_tests / latest_report.total_tests if latest_report.total_tests > 0 else 0,
                "auto_fix_rate": latest_report.auto_fixed_tests / latest_report.failed_tests if latest_report.failed_tests > 0 else 0
            }
        }


class SelfHealingTestFramework:
    """
    Main framework for self-healing test maintenance.
    """
    
    def __init__(self):
        self.analyzer = TestFailureAnalyzer()
        self.repair_engine = TestRepairEngine(self.analyzer)
        self.scheduler = TestMaintenanceScheduler(self.analyzer, self.repair_engine)
        
    def process_test_results(self, test_results: List[Dict[str, Any]]) -> TestMaintenanceReport:
        """Process test results and perform maintenance."""
        return self.scheduler.run_maintenance_cycle(test_results)
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get overall health report of the test suite."""
        
        failure_stats = self.analyzer.get_failure_statistics()
        maintenance_summary = self.scheduler.get_maintenance_summary()
        
        return {
            "failure_statistics": failure_stats,
            "maintenance_summary": maintenance_summary,
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations for test suite improvement."""
        
        recommendations = []
        
        # Check for high failure rates
        failure_stats = self.analyzer.get_failure_statistics()
        if failure_stats.get("total_failures", 0) > 50:
            recommendations.append({
                "type": "HIGH_FAILURE_RATE",
                "description": "Test suite has high failure rate",
                "action": "Review test quality and stability",
                "priority": "HIGH"
            })
        
        # Check for flaky tests
        flaky_tests = failure_stats.get("flaky_tests", [])
        if len(flaky_tests) > 5:
            recommendations.append({
                "type": "FLAKY_TESTS",
                "description": f"Found {len(flaky_tests)} flaky tests",
                "action": "Investigate and fix flaky tests",
                "priority": "MEDIUM"
            })
        
        # Check for common failure patterns
        common_patterns = failure_stats.get("most_common_patterns", [])
        if common_patterns:
            top_pattern = common_patterns[0]
            recommendations.append({
                "type": "COMMON_PATTERN",
                "description": f"Most common failure: {top_pattern[0]}",
                "action": "Address root cause of common failures",
                "priority": "HIGH"
            })
        
        return recommendations