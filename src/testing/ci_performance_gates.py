"""
CI/CD Performance Gates - Agent 3

This module provides CI/CD integration for performance gates that fail builds
on performance regression, with build comparison and automated bisection.
"""

import os
import json
import sqlite3
import subprocess
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import structlog
import tempfile
import shutil

from .performance_regression_system import (
    PerformanceRegressionDetector,
    PerformanceBenchmark,
    performance_detector
)
from .performance_alerting_system import alerting_system

logger = structlog.get_logger()

@dataclass
class BuildInfo:
    """Build information"""
    build_id: str
    commit_hash: str
    branch: str
    timestamp: datetime
    environment: str
    build_url: Optional[str] = None
    pull_request: Optional[str] = None

@dataclass
class PerformanceGate:
    """Performance gate configuration"""
    name: str
    test_pattern: str
    max_time_ms: float
    max_regression_percent: float
    min_samples_for_comparison: int = 5
    fail_on_regression: bool = True
    warning_threshold_percent: float = 15.0
    enabled: bool = True

@dataclass
class GateResult:
    """Performance gate result"""
    gate_name: str
    passed: bool
    test_results: List[Dict] = field(default_factory=list)
    failure_reason: Optional[str] = None
    recommendation: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class BisectionResult:
    """Performance bisection result"""
    regression_introduced_at: str
    regression_fixed_at: Optional[str]
    commits_analyzed: int
    performance_impact: float
    root_cause_commits: List[str]
    analysis_time: float

class CIPerformanceGates:
    """
    CI/CD Performance Gates system with automated build comparison
    and performance bisection for regression root cause analysis
    """
    
    def __init__(self, db_path: str = "ci_performance.db"):
        self.db_path = db_path
        self.detector = performance_detector
        self.gates = {}
        self.build_cache = {}
        self._init_database()
        self._setup_default_gates()
        
        # CI/CD environment detection
        self.ci_environment = self._detect_ci_environment()
        
        logger.info("CIPerformanceGates initialized", 
                   ci_environment=self.ci_environment)
    
    def _init_database(self):
        """Initialize database for CI performance data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS build_performance (
                build_id TEXT NOT NULL,
                commit_hash TEXT NOT NULL,
                branch TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                test_name TEXT NOT NULL,
                mean_time REAL NOT NULL,
                median_time REAL NOT NULL,
                min_time REAL NOT NULL,
                max_time REAL NOT NULL,
                stddev_time REAL NOT NULL,
                rounds INTEGER NOT NULL,
                iterations INTEGER NOT NULL,
                environment TEXT NOT NULL,
                build_url TEXT,
                pull_request TEXT,
                PRIMARY KEY (build_id, test_name)
            )
        """)
        
        # Performance gates table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_gates (
                name TEXT PRIMARY KEY,
                test_pattern TEXT NOT NULL,
                max_time_ms REAL NOT NULL,
                max_regression_percent REAL NOT NULL,
                min_samples_for_comparison INTEGER NOT NULL,
                fail_on_regression BOOLEAN NOT NULL,
                warning_threshold_percent REAL NOT NULL,
                enabled BOOLEAN NOT NULL
            )
        """)
        
        # Gate results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS gate_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                build_id TEXT NOT NULL,
                gate_name TEXT NOT NULL,
                passed BOOLEAN NOT NULL,
                failure_reason TEXT,
                recommendation TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        
        # Bisection results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bisection_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT NOT NULL,
                regression_introduced_at TEXT NOT NULL,
                regression_fixed_at TEXT,
                commits_analyzed INTEGER NOT NULL,
                performance_impact REAL NOT NULL,
                root_cause_commits TEXT NOT NULL,
                analysis_time REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _setup_default_gates(self):
        """Setup default performance gates"""
        gates = [
            PerformanceGate(
                name="critical_var_performance",
                test_pattern="test_var_calculation*",
                max_time_ms=5.0,
                max_regression_percent=20.0,
                fail_on_regression=True
            ),
            PerformanceGate(
                name="correlation_performance",
                test_pattern="test_correlation*",
                max_time_ms=2.0,
                max_regression_percent=25.0,
                fail_on_regression=True
            ),
            PerformanceGate(
                name="agent_inference_performance",
                test_pattern="test_*_agent_inference",
                max_time_ms=100.0,
                max_regression_percent=30.0,
                fail_on_regression=False,
                warning_threshold_percent=20.0
            ),
            PerformanceGate(
                name="api_response_performance",
                test_pattern="test_api_*",
                max_time_ms=200.0,
                max_regression_percent=40.0,
                fail_on_regression=False,
                warning_threshold_percent=25.0
            ),
            PerformanceGate(
                name="matrix_assembly_performance",
                test_pattern="test_matrix_*",
                max_time_ms=10.0,
                max_regression_percent=20.0,
                fail_on_regression=True
            )
        ]
        
        for gate in gates:
            self.add_gate(gate)
    
    def _detect_ci_environment(self) -> str:
        """Detect CI/CD environment"""
        if os.getenv('GITHUB_ACTIONS'):
            return 'github_actions'
        elif os.getenv('GITLAB_CI'):
            return 'gitlab_ci'
        elif os.getenv('JENKINS_URL'):
            return 'jenkins'
        elif os.getenv('BUILDKITE'):
            return 'buildkite'
        elif os.getenv('CIRCLECI'):
            return 'circleci'
        elif os.getenv('TRAVIS'):
            return 'travis'
        else:
            return 'unknown'
    
    def add_gate(self, gate: PerformanceGate):
        """Add performance gate"""
        self.gates[gate.name] = gate
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO performance_gates 
            (name, test_pattern, max_time_ms, max_regression_percent, 
             min_samples_for_comparison, fail_on_regression, 
             warning_threshold_percent, enabled)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            gate.name,
            gate.test_pattern,
            gate.max_time_ms,
            gate.max_regression_percent,
            gate.min_samples_for_comparison,
            gate.fail_on_regression,
            gate.warning_threshold_percent,
            gate.enabled
        ))
        
        conn.commit()
        conn.close()
        
        logger.info("Performance gate added", gate_name=gate.name)
    
    def get_current_build_info(self) -> BuildInfo:
        """Get current build information from CI environment"""
        if self.ci_environment == 'github_actions':
            return BuildInfo(
                build_id=os.getenv('GITHUB_RUN_ID', 'unknown'),
                commit_hash=os.getenv('GITHUB_SHA', 'unknown'),
                branch=os.getenv('GITHUB_REF_NAME', 'unknown'),
                timestamp=datetime.now(),
                environment='github_actions',
                build_url=f"https://github.com/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/{os.getenv('GITHUB_RUN_ID')}",
                pull_request=os.getenv('GITHUB_EVENT_NUMBER')
            )
        elif self.ci_environment == 'gitlab_ci':
            return BuildInfo(
                build_id=os.getenv('CI_PIPELINE_ID', 'unknown'),
                commit_hash=os.getenv('CI_COMMIT_SHA', 'unknown'),
                branch=os.getenv('CI_COMMIT_REF_NAME', 'unknown'),
                timestamp=datetime.now(),
                environment='gitlab_ci',
                build_url=os.getenv('CI_PIPELINE_URL'),
                pull_request=os.getenv('CI_MERGE_REQUEST_ID')
            )
        else:
            # Default/local build
            commit_hash = 'unknown'
            branch = 'unknown'
            
            try:
                commit_hash = subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'],
                    text=True
                ).strip()
                branch = subprocess.check_output(
                    ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                    text=True
                ).strip()
            except subprocess.CalledProcessError:
                pass
            
            return BuildInfo(
                build_id=f"local_{int(datetime.now().timestamp())}",
                commit_hash=commit_hash,
                branch=branch,
                timestamp=datetime.now(),
                environment='local'
            )
    
    def record_build_performance(self, build_info: BuildInfo, benchmarks: List[PerformanceBenchmark]):
        """Record performance data for a build"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for benchmark in benchmarks:
            cursor.execute("""
                INSERT OR REPLACE INTO build_performance 
                (build_id, commit_hash, branch, timestamp, test_name, 
                 mean_time, median_time, min_time, max_time, stddev_time, 
                 rounds, iterations, environment, build_url, pull_request)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                build_info.build_id,
                build_info.commit_hash,
                build_info.branch,
                build_info.timestamp.isoformat(),
                benchmark.test_name,
                benchmark.mean_time,
                benchmark.median_time,
                benchmark.min_time,
                benchmark.max_time,
                benchmark.stddev_time,
                benchmark.rounds,
                benchmark.iterations,
                build_info.environment,
                build_info.build_url,
                build_info.pull_request
            ))
        
        conn.commit()
        conn.close()
        
        logger.info("Build performance recorded", 
                   build_id=build_info.build_id,
                   benchmarks=len(benchmarks))
    
    async def run_performance_gates(self, build_info: BuildInfo, benchmarks: List[PerformanceBenchmark]) -> List[GateResult]:
        """Run performance gates for a build"""
        results = []
        
        # Record build performance first
        self.record_build_performance(build_info, benchmarks)
        
        # Run each gate
        for gate_name, gate in self.gates.items():
            if not gate.enabled:
                continue
            
            result = await self._run_single_gate(gate, build_info, benchmarks)
            results.append(result)
            
            # Record gate result
            self._record_gate_result(build_info, result)
        
        return results
    
    async def _run_single_gate(self, gate: PerformanceGate, build_info: BuildInfo, benchmarks: List[PerformanceBenchmark]) -> GateResult:
        """Run a single performance gate"""
        # Find matching benchmarks
        matching_benchmarks = [
            b for b in benchmarks 
            if self._matches_pattern(b.test_name, gate.test_pattern)
        ]
        
        if not matching_benchmarks:
            return GateResult(
                gate_name=gate.name,
                passed=True,
                failure_reason="No matching tests found"
            )
        
        gate_passed = True
        test_results = []
        failure_reasons = []
        
        for benchmark in matching_benchmarks:
            test_result = {
                'test_name': benchmark.test_name,
                'mean_time': benchmark.mean_time,
                'passed': True,
                'issues': []
            }
            
            # Check absolute time threshold
            if benchmark.mean_time > gate.max_time_ms / 1000:  # Convert ms to s
                test_result['passed'] = False
                test_result['issues'].append(f"Exceeds time threshold: {benchmark.mean_time:.4f}s > {gate.max_time_ms/1000:.4f}s")
                if gate.fail_on_regression:
                    gate_passed = False
                    failure_reasons.append(f"{benchmark.test_name} exceeds time threshold")
            
            # Check regression against baseline
            baseline_performance = await self._get_baseline_performance(benchmark.test_name, build_info.branch)
            if baseline_performance:
                regression_percent = ((benchmark.mean_time - baseline_performance) / baseline_performance) * 100
                
                if regression_percent > gate.max_regression_percent:
                    test_result['passed'] = False
                    test_result['issues'].append(f"Performance regression: {regression_percent:.1f}% > {gate.max_regression_percent:.1f}%")
                    if gate.fail_on_regression:
                        gate_passed = False
                        failure_reasons.append(f"{benchmark.test_name} has {regression_percent:.1f}% regression")
                elif regression_percent > gate.warning_threshold_percent:
                    test_result['issues'].append(f"Performance warning: {regression_percent:.1f}% regression")
                
                test_result['baseline_performance'] = baseline_performance
                test_result['regression_percent'] = regression_percent
            
            test_results.append(test_result)
        
        # Generate recommendation
        recommendation = self._generate_gate_recommendation(gate, test_results)
        
        return GateResult(
            gate_name=gate.name,
            passed=gate_passed,
            test_results=test_results,
            failure_reason='; '.join(failure_reasons) if failure_reasons else None,
            recommendation=recommendation
        )
    
    def _matches_pattern(self, test_name: str, pattern: str) -> bool:
        """Check if test name matches pattern"""
        import fnmatch
        return fnmatch.fnmatch(test_name, pattern)
    
    async def _get_baseline_performance(self, test_name: str, branch: str) -> Optional[float]:
        """Get baseline performance for comparison"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent successful builds from main branches
        main_branches = ['main', 'master', 'production', 'release']
        if branch not in main_branches:
            # For feature branches, compare against main
            compare_branches = main_branches
        else:
            # For main branches, compare against recent history
            compare_branches = [branch]
        
        cursor.execute("""
            SELECT mean_time FROM build_performance 
            WHERE test_name = ? AND branch IN ({})
            ORDER BY timestamp DESC 
            LIMIT 10
        """.format(','.join(['?' for _ in compare_branches])), 
        [test_name] + compare_branches)
        
        results = cursor.fetchall()
        conn.close()
        
        if len(results) >= 3:
            # Use median of recent runs
            import statistics
            return statistics.median([r[0] for r in results])
        
        return None
    
    def _generate_gate_recommendation(self, gate: PerformanceGate, test_results: List[Dict]) -> str:
        """Generate recommendation for gate result"""
        recommendations = []
        
        failed_tests = [t for t in test_results if not t['passed']]
        warning_tests = [t for t in test_results if t['passed'] and t.get('issues')]
        
        if failed_tests:
            recommendations.append(f"FAILED: {len(failed_tests)} tests failed performance gates")
            for test in failed_tests:
                recommendations.append(f"- {test['test_name']}: {', '.join(test['issues'])}")
        
        if warning_tests:
            recommendations.append(f"WARNING: {len(warning_tests)} tests have performance warnings")
        
        if not failed_tests and not warning_tests:
            recommendations.append("All tests passed performance gates successfully")
        
        return '\n'.join(recommendations)
    
    def _record_gate_result(self, build_info: BuildInfo, result: GateResult):
        """Record gate result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO gate_results 
            (build_id, gate_name, passed, failure_reason, recommendation, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            build_info.build_id,
            result.gate_name,
            result.passed,
            result.failure_reason,
            result.recommendation,
            result.timestamp.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    async def run_performance_bisection(self, test_name: str, start_commit: str, end_commit: str) -> BisectionResult:
        """Run performance bisection to find regression root cause"""
        logger.info("Starting performance bisection", 
                   test_name=test_name,
                   start_commit=start_commit,
                   end_commit=end_commit)
        
        start_time = datetime.now()
        
        # Get commit list
        commits = self._get_commit_range(start_commit, end_commit)
        if len(commits) < 2:
            raise ValueError("Need at least 2 commits for bisection")
        
        # Binary search for regression
        left, right = 0, len(commits) - 1
        regression_commit = None
        commits_analyzed = 0
        
        while left < right:
            mid = (left + right) // 2
            commit = commits[mid]
            
            # Test performance at this commit
            performance = await self._test_performance_at_commit(test_name, commit)
            commits_analyzed += 1
            
            if performance is None:
                # Skip this commit if we can't test it
                left = mid + 1
                continue
            
            # Get baseline performance
            baseline = await self._get_baseline_performance(test_name, 'main')
            if baseline is None:
                # Use first commit as baseline
                baseline = await self._test_performance_at_commit(test_name, commits[0])
            
            if baseline and performance > baseline * 1.1:  # 10% threshold
                # Regression found, search earlier
                regression_commit = commit
                right = mid
            else:
                # No regression, search later
                left = mid + 1
        
        # Calculate performance impact
        start_performance = await self._test_performance_at_commit(test_name, start_commit)
        end_performance = await self._test_performance_at_commit(test_name, end_commit)
        
        performance_impact = 0.0
        if start_performance and end_performance:
            performance_impact = ((end_performance - start_performance) / start_performance) * 100
        
        analysis_time = (datetime.now() - start_time).total_seconds()
        
        result = BisectionResult(
            regression_introduced_at=regression_commit or end_commit,
            regression_fixed_at=None,
            commits_analyzed=commits_analyzed,
            performance_impact=performance_impact,
            root_cause_commits=[regression_commit] if regression_commit else [],
            analysis_time=analysis_time
        )
        
        # Record bisection result
        self._record_bisection_result(test_name, result)
        
        logger.info("Performance bisection completed",
                   test_name=test_name,
                   regression_commit=regression_commit,
                   commits_analyzed=commits_analyzed,
                   performance_impact=performance_impact)
        
        return result
    
    def _get_commit_range(self, start_commit: str, end_commit: str) -> List[str]:
        """Get list of commits between start and end"""
        try:
            result = subprocess.run(
                ['git', 'rev-list', '--reverse', f'{start_commit}..{end_commit}'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip().split('\n')
        except subprocess.CalledProcessError:
            return []
    
    async def _test_performance_at_commit(self, test_name: str, commit: str) -> Optional[float]:
        """Test performance at a specific commit"""
        # This is a simplified version - in practice, you'd need to:
        # 1. Checkout the commit
        # 2. Run the specific test
        # 3. Extract performance metrics
        # 4. Restore original state
        
        # For now, return simulated data from database if available
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT mean_time FROM build_performance 
            WHERE test_name = ? AND commit_hash = ?
        """, (test_name, commit))
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
    
    def _record_bisection_result(self, test_name: str, result: BisectionResult):
        """Record bisection result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO bisection_results 
            (test_name, regression_introduced_at, regression_fixed_at, 
             commits_analyzed, performance_impact, root_cause_commits, 
             analysis_time, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            test_name,
            result.regression_introduced_at,
            result.regression_fixed_at,
            result.commits_analyzed,
            result.performance_impact,
            json.dumps(result.root_cause_commits),
            result.analysis_time,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_build_comparison_report(self, build_id: str, comparison_builds: int = 5) -> Dict:
        """Generate build comparison report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current build data
        cursor.execute("""
            SELECT * FROM build_performance 
            WHERE build_id = ?
        """, (build_id,))
        
        current_build = cursor.fetchall()
        
        if not current_build:
            return {"error": "Build not found"}
        
        # Get comparison builds
        cursor.execute("""
            SELECT DISTINCT build_id, commit_hash, branch, timestamp 
            FROM build_performance 
            WHERE build_id != ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (build_id, comparison_builds))
        
        comparison_build_info = cursor.fetchall()
        
        # Compare performance
        comparisons = []
        for comp_build_id, comp_commit, comp_branch, comp_timestamp in comparison_build_info:
            cursor.execute("""
                SELECT test_name, mean_time FROM build_performance 
                WHERE build_id = ?
            """, (comp_build_id,))
            
            comp_results = dict(cursor.fetchall())
            
            # Calculate differences
            test_comparisons = []
            for current_row in current_build:
                test_name = current_row[4]  # test_name column
                current_time = current_row[5]  # mean_time column
                
                if test_name in comp_results:
                    comp_time = comp_results[test_name]
                    diff_percent = ((current_time - comp_time) / comp_time) * 100
                    
                    test_comparisons.append({
                        'test_name': test_name,
                        'current_time': current_time,
                        'comparison_time': comp_time,
                        'difference_percent': diff_percent,
                        'regression': diff_percent > 10
                    })
            
            comparisons.append({
                'build_id': comp_build_id,
                'commit_hash': comp_commit,
                'branch': comp_branch,
                'timestamp': comp_timestamp,
                'test_comparisons': test_comparisons
            })
        
        conn.close()
        
        return {
            'current_build_id': build_id,
            'comparison_count': len(comparisons),
            'comparisons': comparisons,
            'summary': {
                'total_tests': len(current_build),
                'regressions_found': sum(
                    len([t for t in comp['test_comparisons'] if t['regression']])
                    for comp in comparisons
                )
            }
        }

# Global instance
ci_gates = CIPerformanceGates()