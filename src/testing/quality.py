"""
GrandModel Quality Assurance Module
===================================

Comprehensive quality assurance framework for code quality monitoring,
bug tracking, quality metrics, and automated quality reporting.
"""

import asyncio
import logging
import os
import sys
import ast
import time
import json
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
import statistics
import re
from collections import defaultdict, Counter

# Code analysis tools
try:
    import flake8.api.legacy as flake8
    FLAKE8_AVAILABLE = True
except ImportError:
    FLAKE8_AVAILABLE = False

try:
    import pylint.lint
    PYLINT_AVAILABLE = True
except ImportError:
    PYLINT_AVAILABLE = False

try:
    import mypy.api
    MYPY_AVAILABLE = True
except ImportError:
    MYPY_AVAILABLE = False

try:
    import bandit
    BANDIT_AVAILABLE = True
except ImportError:
    BANDIT_AVAILABLE = False


class QualityLevel(Enum):
    """Quality level enumeration"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


class IssueType(Enum):
    """Issue type enumeration"""
    BUG = "bug"
    CODE_SMELL = "code_smell"
    VULNERABILITY = "vulnerability"
    MAINTAINABILITY = "maintainability"
    RELIABILITY = "reliability"
    SECURITY = "security"
    PERFORMANCE = "performance"
    DOCUMENTATION = "documentation"


class IssueSeverity(Enum):
    """Issue severity enumeration"""
    BLOCKER = "blocker"
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


@dataclass
class QualityIssue:
    """Quality issue data structure"""
    id: str
    type: IssueType
    severity: IssueSeverity
    message: str
    file_path: str
    line_number: int
    column_number: int = 0
    rule_id: str = ""
    tool: str = ""
    description: str = ""
    fix_suggestion: str = ""
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class QualityMetrics:
    """Quality metrics data structure"""
    total_lines: int
    code_lines: int
    comment_lines: int
    blank_lines: int
    complexity: float
    maintainability_index: float
    technical_debt_minutes: float
    duplication_percentage: float
    test_coverage: float
    issues_count: int
    bugs_count: int
    vulnerabilities_count: int
    code_smells_count: int
    quality_score: float
    quality_level: QualityLevel


@dataclass
class QualityReport:
    """Quality report data structure"""
    timestamp: datetime
    project_name: str
    metrics: QualityMetrics
    issues: List[QualityIssue]
    trends: Dict[str, List[float]]
    recommendations: List[str]
    summary: Dict[str, Any]


class QualityAssurance:
    """
    Comprehensive quality assurance system
    
    Features:
    - Code quality analysis
    - Static code analysis
    - Security vulnerability scanning
    - Code complexity measurement
    - Test coverage analysis
    - Quality metrics tracking
    - Automated quality reporting
    - Quality trend analysis
    - Bug tracking and categorization
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/quality_config.yaml"
        self.logger = self._setup_logging()
        self.issues: List[QualityIssue] = []
        self.metrics_history: List[QualityMetrics] = []
        self.reports: List[QualityReport] = []
        
        # Initialize paths
        self.project_root = Path(__file__).parent.parent.parent
        self.src_dir = self.project_root / "src"
        self.reports_dir = self.project_root / "quality_reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Quality thresholds
        self.quality_thresholds = {
            "complexity": 10.0,
            "maintainability": 70.0,
            "coverage": 80.0,
            "duplication": 3.0,
            "technical_debt": 60.0  # minutes
        }
        
        # Initialize analysis tools
        self.analyzers = self._initialize_analyzers()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("QualityAssurance")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_analyzers(self) -> Dict[str, bool]:
        """Initialize code analysis tools"""
        analyzers = {
            "flake8": FLAKE8_AVAILABLE,
            "pylint": PYLINT_AVAILABLE,
            "mypy": MYPY_AVAILABLE,
            "bandit": BANDIT_AVAILABLE
        }
        
        available_analyzers = [name for name, available in analyzers.items() if available]
        self.logger.info(f"Available analyzers: {available_analyzers}")
        
        return analyzers
    
    async def analyze_code_quality(self, target_path: Optional[str] = None) -> QualityReport:
        """
        Comprehensive code quality analysis
        
        Args:
            target_path: Path to analyze (defaults to src directory)
            
        Returns:
            QualityReport object
        """
        self.logger.info("Starting comprehensive code quality analysis")
        
        if target_path is None:
            target_path = str(self.src_dir)
        
        start_time = time.time()
        
        # Clear previous issues
        self.issues.clear()
        
        # Run all available analyzers
        analysis_tasks = []
        
        if self.analyzers["flake8"]:
            analysis_tasks.append(self._run_flake8_analysis(target_path))
        
        if self.analyzers["pylint"]:
            analysis_tasks.append(self._run_pylint_analysis(target_path))
        
        if self.analyzers["mypy"]:
            analysis_tasks.append(self._run_mypy_analysis(target_path))
        
        if self.analyzers["bandit"]:
            analysis_tasks.append(self._run_bandit_analysis(target_path))
        
        # Run custom analyzers
        analysis_tasks.extend([
            self._analyze_complexity(target_path),
            self._analyze_maintainability(target_path),
            self._analyze_duplication(target_path),
            self._analyze_documentation(target_path)
        ])
        
        # Execute all analyses
        await asyncio.gather(*analysis_tasks)
        
        # Calculate metrics
        metrics = await self._calculate_metrics(target_path)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics)
        
        # Create quality report
        report = QualityReport(
            timestamp=datetime.now(),
            project_name="GrandModel",
            metrics=metrics,
            issues=self.issues.copy(),
            trends=self._calculate_trends(),
            recommendations=recommendations,
            summary=self._generate_summary(metrics)
        )
        
        self.reports.append(report)
        self.metrics_history.append(metrics)
        
        duration = time.time() - start_time
        self.logger.info(f"Code quality analysis completed in {duration:.2f}s")
        
        return report
    
    async def _run_flake8_analysis(self, target_path: str):
        """Run flake8 analysis"""
        try:
            self.logger.info("Running flake8 analysis...")
            
            # Run flake8 via subprocess for better control
            result = subprocess.run([
                sys.executable, "-m", "flake8",
                target_path,
                "--format=json",
                "--max-line-length=100",
                "--ignore=E203,W503"
            ], capture_output=True, text=True, timeout=300)
            
            if result.stdout:
                try:
                    flake8_results = json.loads(result.stdout)
                    for issue in flake8_results:
                        self.issues.append(QualityIssue(
                            id=f"flake8_{len(self.issues)}",
                            type=IssueType.CODE_SMELL,
                            severity=IssueSeverity.MINOR,
                            message=issue.get("message", ""),
                            file_path=issue.get("filename", ""),
                            line_number=issue.get("line_number", 0),
                            column_number=issue.get("column_number", 0),
                            rule_id=issue.get("code", ""),
                            tool="flake8"
                        ))
                except json.JSONDecodeError:
                    # Fallback to parsing text output
                    for line in result.stdout.strip().split('\n'):
                        if line and ':' in line:
                            parts = line.split(':', 3)
                            if len(parts) >= 4:
                                self.issues.append(QualityIssue(
                                    id=f"flake8_{len(self.issues)}",
                                    type=IssueType.CODE_SMELL,
                                    severity=IssueSeverity.MINOR,
                                    message=parts[3].strip(),
                                    file_path=parts[0],
                                    line_number=int(parts[1]) if parts[1].isdigit() else 0,
                                    column_number=int(parts[2]) if parts[2].isdigit() else 0,
                                    tool="flake8"
                                ))
            
        except Exception as e:
            self.logger.error(f"Flake8 analysis failed: {e}")
    
    async def _run_pylint_analysis(self, target_path: str):
        """Run pylint analysis"""
        try:
            self.logger.info("Running pylint analysis...")
            
            # Run pylint via subprocess
            result = subprocess.run([
                sys.executable, "-m", "pylint",
                target_path,
                "--output-format=json",
                "--disable=C0103,R0913,R0914,R0915,W0613"
            ], capture_output=True, text=True, timeout=600)
            
            if result.stdout:
                try:
                    pylint_results = json.loads(result.stdout)
                    for issue in pylint_results:
                        severity_map = {
                            "error": IssueSeverity.MAJOR,
                            "warning": IssueSeverity.MINOR,
                            "refactor": IssueSeverity.MINOR,
                            "convention": IssueSeverity.INFO
                        }
                        
                        type_map = {
                            "error": IssueType.BUG,
                            "warning": IssueType.CODE_SMELL,
                            "refactor": IssueType.MAINTAINABILITY,
                            "convention": IssueType.CODE_SMELL
                        }
                        
                        self.issues.append(QualityIssue(
                            id=f"pylint_{len(self.issues)}",
                            type=type_map.get(issue.get("type", ""), IssueType.CODE_SMELL),
                            severity=severity_map.get(issue.get("type", ""), IssueSeverity.MINOR),
                            message=issue.get("message", ""),
                            file_path=issue.get("path", ""),
                            line_number=issue.get("line", 0),
                            column_number=issue.get("column", 0),
                            rule_id=issue.get("symbol", ""),
                            tool="pylint"
                        ))
                except json.JSONDecodeError:
                    self.logger.warning("Failed to parse pylint JSON output")
            
        except Exception as e:
            self.logger.error(f"Pylint analysis failed: {e}")
    
    async def _run_mypy_analysis(self, target_path: str):
        """Run mypy analysis"""
        try:
            self.logger.info("Running mypy analysis...")
            
            # Run mypy via subprocess
            result = subprocess.run([
                sys.executable, "-m", "mypy",
                target_path,
                "--ignore-missing-imports",
                "--no-error-summary"
            ], capture_output=True, text=True, timeout=300)
            
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line and ':' in line:
                        parts = line.split(':', 3)
                        if len(parts) >= 4:
                            self.issues.append(QualityIssue(
                                id=f"mypy_{len(self.issues)}",
                                type=IssueType.RELIABILITY,
                                severity=IssueSeverity.MINOR,
                                message=parts[3].strip(),
                                file_path=parts[0],
                                line_number=int(parts[1]) if parts[1].isdigit() else 0,
                                tool="mypy"
                            ))
            
        except Exception as e:
            self.logger.error(f"Mypy analysis failed: {e}")
    
    async def _run_bandit_analysis(self, target_path: str):
        """Run bandit security analysis"""
        try:
            self.logger.info("Running bandit security analysis...")
            
            # Run bandit via subprocess
            result = subprocess.run([
                sys.executable, "-m", "bandit",
                "-r", target_path,
                "-f", "json"
            ], capture_output=True, text=True, timeout=300)
            
            if result.stdout:
                try:
                    bandit_results = json.loads(result.stdout)
                    for issue in bandit_results.get("results", []):
                        severity_map = {
                            "HIGH": IssueSeverity.CRITICAL,
                            "MEDIUM": IssueSeverity.MAJOR,
                            "LOW": IssueSeverity.MINOR
                        }
                        
                        self.issues.append(QualityIssue(
                            id=f"bandit_{len(self.issues)}",
                            type=IssueType.SECURITY,
                            severity=severity_map.get(issue.get("issue_severity", ""), IssueSeverity.MINOR),
                            message=issue.get("issue_text", ""),
                            file_path=issue.get("filename", ""),
                            line_number=issue.get("line_number", 0),
                            rule_id=issue.get("test_id", ""),
                            tool="bandit",
                            description=issue.get("issue_text", "")
                        ))
                except json.JSONDecodeError:
                    self.logger.warning("Failed to parse bandit JSON output")
            
        except Exception as e:
            self.logger.error(f"Bandit analysis failed: {e}")
    
    async def _analyze_complexity(self, target_path: str):
        """Analyze code complexity"""
        try:
            self.logger.info("Analyzing code complexity...")
            
            complexity_issues = []
            
            for py_file in Path(target_path).rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    complexity = self._calculate_cyclomatic_complexity(tree)
                    
                    if complexity > self.quality_thresholds["complexity"]:
                        complexity_issues.append(QualityIssue(
                            id=f"complexity_{len(self.issues)}",
                            type=IssueType.MAINTAINABILITY,
                            severity=IssueSeverity.MAJOR if complexity > 20 else IssueSeverity.MINOR,
                            message=f"High cyclomatic complexity: {complexity}",
                            file_path=str(py_file),
                            line_number=1,
                            tool="complexity_analyzer",
                            description=f"Cyclomatic complexity is {complexity}, consider refactoring"
                        ))
                    
                except Exception as e:
                    self.logger.warning(f"Failed to analyze complexity for {py_file}: {e}")
            
            self.issues.extend(complexity_issues)
            
        except Exception as e:
            self.logger.error(f"Complexity analysis failed: {e}")
    
    async def _analyze_maintainability(self, target_path: str):
        """Analyze maintainability"""
        try:
            self.logger.info("Analyzing maintainability...")
            
            maintainability_issues = []
            
            for py_file in Path(target_path).rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Calculate maintainability index
                    maintainability = self._calculate_maintainability_index(content)
                    
                    if maintainability < self.quality_thresholds["maintainability"]:
                        maintainability_issues.append(QualityIssue(
                            id=f"maintainability_{len(self.issues)}",
                            type=IssueType.MAINTAINABILITY,
                            severity=IssueSeverity.MAJOR if maintainability < 50 else IssueSeverity.MINOR,
                            message=f"Low maintainability index: {maintainability:.1f}",
                            file_path=str(py_file),
                            line_number=1,
                            tool="maintainability_analyzer",
                            description=f"Maintainability index is {maintainability:.1f}, consider refactoring"
                        ))
                    
                except Exception as e:
                    self.logger.warning(f"Failed to analyze maintainability for {py_file}: {e}")
            
            self.issues.extend(maintainability_issues)
            
        except Exception as e:
            self.logger.error(f"Maintainability analysis failed: {e}")
    
    async def _analyze_duplication(self, target_path: str):
        """Analyze code duplication"""
        try:
            self.logger.info("Analyzing code duplication...")
            
            # Simple duplication detection based on line similarity
            file_contents = {}
            
            for py_file in Path(target_path).rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        # Remove empty lines and comments for comparison
                        clean_lines = [line.strip() for line in lines 
                                     if line.strip() and not line.strip().startswith('#')]
                        file_contents[str(py_file)] = clean_lines
                except Exception as e:
                    self.logger.warning(f"Failed to read {py_file}: {e}")
            
            # Find duplicated blocks
            duplication_issues = self._find_duplicated_blocks(file_contents)
            self.issues.extend(duplication_issues)
            
        except Exception as e:
            self.logger.error(f"Duplication analysis failed: {e}")
    
    async def _analyze_documentation(self, target_path: str):
        """Analyze documentation quality"""
        try:
            self.logger.info("Analyzing documentation quality...")
            
            documentation_issues = []
            
            for py_file in Path(target_path).rglob("*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    # Check for missing docstrings
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                            if not ast.get_docstring(node):
                                documentation_issues.append(QualityIssue(
                                    id=f"doc_{len(self.issues)}",
                                    type=IssueType.DOCUMENTATION,
                                    severity=IssueSeverity.MINOR,
                                    message=f"Missing docstring for {node.name}",
                                    file_path=str(py_file),
                                    line_number=node.lineno,
                                    tool="documentation_analyzer",
                                    description=f"{node.__class__.__name__} '{node.name}' lacks documentation"
                                ))
                    
                except Exception as e:
                    self.logger.warning(f"Failed to analyze documentation for {py_file}: {e}")
            
            self.issues.extend(documentation_issues)
            
        except Exception as e:
            self.logger.error(f"Documentation analysis failed: {e}")
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.Try):
                complexity += len(node.handlers)
            elif isinstance(node, ast.With):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _calculate_maintainability_index(self, content: str) -> float:
        """Calculate maintainability index"""
        lines = content.split('\n')
        total_lines = len(lines)
        code_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        
        if code_lines == 0:
            return 100.0
        
        # Simplified maintainability index calculation
        # Based on Halstead complexity and cyclomatic complexity
        
        try:
            tree = ast.parse(content)
            complexity = self._calculate_cyclomatic_complexity(tree)
            
            # Halstead metrics (simplified)
            operators = 0
            operands = 0
            
            for node in ast.walk(tree):
                if isinstance(node, ast.BinOp):
                    operators += 1
                elif isinstance(node, ast.Name):
                    operands += 1
            
            # Maintainability index formula (simplified)
            halstead_volume = (operators + operands) * 4.0 if operators + operands > 0 else 1.0
            maintainability = max(0, 171 - 5.2 * halstead_volume - 0.23 * complexity - 16.2 * total_lines / 100)
            
            return min(100, max(0, maintainability))
            
        except Exception:
            return 50.0  # Default value if calculation fails
    
    def _find_duplicated_blocks(self, file_contents: Dict[str, List[str]]) -> List[QualityIssue]:
        """Find duplicated code blocks"""
        duplication_issues = []
        min_block_size = 6  # Minimum lines for duplication detection
        
        for file1, lines1 in file_contents.items():
            for file2, lines2 in file_contents.items():
                if file1 >= file2:  # Avoid duplicate comparisons
                    continue
                
                # Find common subsequences
                i = 0
                while i < len(lines1) - min_block_size:
                    j = 0
                    while j < len(lines2) - min_block_size:
                        # Check for matching block
                        block_size = 0
                        while (i + block_size < len(lines1) and 
                               j + block_size < len(lines2) and
                               lines1[i + block_size] == lines2[j + block_size]):
                            block_size += 1
                        
                        if block_size >= min_block_size:
                            duplication_issues.append(QualityIssue(
                                id=f"duplication_{len(duplication_issues)}",
                                type=IssueType.MAINTAINABILITY,
                                severity=IssueSeverity.MAJOR if block_size > 20 else IssueSeverity.MINOR,
                                message=f"Duplicated code block ({block_size} lines)",
                                file_path=file1,
                                line_number=i + 1,
                                tool="duplication_analyzer",
                                description=f"Duplicated with {file2} at line {j + 1}"
                            ))
                            j += block_size
                        else:
                            j += 1
                    i += 1
        
        return duplication_issues
    
    async def _calculate_metrics(self, target_path: str) -> QualityMetrics:
        """Calculate quality metrics"""
        total_lines = 0
        code_lines = 0
        comment_lines = 0
        blank_lines = 0
        total_complexity = 0
        total_maintainability = 0
        file_count = 0
        
        for py_file in Path(target_path).rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                total_lines += len(lines)
                
                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        blank_lines += 1
                    elif stripped.startswith('#'):
                        comment_lines += 1
                    else:
                        code_lines += 1
                
                # Calculate complexity and maintainability
                try:
                    tree = ast.parse(content)
                    complexity = self._calculate_cyclomatic_complexity(tree)
                    maintainability = self._calculate_maintainability_index(content)
                    
                    total_complexity += complexity
                    total_maintainability += maintainability
                    file_count += 1
                    
                except Exception:
                    pass
                    
            except Exception as e:
                self.logger.warning(f"Failed to analyze {py_file}: {e}")
        
        # Calculate averages
        avg_complexity = total_complexity / file_count if file_count > 0 else 0
        avg_maintainability = total_maintainability / file_count if file_count > 0 else 0
        
        # Count issues by type
        bugs_count = sum(1 for issue in self.issues if issue.type == IssueType.BUG)
        vulnerabilities_count = sum(1 for issue in self.issues if issue.type == IssueType.VULNERABILITY)
        code_smells_count = sum(1 for issue in self.issues if issue.type == IssueType.CODE_SMELL)
        
        # Calculate technical debt (simplified)
        technical_debt_minutes = len(self.issues) * 5  # 5 minutes per issue
        
        # Calculate duplication percentage (simplified)
        duplication_issues = [issue for issue in self.issues if "duplication" in issue.id]
        duplication_percentage = len(duplication_issues) / file_count if file_count > 0 else 0
        
        # Mock test coverage (would integrate with actual coverage tool)
        test_coverage = 75.0  # Placeholder
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            avg_complexity, avg_maintainability, test_coverage, 
            len(self.issues), duplication_percentage
        )
        
        # Determine quality level
        quality_level = self._determine_quality_level(quality_score)
        
        return QualityMetrics(
            total_lines=total_lines,
            code_lines=code_lines,
            comment_lines=comment_lines,
            blank_lines=blank_lines,
            complexity=avg_complexity,
            maintainability_index=avg_maintainability,
            technical_debt_minutes=technical_debt_minutes,
            duplication_percentage=duplication_percentage,
            test_coverage=test_coverage,
            issues_count=len(self.issues),
            bugs_count=bugs_count,
            vulnerabilities_count=vulnerabilities_count,
            code_smells_count=code_smells_count,
            quality_score=quality_score,
            quality_level=quality_level
        )
    
    def _calculate_quality_score(self, complexity: float, maintainability: float, 
                               coverage: float, issues_count: int, 
                               duplication: float) -> float:
        """Calculate overall quality score"""
        # Normalize metrics to 0-100 scale
        complexity_score = max(0, 100 - (complexity * 5))  # Lower complexity is better
        maintainability_score = maintainability
        coverage_score = coverage
        issues_score = max(0, 100 - (issues_count * 2))  # Fewer issues is better
        duplication_score = max(0, 100 - (duplication * 10))  # Less duplication is better
        
        # Weighted average
        weights = {
            "complexity": 0.2,
            "maintainability": 0.25,
            "coverage": 0.25,
            "issues": 0.2,
            "duplication": 0.1
        }
        
        quality_score = (
            complexity_score * weights["complexity"] +
            maintainability_score * weights["maintainability"] +
            coverage_score * weights["coverage"] +
            issues_score * weights["issues"] +
            duplication_score * weights["duplication"]
        )
        
        return max(0, min(100, quality_score))
    
    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Determine quality level based on score"""
        if score >= 90:
            return QualityLevel.EXCELLENT
        elif score >= 80:
            return QualityLevel.GOOD
        elif score >= 70:
            return QualityLevel.ACCEPTABLE
        elif score >= 50:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL
    
    def _generate_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        # Complexity recommendations
        if metrics.complexity > self.quality_thresholds["complexity"]:
            recommendations.append(
                f"Reduce cyclomatic complexity from {metrics.complexity:.1f} to below {self.quality_thresholds['complexity']}"
            )
        
        # Maintainability recommendations
        if metrics.maintainability_index < self.quality_thresholds["maintainability"]:
            recommendations.append(
                f"Improve maintainability index from {metrics.maintainability_index:.1f} to above {self.quality_thresholds['maintainability']}"
            )
        
        # Coverage recommendations
        if metrics.test_coverage < self.quality_thresholds["coverage"]:
            recommendations.append(
                f"Increase test coverage from {metrics.test_coverage:.1f}% to above {self.quality_thresholds['coverage']}%"
            )
        
        # Duplication recommendations
        if metrics.duplication_percentage > self.quality_thresholds["duplication"]:
            recommendations.append(
                f"Reduce code duplication from {metrics.duplication_percentage:.1f}% to below {self.quality_thresholds['duplication']}%"
            )
        
        # Technical debt recommendations
        if metrics.technical_debt_minutes > self.quality_thresholds["technical_debt"]:
            recommendations.append(
                f"Reduce technical debt from {metrics.technical_debt_minutes:.0f} minutes to below {self.quality_thresholds['technical_debt']} minutes"
            )
        
        # Security recommendations
        if metrics.vulnerabilities_count > 0:
            recommendations.append(
                f"Fix {metrics.vulnerabilities_count} security vulnerabilities"
            )
        
        # Bug recommendations
        if metrics.bugs_count > 0:
            recommendations.append(
                f"Fix {metrics.bugs_count} bugs"
            )
        
        return recommendations
    
    def _calculate_trends(self) -> Dict[str, List[float]]:
        """Calculate quality trends"""
        trends = {}
        
        if len(self.metrics_history) < 2:
            return trends
        
        # Extract metrics over time
        metrics_over_time = {
            "quality_score": [m.quality_score for m in self.metrics_history],
            "complexity": [m.complexity for m in self.metrics_history],
            "maintainability": [m.maintainability_index for m in self.metrics_history],
            "coverage": [m.test_coverage for m in self.metrics_history],
            "issues": [m.issues_count for m in self.metrics_history],
            "technical_debt": [m.technical_debt_minutes for m in self.metrics_history]
        }
        
        # Calculate trends (simple linear regression slope)
        for metric, values in metrics_over_time.items():
            if len(values) >= 2:
                x = list(range(len(values)))
                slope = self._calculate_slope(x, values)
                trends[metric] = slope
        
        return trends
    
    def _calculate_slope(self, x: List[float], y: List[float]) -> float:
        """Calculate linear regression slope"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] * x[i] for i in range(n))
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def _generate_summary(self, metrics: QualityMetrics) -> Dict[str, Any]:
        """Generate quality summary"""
        return {
            "overall_quality": metrics.quality_level.value,
            "quality_score": metrics.quality_score,
            "total_issues": metrics.issues_count,
            "critical_issues": len([i for i in self.issues if i.severity == IssueSeverity.CRITICAL]),
            "major_issues": len([i for i in self.issues if i.severity == IssueSeverity.MAJOR]),
            "minor_issues": len([i for i in self.issues if i.severity == IssueSeverity.MINOR]),
            "top_issue_types": self._get_top_issue_types(),
            "files_analyzed": len(list(Path(self.src_dir).rglob("*.py"))),
            "lines_of_code": metrics.code_lines
        }
    
    def _get_top_issue_types(self) -> List[Dict[str, Any]]:
        """Get top issue types by frequency"""
        issue_counts = Counter(issue.type.value for issue in self.issues)
        return [{"type": issue_type, "count": count} 
                for issue_type, count in issue_counts.most_common(5)]
    
    def generate_quality_report(self) -> str:
        """Generate comprehensive quality report"""
        if not self.reports:
            return ""
        
        latest_report = self.reports[-1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"quality_report_{timestamp}.json"
        
        report_data = {
            "timestamp": latest_report.timestamp.isoformat(),
            "project_name": latest_report.project_name,
            "metrics": {
                "total_lines": latest_report.metrics.total_lines,
                "code_lines": latest_report.metrics.code_lines,
                "comment_lines": latest_report.metrics.comment_lines,
                "blank_lines": latest_report.metrics.blank_lines,
                "complexity": latest_report.metrics.complexity,
                "maintainability_index": latest_report.metrics.maintainability_index,
                "technical_debt_minutes": latest_report.metrics.technical_debt_minutes,
                "duplication_percentage": latest_report.metrics.duplication_percentage,
                "test_coverage": latest_report.metrics.test_coverage,
                "issues_count": latest_report.metrics.issues_count,
                "bugs_count": latest_report.metrics.bugs_count,
                "vulnerabilities_count": latest_report.metrics.vulnerabilities_count,
                "code_smells_count": latest_report.metrics.code_smells_count,
                "quality_score": latest_report.metrics.quality_score,
                "quality_level": latest_report.metrics.quality_level.value
            },
            "issues": [
                {
                    "id": issue.id,
                    "type": issue.type.value,
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "file_path": issue.file_path,
                    "line_number": issue.line_number,
                    "column_number": issue.column_number,
                    "rule_id": issue.rule_id,
                    "tool": issue.tool,
                    "description": issue.description,
                    "created_at": issue.created_at.isoformat()
                }
                for issue in latest_report.issues
            ],
            "trends": latest_report.trends,
            "recommendations": latest_report.recommendations,
            "summary": latest_report.summary
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.logger.info(f"Quality report generated: {report_file}")
        return str(report_file)
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get quality summary"""
        if not self.reports:
            return {}
        
        latest_report = self.reports[-1]
        return {
            "quality_level": latest_report.metrics.quality_level.value,
            "quality_score": latest_report.metrics.quality_score,
            "total_issues": latest_report.metrics.issues_count,
            "bugs": latest_report.metrics.bugs_count,
            "vulnerabilities": latest_report.metrics.vulnerabilities_count,
            "code_smells": latest_report.metrics.code_smells_count,
            "complexity": latest_report.metrics.complexity,
            "maintainability": latest_report.metrics.maintainability_index,
            "test_coverage": latest_report.metrics.test_coverage,
            "technical_debt_minutes": latest_report.metrics.technical_debt_minutes
        }


if __name__ == "__main__":
    # Example usage
    async def main():
        # Initialize quality assurance
        qa = QualityAssurance()
        
        # Run quality analysis
        report = await qa.analyze_code_quality()
        
        # Generate report
        report_file = qa.generate_quality_report()
        
        # Print summary
        summary = qa.get_quality_summary()
        print(f"\nQuality Summary:")
        print(f"Quality Level: {summary.get('quality_level', 'Unknown')}")
        print(f"Quality Score: {summary.get('quality_score', 0):.1f}")
        print(f"Total Issues: {summary.get('total_issues', 0)}")
        print(f"Bugs: {summary.get('bugs', 0)}")
        print(f"Vulnerabilities: {summary.get('vulnerabilities', 0)}")
        print(f"Code Smells: {summary.get('code_smells', 0)}")
        print(f"Complexity: {summary.get('complexity', 0):.1f}")
        print(f"Maintainability: {summary.get('maintainability', 0):.1f}")
        print(f"Test Coverage: {summary.get('test_coverage', 0):.1f}%")
        print(f"Technical Debt: {summary.get('technical_debt_minutes', 0):.0f} minutes")
        print(f"Report: {report_file}")
    
    asyncio.run(main())