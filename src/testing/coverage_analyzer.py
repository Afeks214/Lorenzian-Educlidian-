"""
Advanced Test Coverage Reporting and Analysis System
Provides detailed coverage analysis, trends, and insights
"""

import json
import xml.etree.ElementTree as ET
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
import subprocess
import logging
import ast
import re
from dataclasses import dataclass, asdict
from enum import Enum
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import coverage
import statistics
import asyncio
from concurrent.futures import ThreadPoolExecutor


class CoverageType(Enum):
    """Types of coverage analysis"""
    LINE = "line"
    BRANCH = "branch"
    FUNCTION = "function"
    STATEMENT = "statement"


class CoverageQuality(Enum):
    """Coverage quality levels"""
    EXCELLENT = "excellent"  # 95-100%
    GOOD = "good"           # 80-94%
    FAIR = "fair"           # 60-79%
    POOR = "poor"           # 40-59%
    CRITICAL = "critical"   # <40%


@dataclass
class FileCoverage:
    """Coverage information for a single file"""
    file_path: str
    total_lines: int
    covered_lines: int
    missed_lines: int
    coverage_percentage: float
    branch_coverage: float
    function_coverage: float
    missing_line_ranges: List[Tuple[int, int]]
    covered_functions: List[str]
    uncovered_functions: List[str]
    complexity_score: float
    maintainability_index: float
    test_files: List[str]
    last_modified: datetime
    coverage_trend: List[float]
    hotspots: List[Dict[str, Any]]


@dataclass
class ModuleCoverage:
    """Coverage information for a module"""
    module_name: str
    total_files: int
    covered_files: int
    overall_coverage: float
    line_coverage: float
    branch_coverage: float
    function_coverage: float
    files: List[FileCoverage]
    test_count: int
    test_quality_score: float
    coverage_trend: List[float]
    recommendations: List[str]


@dataclass
class CoverageReport:
    """Complete coverage report"""
    report_id: str
    generated_at: datetime
    overall_coverage: float
    line_coverage: float
    branch_coverage: float
    function_coverage: float
    total_files: int
    covered_files: int
    total_lines: int
    covered_lines: int
    modules: List[ModuleCoverage]
    uncovered_hotspots: List[Dict[str, Any]]
    coverage_trends: Dict[str, List[float]]
    quality_metrics: Dict[str, float]
    recommendations: List[str]


class CoverageAnalyzer:
    """Advanced coverage analysis and reporting"""
    
    def __init__(self, 
                 source_dir: str = "src",
                 test_dir: str = "tests", 
                 output_dir: str = "coverage_reports"):
        self.source_dir = Path(source_dir)
        self.test_dir = Path(test_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.db_path = self.output_dir / "coverage_history.db"
        self._init_database()
        
        # Initialize coverage instance
        self.coverage = coverage.Coverage(
            source=[str(self.source_dir)],
            config_file=".coveragerc" if Path(".coveragerc").exists() else None
        )
    
    def _init_database(self):
        """Initialize database for coverage history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS coverage_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE,
                generated_at TIMESTAMP,
                overall_coverage REAL,
                line_coverage REAL,
                branch_coverage REAL,
                function_coverage REAL,
                total_files INTEGER,
                covered_files INTEGER,
                total_lines INTEGER,
                covered_lines INTEGER,
                quality_score REAL,
                test_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS file_coverage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                file_path TEXT,
                total_lines INTEGER,
                covered_lines INTEGER,
                missed_lines INTEGER,
                coverage_percentage REAL,
                branch_coverage REAL,
                function_coverage REAL,
                complexity_score REAL,
                maintainability_index REAL,
                missing_line_ranges TEXT,
                covered_functions TEXT,
                uncovered_functions TEXT,
                test_files TEXT,
                last_modified TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES coverage_runs (run_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS coverage_trends (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT,
                date TEXT,
                coverage_percentage REAL,
                trend_direction TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS coverage_hotspots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                file_path TEXT,
                line_number INTEGER,
                function_name TEXT,
                complexity_score REAL,
                coverage_importance REAL,
                priority_level TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES coverage_runs (run_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def generate_coverage_report(self, 
                                     run_tests: bool = True,
                                     include_trends: bool = True) -> CoverageReport:
        """Generate comprehensive coverage report"""
        
        # Start coverage collection
        self.coverage.start()
        
        try:
            # Run tests if requested
            if run_tests:
                await self._run_tests_with_coverage()
            
            # Stop coverage collection
            self.coverage.stop()
            self.coverage.save()
            
            # Generate report
            report = await self._analyze_coverage()
            
            # Add historical trends
            if include_trends:
                await self._add_coverage_trends(report)
            
            # Store in database
            await self._store_coverage_report(report)
            
            # Generate visualizations
            await self._generate_coverage_visualizations(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating coverage report: {e}")
            raise
    
    async def _run_tests_with_coverage(self):
        """Run tests with coverage collection"""
        try:
            # Run pytest with coverage
            cmd = [
                "python", "-m", "pytest",
                str(self.test_dir),
                "--cov=" + str(self.source_dir),
                "--cov-report=json:coverage.json",
                "--cov-report=xml:coverage.xml",
                "--cov-report=html:htmlcov",
                "--cov-branch",
                "-v"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            if result.returncode != 0:
                self.logger.warning(f"Tests failed but coverage collected: {result.stderr}")
            
        except subprocess.TimeoutExpired:
            self.logger.error("Test execution timeout")
        except Exception as e:
            self.logger.error(f"Error running tests: {e}")
    
    async def _analyze_coverage(self) -> CoverageReport:
        """Analyze coverage data and generate report"""
        
        # Load coverage data
        coverage_data = await self._load_coverage_data()
        
        # Analyze files
        file_analyses = await self._analyze_files(coverage_data)
        
        # Group by modules
        modules = self._group_by_modules(file_analyses)
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(file_analyses)
        
        # Identify hotspots
        hotspots = self._identify_coverage_hotspots(file_analyses)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(file_analyses, hotspots)
        
        # Create report
        report = CoverageReport(
            report_id=f"COV_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.now(),
            overall_coverage=overall_metrics['overall_coverage'],
            line_coverage=overall_metrics['line_coverage'],
            branch_coverage=overall_metrics['branch_coverage'],
            function_coverage=overall_metrics['function_coverage'],
            total_files=overall_metrics['total_files'],
            covered_files=overall_metrics['covered_files'],
            total_lines=overall_metrics['total_lines'],
            covered_lines=overall_metrics['covered_lines'],
            modules=modules,
            uncovered_hotspots=hotspots,
            coverage_trends={},
            quality_metrics=self._calculate_quality_metrics(file_analyses),
            recommendations=recommendations
        )
        
        return report
    
    async def _load_coverage_data(self) -> Dict[str, Any]:
        """Load coverage data from various sources"""
        coverage_data = {}
        
        # Load from coverage.py
        try:
            self.coverage.load()
            coverage_data['coverage_py'] = self.coverage.get_data()
        except Exception as e:
            self.logger.warning(f"Could not load coverage.py data: {e}")
        
        # Load from JSON report
        json_path = Path("coverage.json")
        if json_path.exists():
            with open(json_path, 'r') as f:
                coverage_data['json'] = json.load(f)
        
        # Load from XML report
        xml_path = Path("coverage.xml")
        if xml_path.exists():
            coverage_data['xml'] = ET.parse(xml_path)
        
        return coverage_data
    
    async def _analyze_files(self, coverage_data: Dict[str, Any]) -> List[FileCoverage]:
        """Analyze coverage for individual files"""
        file_analyses = []
        
        # Get file list from coverage data
        files_to_analyze = set()
        
        if 'json' in coverage_data:
            files_to_analyze.update(coverage_data['json'].get('files', {}).keys())
        
        if 'coverage_py' in coverage_data:
            files_to_analyze.update(coverage_data['coverage_py'].measured_files())
        
        # Analyze each file
        tasks = []
        for file_path in files_to_analyze:
            if Path(file_path).exists():
                tasks.append(self._analyze_single_file(file_path, coverage_data))
        
        file_analyses = await asyncio.gather(*tasks)
        
        return [analysis for analysis in file_analyses if analysis]
    
    async def _analyze_single_file(self, file_path: str, coverage_data: Dict[str, Any]) -> Optional[FileCoverage]:
        """Analyze coverage for a single file"""
        try:
            path = Path(file_path)
            
            # Get coverage info from JSON data
            file_info = {}
            if 'json' in coverage_data:
                file_info = coverage_data['json'].get('files', {}).get(file_path, {})
            
            # Calculate basic metrics
            total_lines = len(file_info.get('executed_lines', [])) + len(file_info.get('missing_lines', []))
            covered_lines = len(file_info.get('executed_lines', []))
            missed_lines = len(file_info.get('missing_lines', []))
            coverage_percentage = (covered_lines / total_lines * 100) if total_lines > 0 else 0
            
            # Get missing line ranges
            missing_line_ranges = self._get_missing_line_ranges(file_info.get('missing_lines', []))
            
            # Analyze functions
            covered_functions, uncovered_functions = await self._analyze_functions(file_path, file_info)
            
            # Calculate complexity
            complexity_score = await self._calculate_complexity(file_path)
            
            # Calculate maintainability index
            maintainability_index = await self._calculate_maintainability(file_path)
            
            # Find test files
            test_files = await self._find_test_files(file_path)
            
            # Get file stats
            file_stats = path.stat()
            
            return FileCoverage(
                file_path=file_path,
                total_lines=total_lines,
                covered_lines=covered_lines,
                missed_lines=missed_lines,
                coverage_percentage=coverage_percentage,
                branch_coverage=self._get_branch_coverage(file_info),
                function_coverage=self._get_function_coverage(covered_functions, uncovered_functions),
                missing_line_ranges=missing_line_ranges,
                covered_functions=covered_functions,
                uncovered_functions=uncovered_functions,
                complexity_score=complexity_score,
                maintainability_index=maintainability_index,
                test_files=test_files,
                last_modified=datetime.fromtimestamp(file_stats.st_mtime),
                coverage_trend=[],
                hotspots=[]
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {e}")
            return None
    
    def _get_missing_line_ranges(self, missing_lines: List[int]) -> List[Tuple[int, int]]:
        """Convert missing lines to ranges"""
        if not missing_lines:
            return []
        
        ranges = []
        missing_lines.sort()
        
        start = missing_lines[0]
        end = missing_lines[0]
        
        for line in missing_lines[1:]:
            if line == end + 1:
                end = line
            else:
                ranges.append((start, end))
                start = line
                end = line
        
        ranges.append((start, end))
        return ranges
    
    async def _analyze_functions(self, file_path: str, file_info: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Analyze function coverage"""
        covered_functions = []
        uncovered_functions = []
        
        try:
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read())
            
            executed_lines = set(file_info.get('executed_lines', []))
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_name = node.name
                    function_start = node.lineno
                    
                    # Check if function has any executed lines
                    if function_start in executed_lines:
                        covered_functions.append(function_name)
                    else:
                        uncovered_functions.append(function_name)
        
        except Exception as e:
            self.logger.error(f"Error analyzing functions in {file_path}: {e}")
        
        return covered_functions, uncovered_functions
    
    async def _calculate_complexity(self, file_path: str) -> float:
        """Calculate cyclomatic complexity"""
        try:
            # Use radon or similar tool for complexity calculation
            result = subprocess.run(
                ["python", "-m", "radon", "cc", file_path, "--json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                complexity_data = json.loads(result.stdout)
                if file_path in complexity_data:
                    complexities = [item['complexity'] for item in complexity_data[file_path]]
                    return statistics.mean(complexities) if complexities else 1.0
            
        except Exception as e:
            self.logger.debug(f"Could not calculate complexity for {file_path}: {e}")
        
        return 1.0  # Default complexity
    
    async def _calculate_maintainability(self, file_path: str) -> float:
        """Calculate maintainability index"""
        try:
            # Use radon for maintainability index
            result = subprocess.run(
                ["python", "-m", "radon", "mi", file_path, "--json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                mi_data = json.loads(result.stdout)
                if file_path in mi_data:
                    return mi_data[file_path]['mi']
            
        except Exception as e:
            self.logger.debug(f"Could not calculate maintainability for {file_path}: {e}")
        
        return 50.0  # Default maintainability
    
    async def _find_test_files(self, source_file: str) -> List[str]:
        """Find test files that test the given source file"""
        test_files = []
        
        # Extract module name
        source_path = Path(source_file)
        module_name = source_path.stem
        
        # Look for test files
        for test_file in self.test_dir.rglob("test_*.py"):
            if module_name in test_file.name:
                test_files.append(str(test_file))
        
        for test_file in self.test_dir.rglob("*_test.py"):
            if module_name in test_file.name:
                test_files.append(str(test_file))
        
        return test_files
    
    def _get_branch_coverage(self, file_info: Dict[str, Any]) -> float:
        """Calculate branch coverage percentage"""
        summary = file_info.get('summary', {})
        
        covered_branches = summary.get('covered_branches', 0)
        num_branches = summary.get('num_branches', 0)
        
        if num_branches > 0:
            return (covered_branches / num_branches) * 100
        return 100.0
    
    def _get_function_coverage(self, covered_functions: List[str], uncovered_functions: List[str]) -> float:
        """Calculate function coverage percentage"""
        total_functions = len(covered_functions) + len(uncovered_functions)
        
        if total_functions > 0:
            return (len(covered_functions) / total_functions) * 100
        return 100.0
    
    def _group_by_modules(self, file_analyses: List[FileCoverage]) -> List[ModuleCoverage]:
        """Group files by modules"""
        modules = {}
        
        for file_analysis in file_analyses:
            # Extract module name from file path
            path_parts = Path(file_analysis.file_path).parts
            module_name = path_parts[0] if path_parts else "root"
            
            if module_name not in modules:
                modules[module_name] = []
            
            modules[module_name].append(file_analysis)
        
        module_coverages = []
        for module_name, files in modules.items():
            module_coverage = ModuleCoverage(
                module_name=module_name,
                total_files=len(files),
                covered_files=sum(1 for f in files if f.coverage_percentage > 0),
                overall_coverage=statistics.mean(f.coverage_percentage for f in files),
                line_coverage=statistics.mean(f.coverage_percentage for f in files),
                branch_coverage=statistics.mean(f.branch_coverage for f in files),
                function_coverage=statistics.mean(f.function_coverage for f in files),
                files=files,
                test_count=sum(len(f.test_files) for f in files),
                test_quality_score=self._calculate_test_quality_score(files),
                coverage_trend=[],
                recommendations=self._generate_module_recommendations(files)
            )
            
            module_coverages.append(module_coverage)
        
        return module_coverages
    
    def _calculate_overall_metrics(self, file_analyses: List[FileCoverage]) -> Dict[str, Any]:
        """Calculate overall coverage metrics"""
        if not file_analyses:
            return {
                'overall_coverage': 0.0,
                'line_coverage': 0.0,
                'branch_coverage': 0.0,
                'function_coverage': 0.0,
                'total_files': 0,
                'covered_files': 0,
                'total_lines': 0,
                'covered_lines': 0
            }
        
        total_lines = sum(f.total_lines for f in file_analyses)
        covered_lines = sum(f.covered_lines for f in file_analyses)
        
        return {
            'overall_coverage': (covered_lines / total_lines * 100) if total_lines > 0 else 0,
            'line_coverage': statistics.mean(f.coverage_percentage for f in file_analyses),
            'branch_coverage': statistics.mean(f.branch_coverage for f in file_analyses),
            'function_coverage': statistics.mean(f.function_coverage for f in file_analyses),
            'total_files': len(file_analyses),
            'covered_files': sum(1 for f in file_analyses if f.coverage_percentage > 0),
            'total_lines': total_lines,
            'covered_lines': covered_lines
        }
    
    def _identify_coverage_hotspots(self, file_analyses: List[FileCoverage]) -> List[Dict[str, Any]]:
        """Identify coverage hotspots that need attention"""
        hotspots = []
        
        for file_analysis in file_analyses:
            # Low coverage files
            if file_analysis.coverage_percentage < 50:
                hotspots.append({
                    'file_path': file_analysis.file_path,
                    'type': 'low_coverage',
                    'severity': 'high',
                    'coverage_percentage': file_analysis.coverage_percentage,
                    'description': f'Low coverage: {file_analysis.coverage_percentage:.1f}%',
                    'recommendation': 'Add more comprehensive tests'
                })
            
            # High complexity, low coverage
            if file_analysis.complexity_score > 5 and file_analysis.coverage_percentage < 80:
                hotspots.append({
                    'file_path': file_analysis.file_path,
                    'type': 'complex_uncovered',
                    'severity': 'high',
                    'coverage_percentage': file_analysis.coverage_percentage,
                    'complexity_score': file_analysis.complexity_score,
                    'description': f'High complexity ({file_analysis.complexity_score:.1f}) with low coverage',
                    'recommendation': 'Focus testing on complex functions'
                })
            
            # Missing function coverage
            if file_analysis.uncovered_functions:
                hotspots.append({
                    'file_path': file_analysis.file_path,
                    'type': 'uncovered_functions',
                    'severity': 'medium',
                    'uncovered_functions': file_analysis.uncovered_functions,
                    'description': f'{len(file_analysis.uncovered_functions)} uncovered functions',
                    'recommendation': 'Add tests for uncovered functions'
                })
        
        # Sort by severity and coverage
        hotspots.sort(key=lambda x: (x['severity'], x.get('coverage_percentage', 0)))
        
        return hotspots[:20]  # Return top 20 hotspots
    
    def _generate_recommendations(self, file_analyses: List[FileCoverage], hotspots: List[Dict[str, Any]]) -> List[str]:
        """Generate coverage improvement recommendations"""
        recommendations = []
        
        overall_coverage = statistics.mean(f.coverage_percentage for f in file_analyses)
        
        # Overall coverage recommendations
        if overall_coverage < 70:
            recommendations.append("Overall coverage is below 70%. Consider implementing a coverage gate in CI/CD.")
        elif overall_coverage < 85:
            recommendations.append("Good coverage! Focus on improving test quality and edge cases.")
        else:
            recommendations.append("Excellent coverage! Focus on maintaining quality and performance.")
        
        # Specific recommendations based on hotspots
        low_coverage_files = [h for h in hotspots if h['type'] == 'low_coverage']
        if low_coverage_files:
            recommendations.append(f"Priority: {len(low_coverage_files)} files have coverage below 50%")
        
        complex_files = [h for h in hotspots if h['type'] == 'complex_uncovered']
        if complex_files:
            recommendations.append(f"Focus on {len(complex_files)} complex files with insufficient coverage")
        
        # Module-specific recommendations
        modules = {}
        for file_analysis in file_analyses:
            module = Path(file_analysis.file_path).parts[0]
            if module not in modules:
                modules[module] = []
            modules[module].append(file_analysis)
        
        for module, files in modules.items():
            module_coverage = statistics.mean(f.coverage_percentage for f in files)
            if module_coverage < 60:
                recommendations.append(f"Module '{module}' needs attention (coverage: {module_coverage:.1f}%)")
        
        return recommendations
    
    def _calculate_test_quality_score(self, files: List[FileCoverage]) -> float:
        """Calculate test quality score for a module"""
        if not files:
            return 0.0
        
        score = 0.0
        
        # Coverage score (40%)
        avg_coverage = statistics.mean(f.coverage_percentage for f in files)
        score += (avg_coverage / 100) * 40
        
        # Branch coverage score (30%)
        avg_branch = statistics.mean(f.branch_coverage for f in files)
        score += (avg_branch / 100) * 30
        
        # Function coverage score (20%)
        avg_function = statistics.mean(f.function_coverage for f in files)
        score += (avg_function / 100) * 20
        
        # Test file presence (10%)
        files_with_tests = sum(1 for f in files if f.test_files)
        test_ratio = files_with_tests / len(files)
        score += test_ratio * 10
        
        return score
    
    def _generate_module_recommendations(self, files: List[FileCoverage]) -> List[str]:
        """Generate recommendations for a module"""
        recommendations = []
        
        avg_coverage = statistics.mean(f.coverage_percentage for f in files)
        
        if avg_coverage < 60:
            recommendations.append("Increase overall test coverage")
        
        files_without_tests = [f for f in files if not f.test_files]
        if files_without_tests:
            recommendations.append(f"Add test files for {len(files_without_tests)} source files")
        
        complex_files = [f for f in files if f.complexity_score > 5]
        if complex_files:
            recommendations.append(f"Focus on testing {len(complex_files)} complex files")
        
        return recommendations
    
    def _calculate_quality_metrics(self, file_analyses: List[FileCoverage]) -> Dict[str, float]:
        """Calculate quality metrics"""
        if not file_analyses:
            return {}
        
        return {
            'coverage_consistency': self._calculate_coverage_consistency(file_analyses),
            'test_distribution': self._calculate_test_distribution(file_analyses),
            'complexity_coverage_ratio': self._calculate_complexity_coverage_ratio(file_analyses),
            'maintainability_score': statistics.mean(f.maintainability_index for f in file_analyses),
            'hotspot_density': self._calculate_hotspot_density(file_analyses)
        }
    
    def _calculate_coverage_consistency(self, file_analyses: List[FileCoverage]) -> float:
        """Calculate coverage consistency across files"""
        coverages = [f.coverage_percentage for f in file_analyses]
        
        if len(coverages) < 2:
            return 100.0
        
        std_dev = statistics.stdev(coverages)
        mean_coverage = statistics.mean(coverages)
        
        # Lower standard deviation relative to mean indicates better consistency
        if mean_coverage > 0:
            consistency = max(0, 100 - (std_dev / mean_coverage * 100))
        else:
            consistency = 0.0
        
        return consistency
    
    def _calculate_test_distribution(self, file_analyses: List[FileCoverage]) -> float:
        """Calculate test distribution quality"""
        files_with_tests = sum(1 for f in file_analyses if f.test_files)
        total_files = len(file_analyses)
        
        if total_files > 0:
            return (files_with_tests / total_files) * 100
        return 0.0
    
    def _calculate_complexity_coverage_ratio(self, file_analyses: List[FileCoverage]) -> float:
        """Calculate how well complex code is covered"""
        total_weighted_coverage = 0
        total_weight = 0
        
        for file_analysis in file_analyses:
            weight = file_analysis.complexity_score
            total_weighted_coverage += file_analysis.coverage_percentage * weight
            total_weight += weight
        
        if total_weight > 0:
            return total_weighted_coverage / total_weight
        return 0.0
    
    def _calculate_hotspot_density(self, file_analyses: List[FileCoverage]) -> float:
        """Calculate density of coverage hotspots"""
        hotspot_count = 0
        
        for file_analysis in file_analyses:
            if file_analysis.coverage_percentage < 50:
                hotspot_count += 1
            if file_analysis.complexity_score > 5 and file_analysis.coverage_percentage < 80:
                hotspot_count += 1
        
        total_files = len(file_analyses)
        
        if total_files > 0:
            return (hotspot_count / total_files) * 100
        return 0.0
    
    async def _add_coverage_trends(self, report: CoverageReport):
        """Add coverage trends to report"""
        conn = sqlite3.connect(self.db_path)
        
        # Get historical data
        for module in report.modules:
            for file_coverage in module.files:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT coverage_percentage, created_at 
                    FROM file_coverage 
                    WHERE file_path = ? 
                    ORDER BY created_at DESC 
                    LIMIT 10
                ''', (file_coverage.file_path,))
                
                results = cursor.fetchall()
                if results:
                    file_coverage.coverage_trend = [row[0] for row in results[::-1]]
        
        conn.close()
    
    async def _store_coverage_report(self, report: CoverageReport):
        """Store coverage report in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store run summary
        cursor.execute('''
            INSERT INTO coverage_runs 
            (run_id, generated_at, overall_coverage, line_coverage, branch_coverage,
             function_coverage, total_files, covered_files, total_lines, covered_lines,
             quality_score, test_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            report.report_id, report.generated_at, report.overall_coverage,
            report.line_coverage, report.branch_coverage, report.function_coverage,
            report.total_files, report.covered_files, report.total_lines,
            report.covered_lines, report.quality_metrics.get('maintainability_score', 0),
            sum(module.test_count for module in report.modules)
        ))
        
        # Store file coverage
        for module in report.modules:
            for file_coverage in module.files:
                cursor.execute('''
                    INSERT INTO file_coverage 
                    (run_id, file_path, total_lines, covered_lines, missed_lines,
                     coverage_percentage, branch_coverage, function_coverage,
                     complexity_score, maintainability_index, missing_line_ranges,
                     covered_functions, uncovered_functions, test_files, last_modified)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    report.report_id, file_coverage.file_path, file_coverage.total_lines,
                    file_coverage.covered_lines, file_coverage.missed_lines,
                    file_coverage.coverage_percentage, file_coverage.branch_coverage,
                    file_coverage.function_coverage, file_coverage.complexity_score,
                    file_coverage.maintainability_index, json.dumps(file_coverage.missing_line_ranges),
                    json.dumps(file_coverage.covered_functions), json.dumps(file_coverage.uncovered_functions),
                    json.dumps(file_coverage.test_files), file_coverage.last_modified
                ))
        
        # Store hotspots
        for hotspot in report.uncovered_hotspots:
            cursor.execute('''
                INSERT INTO coverage_hotspots 
                (run_id, file_path, line_number, function_name, complexity_score,
                 coverage_importance, priority_level)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                report.report_id, hotspot['file_path'], hotspot.get('line_number', 0),
                hotspot.get('function_name', ''), hotspot.get('complexity_score', 0),
                hotspot.get('coverage_percentage', 0), hotspot.get('severity', 'medium')
            ))
        
        conn.commit()
        conn.close()
    
    async def _generate_coverage_visualizations(self, report: CoverageReport):
        """Generate coverage visualizations"""
        
        # Overall coverage pie chart
        fig_overall = go.Figure(data=[go.Pie(
            labels=['Covered', 'Uncovered'],
            values=[report.covered_lines, report.total_lines - report.covered_lines],
            hole=0.3,
            marker_colors=['#28a745', '#dc3545']
        )])
        fig_overall.update_layout(title="Overall Line Coverage")
        
        # Module coverage bar chart
        module_names = [m.module_name for m in report.modules]
        module_coverages = [m.overall_coverage for m in report.modules]
        
        fig_modules = go.Figure(data=[go.Bar(
            x=module_names,
            y=module_coverages,
            marker_color=['#28a745' if c >= 80 else '#ffc107' if c >= 60 else '#dc3545' for c in module_coverages]
        )])
        fig_modules.update_layout(
            title="Coverage by Module",
            xaxis_title="Module",
            yaxis_title="Coverage %"
        )
        
        # Coverage trends
        if report.coverage_trends:
            fig_trends = go.Figure()
            for module_name, trend in report.coverage_trends.items():
                fig_trends.add_trace(go.Scatter(
                    x=list(range(len(trend))),
                    y=trend,
                    mode='lines+markers',
                    name=module_name
                ))
            fig_trends.update_layout(
                title="Coverage Trends",
                xaxis_title="Time",
                yaxis_title="Coverage %"
            )
        
        # Save visualizations
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        fig_overall.write_html(viz_dir / f"overall_coverage_{report.report_id}.html")
        fig_modules.write_html(viz_dir / f"module_coverage_{report.report_id}.html")
        
        if report.coverage_trends:
            fig_trends.write_html(viz_dir / f"coverage_trends_{report.report_id}.html")
    
    def generate_html_report(self, report: CoverageReport) -> str:
        """Generate HTML coverage report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Coverage Report - {report.report_id}</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                .coverage-excellent {{ background-color: #d4edda; }}
                .coverage-good {{ background-color: #d1ecf1; }}
                .coverage-fair {{ background-color: #fff3cd; }}
                .coverage-poor {{ background-color: #f8d7da; }}
                .coverage-critical {{ background-color: #f5c6cb; }}
                .hotspot-high {{ border-left: 4px solid #dc3545; }}
                .hotspot-medium {{ border-left: 4px solid #ffc107; }}
                .hotspot-low {{ border-left: 4px solid #28a745; }}
            </style>
        </head>
        <body>
            <div class="container-fluid">
                <h1>Coverage Report</h1>
                <p>Generated: {report.generated_at}</p>
                
                <div class="row">
                    <div class="col-md-3">
                        <div class="card">
                            <div class="card-body text-center">
                                <h3>{report.overall_coverage:.1f}%</h3>
                                <p>Overall Coverage</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card">
                            <div class="card-body text-center">
                                <h3>{report.line_coverage:.1f}%</h3>
                                <p>Line Coverage</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card">
                            <div class="card-body text-center">
                                <h3>{report.branch_coverage:.1f}%</h3>
                                <p>Branch Coverage</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card">
                            <div class="card-body text-center">
                                <h3>{report.function_coverage:.1f}%</h3>
                                <p>Function Coverage</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row mt-4">
                    <div class="col-12">
                        <h3>Modules</h3>
                        <div class="table-responsive">
                            <table class="table table-striped">
                                <thead>
                                    <tr>
                                        <th>Module</th>
                                        <th>Files</th>
                                        <th>Coverage</th>
                                        <th>Tests</th>
                                        <th>Quality Score</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {self._generate_module_rows(report.modules)}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
                
                <div class="row mt-4">
                    <div class="col-12">
                        <h3>Coverage Hotspots</h3>
                        {self._generate_hotspot_cards(report.uncovered_hotspots)}
                    </div>
                </div>
                
                <div class="row mt-4">
                    <div class="col-12">
                        <h3>Recommendations</h3>
                        <ul>
                            {' '.join([f"<li>{rec}</li>" for rec in report.recommendations])}
                        </ul>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        html_path = self.output_dir / f"coverage_report_{report.report_id}.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        return str(html_path)
    
    def _generate_module_rows(self, modules: List[ModuleCoverage]) -> str:
        """Generate HTML rows for modules"""
        rows = []
        
        for module in modules:
            coverage_class = self._get_coverage_class(module.overall_coverage)
            
            row = f"""
            <tr class="{coverage_class}">
                <td>{module.module_name}</td>
                <td>{module.total_files}</td>
                <td>{module.overall_coverage:.1f}%</td>
                <td>{module.test_count}</td>
                <td>{module.test_quality_score:.1f}</td>
            </tr>
            """
            rows.append(row)
        
        return '\n'.join(rows)
    
    def _generate_hotspot_cards(self, hotspots: List[Dict[str, Any]]) -> str:
        """Generate HTML cards for hotspots"""
        cards = []
        
        for hotspot in hotspots[:10]:  # Show top 10
            severity_class = f"hotspot-{hotspot['severity']}"
            
            card = f"""
            <div class="card {severity_class} mb-2">
                <div class="card-body">
                    <h5>{hotspot['file_path']}</h5>
                    <p>{hotspot['description']}</p>
                    <small class="text-muted">Recommendation: {hotspot['recommendation']}</small>
                </div>
            </div>
            """
            cards.append(card)
        
        return '\n'.join(cards)
    
    def _get_coverage_class(self, coverage: float) -> str:
        """Get CSS class for coverage level"""
        if coverage >= 95:
            return "coverage-excellent"
        elif coverage >= 80:
            return "coverage-good"
        elif coverage >= 60:
            return "coverage-fair"
        elif coverage >= 40:
            return "coverage-poor"
        else:
            return "coverage-critical"