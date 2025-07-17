"""
Automatic Test Documentation Generation System
Generates comprehensive documentation from test code, results, and metadata
"""

import ast
import inspect
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import markdown
import pandas as pd
from jinja2 import Environment, FileSystemLoader, Template
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess
import asyncio
import aiofiles
from .advanced_test_reporting import TestResult, TestSuite, TestStatus
from .test_result_aggregator import TestResultAggregator


@dataclass
class TestDocumentation:
    """Test documentation data structure"""
    test_name: str
    test_module: str
    test_class: str
    description: str
    purpose: str
    parameters: List[Dict[str, Any]]
    markers: List[str]
    dependencies: List[str]
    expected_behavior: str
    test_type: str
    complexity: str
    maintainer: str
    created_date: datetime
    last_modified: datetime
    execution_history: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    coverage_impact: Dict[str, float]
    related_tests: List[str]
    known_issues: List[str]
    documentation_score: float


class TestComplexity(Enum):
    """Test complexity levels"""
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class TestType(Enum):
    """Test type categories"""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    REGRESSION = "regression"
    SMOKE = "smoke"
    ACCEPTANCE = "acceptance"
    STRESS = "stress"
    LOAD = "load"
    FUNCTIONAL = "functional"


class TestCodeAnalyzer:
    """Analyzes test code to extract documentation information"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_test_file(self, file_path: str) -> List[TestDocumentation]:
        """Analyze a test file and extract documentation"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            file_stats = Path(file_path).stat()
            
            test_docs = []
            
            # Find all test classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    doc = self._analyze_test_function(node, file_path, file_stats)
                    if doc:
                        test_docs.append(doc)
                elif isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
                    class_docs = self._analyze_test_class(node, file_path, file_stats)
                    test_docs.extend(class_docs)
            
            return test_docs
            
        except Exception as e:
            self.logger.error(f"Error analyzing test file {file_path}: {e}")
            return []
    
    def _analyze_test_function(self, node: ast.FunctionDef, file_path: str, file_stats) -> Optional[TestDocumentation]:
        """Analyze a single test function"""
        try:
            # Extract docstring
            docstring = ast.get_docstring(node) or ""
            
            # Extract function metadata
            test_name = node.name
            test_module = self._extract_module_name(file_path)
            test_class = ""
            
            # Parse docstring for structured information
            description, purpose, expected_behavior = self._parse_docstring(docstring)
            
            # Extract parameters
            parameters = self._extract_parameters(node)
            
            # Extract markers from decorators
            markers = self._extract_markers(node)
            
            # Analyze dependencies
            dependencies = self._analyze_dependencies(node)
            
            # Determine test type and complexity
            test_type = self._determine_test_type(test_name, markers, dependencies)
            complexity = self._calculate_complexity(node, dependencies)
            
            # Extract maintainer information
            maintainer = self._extract_maintainer(file_path)
            
            return TestDocumentation(
                test_name=test_name,
                test_module=test_module,
                test_class=test_class,
                description=description,
                purpose=purpose,
                parameters=parameters,
                markers=markers,
                dependencies=dependencies,
                expected_behavior=expected_behavior,
                test_type=test_type,
                complexity=complexity,
                maintainer=maintainer,
                created_date=datetime.fromtimestamp(file_stats.st_ctime),
                last_modified=datetime.fromtimestamp(file_stats.st_mtime),
                execution_history=[],
                performance_metrics={},
                coverage_impact={},
                related_tests=[],
                known_issues=[],
                documentation_score=0.0
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing test function {node.name}: {e}")
            return None
    
    def _analyze_test_class(self, node: ast.ClassDef, file_path: str, file_stats) -> List[TestDocumentation]:
        """Analyze a test class and its methods"""
        test_docs = []
        class_docstring = ast.get_docstring(node) or ""
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name.startswith('test_'):
                doc = self._analyze_test_function(item, file_path, file_stats)
                if doc:
                    doc.test_class = node.name
                    # Combine class and method docstrings
                    if class_docstring:
                        doc.description = f"{class_docstring}\n\n{doc.description}"
                    test_docs.append(doc)
        
        return test_docs
    
    def _parse_docstring(self, docstring: str) -> Tuple[str, str, str]:
        """Parse docstring to extract description, purpose, and expected behavior"""
        if not docstring:
            return "", "", ""
        
        lines = docstring.strip().split('\n')
        description = ""
        purpose = ""
        expected_behavior = ""
        
        current_section = "description"
        
        for line in lines:
            line = line.strip()
            
            if line.lower().startswith('purpose:'):
                current_section = "purpose"
                purpose = line[8:].strip()
            elif line.lower().startswith('expected:'):
                current_section = "expected"
                expected_behavior = line[9:].strip()
            elif line.lower().startswith('behavior:'):
                current_section = "expected"
                expected_behavior = line[9:].strip()
            elif current_section == "description" and line:
                description += line + " "
            elif current_section == "purpose" and line:
                purpose += " " + line
            elif current_section == "expected" and line:
                expected_behavior += " " + line
        
        return description.strip(), purpose.strip(), expected_behavior.strip()
    
    def _extract_parameters(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Extract function parameters and their types"""
        parameters = []
        
        for arg in node.args.args:
            if arg.arg == 'self':
                continue
            
            param_info = {
                'name': arg.arg,
                'type': self._get_type_annotation(arg.annotation) if arg.annotation else 'Any',
                'default': None,
                'description': ""
            }
            
            parameters.append(param_info)
        
        # Extract default values
        defaults = node.args.defaults
        if defaults:
            # Map defaults to parameters (from right to left)
            param_count = len(parameters)
            default_count = len(defaults)
            start_index = param_count - default_count
            
            for i, default in enumerate(defaults):
                if start_index + i < len(parameters):
                    parameters[start_index + i]['default'] = ast.unparse(default)
        
        return parameters
    
    def _get_type_annotation(self, annotation) -> str:
        """Get string representation of type annotation"""
        try:
            return ast.unparse(annotation)
        except (ImportError, ModuleNotFoundError) as e:
            return 'Any'
    
    def _extract_markers(self, node: ast.FunctionDef) -> List[str]:
        """Extract pytest markers from decorators"""
        markers = []
        
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Attribute):
                    # Handle pytest.mark.marker_name
                    if (isinstance(decorator.func.value, ast.Attribute) and
                        decorator.func.value.attr == 'mark'):
                        markers.append(decorator.func.attr)
                elif isinstance(decorator.func, ast.Name):
                    # Handle direct marker names
                    markers.append(decorator.func.id)
            elif isinstance(decorator, ast.Attribute):
                # Handle pytest.mark.marker_name without parentheses
                if (isinstance(decorator.value, ast.Attribute) and
                    decorator.value.attr == 'mark'):
                    markers.append(decorator.attr)
        
        return markers
    
    def _analyze_dependencies(self, node: ast.FunctionDef) -> List[str]:
        """Analyze test dependencies from imports and function calls"""
        dependencies = []
        
        # Analyze function body for dependencies
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Call):
                if isinstance(stmt.func, ast.Name):
                    dependencies.append(stmt.func.id)
                elif isinstance(stmt.func, ast.Attribute):
                    # Handle method calls
                    if isinstance(stmt.func.value, ast.Name):
                        dependencies.append(f"{stmt.func.value.id}.{stmt.func.attr}")
        
        return list(set(dependencies))
    
    def _determine_test_type(self, test_name: str, markers: List[str], dependencies: List[str]) -> str:
        """Determine test type based on name, markers, and dependencies"""
        test_name_lower = test_name.lower()
        
        # Check markers first
        for marker in markers:
            if marker in [t.value for t in TestType]:
                return marker
        
        # Check test name patterns
        if any(word in test_name_lower for word in ['unit', 'isolated']):
            return TestType.UNIT.value
        elif any(word in test_name_lower for word in ['integration', 'e2e', 'end_to_end']):
            return TestType.INTEGRATION.value
        elif any(word in test_name_lower for word in ['performance', 'perf', 'benchmark']):
            return TestType.PERFORMANCE.value
        elif any(word in test_name_lower for word in ['security', 'auth', 'permission']):
            return TestType.SECURITY.value
        elif any(word in test_name_lower for word in ['regression', 'bug', 'fix']):
            return TestType.REGRESSION.value
        elif any(word in test_name_lower for word in ['smoke', 'basic', 'sanity']):
            return TestType.SMOKE.value
        elif any(word in test_name_lower for word in ['acceptance', 'accept', 'user']):
            return TestType.ACCEPTANCE.value
        elif any(word in test_name_lower for word in ['stress', 'load', 'volume']):
            return TestType.STRESS.value
        
        # Default to functional if no specific type is identified
        return TestType.FUNCTIONAL.value
    
    def _calculate_complexity(self, node: ast.FunctionDef, dependencies: List[str]) -> str:
        """Calculate test complexity based on various factors"""
        complexity_score = 0
        
        # Factor 1: Number of lines
        if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
            line_count = node.end_lineno - node.lineno
            if line_count > 100:
                complexity_score += 4
            elif line_count > 50:
                complexity_score += 3
            elif line_count > 20:
                complexity_score += 2
            elif line_count > 10:
                complexity_score += 1
        
        # Factor 2: Number of dependencies
        dep_count = len(dependencies)
        if dep_count > 20:
            complexity_score += 3
        elif dep_count > 10:
            complexity_score += 2
        elif dep_count > 5:
            complexity_score += 1
        
        # Factor 3: Control flow complexity
        for stmt in ast.walk(node):
            if isinstance(stmt, (ast.If, ast.While, ast.For)):
                complexity_score += 1
            elif isinstance(stmt, ast.Try):
                complexity_score += 2
            elif isinstance(stmt, ast.With):
                complexity_score += 1
        
        # Factor 4: Number of assertions
        assertion_count = 0
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assert):
                assertion_count += 1
            elif isinstance(stmt, ast.Call):
                if isinstance(stmt.func, ast.Attribute) and stmt.func.attr.startswith('assert'):
                    assertion_count += 1
        
        if assertion_count > 10:
            complexity_score += 2
        elif assertion_count > 5:
            complexity_score += 1
        
        # Map score to complexity level
        if complexity_score <= 2:
            return TestComplexity.TRIVIAL.value
        elif complexity_score <= 4:
            return TestComplexity.SIMPLE.value
        elif complexity_score <= 8:
            return TestComplexity.MODERATE.value
        elif complexity_score <= 12:
            return TestComplexity.COMPLEX.value
        else:
            return TestComplexity.VERY_COMPLEX.value
    
    def _extract_maintainer(self, file_path: str) -> str:
        """Extract maintainer information from file or git"""
        try:
            # Try to get from git
            result = subprocess.run(
                ['git', 'log', '--format=%an', '-1', file_path],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.error(f'Error occurred: {e}')
        
        return "Unknown"
    
    def _extract_module_name(self, file_path: str) -> str:
        """Extract module name from file path"""
        path = Path(file_path)
        # Convert file path to module path
        module_parts = []
        
        for part in path.parts:
            if part.endswith('.py'):
                module_parts.append(part[:-3])  # Remove .py extension
            elif part not in ['tests', 'test', '__pycache__']:
                module_parts.append(part)
        
        return '.'.join(module_parts)


class TestDocumentationGenerator:
    """Generates comprehensive test documentation"""
    
    def __init__(self, output_dir: str = "test_docs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.analyzer = TestCodeAnalyzer()
        self.logger = logging.getLogger(__name__)
        self.db_path = self.output_dir / "test_documentation.db"
        self._init_database()
        self._setup_templates()
    
    def _init_database(self):
        """Initialize database for test documentation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_documentation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT,
                test_module TEXT,
                test_class TEXT,
                description TEXT,
                purpose TEXT,
                parameters TEXT,
                markers TEXT,
                dependencies TEXT,
                expected_behavior TEXT,
                test_type TEXT,
                complexity TEXT,
                maintainer TEXT,
                created_date TIMESTAMP,
                last_modified TIMESTAMP,
                execution_history TEXT,
                performance_metrics TEXT,
                coverage_impact TEXT,
                related_tests TEXT,
                known_issues TEXT,
                documentation_score REAL,
                file_path TEXT,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documentation_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT,
                metric_value REAL,
                metric_type TEXT,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _setup_templates(self):
        """Setup Jinja2 templates for documentation generation"""
        template_dir = self.output_dir / "templates"
        template_dir.mkdir(exist_ok=True)
        
        # Create default templates if they don't exist
        self._create_default_templates(template_dir)
        
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=True
        )
    
    def _create_default_templates(self, template_dir: Path):
        """Create default documentation templates"""
        
        # Main documentation template
        main_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Documentation - {{ module_name }}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/themes/prism.min.css" rel="stylesheet">
    <style>
        .test-card { margin: 20px 0; border-left: 4px solid #007bff; }
        .complexity-trivial { border-left-color: #28a745; }
        .complexity-simple { border-left-color: #17a2b8; }
        .complexity-moderate { border-left-color: #ffc107; }
        .complexity-complex { border-left-color: #fd7e14; }
        .complexity-very-complex { border-left-color: #dc3545; }
        .test-type { font-size: 12px; padding: 2px 8px; border-radius: 12px; }
        .sidebar { position: fixed; top: 0; left: 0; height: 100vh; width: 250px; background: #f8f9fa; overflow-y: auto; }
        .main-content { margin-left: 260px; padding: 20px; }
        .toc-link { text-decoration: none; color: #495057; }
        .toc-link:hover { color: #007bff; }
        .metric-badge { display: inline-block; margin: 2px; }
        .performance-chart { height: 300px; }
        .coverage-indicator { width: 100%; height: 20px; background: #e9ecef; border-radius: 10px; }
        .coverage-fill { height: 100%; background: #28a745; border-radius: 10px; }
    </style>
</head>
<body>
    <nav class="sidebar p-3">
        <h5>Test Documentation</h5>
        <hr>
        <div class="mb-3">
            <strong>Module:</strong> {{ module_name }}<br>
            <strong>Generated:</strong> {{ generated_at }}<br>
            <strong>Total Tests:</strong> {{ total_tests }}
        </div>
        <hr>
        <h6>Tests</h6>
        <ul class="list-unstyled">
            {% for test in tests %}
            <li><a href="#{{ test.test_name }}" class="toc-link">{{ test.test_name }}</a></li>
            {% endfor %}
        </ul>
        <hr>
        <h6>Metrics</h6>
        <div class="small">
            <strong>Avg Complexity:</strong> {{ avg_complexity }}<br>
            <strong>Coverage:</strong> {{ coverage_percentage }}%<br>
            <strong>Documentation Score:</strong> {{ documentation_score }}
        </div>
    </nav>
    
    <div class="main-content">
        <div class="row">
            <div class="col-12">
                <h1>Test Documentation: {{ module_name }}</h1>
                <p class="text-muted">Generated on {{ generated_at }}</p>
            </div>
        </div>
        
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card">
                    <div class="card-body text-center">
                        <h3>{{ total_tests }}</h3>
                        <p>Total Tests</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card">
                    <div class="card-body text-center">
                        <h3>{{ coverage_percentage }}%</h3>
                        <p>Coverage</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card">
                    <div class="card-body text-center">
                        <h3>{{ documentation_score }}</h3>
                        <p>Doc Score</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card">
                    <div class="card-body text-center">
                        <h3>{{ avg_complexity }}</h3>
                        <p>Avg Complexity</p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mb-4">
            <div class="col-12">
                <h3>Test Type Distribution</h3>
                <div id="testTypeChart" class="performance-chart"></div>
            </div>
        </div>
        
        <div class="row mb-4">
            <div class="col-12">
                <h3>Complexity Distribution</h3>
                <div id="complexityChart" class="performance-chart"></div>
            </div>
        </div>
        
        {% for test in tests %}
        <div class="card test-card complexity-{{ test.complexity }}" id="{{ test.test_name }}">
            <div class="card-header">
                <h4>{{ test.test_name }}</h4>
                <div>
                    <span class="badge bg-primary test-type">{{ test.test_type }}</span>
                    <span class="badge bg-secondary">{{ test.complexity }}</span>
                    {% for marker in test.markers %}
                    <span class="badge bg-info">{{ marker }}</span>
                    {% endfor %}
                </div>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-8">
                        <h5>Description</h5>
                        <p>{{ test.description or 'No description available' }}</p>
                        
                        {% if test.purpose %}
                        <h5>Purpose</h5>
                        <p>{{ test.purpose }}</p>
                        {% endif %}
                        
                        {% if test.expected_behavior %}
                        <h5>Expected Behavior</h5>
                        <p>{{ test.expected_behavior }}</p>
                        {% endif %}
                        
                        {% if test.parameters %}
                        <h5>Parameters</h5>
                        <ul>
                            {% for param in test.parameters %}
                            <li><strong>{{ param.name }}</strong> ({{ param.type }}): {{ param.description or 'No description' }}</li>
                            {% endfor %}
                        </ul>
                        {% endif %}
                        
                        {% if test.dependencies %}
                        <h5>Dependencies</h5>
                        <ul>
                            {% for dep in test.dependencies %}
                            <li><code>{{ dep }}</code></li>
                            {% endfor %}
                        </ul>
                        {% endif %}
                        
                        {% if test.known_issues %}
                        <h5>Known Issues</h5>
                        <ul>
                            {% for issue in test.known_issues %}
                            <li>{{ issue }}</li>
                            {% endfor %}
                        </ul>
                        {% endif %}
                    </div>
                    <div class="col-md-4">
                        <h5>Metadata</h5>
                        <table class="table table-sm">
                            <tr><td>Module:</td><td>{{ test.test_module }}</td></tr>
                            <tr><td>Class:</td><td>{{ test.test_class or 'N/A' }}</td></tr>
                            <tr><td>Type:</td><td>{{ test.test_type }}</td></tr>
                            <tr><td>Complexity:</td><td>{{ test.complexity }}</td></tr>
                            <tr><td>Maintainer:</td><td>{{ test.maintainer }}</td></tr>
                            <tr><td>Created:</td><td>{{ test.created_date.strftime('%Y-%m-%d') }}</td></tr>
                            <tr><td>Modified:</td><td>{{ test.last_modified.strftime('%Y-%m-%d') }}</td></tr>
                        </table>
                        
                        {% if test.performance_metrics %}
                        <h5>Performance</h5>
                        <div class="small">
                            {% for metric, value in test.performance_metrics.items() %}
                            <div>{{ metric }}: {{ value }}</div>
                            {% endfor %}
                        </div>
                        {% endif %}
                        
                        {% if test.coverage_impact %}
                        <h5>Coverage Impact</h5>
                        <div class="coverage-indicator">
                            <div class="coverage-fill" style="width: {{ test.coverage_impact.get('percentage', 0) }}%"></div>
                        </div>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
        {% endfor %}
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script>
        // Test type distribution chart
        const testTypes = {{ test_type_data | tojson }};
        const testTypeChart = {
            data: [{
                type: 'pie',
                values: testTypes.values,
                labels: testTypes.labels,
                hole: 0.3
            }],
            layout: {
                title: 'Test Type Distribution',
                height: 300
            }
        };
        Plotly.newPlot('testTypeChart', testTypeChart.data, testTypeChart.layout);
        
        // Complexity distribution chart
        const complexityData = {{ complexity_data | tojson }};
        const complexityChart = {
            data: [{
                type: 'bar',
                x: complexityData.labels,
                y: complexityData.values,
                marker: {
                    color: ['#28a745', '#17a2b8', '#ffc107', '#fd7e14', '#dc3545']
                }
            }],
            layout: {
                title: 'Complexity Distribution',
                height: 300,
                xaxis: { title: 'Complexity Level' },
                yaxis: { title: 'Number of Tests' }
            }
        };
        Plotly.newPlot('complexityChart', complexityChart.data, complexityChart.layout);
    </script>
</body>
</html>
        '''
        
        with open(template_dir / "test_documentation.html", 'w') as f:
            f.write(main_template)
        
        # Markdown template for simple documentation
        markdown_template = '''
# Test Documentation: {{ module_name }}

**Generated:** {{ generated_at }}
**Total Tests:** {{ total_tests }}
**Coverage:** {{ coverage_percentage }}%
**Documentation Score:** {{ documentation_score }}

## Summary

This module contains {{ total_tests }} tests with an average complexity of {{ avg_complexity }}.

## Test Overview

{% for test in tests %}
### {{ test.test_name }}

**Type:** {{ test.test_type }}
**Complexity:** {{ test.complexity }}
**Maintainer:** {{ test.maintainer }}

{{ test.description or 'No description available' }}

{% if test.purpose %}
**Purpose:** {{ test.purpose }}
{% endif %}

{% if test.expected_behavior %}
**Expected Behavior:** {{ test.expected_behavior }}
{% endif %}

{% if test.parameters %}
**Parameters:**
{% for param in test.parameters %}
- `{{ param.name }}` ({{ param.type }}): {{ param.description or 'No description' }}
{% endfor %}
{% endif %}

{% if test.dependencies %}
**Dependencies:**
{% for dep in test.dependencies %}
- {{ dep }}
{% endfor %}
{% endif %}

---

{% endfor %}

## Metrics

- **Average Complexity:** {{ avg_complexity }}
- **Coverage Percentage:** {{ coverage_percentage }}%
- **Documentation Score:** {{ documentation_score }}

*Generated by Test Documentation Generator*
        '''
        
        with open(template_dir / "test_documentation.md", 'w') as f:
            f.write(markdown_template)
    
    async def generate_documentation(self, test_directories: List[str]) -> Dict[str, str]:
        """Generate documentation for all tests in specified directories"""
        all_docs = []
        
        # Process test directories
        for test_dir in test_directories:
            test_path = Path(test_dir)
            if test_path.exists():
                docs = await self._process_test_directory(test_path)
                all_docs.extend(docs)
        
        # Store in database
        await self._store_documentation(all_docs)
        
        # Enhance with execution history and metrics
        await self._enhance_with_metrics(all_docs)
        
        # Generate reports
        reports = await self._generate_reports(all_docs)
        
        return reports
    
    async def _process_test_directory(self, test_dir: Path) -> List[TestDocumentation]:
        """Process all test files in a directory"""
        docs = []
        
        # Find all Python test files
        test_files = list(test_dir.rglob("test_*.py")) + list(test_dir.rglob("*_test.py"))
        
        # Process files concurrently
        tasks = []
        for test_file in test_files:
            tasks.append(self._process_test_file(test_file))
        
        results = await asyncio.gather(*tasks)
        
        for file_docs in results:
            docs.extend(file_docs)
        
        return docs
    
    async def _process_test_file(self, test_file: Path) -> List[TestDocumentation]:
        """Process a single test file"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.analyzer.analyze_test_file, str(test_file)
        )
    
    async def _store_documentation(self, docs: List[TestDocumentation]):
        """Store documentation in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for doc in docs:
            cursor.execute('''
                INSERT OR REPLACE INTO test_documentation 
                (test_name, test_module, test_class, description, purpose, parameters,
                 markers, dependencies, expected_behavior, test_type, complexity,
                 maintainer, created_date, last_modified, execution_history,
                 performance_metrics, coverage_impact, related_tests, known_issues,
                 documentation_score, file_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                doc.test_name, doc.test_module, doc.test_class, doc.description,
                doc.purpose, json.dumps(doc.parameters), json.dumps(doc.markers),
                json.dumps(doc.dependencies), doc.expected_behavior, doc.test_type,
                doc.complexity, doc.maintainer, doc.created_date, doc.last_modified,
                json.dumps(doc.execution_history), json.dumps(doc.performance_metrics),
                json.dumps(doc.coverage_impact), json.dumps(doc.related_tests),
                json.dumps(doc.known_issues), doc.documentation_score, ""
            ))
        
        conn.commit()
        conn.close()
    
    async def _enhance_with_metrics(self, docs: List[TestDocumentation]):
        """Enhance documentation with execution history and metrics"""
        # This would integrate with the test result aggregator
        # For now, we'll calculate basic metrics
        
        for doc in docs:
            # Calculate documentation score
            doc.documentation_score = self._calculate_documentation_score(doc)
            
            # Find related tests
            doc.related_tests = self._find_related_tests(doc, docs)
    
    def _calculate_documentation_score(self, doc: TestDocumentation) -> float:
        """Calculate documentation completeness score"""
        score = 0.0
        
        # Description (25 points)
        if doc.description:
            score += 25 * min(1.0, len(doc.description) / 100)
        
        # Purpose (20 points)
        if doc.purpose:
            score += 20
        
        # Expected behavior (20 points)
        if doc.expected_behavior:
            score += 20
        
        # Parameters documented (15 points)
        if doc.parameters:
            documented_params = sum(1 for p in doc.parameters if p.get('description'))
            score += 15 * (documented_params / len(doc.parameters))
        
        # Markers (10 points)
        if doc.markers:
            score += 10
        
        # Maintainer (10 points)
        if doc.maintainer and doc.maintainer != "Unknown":
            score += 10
        
        return round(score, 1)
    
    def _find_related_tests(self, doc: TestDocumentation, all_docs: List[TestDocumentation]) -> List[str]:
        """Find related tests based on various criteria"""
        related = []
        
        for other_doc in all_docs:
            if other_doc.test_name == doc.test_name:
                continue
            
            # Same module
            if other_doc.test_module == doc.test_module:
                related.append(other_doc.test_name)
            
            # Similar dependencies
            common_deps = set(doc.dependencies) & set(other_doc.dependencies)
            if len(common_deps) > 2:
                related.append(other_doc.test_name)
            
            # Similar name patterns
            if any(word in other_doc.test_name.lower() for word in doc.test_name.lower().split('_')):
                related.append(other_doc.test_name)
        
        return list(set(related))[:5]  # Limit to top 5
    
    async def _generate_reports(self, docs: List[TestDocumentation]) -> Dict[str, str]:
        """Generate documentation reports in multiple formats"""
        reports = {}
        
        # Group by module
        modules = {}
        for doc in docs:
            if doc.test_module not in modules:
                modules[doc.test_module] = []
            modules[doc.test_module].append(doc)
        
        # Generate reports for each module
        for module_name, module_docs in modules.items():
            # Calculate module metrics
            metrics = self._calculate_module_metrics(module_docs)
            
            # Generate HTML report
            html_report = await self._generate_html_report(module_name, module_docs, metrics)
            reports[f"{module_name}_html"] = html_report
            
            # Generate Markdown report
            md_report = await self._generate_markdown_report(module_name, module_docs, metrics)
            reports[f"{module_name}_md"] = md_report
        
        # Generate overall summary
        summary_report = await self._generate_summary_report(docs)
        reports["summary"] = summary_report
        
        return reports
    
    def _calculate_module_metrics(self, docs: List[TestDocumentation]) -> Dict[str, Any]:
        """Calculate metrics for a module"""
        if not docs:
            return {}
        
        # Test type distribution
        test_types = {}
        for doc in docs:
            test_types[doc.test_type] = test_types.get(doc.test_type, 0) + 1
        
        # Complexity distribution
        complexity_dist = {}
        for doc in docs:
            complexity_dist[doc.complexity] = complexity_dist.get(doc.complexity, 0) + 1
        
        return {
            'total_tests': len(docs),
            'avg_complexity': self._calculate_avg_complexity(docs),
            'coverage_percentage': 85.0,  # This would come from actual coverage data
            'documentation_score': sum(doc.documentation_score for doc in docs) / len(docs),
            'test_type_data': {
                'labels': list(test_types.keys()),
                'values': list(test_types.values())
            },
            'complexity_data': {
                'labels': list(complexity_dist.keys()),
                'values': list(complexity_dist.values())
            }
        }
    
    def _calculate_avg_complexity(self, docs: List[TestDocumentation]) -> str:
        """Calculate average complexity level"""
        complexity_scores = {
            'trivial': 1,
            'simple': 2,
            'moderate': 3,
            'complex': 4,
            'very_complex': 5
        }
        
        total_score = sum(complexity_scores.get(doc.complexity, 3) for doc in docs)
        avg_score = total_score / len(docs)
        
        # Map back to complexity level
        if avg_score <= 1.5:
            return 'trivial'
        elif avg_score <= 2.5:
            return 'simple'
        elif avg_score <= 3.5:
            return 'moderate'
        elif avg_score <= 4.5:
            return 'complex'
        else:
            return 'very_complex'
    
    async def _generate_html_report(self, module_name: str, docs: List[TestDocumentation], metrics: Dict[str, Any]) -> str:
        """Generate HTML documentation report"""
        template = self.jinja_env.get_template('test_documentation.html')
        
        html_content = template.render(
            module_name=module_name,
            tests=docs,
            generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **metrics
        )
        
        # Save HTML report
        html_path = self.output_dir / f"{module_name}_documentation.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        return str(html_path)
    
    async def _generate_markdown_report(self, module_name: str, docs: List[TestDocumentation], metrics: Dict[str, Any]) -> str:
        """Generate Markdown documentation report"""
        template = self.jinja_env.get_template('test_documentation.md')
        
        md_content = template.render(
            module_name=module_name,
            tests=docs,
            generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **metrics
        )
        
        # Save Markdown report
        md_path = self.output_dir / f"{module_name}_documentation.md"
        with open(md_path, 'w') as f:
            f.write(md_content)
        
        return str(md_path)
    
    async def _generate_summary_report(self, docs: List[TestDocumentation]) -> str:
        """Generate overall summary report"""
        summary_data = {
            'total_tests': len(docs),
            'modules': len(set(doc.test_module for doc in docs)),
            'avg_documentation_score': sum(doc.documentation_score for doc in docs) / len(docs) if docs else 0,
            'complexity_distribution': self._get_complexity_distribution(docs),
            'type_distribution': self._get_type_distribution(docs),
            'top_documented_tests': sorted(docs, key=lambda x: x.documentation_score, reverse=True)[:10],
            'needs_improvement': sorted(docs, key=lambda x: x.documentation_score)[:10]
        }
        
        # Generate summary HTML
        summary_html = f"""
        <html>
        <head><title>Test Documentation Summary</title></head>
        <body>
            <h1>Test Documentation Summary</h1>
            <h2>Overview</h2>
            <ul>
                <li>Total Tests: {summary_data['total_tests']}</li>
                <li>Modules: {summary_data['modules']}</li>
                <li>Average Documentation Score: {summary_data['avg_documentation_score']:.1f}</li>
            </ul>
            
            <h2>Top Documented Tests</h2>
            <ul>
                {' '.join([f"<li>{test.test_name} ({test.documentation_score})</li>" for test in summary_data['top_documented_tests']])}
            </ul>
            
            <h2>Tests Needing Improvement</h2>
            <ul>
                {' '.join([f"<li>{test.test_name} ({test.documentation_score})</li>" for test in summary_data['needs_improvement']])}
            </ul>
        </body>
        </html>
        """
        
        summary_path = self.output_dir / "test_documentation_summary.html"
        with open(summary_path, 'w') as f:
            f.write(summary_html)
        
        return str(summary_path)
    
    def _get_complexity_distribution(self, docs: List[TestDocumentation]) -> Dict[str, int]:
        """Get complexity distribution"""
        distribution = {}
        for doc in docs:
            distribution[doc.complexity] = distribution.get(doc.complexity, 0) + 1
        return distribution
    
    def _get_type_distribution(self, docs: List[TestDocumentation]) -> Dict[str, int]:
        """Get test type distribution"""
        distribution = {}
        for doc in docs:
            distribution[doc.test_type] = distribution.get(doc.test_type, 0) + 1
        return distribution