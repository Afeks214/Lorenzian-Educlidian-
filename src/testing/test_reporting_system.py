"""
Comprehensive Test Reporting System - Main Integration Module
Coordinates all test reporting components into a unified system
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from .advanced_test_reporting import TestReportGenerator, TestResult, TestSuite, TestStatus
from .test_result_aggregator import TestResultAggregator, TestAggregationConfig
from .test_documentation_generator import TestDocumentationGenerator
from .coverage_analyzer import CoverageAnalyzer, CoverageReport
from .multi_format_reporter import MultiFormatReporter, ReportConfig
from .report_distributor import ReportDistributor, DistributionConfig, NotificationRecipient
from .report_archiver import ReportArchiver, ArchiveConfig
from .performance_trend_analyzer import PerformanceTrendAnalyzer


class TestReportingSystem:
    """
    Comprehensive Test Reporting System
    
    Integrates all test reporting components into a unified system that provides:
    - Advanced HTML reports with visualizations
    - Test result aggregation from multiple sources
    - Automatic test documentation generation
    - Detailed coverage analysis
    - Multi-format report generation (HTML, JSON, CSV, PDF, JUnit XML)
    - Automated report distribution via email/Slack
    - Long-term archival and historical access
    - Performance trend analysis and comparison
    """
    
    def __init__(self, config_path: str = "config/test_reporting_config.json"):
        """Initialize the comprehensive test reporting system"""
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("Test Reporting System initialized successfully")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        
        default_config = {
            "enabled": True,
            "output_dir": "test_reports",
            "aggregation": {
                "enabled": True,
                "source_directories": ["test_results", "coverage_reports"],
                "merge_duplicates": True,
                "timeout_threshold": 300
            },
            "documentation": {
                "enabled": True,
                "test_directories": ["tests"],
                "output_dir": "test_docs",
                "template_dir": "templates"
            },
            "coverage": {
                "enabled": True,
                "source_dir": "src",
                "test_dir": "tests",
                "run_tests": True,
                "include_trends": True
            },
            "multi_format": {
                "enabled": True,
                "formats": ["html", "json", "csv", "pdf", "junit_xml"],
                "include_charts": True,
                "include_trends": True,
                "include_coverage": True,
                "compress_output": True
            },
            "distribution": {
                "enabled": True,
                "recipients": [],
                "email_config": {
                    "enabled": False,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "use_tls": True
                },
                "slack_config": {
                    "enabled": False
                }
            },
            "archival": {
                "enabled": True,
                "archive_format": "zip",
                "retention_days": 90,
                "auto_cleanup": True,
                "backup_locations": []
            },
            "performance_analysis": {
                "enabled": True,
                "trend_analysis_days": 30,
                "comparison_periods": 7
            }
        }
        
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                user_config = json.load(f)
                # Merge with defaults
                default_config.update(user_config)
        else:
            # Create default config file
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        
        logger = logging.getLogger('test_reporting_system')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_dir = Path(self.config.get('output_dir', 'test_reports')) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / 'test_reporting.log')
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _initialize_components(self):
        """Initialize all reporting components"""
        
        output_dir = self.config.get('output_dir', 'test_reports')
        
        # Basic test report generator
        self.test_reporter = TestReportGenerator(output_dir)
        
        # Test result aggregator
        if self.config.get('aggregation', {}).get('enabled', True):
            aggregation_config = TestAggregationConfig(
                config_path=str(self.config_path.parent / 'aggregation_config.yaml')
            )
            self.aggregator = TestResultAggregator(aggregation_config)
        else:
            self.aggregator = None
        
        # Test documentation generator
        if self.config.get('documentation', {}).get('enabled', True):
            doc_config = self.config.get('documentation', {})
            self.doc_generator = TestDocumentationGenerator(
                output_dir=doc_config.get('output_dir', 'test_docs')
            )
        else:
            self.doc_generator = None
        
        # Coverage analyzer
        if self.config.get('coverage', {}).get('enabled', True):
            coverage_config = self.config.get('coverage', {})
            self.coverage_analyzer = CoverageAnalyzer(
                source_dir=coverage_config.get('source_dir', 'src'),
                test_dir=coverage_config.get('test_dir', 'tests'),
                output_dir=f"{output_dir}/coverage"
            )
        else:
            self.coverage_analyzer = None
        
        # Multi-format reporter
        if self.config.get('multi_format', {}).get('enabled', True):
            multi_config = self.config.get('multi_format', {})
            report_config = ReportConfig(
                formats=multi_config.get('formats', ['html', 'json']),
                output_dir=f"{output_dir}/reports",
                template_dir=f"{output_dir}/templates",
                include_charts=multi_config.get('include_charts', True),
                include_trends=multi_config.get('include_trends', True),
                include_coverage=multi_config.get('include_coverage', True),
                compress_output=multi_config.get('compress_output', False)
            )
            self.multi_reporter = MultiFormatReporter(report_config)
        else:
            self.multi_reporter = None
        
        # Report distributor
        if self.config.get('distribution', {}).get('enabled', True):
            dist_config = self.config.get('distribution', {})
            distribution_config = DistributionConfig(
                enabled=True,
                recipients=[NotificationRecipient(**r) for r in dist_config.get('recipients', [])],
                email_config=dist_config.get('email_config', {}),
                slack_config=dist_config.get('slack_config', {})
            )
            self.distributor = ReportDistributor(
                distribution_config, 
                f"{output_dir}/distribution"
            )
        else:
            self.distributor = None
        
        # Report archiver
        if self.config.get('archival', {}).get('enabled', True):
            archive_config_data = self.config.get('archival', {})
            archive_config = ArchiveConfig(
                enabled=True,
                archive_format=archive_config_data.get('archive_format', 'zip'),
                retention_days=archive_config_data.get('retention_days', 90),
                auto_cleanup=archive_config_data.get('auto_cleanup', True),
                storage_path=f"{output_dir}/archives"
            )
            self.archiver = ReportArchiver(archive_config)
        else:
            self.archiver = None
        
        # Performance trend analyzer
        if self.config.get('performance_analysis', {}).get('enabled', True):
            self.performance_analyzer = PerformanceTrendAnalyzer(
                f"{output_dir}/performance_trends.db"
            )
        else:
            self.performance_analyzer = None
    
    async def generate_comprehensive_report(self, 
                                          suite: TestSuite,
                                          previous_results: Optional[List[TestSuite]] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive test report using all available components
        
        Args:
            suite: Test suite results
            previous_results: Optional list of previous test results for trend analysis
            
        Returns:
            Dictionary containing paths to all generated reports and metadata
        """
        
        if not self.config.get('enabled', True):
            self.logger.info("Test reporting system is disabled")
            return {"status": "disabled"}
        
        self.logger.info(f"Generating comprehensive report for suite: {suite.suite_name}")
        
        results = {
            "suite_name": suite.suite_name,
            "generated_at": datetime.now().isoformat(),
            "reports": {},
            "metadata": {},
            "errors": []
        }
        
        try:
            # Step 1: Generate coverage report
            coverage_report = None
            if self.coverage_analyzer:
                self.logger.info("Generating coverage report...")
                try:
                    coverage_report = await self.coverage_analyzer.generate_coverage_report(
                        run_tests=self.config.get('coverage', {}).get('run_tests', True),
                        include_trends=self.config.get('coverage', {}).get('include_trends', True)
                    )
                    results["reports"]["coverage"] = coverage_report
                    self.logger.info("Coverage report generated successfully")
                except Exception as e:
                    error_msg = f"Coverage analysis failed: {str(e)}"
                    self.logger.error(error_msg)
                    results["errors"].append(error_msg)
            
            # Step 2: Generate test documentation
            if self.doc_generator:
                self.logger.info("Generating test documentation...")
                try:
                    doc_config = self.config.get('documentation', {})
                    doc_reports = await self.doc_generator.generate_documentation(
                        doc_config.get('test_directories', ['tests'])
                    )
                    results["reports"]["documentation"] = doc_reports
                    self.logger.info("Test documentation generated successfully")
                except Exception as e:
                    error_msg = f"Documentation generation failed: {str(e)}"
                    self.logger.error(error_msg)
                    results["errors"].append(error_msg)
            
            # Step 3: Generate multi-format reports
            if self.multi_reporter:
                self.logger.info("Generating multi-format reports...")
                try:
                    multi_reports = await self.multi_reporter.generate_reports(
                        suite, coverage_report, previous_results
                    )
                    results["reports"]["multi_format"] = multi_reports
                    self.logger.info("Multi-format reports generated successfully")
                except Exception as e:
                    error_msg = f"Multi-format report generation failed: {str(e)}"
                    self.logger.error(error_msg)
                    results["errors"].append(error_msg)
            
            # Step 4: Record performance metrics
            if self.performance_analyzer:
                self.logger.info("Recording performance metrics...")
                try:
                    await self.performance_analyzer.record_performance_metrics(
                        suite, coverage_report
                    )
                    
                    # Generate trend analysis
                    perf_config = self.config.get('performance_analysis', {})
                    trend_days = perf_config.get('trend_analysis_days', 30)
                    
                    key_metrics = ['execution_time', 'success_rate', 'coverage', 'test_count']
                    trends = await self.performance_analyzer.analyze_trends(
                        suite.suite_name, key_metrics, trend_days
                    )
                    
                    results["reports"]["performance_trends"] = trends
                    self.logger.info("Performance analysis completed successfully")
                except Exception as e:
                    error_msg = f"Performance analysis failed: {str(e)}"
                    self.logger.error(error_msg)
                    results["errors"].append(error_msg)
            
            # Step 5: Archive reports
            if self.archiver:
                self.logger.info("Archiving reports...")
                try:
                    archive_reports = results["reports"].get("multi_format", {})
                    if archive_reports:
                        archive_entry = await self.archiver.archive_reports(
                            suite, archive_reports, coverage_report
                        )
                        results["reports"]["archive"] = archive_entry
                        self.logger.info("Reports archived successfully")
                except Exception as e:
                    error_msg = f"Report archival failed: {str(e)}"
                    self.logger.error(error_msg)
                    results["errors"].append(error_msg)
            
            # Step 6: Distribute reports
            if self.distributor:
                self.logger.info("Distributing reports...")
                try:
                    distribution_reports = results["reports"].get("multi_format", {})
                    if distribution_reports:
                        # Generate recommendations
                        recommendations = self._generate_recommendations(suite, coverage_report)
                        
                        distribution_results = await self.distributor.distribute_reports(
                            suite, distribution_reports, coverage_report, recommendations
                        )
                        results["reports"]["distribution"] = distribution_results
                        self.logger.info("Reports distributed successfully")
                except Exception as e:
                    error_msg = f"Report distribution failed: {str(e)}"
                    self.logger.error(error_msg)
                    results["errors"].append(error_msg)
            
            # Step 7: Generate metadata
            results["metadata"] = {
                "components_used": self._get_enabled_components(),
                "total_processing_time": self._calculate_processing_time(results),
                "report_sizes": self._calculate_report_sizes(results),
                "success_indicators": self._calculate_success_indicators(suite, coverage_report)
            }
            
            self.logger.info("Comprehensive report generation completed")
            
        except Exception as e:
            error_msg = f"Critical error in report generation: {str(e)}"
            self.logger.error(error_msg)
            results["errors"].append(error_msg)
            results["status"] = "failed"
        
        return results
    
    def _generate_recommendations(self, suite: TestSuite, coverage_report: Optional[CoverageReport]) -> List[str]:
        """Generate recommendations based on test results"""
        
        recommendations = []
        
        # Test result recommendations
        if suite.success_rate < 80:
            recommendations.append("Test success rate is below 80%. Consider investigating failing tests.")
        
        if suite.failed > 0:
            recommendations.append(f"Address {suite.failed} failing tests to improve quality.")
        
        if suite.skipped > 0:
            recommendations.append(f"Review {suite.skipped} skipped tests.")
        
        # Performance recommendations
        if suite.total_duration > 300:  # 5 minutes
            recommendations.append("Test execution time is high. Consider optimizing slow tests.")
        
        # Coverage recommendations
        if coverage_report and coverage_report.overall_coverage < 80:
            recommendations.append("Code coverage is below 80%. Add more comprehensive tests.")
        
        return recommendations
    
    def _get_enabled_components(self) -> List[str]:
        """Get list of enabled components"""
        
        components = []
        
        if self.aggregator:
            components.append("aggregation")
        if self.doc_generator:
            components.append("documentation")
        if self.coverage_analyzer:
            components.append("coverage")
        if self.multi_reporter:
            components.append("multi_format")
        if self.distributor:
            components.append("distribution")
        if self.archiver:
            components.append("archival")
        if self.performance_analyzer:
            components.append("performance_analysis")
        
        return components
    
    def _calculate_processing_time(self, results: Dict[str, Any]) -> float:
        """Calculate total processing time"""
        # This would be implemented with actual timing
        return 0.0
    
    def _calculate_report_sizes(self, results: Dict[str, Any]) -> Dict[str, int]:
        """Calculate sizes of generated reports"""
        
        sizes = {}
        
        # This would calculate actual file sizes
        multi_format_reports = results.get("reports", {}).get("multi_format", {})
        for format_name, report_path in multi_format_reports.items():
            if isinstance(report_path, str) and Path(report_path).exists():
                sizes[format_name] = Path(report_path).stat().st_size
        
        return sizes
    
    def _calculate_success_indicators(self, suite: TestSuite, coverage_report: Optional[CoverageReport]) -> Dict[str, Any]:
        """Calculate success indicators"""
        
        indicators = {
            "test_success_rate": suite.success_rate,
            "test_count": suite.total_tests,
            "execution_time": suite.total_duration,
            "coverage_percentage": coverage_report.overall_coverage if coverage_report else 0
        }
        
        # Overall health score
        health_score = 0
        
        if suite.success_rate >= 95:
            health_score += 30
        elif suite.success_rate >= 80:
            health_score += 20
        elif suite.success_rate >= 60:
            health_score += 10
        
        if coverage_report and coverage_report.overall_coverage >= 90:
            health_score += 30
        elif coverage_report and coverage_report.overall_coverage >= 80:
            health_score += 20
        elif coverage_report and coverage_report.overall_coverage >= 60:
            health_score += 10
        
        if suite.total_duration < 60:  # Under 1 minute
            health_score += 20
        elif suite.total_duration < 300:  # Under 5 minutes
            health_score += 15
        elif suite.total_duration < 600:  # Under 10 minutes
            health_score += 10
        
        if suite.total_tests >= 100:
            health_score += 20
        elif suite.total_tests >= 50:
            health_score += 15
        elif suite.total_tests >= 20:
            health_score += 10
        
        indicators["health_score"] = health_score
        
        return indicators
    
    async def process_test_results_from_directory(self, directory: str) -> Dict[str, Any]:
        """Process test results from a directory using aggregation"""
        
        if not self.aggregator:
            raise ValueError("Aggregation component is not enabled")
        
        self.logger.info(f"Processing test results from directory: {directory}")
        
        # Aggregate results
        aggregated_suite = await self.aggregator.aggregate_results([directory])
        
        # Generate comprehensive report
        return await self.generate_comprehensive_report(aggregated_suite)
    
    async def compare_performance_periods(self,
                                        suite_name: str,
                                        baseline_days: int = 30,
                                        comparison_days: int = 7) -> Dict[str, Any]:
        """Compare performance between two periods"""
        
        if not self.performance_analyzer:
            raise ValueError("Performance analysis component is not enabled")
        
        end_date = datetime.now()
        comparison_start = end_date - timedelta(days=comparison_days)
        baseline_start = end_date - timedelta(days=baseline_days)
        baseline_end = comparison_start
        
        comparison = await self.performance_analyzer.compare_performance_periods(
            suite_name, baseline_start, baseline_end, comparison_start, end_date
        )
        
        return comparison
    
    async def generate_historical_report(self, suite_name: str, days: int = 30) -> Dict[str, Any]:
        """Generate historical analysis report"""
        
        results = {}
        
        # Get archival history
        if self.archiver:
            archive_stats = await self.archiver.get_archive_statistics()
            archive_history = await self.archiver.generate_historical_report(suite_name, days)
            results["archive_history"] = archive_history
            results["archive_statistics"] = archive_stats
        
        # Get performance trends
        if self.performance_analyzer:
            performance_dashboard = await self.performance_analyzer.generate_performance_dashboard(suite_name)
            results["performance_dashboard"] = performance_dashboard
        
        return results
    
    async def search_historical_data(self, 
                                   suite_name: Optional[str] = None,
                                   date_range: Optional[tuple] = None,
                                   success_rate_range: Optional[tuple] = None) -> Dict[str, Any]:
        """Search historical test data"""
        
        results = {}
        
        if self.archiver:
            archives = await self.archiver.search_archives(
                suite_name=suite_name,
                date_range=date_range,
                success_rate_range=success_rate_range
            )
            results["archives"] = archives
        
        return results
    
    async def cleanup_old_data(self) -> Dict[str, Any]:
        """Cleanup old data from all components"""
        
        cleanup_results = {}
        
        if self.archiver:
            archive_cleanup = await self.archiver.cleanup_expired_archives()
            cleanup_results["archive_cleanup"] = archive_cleanup
        
        # Additional cleanup for other components can be added here
        
        return cleanup_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all system components"""
        
        status = {
            "system_enabled": self.config.get('enabled', True),
            "components": {
                "aggregation": self.aggregator is not None,
                "documentation": self.doc_generator is not None,
                "coverage": self.coverage_analyzer is not None,
                "multi_format": self.multi_reporter is not None,
                "distribution": self.distributor is not None,
                "archival": self.archiver is not None,
                "performance_analysis": self.performance_analyzer is not None
            },
            "configuration": {
                "output_dir": self.config.get('output_dir'),
                "formats_enabled": self.config.get('multi_format', {}).get('formats', []),
                "distribution_enabled": self.config.get('distribution', {}).get('enabled', False),
                "archival_enabled": self.config.get('archival', {}).get('enabled', False)
            }
        }
        
        return status
    
    def update_configuration(self, config_updates: Dict[str, Any]):
        """Update system configuration"""
        
        self.config.update(config_updates)
        
        # Save updated configuration
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Reinitialize components with new configuration
        self._initialize_components()
        
        self.logger.info("Configuration updated and components reinitialized")


# Convenience function for quick setup
def create_test_reporting_system(config_path: Optional[str] = None) -> TestReportingSystem:
    """Create a test reporting system with default configuration"""
    
    if config_path is None:
        config_path = "config/test_reporting_config.json"
    
    return TestReportingSystem(config_path)


# Example usage
async def main():
    """Example usage of the test reporting system"""
    
    # Create system
    reporting_system = create_test_reporting_system()
    
    # Check status
    status = reporting_system.get_system_status()
    print(f"System status: {status}")
    
    # Example test suite (this would normally come from actual test execution)
    from datetime import datetime
    
    # Create sample test results
    test_results = [
        TestResult(
            test_id="test_1",
            test_name="test_example_function",
            test_class="TestExampleClass",
            test_module="test_example",
            status=TestStatus.PASSED,
            duration=0.5,
            start_time=datetime.now(),
            end_time=datetime.now()
        ),
        TestResult(
            test_id="test_2",
            test_name="test_another_function",
            test_class="TestExampleClass",
            test_module="test_example",
            status=TestStatus.FAILED,
            duration=1.2,
            start_time=datetime.now(),
            end_time=datetime.now(),
            error_message="AssertionError: Expected True but got False"
        )
    ]
    
    # Create test suite
    test_suite = TestSuite(
        suite_name="example_test_suite",
        total_tests=2,
        passed=1,
        failed=1,
        skipped=0,
        errors=0,
        total_duration=1.7,
        start_time=datetime.now(),
        end_time=datetime.now(),
        coverage_percentage=85.0,
        results=test_results
    )
    
    # Generate comprehensive report
    report_results = await reporting_system.generate_comprehensive_report(test_suite)
    
    print(f"Report generation results: {report_results}")


if __name__ == "__main__":
    asyncio.run(main())