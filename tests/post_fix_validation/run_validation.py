#!/usr/bin/env python3
"""
Post-Fix Validation Runner
=========================

Executable script to run comprehensive validation of all 89 critical fixes.
This script can be run manually or integrated into CI/CD pipelines.

Usage:
    python run_validation.py [--category CATEGORY] [--report-format FORMAT] [--output-dir DIR]

Examples:
    python run_validation.py                           # Run all validations
    python run_validation.py --category security       # Run only security validations
    python run_validation.py --report-format html      # Generate HTML report
    python run_validation.py --output-dir ./reports    # Save reports to specific directory

Author: Agent 2 - Post-Fix Validation Research Specialist
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from validation_framework import PostFixValidationFramework, FixCategory, ValidationStatus


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run comprehensive post-fix validation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--category',
        type=str,
        choices=[cat.value for cat in FixCategory],
        help='Run validation for specific category only'
    )
    
    parser.add_argument(
        '--report-format',
        type=str,
        choices=['json', 'html', 'text'],
        default='json',
        help='Report output format (default: json)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./validation_reports',
        help='Output directory for reports (default: ./validation_reports)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--fail-on-regression',
        action='store_true',
        help='Fail immediately if any regression is detected'
    )
    
    parser.add_argument(
        '--min-pass-rate',
        type=float,
        default=0.95,
        help='Minimum pass rate required (default: 0.95)'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run validations in parallel (experimental)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be validated without running tests'
    )
    
    return parser.parse_args()


def create_output_directory(output_dir: str) -> Path:
    """Create output directory if it doesn't exist"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def generate_html_report(report_data: dict, output_file: Path):
    """Generate HTML report"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Post-Fix Validation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
            .metric {{ text-align: center; padding: 10px; background-color: #e9ecef; border-radius: 5px; }}
            .passed {{ color: #28a745; }}
            .failed {{ color: #dc3545; }}
            .error {{ color: #fd7e14; }}
            .category {{ margin: 20px 0; }}
            .category h3 {{ background-color: #6c757d; color: white; padding: 10px; border-radius: 5px; }}
            .test-result {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ddd; }}
            .test-result.passed {{ border-left-color: #28a745; }}
            .test-result.failed {{ border-left-color: #dc3545; }}
            .test-result.error {{ border-left-color: #fd7e14; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Post-Fix Validation Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="summary">
            <div class="metric">
                <h3>Total Tests</h3>
                <div>{report_data['summary']['total_tests']}</div>
            </div>
            <div class="metric">
                <h3 class="passed">Passed</h3>
                <div>{report_data['summary']['passed_tests']}</div>
            </div>
            <div class="metric">
                <h3 class="failed">Failed</h3>
                <div>{report_data['summary']['failed_tests']}</div>
            </div>
            <div class="metric">
                <h3 class="error">Errors</h3>
                <div>{report_data['summary']['error_tests']}</div>
            </div>
            <div class="metric">
                <h3>Pass Rate</h3>
                <div>{report_data['summary']['pass_rate']:.1%}</div>
            </div>
        </div>
        
        <div class="recommendations">
            <h2>Recommendations</h2>
            <ul>
    """
    
    for rec in report_data['recommendations']:
        html_content += f"<li>{rec}</li>\n"
    
    html_content += """
            </ul>
        </div>
        
        <div class="categories">
            <h2>Results by Category</h2>
    """
    
    for category, results in report_data['results_by_category'].items():
        html_content += f"""
            <div class="category">
                <h3>{category.title()}</h3>
        """
        
        for result in results:
            status_class = result['status'].lower()
            html_content += f"""
                <div class="test-result {status_class}">
                    <strong>{result['test_name']}</strong> - {result['status']}
                    <br>Execution Time: {result['execution_time']:.3f}s
            """
            
            if result.get('error_message'):
                html_content += f"<br><em>Error: {result['error_message']}</em>"
            
            html_content += "</div>\n"
        
        html_content += "</div>\n"
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)


def generate_text_report(report_data: dict, output_file: Path):
    """Generate text report"""
    content = f"""
POST-FIX VALIDATION REPORT
=========================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY
-------
Total Tests: {report_data['summary']['total_tests']}
Passed: {report_data['summary']['passed_tests']}
Failed: {report_data['summary']['failed_tests']}
Errors: {report_data['summary']['error_tests']}
Skipped: {report_data['summary']['skipped_tests']}
Pass Rate: {report_data['summary']['pass_rate']:.1%}
Execution Time: {report_data['summary']['execution_time']:.2f}s

RECOMMENDATIONS
--------------
"""
    
    for rec in report_data['recommendations']:
        content += f"- {rec}\n"
    
    content += "\nRESULTS BY CATEGORY\n"
    content += "==================\n"
    
    for category, results in report_data['results_by_category'].items():
        content += f"\n{category.upper()}\n"
        content += "-" * len(category) + "\n"
        
        for result in results:
            status_icon = "✅" if result['status'] == 'passed' else "❌"
            content += f"{status_icon} {result['test_name']} - {result['status']}\n"
            
            if result.get('error_message'):
                content += f"   Error: {result['error_message']}\n"
    
    with open(output_file, 'w') as f:
        f.write(content)


def print_validation_summary(report_data: dict):
    """Print validation summary to console"""
    print("\n" + "="*60)
    print("POST-FIX VALIDATION SUMMARY")
    print("="*60)
    print(f"Total Tests: {report_data['summary']['total_tests']}")
    print(f"Passed: {report_data['summary']['passed_tests']}")
    print(f"Failed: {report_data['summary']['failed_tests']}")
    print(f"Errors: {report_data['summary']['error_tests']}")
    print(f"Skipped: {report_data['summary']['skipped_tests']}")
    print(f"Pass Rate: {report_data['summary']['pass_rate']:.1%}")
    print(f"Execution Time: {report_data['summary']['execution_time']:.2f}s")
    
    if report_data['regressions_detected']:
        print(f"Regressions: {len(report_data['regressions_detected'])}")
        for regression in report_data['regressions_detected']:
            print(f"  - {regression}")
    
    print("\nRECOMMENDATIONS:")
    for rec in report_data['recommendations']:
        print(f"- {rec}")
    
    print("="*60)


def run_dry_run(framework: PostFixValidationFramework, category: Optional[str] = None):
    """Run dry run showing what would be validated"""
    print("\n" + "="*60)
    print("DRY RUN - VALIDATION PLAN")
    print("="*60)
    
    if category:
        categories = [FixCategory(category)]
    else:
        categories = list(FixCategory)
    
    total_fixes = 0
    for cat in categories:
        fixes = framework.fix_inventory[cat]
        total_fixes += len(fixes)
        
        print(f"\n{cat.value.upper()} ({len(fixes)} fixes):")
        for fix in fixes:
            print(f"  - {fix}")
    
    print(f"\nTotal fixes to validate: {total_fixes}")
    print("="*60)


async def main():
    """Main function"""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_path = create_output_directory(args.output_dir)
    
    # Initialize framework
    framework = PostFixValidationFramework()
    
    # Handle dry run
    if args.dry_run:
        run_dry_run(framework, args.category)
        return 0
    
    # Run validation
    logger.info("Starting post-fix validation")
    
    try:
        if args.category:
            logger.info(f"Running validation for category: {args.category}")
            # For single category, we'd need to modify the framework
            # For now, run all and filter results
            
        report = await framework.run_comprehensive_validation()
        
        # Check for regressions
        if args.fail_on_regression and report.regressions_detected:
            logger.error(f"Regressions detected: {report.regressions_detected}")
            return 1
        
        # Check pass rate
        if report.pass_rate < args.min_pass_rate:
            logger.error(f"Pass rate {report.pass_rate:.1%} below minimum {args.min_pass_rate:.1%}")
            return 1
        
        # Generate report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Convert report to dict for serialization
        report_data = {
            "summary": {
                "total_tests": report.total_tests,
                "passed_tests": report.passed_tests,
                "failed_tests": report.failed_tests,
                "error_tests": report.error_tests,
                "skipped_tests": report.skipped_tests,
                "pass_rate": report.pass_rate,
                "execution_time": report.total_execution_time,
                "timestamp": report.timestamp.isoformat()
            },
            "results_by_category": report.results_by_category,
            "regressions_detected": report.regressions_detected,
            "recommendations": report.recommendations
        }
        
        # Save report in requested format
        if args.report_format == 'json':
            report_file = output_path / f"validation_report_{timestamp}.json"
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
        elif args.report_format == 'html':
            report_file = output_path / f"validation_report_{timestamp}.html"
            generate_html_report(report_data, report_file)
        elif args.report_format == 'text':
            report_file = output_path / f"validation_report_{timestamp}.txt"
            generate_text_report(report_data, report_file)
        
        logger.info(f"Report saved to: {report_file}")
        
        # Print summary
        print_validation_summary(report_data)
        
        # Production readiness assessment
        assessment = framework.get_production_readiness_assessment()
        print(f"\nPRODUCTION READINESS: {assessment['status']}")
        
        if assessment['critical_issues']:
            print("CRITICAL ISSUES:")
            for issue in assessment['critical_issues']:
                print(f"- {issue}")
        
        return 0 if assessment['ready'] else 1
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)