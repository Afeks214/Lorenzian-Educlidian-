#!/usr/bin/env python3
"""
Generate comprehensive test report from test results.
"""

import argparse
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
import sys

def parse_junit_xml(xml_file):
    """Parse JUnit XML test results."""
    if not xml_file.exists():
        return {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'skipped': 0,
            'time': 0.0,
            'tests': []
        }
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Handle different JUnit XML formats
    if root.tag == 'testsuites':
        testsuites = root.findall('testsuite')
    elif root.tag == 'testsuite':
        testsuites = [root]
    else:
        testsuites = []
    
    total = 0
    passed = 0
    failed = 0
    errors = 0
    skipped = 0
    time = 0.0
    tests = []
    
    for testsuite in testsuites:
        suite_name = testsuite.get('name', 'Unknown')
        
        for testcase in testsuite.findall('testcase'):
            test_name = testcase.get('name', 'Unknown')
            test_time = float(testcase.get('time', 0))
            
            total += 1
            time += test_time
            
            failure = testcase.find('failure')
            error = testcase.find('error')
            skip = testcase.find('skipped')
            
            if failure is not None:
                failed += 1
                status = 'failed'
                message = failure.get('message', '')
            elif error is not None:
                errors += 1
                status = 'error'
                message = error.get('message', '')
            elif skip is not None:
                skipped += 1
                status = 'skipped'
                message = skip.get('message', '')
            else:
                passed += 1
                status = 'passed'
                message = ''
            
            tests.append({
                'suite': suite_name,
                'name': test_name,
                'status': status,
                'time': test_time,
                'message': message
            })
    
    return {
        'total': total,
        'passed': passed,
        'failed': failed,
        'errors': errors,
        'skipped': skipped,
        'time': time,
        'tests': tests
    }

def parse_coverage_xml(xml_file):
    """Parse coverage XML results."""
    if not xml_file.exists():
        return {
            'line_rate': 0.0,
            'branch_rate': 0.0,
            'lines_covered': 0,
            'lines_valid': 0,
            'branches_covered': 0,
            'branches_valid': 0,
            'packages': []
        }
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Parse coverage summary
    line_rate = float(root.get('line-rate', 0))
    branch_rate = float(root.get('branch-rate', 0))
    lines_covered = int(root.get('lines-covered', 0))
    lines_valid = int(root.get('lines-valid', 0))
    branches_covered = int(root.get('branches-covered', 0))
    branches_valid = int(root.get('branches-valid', 0))
    
    packages = []
    for package in root.findall('.//package'):
        package_name = package.get('name', 'Unknown')
        package_line_rate = float(package.get('line-rate', 0))
        package_branch_rate = float(package.get('branch-rate', 0))
        
        packages.append({
            'name': package_name,
            'line_rate': package_line_rate,
            'branch_rate': package_branch_rate
        })
    
    return {
        'line_rate': line_rate,
        'branch_rate': branch_rate,
        'lines_covered': lines_covered,
        'lines_valid': lines_valid,
        'branches_covered': branches_covered,
        'branches_valid': branches_valid,
        'packages': packages
    }

def generate_html_report(results, output_file):
    """Generate HTML test report."""
    
    # Calculate overall statistics
    total_tests = sum(r['total'] for r in results.values())
    total_passed = sum(r['passed'] for r in results.values())
    total_failed = sum(r['failed'] for r in results.values())
    total_errors = sum(r['errors'] for r in results.values())
    total_skipped = sum(r['skipped'] for r in results.values())
    total_time = sum(r['time'] for r in results.values())
    
    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    # Get coverage info
    coverage = results.get('coverage', {})
    line_coverage = coverage.get('line_rate', 0) * 100
    branch_coverage = coverage.get('branch_rate', 0) * 100
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>GrandModel MARL System - Comprehensive Test Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; margin-bottom: 30px; color: #333; }}
            .summary {{ display: flex; justify-content: space-around; margin-bottom: 30px; }}
            .metric {{ text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 8px; }}
            .metric-value {{ font-size: 2em; font-weight: bold; color: #007bff; }}
            .metric-label {{ font-size: 0.9em; color: #666; }}
            .test-category {{ margin-bottom: 30px; }}
            .test-category h3 {{ background-color: #e9ecef; padding: 10px; margin: 0; border-radius: 4px; }}
            .test-results {{ background-color: #f8f9fa; padding: 15px; border-radius: 4px; }}
            .test-item {{ margin-bottom: 5px; padding: 5px; }}
            .passed {{ color: #28a745; }}
            .failed {{ color: #dc3545; }}
            .error {{ color: #fd7e14; }}
            .skipped {{ color: #6c757d; }}
            .coverage-bar {{ width: 100%; height: 20px; background-color: #e9ecef; border-radius: 10px; overflow: hidden; }}
            .coverage-fill {{ height: 100%; background-color: #28a745; transition: width 0.3s ease; }}
            .requirements {{ background-color: #d4edda; padding: 15px; border-radius: 4px; border-left: 4px solid #28a745; }}
            .timestamp {{ text-align: right; color: #666; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧪 GrandModel MARL System</h1>
                <h2>Comprehensive Test Report</h2>
                <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</div>
            </div>
            
            <div class="summary">
                <div class="metric">
                    <div class="metric-value">{total_tests}</div>
                    <div class="metric-label">Total Tests</div>
                </div>
                <div class="metric">
                    <div class="metric-value passed">{total_passed}</div>
                    <div class="metric-label">Passed</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{success_rate:.1f}%</div>
                    <div class="metric-label">Success Rate</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{line_coverage:.1f}%</div>
                    <div class="metric-label">Line Coverage</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{total_time:.2f}s</div>
                    <div class="metric-label">Total Time</div>
                </div>
            </div>
            
            <div class="requirements">
                <h3>✅ Testing Requirements Met</h3>
                <ul>
                    <li>{'✅' if line_coverage >= 95 else '⚠️'} Line Coverage: {line_coverage:.1f}% (Target: 95%)</li>
                    <li>{'✅' if success_rate >= 95 else '⚠️'} Success Rate: {success_rate:.1f}% (Target: 95%)</li>
                    <li>{'✅' if total_time < 300 else '⚠️'} Execution Time: {total_time:.1f}s (Target: <300s)</li>
                    <li>{'✅' if total_failed == 0 else '⚠️'} Failed Tests: {total_failed} (Target: 0)</li>
                    <li>{'✅' if total_errors == 0 else '⚠️'} Error Tests: {total_errors} (Target: 0)</li>
                </ul>
            </div>
    """
    
    # Add test categories
    test_categories = [
        ('unit', 'Unit Tests', 'Core system components and basic functionality'),
        ('integration', 'Integration Tests', 'Agent coordination and system integration'),
        ('performance', 'Performance Tests', 'High-frequency trading requirements'),
        ('load', 'Load Tests', 'System scalability and throughput'),
        ('security', 'Security Tests', 'Vulnerability detection and validation'),
        ('regression', 'Regression Tests', 'Backward compatibility')
    ]
    
    for category, title, description in test_categories:
        if category in results:
            result = results[category]
            category_success = (result['passed'] / result['total'] * 100) if result['total'] > 0 else 0
            
            html_template += f"""
            <div class="test-category">
                <h3>{title}</h3>
                <div class="test-results">
                    <p>{description}</p>
                    <div style="margin-bottom: 10px;">
                        <strong>Results:</strong> 
                        <span class="passed">{result['passed']} passed</span>, 
                        <span class="failed">{result['failed']} failed</span>, 
                        <span class="error">{result['errors']} errors</span>, 
                        <span class="skipped">{result['skipped']} skipped</span>
                        ({category_success:.1f}% success rate)
                    </div>
                    <div style="margin-bottom: 10px;">
                        <strong>Execution Time:</strong> {result['time']:.2f}s
                    </div>
                    <div class="coverage-bar">
                        <div class="coverage-fill" style="width: {category_success}%"></div>
                    </div>
                </div>
            </div>
            """
    
    # Add coverage details
    if coverage:
        html_template += f"""
        <div class="test-category">
            <h3>📊 Coverage Analysis</h3>
            <div class="test-results">
                <div style="margin-bottom: 15px;">
                    <strong>Line Coverage:</strong> {coverage['lines_covered']}/{coverage['lines_valid']} lines ({line_coverage:.1f}%)
                    <div class="coverage-bar">
                        <div class="coverage-fill" style="width: {line_coverage}%"></div>
                    </div>
                </div>
                <div style="margin-bottom: 15px;">
                    <strong>Branch Coverage:</strong> {coverage['branches_covered']}/{coverage['branches_valid']} branches ({branch_coverage:.1f}%)
                    <div class="coverage-bar">
                        <div class="coverage-fill" style="width: {branch_coverage}%"></div>
                    </div>
                </div>
        """
        
        # Add package coverage
        if coverage.get('packages'):
            html_template += "<h4>Package Coverage:</h4><ul>"
            for package in coverage['packages']:
                pkg_line_coverage = package['line_rate'] * 100
                html_template += f"<li>{package['name']}: {pkg_line_coverage:.1f}% line coverage</li>"
            html_template += "</ul>"
        
        html_template += "</div></div>"
    
    # Add recommendations
    recommendations = []
    if line_coverage < 95:
        recommendations.append("Increase test coverage to reach 95% target")
    if total_failed > 0:
        recommendations.append("Fix failing tests before deployment")
    if total_errors > 0:
        recommendations.append("Resolve test errors")
    if total_time > 300:
        recommendations.append("Optimize test execution time")
    
    if recommendations:
        html_template += f"""
        <div class="test-category">
            <h3>🔧 Recommendations</h3>
            <div class="test-results">
                <ul>
                    {''.join(f'<li>{rec}</li>' for rec in recommendations)}
                </ul>
            </div>
        </div>
        """
    
    html_template += """
            <div class="test-category">
                <h3>🚀 Production Readiness</h3>
                <div class="test-results">
                    <p>System status based on test results:</p>
    """
    
    if success_rate >= 95 and line_coverage >= 95 and total_failed == 0 and total_errors == 0:
        html_template += '<p style="color: #28a745; font-weight: bold;">✅ READY FOR DEPLOYMENT</p>'
    else:
        html_template += '<p style="color: #dc3545; font-weight: bold;">⚠️ NOT READY FOR DEPLOYMENT</p>'
    
    html_template += """
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_template)

def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive test report')
    parser.add_argument('--unit-results', type=Path, help='Unit test results XML file')
    parser.add_argument('--integration-results', type=Path, help='Integration test results XML file')
    parser.add_argument('--performance-results', type=Path, help='Performance test results XML file')
    parser.add_argument('--load-results', type=Path, help='Load test results XML file')
    parser.add_argument('--security-results', type=Path, help='Security test results XML file')
    parser.add_argument('--regression-results', type=Path, help='Regression test results XML file')
    parser.add_argument('--coverage-results', type=Path, help='Coverage results XML file')
    parser.add_argument('--output', type=Path, required=True, help='Output HTML file')
    
    args = parser.parse_args()
    
    # Parse all test results
    results = {}
    
    if args.unit_results:
        results['unit'] = parse_junit_xml(args.unit_results)
    
    if args.integration_results:
        results['integration'] = parse_junit_xml(args.integration_results)
    
    if args.performance_results:
        results['performance'] = parse_junit_xml(args.performance_results)
    
    if args.load_results:
        results['load'] = parse_junit_xml(args.load_results)
    
    if args.security_results:
        results['security'] = parse_junit_xml(args.security_results)
    
    if args.regression_results:
        results['regression'] = parse_junit_xml(args.regression_results)
    
    if args.coverage_results:
        results['coverage'] = parse_coverage_xml(args.coverage_results)
    
    # Generate HTML report
    generate_html_report(results, args.output)
    
    print(f"Test report generated: {args.output}")
    
    # Print summary
    total_tests = sum(r['total'] for r in results.values() if 'total' in r)
    total_passed = sum(r['passed'] for r in results.values() if 'passed' in r)
    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Total tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Success rate: {success_rate:.1f}%")
    
    if 'coverage' in results:
        line_coverage = results['coverage']['line_rate'] * 100
        print(f"Line coverage: {line_coverage:.1f}%")

if __name__ == '__main__':
    main()