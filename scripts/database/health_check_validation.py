#!/usr/bin/env python3
"""
Automated Health Check Validation System
AGENT 1: DATABASE RTO SPECIALIST - Health Check Validation
Target: Validate health check effectiveness and RTO improvements
"""

import asyncio
import asyncpg
import aiohttp
import time
import json
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
import statistics


@dataclass
class HealthCheckResult:
    """Result of a health check validation"""
    timestamp: datetime
    check_type: str
    target: str
    response_time_ms: float
    success: bool
    error_message: Optional[str] = None
    expected_max_time_ms: Optional[float] = None


@dataclass
class RTOTestResult:
    """Result of RTO testing"""
    test_name: str
    target_rto_ms: float
    actual_rto_ms: float
    success: bool
    details: Dict
    timestamp: datetime


class HealthCheckValidator:
    """
    Automated validation system for database health checks
    """
    
    def __init__(self, config_file: str = "health_check_validation.yml"):
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()
        self.results = []
        self.rto_results = []
        
        # Test configuration
        self.test_iterations = self.config.get('test_iterations', 10)
        self.test_interval = self.config.get('test_interval', 1)
        self.timeout = self.config.get('timeout', 5)
        
        # Expected performance thresholds
        self.expected_thresholds = {
            'pg_isready': 100,  # 100ms
            'patroni_health': 200,  # 200ms
            'connection_test': 500,  # 500ms
            'pgbouncer_health': 100,  # 100ms
            'etcd_health': 100,  # 100ms
        }
        
        # Database endpoints
        self.db_endpoints = self.config.get('database_endpoints', {})
        self.patroni_endpoints = self.config.get('patroni_endpoints', {})
        
        # RTO targets
        self.rto_targets = {
            'health_check_detection': 1000,  # 1 second
            'patroni_failover': 15000,  # 15 seconds
            'connection_recovery': 30000,  # 30 seconds
        }
    
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Default configuration
            return {
                'database_endpoints': {
                    'primary': {
                        'host': 'postgres-primary',
                        'port': 5432,
                        'database': 'grandmodel',
                        'user': 'grandmodel',
                        'password': 'password'
                    },
                    'standby': {
                        'host': 'postgres-standby',
                        'port': 5432,
                        'database': 'grandmodel',
                        'user': 'grandmodel',
                        'password': 'password'
                    }
                },
                'patroni_endpoints': {
                    'primary': 'http://patroni-primary:8008',
                    'standby': 'http://patroni-standby:8009'
                },
                'test_iterations': 10,
                'test_interval': 1,
                'timeout': 5
            }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('health_check_validator')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler('/var/log/health_check_validation.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    async def test_pg_isready(self, endpoint_name: str, endpoint_config: Dict) -> HealthCheckResult:
        """Test pg_isready command performance"""
        start_time = time.time()
        
        try:
            # Simulate pg_isready by attempting a connection
            connection = await asyncio.wait_for(
                asyncpg.connect(
                    host=endpoint_config['host'],
                    port=endpoint_config['port'],
                    database=endpoint_config['database'],
                    user=endpoint_config['user'],
                    password=endpoint_config['password']
                ),
                timeout=self.timeout
            )
            
            await connection.close()
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                timestamp=datetime.now(),
                check_type='pg_isready',
                target=endpoint_name,
                response_time_ms=response_time,
                success=True,
                expected_max_time_ms=self.expected_thresholds.get('pg_isready', 100)
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                timestamp=datetime.now(),
                check_type='pg_isready',
                target=endpoint_name,
                response_time_ms=response_time,
                success=False,
                error_message=str(e),
                expected_max_time_ms=self.expected_thresholds.get('pg_isready', 100)
            )
    
    async def test_patroni_health(self, endpoint_name: str, endpoint_url: str) -> HealthCheckResult:
        """Test Patroni health endpoint performance"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(f"{endpoint_url}/health") as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        return HealthCheckResult(
                            timestamp=datetime.now(),
                            check_type='patroni_health',
                            target=endpoint_name,
                            response_time_ms=response_time,
                            success=True,
                            expected_max_time_ms=self.expected_thresholds.get('patroni_health', 200)
                        )
                    else:
                        return HealthCheckResult(
                            timestamp=datetime.now(),
                            check_type='patroni_health',
                            target=endpoint_name,
                            response_time_ms=response_time,
                            success=False,
                            error_message=f"HTTP {response.status}",
                            expected_max_time_ms=self.expected_thresholds.get('patroni_health', 200)
                        )
                        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                timestamp=datetime.now(),
                check_type='patroni_health',
                target=endpoint_name,
                response_time_ms=response_time,
                success=False,
                error_message=str(e),
                expected_max_time_ms=self.expected_thresholds.get('patroni_health', 200)
            )
    
    async def test_connection_pool_health(self, endpoint_name: str, endpoint_config: Dict) -> HealthCheckResult:
        """Test connection pool health"""
        start_time = time.time()
        
        try:
            # Create a small connection pool for testing
            pool = await asyncpg.create_pool(
                host=endpoint_config['host'],
                port=endpoint_config['port'],
                database=endpoint_config['database'],
                user=endpoint_config['user'],
                password=endpoint_config['password'],
                min_size=1,
                max_size=5,
                command_timeout=self.timeout
            )
            
            # Test connection acquisition
            async with pool.acquire() as conn:
                await conn.execute("SELECT 1")
            
            await pool.close()
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                timestamp=datetime.now(),
                check_type='connection_pool',
                target=endpoint_name,
                response_time_ms=response_time,
                success=True,
                expected_max_time_ms=self.expected_thresholds.get('connection_test', 500)
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                timestamp=datetime.now(),
                check_type='connection_pool',
                target=endpoint_name,
                response_time_ms=response_time,
                success=False,
                error_message=str(e),
                expected_max_time_ms=self.expected_thresholds.get('connection_test', 500)
            )
    
    async def run_health_check_validation(self):
        """Run comprehensive health check validation"""
        self.logger.info("Starting health check validation...")
        
        for iteration in range(self.test_iterations):
            self.logger.info(f"Running iteration {iteration + 1}/{self.test_iterations}")
            
            # Test database endpoints
            for endpoint_name, endpoint_config in self.db_endpoints.items():
                # Test pg_isready
                result = await self.test_pg_isready(endpoint_name, endpoint_config)
                self.results.append(result)
                
                # Test connection pool
                result = await self.test_connection_pool_health(endpoint_name, endpoint_config)
                self.results.append(result)
            
            # Test Patroni endpoints
            for endpoint_name, endpoint_url in self.patroni_endpoints.items():
                result = await self.test_patroni_health(endpoint_name, endpoint_url)
                self.results.append(result)
            
            # Wait before next iteration
            await asyncio.sleep(self.test_interval)
        
        self.logger.info("Health check validation completed")
    
    async def test_rto_health_detection(self) -> RTOTestResult:
        """Test RTO for health check detection"""
        start_time = time.time()
        
        try:
            # Simulate a health check failure detection
            failed_checks = 0
            detection_time = None
            
            for i in range(10):  # Test up to 10 iterations
                # Test primary database
                try:
                    connection = await asyncio.wait_for(
                        asyncpg.connect(
                            host=self.db_endpoints['primary']['host'],
                            port=self.db_endpoints['primary']['port'],
                            database=self.db_endpoints['primary']['database'],
                            user=self.db_endpoints['primary']['user'],
                            password=self.db_endpoints['primary']['password']
                        ),
                        timeout=1
                    )
                    await connection.close()
                    break  # Success
                except:
                    failed_checks += 1
                    if failed_checks >= 3:  # Failure threshold
                        detection_time = (time.time() - start_time) * 1000
                        break
                
                await asyncio.sleep(0.5)  # Sub-second check interval
            
            if detection_time:
                return RTOTestResult(
                    test_name='health_detection',
                    target_rto_ms=self.rto_targets['health_check_detection'],
                    actual_rto_ms=detection_time,
                    success=detection_time <= self.rto_targets['health_check_detection'],
                    details={'failed_checks': failed_checks, 'detection_time_ms': detection_time},
                    timestamp=datetime.now()
                )
            else:
                return RTOTestResult(
                    test_name='health_detection',
                    target_rto_ms=self.rto_targets['health_check_detection'],
                    actual_rto_ms=0,
                    success=False,
                    details={'error': 'No failure detected'},
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            return RTOTestResult(
                test_name='health_detection',
                target_rto_ms=self.rto_targets['health_check_detection'],
                actual_rto_ms=0,
                success=False,
                details={'error': str(e)},
                timestamp=datetime.now()
            )
    
    def generate_report(self) -> Dict:
        """Generate comprehensive validation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_summary': {
                'total_tests': len(self.results),
                'successful_tests': sum(1 for r in self.results if r.success),
                'failed_tests': sum(1 for r in self.results if not r.success),
                'success_rate': (sum(1 for r in self.results if r.success) / len(self.results)) * 100 if self.results else 0
            },
            'performance_analysis': {},
            'rto_analysis': {},
            'recommendations': []
        }
        
        # Analyze performance by check type
        check_types = set(r.check_type for r in self.results)
        for check_type in check_types:
            type_results = [r for r in self.results if r.check_type == check_type]
            successful_results = [r for r in type_results if r.success]
            
            if successful_results:
                response_times = [r.response_time_ms for r in successful_results]
                expected_max = successful_results[0].expected_max_time_ms
                
                report['performance_analysis'][check_type] = {
                    'total_tests': len(type_results),
                    'successful_tests': len(successful_results),
                    'success_rate': (len(successful_results) / len(type_results)) * 100,
                    'avg_response_time_ms': statistics.mean(response_times),
                    'median_response_time_ms': statistics.median(response_times),
                    'min_response_time_ms': min(response_times),
                    'max_response_time_ms': max(response_times),
                    'expected_max_time_ms': expected_max,
                    'performance_target_met': max(response_times) <= expected_max if expected_max else True
                }
        
        # Analyze RTO results
        for rto_result in self.rto_results:
            report['rto_analysis'][rto_result.test_name] = {
                'target_rto_ms': rto_result.target_rto_ms,
                'actual_rto_ms': rto_result.actual_rto_ms,
                'success': rto_result.success,
                'improvement_needed_ms': max(0, rto_result.actual_rto_ms - rto_result.target_rto_ms),
                'details': rto_result.details
            }
        
        # Generate recommendations
        recommendations = []
        
        # Check for performance issues
        for check_type, analysis in report['performance_analysis'].items():
            if not analysis['performance_target_met']:
                recommendations.append(f"Performance issue: {check_type} exceeds target time ({analysis['max_response_time_ms']:.1f}ms > {analysis['expected_max_time_ms']}ms)")
            
            if analysis['success_rate'] < 95:
                recommendations.append(f"Reliability issue: {check_type} has low success rate ({analysis['success_rate']:.1f}%)")
        
        # Check for RTO issues
        for test_name, analysis in report['rto_analysis'].items():
            if not analysis['success']:
                recommendations.append(f"RTO issue: {test_name} exceeds target ({analysis['actual_rto_ms']:.1f}ms > {analysis['target_rto_ms']}ms)")
        
        # General recommendations
        if report['test_summary']['success_rate'] < 95:
            recommendations.append("Overall health check reliability needs improvement")
        
        report['recommendations'] = recommendations
        
        return report
    
    async def run_full_validation(self):
        """Run full validation including RTO tests"""
        self.logger.info("Starting full health check validation...")
        
        # Run health check validation
        await self.run_health_check_validation()
        
        # Run RTO tests
        self.logger.info("Running RTO tests...")
        rto_result = await self.test_rto_health_detection()
        self.rto_results.append(rto_result)
        
        # Generate and save report
        report = self.generate_report()
        
        # Save report to file
        report_file = f'/var/log/health_check_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Validation report saved to {report_file}")
        
        # Log summary
        self.logger.info("=== VALIDATION SUMMARY ===")
        self.logger.info(f"Total tests: {report['test_summary']['total_tests']}")
        self.logger.info(f"Success rate: {report['test_summary']['success_rate']:.1f}%")
        self.logger.info(f"Recommendations: {len(report['recommendations'])}")
        
        for rec in report['recommendations']:
            self.logger.warning(f"RECOMMENDATION: {rec}")
        
        return report


async def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick validation
        validator = HealthCheckValidator()
        validator.test_iterations = 3
        await validator.run_health_check_validation()
        report = validator.generate_report()
        print(json.dumps(report, indent=2))
    else:
        # Full validation
        validator = HealthCheckValidator()
        report = await validator.run_full_validation()
        
        # Exit with error code if validation fails
        if report['test_summary']['success_rate'] < 95:
            sys.exit(1)
        else:
            sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())