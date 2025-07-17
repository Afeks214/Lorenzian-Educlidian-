#!/usr/bin/env python3
"""
Automated Patroni Failover Testing System
Target: <30s Database RTO Achievement
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import psycopg2
import requests
from dataclasses import dataclass
import subprocess
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class FailoverTestResult:
    """Results from a failover test"""
    test_id: str
    start_time: datetime
    end_time: datetime
    rto_seconds: float
    primary_down_time: datetime
    new_primary_ready_time: datetime
    client_reconnect_time: datetime
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'test_id': self.test_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'rto_seconds': self.rto_seconds,
            'primary_down_time': self.primary_down_time.isoformat(),
            'new_primary_ready_time': self.new_primary_ready_time.isoformat(),
            'client_reconnect_time': self.client_reconnect_time.isoformat(),
            'success': self.success,
            'error_message': self.error_message
        }

class PatroniFailoverTester:
    """Automated failover testing system for Patroni clusters"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.patroni_api_url = config.get('patroni_api_url', 'http://localhost:8008')
        self.postgres_config = config.get('postgres', {})
        self.test_results: List[FailoverTestResult] = []
        
    async def get_cluster_status(self) -> Dict:
        """Get current cluster status from Patroni API"""
        try:
            response = requests.get(f"{self.patroni_api_url}/cluster", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get cluster status: {e}")
            return {}
    
    async def get_primary_node(self) -> Optional[str]:
        """Get the current primary node"""
        cluster_status = await self.get_cluster_status()
        for member in cluster_status.get('members', []):
            if member.get('role') == 'Leader':
                return member.get('name')
        return None
    
    async def trigger_failover(self, target_node: Optional[str] = None) -> bool:
        """Trigger a failover using Patroni API"""
        try:
            failover_data = {'leader': await self.get_primary_node()}
            if target_node:
                failover_data['candidate'] = target_node
            
            response = requests.post(
                f"{self.patroni_api_url}/failover",
                json=failover_data,
                timeout=30
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to trigger failover: {e}")
            return False
    
    async def simulate_primary_failure(self) -> bool:
        """Simulate primary node failure by stopping the container"""
        try:
            result = subprocess.run([
                'docker', 'stop', 'patroni-primary'
            ], capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to simulate primary failure: {e}")
            return False
    
    async def restore_primary_node(self) -> bool:
        """Restore the primary node after failure simulation"""
        try:
            result = subprocess.run([
                'docker', 'start', 'patroni-primary'
            ], capture_output=True, text=True, timeout=30)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to restore primary node: {e}")
            return False
    
    async def test_database_connectivity(self) -> bool:
        """Test database connectivity"""
        try:
            conn = psycopg2.connect(
                host=self.postgres_config.get('host', 'localhost'),
                port=self.postgres_config.get('port', 5432),
                database=self.postgres_config.get('database', 'grandmodel'),
                user=self.postgres_config.get('user', 'grandmodel'),
                password=self.postgres_config.get('password'),
                connect_timeout=5
            )
            cursor = conn.cursor()
            cursor.execute('SELECT 1')
            cursor.fetchone()
            conn.close()
            return True
        except Exception as e:
            logger.debug(f"Database connectivity test failed: {e}")
            return False
    
    async def wait_for_new_primary(self, timeout: int = 60) -> Tuple[bool, Optional[str]]:
        """Wait for a new primary to be elected"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            cluster_status = await self.get_cluster_status()
            for member in cluster_status.get('members', []):
                if member.get('role') == 'Leader' and member.get('state') == 'running':
                    return True, member.get('name')
            await asyncio.sleep(1)
        return False, None
    
    async def wait_for_database_ready(self, timeout: int = 60) -> bool:
        """Wait for database to be ready for connections"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if await self.test_database_connectivity():
                return True
            await asyncio.sleep(0.5)
        return False
    
    async def run_failover_test(self, test_type: str = 'graceful') -> FailoverTestResult:
        """Run a complete failover test"""
        test_id = f"failover_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        logger.info(f"Starting failover test: {test_id} (type: {test_type})")
        
        try:
            # Get initial primary
            initial_primary = await self.get_primary_node()
            if not initial_primary:
                raise Exception("No primary node found")
            
            logger.info(f"Initial primary: {initial_primary}")
            
            # Trigger failover based on test type
            if test_type == 'graceful':
                failover_success = await self.trigger_failover()
            elif test_type == 'crash':
                failover_success = await self.simulate_primary_failure()
            else:
                raise ValueError(f"Unknown test type: {test_type}")
            
            if not failover_success:
                raise Exception(f"Failed to trigger {test_type} failover")
            
            primary_down_time = datetime.now()
            
            # Wait for new primary election
            new_primary_found, new_primary = await self.wait_for_new_primary()
            if not new_primary_found:
                raise Exception("New primary not elected within timeout")
            
            new_primary_ready_time = datetime.now()
            logger.info(f"New primary elected: {new_primary}")
            
            # Wait for database to be ready
            db_ready = await self.wait_for_database_ready()
            if not db_ready:
                raise Exception("Database not ready within timeout")
            
            client_reconnect_time = datetime.now()
            end_time = datetime.now()
            
            # Calculate RTO
            rto_seconds = (client_reconnect_time - primary_down_time).total_seconds()
            
            # Restore primary node if it was crashed
            if test_type == 'crash':
                await self.restore_primary_node()
            
            result = FailoverTestResult(
                test_id=test_id,
                start_time=start_time,
                end_time=end_time,
                rto_seconds=rto_seconds,
                primary_down_time=primary_down_time,
                new_primary_ready_time=new_primary_ready_time,
                client_reconnect_time=client_reconnect_time,
                success=True
            )
            
            logger.info(f"Failover test completed successfully. RTO: {rto_seconds:.2f}s")
            return result
            
        except Exception as e:
            end_time = datetime.now()
            logger.error(f"Failover test failed: {e}")
            
            return FailoverTestResult(
                test_id=test_id,
                start_time=start_time,
                end_time=end_time,
                rto_seconds=float('inf'),
                primary_down_time=datetime.now(),
                new_primary_ready_time=datetime.now(),
                client_reconnect_time=datetime.now(),
                success=False,
                error_message=str(e)
            )
    
    async def run_continuous_testing(self, 
                                   test_count: int = 10,
                                   interval_minutes: int = 30,
                                   test_types: List[str] = ['graceful', 'crash']) -> List[FailoverTestResult]:
        """Run continuous failover testing"""
        results = []
        
        for i in range(test_count):
            test_type = test_types[i % len(test_types)]
            
            logger.info(f"Running test {i+1}/{test_count} (type: {test_type})")
            result = await self.run_failover_test(test_type)
            results.append(result)
            
            # Wait before next test
            if i < test_count - 1:
                logger.info(f"Waiting {interval_minutes} minutes before next test...")
                await asyncio.sleep(interval_minutes * 60)
        
        return results
    
    def generate_test_report(self, results: List[FailoverTestResult]) -> Dict:
        """Generate comprehensive test report"""
        successful_tests = [r for r in results if r.success]
        failed_tests = [r for r in results if not r.success]
        
        if successful_tests:
            rto_times = [r.rto_seconds for r in successful_tests]
            avg_rto = statistics.mean(rto_times)
            median_rto = statistics.median(rto_times)
            min_rto = min(rto_times)
            max_rto = max(rto_times)
            rto_std = statistics.stdev(rto_times) if len(rto_times) > 1 else 0
        else:
            avg_rto = median_rto = min_rto = max_rto = rto_std = 0
        
        target_achieved = all(r.rto_seconds < 30 for r in successful_tests)
        
        report = {
            'test_summary': {
                'total_tests': len(results),
                'successful_tests': len(successful_tests),
                'failed_tests': len(failed_tests),
                'success_rate': len(successful_tests) / len(results) * 100 if results else 0,
                'target_achieved': target_achieved,
                'target_rto_seconds': 30
            },
            'rto_statistics': {
                'average_rto_seconds': avg_rto,
                'median_rto_seconds': median_rto,
                'min_rto_seconds': min_rto,
                'max_rto_seconds': max_rto,
                'std_deviation': rto_std
            },
            'test_results': [r.to_dict() for r in results],
            'recommendations': self._generate_recommendations(successful_tests)
        }
        
        return report
    
    def _generate_recommendations(self, successful_tests: List[FailoverTestResult]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        if not successful_tests:
            recommendations.append("All tests failed - investigate cluster configuration")
            return recommendations
        
        rto_times = [r.rto_seconds for r in successful_tests]
        avg_rto = statistics.mean(rto_times)
        
        if avg_rto > 30:
            recommendations.append("Average RTO exceeds 30s target - consider further tuning")
        
        if avg_rto > 25:
            recommendations.append("Consider reducing loop_wait to 3s")
            recommendations.append("Consider reducing TTL to 10s")
        
        if avg_rto > 20:
            recommendations.append("Optimize etcd performance")
            recommendations.append("Consider using faster storage for PostgreSQL")
        
        if max(rto_times) > 45:
            recommendations.append("Some tests had high RTO - investigate network latency")
        
        return recommendations

# Configuration
DEFAULT_CONFIG = {
    'patroni_api_url': 'http://localhost:8008',
    'postgres': {
        'host': 'localhost',
        'port': 5432,
        'database': 'grandmodel',
        'user': 'grandmodel',
        'password': 'your_password_here'
    }
}

async def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Patroni Failover Testing System')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--test-count', type=int, default=5, help='Number of tests to run')
    parser.add_argument('--interval', type=int, default=10, help='Interval between tests (minutes)')
    parser.add_argument('--output', help='Output file for test results')
    parser.add_argument('--single-test', action='store_true', help='Run single test')
    parser.add_argument('--test-type', choices=['graceful', 'crash'], default='graceful', help='Test type')
    
    args = parser.parse_args()
    
    # Load configuration
    config = DEFAULT_CONFIG
    if args.config:
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # Create tester
    tester = PatroniFailoverTester(config)
    
    # Run tests
    if args.single_test:
        result = await tester.run_failover_test(args.test_type)
        results = [result]
    else:
        results = await tester.run_continuous_testing(
            test_count=args.test_count,
            interval_minutes=args.interval
        )
    
    # Generate report
    report = tester.generate_test_report(results)
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Test report saved to {args.output}")
    else:
        print(json.dumps(report, indent=2))
    
    # Print summary
    print(f"\n--- Test Summary ---")
    print(f"Total tests: {report['test_summary']['total_tests']}")
    print(f"Success rate: {report['test_summary']['success_rate']:.1f}%")
    print(f"Average RTO: {report['rto_statistics']['average_rto_seconds']:.2f}s")
    print(f"Target achieved: {report['test_summary']['target_achieved']}")
    
    if report['recommendations']:
        print(f"\n--- Recommendations ---")
        for rec in report['recommendations']:
            print(f"- {rec}")

if __name__ == "__main__":
    asyncio.run(main())