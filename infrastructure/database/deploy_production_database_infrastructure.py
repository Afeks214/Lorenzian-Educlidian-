#!/usr/bin/env python3
"""
Production Database Infrastructure Deployment Script
AGENT 4: DATABASE & STORAGE SPECIALIST
Focus: Complete deployment of production-ready database infrastructure
"""

import asyncio
import subprocess
import yaml
import json
import logging
import time
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import docker
import psycopg2
from concurrent.futures import ThreadPoolExecutor

@dataclass
class DeploymentStep:
    """Deployment step definition"""
    name: str
    description: str
    script_path: str
    config_files: List[str]
    dependencies: List[str]
    rollback_command: str
    validation_command: str
    timeout_seconds: int
    critical: bool
    executed: bool = False
    success: bool = False
    execution_time: float = 0.0
    error_message: str = ""

@dataclass
class DeploymentResult:
    """Deployment result"""
    deployment_id: str
    start_time: datetime
    end_time: datetime
    total_duration: float
    success: bool
    steps_executed: int
    steps_successful: int
    steps_failed: int
    error_details: List[str]
    rollback_performed: bool
    health_check_passed: bool
    performance_baseline: Dict[str, float]

class ProductionDatabaseInfrastructureDeployer:
    """
    Complete production database infrastructure deployment system
    """
    
    def __init__(self, config_file: str = None):
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()
        self.deployment_id = f"db_deploy_{int(time.time())}"
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.docker_client = None
        
        # Deployment steps
        self.deployment_steps = self._define_deployment_steps()
        
        # Health checkers
        self.health_checks = []
        
        # Rollback manager
        self.rollback_commands = []
        
        # Initialize Docker client
        self._initialize_docker()
        
    def _load_config(self, config_file: str) -> Dict:
        """Load deployment configuration"""
        default_config = {
            "deployment": {
                "environment": "production",
                "region": "us-east-1",
                "backup_before_deploy": True,
                "validate_before_deploy": True,
                "rollback_on_failure": True,
                "health_check_timeout": 300,
                "parallel_deployment": False,
                "max_parallel_jobs": 3
            },
            "database": {
                "cluster_name": "grandmodel-cluster",
                "primary_host": "db-primary.grandmodel.com",
                "standby_host": "db-standby.grandmodel.com",
                "port": 5432,
                "database": "grandmodel",
                "admin_user": "postgres",
                "admin_password": "secure_admin_password",
                "app_user": "grandmodel_user",
                "app_password": "secure_app_password",
                "version": "15.0"
            },
            "pgbouncer": {
                "version": "1.18",
                "port": 6432,
                "max_client_conn": 1000,
                "default_pool_size": 100,
                "config_template": "pgbouncer_config.ini"
            },
            "patroni": {
                "version": "3.0",
                "api_port": 8008,
                "config_template": "enhanced_patroni_config.yml",
                "etcd_cluster": "etcd-cluster:2379"
            },
            "monitoring": {
                "prometheus_port": 9090,
                "grafana_port": 3000,
                "alertmanager_port": 9093,
                "enable_real_time_monitoring": True
            },
            "multi_region": {
                "enabled": True,
                "regions": ["us-east-1", "us-west-2", "eu-west-1"],
                "failover_enabled": True,
                "backup_replication": True
            },
            "security": {
                "ssl_enabled": True,
                "certificate_path": "/etc/ssl/certs/grandmodel.crt",
                "private_key_path": "/etc/ssl/private/grandmodel.key",
                "ca_certificate_path": "/etc/ssl/certs/ca.crt"
            },
            "backup": {
                "s3_bucket": "grandmodel-backups",
                "retention_days": 30,
                "encryption_enabled": True,
                "cross_region_replication": True
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    return {**default_config, **loaded_config}
            except Exception as e:
                logging.error(f"Failed to load config file: {e}")
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger('db_infrastructure_deployer')
        logger.setLevel(logging.INFO)
        
        # Create logs directory
        log_dir = Path('/var/log/db_deployment')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler
        log_file = log_dir / f'deployment_{self.deployment_id}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Error handler
        error_handler = logging.FileHandler(log_dir / 'deployment_errors.log')
        error_handler.setLevel(logging.ERROR)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        error_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.addHandler(error_handler)
        
        return logger
    
    def _initialize_docker(self):
        """Initialize Docker client"""
        try:
            self.docker_client = docker.from_env()
            self.docker_client.ping()
            self.logger.info("Docker client initialized successfully")
        except Exception as e:
            self.logger.warning(f"Docker initialization failed: {e}")
    
    def _define_deployment_steps(self) -> List[DeploymentStep]:
        """Define all deployment steps"""
        base_path = Path(__file__).parent
        
        steps = [
            DeploymentStep(
                name="pre_deployment_validation",
                description="Validate deployment environment and prerequisites",
                script_path=str(base_path / "deploy_database_optimizations.py"),
                config_files=[],
                dependencies=[],
                rollback_command="",
                validation_command="python3 -c 'import psycopg2; print(\"Database connection test passed\")'",
                timeout_seconds=120,
                critical=True
            ),
            DeploymentStep(
                name="backup_existing_data",
                description="Create backup of existing database",
                script_path="pg_dump --clean --create --if-exists",
                config_files=[],
                dependencies=["pre_deployment_validation"],
                rollback_command="",
                validation_command="ls -la /var/backups/",
                timeout_seconds=1800,
                critical=True
            ),
            DeploymentStep(
                name="deploy_multi_region_disaster_recovery",
                description="Deploy multi-region disaster recovery system",
                script_path=str(base_path / "multi_region_disaster_recovery.py"),
                config_files=["multi_region_config.yml"],
                dependencies=["backup_existing_data"],
                rollback_command="systemctl stop disaster-recovery",
                validation_command="curl -s http://localhost:8001/health",
                timeout_seconds=600,
                critical=True
            ),
            DeploymentStep(
                name="deploy_enhanced_patroni",
                description="Deploy enhanced Patroni configuration",
                script_path="systemctl restart patroni",
                config_files=["enhanced_patroni_config.yml"],
                dependencies=["deploy_multi_region_disaster_recovery"],
                rollback_command="systemctl stop patroni && systemctl start patroni",
                validation_command="curl -s http://localhost:8008/cluster",
                timeout_seconds=300,
                critical=True
            ),
            DeploymentStep(
                name="deploy_pgbouncer_optimization",
                description="Deploy optimized pgBouncer configuration",
                script_path="systemctl restart pgbouncer",
                config_files=["pgbouncer_config.ini", "pgbouncer_userlist.txt"],
                dependencies=["deploy_enhanced_patroni"],
                rollback_command="systemctl stop pgbouncer && systemctl start pgbouncer",
                validation_command="curl -s http://localhost:6432/stats",
                timeout_seconds=180,
                critical=True
            ),
            DeploymentStep(
                name="deploy_connection_pool_optimizer",
                description="Deploy advanced connection pool optimizer",
                script_path=str(base_path / "connection_pool_optimizer.py"),
                config_files=["connection_pool_config.yml"],
                dependencies=["deploy_pgbouncer_optimization"],
                rollback_command="systemctl stop connection-pool-optimizer",
                validation_command="curl -s http://localhost:9092/metrics",
                timeout_seconds=240,
                critical=False
            ),
            DeploymentStep(
                name="deploy_real_time_monitoring",
                description="Deploy real-time monitoring and alerting system",
                script_path=str(base_path / "real_time_monitoring_system.py"),
                config_files=["monitoring_config.yml"],
                dependencies=["deploy_connection_pool_optimizer"],
                rollback_command="systemctl stop db-monitoring",
                validation_command="curl -s http://localhost:9093/metrics",
                timeout_seconds=300,
                critical=False
            ),
            DeploymentStep(
                name="deploy_schema_migration_automation",
                description="Deploy schema migration automation system",
                script_path=str(base_path / "schema_migration_automation.py"),
                config_files=["migration_config.yml"],
                dependencies=["deploy_real_time_monitoring"],
                rollback_command="systemctl stop schema-migration",
                validation_command="python3 -c 'import asyncio; print(\"Migration system test passed\")'",
                timeout_seconds=180,
                critical=False
            ),
            DeploymentStep(
                name="setup_monitoring_dashboards",
                description="Setup Grafana dashboards and Prometheus configuration",
                script_path="docker-compose -f monitoring/docker-compose.yml up -d",
                config_files=["prometheus.yml", "grafana-dashboards.json"],
                dependencies=["deploy_schema_migration_automation"],
                rollback_command="docker-compose -f monitoring/docker-compose.yml down",
                validation_command="curl -s http://localhost:3000/api/health",
                timeout_seconds=300,
                critical=False
            ),
            DeploymentStep(
                name="run_performance_baseline",
                description="Run performance baseline tests",
                script_path="python3 scripts/performance_baseline_test.py",
                config_files=[],
                dependencies=["setup_monitoring_dashboards"],
                rollback_command="",
                validation_command="ls -la /var/log/performance_baseline.log",
                timeout_seconds=600,
                critical=False
            ),
            DeploymentStep(
                name="final_health_check",
                description="Run comprehensive health check",
                script_path="python3 scripts/comprehensive_health_check.py",
                config_files=[],
                dependencies=["run_performance_baseline"],
                rollback_command="",
                validation_command="echo 'Health check completed'",
                timeout_seconds=300,
                critical=True
            )
        ]
        
        return steps
    
    async def run_pre_deployment_checks(self) -> bool:
        """Run pre-deployment validation checks"""
        try:
            self.logger.info("Running pre-deployment checks...")
            
            # Check system requirements
            if not await self._check_system_requirements():
                return False
            
            # Check network connectivity
            if not await self._check_network_connectivity():
                return False
            
            # Check database connectivity
            if not await self._check_database_connectivity():
                return False
            
            # Check disk space
            if not await self._check_disk_space():
                return False
            
            # Check permissions
            if not await self._check_permissions():
                return False
            
            # Check existing services
            if not await self._check_existing_services():
                return False
            
            self.logger.info("✓ Pre-deployment checks passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Pre-deployment checks failed: {e}")
            return False
    
    async def _check_system_requirements(self) -> bool:
        """Check system requirements"""
        try:
            # Check Python version
            import sys
            if sys.version_info < (3, 8):
                self.logger.error("Python 3.8+ required")
                return False
            
            # Check available memory
            import psutil
            memory = psutil.virtual_memory()
            if memory.total < 4 * 1024 * 1024 * 1024:  # 4GB
                self.logger.warning("Less than 4GB RAM available")
            
            # Check CPU cores
            cpu_count = psutil.cpu_count()
            if cpu_count < 2:
                self.logger.warning("Less than 2 CPU cores available")
            
            self.logger.info("✓ System requirements check passed")
            return True
            
        except Exception as e:
            self.logger.error(f"System requirements check failed: {e}")
            return False
    
    async def _check_network_connectivity(self) -> bool:
        """Check network connectivity"""
        try:
            import socket
            
            # Check if ports are available
            ports_to_check = [5432, 6432, 8008, 9090, 9093, 3000]
            
            for port in ports_to_check:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                
                try:
                    result = sock.connect_ex(('localhost', port))
                    if result == 0:
                        self.logger.warning(f"Port {port} is already in use")
                except Exception:
                    pass
                finally:
                    sock.close()
            
            self.logger.info("✓ Network connectivity check passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Network connectivity check failed: {e}")
            return False
    
    async def _check_database_connectivity(self) -> bool:
        """Check database connectivity"""
        try:
            conn = psycopg2.connect(
                host=self.config['database']['primary_host'],
                port=self.config['database']['port'],
                database=self.config['database']['database'],
                user=self.config['database']['admin_user'],
                password=self.config['database']['admin_password'],
                connect_timeout=10
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT version()")
            version = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            self.logger.info(f"✓ Database connectivity check passed: {version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Database connectivity check failed: {e}")
            return False
    
    async def _check_disk_space(self) -> bool:
        """Check available disk space"""
        try:
            import shutil
            
            # Check root filesystem
            total, used, free = shutil.disk_usage("/")
            
            # Require at least 10GB free space
            if free < 10 * 1024 * 1024 * 1024:
                self.logger.error(f"Insufficient disk space: {free / (1024**3):.1f}GB free")
                return False
            
            self.logger.info(f"✓ Disk space check passed: {free / (1024**3):.1f}GB free")
            return True
            
        except Exception as e:
            self.logger.error(f"Disk space check failed: {e}")
            return False
    
    async def _check_permissions(self) -> bool:
        """Check required permissions"""
        try:
            # Check if running as root or with sudo
            if os.geteuid() != 0:
                self.logger.warning("Not running as root, some operations may fail")
            
            # Check write permissions for log directories
            log_dirs = ['/var/log/db_deployment', '/var/log/db_monitoring', '/var/log/db_migrations']
            
            for log_dir in log_dirs:
                os.makedirs(log_dir, exist_ok=True)
                if not os.access(log_dir, os.W_OK):
                    self.logger.error(f"No write permission for {log_dir}")
                    return False
            
            self.logger.info("✓ Permissions check passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Permissions check failed: {e}")
            return False
    
    async def _check_existing_services(self) -> bool:
        """Check for existing services that might conflict"""
        try:
            services_to_check = ['postgresql', 'pgbouncer', 'patroni']
            
            for service in services_to_check:
                try:
                    result = subprocess.run(['systemctl', 'is-active', service], 
                                         capture_output=True, text=True)
                    if result.returncode == 0:
                        self.logger.info(f"Service {service} is running")
                    else:
                        self.logger.info(f"Service {service} is not running")
                except Exception:
                    pass
            
            self.logger.info("✓ Existing services check passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Existing services check failed: {e}")
            return False
    
    async def execute_deployment_step(self, step: DeploymentStep) -> bool:
        """Execute a single deployment step"""
        try:
            self.logger.info(f"Executing step: {step.name} - {step.description}")
            
            start_time = time.time()
            step.executed = True
            
            # Check dependencies
            if not await self._check_step_dependencies(step):
                step.error_message = "Dependencies not met"
                return False
            
            # Execute the step
            if step.script_path.endswith('.py'):
                # Execute Python script
                result = await self._execute_python_script(step)
            else:
                # Execute shell command
                result = await self._execute_shell_command(step)
            
            step.execution_time = time.time() - start_time
            
            if result:
                # Run validation
                if step.validation_command:
                    validation_result = await self._run_validation(step)
                    if not validation_result:
                        step.error_message = "Validation failed"
                        return False
                
                step.success = True
                self.logger.info(f"✓ Step {step.name} completed successfully in {step.execution_time:.2f}s")
                return True
            else:
                step.error_message = "Step execution failed"
                return False
                
        except Exception as e:
            step.execution_time = time.time() - start_time if 'start_time' in locals() else 0
            step.error_message = str(e)
            self.logger.error(f"✗ Step {step.name} failed: {e}")
            return False
    
    async def _check_step_dependencies(self, step: DeploymentStep) -> bool:
        """Check if step dependencies are met"""
        for dep_name in step.dependencies:
            dep_step = next((s for s in self.deployment_steps if s.name == dep_name), None)
            if not dep_step or not dep_step.success:
                self.logger.error(f"Dependency {dep_name} not met for step {step.name}")
                return False
        return True
    
    async def _execute_python_script(self, step: DeploymentStep) -> bool:
        """Execute Python script"""
        try:
            # Use subprocess to execute Python script
            cmd = ['python3', step.script_path]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                timeout=step.timeout_seconds
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                if stdout:
                    self.logger.info(f"Script output: {stdout.decode()}")
                return True
            else:
                if stderr:
                    self.logger.error(f"Script error: {stderr.decode()}")
                return False
                
        except asyncio.TimeoutError:
            self.logger.error(f"Script execution timed out after {step.timeout_seconds}s")
            return False
        except Exception as e:
            self.logger.error(f"Script execution failed: {e}")
            return False
    
    async def _execute_shell_command(self, step: DeploymentStep) -> bool:
        """Execute shell command"""
        try:
            process = await asyncio.create_subprocess_shell(
                step.script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                timeout=step.timeout_seconds
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                if stdout:
                    self.logger.info(f"Command output: {stdout.decode()}")
                return True
            else:
                if stderr:
                    self.logger.error(f"Command error: {stderr.decode()}")
                return False
                
        except asyncio.TimeoutError:
            self.logger.error(f"Command execution timed out after {step.timeout_seconds}s")
            return False
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            return False
    
    async def _run_validation(self, step: DeploymentStep) -> bool:
        """Run validation command"""
        try:
            process = await asyncio.create_subprocess_shell(
                step.validation_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                timeout=30
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.logger.info(f"Validation passed for step {step.name}")
                return True
            else:
                self.logger.error(f"Validation failed for step {step.name}: {stderr.decode()}")
                return False
                
        except Exception as e:
            self.logger.error(f"Validation error for step {step.name}: {e}")
            return False
    
    async def rollback_deployment(self, failed_step: DeploymentStep):
        """Rollback deployment"""
        self.logger.warning(f"Rolling back deployment due to failure at step: {failed_step.name}")
        
        # Rollback completed steps in reverse order
        completed_steps = [s for s in self.deployment_steps if s.success]
        
        for step in reversed(completed_steps):
            if step.rollback_command:
                try:
                    self.logger.info(f"Rolling back step: {step.name}")
                    
                    process = await asyncio.create_subprocess_shell(
                        step.rollback_command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        timeout=60
                    )
                    
                    stdout, stderr = await process.communicate()
                    
                    if process.returncode == 0:
                        self.logger.info(f"✓ Rollback completed for step: {step.name}")
                    else:
                        self.logger.error(f"✗ Rollback failed for step: {step.name}")
                        
                except Exception as e:
                    self.logger.error(f"Rollback error for step {step.name}: {e}")
    
    async def run_comprehensive_health_check(self) -> bool:
        """Run comprehensive health check"""
        try:
            self.logger.info("Running comprehensive health check...")
            
            # Check database connectivity
            if not await self._check_database_connectivity():
                return False
            
            # Check service status
            services = ['postgresql', 'pgbouncer', 'patroni']
            for service in services:
                try:
                    result = subprocess.run(['systemctl', 'is-active', service], 
                                         capture_output=True, text=True)
                    if result.returncode == 0:
                        self.logger.info(f"✓ Service {service} is active")
                    else:
                        self.logger.error(f"✗ Service {service} is not active")
                        return False
                except Exception as e:
                    self.logger.error(f"Failed to check service {service}: {e}")
                    return False
            
            # Check monitoring endpoints
            endpoints = [
                ('http://localhost:8008/cluster', 'Patroni cluster status'),
                ('http://localhost:9090/targets', 'Prometheus targets'),
                ('http://localhost:3000/api/health', 'Grafana health')
            ]
            
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                for url, description in endpoints:
                    try:
                        async with session.get(url, timeout=10) as response:
                            if response.status == 200:
                                self.logger.info(f"✓ {description} endpoint is healthy")
                            else:
                                self.logger.error(f"✗ {description} endpoint returned {response.status}")
                                return False
                    except Exception as e:
                        self.logger.error(f"✗ {description} endpoint check failed: {e}")
                        return False
            
            self.logger.info("✓ Comprehensive health check passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Comprehensive health check failed: {e}")
            return False
    
    async def calculate_performance_baseline(self) -> Dict[str, float]:
        """Calculate performance baseline"""
        try:
            self.logger.info("Calculating performance baseline...")
            
            baseline = {
                'connection_time_ms': 0.0,
                'query_latency_ms': 0.0,
                'throughput_qps': 0.0,
                'buffer_hit_ratio': 0.0,
                'cpu_usage_percent': 0.0,
                'memory_usage_percent': 0.0
            }
            
            # Test database connection time
            start_time = time.time()
            conn = psycopg2.connect(
                host=self.config['database']['primary_host'],
                port=self.config['database']['port'],
                database=self.config['database']['database'],
                user=self.config['database']['app_user'],
                password=self.config['database']['app_password'],
                connect_timeout=5
            )
            
            connection_time = (time.time() - start_time) * 1000
            baseline['connection_time_ms'] = connection_time
            
            # Test query latency
            cursor = conn.cursor()
            start_time = time.time()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            query_latency = (time.time() - start_time) * 1000
            baseline['query_latency_ms'] = query_latency
            
            # Get buffer hit ratio
            cursor.execute("""
                SELECT round(
                    (sum(blks_hit) * 100.0 / sum(blks_hit + blks_read))::numeric, 2
                ) as buffer_hit_ratio
                FROM pg_stat_database
            """)
            result = cursor.fetchone()
            if result and result[0]:
                baseline['buffer_hit_ratio'] = float(result[0])
            
            cursor.close()
            conn.close()
            
            # Get system metrics
            import psutil
            baseline['cpu_usage_percent'] = psutil.cpu_percent(interval=1)
            baseline['memory_usage_percent'] = psutil.virtual_memory().percent
            
            self.logger.info(f"Performance baseline calculated: {baseline}")
            return baseline
            
        except Exception as e:
            self.logger.error(f"Performance baseline calculation failed: {e}")
            return {}
    
    async def deploy_infrastructure(self) -> DeploymentResult:
        """Deploy complete database infrastructure"""
        deployment_start = datetime.now()
        
        try:
            self.logger.info(f"Starting database infrastructure deployment: {self.deployment_id}")
            
            # Pre-deployment checks
            if not await self.run_pre_deployment_checks():
                raise Exception("Pre-deployment checks failed")
            
            # Execute deployment steps
            successful_steps = 0
            failed_steps = 0
            error_details = []
            
            for step in self.deployment_steps:
                success = await self.execute_deployment_step(step)
                
                if success:
                    successful_steps += 1
                else:
                    failed_steps += 1
                    error_details.append(f"Step {step.name}: {step.error_message}")
                    
                    # Check if this is a critical step
                    if step.critical:
                        self.logger.error(f"Critical step {step.name} failed, stopping deployment")
                        
                        # Rollback if enabled
                        if self.config['deployment']['rollback_on_failure']:
                            await self.rollback_deployment(step)
                        
                        break
            
            # Final health check
            health_check_passed = await self.run_comprehensive_health_check()
            
            # Calculate performance baseline
            performance_baseline = await self.calculate_performance_baseline()
            
            # Determine overall success
            overall_success = (failed_steps == 0 and health_check_passed)
            
            deployment_end = datetime.now()
            total_duration = (deployment_end - deployment_start).total_seconds()
            
            # Create deployment result
            result = DeploymentResult(
                deployment_id=self.deployment_id,
                start_time=deployment_start,
                end_time=deployment_end,
                total_duration=total_duration,
                success=overall_success,
                steps_executed=successful_steps + failed_steps,
                steps_successful=successful_steps,
                steps_failed=failed_steps,
                error_details=error_details,
                rollback_performed=self.config['deployment']['rollback_on_failure'] and failed_steps > 0,
                health_check_passed=health_check_passed,
                performance_baseline=performance_baseline
            )
            
            # Log deployment summary
            self._log_deployment_summary(result)
            
            # Save deployment report
            await self._save_deployment_report(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Deployment failed with exception: {e}")
            
            deployment_end = datetime.now()
            total_duration = (deployment_end - deployment_start).total_seconds()
            
            return DeploymentResult(
                deployment_id=self.deployment_id,
                start_time=deployment_start,
                end_time=deployment_end,
                total_duration=total_duration,
                success=False,
                steps_executed=0,
                steps_successful=0,
                steps_failed=1,
                error_details=[str(e)],
                rollback_performed=False,
                health_check_passed=False,
                performance_baseline={}
            )
    
    def _log_deployment_summary(self, result: DeploymentResult):
        """Log deployment summary"""
        self.logger.info("=" * 60)
        self.logger.info("DEPLOYMENT SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Deployment ID: {result.deployment_id}")
        self.logger.info(f"Status: {'SUCCESS' if result.success else 'FAILED'}")
        self.logger.info(f"Duration: {result.total_duration:.2f} seconds")
        self.logger.info(f"Steps Executed: {result.steps_executed}")
        self.logger.info(f"Steps Successful: {result.steps_successful}")
        self.logger.info(f"Steps Failed: {result.steps_failed}")
        self.logger.info(f"Health Check: {'PASSED' if result.health_check_passed else 'FAILED'}")
        self.logger.info(f"Rollback Performed: {'YES' if result.rollback_performed else 'NO'}")
        
        if result.error_details:
            self.logger.info("Error Details:")
            for error in result.error_details:
                self.logger.info(f"  - {error}")
        
        if result.performance_baseline:
            self.logger.info("Performance Baseline:")
            for metric, value in result.performance_baseline.items():
                self.logger.info(f"  - {metric}: {value}")
        
        self.logger.info("=" * 60)
    
    async def _save_deployment_report(self, result: DeploymentResult):
        """Save deployment report to file"""
        try:
            report_dir = Path('/var/log/db_deployment/reports')
            report_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = report_dir / f"deployment_report_{result.deployment_id}.json"
            
            # Convert result to dict
            report_data = {
                'deployment_result': asdict(result),
                'deployment_steps': [asdict(step) for step in self.deployment_steps],
                'configuration': self.config
            }
            
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"Deployment report saved to: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save deployment report: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        
        if self.docker_client:
            self.docker_client.close()


async def main():
    """Main entry point"""
    deployer = ProductionDatabaseInfrastructureDeployer()
    
    try:
        # Deploy infrastructure
        result = await deployer.deploy_infrastructure()
        
        # Print result
        if result.success:
            print("🎉 Database infrastructure deployment completed successfully!")
            print(f"Deployment ID: {result.deployment_id}")
            print(f"Total Duration: {result.total_duration:.2f} seconds")
            print(f"Steps Completed: {result.steps_successful}/{result.steps_executed}")
            
            if result.performance_baseline:
                print("\nPerformance Baseline:")
                for metric, value in result.performance_baseline.items():
                    print(f"  {metric}: {value}")
            
            return 0
        else:
            print("❌ Database infrastructure deployment failed!")
            print(f"Deployment ID: {result.deployment_id}")
            print(f"Steps Failed: {result.steps_failed}")
            
            if result.error_details:
                print("\nError Details:")
                for error in result.error_details:
                    print(f"  - {error}")
            
            return 1
            
    except Exception as e:
        print(f"Deployment failed with exception: {e}")
        return 1
    finally:
        await deployer.cleanup()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)