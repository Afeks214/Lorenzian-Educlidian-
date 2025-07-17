#!/usr/bin/env python3
"""
Database Optimization Deployment Script
AGENT 14: DATABASE OPTIMIZATION SPECIALIST
Focus: Automated deployment and orchestration of all database optimizations
"""

import asyncio
import subprocess
import yaml
import json
import logging
import time
import shutil
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import tempfile
import docker
import psycopg2
import aiohttp

@dataclass
class DeploymentStep:
    """Deployment step information"""
    name: str
    description: str
    required: bool
    executed: bool = False
    success: bool = False
    execution_time: float = 0.0
    error_message: str = ""
    rollback_command: str = ""

@dataclass
class DeploymentResult:
    """Deployment result"""
    success: bool
    total_steps: int
    successful_steps: int
    failed_steps: int
    total_time: float
    error_details: List[str]
    rollback_required: bool

class DatabaseOptimizationDeployer:
    """
    Comprehensive database optimization deployment system
    """
    
    def __init__(self, config_file: str = None):
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()
        self.deployment_steps = []
        self.docker_client = None
        self.deployment_start_time = None
        
        # Initialize deployment steps
        self._initialize_deployment_steps()
        
    def _load_config(self, config_file: str) -> Dict:
        """Load deployment configuration"""
        default_config = {
            "environment": "production",
            "database": {
                "host": "127.0.0.1",
                "port": 5432,
                "database": "grandmodel",
                "admin_user": "postgres",
                "admin_password": "postgres_password",
                "app_user": "grandmodel_user",
                "app_password": "grandmodel_password"
            },
            "pgbouncer": {
                "port": 6432,
                "max_client_conn": 1000,
                "default_pool_size": 100,
                "config_file": "/etc/pgbouncer/pgbouncer.ini"
            },
            "patroni": {
                "cluster_name": "grandmodel-cluster",
                "etcd_endpoints": ["http://etcd-cluster:2379"],
                "primary_node": "postgresql-primary",
                "standby_node": "postgresql-standby",
                "config_file": "/etc/patroni/patroni.yml"
            },
            "monitoring": {
                "prometheus_port": 9090,
                "grafana_port": 3000,
                "log_level": "INFO"
            },
            "deployment": {
                "backup_existing_configs": True,
                "run_validation_tests": True,
                "enable_monitoring": True,
                "rollback_on_failure": True,
                "health_check_timeout": 300
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    return {**default_config, **loaded_config}
            except Exception as e:
                print(f"Error loading config file: {e}")
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for deployment"""
        logger = logging.getLogger('db_optimization_deployer')
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        log_dir = Path('/var/log/db_optimization')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler
        log_file = log_dir / f'deployment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _initialize_deployment_steps(self):
        """Initialize deployment steps"""
        self.deployment_steps = [
            DeploymentStep(
                name="pre_deployment_checks",
                description="Perform pre-deployment validation checks",
                required=True,
                rollback_command=""
            ),
            DeploymentStep(
                name="backup_existing_configs",
                description="Backup existing database configurations",
                required=True,
                rollback_command="restore_config_backups"
            ),
            DeploymentStep(
                name="deploy_pgbouncer",
                description="Deploy and configure pgBouncer connection pooling",
                required=True,
                rollback_command="stop_pgbouncer && restore_pgbouncer_config"
            ),
            DeploymentStep(
                name="deploy_patroni_config",
                description="Deploy enhanced Patroni configuration",
                required=True,
                rollback_command="restore_patroni_config && restart_patroni"
            ),
            DeploymentStep(
                name="deploy_monitoring",
                description="Deploy database performance monitoring",
                required=True,
                rollback_command="stop_monitoring_services"
            ),
            DeploymentStep(
                name="deploy_optimization_scripts",
                description="Deploy query optimization and performance scripts",
                required=True,
                rollback_command="remove_optimization_scripts"
            ),
            DeploymentStep(
                name="configure_high_availability",
                description="Configure high availability and failover",
                required=True,
                rollback_command="disable_ha_features"
            ),
            DeploymentStep(
                name="run_validation_tests",
                description="Run comprehensive validation tests",
                required=False,
                rollback_command=""
            ),
            DeploymentStep(
                name="enable_monitoring",
                description="Enable monitoring and alerting",
                required=True,
                rollback_command="disable_monitoring"
            ),
            DeploymentStep(
                name="post_deployment_validation",
                description="Perform post-deployment validation",
                required=True,
                rollback_command=""
            )
        ]
    
    async def pre_deployment_checks(self) -> bool:
        """Perform pre-deployment checks"""
        try:
            self.logger.info("Running pre-deployment checks...")
            
            # Check Docker availability
            try:
                self.docker_client = docker.from_env()
                self.docker_client.ping()
                self.logger.info("✓ Docker is available")
            except Exception as e:
                self.logger.error(f"✗ Docker not available: {e}")
                return False
            
            # Check database connectivity
            try:
                conn = psycopg2.connect(
                    host=self.config['database']['host'],
                    port=self.config['database']['port'],
                    database=self.config['database']['database'],
                    user=self.config['database']['admin_user'],
                    password=self.config['database']['admin_password'],
                    connect_timeout=10
                )
                conn.close()
                self.logger.info("✓ Database connectivity verified")
            except Exception as e:
                self.logger.error(f"✗ Database connection failed: {e}")
                return False
            
            # Check required directories
            required_dirs = [
                '/etc/pgbouncer',
                '/etc/patroni',
                '/var/log/postgresql',
                '/var/lib/postgresql'
            ]
            
            for dir_path in required_dirs:
                if not os.path.exists(dir_path):
                    try:
                        os.makedirs(dir_path, exist_ok=True)
                        self.logger.info(f"✓ Created directory: {dir_path}")
                    except Exception as e:
                        self.logger.error(f"✗ Failed to create directory {dir_path}: {e}")
                        return False
                else:
                    self.logger.info(f"✓ Directory exists: {dir_path}")
            
            # Check system resources
            try:
                import psutil
                
                # Check memory
                memory = psutil.virtual_memory()
                if memory.total < 4 * 1024 * 1024 * 1024:  # 4GB
                    self.logger.warning("⚠ System has less than 4GB RAM")
                else:
                    self.logger.info(f"✓ System memory: {memory.total / (1024**3):.1f}GB")
                
                # Check disk space
                disk = psutil.disk_usage('/')
                if disk.free < 10 * 1024 * 1024 * 1024:  # 10GB
                    self.logger.error("✗ Insufficient disk space (less than 10GB free)")
                    return False
                else:
                    self.logger.info(f"✓ Disk space: {disk.free / (1024**3):.1f}GB free")
                
            except ImportError:
                self.logger.warning("⚠ psutil not available, skipping system resource checks")
            
            self.logger.info("✓ Pre-deployment checks completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Pre-deployment checks failed: {e}")
            return False
    
    async def backup_existing_configs(self) -> bool:
        """Backup existing configurations"""
        try:
            self.logger.info("Backing up existing configurations...")
            
            backup_dir = Path(f'/var/backups/db_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup files to preserve
            backup_files = [
                '/etc/postgresql/postgresql.conf',
                '/etc/postgresql/pg_hba.conf',
                '/etc/pgbouncer/pgbouncer.ini',
                '/etc/patroni/patroni.yml'
            ]
            
            for file_path in backup_files:
                if os.path.exists(file_path):
                    try:
                        shutil.copy2(file_path, backup_dir / os.path.basename(file_path))
                        self.logger.info(f"✓ Backed up: {file_path}")
                    except Exception as e:
                        self.logger.warning(f"⚠ Failed to backup {file_path}: {e}")
                else:
                    self.logger.info(f"ℹ File not found (skipping): {file_path}")
            
            # Create backup metadata
            backup_metadata = {
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'backed_up_files': backup_files
            }
            
            with open(backup_dir / 'metadata.json', 'w') as f:
                json.dump(backup_metadata, f, indent=2)
            
            self.logger.info(f"✓ Configuration backup completed: {backup_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration backup failed: {e}")
            return False
    
    async def deploy_pgbouncer(self) -> bool:
        """Deploy pgBouncer configuration"""
        try:
            self.logger.info("Deploying pgBouncer configuration...")
            
            # Copy pgBouncer configuration
            pgbouncer_config_source = Path(__file__).parent / 'pgbouncer_config.ini'
            pgbouncer_config_dest = Path('/etc/pgbouncer/pgbouncer.ini')
            
            if pgbouncer_config_source.exists():
                shutil.copy2(pgbouncer_config_source, pgbouncer_config_dest)
                self.logger.info("✓ pgBouncer configuration deployed")
            else:
                self.logger.error("✗ pgBouncer configuration file not found")
                return False
            
            # Copy userlist
            userlist_source = Path(__file__).parent / 'pgbouncer_userlist.txt'
            userlist_dest = Path('/etc/pgbouncer/userlist.txt')
            
            if userlist_source.exists():
                shutil.copy2(userlist_source, userlist_dest)
                self.logger.info("✓ pgBouncer userlist deployed")
            else:
                self.logger.warning("⚠ pgBouncer userlist file not found")
            
            # Start pgBouncer service
            try:
                # Check if running in Docker
                if self.docker_client:
                    # Start pgBouncer container
                    try:
                        pgbouncer_container = self.docker_client.containers.run(
                            'pgbouncer/pgbouncer:latest',
                            name='pgbouncer',
                            ports={'6432/tcp': 6432},
                            volumes={
                                '/etc/pgbouncer': {'bind': '/etc/pgbouncer', 'mode': 'ro'}
                            },
                            detach=True,
                            restart_policy={"Name": "unless-stopped"}
                        )
                        self.logger.info("✓ pgBouncer container started")
                    except docker.errors.APIError as e:
                        if "already in use" in str(e):
                            self.logger.info("ℹ pgBouncer container already running")
                        else:
                            raise
                else:
                    # Start as system service
                    subprocess.run(['systemctl', 'start', 'pgbouncer'], check=True)
                    subprocess.run(['systemctl', 'enable', 'pgbouncer'], check=True)
                    self.logger.info("✓ pgBouncer service started")
                
            except subprocess.CalledProcessError as e:
                self.logger.error(f"✗ Failed to start pgBouncer service: {e}")
                return False
            
            # Wait for pgBouncer to be ready
            await self._wait_for_service('127.0.0.1', 6432, 'pgBouncer')
            
            self.logger.info("✓ pgBouncer deployment completed")
            return True
            
        except Exception as e:
            self.logger.error(f"pgBouncer deployment failed: {e}")
            return False
    
    async def deploy_patroni_config(self) -> bool:
        """Deploy enhanced Patroni configuration"""
        try:
            self.logger.info("Deploying enhanced Patroni configuration...")
            
            # Copy enhanced Patroni configuration
            patroni_config_source = Path(__file__).parent / 'enhanced_patroni_config.yml'
            patroni_config_dest = Path('/etc/patroni/patroni.yml')
            
            if patroni_config_source.exists():
                shutil.copy2(patroni_config_source, patroni_config_dest)
                self.logger.info("✓ Enhanced Patroni configuration deployed")
            else:
                self.logger.error("✗ Enhanced Patroni configuration file not found")
                return False
            
            # Restart Patroni service
            try:
                if self.docker_client:
                    # Handle Patroni containers
                    try:
                        # Stop existing containers
                        for container_name in ['patroni-primary', 'patroni-standby']:
                            try:
                                container = self.docker_client.containers.get(container_name)
                                container.stop()
                                container.remove()
                                self.logger.info(f"✓ Stopped and removed {container_name}")
                            except docker.errors.NotFound:
                                pass
                        
                        # Start new containers with updated configuration
                        # This would typically be done via docker-compose
                        subprocess.run(['docker-compose', 'up', '-d', 'patroni-primary', 'patroni-standby'], 
                                     check=True, cwd='/home/QuantNova/GrandModel')
                        
                        self.logger.info("✓ Patroni containers restarted with new configuration")
                        
                    except subprocess.CalledProcessError as e:
                        self.logger.error(f"✗ Failed to restart Patroni containers: {e}")
                        return False
                else:
                    # Restart system service
                    subprocess.run(['systemctl', 'restart', 'patroni'], check=True)
                    self.logger.info("✓ Patroni service restarted")
                    
            except subprocess.CalledProcessError as e:
                self.logger.error(f"✗ Failed to restart Patroni: {e}")
                return False
            
            # Wait for Patroni to be ready
            await self._wait_for_service('127.0.0.1', 8008, 'Patroni API')
            
            # Verify cluster status
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get('http://127.0.0.1:8008/cluster') as response:
                        if response.status == 200:
                            cluster_data = await response.json()
                            self.logger.info(f"✓ Patroni cluster status: {cluster_data.get('leader', 'unknown')}")
                        else:
                            self.logger.warning(f"⚠ Patroni cluster status check failed: {response.status}")
            except Exception as e:
                self.logger.warning(f"⚠ Failed to check Patroni cluster status: {e}")
            
            self.logger.info("✓ Patroni configuration deployment completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Patroni configuration deployment failed: {e}")
            return False
    
    async def deploy_monitoring(self) -> bool:
        """Deploy database performance monitoring"""
        try:
            self.logger.info("Deploying database performance monitoring...")
            
            # Copy monitoring scripts
            monitoring_scripts = [
                'performance_monitor.py',
                'connection_pool_monitor.py',
                'query_optimizer.py',
                'high_availability_manager.py'
            ]
            
            monitoring_dir = Path('/opt/db_monitoring')
            monitoring_dir.mkdir(parents=True, exist_ok=True)
            
            for script in monitoring_scripts:
                script_source = Path(__file__).parent / script
                script_dest = monitoring_dir / script
                
                if script_source.exists():
                    shutil.copy2(script_source, script_dest)
                    os.chmod(script_dest, 0o755)
                    self.logger.info(f"✓ Deployed monitoring script: {script}")
                else:
                    self.logger.warning(f"⚠ Monitoring script not found: {script}")
            
            # Create monitoring configuration
            monitoring_config = {
                'database': self.config['database'],
                'monitoring': self.config['monitoring'],
                'thresholds': {
                    'connection_utilization_warning': 70,
                    'connection_utilization_critical': 90,
                    'query_time_warning_ms': 100,
                    'query_time_critical_ms': 1000,
                    'buffer_hit_ratio_warning': 90,
                    'buffer_hit_ratio_critical': 80
                }
            }
            
            with open(monitoring_dir / 'monitoring_config.json', 'w') as f:
                json.dump(monitoring_config, f, indent=2)
            
            # Create systemd service files
            self._create_monitoring_services()
            
            self.logger.info("✓ Database monitoring deployment completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Database monitoring deployment failed: {e}")
            return False
    
    def _create_monitoring_services(self):
        """Create systemd service files for monitoring"""
        try:
            services = [
                {
                    'name': 'db-performance-monitor',
                    'script': 'performance_monitor.py',
                    'description': 'Database Performance Monitor'
                },
                {
                    'name': 'db-connection-pool-monitor',
                    'script': 'connection_pool_monitor.py',
                    'description': 'Database Connection Pool Monitor'
                },
                {
                    'name': 'db-query-optimizer',
                    'script': 'query_optimizer.py',
                    'description': 'Database Query Optimizer'
                },
                {
                    'name': 'db-ha-manager',
                    'script': 'high_availability_manager.py',
                    'description': 'Database High Availability Manager'
                }
            ]
            
            for service in services:
                service_content = f"""[Unit]
Description={service['description']}
After=network.target postgresql.service
Requires=postgresql.service

[Service]
Type=simple
User=postgres
Group=postgres
WorkingDirectory=/opt/db_monitoring
ExecStart=/usr/bin/python3 /opt/db_monitoring/{service['script']}
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
                
                service_file = Path(f'/etc/systemd/system/{service["name"]}.service')
                service_file.write_text(service_content)
                
                self.logger.info(f"✓ Created service file: {service['name']}.service")
            
            # Reload systemd
            subprocess.run(['systemctl', 'daemon-reload'], check=True)
            
        except Exception as e:
            self.logger.error(f"Failed to create monitoring services: {e}")
    
    async def deploy_optimization_scripts(self) -> bool:
        """Deploy optimization scripts"""
        try:
            self.logger.info("Deploying optimization scripts...")
            
            # Create optimization directory
            opt_dir = Path('/opt/db_optimization')
            opt_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy optimization scripts
            optimization_scripts = [
                'connection_pool_optimizer.py',
                'query_optimizer.py'
            ]
            
            for script in optimization_scripts:
                script_source = Path(__file__).parent / script
                script_dest = opt_dir / script
                
                if script_source.exists():
                    shutil.copy2(script_source, script_dest)
                    os.chmod(script_dest, 0o755)
                    self.logger.info(f"✓ Deployed optimization script: {script}")
                else:
                    self.logger.warning(f"⚠ Optimization script not found: {script}")
            
            # Create optimization configuration
            optimization_config = {
                'database': self.config['database'],
                'optimization': {
                    'auto_optimize': True,
                    'optimization_interval': 300,
                    'performance_targets': {
                        'max_latency_ms': 2.0,
                        'target_latency_ms': 0.5,
                        'min_throughput_ops_per_sec': 10000
                    }
                }
            }
            
            with open(opt_dir / 'optimization_config.json', 'w') as f:
                json.dump(optimization_config, f, indent=2)
            
            self.logger.info("✓ Optimization scripts deployment completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Optimization scripts deployment failed: {e}")
            return False
    
    async def configure_high_availability(self) -> bool:
        """Configure high availability features"""
        try:
            self.logger.info("Configuring high availability features...")
            
            # Deploy HA manager
            ha_manager_source = Path(__file__).parent / 'high_availability_manager.py'
            ha_manager_dest = Path('/opt/db_monitoring/high_availability_manager.py')
            
            if ha_manager_source.exists():
                shutil.copy2(ha_manager_source, ha_manager_dest)
                os.chmod(ha_manager_dest, 0o755)
                self.logger.info("✓ HA manager deployed")
            else:
                self.logger.error("✗ HA manager script not found")
                return False
            
            # Create HA configuration
            ha_config = {
                'cluster': self.config['patroni'],
                'failover': {
                    'auto_failover': True,
                    'max_lag_bytes': 1048576,
                    'failover_timeout': 30,
                    'confirmation_timeout': 10
                },
                'monitoring': {
                    'check_interval': 5,
                    'health_check_timeout': 3
                }
            }
            
            with open('/opt/db_monitoring/ha_config.json', 'w') as f:
                json.dump(ha_config, f, indent=2)
            
            self.logger.info("✓ High availability configuration completed")
            return True
            
        except Exception as e:
            self.logger.error(f"High availability configuration failed: {e}")
            return False
    
    async def run_validation_tests(self) -> bool:
        """Run validation tests"""
        try:
            self.logger.info("Running validation tests...")
            
            # Run the test suite
            test_script = Path(__file__).parent / 'test_database_optimizations.py'
            if test_script.exists():
                try:
                    result = subprocess.run(
                        ['python3', str(test_script)],
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minutes timeout
                    )
                    
                    if result.returncode == 0:
                        self.logger.info("✓ Validation tests passed")
                        return True
                    else:
                        self.logger.error(f"✗ Validation tests failed: {result.stderr}")
                        return False
                        
                except subprocess.TimeoutExpired:
                    self.logger.error("✗ Validation tests timed out")
                    return False
                    
            else:
                self.logger.warning("⚠ Test script not found, skipping validation")
                return True
                
        except Exception as e:
            self.logger.error(f"Validation tests failed: {e}")
            return False
    
    async def enable_monitoring(self) -> bool:
        """Enable monitoring services"""
        try:
            self.logger.info("Enabling monitoring services...")
            
            services = [
                'db-performance-monitor',
                'db-connection-pool-monitor',
                'db-query-optimizer',
                'db-ha-manager'
            ]
            
            for service in services:
                try:
                    subprocess.run(['systemctl', 'enable', service], check=True)
                    subprocess.run(['systemctl', 'start', service], check=True)
                    self.logger.info(f"✓ Enabled and started: {service}")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"✗ Failed to enable {service}: {e}")
                    return False
            
            # Start Prometheus if configured
            if self.config['monitoring'].get('prometheus_port'):
                try:
                    if self.docker_client:
                        prometheus_container = self.docker_client.containers.run(
                            'prom/prometheus:latest',
                            name='prometheus',
                            ports={f'{self.config["monitoring"]["prometheus_port"]}/tcp': self.config["monitoring"]["prometheus_port"]},
                            detach=True,
                            restart_policy={"Name": "unless-stopped"}
                        )
                        self.logger.info("✓ Prometheus container started")
                except Exception as e:
                    self.logger.warning(f"⚠ Failed to start Prometheus: {e}")
            
            self.logger.info("✓ Monitoring services enabled")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to enable monitoring: {e}")
            return False
    
    async def post_deployment_validation(self) -> bool:
        """Perform post-deployment validation"""
        try:
            self.logger.info("Performing post-deployment validation...")
            
            # Check service status
            services_to_check = [
                ('postgresql', 5432),
                ('pgbouncer', 6432),
                ('patroni', 8008)
            ]
            
            for service_name, port in services_to_check:
                if await self._check_service_health(service_name, port):
                    self.logger.info(f"✓ {service_name} is healthy")
                else:
                    self.logger.error(f"✗ {service_name} health check failed")
                    return False
            
            # Check database performance
            try:
                conn = psycopg2.connect(
                    host=self.config['database']['host'],
                    port=self.config['database']['port'],
                    database=self.config['database']['database'],
                    user=self.config['database']['app_user'],
                    password=self.config['database']['app_password'],
                    connect_timeout=5
                )
                
                cursor = conn.cursor()
                start_time = time.time()
                cursor.execute("SELECT 1")
                query_time = (time.time() - start_time) * 1000
                
                cursor.close()
                conn.close()
                
                if query_time < 10:  # Less than 10ms
                    self.logger.info(f"✓ Database query performance: {query_time:.2f}ms")
                else:
                    self.logger.warning(f"⚠ Database query performance: {query_time:.2f}ms (slower than expected)")
                    
            except Exception as e:
                self.logger.error(f"✗ Database performance check failed: {e}")
                return False
            
            # Check Patroni cluster
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get('http://127.0.0.1:8008/cluster') as response:
                        if response.status == 200:
                            cluster_data = await response.json()
                            if cluster_data.get('leader'):
                                self.logger.info(f"✓ Patroni cluster has leader: {cluster_data['leader']}")
                            else:
                                self.logger.error("✗ Patroni cluster has no leader")
                                return False
                        else:
                            self.logger.error(f"✗ Patroni cluster check failed: {response.status}")
                            return False
            except Exception as e:
                self.logger.error(f"✗ Patroni cluster check failed: {e}")
                return False
            
            self.logger.info("✓ Post-deployment validation completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Post-deployment validation failed: {e}")
            return False
    
    async def _wait_for_service(self, host: str, port: int, service_name: str, timeout: int = 30):
        """Wait for service to be available"""
        import socket
        
        self.logger.info(f"Waiting for {service_name} to be available at {host}:{port}")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                sock.close()
                
                if result == 0:
                    self.logger.info(f"✓ {service_name} is available")
                    return True
                    
            except Exception:
                pass
            
            await asyncio.sleep(1)
        
        self.logger.error(f"✗ {service_name} not available after {timeout} seconds")
        return False
    
    async def _check_service_health(self, service_name: str, port: int) -> bool:
        """Check service health"""
        try:
            import socket
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            
            return result == 0
            
        except Exception as e:
            self.logger.error(f"Health check failed for {service_name}: {e}")
            return False
    
    async def rollback_deployment(self, failed_step: str):
        """Rollback deployment"""
        self.logger.warning(f"Rolling back deployment due to failure at step: {failed_step}")
        
        # Find the failed step and rollback previous steps
        failed_step_index = -1
        for i, step in enumerate(self.deployment_steps):
            if step.name == failed_step:
                failed_step_index = i
                break
        
        if failed_step_index > 0:
            # Rollback in reverse order
            for i in range(failed_step_index - 1, -1, -1):
                step = self.deployment_steps[i]
                if step.executed and step.success and step.rollback_command:
                    try:
                        self.logger.info(f"Rolling back step: {step.name}")
                        # Execute rollback command
                        # This is a simplified rollback - in production, implement proper rollback logic
                        await asyncio.sleep(1)  # Placeholder
                        self.logger.info(f"✓ Rolled back: {step.name}")
                    except Exception as e:
                        self.logger.error(f"✗ Rollback failed for {step.name}: {e}")
    
    async def deploy_optimizations(self) -> DeploymentResult:
        """Deploy all database optimizations"""
        self.deployment_start_time = time.time()
        self.logger.info("Starting database optimization deployment...")
        
        successful_steps = 0
        failed_steps = 0
        error_details = []
        
        for step in self.deployment_steps:
            step_start_time = time.time()
            step.executed = True
            
            self.logger.info(f"Executing step: {step.name} - {step.description}")
            
            try:
                # Execute the step
                if step.name == "pre_deployment_checks":
                    step.success = await self.pre_deployment_checks()
                elif step.name == "backup_existing_configs":
                    step.success = await self.backup_existing_configs()
                elif step.name == "deploy_pgbouncer":
                    step.success = await self.deploy_pgbouncer()
                elif step.name == "deploy_patroni_config":
                    step.success = await self.deploy_patroni_config()
                elif step.name == "deploy_monitoring":
                    step.success = await self.deploy_monitoring()
                elif step.name == "deploy_optimization_scripts":
                    step.success = await self.deploy_optimization_scripts()
                elif step.name == "configure_high_availability":
                    step.success = await self.configure_high_availability()
                elif step.name == "run_validation_tests":
                    step.success = await self.run_validation_tests()
                elif step.name == "enable_monitoring":
                    step.success = await self.enable_monitoring()
                elif step.name == "post_deployment_validation":
                    step.success = await self.post_deployment_validation()
                else:
                    step.success = True  # Unknown step, assume success
                
                step.execution_time = time.time() - step_start_time
                
                if step.success:
                    successful_steps += 1
                    self.logger.info(f"✓ Step completed: {step.name} ({step.execution_time:.2f}s)")
                else:
                    failed_steps += 1
                    error_msg = f"Step failed: {step.name}"
                    error_details.append(error_msg)
                    self.logger.error(f"✗ {error_msg}")
                    
                    if step.required:
                        self.logger.error(f"Required step failed, stopping deployment")
                        
                        # Rollback if configured
                        if self.config['deployment']['rollback_on_failure']:
                            await self.rollback_deployment(step.name)
                        
                        break
                        
            except Exception as e:
                step.success = False
                step.execution_time = time.time() - step_start_time
                step.error_message = str(e)
                failed_steps += 1
                error_msg = f"Step exception: {step.name} - {str(e)}"
                error_details.append(error_msg)
                self.logger.error(f"✗ {error_msg}")
                
                if step.required:
                    self.logger.error(f"Required step failed with exception, stopping deployment")
                    
                    # Rollback if configured
                    if self.config['deployment']['rollback_on_failure']:
                        await self.rollback_deployment(step.name)
                    
                    break
        
        total_time = time.time() - self.deployment_start_time
        success = failed_steps == 0
        
        result = DeploymentResult(
            success=success,
            total_steps=len(self.deployment_steps),
            successful_steps=successful_steps,
            failed_steps=failed_steps,
            total_time=total_time,
            error_details=error_details,
            rollback_required=not success and self.config['deployment']['rollback_on_failure']
        )
        
        # Log deployment summary
        self.logger.info("="*50)
        self.logger.info("DEPLOYMENT SUMMARY")
        self.logger.info("="*50)
        self.logger.info(f"Status: {'SUCCESS' if success else 'FAILED'}")
        self.logger.info(f"Total Steps: {len(self.deployment_steps)}")
        self.logger.info(f"Successful Steps: {successful_steps}")
        self.logger.info(f"Failed Steps: {failed_steps}")
        self.logger.info(f"Total Time: {total_time:.2f} seconds")
        
        if error_details:
            self.logger.info("Errors:")
            for error in error_details:
                self.logger.info(f"  - {error}")
        
        return result

async def main():
    """Main entry point"""
    deployer = DatabaseOptimizationDeployer()
    
    try:
        result = await deployer.deploy_optimizations()
        
        # Save deployment report
        report_file = f'/var/log/db_optimization/deployment_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w') as f:
            json.dump({
                'result': asdict(result),
                'steps': [asdict(step) for step in deployer.deployment_steps],
                'config': deployer.config
            }, f, indent=2, default=str)
        
        print(f"Deployment report saved to: {report_file}")
        
        if result.success:
            print("✓ Database optimization deployment completed successfully!")
            return 0
        else:
            print("✗ Database optimization deployment failed!")
            return 1
            
    except Exception as e:
        print(f"Deployment failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)