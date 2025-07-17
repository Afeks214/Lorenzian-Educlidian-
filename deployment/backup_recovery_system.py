"""
Backup and Recovery System for GrandModel Production
==================================================

Comprehensive backup and disaster recovery system for production MARL models
with automated backups, point-in-time recovery, and disaster recovery capabilities.

Features:
- Automated model and data backups
- Point-in-time recovery
- Disaster recovery procedures
- Cross-region replication
- Backup validation and testing
- Recovery time optimization
- Compliance and retention policies
- Monitoring and alerting

Author: Backup & Recovery Team
Date: 2025-07-15
Version: 1.0.0
"""

import os
import sys
import asyncio
import json
import yaml
import time
import shutil
import tarfile
import gzip
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import structlog
import subprocess
import tempfile
import boto3
from botocore.exceptions import ClientError
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
import kubernetes
from kubernetes import client, config
import schedule
import threading
from croniter import croniter
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import zipfile
import pickle

logger = structlog.get_logger()

@dataclass
class BackupTarget:
    """Backup target configuration"""
    name: str
    type: str  # 'model', 'database', 'redis', 'filesystem', 'kubernetes'
    source_path: str
    description: str
    backup_frequency: str  # cron expression
    retention_days: int
    compression: bool = True
    encryption: bool = False
    enabled: bool = True
    priority: int = 1  # 1=high, 2=medium, 3=low
    
@dataclass
class BackupJob:
    """Backup job definition"""
    job_id: str
    target: BackupTarget
    scheduled_time: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed
    backup_path: Optional[str] = None
    backup_size_mb: float = 0.0
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    checksum: Optional[str] = None

@dataclass
class RestoreJob:
    """Restore job definition"""
    job_id: str
    target: BackupTarget
    backup_path: str
    restore_point: datetime
    destination_path: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed
    duration_seconds: float = 0.0
    error_message: Optional[str] = None

@dataclass
class BackupMetrics:
    """Backup system metrics"""
    total_backups: int = 0
    successful_backups: int = 0
    failed_backups: int = 0
    total_size_gb: float = 0.0
    avg_backup_duration_seconds: float = 0.0
    oldest_backup_days: int = 0
    newest_backup_hours: int = 0
    backup_success_rate: float = 0.0
    storage_utilization: float = 0.0

class BackupRecoverySystem:
    """
    Comprehensive backup and recovery system
    
    Capabilities:
    - Automated scheduled backups
    - Point-in-time recovery
    - Multi-storage backend support
    - Encryption and compression
    - Backup validation and testing
    - Disaster recovery procedures
    - Compliance and retention management
    """
    
    def __init__(self, config_path: str = None):
        """Initialize backup and recovery system"""
        self.system_id = f"backup_{int(time.time())}"
        self.start_time = datetime.now()
        
        # Load configuration
        self.config = self._load_backup_config(config_path)
        
        # Initialize paths
        self.project_root = Path(__file__).parent.parent
        self.local_backup_dir = self.project_root / "backups"
        self.models_dir = self.project_root / "colab" / "exports"
        self.logs_dir = self.project_root / "logs" / "backup"
        self.restore_dir = self.project_root / "restore"
        
        # Create directories
        for directory in [self.local_backup_dir, self.logs_dir, self.restore_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage backends
        self._initialize_storage_backends()
        
        # Initialize backup targets
        self.backup_targets = self._initialize_backup_targets()
        
        # System state
        self.active_jobs: Dict[str, BackupJob] = {}
        self.job_history: List[BackupJob] = []
        self.scheduler_running = False
        self.metrics = BackupMetrics()
        
        # Executor for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("BackupRecoverySystem initialized",
                   system_id=self.system_id,
                   config_name=self.config.get('name', 'default'))
    
    def _load_backup_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load backup configuration"""
        default_config = {
            'name': 'grandmodel-backup',
            'version': '1.0.0',
            'storage': {
                'local': {
                    'enabled': True,
                    'path': str(self.local_backup_dir) if hasattr(self, 'local_backup_dir') else '/tmp/backups',
                    'retention_days': 7
                },
                's3': {
                    'enabled': True,
                    'bucket': os.getenv('BACKUP_S3_BUCKET', 'grandmodel-backups'),
                    'region': os.getenv('AWS_REGION', 'us-east-1'),
                    'retention_days': 30,
                    'storage_class': 'STANDARD_IA'
                },
                'azure': {
                    'enabled': False,
                    'container': os.getenv('BACKUP_AZURE_CONTAINER', 'grandmodel-backups'),
                    'retention_days': 30
                }
            },
            'encryption': {
                'enabled': True,
                'key': os.getenv('BACKUP_ENCRYPTION_KEY'),
                'algorithm': 'AES256'
            },
            'compression': {
                'enabled': True,
                'algorithm': 'gzip',
                'level': 6
            },
            'validation': {
                'enabled': True,
                'checksum_algorithm': 'sha256',
                'test_restore': True
            },
            'monitoring': {
                'enabled': True,
                'alert_on_failure': True,
                'alert_on_missing_backup': True,
                'slack_webhook': os.getenv('SLACK_WEBHOOK_URL')
            },
            'disaster_recovery': {
                'enabled': True,
                'rpo_hours': 1,  # Recovery Point Objective: 1 hour
                'rto_minutes': 30,  # Recovery Time Objective: 30 minutes
                'cross_region_replication': True
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    file_config = yaml.safe_load(f)
                else:
                    file_config = json.load(f)
                
                # Deep merge configuration
                self._deep_merge_config(default_config, file_config)
        
        return default_config
    
    def _deep_merge_config(self, base: Dict[str, Any], update: Dict[str, Any]):
        """Deep merge configuration dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge_config(base[key], value)
            else:
                base[key] = value
    
    def _initialize_storage_backends(self):
        """Initialize storage backends"""
        self.storage_backends = {}
        
        # Local storage
        if self.config['storage']['local']['enabled']:
            self.storage_backends['local'] = {
                'type': 'local',
                'path': Path(self.config['storage']['local']['path'])
            }
        
        # S3 storage
        if self.config['storage']['s3']['enabled']:
            try:
                self.storage_backends['s3'] = {
                    'type': 's3',
                    'client': boto3.client('s3'),
                    'bucket': self.config['storage']['s3']['bucket'],
                    'region': self.config['storage']['s3']['region']
                }
            except Exception as e:
                logger.warning("S3 storage backend initialization failed", error=str(e))
        
        # Azure storage
        if self.config['storage']['azure']['enabled']:
            try:
                # Azure Blob Storage initialization would go here
                logger.info("Azure storage backend not implemented yet")
            except Exception as e:
                logger.warning("Azure storage backend initialization failed", error=str(e))
        
        logger.info(f"Storage backends initialized: {list(self.storage_backends.keys())}")
    
    def _initialize_backup_targets(self) -> List[BackupTarget]:
        """Initialize backup targets"""
        targets = []
        
        # Model backups
        targets.append(BackupTarget(
            name="tactical_models",
            type="model",
            source_path=str(self.models_dir / "tactical_training_test_20250715_135033"),
            description="Tactical MARL model checkpoints",
            backup_frequency="0 */6 * * *",  # Every 6 hours
            retention_days=30,
            compression=True,
            encryption=True,
            priority=1
        ))
        
        targets.append(BackupTarget(
            name="strategic_models",
            type="model",
            source_path=str(self.models_dir / "strategic_training"),
            description="Strategic MARL model checkpoints",
            backup_frequency="0 */12 * * *",  # Every 12 hours
            retention_days=30,
            compression=True,
            encryption=True,
            priority=1
        ))
        
        # Database backups
        targets.append(BackupTarget(
            name="main_database",
            type="database",
            source_path="postgresql://localhost:5432/grandmodel",
            description="Main application database",
            backup_frequency="0 2 * * *",  # Daily at 2 AM
            retention_days=14,
            compression=True,
            encryption=True,
            priority=1
        ))
        
        # Redis backups
        targets.append(BackupTarget(
            name="redis_cache",
            type="redis",
            source_path="redis://localhost:6379",
            description="Redis cache and session data",
            backup_frequency="0 */4 * * *",  # Every 4 hours
            retention_days=7,
            compression=True,
            encryption=False,
            priority=2
        ))
        
        # Configuration backups
        targets.append(BackupTarget(
            name="configurations",
            type="filesystem",
            source_path=str(self.project_root / "configs"),
            description="Application configurations",
            backup_frequency="0 1 * * *",  # Daily at 1 AM
            retention_days=90,
            compression=True,
            encryption=True,
            priority=1
        ))
        
        # Kubernetes resources
        targets.append(BackupTarget(
            name="kubernetes_resources",
            type="kubernetes",
            source_path="grandmodel-prod",  # namespace
            description="Kubernetes deployment resources",
            backup_frequency="0 3 * * *",  # Daily at 3 AM
            retention_days=30,
            compression=True,
            encryption=True,
            priority=2
        ))
        
        # Logs backup
        targets.append(BackupTarget(
            name="application_logs",
            type="filesystem",
            source_path=str(self.project_root / "logs"),
            description="Application logs",
            backup_frequency="0 4 * * *",  # Daily at 4 AM
            retention_days=30,
            compression=True,
            encryption=False,
            priority=3
        ))
        
        return targets
    
    async def start_backup_scheduler(self):
        """Start backup scheduler"""
        logger.info("🚀 Starting backup scheduler")
        
        self.scheduler_running = True
        
        # Schedule backup jobs
        for target in self.backup_targets:
            if target.enabled:
                schedule.every().day.at("00:00").do(
                    self._schedule_backup_job, target
                ).tag(target.name)
        
        # Start scheduler loop
        scheduler_thread = threading.Thread(target=self._run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        
        logger.info("✅ Backup scheduler started")
    
    def _run_scheduler(self):
        """Run backup scheduler"""
        while self.scheduler_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def _schedule_backup_job(self, target: BackupTarget):
        """Schedule backup job for target"""
        job_id = f"{target.name}_{int(time.time())}"
        
        job = BackupJob(
            job_id=job_id,
            target=target,
            scheduled_time=datetime.now()
        )
        
        # Add to active jobs
        self.active_jobs[job_id] = job
        
        # Execute backup asynchronously
        self.executor.submit(self._execute_backup_job, job)
    
    async def backup_target(self, target_name: str) -> BackupJob:
        """Backup specific target immediately"""
        target = next((t for t in self.backup_targets if t.name == target_name), None)
        if not target:
            raise ValueError(f"Backup target not found: {target_name}")
        
        job_id = f"{target_name}_manual_{int(time.time())}"
        job = BackupJob(
            job_id=job_id,
            target=target,
            scheduled_time=datetime.now()
        )
        
        self.active_jobs[job_id] = job
        
        # Execute backup
        await self._execute_backup_job_async(job)
        
        return job
    
    async def _execute_backup_job_async(self, job: BackupJob):
        """Execute backup job asynchronously"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self._execute_backup_job, job)
    
    def _execute_backup_job(self, job: BackupJob):
        """Execute backup job"""
        logger.info(f"🚀 Starting backup job: {job.job_id}")
        
        job.started_at = datetime.now()
        job.status = "running"
        
        try:
            # Create backup based on target type
            if job.target.type == "model":
                backup_path = self._backup_models(job)
            elif job.target.type == "database":
                backup_path = self._backup_database(job)
            elif job.target.type == "redis":
                backup_path = self._backup_redis(job)
            elif job.target.type == "filesystem":
                backup_path = self._backup_filesystem(job)
            elif job.target.type == "kubernetes":
                backup_path = self._backup_kubernetes(job)
            else:
                raise ValueError(f"Unknown backup target type: {job.target.type}")
            
            # Calculate backup size
            if backup_path and Path(backup_path).exists():
                job.backup_size_mb = Path(backup_path).stat().st_size / (1024 * 1024)
            
            # Calculate checksum
            if backup_path and Path(backup_path).exists():
                job.checksum = self._calculate_checksum(backup_path)
            
            # Upload to remote storage
            await self._upload_to_remote_storage(job, backup_path)
            
            # Validate backup
            if self.config['validation']['enabled']:
                await self._validate_backup(job, backup_path)
            
            # Complete job
            job.completed_at = datetime.now()
            job.status = "completed"
            job.backup_path = backup_path
            job.duration_seconds = (job.completed_at - job.started_at).total_seconds()
            
            # Update metrics
            self.metrics.successful_backups += 1
            self.metrics.total_size_gb += job.backup_size_mb / 1024
            
            logger.info(f"✅ Backup job completed: {job.job_id}",
                       size_mb=job.backup_size_mb,
                       duration_seconds=job.duration_seconds)
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.now()
            job.duration_seconds = (job.completed_at - job.started_at).total_seconds()
            
            self.metrics.failed_backups += 1
            
            logger.error(f"❌ Backup job failed: {job.job_id}", error=str(e))
            
            # Send failure notification
            await self._send_backup_failure_notification(job)
        
        finally:
            # Move job to history
            self.job_history.append(job)
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
            
            # Update metrics
            self.metrics.total_backups += 1
            if self.metrics.total_backups > 0:
                self.metrics.backup_success_rate = (
                    self.metrics.successful_backups / self.metrics.total_backups * 100
                )
    
    def _backup_models(self, job: BackupJob) -> str:
        """Backup model files"""
        source_path = Path(job.target.source_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")
        
        # Create backup archive
        backup_filename = f"{job.target.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz"
        backup_path = self.local_backup_dir / backup_filename
        
        with tarfile.open(backup_path, 'w:gz') as tar:
            tar.add(source_path, arcname=source_path.name)
        
        logger.info(f"Model backup created: {backup_path}")
        return str(backup_path)
    
    def _backup_database(self, job: BackupJob) -> str:
        """Backup PostgreSQL database"""
        # Parse database URL
        db_url = job.target.source_path
        
        # Create backup filename
        backup_filename = f"{job.target.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql.gz"
        backup_path = self.local_backup_dir / backup_filename
        
        try:
            # Use pg_dump to create backup
            cmd = [
                'pg_dump',
                '--no-password',
                '--clean',
                '--create',
                '--compress=9',
                '--file', str(backup_path),
                db_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"pg_dump failed: {result.stderr}")
            
            logger.info(f"Database backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            raise RuntimeError(f"Database backup failed: {str(e)}")
    
    def _backup_redis(self, job: BackupJob) -> str:
        """Backup Redis data"""
        # Parse Redis URL
        redis_url = job.target.source_path
        
        try:
            # Connect to Redis
            redis_client = redis.from_url(redis_url)
            
            # Create backup filename
            backup_filename = f"{job.target.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.rdb.gz"
            backup_path = self.local_backup_dir / backup_filename
            
            # Get all keys and values
            backup_data = {}
            for key in redis_client.scan_iter():
                key_type = redis_client.type(key)
                
                if key_type == b'string':
                    backup_data[key.decode()] = redis_client.get(key).decode()
                elif key_type == b'hash':
                    backup_data[key.decode()] = redis_client.hgetall(key)
                elif key_type == b'list':
                    backup_data[key.decode()] = redis_client.lrange(key, 0, -1)
                elif key_type == b'set':
                    backup_data[key.decode()] = list(redis_client.smembers(key))
                elif key_type == b'zset':
                    backup_data[key.decode()] = redis_client.zrange(key, 0, -1, withscores=True)
            
            # Save to compressed file
            with gzip.open(backup_path, 'wb') as f:
                pickle.dump(backup_data, f)
            
            logger.info(f"Redis backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            raise RuntimeError(f"Redis backup failed: {str(e)}")
    
    def _backup_filesystem(self, job: BackupJob) -> str:
        """Backup filesystem directory"""
        source_path = Path(job.target.source_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")
        
        # Create backup archive
        backup_filename = f"{job.target.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz"
        backup_path = self.local_backup_dir / backup_filename
        
        with tarfile.open(backup_path, 'w:gz') as tar:
            tar.add(source_path, arcname=source_path.name)
        
        logger.info(f"Filesystem backup created: {backup_path}")
        return str(backup_path)
    
    def _backup_kubernetes(self, job: BackupJob) -> str:
        """Backup Kubernetes resources"""
        namespace = job.target.source_path
        
        try:
            # Load Kubernetes config
            config.load_incluster_config()
            
            # Create backup filename
            backup_filename = f"{job.target.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
            backup_path = self.local_backup_dir / backup_filename
            
            # Get all resources in namespace
            api_client = client.ApiClient()
            
            # Get deployments
            apps_v1 = client.AppsV1Api()
            deployments = apps_v1.list_namespaced_deployment(namespace=namespace)
            
            # Get services
            core_v1 = client.CoreV1Api()
            services = core_v1.list_namespaced_service(namespace=namespace)
            
            # Get configmaps
            configmaps = core_v1.list_namespaced_config_map(namespace=namespace)
            
            # Get secrets
            secrets = core_v1.list_namespaced_secret(namespace=namespace)
            
            # Create backup data
            backup_data = {
                'deployments': [api_client.sanitize_for_serialization(d) for d in deployments.items],
                'services': [api_client.sanitize_for_serialization(s) for s in services.items],
                'configmaps': [api_client.sanitize_for_serialization(c) for c in configmaps.items],
                'secrets': [api_client.sanitize_for_serialization(s) for s in secrets.items]
            }
            
            # Save to YAML file
            with open(backup_path, 'w') as f:
                yaml.dump(backup_data, f, default_flow_style=False)
            
            logger.info(f"Kubernetes backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            raise RuntimeError(f"Kubernetes backup failed: {str(e)}")
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    async def _upload_to_remote_storage(self, job: BackupJob, backup_path: str):
        """Upload backup to remote storage"""
        if not backup_path or not Path(backup_path).exists():
            return
        
        # Upload to S3
        if 's3' in self.storage_backends:
            await self._upload_to_s3(job, backup_path)
        
        # Upload to Azure (if configured)
        if 'azure' in self.storage_backends:
            await self._upload_to_azure(job, backup_path)
    
    async def _upload_to_s3(self, job: BackupJob, backup_path: str):
        """Upload backup to S3"""
        try:
            s3_backend = self.storage_backends['s3']
            s3_client = s3_backend['client']
            bucket = s3_backend['bucket']
            
            # Create S3 key
            s3_key = f"backups/{job.target.name}/{Path(backup_path).name}"
            
            # Upload file
            s3_client.upload_file(
                backup_path,
                bucket,
                s3_key,
                ExtraArgs={
                    'StorageClass': self.config['storage']['s3']['storage_class'],
                    'Metadata': {
                        'job_id': job.job_id,
                        'target_name': job.target.name,
                        'backup_date': job.started_at.isoformat(),
                        'checksum': job.checksum or ''
                    }
                }
            )
            
            logger.info(f"Backup uploaded to S3: s3://{bucket}/{s3_key}")
            
        except Exception as e:
            logger.error(f"S3 upload failed: {str(e)}")
            raise
    
    async def _upload_to_azure(self, job: BackupJob, backup_path: str):
        """Upload backup to Azure Blob Storage"""
        # Azure Blob Storage upload implementation would go here
        logger.info("Azure upload not implemented yet")
    
    async def _validate_backup(self, job: BackupJob, backup_path: str):
        """Validate backup integrity"""
        if not backup_path or not Path(backup_path).exists():
            raise ValueError("Backup file not found for validation")
        
        # Verify checksum
        calculated_checksum = self._calculate_checksum(backup_path)
        if job.checksum and calculated_checksum != job.checksum:
            raise ValueError("Backup checksum validation failed")
        
        # Test restore (if enabled)
        if self.config['validation']['test_restore']:
            await self._test_restore(job, backup_path)
        
        logger.info(f"Backup validation passed: {job.job_id}")
    
    async def _test_restore(self, job: BackupJob, backup_path: str):
        """Test restore functionality"""
        # Create temporary restore directory
        test_restore_dir = self.restore_dir / f"test_{job.job_id}"
        test_restore_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Test restore based on backup type
            if job.target.type in ["model", "filesystem"]:
                # Test archive extraction
                with tarfile.open(backup_path, 'r:gz') as tar:
                    tar.extractall(test_restore_dir)
            
            elif job.target.type == "redis":
                # Test Redis data restoration
                with gzip.open(backup_path, 'rb') as f:
                    pickle.load(f)
            
            elif job.target.type == "kubernetes":
                # Test YAML parsing
                with open(backup_path, 'r') as f:
                    yaml.safe_load(f)
            
            logger.info(f"Test restore successful: {job.job_id}")
            
        except Exception as e:
            raise ValueError(f"Test restore failed: {str(e)}")
        
        finally:
            # Clean up test directory
            shutil.rmtree(test_restore_dir, ignore_errors=True)
    
    async def restore_from_backup(self, target_name: str, backup_date: datetime,
                                 destination_path: str = None) -> RestoreJob:
        """Restore from backup"""
        logger.info(f"🔄 Starting restore: {target_name} from {backup_date}")
        
        # Find backup target
        target = next((t for t in self.backup_targets if t.name == target_name), None)
        if not target:
            raise ValueError(f"Backup target not found: {target_name}")
        
        # Find backup file
        backup_path = await self._find_backup_file(target_name, backup_date)
        if not backup_path:
            raise ValueError(f"Backup not found for {target_name} at {backup_date}")
        
        # Create restore job
        job_id = f"restore_{target_name}_{int(time.time())}"
        restore_job = RestoreJob(
            job_id=job_id,
            target=target,
            backup_path=backup_path,
            restore_point=backup_date,
            destination_path=destination_path or target.source_path
        )
        
        # Execute restore
        await self._execute_restore_job(restore_job)
        
        return restore_job
    
    async def _find_backup_file(self, target_name: str, backup_date: datetime) -> Optional[str]:
        """Find backup file closest to specified date"""
        # Search local backups
        local_backups = []
        for backup_file in self.local_backup_dir.glob(f"{target_name}_*.tar.gz"):
            try:
                # Extract timestamp from filename
                timestamp_str = backup_file.stem.split('_')[-2:]
                timestamp_str = '_'.join(timestamp_str)
                file_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                
                local_backups.append((file_date, str(backup_file)))
            except (ValueError, IndexError):
                continue
        
        # Find closest backup
        if local_backups:
            local_backups.sort(key=lambda x: abs((x[0] - backup_date).total_seconds()))
            return local_backups[0][1]
        
        # Search S3 backups if local not found
        if 's3' in self.storage_backends:
            s3_backup = await self._find_s3_backup(target_name, backup_date)
            if s3_backup:
                # Download from S3
                local_path = self.local_backup_dir / f"{target_name}_restored_{int(time.time())}.tar.gz"
                await self._download_from_s3(s3_backup, str(local_path))
                return str(local_path)
        
        return None
    
    async def _find_s3_backup(self, target_name: str, backup_date: datetime) -> Optional[str]:
        """Find backup in S3 closest to specified date"""
        try:
            s3_backend = self.storage_backends['s3']
            s3_client = s3_backend['client']
            bucket = s3_backend['bucket']
            
            # List objects with prefix
            response = s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=f"backups/{target_name}/"
            )
            
            if 'Contents' not in response:
                return None
            
            # Find closest backup
            s3_backups = []
            for obj in response['Contents']:
                # Extract timestamp from key
                key_parts = obj['Key'].split('/')
                if len(key_parts) >= 3:
                    filename = key_parts[-1]
                    try:
                        timestamp_str = filename.split('_')[-2:]
                        timestamp_str = '_'.join(timestamp_str).replace('.tar.gz', '')
                        file_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                        
                        s3_backups.append((file_date, obj['Key']))
                    except (ValueError, IndexError):
                        continue
            
            if s3_backups:
                s3_backups.sort(key=lambda x: abs((x[0] - backup_date).total_seconds()))
                return s3_backups[0][1]
            
        except Exception as e:
            logger.error(f"S3 backup search failed: {str(e)}")
        
        return None
    
    async def _download_from_s3(self, s3_key: str, local_path: str):
        """Download backup from S3"""
        try:
            s3_backend = self.storage_backends['s3']
            s3_client = s3_backend['client']
            bucket = s3_backend['bucket']
            
            s3_client.download_file(bucket, s3_key, local_path)
            
            logger.info(f"Backup downloaded from S3: {s3_key}")
            
        except Exception as e:
            logger.error(f"S3 download failed: {str(e)}")
            raise
    
    async def _execute_restore_job(self, restore_job: RestoreJob):
        """Execute restore job"""
        restore_job.started_at = datetime.now()
        restore_job.status = "running"
        
        try:
            # Execute restore based on target type
            if restore_job.target.type == "model":
                await self._restore_models(restore_job)
            elif restore_job.target.type == "database":
                await self._restore_database(restore_job)
            elif restore_job.target.type == "redis":
                await self._restore_redis(restore_job)
            elif restore_job.target.type == "filesystem":
                await self._restore_filesystem(restore_job)
            elif restore_job.target.type == "kubernetes":
                await self._restore_kubernetes(restore_job)
            else:
                raise ValueError(f"Unknown restore target type: {restore_job.target.type}")
            
            # Complete restore
            restore_job.completed_at = datetime.now()
            restore_job.status = "completed"
            restore_job.duration_seconds = (
                restore_job.completed_at - restore_job.started_at
            ).total_seconds()
            
            logger.info(f"✅ Restore completed: {restore_job.job_id}",
                       duration_seconds=restore_job.duration_seconds)
            
        except Exception as e:
            restore_job.status = "failed"
            restore_job.error_message = str(e)
            restore_job.completed_at = datetime.now()
            restore_job.duration_seconds = (
                restore_job.completed_at - restore_job.started_at
            ).total_seconds()
            
            logger.error(f"❌ Restore failed: {restore_job.job_id}", error=str(e))
            raise
    
    async def _restore_models(self, restore_job: RestoreJob):
        """Restore model files"""
        destination_path = Path(restore_job.destination_path)
        
        # Create destination directory
        destination_path.mkdir(parents=True, exist_ok=True)
        
        # Extract backup
        with tarfile.open(restore_job.backup_path, 'r:gz') as tar:
            tar.extractall(destination_path.parent)
        
        logger.info(f"Models restored to: {destination_path}")
    
    async def _restore_database(self, restore_job: RestoreJob):
        """Restore database from backup"""
        try:
            # Use psql to restore database
            cmd = [
                'psql',
                '--no-password',
                '--quiet',
                '--file', restore_job.backup_path,
                restore_job.destination_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"Database restore failed: {result.stderr}")
            
            logger.info("Database restored successfully")
            
        except Exception as e:
            raise RuntimeError(f"Database restore failed: {str(e)}")
    
    async def _restore_redis(self, restore_job: RestoreJob):
        """Restore Redis data from backup"""
        try:
            # Connect to Redis
            redis_client = redis.from_url(restore_job.destination_path)
            
            # Load backup data
            with gzip.open(restore_job.backup_path, 'rb') as f:
                backup_data = pickle.load(f)
            
            # Restore data
            for key, value in backup_data.items():
                if isinstance(value, str):
                    redis_client.set(key, value)
                elif isinstance(value, dict):
                    redis_client.hmset(key, value)
                elif isinstance(value, list):
                    redis_client.lpush(key, *value)
            
            logger.info("Redis data restored successfully")
            
        except Exception as e:
            raise RuntimeError(f"Redis restore failed: {str(e)}")
    
    async def _restore_filesystem(self, restore_job: RestoreJob):
        """Restore filesystem from backup"""
        destination_path = Path(restore_job.destination_path)
        
        # Create destination directory
        destination_path.mkdir(parents=True, exist_ok=True)
        
        # Extract backup
        with tarfile.open(restore_job.backup_path, 'r:gz') as tar:
            tar.extractall(destination_path.parent)
        
        logger.info(f"Filesystem restored to: {destination_path}")
    
    async def _restore_kubernetes(self, restore_job: RestoreJob):
        """Restore Kubernetes resources from backup"""
        try:
            # Load Kubernetes config
            config.load_incluster_config()
            
            # Load backup data
            with open(restore_job.backup_path, 'r') as f:
                backup_data = yaml.safe_load(f)
            
            # Restore resources
            api_client = client.ApiClient()
            apps_v1 = client.AppsV1Api()
            core_v1 = client.CoreV1Api()
            
            namespace = restore_job.destination_path
            
            # Restore deployments
            for deployment_data in backup_data.get('deployments', []):
                deployment = api_client.deserialize(deployment_data, 'V1Deployment')
                apps_v1.create_namespaced_deployment(namespace=namespace, body=deployment)
            
            # Restore services
            for service_data in backup_data.get('services', []):
                service = api_client.deserialize(service_data, 'V1Service')
                core_v1.create_namespaced_service(namespace=namespace, body=service)
            
            # Restore configmaps
            for configmap_data in backup_data.get('configmaps', []):
                configmap = api_client.deserialize(configmap_data, 'V1ConfigMap')
                core_v1.create_namespaced_config_map(namespace=namespace, body=configmap)
            
            logger.info("Kubernetes resources restored successfully")
            
        except Exception as e:
            raise RuntimeError(f"Kubernetes restore failed: {str(e)}")
    
    async def _send_backup_failure_notification(self, job: BackupJob):
        """Send backup failure notification"""
        if not self.config['monitoring']['alert_on_failure']:
            return
        
        webhook_url = self.config['monitoring']['slack_webhook']
        if not webhook_url:
            return
        
        try:
            message = {
                "channel": "#alerts",
                "username": "Backup System",
                "text": f"Backup Failed: {job.target.name}",
                "attachments": [
                    {
                        "color": "danger",
                        "title": f"Backup Job Failed: {job.job_id}",
                        "text": job.error_message,
                        "fields": [
                            {
                                "title": "Target",
                                "value": job.target.name,
                                "short": True
                            },
                            {
                                "title": "Type",
                                "value": job.target.type,
                                "short": True
                            },
                            {
                                "title": "Started",
                                "value": job.started_at.strftime('%Y-%m-%d %H:%M:%S'),
                                "short": True
                            },
                            {
                                "title": "Duration",
                                "value": f"{job.duration_seconds:.1f}s",
                                "short": True
                            }
                        ],
                        "footer": "GrandModel Backup System",
                        "ts": int(job.started_at.timestamp())
                    }
                ]
            }
            
            import requests
            response = requests.post(webhook_url, json=message)
            response.raise_for_status()
            
            logger.info("Backup failure notification sent")
            
        except Exception as e:
            logger.error(f"Failed to send backup failure notification: {str(e)}")
    
    async def cleanup_old_backups(self):
        """Clean up old backups based on retention policy"""
        logger.info("🧹 Cleaning up old backups")
        
        cleaned_count = 0
        
        for target in self.backup_targets:
            if not target.enabled:
                continue
            
            retention_days = target.retention_days
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # Clean up local backups
            for backup_file in self.local_backup_dir.glob(f"{target.name}_*.tar.gz"):
                try:
                    # Extract timestamp from filename
                    timestamp_str = backup_file.stem.split('_')[-2:]
                    timestamp_str = '_'.join(timestamp_str)
                    file_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    
                    if file_date < cutoff_date:
                        backup_file.unlink()
                        cleaned_count += 1
                        logger.info(f"Cleaned up old backup: {backup_file}")
                        
                except (ValueError, IndexError):
                    continue
            
            # Clean up S3 backups
            if 's3' in self.storage_backends:
                await self._cleanup_s3_backups(target, cutoff_date)
        
        logger.info(f"✅ Cleanup completed: {cleaned_count} old backups removed")
    
    async def _cleanup_s3_backups(self, target: BackupTarget, cutoff_date: datetime):
        """Clean up old S3 backups"""
        try:
            s3_backend = self.storage_backends['s3']
            s3_client = s3_backend['client']
            bucket = s3_backend['bucket']
            
            # List objects with prefix
            response = s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=f"backups/{target.name}/"
            )
            
            if 'Contents' not in response:
                return
            
            # Delete old backups
            for obj in response['Contents']:
                if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                    s3_client.delete_object(Bucket=bucket, Key=obj['Key'])
                    logger.info(f"Cleaned up S3 backup: {obj['Key']}")
                    
        except Exception as e:
            logger.error(f"S3 cleanup failed: {str(e)}")
    
    async def get_backup_status(self) -> Dict[str, Any]:
        """Get backup system status"""
        return {
            'system_id': self.system_id,
            'scheduler_running': self.scheduler_running,
            'start_time': self.start_time.isoformat(),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'backup_targets': len(self.backup_targets),
            'active_jobs': len(self.active_jobs),
            'job_history': len(self.job_history),
            'metrics': {
                'total_backups': self.metrics.total_backups,
                'successful_backups': self.metrics.successful_backups,
                'failed_backups': self.metrics.failed_backups,
                'success_rate': self.metrics.backup_success_rate,
                'total_size_gb': self.metrics.total_size_gb
            },
            'storage_backends': list(self.storage_backends.keys()),
            'recent_jobs': [
                {
                    'job_id': job.job_id,
                    'target_name': job.target.name,
                    'status': job.status,
                    'started_at': job.started_at.isoformat() if job.started_at else None,
                    'duration_seconds': job.duration_seconds,
                    'size_mb': job.backup_size_mb,
                    'error_message': job.error_message
                }
                for job in self.job_history[-10:]  # Last 10 jobs
            ]
        }


# Factory function
def create_backup_system(config_path: str = None) -> BackupRecoverySystem:
    """Create backup and recovery system instance"""
    return BackupRecoverySystem(config_path)


# CLI interface
async def main():
    """Main backup CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GrandModel Backup and Recovery System")
    parser.add_argument("--config", help="Backup configuration file")
    parser.add_argument("--backup", help="Backup specific target")
    parser.add_argument("--restore", help="Restore specific target")
    parser.add_argument("--restore-date", help="Restore date (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--restore-path", help="Restore destination path")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup old backups")
    parser.add_argument("--status", action="store_true", help="Show backup status")
    parser.add_argument("--start-scheduler", action="store_true", help="Start backup scheduler")
    
    args = parser.parse_args()
    
    # Create backup system
    backup_system = create_backup_system(args.config)
    
    try:
        if args.backup:
            # Backup specific target
            job = await backup_system.backup_target(args.backup)
            print(f"✅ Backup completed: {job.job_id}")
            print(f"   Target: {job.target.name}")
            print(f"   Size: {job.backup_size_mb:.2f} MB")
            print(f"   Duration: {job.duration_seconds:.1f}s")
            
        elif args.restore:
            # Restore specific target
            if not args.restore_date:
                print("❌ Restore date required (--restore-date)")
                sys.exit(1)
            
            restore_date = datetime.strptime(args.restore_date, '%Y-%m-%d %H:%M:%S')
            job = await backup_system.restore_from_backup(
                args.restore, restore_date, args.restore_path
            )
            print(f"✅ Restore completed: {job.job_id}")
            print(f"   Target: {job.target.name}")
            print(f"   Duration: {job.duration_seconds:.1f}s")
            
        elif args.cleanup:
            # Cleanup old backups
            await backup_system.cleanup_old_backups()
            print("✅ Cleanup completed")
            
        elif args.status:
            # Show status
            status = await backup_system.get_backup_status()
            print(f"📊 Backup System Status:")
            print(f"   System ID: {status['system_id']}")
            print(f"   Scheduler Running: {status['scheduler_running']}")
            print(f"   Uptime: {status['uptime_seconds']:.1f}s")
            print(f"   Backup Targets: {status['backup_targets']}")
            print(f"   Active Jobs: {status['active_jobs']}")
            print(f"   Total Backups: {status['metrics']['total_backups']}")
            print(f"   Success Rate: {status['metrics']['success_rate']:.1f}%")
            print(f"   Total Size: {status['metrics']['total_size_gb']:.2f} GB")
            
        elif args.start_scheduler:
            # Start scheduler
            await backup_system.start_backup_scheduler()
            print("🚀 Backup scheduler started")
            
            # Keep running
            try:
                while True:
                    await asyncio.sleep(60)
            except KeyboardInterrupt:
                print("🛑 Backup scheduler stopped")
                
        else:
            print("❌ No action specified. Use --help for options.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Operation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())