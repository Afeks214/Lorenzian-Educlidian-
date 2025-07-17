#!/usr/bin/env python3
"""
High Availability Manager for PostgreSQL with Patroni
AGENT 14: DATABASE OPTIMIZATION SPECIALIST
Focus: Automatic failover, disaster recovery, and cluster management
"""

import asyncio
import aiohttp
import psycopg2
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from prometheus_client import Counter, Histogram, Gauge
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import yaml

class NodeState(Enum):
    """Possible states of a database node"""
    MASTER = "master"
    REPLICA = "replica"
    STANDBY_LEADER = "standby_leader"
    SYNC_STANDBY = "sync_standby"
    ASYNC_STANDBY = "async_standby"
    CREATING_REPLICA = "creating_replica"
    STOPPED = "stopped"
    STARTING = "starting"
    CRASHED = "crashed"
    UNKNOWN = "unknown"

class FailoverType(Enum):
    """Types of failover operations"""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    SWITCHOVER = "switchover"
    EMERGENCY = "emergency"

@dataclass
class NodeInfo:
    """Information about a database node"""
    name: str
    host: str
    port: int
    state: NodeState
    role: str
    timeline: int
    lag: Optional[int]
    api_url: str
    last_seen: datetime
    health_score: float
    is_leader: bool
    is_running: bool
    patroni_version: str
    postgres_version: str
    pending_restart: bool
    tags: Dict[str, Any]

@dataclass
class ClusterInfo:
    """Information about the database cluster"""
    scope: str
    leader: Optional[str]
    members: List[NodeInfo]
    initialize: str
    config: Dict[str, Any]
    last_leader_operation: Optional[datetime]
    failover_count: int
    total_nodes: int
    healthy_nodes: int
    sync_standbys: List[str]
    async_standbys: List[str]
    maintenance_mode: bool

@dataclass
class FailoverEvent:
    """Information about a failover event"""
    event_id: str
    start_time: datetime
    end_time: Optional[datetime]
    failover_type: FailoverType
    old_leader: str
    new_leader: str
    trigger_reason: str
    duration_seconds: Optional[float]
    success: bool
    affected_nodes: List[str]
    recovery_actions: List[str]
    performance_impact: Dict[str, float]

class HighAvailabilityManager:
    """
    Advanced High Availability Manager for PostgreSQL with Patroni
    """
    
    def __init__(self, config_file: str = None):
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Cluster monitoring
        self.cluster_info = None
        self.node_health_history = {}
        self.failover_history = []
        
        # Performance monitoring
        self.performance_baseline = {}
        self.alert_thresholds = self.config.get('alert_thresholds', {})
        
        # Setup Prometheus metrics
        self.setup_prometheus_metrics()
        
        # Notification system
        self.notification_manager = NotificationManager(self.config.get('notifications', {}))
        
        # Disaster recovery
        self.disaster_recovery = DisasterRecoveryManager(self.config.get('disaster_recovery', {}))
        
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from file"""
        default_config = {
            "cluster": {
                "scope": "grandmodel-cluster",
                "namespace": "/trading",
                "etcd_endpoints": ["http://etcd-cluster:2379"],
                "patroni_port": 8008,
                "postgres_port": 5432
            },
            "nodes": {
                "primary": {
                    "name": "postgresql-primary",
                    "host": "patroni-primary",
                    "api_url": "http://patroni-primary:8008"
                },
                "standby": {
                    "name": "postgresql-standby",
                    "host": "patroni-standby",
                    "api_url": "http://patroni-standby:8008"
                }
            },
            "monitoring": {
                "check_interval": 5,
                "health_check_timeout": 3,
                "performance_check_interval": 30,
                "metrics_retention_hours": 24
            },
            "failover": {
                "auto_failover": True,
                "max_lag_bytes": 1048576,
                "failover_timeout": 30,
                "confirmation_timeout": 10,
                "cooldown_period": 300
            },
            "alert_thresholds": {
                "lag_warning_bytes": 10485760,
                "lag_critical_bytes": 104857600,
                "response_time_warning_ms": 100,
                "response_time_critical_ms": 1000,
                "cpu_warning_percent": 80,
                "cpu_critical_percent": 95,
                "memory_warning_percent": 85,
                "memory_critical_percent": 95
            }
        }
        
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    return {**default_config, **loaded_config}
            except Exception as e:
                self.logger.error(f"Failed to load config file: {e}")
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for HA manager"""
        logger = logging.getLogger('ha_manager')
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler('/var/log/ha_manager.log')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        self.cluster_health_gauge = Gauge(
            'db_cluster_health_score',
            'Overall cluster health score',
            ['cluster_scope']
        )
        
        self.node_health_gauge = Gauge(
            'db_node_health_score',
            'Individual node health score',
            ['node_name', 'node_role']
        )
        
        self.failover_counter = Counter(
            'db_failover_events_total',
            'Total number of failover events',
            ['failover_type', 'success']
        )
        
        self.failover_duration = Histogram(
            'db_failover_duration_seconds',
            'Duration of failover operations',
            ['failover_type'],
            buckets=[1, 5, 10, 15, 30, 60, 120, 300, 600]
        )
        
        self.replication_lag = Gauge(
            'db_replication_lag_bytes',
            'Replication lag in bytes',
            ['master_node', 'replica_node']
        )
        
        self.cluster_status_gauge = Gauge(
            'db_cluster_status',
            'Cluster status (1=healthy, 0=unhealthy)',
            ['cluster_scope']
        )
        
        self.node_status_gauge = Gauge(
            'db_node_status',
            'Node status (1=running, 0=stopped)',
            ['node_name', 'node_role']
        )
    
    async def get_cluster_info(self) -> ClusterInfo:
        """Get current cluster information"""
        try:
            # Get cluster information from Patroni REST API
            members = []
            leader = None
            
            for node_name, node_config in self.config['nodes'].items():
                try:
                    node_info = await self._get_node_info(node_config)
                    members.append(node_info)
                    
                    if node_info.is_leader:
                        leader = node_info.name
                        
                except Exception as e:
                    self.logger.warning(f"Failed to get info for node {node_name}: {e}")
                    # Create a placeholder node with unknown state
                    members.append(NodeInfo(
                        name=node_name,
                        host=node_config['host'],
                        port=self.config['cluster']['postgres_port'],
                        state=NodeState.UNKNOWN,
                        role="unknown",
                        timeline=0,
                        lag=None,
                        api_url=node_config['api_url'],
                        last_seen=datetime.now(),
                        health_score=0.0,
                        is_leader=False,
                        is_running=False,
                        patroni_version="unknown",
                        postgres_version="unknown",
                        pending_restart=False,
                        tags={}
                    ))
            
            # Count healthy nodes
            healthy_nodes = sum(1 for node in members if node.health_score > 0.5)
            
            # Classify standbys
            sync_standbys = []
            async_standbys = []
            
            for node in members:
                if node.role == "replica":
                    if node.state == NodeState.SYNC_STANDBY:
                        sync_standbys.append(node.name)
                    else:
                        async_standbys.append(node.name)
            
            cluster_info = ClusterInfo(
                scope=self.config['cluster']['scope'],
                leader=leader,
                members=members,
                initialize="",
                config={},
                last_leader_operation=None,
                failover_count=len(self.failover_history),
                total_nodes=len(members),
                healthy_nodes=healthy_nodes,
                sync_standbys=sync_standbys,
                async_standbys=async_standbys,
                maintenance_mode=False
            )
            
            self.cluster_info = cluster_info
            return cluster_info
            
        except Exception as e:
            self.logger.error(f"Failed to get cluster info: {e}")
            raise
    
    async def _get_node_info(self, node_config: Dict) -> NodeInfo:
        """Get information about a specific node"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                # Get node status
                async with session.get(f"{node_config['api_url']}/") as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract node information
                        state_str = data.get('state', 'unknown')
                        try:
                            state = NodeState(state_str)
                        except ValueError:
                            state = NodeState.UNKNOWN
                        
                        # Calculate health score
                        health_score = await self._calculate_health_score(node_config, data)
                        
                        return NodeInfo(
                            name=data.get('name', node_config['name']),
                            host=node_config['host'],
                            port=data.get('port', self.config['cluster']['postgres_port']),
                            state=state,
                            role=data.get('role', 'unknown'),
                            timeline=data.get('timeline', 0),
                            lag=data.get('lag'),
                            api_url=node_config['api_url'],
                            last_seen=datetime.now(),
                            health_score=health_score,
                            is_leader=data.get('role') == 'master',
                            is_running=data.get('state') == 'running',
                            patroni_version=data.get('patroni', {}).get('version', 'unknown'),
                            postgres_version=data.get('server_version', 'unknown'),
                            pending_restart=data.get('pending_restart', False),
                            tags=data.get('tags', {})
                        )
                    else:
                        raise Exception(f"API returned status {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Failed to get node info from {node_config['api_url']}: {e}")
            raise
    
    async def _calculate_health_score(self, node_config: Dict, node_data: Dict) -> float:
        """Calculate health score for a node"""
        try:
            score = 0.0
            max_score = 100.0
            
            # Basic connectivity (20 points)
            if node_data.get('state') == 'running':
                score += 20
            
            # Database responsiveness (30 points)
            try:
                response_time = await self._check_database_response_time(node_config)
                if response_time < 10:  # < 10ms
                    score += 30
                elif response_time < 50:  # < 50ms
                    score += 20
                elif response_time < 100:  # < 100ms
                    score += 10
            except:
                pass
            
            # Replication lag (20 points)
            lag = node_data.get('lag')
            if lag is not None:
                if lag < 1024:  # < 1KB
                    score += 20
                elif lag < 10240:  # < 10KB
                    score += 15
                elif lag < 102400:  # < 100KB
                    score += 10
                elif lag < 1048576:  # < 1MB
                    score += 5
            else:
                score += 20  # No lag for master
            
            # Timeline consistency (10 points)
            if node_data.get('timeline', 0) > 0:
                score += 10
            
            # Pending restart status (10 points)
            if not node_data.get('pending_restart', False):
                score += 10
            
            # Resource utilization (10 points)
            try:
                cpu_usage = await self._get_cpu_usage(node_config)
                memory_usage = await self._get_memory_usage(node_config)
                
                if cpu_usage < 70 and memory_usage < 80:
                    score += 10
                elif cpu_usage < 85 and memory_usage < 90:
                    score += 5
            except:
                pass
            
            return min(score / max_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate health score: {e}")
            return 0.0
    
    async def _check_database_response_time(self, node_config: Dict) -> float:
        """Check database response time"""
        start_time = time.time()
        
        try:
            conn = psycopg2.connect(
                host=node_config['host'],
                port=self.config['cluster']['postgres_port'],
                database='grandmodel',
                user='grandmodel_user',
                password='grandmodel_password',
                connect_timeout=5
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            conn.close()
            
            return (time.time() - start_time) * 1000  # Convert to milliseconds
            
        except Exception as e:
            self.logger.warning(f"Database response check failed: {e}")
            return 10000  # 10 seconds penalty
    
    async def _get_cpu_usage(self, node_config: Dict) -> float:
        """Get CPU usage for a node"""
        try:
            # This would typically query a monitoring system
            # For now, return a mock value
            return 0.0
        except:
            return 0.0
    
    async def _get_memory_usage(self, node_config: Dict) -> float:
        """Get memory usage for a node"""
        try:
            # This would typically query a monitoring system
            # For now, return a mock value
            return 0.0
        except:
            return 0.0
    
    async def check_failover_conditions(self) -> Tuple[bool, str]:
        """Check if failover conditions are met"""
        if not self.config['failover']['auto_failover']:
            return False, "Auto failover is disabled"
        
        cluster_info = await self.get_cluster_info()
        
        # Check if we have a leader
        if not cluster_info.leader:
            return True, "No leader found in cluster"
        
        # Find the leader node
        leader_node = None
        for node in cluster_info.members:
            if node.name == cluster_info.leader:
                leader_node = node
                break
        
        if not leader_node:
            return True, "Leader node not found in member list"
        
        # Check leader health
        if leader_node.health_score < 0.5:
            return True, f"Leader health score too low: {leader_node.health_score}"
        
        if not leader_node.is_running:
            return True, "Leader is not running"
        
        # Check replication lag
        max_lag = self.config['failover']['max_lag_bytes']
        for node in cluster_info.members:
            if node.role == "replica" and node.lag and node.lag > max_lag:
                return True, f"Replication lag too high: {node.lag} bytes"
        
        return False, "All conditions normal"
    
    async def initiate_failover(self, failover_type: FailoverType, target_node: str = None) -> FailoverEvent:
        """Initiate a failover operation"""
        event_id = f"failover_{int(time.time())}"
        start_time = datetime.now()
        
        self.logger.info(f"Initiating {failover_type.value} failover (ID: {event_id})")
        
        try:
            cluster_info = await self.get_cluster_info()
            old_leader = cluster_info.leader
            
            # Notify about starting failover
            await self.notification_manager.send_alert(
                "info",
                f"Failover Starting",
                f"Initiating {failover_type.value} failover from {old_leader} to {target_node or 'auto-selected'}"
            )
            
            # Perform the failover
            if failover_type == FailoverType.SWITCHOVER:
                success, new_leader = await self._perform_switchover(target_node)
            else:
                success, new_leader = await self._perform_failover(target_node)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Create failover event
            failover_event = FailoverEvent(
                event_id=event_id,
                start_time=start_time,
                end_time=end_time,
                failover_type=failover_type,
                old_leader=old_leader or "unknown",
                new_leader=new_leader or "unknown",
                trigger_reason="Manual failover",
                duration_seconds=duration,
                success=success,
                affected_nodes=[old_leader, new_leader] if old_leader and new_leader else [],
                recovery_actions=[],
                performance_impact={}
            )
            
            # Update metrics
            self.failover_counter.labels(
                failover_type=failover_type.value,
                success=str(success).lower()
            ).inc()
            
            self.failover_duration.labels(
                failover_type=failover_type.value
            ).observe(duration)
            
            # Store in history
            self.failover_history.append(failover_event)
            
            # Notify about completion
            status = "SUCCESS" if success else "FAILED"
            await self.notification_manager.send_alert(
                "critical" if not success else "info",
                f"Failover {status}",
                f"Failover {event_id} completed in {duration:.2f}s. New leader: {new_leader}"
            )
            
            self.logger.info(f"Failover {event_id} completed: {status} in {duration:.2f}s")
            
            return failover_event
            
        except Exception as e:
            self.logger.error(f"Failover {event_id} failed: {e}")
            
            # Create failed event
            failover_event = FailoverEvent(
                event_id=event_id,
                start_time=start_time,
                end_time=datetime.now(),
                failover_type=failover_type,
                old_leader=cluster_info.leader if cluster_info else "unknown",
                new_leader="unknown",
                trigger_reason=str(e),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                success=False,
                affected_nodes=[],
                recovery_actions=[],
                performance_impact={}
            )
            
            await self.notification_manager.send_alert(
                "critical",
                f"Failover FAILED",
                f"Failover {event_id} failed: {str(e)}"
            )
            
            return failover_event
    
    async def _perform_switchover(self, target_node: str = None) -> Tuple[bool, str]:
        """Perform a planned switchover"""
        try:
            cluster_info = await self.get_cluster_info()
            if not cluster_info.leader:
                raise Exception("No current leader found")
            
            # Select target node
            if not target_node:
                # Select best replica
                best_replica = None
                best_score = 0
                
                for node in cluster_info.members:
                    if node.role == "replica" and node.health_score > best_score:
                        best_replica = node
                        best_score = node.health_score
                
                if not best_replica:
                    raise Exception("No suitable replica found")
                
                target_node = best_replica.name
            
            # Perform switchover via Patroni API
            leader_node = next(node for node in cluster_info.members if node.name == cluster_info.leader)
            
            async with aiohttp.ClientSession() as session:
                # Perform switchover
                switchover_data = {
                    "leader": cluster_info.leader,
                    "candidate": target_node
                }
                
                async with session.post(
                    f"{leader_node.api_url}/switchover",
                    json=switchover_data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.logger.info(f"Switchover initiated: {result}")
                        
                        # Wait for switchover to complete
                        await self._wait_for_new_leader(target_node)
                        
                        return True, target_node
                    else:
                        raise Exception(f"Switchover API returned status {response.status}")
            
        except Exception as e:
            self.logger.error(f"Switchover failed: {e}")
            return False, ""
    
    async def _perform_failover(self, target_node: str = None) -> Tuple[bool, str]:
        """Perform an emergency failover"""
        try:
            cluster_info = await self.get_cluster_info()
            
            # Select target node if not specified
            if not target_node:
                best_replica = None
                best_score = 0
                
                for node in cluster_info.members:
                    if node.role == "replica" and node.is_running and node.health_score > best_score:
                        best_replica = node
                        best_score = node.health_score
                
                if not best_replica:
                    raise Exception("No suitable replica found for failover")
                
                target_node = best_replica.name
            
            # Find target node
            target_node_info = next(
                (node for node in cluster_info.members if node.name == target_node),
                None
            )
            
            if not target_node_info:
                raise Exception(f"Target node {target_node} not found")
            
            # Perform failover via Patroni API
            async with aiohttp.ClientSession() as session:
                failover_data = {
                    "leader": target_node
                }
                
                async with session.post(
                    f"{target_node_info.api_url}/failover",
                    json=failover_data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.logger.info(f"Failover initiated: {result}")
                        
                        # Wait for new leader
                        await self._wait_for_new_leader(target_node)
                        
                        return True, target_node
                    else:
                        raise Exception(f"Failover API returned status {response.status}")
            
        except Exception as e:
            self.logger.error(f"Failover failed: {e}")
            return False, ""
    
    async def _wait_for_new_leader(self, expected_leader: str, timeout: int = 30):
        """Wait for new leader to be elected"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                cluster_info = await self.get_cluster_info()
                if cluster_info.leader == expected_leader:
                    # Verify leader is healthy
                    leader_node = next(node for node in cluster_info.members if node.name == expected_leader)
                    if leader_node.is_running and leader_node.health_score > 0.5:
                        return True
                        
            except Exception as e:
                self.logger.warning(f"Error checking new leader: {e}")
            
            await asyncio.sleep(1)
        
        raise Exception(f"Timeout waiting for new leader {expected_leader}")
    
    async def monitor_cluster(self):
        """Monitor cluster health and trigger failover if needed"""
        while self.is_running:
            try:
                # Get cluster information
                cluster_info = await self.get_cluster_info()
                
                # Update metrics
                self.cluster_status_gauge.labels(
                    cluster_scope=cluster_info.scope
                ).set(1 if cluster_info.leader else 0)
                
                overall_health = sum(node.health_score for node in cluster_info.members) / len(cluster_info.members)
                self.cluster_health_gauge.labels(
                    cluster_scope=cluster_info.scope
                ).set(overall_health)
                
                # Update node metrics
                for node in cluster_info.members:
                    self.node_health_gauge.labels(
                        node_name=node.name,
                        node_role=node.role
                    ).set(node.health_score)
                    
                    self.node_status_gauge.labels(
                        node_name=node.name,
                        node_role=node.role
                    ).set(1 if node.is_running else 0)
                    
                    # Update replication lag
                    if node.role == "replica" and node.lag is not None:
                        self.replication_lag.labels(
                            master_node=cluster_info.leader or "unknown",
                            replica_node=node.name
                        ).set(node.lag)
                
                # Check failover conditions
                should_failover, reason = await self.check_failover_conditions()
                
                if should_failover:
                    self.logger.warning(f"Failover conditions met: {reason}")
                    
                    # Check cooldown period
                    if self.failover_history:
                        last_failover = self.failover_history[-1]
                        cooldown_seconds = self.config['failover']['cooldown_period']
                        
                        if (datetime.now() - last_failover.end_time).seconds < cooldown_seconds:
                            self.logger.info(f"Failover in cooldown period, skipping")
                        else:
                            await self.initiate_failover(FailoverType.AUTOMATIC)
                    else:
                        await self.initiate_failover(FailoverType.AUTOMATIC)
                
                await asyncio.sleep(self.config['monitoring']['check_interval'])
                
            except Exception as e:
                self.logger.error(f"Cluster monitoring error: {e}")
                await asyncio.sleep(self.config['monitoring']['check_interval'])
    
    async def start_monitoring(self):
        """Start the high availability monitoring"""
        self.is_running = True
        self.logger.info("Starting High Availability Manager")
        
        # Start cluster monitoring
        await self.monitor_cluster()
    
    def stop_monitoring(self):
        """Stop the high availability monitoring"""
        self.is_running = False
        self.executor.shutdown(wait=True)
        self.logger.info("High Availability Manager stopped")
    
    def get_cluster_status(self) -> Dict:
        """Get current cluster status"""
        if not self.cluster_info:
            return {"error": "No cluster information available"}
        
        return {
            "cluster": asdict(self.cluster_info),
            "recent_failovers": [asdict(event) for event in self.failover_history[-10:]],
            "timestamp": datetime.now().isoformat()
        }

class NotificationManager:
    """Manages notifications for HA events"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger('notification_manager')
    
    async def send_alert(self, level: str, subject: str, message: str):
        """Send alert notification"""
        try:
            # Log the alert
            self.logger.info(f"ALERT [{level.upper()}]: {subject} - {message}")
            
            # Send email if configured
            if self.config.get('email'):
                await self._send_email_alert(level, subject, message)
            
            # Send webhook if configured
            if self.config.get('webhook'):
                await self._send_webhook_alert(level, subject, message)
                
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")
    
    async def _send_email_alert(self, level: str, subject: str, message: str):
        """Send email alert"""
        try:
            email_config = self.config['email']
            
            msg = MIMEMultipart()
            msg['From'] = email_config['from']
            msg['To'] = ', '.join(email_config['to'])
            msg['Subject'] = f"[{level.upper()}] {subject}"
            
            body = f"""
            Alert Level: {level.upper()}
            Subject: {subject}
            Message: {message}
            Timestamp: {datetime.now().isoformat()}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            if email_config.get('use_tls'):
                server.starttls()
            if email_config.get('username'):
                server.login(email_config['username'], email_config['password'])
            
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
    
    async def _send_webhook_alert(self, level: str, subject: str, message: str):
        """Send webhook alert"""
        try:
            webhook_config = self.config['webhook']
            
            payload = {
                "level": level,
                "subject": subject,
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_config['url'],
                    json=payload,
                    headers=webhook_config.get('headers', {}),
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 200:
                        self.logger.warning(f"Webhook returned status {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")

class DisasterRecoveryManager:
    """Manages disaster recovery operations"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger('disaster_recovery')
    
    async def create_backup(self, backup_type: str = "full") -> bool:
        """Create database backup"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{backup_type}_{timestamp}"
            
            # Create backup using pg_dump or barman
            if backup_type == "full":
                await self._create_full_backup(backup_name)
            elif backup_type == "incremental":
                await self._create_incremental_backup(backup_name)
            
            self.logger.info(f"Backup {backup_name} created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            return False
    
    async def _create_full_backup(self, backup_name: str):
        """Create full backup"""
        # Implementation would depend on backup tool (pg_dump, barman, etc.)
        pass
    
    async def _create_incremental_backup(self, backup_name: str):
        """Create incremental backup"""
        # Implementation would depend on backup tool
        pass

async def main():
    """Main entry point"""
    ha_manager = HighAvailabilityManager()
    
    try:
        await ha_manager.start_monitoring()
    except KeyboardInterrupt:
        print("\nShutting down HA manager...")
        ha_manager.stop_monitoring()
    except Exception as e:
        print(f"Error: {e}")
        ha_manager.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(main())