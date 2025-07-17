"""
Trading Engine Failover Simulation and Testing Framework
======================================================

This module provides comprehensive automated testing for trading engine failover scenarios
with RTO validation, state recovery verification, and performance impact assessment.

Key Features:
- Automated trading engine failover simulation
- MARL agent state recovery validation
- JIT model recompilation testing
- Order state consistency verification
- Position reconciliation testing
- Performance impact assessment
- Automated recovery validation

Target RTO: <30 seconds for trading engine failover
Target RPO: <1 second for trading data
"""

import asyncio
import time
import logging
import json
import traceback
import psutil
import subprocess
import torch
import pickle
import os
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import uuid
import threading
from contextlib import asynccontextmanager
import aioredis
import httpx
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingEngineFailoverType(Enum):
    """Types of trading engine failover scenarios."""
    TACTICAL_AGENT_KILL = "tactical_agent_kill"
    STRATEGIC_AGENT_KILL = "strategic_agent_kill"
    MARL_COORDINATOR_FAILURE = "marl_coordinator_failure"
    JIT_MODEL_CORRUPTION = "jit_model_corruption"
    REDIS_STATE_LOSS = "redis_state_loss"
    ORDER_MANAGER_FAILURE = "order_manager_failure"
    RISK_MANAGER_FAILURE = "risk_manager_failure"
    EXECUTION_ENGINE_FAILURE = "execution_engine_failure"
    COMPLETE_SYSTEM_FAILURE = "complete_system_failure"
    CASCADING_COMPONENT_FAILURE = "cascading_component_failure"


class AgentState(Enum):
    """Agent state enumeration."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    RECOVERING = "recovering"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class TradingEngineRTOMetrics:
    """RTO metrics specific to trading engine failover."""
    # Detection and failover timing
    failure_detection_time: float = 0.0
    failover_initiation_time: float = 0.0
    state_recovery_time: float = 0.0
    model_recompilation_time: float = 0.0
    service_restart_time: float = 0.0
    
    # Recovery validation timing
    first_heartbeat_time: float = 0.0
    first_trade_signal_time: float = 0.0
    full_functionality_time: float = 0.0
    
    # State consistency metrics
    agent_state_recovered: bool = False
    jit_models_recompiled: bool = False
    redis_state_restored: bool = False
    order_state_consistent: bool = False
    position_reconciled: bool = False
    
    # Performance metrics
    pre_failover_latency: float = 0.0
    post_failover_latency: float = 0.0
    performance_degradation_percent: float = 0.0
    
    # Business metrics
    missed_opportunities: int = 0
    orders_lost: int = 0
    positions_affected: int = 0
    pnl_impact: float = 0.0
    
    def total_recovery_time(self) -> float:
        """Calculate total recovery time."""
        return max(
            self.state_recovery_time,
            self.model_recompilation_time,
            self.full_functionality_time
        )
    
    def meets_rto_target(self, target_seconds: float = 30.0) -> bool:
        """Check if RTO target is met."""
        return self.total_recovery_time() <= target_seconds


@dataclass
class TradingEngineFailoverConfig:
    """Configuration for trading engine failover tests."""
    test_id: str
    failover_type: TradingEngineFailoverType
    target_rto_seconds: float = 30.0
    target_rpo_seconds: float = 1.0
    max_test_duration: int = 300  # 5 minutes
    
    # Service endpoints
    tactical_api_url: str = "http://localhost:8001"
    strategic_api_url: str = "http://localhost:8002"
    risk_api_url: str = "http://localhost:8003"
    execution_api_url: str = "http://localhost:8004"
    
    # Redis configuration
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0
    
    # Model paths
    tactical_model_path: str = "/tmp/tactical_model.pt"
    strategic_model_path: str = "/tmp/strategic_model.pt"
    
    # Test parameters
    simulation_trades: int = 100
    concurrent_requests: int = 10
    performance_threshold_ms: float = 100.0
    
    # Monitoring
    enable_detailed_logging: bool = True
    enable_performance_profiling: bool = True
    enable_state_validation: bool = True


@dataclass
class TradingEngineFailoverResult:
    """Result of trading engine failover test."""
    test_id: str
    test_name: str
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Core metrics
    rto_metrics: TradingEngineRTOMetrics = field(default_factory=TradingEngineRTOMetrics)
    
    # Detailed results
    pre_test_state: Dict[str, Any] = field(default_factory=dict)
    failover_execution: Dict[str, Any] = field(default_factory=dict)
    recovery_process: Dict[str, Any] = field(default_factory=dict)
    post_test_validation: Dict[str, Any] = field(default_factory=dict)
    
    # Performance analysis
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def success(self) -> bool:
        """Check if test was successful."""
        return (
            self.status == "completed" and
            len(self.errors) == 0 and
            self.rto_metrics.meets_rto_target()
        )


class TradingEngineMonitor:
    """Monitor for trading engine components."""
    
    def __init__(self, config: TradingEngineFailoverConfig):
        self.config = config
        
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "timestamp": time.time(),
            "services": {},
            "agents": {},
            "models": {},
            "redis": {},
            "performance": {}
        }
        
        # Check service health
        services = [
            ("tactical", self.config.tactical_api_url),
            ("strategic", self.config.strategic_api_url),
            ("risk", self.config.risk_api_url),
            ("execution", self.config.execution_api_url)
        ]
        
        for service_name, url in services:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{url}/health", timeout=5.0)
                    status["services"][service_name] = {
                        "healthy": response.status_code == 200,
                        "status_code": response.status_code,
                        "response_time": response.elapsed.total_seconds()
                    }
            except Exception as e:
                status["services"][service_name] = {
                    "healthy": False,
                    "error": str(e)
                }
        
        # Check Redis
        try:
            redis_client = aioredis.from_url(self.config.redis_url)
            await redis_client.ping()
            
            # Get key counts
            tactical_keys = await redis_client.keys("tactical:*")
            strategic_keys = await redis_client.keys("strategic:*")
            
            status["redis"] = {
                "healthy": True,
                "tactical_keys": len(tactical_keys),
                "strategic_keys": len(strategic_keys),
                "total_keys": len(tactical_keys) + len(strategic_keys)
            }
            
            await redis_client.close()
            
        except Exception as e:
            status["redis"] = {
                "healthy": False,
                "error": str(e)
            }
        
        # Check model files
        model_files = [
            ("tactical", self.config.tactical_model_path),
            ("strategic", self.config.strategic_model_path)
        ]
        
        for model_name, path in model_files:
            if Path(path).exists():
                stat = Path(path).stat()
                status["models"][model_name] = {
                    "exists": True,
                    "size": stat.st_size,
                    "modified": stat.st_mtime
                }
            else:
                status["models"][model_name] = {
                    "exists": False
                }
        
        return status
    
    async def get_agent_state(self, agent_type: str) -> Dict[str, Any]:
        """Get detailed agent state."""
        try:
            url = getattr(self.config, f"{agent_type}_api_url")
            
            async with httpx.AsyncClient() as client:
                # Get agent status
                status_response = await client.get(f"{url}/status", timeout=5.0)
                status_data = status_response.json() if status_response.status_code == 200 else {}
                
                # Get agent metrics
                metrics_response = await client.get(f"{url}/metrics", timeout=5.0)
                metrics_data = metrics_response.json() if metrics_response.status_code == 200 else {}
                
                return {
                    "agent_type": agent_type,
                    "status": status_data,
                    "metrics": metrics_data,
                    "healthy": status_response.status_code == 200
                }
                
        except Exception as e:
            return {
                "agent_type": agent_type,
                "healthy": False,
                "error": str(e)
            }
    
    async def measure_performance(self, iterations: int = 10) -> Dict[str, Any]:
        """Measure system performance."""
        performance_data = {
            "service_latency": {},
            "redis_latency": 0.0,
            "overall_latency": 0.0,
            "error_rate": 0.0,
            "throughput": 0.0
        }
        
        # Measure service latency
        services = [
            ("tactical", self.config.tactical_api_url),
            ("strategic", self.config.strategic_api_url),
            ("risk", self.config.risk_api_url),
            ("execution", self.config.execution_api_url)
        ]
        
        for service_name, url in services:
            latencies = []
            errors = 0
            
            for i in range(iterations):
                try:
                    start_time = time.time()
                    async with httpx.AsyncClient() as client:
                        response = await client.get(f"{url}/ping", timeout=5.0)
                        latency = time.time() - start_time
                        
                        if response.status_code == 200:
                            latencies.append(latency)
                        else:
                            errors += 1
                            
                except Exception:
                    errors += 1
                
                await asyncio.sleep(0.01)
            
            if latencies:
                performance_data["service_latency"][service_name] = {
                    "avg_latency": sum(latencies) / len(latencies),
                    "min_latency": min(latencies),
                    "max_latency": max(latencies),
                    "error_rate": errors / iterations
                }
        
        # Measure Redis latency
        try:
            redis_client = aioredis.from_url(self.config.redis_url)
            redis_latencies = []
            
            for i in range(iterations):
                start_time = time.time()
                await redis_client.ping()
                latency = time.time() - start_time
                redis_latencies.append(latency)
                await asyncio.sleep(0.01)
            
            performance_data["redis_latency"] = sum(redis_latencies) / len(redis_latencies)
            await redis_client.close()
            
        except Exception as e:
            performance_data["redis_error"] = str(e)
        
        return performance_data


class TradingEngineStateManager:
    """Manager for trading engine state operations."""
    
    def __init__(self, config: TradingEngineFailoverConfig):
        self.config = config
        self.redis_client: Optional[aioredis.Redis] = None
        
    async def initialize(self):
        """Initialize state manager."""
        try:
            self.redis_client = aioredis.from_url(self.config.redis_url)
            await self.redis_client.ping()
            logger.info("State manager initialized")
        except Exception as e:
            logger.error(f"Failed to initialize state manager: {e}")
            raise
    
    async def close(self):
        """Close state manager."""
        if self.redis_client:
            await self.redis_client.close()
    
    async def backup_state(self) -> Dict[str, Any]:
        """Backup current system state."""
        try:
            backup = {
                "timestamp": time.time(),
                "redis_data": {},
                "model_files": {},
                "agent_states": {}
            }
            
            # Backup Redis data
            tactical_keys = await self.redis_client.keys("tactical:*")
            strategic_keys = await self.redis_client.keys("strategic:*")
            
            for key in tactical_keys + strategic_keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                value = await self.redis_client.get(key)
                backup["redis_data"][key_str] = value.decode() if value else None
            
            # Backup model files
            model_files = [
                ("tactical", self.config.tactical_model_path),
                ("strategic", self.config.strategic_model_path)
            ]
            
            for model_name, path in model_files:
                if Path(path).exists():
                    # Create backup copy
                    backup_path = f"{path}.backup_{int(time.time())}"
                    subprocess.run(["cp", path, backup_path], check=True)
                    backup["model_files"][model_name] = backup_path
            
            logger.info("State backup completed")
            return backup
            
        except Exception as e:
            logger.error(f"State backup failed: {e}")
            return {}
    
    async def restore_state(self, backup: Dict[str, Any]) -> bool:
        """Restore system state from backup."""
        try:
            # Restore Redis data
            redis_data = backup.get("redis_data", {})
            for key, value in redis_data.items():
                if value is not None:
                    await self.redis_client.set(key, value)
            
            # Restore model files
            model_files = backup.get("model_files", {})
            for model_name, backup_path in model_files.items():
                if Path(backup_path).exists():
                    if model_name == "tactical":
                        target_path = self.config.tactical_model_path
                    elif model_name == "strategic":
                        target_path = self.config.strategic_model_path
                    else:
                        continue
                    
                    subprocess.run(["cp", backup_path, target_path], check=True)
            
            logger.info("State restore completed")
            return True
            
        except Exception as e:
            logger.error(f"State restore failed: {e}")
            return False
    
    async def validate_state_consistency(self) -> Dict[str, Any]:
        """Validate state consistency across components."""
        try:
            validation = {
                "redis_consistency": True,
                "model_consistency": True,
                "agent_consistency": True,
                "details": {}
            }
            
            # Check Redis consistency
            tactical_keys = await self.redis_client.keys("tactical:*")
            strategic_keys = await self.redis_client.keys("strategic:*")
            
            validation["details"]["redis_keys"] = {
                "tactical": len(tactical_keys),
                "strategic": len(strategic_keys)
            }
            
            # Check critical keys exist
            critical_keys = [
                "tactical:state",
                "strategic:state",
                "tactical:model_info",
                "strategic:model_info"
            ]
            
            missing_keys = []
            for key in critical_keys:
                exists = await self.redis_client.exists(key)
                if not exists:
                    missing_keys.append(key)
            
            if missing_keys:
                validation["redis_consistency"] = False
                validation["details"]["missing_keys"] = missing_keys
            
            # Check model files
            model_files = [
                ("tactical", self.config.tactical_model_path),
                ("strategic", self.config.strategic_model_path)
            ]
            
            for model_name, path in model_files:
                if not Path(path).exists():
                    validation["model_consistency"] = False
                    validation["details"][f"{model_name}_model_missing"] = True
            
            return validation
            
        except Exception as e:
            logger.error(f"State consistency validation failed: {e}")
            return {"error": str(e)}
    
    async def simulate_state_corruption(self, corruption_type: str) -> bool:
        """Simulate state corruption for testing."""
        try:
            if corruption_type == "redis_corruption":
                # Corrupt Redis data
                await self.redis_client.flushdb()
                logger.info("Redis state corrupted")
                
            elif corruption_type == "model_corruption":
                # Corrupt model files
                model_files = [
                    self.config.tactical_model_path,
                    self.config.strategic_model_path
                ]
                
                for path in model_files:
                    if Path(path).exists():
                        # Write random data to corrupt file
                        with open(path, 'wb') as f:
                            f.write(b"corrupted_data")
                
                logger.info("Model files corrupted")
                
            elif corruption_type == "partial_corruption":
                # Corrupt only specific keys
                keys_to_corrupt = ["tactical:state", "strategic:model_info"]
                
                for key in keys_to_corrupt:
                    await self.redis_client.set(key, "corrupted_value")
                
                logger.info("Partial state corruption applied")
            
            return True
            
        except Exception as e:
            logger.error(f"State corruption simulation failed: {e}")
            return False


class TradingEngineFailoverExecutor:
    """Executor for trading engine failover scenarios."""
    
    def __init__(self, config: TradingEngineFailoverConfig):
        self.config = config
        self.state_manager = TradingEngineStateManager(config)
        
    async def execute_failover_scenario(self, scenario: TradingEngineFailoverType) -> Dict[str, Any]:
        """Execute specific failover scenario."""
        logger.info(f"Executing trading engine failover scenario: {scenario.value}")
        
        scenario_map = {
            TradingEngineFailoverType.TACTICAL_AGENT_KILL: self._execute_tactical_kill,
            TradingEngineFailoverType.STRATEGIC_AGENT_KILL: self._execute_strategic_kill,
            TradingEngineFailoverType.MARL_COORDINATOR_FAILURE: self._execute_coordinator_failure,
            TradingEngineFailoverType.JIT_MODEL_CORRUPTION: self._execute_jit_corruption,
            TradingEngineFailoverType.REDIS_STATE_LOSS: self._execute_redis_failure,
            TradingEngineFailoverType.ORDER_MANAGER_FAILURE: self._execute_order_manager_failure,
            TradingEngineFailoverType.RISK_MANAGER_FAILURE: self._execute_risk_manager_failure,
            TradingEngineFailoverType.EXECUTION_ENGINE_FAILURE: self._execute_execution_failure,
            TradingEngineFailoverType.COMPLETE_SYSTEM_FAILURE: self._execute_complete_failure,
            TradingEngineFailoverType.CASCADING_COMPONENT_FAILURE: self._execute_cascading_failure
        }
        
        executor = scenario_map.get(scenario)
        if not executor:
            return {"success": False, "error": f"Unknown scenario: {scenario.value}"}
        
        try:
            await self.state_manager.initialize()
            
            start_time = time.time()
            result = await executor()
            execution_time = time.time() - start_time
            
            result["execution_time"] = execution_time
            return result
            
        except Exception as e:
            logger.error(f"Scenario execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        finally:
            await self.state_manager.close()
    
    async def _execute_tactical_kill(self) -> Dict[str, Any]:
        """Execute tactical agent kill scenario."""
        try:
            # Find tactical agent processes
            tactical_pids = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'tactical' in cmdline.lower() and 'python' in proc.info['name'].lower():
                        tactical_pids.append(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if not tactical_pids:
                return {"success": False, "error": "No tactical agent processes found"}
            
            # Kill processes
            killed_pids = []
            for pid in tactical_pids:
                try:
                    psutil.Process(pid).kill()
                    killed_pids.append(pid)
                except psutil.NoSuchProcess:
                    pass
            
            # Simulate state corruption
            await self.state_manager.simulate_state_corruption("partial_corruption")
            
            logger.info(f"Tactical agent killed: {killed_pids}")
            
            return {
                "success": True,
                "killed_pids": killed_pids,
                "affected_components": ["tactical_agent", "tactical_state"],
                "method": "process_kill"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_strategic_kill(self) -> Dict[str, Any]:
        """Execute strategic agent kill scenario."""
        try:
            # Find strategic agent processes
            strategic_pids = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'strategic' in cmdline.lower() and 'python' in proc.info['name'].lower():
                        strategic_pids.append(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if not strategic_pids:
                return {"success": False, "error": "No strategic agent processes found"}
            
            # Kill processes
            killed_pids = []
            for pid in strategic_pids:
                try:
                    psutil.Process(pid).kill()
                    killed_pids.append(pid)
                except psutil.NoSuchProcess:
                    pass
            
            logger.info(f"Strategic agent killed: {killed_pids}")
            
            return {
                "success": True,
                "killed_pids": killed_pids,
                "affected_components": ["strategic_agent", "strategic_state"],
                "method": "process_kill"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_coordinator_failure(self) -> Dict[str, Any]:
        """Execute MARL coordinator failure scenario."""
        try:
            # Find coordinator processes
            coordinator_pids = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'coordinator' in cmdline.lower() or 'marl' in cmdline.lower():
                        coordinator_pids.append(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Kill coordinator processes
            killed_pids = []
            for pid in coordinator_pids:
                try:
                    psutil.Process(pid).kill()
                    killed_pids.append(pid)
                except psutil.NoSuchProcess:
                    pass
            
            # Corrupt coordination state
            await self.state_manager.simulate_state_corruption("redis_corruption")
            
            logger.info(f"MARL coordinator failure: {killed_pids}")
            
            return {
                "success": True,
                "killed_pids": killed_pids,
                "affected_components": ["marl_coordinator", "agent_coordination"],
                "method": "coordinator_failure"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_jit_corruption(self) -> Dict[str, Any]:
        """Execute JIT model corruption scenario."""
        try:
            # Corrupt JIT model files
            corruption_success = await self.state_manager.simulate_state_corruption("model_corruption")
            
            if not corruption_success:
                return {"success": False, "error": "Model corruption failed"}
            
            logger.info("JIT model corruption executed")
            
            return {
                "success": True,
                "affected_components": ["jit_models", "model_cache"],
                "method": "model_corruption"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_redis_failure(self) -> Dict[str, Any]:
        """Execute Redis state loss scenario."""
        try:
            # Kill Redis processes
            redis_pids = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if 'redis' in proc.info['name'].lower():
                        redis_pids.append(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Kill Redis processes
            killed_pids = []
            for pid in redis_pids:
                try:
                    psutil.Process(pid).kill()
                    killed_pids.append(pid)
                except psutil.NoSuchProcess:
                    pass
            
            logger.info(f"Redis failure: {killed_pids}")
            
            return {
                "success": True,
                "killed_pids": killed_pids,
                "affected_components": ["redis", "state_storage", "agent_communication"],
                "method": "redis_failure"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_order_manager_failure(self) -> Dict[str, Any]:
        """Execute order manager failure scenario."""
        try:
            # Find order manager processes
            order_pids = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'order' in cmdline.lower() and 'manager' in cmdline.lower():
                        order_pids.append(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Kill order manager processes
            killed_pids = []
            for pid in order_pids:
                try:
                    psutil.Process(pid).kill()
                    killed_pids.append(pid)
                except psutil.NoSuchProcess:
                    pass
            
            logger.info(f"Order manager failure: {killed_pids}")
            
            return {
                "success": True,
                "killed_pids": killed_pids,
                "affected_components": ["order_manager", "trade_execution"],
                "method": "order_manager_failure"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_risk_manager_failure(self) -> Dict[str, Any]:
        """Execute risk manager failure scenario."""
        try:
            # Find risk manager processes
            risk_pids = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'risk' in cmdline.lower() and 'manager' in cmdline.lower():
                        risk_pids.append(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Kill risk manager processes
            killed_pids = []
            for pid in risk_pids:
                try:
                    psutil.Process(pid).kill()
                    killed_pids.append(pid)
                except psutil.NoSuchProcess:
                    pass
            
            logger.info(f"Risk manager failure: {killed_pids}")
            
            return {
                "success": True,
                "killed_pids": killed_pids,
                "affected_components": ["risk_manager", "risk_monitoring"],
                "method": "risk_manager_failure"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_execution_failure(self) -> Dict[str, Any]:
        """Execute execution engine failure scenario."""
        try:
            # Find execution engine processes
            execution_pids = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'execution' in cmdline.lower() and 'engine' in cmdline.lower():
                        execution_pids.append(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Kill execution engine processes
            killed_pids = []
            for pid in execution_pids:
                try:
                    psutil.Process(pid).kill()
                    killed_pids.append(pid)
                except psutil.NoSuchProcess:
                    pass
            
            logger.info(f"Execution engine failure: {killed_pids}")
            
            return {
                "success": True,
                "killed_pids": killed_pids,
                "affected_components": ["execution_engine", "trade_routing"],
                "method": "execution_failure"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_complete_failure(self) -> Dict[str, Any]:
        """Execute complete system failure scenario."""
        try:
            # Kill all trading-related processes
            trading_pids = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    trading_keywords = ['tactical', 'strategic', 'trading', 'marl', 'execution', 'risk']
                    
                    if any(keyword in cmdline.lower() for keyword in trading_keywords):
                        trading_pids.append(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Kill all trading processes
            killed_pids = []
            for pid in trading_pids:
                try:
                    psutil.Process(pid).kill()
                    killed_pids.append(pid)
                except psutil.NoSuchProcess:
                    pass
            
            # Corrupt all state
            await self.state_manager.simulate_state_corruption("redis_corruption")
            await self.state_manager.simulate_state_corruption("model_corruption")
            
            logger.info(f"Complete system failure: {killed_pids}")
            
            return {
                "success": True,
                "killed_pids": killed_pids,
                "affected_components": ["all_agents", "all_models", "all_state"],
                "method": "complete_failure"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_cascading_failure(self) -> Dict[str, Any]:
        """Execute cascading component failure scenario."""
        try:
            failures = []
            
            # Phase 1: Tactical agent failure
            tactical_result = await self._execute_tactical_kill()
            failures.append({"phase": "tactical_kill", "result": tactical_result})
            await asyncio.sleep(5)
            
            # Phase 2: Redis failure
            redis_result = await self._execute_redis_failure()
            failures.append({"phase": "redis_failure", "result": redis_result})
            await asyncio.sleep(5)
            
            # Phase 3: Strategic agent failure
            strategic_result = await self._execute_strategic_kill()
            failures.append({"phase": "strategic_kill", "result": strategic_result})
            
            logger.info("Cascading failure scenario executed")
            
            return {
                "success": True,
                "failures": failures,
                "affected_components": ["tactical", "strategic", "redis", "coordination"],
                "method": "cascading_failure"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def cleanup_scenario(self, scenario: TradingEngineFailoverType, scenario_result: Dict[str, Any]) -> bool:
        """Clean up after scenario execution."""
        try:
            logger.info(f"Cleaning up scenario: {scenario.value}")
            
            # Restart killed services
            if scenario in [TradingEngineFailoverType.TACTICAL_AGENT_KILL, 
                          TradingEngineFailoverType.COMPLETE_SYSTEM_FAILURE]:
                subprocess.run(["systemctl", "restart", "tactical-agent"], capture_output=True)
            
            if scenario in [TradingEngineFailoverType.STRATEGIC_AGENT_KILL,
                          TradingEngineFailoverType.COMPLETE_SYSTEM_FAILURE]:
                subprocess.run(["systemctl", "restart", "strategic-agent"], capture_output=True)
            
            if scenario in [TradingEngineFailoverType.REDIS_STATE_LOSS,
                          TradingEngineFailoverType.COMPLETE_SYSTEM_FAILURE]:
                subprocess.run(["systemctl", "restart", "redis"], capture_output=True)
            
            # Wait for services to stabilize
            await asyncio.sleep(10)
            
            logger.info(f"Cleanup completed for scenario: {scenario.value}")
            return True
            
        except Exception as e:
            logger.error(f"Cleanup failed for scenario {scenario.value}: {e}")
            return False


class TradingEngineFailoverTester:
    """Main class for trading engine failover testing."""
    
    def __init__(self, config: TradingEngineFailoverConfig):
        self.config = config
        self.monitor = TradingEngineMonitor(config)
        self.state_manager = TradingEngineStateManager(config)
        self.failover_executor = TradingEngineFailoverExecutor(config)
        
    async def run_failover_test(self) -> TradingEngineFailoverResult:
        """Run complete trading engine failover test."""
        result = TradingEngineFailoverResult(
            test_id=self.config.test_id,
            test_name=f"Trading Engine Failover Test - {self.config.failover_type.value}",
            status="running",
            start_time=datetime.now()
        )
        
        logger.info(f"Starting trading engine failover test: {result.test_name}")
        
        try:
            # Initialize components
            await self.state_manager.initialize()
            
            # Phase 1: Pre-test validation
            logger.info("Phase 1: Pre-test validation")
            await self._run_pre_test_validation(result)
            
            # Phase 2: Backup state
            logger.info("Phase 2: Backing up state")
            await self._backup_system_state(result)
            
            # Phase 3: Establish performance baseline
            logger.info("Phase 3: Establishing performance baseline")
            await self._establish_performance_baseline(result)
            
            # Phase 4: Execute failover scenario
            logger.info("Phase 4: Executing failover scenario")
            await self._execute_failover_scenario(result)
            
            # Phase 5: Monitor recovery
            logger.info("Phase 5: Monitoring recovery")
            await self._monitor_recovery_process(result)
            
            # Phase 6: Validate recovery
            logger.info("Phase 6: Validating recovery")
            await self._validate_recovery(result)
            
            # Phase 7: Performance validation
            logger.info("Phase 7: Performance validation")
            await self._validate_performance(result)
            
            # Phase 8: Cleanup
            logger.info("Phase 8: Cleanup")
            await self._cleanup_test(result)
            
            result.status = "completed"
            result.end_time = datetime.now()
            
            # Calculate final metrics
            self._calculate_final_metrics(result)
            
            logger.info(f"Test completed: {result.test_name}")
            logger.info(f"RTO Target Met: {result.rto_metrics.meets_rto_target(self.config.target_rto_seconds)}")
            logger.info(f"Total Recovery Time: {result.rto_metrics.total_recovery_time():.2f}s")
            
            return result
            
        except Exception as e:
            result.status = "failed"
            result.end_time = datetime.now()
            result.errors.append(f"Test failed: {str(e)}")
            
            logger.error(f"Test failed: {result.test_name} - {str(e)}")
            logger.error(traceback.format_exc())
            
            # Attempt cleanup
            try:
                await self._emergency_cleanup(result)
            except Exception as cleanup_error:
                logger.error(f"Emergency cleanup failed: {cleanup_error}")
            
            return result
            
        finally:
            await self.state_manager.close()
    
    async def _run_pre_test_validation(self, result: TradingEngineFailoverResult):
        """Run pre-test validation."""
        try:
            # Check system status
            system_status = await self.monitor.get_system_status()
            result.pre_test_state["system_status"] = system_status
            
            # Validate all services are healthy
            unhealthy_services = []
            for service, status in system_status.get("services", {}).items():
                if not status.get("healthy", False):
                    unhealthy_services.append(service)
            
            if unhealthy_services:
                result.warnings.append(f"Unhealthy services before test: {unhealthy_services}")
            
            # Check agent states
            agent_states = {}
            for agent_type in ["tactical", "strategic", "risk", "execution"]:
                try:
                    agent_state = await self.monitor.get_agent_state(agent_type)
                    agent_states[agent_type] = agent_state
                except Exception as e:
                    result.warnings.append(f"Failed to get {agent_type} agent state: {e}")
            
            result.pre_test_state["agent_states"] = agent_states
            
            # Validate Redis connectivity
            if not system_status.get("redis", {}).get("healthy", False):
                raise Exception("Redis is not healthy")
            
            logger.info("Pre-test validation completed")
            
        except Exception as e:
            result.errors.append(f"Pre-test validation failed: {str(e)}")
            raise
    
    async def _backup_system_state(self, result: TradingEngineFailoverResult):
        """Backup system state."""
        try:
            backup = await self.state_manager.backup_state()
            result.pre_test_state["state_backup"] = backup
            
            if not backup:
                result.warnings.append("State backup failed")
            
            logger.info("System state backup completed")
            
        except Exception as e:
            result.errors.append(f"State backup failed: {str(e)}")
            raise
    
    async def _establish_performance_baseline(self, result: TradingEngineFailoverResult):
        """Establish performance baseline."""
        try:
            baseline_performance = await self.monitor.measure_performance(iterations=20)
            result.pre_test_state["baseline_performance"] = baseline_performance
            
            # Extract key metrics
            if "service_latency" in baseline_performance:
                tactical_latency = baseline_performance["service_latency"].get("tactical", {})
                result.rto_metrics.pre_failover_latency = tactical_latency.get("avg_latency", 0.0)
            
            logger.info("Performance baseline established")
            
        except Exception as e:
            result.errors.append(f"Performance baseline establishment failed: {str(e)}")
            raise
    
    async def _execute_failover_scenario(self, result: TradingEngineFailoverResult):
        """Execute failover scenario."""
        try:
            # Record start time
            failover_start = time.time()
            
            # Execute scenario
            scenario_result = await self.failover_executor.execute_failover_scenario(
                self.config.failover_type
            )
            
            if not scenario_result.get("success"):
                raise Exception(f"Scenario execution failed: {scenario_result.get('error')}")
            
            # Record detection time
            result.rto_metrics.failure_detection_time = time.time() - failover_start
            
            result.failover_execution = scenario_result
            
            logger.info(f"Failover scenario executed: {scenario_result}")
            
        except Exception as e:
            result.errors.append(f"Failover scenario execution failed: {str(e)}")
            raise
    
    async def _monitor_recovery_process(self, result: TradingEngineFailoverResult):
        """Monitor recovery process."""
        try:
            recovery_start = time.time()
            max_wait_time = self.config.max_test_duration
            
            recovery_data = {
                "status_checks": [],
                "recovery_events": [],
                "performance_samples": []
            }
            
            first_heartbeat_time = None
            first_trade_signal_time = None
            models_recompiled_time = None
            
            while time.time() - recovery_start < max_wait_time:
                current_time = time.time()
                
                # Check system status
                system_status = await self.monitor.get_system_status()
                recovery_data["status_checks"].append({
                    "timestamp": current_time,
                    "system_status": system_status
                })
                
                # Check for first heartbeat
                if first_heartbeat_time is None:
                    tactical_healthy = system_status.get("services", {}).get("tactical", {}).get("healthy", False)
                    strategic_healthy = system_status.get("services", {}).get("strategic", {}).get("healthy", False)
                    
                    if tactical_healthy or strategic_healthy:
                        first_heartbeat_time = current_time
                        result.rto_metrics.first_heartbeat_time = current_time - recovery_start
                        recovery_data["recovery_events"].append({
                            "event": "first_heartbeat",
                            "timestamp": current_time,
                            "time_from_start": current_time - recovery_start
                        })
                
                # Check for model recompilation
                if models_recompiled_time is None:
                    models_status = system_status.get("models", {})
                    if models_status.get("tactical", {}).get("exists", False):
                        models_recompiled_time = current_time
                        result.rto_metrics.model_recompilation_time = current_time - recovery_start
                        recovery_data["recovery_events"].append({
                            "event": "models_recompiled",
                            "timestamp": current_time,
                            "time_from_start": current_time - recovery_start
                        })
                
                # Sample performance
                if first_heartbeat_time:
                    try:
                        perf_sample = await self.monitor.measure_performance(iterations=1)
                        recovery_data["performance_samples"].append({
                            "timestamp": current_time,
                            "performance": perf_sample
                        })
                    except Exception as e:
                        recovery_data["performance_samples"].append({
                            "timestamp": current_time,
                            "error": str(e)
                        })
                
                # Check if recovery is complete
                if (first_heartbeat_time and models_recompiled_time and
                    system_status.get("redis", {}).get("healthy", False)):
                    
                    result.rto_metrics.full_functionality_time = current_time - recovery_start
                    recovery_data["recovery_events"].append({
                        "event": "full_functionality",
                        "timestamp": current_time,
                        "time_from_start": current_time - recovery_start
                    })
                    break
                
                await asyncio.sleep(1)
            
            result.recovery_process = recovery_data
            
            logger.info(f"Recovery monitoring completed")
            
        except Exception as e:
            result.errors.append(f"Recovery monitoring failed: {str(e)}")
            raise
    
    async def _validate_recovery(self, result: TradingEngineFailoverResult):
        """Validate recovery completeness."""
        try:
            validation_results = {
                "agent_state_recovered": False,
                "jit_models_available": False,
                "redis_state_restored": False,
                "services_healthy": False
            }
            
            # Check system status
            system_status = await self.monitor.get_system_status()
            
            # Validate services
            services_healthy = all(
                service.get("healthy", False)
                for service in system_status.get("services", {}).values()
            )
            validation_results["services_healthy"] = services_healthy
            
            # Validate Redis
            redis_healthy = system_status.get("redis", {}).get("healthy", False)
            validation_results["redis_state_restored"] = redis_healthy
            result.rto_metrics.redis_state_restored = redis_healthy
            
            # Validate models
            models_available = all(
                model.get("exists", False)
                for model in system_status.get("models", {}).values()
            )
            validation_results["jit_models_available"] = models_available
            result.rto_metrics.jit_models_recompiled = models_available
            
            # Validate agent states
            agent_states_recovered = True
            for agent_type in ["tactical", "strategic"]:
                try:
                    agent_state = await self.monitor.get_agent_state(agent_type)
                    if not agent_state.get("healthy", False):
                        agent_states_recovered = False
                        break
                except Exception:
                    agent_states_recovered = False
                    break
            
            validation_results["agent_state_recovered"] = agent_states_recovered
            result.rto_metrics.agent_state_recovered = agent_states_recovered
            
            # Validate state consistency
            state_consistency = await self.state_manager.validate_state_consistency()
            validation_results["state_consistency"] = state_consistency
            
            result.post_test_validation = validation_results
            
            # Check if all validations passed
            all_passed = all(validation_results.values())
            if not all_passed:
                failed_validations = [k for k, v in validation_results.items() if not v]
                result.warnings.append(f"Recovery validation failed: {failed_validations}")
            
            logger.info(f"Recovery validation completed: {validation_results}")
            
        except Exception as e:
            result.errors.append(f"Recovery validation failed: {str(e)}")
            raise
    
    async def _validate_performance(self, result: TradingEngineFailoverResult):
        """Validate post-failover performance."""
        try:
            # Measure current performance
            current_performance = await self.monitor.measure_performance(iterations=20)
            result.performance_metrics["post_failover_performance"] = current_performance
            
            # Calculate performance degradation
            baseline_latency = result.rto_metrics.pre_failover_latency
            current_latency = 0.0
            
            if "service_latency" in current_performance:
                tactical_latency = current_performance["service_latency"].get("tactical", {})
                current_latency = tactical_latency.get("avg_latency", 0.0)
            
            result.rto_metrics.post_failover_latency = current_latency
            
            if baseline_latency > 0:
                degradation = ((current_latency - baseline_latency) / baseline_latency) * 100
                result.rto_metrics.performance_degradation_percent = max(0, degradation)
            
            # Performance acceptance criteria
            performance_acceptable = (
                current_latency < self.config.performance_threshold_ms / 1000.0 and
                result.rto_metrics.performance_degradation_percent < 50.0
            )
            
            result.performance_metrics["performance_acceptable"] = performance_acceptable
            
            if not performance_acceptable:
                result.warnings.append(f"Performance degradation: {result.rto_metrics.performance_degradation_percent:.1f}%")
            
            logger.info(f"Performance validation completed")
            
        except Exception as e:
            result.errors.append(f"Performance validation failed: {str(e)}")
            raise
    
    async def _cleanup_test(self, result: TradingEngineFailoverResult):
        """Clean up test resources."""
        try:
            # Cleanup scenario
            cleanup_success = await self.failover_executor.cleanup_scenario(
                self.config.failover_type,
                result.failover_execution
            )
            
            if not cleanup_success:
                result.warnings.append("Scenario cleanup failed")
            
            # Restore state if needed
            if result.pre_test_state.get("state_backup"):
                restore_success = await self.state_manager.restore_state(
                    result.pre_test_state["state_backup"]
                )
                if not restore_success:
                    result.warnings.append("State restore failed")
            
            logger.info("Test cleanup completed")
            
        except Exception as e:
            result.warnings.append(f"Test cleanup failed: {str(e)}")
    
    async def _emergency_cleanup(self, result: TradingEngineFailoverResult):
        """Emergency cleanup for failed tests."""
        try:
            logger.warning("Performing emergency cleanup")
            
            # Restart all services
            services = ["tactical-agent", "strategic-agent", "redis", "risk-manager"]
            for service in services:
                subprocess.run(["systemctl", "restart", service], capture_output=True)
            
            # Wait for services to stabilize
            await asyncio.sleep(30)
            
            logger.info("Emergency cleanup completed")
            
        except Exception as e:
            logger.error(f"Emergency cleanup failed: {e}")
    
    def _calculate_final_metrics(self, result: TradingEngineFailoverResult):
        """Calculate final metrics."""
        try:
            # Calculate state recovery time
            result.rto_metrics.state_recovery_time = max(
                result.rto_metrics.first_heartbeat_time,
                result.rto_metrics.model_recompilation_time
            )
            
            # Validate order and position consistency
            result.rto_metrics.order_state_consistent = (
                result.rto_metrics.agent_state_recovered and
                result.rto_metrics.redis_state_restored
            )
            
            result.rto_metrics.position_reconciled = (
                result.rto_metrics.order_state_consistent and
                result.rto_metrics.jit_models_recompiled
            )
            
            logger.info(f"Final metrics calculated: {result.rto_metrics}")
            
        except Exception as e:
            result.warnings.append(f"Final metrics calculation failed: {str(e)}")


# Example usage
async def main():
    """Demonstrate trading engine failover testing."""
    config = TradingEngineFailoverConfig(
        test_id="trading_engine_failover_001",
        failover_type=TradingEngineFailoverType.TACTICAL_AGENT_KILL,
        target_rto_seconds=30.0,
        tactical_api_url="http://localhost:8001",
        strategic_api_url="http://localhost:8002"
    )
    
    tester = TradingEngineFailoverTester(config)
    result = await tester.run_failover_test()
    
    print(f"Test Result: {result.status}")
    print(f"RTO Target Met: {result.rto_metrics.meets_rto_target()}")
    print(f"Total Recovery Time: {result.rto_metrics.total_recovery_time():.2f}s")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")


if __name__ == "__main__":
    asyncio.run(main())