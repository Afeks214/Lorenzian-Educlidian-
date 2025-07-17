"""
Service Health State Machine

Advanced health monitoring with state transitions and automated recovery actions.
"""

import asyncio
import logging
import time
import json
from enum import Enum
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class HealthState(Enum):
    """Service health states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"
    RECOVERING = "recovering"
    MAINTENANCE = "maintenance"

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class HealthCheck:
    """Health check configuration."""
    name: str
    check_function: Callable
    interval_seconds: int
    timeout_seconds: int
    failure_threshold: int
    recovery_threshold: int
    alert_level: AlertLevel
    enabled: bool = True

@dataclass
class StateTransition:
    """State transition record."""
    from_state: HealthState
    to_state: HealthState
    timestamp: float
    trigger: str
    details: Dict[str, Any]

@dataclass
class HealthMetrics:
    """Health metrics for a service."""
    state: HealthState
    last_check_time: float
    consecutive_failures: int
    consecutive_successes: int
    total_checks: int
    total_failures: int
    uptime_seconds: float
    last_failure_time: Optional[float] = None
    last_recovery_time: Optional[float] = None

class ServiceHealthStateMachine:
    """
    Advanced service health state machine with automated recovery.
    
    Features:
    - State-based health monitoring
    - Automated recovery actions
    - Alert escalation
    - Health metrics tracking
    - Configurable thresholds
    """
    
    def __init__(self, service_name: str, redis_url: str = "redis://localhost:6379/2"):
        """Initialize health state machine."""
        self.service_name = service_name
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        
        # Current state
        self.current_state = HealthState.HEALTHY
        self.state_entered_time = time.time()
        self.state_history: List[StateTransition] = []
        
        # Health checks
        self.health_checks: Dict[str, HealthCheck] = {}
        self.check_results: Dict[str, Dict[str, Any]] = {}
        
        # Metrics
        self.metrics = HealthMetrics(
            state=HealthState.HEALTHY,
            last_check_time=time.time(),
            consecutive_failures=0,
            consecutive_successes=0,
            total_checks=0,
            total_failures=0,
            uptime_seconds=0.0
        )
        
        # Configuration
        self.config = {
            "state_transition_timeout": 300,  # 5 minutes
            "max_state_history": 100,
            "alert_cooldown_seconds": 300,
            "recovery_validation_checks": 3,
            "maintenance_mode_duration": 1800,  # 30 minutes
        }
        
        # Alert tracking
        self.last_alert_time: Dict[str, float] = {}
        
        # Recovery actions
        self.recovery_actions: Dict[HealthState, List[Callable]] = {
            HealthState.DEGRADED: [],
            HealthState.CRITICAL: [],
            HealthState.FAILED: []
        }
        
        self.running = False
        self.monitor_task = None
        
        logger.info(f"Health state machine initialized for service: {service_name}")
    
    async def initialize(self):
        """Initialize Redis connection and restore state."""
        self.redis_client = redis.from_url(self.redis_url)
        await self.redis_client.ping()
        
        # Restore state from Redis
        await self._restore_state()
        
        # Register default health checks
        await self._register_default_health_checks()
        
        logger.info(f"Health state machine initialized for {self.service_name}")
    
    async def _restore_state(self):
        """Restore state from Redis."""
        try:
            state_key = f"health_state:{self.service_name}"
            state_data = await self.redis_client.get(state_key)
            
            if state_data:
                data = json.loads(state_data)
                self.current_state = HealthState(data.get("state", "healthy"))
                self.state_entered_time = data.get("state_entered_time", time.time())
                
                # Restore metrics
                if "metrics" in data:
                    metrics_data = data["metrics"]
                    self.metrics = HealthMetrics(
                        state=HealthState(metrics_data.get("state", "healthy")),
                        last_check_time=metrics_data.get("last_check_time", time.time()),
                        consecutive_failures=metrics_data.get("consecutive_failures", 0),
                        consecutive_successes=metrics_data.get("consecutive_successes", 0),
                        total_checks=metrics_data.get("total_checks", 0),
                        total_failures=metrics_data.get("total_failures", 0),
                        uptime_seconds=metrics_data.get("uptime_seconds", 0.0),
                        last_failure_time=metrics_data.get("last_failure_time"),
                        last_recovery_time=metrics_data.get("last_recovery_time")
                    )
                
                logger.info(f"Restored health state: {self.current_state.value}")
            else:
                logger.info("No previous state found, starting fresh")
                
        except Exception as e:
            logger.error(f"Failed to restore state: {e}")
    
    async def _persist_state(self):
        """Persist current state to Redis."""
        try:
            state_key = f"health_state:{self.service_name}"
            state_data = {
                "state": self.current_state.value,
                "state_entered_time": self.state_entered_time,
                "timestamp": time.time(),
                "metrics": {
                    "state": self.metrics.state.value,
                    "last_check_time": self.metrics.last_check_time,
                    "consecutive_failures": self.metrics.consecutive_failures,
                    "consecutive_successes": self.metrics.consecutive_successes,
                    "total_checks": self.metrics.total_checks,
                    "total_failures": self.metrics.total_failures,
                    "uptime_seconds": self.metrics.uptime_seconds,
                    "last_failure_time": self.metrics.last_failure_time,
                    "last_recovery_time": self.metrics.last_recovery_time
                }
            }
            
            await self.redis_client.setex(state_key, 3600, json.dumps(state_data))
            
        except Exception as e:
            logger.error(f"Failed to persist state: {e}")
    
    async def _register_default_health_checks(self):
        """Register default health checks."""
        # HTTP health check
        await self.register_health_check(
            name="http_health_check",
            check_function=self._http_health_check,
            interval_seconds=30,
            timeout_seconds=10,
            failure_threshold=3,
            recovery_threshold=2,
            alert_level=AlertLevel.CRITICAL
        )
        
        # Redis connectivity check
        await self.register_health_check(
            name="redis_connectivity",
            check_function=self._redis_connectivity_check,
            interval_seconds=60,
            timeout_seconds=5,
            failure_threshold=2,
            recovery_threshold=1,
            alert_level=AlertLevel.WARNING
        )
        
        # Event processing check
        await self.register_health_check(
            name="event_processing",
            check_function=self._event_processing_check,
            interval_seconds=120,
            timeout_seconds=15,
            failure_threshold=2,
            recovery_threshold=1,
            alert_level=AlertLevel.WARNING
        )
    
    async def register_health_check(
        self,
        name: str,
        check_function: Callable,
        interval_seconds: int,
        timeout_seconds: int,
        failure_threshold: int,
        recovery_threshold: int,
        alert_level: AlertLevel
    ):
        """Register a health check."""
        health_check = HealthCheck(
            name=name,
            check_function=check_function,
            interval_seconds=interval_seconds,
            timeout_seconds=timeout_seconds,
            failure_threshold=failure_threshold,
            recovery_threshold=recovery_threshold,
            alert_level=alert_level
        )
        
        self.health_checks[name] = health_check
        self.check_results[name] = {
            "last_check": 0,
            "consecutive_failures": 0,
            "consecutive_successes": 0,
            "last_result": None,
            "last_error": None
        }
        
        logger.info(f"Registered health check: {name}")
    
    async def start_monitoring(self):
        """Start health monitoring."""
        if self.running:
            return
        
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info(f"Started health monitoring for {self.service_name}")
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        self.running = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"Stopped health monitoring for {self.service_name}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Run health checks
                await self._run_health_checks()
                
                # Evaluate state transitions
                await self._evaluate_state_transitions()
                
                # Update metrics
                await self._update_metrics()
                
                # Persist state
                await self._persist_state()
                
                # Brief pause
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def _run_health_checks(self):
        """Run all enabled health checks."""
        current_time = time.time()
        
        for name, check in self.health_checks.items():
            if not check.enabled:
                continue
            
            result = self.check_results[name]
            
            # Check if it's time to run this check
            if current_time - result["last_check"] < check.interval_seconds:
                continue
            
            # Run the health check
            try:
                check_result = await asyncio.wait_for(
                    check.check_function(),
                    timeout=check.timeout_seconds
                )
                
                # Update results
                result["last_check"] = current_time
                result["last_result"] = check_result
                result["last_error"] = None
                
                if check_result:
                    result["consecutive_successes"] += 1
                    result["consecutive_failures"] = 0
                else:
                    result["consecutive_failures"] += 1
                    result["consecutive_successes"] = 0
                
                logger.debug(f"Health check {name}: {'PASS' if check_result else 'FAIL'}")
                
            except asyncio.TimeoutError:
                result["last_check"] = current_time
                result["last_result"] = False
                result["last_error"] = "timeout"
                result["consecutive_failures"] += 1
                result["consecutive_successes"] = 0
                
                logger.warning(f"Health check {name} timed out")
                
            except Exception as e:
                result["last_check"] = current_time
                result["last_result"] = False
                result["last_error"] = str(e)
                result["consecutive_failures"] += 1
                result["consecutive_successes"] = 0
                
                logger.error(f"Health check {name} failed: {e}")
    
    async def _evaluate_state_transitions(self):
        """Evaluate and execute state transitions."""
        current_time = time.time()
        
        # Calculate overall health score
        health_score = await self._calculate_health_score()
        
        # Determine new state based on health score and current state
        new_state = await self._determine_new_state(health_score)
        
        # Execute state transition if needed
        if new_state != self.current_state:
            await self._transition_to_state(new_state, f"health_score_{health_score}")
    
    async def _calculate_health_score(self) -> float:
        """Calculate overall health score (0.0 = failed, 1.0 = healthy)."""
        if not self.health_checks:
            return 1.0
        
        total_weight = 0
        weighted_score = 0
        
        for name, check in self.health_checks.items():
            if not check.enabled:
                continue
            
            result = self.check_results[name]
            
            # Weight based on alert level
            weight = {
                AlertLevel.INFO: 0.1,
                AlertLevel.WARNING: 0.3,
                AlertLevel.CRITICAL: 0.6,
                AlertLevel.EMERGENCY: 1.0
            }.get(check.alert_level, 0.5)
            
            # Calculate individual check score
            if result["consecutive_failures"] >= check.failure_threshold:
                check_score = 0.0
            elif result["consecutive_successes"] >= check.recovery_threshold:
                check_score = 1.0
            else:
                # Partial score during transition
                check_score = 0.5
            
            weighted_score += weight * check_score
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 1.0
    
    async def _determine_new_state(self, health_score: float) -> HealthState:
        """Determine new state based on health score."""
        current_state = self.current_state
        
        # State transition logic
        if health_score >= 0.9:
            if current_state in [HealthState.DEGRADED, HealthState.CRITICAL, HealthState.FAILED, HealthState.RECOVERING]:
                return HealthState.HEALTHY
            return current_state
        
        elif health_score >= 0.7:
            if current_state == HealthState.HEALTHY:
                return HealthState.DEGRADED
            elif current_state in [HealthState.CRITICAL, HealthState.FAILED]:
                return HealthState.RECOVERING
            return current_state
        
        elif health_score >= 0.3:
            if current_state in [HealthState.HEALTHY, HealthState.DEGRADED]:
                return HealthState.CRITICAL
            return current_state
        
        else:
            return HealthState.FAILED
    
    async def _transition_to_state(self, new_state: HealthState, trigger: str):
        """Execute state transition."""
        old_state = self.current_state
        current_time = time.time()
        
        # Record transition
        transition = StateTransition(
            from_state=old_state,
            to_state=new_state,
            timestamp=current_time,
            trigger=trigger,
            details={}
        )
        
        self.state_history.append(transition)
        if len(self.state_history) > self.config["max_state_history"]:
            self.state_history = self.state_history[-self.config["max_state_history"]:]
        
        # Update state
        self.current_state = new_state
        self.state_entered_time = current_time
        
        # Update metrics
        self.metrics.state = new_state
        if new_state == HealthState.HEALTHY and old_state != HealthState.HEALTHY:
            self.metrics.last_recovery_time = current_time
        elif new_state in [HealthState.CRITICAL, HealthState.FAILED] and old_state not in [HealthState.CRITICAL, HealthState.FAILED]:
            self.metrics.last_failure_time = current_time
        
        logger.info(f"State transition: {old_state.value} -> {new_state.value} (trigger: {trigger})")
        
        # Execute recovery actions
        await self._execute_recovery_actions(new_state)
        
        # Send alerts
        await self._send_state_transition_alert(old_state, new_state, trigger)
    
    async def _execute_recovery_actions(self, state: HealthState):
        """Execute recovery actions for the given state."""
        actions = self.recovery_actions.get(state, [])
        
        for action in actions:
            try:
                await action()
                logger.info(f"Executed recovery action for state: {state.value}")
            except Exception as e:
                logger.error(f"Recovery action failed for state {state.value}: {e}")
    
    async def _send_state_transition_alert(self, old_state: HealthState, new_state: HealthState, trigger: str):
        """Send alert for state transition."""
        alert_level = self._get_alert_level_for_state(new_state)
        
        # Check alert cooldown
        alert_key = f"state_transition_{old_state.value}_{new_state.value}"
        if alert_key in self.last_alert_time:
            if time.time() - self.last_alert_time[alert_key] < self.config["alert_cooldown_seconds"]:
                return
        
        # Send alert
        alert_data = {
            "service": self.service_name,
            "alert_type": "state_transition",
            "alert_level": alert_level.value,
            "old_state": old_state.value,
            "new_state": new_state.value,
            "trigger": trigger,
            "timestamp": time.time(),
            "metrics": {
                "uptime_seconds": self.metrics.uptime_seconds,
                "total_checks": self.metrics.total_checks,
                "total_failures": self.metrics.total_failures,
                "consecutive_failures": self.metrics.consecutive_failures
            }
        }
        
        await self.redis_client.xadd("tactical_alerts", alert_data)
        self.last_alert_time[alert_key] = time.time()
        
        logger.warning(f"Sent {alert_level.value} alert for state transition: {old_state.value} -> {new_state.value}")
    
    def _get_alert_level_for_state(self, state: HealthState) -> AlertLevel:
        """Get alert level for a state."""
        return {
            HealthState.HEALTHY: AlertLevel.INFO,
            HealthState.DEGRADED: AlertLevel.WARNING,
            HealthState.CRITICAL: AlertLevel.CRITICAL,
            HealthState.FAILED: AlertLevel.EMERGENCY,
            HealthState.RECOVERING: AlertLevel.INFO,
            HealthState.MAINTENANCE: AlertLevel.INFO
        }.get(state, AlertLevel.WARNING)
    
    async def _update_metrics(self):
        """Update health metrics."""
        current_time = time.time()
        
        self.metrics.last_check_time = current_time
        self.metrics.total_checks += 1
        
        # Update uptime
        if self.current_state == HealthState.HEALTHY:
            self.metrics.uptime_seconds = current_time - self.state_entered_time
        
        # Update consecutive counters
        if self.current_state in [HealthState.CRITICAL, HealthState.FAILED]:
            self.metrics.consecutive_failures += 1
            self.metrics.consecutive_successes = 0
        else:
            self.metrics.consecutive_successes += 1
            if self.current_state == HealthState.HEALTHY:
                self.metrics.consecutive_failures = 0
    
    # Default health check implementations
    async def _http_health_check(self) -> bool:
        """Check HTTP health endpoint."""
        try:
            import subprocess
            result = subprocess.run(
                ["curl", "-f", "http://localhost:8001/health"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
    
    async def _redis_connectivity_check(self) -> bool:
        """Check Redis connectivity."""
        try:
            await self.redis_client.ping()
            return True
        except Exception:
            return False
    
    async def _event_processing_check(self) -> bool:
        """Check if event processing is active."""
        try:
            # Check if there are recent events processed
            processing_key = "tactical:last_event_processed"
            last_processed = await self.redis_client.get(processing_key)
            
            if last_processed:
                last_time = float(last_processed)
                # Consider healthy if processed event within last 5 minutes
                return time.time() - last_time < 300
            
            return False
        except Exception:
            return False
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        current_time = time.time()
        
        return {
            "service": self.service_name,
            "state": self.current_state.value,
            "state_duration_seconds": current_time - self.state_entered_time,
            "health_score": await self._calculate_health_score(),
            "timestamp": current_time,
            "metrics": {
                "uptime_seconds": self.metrics.uptime_seconds,
                "total_checks": self.metrics.total_checks,
                "total_failures": self.metrics.total_failures,
                "consecutive_failures": self.metrics.consecutive_failures,
                "consecutive_successes": self.metrics.consecutive_successes,
                "last_failure_time": self.metrics.last_failure_time,
                "last_recovery_time": self.metrics.last_recovery_time
            },
            "health_checks": {
                name: {
                    "enabled": check.enabled,
                    "last_result": self.check_results[name]["last_result"],
                    "consecutive_failures": self.check_results[name]["consecutive_failures"],
                    "consecutive_successes": self.check_results[name]["consecutive_successes"],
                    "last_error": self.check_results[name]["last_error"]
                }
                for name, check in self.health_checks.items()
            },
            "recent_transitions": [
                {
                    "from_state": t.from_state.value,
                    "to_state": t.to_state.value,
                    "timestamp": t.timestamp,
                    "trigger": t.trigger
                }
                for t in self.state_history[-10:]
            ]
        }
    
    async def force_state_transition(self, new_state: HealthState, reason: str = "manual"):
        """Force a state transition (for maintenance, etc.)."""
        await self._transition_to_state(new_state, f"manual_{reason}")
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.stop_monitoring()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info(f"Health state machine cleanup complete for {self.service_name}")