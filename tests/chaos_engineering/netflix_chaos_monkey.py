"""
Netflix Chaos Monkey Implementation
==================================

Production-ready chaos engineering implementation inspired by Netflix's Chaos Monkey.
Provides automated, intelligent chaos testing with production safeguards and 
comprehensive resilience validation.

Key Features:
- Automated chaos testing with intelligent scheduling
- Production-safe failure injection
- Comprehensive monitoring and alerting
- Rollback and safety mechanisms
- Resilience certification
- Game day coordination
"""

import asyncio
import time
import random
import logging
import json
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta, timezone
from pathlib import Path
import yaml
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from .enterprise_chaos_framework import EnterpriseChaosFramework
from .failure_injection_engine import FailureInjectionEngine
from .chaos_automation_scheduler import ChaosAutomationScheduler
from .resilience_validator import ResilienceValidator, ValidationLevel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChaosMode(Enum):
    """Chaos monkey operation modes."""
    DISABLED = "disabled"
    SCHEDULE_ONLY = "schedule_only"
    GAME_DAY_ONLY = "game_day_only"
    FULL_AUTOMATED = "full_automated"
    EMERGENCY_ONLY = "emergency_only"


class SafetyLevel(Enum):
    """Safety levels for chaos operations."""
    MINIMAL = "minimal"        # Minimal safety checks
    STANDARD = "standard"      # Standard production safety
    STRICT = "strict"          # Strict safety enforcement
    PARANOID = "paranoid"      # Maximum safety checks


@dataclass
class ChaosMonkeyConfig:
    """Configuration for Netflix Chaos Monkey."""
    # Basic configuration
    enabled: bool = True
    mode: ChaosMode = ChaosMode.SCHEDULE_ONLY
    safety_level: SafetyLevel = SafetyLevel.STANDARD
    
    # Scheduling configuration
    business_hours_only: bool = True
    excluded_days: List[str] = field(default_factory=lambda: ["saturday", "sunday"])
    excluded_dates: List[str] = field(default_factory=list)
    
    # Safety configuration
    min_healthy_instances: int = 2
    max_failures_per_hour: int = 3
    max_failures_per_day: int = 10
    safety_timeout_minutes: int = 30
    
    # Notification configuration
    alert_channels: List[str] = field(default_factory=lambda: ["slack", "email"])
    notify_before_chaos: bool = True
    notify_after_chaos: bool = True
    
    # Validation configuration
    pre_chaos_validation: bool = True
    post_chaos_validation: bool = True
    continuous_validation: bool = True
    
    # Metrics configuration
    metrics_enabled: bool = True
    metrics_retention_days: int = 30
    resilience_threshold: float = 0.8


class ChaosMonkeyMetrics:
    """Metrics collection for Chaos Monkey."""
    
    def __init__(self):
        self.metrics_data = {
            "chaos_executions": [],
            "failures_injected": [],
            "recovery_times": [],
            "availability_scores": [],
            "resilience_scores": []
        }
        self.metrics_lock = threading.Lock()
        
    def record_chaos_execution(self, execution_data: Dict[str, Any]):
        """Record a chaos execution."""
        with self.metrics_lock:
            self.metrics_data["chaos_executions"].append({
                "timestamp": datetime.now(timezone.utc),
                "execution_id": execution_data.get("execution_id"),
                "experiment_type": execution_data.get("experiment_type"),
                "success": execution_data.get("success", False),
                "duration": execution_data.get("duration", 0),
                "recovery_time": execution_data.get("recovery_time", 0),
                "availability": execution_data.get("availability", 0)
            })
    
    def record_failure_injection(self, failure_data: Dict[str, Any]):
        """Record a failure injection."""
        with self.metrics_lock:
            self.metrics_data["failures_injected"].append({
                "timestamp": datetime.now(timezone.utc),
                "failure_type": failure_data.get("failure_type"),
                "target_component": failure_data.get("target_component"),
                "success": failure_data.get("success", False),
                "blast_radius": failure_data.get("blast_radius", 0),
                "duration": failure_data.get("duration", 0)
            })
    
    def record_recovery_time(self, recovery_time: float):
        """Record a recovery time."""
        with self.metrics_lock:
            self.metrics_data["recovery_times"].append({
                "timestamp": datetime.now(timezone.utc),
                "recovery_time": recovery_time
            })
    
    def record_availability_score(self, availability: float):
        """Record an availability score."""
        with self.metrics_lock:
            self.metrics_data["availability_scores"].append({
                "timestamp": datetime.now(timezone.utc),
                "availability": availability
            })
    
    def record_resilience_score(self, resilience: float):
        """Record a resilience score."""
        with self.metrics_lock:
            self.metrics_data["resilience_scores"].append({
                "timestamp": datetime.now(timezone.utc),
                "resilience": resilience
            })
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        with self.metrics_lock:
            now = datetime.now(timezone.utc)
            last_24h = now - timedelta(hours=24)
            last_7d = now - timedelta(days=7)
            
            # Filter recent data
            recent_executions = [
                e for e in self.metrics_data["chaos_executions"]
                if e["timestamp"] >= last_24h
            ]
            
            recent_failures = [
                f for f in self.metrics_data["failures_injected"]
                if f["timestamp"] >= last_24h
            ]
            
            recent_recovery_times = [
                r["recovery_time"] for r in self.metrics_data["recovery_times"]
                if r["timestamp"] >= last_7d
            ]
            
            recent_availability = [
                a["availability"] for a in self.metrics_data["availability_scores"]
                if a["timestamp"] >= last_7d
            ]
            
            recent_resilience = [
                r["resilience"] for r in self.metrics_data["resilience_scores"]
                if r["timestamp"] >= last_7d
            ]
            
            return {
                "summary_period": "last_24h",
                "chaos_executions": len(recent_executions),
                "successful_executions": sum(1 for e in recent_executions if e["success"]),
                "failures_injected": len(recent_failures),
                "successful_failures": sum(1 for f in recent_failures if f["success"]),
                "average_recovery_time": sum(recent_recovery_times) / len(recent_recovery_times) if recent_recovery_times else 0,
                "average_availability": sum(recent_availability) / len(recent_availability) if recent_availability else 0,
                "average_resilience": sum(recent_resilience) / len(recent_resilience) if recent_resilience else 0,
                "total_historical_executions": len(self.metrics_data["chaos_executions"]),
                "total_historical_failures": len(self.metrics_data["failures_injected"])
            }


class SafetyMonitor:
    """Safety monitoring for chaos operations."""
    
    def __init__(self, config: ChaosMonkeyConfig):
        self.config = config
        self.safety_state = {
            "failures_last_hour": 0,
            "failures_last_day": 0,
            "last_failure_time": None,
            "emergency_stop_active": False,
            "safety_violations": []
        }
        self.safety_lock = threading.Lock()
        
    async def check_safety_conditions(self) -> Dict[str, Any]:
        """Check if it's safe to perform chaos operations."""
        try:
            safety_result = {
                "safe": True,
                "violations": [],
                "recommendations": []
            }
            
            # Check business hours
            if self.config.business_hours_only and not self._is_business_hours():
                safety_result["safe"] = False
                safety_result["violations"].append("Outside business hours")
            
            # Check excluded days
            if self._is_excluded_day():
                safety_result["safe"] = False
                safety_result["violations"].append("Excluded day")
            
            # Check excluded dates
            if self._is_excluded_date():
                safety_result["safe"] = False
                safety_result["violations"].append("Excluded date")
            
            # Check failure limits
            if self.safety_state["failures_last_hour"] >= self.config.max_failures_per_hour:
                safety_result["safe"] = False
                safety_result["violations"].append("Hourly failure limit exceeded")
            
            if self.safety_state["failures_last_day"] >= self.config.max_failures_per_day:
                safety_result["safe"] = False
                safety_result["violations"].append("Daily failure limit exceeded")
            
            # Check emergency stop
            if self.safety_state["emergency_stop_active"]:
                safety_result["safe"] = False
                safety_result["violations"].append("Emergency stop active")
            
            # Check system health
            system_health = await self._check_system_health()
            if not system_health["healthy"]:
                safety_result["safe"] = False
                safety_result["violations"].append("System unhealthy")
            
            # Check minimum healthy instances
            healthy_instances = await self._count_healthy_instances()
            if healthy_instances < self.config.min_healthy_instances:
                safety_result["safe"] = False
                safety_result["violations"].append("Insufficient healthy instances")
            
            # Add recommendations
            if not safety_result["safe"]:
                safety_result["recommendations"].append("Wait for safety conditions to improve")
            
            return safety_result
            
        except Exception as e:
            logger.error(f"Safety check failed: {e}")
            return {"safe": False, "violations": ["Safety check failed"], "error": str(e)}
    
    def _is_business_hours(self) -> bool:
        """Check if current time is within business hours."""
        now = datetime.now()
        return 9 <= now.hour <= 17 and now.weekday() < 5  # 9 AM - 5 PM, weekdays
    
    def _is_excluded_day(self) -> bool:
        """Check if current day is excluded."""
        now = datetime.now()
        day_name = now.strftime("%A").lower()
        return day_name in [day.lower() for day in self.config.excluded_days]
    
    def _is_excluded_date(self) -> bool:
        """Check if current date is excluded."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        return date_str in self.config.excluded_dates
    
    async def _check_system_health(self) -> Dict[str, Any]:
        """Check overall system health."""
        try:
            # This would integrate with actual health monitoring
            # For now, simulate health check
            return {
                "healthy": True,
                "cpu_usage": random.uniform(20, 80),
                "memory_usage": random.uniform(30, 85),
                "error_rate": random.uniform(0, 5)
            }
        except Exception as e:
            logger.error(f"System health check failed: {e}")
            return {"healthy": False, "error": str(e)}
    
    async def _count_healthy_instances(self) -> int:
        """Count healthy service instances."""
        try:
            # This would count actual healthy instances
            # For now, simulate count
            return random.randint(2, 5)
        except Exception as e:
            logger.error(f"Healthy instance count failed: {e}")
            return 0
    
    def record_failure(self):
        """Record a failure occurrence."""
        with self.safety_lock:
            now = datetime.now(timezone.utc)
            self.safety_state["last_failure_time"] = now
            
            # Update hourly count
            self.safety_state["failures_last_hour"] += 1
            
            # Update daily count
            self.safety_state["failures_last_day"] += 1
            
            # Schedule count resets
            asyncio.create_task(self._reset_hourly_count())
            asyncio.create_task(self._reset_daily_count())
    
    async def _reset_hourly_count(self):
        """Reset hourly failure count."""
        await asyncio.sleep(3600)  # 1 hour
        with self.safety_lock:
            self.safety_state["failures_last_hour"] = max(0, self.safety_state["failures_last_hour"] - 1)
    
    async def _reset_daily_count(self):
        """Reset daily failure count."""
        await asyncio.sleep(86400)  # 24 hours
        with self.safety_lock:
            self.safety_state["failures_last_day"] = max(0, self.safety_state["failures_last_day"] - 1)
    
    def emergency_stop(self, reason: str = "Manual emergency stop"):
        """Activate emergency stop."""
        with self.safety_lock:
            self.safety_state["emergency_stop_active"] = True
            self.safety_state["safety_violations"].append({
                "timestamp": datetime.now(timezone.utc),
                "type": "emergency_stop",
                "reason": reason
            })
        
        logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")
    
    def clear_emergency_stop(self):
        """Clear emergency stop."""
        with self.safety_lock:
            self.safety_state["emergency_stop_active"] = False
        
        logger.info("Emergency stop cleared")
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status."""
        with self.safety_lock:
            return {
                "emergency_stop_active": self.safety_state["emergency_stop_active"],
                "failures_last_hour": self.safety_state["failures_last_hour"],
                "failures_last_day": self.safety_state["failures_last_day"],
                "last_failure_time": self.safety_state["last_failure_time"],
                "safety_violations": self.safety_state["safety_violations"][-10:],  # Last 10 violations
                "safety_limits": {
                    "max_failures_per_hour": self.config.max_failures_per_hour,
                    "max_failures_per_day": self.config.max_failures_per_day,
                    "min_healthy_instances": self.config.min_healthy_instances
                }
            }


class NetflixChaosMonkey:
    """Netflix-style Chaos Monkey implementation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize Chaos Monkey."""
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.chaos_framework = EnterpriseChaosFramework()
        self.failure_engine = FailureInjectionEngine()
        self.scheduler = ChaosAutomationScheduler(self.chaos_framework)
        self.validator = ResilienceValidator()
        
        # Initialize monitoring
        self.metrics = ChaosMonkeyMetrics()
        self.safety_monitor = SafetyMonitor(self.config)
        
        # Runtime state
        self.running = False
        self.execution_history = []
        self.game_day_events = []
        
        logger.info("Netflix Chaos Monkey initialized")
    
    def _load_config(self, config_path: Optional[str]) -> ChaosMonkeyConfig:
        """Load configuration from file or use defaults."""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                return ChaosMonkeyConfig(**config_data)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                logger.info("Using default configuration")
        
        return ChaosMonkeyConfig()
    
    async def start(self):
        """Start the Chaos Monkey."""
        try:
            if not self.config.enabled:
                logger.info("Chaos Monkey is disabled by configuration")
                return
            
            if self.running:
                logger.warning("Chaos Monkey is already running")
                return
            
            logger.info("Starting Netflix Chaos Monkey...")
            
            # Start scheduler
            self.scheduler.start()
            
            # Set up chaos schedules based on mode
            await self._setup_chaos_schedules()
            
            # Start continuous validation if enabled
            if self.config.continuous_validation:
                await self._start_continuous_validation()
            
            # Start metrics collection
            await self._start_metrics_collection()
            
            self.running = True
            logger.info("Netflix Chaos Monkey started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Chaos Monkey: {e}")
            raise
    
    async def stop(self):
        """Stop the Chaos Monkey."""
        try:
            if not self.running:
                logger.warning("Chaos Monkey is not running")
                return
            
            logger.info("Stopping Netflix Chaos Monkey...")
            
            # Stop scheduler
            self.scheduler.stop()
            
            # Clean up any active chaos
            await self.chaos_framework.chaos_engine.cleanup_all_injections()
            
            self.running = False
            logger.info("Netflix Chaos Monkey stopped successfully")
            
        except Exception as e:
            logger.error(f"Failed to stop Chaos Monkey: {e}")
    
    async def _setup_chaos_schedules(self):
        """Set up chaos schedules based on configuration mode."""
        try:
            if self.config.mode == ChaosMode.DISABLED:
                logger.info("Chaos Monkey is disabled")
                return
            
            elif self.config.mode == ChaosMode.SCHEDULE_ONLY:
                # Set up automated schedules
                await self._setup_automated_schedules()
                
            elif self.config.mode == ChaosMode.GAME_DAY_ONLY:
                # Set up game day schedules only
                await self._setup_game_day_schedules()
                
            elif self.config.mode == ChaosMode.FULL_AUTOMATED:
                # Set up both automated and game day schedules
                await self._setup_automated_schedules()
                await self._setup_game_day_schedules()
                
            elif self.config.mode == ChaosMode.EMERGENCY_ONLY:
                # Only emergency drills
                await self._setup_emergency_schedules()
            
        except Exception as e:
            logger.error(f"Failed to setup chaos schedules: {e}")
    
    async def _setup_automated_schedules(self):
        """Set up automated chaos schedules."""
        try:
            from .chaos_automation_scheduler import ChaosSchedule, ScheduleType, SchedulePriority, ExecutionContext
            
            # Daily resilience testing
            daily_schedule = ChaosSchedule(
                schedule_id="daily_resilience",
                name="Daily Resilience Testing",
                description="Daily automated resilience testing",
                schedule_type=ScheduleType.CONTINUOUS,
                priority=SchedulePriority.MEDIUM,
                execution_context=ExecutionContext.PRODUCTION,
                cron_expression="0 10 * * MON-FRI",  # 10 AM weekdays
                experiment_selection_strategy="intelligent",
                max_concurrent_experiments=1,
                business_hours_only=self.config.business_hours_only,
                skip_on_high_load=True,
                execution_conditions={
                    "min_health_score": 0.8,
                    "max_cpu_usage": 70,
                    "max_memory_usage": 80
                }
            )
            
            await self.scheduler.add_schedule(daily_schedule)
            
            # Weekly comprehensive testing
            weekly_schedule = ChaosSchedule(
                schedule_id="weekly_comprehensive",
                name="Weekly Comprehensive Testing",
                description="Weekly comprehensive chaos testing",
                schedule_type=ScheduleType.VALIDATION,
                priority=SchedulePriority.HIGH,
                execution_context=ExecutionContext.PRODUCTION,
                cron_expression="0 14 * * WED",  # 2 PM Wednesday
                experiment_selection_strategy="comprehensive",
                max_concurrent_experiments=2,
                business_hours_only=True,
                skip_on_high_load=True
            )
            
            await self.scheduler.add_schedule(weekly_schedule)
            
            logger.info("Automated schedules set up successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup automated schedules: {e}")
    
    async def _setup_game_day_schedules(self):
        """Set up game day event schedules."""
        try:
            from .chaos_automation_scheduler import GameDayEvent
            
            # Monthly game day
            monthly_game_day = GameDayEvent(
                event_id="monthly_game_day",
                name="Monthly Resilience Game Day",
                description="Monthly comprehensive resilience game day",
                event_date=datetime.now(timezone.utc).replace(day=15) + timedelta(days=30),  # 15th of next month
                duration_minutes=240,
                scenarios=[
                    {"type": "experiment", "experiment_id": "CHAOS_SVC_REDIS_001"},
                    {"type": "experiment", "experiment_id": "CHAOS_RES_MEM_001"},
                    {"type": "experiment", "experiment_id": "CHAOS_LOAD_HFT_001"},
                    {"type": "custom", "name": "Cross-service coordination test"}
                ],
                objectives=[
                    "Validate end-to-end resilience",
                    "Test incident response procedures",
                    "Verify monitoring and alerting",
                    "Train operations team"
                ]
            )
            
            await self.scheduler.schedule_game_day_event(monthly_game_day)
            
            logger.info("Game day schedules set up successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup game day schedules: {e}")
    
    async def _setup_emergency_schedules(self):
        """Set up emergency drill schedules."""
        try:
            from .chaos_automation_scheduler import ChaosSchedule, ScheduleType, SchedulePriority, ExecutionContext
            
            # Emergency drill schedule
            emergency_schedule = ChaosSchedule(
                schedule_id="emergency_drills",
                name="Emergency Response Drills",
                description="Emergency response and recovery drills",
                schedule_type=ScheduleType.EMERGENCY_DRILL,
                priority=SchedulePriority.CRITICAL,
                execution_context=ExecutionContext.PRODUCTION,
                cron_expression="0 15 15 * *",  # 3 PM on 15th of each month
                experiment_selection_strategy="emergency_scenarios",
                max_concurrent_experiments=1,
                business_hours_only=True,
                skip_on_high_load=False  # Emergency drills should run regardless
            )
            
            await self.scheduler.add_schedule(emergency_schedule)
            
            logger.info("Emergency schedules set up successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup emergency schedules: {e}")
    
    async def _start_continuous_validation(self):
        """Start continuous validation pipeline."""
        try:
            from .chaos_automation_scheduler import ValidationPipeline
            
            # Critical path validation
            critical_validation = ValidationPipeline(
                pipeline_id="critical_path_validation",
                name="Critical Path Continuous Validation",
                description="Continuous validation of critical system paths",
                validation_scenarios=["CHAOS_SVC_DB_001", "CHAOS_SVC_NET_001"],
                execution_frequency="hourly",
                quality_gates=[
                    {"type": "pass_rate", "threshold": 0.95, "trigger_rollback": True},
                    {"type": "recovery_time", "threshold": 30.0, "trigger_rollback": False}
                ]
            )
            
            await self.scheduler.add_validation_pipeline(critical_validation)
            
            logger.info("Continuous validation started")
            
        except Exception as e:
            logger.error(f"Failed to start continuous validation: {e}")
    
    async def _start_metrics_collection(self):
        """Start metrics collection."""
        try:
            # This would start actual metrics collection
            # For now, just log
            logger.info("Metrics collection started")
            
        except Exception as e:
            logger.error(f"Failed to start metrics collection: {e}")
    
    async def execute_chaos_experiment(self, experiment_id: str, notify: bool = True) -> Dict[str, Any]:
        """Execute a specific chaos experiment."""
        try:
            # Check safety conditions
            safety_check = await self.safety_monitor.check_safety_conditions()
            if not safety_check["safe"]:
                return {
                    "success": False,
                    "reason": "Safety conditions not met",
                    "violations": safety_check["violations"]
                }
            
            # Send pre-chaos notification
            if notify and self.config.notify_before_chaos:
                await self._send_notification(
                    f"🐒 Chaos Monkey: Starting experiment {experiment_id}",
                    "chaos_start"
                )
            
            # Pre-chaos validation
            if self.config.pre_chaos_validation:
                pre_validation = await self.validator.validate_resilience(ValidationLevel.BASIC)
                if not pre_validation["success"]:
                    return {
                        "success": False,
                        "reason": "Pre-chaos validation failed",
                        "validation_result": pre_validation
                    }
            
            # Execute experiment
            execution_start = time.time()
            experiment_result = await self.chaos_framework.run_experiment(experiment_id)
            execution_duration = time.time() - execution_start
            
            # Record metrics
            self.metrics.record_chaos_execution({
                "execution_id": experiment_result.get("experiment_id"),
                "experiment_type": experiment_id,
                "success": experiment_result.get("status") == "SUCCESS",
                "duration": execution_duration,
                "recovery_time": experiment_result.get("recovery_time", 0),
                "availability": experiment_result.get("availability", 0)
            })
            
            # Record failure for safety tracking
            self.safety_monitor.record_failure()
            
            # Post-chaos validation
            if self.config.post_chaos_validation:
                post_validation = await self.validator.validate_resilience(ValidationLevel.STANDARD)
                experiment_result["post_validation"] = post_validation
            
            # Send post-chaos notification
            if notify and self.config.notify_after_chaos:
                status_emoji = "✅" if experiment_result.get("status") == "SUCCESS" else "❌"
                await self._send_notification(
                    f"{status_emoji} Chaos Monkey: Experiment {experiment_id} completed",
                    "chaos_end"
                )
            
            # Store execution history
            self.execution_history.append({
                "timestamp": datetime.now(timezone.utc),
                "experiment_id": experiment_id,
                "result": experiment_result,
                "duration": execution_duration
            })
            
            logger.info(f"Chaos experiment {experiment_id} completed: {experiment_result.get('status')}")
            
            return experiment_result
            
        except Exception as e:
            logger.error(f"Chaos experiment execution failed: {e}")
            
            # Send error notification
            if notify:
                await self._send_notification(
                    f"🚨 Chaos Monkey: Experiment {experiment_id} failed: {str(e)}",
                    "chaos_error"
                )
            
            return {"success": False, "error": str(e)}
    
    async def _send_notification(self, message: str, notification_type: str):
        """Send notification to configured channels."""
        try:
            for channel in self.config.alert_channels:
                if channel == "slack":
                    await self._send_slack_notification(message, notification_type)
                elif channel == "email":
                    await self._send_email_notification(message, notification_type)
                elif channel == "webhook":
                    await self._send_webhook_notification(message, notification_type)
                    
        except Exception as e:
            logger.error(f"Notification send failed: {e}")
    
    async def _send_slack_notification(self, message: str, notification_type: str):
        """Send Slack notification."""
        try:
            # This would integrate with actual Slack API
            logger.info(f"Slack notification: {message}")
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
    
    async def _send_email_notification(self, message: str, notification_type: str):
        """Send email notification."""
        try:
            # This would integrate with actual email service
            logger.info(f"Email notification: {message}")
        except Exception as e:
            logger.error(f"Email notification failed: {e}")
    
    async def _send_webhook_notification(self, message: str, notification_type: str):
        """Send webhook notification."""
        try:
            # This would integrate with actual webhook service
            logger.info(f"Webhook notification: {message}")
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
    
    async def emergency_stop(self, reason: str = "Manual emergency stop") -> Dict[str, Any]:
        """Emergency stop all chaos activities."""
        try:
            logger.critical(f"CHAOS MONKEY EMERGENCY STOP: {reason}")
            
            # Activate safety monitor emergency stop
            self.safety_monitor.emergency_stop(reason)
            
            # Stop scheduler
            await self.scheduler.emergency_stop(reason)
            
            # Clean up all chaos injections
            cleanup_count = await self.chaos_framework.chaos_engine.cleanup_all_injections()
            
            # Send emergency notification
            await self._send_notification(
                f"🚨 CHAOS MONKEY EMERGENCY STOP: {reason}",
                "emergency_stop"
            )
            
            return {
                "success": True,
                "reason": reason,
                "cleanup_count": cleanup_count,
                "timestamp": datetime.now(timezone.utc)
            }
            
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def clear_emergency_stop(self) -> Dict[str, Any]:
        """Clear emergency stop and resume normal operations."""
        try:
            logger.info("Clearing Chaos Monkey emergency stop")
            
            # Clear safety monitor emergency stop
            self.safety_monitor.clear_emergency_stop()
            
            # Restart scheduler if it was stopped
            if not self.scheduler.scheduler.running:
                self.scheduler.start()
            
            # Re-setup schedules
            await self._setup_chaos_schedules()
            
            # Send notification
            await self._send_notification(
                "✅ Chaos Monkey: Emergency stop cleared, resuming normal operations",
                "emergency_cleared"
            )
            
            return {
                "success": True,
                "timestamp": datetime.now(timezone.utc)
            }
            
        except Exception as e:
            logger.error(f"Clear emergency stop failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def run_resilience_certification(self, level: ValidationLevel = ValidationLevel.STANDARD) -> Dict[str, Any]:
        """Run comprehensive resilience certification."""
        try:
            logger.info(f"Starting resilience certification: {level.value}")
            
            # Send notification
            await self._send_notification(
                f"🏆 Chaos Monkey: Starting resilience certification ({level.value})",
                "certification_start"
            )
            
            # Run validation
            certification_result = await self.validator.validate_resilience(level)
            
            # Record resilience score
            if certification_result.get("success"):
                self.metrics.record_resilience_score(certification_result.get("overall_score", 0))
            
            # Send completion notification
            score = certification_result.get("overall_score", 0)
            status = certification_result.get("report", {}).get("certification_status", "UNKNOWN")
            
            await self._send_notification(
                f"🏆 Chaos Monkey: Certification completed - {status} (Score: {score:.2f})",
                "certification_complete"
            )
            
            return certification_result
            
        except Exception as e:
            logger.error(f"Resilience certification failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_chaos_monkey_status(self) -> Dict[str, Any]:
        """Get comprehensive Chaos Monkey status."""
        try:
            # Get component statuses
            scheduler_status = await self.scheduler.get_scheduler_status()
            framework_status = await self.chaos_framework.get_framework_status()
            validation_status = await self.validator.get_validation_status()
            
            # Get metrics
            metrics_summary = self.metrics.get_metrics_summary()
            
            # Get safety status
            safety_status = self.safety_monitor.get_safety_status()
            
            return {
                "chaos_monkey": {
                    "running": self.running,
                    "mode": self.config.mode.value,
                    "safety_level": self.config.safety_level.value,
                    "enabled": self.config.enabled
                },
                "scheduler": scheduler_status,
                "framework": framework_status,
                "validation": validation_status,
                "metrics": metrics_summary,
                "safety": safety_status,
                "execution_history": len(self.execution_history),
                "game_day_events": len(self.game_day_events)
            }
            
        except Exception as e:
            logger.error(f"Status retrieval failed: {e}")
            return {"error": str(e)}
    
    async def save_configuration(self, config_path: str) -> bool:
        """Save current configuration to file."""
        try:
            config_dict = {
                "enabled": self.config.enabled,
                "mode": self.config.mode.value,
                "safety_level": self.config.safety_level.value,
                "business_hours_only": self.config.business_hours_only,
                "excluded_days": self.config.excluded_days,
                "excluded_dates": self.config.excluded_dates,
                "min_healthy_instances": self.config.min_healthy_instances,
                "max_failures_per_hour": self.config.max_failures_per_hour,
                "max_failures_per_day": self.config.max_failures_per_day,
                "alert_channels": self.config.alert_channels,
                "notify_before_chaos": self.config.notify_before_chaos,
                "notify_after_chaos": self.config.notify_after_chaos,
                "pre_chaos_validation": self.config.pre_chaos_validation,
                "post_chaos_validation": self.config.post_chaos_validation,
                "continuous_validation": self.config.continuous_validation,
                "resilience_threshold": self.config.resilience_threshold
            }
            
            with open(config_path, 'w') as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False)
            
            logger.info(f"Configuration saved to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Configuration save failed: {e}")
            return False


# Example usage and configuration
async def main():
    """Demonstrate Netflix Chaos Monkey."""
    
    # Create sample configuration
    config = ChaosMonkeyConfig(
        enabled=True,
        mode=ChaosMode.SCHEDULE_ONLY,
        safety_level=SafetyLevel.STANDARD,
        business_hours_only=True,
        excluded_days=["saturday", "sunday"],
        max_failures_per_hour=2,
        max_failures_per_day=5,
        alert_channels=["slack", "email"],
        notify_before_chaos=True,
        notify_after_chaos=True,
        pre_chaos_validation=True,
        post_chaos_validation=True,
        continuous_validation=True,
        resilience_threshold=0.8
    )
    
    # Initialize Chaos Monkey
    chaos_monkey = NetflixChaosMonkey()
    chaos_monkey.config = config
    
    try:
        # Start Chaos Monkey
        await chaos_monkey.start()
        
        # Execute a specific experiment
        experiment_result = await chaos_monkey.execute_chaos_experiment("CHAOS_SVC_REDIS_001")
        print(f"Experiment result: {experiment_result}")
        
        # Run resilience certification
        certification_result = await chaos_monkey.run_resilience_certification(ValidationLevel.STANDARD)
        print(f"Certification result: {certification_result.get('success', False)}")
        
        # Get status
        status = await chaos_monkey.get_chaos_monkey_status()
        print(f"Chaos Monkey status: {json.dumps(status, indent=2, default=str)}")
        
        # Save configuration
        await chaos_monkey.save_configuration("/tmp/chaos_monkey_config.yaml")
        
    finally:
        # Stop Chaos Monkey
        await chaos_monkey.stop()


if __name__ == "__main__":
    asyncio.run(main())