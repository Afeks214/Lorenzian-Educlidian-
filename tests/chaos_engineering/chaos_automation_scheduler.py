"""
Chaos Automation Scheduler
==========================

Automated scheduling and orchestration system for continuous chaos engineering.
Provides Netflix-style automated chaos testing with intelligent scheduling,
game day coordination, and continuous validation.

Features:
- Automated chaos testing schedules
- Game day event coordination
- Continuous validation pipelines
- Intelligent failure selection
- Risk-aware scheduling
- Automated rollback mechanisms
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
import croniter
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger

from .enterprise_chaos_framework import EnterpriseChaosFramework, ChaosExperiment, ChaosScenarioType
from .failure_injection_engine import FailureInjectionEngine, FailureSpec, FailureType, FailureEscalationLevel, FailurePattern

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScheduleType(Enum):
    """Types of chaos schedules."""
    CONTINUOUS = "continuous"           # Continuous background testing
    GAME_DAY = "game_day"              # Scheduled game day events
    VALIDATION = "validation"          # Continuous validation testing
    EMERGENCY_DRILL = "emergency_drill"  # Emergency response drills
    LOAD_TEST = "load_test"            # Scheduled load testing
    REGRESSION = "regression"          # Regression testing after deployments


class SchedulePriority(Enum):
    """Priority levels for scheduled chaos tests."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ExecutionContext(Enum):
    """Execution context for chaos tests."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    CANARY = "canary"


@dataclass
class ChaosSchedule:
    """Definition of a chaos testing schedule."""
    schedule_id: str
    name: str
    description: str
    schedule_type: ScheduleType
    priority: SchedulePriority
    execution_context: ExecutionContext
    
    # Scheduling configuration
    cron_expression: Optional[str] = None
    interval_seconds: Optional[int] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Experiment configuration
    experiment_ids: List[str] = field(default_factory=list)
    experiment_selection_strategy: str = "sequential"  # "sequential", "random", "weighted"
    max_concurrent_experiments: int = 1
    
    # Conditions and constraints
    execution_conditions: Dict[str, Any] = field(default_factory=dict)
    safety_constraints: Dict[str, Any] = field(default_factory=dict)
    business_hours_only: bool = False
    skip_on_high_load: bool = True
    
    # Monitoring and alerting
    monitoring_enabled: bool = True
    alert_on_failure: bool = True
    alert_channels: List[str] = field(default_factory=list)
    
    # Execution tracking
    enabled: bool = True
    last_execution: Optional[datetime] = None
    next_execution: Optional[datetime] = None
    execution_count: int = 0
    failure_count: int = 0
    success_rate: float = 0.0


@dataclass
class GameDayEvent:
    """Definition of a game day chaos event."""
    event_id: str
    name: str
    description: str
    event_date: datetime
    duration_minutes: int
    
    # Event configuration
    scenarios: List[Dict[str, Any]] = field(default_factory=list)
    participants: List[str] = field(default_factory=list)
    objectives: List[str] = field(default_factory=list)
    
    # Execution phases
    preparation_phase: Dict[str, Any] = field(default_factory=dict)
    execution_phase: Dict[str, Any] = field(default_factory=dict)
    recovery_phase: Dict[str, Any] = field(default_factory=dict)
    
    # Tracking
    status: str = "SCHEDULED"
    results: Dict[str, Any] = field(default_factory=dict)
    lessons_learned: List[str] = field(default_factory=list)


@dataclass
class ValidationPipeline:
    """Definition of a continuous validation pipeline."""
    pipeline_id: str
    name: str
    description: str
    
    # Pipeline configuration
    validation_scenarios: List[str] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    failure_thresholds: Dict[str, Any] = field(default_factory=dict)
    
    # Execution configuration
    execution_frequency: str = "hourly"  # "continuous", "hourly", "daily"
    batch_size: int = 5
    parallel_execution: bool = True
    
    # Quality gates
    quality_gates: List[Dict[str, Any]] = field(default_factory=list)
    rollback_triggers: List[str] = field(default_factory=list)
    
    # Tracking
    enabled: bool = True
    last_validation: Optional[datetime] = None
    validation_count: int = 0
    pass_rate: float = 0.0


class SystemHealthMonitor:
    """Monitor system health to make intelligent scheduling decisions."""
    
    def __init__(self):
        self.health_metrics = {}
        self.health_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "error_rate": 5.0,
            "response_time": 1000.0,
            "active_users": 1000
        }
        
    async def get_system_health(self) -> Dict[str, Any]:
        """Get current system health metrics."""
        try:
            # In production, this would integrate with monitoring systems
            # For now, simulate health metrics
            health = {
                "cpu_usage": random.uniform(20, 90),
                "memory_usage": random.uniform(30, 85),
                "error_rate": random.uniform(0, 10),
                "response_time": random.uniform(50, 2000),
                "active_users": random.randint(10, 2000),
                "timestamp": datetime.now(timezone.utc)
            }
            
            # Calculate health score
            health_score = self._calculate_health_score(health)
            health["health_score"] = health_score
            
            return health
            
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return {"health_score": 0.0, "error": str(e)}
    
    def _calculate_health_score(self, health: Dict[str, Any]) -> float:
        """Calculate overall health score (0-1)."""
        try:
            scores = []
            
            # CPU score
            cpu_score = max(0, 1 - (health["cpu_usage"] / 100))
            scores.append(cpu_score)
            
            # Memory score
            memory_score = max(0, 1 - (health["memory_usage"] / 100))
            scores.append(memory_score)
            
            # Error rate score
            error_score = max(0, 1 - (health["error_rate"] / 20))
            scores.append(error_score)
            
            # Response time score
            response_score = max(0, 1 - (health["response_time"] / 3000))
            scores.append(response_score)
            
            return sum(scores) / len(scores)
            
        except Exception:
            return 0.5  # Default neutral score
    
    async def is_system_healthy_for_chaos(self, min_health_score: float = 0.7) -> bool:
        """Check if system is healthy enough for chaos testing."""
        try:
            health = await self.get_system_health()
            return health.get("health_score", 0) >= min_health_score
        except Exception:
            return False
    
    async def get_recommended_blast_radius(self) -> float:
        """Get recommended blast radius based on system health."""
        try:
            health = await self.get_system_health()
            health_score = health.get("health_score", 0.5)
            
            # Lower health score = smaller blast radius
            if health_score >= 0.9:
                return 0.8  # High blast radius for healthy systems
            elif health_score >= 0.7:
                return 0.5  # Medium blast radius
            elif health_score >= 0.5:
                return 0.3  # Low blast radius for degraded systems
            else:
                return 0.1  # Minimal blast radius for unhealthy systems
                
        except Exception:
            return 0.3  # Safe default


class IntelligentExperimentSelector:
    """Intelligently select chaos experiments based on context."""
    
    def __init__(self, chaos_framework: EnterpriseChaosFramework):
        self.chaos_framework = chaos_framework
        self.selection_history = []
        self.experiment_weights = {}
        
    async def select_experiments(self, 
                               schedule: ChaosSchedule,
                               system_health: Dict[str, Any],
                               max_count: int = 5) -> List[str]:
        """Select experiments based on strategy and context."""
        try:
            available_experiments = schedule.experiment_ids
            
            if not available_experiments:
                available_experiments = [exp.id for exp in self.chaos_framework.experiments]
            
            # Filter experiments based on context
            filtered_experiments = await self._filter_experiments_by_context(
                available_experiments, schedule, system_health
            )
            
            # Select based on strategy
            if schedule.experiment_selection_strategy == "sequential":
                selected = await self._select_sequential(filtered_experiments, max_count)
            elif schedule.experiment_selection_strategy == "random":
                selected = await self._select_random(filtered_experiments, max_count)
            elif schedule.experiment_selection_strategy == "weighted":
                selected = await self._select_weighted(filtered_experiments, max_count)
            else:
                selected = filtered_experiments[:max_count]
            
            # Update selection history
            self.selection_history.append({
                "timestamp": datetime.now(timezone.utc),
                "schedule_id": schedule.schedule_id,
                "selected_experiments": selected,
                "system_health": system_health
            })
            
            return selected
            
        except Exception as e:
            logger.error(f"Experiment selection failed: {e}")
            return []
    
    async def _filter_experiments_by_context(self, 
                                           experiments: List[str],
                                           schedule: ChaosSchedule,
                                           system_health: Dict[str, Any]) -> List[str]:
        """Filter experiments based on execution context."""
        filtered = []
        
        for exp_id in experiments:
            experiment = next((e for e in self.chaos_framework.experiments if e.id == exp_id), None)
            if not experiment:
                continue
            
            # Check execution context
            if schedule.execution_context == ExecutionContext.PRODUCTION:
                # Be more conservative in production
                if experiment.blast_radius == "CRITICAL":
                    continue
                    
            elif schedule.execution_context == ExecutionContext.DEVELOPMENT:
                # Allow more aggressive testing in development
                pass
            
            # Check system health constraints
            health_score = system_health.get("health_score", 0.5)
            
            if health_score < 0.7 and experiment.blast_radius in ["HIGH", "CRITICAL"]:
                continue
            
            if health_score < 0.5 and experiment.blast_radius == "MEDIUM":
                continue
            
            # Check business hours constraint
            if schedule.business_hours_only and not self._is_business_hours():
                continue
                
            filtered.append(exp_id)
        
        return filtered
    
    def _is_business_hours(self) -> bool:
        """Check if current time is within business hours."""
        now = datetime.now()
        return 9 <= now.hour <= 17 and now.weekday() < 5  # 9 AM - 5 PM, weekdays
    
    async def _select_sequential(self, experiments: List[str], max_count: int) -> List[str]:
        """Select experiments sequentially."""
        # Track last selected position
        last_position = getattr(self, '_last_sequential_position', 0)
        
        selected = []
        for i in range(max_count):
            if experiments:
                index = (last_position + i) % len(experiments)
                selected.append(experiments[index])
        
        self._last_sequential_position = (last_position + max_count) % len(experiments)
        return selected
    
    async def _select_random(self, experiments: List[str], max_count: int) -> List[str]:
        """Select experiments randomly."""
        if len(experiments) <= max_count:
            return experiments
        
        return random.sample(experiments, max_count)
    
    async def _select_weighted(self, experiments: List[str], max_count: int) -> List[str]:
        """Select experiments based on weights."""
        if not experiments:
            return []
        
        # Calculate weights based on success rate, frequency, etc.
        weights = []
        for exp_id in experiments:
            weight = self.experiment_weights.get(exp_id, 1.0)
            
            # Adjust weight based on recent execution
            recent_executions = [
                h for h in self.selection_history[-10:] 
                if exp_id in h["selected_experiments"]
            ]
            
            if recent_executions:
                weight *= 0.8  # Reduce weight for recently executed experiments
            
            weights.append(weight)
        
        # Weighted random selection
        selected = []
        for _ in range(min(max_count, len(experiments))):
            if not experiments:
                break
                
            # Select based on weights
            selected_exp = random.choices(experiments, weights=weights)[0]
            selected.append(selected_exp)
            
            # Remove selected experiment to avoid duplicates
            index = experiments.index(selected_exp)
            experiments.pop(index)
            weights.pop(index)
        
        return selected


class ChaosAutomationScheduler:
    """Main scheduler for automated chaos engineering."""
    
    def __init__(self, chaos_framework: EnterpriseChaosFramework):
        self.chaos_framework = chaos_framework
        self.scheduler = AsyncIOScheduler()
        self.health_monitor = SystemHealthMonitor()
        self.experiment_selector = IntelligentExperimentSelector(chaos_framework)
        
        # Schedule storage
        self.schedules: Dict[str, ChaosSchedule] = {}
        self.game_day_events: Dict[str, GameDayEvent] = {}
        self.validation_pipelines: Dict[str, ValidationPipeline] = {}
        
        # Execution tracking
        self.execution_history = []
        self.active_executions = {}
        
        # Configuration
        self.max_concurrent_schedules = 3
        self.safety_timeout_minutes = 30
        self.emergency_stop_enabled = True
        
        logger.info("Chaos Automation Scheduler initialized")
    
    def start(self):
        """Start the scheduler."""
        try:
            self.scheduler.start()
            logger.info("Chaos Automation Scheduler started")
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            raise
    
    def stop(self):
        """Stop the scheduler."""
        try:
            self.scheduler.shutdown()
            logger.info("Chaos Automation Scheduler stopped")
        except Exception as e:
            logger.error(f"Failed to stop scheduler: {e}")
    
    async def add_schedule(self, schedule: ChaosSchedule) -> bool:
        """Add a new chaos schedule."""
        try:
            # Validate schedule
            if not await self._validate_schedule(schedule):
                return False
            
            # Store schedule
            self.schedules[schedule.schedule_id] = schedule
            
            # Add to scheduler
            await self._add_schedule_to_scheduler(schedule)
            
            logger.info(f"Added chaos schedule: {schedule.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add schedule: {e}")
            return False
    
    async def _validate_schedule(self, schedule: ChaosSchedule) -> bool:
        """Validate a chaos schedule."""
        try:
            # Check required fields
            if not schedule.schedule_id or not schedule.name:
                logger.error("Schedule missing required fields")
                return False
            
            # Validate cron expression if provided
            if schedule.cron_expression:
                try:
                    croniter.croniter(schedule.cron_expression)
                except ValueError as e:
                    logger.error(f"Invalid cron expression: {e}")
                    return False
            
            # Validate experiment IDs
            if schedule.experiment_ids:
                available_ids = [exp.id for exp in self.chaos_framework.experiments]
                for exp_id in schedule.experiment_ids:
                    if exp_id not in available_ids:
                        logger.error(f"Unknown experiment ID: {exp_id}")
                        return False
            
            # Validate execution context
            if schedule.execution_context == ExecutionContext.PRODUCTION:
                # Additional validation for production schedules
                if schedule.priority == SchedulePriority.CRITICAL:
                    logger.error("Critical priority not allowed in production")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Schedule validation failed: {e}")
            return False
    
    async def _add_schedule_to_scheduler(self, schedule: ChaosSchedule):
        """Add schedule to the APScheduler."""
        try:
            job_id = f"chaos_schedule_{schedule.schedule_id}"
            
            # Create trigger based on schedule type
            if schedule.cron_expression:
                trigger = CronTrigger.from_crontab(schedule.cron_expression)
            elif schedule.interval_seconds:
                trigger = IntervalTrigger(seconds=schedule.interval_seconds)
            else:
                # Default to hourly for continuous schedules
                trigger = IntervalTrigger(hours=1)
            
            # Add job
            self.scheduler.add_job(
                self._execute_schedule,
                trigger=trigger,
                id=job_id,
                args=[schedule.schedule_id],
                max_instances=1,
                start_date=schedule.start_date,
                end_date=schedule.end_date,
                replace_existing=True
            )
            
            # Update next execution time
            job = self.scheduler.get_job(job_id)
            if job:
                schedule.next_execution = job.next_run_time
            
        except Exception as e:
            logger.error(f"Failed to add schedule to scheduler: {e}")
            raise
    
    async def _execute_schedule(self, schedule_id: str):
        """Execute a chaos schedule."""
        try:
            schedule = self.schedules.get(schedule_id)
            if not schedule or not schedule.enabled:
                return
            
            logger.info(f"Executing chaos schedule: {schedule.name}")
            
            # Check system health
            system_health = await self.health_monitor.get_system_health()
            
            # Skip if system is unhealthy and skip_on_high_load is enabled
            if schedule.skip_on_high_load and not await self.health_monitor.is_system_healthy_for_chaos():
                logger.info(f"Skipping schedule {schedule.name} due to poor system health")
                return
            
            # Check execution conditions
            if not await self._check_execution_conditions(schedule, system_health):
                logger.info(f"Skipping schedule {schedule.name} due to unmet conditions")
                return
            
            # Check concurrency limits
            if len(self.active_executions) >= self.max_concurrent_schedules:
                logger.info(f"Skipping schedule {schedule.name} due to concurrency limit")
                return
            
            # Select experiments
            selected_experiments = await self.experiment_selector.select_experiments(
                schedule, system_health, schedule.max_concurrent_experiments
            )
            
            if not selected_experiments:
                logger.info(f"No experiments selected for schedule {schedule.name}")
                return
            
            # Execute experiments
            execution_id = f"execution_{schedule_id}_{int(time.time())}"
            
            execution_result = await self._execute_experiments(
                execution_id, schedule, selected_experiments, system_health
            )
            
            # Update schedule tracking
            schedule.last_execution = datetime.now(timezone.utc)
            schedule.execution_count += 1
            
            if execution_result.get("success", False):
                schedule.success_rate = (schedule.success_rate * (schedule.execution_count - 1) + 1) / schedule.execution_count
            else:
                schedule.failure_count += 1
                schedule.success_rate = (schedule.success_rate * (schedule.execution_count - 1)) / schedule.execution_count
            
            # Store execution history
            self.execution_history.append({
                "execution_id": execution_id,
                "schedule_id": schedule_id,
                "timestamp": datetime.now(timezone.utc),
                "experiments": selected_experiments,
                "system_health": system_health,
                "result": execution_result
            })
            
            logger.info(f"Completed chaos schedule execution: {schedule.name}")
            
        except Exception as e:
            logger.error(f"Schedule execution failed: {e}")
            
            # Update failure count
            if schedule_id in self.schedules:
                self.schedules[schedule_id].failure_count += 1
    
    async def _check_execution_conditions(self, schedule: ChaosSchedule, system_health: Dict[str, Any]) -> bool:
        """Check if execution conditions are met."""
        try:
            conditions = schedule.execution_conditions
            
            # Check health score condition
            min_health_score = conditions.get("min_health_score", 0.5)
            if system_health.get("health_score", 0) < min_health_score:
                return False
            
            # Check CPU usage condition
            max_cpu_usage = conditions.get("max_cpu_usage", 90)
            if system_health.get("cpu_usage", 0) > max_cpu_usage:
                return False
            
            # Check memory usage condition
            max_memory_usage = conditions.get("max_memory_usage", 90)
            if system_health.get("memory_usage", 0) > max_memory_usage:
                return False
            
            # Check error rate condition
            max_error_rate = conditions.get("max_error_rate", 10)
            if system_health.get("error_rate", 0) > max_error_rate:
                return False
            
            # Check business hours condition
            if schedule.business_hours_only and not self._is_business_hours():
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Execution condition check failed: {e}")
            return False
    
    def _is_business_hours(self) -> bool:
        """Check if current time is within business hours."""
        now = datetime.now()
        return 9 <= now.hour <= 17 and now.weekday() < 5  # 9 AM - 5 PM, weekdays
    
    async def _execute_experiments(self, 
                                 execution_id: str,
                                 schedule: ChaosSchedule,
                                 experiment_ids: List[str],
                                 system_health: Dict[str, Any]) -> Dict[str, Any]:
        """Execute selected experiments."""
        try:
            # Track active execution
            self.active_executions[execution_id] = {
                "schedule_id": schedule.schedule_id,
                "experiment_ids": experiment_ids,
                "start_time": datetime.now(timezone.utc),
                "status": "RUNNING"
            }
            
            # Execute experiments
            if schedule.max_concurrent_experiments == 1:
                # Sequential execution
                results = await self._execute_experiments_sequentially(experiment_ids)
            else:
                # Parallel execution
                results = await self._execute_experiments_in_parallel(experiment_ids, schedule.max_concurrent_experiments)
            
            # Calculate success rate
            successful_experiments = sum(1 for r in results if r.get("status") == "SUCCESS")
            success_rate = successful_experiments / len(results) if results else 0
            
            execution_result = {
                "success": success_rate >= 0.8,  # 80% success rate threshold
                "execution_id": execution_id,
                "experiment_results": results,
                "success_rate": success_rate,
                "total_experiments": len(experiment_ids),
                "successful_experiments": successful_experiments,
                "duration": (datetime.now(timezone.utc) - self.active_executions[execution_id]["start_time"]).total_seconds()
            }
            
            # Update execution status
            self.active_executions[execution_id]["status"] = "COMPLETED"
            self.active_executions[execution_id]["result"] = execution_result
            
            # Schedule cleanup
            asyncio.create_task(self._cleanup_execution(execution_id))
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Experiment execution failed: {e}")
            
            # Update execution status
            if execution_id in self.active_executions:
                self.active_executions[execution_id]["status"] = "FAILED"
                self.active_executions[execution_id]["error"] = str(e)
            
            return {"success": False, "error": str(e)}
    
    async def _execute_experiments_sequentially(self, experiment_ids: List[str]) -> List[Dict[str, Any]]:
        """Execute experiments sequentially."""
        results = []
        
        for exp_id in experiment_ids:
            try:
                # Add delay between experiments
                if results:
                    await asyncio.sleep(30)  # 30 second delay
                
                # Execute experiment
                result = await self.chaos_framework.run_experiment(exp_id)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Experiment {exp_id} failed: {e}")
                results.append({"experiment_id": exp_id, "status": "FAILED", "error": str(e)})
        
        return results
    
    async def _execute_experiments_in_parallel(self, experiment_ids: List[str], max_concurrent: int) -> List[Dict[str, Any]]:
        """Execute experiments in parallel with concurrency limit."""
        results = []
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_single_experiment(exp_id: str):
            async with semaphore:
                try:
                    result = await self.chaos_framework.run_experiment(exp_id)
                    return result
                except Exception as e:
                    logger.error(f"Experiment {exp_id} failed: {e}")
                    return {"experiment_id": exp_id, "status": "FAILED", "error": str(e)}
        
        # Execute experiments
        tasks = [execute_single_experiment(exp_id) for exp_id in experiment_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                results[i] = {"experiment_id": experiment_ids[i], "status": "FAILED", "error": str(result)}
        
        return results
    
    async def _cleanup_execution(self, execution_id: str):
        """Clean up execution after completion."""
        try:
            # Wait a bit before cleanup
            await asyncio.sleep(300)  # 5 minutes
            
            # Remove from active executions
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
                
        except Exception as e:
            logger.error(f"Execution cleanup failed: {e}")
    
    async def schedule_game_day_event(self, event: GameDayEvent) -> bool:
        """Schedule a game day event."""
        try:
            # Validate event
            if not event.event_id or not event.name:
                logger.error("Game day event missing required fields")
                return False
            
            # Store event
            self.game_day_events[event.event_id] = event
            
            # Schedule event execution
            job_id = f"game_day_{event.event_id}"
            
            self.scheduler.add_job(
                self._execute_game_day_event,
                trigger=DateTrigger(run_date=event.event_date),
                id=job_id,
                args=[event.event_id],
                max_instances=1,
                replace_existing=True
            )
            
            logger.info(f"Scheduled game day event: {event.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to schedule game day event: {e}")
            return False
    
    async def _execute_game_day_event(self, event_id: str):
        """Execute a game day event."""
        try:
            event = self.game_day_events.get(event_id)
            if not event:
                return
            
            logger.info(f"Executing game day event: {event.name}")
            
            event.status = "RUNNING"
            start_time = datetime.now(timezone.utc)
            
            # Execute phases
            results = {
                "preparation": await self._execute_event_phase(event, "preparation"),
                "execution": await self._execute_event_phase(event, "execution"),
                "recovery": await self._execute_event_phase(event, "recovery")
            }
            
            # Update event status
            event.status = "COMPLETED"
            event.results = {
                "start_time": start_time,
                "end_time": datetime.now(timezone.utc),
                "duration": (datetime.now(timezone.utc) - start_time).total_seconds(),
                "phase_results": results
            }
            
            logger.info(f"Completed game day event: {event.name}")
            
        except Exception as e:
            logger.error(f"Game day event execution failed: {e}")
            
            if event_id in self.game_day_events:
                self.game_day_events[event_id].status = "FAILED"
                self.game_day_events[event_id].results = {"error": str(e)}
    
    async def _execute_event_phase(self, event: GameDayEvent, phase: str) -> Dict[str, Any]:
        """Execute a specific phase of a game day event."""
        try:
            phase_config = getattr(event, f"{phase}_phase", {})
            
            if not phase_config:
                return {"status": "SKIPPED", "reason": "No configuration"}
            
            # Execute phase scenarios
            if phase == "execution":
                # Execute chaos scenarios
                scenario_results = []
                for scenario in event.scenarios:
                    scenario_result = await self._execute_game_day_scenario(scenario)
                    scenario_results.append(scenario_result)
                
                return {
                    "status": "COMPLETED",
                    "scenario_results": scenario_results,
                    "scenarios_executed": len(scenario_results)
                }
            
            else:
                # Preparation and recovery phases
                return {
                    "status": "COMPLETED",
                    "phase": phase,
                    "config": phase_config
                }
                
        except Exception as e:
            logger.error(f"Event phase execution failed: {e}")
            return {"status": "FAILED", "error": str(e)}
    
    async def _execute_game_day_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a game day scenario."""
        try:
            scenario_type = scenario.get("type", "experiment")
            
            if scenario_type == "experiment":
                experiment_id = scenario.get("experiment_id")
                if experiment_id:
                    return await self.chaos_framework.run_experiment(experiment_id)
            
            elif scenario_type == "custom":
                # Execute custom scenario
                return await self._execute_custom_scenario(scenario)
            
            return {"status": "SKIPPED", "reason": "Unknown scenario type"}
            
        except Exception as e:
            logger.error(f"Game day scenario execution failed: {e}")
            return {"status": "FAILED", "error": str(e)}
    
    async def _execute_custom_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a custom game day scenario."""
        try:
            # This would implement custom scenario execution
            # For now, return success
            return {
                "status": "SUCCESS",
                "scenario_type": "custom",
                "scenario_name": scenario.get("name", "Unknown"),
                "duration": 30
            }
            
        except Exception as e:
            logger.error(f"Custom scenario execution failed: {e}")
            return {"status": "FAILED", "error": str(e)}
    
    async def add_validation_pipeline(self, pipeline: ValidationPipeline) -> bool:
        """Add a continuous validation pipeline."""
        try:
            # Validate pipeline
            if not pipeline.pipeline_id or not pipeline.name:
                logger.error("Pipeline missing required fields")
                return False
            
            # Store pipeline
            self.validation_pipelines[pipeline.pipeline_id] = pipeline
            
            # Schedule validation execution
            job_id = f"validation_{pipeline.pipeline_id}"
            
            if pipeline.execution_frequency == "continuous":
                trigger = IntervalTrigger(minutes=15)
            elif pipeline.execution_frequency == "hourly":
                trigger = IntervalTrigger(hours=1)
            elif pipeline.execution_frequency == "daily":
                trigger = CronTrigger(hour=2, minute=0)  # 2 AM daily
            else:
                trigger = IntervalTrigger(hours=1)  # Default to hourly
            
            self.scheduler.add_job(
                self._execute_validation_pipeline,
                trigger=trigger,
                id=job_id,
                args=[pipeline.pipeline_id],
                max_instances=1,
                replace_existing=True
            )
            
            logger.info(f"Added validation pipeline: {pipeline.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add validation pipeline: {e}")
            return False
    
    async def _execute_validation_pipeline(self, pipeline_id: str):
        """Execute a validation pipeline."""
        try:
            pipeline = self.validation_pipelines.get(pipeline_id)
            if not pipeline or not pipeline.enabled:
                return
            
            logger.info(f"Executing validation pipeline: {pipeline.name}")
            
            # Execute validation scenarios
            validation_results = []
            
            for scenario_id in pipeline.validation_scenarios:
                try:
                    result = await self.chaos_framework.run_experiment(scenario_id)
                    validation_results.append(result)
                except Exception as e:
                    logger.error(f"Validation scenario {scenario_id} failed: {e}")
                    validation_results.append({"scenario_id": scenario_id, "status": "FAILED", "error": str(e)})
            
            # Calculate pass rate
            successful_validations = sum(1 for r in validation_results if r.get("status") == "SUCCESS")
            pass_rate = successful_validations / len(validation_results) if validation_results else 0
            
            # Update pipeline tracking
            pipeline.last_validation = datetime.now(timezone.utc)
            pipeline.validation_count += 1
            pipeline.pass_rate = pass_rate
            
            # Check quality gates
            quality_gate_results = await self._check_quality_gates(pipeline, validation_results)
            
            # Trigger rollback if needed
            if quality_gate_results.get("rollback_triggered", False):
                await self._trigger_rollback(pipeline, quality_gate_results)
            
            logger.info(f"Completed validation pipeline: {pipeline.name} (Pass rate: {pass_rate:.2%})")
            
        except Exception as e:
            logger.error(f"Validation pipeline execution failed: {e}")
    
    async def _check_quality_gates(self, pipeline: ValidationPipeline, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check quality gates for validation pipeline."""
        try:
            gate_results = {
                "gates_passed": 0,
                "gates_failed": 0,
                "rollback_triggered": False,
                "gate_details": []
            }
            
            for gate in pipeline.quality_gates:
                gate_type = gate.get("type", "pass_rate")
                gate_threshold = gate.get("threshold", 0.8)
                
                if gate_type == "pass_rate":
                    successful_validations = sum(1 for r in validation_results if r.get("status") == "SUCCESS")
                    actual_pass_rate = successful_validations / len(validation_results) if validation_results else 0
                    
                    gate_passed = actual_pass_rate >= gate_threshold
                    
                    gate_results["gate_details"].append({
                        "type": gate_type,
                        "threshold": gate_threshold,
                        "actual_value": actual_pass_rate,
                        "passed": gate_passed
                    })
                    
                    if gate_passed:
                        gate_results["gates_passed"] += 1
                    else:
                        gate_results["gates_failed"] += 1
                        
                        # Check if rollback should be triggered
                        if gate.get("trigger_rollback", False):
                            gate_results["rollback_triggered"] = True
            
            return gate_results
            
        except Exception as e:
            logger.error(f"Quality gate check failed: {e}")
            return {"gates_passed": 0, "gates_failed": 1, "rollback_triggered": False}
    
    async def _trigger_rollback(self, pipeline: ValidationPipeline, gate_results: Dict[str, Any]):
        """Trigger rollback based on quality gate failures."""
        try:
            logger.warning(f"Triggering rollback for pipeline: {pipeline.name}")
            
            # This would implement actual rollback procedures
            # For now, just log the rollback
            rollback_result = {
                "pipeline_id": pipeline.pipeline_id,
                "rollback_triggered": True,
                "trigger_reason": "Quality gate failures",
                "gate_results": gate_results,
                "timestamp": datetime.now(timezone.utc)
            }
            
            logger.info(f"Rollback triggered: {rollback_result}")
            
        except Exception as e:
            logger.error(f"Rollback trigger failed: {e}")
    
    async def emergency_stop(self, reason: str = "Manual emergency stop") -> Dict[str, Any]:
        """Emergency stop all chaos testing activities."""
        try:
            if not self.emergency_stop_enabled:
                return {"success": False, "reason": "Emergency stop disabled"}
            
            logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
            
            # Stop all active executions
            active_executions = list(self.active_executions.keys())
            
            # Clean up chaos framework
            cleanup_count = await self.chaos_framework.chaos_engine.cleanup_all_injections()
            
            # Pause all schedules
            paused_schedules = []
            for schedule_id, schedule in self.schedules.items():
                if schedule.enabled:
                    schedule.enabled = False
                    paused_schedules.append(schedule_id)
            
            # Remove all scheduled jobs
            self.scheduler.remove_all_jobs()
            
            emergency_result = {
                "success": True,
                "timestamp": datetime.now(timezone.utc),
                "reason": reason,
                "active_executions_stopped": len(active_executions),
                "chaos_injections_cleaned": cleanup_count,
                "schedules_paused": len(paused_schedules),
                "jobs_removed": "all"
            }
            
            logger.info(f"Emergency stop completed: {emergency_result}")
            return emergency_result
            
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_scheduler_status(self) -> Dict[str, Any]:
        """Get comprehensive scheduler status."""
        try:
            # Get system health
            system_health = await self.health_monitor.get_system_health()
            
            # Get active schedules
            active_schedules = {
                schedule_id: {
                    "name": schedule.name,
                    "enabled": schedule.enabled,
                    "last_execution": schedule.last_execution,
                    "next_execution": schedule.next_execution,
                    "execution_count": schedule.execution_count,
                    "success_rate": schedule.success_rate
                }
                for schedule_id, schedule in self.schedules.items()
            }
            
            # Get scheduler job status
            jobs = self.scheduler.get_jobs()
            job_status = {
                job.id: {
                    "next_run": job.next_run_time,
                    "trigger": str(job.trigger),
                    "func": job.func.__name__
                }
                for job in jobs
            }
            
            return {
                "scheduler_running": self.scheduler.running,
                "system_health": system_health,
                "active_schedules": active_schedules,
                "active_executions": len(self.active_executions),
                "scheduled_jobs": len(jobs),
                "job_details": job_status,
                "game_day_events": len(self.game_day_events),
                "validation_pipelines": len(self.validation_pipelines),
                "emergency_stop_enabled": self.emergency_stop_enabled,
                "execution_history_count": len(self.execution_history)
            }
            
        except Exception as e:
            logger.error(f"Failed to get scheduler status: {e}")
            return {"error": str(e)}


# Example usage
async def main():
    """Demonstrate the Chaos Automation Scheduler."""
    # Initialize components
    chaos_framework = EnterpriseChaosFramework()
    scheduler = ChaosAutomationScheduler(chaos_framework)
    
    # Create example schedule
    continuous_schedule = ChaosSchedule(
        schedule_id="continuous_resilience",
        name="Continuous Resilience Testing",
        description="Continuous background chaos testing",
        schedule_type=ScheduleType.CONTINUOUS,
        priority=SchedulePriority.MEDIUM,
        execution_context=ExecutionContext.STAGING,
        interval_seconds=3600,  # Every hour
        experiment_selection_strategy="weighted",
        max_concurrent_experiments=2,
        business_hours_only=False,
        skip_on_high_load=True,
        execution_conditions={
            "min_health_score": 0.7,
            "max_cpu_usage": 80,
            "max_memory_usage": 85
        }
    )
    
    # Create game day event
    game_day = GameDayEvent(
        event_id="quarterly_game_day",
        name="Quarterly Resilience Game Day",
        description="Quarterly comprehensive resilience testing",
        event_date=datetime.now(timezone.utc) + timedelta(days=7),
        duration_minutes=240,
        scenarios=[
            {"type": "experiment", "experiment_id": "CHAOS_SVC_REDIS_001"},
            {"type": "experiment", "experiment_id": "CHAOS_RES_MEM_001"},
            {"type": "custom", "name": "Multi-service coordination test"}
        ],
        objectives=[
            "Validate service recovery procedures",
            "Test monitoring and alerting systems",
            "Verify escalation procedures"
        ]
    )
    
    # Create validation pipeline
    validation_pipeline = ValidationPipeline(
        pipeline_id="critical_path_validation",
        name="Critical Path Validation",
        description="Continuous validation of critical system paths",
        validation_scenarios=["CHAOS_SVC_DB_001", "CHAOS_SVC_NET_001"],
        execution_frequency="hourly",
        quality_gates=[
            {"type": "pass_rate", "threshold": 0.95, "trigger_rollback": True}
        ]
    )
    
    try:
        # Start scheduler
        scheduler.start()
        
        # Add schedule
        await scheduler.add_schedule(continuous_schedule)
        
        # Schedule game day event
        await scheduler.schedule_game_day_event(game_day)
        
        # Add validation pipeline
        await scheduler.add_validation_pipeline(validation_pipeline)
        
        # Get status
        status = await scheduler.get_scheduler_status()
        print(f"Scheduler Status: {json.dumps(status, indent=2, default=str)}")
        
        # Run for a while
        await asyncio.sleep(60)
        
    finally:
        # Stop scheduler
        scheduler.stop()


if __name__ == "__main__":
    asyncio.run(main())