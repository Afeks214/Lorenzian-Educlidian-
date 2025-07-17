#!/usr/bin/env python3
"""
AGENT 7: Trading Decision Logger
Comprehensive logging system for trading decisions, performance attribution, and audit trails.
"""

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import asyncio
import threading
from contextlib import contextmanager
from datetime import timedelta

from src.monitoring.structured_logging import (
    StructuredLogger, 
    correlation_context, 
    LogContext, 
    LogComponent,
    get_logger
)
from src.core.errors.base_exceptions import BaseGrandModelError


class TradingDecisionType(Enum):
    """Types of trading decisions to log."""
    SIGNAL_GENERATION = "signal_generation"
    POSITION_SIZING = "position_sizing"
    RISK_ASSESSMENT = "risk_assessment"
    EXECUTION_DECISION = "execution_decision"
    STOP_LOSS_PLACEMENT = "stop_loss_placement"
    TAKE_PROFIT_PLACEMENT = "take_profit_placement"
    PORTFOLIO_REBALANCING = "portfolio_rebalancing"
    AGENT_COORDINATION = "agent_coordination"
    REGIME_DETECTION = "regime_detection"
    EMERGENCY_ACTION = "emergency_action"


class TradingDecisionOutcome(Enum):
    """Outcomes of trading decisions."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL_SUCCESS = "partial_success"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    PENDING = "pending"


@dataclass
class TradingDecisionMetrics:
    """Metrics for trading decision performance."""
    decision_latency_ms: float
    confidence_score: float
    risk_score: float
    expected_return: float
    expected_risk: float
    sharpe_ratio: Optional[float] = None
    kelly_fraction: Optional[float] = None
    var_impact: Optional[float] = None
    correlation_impact: Optional[float] = None


@dataclass
class TradingDecisionContext:
    """Context information for trading decisions."""
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    decision_type: TradingDecisionType = TradingDecisionType.SIGNAL_GENERATION
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    agent_id: str = "unknown"
    strategy_id: str = "unknown"
    symbol: Optional[str] = None
    market_regime: Optional[str] = None
    portfolio_value: Optional[float] = None
    position_size: Optional[float] = None
    price: Optional[float] = None
    market_data: Dict[str, Any] = field(default_factory=dict)
    agent_states: Dict[str, Any] = field(default_factory=dict)
    external_signals: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingDecisionRecord:
    """Complete record of a trading decision."""
    context: TradingDecisionContext
    decision_logic: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    metrics: TradingDecisionMetrics
    outcome: TradingDecisionOutcome
    error_details: Optional[Dict[str, Any]] = None
    system_state: Optional[Dict[str, Any]] = None
    performance_attribution: Optional[Dict[str, Any]] = None
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'context': asdict(self.context),
            'decision_logic': self.decision_logic,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'metrics': asdict(self.metrics),
            'outcome': self.outcome.value,
            'error_details': self.error_details,
            'system_state': self.system_state,
            'performance_attribution': self.performance_attribution,
            'audit_trail': self.audit_trail,
            'correlation_id': self.correlation_id
        }


class TradingDecisionLogger:
    """
    Comprehensive trading decision logger with audit trail and performance attribution.
    """
    
    def __init__(self, logger_name: str = "trading_decisions"):
        self.logger = get_logger(logger_name)
        self.decision_history: List[TradingDecisionRecord] = []
        self.decision_index: Dict[str, TradingDecisionRecord] = {}
        self.performance_tracker: Dict[str, List[float]] = {}
        
        # Thread-safe decision tracking
        self._lock = threading.RLock()
        
        # Performance attribution settings
        self.track_performance_attribution = True
        self.attribution_window_hours = 24
        
        # Audit trail settings
        self.max_audit_trail_entries = 100
        self.max_decision_history = 10000
        
        # Metrics tracking
        self.decision_counters = {
            decision_type: 0 for decision_type in TradingDecisionType
        }
        self.outcome_counters = {
            outcome: 0 for outcome in TradingDecisionOutcome
        }
        
        self.logger.info("TradingDecisionLogger initialized")
    
    @contextmanager
    def decision_context(
        self, 
        decision_type: TradingDecisionType,
        agent_id: str,
        strategy_id: str,
        **kwargs
    ):
        """Context manager for tracking trading decisions."""
        context = TradingDecisionContext(
            decision_type=decision_type,
            agent_id=agent_id,
            strategy_id=strategy_id,
            **kwargs
        )
        
        # Set correlation context for logging
        with correlation_context.context(
            correlation_id=context.decision_id,
            component=LogComponent.TRADING_ENGINE,
            strategy_id=strategy_id
        ):
            decision_record = TradingDecisionRecord(
                context=context,
                decision_logic="",
                inputs={},
                outputs={},
                metrics=TradingDecisionMetrics(0, 0, 0, 0, 0),
                outcome=TradingDecisionOutcome.PENDING
            )
            
            start_time = time.time()
            
            try:
                yield DecisionTracker(decision_record, self)
            except Exception as e:
                decision_record.outcome = TradingDecisionOutcome.FAILURE
                decision_record.error_details = {
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'error_context': getattr(e, 'context', None) if hasattr(e, 'context') else None
                }
                
                # Log error with context
                self.logger.error(
                    f"Trading decision failed: {decision_type.value}",
                    decision_id=context.decision_id,
                    agent_id=agent_id,
                    strategy_id=strategy_id,
                    error=str(e),
                    exc_info=True
                )
                
                raise
            finally:
                # Update metrics
                decision_record.metrics.decision_latency_ms = (time.time() - start_time) * 1000
                
                # Record decision
                self._record_decision(decision_record)
    
    def _record_decision(self, decision_record: TradingDecisionRecord):
        """Record a trading decision."""
        with self._lock:
            # Add to history
            self.decision_history.append(decision_record)
            
            # Maintain history size
            if len(self.decision_history) > self.max_decision_history:
                old_record = self.decision_history.pop(0)
                self.decision_index.pop(old_record.context.decision_id, None)
            
            # Index by decision ID
            self.decision_index[decision_record.context.decision_id] = decision_record
            
            # Update counters
            self.decision_counters[decision_record.context.decision_type] += 1
            self.outcome_counters[decision_record.outcome] += 1
            
            # Log decision
            self._log_decision(decision_record)
            
            # Update performance attribution if enabled
            if self.track_performance_attribution:
                self._update_performance_attribution(decision_record)
    
    def _log_decision(self, decision_record: TradingDecisionRecord):
        """Log trading decision with structured format."""
        context = decision_record.context
        
        # Create structured log entry
        log_data = {
            'decision_type': context.decision_type.value,
            'decision_id': context.decision_id,
            'agent_id': context.agent_id,
            'strategy_id': context.strategy_id,
            'symbol': context.symbol,
            'outcome': decision_record.outcome.value,
            'decision_logic': decision_record.decision_logic,
            'metrics': asdict(decision_record.metrics),
            'inputs': decision_record.inputs,
            'outputs': decision_record.outputs
        }
        
        # Log based on outcome
        if decision_record.outcome == TradingDecisionOutcome.SUCCESS:
            self.logger.info(
                f"Trading decision successful: {context.decision_type.value}",
                extra_fields=log_data
            )
        elif decision_record.outcome == TradingDecisionOutcome.FAILURE:
            self.logger.error(
                f"Trading decision failed: {context.decision_type.value}",
                extra_fields=log_data,
                error_details=decision_record.error_details
            )
        else:
            self.logger.warning(
                f"Trading decision outcome: {decision_record.outcome.value}",
                extra_fields=log_data
            )
    
    def _update_performance_attribution(self, decision_record: TradingDecisionRecord):
        """Update performance attribution tracking."""
        context = decision_record.context
        
        # Track by agent and strategy
        agent_key = f"{context.agent_id}:{context.strategy_id}"
        
        if agent_key not in self.performance_tracker:
            self.performance_tracker[agent_key] = []
        
        # Calculate performance score
        score = self._calculate_performance_score(decision_record)
        
        self.performance_tracker[agent_key].append(score)
        
        # Maintain window size
        max_entries = int(self.attribution_window_hours * 60 / 5)  # Assuming 5-minute decisions
        if len(self.performance_tracker[agent_key]) > max_entries:
            self.performance_tracker[agent_key] = self.performance_tracker[agent_key][-max_entries:]
    
    def _calculate_performance_score(self, decision_record: TradingDecisionRecord) -> float:
        """Calculate performance score for a decision."""
        metrics = decision_record.metrics
        
        # Base score from outcome
        outcome_scores = {
            TradingDecisionOutcome.SUCCESS: 1.0,
            TradingDecisionOutcome.PARTIAL_SUCCESS: 0.5,
            TradingDecisionOutcome.FAILURE: -1.0,
            TradingDecisionOutcome.TIMEOUT: -0.5,
            TradingDecisionOutcome.CANCELLED: 0.0,
            TradingDecisionOutcome.PENDING: 0.0
        }
        
        base_score = outcome_scores.get(decision_record.outcome, 0.0)
        
        # Adjust for metrics
        confidence_adjustment = (metrics.confidence_score - 0.5) * 0.5
        risk_adjustment = (0.5 - metrics.risk_score) * 0.3
        
        # Speed bonus/penalty
        speed_adjustment = max(0, (1000 - metrics.decision_latency_ms) / 1000) * 0.2
        
        return base_score + confidence_adjustment + risk_adjustment + speed_adjustment
    
    def get_decision_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of trading decisions in the last N hours."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        recent_decisions = [
            record for record in self.decision_history
            if record.context.timestamp >= cutoff_time
        ]
        
        if not recent_decisions:
            return {"message": "No recent decisions found"}
        
        # Aggregate statistics
        total_decisions = len(recent_decisions)
        outcome_counts = {}
        type_counts = {}
        
        for record in recent_decisions:
            outcome = record.outcome.value
            decision_type = record.context.decision_type.value
            
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            type_counts[decision_type] = type_counts.get(decision_type, 0) + 1
        
        # Calculate success rate
        success_count = outcome_counts.get('success', 0)
        success_rate = success_count / total_decisions if total_decisions > 0 else 0
        
        # Performance metrics
        avg_latency = sum(r.metrics.decision_latency_ms for r in recent_decisions) / total_decisions
        avg_confidence = sum(r.metrics.confidence_score for r in recent_decisions) / total_decisions
        avg_risk = sum(r.metrics.risk_score for r in recent_decisions) / total_decisions
        
        return {
            "time_period_hours": hours,
            "total_decisions": total_decisions,
            "success_rate": success_rate,
            "outcome_distribution": outcome_counts,
            "decision_type_distribution": type_counts,
            "average_metrics": {
                "decision_latency_ms": avg_latency,
                "confidence_score": avg_confidence,
                "risk_score": avg_risk
            }
        }
    
    def get_performance_attribution(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance attribution for agents and strategies."""
        attribution = {}
        
        for agent_key, scores in self.performance_tracker.items():
            if not scores:
                continue
            
            # Calculate recent performance
            recent_scores = scores[-int(hours * 12):]  # Last N hours (5-min intervals)
            
            attribution[agent_key] = {
                "average_score": sum(recent_scores) / len(recent_scores),
                "total_decisions": len(recent_scores),
                "score_trend": self._calculate_trend(recent_scores),
                "volatility": self._calculate_volatility(recent_scores),
                "best_score": max(recent_scores) if recent_scores else 0,
                "worst_score": min(recent_scores) if recent_scores else 0
            }
        
        return attribution
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """Calculate trend direction for performance scores."""
        if len(scores) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        n = len(scores)
        x = list(range(n))
        y = scores
        
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "flat"
        
        slope = numerator / denominator
        
        if slope > 0.1:
            return "improving"
        elif slope < -0.1:
            return "declining"
        else:
            return "stable"
    
    def _calculate_volatility(self, scores: List[float]) -> float:
        """Calculate volatility of performance scores."""
        if len(scores) < 2:
            return 0.0
        
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        
        return variance ** 0.5
    
    def get_audit_trail(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Get complete audit trail for a specific decision."""
        decision_record = self.decision_index.get(decision_id)
        
        if not decision_record:
            return None
        
        return {
            "decision_id": decision_id,
            "audit_trail": decision_record.audit_trail,
            "decision_record": decision_record.to_dict(),
            "related_decisions": self._find_related_decisions(decision_record)
        }
    
    def _find_related_decisions(self, decision_record: TradingDecisionRecord) -> List[str]:
        """Find decisions related to the given decision."""
        related = []
        context = decision_record.context
        
        # Find decisions with same symbol and strategy within time window
        time_window = timedelta(minutes=30)
        
        for record in self.decision_history:
            if record.context.decision_id == context.decision_id:
                continue
            
            # Check if related
            if (record.context.symbol == context.symbol and
                record.context.strategy_id == context.strategy_id and
                abs((record.context.timestamp - context.timestamp).total_seconds()) < time_window.total_seconds()):
                related.append(record.context.decision_id)
        
        return related
    
    def log_system_state_snapshot(self, state: Dict[str, Any]):
        """Log system state snapshot for debugging."""
        self.logger.info(
            "System state snapshot",
            extra_fields={
                "snapshot_type": "system_state",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "state": state
            }
        )
    
    def get_decision_statistics(self) -> Dict[str, Any]:
        """Get comprehensive decision statistics."""
        with self._lock:
            return {
                "total_decisions": len(self.decision_history),
                "decision_type_counts": dict(self.decision_counters),
                "outcome_counts": dict(self.outcome_counters),
                "performance_tracker_size": len(self.performance_tracker),
                "average_decision_latency": self._calculate_average_latency(),
                "top_performing_agents": self._get_top_performers(),
                "recent_error_patterns": self._analyze_error_patterns()
            }
    
    def _calculate_average_latency(self) -> float:
        """Calculate average decision latency."""
        if not self.decision_history:
            return 0.0
        
        total_latency = sum(record.metrics.decision_latency_ms for record in self.decision_history)
        return total_latency / len(self.decision_history)
    
    def _get_top_performers(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top performing agents/strategies."""
        performance_data = []
        
        for agent_key, scores in self.performance_tracker.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                performance_data.append({
                    "agent_key": agent_key,
                    "average_score": avg_score,
                    "total_decisions": len(scores)
                })
        
        return sorted(performance_data, key=lambda x: x["average_score"], reverse=True)[:limit]
    
    def _analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze recent error patterns."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        
        error_records = [
            record for record in self.decision_history
            if (record.context.timestamp >= cutoff_time and 
                record.outcome == TradingDecisionOutcome.FAILURE and
                record.error_details)
        ]
        
        if not error_records:
            return {"message": "No recent errors found"}
        
        # Group by error type
        error_types = {}
        for record in error_records:
            error_type = record.error_details.get('error_type', 'unknown')
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(record)
        
        # Analyze patterns
        patterns = {}
        for error_type, records in error_types.items():
            patterns[error_type] = {
                "count": len(records),
                "affected_agents": list(set(r.context.agent_id for r in records)),
                "affected_strategies": list(set(r.context.strategy_id for r in records)),
                "common_symbols": list(set(r.context.symbol for r in records if r.context.symbol)),
                "average_latency": sum(r.metrics.decision_latency_ms for r in records) / len(records)
            }
        
        return patterns


class DecisionTracker:
    """Helper class for tracking decision details within context."""
    
    def __init__(self, decision_record: TradingDecisionRecord, logger: TradingDecisionLogger):
        self.decision_record = decision_record
        self.logger = logger
        self.audit_entries = []
    
    def set_decision_logic(self, logic: str):
        """Set the decision logic description."""
        self.decision_record.decision_logic = logic
        self.add_audit_entry("decision_logic_set", {"logic": logic})
    
    def set_inputs(self, inputs: Dict[str, Any]):
        """Set decision inputs."""
        self.decision_record.inputs = inputs
        self.add_audit_entry("inputs_set", {"input_keys": list(inputs.keys())})
    
    def set_outputs(self, outputs: Dict[str, Any]):
        """Set decision outputs."""
        self.decision_record.outputs = outputs
        self.add_audit_entry("outputs_set", {"output_keys": list(outputs.keys())})
    
    def update_metrics(self, **metrics):
        """Update decision metrics."""
        for key, value in metrics.items():
            if hasattr(self.decision_record.metrics, key):
                setattr(self.decision_record.metrics, key, value)
        
        self.add_audit_entry("metrics_updated", {"metrics": metrics})
    
    def set_outcome(self, outcome: TradingDecisionOutcome):
        """Set decision outcome."""
        self.decision_record.outcome = outcome
        self.add_audit_entry("outcome_set", {"outcome": outcome.value})
    
    def add_audit_entry(self, action: str, details: Dict[str, Any]):
        """Add entry to audit trail."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "details": details
        }
        
        self.audit_entries.append(entry)
        self.decision_record.audit_trail.append(entry)
        
        # Maintain audit trail size
        if len(self.decision_record.audit_trail) > self.logger.max_audit_trail_entries:
            self.decision_record.audit_trail = self.decision_record.audit_trail[-self.logger.max_audit_trail_entries:]
    
    def log_intermediate_step(self, step_name: str, data: Dict[str, Any]):
        """Log intermediate decision step."""
        self.logger.logger.debug(
            f"Decision step: {step_name}",
            extra_fields={
                "decision_id": self.decision_record.context.decision_id,
                "step_name": step_name,
                "step_data": data
            }
        )
        
        self.add_audit_entry("intermediate_step", {"step_name": step_name, "data": data})
    
    def capture_system_state(self, state: Dict[str, Any]):
        """Capture system state for debugging."""
        self.decision_record.system_state = state
        self.add_audit_entry("system_state_captured", {"state_keys": list(state.keys())})


# Global trading decision logger instance
_trading_decision_logger = None


def get_trading_decision_logger() -> TradingDecisionLogger:
    """Get global trading decision logger instance."""
    global _trading_decision_logger
    if _trading_decision_logger is None:
        _trading_decision_logger = TradingDecisionLogger()
    return _trading_decision_logger


def log_trading_decision(
    decision_type: TradingDecisionType,
    agent_id: str,
    strategy_id: str,
    **kwargs
):
    """Decorator for logging trading decisions."""
    def decorator(func):
        def wrapper(*args, **func_kwargs):
            logger = get_trading_decision_logger()
            
            with logger.decision_context(
                decision_type=decision_type,
                agent_id=agent_id,
                strategy_id=strategy_id,
                **kwargs
            ) as tracker:
                # Set decision logic from function name and docstring
                logic = f"{func.__name__}: {func.__doc__ or 'No description available'}"
                tracker.set_decision_logic(logic)
                
                # Capture inputs
                inputs = {"args": args, "kwargs": func_kwargs}
                tracker.set_inputs(inputs)
                
                try:
                    # Execute function
                    result = func(*args, **func_kwargs)
                    
                    # Capture outputs
                    outputs = {"result": result}
                    tracker.set_outputs(outputs)
                    
                    # Mark as successful
                    tracker.set_outcome(TradingDecisionOutcome.SUCCESS)
                    
                    return result
                
                except Exception as e:
                    # Mark as failed
                    tracker.set_outcome(TradingDecisionOutcome.FAILURE)
                    raise
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Demo usage
    logger = TradingDecisionLogger()
    
    # Example trading decision
    with logger.decision_context(
        decision_type=TradingDecisionType.POSITION_SIZING,
        agent_id="risk_agent_1",
        strategy_id="momentum_strategy",
        symbol="BTCUSD",
        portfolio_value=100000.0
    ) as tracker:
        
        # Set decision logic
        tracker.set_decision_logic("Calculate optimal position size using Kelly criterion")
        
        # Set inputs
        tracker.set_inputs({
            "expected_return": 0.15,
            "volatility": 0.25,
            "win_rate": 0.6,
            "risk_free_rate": 0.02
        })
        
        # Simulate decision process
        tracker.log_intermediate_step("kelly_calculation", {"kelly_fraction": 0.25})
        tracker.log_intermediate_step("risk_adjustment", {"adjusted_fraction": 0.20})
        
        # Update metrics
        tracker.update_metrics(
            confidence_score=0.85,
            risk_score=0.3,
            expected_return=0.12,
            expected_risk=0.08
        )
        
        # Set outputs
        tracker.set_outputs({
            "position_size": 20000.0,
            "risk_percentage": 0.08,
            "expected_pnl": 2400.0
        })
        
        # Mark as successful
        tracker.set_outcome(TradingDecisionOutcome.SUCCESS)
    
    # Get summary
    summary = logger.get_decision_summary()
    print("Decision Summary:", json.dumps(summary, indent=2))
    
    # Get performance attribution
    attribution = logger.get_performance_attribution()
    print("Performance Attribution:", json.dumps(attribution, indent=2))