"""
EXECUTION PIPELINE COORDINATOR - AGENT 4
Maximum Velocity Deployment

This module coordinates the complete execution pipeline:
MARL → Main MAPPO → MC Dropout → Execution/Rejection → Feedback Loop

Single decision point with clear handoffs and comprehensive feedback.
"""

import torch
import asyncio
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# Import components
from .mc_dropout_execution_integration import (
    SingleMCDropoutEngine,
    TradeExecutionContext,
    BinaryExecutionResult,
    get_mc_dropout_engine
)
from ..training.enhanced_centralized_critic_with_mc_dropout import (
    EnhancedCentralizedCriticWithMC,
    MCDropoutFeatures,
    EnhancedCombinedStateWithMC,
    get_enhanced_critic_with_mc
)

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of execution pipeline."""
    PENDING = "pending"
    MAPPO_EVALUATING = "mappo_evaluating"
    MC_DROPOUT_EVALUATING = "mc_dropout_evaluating"
    EXECUTING = "executing"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class MAPPORecommendation:
    """MAPPO system recommendation for trade execution."""
    # Action recommendation
    recommended_action: str  # "buy", "sell", "hold"
    action_confidence: float
    value_estimate: float
    
    # Policy outputs
    policy_entropy: float
    critic_uncertainty: float
    
    # Multi-agent coordination
    agent_consensus: Dict[str, float]
    coordination_score: float
    
    # Risk assessment
    risk_score: float
    position_sizing: float
    
    # Market analysis
    regime_assessment: str
    timing_score: float
    
    # Metadata
    timestamp: float
    processing_time_ms: float


@dataclass
class ExecutionDecision:
    """Final execution decision with complete pipeline analysis."""
    # Decision outcome
    decision_made: str  # "execute", "reject", "delay"
    final_confidence: float
    
    # Pipeline components
    mappo_recommendation: MAPPORecommendation
    mc_dropout_result: BinaryExecutionResult
    
    # Execution details
    execution_context: TradeExecutionContext
    pipeline_status: ExecutionStatus
    
    # Performance metrics
    total_pipeline_time_ms: float
    mappo_time_ms: float
    mc_dropout_time_us: float
    
    # Feedback preparation
    feedback_data: Dict[str, Any]


@dataclass
class ExecutionOutcome:
    """Actual execution outcome for feedback."""
    # Execution results
    execution_successful: bool
    actual_pnl: float
    execution_cost: float
    slippage: float
    
    # Market impact
    market_impact_bps: float
    liquidity_consumed: float
    
    # Timing
    execution_time_ms: float
    fill_quality_score: float
    
    # Risk outcome
    risk_realized: float
    drawdown_contribution: float
    
    # Performance attribution
    alpha_contribution: float
    sharpe_contribution: float
    
    # Metadata
    execution_timestamp: float
    completion_timestamp: float


class ExecutionPipelineCoordinator:
    """
    EXECUTION PIPELINE COORDINATOR
    
    Orchestrates the complete flow:
    1. Receive MARL system outputs
    2. Generate MAPPO recommendation
    3. Forward to MC dropout for 1000-sample analysis
    4. Execute or reject based on MC dropout decision
    5. Collect outcomes and provide feedback to MAPPO
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize components
        self.mc_dropout_engine = get_mc_dropout_engine()
        self.enhanced_critic = get_enhanced_critic_with_mc()
        
        # Pipeline settings
        self.max_pipeline_time_ms = config.get('max_pipeline_time_ms', 1000)  # 1 second max
        self.require_mc_approval = config.get('require_mc_approval', True)
        self.confidence_threshold = config.get('confidence_threshold', 0.75)
        
        # Performance tracking
        self.pipeline_stats = {
            'total_recommendations': 0,
            'executed_trades': 0,
            'rejected_trades': 0,
            'pipeline_timeouts': 0,
            'average_pipeline_time_ms': 0.0,
            'mappo_mc_alignment_rate': 0.0,
            'execution_success_rate': 0.0
        }
        
        # Feedback system
        self.pending_executions = {}  # Track executions awaiting outcomes
        self.feedback_buffer = []     # Buffer for MAPPO learning
        self.alignment_history = []   # Track MAPPO-MC dropout alignment
        
        logger.info("Execution Pipeline Coordinator initialized")
    
    async def process_marl_outputs(self, 
                                 marl_outputs: Dict[str, Any],
                                 market_context: Dict[str, Any],
                                 portfolio_context: Dict[str, Any]) -> ExecutionDecision:
        """
        Main pipeline entry point - process MARL outputs through complete pipeline
        
        Args:
            marl_outputs: Outputs from MARL systems (strategic, tactical, execution)
            market_context: Current market conditions
            portfolio_context: Current portfolio state
            
        Returns:
            Complete execution decision with pipeline analysis
        """
        pipeline_start = time.perf_counter()
        
        try:
            # Step 1: Generate MAPPO recommendation
            mappo_start = time.perf_counter()
            mappo_recommendation = await self._generate_mappo_recommendation(
                marl_outputs, market_context, portfolio_context
            )
            mappo_time = (time.perf_counter() - mappo_start) * 1000
            
            # Step 2: Prepare execution context
            execution_context = self._prepare_execution_context(
                mappo_recommendation, market_context, portfolio_context
            )
            
            # Step 3: MC dropout evaluation
            mc_start = time.perf_counter()
            mc_dropout_result = await self.mc_dropout_engine.evaluate_trade_execution(
                execution_context
            )
            mc_time_us = mc_dropout_result.processing_time_us
            
            # Step 4: Make final decision
            final_decision = self._make_final_decision(
                mappo_recommendation, mc_dropout_result
            )
            
            # Step 5: Update pipeline tracking
            total_time = (time.perf_counter() - pipeline_start) * 1000
            self._update_pipeline_stats(total_time, mappo_recommendation, mc_dropout_result)
            
            # Step 6: Prepare feedback data
            feedback_data = self._prepare_feedback_data(
                mappo_recommendation, mc_dropout_result, final_decision
            )
            
            # Create execution decision
            execution_decision = ExecutionDecision(
                decision_made=final_decision,
                final_confidence=mc_dropout_result.confidence,
                mappo_recommendation=mappo_recommendation,
                mc_dropout_result=mc_dropout_result,
                execution_context=execution_context,
                pipeline_status=ExecutionStatus.COMPLETED if final_decision == "execute" else ExecutionStatus.REJECTED,
                total_pipeline_time_ms=total_time,
                mappo_time_ms=mappo_time,
                mc_dropout_time_us=mc_time_us,
                feedback_data=feedback_data
            )
            
            # Track for feedback
            if final_decision == "execute":
                self.pending_executions[id(execution_decision)] = execution_decision
            
            logger.info(f"Pipeline decision: {final_decision.upper()} "
                       f"(MAPPO: {mappo_recommendation.action_confidence:.3f}, "
                       f"MC: {mc_dropout_result.confidence:.3f}, "
                       f"time: {total_time:.1f}ms)")
            
            return execution_decision
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            # Return safe rejection decision
            return self._create_error_decision(str(e), pipeline_start)
    
    async def _generate_mappo_recommendation(self,
                                           marl_outputs: Dict[str, Any],
                                           market_context: Dict[str, Any],
                                           portfolio_context: Dict[str, Any]) -> MAPPORecommendation:
        """Generate MAPPO recommendation from MARL outputs."""
        
        # Combine MARL outputs into unified state
        base_features = self._combine_marl_outputs(marl_outputs, market_context, portfolio_context)
        
        # Get value estimate from enhanced critic
        with torch.no_grad():
            if base_features.dim() == 1:
                base_features = base_features.unsqueeze(0)
            
            # Forward through critic (uses backward compatibility for 112D input)
            value_output = self.enhanced_critic(base_features)
            
            if isinstance(value_output, tuple):
                value_estimate, uncertainty = value_output
                critic_uncertainty = uncertainty.item()
            else:
                value_estimate = value_output
                critic_uncertainty = 0.0
            
            value_estimate = value_estimate.item()
        
        # Extract recommendation from MARL outputs
        strategic_output = marl_outputs.get('strategic', {})
        tactical_output = marl_outputs.get('tactical', {})
        execution_output = marl_outputs.get('execution', {})
        
        # Determine recommended action
        strategic_action = strategic_output.get('action', 'hold')
        tactical_action = tactical_output.get('action', 'hold')
        execution_action = execution_output.get('action', 'hold')
        
        # Simple voting mechanism
        actions = [strategic_action, tactical_action, execution_action]
        recommended_action = max(set(actions), key=actions.count)
        
        # Calculate action confidence
        action_confidence = sum([
            strategic_output.get('confidence', 0.5),
            tactical_output.get('confidence', 0.5),
            execution_output.get('confidence', 0.5)
        ]) / 3.0
        
        # Agent consensus
        agent_consensus = {
            'strategic': strategic_output.get('confidence', 0.5),
            'tactical': tactical_output.get('confidence', 0.5),
            'execution': execution_output.get('confidence', 0.5)
        }
        
        # Coordination score (agreement between agents)
        consensus_values = list(agent_consensus.values())
        coordination_score = 1.0 - (max(consensus_values) - min(consensus_values))
        
        return MAPPORecommendation(
            recommended_action=recommended_action,
            action_confidence=action_confidence,
            value_estimate=value_estimate,
            policy_entropy=marl_outputs.get('policy_entropy', 0.5),
            critic_uncertainty=critic_uncertainty,
            agent_consensus=agent_consensus,
            coordination_score=coordination_score,
            risk_score=portfolio_context.get('current_risk_score', 0.5),
            position_sizing=execution_output.get('position_size', 1.0),
            regime_assessment=market_context.get('regime', 'normal'),
            timing_score=tactical_output.get('timing_score', 0.5),
            timestamp=time.time(),
            processing_time_ms=10.0  # Estimated MAPPO processing time
        )
    
    def _combine_marl_outputs(self,
                            marl_outputs: Dict[str, Any],
                            market_context: Dict[str, Any],
                            portfolio_context: Dict[str, Any]) -> torch.Tensor:
        """Combine MARL outputs into 112D feature vector."""
        
        features = []
        
        # Strategic features (30D)
        strategic = marl_outputs.get('strategic', {})
        features.extend([
            strategic.get('regime_confidence', 0.5),
            strategic.get('trend_strength', 0.0),
            strategic.get('volatility_forecast', 0.0),
            strategic.get('correlation_score', 0.0),
            strategic.get('macro_alignment', 0.5)
        ])
        features.extend([0.0] * 25)  # Pad to 30D
        
        # Tactical features (35D)
        tactical = marl_outputs.get('tactical', {})
        features.extend([
            tactical.get('momentum_score', 0.0),
            tactical.get('mean_reversion_score', 0.0),
            tactical.get('breakout_probability', 0.0),
            tactical.get('support_resistance_score', 0.0),
            tactical.get('volume_profile_score', 0.0)
        ])
        features.extend([0.0] * 30)  # Pad to 35D
        
        # Execution features (25D)
        execution = marl_outputs.get('execution', {})
        features.extend([
            execution.get('liquidity_score', 0.5),
            execution.get('impact_estimate', 0.0),
            execution.get('timing_urgency', 0.5),
            execution.get('venue_preference', 0.5),
            execution.get('cost_estimate', 0.0)
        ])
        features.extend([0.0] * 20)  # Pad to 25D
        
        # Market features (12D)
        features.extend([
            market_context.get('volatility', 0.0),
            market_context.get('volume', 0.0),
            market_context.get('bid_ask_spread', 0.0),
            market_context.get('momentum', 0.0),
            market_context.get('trend_strength', 0.0),
            market_context.get('regime_stability', 0.5)
        ])
        features.extend([0.0] * 6)  # Pad to 12D
        
        # Portfolio features (10D)
        features.extend([
            portfolio_context.get('current_position', 0.0),
            portfolio_context.get('available_capital', 1.0),
            portfolio_context.get('risk_usage', 0.0),
            portfolio_context.get('concentration', 0.0),
            portfolio_context.get('correlation_exposure', 0.0)
        ])
        features.extend([0.0] * 5)  # Pad to 10D
        
        # Ensure exactly 112D
        if len(features) < 112:
            features.extend([0.0] * (112 - len(features)))
        else:
            features = features[:112]
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def _prepare_execution_context(self,
                                 mappo_rec: MAPPORecommendation,
                                 market_context: Dict[str, Any],
                                 portfolio_context: Dict[str, Any]) -> TradeExecutionContext:
        """Prepare execution context for MC dropout evaluation."""
        
        return TradeExecutionContext(
            mappo_recommendation={
                'action_confidence': mappo_rec.action_confidence,
                'value_estimate': mappo_rec.value_estimate,
                'policy_entropy': mappo_rec.policy_entropy,
                'critic_uncertainty': mappo_rec.critic_uncertainty
            },
            market_data={
                'volatility': market_context.get('volatility', 0.0),
                'bid_ask_spread': market_context.get('bid_ask_spread', 0.0),
                'volume': market_context.get('volume', 0.0),
                'momentum': market_context.get('momentum', 0.0),
                'market_impact': market_context.get('market_impact', 0.0),
                'liquidity': market_context.get('liquidity', 0.0)
            },
            portfolio_state={
                'current_position': portfolio_context.get('current_position', 0.0),
                'available_capital': portfolio_context.get('available_capital', 1.0),
                'var_usage': portfolio_context.get('var_usage', 0.0),
                'concentration_risk': portfolio_context.get('concentration_risk', 0.0),
                'correlation_exposure': portfolio_context.get('correlation_exposure', 0.0)
            },
            risk_metrics={
                'var_estimate': portfolio_context.get('var_estimate', 0.0),
                'stress_test_result': portfolio_context.get('stress_test_result', 0.0),
                'drawdown_risk': portfolio_context.get('drawdown_risk', 0.0),
                'regime_risk': market_context.get('regime_risk', 0.0)
            },
            trade_details={
                'notional_value': portfolio_context.get('trade_notional', 0.0),
                'time_horizon': mappo_rec.timing_score,
                'urgency_score': 0.5,
                'execution_cost_estimate': 0.0
            },
            timestamp=time.time()
        )
    
    def _make_final_decision(self,
                           mappo_rec: MAPPORecommendation,
                           mc_result: BinaryExecutionResult) -> str:
        """Make final execution decision combining MAPPO and MC dropout."""
        
        # MC dropout has final veto power
        if not mc_result.execute_trade:
            return "reject"
        
        # Check MAPPO recommendation alignment
        mappo_wants_action = mappo_rec.recommended_action in ['buy', 'sell']
        
        if not mappo_wants_action:
            return "reject"
        
        # Check combined confidence
        combined_confidence = (mappo_rec.action_confidence + mc_result.confidence) / 2.0
        
        if combined_confidence < self.confidence_threshold:
            return "reject"
        
        return "execute"
    
    def _update_pipeline_stats(self,
                             pipeline_time: float,
                             mappo_rec: MAPPORecommendation,
                             mc_result: BinaryExecutionResult):
        """Update pipeline performance statistics."""
        
        self.pipeline_stats['total_recommendations'] += 1
        total = self.pipeline_stats['total_recommendations']
        
        # Update average pipeline time
        current_avg = self.pipeline_stats['average_pipeline_time_ms']
        self.pipeline_stats['average_pipeline_time_ms'] = (
            (current_avg * (total - 1) + pipeline_time) / total
        )
        
        # Track execution/rejection
        if mc_result.execute_trade:
            self.pipeline_stats['executed_trades'] += 1
        else:
            self.pipeline_stats['rejected_trades'] += 1
        
        # Track MAPPO-MC alignment
        mappo_wants_action = mappo_rec.recommended_action in ['buy', 'sell']
        alignment = mappo_wants_action == mc_result.execute_trade
        
        self.alignment_history.append(alignment)
        if len(self.alignment_history) > 100:
            self.alignment_history = self.alignment_history[-100:]
        
        self.pipeline_stats['mappo_mc_alignment_rate'] = sum(self.alignment_history) / len(self.alignment_history)
    
    def _prepare_feedback_data(self,
                             mappo_rec: MAPPORecommendation,
                             mc_result: BinaryExecutionResult,
                             final_decision: str) -> Dict[str, Any]:
        """Prepare feedback data for MAPPO learning."""
        
        return {
            'mappo_recommendation': {
                'action': mappo_rec.recommended_action,
                'confidence': mappo_rec.action_confidence,
                'value_estimate': mappo_rec.value_estimate
            },
            'mc_dropout_result': {
                'approved': mc_result.execute_trade,
                'confidence': mc_result.confidence,
                'uncertainty': mc_result.uncertainty_metrics.total_uncertainty,
                'sample_agreement': mc_result.sample_statistics.samples_above_threshold / 1000.0
            },
            'final_decision': final_decision,
            'alignment_score': float(
                (mappo_rec.recommended_action in ['buy', 'sell']) == mc_result.execute_trade
            ),
            'timestamp': time.time()
        }
    
    def _create_error_decision(self, error_msg: str, start_time: float) -> ExecutionDecision:
        """Create error decision for pipeline failures."""
        
        pipeline_time = (time.perf_counter() - start_time) * 1000
        
        # Create dummy objects for error case
        dummy_mappo = MAPPORecommendation(
            recommended_action="hold",
            action_confidence=0.0,
            value_estimate=0.0,
            policy_entropy=0.0,
            critic_uncertainty=1.0,
            agent_consensus={},
            coordination_score=0.0,
            risk_score=1.0,
            position_sizing=0.0,
            regime_assessment="unknown",
            timing_score=0.0,
            timestamp=time.time(),
            processing_time_ms=pipeline_time
        )
        
        return ExecutionDecision(
            decision_made="reject",
            final_confidence=0.0,
            mappo_recommendation=dummy_mappo,
            mc_dropout_result=None,
            execution_context=None,
            pipeline_status=ExecutionStatus.FAILED,
            total_pipeline_time_ms=pipeline_time,
            mappo_time_ms=0.0,
            mc_dropout_time_us=0.0,
            feedback_data={'error': error_msg}
        )
    
    async def provide_execution_outcome(self,
                                      execution_decision_id: int,
                                      outcome: ExecutionOutcome):
        """
        Provide execution outcome for feedback learning
        
        Args:
            execution_decision_id: ID of the execution decision
            outcome: Actual execution outcome
        """
        
        if execution_decision_id not in self.pending_executions:
            logger.warning(f"Unknown execution decision ID: {execution_decision_id}")
            return
        
        execution_decision = self.pending_executions.pop(execution_decision_id)
        
        # Update enhanced critic with MC dropout feedback
        await self._update_critic_feedback(execution_decision, outcome)
        
        # Update execution success rate
        total_outcomes = len(self.feedback_buffer) + 1
        current_success_rate = self.pipeline_stats['execution_success_rate']
        
        new_success_rate = (
            (current_success_rate * (total_outcomes - 1) + float(outcome.execution_successful)) /
            total_outcomes
        )
        self.pipeline_stats['execution_success_rate'] = new_success_rate
        
        # Store feedback for MAPPO learning
        feedback_sample = {
            'execution_decision': execution_decision,
            'outcome': outcome,
            'timestamp': time.time()
        }
        
        self.feedback_buffer.append(feedback_sample)
        
        # Keep limited buffer
        if len(self.feedback_buffer) > 500:
            self.feedback_buffer = self.feedback_buffer[-250:]
        
        logger.info(f"Execution outcome recorded: {'SUCCESS' if outcome.execution_successful else 'FAILURE'} "
                   f"(PnL: {outcome.actual_pnl:.4f})")
    
    async def _update_critic_feedback(self,
                                    execution_decision: ExecutionDecision,
                                    outcome: ExecutionOutcome):
        """Update enhanced critic with execution outcome feedback."""
        
        if execution_decision.mc_dropout_result is None:
            return
        
        # Convert outcome to feedback format
        actual_outcome = {
            'success': outcome.execution_successful,
            'pnl': outcome.actual_pnl,
            'execution_cost': outcome.execution_cost,
            'slippage': outcome.slippage,
            'risk_realized': outcome.risk_realized
        }
        
        # Update critic feedback
        self.enhanced_critic.update_mc_feedback(
            execution_decision.mc_dropout_result,
            actual_outcome
        )
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline performance metrics."""
        
        stats = self.pipeline_stats.copy()
        
        # Add derived metrics
        total = stats['total_recommendations']
        if total > 0:
            stats['execution_rate'] = stats['executed_trades'] / total
            stats['rejection_rate'] = stats['rejected_trades'] / total
        else:
            stats['execution_rate'] = 0.0
            stats['rejection_rate'] = 0.0
        
        # Add component metrics
        stats['mc_dropout_metrics'] = self.mc_dropout_engine.get_performance_metrics()
        stats['critic_learning_metrics'] = self.enhanced_critic.get_mc_learning_metrics()
        
        # Add feedback buffer status
        stats['pending_executions'] = len(self.pending_executions)
        stats['feedback_buffer_size'] = len(self.feedback_buffer)
        
        return stats


# Factory function
def create_execution_pipeline_coordinator(config: Dict[str, Any] = None) -> ExecutionPipelineCoordinator:
    """Create execution pipeline coordinator with optimal configuration."""
    
    if config is None:
        config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'max_pipeline_time_ms': 1000,
            'require_mc_approval': True,
            'confidence_threshold': 0.75
        }
    
    return ExecutionPipelineCoordinator(config)


# Global instance
_global_pipeline_coordinator = None

def get_execution_pipeline_coordinator() -> ExecutionPipelineCoordinator:
    """Get the global execution pipeline coordinator instance."""
    global _global_pipeline_coordinator
    if _global_pipeline_coordinator is None:
        _global_pipeline_coordinator = create_execution_pipeline_coordinator()
    return _global_pipeline_coordinator