"""
Risk Management MARL System Demo

Demonstrates the complete 4-agent MARL system for risk management:
- Position Sizing Agent (π₁)
- Stop/Target Agent (π₂) 
- Risk Monitor Agent (π₃)
- Portfolio Optimizer Agent (π₄)

Features demonstrated:
- Multi-agent coordination
- Centralized critic evaluation
- Real-time risk management
- Performance optimization
- Emergency protocols
"""

import time
import numpy as np
from datetime import datetime
from typing import Dict, Any
import structlog

from src.core.events import EventBus
from src.risk.agents import (
    PositionSizingAgent, StopTargetAgent, RiskMonitorAgent, 
    PortfolioOptimizerAgent, RiskState
)
from src.risk.marl.centralized_critic import CentralizedCritic, GlobalRiskState
from src.risk.marl.agent_coordinator import AgentCoordinator, CoordinatorConfig
from src.risk.marl.risk_environment import RiskEnvironment
from src.risk.core.state_processor import RiskStateProcessor, StateProcessingConfig
from src.risk.core.performance_optimizer import PerformanceOptimizer, setup_performance_optimization

logger = structlog.get_logger()


class RiskManagementMARLSystem:
    """
    Complete Risk Management MARL System
    
    Integrates all components for coordinated risk management:
    - 4 specialized risk agents
    - Centralized critic for global assessment
    - Agent coordinator for consensus
    - Performance optimization
    - Real-time monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the complete MARL system"""
        
        self.config = config
        self.system_start_time = datetime.now()
        
        # Initialize event bus for system communication
        self.event_bus = EventBus()
        
        # Initialize performance optimizer
        perf_config = config.get('performance', {})
        self.performance_optimizer = setup_performance_optimization(perf_config)
        
        # Initialize state processor
        state_config = StateProcessingConfig()
        self.state_processor = RiskStateProcessor(state_config, self.event_bus)
        
        # Initialize centralized critic
        critic_config = config.get('critic', {})
        self.centralized_critic = CentralizedCritic(critic_config, self.event_bus)
        
        # Initialize agent coordinator
        coordinator_config = CoordinatorConfig()
        self.agent_coordinator = AgentCoordinator(
            coordinator_config, self.centralized_critic, self.event_bus
        )
        
        # Initialize the 4 risk agents
        self.agents = self._initialize_agents()
        
        # Register agents with coordinator
        for agent in self.agents.values():
            self.agent_coordinator.register_agent(agent)
        
        # System statistics
        self.total_decisions = 0
        self.emergency_stops = 0
        self.consensus_achieved = 0
        self.performance_violations = 0
        
        logger.info("Risk Management MARL System initialized",
                   agents=len(self.agents),
                   system_start=self.system_start_time.isoformat())
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all 4 risk agents"""
        
        agents = {}
        
        # π₁ Position Sizing Agent
        position_config = self.config.get('position_sizing', {
            'max_leverage': 3.0,
            'var_limit': 0.02,
            'correlation_threshold': 0.7
        })
        agents['position_sizing'] = PositionSizingAgent(position_config, self.event_bus)
        
        # π₂ Stop/Target Agent
        stop_target_config = self.config.get('stop_target', {
            'base_stop_distance': 0.02,
            'base_target_distance': 0.04,
            'volatility_sensitivity': 1.5
        })
        agents['stop_target'] = StopTargetAgent(stop_target_config, self.event_bus)
        
        # π₃ Risk Monitor Agent
        risk_monitor_config = self.config.get('risk_monitor', {
            'alert_threshold': 0.6,
            'reduce_threshold': 0.75,
            'emergency_threshold': 0.9
        })
        agents['risk_monitor'] = RiskMonitorAgent(risk_monitor_config, self.event_bus)
        
        # π₄ Portfolio Optimizer Agent
        portfolio_config = self.config.get('portfolio_optimizer', {
            'target_volatility': 0.12,
            'max_correlation': 0.8,
            'rebalance_threshold': 0.05
        })
        agents['portfolio_optimizer'] = PortfolioOptimizerAgent(portfolio_config, self.event_bus)
        
        return agents
    
    def process_risk_situation(self, risk_vector: np.ndarray) -> Dict[str, Any]:
        """
        Process a risk situation through the complete MARL system
        
        Args:
            risk_vector: 10-dimensional risk state vector
            
        Returns:
            System response with actions, consensus, and performance metrics
        """
        
        start_time = time.perf_counter()
        
        try:
            # Step 1: Process and normalize risk state
            normalized_state, processing_metadata = self.state_processor.process_state(risk_vector)
            
            # Step 2: Create risk state object
            risk_state = RiskState.from_vector(normalized_state)
            
            # Step 3: Coordinate agent decisions
            consensus_results = self.agent_coordinator.coordinate_decision(risk_state)
            
            # Step 4: Evaluate global risk with centralized critic
            global_risk_state = self._create_global_risk_state(risk_state, consensus_results)
            global_risk_value, operating_mode = self.centralized_critic.evaluate_global_risk(global_risk_state)
            
            # Step 5: Extract individual agent actions
            agent_actions = self._extract_agent_actions(consensus_results)
            
            # Calculate total response time
            response_time = (time.perf_counter() - start_time) * 1000
            
            # Update statistics
            self.total_decisions += 1
            if operating_mode.value == 'emergency':
                self.emergency_stops += 1
            if consensus_results:
                self.consensus_achieved += 1
            if response_time > 10.0:
                self.performance_violations += 1
            
            # Compile response
            response = {
                'timestamp': datetime.now(),
                'response_time_ms': response_time,
                'risk_state': {
                    'original': risk_vector.tolist(),
                    'normalized': normalized_state.tolist(),
                    'processing_metadata': processing_metadata
                },
                'agent_actions': agent_actions,
                'consensus_results': self._format_consensus_results(consensus_results),
                'global_assessment': {
                    'risk_value': global_risk_value,
                    'operating_mode': operating_mode.value
                },
                'performance': {
                    'target_met': response_time <= 10.0,
                    'response_time_ms': response_time
                },
                'system_health': self._get_system_health()
            }
            
            # Log the decision
            self._log_system_decision(response)
            
            return response
            
        except Exception as e:
            error_response_time = (time.perf_counter() - start_time) * 1000
            logger.error("Error processing risk situation", 
                        error=str(e),
                        response_time=error_response_time)
            
            # Return emergency response
            return self._get_emergency_response(risk_vector, str(e))
    
    def _create_global_risk_state(self, risk_state: RiskState, consensus_results: Dict[str, Any]) -> GlobalRiskState:
        """Create global risk state for centralized critic"""
        
        base_vector = risk_state.to_vector()
        
        return GlobalRiskState(
            position_sizing_risk=base_vector,
            stop_target_risk=base_vector,
            risk_monitor_risk=base_vector,
            portfolio_optimizer_risk=base_vector,
            total_portfolio_var=risk_state.var_estimate_5pct,
            portfolio_correlation_max=risk_state.correlation_risk,
            aggregate_leverage=risk_state.margin_usage_pct * 4.0,  # Approximate leverage
            liquidity_risk_score=1.0 - risk_state.liquidity_conditions,
            systemic_risk_level=risk_state.market_stress_level,
            timestamp=datetime.now(),
            market_hours_factor=risk_state.time_of_day_risk
        )
    
    def _extract_agent_actions(self, consensus_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract individual agent actions from consensus results"""
        
        actions = {}
        
        for action_type, consensus in consensus_results.items():
            if hasattr(consensus, 'consensus_action'):
                if isinstance(consensus.consensus_action, np.ndarray):
                    actions[action_type] = consensus.consensus_action.tolist()
                else:
                    actions[action_type] = consensus.consensus_action
            else:
                actions[action_type] = None
        
        return actions
    
    def _format_consensus_results(self, consensus_results: Dict[str, Any]) -> Dict[str, Any]:
        """Format consensus results for response"""
        
        formatted = {}
        
        for action_type, consensus in consensus_results.items():
            if hasattr(consensus, 'method_used'):
                formatted[action_type] = {
                    'method_used': consensus.method_used.value,
                    'confidence_score': consensus.confidence_score,
                    'participating_agents': consensus.participating_agents,
                    'execution_time_ms': consensus.execution_time_ms,
                    'conflicts_detected': consensus.conflicts_detected,
                    'overrides_applied': consensus.overrides_applied
                }
            else:
                formatted[action_type] = {'status': 'no_consensus'}
        
        return formatted
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get current system health metrics"""
        
        # Agent health
        agent_health = {}
        for name, agent in self.agents.items():
            metrics = agent.get_risk_metrics()
            agent_health[name] = {
                'decisions_made': metrics.total_risk_decisions,
                'avg_response_time': metrics.avg_response_time_ms,
                'active': True
            }
        
        # Coordinator health
        coordinator_metrics = self.agent_coordinator.get_performance_metrics()
        
        # Performance optimizer health
        performance_metrics = self.performance_optimizer.get_performance_metrics()
        
        return {
            'agents': agent_health,
            'coordinator': {
                'coordination_count': coordinator_metrics['coordination_count'],
                'consensus_failures': coordinator_metrics['consensus_failures'],
                'emergency_activations': coordinator_metrics['emergency_activations'],
                'current_mode': coordinator_metrics['current_mode']
            },
            'performance': {
                'avg_response_time_ms': performance_metrics.avg_response_time_ms,
                'target_achieved': performance_metrics.target_achieved,
                'throughput_ops_per_sec': performance_metrics.throughput_ops_per_sec
            },
            'system_stats': {
                'total_decisions': self.total_decisions,
                'emergency_stops': self.emergency_stops,
                'consensus_achieved': self.consensus_achieved,
                'performance_violations': self.performance_violations,
                'uptime_seconds': (datetime.now() - self.system_start_time).total_seconds()
            }
        }
    
    def _get_emergency_response(self, risk_vector: np.ndarray, error: str) -> Dict[str, Any]:
        """Get emergency response for error cases"""
        
        return {
            'timestamp': datetime.now(),
            'response_time_ms': 0.0,
            'risk_state': {
                'original': risk_vector.tolist(),
                'normalized': None,
                'processing_metadata': {'error': error}
            },
            'agent_actions': {
                'position_sizing': 1,  # REDUCE_SMALL
                'stop_target': [0.8, 1.0],  # Conservative stops
                'risk_monitor': 3,  # EMERGENCY_STOP
                'portfolio_optimizer': [0.2, 0.4, 0.05, 0.3, 0.05]  # Conservative allocation
            },
            'consensus_results': {'status': 'emergency_override'},
            'global_assessment': {
                'risk_value': -1.0,
                'operating_mode': 'emergency'
            },
            'performance': {
                'target_met': False,
                'response_time_ms': 0.0
            },
            'system_health': {'status': 'error', 'error': error}
        }
    
    def _log_system_decision(self, response: Dict[str, Any]):
        """Log system decision for monitoring"""
        
        logger.info("MARL system decision",
                   response_time=response['response_time_ms'],
                   operating_mode=response['global_assessment']['operating_mode'],
                   consensus_count=len(response['consensus_results']),
                   performance_target_met=response['performance']['target_met'],
                   total_decisions=self.total_decisions)
    
    def run_demo_scenario(self, num_scenarios: int = 10) -> Dict[str, Any]:
        """
        Run demo scenarios to showcase the system
        
        Args:
            num_scenarios: Number of scenarios to run
            
        Returns:
            Demo results and statistics
        """
        
        logger.info("Starting MARL system demo", scenarios=num_scenarios)
        
        demo_results = []
        scenario_types = [
            'normal_conditions',
            'high_volatility',
            'correlation_spike',
            'market_stress',
            'emergency_conditions'
        ]
        
        for i in range(num_scenarios):
            scenario_type = scenario_types[i % len(scenario_types)]
            
            # Generate scenario-specific risk vector
            risk_vector = self._generate_scenario_risk_vector(scenario_type)
            
            # Process through MARL system
            start_time = time.perf_counter()
            response = self.process_risk_situation(risk_vector)
            processing_time = (time.perf_counter() - start_time) * 1000
            
            # Store results
            demo_results.append({
                'scenario_id': i + 1,
                'scenario_type': scenario_type,
                'processing_time_ms': processing_time,
                'response': response
            })
            
            logger.info("Demo scenario completed",
                       scenario_id=i + 1,
                       scenario_type=scenario_type,
                       processing_time=processing_time,
                       operating_mode=response['global_assessment']['operating_mode'])
        
        # Compile demo summary
        demo_summary = self._compile_demo_summary(demo_results)
        
        logger.info("MARL system demo completed", 
                   scenarios_run=num_scenarios,
                   avg_response_time=demo_summary['avg_response_time_ms'],
                   performance_target_met=demo_summary['performance_target_met_rate'])
        
        return {
            'demo_results': demo_results,
            'demo_summary': demo_summary,
            'system_health': self._get_system_health()
        }
    
    def _generate_scenario_risk_vector(self, scenario_type: str) -> np.ndarray:
        """Generate risk vector for specific scenario type"""
        
        if scenario_type == 'normal_conditions':
            return np.array([1.0, 5, 0.3, 0.3, 0.01, 0.0, 0.3, 0.3, 0.2, 0.8])
        
        elif scenario_type == 'high_volatility':
            return np.array([0.95, 7, 0.8, 0.4, 0.03, 0.05, 0.5, 0.6, 0.5, 0.6])
        
        elif scenario_type == 'correlation_spike':
            return np.array([0.9, 8, 0.6, 0.9, 0.04, 0.08, 0.6, 0.5, 0.6, 0.5])
        
        elif scenario_type == 'market_stress':
            return np.array([0.85, 10, 0.7, 0.7, 0.06, 0.12, 0.7, 0.7, 0.8, 0.3])
        
        elif scenario_type == 'emergency_conditions':
            return np.array([0.7, 15, 0.9, 0.95, 0.1, 0.2, 0.9, 0.8, 0.9, 0.2])
        
        else:
            # Default to normal conditions
            return np.array([1.0, 5, 0.3, 0.3, 0.01, 0.0, 0.3, 0.3, 0.2, 0.8])
    
    def _compile_demo_summary(self, demo_results: list) -> Dict[str, Any]:
        """Compile summary statistics from demo results"""
        
        response_times = [r['processing_time_ms'] for r in demo_results]
        target_met_count = sum(1 for r in demo_results 
                              if r['response']['performance']['target_met'])
        
        operating_modes = [r['response']['global_assessment']['operating_mode'] 
                          for r in demo_results]
        mode_counts = {mode: operating_modes.count(mode) for mode in set(operating_modes)}
        
        return {
            'total_scenarios': len(demo_results),
            'avg_response_time_ms': np.mean(response_times),
            'max_response_time_ms': np.max(response_times),
            'min_response_time_ms': np.min(response_times),
            'performance_target_met_rate': target_met_count / len(demo_results),
            'operating_mode_distribution': mode_counts,
            'system_performance': {
                'decisions_processed': self.total_decisions,
                'emergency_stops': self.emergency_stops,
                'consensus_achieved_rate': self.consensus_achieved / max(1, self.total_decisions)
            }
        }
    
    def shutdown(self):
        """Shutdown the MARL system"""
        
        logger.info("Shutting down Risk Management MARL System")
        
        # Shutdown components
        self.agent_coordinator.shutdown()
        self.state_processor.shutdown()
        self.performance_optimizer.shutdown()
        
        # Reset agents
        for agent in self.agents.values():
            agent.reset()
        
        logger.info("MARL system shutdown complete")


def main():
    """Main demo function"""
    
    # Configuration for the complete system
    system_config = {
        'performance': {
            'target_response_time_ms': 10.0,
            'optimization_level': 'balanced'
        },
        'critic': {
            'learning_rate': 1e-4,
            'stress_threshold': 0.15,
            'emergency_threshold': 0.25
        },
        'position_sizing': {
            'max_leverage': 3.0,
            'var_limit': 0.02,
            'correlation_threshold': 0.7
        },
        'stop_target': {
            'base_stop_distance': 0.02,
            'base_target_distance': 0.04,
            'volatility_sensitivity': 1.5
        },
        'risk_monitor': {
            'alert_threshold': 0.6,
            'reduce_threshold': 0.75,
            'emergency_threshold': 0.9
        },
        'portfolio_optimizer': {
            'target_volatility': 0.12,
            'max_correlation': 0.8,
            'rebalance_threshold': 0.05
        }
    }
    
    print("🤖 Risk Management MARL System Demo")
    print("=" * 50)
    
    # Initialize system
    print("Initializing MARL system...")
    marl_system = RiskManagementMARLSystem(system_config)
    
    # Run demo scenarios
    print("Running demo scenarios...")
    demo_results = marl_system.run_demo_scenario(num_scenarios=15)
    
    # Display results
    print("\n📊 Demo Results Summary:")
    print("-" * 30)
    summary = demo_results['demo_summary']
    print(f"Scenarios processed: {summary['total_scenarios']}")
    print(f"Average response time: {summary['avg_response_time_ms']:.2f}ms")
    print(f"Performance target met: {summary['performance_target_met_rate']:.1%}")
    print(f"Operating modes: {summary['operating_mode_distribution']}")
    
    print("\n🎯 System Performance:")
    print("-" * 20)
    health = demo_results['system_health']
    print(f"Total decisions: {health['system_stats']['total_decisions']}")
    print(f"Emergency stops: {health['system_stats']['emergency_stops']}")
    print(f"Consensus rate: {summary['system_performance']['consensus_achieved_rate']:.1%}")
    print(f"Performance violations: {health['system_stats']['performance_violations']}")
    
    # Shutdown system
    print("\n🛑 Shutting down system...")
    marl_system.shutdown()
    
    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    main()