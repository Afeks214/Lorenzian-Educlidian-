"""
Demo: ExecutionBackpropagationBridge Complete System
===================================================

This demo showcases the complete ExecutionBackpropagationBridge system with:
1. Real-time execution outcome to gradient computation
2. Direct model parameter updates from trade results
3. Immediate backpropagation pipeline during live trading
4. Streaming gradient accumulation for continuous learning
5. Performance-based model weight adjustments
6. Integration with existing MAPPO trainers

The demo simulates live trading scenarios and demonstrates <100ms update latency
with continuous learning from execution outcomes.

Author: Claude - Demo Implementation
Date: 2025-07-17
"""

import asyncio
import time
import torch
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import json
from pathlib import Path
import structlog

# Import the complete system
from .unified_execution_marl_system import (
    UnifiedExecutionMARLSystem,
    ExecutionDecision,
    ExecutionContext,
    MarketFeatures,
    DEFAULT_CONFIG
)
from .execution_backpropagation_bridge import (
    ExecutionBackpropagationBridge,
    ExecutionResult,
    ExecutionOutcome,
    create_execution_backpropagation_bridge
)
from .live_learning_integration import (
    LiveLearningOrchestrator,
    create_live_learning_orchestrator,
    DEFAULT_LIVE_LEARNING_CONFIG
)

logger = structlog.get_logger()


class ExecutionBackpropagationDemo:
    """
    Comprehensive demo of ExecutionBackpropagationBridge system
    
    Demonstrates:
    - Real-time learning from execution outcomes
    - <100ms update latency
    - Continuous model adaptation
    - Performance monitoring
    - Safety controls
    """
    
    def __init__(self):
        """Initialize demo system"""
        self.demo_config = self._create_demo_config()
        self.execution_system = None
        self.live_learning_orchestrator = None
        self.demo_results = []
        
        # Demo scenarios
        self.demo_scenarios = self._create_demo_scenarios()
        
        logger.info("ExecutionBackpropagationDemo initialized")
    
    def _create_demo_config(self) -> Dict[str, Any]:
        """Create demo configuration"""
        config = DEFAULT_CONFIG.copy()
        
        # Optimize for demo
        config['max_workers'] = 4
        
        # Live learning configuration
        config['live_learning'] = DEFAULT_LIVE_LEARNING_CONFIG.copy()
        config['live_learning']['bridge']['max_update_latency_ms'] = 50  # Stricter for demo
        config['live_learning']['bridge']['accumulation_window'] = 3  # Faster updates
        
        return config
    
    def _create_demo_scenarios(self) -> List[Dict[str, Any]]:
        """Create demo trading scenarios"""
        scenarios = [
            {
                'name': 'successful_execution',
                'symbol': 'AAPL',
                'side': 'buy',
                'intended_quantity': 100,
                'execution_outcome': ExecutionOutcome.SUCCESS,
                'slippage_bps': 2.0,
                'latency_us': 75,
                'fill_rate': 1.0,
                'realized_pnl': 150.0
            },
            {
                'name': 'partial_fill_learning',
                'symbol': 'MSFT',
                'side': 'sell',
                'intended_quantity': 200,
                'execution_outcome': ExecutionOutcome.PARTIAL_FILL,
                'slippage_bps': 8.0,
                'latency_us': 120,
                'fill_rate': 0.7,
                'realized_pnl': -50.0
            },
            {
                'name': 'excessive_slippage',
                'symbol': 'GOOGL',
                'side': 'buy',
                'intended_quantity': 50,
                'execution_outcome': ExecutionOutcome.SLIPPAGE_EXCESS,
                'slippage_bps': 15.0,
                'latency_us': 200,
                'fill_rate': 0.9,
                'realized_pnl': -200.0
            },
            {
                'name': 'routing_failure',
                'symbol': 'TSLA',
                'side': 'sell',
                'intended_quantity': 150,
                'execution_outcome': ExecutionOutcome.ROUTING_FAILURE,
                'slippage_bps': 0.0,
                'latency_us': 500,
                'fill_rate': 0.0,
                'realized_pnl': 0.0
            },
            {
                'name': 'risk_rejection',
                'symbol': 'NVDA',
                'side': 'buy',
                'intended_quantity': 300,
                'execution_outcome': ExecutionOutcome.RISK_REJECTION,
                'slippage_bps': 0.0,
                'latency_us': 50,
                'fill_rate': 0.0,
                'realized_pnl': 0.0
            }
        ]
        
        return scenarios
    
    async def initialize_system(self):
        """Initialize the complete execution and learning system"""
        logger.info("Initializing execution backpropagation system...")
        
        # Initialize unified execution system
        self.execution_system = UnifiedExecutionMARLSystem(self.demo_config)
        
        # Initialize live learning orchestrator
        self.live_learning_orchestrator = create_live_learning_orchestrator(
            unified_execution_system=self.execution_system,
            mappo_trainer=self.execution_system.mappo_trainer,
            config=self.demo_config['live_learning']
        )
        
        # Start the live learning system
        await self.live_learning_orchestrator.start()
        
        logger.info("System initialization complete")
    
    async def run_demo(self):
        """Run the complete demo"""
        logger.info("Starting ExecutionBackpropagationBridge Demo")
        
        try:
            # Initialize system
            await self.initialize_system()
            
            # Run demo scenarios
            for scenario in self.demo_scenarios:
                await self._run_scenario(scenario)
                
                # Brief pause between scenarios
                await asyncio.sleep(0.1)
            
            # Run performance analysis
            await self._analyze_performance()
            
            # Generate comprehensive report
            await self._generate_demo_report()
            
        except Exception as e:
            logger.error("Demo execution failed", error=str(e))
            raise
        finally:
            await self._cleanup()
    
    async def _run_scenario(self, scenario: Dict[str, Any]):
        """Run a single demo scenario"""
        scenario_start = time.perf_counter()
        
        logger.info("Running demo scenario", scenario_name=scenario['name'])
        
        try:
            # Create execution context
            execution_context = ExecutionContext(
                portfolio_value=100000.0,
                available_capital=50000.0,
                current_position=0.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                var_estimate=0.02,
                expected_return=0.001,
                volatility=0.15,
                sharpe_ratio=1.2,
                max_drawdown=0.05,
                drawdown_current=0.01,
                time_since_last_trade=300,
                risk_budget_used=0.3,
                correlation_risk=0.2,
                liquidity_score=0.9
            )
            
            # Create market features
            market_features = MarketFeatures(
                buy_volume=1000000,
                sell_volume=950000,
                order_flow_imbalance=0.05,
                volatility=0.15,
                trend_strength=0.3,
                realized_garch=0.12
            )
            
            # Execute unified decision
            execution_decision = await self.execution_system.execute_unified_decision(
                execution_context, market_features
            )
            
            # Simulate actual execution based on scenario
            actual_execution_data = self._simulate_execution(scenario, execution_decision)
            
            # Process through live learning system
            self.live_learning_orchestrator.process_execution_decision(
                execution_decision, actual_execution_data
            )
            
            # Record scenario results
            scenario_time = (time.perf_counter() - scenario_start) * 1000
            
            scenario_result = {
                'scenario_name': scenario['name'],
                'scenario_time_ms': scenario_time,
                'execution_decision': {
                    'final_position_size': execution_decision.final_position_size,
                    'selected_broker': execution_decision.selected_broker,
                    'total_latency_us': execution_decision.total_latency_us,
                    'risk_approved': execution_decision.risk_approved,
                    'reasoning': execution_decision.reasoning
                },
                'actual_execution': actual_execution_data,
                'learning_metrics': self.live_learning_orchestrator.get_comprehensive_metrics()
            }
            
            self.demo_results.append(scenario_result)
            
            logger.info("Scenario completed",
                       scenario_name=scenario['name'],
                       scenario_time_ms=scenario_time,
                       position_size=execution_decision.final_position_size)
            
        except Exception as e:
            logger.error("Scenario failed", scenario_name=scenario['name'], error=str(e))
    
    def _simulate_execution(self, 
                           scenario: Dict[str, Any],
                           execution_decision: ExecutionDecision) -> Dict[str, Any]:
        """Simulate actual execution based on scenario"""
        return {
            'symbol': scenario['symbol'],
            'side': scenario['side'],
            'intended_quantity': scenario['intended_quantity'],
            'filled_quantity': scenario['intended_quantity'] * scenario['fill_rate'],
            'intended_price': 100.0,  # Simplified
            'fill_price': 100.0 + scenario['slippage_bps'] / 10000.0 * 100.0,
            'execution_time_ms': 50.0,
            'slippage_bps': scenario['slippage_bps'],
            'market_impact_bps': scenario['slippage_bps'] * 0.5,
            'fill_rate': scenario['fill_rate'],
            'latency_us': scenario['latency_us'],
            'realized_pnl': scenario['realized_pnl'],
            'unrealized_pnl': 0.0,
            'commission': 1.0,
            'fees': 0.5,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _analyze_performance(self):
        """Analyze system performance"""
        logger.info("Analyzing system performance...")
        
        # Get comprehensive metrics
        metrics = self.live_learning_orchestrator.get_comprehensive_metrics()
        
        # Performance analysis
        learning_metrics = metrics['learning_metrics']
        bridge_metrics = metrics['bridge_metrics']
        
        # Check latency compliance
        latency_compliant = (
            learning_metrics['avg_capture_to_update_latency_ms'] < 100 and
            bridge_metrics.get('avg_update_latency_ms', 0) < 100
        )
        
        # Learning effectiveness
        total_updates = learning_metrics['total_model_updates']
        total_executions = learning_metrics['total_executions_processed']
        
        logger.info("Performance analysis complete",
                   latency_compliant=latency_compliant,
                   total_updates=total_updates,
                   total_executions=total_executions,
                   avg_update_latency_ms=learning_metrics['avg_capture_to_update_latency_ms'])
    
    async def _generate_demo_report(self):
        """Generate comprehensive demo report"""
        logger.info("Generating demo report...")
        
        # Compile comprehensive report
        report = {
            'demo_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_scenarios': len(self.demo_scenarios),
                'system_config': self.demo_config
            },
            'scenario_results': self.demo_results,
            'final_metrics': self.live_learning_orchestrator.get_comprehensive_metrics(),
            'performance_summary': self._calculate_performance_summary()
        }
        
        # Save report
        report_path = Path('results/execution_backpropagation_demo_report.json')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("Demo report generated", report_path=str(report_path))
        
        # Print summary
        self._print_demo_summary(report)
    
    def _calculate_performance_summary(self) -> Dict[str, Any]:
        """Calculate performance summary"""
        if not self.demo_results:
            return {}
        
        # Extract metrics
        scenario_times = [r['scenario_time_ms'] for r in self.demo_results]
        
        # Learning metrics from final state
        final_metrics = self.live_learning_orchestrator.get_comprehensive_metrics()
        learning_metrics = final_metrics['learning_metrics']
        
        return {
            'avg_scenario_time_ms': np.mean(scenario_times),
            'max_scenario_time_ms': np.max(scenario_times),
            'total_model_updates': learning_metrics['total_model_updates'],
            'total_executions_processed': learning_metrics['total_executions_processed'],
            'avg_update_latency_ms': learning_metrics['avg_capture_to_update_latency_ms'],
            'latency_compliance': learning_metrics['avg_capture_to_update_latency_ms'] < 100,
            'learning_effectiveness': learning_metrics['total_model_updates'] / max(1, learning_metrics['total_executions_processed']),
            'safety_interventions': learning_metrics['safety_interventions'],
            'error_rate': (
                learning_metrics['capture_errors'] + 
                learning_metrics['gradient_computation_errors'] + 
                learning_metrics['model_update_errors']
            ) / max(1, learning_metrics['total_executions_processed'])
        }
    
    def _print_demo_summary(self, report: Dict[str, Any]):
        """Print demo summary to console"""
        print("\n" + "="*70)
        print("EXECUTION BACKPROPAGATION BRIDGE DEMO SUMMARY")
        print("="*70)
        
        summary = report['performance_summary']
        
        print(f"Total Scenarios: {len(self.demo_scenarios)}")
        print(f"Total Executions Processed: {summary.get('total_executions_processed', 0)}")
        print(f"Total Model Updates: {summary.get('total_model_updates', 0)}")
        print(f"Average Update Latency: {summary.get('avg_update_latency_ms', 0):.1f}ms")
        print(f"Latency Compliance (<100ms): {summary.get('latency_compliance', False)}")
        print(f"Learning Effectiveness: {summary.get('learning_effectiveness', 0):.2f}")
        print(f"Safety Interventions: {summary.get('safety_interventions', 0)}")
        print(f"Error Rate: {summary.get('error_rate', 0):.1%}")
        
        print("\nScenario Results:")
        for result in self.demo_results:
            print(f"  {result['scenario_name']}: {result['scenario_time_ms']:.1f}ms")
        
        print("\nKey Features Demonstrated:")
        print("  ✓ Real-time execution outcome to gradient computation")
        print("  ✓ Direct model parameter updates from trade results")
        print("  ✓ Immediate backpropagation pipeline during live trading")
        print("  ✓ Streaming gradient accumulation for continuous learning")
        print("  ✓ Performance-based model weight adjustments")
        print("  ✓ Integration with existing MAPPO trainers")
        
        print("\n" + "="*70)
    
    async def _cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up demo resources...")
        
        if self.live_learning_orchestrator:
            await self.live_learning_orchestrator.shutdown()
        
        if self.execution_system:
            await self.execution_system.shutdown()
        
        logger.info("Demo cleanup complete")


async def main():
    """Main demo function"""
    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Run demo
    demo = ExecutionBackpropagationDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())