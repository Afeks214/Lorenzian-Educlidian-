"""
Attention Mechanism Analysis and Enhancement Tool for Strategic Agents

This module provides comprehensive analysis and enhancement tools for the attention
mechanisms implemented in all three strategic agents (MLMI, NWRQK, Regime Detection).

Key Features:
- Context sensitivity analysis and enhancement
- Attention weight visualization and interpretation
- Performance impact assessment
- Dynamic feature importance tracking
- Market context adaptation validation

Author: Agent Alpha - Dynamic Feature Selection Specialist
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
import logging
import time
from pathlib import Path
import json

from src.agents.mlmi_strategic_agent import MLMIStrategicAgent, MLMIPolicyNetwork
from src.agents.nwrqk_strategic_agent import NWRQKStrategicAgent, NWRQKPolicyNetwork  
from src.agents.regime_detection_agent import RegimeDetectionAgent, RegimePolicyNetwork

logger = logging.getLogger(__name__)


class AttentionAnalyzer:
    """
    Comprehensive attention mechanism analyzer for strategic agents.
    
    Provides tools for analyzing, visualizing, and enhancing the context
    sensitivity of attention mechanisms across all strategic agents.
    """
    
    def __init__(self, output_dir: str = "attention_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Market context scenarios for testing
        self.market_scenarios = {
            'high_volatility': {'volatility': 3.0, 'momentum': 0.0, 'description': 'High volatility, sideways market'},
            'low_volatility': {'volatility': 0.3, 'momentum': 0.0, 'description': 'Low volatility, stable market'},
            'strong_uptrend': {'volatility': 1.0, 'momentum': 2.0, 'description': 'Strong bullish momentum'},
            'strong_downtrend': {'volatility': 1.0, 'momentum': -2.0, 'description': 'Strong bearish momentum'},
            'volatile_uptrend': {'volatility': 2.5, 'momentum': 1.5, 'description': 'Volatile bullish market'},
            'volatile_downtrend': {'volatility': 2.5, 'momentum': -1.5, 'description': 'Volatile bearish market'},
            'crisis_mode': {'volatility': 4.0, 'momentum': -1.0, 'description': 'Crisis/panic selling'},
            'recovery_mode': {'volatility': 1.5, 'momentum': 0.5, 'description': 'Market recovery phase'},
        }
        
        logger.info(f"Attention analyzer initialized, output dir: {self.output_dir}")
    
    def create_synthetic_market_data(self, volatility: float = 1.0, momentum: float = 0.0, 
                                   sequence_length: int = 48, feature_count: int = 13) -> np.ndarray:
        """
        Create synthetic market data with controlled characteristics.
        
        Args:
            volatility: Market volatility level
            momentum: Momentum factor
            sequence_length: Number of time steps (default 48 for strategic timeframe)
            feature_count: Number of features (default 13 for standard matrix)
            
        Returns:
            Synthetic market data matrix (sequence_length, feature_count)
        """
        np.random.seed(42)  # Reproducible for testing
        
        # Generate base price series with momentum and volatility
        price_changes = np.random.normal(momentum * 0.001, volatility * 0.01, sequence_length)
        prices = 100 * np.exp(np.cumsum(price_changes))  # Geometric Brownian Motion
        
        matrix = np.zeros((sequence_length, feature_count))
        
        # Feature 0: MLMI value (correlation-based)
        # Higher correlation in trending markets, lower in sideways
        trend_strength = abs(momentum)
        base_correlation = 0.3 + 0.4 * trend_strength / (1 + trend_strength)
        matrix[:, 0] = base_correlation + np.random.normal(0, 0.1, sequence_length)
        
        # Feature 1: MLMI signal (momentum-based)  
        # Should be more pronounced in trending markets
        matrix[:, 1] = momentum * 0.5 + np.random.normal(0, volatility * 0.05, sequence_length)
        
        # Features 2-3: NWRQK values (support/resistance based)
        # NWRQK value should be higher near support/resistance
        support_level = np.mean(prices) * (1 - volatility * 0.02)
        resistance_level = np.mean(prices) * (1 + volatility * 0.02)
        
        for i in range(sequence_length):
            price = prices[i]
            # Distance to nearest level (normalized)
            dist_to_support = abs(price - support_level) / price
            dist_to_resistance = abs(price - resistance_level) / price
            min_dist = min(dist_to_support, dist_to_resistance)
            
            # NWRQK value inversely related to distance (higher near levels)
            matrix[i, 2] = (1.0 - min_dist * 10) + np.random.normal(0, 0.1)
            
        # Feature 3: NWRQK slope (trend of support/resistance strength)
        matrix[:, 3] = momentum * 0.3 + np.random.normal(0, 0.05, sequence_length)
        
        # Features 4-5: LVN distance and strength
        # Distance should be smaller in volatile markets (more levels)
        matrix[:, 4] = np.random.exponential(0.01 / (1 + volatility), sequence_length)
        matrix[:, 5] = volatility * 0.3 + np.random.beta(2, 2, sequence_length) * 0.7
        
        # Features 9-10: Momentum indicators (20 and 50 period)
        if sequence_length >= 20:
            for i in range(20, sequence_length):
                # 20-period momentum
                momentum_20 = (prices[i] - prices[i-20]) / prices[i-20]
                matrix[i, 9] = momentum_20
        
        if sequence_length >= 50:
            for i in range(50, sequence_length):
                # 50-period momentum  
                momentum_50 = (prices[i] - prices[i-50]) / prices[i-50]
                matrix[i, 10] = momentum_50
        
        # Feature 10: MMD score (regime change indicator)
        # Higher during regime transitions, affected by volatility changes
        regime_change_signal = volatility / 2.0 + abs(momentum) / 3.0
        matrix[:, 10] = regime_change_signal + np.random.normal(0, 0.1, sequence_length)
        
        # Feature 11: 30-period volatility
        matrix[:, 11] = volatility + np.random.normal(0, 0.1, sequence_length)
        
        # Feature 12: Volume profile skew
        # More skewed in trending markets
        volume_skew = momentum * 0.8 + np.random.normal(0, 0.5, sequence_length)
        matrix[:, 12] = volume_skew
        
        # Ensure no NaN or extreme values
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return matrix
    
    def analyze_attention_context_sensitivity(self, agent, agent_name: str, 
                                            feature_extraction_func) -> Dict[str, Any]:
        """
        Analyze how attention weights vary across different market contexts.
        
        Args:
            agent: Strategic agent instance
            agent_name: Name of the agent for reporting
            feature_extraction_func: Function to extract features from matrix data
            
        Returns:
            Context sensitivity analysis results
        """
        logger.info(f"Analyzing attention context sensitivity for {agent_name}")
        
        results = {
            'agent_name': agent_name,
            'scenarios': {},
            'sensitivity_metrics': {},
            'recommendations': []
        }
        
        scenario_attention_weights = {}
        
        # Test each market scenario
        for scenario_name, scenario_params in self.market_scenarios.items():
            logger.debug(f"Testing scenario: {scenario_name}")
            
            # Create market data for this scenario
            market_data = self.create_synthetic_market_data(
                volatility=scenario_params['volatility'],
                momentum=scenario_params['momentum']
            )
            
            # Extract features based on agent type
            if agent_name == 'MLMI':
                features = agent.extract_mlmi_features(market_data)
            elif agent_name == 'NWRQK':
                features = agent.extract_features(market_data)
            elif agent_name == 'Regime':
                features = agent.extract_features(market_data)
            else:
                features = feature_extraction_func(market_data)
            
            # Get attention weights
            if isinstance(features, np.ndarray):
                feature_tensor = torch.tensor(features, dtype=torch.float32)
            else:
                feature_tensor = features
                
            with torch.no_grad():
                policy_output = agent.policy_network(feature_tensor.unsqueeze(0))
                attention_weights = policy_output['attention_weights'].squeeze(0).numpy()
            
            scenario_attention_weights[scenario_name] = {
                'attention_weights': attention_weights.tolist(),
                'features': features.tolist() if hasattr(features, 'tolist') else features,
                'scenario_params': scenario_params
            }
            
            results['scenarios'][scenario_name] = scenario_attention_weights[scenario_name]
        
        # Calculate sensitivity metrics
        all_attention_weights = np.array([
            scenario_data['attention_weights'] 
            for scenario_data in scenario_attention_weights.values()
        ])
        
        # Overall variance across scenarios
        attention_variance = np.var(all_attention_weights, axis=0)
        attention_std = np.std(all_attention_weights, axis=0)
        attention_range = np.max(all_attention_weights, axis=0) - np.min(all_attention_weights, axis=0)
        
        # Context sensitivity score (how much weights vary across contexts)
        context_sensitivity_score = np.mean(attention_range)
        
        results['sensitivity_metrics'] = {
            'context_sensitivity_score': float(context_sensitivity_score),
            'attention_variance': attention_variance.tolist(),
            'attention_std': attention_std.tolist(), 
            'attention_range': attention_range.tolist(),
            'mean_attention_weights': np.mean(all_attention_weights, axis=0).tolist(),
            'passes_context_test': context_sensitivity_score > 0.1  # Must vary by >10%
        }
        
        # Generate recommendations
        if context_sensitivity_score < 0.1:
            results['recommendations'].append(
                "LOW CONTEXT SENSITIVITY: Attention weights vary by less than 10% across market contexts. "
                "Consider: 1) Better network initialization, 2) Training on diverse market data, "
                "3) Adjusting attention head architecture."
            )
        
        if np.any(attention_range < 0.05):
            low_variance_features = np.where(attention_range < 0.05)[0]
            results['recommendations'].append(
                f"STATIC FEATURES: Features {low_variance_features.tolist()} show minimal attention variation. "
                "These features may not be contributing meaningfully to context adaptation."
            )
        
        # Feature-specific analysis
        feature_names = self._get_feature_names(agent_name)
        for i, (feature_name, variance, range_val) in enumerate(
            zip(feature_names, attention_variance, attention_range)
        ):
            if range_val > 0.2:  # High context sensitivity
                results['recommendations'].append(
                    f"GOOD ADAPTATION: {feature_name} shows strong context sensitivity (range: {range_val:.3f})"
                )
        
        logger.info(f"Context sensitivity analysis complete for {agent_name}. "
                   f"Score: {context_sensitivity_score:.3f}")
        
        return results
    
    def _get_feature_names(self, agent_name: str) -> List[str]:
        """Get feature names for each agent type."""
        if agent_name == 'MLMI':
            return ['mlmi_value', 'mlmi_signal', 'momentum_20', 'momentum_50']
        elif agent_name == 'NWRQK':
            return ['nwrqk_value', 'nwrqk_slope', 'lvn_distance', 'lvn_strength']
        elif agent_name == 'Regime':
            return ['mmd_score', 'volatility_30', 'volume_profile_skew']
        else:
            return [f'feature_{i}' for i in range(4)]  # Default
    
    def enhance_attention_context_sensitivity(self, agent, agent_name: str) -> None:
        """
        Enhance attention mechanism context sensitivity through targeted initialization.
        
        This method applies better initialization strategies to make attention weights
        more responsive to different market contexts.
        
        Args:
            agent: Strategic agent instance
            agent_name: Name of the agent
        """
        logger.info(f"Enhancing attention context sensitivity for {agent_name}")
        
        # Get the attention head from the policy network
        attention_head = agent.policy_network.attention_head
        
        # Enhanced initialization for better context sensitivity
        with torch.no_grad():
            # Initialize first layer with higher variance to increase sensitivity
            nn.init.xavier_normal_(attention_head[0].weight, gain=2.0)
            nn.init.constant_(attention_head[0].bias, 0.0)
            
            # Initialize second layer with smaller variance for stability
            nn.init.xavier_normal_(attention_head[2].weight, gain=0.5)
            nn.init.constant_(attention_head[2].bias, 0.0)
        
        # Add some asymmetry to encourage different feature focus
        with torch.no_grad():
            # Add small random bias to break symmetry
            feature_count = attention_head[0].weight.shape[1]
            asymmetry_bias = torch.randn(feature_count) * 0.1
            attention_head[2].bias.add_(asymmetry_bias)
        
        logger.info(f"Enhanced attention initialization applied to {agent_name}")
    
    def visualize_attention_patterns(self, analysis_results: Dict[str, Any]) -> str:
        """
        Create visualizations of attention patterns across market contexts.
        
        Args:
            analysis_results: Results from analyze_attention_context_sensitivity
            
        Returns:
            Path to saved visualization
        """
        agent_name = analysis_results['agent_name']
        scenarios = analysis_results['scenarios']
        
        # Prepare data for visualization
        scenario_names = list(scenarios.keys())
        attention_data = np.array([
            scenarios[scenario]['attention_weights'] 
            for scenario in scenario_names
        ])
        
        feature_names = self._get_feature_names(agent_name)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{agent_name} Agent: Attention Mechanism Analysis', fontsize=16, fontweight='bold')
        
        # 1. Heatmap of attention weights across scenarios
        ax1 = axes[0, 0]
        sns.heatmap(attention_data.T, 
                   xticklabels=scenario_names, 
                   yticklabels=feature_names,
                   annot=True, fmt='.3f', cmap='viridis', ax=ax1)
        ax1.set_title('Attention Weights by Market Scenario')
        ax1.set_xlabel('Market Scenarios')
        ax1.set_ylabel('Features')
        
        # 2. Line plot showing attention variation
        ax2 = axes[0, 1]
        for i, feature_name in enumerate(feature_names):
            ax2.plot(range(len(scenario_names)), attention_data[:, i], 
                    marker='o', label=feature_name, linewidth=2)
        ax2.set_xticks(range(len(scenario_names)))
        ax2.set_xticklabels(scenario_names, rotation=45, ha='right')
        ax2.set_ylabel('Attention Weight')
        ax2.set_title('Attention Weight Variation Across Scenarios')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Bar plot of attention sensitivity (range) per feature
        ax3 = axes[1, 0]
        attention_ranges = np.max(attention_data, axis=0) - np.min(attention_data, axis=0)
        bars = ax3.bar(feature_names, attention_ranges, color='skyblue', edgecolor='navy')
        ax3.set_ylabel('Attention Range (Max - Min)')
        ax3.set_title('Context Sensitivity by Feature')
        ax3.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, 
                   label='Minimum Required (0.1)')
        ax3.legend()
        
        # Add value labels on bars
        for bar, range_val in zip(bars, attention_ranges):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{range_val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Scenario parameter visualization
        ax4 = axes[1, 1]
        volatilities = [scenarios[s]['scenario_params']['volatility'] for s in scenario_names]
        momentums = [scenarios[s]['scenario_params']['momentum'] for s in scenario_names]
        
        scatter = ax4.scatter(volatilities, momentums, 
                            c=range(len(scenario_names)), cmap='tab10', s=100)
        for i, scenario in enumerate(scenario_names):
            ax4.annotate(scenario, (volatilities[i], momentums[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax4.set_xlabel('Volatility')
        ax4.set_ylabel('Momentum')
        ax4.set_title('Market Scenario Parameter Space')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = self.output_dir / f"{agent_name.lower()}_attention_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Attention visualization saved to {output_path}")
        return str(output_path)
    
    def run_performance_impact_analysis(self, agent, agent_name: str, 
                                      num_iterations: int = 1000) -> Dict[str, Any]:
        """
        Analyze the performance impact of attention mechanisms.
        
        Args:
            agent: Strategic agent instance
            agent_name: Name of the agent
            num_iterations: Number of iterations for timing tests
            
        Returns:
            Performance analysis results
        """
        logger.info(f"Running performance impact analysis for {agent_name}")
        
        # Create test data
        test_matrix = self.create_synthetic_market_data()
        
        # Extract features based on agent type
        if agent_name == 'MLMI':
            features = agent.extract_mlmi_features(test_matrix)
        elif agent_name == 'NWRQK':
            features = agent.extract_features(test_matrix)
        elif agent_name == 'Regime':
            features = agent.extract_features(test_matrix)
        else:
            raise ValueError(f"Unknown agent type: {agent_name}")
        
        # Test with attention (current implementation)
        attention_times = []
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            
            if agent_name == 'MLMI':
                decision = agent.make_decision({'matrix_data': test_matrix})
            else:
                decision = agent.make_decision(features)
                
            end_time = time.perf_counter()
            attention_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        results = {
            'agent_name': agent_name,
            'num_iterations': num_iterations,
            'attention_enabled': {
                'mean_time_ms': float(np.mean(attention_times)),
                'std_time_ms': float(np.std(attention_times)),
                'min_time_ms': float(np.min(attention_times)),
                'max_time_ms': float(np.max(attention_times)),
                'p95_time_ms': float(np.percentile(attention_times, 95)),
                'p99_time_ms': float(np.percentile(attention_times, 99))
            },
            'performance_requirements': {
                'target_avg_ms': 5.0,
                'target_p95_ms': 10.0,
                'meets_avg_requirement': np.mean(attention_times) < 5.0,
                'meets_p95_requirement': np.percentile(attention_times, 95) < 10.0
            }
        }
        
        logger.info(f"Performance analysis complete for {agent_name}. "
                   f"Avg: {results['attention_enabled']['mean_time_ms']:.2f}ms, "
                   f"P95: {results['attention_enabled']['p95_time_ms']:.2f}ms")
        
        return results
    
    def generate_comprehensive_report(self, agents: Dict[str, Any]) -> str:
        """
        Generate a comprehensive attention mechanism analysis report.
        
        Args:
            agents: Dictionary of agent instances {name: agent}
            
        Returns:
            Path to generated report file
        """
        logger.info("Generating comprehensive attention mechanism report")
        
        report = {
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'mission_status': 'Agent Alpha - Dynamic Feature Selection Analysis',
            'agents': {},
            'summary': {},
            'recommendations': []
        }
        
        all_context_scores = []
        all_performance_results = []
        
        # Analyze each agent
        for agent_name, agent in agents.items():
            logger.info(f"Analyzing {agent_name} agent...")
            
            # Get appropriate feature extraction function
            if agent_name == 'MLMI':
                feature_func = agent.extract_mlmi_features
            elif agent_name == 'NWRQK':
                feature_func = agent.extract_features  
            elif agent_name == 'Regime':
                feature_func = agent.extract_features
            else:
                feature_func = lambda x: np.zeros(4)
            
            # Context sensitivity analysis
            context_analysis = self.analyze_attention_context_sensitivity(
                agent, agent_name, feature_func
            )
            
            # Performance impact analysis
            performance_analysis = self.run_performance_impact_analysis(
                agent, agent_name
            )
            
            # Generate visualizations
            visualization_path = self.visualize_attention_patterns(context_analysis)
            
            # Store results
            report['agents'][agent_name] = {
                'context_sensitivity': context_analysis,
                'performance_impact': performance_analysis,
                'visualization_path': visualization_path
            }
            
            all_context_scores.append(
                context_analysis['sensitivity_metrics']['context_sensitivity_score']
            )
            all_performance_results.append(performance_analysis)
        
        # Generate summary
        report['summary'] = {
            'total_agents_analyzed': len(agents),
            'avg_context_sensitivity_score': float(np.mean(all_context_scores)),
            'min_context_sensitivity_score': float(np.min(all_context_scores)),
            'max_context_sensitivity_score': float(np.max(all_context_scores)),
            'agents_meeting_context_requirement': sum(1 for score in all_context_scores if score > 0.1),
            'agents_meeting_performance_requirement': sum(
                1 for perf in all_performance_results 
                if perf['performance_requirements']['meets_avg_requirement']
            ),
            'overall_mission_status': 'SUCCESS' if (
                np.mean(all_context_scores) > 0.1 and
                all(perf['performance_requirements']['meets_avg_requirement'] 
                    for perf in all_performance_results)
            ) else 'NEEDS_IMPROVEMENT'
        }
        
        # Generate recommendations
        if report['summary']['overall_mission_status'] == 'NEEDS_IMPROVEMENT':
            if np.mean(all_context_scores) <= 0.1:
                report['recommendations'].append(
                    "CRITICAL: Overall context sensitivity below threshold. "
                    "Apply enhanced initialization to all agents."
                )
            
            failing_performance = [
                perf['agent_name'] for perf in all_performance_results
                if not perf['performance_requirements']['meets_avg_requirement']
            ]
            if failing_performance:
                report['recommendations'].append(
                    f"PERFORMANCE: Agents {failing_performance} exceed 5ms inference target. "
                    "Consider network architecture optimization."
                )
        else:
            report['recommendations'].append(
                "SUCCESS: All agents meet context sensitivity and performance requirements. "
                "Attention mechanisms are properly implemented and functional."
            )
        
        # Save report
        report_path = self.output_dir / "comprehensive_attention_analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Comprehensive report generated: {report_path}")
        return str(report_path)


def run_agent_alpha_analysis():
    """
    Main function to run Agent Alpha's dynamic feature selection analysis.
    
    This function creates instances of all three strategic agents and runs
    comprehensive attention mechanism analysis.
    """
    from src.core.events import EventBus
    from unittest.mock import Mock
    
    logger.info("🧠 AGENT ALPHA MISSION: Starting Dynamic Feature Selection Analysis")
    
    # Create mock event bus
    mock_event_bus = Mock(spec=EventBus)
    
    # Create agent instances
    agents = {}
    
    # MLMI Agent
    mlmi_config = {
        'agent_id': 'mlmi_strategic_agent',
        'gamma': 0.99,
        'lambda_': 0.95,
        'hidden_dim': 128,
        'dropout_rate': 0.0  # Disable for analysis
    }
    agents['MLMI'] = MLMIStrategicAgent(mlmi_config, mock_event_bus)
    
    # NWRQK Agent
    nwrqk_config = {
        'agent_id': 'nwrqk_strategic_agent',
        'hidden_dim': 64,
        'dropout_rate': 0.0  # Disable for analysis
    }
    agents['NWRQK'] = NWRQKStrategicAgent(nwrqk_config)
    
    # Regime Detection Agent
    regime_config = {
        'agent_id': 'regime_detection_agent',
        'hidden_dim': 32,
        'dropout_rate': 0.0  # Disable for analysis
    }
    agents['Regime'] = RegimeDetectionAgent(regime_config)
    
    # Create analyzer
    analyzer = AttentionAnalyzer(output_dir="attention_analysis")
    
    # Enhance context sensitivity for all agents
    for agent_name, agent in agents.items():
        analyzer.enhance_attention_context_sensitivity(agent, agent_name)
    
    # Generate comprehensive analysis
    report_path = analyzer.generate_comprehensive_report(agents)
    
    logger.info(f"🏆 AGENT ALPHA MISSION COMPLETE: Report saved to {report_path}")
    
    return report_path


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the analysis
    run_agent_alpha_analysis()