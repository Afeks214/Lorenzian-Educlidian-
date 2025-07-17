"""
Comprehensive Validation Suite for Enhanced Centralized Critic
Agent 3 - The Learning Optimization Specialist

This module provides comprehensive validation and testing for the enhanced
centralized critic and MAPPO training pipeline, ensuring improved performance
and convergence.

Features:
- Value function accuracy validation
- Convergence speed benchmarking
- Uncertainty calibration testing
- Attention mechanism analysis
- Backward compatibility verification
- Performance regression testing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import structlog
from datetime import datetime
import time
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

from .enhanced_centralized_critic import (
    EnhancedCentralizedCritic, 
    EnhancedCombinedState, 
    SuperpositionFeatures,
    create_enhanced_centralized_critic
)
from .enhanced_mappo_trainer import (
    EnhancedMAPPOTrainer,
    EnhancedMAPPOConfig,
    create_enhanced_mappo_trainer
)

logger = structlog.get_logger()


@dataclass
class ValidationResults:
    """Container for validation results"""
    accuracy_metrics: Dict[str, float]
    convergence_metrics: Dict[str, float]
    uncertainty_metrics: Dict[str, float]
    attention_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    compatibility_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'accuracy_metrics': self.accuracy_metrics,
            'convergence_metrics': self.convergence_metrics,
            'uncertainty_metrics': self.uncertainty_metrics,
            'attention_metrics': self.attention_metrics,
            'performance_metrics': self.performance_metrics,
            'compatibility_metrics': self.compatibility_metrics
        }
    
    def save(self, filepath: str):
        """Save validation results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info("Validation results saved", filepath=filepath)


class ValueFunctionAccuracyValidator:
    """Validates value function accuracy improvements"""
    
    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device
        
    def generate_synthetic_data(self, num_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic data for value function testing"""
        # Generate base features (102D)
        base_features = torch.randn(num_samples, 102)
        
        # Generate superposition features (10D)
        superposition_features = torch.randn(num_samples, 10)
        
        # Combine features
        combined_features = torch.cat([base_features, superposition_features], dim=1)
        
        # Generate ground truth values with complex dependencies
        # Simulate realistic value function dependencies
        market_factor = base_features[:, :32].mean(dim=1) * 0.3  # Market features
        execution_factor = base_features[:, 32:47].mean(dim=1) * 0.2  # Execution features
        routing_factor = base_features[:, 47:102].mean(dim=1) * 0.1  # Routing features
        
        # Superposition contributions
        superposition_factor = superposition_features.mean(dim=1) * 0.4
        
        # Non-linear interactions
        interaction_factor = torch.sigmoid(market_factor * execution_factor) * 0.1
        
        # Ground truth values
        true_values = (market_factor + execution_factor + routing_factor + 
                      superposition_factor + interaction_factor)
        
        return combined_features.to(self.device), true_values.to(self.device)
    
    def validate_accuracy(self, 
                         enhanced_critic: EnhancedCentralizedCritic,
                         baseline_critic: Optional[nn.Module] = None) -> Dict[str, float]:
        """Validate value function accuracy"""
        logger.info("Starting value function accuracy validation")
        
        # Generate test data
        test_features, true_values = self.generate_synthetic_data(1000)
        
        # Test enhanced critic
        enhanced_critic.eval()
        with torch.no_grad():
            if enhanced_critic.use_uncertainty:
                enhanced_predictions, enhanced_uncertainties = enhanced_critic(test_features)
            else:
                enhanced_predictions = enhanced_critic(test_features)
                enhanced_uncertainties = torch.zeros_like(enhanced_predictions)
        
        enhanced_predictions = enhanced_predictions.squeeze().cpu().numpy()
        true_values_np = true_values.cpu().numpy()
        
        # Compute accuracy metrics for enhanced critic
        enhanced_mse = mean_squared_error(true_values_np, enhanced_predictions)
        enhanced_mae = mean_absolute_error(true_values_np, enhanced_predictions)
        enhanced_r2 = r2_score(true_values_np, enhanced_predictions)
        enhanced_corr, _ = pearsonr(true_values_np, enhanced_predictions)
        
        metrics = {
            'enhanced_mse': enhanced_mse,
            'enhanced_mae': enhanced_mae,
            'enhanced_r2': enhanced_r2,
            'enhanced_correlation': enhanced_corr,
            'enhanced_uncertainty_mean': enhanced_uncertainties.mean().item(),
            'enhanced_uncertainty_std': enhanced_uncertainties.std().item()
        }
        
        # Compare with baseline if provided
        if baseline_critic is not None:
            baseline_critic.eval()
            with torch.no_grad():
                # Handle different input dimensions
                if hasattr(baseline_critic, 'input_dim'):
                    baseline_input = test_features[:, :baseline_critic.input_dim]
                else:
                    baseline_input = test_features[:, :102]  # Assume 102D baseline
                
                baseline_predictions = baseline_critic(baseline_input)
                if isinstance(baseline_predictions, tuple):
                    baseline_predictions = baseline_predictions[0]
            
            baseline_predictions = baseline_predictions.squeeze().cpu().numpy()
            
            # Compute baseline metrics
            baseline_mse = mean_squared_error(true_values_np, baseline_predictions)
            baseline_mae = mean_absolute_error(true_values_np, baseline_predictions)
            baseline_r2 = r2_score(true_values_np, baseline_predictions)
            baseline_corr, _ = pearsonr(true_values_np, baseline_predictions)
            
            metrics.update({
                'baseline_mse': baseline_mse,
                'baseline_mae': baseline_mae,
                'baseline_r2': baseline_r2,
                'baseline_correlation': baseline_corr,
                'improvement_mse': (baseline_mse - enhanced_mse) / baseline_mse,
                'improvement_mae': (baseline_mae - enhanced_mae) / baseline_mae,
                'improvement_r2': enhanced_r2 - baseline_r2,
                'improvement_correlation': enhanced_corr - baseline_corr
            })
        
        logger.info("Value function accuracy validation completed", metrics=metrics)
        return metrics


class ConvergenceSpeedBenchmark:
    """Benchmarks convergence speed improvements"""
    
    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device
        
    def create_training_environment(self) -> Dict[str, Any]:
        """Create synthetic training environment"""
        class DummyAgent(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(112, 3)
                
            def forward(self, x):
                return {'action_logits': self.linear(x)}
        
        agents = {
            f'agent_{i}': DummyAgent() for i in range(3)
        }
        
        return agents
    
    def benchmark_convergence(self, 
                            enhanced_trainer: EnhancedMAPPOTrainer,
                            baseline_trainer: Optional[Any] = None,
                            num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark convergence speed"""
        logger.info("Starting convergence speed benchmark")
        
        # Generate training data
        batch_size = 64
        training_data = {
            'observations': torch.randn(batch_size, 112),
            'actions': torch.randint(0, 3, (batch_size,)),
            'log_probs': torch.randn(batch_size),
            'values': torch.randn(batch_size),
            'rewards': torch.randn(batch_size),
            'masks': torch.ones(batch_size)
        }
        
        # Track convergence metrics
        enhanced_losses = []
        enhanced_times = []
        
        # Benchmark enhanced trainer
        start_time = time.time()
        for iteration in range(num_iterations):
            iter_start = time.time()
            
            metrics = enhanced_trainer.train_step(training_data)
            
            iter_end = time.time()
            enhanced_times.append(iter_end - iter_start)
            
            # Extract loss
            policy_losses = [v for k, v in metrics.items() if 'policy_loss' in k]
            value_losses = [v for k, v in metrics.items() if 'value_loss' in k]
            
            avg_policy_loss = np.mean(policy_losses) if policy_losses else 0.0
            avg_value_loss = np.mean(value_losses) if value_losses else 0.0
            
            enhanced_losses.append(avg_policy_loss + avg_value_loss)
        
        total_time = time.time() - start_time
        
        # Compute convergence metrics
        final_loss = np.mean(enhanced_losses[-10:])  # Average of last 10 iterations
        initial_loss = np.mean(enhanced_losses[:10])   # Average of first 10 iterations
        
        convergence_rate = (initial_loss - final_loss) / initial_loss if initial_loss != 0 else 0
        
        # Stability metrics
        loss_stability = np.std(enhanced_losses[-20:]) if len(enhanced_losses) >= 20 else np.std(enhanced_losses)
        
        metrics = {
            'enhanced_final_loss': final_loss,
            'enhanced_initial_loss': initial_loss,
            'enhanced_convergence_rate': convergence_rate,
            'enhanced_loss_stability': loss_stability,
            'enhanced_avg_iteration_time': np.mean(enhanced_times),
            'enhanced_total_time': total_time,
            'enhanced_iterations_per_second': num_iterations / total_time
        }
        
        logger.info("Convergence speed benchmark completed", metrics=metrics)
        return metrics


class UncertaintyCalibrationTester:
    """Tests uncertainty calibration and quality"""
    
    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device
        
    def test_uncertainty_calibration(self, 
                                   enhanced_critic: EnhancedCentralizedCritic,
                                   num_samples: int = 1000) -> Dict[str, float]:
        """Test uncertainty calibration quality"""
        logger.info("Starting uncertainty calibration test")
        
        if not enhanced_critic.use_uncertainty:
            logger.warning("Critic does not use uncertainty estimation")
            return {}
        
        # Generate diverse test data
        test_features = torch.randn(num_samples, 112).to(self.device)
        
        # Get predictions with uncertainty
        enhanced_critic.eval()
        with torch.no_grad():
            predictions, uncertainties = enhanced_critic(test_features)
        
        predictions = predictions.squeeze().cpu().numpy()
        uncertainties = uncertainties.squeeze().cpu().numpy()
        
        # Calibration metrics
        # Higher uncertainty should correlate with higher prediction variance
        
        # Sort by uncertainty
        sorted_indices = np.argsort(uncertainties)
        sorted_predictions = predictions[sorted_indices]
        sorted_uncertainties = uncertainties[sorted_indices]
        
        # Split into quantiles
        n_quantiles = 10
        quantile_size = len(sorted_predictions) // n_quantiles
        
        quantile_variances = []
        quantile_uncertainties = []
        
        for i in range(n_quantiles):
            start_idx = i * quantile_size
            end_idx = min((i + 1) * quantile_size, len(sorted_predictions))
            
            quantile_preds = sorted_predictions[start_idx:end_idx]
            quantile_uncs = sorted_uncertainties[start_idx:end_idx]
            
            quantile_variances.append(np.var(quantile_preds))
            quantile_uncertainties.append(np.mean(quantile_uncs))
        
        # Correlation between uncertainty and prediction variance
        uncertainty_variance_corr, _ = pearsonr(quantile_uncertainties, quantile_variances)
        
        # Sharpness (average uncertainty)
        sharpness = np.mean(uncertainties)
        
        # Dispersion (uncertainty distribution)
        uncertainty_std = np.std(uncertainties)
        
        metrics = {
            'uncertainty_variance_correlation': uncertainty_variance_corr,
            'uncertainty_sharpness': sharpness,
            'uncertainty_dispersion': uncertainty_std,
            'uncertainty_range': np.max(uncertainties) - np.min(uncertainties),
            'uncertainty_mean': np.mean(uncertainties),
            'uncertainty_median': np.median(uncertainties)
        }
        
        logger.info("Uncertainty calibration test completed", metrics=metrics)
        return metrics


class AttentionMechanismAnalyzer:
    """Analyzes attention mechanism behavior and effectiveness"""
    
    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device
        
    def analyze_attention_patterns(self, 
                                 enhanced_critic: EnhancedCentralizedCritic,
                                 num_samples: int = 500) -> Dict[str, float]:
        """Analyze attention mechanism patterns"""
        logger.info("Starting attention mechanism analysis")
        
        # Generate test data
        test_features = torch.randn(num_samples, 112).to(self.device)
        
        # Forward pass to generate attention weights
        enhanced_critic.eval()
        with torch.no_grad():
            predictions = enhanced_critic(test_features)
        
        # Extract attention weights
        if hasattr(enhanced_critic, 'superposition_attention'):
            attention_weights = enhanced_critic.superposition_attention.attention_weights
        else:
            logger.warning("No attention weights found")
            return {}
        
        if attention_weights is None:
            logger.warning("Attention weights are None")
            return {}
        
        attention_weights = attention_weights.cpu().numpy()
        
        # Attention analysis
        # 1. Entropy - how diverse are the attention patterns
        attention_entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-8), axis=-1)
        
        # 2. Concentration - how focused is the attention
        attention_max = np.max(attention_weights, axis=-1)
        
        # 3. Consistency - how consistent are attention patterns
        attention_std = np.std(attention_weights, axis=0)
        
        # 4. Diversity - how much do different samples vary in attention
        sample_diversity = np.std(attention_weights, axis=0)
        
        metrics = {
            'attention_entropy_mean': np.mean(attention_entropy),
            'attention_entropy_std': np.std(attention_entropy),
            'attention_concentration_mean': np.mean(attention_max),
            'attention_concentration_std': np.std(attention_max),
            'attention_consistency': np.mean(attention_std),
            'attention_diversity': np.mean(sample_diversity),
            'attention_distribution_uniformity': 1.0 - np.std(np.mean(attention_weights, axis=0))
        }
        
        logger.info("Attention mechanism analysis completed", metrics=metrics)
        return metrics


class PerformanceRegressionTester:
    """Tests for performance regressions"""
    
    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device
        
    def test_inference_performance(self, 
                                 enhanced_critic: EnhancedCentralizedCritic,
                                 num_samples: int = 1000,
                                 num_iterations: int = 100) -> Dict[str, float]:
        """Test inference performance"""
        logger.info("Starting inference performance test")
        
        # Generate test data
        test_features = torch.randn(num_samples, 112).to(self.device)
        
        # Warmup
        enhanced_critic.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = enhanced_critic(test_features)
        
        # Benchmark inference
        times = []
        for _ in range(num_iterations):
            start_time = time.time()
            with torch.no_grad():
                _ = enhanced_critic(test_features)
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Compute performance metrics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        samples_per_second = num_samples / avg_time
        
        metrics = {
            'avg_inference_time': avg_time,
            'std_inference_time': std_time,
            'min_inference_time': min_time,
            'max_inference_time': max_time,
            'samples_per_second': samples_per_second,
            'time_per_sample_ms': (avg_time / num_samples) * 1000
        }
        
        logger.info("Inference performance test completed", metrics=metrics)
        return metrics


class BackwardCompatibilityValidator:
    """Validates backward compatibility with existing systems"""
    
    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device
        
    def validate_compatibility(self, 
                             enhanced_critic: EnhancedCentralizedCritic) -> Dict[str, float]:
        """Validate backward compatibility"""
        logger.info("Starting backward compatibility validation")
        
        # Test 102D input compatibility
        test_features_102d = torch.randn(100, 102).to(self.device)
        
        try:
            enhanced_critic.eval()
            with torch.no_grad():
                predictions_102d = enhanced_critic(test_features_102d)
            
            if isinstance(predictions_102d, tuple):
                predictions_102d = predictions_102d[0]
            
            compatibility_102d = True
            prediction_shape_valid = predictions_102d.shape[1] == 1
            
        except Exception as e:
            logger.error("102D compatibility test failed", error=str(e))
            compatibility_102d = False
            prediction_shape_valid = False
        
        # Test 112D input
        test_features_112d = torch.randn(100, 112).to(self.device)
        
        try:
            with torch.no_grad():
                predictions_112d = enhanced_critic(test_features_112d)
            
            if isinstance(predictions_112d, tuple):
                predictions_112d = predictions_112d[0]
            
            compatibility_112d = True
            prediction_shape_valid_112d = predictions_112d.shape[1] == 1
            
        except Exception as e:
            logger.error("112D compatibility test failed", error=str(e))
            compatibility_112d = False
            prediction_shape_valid_112d = False
        
        # Test parameter count (should be reasonable)
        total_params = sum(p.numel() for p in enhanced_critic.parameters())
        param_count_reasonable = total_params < 50_000_000  # Less than 50M parameters
        
        metrics = {
            'compatibility_102d': float(compatibility_102d),
            'compatibility_112d': float(compatibility_112d),
            'prediction_shape_valid_102d': float(prediction_shape_valid),
            'prediction_shape_valid_112d': float(prediction_shape_valid_112d),
            'total_parameters': float(total_params),
            'parameter_count_reasonable': float(param_count_reasonable),
            'overall_compatibility_score': float(
                compatibility_102d and compatibility_112d and 
                prediction_shape_valid and prediction_shape_valid_112d and
                param_count_reasonable
            )
        }
        
        logger.info("Backward compatibility validation completed", metrics=metrics)
        return metrics


class ComprehensiveValidationSuite:
    """Comprehensive validation suite for enhanced centralized critic"""
    
    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device
        
        # Initialize validators
        self.accuracy_validator = ValueFunctionAccuracyValidator(device)
        self.convergence_benchmark = ConvergenceSpeedBenchmark(device)
        self.uncertainty_tester = UncertaintyCalibrationTester(device)
        self.attention_analyzer = AttentionMechanismAnalyzer(device)
        self.performance_tester = PerformanceRegressionTester(device)
        self.compatibility_validator = BackwardCompatibilityValidator(device)
        
    def run_full_validation(self, 
                          enhanced_critic: EnhancedCentralizedCritic,
                          enhanced_trainer: Optional[EnhancedMAPPOTrainer] = None,
                          baseline_critic: Optional[nn.Module] = None,
                          baseline_trainer: Optional[Any] = None) -> ValidationResults:
        """Run comprehensive validation suite"""
        logger.info("Starting comprehensive validation suite")
        
        # Run all validation tests
        accuracy_metrics = self.accuracy_validator.validate_accuracy(
            enhanced_critic, baseline_critic
        )
        
        convergence_metrics = {}
        if enhanced_trainer is not None:
            convergence_metrics = self.convergence_benchmark.benchmark_convergence(
                enhanced_trainer, baseline_trainer
            )
        
        uncertainty_metrics = self.uncertainty_tester.test_uncertainty_calibration(
            enhanced_critic
        )
        
        attention_metrics = self.attention_analyzer.analyze_attention_patterns(
            enhanced_critic
        )
        
        performance_metrics = self.performance_tester.test_inference_performance(
            enhanced_critic
        )
        
        compatibility_metrics = self.compatibility_validator.validate_compatibility(
            enhanced_critic
        )
        
        # Create validation results
        results = ValidationResults(
            accuracy_metrics=accuracy_metrics,
            convergence_metrics=convergence_metrics,
            uncertainty_metrics=uncertainty_metrics,
            attention_metrics=attention_metrics,
            performance_metrics=performance_metrics,
            compatibility_metrics=compatibility_metrics
        )
        
        logger.info("Comprehensive validation suite completed")
        return results
    
    def generate_validation_report(self, 
                                 results: ValidationResults,
                                 output_dir: str = "validation_results") -> str:
        """Generate comprehensive validation report"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save results
        results.save(output_path / "validation_results.json")
        
        # Generate report
        report_path = output_path / "validation_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Enhanced Centralized Critic Validation Report\n\n")
            f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Accuracy metrics
            f.write("## Value Function Accuracy\n\n")
            for key, value in results.accuracy_metrics.items():
                f.write(f"- **{key}**: {value:.4f}\n")
            f.write("\n")
            
            # Convergence metrics
            if results.convergence_metrics:
                f.write("## Convergence Speed\n\n")
                for key, value in results.convergence_metrics.items():
                    f.write(f"- **{key}**: {value:.4f}\n")
                f.write("\n")
            
            # Uncertainty metrics
            if results.uncertainty_metrics:
                f.write("## Uncertainty Calibration\n\n")
                for key, value in results.uncertainty_metrics.items():
                    f.write(f"- **{key}**: {value:.4f}\n")
                f.write("\n")
            
            # Attention metrics
            if results.attention_metrics:
                f.write("## Attention Mechanism\n\n")
                for key, value in results.attention_metrics.items():
                    f.write(f"- **{key}**: {value:.4f}\n")
                f.write("\n")
            
            # Performance metrics
            f.write("## Performance\n\n")
            for key, value in results.performance_metrics.items():
                f.write(f"- **{key}**: {value:.4f}\n")
            f.write("\n")
            
            # Compatibility metrics
            f.write("## Backward Compatibility\n\n")
            for key, value in results.compatibility_metrics.items():
                f.write(f"- **{key}**: {value:.4f}\n")
            f.write("\n")
            
            # Summary
            f.write("## Summary\n\n")
            
            # Overall score
            overall_score = (
                results.accuracy_metrics.get('enhanced_r2', 0) * 0.3 +
                results.uncertainty_metrics.get('uncertainty_variance_correlation', 0) * 0.2 +
                results.attention_metrics.get('attention_diversity', 0) * 0.2 +
                (1.0 - results.performance_metrics.get('avg_inference_time', 1.0)) * 0.15 +
                results.compatibility_metrics.get('overall_compatibility_score', 0) * 0.15
            )
            
            f.write(f"**Overall Validation Score**: {overall_score:.4f}\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            
            if results.accuracy_metrics.get('enhanced_r2', 0) < 0.8:
                f.write("- Consider increasing model capacity or adjusting architecture\n")
            
            if results.uncertainty_metrics.get('uncertainty_variance_correlation', 0) < 0.5:
                f.write("- Improve uncertainty calibration through better training\n")
            
            if results.performance_metrics.get('avg_inference_time', 1.0) > 0.1:
                f.write("- Optimize inference performance for production deployment\n")
            
            if results.compatibility_metrics.get('overall_compatibility_score', 0) < 1.0:
                f.write("- Address backward compatibility issues\n")
        
        logger.info("Validation report generated", report_path=str(report_path))
        return str(report_path)


# Factory function
def run_validation_suite(
    critic_config: Dict[str, Any],
    training_config: Dict[str, Any],
    device: torch.device = torch.device('cpu')
) -> ValidationResults:
    """Run comprehensive validation suite"""
    
    # Create enhanced critic
    enhanced_critic = create_enhanced_centralized_critic(critic_config)
    
    # Create enhanced trainer
    agents = {
        f'agent_{i}': nn.Sequential(
            nn.Linear(112, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        ) for i in range(3)
    }
    
    enhanced_trainer = create_enhanced_mappo_trainer(
        agents=agents,
        critic_config=critic_config,
        training_config=training_config,
        device=device
    )
    
    # Run validation
    validation_suite = ComprehensiveValidationSuite(device)
    results = validation_suite.run_full_validation(
        enhanced_critic=enhanced_critic,
        enhanced_trainer=enhanced_trainer
    )
    
    return results


if __name__ == "__main__":
    # Run validation suite
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    critic_config = {
        'base_input_dim': 102,
        'superposition_dim': 10,
        'hidden_dims': [512, 256, 128, 64],
        'use_uncertainty': True,
        'num_ensembles': 5
    }
    
    training_config = {
        'learning_rate': 3e-4,
        'uncertainty_loss_coef': 0.1,
        'adaptive_lr_enabled': True
    }
    
    results = run_validation_suite(
        critic_config=critic_config,
        training_config=training_config,
        device=device
    )
    
    # Generate report
    validation_suite = ComprehensiveValidationSuite(device)
    report_path = validation_suite.generate_validation_report(results)
    
    print(f"Validation completed. Report saved to: {report_path}")
    print(f"Overall validation score: {results.accuracy_metrics.get('enhanced_r2', 0):.4f}")