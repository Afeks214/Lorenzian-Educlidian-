"""
Superposition Factory for MARL Agent Superposition Management.

This module provides a factory pattern for creating appropriate superposition types
based on agent specifications and requirements, with advanced configuration and
optimization capabilities.
"""

from typing import Dict, Any, Optional, Type, List, Union
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
import structlog
from datetime import datetime

from .base_superposition import UniversalSuperposition
from .strategic_superpositions import (
    MLMISuperposition, 
    NWRQKSuperposition, 
    RegimeSuperposition
)
from .tactical_superpositions import (
    FVGSuperposition, 
    MomentumSuperposition, 
    EntryOptSuperposition
)
from .risk_superpositions import (
    PositionSizingSuperposition,
    StopTargetSuperposition,
    RiskMonitorSuperposition,
    PortfolioOptimizerSuperposition
)
from .execution_superpositions import (
    ExecutionTimingSuperposition,
    RoutingSuperposition,
    RiskManagementSuperposition
)

logger = structlog.get_logger()


class SuperpositionType(Enum):
    """Supported superposition types"""
    # Strategic
    MLMI = "mlmi"
    NWRQK = "nwrqk"
    REGIME = "regime"
    
    # Tactical
    FVG = "fvg"
    MOMENTUM = "momentum"
    ENTRY_OPT = "entry_opt"
    
    # Risk Management
    POSITION_SIZING = "position_sizing"
    STOP_TARGET = "stop_target"
    RISK_MONITOR = "risk_monitor"
    PORTFOLIO_OPTIMIZER = "portfolio_optimizer"
    
    # Execution
    EXECUTION_TIMING = "execution_timing"
    ROUTING = "routing"
    EXECUTION_RISK_MANAGEMENT = "execution_risk_management"


class SuperpositionCategory(Enum):
    """Superposition categories"""
    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    RISK_MANAGEMENT = "risk_management"
    EXECUTION = "execution"


class SuperpositionRegistry:
    """Registry for managing superposition types and their configurations"""
    
    def __init__(self):
        self._registry: Dict[SuperpositionType, Type[UniversalSuperposition]] = {
            # Strategic superpositions
            SuperpositionType.MLMI: MLMISuperposition,
            SuperpositionType.NWRQK: NWRQKSuperposition,
            SuperpositionType.REGIME: RegimeSuperposition,
            
            # Tactical superpositions
            SuperpositionType.FVG: FVGSuperposition,
            SuperpositionType.MOMENTUM: MomentumSuperposition,
            SuperpositionType.ENTRY_OPT: EntryOptSuperposition,
            
            # Risk management superpositions
            SuperpositionType.POSITION_SIZING: PositionSizingSuperposition,
            SuperpositionType.STOP_TARGET: StopTargetSuperposition,
            SuperpositionType.RISK_MONITOR: RiskMonitorSuperposition,
            SuperpositionType.PORTFOLIO_OPTIMIZER: PortfolioOptimizerSuperposition,
            
            # Execution superpositions
            SuperpositionType.EXECUTION_TIMING: ExecutionTimingSuperposition,
            SuperpositionType.ROUTING: RoutingSuperposition,
            SuperpositionType.EXECUTION_RISK_MANAGEMENT: RiskManagementSuperposition,
        }
        
        self._category_mapping: Dict[SuperpositionType, SuperpositionCategory] = {
            # Strategic
            SuperpositionType.MLMI: SuperpositionCategory.STRATEGIC,
            SuperpositionType.NWRQK: SuperpositionCategory.STRATEGIC,
            SuperpositionType.REGIME: SuperpositionCategory.STRATEGIC,
            
            # Tactical
            SuperpositionType.FVG: SuperpositionCategory.TACTICAL,
            SuperpositionType.MOMENTUM: SuperpositionCategory.TACTICAL,
            SuperpositionType.ENTRY_OPT: SuperpositionCategory.TACTICAL,
            
            # Risk Management
            SuperpositionType.POSITION_SIZING: SuperpositionCategory.RISK_MANAGEMENT,
            SuperpositionType.STOP_TARGET: SuperpositionCategory.RISK_MANAGEMENT,
            SuperpositionType.RISK_MONITOR: SuperpositionCategory.RISK_MANAGEMENT,
            SuperpositionType.PORTFOLIO_OPTIMIZER: SuperpositionCategory.RISK_MANAGEMENT,
            
            # Execution
            SuperpositionType.EXECUTION_TIMING: SuperpositionCategory.EXECUTION,
            SuperpositionType.ROUTING: SuperpositionCategory.EXECUTION,
            SuperpositionType.EXECUTION_RISK_MANAGEMENT: SuperpositionCategory.EXECUTION,
        }
        
        self._default_configs: Dict[SuperpositionType, Dict[str, Any]] = self._initialize_default_configs()
    
    def get_superposition_class(self, superposition_type: SuperpositionType) -> Type[UniversalSuperposition]:
        """Get superposition class for given type"""
        return self._registry.get(superposition_type)
    
    def get_category(self, superposition_type: SuperpositionType) -> SuperpositionCategory:
        """Get category for given superposition type"""
        return self._category_mapping.get(superposition_type)
    
    def get_default_config(self, superposition_type: SuperpositionType) -> Dict[str, Any]:
        """Get default configuration for given superposition type"""
        return self._default_configs.get(superposition_type, {}).copy()
    
    def get_types_by_category(self, category: SuperpositionCategory) -> List[SuperpositionType]:
        """Get all superposition types in a category"""
        return [
            stype for stype, scat in self._category_mapping.items()
            if scat == category
        ]
    
    def register_superposition(self, 
                             superposition_type: SuperpositionType,
                             superposition_class: Type[UniversalSuperposition],
                             category: SuperpositionCategory,
                             default_config: Optional[Dict[str, Any]] = None) -> None:
        """Register a new superposition type"""
        self._registry[superposition_type] = superposition_class
        self._category_mapping[superposition_type] = category
        if default_config:
            self._default_configs[superposition_type] = default_config
    
    def _initialize_default_configs(self) -> Dict[SuperpositionType, Dict[str, Any]]:
        """Initialize default configurations for all superposition types"""
        return {
            # Strategic configurations
            SuperpositionType.MLMI: {
                'coherence_decay_rate': 0.001,
                'entanglement_strength': 0.6,
                'volume_analysis_weight': 0.3,
                'liquidity_analysis_weight': 0.25,
                'momentum_analysis_weight': 0.3,
                'pattern_recognition_weight': 0.15,
                'enable_volume_profile': True,
                'enable_liquidity_detection': True,
                'enable_momentum_tracking': True,
                'enable_pattern_recognition': True,
            },
            
            SuperpositionType.NWRQK: {
                'coherence_decay_rate': 0.0008,
                'entanglement_strength': 0.7,
                'level_analysis_weight': 0.35,
                'wave_analysis_weight': 0.25,
                'pattern_recognition_weight': 0.2,
                'kernel_processing_weight': 0.2,
                'enable_level_identification': True,
                'enable_wave_analysis': True,
                'enable_kernel_density': True,
                'kernel_bandwidth_auto': True,
            },
            
            SuperpositionType.REGIME: {
                'coherence_decay_rate': 0.0005,
                'entanglement_strength': 0.5,
                'regime_identification_weight': 0.4,
                'transition_detection_weight': 0.3,
                'regime_modeling_weight': 0.2,
                'signal_processing_weight': 0.1,
                'enable_hmm_modeling': True,
                'enable_transition_detection': True,
                'enable_regime_signals': True,
                'regime_persistence_threshold': 0.7,
            },
            
            # Tactical configurations
            SuperpositionType.FVG: {
                'coherence_decay_rate': 0.002,
                'entanglement_strength': 0.4,
                'fvg_identification_weight': 0.3,
                'fvg_validation_weight': 0.25,
                'fvg_trading_weight': 0.25,
                'fvg_monitoring_weight': 0.2,
                'enable_fvg_detection': True,
                'enable_quality_scoring': True,
                'enable_fill_probability': True,
                'min_fvg_size': 0.0001,
                'max_fvg_age': 100,
            },
            
            SuperpositionType.MOMENTUM: {
                'coherence_decay_rate': 0.0015,
                'entanglement_strength': 0.45,
                'momentum_analysis_weight': 0.4,
                'momentum_indicators_weight': 0.3,
                'momentum_patterns_weight': 0.2,
                'momentum_trading_weight': 0.1,
                'enable_multi_timeframe': True,
                'enable_momentum_indicators': True,
                'enable_phase_detection': True,
                'momentum_periods': [5, 10, 20],
            },
            
            SuperpositionType.ENTRY_OPT: {
                'coherence_decay_rate': 0.0012,
                'entanglement_strength': 0.4,
                'timing_analysis_weight': 0.3,
                'signal_confirmation_weight': 0.3,
                'risk_assessment_weight': 0.25,
                'quality_evaluation_weight': 0.15,
                'enable_timing_optimization': True,
                'enable_signal_confirmation': True,
                'enable_risk_assessment': True,
                'min_confirmation_signals': 2,
            },
            
            # Risk management configurations
            SuperpositionType.POSITION_SIZING: {
                'coherence_decay_rate': 0.0008,
                'entanglement_strength': 0.6,
                'size_calculation_weight': 0.35,
                'risk_assessment_weight': 0.3,
                'kelly_analysis_weight': 0.2,
                'adjustment_factors_weight': 0.15,
                'enable_kelly_criterion': True,
                'enable_risk_parity': True,
                'enable_volatility_adjustment': True,
                'enable_correlation_adjustment': True,
                'max_kelly_fraction': 0.25,
            },
            
            SuperpositionType.STOP_TARGET: {
                'coherence_decay_rate': 0.001,
                'entanglement_strength': 0.5,
                'stop_management_weight': 0.3,
                'target_management_weight': 0.3,
                'risk_reward_optimization_weight': 0.25,
                'dynamic_adjustment_weight': 0.15,
                'enable_trailing_stops': True,
                'enable_partial_profits': True,
                'enable_dynamic_targets': True,
                'default_risk_reward_ratio': 2.0,
            },
            
            SuperpositionType.RISK_MONITOR: {
                'coherence_decay_rate': 0.0005,
                'entanglement_strength': 0.7,
                'risk_measurement_weight': 0.3,
                'alert_management_weight': 0.25,
                'limit_monitoring_weight': 0.25,
                'risk_forecasting_weight': 0.2,
                'enable_real_time_monitoring': True,
                'enable_risk_alerts': True,
                'enable_limit_enforcement': True,
                'alert_threshold': 0.8,
            },
            
            SuperpositionType.PORTFOLIO_OPTIMIZER: {
                'coherence_decay_rate': 0.0006,
                'entanglement_strength': 0.6,
                'weight_optimization_weight': 0.3,
                'risk_management_weight': 0.25,
                'rebalancing_weight': 0.25,
                'performance_monitoring_weight': 0.2,
                'enable_mean_variance_optimization': True,
                'enable_risk_parity': True,
                'enable_dynamic_rebalancing': True,
                'rebalancing_threshold': 0.05,
            },
            
            # Execution configurations
            SuperpositionType.EXECUTION_TIMING: {
                'coherence_decay_rate': 0.002,
                'entanglement_strength': 0.4,
                'timing_optimization_weight': 0.3,
                'market_impact_analysis_weight': 0.25,
                'cost_optimization_weight': 0.25,
                'strategy_selection_weight': 0.2,
                'enable_impact_modeling': True,
                'enable_cost_optimization': True,
                'enable_adaptive_execution': True,
                'max_participation_rate': 0.3,
            },
            
            SuperpositionType.ROUTING: {
                'coherence_decay_rate': 0.0018,
                'entanglement_strength': 0.45,
                'venue_selection_weight': 0.35,
                'order_fragmentation_weight': 0.25,
                'cost_optimization_weight': 0.25,
                'performance_monitoring_weight': 0.15,
                'enable_venue_optimization': True,
                'enable_order_fragmentation': True,
                'enable_cost_analysis': True,
                'max_venue_fragments': 5,
            },
            
            SuperpositionType.EXECUTION_RISK_MANAGEMENT: {
                'coherence_decay_rate': 0.0005,
                'entanglement_strength': 0.8,
                'pretrade_risk_checks_weight': 0.3,
                'risk_monitoring_weight': 0.3,
                'risk_controls_weight': 0.25,
                'performance_assessment_weight': 0.15,
                'enable_pretrade_checks': True,
                'enable_real_time_monitoring': True,
                'enable_risk_controls': True,
                'enable_kill_switch': True,
                'slippage_threshold': 0.005,
            },
        }


class SuperpositionFactory:
    """
    Factory for creating and managing superposition instances.
    
    Provides centralized creation, configuration, and optimization of 
    superposition instances for MARL agents.
    """
    
    def __init__(self):
        self.registry = SuperpositionRegistry()
        self.logger = logger.bind(component="SuperpositionFactory")
        self._instance_cache: Dict[str, UniversalSuperposition] = {}
        self._performance_metrics: Dict[str, Dict[str, float]] = {}
    
    def create_superposition(self,
                           agent_id: str,
                           superposition_type: Union[SuperpositionType, str],
                           config: Optional[Dict[str, Any]] = None,
                           initial_state: Optional[np.ndarray] = None,
                           use_cache: bool = True) -> UniversalSuperposition:
        """
        Create a superposition instance for an agent
        
        Args:
            agent_id: Unique identifier for the agent
            superposition_type: Type of superposition to create
            config: Configuration parameters (optional)
            initial_state: Initial state vector (optional)
            use_cache: Whether to use cached instances
            
        Returns:
            Configured superposition instance
        """
        # Convert string type to enum if necessary
        if isinstance(superposition_type, str):
            try:
                superposition_type = SuperpositionType(superposition_type.lower())
            except ValueError:
                raise ValueError(f"Unknown superposition type: {superposition_type}")
        
        # Check cache first
        cache_key = f"{agent_id}_{superposition_type.value}"
        if use_cache and cache_key in self._instance_cache:
            self.logger.debug("Returning cached superposition", agent_id=agent_id, type=superposition_type.value)
            return self._instance_cache[cache_key]
        
        # Get superposition class
        superposition_class = self.registry.get_superposition_class(superposition_type)
        if not superposition_class:
            raise ValueError(f"No superposition class found for type: {superposition_type}")
        
        # Prepare configuration
        final_config = self._prepare_config(superposition_type, config)
        
        # Create instance
        try:
            instance = superposition_class(
                agent_id=agent_id,
                config=final_config,
                initial_state=initial_state
            )
            
            # Cache instance if requested
            if use_cache:
                self._instance_cache[cache_key] = instance
            
            # Initialize performance tracking
            self._performance_metrics[cache_key] = {
                'creation_time': datetime.now().timestamp(),
                'usage_count': 0,
                'average_performance': 0.0,
                'error_count': 0
            }
            
            self.logger.info(
                "Created superposition instance",
                agent_id=agent_id,
                type=superposition_type.value,
                config_keys=list(final_config.keys())
            )
            
            return instance
            
        except Exception as e:
            self.logger.error(
                "Failed to create superposition",
                agent_id=agent_id,
                type=superposition_type.value,
                error=str(e)
            )
            raise
    
    def create_multi_superposition(self,
                                 agent_id: str,
                                 superposition_types: List[Union[SuperpositionType, str]],
                                 config: Optional[Dict[str, Any]] = None) -> List[UniversalSuperposition]:
        """
        Create multiple superposition instances for an agent
        
        Args:
            agent_id: Unique identifier for the agent
            superposition_types: List of superposition types to create
            config: Shared configuration parameters
            
        Returns:
            List of configured superposition instances
        """
        instances = []
        
        for stype in superposition_types:
            try:
                instance = self.create_superposition(
                    agent_id=agent_id,
                    superposition_type=stype,
                    config=config
                )
                instances.append(instance)
            except Exception as e:
                self.logger.error(
                    "Failed to create superposition in multi-creation",
                    agent_id=agent_id,
                    type=str(stype),
                    error=str(e)
                )
                continue
        
        return instances
    
    def create_by_category(self,
                          agent_id: str,
                          category: SuperpositionCategory,
                          config: Optional[Dict[str, Any]] = None) -> List[UniversalSuperposition]:
        """
        Create all superposition instances for a category
        
        Args:
            agent_id: Unique identifier for the agent
            category: Superposition category
            config: Configuration parameters
            
        Returns:
            List of superposition instances for the category
        """
        superposition_types = self.registry.get_types_by_category(category)
        return self.create_multi_superposition(agent_id, superposition_types, config)
    
    def optimize_superposition_config(self,
                                    superposition_type: SuperpositionType,
                                    performance_data: Dict[str, Any],
                                    optimization_target: str = 'sharpe_ratio') -> Dict[str, Any]:
        """
        Optimize superposition configuration based on performance data
        
        Args:
            superposition_type: Type of superposition to optimize
            performance_data: Historical performance data
            optimization_target: Target metric to optimize
            
        Returns:
            Optimized configuration parameters
        """
        self.logger.info(
            "Starting superposition optimization",
            type=superposition_type.value,
            target=optimization_target
        )
        
        # Get base configuration
        base_config = self.registry.get_default_config(superposition_type)
        
        # Optimization logic (simplified)
        optimized_config = base_config.copy()
        
        # Adjust coherence decay rate based on performance stability
        stability = performance_data.get('stability', 0.5)
        if stability < 0.3:
            optimized_config['coherence_decay_rate'] *= 0.8  # Slower decay for unstable performance
        elif stability > 0.7:
            optimized_config['coherence_decay_rate'] *= 1.2  # Faster decay for stable performance
        
        # Adjust entanglement strength based on correlation with other agents
        correlation = performance_data.get('agent_correlation', 0.5)
        if correlation > 0.8:
            optimized_config['entanglement_strength'] *= 0.8  # Reduce entanglement for high correlation
        elif correlation < 0.2:
            optimized_config['entanglement_strength'] *= 1.2  # Increase entanglement for low correlation
        
        # Adjust attention weights based on component performance
        component_performance = performance_data.get('component_performance', {})
        for component, performance in component_performance.items():
            weight_key = f"{component}_weight"
            if weight_key in optimized_config:
                if performance > 0.7:
                    optimized_config[weight_key] *= 1.1  # Increase weight for good performance
                elif performance < 0.3:
                    optimized_config[weight_key] *= 0.9  # Decrease weight for poor performance
        
        # Normalize weights to sum to 1.0
        self._normalize_weights(optimized_config)
        
        self.logger.info(
            "Completed superposition optimization",
            type=superposition_type.value,
            changes=self._config_diff(base_config, optimized_config)
        )
        
        return optimized_config
    
    def get_superposition_info(self, 
                             superposition_type: SuperpositionType) -> Dict[str, Any]:
        """
        Get information about a superposition type
        
        Args:
            superposition_type: Type of superposition
            
        Returns:
            Information dictionary
        """
        superposition_class = self.registry.get_superposition_class(superposition_type)
        default_config = self.registry.get_default_config(superposition_type)
        category = self.registry.get_category(superposition_type)
        
        return {
            'type': superposition_type.value,
            'category': category.value,
            'class_name': superposition_class.__name__,
            'state_dimension': superposition_class.get_state_dimension(superposition_class),
            'default_config': default_config,
            'required_features': self._get_required_features(superposition_type),
            'performance_characteristics': self._get_performance_characteristics(superposition_type)
        }
    
    def get_factory_status(self) -> Dict[str, Any]:
        """Get factory status and statistics"""
        return {
            'cached_instances': len(self._instance_cache),
            'registered_types': len(self.registry._registry),
            'performance_tracking': {
                cache_key: {
                    'usage_count': metrics['usage_count'],
                    'average_performance': metrics['average_performance'],
                    'error_rate': metrics['error_count'] / max(metrics['usage_count'], 1)
                }
                for cache_key, metrics in self._performance_metrics.items()
            },
            'memory_usage': self._estimate_memory_usage()
        }
    
    def clear_cache(self, agent_id: Optional[str] = None) -> None:
        """Clear superposition cache"""
        if agent_id:
            # Clear cache for specific agent
            keys_to_remove = [key for key in self._instance_cache.keys() if key.startswith(f"{agent_id}_")]
            for key in keys_to_remove:
                del self._instance_cache[key]
                if key in self._performance_metrics:
                    del self._performance_metrics[key]
        else:
            # Clear all cache
            self._instance_cache.clear()
            self._performance_metrics.clear()
        
        self.logger.info("Cache cleared", agent_id=agent_id)
    
    def _prepare_config(self, 
                       superposition_type: SuperpositionType,
                       user_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare final configuration by merging default and user configs"""
        default_config = self.registry.get_default_config(superposition_type)
        
        if user_config:
            # Merge configurations
            final_config = default_config.copy()
            final_config.update(user_config)
            
            # Validate configuration
            self._validate_config(superposition_type, final_config)
            
            return final_config
        
        return default_config
    
    def _validate_config(self, 
                        superposition_type: SuperpositionType,
                        config: Dict[str, Any]) -> None:
        """Validate configuration parameters"""
        # Check required parameters
        required_params = ['coherence_decay_rate', 'entanglement_strength']
        for param in required_params:
            if param not in config:
                raise ValueError(f"Missing required parameter: {param}")
        
        # Check parameter ranges
        if not 0 < config['coherence_decay_rate'] < 1:
            raise ValueError("coherence_decay_rate must be between 0 and 1")
        
        if not 0 < config['entanglement_strength'] <= 1:
            raise ValueError("entanglement_strength must be between 0 and 1")
        
        # Check weight parameters sum to reasonable values
        weight_params = [key for key in config.keys() if key.endswith('_weight')]
        if weight_params:
            total_weight = sum(config[param] for param in weight_params)
            if not 0.8 <= total_weight <= 1.2:
                self.logger.warning(
                    "Weight parameters sum to unusual value",
                    total_weight=total_weight,
                    type=superposition_type.value
                )
    
    def _normalize_weights(self, config: Dict[str, Any]) -> None:
        """Normalize weight parameters to sum to 1.0"""
        weight_params = [key for key in config.keys() if key.endswith('_weight')]
        if weight_params:
            total_weight = sum(config[param] for param in weight_params)
            if total_weight > 0:
                for param in weight_params:
                    config[param] /= total_weight
    
    def _config_diff(self, 
                    config1: Dict[str, Any],
                    config2: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate differences between configurations"""
        diff = {}
        all_keys = set(config1.keys()) | set(config2.keys())
        
        for key in all_keys:
            val1 = config1.get(key)
            val2 = config2.get(key)
            
            if val1 != val2:
                diff[key] = {'old': val1, 'new': val2}
        
        return diff
    
    def _get_required_features(self, superposition_type: SuperpositionType) -> List[str]:
        """Get required features for superposition type"""
        feature_map = {
            SuperpositionType.MLMI: ['volume_data', 'price_data', 'liquidity_data'],
            SuperpositionType.NWRQK: ['ohlc_data', 'volume_data'],
            SuperpositionType.REGIME: ['market_data', 'volatility_data'],
            SuperpositionType.FVG: ['ohlc_data', 'volume_data'],
            SuperpositionType.MOMENTUM: ['price_data', 'volume_data'],
            SuperpositionType.ENTRY_OPT: ['signal_data', 'market_data'],
            SuperpositionType.POSITION_SIZING: ['trade_data', 'portfolio_data'],
            SuperpositionType.STOP_TARGET: ['position_data', 'market_data'],
            SuperpositionType.RISK_MONITOR: ['portfolio_data', 'market_data'],
            SuperpositionType.PORTFOLIO_OPTIMIZER: ['returns_data', 'covariance_data'],
            SuperpositionType.EXECUTION_TIMING: ['order_data', 'market_data'],
            SuperpositionType.ROUTING: ['venue_data', 'order_data'],
            SuperpositionType.EXECUTION_RISK_MANAGEMENT: ['execution_data', 'risk_data']
        }
        
        return feature_map.get(superposition_type, [])
    
    def _get_performance_characteristics(self, superposition_type: SuperpositionType) -> Dict[str, Any]:
        """Get performance characteristics for superposition type"""
        characteristics = {
            'computational_complexity': 'medium',
            'memory_usage': 'low',
            'update_frequency': 'high',
            'scalability': 'good'
        }
        
        # Adjust based on type
        if superposition_type in [SuperpositionType.PORTFOLIO_OPTIMIZER, SuperpositionType.RISK_MONITOR]:
            characteristics['computational_complexity'] = 'high'
            characteristics['memory_usage'] = 'medium'
        elif superposition_type in [SuperpositionType.FVG, SuperpositionType.EXECUTION_TIMING]:
            characteristics['update_frequency'] = 'very_high'
        
        return characteristics
    
    def _estimate_memory_usage(self) -> Dict[str, Any]:
        """Estimate memory usage of cached instances"""
        # Simplified memory estimation
        instance_count = len(self._instance_cache)
        estimated_mb = instance_count * 0.5  # Rough estimate
        
        return {
            'instance_count': instance_count,
            'estimated_mb': estimated_mb,
            'cache_efficiency': min(instance_count / 100, 1.0)  # Normalize to 0-1
        }


# Convenience functions for common use cases
def create_strategic_superposition(agent_id: str, 
                                 superposition_type: str,
                                 config: Optional[Dict[str, Any]] = None) -> UniversalSuperposition:
    """Create a strategic superposition instance"""
    factory = SuperpositionFactory()
    return factory.create_superposition(agent_id, superposition_type, config)


def create_tactical_superposition(agent_id: str,
                                superposition_type: str,
                                config: Optional[Dict[str, Any]] = None) -> UniversalSuperposition:
    """Create a tactical superposition instance"""
    factory = SuperpositionFactory()
    return factory.create_superposition(agent_id, superposition_type, config)


def create_risk_superposition(agent_id: str,
                            superposition_type: str,
                            config: Optional[Dict[str, Any]] = None) -> UniversalSuperposition:
    """Create a risk management superposition instance"""
    factory = SuperpositionFactory()
    return factory.create_superposition(agent_id, superposition_type, config)


def create_execution_superposition(agent_id: str,
                                 superposition_type: str,
                                 config: Optional[Dict[str, Any]] = None) -> UniversalSuperposition:
    """Create an execution superposition instance"""
    factory = SuperpositionFactory()
    return factory.create_superposition(agent_id, superposition_type, config)


def create_full_agent_superposition_suite(agent_id: str,
                                         agent_category: str,
                                         config: Optional[Dict[str, Any]] = None) -> List[UniversalSuperposition]:
    """Create a complete superposition suite for an agent"""
    factory = SuperpositionFactory()
    
    category_map = {
        'strategic': SuperpositionCategory.STRATEGIC,
        'tactical': SuperpositionCategory.TACTICAL,
        'risk': SuperpositionCategory.RISK_MANAGEMENT,
        'execution': SuperpositionCategory.EXECUTION
    }
    
    category = category_map.get(agent_category)
    if not category:
        raise ValueError(f"Unknown agent category: {agent_category}")
    
    return factory.create_by_category(agent_id, category, config)