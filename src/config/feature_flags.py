"""
Feature Flag Management System
Dynamic feature flags and A/B testing configuration management.
"""

import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import random
import hashlib


class FeatureState(Enum):
    """Feature flag states"""
    ENABLED = "enabled"
    DISABLED = "disabled"
    ROLLOUT = "rollout"  # Gradual rollout
    EXPERIMENT = "experiment"  # A/B testing


class RolloutStrategy(Enum):
    """Rollout strategies"""
    PERCENTAGE = "percentage"
    USER_LIST = "user_list"
    HASH_BASED = "hash_based"
    TIME_BASED = "time_based"


@dataclass
class FeatureFlag:
    """Feature flag definition"""
    name: str
    state: FeatureState
    description: str
    owner: str
    created_at: datetime
    updated_at: datetime
    enabled_percentage: float = 100.0
    enabled_users: List[str] = None
    disabled_users: List[str] = None
    rollout_strategy: RolloutStrategy = RolloutStrategy.PERCENTAGE
    conditions: Dict[str, Any] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.enabled_users is None:
            self.enabled_users = []
        if self.disabled_users is None:
            self.disabled_users = []
        if self.conditions is None:
            self.conditions = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ABTestConfig:
    """A/B testing configuration"""
    name: str
    feature_flag: str
    variants: Dict[str, Dict[str, Any]]
    traffic_allocation: Dict[str, float]
    success_metrics: List[str]
    start_date: datetime
    end_date: Optional[datetime]
    minimum_sample_size: int
    confidence_level: float = 0.95
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class FeatureEvaluation:
    """Feature evaluation result"""
    feature_name: str
    enabled: bool
    variant: Optional[str]
    reason: str
    user_id: Optional[str]
    context: Dict[str, Any]
    timestamp: datetime


class FeatureFlagManager:
    """
    Feature flag management system with:
    - Dynamic feature flags
    - A/B testing support
    - Rollout strategies
    - Real-time updates
    - Analytics and reporting
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(__file__).parent.parent.parent / "config" / "feature_flags"
        self.logger = logging.getLogger(__name__)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Storage
        self.features: Dict[str, FeatureFlag] = {}
        self.ab_tests: Dict[str, ABTestConfig] = {}
        self.evaluations: List[FeatureEvaluation] = []
        
        # Event listeners
        self.evaluation_listeners: List[Callable[[FeatureEvaluation], None]] = []
        self.flag_change_listeners: List[Callable[[str, FeatureFlag], None]] = []
        
        # Ensure directories exist
        self.config_path.mkdir(parents=True, exist_ok=True)
        
        # Load configurations
        self._load_feature_flags()
        self._load_ab_tests()
        
        self.logger.info("FeatureFlagManager initialized")

    def _load_feature_flags(self):
        """Load feature flags from disk"""
        flags_file = self.config_path / "feature_flags.json"
        
        if flags_file.exists():
            try:
                with open(flags_file, 'r') as f:
                    data = json.load(f)
                
                for name, flag_data in data.items():
                    # Convert datetime strings back to datetime objects
                    flag_data['created_at'] = datetime.fromisoformat(flag_data['created_at'])
                    flag_data['updated_at'] = datetime.fromisoformat(flag_data['updated_at'])
                    
                    # Convert enum strings back to enums
                    flag_data['state'] = FeatureState(flag_data['state'])
                    flag_data['rollout_strategy'] = RolloutStrategy(flag_data['rollout_strategy'])
                    
                    self.features[name] = FeatureFlag(**flag_data)
                
                self.logger.info(f"Loaded {len(self.features)} feature flags")
                
            except Exception as e:
                self.logger.error(f"Failed to load feature flags: {e}")

    def _save_feature_flags(self):
        """Save feature flags to disk"""
        flags_file = self.config_path / "feature_flags.json"
        
        # Convert to serializable format
        data = {}
        for name, flag in self.features.items():
            flag_dict = asdict(flag)
            
            # Convert datetime objects to ISO strings
            flag_dict['created_at'] = flag_dict['created_at'].isoformat()
            flag_dict['updated_at'] = flag_dict['updated_at'].isoformat()
            
            # Convert enums to strings
            flag_dict['state'] = flag_dict['state'].value
            flag_dict['rollout_strategy'] = flag_dict['rollout_strategy'].value
            
            data[name] = flag_dict
        
        with open(flags_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_ab_tests(self):
        """Load A/B tests from disk"""
        ab_tests_file = self.config_path / "ab_tests.json"
        
        if ab_tests_file.exists():
            try:
                with open(ab_tests_file, 'r') as f:
                    data = json.load(f)
                
                for name, test_data in data.items():
                    # Convert datetime strings back to datetime objects
                    test_data['start_date'] = datetime.fromisoformat(test_data['start_date'])
                    if test_data.get('end_date'):
                        test_data['end_date'] = datetime.fromisoformat(test_data['end_date'])
                    
                    self.ab_tests[name] = ABTestConfig(**test_data)
                
                self.logger.info(f"Loaded {len(self.ab_tests)} A/B tests")
                
            except Exception as e:
                self.logger.error(f"Failed to load A/B tests: {e}")

    def _save_ab_tests(self):
        """Save A/B tests to disk"""
        ab_tests_file = self.config_path / "ab_tests.json"
        
        # Convert to serializable format
        data = {}
        for name, test in self.ab_tests.items():
            test_dict = asdict(test)
            
            # Convert datetime objects to ISO strings
            test_dict['start_date'] = test_dict['start_date'].isoformat()
            if test_dict.get('end_date'):
                test_dict['end_date'] = test_dict['end_date'].isoformat()
            
            data[name] = test_dict
        
        with open(ab_tests_file, 'w') as f:
            json.dump(data, f, indent=2)

    def create_feature_flag(self, name: str, description: str, owner: str,
                          state: FeatureState = FeatureState.DISABLED,
                          enabled_percentage: float = 100.0,
                          rollout_strategy: RolloutStrategy = RolloutStrategy.PERCENTAGE,
                          conditions: Dict[str, Any] = None,
                          metadata: Dict[str, Any] = None) -> bool:
        """
        Create a new feature flag
        
        Args:
            name: Feature flag name
            description: Description of the feature
            owner: Owner of the feature flag
            state: Initial state
            enabled_percentage: Percentage of users to enable for
            rollout_strategy: Strategy for rollout
            conditions: Additional conditions
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        with self._lock:
            if name in self.features:
                self.logger.warning(f"Feature flag '{name}' already exists")
                return False
            
            try:
                flag = FeatureFlag(
                    name=name,
                    state=state,
                    description=description,
                    owner=owner,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    enabled_percentage=enabled_percentage,
                    rollout_strategy=rollout_strategy,
                    conditions=conditions or {},
                    metadata=metadata or {}
                )
                
                self.features[name] = flag
                self._save_feature_flags()
                
                # Notify listeners
                for listener in self.flag_change_listeners:
                    try:
                        listener(name, flag)
                    except Exception as e:
                        self.logger.error(f"Error in flag change listener: {e}")
                
                self.logger.info(f"Created feature flag '{name}' by {owner}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to create feature flag '{name}': {e}")
                return False

    def update_feature_flag(self, name: str, **kwargs) -> bool:
        """
        Update an existing feature flag
        
        Args:
            name: Feature flag name
            **kwargs: Fields to update
            
        Returns:
            True if successful
        """
        with self._lock:
            if name not in self.features:
                self.logger.error(f"Feature flag '{name}' not found")
                return False
            
            try:
                flag = self.features[name]
                
                # Update fields
                for field, value in kwargs.items():
                    if hasattr(flag, field):
                        setattr(flag, field, value)
                
                flag.updated_at = datetime.now()
                
                self._save_feature_flags()
                
                # Notify listeners
                for listener in self.flag_change_listeners:
                    try:
                        listener(name, flag)
                    except Exception as e:
                        self.logger.error(f"Error in flag change listener: {e}")
                
                self.logger.info(f"Updated feature flag '{name}'")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to update feature flag '{name}': {e}")
                return False

    def delete_feature_flag(self, name: str) -> bool:
        """
        Delete a feature flag
        
        Args:
            name: Feature flag name
            
        Returns:
            True if successful
        """
        with self._lock:
            if name not in self.features:
                self.logger.error(f"Feature flag '{name}' not found")
                return False
            
            try:
                del self.features[name]
                self._save_feature_flags()
                
                self.logger.info(f"Deleted feature flag '{name}'")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to delete feature flag '{name}': {e}")
                return False

    def is_feature_enabled(self, feature_name: str, user_id: Optional[str] = None,
                          context: Dict[str, Any] = None) -> bool:
        """
        Check if a feature is enabled
        
        Args:
            feature_name: Name of the feature
            user_id: User ID for evaluation
            context: Additional context
            
        Returns:
            True if feature is enabled
        """
        evaluation = self.evaluate_feature(feature_name, user_id, context)
        return evaluation.enabled

    def evaluate_feature(self, feature_name: str, user_id: Optional[str] = None,
                        context: Dict[str, Any] = None) -> FeatureEvaluation:
        """
        Evaluate a feature flag
        
        Args:
            feature_name: Name of the feature
            user_id: User ID for evaluation
            context: Additional context
            
        Returns:
            FeatureEvaluation result
        """
        with self._lock:
            context = context or {}
            
            if feature_name not in self.features:
                evaluation = FeatureEvaluation(
                    feature_name=feature_name,
                    enabled=False,
                    variant=None,
                    reason="Feature not found",
                    user_id=user_id,
                    context=context,
                    timestamp=datetime.now()
                )
                self._log_evaluation(evaluation)
                return evaluation
            
            flag = self.features[feature_name]
            
            # Check basic state
            if flag.state == FeatureState.DISABLED:
                evaluation = FeatureEvaluation(
                    feature_name=feature_name,
                    enabled=False,
                    variant=None,
                    reason="Feature disabled",
                    user_id=user_id,
                    context=context,
                    timestamp=datetime.now()
                )
                self._log_evaluation(evaluation)
                return evaluation
            
            if flag.state == FeatureState.ENABLED:
                evaluation = FeatureEvaluation(
                    feature_name=feature_name,
                    enabled=True,
                    variant=None,
                    reason="Feature enabled",
                    user_id=user_id,
                    context=context,
                    timestamp=datetime.now()
                )
                self._log_evaluation(evaluation)
                return evaluation
            
            # Handle rollout and experiment states
            enabled = self._evaluate_rollout(flag, user_id, context)
            variant = self._evaluate_variant(feature_name, user_id, context)
            
            reason = f"Rollout evaluation: {flag.rollout_strategy.value}"
            if variant:
                reason += f", variant: {variant}"
            
            evaluation = FeatureEvaluation(
                feature_name=feature_name,
                enabled=enabled,
                variant=variant,
                reason=reason,
                user_id=user_id,
                context=context,
                timestamp=datetime.now()
            )
            
            self._log_evaluation(evaluation)
            return evaluation

    def _evaluate_rollout(self, flag: FeatureFlag, user_id: Optional[str],
                         context: Dict[str, Any]) -> bool:
        """Evaluate rollout strategy"""
        # Check explicit user lists first
        if user_id:
            if user_id in flag.disabled_users:
                return False
            if user_id in flag.enabled_users:
                return True
        
        # Evaluate based on rollout strategy
        if flag.rollout_strategy == RolloutStrategy.PERCENTAGE:
            return self._evaluate_percentage_rollout(flag, user_id)
        elif flag.rollout_strategy == RolloutStrategy.HASH_BASED:
            return self._evaluate_hash_based_rollout(flag, user_id)
        elif flag.rollout_strategy == RolloutStrategy.TIME_BASED:
            return self._evaluate_time_based_rollout(flag, context)
        elif flag.rollout_strategy == RolloutStrategy.USER_LIST:
            return user_id in flag.enabled_users if user_id else False
        
        return False

    def _evaluate_percentage_rollout(self, flag: FeatureFlag, user_id: Optional[str]) -> bool:
        """Evaluate percentage-based rollout"""
        if user_id:
            # Use consistent hash for stable rollout
            hash_value = int(hashlib.md5(f"{flag.name}:{user_id}".encode()).hexdigest(), 16)
            percentage = (hash_value % 100) + 1
            return percentage <= flag.enabled_percentage
        else:
            # Random rollout for anonymous users
            return random.random() * 100 <= flag.enabled_percentage

    def _evaluate_hash_based_rollout(self, flag: FeatureFlag, user_id: Optional[str]) -> bool:
        """Evaluate hash-based rollout"""
        if not user_id:
            return False
        
        hash_value = int(hashlib.md5(f"{flag.name}:{user_id}".encode()).hexdigest(), 16)
        percentage = (hash_value % 100) + 1
        return percentage <= flag.enabled_percentage

    def _evaluate_time_based_rollout(self, flag: FeatureFlag, context: Dict[str, Any]) -> bool:
        """Evaluate time-based rollout"""
        conditions = flag.conditions
        
        if 'start_time' in conditions and 'end_time' in conditions:
            now = datetime.now()
            start_time = datetime.fromisoformat(conditions['start_time'])
            end_time = datetime.fromisoformat(conditions['end_time'])
            
            if start_time <= now <= end_time:
                return random.random() * 100 <= flag.enabled_percentage
        
        return False

    def _evaluate_variant(self, feature_name: str, user_id: Optional[str],
                         context: Dict[str, Any]) -> Optional[str]:
        """Evaluate A/B test variant"""
        # Find active A/B test for this feature
        for test_name, test in self.ab_tests.items():
            if test.feature_flag == feature_name:
                now = datetime.now()
                
                # Check if test is active
                if test.start_date <= now and (not test.end_date or now <= test.end_date):
                    return self._assign_variant(test, user_id)
        
        return None

    def _assign_variant(self, test: ABTestConfig, user_id: Optional[str]) -> str:
        """Assign user to A/B test variant"""
        if not user_id:
            # Random assignment for anonymous users
            rand = random.random()
        else:
            # Consistent assignment based on user ID
            hash_value = int(hashlib.md5(f"{test.name}:{user_id}".encode()).hexdigest(), 16)
            rand = (hash_value % 10000) / 10000.0
        
        # Assign based on traffic allocation
        cumulative = 0.0
        for variant, allocation in test.traffic_allocation.items():
            cumulative += allocation
            if rand <= cumulative:
                return variant
        
        # Default to first variant if no match
        return list(test.variants.keys())[0]

    def _log_evaluation(self, evaluation: FeatureEvaluation):
        """Log feature evaluation"""
        self.evaluations.append(evaluation)
        
        # Notify listeners
        for listener in self.evaluation_listeners:
            try:
                listener(evaluation)
            except Exception as e:
                self.logger.error(f"Error in evaluation listener: {e}")
        
        # Keep only recent evaluations (last 10000)
        if len(self.evaluations) > 10000:
            self.evaluations = self.evaluations[-10000:]

    def create_ab_test(self, name: str, feature_flag: str, variants: Dict[str, Dict[str, Any]],
                      traffic_allocation: Dict[str, float], success_metrics: List[str],
                      start_date: datetime, end_date: Optional[datetime] = None,
                      minimum_sample_size: int = 1000,
                      confidence_level: float = 0.95) -> bool:
        """
        Create A/B test
        
        Args:
            name: Test name
            feature_flag: Associated feature flag
            variants: Test variants with their configurations
            traffic_allocation: Traffic allocation per variant
            success_metrics: Metrics to track
            start_date: Test start date
            end_date: Test end date (optional)
            minimum_sample_size: Minimum sample size
            confidence_level: Statistical confidence level
            
        Returns:
            True if successful
        """
        with self._lock:
            if name in self.ab_tests:
                self.logger.warning(f"A/B test '{name}' already exists")
                return False
            
            # Validate traffic allocation
            if abs(sum(traffic_allocation.values()) - 1.0) > 0.01:
                self.logger.error("Traffic allocation must sum to 1.0")
                return False
            
            try:
                test = ABTestConfig(
                    name=name,
                    feature_flag=feature_flag,
                    variants=variants,
                    traffic_allocation=traffic_allocation,
                    success_metrics=success_metrics,
                    start_date=start_date,
                    end_date=end_date,
                    minimum_sample_size=minimum_sample_size,
                    confidence_level=confidence_level
                )
                
                self.ab_tests[name] = test
                self._save_ab_tests()
                
                self.logger.info(f"Created A/B test '{name}' for feature '{feature_flag}'")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to create A/B test '{name}': {e}")
                return False

    def get_feature_flag(self, name: str) -> Optional[FeatureFlag]:
        """Get feature flag by name"""
        return self.features.get(name)

    def get_all_feature_flags(self) -> Dict[str, FeatureFlag]:
        """Get all feature flags"""
        return self.features.copy()

    def get_ab_test(self, name: str) -> Optional[ABTestConfig]:
        """Get A/B test by name"""
        return self.ab_tests.get(name)

    def get_all_ab_tests(self) -> Dict[str, ABTestConfig]:
        """Get all A/B tests"""
        return self.ab_tests.copy()

    def get_feature_analytics(self, feature_name: str, hours_back: int = 24) -> Dict[str, Any]:
        """
        Get analytics for a feature flag
        
        Args:
            feature_name: Feature name
            hours_back: Hours to look back
            
        Returns:
            Analytics data
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        # Filter evaluations for this feature
        feature_evaluations = [
            e for e in self.evaluations
            if e.feature_name == feature_name and e.timestamp >= cutoff_time
        ]
        
        if not feature_evaluations:
            return {
                'total_evaluations': 0,
                'enabled_count': 0,
                'disabled_count': 0,
                'enable_rate': 0.0,
                'unique_users': 0,
                'variants': {}
            }
        
        enabled_count = sum(1 for e in feature_evaluations if e.enabled)
        disabled_count = len(feature_evaluations) - enabled_count
        
        unique_users = len(set(e.user_id for e in feature_evaluations if e.user_id))
        
        # Variant analytics
        variant_counts = {}
        for evaluation in feature_evaluations:
            if evaluation.variant:
                variant_counts[evaluation.variant] = variant_counts.get(evaluation.variant, 0) + 1
        
        return {
            'total_evaluations': len(feature_evaluations),
            'enabled_count': enabled_count,
            'disabled_count': disabled_count,
            'enable_rate': enabled_count / len(feature_evaluations),
            'unique_users': unique_users,
            'variants': variant_counts,
            'time_range': {
                'start': cutoff_time.isoformat(),
                'end': datetime.now().isoformat()
            }
        }

    def add_evaluation_listener(self, listener: Callable[[FeatureEvaluation], None]):
        """Add evaluation listener"""
        self.evaluation_listeners.append(listener)

    def remove_evaluation_listener(self, listener: Callable[[FeatureEvaluation], None]):
        """Remove evaluation listener"""
        if listener in self.evaluation_listeners:
            self.evaluation_listeners.remove(listener)

    def add_flag_change_listener(self, listener: Callable[[str, FeatureFlag], None]):
        """Add flag change listener"""
        self.flag_change_listeners.append(listener)

    def remove_flag_change_listener(self, listener: Callable[[str, FeatureFlag], None]):
        """Remove flag change listener"""
        if listener in self.flag_change_listeners:
            self.flag_change_listeners.remove(listener)

    def reload_config(self):
        """Reload configuration from disk"""
        with self._lock:
            self.features.clear()
            self.ab_tests.clear()
            
            self._load_feature_flags()
            self._load_ab_tests()
            
            self.logger.info("Feature flag configuration reloaded")

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        now = datetime.now()
        
        # Count active tests
        active_tests = 0
        for test in self.ab_tests.values():
            if test.start_date <= now and (not test.end_date or now <= test.end_date):
                active_tests += 1
        
        # Count flags by state
        state_counts = {}
        for flag in self.features.values():
            state_counts[flag.state.value] = state_counts.get(flag.state.value, 0) + 1
        
        return {
            'total_flags': len(self.features),
            'total_ab_tests': len(self.ab_tests),
            'active_ab_tests': active_tests,
            'flag_states': state_counts,
            'recent_evaluations': len([e for e in self.evaluations 
                                     if e.timestamp >= now - timedelta(hours=1)]),
            'evaluation_listeners': len(self.evaluation_listeners),
            'flag_change_listeners': len(self.flag_change_listeners)
        }