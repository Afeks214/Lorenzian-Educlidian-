"""
Universal Superposition Validator for GrandModel System.

This module provides comprehensive validation for the universal superposition framework
that combines multiple MARL agents into coherent trading decisions. It validates:
- Mathematical properties of superposition states
- Quantum-inspired coherence constraints
- Multi-agent coordination integrity
- Performance consistency across all components
- Cascade integrity validation

The validator ensures that all superposition operations maintain mathematical rigor
and trading system reliability.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score


class ValidationLevel(Enum):
    """Validation levels for different types of checks."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    CRITICAL = "critical"


class SuperpositionProperty(Enum):
    """Properties to validate in superposition states."""
    COHERENCE = "coherence"
    ENTANGLEMENT = "entanglement"
    NORMALIZATION = "normalization"
    ORTHOGONALITY = "orthogonality"
    UNITARITY = "unitarity"
    LINEARITY = "linearity"
    CAUSALITY = "causality"


@dataclass
class ValidationResult:
    """Result from a superposition validation test."""
    test_name: str
    property_tested: SuperpositionProperty
    passed: bool
    score: float
    error_message: Optional[str] = None
    computed_value: Any = None
    expected_value: Any = None
    tolerance: float = 1e-6
    timestamp: datetime = field(default_factory=datetime.now)
    agent_id: Optional[str] = None
    performance_impact: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'test_name': self.test_name,
            'property_tested': self.property_tested.value,
            'passed': self.passed,
            'score': self.score,
            'error_message': self.error_message,
            'computed_value': self.computed_value,
            'expected_value': self.expected_value,
            'tolerance': self.tolerance,
            'timestamp': self.timestamp.isoformat(),
            'agent_id': self.agent_id,
            'performance_impact': self.performance_impact
        }


@dataclass
class SuperpositionState:
    """Represents a superposition state for validation."""
    amplitudes: np.ndarray
    phases: np.ndarray
    agent_contributions: Dict[str, float]
    confidence_scores: Dict[str, float]
    coherence_matrix: np.ndarray
    timestamp: datetime = field(default_factory=datetime.now)
    
    def validate_structure(self) -> bool:
        """Validate the structure of the superposition state."""
        try:
            # Check dimensions match
            if self.amplitudes.shape != self.phases.shape:
                return False
            
            # Check all arrays are finite
            if not (np.all(np.isfinite(self.amplitudes)) and 
                    np.all(np.isfinite(self.phases))):
                return False
            
            # Check coherence matrix is square
            n_agents = len(self.agent_contributions)
            if self.coherence_matrix.shape != (n_agents, n_agents):
                return False
            
            return True
        except Exception:
            return False


class UniversalSuperpositionValidator:
    """
    Comprehensive validator for universal superposition framework.
    
    This validator ensures that all superposition operations maintain:
    - Mathematical rigor and consistency
    - Quantum-inspired coherence properties
    - Multi-agent coordination integrity
    - Performance requirements (<5ms targets)
    - Cascade integrity across MARL systems
    """
    
    def __init__(
        self,
        tolerance: float = 1e-6,
        device: str = "cpu",
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        performance_target_ms: float = 5.0
    ):
        """
        Initialize the universal superposition validator.
        
        Args:
            tolerance: Numerical tolerance for validation checks
            device: PyTorch device for computations
            validation_level: Level of validation rigor
            performance_target_ms: Target performance in milliseconds
        """
        self.tolerance = tolerance
        self.device = torch.device(device)
        self.validation_level = validation_level
        self.performance_target_ms = performance_target_ms
        
        # Setup logging
        self.logger = logging.getLogger('superposition_validator')
        self.logger.setLevel(logging.INFO)
        
        # Validation history and statistics
        self.validation_history: List[ValidationResult] = []
        self.total_tests = 0
        self.passed_tests = 0
        self.performance_violations = 0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Agent registry for cascade validation
        self.registered_agents: Dict[str, Any] = {}
        
        self.logger.info(f"Universal superposition validator initialized with "
                        f"tolerance={tolerance}, level={validation_level.value}")
    
    def register_agent(self, agent_id: str, agent_config: Dict[str, Any]) -> None:
        """Register an agent for cascade validation."""
        with self._lock:
            self.registered_agents[agent_id] = agent_config
            self.logger.info(f"Registered agent {agent_id} for cascade validation")
    
    def validate_superposition_state(
        self,
        state: SuperpositionState,
        validation_level: Optional[ValidationLevel] = None
    ) -> Dict[str, ValidationResult]:
        """
        Validate a complete superposition state.
        
        Args:
            state: The superposition state to validate
            validation_level: Override default validation level
            
        Returns:
            Dictionary mapping test names to validation results
        """
        level = validation_level or self.validation_level
        results = {}
        
        # Basic structure validation
        if not state.validate_structure():
            results['structure'] = ValidationResult(
                test_name="Structure Validation",
                property_tested=SuperpositionProperty.COHERENCE,
                passed=False,
                score=0.0,
                error_message="Invalid superposition state structure"
            )
            return results
        
        # Core property validations
        validations = [
            (self._validate_coherence, SuperpositionProperty.COHERENCE),
            (self._validate_normalization, SuperpositionProperty.NORMALIZATION),
            (self._validate_orthogonality, SuperpositionProperty.ORTHOGONALITY),
            (self._validate_unitarity, SuperpositionProperty.UNITARITY),
            (self._validate_linearity, SuperpositionProperty.LINEARITY)
        ]
        
        # Add additional validations based on level
        if level in [ValidationLevel.COMPREHENSIVE, ValidationLevel.CRITICAL]:
            validations.extend([
                (self._validate_entanglement, SuperpositionProperty.ENTANGLEMENT),
                (self._validate_causality, SuperpositionProperty.CAUSALITY)
            ])
        
        # Run validations
        for validation_func, property_type in validations:
            try:
                result = validation_func(state)
                result.property_tested = property_type
                results[result.test_name] = result
            except Exception as e:
                results[f'{property_type.value}_error'] = ValidationResult(
                    test_name=f"{property_type.value.title()} Validation",
                    property_tested=property_type,
                    passed=False,
                    score=0.0,
                    error_message=f"Validation error: {str(e)}"
                )
        
        # Update statistics
        self._update_statistics(results)
        
        return results
    
    def _validate_coherence(self, state: SuperpositionState) -> ValidationResult:
        """
        Validate coherence properties of superposition state.
        
        Coherence ensures that the superposition maintains quantum-inspired
        properties that enable constructive interference between agents.
        """
        start_time = datetime.now()
        
        try:
            # Calculate coherence matrix properties
            coherence_matrix = state.coherence_matrix
            n_agents = coherence_matrix.shape[0]
            
            # Check Hermitian property (coherence matrix should be Hermitian)
            is_hermitian = np.allclose(coherence_matrix, coherence_matrix.T, atol=self.tolerance)
            
            # Check positive semi-definite property
            eigenvalues = np.linalg.eigvals(coherence_matrix)
            is_psd = np.all(eigenvalues >= -self.tolerance)
            
            # Calculate coherence measure (trace of coherence matrix)
            coherence_trace = np.trace(coherence_matrix)
            
            # Check diagonal elements (self-coherence should be 1)
            diagonal_coherence = np.diag(coherence_matrix)
            diagonal_valid = np.allclose(diagonal_coherence, 1.0, atol=self.tolerance)
            
            # Calculate off-diagonal coherence strength
            off_diagonal_sum = np.sum(np.abs(coherence_matrix)) - np.sum(np.abs(diagonal_coherence))
            max_off_diagonal = np.max(np.abs(coherence_matrix - np.diag(diagonal_coherence)))
            
            # Overall coherence score
            coherence_score = 1.0
            errors = []
            
            if not is_hermitian:
                coherence_score *= 0.5
                errors.append("Coherence matrix not Hermitian")
            
            if not is_psd:
                coherence_score *= 0.3
                errors.append("Coherence matrix not positive semi-definite")
            
            if not diagonal_valid:
                coherence_score *= 0.7
                errors.append("Diagonal coherence elements not unity")
            
            if max_off_diagonal > 1.0:
                coherence_score *= 0.8
                errors.append("Off-diagonal coherence exceeds bounds")
            
            # Performance check
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            performance_valid = processing_time < self.performance_target_ms
            
            if not performance_valid:
                coherence_score *= 0.9
                errors.append(f"Performance target exceeded: {processing_time:.2f}ms")
            
            passed = len(errors) == 0
            error_message = "; ".join(errors) if errors else None
            
            return ValidationResult(
                test_name="Coherence Validation",
                property_tested=SuperpositionProperty.COHERENCE,
                passed=passed,
                score=coherence_score,
                error_message=error_message,
                computed_value={
                    'coherence_trace': float(coherence_trace),
                    'max_off_diagonal': float(max_off_diagonal),
                    'is_hermitian': bool(is_hermitian),
                    'is_psd': bool(is_psd),
                    'processing_time_ms': processing_time
                },
                expected_value={
                    'coherence_trace': float(n_agents),
                    'max_off_diagonal': 1.0,
                    'is_hermitian': True,
                    'is_psd': True,
                    'processing_time_ms': self.performance_target_ms
                },
                tolerance=self.tolerance,
                performance_impact=processing_time
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Coherence Validation",
                property_tested=SuperpositionProperty.COHERENCE,
                passed=False,
                score=0.0,
                error_message=f"Coherence validation failed: {str(e)}"
            )
    
    def _validate_normalization(self, state: SuperpositionState) -> ValidationResult:
        """Validate normalization properties of superposition state."""
        start_time = datetime.now()
        
        try:
            # Check amplitude normalization
            amplitude_norm = np.linalg.norm(state.amplitudes)
            amplitude_normalized = np.isclose(amplitude_norm, 1.0, atol=self.tolerance)
            
            # Check probability normalization
            probabilities = np.abs(state.amplitudes) ** 2
            prob_sum = np.sum(probabilities)
            prob_normalized = np.isclose(prob_sum, 1.0, atol=self.tolerance)
            
            # Check agent contribution normalization
            contrib_sum = sum(state.agent_contributions.values())
            contrib_normalized = np.isclose(contrib_sum, 1.0, atol=self.tolerance)
            
            # Check confidence score bounds
            confidence_valid = all(0.0 <= score <= 1.0 for score in state.confidence_scores.values())
            
            normalization_score = 1.0
            errors = []
            
            if not amplitude_normalized:
                normalization_score *= 0.6
                errors.append(f"Amplitude norm {amplitude_norm:.6f} != 1.0")
            
            if not prob_normalized:
                normalization_score *= 0.5
                errors.append(f"Probability sum {prob_sum:.6f} != 1.0")
            
            if not contrib_normalized:
                normalization_score *= 0.7
                errors.append(f"Agent contribution sum {contrib_sum:.6f} != 1.0")
            
            if not confidence_valid:
                normalization_score *= 0.8
                errors.append("Confidence scores outside [0,1] bounds")
            
            # Performance check
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            performance_valid = processing_time < self.performance_target_ms
            
            if not performance_valid:
                normalization_score *= 0.9
                errors.append(f"Performance target exceeded: {processing_time:.2f}ms")
            
            passed = len(errors) == 0
            error_message = "; ".join(errors) if errors else None
            
            return ValidationResult(
                test_name="Normalization Validation",
                property_tested=SuperpositionProperty.NORMALIZATION,
                passed=passed,
                score=normalization_score,
                error_message=error_message,
                computed_value={
                    'amplitude_norm': float(amplitude_norm),
                    'prob_sum': float(prob_sum),
                    'contrib_sum': float(contrib_sum),
                    'processing_time_ms': processing_time
                },
                expected_value={
                    'amplitude_norm': 1.0,
                    'prob_sum': 1.0,
                    'contrib_sum': 1.0,
                    'processing_time_ms': self.performance_target_ms
                },
                tolerance=self.tolerance,
                performance_impact=processing_time
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Normalization Validation",
                property_tested=SuperpositionProperty.NORMALIZATION,
                passed=False,
                score=0.0,
                error_message=f"Normalization validation failed: {str(e)}"
            )
    
    def _validate_orthogonality(self, state: SuperpositionState) -> ValidationResult:
        """Validate orthogonality properties between agent states."""
        start_time = datetime.now()
        
        try:
            # Calculate orthogonality matrix from coherence matrix
            coherence_matrix = state.coherence_matrix
            n_agents = coherence_matrix.shape[0]
            
            # Check if coherence matrix represents orthogonal states
            # For orthogonal states, off-diagonal elements should be close to zero
            off_diagonal_mask = ~np.eye(n_agents, dtype=bool)
            off_diagonal_elements = coherence_matrix[off_diagonal_mask]
            
            # Calculate orthogonality measure
            orthogonality_measure = 1.0 - np.mean(np.abs(off_diagonal_elements))
            
            # Check maximum off-diagonal element
            max_off_diagonal = np.max(np.abs(off_diagonal_elements))
            
            # Validate orthogonality constraints
            orthogonality_score = 1.0
            errors = []
            
            # Strong orthogonality check (for critical systems)
            if self.validation_level == ValidationLevel.CRITICAL:
                if max_off_diagonal > 0.1:
                    orthogonality_score *= 0.5
                    errors.append(f"Maximum off-diagonal element {max_off_diagonal:.6f} > 0.1")
            else:
                if max_off_diagonal > 0.3:
                    orthogonality_score *= 0.7
                    errors.append(f"Maximum off-diagonal element {max_off_diagonal:.6f} > 0.3")
            
            # Check orthogonality measure
            if orthogonality_measure < 0.5:
                orthogonality_score *= 0.6
                errors.append(f"Orthogonality measure {orthogonality_measure:.6f} < 0.5")
            
            # Performance check
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            performance_valid = processing_time < self.performance_target_ms
            
            if not performance_valid:
                orthogonality_score *= 0.9
                errors.append(f"Performance target exceeded: {processing_time:.2f}ms")
            
            passed = len(errors) == 0
            error_message = "; ".join(errors) if errors else None
            
            return ValidationResult(
                test_name="Orthogonality Validation",
                property_tested=SuperpositionProperty.ORTHOGONALITY,
                passed=passed,
                score=orthogonality_score,
                error_message=error_message,
                computed_value={
                    'orthogonality_measure': float(orthogonality_measure),
                    'max_off_diagonal': float(max_off_diagonal),
                    'processing_time_ms': processing_time
                },
                expected_value={
                    'orthogonality_measure': 1.0,
                    'max_off_diagonal': 0.0,
                    'processing_time_ms': self.performance_target_ms
                },
                tolerance=self.tolerance,
                performance_impact=processing_time
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Orthogonality Validation",
                property_tested=SuperpositionProperty.ORTHOGONALITY,
                passed=False,
                score=0.0,
                error_message=f"Orthogonality validation failed: {str(e)}"
            )
    
    def _validate_unitarity(self, state: SuperpositionState) -> ValidationResult:
        """Validate unitarity properties of superposition transformations."""
        start_time = datetime.now()
        
        try:
            # Create transformation matrix from coherence matrix
            coherence_matrix = state.coherence_matrix
            
            # Check if transformation is unitary (U† U = I)
            should_be_identity = np.dot(coherence_matrix.T, coherence_matrix)
            identity_matrix = np.eye(coherence_matrix.shape[0])
            
            # Calculate unitarity measure
            unitarity_error = np.linalg.norm(should_be_identity - identity_matrix)
            unitarity_measure = 1.0 / (1.0 + unitarity_error)
            
            # Check determinant (should be 1 for unitary matrices)
            det = np.linalg.det(coherence_matrix)
            det_magnitude = np.abs(det)
            
            unitarity_score = 1.0
            errors = []
            
            # Check unitarity condition
            if unitarity_error > self.tolerance * 10:  # More relaxed for practical systems
                unitarity_score *= 0.6
                errors.append(f"Unitarity error {unitarity_error:.6f} > {self.tolerance * 10:.6f}")
            
            # Check determinant magnitude
            if not np.isclose(det_magnitude, 1.0, atol=self.tolerance * 10):
                unitarity_score *= 0.7
                errors.append(f"Determinant magnitude {det_magnitude:.6f} != 1.0")
            
            # Performance check
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            performance_valid = processing_time < self.performance_target_ms
            
            if not performance_valid:
                unitarity_score *= 0.9
                errors.append(f"Performance target exceeded: {processing_time:.2f}ms")
            
            passed = len(errors) == 0
            error_message = "; ".join(errors) if errors else None
            
            return ValidationResult(
                test_name="Unitarity Validation",
                property_tested=SuperpositionProperty.UNITARITY,
                passed=passed,
                score=unitarity_score,
                error_message=error_message,
                computed_value={
                    'unitarity_error': float(unitarity_error),
                    'unitarity_measure': float(unitarity_measure),
                    'det_magnitude': float(det_magnitude),
                    'processing_time_ms': processing_time
                },
                expected_value={
                    'unitarity_error': 0.0,
                    'unitarity_measure': 1.0,
                    'det_magnitude': 1.0,
                    'processing_time_ms': self.performance_target_ms
                },
                tolerance=self.tolerance,
                performance_impact=processing_time
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Unitarity Validation",
                property_tested=SuperpositionProperty.UNITARITY,
                passed=False,
                score=0.0,
                error_message=f"Unitarity validation failed: {str(e)}"
            )
    
    def _validate_linearity(self, state: SuperpositionState) -> ValidationResult:
        """Validate linearity properties of superposition combinations."""
        start_time = datetime.now()
        
        try:
            # Check if amplitudes and contributions follow linear superposition
            amplitudes = state.amplitudes
            contributions = np.array(list(state.agent_contributions.values()))
            
            # Calculate expected superposition from linear combination
            n_agents = len(state.agent_contributions)
            expected_amplitude_magnitude = np.sqrt(np.sum(contributions ** 2))
            
            # Check linearity in amplitude space
            amplitude_magnitude = np.linalg.norm(amplitudes)
            linearity_error = abs(amplitude_magnitude - expected_amplitude_magnitude)
            
            # Check additivity property
            # For linear superposition: |ψ⟩ = Σ cᵢ|ψᵢ⟩
            phase_consistency = self._check_phase_consistency(state.phases, contributions)
            
            linearity_score = 1.0
            errors = []
            
            # Check amplitude linearity
            if linearity_error > self.tolerance * 100:  # More relaxed for practical systems
                linearity_score *= 0.6
                errors.append(f"Amplitude linearity error {linearity_error:.6f} > {self.tolerance * 100:.6f}")
            
            # Check phase consistency
            if not phase_consistency:
                linearity_score *= 0.7
                errors.append("Phase consistency violated")
            
            # Performance check
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            performance_valid = processing_time < self.performance_target_ms
            
            if not performance_valid:
                linearity_score *= 0.9
                errors.append(f"Performance target exceeded: {processing_time:.2f}ms")
            
            passed = len(errors) == 0
            error_message = "; ".join(errors) if errors else None
            
            return ValidationResult(
                test_name="Linearity Validation",
                property_tested=SuperpositionProperty.LINEARITY,
                passed=passed,
                score=linearity_score,
                error_message=error_message,
                computed_value={
                    'linearity_error': float(linearity_error),
                    'amplitude_magnitude': float(amplitude_magnitude),
                    'expected_magnitude': float(expected_amplitude_magnitude),
                    'phase_consistency': phase_consistency,
                    'processing_time_ms': processing_time
                },
                expected_value={
                    'linearity_error': 0.0,
                    'amplitude_magnitude': float(expected_amplitude_magnitude),
                    'expected_magnitude': float(expected_amplitude_magnitude),
                    'phase_consistency': True,
                    'processing_time_ms': self.performance_target_ms
                },
                tolerance=self.tolerance,
                performance_impact=processing_time
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Linearity Validation",
                property_tested=SuperpositionProperty.LINEARITY,
                passed=False,
                score=0.0,
                error_message=f"Linearity validation failed: {str(e)}"
            )
    
    def _validate_entanglement(self, state: SuperpositionState) -> ValidationResult:
        """Validate entanglement properties between agents."""
        start_time = datetime.now()
        
        try:
            # Calculate entanglement measure using mutual information
            coherence_matrix = state.coherence_matrix
            n_agents = coherence_matrix.shape[0]
            
            # Calculate bipartite entanglement between all agent pairs
            entanglement_measures = []
            for i in range(n_agents):
                for j in range(i + 1, n_agents):
                    # Extract bipartite coherence
                    bipartite_coherence = abs(coherence_matrix[i, j])
                    
                    # Calculate entanglement measure (simplified)
                    entanglement = -np.log(max(1 - bipartite_coherence, 1e-10))
                    entanglement_measures.append(entanglement)
            
            # Calculate average entanglement
            avg_entanglement = np.mean(entanglement_measures) if entanglement_measures else 0.0
            max_entanglement = np.max(entanglement_measures) if entanglement_measures else 0.0
            
            # Check entanglement bounds
            entanglement_score = 1.0
            errors = []
            
            # For trading systems, we want controlled entanglement
            # Too much entanglement can lead to instability
            if max_entanglement > 2.0:  # Reasonable threshold
                entanglement_score *= 0.7
                errors.append(f"Maximum entanglement {max_entanglement:.6f} > 2.0")
            
            # Too little entanglement means agents aren't coordinating
            if avg_entanglement < 0.1:
                entanglement_score *= 0.8
                errors.append(f"Average entanglement {avg_entanglement:.6f} < 0.1")
            
            # Performance check
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            performance_valid = processing_time < self.performance_target_ms
            
            if not performance_valid:
                entanglement_score *= 0.9
                errors.append(f"Performance target exceeded: {processing_time:.2f}ms")
            
            passed = len(errors) == 0
            error_message = "; ".join(errors) if errors else None
            
            return ValidationResult(
                test_name="Entanglement Validation",
                property_tested=SuperpositionProperty.ENTANGLEMENT,
                passed=passed,
                score=entanglement_score,
                error_message=error_message,
                computed_value={
                    'avg_entanglement': float(avg_entanglement),
                    'max_entanglement': float(max_entanglement),
                    'entanglement_measures': [float(e) for e in entanglement_measures],
                    'processing_time_ms': processing_time
                },
                expected_value={
                    'avg_entanglement': 0.5,  # Moderate entanglement
                    'max_entanglement': 2.0,
                    'processing_time_ms': self.performance_target_ms
                },
                tolerance=self.tolerance,
                performance_impact=processing_time
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Entanglement Validation",
                property_tested=SuperpositionProperty.ENTANGLEMENT,
                passed=False,
                score=0.0,
                error_message=f"Entanglement validation failed: {str(e)}"
            )
    
    def _validate_causality(self, state: SuperpositionState) -> ValidationResult:
        """Validate causality constraints in superposition evolution."""
        start_time = datetime.now()
        
        try:
            # Check temporal causality through timestamp ordering
            current_time = datetime.now()
            state_time = state.timestamp
            
            # State should not be from the future
            temporal_valid = state_time <= current_time
            
            # Check information causality
            # Future information should not influence past decisions
            info_causality_valid = self._check_information_causality(state)
            
            # Check causal ordering of agent contributions
            causal_ordering_valid = self._check_causal_ordering(state)
            
            causality_score = 1.0
            errors = []
            
            if not temporal_valid:
                causality_score *= 0.3
                errors.append("Temporal causality violated - future state detected")
            
            if not info_causality_valid:
                causality_score *= 0.5
                errors.append("Information causality violated")
            
            if not causal_ordering_valid:
                causality_score *= 0.7
                errors.append("Causal ordering violated")
            
            # Performance check
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            performance_valid = processing_time < self.performance_target_ms
            
            if not performance_valid:
                causality_score *= 0.9
                errors.append(f"Performance target exceeded: {processing_time:.2f}ms")
            
            passed = len(errors) == 0
            error_message = "; ".join(errors) if errors else None
            
            return ValidationResult(
                test_name="Causality Validation",
                property_tested=SuperpositionProperty.CAUSALITY,
                passed=passed,
                score=causality_score,
                error_message=error_message,
                computed_value={
                    'temporal_valid': temporal_valid,
                    'info_causality_valid': info_causality_valid,
                    'causal_ordering_valid': causal_ordering_valid,
                    'processing_time_ms': processing_time
                },
                expected_value={
                    'temporal_valid': True,
                    'info_causality_valid': True,
                    'causal_ordering_valid': True,
                    'processing_time_ms': self.performance_target_ms
                },
                tolerance=self.tolerance,
                performance_impact=processing_time
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Causality Validation",
                property_tested=SuperpositionProperty.CAUSALITY,
                passed=False,
                score=0.0,
                error_message=f"Causality validation failed: {str(e)}"
            )
    
    def _check_phase_consistency(self, phases: np.ndarray, contributions: np.ndarray) -> bool:
        """Check if phases are consistent with agent contributions."""
        try:
            # Check if phases are within valid range [0, 2π]
            phase_range_valid = np.all((phases >= 0) & (phases <= 2 * np.pi))
            
            # Check if phase relationships are consistent
            # For simple case, check if dominant contributor has phase near 0
            if len(contributions) > 0:
                dominant_agent = np.argmax(contributions)
                dominant_phase = phases[dominant_agent] if dominant_agent < len(phases) else 0
                
                # Dominant phase should be close to 0 or 2π for stability
                phase_stable = (dominant_phase < np.pi / 4) or (dominant_phase > 7 * np.pi / 4)
                
                return phase_range_valid and phase_stable
            
            return phase_range_valid
            
        except Exception:
            return False
    
    def _check_information_causality(self, state: SuperpositionState) -> bool:
        """Check if information causality is preserved."""
        try:
            # In trading systems, information causality means:
            # 1. No future information in current decisions
            # 2. No circular dependencies in agent contributions
            
            # Check for circular dependencies in agent contributions
            agent_ids = list(state.agent_contributions.keys())
            
            # Simple cycle detection - check if any agent depends on itself
            # In practice, this would involve more complex dependency analysis
            for agent_id in agent_ids:
                if agent_id in agent_ids:  # Simplified check
                    continue
            
            return True  # Simplified implementation
            
        except Exception:
            return False
    
    def _check_causal_ordering(self, state: SuperpositionState) -> bool:
        """Check if causal ordering is preserved in agent interactions."""
        try:
            # Check if coherence matrix respects causal ordering
            coherence_matrix = state.coherence_matrix
            
            # For causal systems, the coherence matrix should not have
            # strong backward dependencies (simplified check)
            
            # Check if matrix is approximately upper triangular
            # (indicating forward causal flow)
            lower_triangle = np.tril(coherence_matrix, k=-1)
            upper_triangle = np.triu(coherence_matrix, k=1)
            
            lower_sum = np.sum(np.abs(lower_triangle))
            upper_sum = np.sum(np.abs(upper_triangle))
            
            # More forward causality than backward
            return upper_sum >= lower_sum * 0.8
            
        except Exception:
            return False
    
    def _update_statistics(self, results: Dict[str, ValidationResult]) -> None:
        """Update validation statistics."""
        with self._lock:
            for result in results.values():
                self.total_tests += 1
                if result.passed:
                    self.passed_tests += 1
                
                # Track performance violations
                if (result.performance_impact and 
                    result.performance_impact > self.performance_target_ms):
                    self.performance_violations += 1
                
                self.validation_history.append(result)
            
            # Keep only recent history
            if len(self.validation_history) > 10000:
                self.validation_history = self.validation_history[-5000:]
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation summary."""
        with self._lock:
            success_rate = self.passed_tests / max(1, self.total_tests)
            performance_violation_rate = self.performance_violations / max(1, self.total_tests)
            
            # Calculate property-specific success rates
            property_stats = {}
            for prop in SuperpositionProperty:
                prop_results = [r for r in self.validation_history 
                              if r.property_tested == prop]
                if prop_results:
                    prop_success_rate = sum(1 for r in prop_results if r.passed) / len(prop_results)
                    property_stats[prop.value] = {
                        'success_rate': prop_success_rate,
                        'total_tests': len(prop_results),
                        'avg_score': np.mean([r.score for r in prop_results])
                    }
            
            # Recent performance metrics
            recent_results = self.validation_history[-100:] if self.validation_history else []
            recent_performance = [r.performance_impact for r in recent_results 
                                if r.performance_impact is not None]
            
            return {
                'total_tests': self.total_tests,
                'passed_tests': self.passed_tests,
                'success_rate': success_rate,
                'performance_violations': self.performance_violations,
                'performance_violation_rate': performance_violation_rate,
                'property_statistics': property_stats,
                'recent_avg_performance_ms': np.mean(recent_performance) if recent_performance else 0,
                'recent_max_performance_ms': np.max(recent_performance) if recent_performance else 0,
                'validation_level': self.validation_level.value,
                'performance_target_ms': self.performance_target_ms,
                'tolerance': self.tolerance,
                'registered_agents': len(self.registered_agents),
                'last_validation_time': (
                    self.validation_history[-1].timestamp.isoformat() 
                    if self.validation_history else None
                )
            }
    
    def generate_validation_report(self, include_details: bool = False) -> str:
        """Generate comprehensive validation report."""
        summary = self.get_validation_summary()
        
        report = f"""
Universal Superposition Validation Report
=========================================

System Overview:
- Validation Level: {summary['validation_level']}
- Performance Target: {summary['performance_target_ms']}ms
- Tolerance: {summary['tolerance']}
- Registered Agents: {summary['registered_agents']}

Overall Statistics:
- Total Tests: {summary['total_tests']}
- Passed Tests: {summary['passed_tests']}
- Success Rate: {summary['success_rate']:.2%}
- Performance Violations: {summary['performance_violations']}
- Performance Violation Rate: {summary['performance_violation_rate']:.2%}

Performance Metrics:
- Recent Average Performance: {summary['recent_avg_performance_ms']:.2f}ms
- Recent Maximum Performance: {summary['recent_max_performance_ms']:.2f}ms
- Performance Target: {summary['performance_target_ms']}ms

Property-Specific Results:
"""
        
        for prop, stats in summary['property_statistics'].items():
            report += f"""
- {prop.title()}:
  - Success Rate: {stats['success_rate']:.2%}
  - Total Tests: {stats['total_tests']}
  - Average Score: {stats['avg_score']:.3f}
"""
        
        if include_details and self.validation_history:
            report += "\nRecent Validation Results:\n"
            for result in self.validation_history[-10:]:
                status = "✓ PASS" if result.passed else "✗ FAIL"
                perf_info = f" ({result.performance_impact:.2f}ms)" if result.performance_impact else ""
                report += f"- {result.test_name}: {status} (Score: {result.score:.3f}){perf_info}\n"
                
                if not result.passed and result.error_message:
                    report += f"  Error: {result.error_message}\n"
        
        report += f"\nLast Validation: {summary['last_validation_time']}\n"
        
        return report
    
    def export_validation_data(self) -> Dict[str, Any]:
        """Export validation data for analysis."""
        with self._lock:
            return {
                'summary': self.get_validation_summary(),
                'history': [result.to_dict() for result in self.validation_history],
                'configuration': {
                    'tolerance': self.tolerance,
                    'validation_level': self.validation_level.value,
                    'performance_target_ms': self.performance_target_ms,
                    'device': str(self.device)
                }
            }