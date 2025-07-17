#!/usr/bin/env python3
"""
Real-time Superposition Monitoring System
Monitors quantum superposition states and effectiveness in MARL agents
"""

import asyncio
import numpy as np
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
from collections import deque
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings

# Monitoring metrics
from prometheus_client import Counter, Histogram, Gauge, Info
import redis
import psutil

# Quantum state monitoring
SUPERPOSITION_COHERENCE = Gauge('superposition_coherence', 'Quantum coherence measure', ['agent_id', 'superposition_type'])
SUPERPOSITION_FIDELITY = Gauge('superposition_fidelity', 'State fidelity measure', ['agent_id', 'superposition_type'])
SUPERPOSITION_ENTROPY = Gauge('superposition_entropy', 'Entanglement entropy', ['agent_id', 'superposition_type'])
SUPERPOSITION_EFFECTIVENESS = Gauge('superposition_effectiveness', 'Overall effectiveness score', ['agent_id'])

# Performance metrics
SUPERPOSITION_EXECUTION_TIME = Histogram('superposition_execution_seconds', 'Execution time for superposition operations', ['agent_id', 'operation'])
SUPERPOSITION_ANOMALY_COUNT = Counter('superposition_anomalies_total', 'Number of detected anomalies', ['agent_id', 'anomaly_type'])
SUPERPOSITION_TRANSITIONS = Counter('superposition_transitions_total', 'State transitions', ['agent_id', 'from_state', 'to_state'])

# System metrics
QUANTUM_DECOHERENCE_RATE = Gauge('quantum_decoherence_rate', 'Decoherence rate per second', ['agent_id'])
QUANTUM_GATE_FIDELITY = Gauge('quantum_gate_fidelity', 'Gate operation fidelity', ['agent_id', 'gate_type'])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SuperpositionState(Enum):
    """Quantum superposition states."""
    COHERENT = "coherent"
    DECOHERENT = "decoherent"
    ENTANGLED = "entangled"
    COLLAPSED = "collapsed"
    UNKNOWN = "unknown"

class AnomalyType(Enum):
    """Types of superposition anomalies."""
    DECOHERENCE_SPIKE = "decoherence_spike"
    FIDELITY_DROP = "fidelity_drop"
    ENTANGLEMENT_LOSS = "entanglement_loss"
    PHASE_DRIFT = "phase_drift"
    EXECUTION_TIMEOUT = "execution_timeout"
    MEASUREMENT_ERROR = "measurement_error"

@dataclass
class SuperpositionMeasurement:
    """Quantum superposition measurement result."""
    agent_id: str
    timestamp: datetime
    state: SuperpositionState
    coherence: float
    fidelity: float
    entropy: float
    phase: complex
    amplitude: np.ndarray
    measurement_duration_ms: float
    quantum_error_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'agent_id': self.agent_id,
            'timestamp': self.timestamp.isoformat(),
            'state': self.state.value,
            'coherence': self.coherence,
            'fidelity': self.fidelity,
            'entropy': self.entropy,
            'phase': {'real': self.phase.real, 'imag': self.phase.imag},
            'amplitude': self.amplitude.tolist(),
            'measurement_duration_ms': self.measurement_duration_ms,
            'quantum_error_rate': self.quantum_error_rate
        }

@dataclass
class SuperpositionAnomaly:
    """Superposition anomaly detection result."""
    agent_id: str
    anomaly_type: AnomalyType
    timestamp: datetime
    severity: float
    description: str
    measurements: List[SuperpositionMeasurement]
    suggested_action: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'agent_id': self.agent_id,
            'anomaly_type': self.anomaly_type.value,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity,
            'description': self.description,
            'measurements': [m.to_dict() for m in self.measurements],
            'suggested_action': self.suggested_action
        }

class QuantumStateAnalyzer:
    """Analyzes quantum superposition states."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.coherence_threshold = config.get('coherence_threshold', 0.7)
        self.fidelity_threshold = config.get('fidelity_threshold', 0.95)
        self.entropy_threshold = config.get('entropy_threshold', 2.0)
        
    def analyze_quantum_state(self, measurement: SuperpositionMeasurement) -> Dict[str, Any]:
        """Analyze quantum state properties."""
        
        analysis = {
            'coherence_score': self._calculate_coherence_score(measurement),
            'fidelity_score': self._calculate_fidelity_score(measurement),
            'entropy_analysis': self._analyze_entropy(measurement),
            'phase_stability': self._analyze_phase_stability(measurement),
            'amplitude_distribution': self._analyze_amplitude_distribution(measurement),
            'quantum_error_analysis': self._analyze_quantum_errors(measurement)
        }
        
        return analysis
    
    def _calculate_coherence_score(self, measurement: SuperpositionMeasurement) -> float:
        """Calculate quantum coherence score."""
        try:
            # Coherence based on off-diagonal elements of density matrix
            amplitude_norm = np.linalg.norm(measurement.amplitude)
            if amplitude_norm == 0:
                return 0.0
            
            normalized_amplitude = measurement.amplitude / amplitude_norm
            
            # Calculate coherence as measure of superposition
            coherence = np.sum(np.abs(normalized_amplitude[:-1] * normalized_amplitude[1:]))
            return min(coherence, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating coherence: {e}")
            return 0.0
    
    def _calculate_fidelity_score(self, measurement: SuperpositionMeasurement) -> float:
        """Calculate quantum state fidelity."""
        try:
            # Fidelity with respect to ideal superposition state
            ideal_state = np.ones(len(measurement.amplitude)) / np.sqrt(len(measurement.amplitude))
            fidelity = np.abs(np.dot(measurement.amplitude.conj(), ideal_state))**2
            return min(fidelity, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating fidelity: {e}")
            return 0.0
    
    def _analyze_entropy(self, measurement: SuperpositionMeasurement) -> Dict[str, float]:
        """Analyze entropy properties."""
        try:
            amplitude_squared = np.abs(measurement.amplitude)**2
            amplitude_squared = amplitude_squared / np.sum(amplitude_squared)  # Normalize
            
            # Von Neumann entropy
            von_neumann_entropy = -np.sum(amplitude_squared * np.log(amplitude_squared + 1e-10))
            
            # Renyi entropy (order 2)
            renyi_entropy = -np.log(np.sum(amplitude_squared**2))
            
            return {
                'von_neumann': von_neumann_entropy,
                'renyi': renyi_entropy,
                'normalized': von_neumann_entropy / np.log(len(measurement.amplitude))
            }
            
        except Exception as e:
            logger.warning(f"Error calculating entropy: {e}")
            return {'von_neumann': 0.0, 'renyi': 0.0, 'normalized': 0.0}
    
    def _analyze_phase_stability(self, measurement: SuperpositionMeasurement) -> Dict[str, float]:
        """Analyze quantum phase stability."""
        try:
            phase_angle = np.angle(measurement.phase)
            phase_magnitude = np.abs(measurement.phase)
            
            # Phase coherence
            phase_coherence = phase_magnitude
            
            # Phase drift (would need historical data)
            phase_drift = 0.0  # Placeholder
            
            return {
                'phase_angle': phase_angle,
                'phase_magnitude': phase_magnitude,
                'phase_coherence': phase_coherence,
                'phase_drift': phase_drift
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing phase: {e}")
            return {'phase_angle': 0.0, 'phase_magnitude': 0.0, 'phase_coherence': 0.0, 'phase_drift': 0.0}
    
    def _analyze_amplitude_distribution(self, measurement: SuperpositionMeasurement) -> Dict[str, float]:
        """Analyze amplitude distribution properties."""
        try:
            amplitudes = np.abs(measurement.amplitude)
            
            # Statistical measures
            mean_amplitude = np.mean(amplitudes)
            std_amplitude = np.std(amplitudes)
            skewness = np.mean(((amplitudes - mean_amplitude) / std_amplitude)**3)
            kurtosis = np.mean(((amplitudes - mean_amplitude) / std_amplitude)**4)
            
            return {
                'mean': mean_amplitude,
                'std': std_amplitude,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'uniformity': 1.0 - (std_amplitude / (mean_amplitude + 1e-10))
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing amplitude distribution: {e}")
            return {'mean': 0.0, 'std': 0.0, 'skewness': 0.0, 'kurtosis': 0.0, 'uniformity': 0.0}
    
    def _analyze_quantum_errors(self, measurement: SuperpositionMeasurement) -> Dict[str, float]:
        """Analyze quantum error characteristics."""
        try:
            # Error rate analysis
            error_rate = measurement.quantum_error_rate
            
            # Normalized error based on measurement duration
            normalized_error = error_rate / (measurement.measurement_duration_ms / 1000.0)
            
            # Error severity classification
            if error_rate < 0.01:
                severity = 'low'
            elif error_rate < 0.05:
                severity = 'medium'
            else:
                severity = 'high'
            
            return {
                'error_rate': error_rate,
                'normalized_error': normalized_error,
                'severity': severity,
                'error_threshold_exceeded': error_rate > 0.05
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing quantum errors: {e}")
            return {'error_rate': 0.0, 'normalized_error': 0.0, 'severity': 'unknown', 'error_threshold_exceeded': False}

class SuperpositionAnomalyDetector:
    """Detects anomalies in quantum superposition states."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.detection_thresholds = config.get('detection_thresholds', {})
        self.measurement_history = deque(maxlen=1000)
        self.anomaly_history = deque(maxlen=100)
        
    def detect_anomalies(self, measurement: SuperpositionMeasurement) -> List[SuperpositionAnomaly]:
        """Detect anomalies in superposition measurement."""
        
        anomalies = []
        self.measurement_history.append(measurement)
        
        # Check for various anomaly types
        anomalies.extend(self._check_decoherence_spike(measurement))
        anomalies.extend(self._check_fidelity_drop(measurement))
        anomalies.extend(self._check_entanglement_loss(measurement))
        anomalies.extend(self._check_phase_drift(measurement))
        anomalies.extend(self._check_execution_timeout(measurement))
        anomalies.extend(self._check_measurement_error(measurement))
        
        # Store detected anomalies
        for anomaly in anomalies:
            self.anomaly_history.append(anomaly)
            
            # Update metrics
            SUPERPOSITION_ANOMALY_COUNT.labels(
                agent_id=anomaly.agent_id,
                anomaly_type=anomaly.anomaly_type.value
            ).inc()
        
        return anomalies
    
    def _check_decoherence_spike(self, measurement: SuperpositionMeasurement) -> List[SuperpositionAnomaly]:
        """Check for sudden decoherence spikes."""
        anomalies = []
        
        if len(self.measurement_history) < 2:
            return anomalies
        
        try:
            current_coherence = measurement.coherence
            previous_coherence = self.measurement_history[-2].coherence
            
            coherence_drop = previous_coherence - current_coherence
            drop_threshold = self.detection_thresholds.get('decoherence_spike_threshold', 0.2)
            
            if coherence_drop > drop_threshold:
                anomaly = SuperpositionAnomaly(
                    agent_id=measurement.agent_id,
                    anomaly_type=AnomalyType.DECOHERENCE_SPIKE,
                    timestamp=measurement.timestamp,
                    severity=min(coherence_drop / drop_threshold, 1.0),
                    description=f"Sudden coherence drop: {coherence_drop:.3f}",
                    measurements=[measurement],
                    suggested_action="Check for environmental interference or quantum noise"
                )
                anomalies.append(anomaly)
                
        except Exception as e:
            logger.warning(f"Error checking decoherence spike: {e}")
        
        return anomalies
    
    def _check_fidelity_drop(self, measurement: SuperpositionMeasurement) -> List[SuperpositionAnomaly]:
        """Check for fidelity drops."""
        anomalies = []
        
        try:
            fidelity_threshold = self.detection_thresholds.get('fidelity_threshold', 0.9)
            
            if measurement.fidelity < fidelity_threshold:
                anomaly = SuperpositionAnomaly(
                    agent_id=measurement.agent_id,
                    anomaly_type=AnomalyType.FIDELITY_DROP,
                    timestamp=measurement.timestamp,
                    severity=1.0 - (measurement.fidelity / fidelity_threshold),
                    description=f"Low fidelity: {measurement.fidelity:.3f}",
                    measurements=[measurement],
                    suggested_action="Recalibrate quantum gates or check system initialization"
                )
                anomalies.append(anomaly)
                
        except Exception as e:
            logger.warning(f"Error checking fidelity drop: {e}")
        
        return anomalies
    
    def _check_entanglement_loss(self, measurement: SuperpositionMeasurement) -> List[SuperpositionAnomaly]:
        """Check for entanglement loss."""
        anomalies = []
        
        try:
            entropy_threshold = self.detection_thresholds.get('entropy_threshold', 0.5)
            
            if measurement.entropy < entropy_threshold:
                anomaly = SuperpositionAnomaly(
                    agent_id=measurement.agent_id,
                    anomaly_type=AnomalyType.ENTANGLEMENT_LOSS,
                    timestamp=measurement.timestamp,
                    severity=1.0 - (measurement.entropy / entropy_threshold),
                    description=f"Low entanglement entropy: {measurement.entropy:.3f}",
                    measurements=[measurement],
                    suggested_action="Check quantum channel integrity and gate sequencing"
                )
                anomalies.append(anomaly)
                
        except Exception as e:
            logger.warning(f"Error checking entanglement loss: {e}")
        
        return anomalies
    
    def _check_phase_drift(self, measurement: SuperpositionMeasurement) -> List[SuperpositionAnomaly]:
        """Check for phase drift."""
        anomalies = []
        
        if len(self.measurement_history) < 5:
            return anomalies
        
        try:
            recent_phases = [np.angle(m.phase) for m in list(self.measurement_history)[-5:]]
            phase_variance = np.var(recent_phases)
            
            drift_threshold = self.detection_thresholds.get('phase_drift_threshold', 0.1)
            
            if phase_variance > drift_threshold:
                anomaly = SuperpositionAnomaly(
                    agent_id=measurement.agent_id,
                    anomaly_type=AnomalyType.PHASE_DRIFT,
                    timestamp=measurement.timestamp,
                    severity=min(phase_variance / drift_threshold, 1.0),
                    description=f"Phase drift detected: variance={phase_variance:.3f}",
                    measurements=[measurement],
                    suggested_action="Check phase reference stability and calibration"
                )
                anomalies.append(anomaly)
                
        except Exception as e:
            logger.warning(f"Error checking phase drift: {e}")
        
        return anomalies
    
    def _check_execution_timeout(self, measurement: SuperpositionMeasurement) -> List[SuperpositionAnomaly]:
        """Check for execution timeouts."""
        anomalies = []
        
        try:
            timeout_threshold = self.detection_thresholds.get('execution_timeout_ms', 1000)
            
            if measurement.measurement_duration_ms > timeout_threshold:
                anomaly = SuperpositionAnomaly(
                    agent_id=measurement.agent_id,
                    anomaly_type=AnomalyType.EXECUTION_TIMEOUT,
                    timestamp=measurement.timestamp,
                    severity=min(measurement.measurement_duration_ms / timeout_threshold, 1.0),
                    description=f"Execution timeout: {measurement.measurement_duration_ms:.1f}ms",
                    measurements=[measurement],
                    suggested_action="Optimize quantum gate sequence or increase computational resources"
                )
                anomalies.append(anomaly)
                
        except Exception as e:
            logger.warning(f"Error checking execution timeout: {e}")
        
        return anomalies
    
    def _check_measurement_error(self, measurement: SuperpositionMeasurement) -> List[SuperpositionAnomaly]:
        """Check for measurement errors."""
        anomalies = []
        
        try:
            error_threshold = self.detection_thresholds.get('measurement_error_threshold', 0.05)
            
            if measurement.quantum_error_rate > error_threshold:
                anomaly = SuperpositionAnomaly(
                    agent_id=measurement.agent_id,
                    anomaly_type=AnomalyType.MEASUREMENT_ERROR,
                    timestamp=measurement.timestamp,
                    severity=min(measurement.quantum_error_rate / error_threshold, 1.0),
                    description=f"High measurement error rate: {measurement.quantum_error_rate:.3f}",
                    measurements=[measurement],
                    suggested_action="Calibrate measurement apparatus or reduce measurement frequency"
                )
                anomalies.append(anomaly)
                
        except Exception as e:
            logger.warning(f"Error checking measurement error: {e}")
        
        return anomalies

class SuperpositionMonitor:
    """Real-time superposition monitoring system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.analyzer = QuantumStateAnalyzer(config.get('analyzer', {}))
        self.anomaly_detector = SuperpositionAnomalyDetector(config.get('anomaly_detector', {}))
        self.redis_client = redis.Redis(**config.get('redis', {}))
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 4))
        
        # Monitoring state
        self.monitoring_active = False
        self.agent_states = {}
        self.performance_metrics = {}
        
    async def start_monitoring(self):
        """Start real-time superposition monitoring."""
        self.monitoring_active = True
        logger.info("Starting superposition monitoring system")
        
        # Start monitoring tasks
        monitoring_tasks = [
            asyncio.create_task(self._monitor_superposition_states()),
            asyncio.create_task(self._monitor_performance_metrics()),
            asyncio.create_task(self._monitor_anomalies()),
            asyncio.create_task(self._update_prometheus_metrics())
        ]
        
        await asyncio.gather(*monitoring_tasks)
    
    async def stop_monitoring(self):
        """Stop monitoring system."""
        self.monitoring_active = False
        logger.info("Stopping superposition monitoring system")
    
    async def _monitor_superposition_states(self):
        """Monitor quantum superposition states."""
        while self.monitoring_active:
            try:
                # Get active agents
                active_agents = self._get_active_agents()
                
                for agent_id in active_agents:
                    # Measure superposition state
                    measurement = await self._measure_superposition_state(agent_id)
                    
                    if measurement:
                        # Analyze state
                        analysis = self.analyzer.analyze_quantum_state(measurement)
                        
                        # Store state
                        self.agent_states[agent_id] = {
                            'measurement': measurement,
                            'analysis': analysis,
                            'timestamp': datetime.utcnow()
                        }
                        
                        # Detect anomalies
                        anomalies = self.anomaly_detector.detect_anomalies(measurement)
                        
                        if anomalies:
                            await self._handle_anomalies(anomalies)
                
                await asyncio.sleep(0.1)  # 100ms monitoring interval
                
            except Exception as e:
                logger.error(f"Error monitoring superposition states: {e}")
                await asyncio.sleep(1)
    
    async def _monitor_performance_metrics(self):
        """Monitor performance metrics."""
        while self.monitoring_active:
            try:
                for agent_id, state_data in self.agent_states.items():
                    measurement = state_data['measurement']
                    analysis = state_data['analysis']
                    
                    # Calculate effectiveness score
                    effectiveness = self._calculate_effectiveness_score(measurement, analysis)
                    
                    # Store performance metrics
                    self.performance_metrics[agent_id] = {
                        'effectiveness': effectiveness,
                        'coherence': measurement.coherence,
                        'fidelity': measurement.fidelity,
                        'entropy': measurement.entropy,
                        'execution_time': measurement.measurement_duration_ms,
                        'error_rate': measurement.quantum_error_rate,
                        'timestamp': datetime.utcnow()
                    }
                
                await asyncio.sleep(1)  # 1s performance monitoring interval
                
            except Exception as e:
                logger.error(f"Error monitoring performance metrics: {e}")
                await asyncio.sleep(1)
    
    async def _monitor_anomalies(self):
        """Monitor for anomalies and trigger alerts."""
        while self.monitoring_active:
            try:
                # Check for critical anomalies
                critical_anomalies = []
                
                for anomaly in self.anomaly_detector.anomaly_history:
                    if anomaly.severity > 0.8:  # Critical threshold
                        critical_anomalies.append(anomaly)
                
                if critical_anomalies:
                    await self._trigger_critical_alerts(critical_anomalies)
                
                await asyncio.sleep(5)  # 5s anomaly monitoring interval
                
            except Exception as e:
                logger.error(f"Error monitoring anomalies: {e}")
                await asyncio.sleep(1)
    
    async def _update_prometheus_metrics(self):
        """Update Prometheus metrics."""
        while self.monitoring_active:
            try:
                for agent_id, metrics in self.performance_metrics.items():
                    # Update Prometheus gauges
                    SUPERPOSITION_EFFECTIVENESS.labels(agent_id=agent_id).set(metrics['effectiveness'])
                    SUPERPOSITION_COHERENCE.labels(agent_id=agent_id, superposition_type='primary').set(metrics['coherence'])
                    SUPERPOSITION_FIDELITY.labels(agent_id=agent_id, superposition_type='primary').set(metrics['fidelity'])
                    SUPERPOSITION_ENTROPY.labels(agent_id=agent_id, superposition_type='primary').set(metrics['entropy'])
                    QUANTUM_DECOHERENCE_RATE.labels(agent_id=agent_id).set(1.0 - metrics['coherence'])
                
                await asyncio.sleep(1)  # 1s metrics update interval
                
            except Exception as e:
                logger.error(f"Error updating Prometheus metrics: {e}")
                await asyncio.sleep(1)
    
    def _get_active_agents(self) -> List[str]:
        """Get list of active agents."""
        # This would typically query the agent registry
        # For now, return a placeholder list
        return ['strategic_agent', 'tactical_agent', 'risk_agent', 'execution_agent']
    
    async def _measure_superposition_state(self, agent_id: str) -> Optional[SuperpositionMeasurement]:
        """Measure quantum superposition state for an agent."""
        start_time = time.time()
        
        try:
            # This would typically interface with the quantum state measurement system
            # For now, generate synthetic measurements
            
            # Simulate quantum measurement
            n_qubits = 4
            amplitude = np.random.complex128(2**n_qubits)
            amplitude /= np.linalg.norm(amplitude)  # Normalize
            
            # Calculate metrics
            coherence = np.random.uniform(0.5, 1.0)
            fidelity = np.random.uniform(0.8, 1.0)
            entropy = np.random.uniform(0.1, 2.0)
            phase = np.random.complex128()
            error_rate = np.random.uniform(0.001, 0.01)
            
            measurement_duration = (time.time() - start_time) * 1000
            
            measurement = SuperpositionMeasurement(
                agent_id=agent_id,
                timestamp=datetime.utcnow(),
                state=SuperpositionState.COHERENT,
                coherence=coherence,
                fidelity=fidelity,
                entropy=entropy,
                phase=phase,
                amplitude=amplitude,
                measurement_duration_ms=measurement_duration,
                quantum_error_rate=error_rate
            )
            
            return measurement
            
        except Exception as e:
            logger.error(f"Error measuring superposition state for {agent_id}: {e}")
            return None
    
    def _calculate_effectiveness_score(self, measurement: SuperpositionMeasurement, analysis: Dict[str, Any]) -> float:
        """Calculate overall superposition effectiveness score."""
        try:
            # Weighted combination of metrics
            weights = {
                'coherence': 0.3,
                'fidelity': 0.3,
                'entropy': 0.2,
                'error_rate': 0.2
            }
            
            score = (
                weights['coherence'] * measurement.coherence +
                weights['fidelity'] * measurement.fidelity +
                weights['entropy'] * min(measurement.entropy / 2.0, 1.0) +
                weights['error_rate'] * (1.0 - measurement.quantum_error_rate)
            )
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.warning(f"Error calculating effectiveness score: {e}")
            return 0.0
    
    async def _handle_anomalies(self, anomalies: List[SuperpositionAnomaly]):
        """Handle detected anomalies."""
        for anomaly in anomalies:
            logger.warning(f"Anomaly detected in {anomaly.agent_id}: {anomaly.description}")
            
            # Store anomaly in Redis
            anomaly_key = f"anomaly:{anomaly.agent_id}:{anomaly.timestamp.timestamp()}"
            await self.redis_client.setex(anomaly_key, 3600, json.dumps(anomaly.to_dict()))
            
            # Trigger alerts for severe anomalies
            if anomaly.severity > 0.7:
                await self._trigger_alert(anomaly)
    
    async def _trigger_alert(self, anomaly: SuperpositionAnomaly):
        """Trigger alert for anomaly."""
        alert_data = {
            'type': 'superposition_anomaly',
            'agent_id': anomaly.agent_id,
            'anomaly_type': anomaly.anomaly_type.value,
            'severity': anomaly.severity,
            'description': anomaly.description,
            'timestamp': anomaly.timestamp.isoformat(),
            'suggested_action': anomaly.suggested_action
        }
        
        # Store alert in Redis
        alert_key = f"alert:superposition:{anomaly.agent_id}:{int(time.time())}"
        await self.redis_client.setex(alert_key, 3600, json.dumps(alert_data))
        
        logger.error(f"ALERT: {alert_data}")
    
    async def _trigger_critical_alerts(self, anomalies: List[SuperpositionAnomaly]):
        """Trigger critical alerts."""
        for anomaly in anomalies:
            critical_alert = {
                'type': 'critical_superposition_anomaly',
                'agent_id': anomaly.agent_id,
                'anomaly_type': anomaly.anomaly_type.value,
                'severity': anomaly.severity,
                'description': anomaly.description,
                'timestamp': anomaly.timestamp.isoformat(),
                'suggested_action': anomaly.suggested_action
            }
            
            # Store critical alert
            alert_key = f"critical_alert:superposition:{anomaly.agent_id}:{int(time.time())}"
            await self.redis_client.setex(alert_key, 7200, json.dumps(critical_alert))
            
            logger.critical(f"CRITICAL ALERT: {critical_alert}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return {
            'monitoring_active': self.monitoring_active,
            'monitored_agents': list(self.agent_states.keys()),
            'performance_metrics': self.performance_metrics,
            'anomaly_count': len(self.anomaly_detector.anomaly_history),
            'last_update': datetime.utcnow().isoformat()
        }

# Factory function
def create_superposition_monitor(config: Dict[str, Any]) -> SuperpositionMonitor:
    """Create superposition monitor instance."""
    return SuperpositionMonitor(config)

# Example configuration
EXAMPLE_CONFIG = {
    'analyzer': {
        'coherence_threshold': 0.7,
        'fidelity_threshold': 0.95,
        'entropy_threshold': 2.0
    },
    'anomaly_detector': {
        'detection_thresholds': {
            'decoherence_spike_threshold': 0.2,
            'fidelity_threshold': 0.9,
            'entropy_threshold': 0.5,
            'phase_drift_threshold': 0.1,
            'execution_timeout_ms': 1000,
            'measurement_error_threshold': 0.05
        }
    },
    'redis': {
        'host': 'localhost',
        'port': 6379,
        'db': 0
    },
    'max_workers': 4
}

# Example usage
async def main():
    """Example usage of superposition monitoring system."""
    config = EXAMPLE_CONFIG
    monitor = create_superposition_monitor(config)
    
    # Start monitoring
    await monitor.start_monitoring()

if __name__ == "__main__":
    asyncio.run(main())