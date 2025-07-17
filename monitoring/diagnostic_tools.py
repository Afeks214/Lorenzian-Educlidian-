#!/usr/bin/env python3
"""
Diagnostic Tools for Superposition Behavior Analysis
Advanced analysis and debugging tools for quantum superposition states
"""

import asyncio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import scipy.stats as stats
from scipy.fft import fft, ifft
from scipy.signal import hilbert, spectrogram
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
from concurrent.futures import ThreadPoolExecutor
import threading

# Import monitoring components
from .superposition_monitoring import SuperpositionMeasurement, SuperpositionState, SuperpositionMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiagnosticType(Enum):
    """Types of diagnostic analyses."""
    COHERENCE_ANALYSIS = "coherence_analysis"
    PHASE_SPACE_ANALYSIS = "phase_space_analysis"
    SPECTRAL_ANALYSIS = "spectral_analysis"
    CORRELATION_ANALYSIS = "correlation_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    PERFORMANCE_PROFILING = "performance_profiling"
    QUANTUM_TOMOGRAPHY = "quantum_tomography"
    ENTANGLEMENT_ANALYSIS = "entanglement_analysis"

@dataclass
class DiagnosticResult:
    """Diagnostic analysis result."""
    diagnostic_type: DiagnosticType
    agent_id: str
    timestamp: datetime
    analysis_duration_ms: float
    findings: Dict[str, Any]
    visualizations: Dict[str, str]  # Base64 encoded plots
    recommendations: List[str]
    severity: str
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'diagnostic_type': self.diagnostic_type.value,
            'agent_id': self.agent_id,
            'timestamp': self.timestamp.isoformat(),
            'analysis_duration_ms': self.analysis_duration_ms,
            'findings': self.findings,
            'visualizations': self.visualizations,
            'recommendations': self.recommendations,
            'severity': self.severity,
            'confidence': self.confidence
        }

class CoherenceAnalyzer:
    """Analyzes quantum coherence patterns."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.coherence_threshold = config.get('coherence_threshold', 0.7)
        
    def analyze_coherence_patterns(self, measurements: List[SuperpositionMeasurement]) -> Dict[str, Any]:
        """Analyze coherence patterns over time."""
        if not measurements:
            return {}
        
        # Extract coherence values
        coherence_values = [m.coherence for m in measurements]
        timestamps = [m.timestamp for m in measurements]
        
        # Statistical analysis
        coherence_stats = {
            'mean': np.mean(coherence_values),
            'std': np.std(coherence_values),
            'min': np.min(coherence_values),
            'max': np.max(coherence_values),
            'median': np.median(coherence_values),
            'skewness': stats.skew(coherence_values),
            'kurtosis': stats.kurtosis(coherence_values)
        }
        
        # Trend analysis
        if len(coherence_values) > 1:
            x = np.arange(len(coherence_values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, coherence_values)
            coherence_stats['trend_slope'] = slope
            coherence_stats['trend_r_squared'] = r_value**2
            coherence_stats['trend_p_value'] = p_value
        
        # Stability analysis
        coherence_stability = self._analyze_stability(coherence_values)
        
        # Decoherence rate analysis
        decoherence_analysis = self._analyze_decoherence_rate(coherence_values, timestamps)
        
        return {
            'statistics': coherence_stats,
            'stability': coherence_stability,
            'decoherence': decoherence_analysis,
            'threshold_violations': sum(1 for c in coherence_values if c < self.coherence_threshold)
        }
    
    def _analyze_stability(self, coherence_values: List[float]) -> Dict[str, Any]:
        """Analyze coherence stability."""
        if len(coherence_values) < 2:
            return {'stability_score': 0.0}
        
        # Calculate variance and coefficient of variation
        variance = np.var(coherence_values)
        cv = np.std(coherence_values) / np.mean(coherence_values) if np.mean(coherence_values) > 0 else 0
        
        # Calculate stability score (1 - normalized variance)
        stability_score = max(0, 1 - (variance / 0.25))  # Normalize by typical variance
        
        # Detect stability periods
        stability_periods = self._detect_stability_periods(coherence_values)
        
        return {
            'stability_score': stability_score,
            'coefficient_of_variation': cv,
            'variance': variance,
            'stability_periods': stability_periods
        }
    
    def _detect_stability_periods(self, coherence_values: List[float]) -> List[Dict[str, Any]]:
        """Detect periods of stable coherence."""
        if len(coherence_values) < 5:
            return []
        
        periods = []
        current_period_start = 0
        stability_threshold = 0.05  # 5% variation threshold
        
        for i in range(1, len(coherence_values)):
            window = coherence_values[max(0, i-4):i+1]
            if np.std(window) > stability_threshold:
                if i - current_period_start > 5:  # Minimum period length
                    periods.append({
                        'start': current_period_start,
                        'end': i-1,
                        'duration': i - current_period_start,
                        'mean_coherence': np.mean(coherence_values[current_period_start:i])
                    })
                current_period_start = i
        
        # Add final period if stable
        if len(coherence_values) - current_period_start > 5:
            periods.append({
                'start': current_period_start,
                'end': len(coherence_values)-1,
                'duration': len(coherence_values) - current_period_start,
                'mean_coherence': np.mean(coherence_values[current_period_start:])
            })
        
        return periods
    
    def _analyze_decoherence_rate(self, coherence_values: List[float], timestamps: List[datetime]) -> Dict[str, Any]:
        """Analyze decoherence rate."""
        if len(coherence_values) < 2:
            return {'decoherence_rate': 0.0}
        
        # Calculate time differences
        time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
        coherence_diffs = [coherence_values[i+1] - coherence_values[i] for i in range(len(coherence_values)-1)]
        
        # Calculate decoherence rates
        decoherence_rates = []
        for i, (time_diff, coherence_diff) in enumerate(zip(time_diffs, coherence_diffs)):
            if time_diff > 0 and coherence_diff < 0:  # Decoherence event
                rate = -coherence_diff / time_diff
                decoherence_rates.append(rate)
        
        if not decoherence_rates:
            return {'decoherence_rate': 0.0}
        
        return {
            'decoherence_rate': np.mean(decoherence_rates),
            'decoherence_std': np.std(decoherence_rates),
            'decoherence_events': len(decoherence_rates),
            'max_decoherence_rate': np.max(decoherence_rates)
        }

class PhaseSpaceAnalyzer:
    """Analyzes quantum phase space dynamics."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def analyze_phase_space(self, measurements: List[SuperpositionMeasurement]) -> Dict[str, Any]:
        """Analyze phase space dynamics."""
        if not measurements:
            return {}
        
        # Extract phase information
        phases = [np.angle(m.phase) for m in measurements]
        amplitudes = [np.abs(m.phase) for m in measurements]
        
        # Phase space trajectory analysis
        trajectory_analysis = self._analyze_trajectory(phases, amplitudes)
        
        # Phase coherence analysis
        phase_coherence = self._analyze_phase_coherence(phases)
        
        # Amplitude dynamics
        amplitude_dynamics = self._analyze_amplitude_dynamics(amplitudes)
        
        return {
            'trajectory': trajectory_analysis,
            'phase_coherence': phase_coherence,
            'amplitude_dynamics': amplitude_dynamics
        }
    
    def _analyze_trajectory(self, phases: List[float], amplitudes: List[float]) -> Dict[str, Any]:
        """Analyze phase space trajectory."""
        if len(phases) < 2:
            return {}
        
        # Calculate trajectory properties
        trajectory_length = sum(
            np.sqrt((phases[i+1] - phases[i])**2 + (amplitudes[i+1] - amplitudes[i])**2)
            for i in range(len(phases)-1)
        )
        
        # Calculate curvature
        curvatures = []
        for i in range(1, len(phases)-1):
            # Approximate curvature using three points
            p1 = np.array([phases[i-1], amplitudes[i-1]])
            p2 = np.array([phases[i], amplitudes[i]])
            p3 = np.array([phases[i+1], amplitudes[i+1]])
            
            # Calculate curvature
            v1 = p2 - p1
            v2 = p3 - p2
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cross_product = v1[0] * v2[1] - v1[1] * v2[0]
                curvature = cross_product / (np.linalg.norm(v1) * np.linalg.norm(v2))
                curvatures.append(abs(curvature))
        
        # Analyze attractors and repellers
        attractor_analysis = self._analyze_attractors(phases, amplitudes)
        
        return {
            'trajectory_length': trajectory_length,
            'mean_curvature': np.mean(curvatures) if curvatures else 0,
            'max_curvature': np.max(curvatures) if curvatures else 0,
            'attractor_analysis': attractor_analysis
        }
    
    def _analyze_phase_coherence(self, phases: List[float]) -> Dict[str, Any]:
        """Analyze phase coherence."""
        if not phases:
            return {}
        
        # Calculate phase coherence order parameter
        complex_phases = [np.exp(1j * phase) for phase in phases]
        coherence_order = abs(np.mean(complex_phases))
        
        # Phase variance
        phase_variance = np.var(phases)
        
        # Phase diffusion coefficient
        phase_diffs = [phases[i+1] - phases[i] for i in range(len(phases)-1)]
        phase_diffusion = np.var(phase_diffs) / 2 if len(phase_diffs) > 1 else 0
        
        return {
            'coherence_order': coherence_order,
            'phase_variance': phase_variance,
            'phase_diffusion': phase_diffusion,
            'phase_spread': np.ptp(phases)  # Peak-to-peak spread
        }
    
    def _analyze_amplitude_dynamics(self, amplitudes: List[float]) -> Dict[str, Any]:
        """Analyze amplitude dynamics."""
        if not amplitudes:
            return {}
        
        # Statistical measures
        amplitude_stats = {
            'mean': np.mean(amplitudes),
            'std': np.std(amplitudes),
            'min': np.min(amplitudes),
            'max': np.max(amplitudes)
        }
        
        # Amplitude fluctuations
        amplitude_fluctuations = [amplitudes[i+1] - amplitudes[i] for i in range(len(amplitudes)-1)]
        fluctuation_stats = {
            'mean_fluctuation': np.mean(amplitude_fluctuations) if amplitude_fluctuations else 0,
            'fluctuation_std': np.std(amplitude_fluctuations) if amplitude_fluctuations else 0
        }
        
        return {
            'statistics': amplitude_stats,
            'fluctuations': fluctuation_stats
        }
    
    def _analyze_attractors(self, phases: List[float], amplitudes: List[float]) -> Dict[str, Any]:
        """Analyze attractors and repellers in phase space."""
        if len(phases) < 10:
            return {}
        
        # Create phase space points
        points = np.column_stack([phases, amplitudes])
        
        # Simple clustering to identify attractors
        from scipy.cluster.hierarchy import linkage, fcluster
        
        try:
            # Hierarchical clustering
            linkage_matrix = linkage(points, method='ward')
            clusters = fcluster(linkage_matrix, t=3, criterion='maxclust')
            
            # Analyze clusters as potential attractors
            attractors = []
            for cluster_id in np.unique(clusters):
                cluster_points = points[clusters == cluster_id]
                if len(cluster_points) > 5:  # Minimum points for attractor
                    center = np.mean(cluster_points, axis=0)
                    spread = np.std(cluster_points, axis=0)
                    attractors.append({
                        'center': center.tolist(),
                        'spread': spread.tolist(),
                        'points': len(cluster_points)
                    })
            
            return {
                'num_attractors': len(attractors),
                'attractors': attractors
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing attractors: {e}")
            return {}

class SpectralAnalyzer:
    """Analyzes frequency domain characteristics."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def analyze_spectrum(self, measurements: List[SuperpositionMeasurement]) -> Dict[str, Any]:
        """Analyze frequency spectrum of measurements."""
        if len(measurements) < 10:
            return {}
        
        # Extract time series
        coherence_series = np.array([m.coherence for m in measurements])
        fidelity_series = np.array([m.fidelity for m in measurements])
        entropy_series = np.array([m.entropy for m in measurements])
        
        # Perform FFT analysis
        coherence_fft = self._perform_fft_analysis(coherence_series)
        fidelity_fft = self._perform_fft_analysis(fidelity_series)
        entropy_fft = self._perform_fft_analysis(entropy_series)
        
        # Cross-correlation analysis
        cross_correlation = self._analyze_cross_correlation(coherence_series, fidelity_series)
        
        # Detect periodic patterns
        periodic_patterns = self._detect_periodic_patterns(coherence_series)
        
        return {
            'coherence_spectrum': coherence_fft,
            'fidelity_spectrum': fidelity_fft,
            'entropy_spectrum': entropy_fft,
            'cross_correlation': cross_correlation,
            'periodic_patterns': periodic_patterns
        }
    
    def _perform_fft_analysis(self, signal: np.ndarray) -> Dict[str, Any]:
        """Perform FFT analysis on signal."""
        # Detrend signal
        signal_detrended = signal - np.mean(signal)
        
        # Apply window function
        window = np.hanning(len(signal_detrended))
        signal_windowed = signal_detrended * window
        
        # Compute FFT
        fft_result = fft(signal_windowed)
        frequencies = np.fft.fftfreq(len(signal_windowed))
        
        # Calculate power spectral density
        psd = np.abs(fft_result)**2
        
        # Find dominant frequencies
        dominant_freqs = frequencies[np.argsort(psd)[-5:]]  # Top 5 frequencies
        
        return {
            'dominant_frequencies': dominant_freqs.tolist(),
            'power_spectral_density': psd[:len(psd)//2].tolist(),  # Only positive frequencies
            'total_power': np.sum(psd),
            'peak_frequency': frequencies[np.argmax(psd)],
            'spectral_centroid': np.sum(frequencies[:len(frequencies)//2] * psd[:len(psd)//2]) / np.sum(psd[:len(psd)//2])
        }
    
    def _analyze_cross_correlation(self, signal1: np.ndarray, signal2: np.ndarray) -> Dict[str, Any]:
        """Analyze cross-correlation between signals."""
        # Normalize signals
        signal1_norm = (signal1 - np.mean(signal1)) / np.std(signal1)
        signal2_norm = (signal2 - np.mean(signal2)) / np.std(signal2)
        
        # Calculate cross-correlation
        cross_corr = np.correlate(signal1_norm, signal2_norm, mode='full')
        
        # Find peak correlation and lag
        peak_idx = np.argmax(np.abs(cross_corr))
        peak_correlation = cross_corr[peak_idx]
        lag = peak_idx - len(signal1_norm) + 1
        
        return {
            'peak_correlation': float(peak_correlation),
            'lag': int(lag),
            'correlation_strength': float(np.abs(peak_correlation))
        }
    
    def _detect_periodic_patterns(self, signal: np.ndarray) -> Dict[str, Any]:
        """Detect periodic patterns in signal."""
        # Autocorrelation
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks in autocorrelation
        peak_indices = []
        for i in range(1, len(autocorr)-1):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                if autocorr[i] > 0.1 * np.max(autocorr):  # Threshold for significant peaks
                    peak_indices.append(i)
        
        # Calculate periods
        periods = []
        for peak_idx in peak_indices[:5]:  # Top 5 periods
            period = peak_idx
            strength = autocorr[peak_idx] / autocorr[0]  # Normalized strength
            periods.append({
                'period': period,
                'strength': float(strength)
            })
        
        return {
            'detected_periods': periods,
            'periodicity_score': float(np.max(autocorr[1:]) / autocorr[0]) if len(autocorr) > 1 else 0
        }

class QuantumTomographyAnalyzer:
    """Performs quantum state tomography analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def perform_tomography(self, measurements: List[SuperpositionMeasurement]) -> Dict[str, Any]:
        """Perform quantum state tomography."""
        if not measurements:
            return {}
        
        # Reconstruct density matrix
        density_matrix = self._reconstruct_density_matrix(measurements)
        
        # Calculate quantum properties
        quantum_properties = self._calculate_quantum_properties(density_matrix)
        
        # Analyze entanglement
        entanglement_analysis = self._analyze_entanglement(density_matrix)
        
        return {
            'density_matrix': density_matrix.tolist() if density_matrix is not None else None,
            'quantum_properties': quantum_properties,
            'entanglement': entanglement_analysis
        }
    
    def _reconstruct_density_matrix(self, measurements: List[SuperpositionMeasurement]) -> Optional[np.ndarray]:
        """Reconstruct density matrix from measurements."""
        try:
            # Extract amplitude data
            amplitudes = [m.amplitude for m in measurements]
            
            if not amplitudes:
                return None
            
            # Average over measurements to get state estimate
            n_qubits = int(np.log2(len(amplitudes[0])))
            state_estimate = np.mean(amplitudes, axis=0)
            state_estimate = state_estimate / np.linalg.norm(state_estimate)
            
            # Construct density matrix
            density_matrix = np.outer(state_estimate, state_estimate.conj())
            
            return density_matrix
            
        except Exception as e:
            logger.warning(f"Error reconstructing density matrix: {e}")
            return None
    
    def _calculate_quantum_properties(self, density_matrix: np.ndarray) -> Dict[str, Any]:
        """Calculate quantum properties from density matrix."""
        if density_matrix is None:
            return {}
        
        try:
            # Purity
            purity = np.trace(density_matrix @ density_matrix).real
            
            # Von Neumann entropy
            eigenvalues = np.linalg.eigvals(density_matrix)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove near-zero eigenvalues
            von_neumann_entropy = -np.sum(eigenvalues * np.log(eigenvalues)).real
            
            # Trace of density matrix (should be 1)
            trace = np.trace(density_matrix).real
            
            # Fidelity with maximally mixed state
            mixed_state = np.eye(density_matrix.shape[0]) / density_matrix.shape[0]
            fidelity_mixed = np.trace(np.sqrt(np.sqrt(density_matrix) @ mixed_state @ np.sqrt(density_matrix))).real
            
            return {
                'purity': float(purity),
                'von_neumann_entropy': float(von_neumann_entropy),
                'trace': float(trace),
                'fidelity_with_mixed': float(fidelity_mixed),
                'rank': int(np.linalg.matrix_rank(density_matrix))
            }
            
        except Exception as e:
            logger.warning(f"Error calculating quantum properties: {e}")
            return {}
    
    def _analyze_entanglement(self, density_matrix: np.ndarray) -> Dict[str, Any]:
        """Analyze entanglement properties."""
        if density_matrix is None:
            return {}
        
        try:
            # For multi-qubit systems, calculate entanglement measures
            n_qubits = int(np.log2(density_matrix.shape[0]))
            
            if n_qubits < 2:
                return {'entanglement_detected': False}
            
            # Calculate concurrence for 2-qubit systems
            if n_qubits == 2:
                concurrence = self._calculate_concurrence(density_matrix)
                return {
                    'entanglement_detected': concurrence > 0.01,
                    'concurrence': float(concurrence),
                    'n_qubits': n_qubits
                }
            
            # For higher dimensions, use entropy-based measures
            entanglement_entropy = self._calculate_entanglement_entropy(density_matrix)
            
            return {
                'entanglement_detected': entanglement_entropy > 0.01,
                'entanglement_entropy': float(entanglement_entropy),
                'n_qubits': n_qubits
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing entanglement: {e}")
            return {'entanglement_detected': False}
    
    def _calculate_concurrence(self, density_matrix: np.ndarray) -> float:
        """Calculate concurrence for 2-qubit state."""
        try:
            # Pauli-Y matrix
            sigma_y = np.array([[0, -1j], [1j, 0]])
            
            # Construct sigma_y tensor sigma_y
            sigma_yy = np.kron(sigma_y, sigma_y)
            
            # Calculate R matrix
            R = density_matrix @ sigma_yy @ density_matrix.conj() @ sigma_yy
            
            # Calculate eigenvalues of R
            eigenvalues = np.linalg.eigvals(R)
            eigenvalues = np.sqrt(np.sort(eigenvalues.real)[::-1])
            
            # Concurrence
            concurrence = max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3])
            
            return concurrence
            
        except Exception as e:
            logger.warning(f"Error calculating concurrence: {e}")
            return 0.0
    
    def _calculate_entanglement_entropy(self, density_matrix: np.ndarray) -> float:
        """Calculate entanglement entropy."""
        try:
            # Partial trace (simplified for demonstration)
            n = density_matrix.shape[0]
            sqrt_n = int(np.sqrt(n))
            
            if sqrt_n * sqrt_n != n:
                return 0.0
            
            # Reshape and trace out one subsystem
            reshaped = density_matrix.reshape(sqrt_n, sqrt_n, sqrt_n, sqrt_n)
            reduced_matrix = np.trace(reshaped, axis1=1, axis2=3)
            
            # Calculate entropy of reduced state
            eigenvalues = np.linalg.eigvals(reduced_matrix)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            
            entropy = -np.sum(eigenvalues * np.log(eigenvalues)).real
            
            return entropy
            
        except Exception as e:
            logger.warning(f"Error calculating entanglement entropy: {e}")
            return 0.0

class SuperpositionDiagnosticSuite:
    """Comprehensive diagnostic suite for superposition analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.coherence_analyzer = CoherenceAnalyzer(config.get('coherence_analyzer', {}))
        self.phase_space_analyzer = PhaseSpaceAnalyzer(config.get('phase_space_analyzer', {}))
        self.spectral_analyzer = SpectralAnalyzer(config.get('spectral_analyzer', {}))
        self.tomography_analyzer = QuantumTomographyAnalyzer(config.get('tomography_analyzer', {}))
        
        # Visualization settings
        self.save_plots = config.get('save_plots', True)
        self.plot_format = config.get('plot_format', 'png')
        
    async def run_comprehensive_analysis(self, measurements: List[SuperpositionMeasurement], agent_id: str) -> DiagnosticResult:
        """Run comprehensive diagnostic analysis."""
        start_time = time.time()
        
        # Run all analyses
        analyses = {}
        visualizations = {}
        
        try:
            # Coherence analysis
            coherence_analysis = self.coherence_analyzer.analyze_coherence_patterns(measurements)
            analyses['coherence'] = coherence_analysis
            
            # Phase space analysis
            phase_space_analysis = self.phase_space_analyzer.analyze_phase_space(measurements)
            analyses['phase_space'] = phase_space_analysis
            
            # Spectral analysis
            spectral_analysis = self.spectral_analyzer.analyze_spectrum(measurements)
            analyses['spectral'] = spectral_analysis
            
            # Quantum tomography
            tomography_analysis = self.tomography_analyzer.perform_tomography(measurements)
            analyses['tomography'] = tomography_analysis
            
            # Generate visualizations
            if self.save_plots:
                visualizations = await self._generate_visualizations(measurements, analyses, agent_id)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(analyses)
            
            # Calculate severity and confidence
            severity = self._calculate_severity(analyses)
            confidence = self._calculate_confidence(analyses, len(measurements))
            
            analysis_duration = (time.time() - start_time) * 1000
            
            return DiagnosticResult(
                diagnostic_type=DiagnosticType.COHERENCE_ANALYSIS,
                agent_id=agent_id,
                timestamp=datetime.utcnow(),
                analysis_duration_ms=analysis_duration,
                findings=analyses,
                visualizations=visualizations,
                recommendations=recommendations,
                severity=severity,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return DiagnosticResult(
                diagnostic_type=DiagnosticType.COHERENCE_ANALYSIS,
                agent_id=agent_id,
                timestamp=datetime.utcnow(),
                analysis_duration_ms=(time.time() - start_time) * 1000,
                findings={'error': str(e)},
                visualizations={},
                recommendations=['Analysis failed - check system logs'],
                severity='high',
                confidence=0.0
            )
    
    async def _generate_visualizations(self, measurements: List[SuperpositionMeasurement], 
                                     analyses: Dict[str, Any], agent_id: str) -> Dict[str, str]:
        """Generate diagnostic visualizations."""
        visualizations = {}
        
        try:
            # Coherence time series plot
            coherence_plot = self._create_coherence_plot(measurements)
            visualizations['coherence_timeseries'] = coherence_plot
            
            # Phase space plot
            phase_space_plot = self._create_phase_space_plot(measurements)
            visualizations['phase_space'] = phase_space_plot
            
            # Spectral analysis plot
            if 'spectral' in analyses and analyses['spectral']:
                spectral_plot = self._create_spectral_plot(analyses['spectral'])
                visualizations['spectrum'] = spectral_plot
            
            # Correlation heatmap
            correlation_plot = self._create_correlation_heatmap(measurements)
            visualizations['correlation'] = correlation_plot
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
        
        return visualizations
    
    def _create_coherence_plot(self, measurements: List[SuperpositionMeasurement]) -> str:
        """Create coherence time series plot."""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
            
            times = [m.timestamp for m in measurements]
            coherence = [m.coherence for m in measurements]
            fidelity = [m.fidelity for m in measurements]
            entropy = [m.entropy for m in measurements]
            
            # Coherence plot
            ax1.plot(times, coherence, 'b-', linewidth=2, label='Coherence')
            ax1.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Threshold')
            ax1.set_ylabel('Coherence')
            ax1.set_title('Quantum Coherence Analysis')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Fidelity plot
            ax2.plot(times, fidelity, 'g-', linewidth=2, label='Fidelity')
            ax2.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='Threshold')
            ax2.set_ylabel('Fidelity')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Entropy plot
            ax3.plot(times, entropy, 'orange', linewidth=2, label='Entropy')
            ax3.set_ylabel('Entropy')
            ax3.set_xlabel('Time')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            return plot_data
            
        except Exception as e:
            logger.error(f"Error creating coherence plot: {e}")
            return ""
    
    def _create_phase_space_plot(self, measurements: List[SuperpositionMeasurement]) -> str:
        """Create phase space plot."""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            phases = [np.angle(m.phase) for m in measurements]
            amplitudes = [np.abs(m.phase) for m in measurements]
            
            # Create scatter plot with trajectory
            scatter = ax.scatter(phases, amplitudes, c=range(len(phases)), 
                               cmap='viridis', alpha=0.7, s=50)
            
            # Add trajectory line
            ax.plot(phases, amplitudes, 'k-', alpha=0.3, linewidth=1)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Time Step')
            
            ax.set_xlabel('Phase (radians)')
            ax.set_ylabel('Amplitude')
            ax.set_title('Quantum Phase Space Trajectory')
            ax.grid(True, alpha=0.3)
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            return plot_data
            
        except Exception as e:
            logger.error(f"Error creating phase space plot: {e}")
            return ""
    
    def _create_spectral_plot(self, spectral_analysis: Dict[str, Any]) -> str:
        """Create spectral analysis plot."""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Power spectral density
            if 'coherence_spectrum' in spectral_analysis:
                psd = spectral_analysis['coherence_spectrum'].get('power_spectral_density', [])
                if psd:
                    frequencies = np.linspace(0, 0.5, len(psd))
                    ax1.plot(frequencies, psd, 'b-', linewidth=2)
                    ax1.set_ylabel('Power Spectral Density')
                    ax1.set_title('Coherence Spectrum')
                    ax1.grid(True, alpha=0.3)
            
            # Dominant frequencies
            if 'coherence_spectrum' in spectral_analysis:
                dominant_freqs = spectral_analysis['coherence_spectrum'].get('dominant_frequencies', [])
                if dominant_freqs:
                    ax2.bar(range(len(dominant_freqs)), dominant_freqs, color='orange', alpha=0.7)
                    ax2.set_ylabel('Frequency')
                    ax2.set_xlabel('Rank')
                    ax2.set_title('Dominant Frequencies')
                    ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            return plot_data
            
        except Exception as e:
            logger.error(f"Error creating spectral plot: {e}")
            return ""
    
    def _create_correlation_heatmap(self, measurements: List[SuperpositionMeasurement]) -> str:
        """Create correlation heatmap."""
        try:
            # Create data matrix
            data = {
                'coherence': [m.coherence for m in measurements],
                'fidelity': [m.fidelity for m in measurements],
                'entropy': [m.entropy for m in measurements],
                'error_rate': [m.quantum_error_rate for m in measurements]
            }
            
            df = pd.DataFrame(data)
            correlation_matrix = df.corr()
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                       center=0, square=True, ax=ax)
            ax.set_title('Quantum Metrics Correlation Matrix')
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            return plot_data
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {e}")
            return ""
    
    def _generate_recommendations(self, analyses: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        try:
            # Coherence recommendations
            if 'coherence' in analyses:
                coherence_analysis = analyses['coherence']
                
                if 'statistics' in coherence_analysis:
                    stats = coherence_analysis['statistics']
                    
                    if stats.get('mean', 0) < 0.7:
                        recommendations.append("Low coherence detected - consider reducing environmental noise")
                    
                    if stats.get('std', 0) > 0.2:
                        recommendations.append("High coherence variance - check system stability")
                    
                    if stats.get('trend_slope', 0) < -0.01:
                        recommendations.append("Declining coherence trend - investigate decoherence sources")
                
                if coherence_analysis.get('threshold_violations', 0) > 0:
                    recommendations.append("Coherence threshold violations detected - system tuning required")
            
            # Phase space recommendations
            if 'phase_space' in analyses:
                phase_analysis = analyses['phase_space']
                
                if 'trajectory' in phase_analysis:
                    trajectory = phase_analysis['trajectory']
                    
                    if trajectory.get('mean_curvature', 0) > 0.5:
                        recommendations.append("High phase space curvature - consider smoother control")
                
                if 'phase_coherence' in phase_analysis:
                    phase_coherence = phase_analysis['phase_coherence']
                    
                    if phase_coherence.get('coherence_order', 0) < 0.5:
                        recommendations.append("Low phase coherence - check phase reference stability")
            
            # Spectral recommendations
            if 'spectral' in analyses:
                spectral_analysis = analyses['spectral']
                
                if 'periodic_patterns' in spectral_analysis:
                    patterns = spectral_analysis['periodic_patterns']
                    
                    if patterns.get('periodicity_score', 0) > 0.8:
                        recommendations.append("Strong periodic behavior detected - potential oscillatory instability")
            
            # Quantum tomography recommendations
            if 'tomography' in analyses:
                tomography_analysis = analyses['tomography']
                
                if 'quantum_properties' in tomography_analysis:
                    properties = tomography_analysis['quantum_properties']
                    
                    if properties.get('purity', 0) < 0.8:
                        recommendations.append("Low quantum purity - increase isolation from environment")
                    
                    if properties.get('von_neumann_entropy', 0) > 2.0:
                        recommendations.append("High quantum entropy - consider state preparation optimization")
            
            # Default recommendation if no issues found
            if not recommendations:
                recommendations.append("System operating within normal parameters")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Analysis incomplete - check system logs for details")
        
        return recommendations
    
    def _calculate_severity(self, analyses: Dict[str, Any]) -> str:
        """Calculate overall severity based on analysis results."""
        severity_score = 0
        
        try:
            # Coherence severity
            if 'coherence' in analyses:
                coherence_analysis = analyses['coherence']
                
                if 'statistics' in coherence_analysis:
                    stats = coherence_analysis['statistics']
                    
                    if stats.get('mean', 1) < 0.5:
                        severity_score += 3
                    elif stats.get('mean', 1) < 0.7:
                        severity_score += 2
                    elif stats.get('mean', 1) < 0.8:
                        severity_score += 1
                    
                    if stats.get('std', 0) > 0.3:
                        severity_score += 2
                    elif stats.get('std', 0) > 0.2:
                        severity_score += 1
                
                if coherence_analysis.get('threshold_violations', 0) > 0:
                    severity_score += 2
            
            # Phase space severity
            if 'phase_space' in analyses:
                phase_analysis = analyses['phase_space']
                
                if 'trajectory' in phase_analysis:
                    trajectory = phase_analysis['trajectory']
                    
                    if trajectory.get('mean_curvature', 0) > 0.7:
                        severity_score += 2
                    elif trajectory.get('mean_curvature', 0) > 0.5:
                        severity_score += 1
            
            # Determine severity level
            if severity_score >= 6:
                return 'critical'
            elif severity_score >= 4:
                return 'high'
            elif severity_score >= 2:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            logger.error(f"Error calculating severity: {e}")
            return 'medium'
    
    def _calculate_confidence(self, analyses: Dict[str, Any], num_measurements: int) -> float:
        """Calculate confidence in analysis results."""
        try:
            # Base confidence on number of measurements
            if num_measurements < 10:
                base_confidence = 0.3
            elif num_measurements < 50:
                base_confidence = 0.6
            elif num_measurements < 100:
                base_confidence = 0.8
            else:
                base_confidence = 0.9
            
            # Adjust based on analysis completeness
            analysis_count = len(analyses)
            completeness_factor = min(analysis_count / 4, 1.0)  # 4 main analyses
            
            # Adjust based on data quality
            quality_factor = 1.0  # Placeholder - would assess data quality
            
            confidence = base_confidence * completeness_factor * quality_factor
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5

# Factory function
def create_diagnostic_suite(config: Dict[str, Any]) -> SuperpositionDiagnosticSuite:
    """Create diagnostic suite instance."""
    return SuperpositionDiagnosticSuite(config)

# Example configuration
EXAMPLE_CONFIG = {
    'coherence_analyzer': {
        'coherence_threshold': 0.7
    },
    'phase_space_analyzer': {},
    'spectral_analyzer': {},
    'tomography_analyzer': {},
    'save_plots': True,
    'plot_format': 'png'
}

# Example usage
async def main():
    """Example usage of diagnostic suite."""
    config = EXAMPLE_CONFIG
    diagnostic_suite = create_diagnostic_suite(config)
    
    # Create sample measurements
    measurements = []
    for i in range(100):
        measurement = SuperpositionMeasurement(
            agent_id='test_agent',
            timestamp=datetime.utcnow(),
            state=SuperpositionState.COHERENT,
            coherence=0.8 + 0.2 * np.random.normal(),
            fidelity=0.9 + 0.1 * np.random.normal(),
            entropy=1.0 + 0.5 * np.random.normal(),
            phase=np.random.complex128(),
            amplitude=np.random.complex128(4),
            measurement_duration_ms=10 + 5 * np.random.normal(),
            quantum_error_rate=0.01 + 0.005 * np.random.normal()
        )
        measurements.append(measurement)
    
    # Run analysis
    result = await diagnostic_suite.run_comprehensive_analysis(measurements, 'test_agent')
    
    print(f"Analysis completed in {result.analysis_duration_ms:.2f}ms")
    print(f"Severity: {result.severity}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Recommendations: {result.recommendations}")

if __name__ == "__main__":
    asyncio.run(main())