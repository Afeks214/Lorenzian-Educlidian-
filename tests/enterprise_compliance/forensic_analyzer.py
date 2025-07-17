#!/usr/bin/env python3
"""
Forensic Analyzer for Enterprise Audit Trails
Agent Zeta: Enterprise Compliance & Chaos Engineering Implementation Specialist

Advanced forensic analysis capabilities for audit trail investigation, 
compliance verification, and security incident response.

Features:
- Timeline reconstruction and correlation analysis
- Pattern detection and anomaly identification
- Evidence collection and chain of custody
- Digital forensics with cryptographic verification
- Automated incident response and root cause analysis
- Compliance verification and violation tracking
"""

import asyncio
import json
import logging
import time
import hashlib
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import sqlite3
from collections import defaultdict, deque
import networkx as nx
import numpy as np
from scipy import stats
import re
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle
import gzip

logger = logging.getLogger(__name__)


class ForensicEventType(Enum):
    """Types of forensic events"""
    SECURITY_INCIDENT = "security_incident"
    COMPLIANCE_VIOLATION = "compliance_violation"
    SYSTEM_ANOMALY = "system_anomaly"
    DATA_BREACH = "data_breach"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    POLICY_VIOLATION = "policy_violation"
    FRAUD_DETECTION = "fraud_detection"
    SYSTEM_COMPROMISE = "system_compromise"


class AnalysisType(Enum):
    """Types of forensic analysis"""
    TIMELINE_RECONSTRUCTION = "timeline_reconstruction"
    PATTERN_ANALYSIS = "pattern_analysis"
    CORRELATION_ANALYSIS = "correlation_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    CAUSALITY_ANALYSIS = "causality_analysis"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"
    NETWORK_ANALYSIS = "network_analysis"
    EVIDENCE_COLLECTION = "evidence_collection"


class EvidenceType(Enum):
    """Types of digital evidence"""
    AUDIT_LOG = "audit_log"
    BLOCKCHAIN_TRANSACTION = "blockchain_transaction"
    SYSTEM_STATE = "system_state"
    USER_ACTION = "user_action"
    NETWORK_TRAFFIC = "network_traffic"
    DATABASE_RECORD = "database_record"
    CRYPTOGRAPHIC_PROOF = "cryptographic_proof"
    WITNESS_TESTIMONY = "witness_testimony"


@dataclass
class ForensicEvidence:
    """Digital evidence record"""
    evidence_id: str
    evidence_type: EvidenceType
    source_system: str
    collected_at: datetime
    
    # Evidence data
    raw_data: Dict[str, Any]
    processed_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Chain of custody
    collected_by: str = "forensic_analyzer"
    custody_chain: List[Dict[str, Any]] = field(default_factory=list)
    
    # Integrity verification
    data_hash: Optional[str] = None
    digital_signature: Optional[str] = None
    verification_status: str = "pending"
    
    # Analysis results
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 0.0
    confidence_score: float = 0.0
    
    def __post_init__(self):
        if self.data_hash is None:
            self.data_hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """Calculate evidence hash for integrity"""
        evidence_data = {
            'evidence_id': self.evidence_id,
            'evidence_type': self.evidence_type.value,
            'source_system': self.source_system,
            'collected_at': self.collected_at.isoformat(),
            'raw_data': self.raw_data,
            'collected_by': self.collected_by
        }
        
        evidence_string = json.dumps(evidence_data, sort_keys=True, default=str)
        return hashlib.sha256(evidence_string.encode()).hexdigest()


@dataclass
class ForensicTimeline:
    """Forensic timeline reconstruction"""
    timeline_id: str
    incident_id: str
    created_at: datetime
    
    # Timeline events
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    # Analysis metadata
    time_range: Tuple[datetime, datetime] = field(default_factory=lambda: (datetime.now(timezone.utc), datetime.now(timezone.utc)))
    correlation_confidence: float = 0.0
    completeness_score: float = 0.0
    
    # Findings
    key_findings: List[str] = field(default_factory=list)
    attack_vectors: List[str] = field(default_factory=list)
    impact_assessment: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ForensicPattern:
    """Detected forensic pattern"""
    pattern_id: str
    pattern_type: str
    description: str
    confidence: float
    
    # Pattern characteristics
    signature: str
    frequency: int
    time_window: timedelta
    
    # Associated events
    matching_events: List[str] = field(default_factory=list)
    related_patterns: List[str] = field(default_factory=list)
    
    # Analysis results
    risk_score: float = 0.0
    severity: str = "medium"
    recommendations: List[str] = field(default_factory=list)


class ForensicAnalyzer:
    """
    Advanced Forensic Analyzer for Enterprise Audit Trails
    
    Provides comprehensive forensic analysis capabilities including:
    - Timeline reconstruction and event correlation
    - Pattern detection and anomaly identification
    - Evidence collection with chain of custody
    - Digital forensics with cryptographic verification
    - Automated incident response and root cause analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Initialize storage
        self.db_connection = self._initialize_database()
        
        # Analysis engines
        self.pattern_detector = PatternDetector()
        self.anomaly_detector = AnomalyDetector()
        self.correlation_engine = CorrelationEngine()
        self.timeline_reconstructor = TimelineReconstructor()
        
        # Evidence management
        self.evidence_store: Dict[str, ForensicEvidence] = {}
        self.custody_chain = []
        
        # Performance tracking
        self.analysis_metrics = {
            'total_analyses': 0,
            'evidence_collected': 0,
            'patterns_detected': 0,
            'anomalies_found': 0,
            'timelines_reconstructed': 0,
            'avg_analysis_time_ms': 0.0
        }
        
        logger.info("ForensicAnalyzer initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'analysis': {
                'pattern_detection_threshold': 0.7,
                'anomaly_detection_threshold': 0.8,
                'correlation_threshold': 0.6,
                'timeline_window_hours': 24,
                'max_evidence_age_days': 90
            },
            'performance': {
                'max_concurrent_analyses': 10,
                'analysis_timeout_seconds': 300,
                'cache_size': 1000
            },
            'storage': {
                'database_path': '/tmp/forensic_analysis.db',
                'evidence_retention_days': 365,
                'compression_enabled': True
            }
        }
    
    def _initialize_database(self) -> sqlite3.Connection:
        """Initialize forensic analysis database"""
        db_path = self.config['storage']['database_path']
        conn = sqlite3.connect(db_path, check_same_thread=False)
        
        # Create forensic tables
        conn.execute('''
            CREATE TABLE IF NOT EXISTS forensic_evidence (
                evidence_id TEXT PRIMARY KEY,
                evidence_type TEXT NOT NULL,
                source_system TEXT NOT NULL,
                collected_at TEXT NOT NULL,
                raw_data TEXT NOT NULL,
                processed_data TEXT,
                metadata TEXT,
                collected_by TEXT,
                custody_chain TEXT,
                data_hash TEXT,
                digital_signature TEXT,
                verification_status TEXT,
                analysis_results TEXT,
                relevance_score REAL,
                confidence_score REAL
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS forensic_timelines (
                timeline_id TEXT PRIMARY KEY,
                incident_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                events TEXT,
                time_range_start TEXT,
                time_range_end TEXT,
                correlation_confidence REAL,
                completeness_score REAL,
                key_findings TEXT,
                attack_vectors TEXT,
                impact_assessment TEXT
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS forensic_patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT NOT NULL,
                description TEXT,
                confidence REAL,
                signature TEXT,
                frequency INTEGER,
                time_window_seconds INTEGER,
                matching_events TEXT,
                related_patterns TEXT,
                risk_score REAL,
                severity TEXT,
                recommendations TEXT
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS forensic_incidents (
                incident_id TEXT PRIMARY KEY,
                incident_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                description TEXT,
                affected_systems TEXT,
                evidence_ids TEXT,
                timeline_id TEXT,
                analysis_results TEXT,
                remediation_actions TEXT
            )
        ''')
        
        # Create indices
        conn.execute('CREATE INDEX IF NOT EXISTS idx_evidence_collected_at ON forensic_evidence(collected_at)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_evidence_type ON forensic_evidence(evidence_type)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_timelines_incident ON forensic_timelines(incident_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_patterns_type ON forensic_patterns(pattern_type)')
        
        conn.commit()
        return conn
    
    async def collect_evidence(
        self,
        evidence_type: EvidenceType,
        source_system: str,
        raw_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        collected_by: str = "forensic_analyzer"
    ) -> ForensicEvidence:
        """
        Collect digital evidence with chain of custody
        
        Args:
            evidence_type: Type of evidence being collected
            source_system: Source system identifier
            raw_data: Raw evidence data
            metadata: Additional metadata
            collected_by: Identifier of collector
            
        Returns:
            ForensicEvidence: Collected evidence record
        """
        evidence = ForensicEvidence(
            evidence_id=f"evidence_{uuid.uuid4().hex}",
            evidence_type=evidence_type,
            source_system=source_system,
            collected_at=datetime.now(timezone.utc),
            raw_data=raw_data,
            metadata=metadata or {},
            collected_by=collected_by
        )
        
        # Add to custody chain
        evidence.custody_chain.append({
            'action': 'collected',
            'timestamp': evidence.collected_at.isoformat(),
            'actor': collected_by,
            'location': source_system,
            'hash': evidence.data_hash
        })
        
        # Store evidence
        await self._store_evidence(evidence)
        self.evidence_store[evidence.evidence_id] = evidence
        
        # Update metrics
        self.analysis_metrics['evidence_collected'] += 1
        
        logger.info(f"Evidence collected: {evidence.evidence_id}")
        return evidence
    
    async def _store_evidence(self, evidence: ForensicEvidence):
        """Store evidence in database"""
        cursor = self.db_connection.cursor()
        cursor.execute('''
            INSERT INTO forensic_evidence (
                evidence_id, evidence_type, source_system, collected_at,
                raw_data, processed_data, metadata, collected_by,
                custody_chain, data_hash, digital_signature,
                verification_status, analysis_results, relevance_score,
                confidence_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            evidence.evidence_id,
            evidence.evidence_type.value,
            evidence.source_system,
            evidence.collected_at.isoformat(),
            json.dumps(evidence.raw_data),
            json.dumps(evidence.processed_data) if evidence.processed_data else None,
            json.dumps(evidence.metadata),
            evidence.collected_by,
            json.dumps(evidence.custody_chain),
            evidence.data_hash,
            evidence.digital_signature,
            evidence.verification_status,
            json.dumps(evidence.analysis_results),
            evidence.relevance_score,
            evidence.confidence_score
        ))
        self.db_connection.commit()
    
    async def analyze_timeline(
        self,
        incident_id: str,
        time_range: Tuple[datetime, datetime],
        evidence_filters: Optional[Dict[str, Any]] = None
    ) -> ForensicTimeline:
        """
        Reconstruct forensic timeline for incident
        
        Args:
            incident_id: Incident identifier
            time_range: Time range for analysis
            evidence_filters: Filters for evidence selection
            
        Returns:
            ForensicTimeline: Reconstructed timeline
        """
        start_time = time.time()
        
        # Collect relevant evidence
        evidence_list = await self._collect_timeline_evidence(time_range, evidence_filters)
        
        # Reconstruct timeline
        timeline = await self.timeline_reconstructor.reconstruct(
            incident_id, evidence_list, time_range
        )
        
        # Store timeline
        await self._store_timeline(timeline)
        
        # Update metrics
        analysis_time = (time.time() - start_time) * 1000
        self.analysis_metrics['timelines_reconstructed'] += 1
        self._update_avg_analysis_time(analysis_time)
        
        logger.info(f"Timeline reconstructed: {timeline.timeline_id}")
        return timeline
    
    async def _collect_timeline_evidence(
        self,
        time_range: Tuple[datetime, datetime],
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ForensicEvidence]:
        """Collect evidence for timeline reconstruction"""
        cursor = self.db_connection.cursor()
        
        # Build query
        where_conditions = ["collected_at BETWEEN ? AND ?"]
        params = [time_range[0].isoformat(), time_range[1].isoformat()]
        
        if filters:
            for key, value in filters.items():
                if key == 'evidence_type':
                    where_conditions.append("evidence_type = ?")
                    params.append(value)
                elif key == 'source_system':
                    where_conditions.append("source_system = ?")
                    params.append(value)
                elif key == 'min_relevance':
                    where_conditions.append("relevance_score >= ?")
                    params.append(value)
        
        where_clause = " AND ".join(where_conditions)
        cursor.execute(f'''
            SELECT * FROM forensic_evidence
            WHERE {where_clause}
            ORDER BY collected_at ASC
        ''', params)
        
        results = cursor.fetchall()
        
        # Convert to evidence objects
        evidence_list = []
        for row in results:
            evidence = ForensicEvidence(
                evidence_id=row[0],
                evidence_type=EvidenceType(row[1]),
                source_system=row[2],
                collected_at=datetime.fromisoformat(row[3]),
                raw_data=json.loads(row[4]),
                processed_data=json.loads(row[5]) if row[5] else None,
                metadata=json.loads(row[6]),
                collected_by=row[7],
                custody_chain=json.loads(row[8]),
                data_hash=row[9],
                digital_signature=row[10],
                verification_status=row[11],
                analysis_results=json.loads(row[12]),
                relevance_score=row[13],
                confidence_score=row[14]
            )
            evidence_list.append(evidence)
        
        return evidence_list
    
    async def _store_timeline(self, timeline: ForensicTimeline):
        """Store timeline in database"""
        cursor = self.db_connection.cursor()
        cursor.execute('''
            INSERT INTO forensic_timelines (
                timeline_id, incident_id, created_at, events,
                time_range_start, time_range_end, correlation_confidence,
                completeness_score, key_findings, attack_vectors,
                impact_assessment
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timeline.timeline_id,
            timeline.incident_id,
            timeline.created_at.isoformat(),
            json.dumps(timeline.events),
            timeline.time_range[0].isoformat(),
            timeline.time_range[1].isoformat(),
            timeline.correlation_confidence,
            timeline.completeness_score,
            json.dumps(timeline.key_findings),
            json.dumps(timeline.attack_vectors),
            json.dumps(timeline.impact_assessment)
        ))
        self.db_connection.commit()
    
    async def detect_patterns(
        self,
        analysis_window: timedelta = timedelta(hours=24),
        pattern_types: Optional[List[str]] = None
    ) -> List[ForensicPattern]:
        """
        Detect forensic patterns in evidence
        
        Args:
            analysis_window: Time window for pattern detection
            pattern_types: Types of patterns to detect
            
        Returns:
            List[ForensicPattern]: Detected patterns
        """
        start_time = time.time()
        
        # Get evidence for analysis
        end_time = datetime.now(timezone.utc)
        start_time_window = end_time - analysis_window
        
        evidence_list = await self._collect_timeline_evidence(
            (start_time_window, end_time),
            {'min_relevance': 0.3}
        )
        
        # Detect patterns
        patterns = await self.pattern_detector.detect_patterns(
            evidence_list, pattern_types
        )
        
        # Store patterns
        for pattern in patterns:
            await self._store_pattern(pattern)
        
        # Update metrics
        analysis_time = (time.time() - start_time) * 1000
        self.analysis_metrics['patterns_detected'] += len(patterns)
        self._update_avg_analysis_time(analysis_time)
        
        logger.info(f"Detected {len(patterns)} forensic patterns")
        return patterns
    
    async def _store_pattern(self, pattern: ForensicPattern):
        """Store pattern in database"""
        cursor = self.db_connection.cursor()
        cursor.execute('''
            INSERT INTO forensic_patterns (
                pattern_id, pattern_type, description, confidence,
                signature, frequency, time_window_seconds,
                matching_events, related_patterns, risk_score,
                severity, recommendations
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pattern.pattern_id,
            pattern.pattern_type,
            pattern.description,
            pattern.confidence,
            pattern.signature,
            pattern.frequency,
            int(pattern.time_window.total_seconds()),
            json.dumps(pattern.matching_events),
            json.dumps(pattern.related_patterns),
            pattern.risk_score,
            pattern.severity,
            json.dumps(pattern.recommendations)
        ))
        self.db_connection.commit()
    
    async def detect_anomalies(
        self,
        baseline_window: timedelta = timedelta(days=7),
        analysis_window: timedelta = timedelta(hours=24)
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies in evidence patterns
        
        Args:
            baseline_window: Window for baseline calculation
            analysis_window: Window for anomaly detection
            
        Returns:
            List[Dict[str, Any]]: Detected anomalies
        """
        start_time = time.time()
        
        # Get baseline and analysis data
        end_time = datetime.now(timezone.utc)
        baseline_start = end_time - baseline_window
        analysis_start = end_time - analysis_window
        
        baseline_evidence = await self._collect_timeline_evidence(
            (baseline_start, analysis_start)
        )
        
        analysis_evidence = await self._collect_timeline_evidence(
            (analysis_start, end_time)
        )
        
        # Detect anomalies
        anomalies = await self.anomaly_detector.detect_anomalies(
            baseline_evidence, analysis_evidence
        )
        
        # Update metrics
        analysis_time = (time.time() - start_time) * 1000
        self.analysis_metrics['anomalies_found'] += len(anomalies)
        self._update_avg_analysis_time(analysis_time)
        
        logger.info(f"Detected {len(anomalies)} anomalies")
        return anomalies
    
    async def correlate_events(
        self,
        event_ids: List[str],
        correlation_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        Correlate events for forensic analysis
        
        Args:
            event_ids: Event IDs to correlate
            correlation_threshold: Minimum correlation threshold
            
        Returns:
            Dict[str, Any]: Correlation analysis results
        """
        # Get evidence for events
        evidence_list = []
        for event_id in event_ids:
            if event_id in self.evidence_store:
                evidence_list.append(self.evidence_store[event_id])
        
        # Perform correlation analysis
        correlation_results = await self.correlation_engine.correlate_events(
            evidence_list, correlation_threshold
        )
        
        return correlation_results
    
    async def generate_forensic_report(
        self,
        incident_id: str,
        include_timeline: bool = True,
        include_patterns: bool = True,
        include_evidence: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive forensic report
        
        Args:
            incident_id: Incident identifier
            include_timeline: Include timeline reconstruction
            include_patterns: Include pattern analysis
            include_evidence: Include evidence details
            
        Returns:
            Dict[str, Any]: Forensic report
        """
        report = {
            'report_id': f"forensic_{incident_id}_{uuid.uuid4().hex[:8]}",
            'incident_id': incident_id,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'analyst': 'forensic_analyzer',
            'summary': {}
        }
        
        # Get incident details
        cursor = self.db_connection.cursor()
        cursor.execute('''
            SELECT * FROM forensic_incidents
            WHERE incident_id = ?
        ''', (incident_id,))
        
        incident_data = cursor.fetchone()
        if incident_data:
            report['incident_details'] = {
                'type': incident_data[1],
                'severity': incident_data[2],
                'status': incident_data[3],
                'created_at': incident_data[4],
                'description': incident_data[6],
                'affected_systems': json.loads(incident_data[7]) if incident_data[7] else []
            }
        
        # Include timeline if requested
        if include_timeline:
            cursor.execute('''
                SELECT * FROM forensic_timelines
                WHERE incident_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            ''', (incident_id,))
            
            timeline_data = cursor.fetchone()
            if timeline_data:
                report['timeline'] = {
                    'timeline_id': timeline_data[0],
                    'events': json.loads(timeline_data[3]),
                    'time_range': [timeline_data[4], timeline_data[5]],
                    'correlation_confidence': timeline_data[6],
                    'completeness_score': timeline_data[7],
                    'key_findings': json.loads(timeline_data[8]),
                    'attack_vectors': json.loads(timeline_data[9]),
                    'impact_assessment': json.loads(timeline_data[10])
                }
        
        # Include patterns if requested
        if include_patterns:
            cursor.execute('''
                SELECT * FROM forensic_patterns
                WHERE pattern_id IN (
                    SELECT pattern_id FROM forensic_patterns
                    WHERE matching_events LIKE ?
                )
                ORDER BY confidence DESC
            ''', (f'%{incident_id}%',))
            
            pattern_data = cursor.fetchall()
            report['patterns'] = []
            for pattern in pattern_data:
                report['patterns'].append({
                    'pattern_id': pattern[0],
                    'pattern_type': pattern[1],
                    'description': pattern[2],
                    'confidence': pattern[3],
                    'risk_score': pattern[9],
                    'severity': pattern[10],
                    'recommendations': json.loads(pattern[11])
                })
        
        # Include evidence if requested
        if include_evidence:
            cursor.execute('''
                SELECT * FROM forensic_evidence
                WHERE evidence_id IN (
                    SELECT evidence_id FROM forensic_evidence
                    WHERE raw_data LIKE ?
                )
                ORDER BY collected_at ASC
            ''', (f'%{incident_id}%',))
            
            evidence_data = cursor.fetchall()
            report['evidence'] = []
            for evidence in evidence_data:
                report['evidence'].append({
                    'evidence_id': evidence[0],
                    'evidence_type': evidence[1],
                    'source_system': evidence[2],
                    'collected_at': evidence[3],
                    'relevance_score': evidence[13],
                    'confidence_score': evidence[14],
                    'verification_status': evidence[11]
                })
        
        # Generate summary
        report['summary'] = {
            'total_evidence': len(report.get('evidence', [])),
            'total_patterns': len(report.get('patterns', [])),
            'timeline_available': include_timeline and 'timeline' in report,
            'overall_confidence': self._calculate_overall_confidence(report),
            'recommendations': self._generate_recommendations(report)
        }
        
        return report
    
    def _calculate_overall_confidence(self, report: Dict[str, Any]) -> float:
        """Calculate overall confidence score"""
        confidence_scores = []
        
        # Timeline confidence
        if 'timeline' in report:
            confidence_scores.append(report['timeline']['correlation_confidence'])
        
        # Pattern confidence
        if 'patterns' in report:
            pattern_confidences = [p['confidence'] for p in report['patterns']]
            if pattern_confidences:
                confidence_scores.append(np.mean(pattern_confidences))
        
        # Evidence confidence
        if 'evidence' in report:
            evidence_confidences = [e['confidence_score'] for e in report['evidence']]
            if evidence_confidences:
                confidence_scores.append(np.mean(evidence_confidences))
        
        return np.mean(confidence_scores) if confidence_scores else 0.0
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Timeline recommendations
        if 'timeline' in report:
            if report['timeline']['completeness_score'] < 0.8:
                recommendations.append("Collect additional evidence to improve timeline completeness")
        
        # Pattern recommendations
        if 'patterns' in report:
            high_risk_patterns = [p for p in report['patterns'] if p['risk_score'] > 0.7]
            if high_risk_patterns:
                recommendations.append("Address high-risk patterns identified in analysis")
        
        # Evidence recommendations
        if 'evidence' in report:
            unverified_evidence = [e for e in report['evidence'] if e['verification_status'] != 'verified']
            if unverified_evidence:
                recommendations.append("Verify digital signatures and integrity of unverified evidence")
        
        return recommendations
    
    def _update_avg_analysis_time(self, analysis_time: float):
        """Update average analysis time metric"""
        total_analyses = self.analysis_metrics['total_analyses']
        old_avg = self.analysis_metrics['avg_analysis_time_ms']
        
        self.analysis_metrics['total_analyses'] += 1
        self.analysis_metrics['avg_analysis_time_ms'] = (
            (old_avg * total_analyses + analysis_time) / (total_analyses + 1)
        )
    
    def get_analysis_metrics(self) -> Dict[str, Any]:
        """Get forensic analysis metrics"""
        return self.analysis_metrics.copy()


# Helper classes for specialized analysis
class PatternDetector:
    """Pattern detection engine"""
    
    async def detect_patterns(
        self,
        evidence_list: List[ForensicEvidence],
        pattern_types: Optional[List[str]] = None
    ) -> List[ForensicPattern]:
        """Detect patterns in evidence"""
        patterns = []
        
        # Detect sequence patterns
        sequence_patterns = await self._detect_sequence_patterns(evidence_list)
        patterns.extend(sequence_patterns)
        
        # Detect frequency patterns
        frequency_patterns = await self._detect_frequency_patterns(evidence_list)
        patterns.extend(frequency_patterns)
        
        # Detect behavioral patterns
        behavioral_patterns = await self._detect_behavioral_patterns(evidence_list)
        patterns.extend(behavioral_patterns)
        
        return patterns
    
    async def _detect_sequence_patterns(self, evidence_list: List[ForensicEvidence]) -> List[ForensicPattern]:
        """Detect sequence patterns"""
        patterns = []
        
        # Group by source system
        system_events = defaultdict(list)
        for evidence in evidence_list:
            system_events[evidence.source_system].append(evidence)
        
        # Look for common sequences
        for system, events in system_events.items():
            if len(events) >= 3:
                # Sort by time
                events.sort(key=lambda x: x.collected_at)
                
                # Look for repeating sequences
                sequences = []
                for i in range(len(events) - 2):
                    sequence = [
                        events[i].evidence_type.value,
                        events[i+1].evidence_type.value,
                        events[i+2].evidence_type.value
                    ]
                    sequences.append(sequence)
                
                # Find common sequences
                sequence_counts = defaultdict(int)
                for seq in sequences:
                    sequence_counts[tuple(seq)] += 1
                
                # Create patterns for frequent sequences
                for sequence, count in sequence_counts.items():
                    if count >= 2:
                        pattern = ForensicPattern(
                            pattern_id=f"seq_{uuid.uuid4().hex[:8]}",
                            pattern_type="sequence",
                            description=f"Repeating sequence pattern: {' -> '.join(sequence)}",
                            confidence=min(count / len(sequences), 1.0),
                            signature=str(sequence),
                            frequency=count,
                            time_window=timedelta(minutes=30),
                            matching_events=[e.evidence_id for e in events],
                            risk_score=0.6,
                            severity="medium"
                        )
                        patterns.append(pattern)
        
        return patterns
    
    async def _detect_frequency_patterns(self, evidence_list: List[ForensicEvidence]) -> List[ForensicPattern]:
        """Detect frequency patterns"""
        patterns = []
        
        # Analyze event frequencies
        event_types = defaultdict(list)
        for evidence in evidence_list:
            event_types[evidence.evidence_type].append(evidence.collected_at)
        
        # Look for unusual frequencies
        for event_type, timestamps in event_types.items():
            if len(timestamps) >= 5:
                # Calculate time differences
                timestamps.sort()
                intervals = []
                for i in range(1, len(timestamps)):
                    interval = (timestamps[i] - timestamps[i-1]).total_seconds()
                    intervals.append(interval)
                
                # Check for regular intervals (potential automation)
                if intervals:
                    mean_interval = np.mean(intervals)
                    std_interval = np.std(intervals)
                    
                    # If standard deviation is low, it might be automated
                    if std_interval < mean_interval * 0.2 and len(intervals) >= 5:
                        pattern = ForensicPattern(
                            pattern_id=f"freq_{uuid.uuid4().hex[:8]}",
                            pattern_type="frequency",
                            description=f"Regular frequency pattern for {event_type.value}",
                            confidence=0.8,
                            signature=f"interval_{mean_interval:.2f}s",
                            frequency=len(timestamps),
                            time_window=timedelta(seconds=mean_interval),
                            risk_score=0.7,
                            severity="high"
                        )
                        patterns.append(pattern)
        
        return patterns
    
    async def _detect_behavioral_patterns(self, evidence_list: List[ForensicEvidence]) -> List[ForensicPattern]:
        """Detect behavioral patterns"""
        patterns = []
        
        # Analyze user behavior patterns
        user_activities = defaultdict(list)
        for evidence in evidence_list:
            if 'user_id' in evidence.raw_data:
                user_id = evidence.raw_data['user_id']
                user_activities[user_id].append(evidence)
        
        # Look for unusual user behavior
        for user_id, activities in user_activities.items():
            if len(activities) >= 3:
                # Check for off-hours activity
                off_hours_count = 0
                for activity in activities:
                    hour = activity.collected_at.hour
                    if hour < 6 or hour > 22:  # Outside business hours
                        off_hours_count += 1
                
                if off_hours_count / len(activities) > 0.5:
                    pattern = ForensicPattern(
                        pattern_id=f"behav_{uuid.uuid4().hex[:8]}",
                        pattern_type="behavioral",
                        description=f"Off-hours activity pattern for user {user_id}",
                        confidence=0.7,
                        signature=f"user_{user_id}_off_hours",
                        frequency=off_hours_count,
                        time_window=timedelta(hours=24),
                        risk_score=0.8,
                        severity="high"
                    )
                    patterns.append(pattern)
        
        return patterns


class AnomalyDetector:
    """Anomaly detection engine"""
    
    async def detect_anomalies(
        self,
        baseline_evidence: List[ForensicEvidence],
        analysis_evidence: List[ForensicEvidence]
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in evidence"""
        anomalies = []
        
        # Volume anomalies
        volume_anomalies = await self._detect_volume_anomalies(baseline_evidence, analysis_evidence)
        anomalies.extend(volume_anomalies)
        
        # Temporal anomalies
        temporal_anomalies = await self._detect_temporal_anomalies(baseline_evidence, analysis_evidence)
        anomalies.extend(temporal_anomalies)
        
        # Source anomalies
        source_anomalies = await self._detect_source_anomalies(baseline_evidence, analysis_evidence)
        anomalies.extend(source_anomalies)
        
        return anomalies
    
    async def _detect_volume_anomalies(self, baseline: List[ForensicEvidence], analysis: List[ForensicEvidence]) -> List[Dict[str, Any]]:
        """Detect volume anomalies"""
        anomalies = []
        
        # Compare evidence volumes
        baseline_count = len(baseline)
        analysis_count = len(analysis)
        
        if baseline_count > 0:
            volume_ratio = analysis_count / baseline_count
            
            # Check for significant volume changes
            if volume_ratio > 3.0:  # 3x increase
                anomalies.append({
                    'type': 'volume_spike',
                    'description': f'Evidence volume increased by {volume_ratio:.1f}x',
                    'severity': 'high',
                    'confidence': 0.9,
                    'baseline_count': baseline_count,
                    'analysis_count': analysis_count
                })
            elif volume_ratio < 0.3:  # 70% decrease
                anomalies.append({
                    'type': 'volume_drop',
                    'description': f'Evidence volume decreased by {(1-volume_ratio)*100:.1f}%',
                    'severity': 'medium',
                    'confidence': 0.8,
                    'baseline_count': baseline_count,
                    'analysis_count': analysis_count
                })
        
        return anomalies
    
    async def _detect_temporal_anomalies(self, baseline: List[ForensicEvidence], analysis: List[ForensicEvidence]) -> List[Dict[str, Any]]:
        """Detect temporal anomalies"""
        anomalies = []
        
        # Analyze time distributions
        if baseline and analysis:
            baseline_hours = [e.collected_at.hour for e in baseline]
            analysis_hours = [e.collected_at.hour for e in analysis]
            
            # Statistical test for different distributions
            if len(baseline_hours) >= 5 and len(analysis_hours) >= 5:
                statistic, p_value = stats.ks_2samp(baseline_hours, analysis_hours)
                
                if p_value < 0.05:  # Significant difference
                    anomalies.append({
                        'type': 'temporal_shift',
                        'description': 'Significant change in temporal distribution of events',
                        'severity': 'medium',
                        'confidence': 1 - p_value,
                        'statistic': statistic,
                        'p_value': p_value
                    })
        
        return anomalies
    
    async def _detect_source_anomalies(self, baseline: List[ForensicEvidence], analysis: List[ForensicEvidence]) -> List[Dict[str, Any]]:
        """Detect source system anomalies"""
        anomalies = []
        
        # Compare source system distributions
        baseline_sources = defaultdict(int)
        analysis_sources = defaultdict(int)
        
        for evidence in baseline:
            baseline_sources[evidence.source_system] += 1
        
        for evidence in analysis:
            analysis_sources[evidence.source_system] += 1
        
        # Check for new sources
        new_sources = set(analysis_sources.keys()) - set(baseline_sources.keys())
        if new_sources:
            anomalies.append({
                'type': 'new_sources',
                'description': f'New evidence sources detected: {", ".join(new_sources)}',
                'severity': 'high',
                'confidence': 0.9,
                'new_sources': list(new_sources)
            })
        
        # Check for missing sources
        missing_sources = set(baseline_sources.keys()) - set(analysis_sources.keys())
        if missing_sources:
            anomalies.append({
                'type': 'missing_sources',
                'description': f'Expected sources not found: {", ".join(missing_sources)}',
                'severity': 'medium',
                'confidence': 0.8,
                'missing_sources': list(missing_sources)
            })
        
        return anomalies


class CorrelationEngine:
    """Event correlation engine"""
    
    async def correlate_events(
        self,
        evidence_list: List[ForensicEvidence],
        threshold: float = 0.6
    ) -> Dict[str, Any]:
        """Correlate events for forensic analysis"""
        correlation_matrix = {}
        correlations = []
        
        # Calculate pairwise correlations
        for i, evidence1 in enumerate(evidence_list):
            for j, evidence2 in enumerate(evidence_list[i+1:], i+1):
                correlation = self._calculate_correlation(evidence1, evidence2)
                if correlation >= threshold:
                    correlations.append({
                        'evidence1': evidence1.evidence_id,
                        'evidence2': evidence2.evidence_id,
                        'correlation': correlation,
                        'factors': self._identify_correlation_factors(evidence1, evidence2)
                    })
        
        # Build correlation graph
        correlation_graph = nx.Graph()
        for evidence in evidence_list:
            correlation_graph.add_node(evidence.evidence_id)
        
        for correlation in correlations:
            correlation_graph.add_edge(
                correlation['evidence1'],
                correlation['evidence2'],
                weight=correlation['correlation']
            )
        
        # Find correlation clusters
        clusters = list(nx.connected_components(correlation_graph))
        
        return {
            'correlations': correlations,
            'clusters': [list(cluster) for cluster in clusters],
            'correlation_score': np.mean([c['correlation'] for c in correlations]) if correlations else 0.0,
            'total_correlations': len(correlations)
        }
    
    def _calculate_correlation(self, evidence1: ForensicEvidence, evidence2: ForensicEvidence) -> float:
        """Calculate correlation between two pieces of evidence"""
        correlation_factors = []
        
        # Time correlation
        time_diff = abs((evidence1.collected_at - evidence2.collected_at).total_seconds())
        time_correlation = max(0, 1 - (time_diff / 3600))  # 1 hour window
        correlation_factors.append(time_correlation)
        
        # Source correlation
        source_correlation = 1.0 if evidence1.source_system == evidence2.source_system else 0.3
        correlation_factors.append(source_correlation)
        
        # Type correlation
        type_correlation = 1.0 if evidence1.evidence_type == evidence2.evidence_type else 0.5
        correlation_factors.append(type_correlation)
        
        # User correlation
        user1 = evidence1.raw_data.get('user_id')
        user2 = evidence2.raw_data.get('user_id')
        user_correlation = 1.0 if user1 and user2 and user1 == user2 else 0.0
        correlation_factors.append(user_correlation)
        
        # Content correlation (simplified)
        content_correlation = self._calculate_content_correlation(evidence1.raw_data, evidence2.raw_data)
        correlation_factors.append(content_correlation)
        
        return np.mean(correlation_factors)
    
    def _calculate_content_correlation(self, data1: Dict[str, Any], data2: Dict[str, Any]) -> float:
        """Calculate content correlation between evidence data"""
        # Simple content correlation based on common keys
        keys1 = set(data1.keys())
        keys2 = set(data2.keys())
        
        common_keys = keys1.intersection(keys2)
        if not common_keys:
            return 0.0
        
        jaccard_similarity = len(common_keys) / len(keys1.union(keys2))
        return jaccard_similarity
    
    def _identify_correlation_factors(self, evidence1: ForensicEvidence, evidence2: ForensicEvidence) -> List[str]:
        """Identify factors contributing to correlation"""
        factors = []
        
        # Time proximity
        time_diff = abs((evidence1.collected_at - evidence2.collected_at).total_seconds())
        if time_diff < 300:  # 5 minutes
            factors.append('temporal_proximity')
        
        # Same source
        if evidence1.source_system == evidence2.source_system:
            factors.append('same_source_system')
        
        # Same user
        if evidence1.raw_data.get('user_id') == evidence2.raw_data.get('user_id'):
            factors.append('same_user')
        
        # Same type
        if evidence1.evidence_type == evidence2.evidence_type:
            factors.append('same_evidence_type')
        
        return factors


class TimelineReconstructor:
    """Timeline reconstruction engine"""
    
    async def reconstruct(
        self,
        incident_id: str,
        evidence_list: List[ForensicEvidence],
        time_range: Tuple[datetime, datetime]
    ) -> ForensicTimeline:
        """Reconstruct forensic timeline"""
        # Sort evidence by time
        evidence_list.sort(key=lambda x: x.collected_at)
        
        # Create timeline events
        events = []
        for evidence in evidence_list:
            event = {
                'timestamp': evidence.collected_at.isoformat(),
                'evidence_id': evidence.evidence_id,
                'evidence_type': evidence.evidence_type.value,
                'source_system': evidence.source_system,
                'description': self._generate_event_description(evidence),
                'impact': self._assess_event_impact(evidence),
                'confidence': evidence.confidence_score
            }
            events.append(event)
        
        # Calculate timeline metrics
        correlation_confidence = self._calculate_correlation_confidence(evidence_list)
        completeness_score = self._calculate_completeness_score(evidence_list, time_range)
        
        # Generate findings
        key_findings = self._generate_key_findings(evidence_list)
        attack_vectors = self._identify_attack_vectors(evidence_list)
        impact_assessment = self._assess_overall_impact(evidence_list)
        
        timeline = ForensicTimeline(
            timeline_id=f"timeline_{uuid.uuid4().hex}",
            incident_id=incident_id,
            created_at=datetime.now(timezone.utc),
            events=events,
            time_range=time_range,
            correlation_confidence=correlation_confidence,
            completeness_score=completeness_score,
            key_findings=key_findings,
            attack_vectors=attack_vectors,
            impact_assessment=impact_assessment
        )
        
        return timeline
    
    def _generate_event_description(self, evidence: ForensicEvidence) -> str:
        """Generate human-readable event description"""
        base_desc = f"{evidence.evidence_type.value} from {evidence.source_system}"
        
        # Add specific details based on evidence type
        if evidence.evidence_type == EvidenceType.USER_ACTION:
            if 'action' in evidence.raw_data:
                return f"User performed {evidence.raw_data['action']} on {evidence.source_system}"
        elif evidence.evidence_type == EvidenceType.SYSTEM_STATE:
            if 'state' in evidence.raw_data:
                return f"System state changed to {evidence.raw_data['state']} on {evidence.source_system}"
        
        return base_desc
    
    def _assess_event_impact(self, evidence: ForensicEvidence) -> str:
        """Assess impact of individual event"""
        if evidence.raw_data.get('financial_impact'):
            return 'high'
        elif evidence.raw_data.get('regulatory_impact'):
            return 'high'
        elif evidence.raw_data.get('risk_level', 0) > 7:
            return 'high'
        elif evidence.raw_data.get('risk_level', 0) > 4:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_correlation_confidence(self, evidence_list: List[ForensicEvidence]) -> float:
        """Calculate correlation confidence for timeline"""
        if len(evidence_list) < 2:
            return 0.0
        
        # Calculate time gaps
        time_gaps = []
        for i in range(1, len(evidence_list)):
            gap = (evidence_list[i].collected_at - evidence_list[i-1].collected_at).total_seconds()
            time_gaps.append(gap)
        
        # Lower confidence for large gaps
        if time_gaps:
            max_gap = max(time_gaps)
            avg_gap = np.mean(time_gaps)
            
            # Confidence decreases with larger gaps
            confidence = max(0.0, 1.0 - (max_gap / 86400))  # 24 hours max
            return confidence
        
        return 0.8  # Default confidence
    
    def _calculate_completeness_score(self, evidence_list: List[ForensicEvidence], time_range: Tuple[datetime, datetime]) -> float:
        """Calculate timeline completeness score"""
        if not evidence_list:
            return 0.0
        
        # Calculate time coverage
        total_time = (time_range[1] - time_range[0]).total_seconds()
        
        # Calculate time gaps
        evidence_times = [e.collected_at for e in evidence_list]
        evidence_times.sort()
        
        covered_time = 0
        for i in range(len(evidence_times) - 1):
            gap = (evidence_times[i+1] - evidence_times[i]).total_seconds()
            if gap < 3600:  # 1 hour max gap for coverage
                covered_time += gap
        
        completeness = min(1.0, covered_time / total_time)
        return completeness
    
    def _generate_key_findings(self, evidence_list: List[ForensicEvidence]) -> List[str]:
        """Generate key findings from evidence"""
        findings = []
        
        # High-confidence evidence
        high_confidence = [e for e in evidence_list if e.confidence_score > 0.8]
        if high_confidence:
            findings.append(f"Found {len(high_confidence)} high-confidence evidence items")
        
        # Multiple source systems
        sources = set(e.source_system for e in evidence_list)
        if len(sources) > 1:
            findings.append(f"Evidence spans {len(sources)} different systems: {', '.join(sources)}")
        
        # Temporal patterns
        if len(evidence_list) >= 3:
            time_diffs = []
            for i in range(1, len(evidence_list)):
                diff = (evidence_list[i].collected_at - evidence_list[i-1].collected_at).total_seconds()
                time_diffs.append(diff)
            
            if time_diffs and np.std(time_diffs) < np.mean(time_diffs) * 0.3:
                findings.append("Regular temporal pattern detected in evidence")
        
        return findings
    
    def _identify_attack_vectors(self, evidence_list: List[ForensicEvidence]) -> List[str]:
        """Identify potential attack vectors"""
        vectors = []
        
        # Check for unauthorized access
        unauthorized_access = [e for e in evidence_list if e.evidence_type == EvidenceType.UNAUTHORIZED_ACCESS]
        if unauthorized_access:
            vectors.append("unauthorized_access")
        
        # Check for system compromises
        compromises = [e for e in evidence_list if e.evidence_type == EvidenceType.SYSTEM_COMPROMISE]
        if compromises:
            vectors.append("system_compromise")
        
        # Check for data breaches
        breaches = [e for e in evidence_list if e.evidence_type == EvidenceType.DATA_BREACH]
        if breaches:
            vectors.append("data_breach")
        
        # Check for policy violations
        violations = [e for e in evidence_list if 'violation' in e.raw_data]
        if violations:
            vectors.append("policy_violation")
        
        return vectors
    
    def _assess_overall_impact(self, evidence_list: List[ForensicEvidence]) -> Dict[str, Any]:
        """Assess overall impact of incident"""
        impact = {
            'financial': 'low',
            'operational': 'low',
            'reputational': 'low',
            'regulatory': 'low'
        }
        
        # Check for financial impact
        financial_evidence = [e for e in evidence_list if e.raw_data.get('financial_impact')]
        if financial_evidence:
            impact['financial'] = 'high'
        
        # Check for regulatory impact
        regulatory_evidence = [e for e in evidence_list if e.raw_data.get('regulatory_impact')]
        if regulatory_evidence:
            impact['regulatory'] = 'high'
        
        # Check for system impact
        system_evidence = [e for e in evidence_list if e.evidence_type == EvidenceType.SYSTEM_STATE]
        if len(system_evidence) > 5:
            impact['operational'] = 'high'
        
        return impact


# Test function
async def test_forensic_analyzer():
    """Test the Forensic Analyzer"""
    print("🔍 Testing Forensic Analyzer")
    
    # Initialize analyzer
    analyzer = ForensicAnalyzer()
    
    # Test evidence collection
    print("\\n📝 Testing evidence collection...")
    
    # Collect audit log evidence
    audit_evidence = await analyzer.collect_evidence(
        evidence_type=EvidenceType.AUDIT_LOG,
        source_system="STRATEGIC_MARL",
        raw_data={
            "event_type": "decision_made",
            "user_id": "trader_001",
            "action": "LONG",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "financial_impact": {"amount": 50000}
        },
        metadata={"incident_id": "inc_001"}
    )
    print(f"Collected audit evidence: {audit_evidence.evidence_id}")
    
    # Collect user action evidence
    user_evidence = await analyzer.collect_evidence(
        evidence_type=EvidenceType.USER_ACTION,
        source_system="TRADING_PLATFORM",
        raw_data={
            "user_id": "trader_001",
            "action": "login",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ip_address": "192.168.1.100"
        },
        metadata={"incident_id": "inc_001"}
    )
    print(f"Collected user evidence: {user_evidence.evidence_id}")
    
    # Test timeline reconstruction
    print("\\n⏱️ Testing timeline reconstruction...")
    time_range = (
        datetime.now(timezone.utc) - timedelta(hours=1),
        datetime.now(timezone.utc)
    )
    
    timeline = await analyzer.analyze_timeline(
        incident_id="inc_001",
        time_range=time_range,
        evidence_filters={"evidence_type": "audit_log"}
    )
    print(f"Timeline reconstructed: {timeline.timeline_id}")
    print(f"Timeline events: {len(timeline.events)}")
    
    # Test pattern detection
    print("\\n🔍 Testing pattern detection...")
    patterns = await analyzer.detect_patterns(
        analysis_window=timedelta(hours=1)
    )
    print(f"Detected {len(patterns)} patterns")
    
    # Test anomaly detection
    print("\\n⚠️ Testing anomaly detection...")
    anomalies = await analyzer.detect_anomalies(
        baseline_window=timedelta(hours=24),
        analysis_window=timedelta(hours=1)
    )
    print(f"Detected {len(anomalies)} anomalies")
    
    # Test event correlation
    print("\\n🔗 Testing event correlation...")
    correlation_results = await analyzer.correlate_events(
        event_ids=[audit_evidence.evidence_id, user_evidence.evidence_id],
        correlation_threshold=0.5
    )
    print(f"Correlation analysis: {correlation_results}")
    
    # Test forensic report generation
    print("\\n📋 Testing forensic report generation...")
    report = await analyzer.generate_forensic_report(
        incident_id="inc_001",
        include_timeline=True,
        include_patterns=True,
        include_evidence=True
    )
    print(f"Generated forensic report: {report['report_id']}")
    print(f"Report summary: {report['summary']}")
    
    # Test metrics
    print("\\n📊 Analysis metrics:")
    metrics = analyzer.get_analysis_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print("\\n✅ Forensic Analyzer test complete!")


if __name__ == "__main__":
    asyncio.run(test_forensic_analyzer())