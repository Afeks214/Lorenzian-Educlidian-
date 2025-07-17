"""
AI Ethics Engine for Advanced XAI
Agent Epsilon: Advanced XAI Implementation Specialist

Industry-first ethics engine for trading AI systems.
Monitors for ethics violations and ensures responsible AI deployment.

Features:
- Comprehensive ethics rules framework
- Real-time ethics monitoring
- Violation detection and reporting
- Automated ethics alerts
- Compliance with AI ethics guidelines
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import uuid
from collections import defaultdict

from .bias_detector import BiasDetector, BiasResult, BiasLevel

logger = logging.getLogger(__name__)


class EthicsViolationType(Enum):
    """Types of ethics violations"""
    BIAS_DISCRIMINATION = "bias_discrimination"
    UNFAIR_TREATMENT = "unfair_treatment"
    LACK_OF_TRANSPARENCY = "lack_of_transparency"
    PRIVACY_VIOLATION = "privacy_violation"
    MANIPULATION = "manipulation"
    HARMFUL_OUTCOMES = "harmful_outcomes"
    CONSENT_VIOLATION = "consent_violation"
    ALGORITHMIC_ACCOUNTABILITY = "algorithmic_accountability"
    HUMAN_OVERSIGHT = "human_oversight"
    SAFETY_VIOLATION = "safety_violation"


class EthicsLevel(Enum):
    """Ethics violation severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EthicsRule:
    """Ethics rule definition"""
    rule_id: str
    rule_name: str
    rule_description: str
    violation_type: EthicsViolationType
    severity_level: EthicsLevel
    rule_function: Callable
    threshold: float
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EthicsViolation:
    """Ethics violation instance"""
    violation_id: str
    timestamp: datetime
    violation_type: EthicsViolationType
    severity_level: EthicsLevel
    rule_id: str
    rule_name: str
    
    # Violation details
    violation_score: float
    threshold: float
    description: str
    affected_entities: List[str]
    
    # Context
    decision_id: Optional[str] = None
    user_id: Optional[str] = None
    system_component: str = "unknown"
    
    # Evidence
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    # Remediation
    remediation_actions: List[str] = field(default_factory=list)
    remediation_required: bool = True
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EthicsAssessment:
    """Comprehensive ethics assessment"""
    assessment_id: str
    timestamp: datetime
    overall_ethics_score: float
    
    # Violations
    violations: List[EthicsViolation]
    violation_summary: Dict[str, int]
    
    # Metrics
    fairness_score: float
    transparency_score: float
    accountability_score: float
    safety_score: float
    
    # Recommendations
    improvement_recommendations: List[str]
    compliance_status: str
    
    # Context
    assessment_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EthicsEngine:
    """
    AI Ethics Engine for Trading Systems
    
    Monitors and enforces ethical AI principles
    """
    
    def __init__(self, bias_detector: BiasDetector, config: Optional[Dict[str, Any]] = None):
        self.bias_detector = bias_detector
        self.config = config or self._get_default_config()
        
        # Ethics rules
        self.ethics_rules: Dict[str, EthicsRule] = {}
        self._initialize_ethics_rules()
        
        # Violation tracking
        self.violation_history: List[EthicsViolation] = []
        self.assessment_history: List[EthicsAssessment] = []
        
        # Performance tracking
        self.performance_stats = {
            'total_assessments': 0,
            'total_violations': 0,
            'violations_by_type': defaultdict(int),
            'violations_by_severity': defaultdict(int),
            'avg_assessment_time_ms': 0.0,
            'avg_ethics_score': 0.0
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            EthicsLevel.LOW: 0.7,
            EthicsLevel.MEDIUM: 0.5,
            EthicsLevel.HIGH: 0.3,
            EthicsLevel.CRITICAL: 0.1
        }
        
        logger.info("EthicsEngine initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'ethics_score_threshold': 0.7,
            'real_time_monitoring': True,
            'automatic_remediation': True,
            'alert_on_violations': True,
            'log_all_assessments': True,
            'compliance_framework': 'EU_AI_ACT',
            'transparency_requirements': True,
            'human_oversight_required': True
        }
    
    def _initialize_ethics_rules(self):
        """Initialize ethics rules"""
        
        # Bias and discrimination rules
        self.add_ethics_rule(EthicsRule(
            rule_id="bias_demographic_parity",
            rule_name="Demographic Parity",
            rule_description="Ensures equal treatment across demographic groups",
            violation_type=EthicsViolationType.BIAS_DISCRIMINATION,
            severity_level=EthicsLevel.HIGH,
            rule_function=self._check_demographic_parity,
            threshold=0.05
        ))
        
        self.add_ethics_rule(EthicsRule(
            rule_id="bias_disparate_impact",
            rule_name="Disparate Impact",
            rule_description="Prevents disparate impact on protected groups",
            violation_type=EthicsViolationType.BIAS_DISCRIMINATION,
            severity_level=EthicsLevel.CRITICAL,
            rule_function=self._check_disparate_impact,
            threshold=0.8
        ))
        
        # Transparency rules
        self.add_ethics_rule(EthicsRule(
            rule_id="transparency_explainability",
            rule_name="Explainability Requirement",
            rule_description="Ensures decisions can be explained",
            violation_type=EthicsViolationType.LACK_OF_TRANSPARENCY,
            severity_level=EthicsLevel.HIGH,
            rule_function=self._check_explainability,
            threshold=0.6
        ))
        
        # Safety rules
        self.add_ethics_rule(EthicsRule(
            rule_id="safety_risk_limits",
            rule_name="Risk Limits",
            rule_description="Ensures decisions don't exceed risk limits",
            violation_type=EthicsViolationType.SAFETY_VIOLATION,
            severity_level=EthicsLevel.CRITICAL,
            rule_function=self._check_risk_limits,
            threshold=0.1
        ))
        
        # Accountability rules
        self.add_ethics_rule(EthicsRule(
            rule_id="accountability_audit_trail",
            rule_name="Audit Trail",
            rule_description="Ensures comprehensive audit trail exists",
            violation_type=EthicsViolationType.ALGORITHMIC_ACCOUNTABILITY,
            severity_level=EthicsLevel.MEDIUM,
            rule_function=self._check_audit_trail,
            threshold=0.9
        ))
        
        # Human oversight rules
        self.add_ethics_rule(EthicsRule(
            rule_id="human_oversight_high_risk",
            rule_name="Human Oversight for High Risk",
            rule_description="Requires human oversight for high-risk decisions",
            violation_type=EthicsViolationType.HUMAN_OVERSIGHT,
            severity_level=EthicsLevel.HIGH,
            rule_function=self._check_human_oversight,
            threshold=0.8
        ))
    
    def add_ethics_rule(self, rule: EthicsRule):
        """Add ethics rule to the engine"""
        self.ethics_rules[rule.rule_id] = rule
        logger.info(f"Added ethics rule: {rule.rule_name}")
    
    def remove_ethics_rule(self, rule_id: str):
        """Remove ethics rule"""
        if rule_id in self.ethics_rules:
            del self.ethics_rules[rule_id]
            logger.info(f"Removed ethics rule: {rule_id}")
    
    def enable_rule(self, rule_id: str):
        """Enable ethics rule"""
        if rule_id in self.ethics_rules:
            self.ethics_rules[rule_id].enabled = True
    
    def disable_rule(self, rule_id: str):
        """Disable ethics rule"""
        if rule_id in self.ethics_rules:
            self.ethics_rules[rule_id].enabled = False
    
    async def assess_ethics(
        self,
        decision_context: Dict[str, Any],
        bias_results: Optional[List[BiasResult]] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> EthicsAssessment:
        """
        Perform comprehensive ethics assessment
        
        Args:
            decision_context: Decision context data
            bias_results: Bias detection results
            user_context: User context information
            
        Returns:
            EthicsAssessment: Comprehensive ethics assessment
        """
        start_time = datetime.now()
        
        # Collect violations
        violations = []
        
        # Check each ethics rule
        for rule_id, rule in self.ethics_rules.items():
            if not rule.enabled:
                continue
            
            try:
                violation = await self._check_ethics_rule(
                    rule, decision_context, bias_results, user_context
                )
                
                if violation:
                    violations.append(violation)
                    
            except Exception as e:
                logger.error(f"Error checking ethics rule {rule_id}: {e}")
        
        # Calculate overall ethics score
        overall_score = self._calculate_ethics_score(violations)
        
        # Calculate component scores
        fairness_score = self._calculate_fairness_score(violations, bias_results)
        transparency_score = self._calculate_transparency_score(violations, decision_context)
        accountability_score = self._calculate_accountability_score(violations, decision_context)
        safety_score = self._calculate_safety_score(violations, decision_context)
        
        # Generate improvement recommendations
        recommendations = self._generate_improvement_recommendations(violations)
        
        # Determine compliance status
        compliance_status = self._determine_compliance_status(overall_score, violations)
        
        # Create assessment
        assessment = EthicsAssessment(
            assessment_id=f"ethics_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(timezone.utc),
            overall_ethics_score=overall_score,
            violations=violations,
            violation_summary=self._summarize_violations(violations),
            fairness_score=fairness_score,
            transparency_score=transparency_score,
            accountability_score=accountability_score,
            safety_score=safety_score,
            improvement_recommendations=recommendations,
            compliance_status=compliance_status,
            assessment_context=decision_context,
            metadata={
                'assessment_time_ms': (datetime.now() - start_time).total_seconds() * 1000,
                'rules_checked': len([r for r in self.ethics_rules.values() if r.enabled]),
                'bias_results_included': bias_results is not None
            }
        )
        
        # Update performance stats
        self._update_performance_stats(assessment)
        
        # Store assessment
        self.assessment_history.append(assessment)
        self.violation_history.extend(violations)
        
        # Send alerts if necessary
        if self.config['alert_on_violations'] and violations:
            await self._send_ethics_alerts(assessment)
        
        return assessment
    
    async def _check_ethics_rule(
        self,
        rule: EthicsRule,
        decision_context: Dict[str, Any],
        bias_results: Optional[List[BiasResult]],
        user_context: Optional[Dict[str, Any]]
    ) -> Optional[EthicsViolation]:
        """Check individual ethics rule"""
        
        # Prepare rule context
        rule_context = {
            'decision_context': decision_context,
            'bias_results': bias_results or [],
            'user_context': user_context or {},
            'rule': rule
        }
        
        # Execute rule function
        try:
            violation_score, evidence = await rule.rule_function(rule_context)
            
            # Check if violation occurred
            if violation_score > rule.threshold:
                violation = EthicsViolation(
                    violation_id=f"violation_{uuid.uuid4().hex[:8]}",
                    timestamp=datetime.now(timezone.utc),
                    violation_type=rule.violation_type,
                    severity_level=rule.severity_level,
                    rule_id=rule.rule_id,
                    rule_name=rule.rule_name,
                    violation_score=violation_score,
                    threshold=rule.threshold,
                    description=self._generate_violation_description(rule, violation_score),
                    affected_entities=evidence.get('affected_entities', []),
                    decision_id=decision_context.get('decision_id'),
                    user_id=user_context.get('user_id') if user_context else None,
                    system_component=decision_context.get('system_component', 'unknown'),
                    evidence=evidence,
                    remediation_actions=self._generate_remediation_actions(rule, violation_score),
                    remediation_required=rule.severity_level in [EthicsLevel.HIGH, EthicsLevel.CRITICAL]
                )
                
                return violation
                
        except Exception as e:
            logger.error(f"Error executing ethics rule {rule.rule_id}: {e}")
        
        return None
    
    async def _check_demographic_parity(self, context: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Check demographic parity rule"""
        bias_results = context.get('bias_results', [])
        
        max_violation = 0.0
        evidence = {'affected_entities': []}
        
        for result in bias_results:
            if result.bias_type.value == 'demographic':
                if result.bias_score > max_violation:
                    max_violation = result.bias_score
                    evidence = {
                        'affected_entities': result.affected_groups,
                        'bias_score': result.bias_score,
                        'statistical_significance': result.statistical_significance
                    }
        
        return max_violation, evidence
    
    async def _check_disparate_impact(self, context: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Check disparate impact rule"""
        bias_results = context.get('bias_results', [])
        
        max_violation = 0.0
        evidence = {'affected_entities': []}
        
        for result in bias_results:
            if 'disparate_impact' in result.detection_method:
                # Convert disparate impact ratio to violation score
                violation_score = 1.0 - min(result.bias_score, 1.0 / result.bias_score)
                
                if violation_score > max_violation:
                    max_violation = violation_score
                    evidence = {
                        'affected_entities': result.affected_groups,
                        'disparate_impact_ratio': result.bias_score,
                        'statistical_significance': result.statistical_significance
                    }
        
        return max_violation, evidence
    
    async def _check_explainability(self, context: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Check explainability requirement"""
        decision_context = context.get('decision_context', {})
        
        # Check if explanation exists
        has_explanation = 'explanation' in decision_context or 'explanation_id' in decision_context
        explanation_quality = decision_context.get('explanation_quality', 0.0)
        
        if not has_explanation:
            violation_score = 1.0
        else:
            # Violation score based on explanation quality
            violation_score = 1.0 - explanation_quality
        
        evidence = {
            'has_explanation': has_explanation,
            'explanation_quality': explanation_quality,
            'affected_entities': [decision_context.get('decision_id', 'unknown')]
        }
        
        return violation_score, evidence
    
    async def _check_risk_limits(self, context: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Check risk limits"""
        decision_context = context.get('decision_context', {})
        
        # Check various risk metrics
        risk_metrics = decision_context.get('risk_metrics', {})
        
        violation_score = 0.0
        evidence = {'affected_entities': [], 'risk_violations': []}
        
        # Check drawdown risk
        drawdown = risk_metrics.get('drawdown', 0.0)
        if drawdown > 0.1:  # 10% drawdown limit
            violation_score = max(violation_score, drawdown)
            evidence['risk_violations'].append(f"Drawdown: {drawdown:.2%}")
        
        # Check volatility risk
        volatility = risk_metrics.get('volatility', 0.0)
        if volatility > 0.3:  # 30% volatility limit
            violation_score = max(violation_score, volatility / 0.3)
            evidence['risk_violations'].append(f"Volatility: {volatility:.2%}")
        
        # Check concentration risk
        concentration = risk_metrics.get('concentration', 0.0)
        if concentration > 0.5:  # 50% concentration limit
            violation_score = max(violation_score, concentration / 0.5)
            evidence['risk_violations'].append(f"Concentration: {concentration:.2%}")
        
        return violation_score, evidence
    
    async def _check_audit_trail(self, context: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Check audit trail completeness"""
        decision_context = context.get('decision_context', {})
        
        # Check audit trail completeness
        required_fields = ['decision_id', 'timestamp', 'user_id', 'system_component']
        present_fields = [field for field in required_fields if field in decision_context]
        
        completeness = len(present_fields) / len(required_fields)
        violation_score = 1.0 - completeness
        
        evidence = {
            'completeness': completeness,
            'missing_fields': [field for field in required_fields if field not in decision_context],
            'affected_entities': [decision_context.get('decision_id', 'unknown')]
        }
        
        return violation_score, evidence
    
    async def _check_human_oversight(self, context: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Check human oversight requirement"""
        decision_context = context.get('decision_context', {})
        
        # Check if human oversight is required and present
        risk_level = decision_context.get('risk_level', 'medium')
        confidence = decision_context.get('confidence', 0.5)
        human_reviewed = decision_context.get('human_reviewed', False)
        
        violation_score = 0.0
        evidence = {'affected_entities': []}
        
        # High risk decisions require human oversight
        if risk_level == 'high' and not human_reviewed:
            violation_score = 1.0
            evidence['reason'] = 'High risk decision without human review'
        
        # Low confidence decisions require human oversight
        elif confidence < 0.5 and not human_reviewed:
            violation_score = 1.0 - confidence
            evidence['reason'] = 'Low confidence decision without human review'
        
        evidence['affected_entities'] = [decision_context.get('decision_id', 'unknown')]
        
        return violation_score, evidence
    
    def _calculate_ethics_score(self, violations: List[EthicsViolation]) -> float:
        """Calculate overall ethics score"""
        if not violations:
            return 1.0
        
        # Weight violations by severity
        severity_weights = {
            EthicsLevel.LOW: 0.1,
            EthicsLevel.MEDIUM: 0.3,
            EthicsLevel.HIGH: 0.7,
            EthicsLevel.CRITICAL: 1.0
        }
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for violation in violations:
            weight = severity_weights[violation.severity_level]
            total_weighted_score += violation.violation_score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 1.0
        
        avg_weighted_violation = total_weighted_score / total_weight
        return max(0.0, 1.0 - avg_weighted_violation)
    
    def _calculate_fairness_score(
        self, 
        violations: List[EthicsViolation], 
        bias_results: Optional[List[BiasResult]]
    ) -> float:
        """Calculate fairness score"""
        
        # Start with bias results
        fairness_score = 1.0
        
        if bias_results:
            bias_scores = [result.bias_score for result in bias_results]
            if bias_scores:
                avg_bias = np.mean(bias_scores)
                fairness_score = max(0.0, 1.0 - avg_bias)
        
        # Adjust for fairness-related violations
        fairness_violations = [
            v for v in violations 
            if v.violation_type in [EthicsViolationType.BIAS_DISCRIMINATION, EthicsViolationType.UNFAIR_TREATMENT]
        ]
        
        if fairness_violations:
            violation_penalty = np.mean([v.violation_score for v in fairness_violations])
            fairness_score = max(0.0, fairness_score - violation_penalty)
        
        return fairness_score
    
    def _calculate_transparency_score(
        self, 
        violations: List[EthicsViolation], 
        decision_context: Dict[str, Any]
    ) -> float:
        """Calculate transparency score"""
        
        # Base score from explanation quality
        explanation_quality = decision_context.get('explanation_quality', 0.5)
        transparency_score = explanation_quality
        
        # Adjust for transparency violations
        transparency_violations = [
            v for v in violations 
            if v.violation_type == EthicsViolationType.LACK_OF_TRANSPARENCY
        ]
        
        if transparency_violations:
            violation_penalty = np.mean([v.violation_score for v in transparency_violations])
            transparency_score = max(0.0, transparency_score - violation_penalty)
        
        return transparency_score
    
    def _calculate_accountability_score(
        self, 
        violations: List[EthicsViolation], 
        decision_context: Dict[str, Any]
    ) -> float:
        """Calculate accountability score"""
        
        # Base score from audit trail completeness
        audit_completeness = decision_context.get('audit_completeness', 0.5)
        accountability_score = audit_completeness
        
        # Adjust for accountability violations
        accountability_violations = [
            v for v in violations 
            if v.violation_type == EthicsViolationType.ALGORITHMIC_ACCOUNTABILITY
        ]
        
        if accountability_violations:
            violation_penalty = np.mean([v.violation_score for v in accountability_violations])
            accountability_score = max(0.0, accountability_score - violation_penalty)
        
        return accountability_score
    
    def _calculate_safety_score(
        self, 
        violations: List[EthicsViolation], 
        decision_context: Dict[str, Any]
    ) -> float:
        """Calculate safety score"""
        
        # Base score from risk metrics
        risk_metrics = decision_context.get('risk_metrics', {})
        safety_score = 1.0 - risk_metrics.get('overall_risk', 0.0)
        
        # Adjust for safety violations
        safety_violations = [
            v for v in violations 
            if v.violation_type == EthicsViolationType.SAFETY_VIOLATION
        ]
        
        if safety_violations:
            violation_penalty = np.mean([v.violation_score for v in safety_violations])
            safety_score = max(0.0, safety_score - violation_penalty)
        
        return safety_score
    
    def _generate_violation_description(self, rule: EthicsRule, violation_score: float) -> str:
        """Generate human-readable violation description"""
        severity_desc = {
            EthicsLevel.LOW: "minor",
            EthicsLevel.MEDIUM: "moderate",
            EthicsLevel.HIGH: "significant",
            EthicsLevel.CRITICAL: "critical"
        }
        
        return (
            f"A {severity_desc[rule.severity_level]} violation of {rule.rule_name} "
            f"was detected with a score of {violation_score:.3f} "
            f"(threshold: {rule.threshold:.3f}). {rule.rule_description}"
        )
    
    def _generate_remediation_actions(self, rule: EthicsRule, violation_score: float) -> List[str]:
        """Generate remediation actions for violation"""
        actions = []
        
        if rule.violation_type == EthicsViolationType.BIAS_DISCRIMINATION:
            actions.extend([
                "Implement bias mitigation techniques",
                "Review and adjust model training data",
                "Apply fairness constraints to decision process"
            ])
        
        elif rule.violation_type == EthicsViolationType.LACK_OF_TRANSPARENCY:
            actions.extend([
                "Generate detailed explanations for decisions",
                "Implement interpretability methods",
                "Provide clear reasoning for outcomes"
            ])
        
        elif rule.violation_type == EthicsViolationType.SAFETY_VIOLATION:
            actions.extend([
                "Implement immediate risk controls",
                "Review and adjust risk limits",
                "Conduct safety assessment"
            ])
        
        elif rule.violation_type == EthicsViolationType.ALGORITHMIC_ACCOUNTABILITY:
            actions.extend([
                "Enhance audit trail documentation",
                "Implement comprehensive logging",
                "Establish clear accountability chains"
            ])
        
        # Add severity-specific actions
        if rule.severity_level == EthicsLevel.CRITICAL:
            actions.insert(0, "Immediate system review required")
            actions.append("Consider temporary system suspension")
        
        return actions
    
    def _generate_improvement_recommendations(self, violations: List[EthicsViolation]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Aggregate recommendations by violation type
        violation_types = set(v.violation_type for v in violations)
        
        if EthicsViolationType.BIAS_DISCRIMINATION in violation_types:
            recommendations.append("Implement comprehensive bias testing and mitigation")
        
        if EthicsViolationType.LACK_OF_TRANSPARENCY in violation_types:
            recommendations.append("Enhance explanation generation and transparency")
        
        if EthicsViolationType.SAFETY_VIOLATION in violation_types:
            recommendations.append("Strengthen safety controls and risk management")
        
        if EthicsViolationType.ALGORITHMIC_ACCOUNTABILITY in violation_types:
            recommendations.append("Improve audit trail and accountability mechanisms")
        
        if EthicsViolationType.HUMAN_OVERSIGHT in violation_types:
            recommendations.append("Implement human-in-the-loop controls")
        
        # Add general recommendations
        if violations:
            recommendations.append("Conduct regular ethics assessments")
            recommendations.append("Implement automated ethics monitoring")
            recommendations.append("Train staff on AI ethics principles")
        
        return recommendations
    
    def _determine_compliance_status(self, ethics_score: float, violations: List[EthicsViolation]) -> str:
        """Determine compliance status"""
        critical_violations = [v for v in violations if v.severity_level == EthicsLevel.CRITICAL]
        high_violations = [v for v in violations if v.severity_level == EthicsLevel.HIGH]
        
        if critical_violations:
            return "NON_COMPLIANT_CRITICAL"
        elif high_violations:
            return "NON_COMPLIANT_HIGH"
        elif ethics_score < 0.7:
            return "NON_COMPLIANT_MODERATE"
        elif ethics_score < 0.9:
            return "PARTIALLY_COMPLIANT"
        else:
            return "COMPLIANT"
    
    def _summarize_violations(self, violations: List[EthicsViolation]) -> Dict[str, int]:
        """Summarize violations by type and severity"""
        summary = {
            'total': len(violations),
            'by_type': defaultdict(int),
            'by_severity': defaultdict(int)
        }
        
        for violation in violations:
            summary['by_type'][violation.violation_type.value] += 1
            summary['by_severity'][violation.severity_level.value] += 1
        
        return dict(summary)
    
    async def _send_ethics_alerts(self, assessment: EthicsAssessment):
        """Send ethics alerts for violations"""
        critical_violations = [v for v in assessment.violations if v.severity_level == EthicsLevel.CRITICAL]
        high_violations = [v for v in assessment.violations if v.severity_level == EthicsLevel.HIGH]
        
        if critical_violations:
            logger.critical(f"CRITICAL ETHICS VIOLATIONS: {len(critical_violations)} violations detected")
        
        if high_violations:
            logger.error(f"HIGH ETHICS VIOLATIONS: {len(high_violations)} violations detected")
        
        # In a real system, this would send alerts to monitoring systems
    
    def _update_performance_stats(self, assessment: EthicsAssessment):
        """Update performance statistics"""
        self.performance_stats['total_assessments'] += 1
        self.performance_stats['total_violations'] += len(assessment.violations)
        
        # Update averages
        total_assessments = self.performance_stats['total_assessments']
        old_avg_score = self.performance_stats['avg_ethics_score']
        self.performance_stats['avg_ethics_score'] = (
            (old_avg_score * (total_assessments - 1) + assessment.overall_ethics_score) / total_assessments
        )
        
        old_avg_time = self.performance_stats['avg_assessment_time_ms']
        new_time = assessment.metadata.get('assessment_time_ms', 0)
        self.performance_stats['avg_assessment_time_ms'] = (
            (old_avg_time * (total_assessments - 1) + new_time) / total_assessments
        )
        
        # Update violation counts
        for violation in assessment.violations:
            self.performance_stats['violations_by_type'][violation.violation_type.value] += 1
            self.performance_stats['violations_by_severity'][violation.severity_level.value] += 1
    
    def get_ethics_summary(self) -> Dict[str, Any]:
        """Get ethics summary"""
        return {
            'total_assessments': len(self.assessment_history),
            'total_violations': len(self.violation_history),
            'avg_ethics_score': self.performance_stats['avg_ethics_score'],
            'violation_summary': dict(self.performance_stats['violations_by_type']),
            'severity_summary': dict(self.performance_stats['violations_by_severity']),
            'compliance_status': self._get_current_compliance_status(),
            'performance_stats': self.performance_stats
        }
    
    def _get_current_compliance_status(self) -> str:
        """Get current compliance status"""
        if not self.assessment_history:
            return "NOT_ASSESSED"
        
        latest_assessment = self.assessment_history[-1]
        return latest_assessment.compliance_status
    
    def get_violation_trends(self) -> Dict[str, Any]:
        """Get violation trends analysis"""
        if not self.violation_history:
            return {}
        
        # Group violations by time period
        recent_violations = [
            v for v in self.violation_history
            if (datetime.now(timezone.utc) - v.timestamp).days <= 7
        ]
        
        return {
            'total_violations': len(self.violation_history),
            'recent_violations': len(recent_violations),
            'trend_direction': 'stable',  # Would calculate actual trend
            'most_common_violation': max(
                self.performance_stats['violations_by_type'].items(),
                key=lambda x: x[1]
            )[0] if self.performance_stats['violations_by_type'] else None
        }


# Test function
async def test_ethics_engine():
    """Test the Ethics Engine"""
    print("🧪 Testing Ethics Engine")
    
    # Initialize bias detector and ethics engine
    bias_detector = BiasDetector()
    ethics_engine = EthicsEngine(bias_detector)
    
    # Create mock decision context
    decision_context = {
        'decision_id': 'dec_123',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'user_id': 'trader_001',
        'system_component': 'strategic_marl',
        'action': 'LONG',
        'confidence': 0.4,  # Low confidence
        'explanation_quality': 0.3,  # Low explanation quality
        'risk_level': 'high',  # High risk
        'human_reviewed': False,  # No human review
        'risk_metrics': {
            'drawdown': 0.15,  # High drawdown
            'volatility': 0.35,  # High volatility
            'overall_risk': 0.2
        }
    }
    
    # Create mock bias results
    from .bias_detector import BiasResult, BiasType, BiasMetric, BiasLevel
    bias_results = [
        BiasResult(
            detection_id='bias_001',
            timestamp=datetime.now(timezone.utc),
            bias_type=BiasType.DEMOGRAPHIC,
            bias_metric=BiasMetric.DEMOGRAPHIC_PARITY,
            bias_score=0.15,
            bias_level=BiasLevel.MODERATE,
            statistical_significance=0.01,
            confidence_interval=(0.05, 0.25),
            affected_groups=['Group A', 'Group B'],
            affected_decisions=['dec_123'],
            mitigation_suggestions=['Implement bias mitigation'],
            decision_context=decision_context,
            sample_size=1000,
            test_statistic=2.5,
            detection_method='demographic_parity',
            detector_version='1.0'
        )
    ]
    
    # Test ethics assessment
    print("\\n🔍 Testing ethics assessment...")
    
    assessment = await ethics_engine.assess_ethics(
        decision_context=decision_context,
        bias_results=bias_results,
        user_context={'user_id': 'trader_001', 'role': 'trader'}
    )
    
    print(f"Ethics Assessment Results:")
    print(f"  Overall Ethics Score: {assessment.overall_ethics_score:.3f}")
    print(f"  Fairness Score: {assessment.fairness_score:.3f}")
    print(f"  Transparency Score: {assessment.transparency_score:.3f}")
    print(f"  Accountability Score: {assessment.accountability_score:.3f}")
    print(f"  Safety Score: {assessment.safety_score:.3f}")
    print(f"  Compliance Status: {assessment.compliance_status}")
    
    print(f"\\n🚨 Violations Detected: {len(assessment.violations)}")
    for violation in assessment.violations:
        print(f"  - {violation.violation_type.value} ({violation.severity_level.value})")
        print(f"    Score: {violation.violation_score:.3f}, Threshold: {violation.threshold:.3f}")
        print(f"    Description: {violation.description}")
        print(f"    Remediation: {violation.remediation_actions[:2]}")
    
    print(f"\\n💡 Improvement Recommendations:")
    for i, rec in enumerate(assessment.improvement_recommendations, 1):
        print(f"  {i}. {rec}")
    
    # Test ethics summary
    print("\\n📊 Ethics Summary:")
    summary = ethics_engine.get_ethics_summary()
    print(f"  Total assessments: {summary['total_assessments']}")
    print(f"  Total violations: {summary['total_violations']}")
    print(f"  Average ethics score: {summary['avg_ethics_score']:.3f}")
    print(f"  Current compliance: {summary['compliance_status']}")
    
    print("\\n✅ Ethics Engine test complete!")


if __name__ == "__main__":
    asyncio.run(test_ethics_engine())