"""
Causal Narrative Generator for Advanced XAI
Agent Epsilon: Advanced XAI Implementation Specialist

Industry-first causal narrative generation system that transforms complex causal graphs
and counterfactual analyses into compelling human-readable explanations.

Features:
- Causal story generation from graph structures
- Multi-audience narrative adaptation
- Counterfactual scenario storytelling
- Causal strength quantification
- Interactive explanation generation
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import uuid
import json
from collections import defaultdict

from .do_calculus_engine import CausalGraph, CausalNode, CausalResult, NodeType
from .counterfactual_engine import CounterfactualResult, CounterfactualScenario, CounterfactualIntervention

logger = logging.getLogger(__name__)


class NarrativeStyle(Enum):
    """Narrative styles for different audiences"""
    TECHNICAL = "technical"
    EXECUTIVE = "executive"
    REGULATORY = "regulatory"
    TRADER = "trader"
    CLIENT = "client"
    ACADEMIC = "academic"


class CausalRelationship(Enum):
    """Types of causal relationships"""
    DIRECT_CAUSE = "direct_cause"
    INDIRECT_CAUSE = "indirect_cause"
    MEDIATOR = "mediator"
    CONFOUNDER = "confounder"
    COLLIDER = "collider"
    MODERATOR = "moderator"


@dataclass
class CausalFactor:
    """Individual causal factor in explanation"""
    variable: str
    relationship: CausalRelationship
    strength: float
    confidence: float
    direction: str  # positive, negative, neutral
    importance_rank: int
    explanation: str
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalStory:
    """Complete causal story for explanation"""
    story_id: str
    title: str
    summary: str
    main_narrative: str
    causal_factors: List[CausalFactor]
    counterfactual_scenarios: List[str]
    key_insights: List[str]
    confidence_assessment: str
    limitations: List[str]
    recommendations: List[str]
    audience: NarrativeStyle
    complexity_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalExplanation:
    """Multi-layered causal explanation"""
    explanation_id: str
    decision_context: Dict[str, Any]
    primary_story: CausalStory
    alternative_stories: List[CausalStory]
    comparative_analysis: str
    robustness_assessment: str
    actionable_insights: List[str]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


class CausalNarrativeGenerator:
    """
    Advanced Causal Narrative Generator
    
    Transforms causal analyses into compelling explanations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Narrative templates for different styles
        self.narrative_templates = self._initialize_narrative_templates()
        
        # Causal relationship descriptors
        self.relationship_descriptors = self._initialize_relationship_descriptors()
        
        # Financial domain vocabulary
        self.financial_vocabulary = self._initialize_financial_vocabulary()
        
        # Performance tracking
        self.performance_stats = {
            "total_narratives": 0,
            "avg_generation_time_ms": 0.0,
            "avg_narrative_length": 0.0,
            "avg_complexity_score": 0.0,
            "audience_distribution": defaultdict(int)
        }
        
        logger.info("CausalNarrativeGenerator initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "max_narrative_length": 500,
            "min_narrative_length": 100,
            "max_causal_factors": 5,
            "complexity_threshold": 0.7,
            "confidence_threshold": 0.6,
            "include_counterfactuals": True,
            "include_recommendations": True,
            "personalization_enabled": True
        }
    
    def _initialize_narrative_templates(self) -> Dict[NarrativeStyle, Dict[str, str]]:
        """Initialize narrative templates for different audiences"""
        return {
            NarrativeStyle.TECHNICAL: {
                "opening": "Causal analysis reveals that {outcome} is primarily driven by {primary_cause} with a causal effect of {effect_size:.3f}.",
                "causal_chain": "The causal pathway follows: {causal_chain}.",
                "strength": "The causal relationship has strength {strength:.2f} with confidence {confidence:.2f}.",
                "counterfactual": "Under counterfactual conditions where {intervention}, the outcome would be {counterfactual_outcome}.",
                "conclusion": "The analysis identifies {key_factors} as the primary causal mechanisms."
            },
            
            NarrativeStyle.EXECUTIVE: {
                "opening": "The trading decision resulted in {outcome} due to {primary_cause}.",
                "causal_chain": "This outcome was influenced by a chain of factors: {causal_chain}.",
                "strength": "The primary driver accounts for {strength_percentage:.0f}% of the outcome.",
                "counterfactual": "Had we {intervention}, the result would likely have been {counterfactual_outcome}.",
                "conclusion": "Key takeaways: {key_insights}."
            },
            
            NarrativeStyle.REGULATORY: {
                "opening": "Algorithmic trading decision analysis shows {outcome} caused by {primary_cause}.",
                "causal_chain": "The decision process followed this sequence: {causal_chain}.",
                "strength": "Causal evidence strength: {strength_level} (confidence: {confidence:.2f}).",
                "counterfactual": "Alternative scenario analysis: {counterfactual_analysis}.",
                "conclusion": "Risk factors and controls: {risk_assessment}."
            },
            
            NarrativeStyle.TRADER: {
                "opening": "Your {outcome} was driven by {primary_cause}.",
                "causal_chain": "Here's what happened: {causal_chain}.",
                "strength": "This factor had {strength_description} impact on the outcome.",
                "counterfactual": "If {intervention}, you would have seen {counterfactual_outcome}.",
                "conclusion": "Next time, watch for: {actionable_insights}."
            },
            
            NarrativeStyle.CLIENT: {
                "opening": "The investment decision led to {outcome} because of {primary_cause}.",
                "causal_chain": "The market conditions created this situation: {causal_chain}.",
                "strength": "This was the {strength_description} factor affecting your investment.",
                "counterfactual": "In different market conditions, the outcome would have been {counterfactual_outcome}.",
                "conclusion": "This means: {client_insights}."
            },
            
            NarrativeStyle.ACADEMIC: {
                "opening": "Causal identification reveals {outcome} as a function of {primary_cause} (β = {effect_size:.3f}).",
                "causal_chain": "The directed acyclic graph shows: {causal_chain}.",
                "strength": "Effect size: {effect_size:.3f} (95% CI: {ci_lower:.3f}, {ci_upper:.3f}).",
                "counterfactual": "Counterfactual analysis: E[Y|do(X=x')] = {counterfactual_outcome}.",
                "conclusion": "Findings: {academic_findings}."
            }
        }
    
    def _initialize_relationship_descriptors(self) -> Dict[CausalRelationship, Dict[str, str]]:
        """Initialize causal relationship descriptors"""
        return {
            CausalRelationship.DIRECT_CAUSE: {
                "description": "directly causes",
                "strength_adjectives": ["weak", "moderate", "strong", "very strong"],
                "explanation": "has a direct causal effect on"
            },
            
            CausalRelationship.INDIRECT_CAUSE: {
                "description": "indirectly influences",
                "strength_adjectives": ["slight", "moderate", "significant", "major"],
                "explanation": "affects through intermediate variables"
            },
            
            CausalRelationship.MEDIATOR: {
                "description": "mediates the effect of",
                "strength_adjectives": ["partial", "substantial", "complete", "strong"],
                "explanation": "explains how the effect occurs"
            },
            
            CausalRelationship.CONFOUNDER: {
                "description": "confounds the relationship between",
                "strength_adjectives": ["weak", "moderate", "strong", "severe"],
                "explanation": "creates spurious association"
            },
            
            CausalRelationship.COLLIDER: {
                "description": "is a collider for",
                "strength_adjectives": ["mild", "moderate", "strong", "severe"],
                "explanation": "creates selection bias"
            },
            
            CausalRelationship.MODERATOR: {
                "description": "moderates the effect of",
                "strength_adjectives": ["weak", "moderate", "strong", "very strong"],
                "explanation": "changes the strength of the relationship"
            }
        }
    
    def _initialize_financial_vocabulary(self) -> Dict[str, Dict[str, str]]:
        """Initialize financial domain vocabulary"""
        return {
            "variables": {
                "decision": "trading decision",
                "confidence": "decision confidence",
                "market_volatility": "market volatility",
                "volume_ratio": "trading volume",
                "momentum": "price momentum",
                "drawdown": "portfolio drawdown",
                "execution_success": "execution quality",
                "slippage": "execution slippage"
            },
            
            "outcomes": {
                "positive": ["favorable", "profitable", "successful", "beneficial"],
                "negative": ["unfavorable", "costly", "problematic", "detrimental"],
                "neutral": ["stable", "unchanged", "neutral", "balanced"]
            },
            
            "actions": {
                "LONG": "buy position",
                "SHORT": "sell position", 
                "HOLD": "hold position"
            },
            
            "market_conditions": {
                "high_volatility": "volatile market conditions",
                "low_volatility": "stable market conditions",
                "high_volume": "active trading conditions",
                "low_volume": "quiet trading conditions"
            }
        }
    
    async def generate_causal_narrative(
        self,
        causal_result: CausalResult,
        graph: CausalGraph,
        decision_context: Dict[str, Any],
        audience: NarrativeStyle = NarrativeStyle.TECHNICAL
    ) -> CausalStory:
        """
        Generate causal narrative from causal analysis result
        
        Args:
            causal_result: Result from causal analysis
            graph: Causal graph
            decision_context: Decision context
            audience: Target audience
            
        Returns:
            CausalStory: Generated narrative
        """
        start_time = datetime.now()
        
        # Extract causal factors
        causal_factors = self._extract_causal_factors(causal_result, graph)
        
        # Generate narrative components
        title = self._generate_title(causal_result, decision_context, audience)
        summary = self._generate_summary(causal_result, causal_factors, audience)
        main_narrative = self._generate_main_narrative(causal_result, causal_factors, graph, audience)
        
        # Generate insights and recommendations
        key_insights = self._generate_key_insights(causal_factors, causal_result)
        recommendations = self._generate_recommendations(causal_factors, decision_context, audience)
        
        # Assess confidence and limitations
        confidence_assessment = self._assess_confidence(causal_result, causal_factors)
        limitations = self._identify_limitations(causal_result, graph)
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(causal_factors, graph)
        
        # Create story
        story = CausalStory(
            story_id=f"story_{uuid.uuid4().hex[:8]}",
            title=title,
            summary=summary,
            main_narrative=main_narrative,
            causal_factors=causal_factors,
            counterfactual_scenarios=[],  # Will be populated separately
            key_insights=key_insights,
            confidence_assessment=confidence_assessment,
            limitations=limitations,
            recommendations=recommendations,
            audience=audience,
            complexity_score=complexity_score,
            metadata={
                "generation_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "narrative_length": len(main_narrative),
                "factors_count": len(causal_factors),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
        # Update performance stats
        self._update_performance_stats(story)
        
        return story
    
    def _extract_causal_factors(self, causal_result: CausalResult, graph: CausalGraph) -> List[CausalFactor]:
        """Extract causal factors from analysis result"""
        factors = []
        
        # Primary causal factor (treatment)
        primary_factor = CausalFactor(
            variable=causal_result.query_id.split("_")[-1] if "_" in causal_result.query_id else "primary_cause",
            relationship=CausalRelationship.DIRECT_CAUSE,
            strength=abs(causal_result.effect_size),
            confidence=1.0 - causal_result.p_value,
            direction="positive" if causal_result.effect_size > 0 else "negative",
            importance_rank=1,
            explanation=f"Primary causal factor with effect size {causal_result.effect_size:.3f}"
        )
        factors.append(primary_factor)
        
        # Confounders
        for i, confounder in enumerate(causal_result.confounders):
            if confounder in graph.nodes:
                factor = CausalFactor(
                    variable=confounder,
                    relationship=CausalRelationship.CONFOUNDER,
                    strength=0.5,  # Default strength
                    confidence=0.7,
                    direction="neutral",
                    importance_rank=i + 2,
                    explanation=f"Confounding variable that affects both treatment and outcome"
                )
                factors.append(factor)
        
        # Mediators
        for i, mediator in enumerate(causal_result.mediators):
            if mediator in graph.nodes:
                factor = CausalFactor(
                    variable=mediator,
                    relationship=CausalRelationship.MEDIATOR,
                    strength=0.6,
                    confidence=0.8,
                    direction="positive",
                    importance_rank=len(factors) + 1,
                    explanation=f"Mediating variable that explains the causal mechanism"
                )
                factors.append(factor)
        
        # Limit number of factors
        return factors[:self.config["max_causal_factors"]]
    
    def _generate_title(
        self, 
        causal_result: CausalResult, 
        decision_context: Dict[str, Any], 
        audience: NarrativeStyle
    ) -> str:
        """Generate narrative title"""
        outcome = decision_context.get("outcome", "outcome")
        primary_cause = decision_context.get("primary_cause", "market conditions")
        
        title_templates = {
            NarrativeStyle.TECHNICAL: f"Causal Analysis: {outcome} driven by {primary_cause}",
            NarrativeStyle.EXECUTIVE: f"Trading Decision Analysis: {outcome} Results",
            NarrativeStyle.REGULATORY: f"Algorithmic Decision Audit: {outcome} Causality",
            NarrativeStyle.TRADER: f"Why Your Trade {outcome}",
            NarrativeStyle.CLIENT: f"Investment Performance: {outcome} Explained",
            NarrativeStyle.ACADEMIC: f"Causal Identification of {outcome} Determinants"
        }
        
        return title_templates.get(audience, f"Causal Analysis: {outcome}")
    
    def _generate_summary(
        self, 
        causal_result: CausalResult, 
        causal_factors: List[CausalFactor],
        audience: NarrativeStyle
    ) -> str:
        """Generate narrative summary"""
        primary_factor = causal_factors[0] if causal_factors else None
        
        if not primary_factor:
            return "No clear causal factors identified in the analysis."
        
        effect_magnitude = "strong" if primary_factor.strength > 0.7 else "moderate" if primary_factor.strength > 0.4 else "weak"
        
        summary_templates = {
            NarrativeStyle.TECHNICAL: (
                f"Analysis identifies {primary_factor.variable} as the primary causal factor "
                f"with {effect_magnitude} effect size ({primary_factor.strength:.3f}) "
                f"and {primary_factor.confidence:.2f} confidence."
            ),
            NarrativeStyle.EXECUTIVE: (
                f"The outcome was primarily driven by {primary_factor.variable}, "
                f"which had a {effect_magnitude} impact on the result."
            ),
            NarrativeStyle.REGULATORY: (
                f"Causal analysis shows {primary_factor.variable} as the primary determinant "
                f"with {effect_magnitude} causal strength and adequate statistical evidence."
            ),
            NarrativeStyle.TRADER: (
                f"Your result was mainly due to {primary_factor.variable}, "
                f"which had a {effect_magnitude} impact on the outcome."
            ),
            NarrativeStyle.CLIENT: (
                f"The investment outcome was primarily influenced by {primary_factor.variable}, "
                f"which created a {effect_magnitude} effect on performance."
            ),
            NarrativeStyle.ACADEMIC: (
                f"Causal identification establishes {primary_factor.variable} as the primary driver "
                f"(β = {primary_factor.strength:.3f}, p < {1-primary_factor.confidence:.3f})."
            )
        }
        
        return summary_templates.get(audience, summary_templates[NarrativeStyle.TECHNICAL])
    
    def _generate_main_narrative(
        self,
        causal_result: CausalResult,
        causal_factors: List[CausalFactor],
        graph: CausalGraph,
        audience: NarrativeStyle
    ) -> str:
        """Generate main narrative text"""
        if not causal_factors:
            return "No significant causal relationships were identified in the analysis."
        
        templates = self.narrative_templates[audience]
        
        # Extract narrative components
        primary_factor = causal_factors[0]
        primary_cause = self._translate_variable_name(primary_factor.variable)
        outcome = "the observed outcome"
        effect_size = primary_factor.strength
        
        # Generate causal chain
        causal_chain = self._generate_causal_chain(causal_factors, graph, audience)
        
        # Build narrative
        narrative_parts = []
        
        # Opening
        opening = templates["opening"].format(
            outcome=outcome,
            primary_cause=primary_cause,
            effect_size=effect_size
        )
        narrative_parts.append(opening)
        
        # Causal chain
        if causal_chain:
            chain_text = templates["causal_chain"].format(causal_chain=causal_chain)
            narrative_parts.append(chain_text)
        
        # Strength assessment
        strength_text = self._format_strength_assessment(primary_factor, audience, templates)
        narrative_parts.append(strength_text)
        
        # Additional factors
        if len(causal_factors) > 1:
            additional_factors = self._describe_additional_factors(causal_factors[1:], audience)
            if additional_factors:
                narrative_parts.append(additional_factors)
        
        # Conclusion
        key_factors = [self._translate_variable_name(f.variable) for f in causal_factors[:3]]
        conclusion = templates["conclusion"].format(
            key_factors=", ".join(key_factors),
            key_insights="; ".join(self._generate_key_insights(causal_factors, causal_result)[:2])
        )
        narrative_parts.append(conclusion)
        
        return " ".join(narrative_parts)
    
    def _generate_causal_chain(
        self, 
        causal_factors: List[CausalFactor], 
        graph: CausalGraph, 
        audience: NarrativeStyle
    ) -> str:
        """Generate causal chain description"""
        if not causal_factors:
            return ""
        
        chain_elements = []
        
        for factor in causal_factors:
            variable_name = self._translate_variable_name(factor.variable)
            relationship_desc = self.relationship_descriptors[factor.relationship]["description"]
            
            if audience == NarrativeStyle.TECHNICAL:
                element = f"{variable_name} ({relationship_desc})"
            else:
                element = variable_name
            
            chain_elements.append(element)
        
        if audience == NarrativeStyle.TECHNICAL:
            return " → ".join(chain_elements)
        else:
            return " influenced by ".join(chain_elements)
    
    def _format_strength_assessment(
        self, 
        factor: CausalFactor, 
        audience: NarrativeStyle, 
        templates: Dict[str, str]
    ) -> str:
        """Format strength assessment for factor"""
        descriptors = self.relationship_descriptors[factor.relationship]
        
        # Map strength to descriptive adjective
        strength_level = min(3, int(factor.strength * 4))
        strength_adjective = descriptors["strength_adjectives"][strength_level]
        
        if audience == NarrativeStyle.TECHNICAL:
            return templates["strength"].format(
                strength=factor.strength,
                confidence=factor.confidence
            )
        elif audience == NarrativeStyle.EXECUTIVE:
            strength_percentage = factor.strength * 100
            return templates["strength"].format(strength_percentage=strength_percentage)
        else:
            return templates["strength"].format(
                strength_description=strength_adjective,
                strength_level=strength_adjective,
                confidence=factor.confidence
            )
    
    def _describe_additional_factors(
        self, 
        factors: List[CausalFactor], 
        audience: NarrativeStyle
    ) -> str:
        """Describe additional causal factors"""
        if not factors:
            return ""
        
        descriptions = []
        for factor in factors:
            variable_name = self._translate_variable_name(factor.variable)
            relationship_desc = self.relationship_descriptors[factor.relationship]["description"]
            
            if audience == NarrativeStyle.TECHNICAL:
                desc = f"{variable_name} {relationship_desc} (strength: {factor.strength:.2f})"
            else:
                desc = f"{variable_name} also played a role"
            
            descriptions.append(desc)
        
        if audience == NarrativeStyle.TECHNICAL:
            return f"Additional factors: {'; '.join(descriptions)}."
        else:
            return f"Other contributing factors include {', '.join(descriptions)}."
    
    def _generate_key_insights(
        self, 
        causal_factors: List[CausalFactor], 
        causal_result: CausalResult
    ) -> List[str]:
        """Generate key insights from causal analysis"""
        insights = []
        
        if causal_factors:
            primary_factor = causal_factors[0]
            insights.append(
                f"{self._translate_variable_name(primary_factor.variable)} "
                f"is the primary driver of the outcome"
            )
        
        if causal_result.confounders:
            insights.append(
                f"Confounding factors include {', '.join(causal_result.confounders)}"
            )
        
        if causal_result.mediators:
            insights.append(
                f"The effect operates through {', '.join(causal_result.mediators)}"
            )
        
        if causal_result.p_value < 0.05:
            insights.append("The causal relationship is statistically significant")
        
        return insights
    
    def _generate_recommendations(
        self,
        causal_factors: List[CausalFactor],
        decision_context: Dict[str, Any],
        audience: NarrativeStyle
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if not causal_factors:
            return recommendations
        
        primary_factor = causal_factors[0]
        
        # Audience-specific recommendations
        if audience == NarrativeStyle.TRADER:
            recommendations.append(
                f"Monitor {self._translate_variable_name(primary_factor.variable)} closely for future trades"
            )
            recommendations.append(
                f"Consider adjusting position sizing based on {primary_factor.variable} strength"
            )
        
        elif audience == NarrativeStyle.EXECUTIVE:
            recommendations.append(
                f"Enhance monitoring of {self._translate_variable_name(primary_factor.variable)}"
            )
            recommendations.append(
                "Review risk management protocols for similar scenarios"
            )
        
        elif audience == NarrativeStyle.REGULATORY:
            recommendations.append(
                "Document causal factors in compliance reports"
            )
            recommendations.append(
                "Enhance model validation for identified causal relationships"
            )
        
        return recommendations
    
    def _assess_confidence(
        self, 
        causal_result: CausalResult, 
        causal_factors: List[CausalFactor]
    ) -> str:
        """Assess confidence in causal analysis"""
        if not causal_factors:
            return "Low confidence due to lack of identifiable causal factors"
        
        primary_confidence = causal_factors[0].confidence
        p_value = causal_result.p_value
        
        if primary_confidence > 0.8 and p_value < 0.01:
            return "High confidence - Strong statistical evidence and clear causal mechanisms"
        elif primary_confidence > 0.6 and p_value < 0.05:
            return "Moderate confidence - Adequate statistical evidence with identifiable causes"
        else:
            return "Low confidence - Weak statistical evidence or unclear causal mechanisms"
    
    def _identify_limitations(self, causal_result: CausalResult, graph: CausalGraph) -> List[str]:
        """Identify limitations in causal analysis"""
        limitations = []
        
        if not causal_result.identifiable:
            limitations.append("Causal effect is not identifiable from observational data")
        
        if causal_result.p_value > 0.05:
            limitations.append("Statistical significance is not achieved")
        
        if len(causal_result.backdoor_sets) == 0:
            limitations.append("No valid backdoor adjustment sets found")
        
        if len(graph.data) < 100:
            limitations.append("Limited sample size may affect reliability")
        
        limitations.extend(causal_result.assumptions)
        
        return limitations
    
    def _calculate_complexity_score(
        self, 
        causal_factors: List[CausalFactor], 
        graph: CausalGraph
    ) -> float:
        """Calculate narrative complexity score"""
        complexity_components = []
        
        # Factor count complexity
        factor_complexity = min(1.0, len(causal_factors) / 5.0)
        complexity_components.append(factor_complexity)
        
        # Graph complexity
        graph_complexity = min(1.0, len(graph.nodes) / 10.0)
        complexity_components.append(graph_complexity)
        
        # Relationship complexity
        relationship_types = set(f.relationship for f in causal_factors)
        relationship_complexity = len(relationship_types) / 6.0  # 6 relationship types
        complexity_components.append(relationship_complexity)
        
        return np.mean(complexity_components)
    
    def _translate_variable_name(self, variable: str) -> str:
        """Translate variable name to human-readable form"""
        vocabulary = self.financial_vocabulary["variables"]
        return vocabulary.get(variable, variable.replace("_", " "))
    
    async def generate_counterfactual_narrative(
        self,
        counterfactual_result: CounterfactualResult,
        decision_context: Dict[str, Any],
        audience: NarrativeStyle = NarrativeStyle.TECHNICAL
    ) -> List[str]:
        """Generate narrative for counterfactual scenarios"""
        narratives = []
        
        for scenario in counterfactual_result.scenarios:
            narrative = self._generate_scenario_narrative(scenario, decision_context, audience)
            narratives.append(narrative)
        
        return narratives
    
    def _generate_scenario_narrative(
        self,
        scenario: CounterfactualScenario,
        decision_context: Dict[str, Any],
        audience: NarrativeStyle
    ) -> str:
        """Generate narrative for single counterfactual scenario"""
        if not scenario.interventions:
            return "In the baseline scenario, no changes were made to the original conditions."
        
        intervention_descriptions = []
        for intervention in scenario.interventions:
            var_name = self._translate_variable_name(intervention.variable)
            intervention_descriptions.append(
                f"{var_name} changed from {intervention.original_value} to {intervention.counterfactual_value}"
            )
        
        interventions_text = ", ".join(intervention_descriptions)
        
        templates = {
            NarrativeStyle.TECHNICAL: (
                f"Counterfactual analysis: If {interventions_text}, "
                f"the predicted outcome would be {scenario.predicted_outcome:.4f} "
                f"with probability {scenario.outcome_probability:.3f}."
            ),
            NarrativeStyle.EXECUTIVE: (
                f"Alternative scenario: Had {interventions_text}, "
                f"the result would likely have been {scenario.predicted_outcome:.2f}."
            ),
            NarrativeStyle.TRADER: (
                f"What if {interventions_text}? "
                f"You would have seen {scenario.predicted_outcome:.2f} instead."
            ),
            NarrativeStyle.CLIENT: (
                f"In different conditions where {interventions_text}, "
                f"your investment would have performed at {scenario.predicted_outcome:.2f}."
            )
        }
        
        return templates.get(audience, templates[NarrativeStyle.TECHNICAL])
    
    async def generate_complete_explanation(
        self,
        causal_result: CausalResult,
        counterfactual_result: Optional[CounterfactualResult],
        graph: CausalGraph,
        decision_context: Dict[str, Any],
        audience: NarrativeStyle = NarrativeStyle.TECHNICAL
    ) -> CausalExplanation:
        """Generate complete causal explanation"""
        
        # Generate primary story
        primary_story = await self.generate_causal_narrative(
            causal_result, graph, decision_context, audience
        )
        
        # Generate counterfactual narratives
        counterfactual_narratives = []
        if counterfactual_result and self.config["include_counterfactuals"]:
            counterfactual_narratives = await self.generate_counterfactual_narrative(
                counterfactual_result, decision_context, audience
            )
        
        primary_story.counterfactual_scenarios = counterfactual_narratives
        
        # Generate alternative stories for different audiences
        alternative_stories = []
        if audience != NarrativeStyle.EXECUTIVE:
            alt_story = await self.generate_causal_narrative(
                causal_result, graph, decision_context, NarrativeStyle.EXECUTIVE
            )
            alternative_stories.append(alt_story)
        
        # Generate comparative analysis
        comparative_analysis = self._generate_comparative_analysis(
            causal_result, counterfactual_result
        )
        
        # Generate robustness assessment
        robustness_assessment = self._generate_robustness_assessment(causal_result)
        
        # Generate actionable insights
        actionable_insights = self._generate_actionable_insights(
            causal_result, counterfactual_result, decision_context
        )
        
        explanation = CausalExplanation(
            explanation_id=f"explanation_{uuid.uuid4().hex[:8]}",
            decision_context=decision_context,
            primary_story=primary_story,
            alternative_stories=alternative_stories,
            comparative_analysis=comparative_analysis,
            robustness_assessment=robustness_assessment,
            actionable_insights=actionable_insights,
            metadata={
                "total_narratives": 1 + len(alternative_stories),
                "counterfactual_scenarios": len(counterfactual_narratives),
                "target_audience": audience.value
            }
        )
        
        return explanation
    
    def _generate_comparative_analysis(
        self, 
        causal_result: CausalResult, 
        counterfactual_result: Optional[CounterfactualResult]
    ) -> str:
        """Generate comparative analysis"""
        if not counterfactual_result:
            return "No counterfactual scenarios available for comparison."
        
        baseline_prob = counterfactual_result.probability
        
        analysis_parts = []
        analysis_parts.append(f"Baseline probability: {baseline_prob:.3f}")
        
        if counterfactual_result.scenarios:
            scenario_probs = [s.outcome_probability for s in counterfactual_result.scenarios]
            best_scenario = max(scenario_probs)
            worst_scenario = min(scenario_probs)
            
            analysis_parts.append(f"Best alternative outcome: {best_scenario:.3f}")
            analysis_parts.append(f"Worst alternative outcome: {worst_scenario:.3f}")
            analysis_parts.append(f"Range of outcomes: {best_scenario - worst_scenario:.3f}")
        
        return "; ".join(analysis_parts)
    
    def _generate_robustness_assessment(self, causal_result: CausalResult) -> str:
        """Generate robustness assessment"""
        robustness_factors = []
        
        # Statistical robustness
        if causal_result.p_value < 0.01:
            robustness_factors.append("Strong statistical significance")
        elif causal_result.p_value < 0.05:
            robustness_factors.append("Adequate statistical significance")
        else:
            robustness_factors.append("Weak statistical significance")
        
        # Identification robustness
        if len(causal_result.backdoor_sets) > 1:
            robustness_factors.append("Multiple valid identification strategies")
        elif len(causal_result.backdoor_sets) == 1:
            robustness_factors.append("Single valid identification strategy")
        else:
            robustness_factors.append("No valid identification strategy")
        
        # Sensitivity analysis
        if causal_result.sensitivity_analysis:
            e_value = causal_result.sensitivity_analysis.get("e_value", 0)
            if e_value > 2:
                robustness_factors.append("Robust to moderate unmeasured confounding")
            elif e_value > 1.5:
                robustness_factors.append("Moderately robust to unmeasured confounding")
            else:
                robustness_factors.append("Sensitive to unmeasured confounding")
        
        return "; ".join(robustness_factors)
    
    def _generate_actionable_insights(
        self,
        causal_result: CausalResult,
        counterfactual_result: Optional[CounterfactualResult],
        decision_context: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable insights"""
        insights = []
        
        # Causal insights
        if causal_result.effect_size > 0.1:
            insights.append("Focus on strengthening the primary causal factor")
        
        if causal_result.confounders:
            insights.append("Control for confounding factors in future decisions")
        
        # Counterfactual insights
        if counterfactual_result and counterfactual_result.feasible_interventions:
            insights.append("Consider feasible alternative interventions")
        
        if counterfactual_result and counterfactual_result.closest_counterfactual:
            closest = counterfactual_result.closest_counterfactual
            if closest.total_cost < 0.5:
                insights.append("Low-cost alternatives are available")
        
        # Decision-specific insights
        decision_type = decision_context.get("action", "")
        if decision_type == "LONG":
            insights.append("Monitor market conditions for long position optimization")
        elif decision_type == "SHORT":
            insights.append("Consider risk management for short positions")
        
        return insights
    
    def _update_performance_stats(self, story: CausalStory):
        """Update performance statistics"""
        self.performance_stats["total_narratives"] += 1
        
        # Update averages
        total = self.performance_stats["total_narratives"]
        old_avg_time = self.performance_stats["avg_generation_time_ms"]
        new_time = story.metadata.get("generation_time_ms", 0)
        self.performance_stats["avg_generation_time_ms"] = (
            (old_avg_time * (total - 1) + new_time) / total
        )
        
        old_avg_length = self.performance_stats["avg_narrative_length"]
        new_length = story.metadata.get("narrative_length", 0)
        self.performance_stats["avg_narrative_length"] = (
            (old_avg_length * (total - 1) + new_length) / total
        )
        
        old_avg_complexity = self.performance_stats["avg_complexity_score"]
        self.performance_stats["avg_complexity_score"] = (
            (old_avg_complexity * (total - 1) + story.complexity_score) / total
        )
        
        # Update audience distribution
        self.performance_stats["audience_distribution"][story.audience.value] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get generator performance statistics"""
        return {
            "performance_stats": self.performance_stats.copy(),
            "narrative_templates": len(self.narrative_templates),
            "relationship_descriptors": len(self.relationship_descriptors),
            "financial_vocabulary": sum(len(v) for v in self.financial_vocabulary.values())
        }


# Test function
async def test_causal_narrative_generator():
    """Test the Causal Narrative Generator"""
    print("🧪 Testing Causal Narrative Generator")
    
    # Initialize generator
    generator = CausalNarrativeGenerator()
    
    # Create mock causal result
    from .do_calculus_engine import CausalResult
    causal_result = CausalResult(
        query_id="test_query",
        effect_size=0.75,
        confidence_interval=(0.2, 1.3),
        p_value=0.01,
        identifiable=True,
        identification_method="backdoor_adjustment",
        backdoor_sets=[{"market_volatility", "volume_ratio"}],
        confounders={"market_volatility"},
        mediators={"mlmi_signal"},
        colliders=set(),
        assumptions=["No unmeasured confounders"],
        sensitivity_analysis={"e_value": 2.1},
        metadata={}
    )
    
    # Create mock graph
    from .do_calculus_engine import CausalGraph, CausalNode, NodeType
    graph = CausalGraph()
    graph.add_node(CausalNode("decision", NodeType.DECISION))
    graph.add_node(CausalNode("market_volatility", NodeType.MARKET_FEATURE))
    graph.add_node(CausalNode("mlmi_signal", NodeType.AGENT_SIGNAL))
    
    # Create mock decision context
    decision_context = {
        "action": "LONG",
        "outcome": "positive return",
        "primary_cause": "strong market momentum"
    }
    
    # Test narrative generation for different audiences
    print("\\n📖 Testing narrative generation...")
    
    for audience in NarrativeStyle:
        print(f"\\n--- {audience.value.upper()} AUDIENCE ---")
        
        story = await generator.generate_causal_narrative(
            causal_result, graph, decision_context, audience
        )
        
        print(f"Title: {story.title}")
        print(f"Summary: {story.summary}")
        print(f"Narrative: {story.main_narrative[:200]}...")
        print(f"Complexity: {story.complexity_score:.2f}")
        print(f"Factors: {len(story.causal_factors)}")
    
    # Test complete explanation
    print("\\n🔍 Testing complete explanation generation...")
    
    explanation = await generator.generate_complete_explanation(
        causal_result, None, graph, decision_context, NarrativeStyle.EXECUTIVE
    )
    
    print(f"Complete explanation generated:")
    print(f"  Primary story: {explanation.primary_story.title}")
    print(f"  Alternative stories: {len(explanation.alternative_stories)}")
    print(f"  Actionable insights: {len(explanation.actionable_insights)}")
    
    # Performance stats
    print("\\n📈 Performance Statistics:")
    stats = generator.get_performance_stats()
    for key, value in stats["performance_stats"].items():
        print(f"  {key}: {value}")
    
    print("\\n✅ Causal Narrative Generator test complete!")


if __name__ == "__main__":
    asyncio.run(test_causal_narrative_generator())