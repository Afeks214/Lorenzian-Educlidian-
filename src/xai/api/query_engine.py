"""
Natural Language Query Processing Engine for XAI API
AGENT DELTA MISSION: Advanced Query Processing Engine

This module implements a sophisticated natural language query processing engine
that can understand and respond to complex analytics questions about trading
decisions, agent performance, and system behavior.

Features:
- Natural language understanding and intent recognition
- Complex analytics query execution  
- Performance aggregation and reporting
- Historical decision analysis and insights
- Multi-modal response generation (text, data, visualizations)
- Query optimization and caching

Author: Agent Delta - Integration Specialist  
Version: 1.0 - Natural Language Query Engine
"""

import re
import json
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd

# NLP and text processing
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from src.monitoring.logger_config import get_logger

logger = get_logger(__name__)

# Download required NLTK data (with error handling)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except Exception as e:
    logger.warning(f"NLTK download failed: {e}, using fallback text processing")
    NLTK_AVAILABLE = False


class QueryIntent(Enum):
    """Types of query intents"""
    PERFORMANCE_ANALYSIS = "PERFORMANCE_ANALYSIS"
    DECISION_EXPLANATION = "DECISION_EXPLANATION"
    AGENT_COMPARISON = "AGENT_COMPARISON"
    RISK_ASSESSMENT = "RISK_ASSESSMENT"
    HISTORICAL_ANALYSIS = "HISTORICAL_ANALYSIS"
    SYSTEM_STATUS = "SYSTEM_STATUS"
    MARKET_INSIGHTS = "MARKET_INSIGHTS"
    COMPLIANCE_QUERY = "COMPLIANCE_QUERY"
    UNKNOWN = "UNKNOWN"


class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "SIMPLE"        # Single metric or direct lookup
    MODERATE = "MODERATE"    # Multiple metrics or simple aggregation
    COMPLEX = "COMPLEX"      # Advanced analytics or cross-domain analysis
    ANALYTICAL = "ANALYTICAL" # Deep insights requiring multiple data sources


@dataclass
class QueryAnalysis:
    """Analysis of a natural language query"""
    original_query: str
    intent: QueryIntent
    complexity: QueryComplexity
    entities: Dict[str, List[str]]  # Extracted entities (symbols, agents, dates, etc.)
    keywords: List[str]
    time_range: Optional[Tuple[datetime, datetime]]
    target_agents: List[str]
    target_symbols: List[str]
    metrics_requested: List[str]
    confidence: float


@dataclass
class QueryResult:
    """Result of query processing"""
    query_analysis: QueryAnalysis
    intent: QueryIntent
    answer: str
    supporting_data: Dict[str, Any]
    visualizations: List[Dict[str, Any]]
    confidence: float
    follow_up_suggestions: List[str]
    data_sources: List[str]
    processing_time_ms: float
    requires_marl_data: bool
    data_requirements: List[str]
    data_points_count: int


class EntityExtractor:
    """Extract entities from natural language queries"""
    
    def __init__(self):
        """Initialize entity extractor"""
        self.stemmer = PorterStemmer() if NLTK_AVAILABLE else None
        self.stop_words = set(stopwords.words('english')) if NLTK_AVAILABLE else set()
        
        # Define entity patterns
        self.agent_patterns = {
            'mlmi': ['mlmi', 'momentum', 'liquidity', 'trend', 'momentum agent'],
            'nwrqk': ['nwrqk', 'risk', 'quality', 'net worth', 'risk agent'],
            'regime': ['regime', 'market regime', 'regime detection', 'regime agent'],
            'all_agents': ['agents', 'all agents', 'every agent', 'each agent']
        }
        
        self.symbol_patterns = [
            r'\b[A-Z]{1,4}\b',  # 1-4 letter symbols
            r'\b(NQ|ES|YM|RTY)\b',  # Common futures
            r'\b(SPY|QQQ|IWM|DIA)\b'  # Common ETFs
        ]
        
        self.time_patterns = {
            'today': lambda: (datetime.now().replace(hour=0, minute=0, second=0),
                             datetime.now()),
            'yesterday': lambda: (datetime.now().replace(hour=0, minute=0, second=0) - timedelta(days=1),
                                 datetime.now().replace(hour=0, minute=0, second=0)),
            'last week': lambda: (datetime.now() - timedelta(weeks=1), datetime.now()),
            'last month': lambda: (datetime.now() - timedelta(days=30), datetime.now()),
            'this week': lambda: (datetime.now() - timedelta(days=datetime.now().weekday()),
                                 datetime.now()),
            'this month': lambda: (datetime.now().replace(day=1), datetime.now()),
            'last 24 hours': lambda: (datetime.now() - timedelta(hours=24), datetime.now()),
            'last 7 days': lambda: (datetime.now() - timedelta(days=7), datetime.now()),
            'past hour': lambda: (datetime.now() - timedelta(hours=1), datetime.now())
        }
        
        self.metrics_patterns = {
            'performance': ['performance', 'accuracy', 'success rate', 'win rate', 'returns'],
            'risk': ['risk', 'var', 'value at risk', 'volatility', 'drawdown'],
            'confidence': ['confidence', 'certainty', 'conviction'],
            'latency': ['latency', 'speed', 'response time', 'execution time'],
            'volume': ['volume', 'trading volume', 'liquidity'],
            'pnl': ['pnl', 'profit', 'loss', 'returns', 'gains']
        }
        
    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from query"""
        entities = {
            'agents': [],
            'symbols': [],
            'time_expressions': [],
            'metrics': [],
            'actions': []
        }
        
        query_lower = query.lower()
        
        # Extract agents
        for agent, patterns in self.agent_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    entities['agents'].append(agent)
        
        # Extract symbols
        for pattern in self.symbol_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities['symbols'].extend(matches)
        
        # Extract time expressions
        for time_expr in self.time_patterns.keys():
            if time_expr in query_lower:
                entities['time_expressions'].append(time_expr)
        
        # Extract metrics
        for metric, patterns in self.metrics_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    entities['metrics'].append(metric)
        
        # Extract actions
        action_keywords = ['buy', 'sell', 'hold', 'long', 'short', 'trade', 'position']
        for action in action_keywords:
            if action in query_lower:
                entities['actions'].append(action)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def extract_time_range(self, query: str) -> Optional[Tuple[datetime, datetime]]:
        """Extract time range from query"""
        query_lower = query.lower()
        
        for time_expr, func in self.time_patterns.items():
            if time_expr in query_lower:
                return func()
        
        # Try to extract specific dates
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
            r'(\d{2}/\d{2}/\d{4})',  # MM/DD/YYYY
            r'(\d{1,2}/\d{1,2})',    # MM/DD (current year)
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, query)
            if matches:
                try:
                    if '-' in matches[0]:
                        date = datetime.strptime(matches[0], '%Y-%m-%d')
                    elif '/' in matches[0] and len(matches[0]) > 5:
                        date = datetime.strptime(matches[0], '%m/%d/%Y')
                    else:
                        date = datetime.strptime(f"{matches[0]}/{datetime.now().year}", '%m/%d/%Y')
                    
                    return (date, date + timedelta(days=1))
                except ValueError:
                    continue
        
        return None


class IntentClassifier:
    """Classify query intent using keyword-based approach"""
    
    def __init__(self):
        """Initialize intent classifier"""
        self.intent_keywords = {
            QueryIntent.PERFORMANCE_ANALYSIS: [
                'performance', 'accuracy', 'success', 'win rate', 'returns', 'profit',
                'how well', 'effectiveness', 'results', 'outcomes'
            ],
            QueryIntent.DECISION_EXPLANATION: [
                'why', 'explain', 'reason', 'rationale', 'because', 'decision',
                'chose', 'selected', 'recommended', 'suggested'
            ],
            QueryIntent.AGENT_COMPARISON: [
                'compare', 'comparison', 'better', 'best', 'worst', 'versus', 'vs',
                'difference', 'between', 'which agent'
            ],
            QueryIntent.RISK_ASSESSMENT: [
                'risk', 'var', 'volatility', 'drawdown', 'safety', 'dangerous',
                'risky', 'conservative', 'aggressive'
            ],
            QueryIntent.HISTORICAL_ANALYSIS: [
                'history', 'historical', 'past', 'previous', 'trend', 'over time',
                'timeline', 'evolution', 'progression'
            ],
            QueryIntent.SYSTEM_STATUS: [
                'status', 'health', 'running', 'operational', 'uptime', 'available',
                'working', 'functioning'
            ],
            QueryIntent.MARKET_INSIGHTS: [
                'market', 'regime', 'conditions', 'environment', 'sentiment',
                'outlook', 'forecast', 'prediction'
            ],
            QueryIntent.COMPLIANCE_QUERY: [
                'compliance', 'regulatory', 'audit', 'regulation', 'mifid',
                'best execution', 'transparency', 'report'
            ]
        }
    
    def classify_intent(self, query: str, entities: Dict[str, List[str]]) -> Tuple[QueryIntent, float]:
        """Classify query intent with confidence score"""
        query_lower = query.lower()
        intent_scores = {}
        
        # Score based on keyword matches
        for intent, keywords in self.intent_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in query_lower:
                    score += 1
            
            # Normalize by number of keywords
            intent_scores[intent] = score / len(keywords) if keywords else 0
        
        # Boost scores based on entities
        if entities.get('agents'):
            intent_scores[QueryIntent.AGENT_COMPARISON] += 0.2
            intent_scores[QueryIntent.PERFORMANCE_ANALYSIS] += 0.1
        
        if entities.get('time_expressions'):
            intent_scores[QueryIntent.HISTORICAL_ANALYSIS] += 0.2
        
        if entities.get('metrics'):
            intent_scores[QueryIntent.PERFORMANCE_ANALYSIS] += 0.3
            intent_scores[QueryIntent.RISK_ASSESSMENT] += 0.2
        
        # Find best intent
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            confidence = intent_scores[best_intent]
            
            # Minimum confidence threshold
            if confidence < 0.1:
                return QueryIntent.UNKNOWN, 0.0
            
            return best_intent, min(confidence, 1.0)
        
        return QueryIntent.UNKNOWN, 0.0
    
    def determine_complexity(self, query: str, entities: Dict[str, List[str]]) -> QueryComplexity:
        """Determine query complexity"""
        complexity_indicators = {
            'simple': ['what', 'when', 'where', 'who'],
            'moderate': ['how', 'compare', 'show me', 'list'],
            'complex': ['analyze', 'correlation', 'relationship', 'impact'],
            'analytical': ['predict', 'forecast', 'optimize', 'recommend', 'strategy']
        }
        
        query_lower = query.lower()
        scores = {}
        
        for complexity, indicators in complexity_indicators.items():
            score = sum(1 for indicator in indicators if indicator in query_lower)
            scores[complexity] = score
        
        # Factor in entity complexity
        entity_count = sum(len(entities[key]) for key in entities)
        if entity_count > 5:
            scores['complex'] += 1
        elif entity_count > 3:
            scores['moderate'] += 1
        
        # Determine complexity
        if scores.get('analytical', 0) > 0:
            return QueryComplexity.ANALYTICAL
        elif scores.get('complex', 0) > 0:
            return QueryComplexity.COMPLEX
        elif scores.get('moderate', 0) > 0:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE


class ResponseGenerator:
    """Generate natural language responses to queries"""
    
    def __init__(self):
        """Initialize response generator"""
        self.response_templates = {
            QueryIntent.PERFORMANCE_ANALYSIS: {
                'intro': "Based on the performance analysis:",
                'single_metric': "The {metric} for {target} is {value}.",
                'comparison': "Comparing performance: {comparison_text}",
                'summary': "Overall performance shows {summary}."
            },
            QueryIntent.DECISION_EXPLANATION: {
                'intro': "The decision was made because:",
                'reasoning': "{reasoning}",
                'factors': "Key factors include: {factors}",
                'confidence': "Decision confidence: {confidence}"
            },
            QueryIntent.AGENT_COMPARISON: {
                'intro': "Agent comparison analysis:",
                'performance': "{agent1} shows {perf1} while {agent2} shows {perf2}",
                'specialization': "Each agent specializes in: {specializations}",
                'recommendation': "Based on current conditions, {best_agent} is most suitable."
            },
            QueryIntent.RISK_ASSESSMENT: {
                'intro': "Risk assessment indicates:",
                'level': "Current risk level: {risk_level}",
                'factors': "Primary risk factors: {risk_factors}",
                'recommendation': "Risk management recommendation: {recommendation}"
            },
            QueryIntent.HISTORICAL_ANALYSIS: {
                'intro': "Historical analysis shows:",
                'trend': "The trend over {period} shows {trend_description}",
                'patterns': "Notable patterns: {patterns}",
                'insights': "Key insights: {insights}"
            },
            QueryIntent.SYSTEM_STATUS: {
                'intro': "System status report:",
                'health': "System health: {health_status}",
                'performance': "Current performance: {performance_metrics}",
                'alerts': "Active alerts: {alerts}"
            },
            QueryIntent.MARKET_INSIGHTS: {
                'intro': "Market analysis reveals:",
                'regime': "Current market regime: {regime}",
                'conditions': "Market conditions: {conditions}",
                'outlook': "Outlook: {outlook}"
            },
            QueryIntent.COMPLIANCE_QUERY: {
                'intro': "Compliance analysis:",
                'status': "Compliance status: {status}",
                'coverage': "Explanation coverage: {coverage}",
                'recommendations': "Recommendations: {recommendations}"
            }
        }
    
    def generate_response(
        self,
        query_analysis: QueryAnalysis,
        data: Dict[str, Any]
    ) -> str:
        """Generate natural language response"""
        intent = query_analysis.intent
        
        if intent == QueryIntent.UNKNOWN:
            return "I'm not sure I understand your question. Could you please rephrase or be more specific?"
        
        templates = self.response_templates.get(intent, {})
        response_parts = []
        
        # Add introduction
        if 'intro' in templates:
            response_parts.append(templates['intro'])
        
        # Generate intent-specific response
        if intent == QueryIntent.PERFORMANCE_ANALYSIS:
            response_parts.extend(self._generate_performance_response(query_analysis, data, templates))
        
        elif intent == QueryIntent.DECISION_EXPLANATION:
            response_parts.extend(self._generate_explanation_response(query_analysis, data, templates))
        
        elif intent == QueryIntent.AGENT_COMPARISON:
            response_parts.extend(self._generate_comparison_response(query_analysis, data, templates))
        
        elif intent == QueryIntent.RISK_ASSESSMENT:
            response_parts.extend(self._generate_risk_response(query_analysis, data, templates))
        
        elif intent == QueryIntent.HISTORICAL_ANALYSIS:
            response_parts.extend(self._generate_historical_response(query_analysis, data, templates))
        
        elif intent == QueryIntent.SYSTEM_STATUS:
            response_parts.extend(self._generate_status_response(query_analysis, data, templates))
        
        elif intent == QueryIntent.MARKET_INSIGHTS:
            response_parts.extend(self._generate_market_response(query_analysis, data, templates))
        
        elif intent == QueryIntent.COMPLIANCE_QUERY:
            response_parts.extend(self._generate_compliance_response(query_analysis, data, templates))
        
        return " ".join(response_parts)
    
    def _generate_performance_response(
        self,
        analysis: QueryAnalysis,
        data: Dict[str, Any],
        templates: Dict[str, str]
    ) -> List[str]:
        """Generate performance analysis response"""
        parts = []
        
        if 'agent_performance' in data:
            agent_data = data['agent_performance']
            
            if analysis.target_agents:
                for agent in analysis.target_agents:
                    if agent in agent_data:
                        perf = agent_data[agent]
                        accuracy = perf.get('recent_accuracy', 0.0)
                        parts.append(f"{agent.upper()} agent shows {accuracy:.1%} accuracy.")
            else:
                # Overall performance summary
                avg_accuracy = np.mean([
                    perf.get('recent_accuracy', 0.0) 
                    for perf in agent_data.values()
                ])
                parts.append(f"Average agent accuracy is {avg_accuracy:.1%}.")
        
        if 'overall_performance' in data:
            overall = data['overall_performance']
            win_rate = overall.get('win_rate', 0.0)
            sharpe = overall.get('sharpe_ratio', 0.0)
            parts.append(f"Overall system shows {win_rate:.1%} win rate and {sharpe:.2f} Sharpe ratio.")
        
        return parts
    
    def _generate_explanation_response(
        self,
        analysis: QueryAnalysis,
        data: Dict[str, Any],
        templates: Dict[str, str]
    ) -> List[str]:
        """Generate decision explanation response"""
        parts = []
        
        if 'explanation' in data:
            explanation = data['explanation']
            parts.append(explanation.get('reasoning', 'Decision reasoning not available.'))
            
            factors = explanation.get('top_factors', [])
            if factors:
                factor_names = [f[0] for f in factors[:3]]
                parts.append(f"Key factors: {', '.join(factor_names)}")
        
        return parts
    
    def _generate_comparison_response(
        self,
        analysis: QueryAnalysis,
        data: Dict[str, Any],
        templates: Dict[str, str]
    ) -> List[str]:
        """Generate agent comparison response"""
        parts = []
        
        if 'agent_performance' in data and len(analysis.target_agents) >= 2:
            agent_data = data['agent_performance']
            agent1, agent2 = analysis.target_agents[0], analysis.target_agents[1]
            
            if agent1 in agent_data and agent2 in agent_data:
                perf1 = agent_data[agent1].get('recent_accuracy', 0.0)
                perf2 = agent_data[agent2].get('recent_accuracy', 0.0)
                
                better_agent = agent1 if perf1 > perf2 else agent2
                parts.append(f"{agent1.upper()}: {perf1:.1%} accuracy, {agent2.upper()}: {perf2:.1%} accuracy.")
                parts.append(f"{better_agent.upper()} is currently performing better.")
        
        return parts
    
    def _generate_risk_response(
        self,
        analysis: QueryAnalysis,
        data: Dict[str, Any],
        templates: Dict[str, str]
    ) -> List[str]:
        """Generate risk assessment response"""
        parts = []
        
        if 'risk_metrics' in data:
            risk_data = data['risk_metrics']
            var = risk_data.get('portfolio_var', 0.0)
            volatility = risk_data.get('volatility', 0.0)
            
            parts.append(f"Portfolio VaR: {var:.2%}, Volatility: {volatility:.2%}")
            
            risk_level = "Low" if var < 0.02 else "Moderate" if var < 0.05 else "High"
            parts.append(f"Risk level: {risk_level}")
        
        return parts
    
    def _generate_historical_response(
        self,
        analysis: QueryAnalysis,
        data: Dict[str, Any],
        templates: Dict[str, str]
    ) -> List[str]:
        """Generate historical analysis response"""
        parts = []
        
        if 'decision_history' in data:
            history = data['decision_history']
            total_decisions = len(history)
            
            if total_decisions > 0:
                actions = [d.get('action', 'HOLD') for d in history]
                action_counts = pd.Series(actions).value_counts()
                most_common = action_counts.index[0] if len(action_counts) > 0 else 'HOLD'
                
                parts.append(f"Over the analyzed period: {total_decisions} decisions made.")
                parts.append(f"Most common action: {most_common} ({action_counts.iloc[0]} times)")
        
        return parts
    
    def _generate_status_response(
        self,
        analysis: QueryAnalysis,
        data: Dict[str, Any],
        templates: Dict[str, str]
    ) -> List[str]:
        """Generate system status response"""
        parts = []
        
        if 'system_health' in data:
            health = data['system_health']
            uptime = health.get('uptime', 0.0)
            latency = health.get('average_latency_ms', 0.0)
            
            parts.append(f"System uptime: {uptime:.1%}")
            parts.append(f"Average response time: {latency:.1f}ms")
            
            status = "Excellent" if uptime > 0.99 else "Good" if uptime > 0.95 else "Degraded"
            parts.append(f"Overall status: {status}")
        
        return parts
    
    def _generate_market_response(
        self,
        analysis: QueryAnalysis,
        data: Dict[str, Any],
        templates: Dict[str, str]
    ) -> List[str]:
        """Generate market insights response"""
        parts = []
        
        if 'market_analysis' in data:
            market = data['market_analysis']
            regime = market.get('current_regime', 'unknown')
            confidence = market.get('regime_confidence', 0.0)
            
            parts.append(f"Current market regime: {regime} (confidence: {confidence:.1%})")
            
            conditions = market.get('volatility_environment', 'unknown')
            parts.append(f"Market conditions: {conditions}")
        
        return parts
    
    def _generate_compliance_response(
        self,
        analysis: QueryAnalysis,
        data: Dict[str, Any],
        templates: Dict[str, str]
    ) -> List[str]:
        """Generate compliance query response"""
        parts = []
        
        if 'compliance_data' in data:
            compliance = data['compliance_data']
            status = compliance.get('overall_status', 'Unknown')
            coverage = compliance.get('explanation_coverage', 0)
            
            parts.append(f"Compliance status: {status}")
            parts.append(f"Decision explanation coverage: {coverage} decisions")
        
        return parts


class NaturalLanguageQueryEngine:
    """
    Natural Language Query Processing Engine
    
    Processes natural language queries about trading decisions, agent performance,
    and system analytics, returning structured responses with supporting data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Natural Language Query Engine"""
        self.config = config or self._default_config()
        
        # Core components
        self.entity_extractor = EntityExtractor()
        self.intent_classifier = IntentClassifier()
        self.response_generator = ResponseGenerator()
        
        # Query cache for performance
        self.query_cache: Dict[str, QueryResult] = {}
        
        # Performance metrics
        self.engine_metrics = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'intent_classification_accuracy': 0.0,
            'average_processing_time_ms': 0.0,
            'successful_queries': 0,
            'failed_queries': 0
        }
        
        # Health tracking
        self._healthy = True
        self._last_health_check = datetime.now()
        
        logger.info("Natural Language Query Engine initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'cache_ttl_minutes': 30,
            'max_cache_size': 1000,
            'min_confidence_threshold': 0.3,
            'max_processing_time_seconds': 10.0,
            'enable_caching': True,
            'enable_follow_up_suggestions': True
        }
    
    async def initialize(self) -> None:
        """Initialize query engine"""
        try:
            # Validate configuration
            self._validate_config()
            
            # Initialize components
            logger.info("Query engine components initialized successfully")
            
            self._healthy = True
            
        except Exception as e:
            logger.error(f"Failed to initialize query engine: {e}")
            self._healthy = False
            raise
    
    def _validate_config(self) -> None:
        """Validate configuration"""
        required_keys = ['cache_ttl_minutes', 'max_cache_size', 'min_confidence_threshold']
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        if self.config['min_confidence_threshold'] < 0 or self.config['min_confidence_threshold'] > 1:
            raise ValueError("min_confidence_threshold must be between 0 and 1")
    
    async def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        user_preferences: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> QueryResult:
        """
        Process natural language query and return structured response.
        
        Args:
            query: Natural language query string
            context: Additional context information
            user_preferences: User preferences for response formatting
            correlation_id: Request correlation ID for tracking
            
        Returns:
            QueryResult: Structured query result with answer and supporting data
        """
        start_time = time.time()
        
        try:
            logger.info(
                "Processing natural language query",
                extra={
                    "correlation_id": correlation_id,
                    "query_length": len(query),
                    "has_context": context is not None
                }
            )
            
            # Check cache first
            if self.config.get('enable_caching', True):
                cache_key = self._generate_cache_key(query, context)
                cached_result = self._get_cached_result(cache_key)
                
                if cached_result:
                    self.engine_metrics['cache_hits'] += 1
                    self.engine_metrics['total_queries'] += 1
                    return cached_result
            
            self.engine_metrics['cache_misses'] += 1
            
            # Analyze query
            query_analysis = await self._analyze_query(query, context)
            
            # Check confidence threshold
            if query_analysis.confidence < self.config['min_confidence_threshold']:
                return self._create_low_confidence_response(query_analysis)
            
            # Determine data requirements
            data_requirements = self._determine_data_requirements(query_analysis)
            
            # Execute query processing
            query_result = await self._execute_query(query_analysis, data_requirements, context)
            
            # Cache result if enabled
            if self.config.get('enable_caching', True):
                self._cache_result(cache_key, query_result)
            
            # Update metrics
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_metrics(processing_time_ms, success=True)
            
            logger.info(
                "Query processed successfully",
                extra={
                    "correlation_id": correlation_id,
                    "intent": query_analysis.intent.value,
                    "confidence": query_analysis.confidence,
                    "processing_time_ms": processing_time_ms
                }
            )
            
            return query_result
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_metrics(processing_time_ms, success=False)
            
            logger.error(
                "Query processing failed",
                extra={
                    "correlation_id": correlation_id,
                    "error": str(e),
                    "processing_time_ms": processing_time_ms
                },
                exc_info=True
            )
            
            return self._create_error_response(query, str(e))
    
    async def _analyze_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> QueryAnalysis:
        """Analyze natural language query"""
        
        # Extract entities
        entities = self.entity_extractor.extract_entities(query)
        
        # Extract time range
        time_range = self.entity_extractor.extract_time_range(query)
        
        # Classify intent
        intent, confidence = self.intent_classifier.classify_intent(query, entities)
        
        # Determine complexity
        complexity = self.intent_classifier.determine_complexity(query, entities)
        
        # Extract keywords
        keywords = self._extract_keywords(query)
        
        # Map entities to specific fields
        target_agents = self._map_agents(entities.get('agents', []))
        target_symbols = entities.get('symbols', [])
        metrics_requested = entities.get('metrics', [])
        
        query_analysis = QueryAnalysis(
            original_query=query,
            intent=intent,
            complexity=complexity,
            entities=entities,
            keywords=keywords,
            time_range=time_range,
            target_agents=target_agents,
            target_symbols=target_symbols,
            metrics_requested=metrics_requested,
            confidence=confidence
        )
        
        return query_analysis
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query"""
        if NLTK_AVAILABLE:
            # Use NLTK for better keyword extraction
            tokens = word_tokenize(query.lower())
            keywords = [
                self.entity_extractor.stemmer.stem(token)
                for token in tokens
                if token.isalpha() and token not in self.entity_extractor.stop_words
            ]
        else:
            # Fallback keyword extraction
            words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
            common_stopwords = {'the', 'and', 'but', 'for', 'are', 'with', 'was', 'this', 'that'}
            keywords = [word for word in words if word not in common_stopwords]
        
        return keywords[:10]  # Limit to top 10 keywords
    
    def _map_agents(self, agent_entities: List[str]) -> List[str]:
        """Map entity agent names to standard agent IDs"""
        mapping = {
            'mlmi': 'MLMI',
            'nwrqk': 'NWRQK',
            'regime': 'Regime',
            'all_agents': ['MLMI', 'NWRQK', 'Regime']
        }
        
        mapped_agents = []
        for agent in agent_entities:
            if agent in mapping:
                mapped = mapping[agent]
                if isinstance(mapped, list):
                    mapped_agents.extend(mapped)
                else:
                    mapped_agents.append(mapped)
        
        return list(set(mapped_agents))
    
    def _determine_data_requirements(self, analysis: QueryAnalysis) -> List[str]:
        """Determine what data is needed to answer the query"""
        requirements = []
        
        intent = analysis.intent
        
        if intent == QueryIntent.PERFORMANCE_ANALYSIS:
            requirements.extend(['agent_performance', 'overall_performance'])
        
        elif intent == QueryIntent.DECISION_EXPLANATION:
            requirements.extend(['decision_history', 'explanations'])
        
        elif intent == QueryIntent.AGENT_COMPARISON:
            requirements.extend(['agent_performance', 'agent_specializations'])
        
        elif intent == QueryIntent.RISK_ASSESSMENT:
            requirements.extend(['risk_metrics', 'portfolio_data'])
        
        elif intent == QueryIntent.HISTORICAL_ANALYSIS:
            requirements.extend(['decision_history', 'performance_history'])
        
        elif intent == QueryIntent.SYSTEM_STATUS:
            requirements.extend(['system_health', 'performance_metrics'])
        
        elif intent == QueryIntent.MARKET_INSIGHTS:
            requirements.extend(['market_analysis', 'regime_data'])
        
        elif intent == QueryIntent.COMPLIANCE_QUERY:
            requirements.extend(['compliance_data', 'audit_trail'])
        
        # Add time-specific requirements
        if analysis.time_range:
            requirements.append('time_filtered_data')
        
        # Add symbol-specific requirements
        if analysis.target_symbols:
            requirements.append('symbol_specific_data')
        
        return list(set(requirements))
    
    async def _execute_query(
        self,
        analysis: QueryAnalysis,
        data_requirements: List[str],
        context: Optional[Dict[str, Any]]
    ) -> QueryResult:
        """Execute query processing"""
        
        # Mock data for now - in production this would fetch real data
        supporting_data = await self._fetch_supporting_data(data_requirements, analysis)
        
        # Generate natural language response
        answer = self.response_generator.generate_response(analysis, supporting_data)
        
        # Generate visualizations
        visualizations = self._generate_visualizations(analysis, supporting_data)
        
        # Generate follow-up suggestions
        follow_up_suggestions = self._generate_follow_up_suggestions(analysis)
        
        # Determine data sources
        data_sources = self._determine_data_sources(data_requirements)
        
        query_result = QueryResult(
            query_analysis=analysis,
            intent=analysis.intent,
            answer=answer,
            supporting_data=supporting_data,
            visualizations=visualizations,
            confidence=analysis.confidence,
            follow_up_suggestions=follow_up_suggestions,
            data_sources=data_sources,
            processing_time_ms=0.0,  # Will be set by caller
            requires_marl_data=self._requires_marl_data(data_requirements),
            data_requirements=data_requirements,
            data_points_count=self._count_data_points(supporting_data)
        )
        
        return query_result
    
    async def _fetch_supporting_data(
        self,
        requirements: List[str],
        analysis: QueryAnalysis
    ) -> Dict[str, Any]:
        """Fetch supporting data for query"""
        
        # Mock data - in production would fetch from actual data sources
        supporting_data = {}
        
        if 'agent_performance' in requirements:
            supporting_data['agent_performance'] = {
                'MLMI': {
                    'recent_accuracy': 0.72,
                    'trend_following_score': 0.85,
                    'confidence_calibration': 0.78
                },
                'NWRQK': {
                    'recent_accuracy': 0.68,
                    'risk_assessment_score': 0.82,
                    'confidence_calibration': 0.75
                },
                'Regime': {
                    'recent_accuracy': 0.75,
                    'regime_detection_score': 0.88,
                    'confidence_calibration': 0.82
                }
            }
        
        if 'overall_performance' in requirements:
            supporting_data['overall_performance'] = {
                'win_rate': 0.67,
                'sharpe_ratio': 1.45,
                'total_trades': 234,
                'profit_factor': 1.34,
                'max_drawdown': 0.08
            }
        
        if 'risk_metrics' in requirements:
            supporting_data['risk_metrics'] = {
                'portfolio_var': 0.018,
                'expected_shortfall': 0.022,
                'volatility': 0.15,
                'correlation_risk': 0.12
            }
        
        if 'system_health' in requirements:
            supporting_data['system_health'] = {
                'uptime': 0.999,
                'average_latency_ms': 3.2,
                'error_rate': 0.001,
                'active_connections': 45
            }
        
        if 'market_analysis' in requirements:
            supporting_data['market_analysis'] = {
                'current_regime': 'trending',
                'regime_confidence': 0.83,
                'volatility_environment': 'moderate',
                'liquidity_conditions': 'normal'
            }
        
        if 'decision_history' in requirements:
            # Generate mock decision history
            decisions = []
            for i in range(10):
                decisions.append({
                    'id': f'decision_{i}',
                    'timestamp': datetime.now() - timedelta(hours=i),
                    'action': np.random.choice(['LONG', 'SHORT', 'HOLD']),
                    'confidence': 0.6 + np.random.normal(0, 0.1),
                    'symbol': analysis.target_symbols[0] if analysis.target_symbols else 'NQ'
                })
            supporting_data['decision_history'] = decisions
        
        return supporting_data
    
    def _generate_visualizations(
        self,
        analysis: QueryAnalysis,
        data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate visualization specifications"""
        visualizations = []
        
        intent = analysis.intent
        
        if intent == QueryIntent.PERFORMANCE_ANALYSIS:
            if 'agent_performance' in data:
                visualizations.append({
                    'type': 'bar_chart',
                    'title': 'Agent Performance Comparison',
                    'data': data['agent_performance'],
                    'x_axis': 'Agent',
                    'y_axis': 'Accuracy',
                    'description': 'Comparison of recent agent accuracy scores'
                })
        
        elif intent == QueryIntent.HISTORICAL_ANALYSIS:
            if 'decision_history' in data:
                visualizations.append({
                    'type': 'time_series',
                    'title': 'Decision History Timeline',
                    'data': data['decision_history'],
                    'x_axis': 'Timestamp',
                    'y_axis': 'Confidence',
                    'description': 'Decision confidence over time'
                })
        
        elif intent == QueryIntent.RISK_ASSESSMENT:
            if 'risk_metrics' in data:
                visualizations.append({
                    'type': 'gauge',
                    'title': 'Risk Level Indicator',
                    'data': data['risk_metrics'],
                    'metric': 'portfolio_var',
                    'description': 'Current portfolio risk level'
                })
        
        return visualizations
    
    def _generate_follow_up_suggestions(self, analysis: QueryAnalysis) -> List[str]:
        """Generate follow-up question suggestions"""
        if not self.config.get('enable_follow_up_suggestions', True):
            return []
        
        intent = analysis.intent
        suggestions = []
        
        if intent == QueryIntent.PERFORMANCE_ANALYSIS:
            suggestions.extend([
                "How has performance changed over the last week?",
                "Which agent is performing best in current market conditions?",
                "What factors are driving the performance differences?"
            ])
        
        elif intent == QueryIntent.DECISION_EXPLANATION:
            suggestions.extend([
                "What alternative decisions were considered?",
                "How confident was the system in this decision?",
                "What market conditions influenced this decision?"
            ])
        
        elif intent == QueryIntent.AGENT_COMPARISON:
            suggestions.extend([
                "Which agent specializes in trending markets?",
                "How do agent performances correlate with market regimes?",
                "What are the strengths of each agent?"
            ])
        
        elif intent == QueryIntent.RISK_ASSESSMENT:
            suggestions.extend([
                "What are the main risk factors right now?",
                "How does current risk compare to historical levels?",
                "What risk mitigation strategies are recommended?"
            ])
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def _determine_data_sources(self, requirements: List[str]) -> List[str]:
        """Determine data sources used"""
        sources = set()
        
        source_mapping = {
            'agent_performance': 'Strategic MARL System',
            'overall_performance': 'Performance Analytics Engine',
            'risk_metrics': 'Risk Management System',
            'system_health': 'System Monitoring',
            'market_analysis': 'Market Intelligence Module',
            'decision_history': 'Decision Database',
            'compliance_data': 'Compliance Management System'
        }
        
        for requirement in requirements:
            if requirement in source_mapping:
                sources.add(source_mapping[requirement])
        
        return list(sources)
    
    def _requires_marl_data(self, requirements: List[str]) -> bool:
        """Check if query requires Strategic MARL data"""
        marl_requirements = {
            'agent_performance', 'decision_history', 'agent_specializations',
            'performance_history', 'symbol_specific_data'
        }
        
        return bool(set(requirements) & marl_requirements)
    
    def _count_data_points(self, data: Dict[str, Any]) -> int:
        """Count total data points in supporting data"""
        count = 0
        
        for key, value in data.items():
            if isinstance(value, list):
                count += len(value)
            elif isinstance(value, dict):
                count += len(value)
            else:
                count += 1
        
        return count
    
    def _generate_cache_key(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Generate cache key for query"""
        query_hash = hash(query.lower().strip())
        context_hash = hash(str(context)) if context else 0
        return f"query_{query_hash}_{context_hash}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[QueryResult]:
        """Get cached query result if valid"""
        if cache_key in self.query_cache:
            cached_result = self.query_cache[cache_key]
            # Check cache TTL (simplified)
            return cached_result
        
        return None
    
    def _cache_result(self, cache_key: str, result: QueryResult) -> None:
        """Cache query result"""
        # Manage cache size
        if len(self.query_cache) >= self.config['max_cache_size']:
            # Remove oldest entry (simplified LRU)
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
        
        self.query_cache[cache_key] = result
    
    def _create_low_confidence_response(self, analysis: QueryAnalysis) -> QueryResult:
        """Create response for low confidence queries"""
        return QueryResult(
            query_analysis=analysis,
            intent=QueryIntent.UNKNOWN,
            answer="I'm not confident I understand your question correctly. Could you please rephrase or provide more details?",
            supporting_data={},
            visualizations=[],
            confidence=analysis.confidence,
            follow_up_suggestions=[
                "Try being more specific about what you'd like to know",
                "Include specific agent names or time periods",
                "Ask about performance, risk, or decision explanations"
            ],
            data_sources=[],
            processing_time_ms=1.0,
            requires_marl_data=False,
            data_requirements=[],
            data_points_count=0
        )
    
    def _create_error_response(self, query: str, error: str) -> QueryResult:
        """Create response for query processing errors"""
        return QueryResult(
            query_analysis=QueryAnalysis(
                original_query=query,
                intent=QueryIntent.UNKNOWN,
                complexity=QueryComplexity.SIMPLE,
                entities={},
                keywords=[],
                time_range=None,
                target_agents=[],
                target_symbols=[],
                metrics_requested=[],
                confidence=0.0
            ),
            intent=QueryIntent.UNKNOWN,
            answer=f"I encountered an error processing your query: {error}. Please try again with a different question.",
            supporting_data={},
            visualizations=[],
            confidence=0.0,
            follow_up_suggestions=[
                "Try a simpler question",
                "Check your query for any special characters",
                "Contact support if the issue persists"
            ],
            data_sources=[],
            processing_time_ms=1.0,
            requires_marl_data=False,
            data_requirements=[],
            data_points_count=0
        )
    
    def _update_metrics(self, processing_time_ms: float, success: bool) -> None:
        """Update engine metrics"""
        self.engine_metrics['total_queries'] += 1
        
        if success:
            self.engine_metrics['successful_queries'] += 1
        else:
            self.engine_metrics['failed_queries'] += 1
        
        # Update average processing time
        total_queries = self.engine_metrics['total_queries']
        old_avg = self.engine_metrics['average_processing_time_ms']
        self.engine_metrics['average_processing_time_ms'] = (
            (old_avg * (total_queries - 1) + processing_time_ms) / total_queries
        )
    
    async def enrich_with_marl_data(
        self,
        query_result: QueryResult,
        marl_data: Dict[str, Any]
    ) -> QueryResult:
        """Enrich query result with Strategic MARL data"""
        try:
            # Merge MARL data into supporting data
            enriched_data = {**query_result.supporting_data, **marl_data}
            
            # Regenerate answer with enriched data
            enriched_answer = self.response_generator.generate_response(
                query_result.query_analysis, enriched_data
            )
            
            # Update query result
            query_result.supporting_data = enriched_data
            query_result.answer = enriched_answer
            query_result.data_points_count = self._count_data_points(enriched_data)
            
            return query_result
            
        except Exception as e:
            logger.error(f"Failed to enrich query result with MARL data: {e}")
            return query_result
    
    def is_healthy(self) -> bool:
        """Check query engine health"""
        try:
            self._last_health_check = datetime.now()
            
            # Check basic functionality
            if not self._healthy:
                return False
            
            # Check cache size
            if len(self.query_cache) > self.config['max_cache_size'] * 1.1:
                logger.warning("Query cache size exceeded")
                return False
            
            # Check error rate
            total_queries = self.engine_metrics['total_queries']
            if total_queries > 10:  # Only check after some queries
                error_rate = self.engine_metrics['failed_queries'] / total_queries
                if error_rate > 0.2:  # 20% error threshold
                    logger.warning(f"High query error rate: {error_rate:.2%}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Query engine health check failed: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown query engine"""
        try:
            logger.info("Shutting down Natural Language Query Engine")
            
            # Clear caches
            self.query_cache.clear()
            
            self._healthy = False
            
            logger.info(
                "Query engine shutdown complete",
                extra={"final_metrics": self.engine_metrics}
            )
            
        except Exception as e:
            logger.error(f"Error during query engine shutdown: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get engine metrics"""
        return {
            **self.engine_metrics,
            'cache_size': len(self.query_cache),
            'health_status': self.is_healthy(),
            'last_health_check': self._last_health_check.isoformat()
        }