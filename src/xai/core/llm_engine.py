"""
Ollama LLM Engine for Trading Explanations
Agent Alpha Mission: Local LLM Inference for <100ms Response Times

Implements Ollama Phi model integration for financial domain-specific
explanation generation with context-aware prompting and caching.

Target: <100ms explanation generation with local inference
"""

import asyncio
import aiohttp
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import numpy as np

logger = logging.getLogger(__name__)


class ExplanationStyle(Enum):
    """Explanation generation styles"""
    CONCISE = "concise"
    DETAILED = "detailed"
    TECHNICAL = "technical"
    REGULATORY = "regulatory"
    CLIENT_FRIENDLY = "client_friendly"


class PromptTemplate(Enum):
    """Pre-defined prompt templates for different use cases"""
    TRADING_DECISION = "trading_decision"
    RISK_ASSESSMENT = "risk_assessment"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    MARKET_CONTEXT = "market_context"
    COMPLIANCE_REPORT = "compliance_report"


@dataclass
class OllamaConfig:
    """Configuration for Ollama LLM Engine"""
    base_url: str = "http://localhost:11434"
    model_name: str = "phi"
    max_tokens: int = 256
    temperature: float = 0.3
    timeout_seconds: int = 10
    cache_size: int = 500
    cache_ttl_minutes: int = 15
    enable_streaming: bool = False
    retry_attempts: int = 3
    retry_delay_seconds: float = 0.5


@dataclass
class ExplanationContext:
    """Context information for explanation generation"""
    symbol: str
    action: str
    confidence: float
    timestamp: datetime
    market_features: Dict[str, float]
    similar_decisions: List[Dict[str, Any]]
    performance_metrics: Optional[Dict[str, Any]] = None
    risk_metrics: Optional[Dict[str, Any]] = None
    user_preferences: Optional[Dict[str, Any]] = None


@dataclass
class ExplanationResult:
    """Result from explanation generation"""
    explanation_text: str
    confidence_score: float
    generation_time_ms: float
    tokens_generated: int
    cached: bool
    template_used: str
    style: ExplanationStyle
    context_factors: List[str]


class OllamaExplanationEngine:
    """
    High-performance Ollama LLM Engine for trading explanations
    
    Provides local LLM inference with <100ms response time target
    """
    
    def __init__(self, config: Optional[OllamaConfig] = None):
        """Initialize Ollama explanation engine"""
        self.config = config or OllamaConfig()
        
        # HTTP client for Ollama API
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Response cache for performance
        self.response_cache: Dict[str, Tuple[ExplanationResult, float]] = {}
        
        # Performance tracking
        self.performance_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'avg_generation_time_ms': 0.0,
            'avg_tokens_generated': 0,
            'error_count': 0,
            'timeout_count': 0,
            'retry_count': 0
        }
        
        # Prompt templates
        self.prompt_templates = self._initialize_prompt_templates()
        
        # Financial domain vocabulary
        self.financial_vocabulary = self._initialize_financial_vocabulary()
        
        logger.info(f"OllamaExplanationEngine initialized with model: {self.config.model_name}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._initialize_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self._cleanup_session()
    
    async def _initialize_session(self):
        """Initialize aiohttp session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def _cleanup_session(self):
        """Clean up aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def _initialize_prompt_templates(self) -> Dict[PromptTemplate, str]:
        """Initialize financial domain-specific prompt templates"""
        return {
            PromptTemplate.TRADING_DECISION: """
You are a professional trading explanation system. Explain this trading decision clearly and concisely.

Trading Decision: {action} {symbol}
Confidence: {confidence:.1%}
Timestamp: {timestamp}

Market Context:
{market_context}

Similar Historical Decisions:
{similar_decisions}

Provide a clear, professional explanation in {style} style focusing on:
1. Primary factors driving the decision
2. Confidence rationale
3. Risk considerations

Keep response under 200 words and use financial terminology appropriately.
            """.strip(),
            
            PromptTemplate.RISK_ASSESSMENT: """
You are a risk management expert. Analyze this trading decision from a risk perspective.

Decision: {action} {symbol} (Confidence: {confidence:.1%})
Risk Metrics: {risk_metrics}
Market Conditions: {market_context}

Provide a risk assessment covering:
1. Primary risk factors
2. Risk mitigation measures
3. Position sizing considerations
4. Monitoring requirements

Response should be professional and actionable for risk managers.
            """.strip(),
            
            PromptTemplate.PERFORMANCE_ANALYSIS: """
You are a performance analyst. Explain this decision in the context of historical performance.

Decision: {action} {symbol}
Performance Metrics: {performance_metrics}
Historical Context: {similar_decisions}

Analyze:
1. Expected performance based on similar decisions
2. Success probability factors
3. Performance attribution
4. Improvement opportunities

Provide data-driven insights for performance optimization.
            """.strip(),
            
            PromptTemplate.MARKET_CONTEXT: """
You are a market analyst. Explain the market context for this trading decision.

Symbol: {symbol}
Action: {action}
Market Features: {market_context}
Timestamp: {timestamp}

Explain:
1. Current market regime
2. Technical indicators significance
3. Market structure factors
4. Timing considerations

Focus on market microstructure and technical analysis.
            """.strip(),
            
            PromptTemplate.COMPLIANCE_REPORT: """
You are a compliance officer. Generate a regulatory-compliant explanation for this algorithmic trading decision.

Decision ID: {decision_id}
Symbol: {symbol}
Action: {action}
Timestamp: {timestamp}
Confidence: {confidence:.1%}

Algorithm Details: Multi-Agent Reinforcement Learning System
Decision Factors: {market_context}

Provide compliance documentation including:
1. Decision methodology
2. Risk controls applied
3. Audit trail information
4. Regulatory compliance status

Format for regulatory submission.
            """.strip()
        }
    
    def _initialize_financial_vocabulary(self) -> Dict[str, str]:
        """Initialize financial domain vocabulary for consistent terminology"""
        return {
            # Actions
            'LONG': 'long position',
            'SHORT': 'short position', 
            'HOLD': 'hold position',
            'BUY': 'buy order',
            'SELL': 'sell order',
            
            # Market conditions
            'VOLATILE': 'elevated volatility',
            'TRENDING': 'trending market',
            'RANGING': 'range-bound market',
            'LIQUID': 'liquid market conditions',
            
            # Technical indicators
            'MOMENTUM': 'price momentum',
            'MEAN_REVERSION': 'mean reversion signal',
            'BREAKOUT': 'breakout pattern',
            'SUPPORT': 'support level',
            'RESISTANCE': 'resistance level',
            
            # Risk terms
            'HIGH_RISK': 'elevated risk environment',
            'MODERATE_RISK': 'moderate risk conditions',
            'LOW_RISK': 'low risk environment'
        }
    
    async def generate_explanation(
        self,
        context: ExplanationContext,
        template: PromptTemplate = PromptTemplate.TRADING_DECISION,
        style: ExplanationStyle = ExplanationStyle.CONCISE,
        use_cache: bool = True
    ) -> ExplanationResult:
        """
        Generate trading explanation using Ollama LLM
        
        Args:
            context: Explanation context with decision details
            template: Prompt template to use
            style: Explanation style
            use_cache: Whether to use cached responses
            
        Returns:
            ExplanationResult: Generated explanation with metadata
        """
        start_time = time.time()
        
        try:
            # Check cache first
            if use_cache:
                cache_key = self._generate_cache_key(context, template, style)
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    self.performance_stats['cache_hits'] += 1
                    return cached_result
            
            # Ensure session is initialized
            if not self.session:
                await self._initialize_session()
            
            # Generate prompt
            prompt = self._build_prompt(context, template, style)
            
            # Generate explanation with retry logic
            explanation_text, tokens_generated = await self._generate_with_retry(prompt)
            
            # Calculate confidence score based on various factors
            confidence_score = self._calculate_confidence_score(
                explanation_text, context, tokens_generated
            )
            
            # Create result
            generation_time_ms = (time.time() - start_time) * 1000
            result = ExplanationResult(
                explanation_text=explanation_text,
                confidence_score=confidence_score,
                generation_time_ms=generation_time_ms,
                tokens_generated=tokens_generated,
                cached=False,
                template_used=template.value,
                style=style,
                context_factors=list(context.market_features.keys())[:5]
            )
            
            # Cache result
            if use_cache:
                self._cache_result(cache_key, result)
            
            # Update performance stats
            self._update_performance_stats(generation_time_ms, tokens_generated)
            
            logger.debug(f"Generated explanation in {generation_time_ms:.1f}ms")
            return result
            
        except asyncio.TimeoutError:
            self.performance_stats['timeout_count'] += 1
            return self._create_fallback_explanation(context, style, "timeout")
            
        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            self.performance_stats['error_count'] += 1
            return self._create_fallback_explanation(context, style, str(e))
    
    def _build_prompt(
        self,
        context: ExplanationContext,
        template: PromptTemplate,
        style: ExplanationStyle
    ) -> str:
        """Build prompt from template and context"""
        
        # Get template
        prompt_template = self.prompt_templates[template]
        
        # Format market context
        market_context = self._format_market_context(context.market_features)
        
        # Format similar decisions
        similar_decisions = self._format_similar_decisions(context.similar_decisions)
        
        # Format risk metrics
        risk_metrics = self._format_risk_metrics(context.risk_metrics)
        
        # Format performance metrics
        performance_metrics = self._format_performance_metrics(context.performance_metrics)
        
        # Build prompt with context
        prompt = prompt_template.format(
            action=context.action,
            symbol=context.symbol,
            confidence=context.confidence,
            timestamp=context.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            market_context=market_context,
            similar_decisions=similar_decisions,
            risk_metrics=risk_metrics,
            performance_metrics=performance_metrics,
            style=style.value,
            decision_id=f"{context.symbol}_{int(context.timestamp.timestamp())}"
        )
        
        return prompt
    
    def _format_market_context(self, features: Dict[str, float]) -> str:
        """Format market features for prompt"""
        if not features:
            return "Standard market conditions"
        
        # Sort features by importance (absolute value)
        sorted_features = sorted(
            features.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )[:5]
        
        context_parts = []
        for feature_name, value in sorted_features:
            # Clean up feature name
            clean_name = feature_name.replace('_', ' ').title()
            
            # Format value with appropriate description
            if abs(value) > 0.5:
                strength = "strong"
            elif abs(value) > 0.3:
                strength = "moderate"
            else:
                strength = "weak"
            
            direction = "positive" if value > 0 else "negative"
            context_parts.append(f"{clean_name}: {strength} {direction} signal ({value:.3f})")
        
        return "; ".join(context_parts)
    
    def _format_similar_decisions(self, similar_decisions: List[Dict[str, Any]]) -> str:
        """Format similar decisions for prompt context"""
        if not similar_decisions:
            return "No similar historical decisions found"
        
        formatted_decisions = []
        for i, decision in enumerate(similar_decisions[:3]):  # Top 3 similar
            metadata = decision.get('metadata', {})
            similarity = decision.get('score', 0)
            
            formatted_decisions.append(
                f"{i+1}. {metadata.get('action', 'UNKNOWN')} {metadata.get('symbol', 'UNKNOWN')} "
                f"(confidence: {metadata.get('confidence', 0):.1%}, similarity: {similarity:.1%})"
            )
        
        return "\n".join(formatted_decisions)
    
    def _format_risk_metrics(self, risk_metrics: Optional[Dict[str, Any]]) -> str:
        """Format risk metrics for prompt"""
        if not risk_metrics:
            return "Standard risk profile"
        
        risk_parts = []
        for metric, value in risk_metrics.items():
            if isinstance(value, (int, float)):
                risk_parts.append(f"{metric.replace('_', ' ').title()}: {value:.3f}")
        
        return "; ".join(risk_parts) if risk_parts else "Standard risk profile"
    
    def _format_performance_metrics(self, performance_metrics: Optional[Dict[str, Any]]) -> str:
        """Format performance metrics for prompt"""
        if not performance_metrics:
            return "No recent performance data"
        
        perf_parts = []
        for metric, value in performance_metrics.items():
            if isinstance(value, (int, float)):
                if 'return' in metric.lower():
                    perf_parts.append(f"{metric.replace('_', ' ').title()}: {value*100:+.2f}%")
                else:
                    perf_parts.append(f"{metric.replace('_', ' ').title()}: {value:.3f}")
        
        return "; ".join(perf_parts) if perf_parts else "No recent performance data"
    
    async def _generate_with_retry(self, prompt: str) -> Tuple[str, int]:
        """Generate explanation with retry logic"""
        
        for attempt in range(self.config.retry_attempts):
            try:
                # Prepare request payload
                payload = {
                    "model": self.config.model_name,
                    "prompt": prompt,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens,
                        "stop": ["Human:", "Assistant:", "\n\n"]
                    },
                    "stream": self.config.enable_streaming
                }
                
                # Send request to Ollama
                explanation_text, tokens_generated = await self._send_ollama_request(payload)
                
                # Post-process explanation
                explanation_text = self._post_process_explanation(explanation_text)
                
                return explanation_text, tokens_generated
                
            except Exception as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                self.performance_stats['retry_count'] += 1
                
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay_seconds)
                else:
                    raise
    
    async def _send_ollama_request(self, payload: Dict[str, Any]) -> Tuple[str, int]:
        """Send request to Ollama API and return response"""
        
        url = f"{self.config.base_url}/api/generate"
        
        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                if self.config.enable_streaming:
                    # Handle streaming response
                    explanation_parts = []
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line.decode().strip())
                                if 'response' in data:
                                    explanation_parts.append(data['response'])
                                if data.get('done', False):
                                    break
                            except json.JSONDecodeError:
                                continue
                    
                    explanation_text = ''.join(explanation_parts)
                    tokens_generated = len(explanation_text.split())
                    
                else:
                    # Handle non-streaming response
                    data = await response.json()
                    explanation_text = data.get('response', '')
                    tokens_generated = len(explanation_text.split())
                
                return explanation_text, tokens_generated
                
            else:
                error_text = await response.text()
                raise Exception(f"Ollama API error: {response.status} - {error_text}")
    
    def _post_process_explanation(self, explanation_text: str) -> str:
        """Post-process generated explanation for quality and consistency"""
        
        # Remove common artifacts
        explanation_text = explanation_text.strip()
        
        # Remove duplicate sentences
        sentences = explanation_text.split('. ')
        unique_sentences = []
        seen_sentences = set()
        
        for sentence in sentences:
            sentence_clean = sentence.strip().lower()
            if sentence_clean not in seen_sentences and len(sentence_clean) > 10:
                unique_sentences.append(sentence.strip())
                seen_sentences.add(sentence_clean)
        
        # Rejoin sentences
        explanation_text = '. '.join(unique_sentences)
        
        # Ensure proper ending
        if not explanation_text.endswith('.'):
            explanation_text += '.'
        
        # Apply financial vocabulary consistency
        for term, replacement in self.financial_vocabulary.items():
            explanation_text = explanation_text.replace(term.lower(), replacement)
        
        return explanation_text
    
    def _calculate_confidence_score(
        self,
        explanation_text: str,
        context: ExplanationContext,
        tokens_generated: int
    ) -> float:
        """Calculate confidence score for generated explanation"""
        
        confidence_factors = []
        
        # Length factor (optimal length around 150-250 characters)
        text_length = len(explanation_text)
        if 150 <= text_length <= 250:
            length_score = 1.0
        elif 100 <= text_length <= 300:
            length_score = 0.8
        else:
            length_score = 0.6
        confidence_factors.append(length_score)
        
        # Token efficiency (characters per token)
        if tokens_generated > 0:
            char_per_token = text_length / tokens_generated
            token_efficiency = min(1.0, char_per_token / 4.0)  # Target ~4 chars/token
            confidence_factors.append(token_efficiency)
        
        # Content quality (presence of key financial terms)
        financial_terms_found = 0
        for term in ['decision', 'confidence', 'market', 'risk', 'position', 'signal']:
            if term.lower() in explanation_text.lower():
                financial_terms_found += 1
        
        content_quality = min(1.0, financial_terms_found / 4.0)
        confidence_factors.append(content_quality)
        
        # Context alignment (mentions context symbol/action)
        context_alignment = 0.5  # Base score
        if context.symbol.lower() in explanation_text.lower():
            context_alignment += 0.25
        if context.action.lower() in explanation_text.lower():
            context_alignment += 0.25
        confidence_factors.append(context_alignment)
        
        # Overall decision confidence influence
        confidence_factors.append(context.confidence)
        
        return np.mean(confidence_factors)
    
    def _generate_cache_key(
        self,
        context: ExplanationContext,
        template: PromptTemplate,
        style: ExplanationStyle
    ) -> str:
        """Generate cache key for explanation request"""
        
        # Create deterministic hash of key context elements
        key_data = f"{context.symbol}_{context.action}_{context.confidence:.2f}_{template.value}_{style.value}"
        
        # Include top market features
        sorted_features = sorted(context.market_features.items(), key=lambda x: abs(x[1]), reverse=True)
        feature_hash = "_".join(f"{k}:{v:.2f}" for k, v in sorted_features[:3])
        key_data += f"_{feature_hash}"
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[ExplanationResult]:
        """Get cached explanation result if valid"""
        if cache_key in self.response_cache:
            result, timestamp = self.response_cache[cache_key]
            
            # Check if cache is still valid
            cache_age_minutes = (time.time() - timestamp) / 60
            if cache_age_minutes < self.config.cache_ttl_minutes:
                # Mark as cached and return
                result.cached = True
                return result
            else:
                # Remove expired cache entry
                del self.response_cache[cache_key]
        
        return None
    
    def _cache_result(self, cache_key: str, result: ExplanationResult):
        """Cache explanation result"""
        
        # Limit cache size
        if len(self.response_cache) >= self.config.cache_size:
            # Remove oldest entry
            oldest_key = min(
                self.response_cache.keys(),
                key=lambda k: self.response_cache[k][1]
            )
            del self.response_cache[oldest_key]
        
        self.response_cache[cache_key] = (result, time.time())
    
    def _create_fallback_explanation(
        self,
        context: ExplanationContext,
        style: ExplanationStyle,
        error_reason: str
    ) -> ExplanationResult:
        """Create fallback explanation when LLM generation fails"""
        
        # Generate basic explanation using templates
        if style == ExplanationStyle.TECHNICAL:
            explanation = (
                f"Technical analysis indicates {context.action} signal for {context.symbol} "
                f"with {context.confidence:.1%} confidence based on multi-factor model output."
            )
        elif style == ExplanationStyle.REGULATORY:
            explanation = (
                f"Algorithmic trading decision: {context.action} {context.symbol} "
                f"generated by Multi-Agent Reinforcement Learning system with "
                f"{context.confidence:.1%} confidence at {context.timestamp}."
            )
        else:
            explanation = (
                f"Trading system recommends {context.action} position in {context.symbol} "
                f"based on current market analysis with {context.confidence:.1%} confidence."
            )
        
        return ExplanationResult(
            explanation_text=explanation,
            confidence_score=0.5,
            generation_time_ms=1.0,
            tokens_generated=len(explanation.split()),
            cached=False,
            template_used="fallback",
            style=style,
            context_factors=[f"fallback_due_to_{error_reason}"]
        )
    
    def _update_performance_stats(self, generation_time_ms: float, tokens_generated: int):
        """Update performance statistics"""
        self.performance_stats['total_requests'] += 1
        total_requests = self.performance_stats['total_requests']
        
        # Update average generation time
        old_avg_time = self.performance_stats['avg_generation_time_ms']
        self.performance_stats['avg_generation_time_ms'] = (
            (old_avg_time * (total_requests - 1) + generation_time_ms) / total_requests
        )
        
        # Update average tokens generated
        old_avg_tokens = self.performance_stats['avg_tokens_generated']
        self.performance_stats['avg_tokens_generated'] = (
            (old_avg_tokens * (total_requests - 1) + tokens_generated) / total_requests
        )
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get Ollama engine health status"""
        try:
            if not self.session:
                await self._initialize_session()
            
            # Test Ollama connectivity
            start_time = time.time()
            url = f"{self.config.base_url}/api/tags"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    models_data = await response.json()
                    available_models = [model['name'] for model in models_data.get('models', [])]
                    
                    latency_ms = (time.time() - start_time) * 1000
                    
                    return {
                        'status': 'healthy',
                        'ollama_available': True,
                        'model_available': self.config.model_name in available_models,
                        'available_models': available_models,
                        'api_latency_ms': latency_ms,
                        'performance_stats': self.performance_stats.copy(),
                        'cache_size': len(self.response_cache),
                        'latency_target_met': self.performance_stats['avg_generation_time_ms'] < 100.0
                    }
                else:
                    return {
                        'status': 'unhealthy',
                        'error': f"Ollama API returned status {response.status}",
                        'ollama_available': False
                    }
                    
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'ollama_available': False
            }
    
    async def clear_cache(self):
        """Clear response cache"""
        self.response_cache.clear()
        logger.info("Ollama explanation cache cleared")
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        return {
            'performance_stats': self.performance_stats.copy(),
            'cache_stats': {
                'cache_size': len(self.response_cache),
                'cache_hit_rate': (
                    self.performance_stats['cache_hits'] / 
                    max(1, self.performance_stats['total_requests'])
                ),
                'cache_ttl_minutes': self.config.cache_ttl_minutes
            },
            'latency_analysis': {
                'avg_generation_time_ms': self.performance_stats['avg_generation_time_ms'],
                'target_latency_ms': 100.0,
                'target_met': self.performance_stats['avg_generation_time_ms'] < 100.0,
                'timeout_rate': (
                    self.performance_stats['timeout_count'] / 
                    max(1, self.performance_stats['total_requests'])
                ),
                'error_rate': (
                    self.performance_stats['error_count'] / 
                    max(1, self.performance_stats['total_requests'])
                )
            },
            'model_stats': {
                'avg_tokens_generated': self.performance_stats['avg_tokens_generated'],
                'model_name': self.config.model_name,
                'temperature': self.config.temperature,
                'max_tokens': self.config.max_tokens
            }
        }