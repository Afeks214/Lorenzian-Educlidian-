"""
Embedding Pipeline for Trading Decision Semantic Search
Agent Alpha Mission: High-Quality Financial Text Embeddings

Implements sentence-transformers pipeline with all-mpnet-base-v2 for
high-quality financial text embeddings and semantic similarity optimization.

Target: Real-time embedding generation for <100ms overall latency
"""

import asyncio
import logging
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading

# Sentence transformers with fallback
try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available, using mock embeddings")

logger = logging.getLogger(__name__)


class EmbeddingType(Enum):
    """Types of embeddings for different use cases"""
    TRADING_DECISION = "trading_decision"
    MARKET_CONTEXT = "market_context"
    PERFORMANCE_METRICS = "performance_metrics"
    EXPLANATION_TEXT = "explanation_text"
    RISK_ASSESSMENT = "risk_assessment"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding pipeline"""
    model_name: str = "all-mpnet-base-v2"
    device: str = "auto"  # auto, cpu, cuda
    batch_size: int = 32
    max_seq_length: int = 512
    normalize_embeddings: bool = True
    cache_size: int = 2000
    cache_ttl_minutes: int = 30
    thread_pool_size: int = 4
    enable_preprocessing: bool = True
    similarity_threshold: float = 0.7


@dataclass
class EmbeddingRequest:
    """Request for embedding generation"""
    text: str
    embedding_type: EmbeddingType
    metadata: Optional[Dict[str, Any]] = None
    priority: int = 1  # 1=high, 2=medium, 3=low


@dataclass
class EmbeddingResult:
    """Result from embedding generation"""
    embedding: np.ndarray
    text: str
    embedding_type: EmbeddingType
    generation_time_ms: float
    cached: bool
    model_used: str
    preprocessing_applied: bool
    similarity_score: Optional[float] = None


class EmbeddingPipeline:
    """
    High-performance embedding pipeline for financial text
    
    Provides real-time embedding generation with caching and optimization
    for semantic search in trading decision systems.
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize embedding pipeline"""
        self.config = config or EmbeddingConfig()
        
        # Sentence transformer model
        self.model: Optional[Any] = None
        self.device = None
        
        # Thread pool for async processing
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.thread_pool_size)
        
        # Embedding cache for performance
        self.embedding_cache: Dict[str, Tuple[EmbeddingResult, float]] = {}
        self.cache_lock = threading.RLock()
        
        # Performance tracking
        self.performance_stats = {
            'total_embeddings': 0,
            'cache_hits': 0,
            'avg_generation_time_ms': 0.0,
            'batch_generations': 0,
            'preprocessing_time_ms': 0.0,
            'model_inference_time_ms': 0.0,
            'error_count': 0
        }
        
        # Text preprocessing rules for financial domain
        self.preprocessing_rules = self._initialize_preprocessing_rules()
        
        # Financial text templates
        self.text_templates = self._initialize_text_templates()
        
        # Initialize model
        self._initialize_model()
        
        logger.info(f"EmbeddingPipeline initialized (available: {SENTENCE_TRANSFORMERS_AVAILABLE})")
    
    def _initialize_model(self):
        """Initialize sentence transformer model"""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                # Determine device
                if self.config.device == "auto":
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                else:
                    self.device = self.config.device
                
                # Load model
                self.model = SentenceTransformer(self.config.model_name, device=self.device)
                
                # Configure model settings
                self.model.max_seq_length = self.config.max_seq_length
                
                logger.info(f"Loaded embedding model: {self.config.model_name} on {self.device}")
                
                # Warm up model with dummy text
                self._warmup_model()
                
            else:
                # Mock model for testing
                self.model = MockEmbeddingModel()
                self.device = "cpu"
                logger.warning("Using mock embedding model")
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.model = MockEmbeddingModel()
            self.device = "cpu"
    
    def _warmup_model(self):
        """Warm up model with dummy inference"""
        try:
            dummy_texts = [
                "Trading decision for NQ futures",
                "Market analysis shows bullish momentum",
                "Risk assessment indicates moderate volatility"
            ]
            
            start_time = time.time()
            if SENTENCE_TRANSFORMERS_AVAILABLE and hasattr(self.model, 'encode'):
                _ = self.model.encode(dummy_texts, show_progress_bar=False)
            
            warmup_time = (time.time() - start_time) * 1000
            logger.info(f"Model warmup completed in {warmup_time:.1f}ms")
            
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def _initialize_preprocessing_rules(self) -> Dict[str, Any]:
        """Initialize text preprocessing rules for financial domain"""
        return {
            'financial_abbreviations': {
                'NQ': 'NASDAQ futures',
                'ES': 'S&P 500 futures', 
                'YM': 'Dow Jones futures',
                'RTY': 'Russell 2000 futures',
                'BTC': 'Bitcoin',
                'ETH': 'Ethereum',
                'SPY': 'SPDR S&P 500 ETF',
                'QQQ': 'Invesco NASDAQ ETF'
            },
            'action_standardization': {
                'LONG': 'long position',
                'SHORT': 'short position',
                'BUY': 'buy order',
                'SELL': 'sell order',
                'HOLD': 'hold position'
            },
            'technical_indicators': {
                'RSI': 'Relative Strength Index',
                'MACD': 'Moving Average Convergence Divergence',
                'EMA': 'Exponential Moving Average',
                'SMA': 'Simple Moving Average',
                'BB': 'Bollinger Bands',
                'ATR': 'Average True Range'
            },
            'market_conditions': {
                'VOLATILE': 'high volatility market conditions',
                'TRENDING': 'trending market environment',
                'RANGING': 'range-bound market conditions',
                'LIQUID': 'high liquidity environment'
            }
        }
    
    def _initialize_text_templates(self) -> Dict[EmbeddingType, str]:
        """Initialize text templates for different embedding types"""
        return {
            EmbeddingType.TRADING_DECISION: (
                "Trading decision: {action} {symbol} with {confidence:.1%} confidence. "
                "Market signals: {signals}. Context: {context}"
            ),
            EmbeddingType.MARKET_CONTEXT: (
                "Market analysis for {symbol}: {conditions}. "
                "Technical indicators: {indicators}. Timestamp: {timestamp}"
            ),
            EmbeddingType.PERFORMANCE_METRICS: (
                "Performance analysis: {return_pct:+.2f}% return, {sharpe:.2f} Sharpe ratio, "
                "{drawdown:.2f}% max drawdown, {win_rate:.1f}% win rate over {period}"
            ),
            EmbeddingType.EXPLANATION_TEXT: (
                "Trading explanation: {explanation}. "
                "Decision factors: {factors}. Confidence: {confidence:.1%}"
            ),
            EmbeddingType.RISK_ASSESSMENT: (
                "Risk assessment for {symbol}: {risk_level} risk environment. "
                "Key factors: {risk_factors}. Position impact: {position_impact}"
            )
        }
    
    async def generate_embedding(
        self,
        request: EmbeddingRequest,
        use_cache: bool = True
    ) -> EmbeddingResult:
        """
        Generate embedding for single text
        
        Args:
            request: Embedding request with text and metadata
            use_cache: Whether to use cached embeddings
            
        Returns:
            EmbeddingResult: Generated embedding with metadata
        """
        start_time = time.time()
        
        try:
            # Check cache first
            if use_cache:
                cache_key = self._generate_cache_key(request)
                cached_result = self._get_cached_embedding(cache_key)
                if cached_result:
                    self.performance_stats['cache_hits'] += 1
                    return cached_result
            
            # Preprocess text
            preprocessing_start = time.time()
            processed_text = self._preprocess_text(request.text, request.embedding_type)
            preprocessing_time = (time.time() - preprocessing_start) * 1000
            
            # Generate embedding
            inference_start = time.time()
            embedding = await self._generate_single_embedding(processed_text)
            inference_time = (time.time() - inference_start) * 1000
            
            # Create result
            total_time = (time.time() - start_time) * 1000
            result = EmbeddingResult(
                embedding=embedding,
                text=processed_text,
                embedding_type=request.embedding_type,
                generation_time_ms=total_time,
                cached=False,
                model_used=self.config.model_name,
                preprocessing_applied=self.config.enable_preprocessing
            )
            
            # Cache result
            if use_cache:
                self._cache_embedding(cache_key, result)
            
            # Update performance stats
            self._update_performance_stats(total_time, preprocessing_time, inference_time)
            
            logger.debug(f"Generated embedding in {total_time:.1f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            self.performance_stats['error_count'] += 1
            return self._create_fallback_embedding(request)
    
    async def generate_batch_embeddings(
        self,
        requests: List[EmbeddingRequest],
        use_cache: bool = True
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for batch of texts (optimized for performance)
        
        Args:
            requests: List of embedding requests
            use_cache: Whether to use cached embeddings
            
        Returns:
            List[EmbeddingResult]: Generated embeddings
        """
        start_time = time.time()
        
        try:
            # Separate cached and uncached requests
            cached_results = {}
            uncached_requests = []
            
            if use_cache:
                for i, request in enumerate(requests):
                    cache_key = self._generate_cache_key(request)
                    cached_result = self._get_cached_embedding(cache_key)
                    if cached_result:
                        cached_results[i] = cached_result
                        self.performance_stats['cache_hits'] += 1
                    else:
                        uncached_requests.append((i, request))
            else:
                uncached_requests = list(enumerate(requests))
            
            # Process uncached requests in batches
            uncached_results = {}
            if uncached_requests:
                # Preprocess texts
                processed_texts = []
                for _, request in uncached_requests:
                    processed_text = self._preprocess_text(request.text, request.embedding_type)
                    processed_texts.append(processed_text)
                
                # Generate embeddings in batches
                embeddings = await self._generate_batch_embeddings_internal(processed_texts)
                
                # Create results
                for j, (i, request) in enumerate(uncached_requests):
                    if j < len(embeddings):
                        result = EmbeddingResult(
                            embedding=embeddings[j],
                            text=processed_texts[j],
                            embedding_type=request.embedding_type,
                            generation_time_ms=0.0,  # Will be set below
                            cached=False,
                            model_used=self.config.model_name,
                            preprocessing_applied=self.config.enable_preprocessing
                        )
                        uncached_results[i] = result
                        
                        # Cache result
                        if use_cache:
                            cache_key = self._generate_cache_key(request)
                            self._cache_embedding(cache_key, result)
            
            # Combine cached and uncached results
            results = []
            total_time = (time.time() - start_time) * 1000
            
            for i in range(len(requests)):
                if i in cached_results:
                    results.append(cached_results[i])
                elif i in uncached_results:
                    result = uncached_results[i]
                    result.generation_time_ms = total_time / len(uncached_requests)
                    results.append(result)
                else:
                    # Fallback
                    results.append(self._create_fallback_embedding(requests[i]))
            
            # Update batch stats
            self.performance_stats['batch_generations'] += 1
            
            logger.debug(f"Generated {len(results)} embeddings in {total_time:.1f}ms")
            return results
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            self.performance_stats['error_count'] += 1
            return [self._create_fallback_embedding(req) for req in requests]
    
    def _preprocess_text(self, text: str, embedding_type: EmbeddingType) -> str:
        """Preprocess text for better embedding quality"""
        
        if not self.config.enable_preprocessing:
            return text
        
        processed_text = text.strip()
        
        # Apply financial domain preprocessing
        rules = self.preprocessing_rules
        
        # Expand financial abbreviations
        for abbrev, expansion in rules['financial_abbreviations'].items():
            processed_text = processed_text.replace(abbrev, expansion)
        
        # Standardize action terms
        for action, standard in rules['action_standardization'].items():
            processed_text = processed_text.replace(action, standard)
        
        # Expand technical indicators
        for indicator, expansion in rules['technical_indicators'].items():
            processed_text = processed_text.replace(indicator, expansion)
        
        # Standardize market conditions
        for condition, standard in rules['market_conditions'].items():
            processed_text = processed_text.replace(condition, standard)
        
        # Remove excessive whitespace
        processed_text = ' '.join(processed_text.split())
        
        # Ensure reasonable length
        if len(processed_text) > self.config.max_seq_length:
            processed_text = processed_text[:self.config.max_seq_length-3] + "..."
        
        return processed_text
    
    async def _generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for single text"""
        
        def _encode():
            if SENTENCE_TRANSFORMERS_AVAILABLE and hasattr(self.model, 'encode'):
                return self.model.encode(
                    [text],
                    normalize_embeddings=self.config.normalize_embeddings,
                    show_progress_bar=False
                )[0]
            else:
                return self.model.encode([text])[0]
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(self.thread_pool, _encode)
        
        return embedding
    
    async def _generate_batch_embeddings_internal(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for batch of texts"""
        
        def _encode_batch():
            if SENTENCE_TRANSFORMERS_AVAILABLE and hasattr(self.model, 'encode'):
                return self.model.encode(
                    texts,
                    batch_size=self.config.batch_size,
                    normalize_embeddings=self.config.normalize_embeddings,
                    show_progress_bar=False
                )
            else:
                return self.model.encode(texts)
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(self.thread_pool, _encode_batch)
        
        return embeddings
    
    def calculate_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """
        Calculate similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            metric: Similarity metric (cosine, euclidean, dot)
            
        Returns:
            float: Similarity score
        """
        try:
            if metric == "cosine":
                # Cosine similarity
                dot_product = np.dot(embedding1, embedding2)
                norm1 = np.linalg.norm(embedding1)
                norm2 = np.linalg.norm(embedding2)
                return dot_product / (norm1 * norm2)
                
            elif metric == "euclidean":
                # Euclidean distance (converted to similarity)
                distance = np.linalg.norm(embedding1 - embedding2)
                return 1.0 / (1.0 + distance)
                
            elif metric == "dot":
                # Dot product similarity
                return np.dot(embedding1, embedding2)
                
            else:
                raise ValueError(f"Unknown similarity metric: {metric}")
                
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0
    
    async def find_most_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: List[np.ndarray],
        candidates_metadata: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 5,
        min_similarity: Optional[float] = None
    ) -> List[Tuple[int, float, Optional[Dict[str, Any]]]]:
        """
        Find most similar embeddings to query
        
        Args:
            query_embedding: Query embedding
            candidate_embeddings: List of candidate embeddings
            candidates_metadata: Optional metadata for candidates
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (index, similarity_score, metadata) tuples
        """
        try:
            # Calculate similarities
            similarities = []
            for i, candidate in enumerate(candidate_embeddings):
                similarity = self.calculate_similarity(query_embedding, candidate)
                
                # Apply minimum similarity filter
                if min_similarity is None or similarity >= min_similarity:
                    metadata = candidates_metadata[i] if candidates_metadata and i < len(candidates_metadata) else None
                    similarities.append((i, similarity, metadata))
            
            # Sort by similarity descending
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top k
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to find similar embeddings: {e}")
            return []
    
    def _generate_cache_key(self, request: EmbeddingRequest) -> str:
        """Generate cache key for embedding request"""
        
        # Include text, embedding type, and preprocessing settings
        key_data = f"{request.text}_{request.embedding_type.value}_{self.config.enable_preprocessing}"
        
        # Include relevant config that affects embeddings
        key_data += f"_{self.config.normalize_embeddings}_{self.config.max_seq_length}"
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_embedding(self, cache_key: str) -> Optional[EmbeddingResult]:
        """Get cached embedding result if valid"""
        
        with self.cache_lock:
            if cache_key in self.embedding_cache:
                result, timestamp = self.embedding_cache[cache_key]
                
                # Check if cache is still valid
                cache_age_minutes = (time.time() - timestamp) / 60
                if cache_age_minutes < self.config.cache_ttl_minutes:
                    # Mark as cached and return
                    result.cached = True
                    return result
                else:
                    # Remove expired cache entry
                    del self.embedding_cache[cache_key]
        
        return None
    
    def _cache_embedding(self, cache_key: str, result: EmbeddingResult):
        """Cache embedding result"""
        
        with self.cache_lock:
            # Limit cache size
            if len(self.embedding_cache) >= self.config.cache_size:
                # Remove oldest entry
                oldest_key = min(
                    self.embedding_cache.keys(),
                    key=lambda k: self.embedding_cache[k][1]
                )
                del self.embedding_cache[oldest_key]
            
            self.embedding_cache[cache_key] = (result, time.time())
    
    def _create_fallback_embedding(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Create fallback embedding when generation fails"""
        
        # Create deterministic fallback embedding based on text hash
        text_hash = hash(request.text) % 10000
        np.random.seed(text_hash)
        
        # Generate mock embedding with appropriate dimensions
        if request.embedding_type == EmbeddingType.TRADING_DECISION:
            dim = 768
        elif request.embedding_type == EmbeddingType.PERFORMANCE_METRICS:
            dim = 384
        elif request.embedding_type == EmbeddingType.MARKET_CONTEXT:
            dim = 512
        else:
            dim = 768
        
        fallback_embedding = np.random.normal(0, 1, dim)
        if self.config.normalize_embeddings:
            fallback_embedding = fallback_embedding / np.linalg.norm(fallback_embedding)
        
        return EmbeddingResult(
            embedding=fallback_embedding,
            text=request.text,
            embedding_type=request.embedding_type,
            generation_time_ms=1.0,
            cached=False,
            model_used="fallback",
            preprocessing_applied=False
        )
    
    def _update_performance_stats(
        self,
        total_time_ms: float,
        preprocessing_time_ms: float,
        inference_time_ms: float
    ):
        """Update performance statistics"""
        
        self.performance_stats['total_embeddings'] += 1
        total_embeddings = self.performance_stats['total_embeddings']
        
        # Update average generation time
        old_avg_time = self.performance_stats['avg_generation_time_ms']
        self.performance_stats['avg_generation_time_ms'] = (
            (old_avg_time * (total_embeddings - 1) + total_time_ms) / total_embeddings
        )
        
        # Update preprocessing time
        old_prep_time = self.performance_stats['preprocessing_time_ms']
        self.performance_stats['preprocessing_time_ms'] = (
            (old_prep_time * (total_embeddings - 1) + preprocessing_time_ms) / total_embeddings
        )
        
        # Update inference time
        old_inf_time = self.performance_stats['model_inference_time_ms']
        self.performance_stats['model_inference_time_ms'] = (
            (old_inf_time * (total_embeddings - 1) + inference_time_ms) / total_embeddings
        )
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get embedding pipeline health status"""
        try:
            # Test embedding generation
            test_request = EmbeddingRequest(
                text="Test embedding generation",
                embedding_type=EmbeddingType.TRADING_DECISION
            )
            
            start_time = time.time()
            test_result = await self.generate_embedding(test_request, use_cache=False)
            test_latency_ms = (time.time() - start_time) * 1000
            
            return {
                'status': 'healthy',
                'sentence_transformers_available': SENTENCE_TRANSFORMERS_AVAILABLE,
                'model_loaded': self.model is not None,
                'model_name': self.config.model_name,
                'device': self.device,
                'test_embedding_latency_ms': test_latency_ms,
                'latency_target_met': test_latency_ms < 50.0,  # Embedding target < 50ms
                'performance_stats': self.performance_stats.copy(),
                'cache_size': len(self.embedding_cache),
                'thread_pool_size': self.config.thread_pool_size
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'sentence_transformers_available': SENTENCE_TRANSFORMERS_AVAILABLE
            }
    
    async def clear_cache(self):
        """Clear embedding cache"""
        with self.cache_lock:
            self.embedding_cache.clear()
        logger.info("Embedding pipeline cache cleared")
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        return {
            'performance_stats': self.performance_stats.copy(),
            'cache_stats': {
                'cache_size': len(self.embedding_cache),
                'cache_hit_rate': (
                    self.performance_stats['cache_hits'] / 
                    max(1, self.performance_stats['total_embeddings'])
                ),
                'cache_ttl_minutes': self.config.cache_ttl_minutes
            },
            'latency_analysis': {
                'avg_total_time_ms': self.performance_stats['avg_generation_time_ms'],
                'avg_preprocessing_time_ms': self.performance_stats['preprocessing_time_ms'],
                'avg_inference_time_ms': self.performance_stats['model_inference_time_ms'],
                'target_latency_ms': 50.0,
                'target_met': self.performance_stats['avg_generation_time_ms'] < 50.0
            },
            'model_stats': {
                'model_name': self.config.model_name,
                'device': self.device,
                'batch_size': self.config.batch_size,
                'max_seq_length': self.config.max_seq_length,
                'normalize_embeddings': self.config.normalize_embeddings
            }
        }
    
    def __del__(self):
        """Cleanup thread pool on deletion"""
        try:
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=False)
        except (FileNotFoundError, IOError, OSError) as e:
            logger.error(f'Error occurred: {e}')


# Mock embedding model for testing

class MockEmbeddingModel:
    """Mock embedding model for testing when sentence-transformers not available"""
    
    def encode(self, texts: Union[str, List[str]], normalize_embeddings: bool = True, 
               show_progress_bar: bool = False, batch_size: int = 32) -> np.ndarray:
        """Generate mock embeddings"""
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            # Create deterministic embedding based on text hash
            text_hash = hash(text) % 10000
            np.random.seed(text_hash)
            
            # Generate embedding (default to 768 dimensions)
            embedding = np.random.normal(0, 1, 768)
            
            if normalize_embeddings:
                embedding = embedding / np.linalg.norm(embedding)
            
            embeddings.append(embedding)
        
        return np.array(embeddings)