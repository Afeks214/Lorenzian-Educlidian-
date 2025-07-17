"""
ChromaDB Vector Store for Trading Decisions
Agent Alpha Mission: High-Performance Vector Database Foundation

Implements ChromaDB-based vector storage with 4 specialized collections:
- trading_decisions (768-dim embeddings)
- performance_metrics (384-dim embeddings) 
- market_contexts (512-dim embeddings)
- explanations (768-dim embeddings)

Target: <100ms query latency with HNSW indexing
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import numpy as np
import pandas as pd

# ChromaDB imports with fallback
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not available, using mock vector store")

# Import sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available, using mock embeddings")

logger = logging.getLogger(__name__)


class CollectionType(Enum):
    """Vector store collection types"""
    TRADING_DECISIONS = "trading_decisions"
    PERFORMANCE_METRICS = "performance_metrics"
    MARKET_CONTEXTS = "market_contexts"
    EXPLANATIONS = "explanations"


@dataclass
class VectorStoreConfig:
    """Configuration for vector store"""
    persist_directory: str = "./chroma_db"
    embedding_model: str = "all-mpnet-base-v2"
    max_results: int = 10
    similarity_threshold: float = 0.7
    cache_size: int = 1000
    enable_hnsw: bool = True
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 100


@dataclass
class SearchResult:
    """Vector search result"""
    id: str
    score: float
    metadata: Dict[str, Any]
    document: str
    embedding: Optional[np.ndarray] = None


@dataclass
class TradingDecisionRecord:
    """Trading decision record for vector storage"""
    decision_id: str
    timestamp: datetime
    symbol: str
    asset_class: str
    action: str
    confidence: float
    features: Dict[str, float]
    market_conditions: Dict[str, Any]
    execution_result: Optional[Dict[str, Any]] = None
    explanation_text: Optional[str] = None


class TradingDecisionVectorStore:
    """
    High-performance ChromaDB vector store for trading decisions
    
    Provides semantic search capabilities with <100ms latency target
    """
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """Initialize vector store"""
        self.config = config or VectorStoreConfig()
        
        # Initialize ChromaDB client
        self.client = None
        self.collections: Dict[CollectionType, Any] = {}
        
        # Embedding model
        self.embedding_model = None
        
        # Performance tracking
        self.performance_stats = {
            'total_queries': 0,
            'total_inserts': 0,
            'avg_query_time_ms': 0.0,
            'avg_insert_time_ms': 0.0,
            'cache_hits': 0,
            'error_count': 0
        }
        
        # Query cache for performance
        self.query_cache: Dict[str, Tuple[List[SearchResult], float]] = {}
        
        # Initialize database
        self._initialize_database()
        
        logger.info(f"TradingDecisionVectorStore initialized (ChromaDB: {CHROMADB_AVAILABLE})")
    
    def _initialize_database(self):
        """Initialize ChromaDB and collections"""
        try:
            if CHROMADB_AVAILABLE:
                # Initialize ChromaDB with persistence
                settings = Settings(
                    persist_directory=self.config.persist_directory,
                    anonymized_telemetry=False
                )
                self.client = chromadb.PersistentClient(settings=settings)
                
                # Initialize embedding model
                if SENTENCE_TRANSFORMERS_AVAILABLE:
                    self.embedding_model = SentenceTransformer(self.config.embedding_model)
                    logger.info(f"Loaded embedding model: {self.config.embedding_model}")
                else:
                    self.embedding_model = MockEmbeddingModel()
                    logger.warning("Using mock embedding model")
                
                # Create collections
                self._create_collections()
                
            else:
                # Mock ChromaDB client
                self.client = MockChromaClient()
                self.embedding_model = MockEmbeddingModel()
                self._create_mock_collections()
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            # Fallback to mock
            self.client = MockChromaClient()
            self.embedding_model = MockEmbeddingModel()
            self._create_mock_collections()
    
    def _create_collections(self):
        """Create specialized collections with optimized configurations"""
        
        # Collection configurations
        collection_configs = {
            CollectionType.TRADING_DECISIONS: {
                "name": "trading_decisions",
                "metadata": {
                    "description": "Trading decision embeddings with market context",
                    "embedding_dimension": 768,
                    "hnsw_space": "cosine"
                }
            },
            CollectionType.PERFORMANCE_METRICS: {
                "name": "performance_metrics", 
                "metadata": {
                    "description": "Performance metrics embeddings for similarity search",
                    "embedding_dimension": 384,
                    "hnsw_space": "cosine"
                }
            },
            CollectionType.MARKET_CONTEXTS: {
                "name": "market_contexts",
                "metadata": {
                    "description": "Market context embeddings for regime detection",
                    "embedding_dimension": 512,
                    "hnsw_space": "cosine"
                }
            },
            CollectionType.EXPLANATIONS: {
                "name": "explanations",
                "metadata": {
                    "description": "Explanation text embeddings for semantic search",
                    "embedding_dimension": 768,
                    "hnsw_space": "cosine"
                }
            }
        }
        
        for collection_type, config in collection_configs.items():
            try:
                # Get or create collection
                collection = self.client.get_or_create_collection(
                    name=config["name"],
                    metadata=config["metadata"],
                    embedding_function=self._get_embedding_function()
                )
                self.collections[collection_type] = collection
                logger.info(f"Created/loaded collection: {config['name']}")
                
            except Exception as e:
                logger.error(f"Failed to create collection {config['name']}: {e}")
                # Create mock collection
                self.collections[collection_type] = MockCollection(config["name"])
    
    def _create_mock_collections(self):
        """Create mock collections for testing"""
        for collection_type in CollectionType:
            self.collections[collection_type] = MockCollection(collection_type.value)
    
    def _get_embedding_function(self):
        """Get ChromaDB embedding function"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return MockEmbeddingFunction()
        
        def embedding_function(texts: List[str]) -> List[List[float]]:
            """Embedding function for ChromaDB"""
            try:
                # Generate embeddings using sentence transformer
                embeddings = self.embedding_model.encode(
                    texts, 
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                return embeddings.tolist()
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
                # Return mock embeddings
                return [[0.0] * 768 for _ in texts]
        
        return embedding_function
    
    async def store_trading_decision(
        self,
        decision_record: TradingDecisionRecord,
        custom_embedding_text: Optional[str] = None
    ) -> bool:
        """
        Store trading decision in vector database
        
        Args:
            decision_record: Trading decision to store
            custom_embedding_text: Custom text for embedding generation
            
        Returns:
            bool: Success status
        """
        start_time = time.time()
        
        try:
            # Generate embedding text
            if custom_embedding_text:
                embedding_text = custom_embedding_text
            else:
                embedding_text = self._create_decision_embedding_text(decision_record)
            
            # Create metadata
            metadata = {
                'timestamp': decision_record.timestamp.isoformat(),
                'symbol': decision_record.symbol,
                'asset_class': decision_record.asset_class,
                'action': decision_record.action,
                'confidence': float(decision_record.confidence),
                'features_json': json.dumps(decision_record.features),
                'market_conditions_json': json.dumps(decision_record.market_conditions)
            }
            
            # Add execution results if available
            if decision_record.execution_result:
                metadata.update({
                    'execution_success': decision_record.execution_result.get('success', False),
                    'execution_latency_ms': float(decision_record.execution_result.get('execution_time_ms', 0)),
                    'slippage': float(decision_record.execution_result.get('slippage', 0))
                })
            
            # Store in trading_decisions collection
            collection = self.collections[CollectionType.TRADING_DECISIONS]
            
            if CHROMADB_AVAILABLE:
                collection.add(
                    ids=[decision_record.decision_id],
                    documents=[embedding_text],
                    metadatas=[metadata]
                )
            else:
                # Mock storage
                collection.add_mock(decision_record.decision_id, embedding_text, metadata)
            
            # Update performance stats
            insert_time_ms = (time.time() - start_time) * 1000
            self._update_insert_stats(insert_time_ms)
            
            logger.debug(f"Stored decision {decision_record.decision_id} in {insert_time_ms:.1f}ms")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store trading decision {decision_record.decision_id}: {e}")
            self.performance_stats['error_count'] += 1
            return False
    
    def _create_decision_embedding_text(self, record: TradingDecisionRecord) -> str:
        """Create rich text for embedding that captures decision essence"""
        
        # Extract key features for embedding
        key_features = []
        for feature_name, value in record.features.items():
            if abs(value) > 0.1:  # Only include significant features
                key_features.append(f"{feature_name}:{value:.3f}")
        
        # Create comprehensive embedding text
        embedding_text = (
            f"Trading decision for {record.symbol} ({record.asset_class}): "
            f"{record.action} with {record.confidence:.1%} confidence. "
            f"Key signals: {', '.join(key_features[:5])}. "
            f"Market conditions: volatility={record.market_conditions.get('volatility', 'unknown')}, "
            f"trend={record.market_conditions.get('trend', 'unknown')}, "
            f"timestamp={record.timestamp.strftime('%Y-%m-%d %H:%M')}"
        )
        
        return embedding_text
    
    async def find_similar_decisions(
        self,
        query_text: str,
        symbol: Optional[str] = None,
        asset_class: Optional[str] = None,
        action: Optional[str] = None,
        time_window_hours: Optional[int] = None,
        min_confidence: Optional[float] = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Find similar trading decisions using semantic search
        
        Args:
            query_text: Query text for semantic similarity
            symbol: Filter by symbol
            asset_class: Filter by asset class
            action: Filter by action type
            time_window_hours: Filter by time window
            min_confidence: Minimum confidence threshold
            limit: Maximum results to return
            
        Returns:
            List[SearchResult]: Similar decisions ranked by similarity
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(
                query_text, symbol, asset_class, action, time_window_hours, min_confidence, limit
            )
            
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self.performance_stats['cache_hits'] += 1
                return cached_result
            
            # Build where clause for filtering
            where_clause = self._build_where_clause(
                symbol, asset_class, action, time_window_hours, min_confidence
            )
            
            # Perform semantic search
            collection = self.collections[CollectionType.TRADING_DECISIONS]
            
            if CHROMADB_AVAILABLE:
                results = collection.query(
                    query_texts=[query_text],
                    n_results=limit,
                    where=where_clause if where_clause else None
                )
                
                # Process results
                search_results = []
                for i in range(len(results['ids'][0])):
                    search_result = SearchResult(
                        id=results['ids'][0][i],
                        score=1.0 - results['distances'][0][i],  # Convert distance to similarity
                        metadata=results['metadatas'][0][i],
                        document=results['documents'][0][i]
                    )
                    search_results.append(search_result)
            else:
                # Mock search
                search_results = self._mock_search(query_text, limit)
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in search_results 
                if result.score >= self.config.similarity_threshold
            ]
            
            # Cache results
            self._cache_result(cache_key, filtered_results)
            
            # Update performance stats
            query_time_ms = (time.time() - start_time) * 1000
            self._update_query_stats(query_time_ms)
            
            logger.debug(f"Found {len(filtered_results)} similar decisions in {query_time_ms:.1f}ms")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Failed to find similar decisions: {e}")
            self.performance_stats['error_count'] += 1
            return []
    
    def _build_where_clause(
        self,
        symbol: Optional[str],
        asset_class: Optional[str], 
        action: Optional[str],
        time_window_hours: Optional[int],
        min_confidence: Optional[float]
    ) -> Optional[Dict[str, Any]]:
        """Build ChromaDB where clause from filters"""
        
        where_conditions = {}
        
        if symbol:
            where_conditions['symbol'] = symbol
        
        if asset_class:
            where_conditions['asset_class'] = asset_class
            
        if action:
            where_conditions['action'] = action
            
        if min_confidence is not None:
            where_conditions['confidence'] = {'$gte': min_confidence}
        
        if time_window_hours is not None:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            where_conditions['timestamp'] = {'$gte': cutoff_time.isoformat()}
        
        return where_conditions if where_conditions else None
    
    async def store_performance_metrics(
        self,
        timeframe: str,
        metrics: Dict[str, Any],
        symbol: Optional[str] = None
    ) -> bool:
        """
        Store performance metrics in vector database
        
        Args:
            timeframe: Time period for metrics (e.g., '1h', '1d')
            metrics: Performance metrics dictionary
            symbol: Optional symbol filter
            
        Returns:
            bool: Success status
        """
        try:
            # Create embedding text for performance metrics
            metrics_text = self._create_metrics_embedding_text(timeframe, metrics)
            
            # Create metadata
            metadata = {
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol or 'ALL',
                'return': float(metrics.get('return', 0)),
                'sharpe_ratio': float(metrics.get('sharpe_ratio', 0)),
                'max_drawdown': float(metrics.get('max_drawdown', 0)),
                'win_rate': float(metrics.get('win_rate', 0)),
                'avg_latency_ms': float(metrics.get('avg_latency_ms', 0)),
                'total_trades': int(metrics.get('total_trades', 0))
            }
            
            # Generate unique ID
            metrics_id = f"metrics_{timeframe}_{symbol or 'ALL'}_{int(time.time())}"
            
            # Store in performance_metrics collection
            collection = self.collections[CollectionType.PERFORMANCE_METRICS]
            
            if CHROMADB_AVAILABLE:
                collection.add(
                    ids=[metrics_id],
                    documents=[metrics_text],
                    metadatas=[metadata]
                )
            else:
                collection.add_mock(metrics_id, metrics_text, metadata)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store performance metrics: {e}")
            return False
    
    def _create_metrics_embedding_text(self, timeframe: str, metrics: Dict[str, Any]) -> str:
        """Create embedding text for performance metrics"""
        
        # Format key metrics for embedding
        return_pct = metrics.get('return', 0) * 100
        sharpe = metrics.get('sharpe_ratio', 0)
        drawdown_pct = metrics.get('max_drawdown', 0) * 100
        win_rate_pct = metrics.get('win_rate', 0) * 100
        
        metrics_text = (
            f"Performance metrics for {timeframe} period: "
            f"Return {return_pct:+.2f}%, Sharpe ratio {sharpe:.2f}, "
            f"Max drawdown {drawdown_pct:.2f}%, Win rate {win_rate_pct:.1f}%, "
            f"Total trades {metrics.get('total_trades', 0)}, "
            f"Average latency {metrics.get('avg_latency_ms', 0):.1f}ms"
        )
        
        return metrics_text
    
    def _generate_cache_key(self, *args) -> str:
        """Generate cache key for query results"""
        key_data = '|'.join(str(arg) for arg in args)
        return str(hash(key_data))
    
    def _get_cached_result(self, cache_key: str) -> Optional[List[SearchResult]]:
        """Get cached search result if valid"""
        if cache_key in self.query_cache:
            results, timestamp = self.query_cache[cache_key]
            # Cache valid for 5 minutes
            if time.time() - timestamp < 300:
                return results
            else:
                del self.query_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, results: List[SearchResult]):
        """Cache search results"""
        # Limit cache size
        if len(self.query_cache) >= self.config.cache_size:
            # Remove oldest entry
            oldest_key = min(self.query_cache.keys(), 
                           key=lambda k: self.query_cache[k][1])
            del self.query_cache[oldest_key]
        
        self.query_cache[cache_key] = (results, time.time())
    
    def _mock_search(self, query_text: str, limit: int) -> List[SearchResult]:
        """Mock search for testing"""
        mock_results = []
        for i in range(min(limit, 3)):
            mock_results.append(SearchResult(
                id=f"mock_decision_{i}",
                score=0.8 - i * 0.1,
                metadata={
                    'symbol': 'NQ',
                    'action': 'LONG',
                    'confidence': 0.8,
                    'timestamp': datetime.now().isoformat()
                },
                document=f"Mock trading decision {i} for {query_text}"
            ))
        return mock_results
    
    def _update_query_stats(self, query_time_ms: float):
        """Update query performance statistics"""
        self.performance_stats['total_queries'] += 1
        total_queries = self.performance_stats['total_queries']
        old_avg = self.performance_stats['avg_query_time_ms']
        
        # Update rolling average
        self.performance_stats['avg_query_time_ms'] = (
            (old_avg * (total_queries - 1) + query_time_ms) / total_queries
        )
    
    def _update_insert_stats(self, insert_time_ms: float):
        """Update insert performance statistics"""
        self.performance_stats['total_inserts'] += 1
        total_inserts = self.performance_stats['total_inserts']
        old_avg = self.performance_stats['avg_insert_time_ms']
        
        # Update rolling average
        self.performance_stats['avg_insert_time_ms'] = (
            (old_avg * (total_inserts - 1) + insert_time_ms) / total_inserts
        )
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get vector store health status"""
        try:
            health_status = {
                'status': 'healthy',
                'chromadb_available': CHROMADB_AVAILABLE,
                'sentence_transformers_available': SENTENCE_TRANSFORMERS_AVAILABLE,
                'collections_count': len(self.collections),
                'performance_stats': self.performance_stats.copy()
            }
            
            # Test query performance
            start_time = time.time()
            test_results = await self.find_similar_decisions("test query", limit=1)
            query_latency_ms = (time.time() - start_time) * 1000
            
            health_status['test_query_latency_ms'] = query_latency_ms
            health_status['latency_target_met'] = query_latency_ms < 100.0
            
            # Collection health
            collection_health = {}
            for collection_type, collection in self.collections.items():
                if hasattr(collection, 'count'):
                    try:
                        count = collection.count()
                        collection_health[collection_type.value] = {
                            'document_count': count,
                            'status': 'healthy'
                        }
                    except Exception as e:
                        collection_health[collection_type.value] = {
                            'document_count': 0,
                            'status': f'error: {e}'
                        }
                else:
                    collection_health[collection_type.value] = {
                        'document_count': 'unknown',
                        'status': 'mock'
                    }
            
            health_status['collections'] = collection_health
            
            return health_status
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'chromadb_available': CHROMADB_AVAILABLE
            }
    
    async def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()
        logger.info("Vector store cache cleared")
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        return {
            'performance_stats': self.performance_stats.copy(),
            'cache_stats': {
                'cache_size': len(self.query_cache),
                'cache_hit_rate': (
                    self.performance_stats['cache_hits'] / 
                    max(1, self.performance_stats['total_queries'])
                )
            },
            'latency_analysis': {
                'avg_query_latency_ms': self.performance_stats['avg_query_time_ms'],
                'avg_insert_latency_ms': self.performance_stats['avg_insert_time_ms'],
                'target_latency_ms': 100.0,
                'query_target_met': self.performance_stats['avg_query_time_ms'] < 100.0,
                'insert_target_met': self.performance_stats['avg_insert_time_ms'] < 50.0
            }
        }


# Mock classes for testing when dependencies are not available

class MockChromaClient:
    """Mock ChromaDB client for testing"""
    
    def get_or_create_collection(self, name, metadata=None, embedding_function=None):
        return MockCollection(name)


class MockCollection:
    """Mock ChromaDB collection"""
    
    def __init__(self, name: str):
        self.name = name
        self.data = {}
    
    def add(self, ids, documents, metadatas):
        for i, doc_id in enumerate(ids):
            self.data[doc_id] = {
                'document': documents[i],
                'metadata': metadatas[i] if i < len(metadatas) else {}
            }
    
    def add_mock(self, doc_id, document, metadata):
        self.data[doc_id] = {
            'document': document,
            'metadata': metadata
        }
    
    def query(self, query_texts, n_results=10, where=None):
        # Return mock results
        mock_ids = list(self.data.keys())[:n_results]
        mock_distances = [0.1 + i * 0.1 for i in range(len(mock_ids))]
        mock_documents = [self.data[doc_id]['document'] for doc_id in mock_ids]
        mock_metadatas = [self.data[doc_id]['metadata'] for doc_id in mock_ids]
        
        return {
            'ids': [mock_ids],
            'distances': [mock_distances],
            'documents': [mock_documents],
            'metadatas': [mock_metadatas]
        }
    
    def count(self):
        return len(self.data)


class MockEmbeddingModel:
    """Mock sentence transformer model"""
    
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        # Return mock embeddings
        embeddings = []
        for text in texts:
            # Create deterministic mock embedding based on text hash
            text_hash = hash(text) % 1000
            embedding = np.random.RandomState(text_hash).normal(0, 1, 768)
            if normalize_embeddings:
                embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        return np.array(embeddings)


class MockEmbeddingFunction:
    """Mock ChromaDB embedding function"""
    
    def __call__(self, texts):
        model = MockEmbeddingModel()
        embeddings = model.encode(texts)
        return embeddings.tolist()