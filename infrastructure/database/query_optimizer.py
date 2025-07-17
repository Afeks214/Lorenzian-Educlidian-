#!/usr/bin/env python3
"""
Intelligent Query Optimizer for High-Frequency Trading Database
AGENT 14: DATABASE OPTIMIZATION SPECIALIST
Focus: Sub-millisecond query optimization and performance monitoring
"""

import asyncio
import asyncpg
import psycopg2
import time
import logging
import json
import hashlib
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from prometheus_client import Counter, Histogram, Gauge
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from sqlparse import parse, format as sql_format
from sqlparse.sql import Statement, Token
from sqlparse.tokens import Keyword, Name

@dataclass
class QueryMetrics:
    """Metrics for query performance"""
    query_hash: str
    query_text: str
    execution_count: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    rows_examined: int
    rows_returned: int
    index_usage: Dict[str, int]
    plan_hash: str
    last_execution: datetime
    error_count: int
    cache_hit_rate: float

@dataclass
class IndexRecommendation:
    """Index recommendation for query optimization"""
    table_name: str
    columns: List[str]
    index_type: str
    estimated_improvement: float
    query_count: int
    cost_benefit_ratio: float
    priority: str

@dataclass
class QueryPlan:
    """Query execution plan analysis"""
    plan_hash: str
    plan_text: str
    cost: float
    rows: int
    width: int
    startup_cost: float
    total_cost: float
    execution_time_ms: float
    buffer_usage: Dict[str, int]
    index_usage: List[str]
    table_scans: List[str]
    joins: List[str]
    sorts: List[str]
    aggregations: List[str]

class IntelligentQueryOptimizer:
    """
    Intelligent query optimizer with machine learning-based recommendations
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.logger = self._setup_logging()
        self.connection_pool = None
        self.is_running = False
        
        # Query tracking
        self.query_metrics = {}
        self.query_cache = {}
        self.slow_queries = deque(maxlen=1000)
        self.query_patterns = defaultdict(list)
        
        # Plan cache
        self.plan_cache = {}
        self.plan_recommendations = {}
        
        # Index recommendations
        self.index_recommendations = {}
        self.index_usage_stats = defaultdict(int)
        
        # Performance monitoring
        self.execution_times = defaultdict(list)
        self.query_frequency = defaultdict(int)
        
        # Setup Prometheus metrics
        self.setup_prometheus_metrics()
        
        # Local SQLite for query analytics
        self.analytics_db = self._setup_analytics_db()
        
        # Background optimization thread
        self.optimization_executor = ThreadPoolExecutor(max_workers=2)
        
    def _default_config(self) -> Dict:
        """Default configuration for query optimizer"""
        return {
            "database": {
                "host": "127.0.0.1",
                "port": 6432,
                "database": "grandmodel",
                "user": "grandmodel_user",
                "password": "grandmodel_password",
                "min_connections": 10,
                "max_connections": 50
            },
            "optimization": {
                "slow_query_threshold_ms": 10.0,
                "plan_cache_size": 10000,
                "query_cache_size": 5000,
                "auto_explain_threshold_ms": 50.0,
                "index_recommendation_threshold": 100,
                "optimization_interval_seconds": 60
            },
            "monitoring": {
                "metrics_retention_hours": 24,
                "alert_threshold_ms": 100.0,
                "error_rate_threshold": 0.05
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for query optimizer"""
        logger = logging.getLogger('query_optimizer')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def setup_prometheus_metrics(self):
        """Setup Prometheus metrics for query monitoring"""
        self.query_execution_time = Histogram(
            'db_query_execution_time_seconds',
            'Query execution time',
            ['query_type', 'table', 'optimization_level'],
            buckets=[0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        )
        
        self.query_count = Counter(
            'db_query_count_total',
            'Total number of queries executed',
            ['query_type', 'table', 'status']
        )
        
        self.slow_query_count = Counter(
            'db_slow_query_count_total',
            'Number of slow queries',
            ['query_type', 'table']
        )
        
        self.cache_hit_rate = Gauge(
            'db_query_cache_hit_rate',
            'Query cache hit rate',
            ['cache_type']
        )
        
        self.index_usage = Counter(
            'db_index_usage_total',
            'Index usage statistics',
            ['table', 'index_name', 'usage_type']
        )
        
        self.optimization_recommendations = Gauge(
            'db_optimization_recommendations_total',
            'Number of optimization recommendations',
            ['recommendation_type', 'priority']
        )
    
    def _setup_analytics_db(self) -> sqlite3.Connection:
        """Setup local SQLite database for query analytics"""
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        
        # Create tables for query analytics
        cursor.execute("""
            CREATE TABLE query_metrics (
                query_hash TEXT PRIMARY KEY,
                query_text TEXT,
                execution_count INTEGER,
                total_time_ms REAL,
                avg_time_ms REAL,
                min_time_ms REAL,
                max_time_ms REAL,
                p95_time_ms REAL,
                p99_time_ms REAL,
                rows_examined INTEGER,
                rows_returned INTEGER,
                plan_hash TEXT,
                last_execution TEXT,
                error_count INTEGER,
                cache_hit_rate REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE execution_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash TEXT,
                execution_time_ms REAL,
                rows_examined INTEGER,
                rows_returned INTEGER,
                plan_hash TEXT,
                timestamp TEXT,
                error_message TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE index_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_name TEXT,
                columns TEXT,
                index_type TEXT,
                estimated_improvement REAL,
                query_count INTEGER,
                cost_benefit_ratio REAL,
                priority TEXT,
                created_at TEXT
            )
        """)
        
        conn.commit()
        return conn
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for pattern matching"""
        # Remove comments and normalize whitespace
        query = re.sub(r'--.*$', '', query, flags=re.MULTILINE)
        query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)
        query = re.sub(r'\s+', ' ', query).strip()
        
        # Replace parameter placeholders
        query = re.sub(r'\$\d+', '?', query)
        query = re.sub(r"'[^']*'", "'?'", query)
        query = re.sub(r'\b\d+\b', '?', query)
        
        return query.upper()
    
    def _get_query_hash(self, query: str) -> str:
        """Generate hash for query identification"""
        normalized = self._normalize_query(query)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _classify_query(self, query: str) -> str:
        """Classify query type"""
        query_upper = query.upper().strip()
        
        if query_upper.startswith('SELECT'):
            return 'SELECT'
        elif query_upper.startswith('INSERT'):
            return 'INSERT'
        elif query_upper.startswith('UPDATE'):
            return 'UPDATE'
        elif query_upper.startswith('DELETE'):
            return 'DELETE'
        elif query_upper.startswith('CREATE'):
            return 'CREATE'
        elif query_upper.startswith('DROP'):
            return 'DROP'
        elif query_upper.startswith('ALTER'):
            return 'ALTER'
        else:
            return 'OTHER'
    
    def _extract_tables(self, query: str) -> List[str]:
        """Extract table names from query"""
        tables = []
        
        try:
            parsed = parse(query)[0]
            
            def extract_from_token(token):
                if token.ttype is Name:
                    tables.append(str(token).strip())
                elif hasattr(token, 'tokens'):
                    for subtoken in token.tokens:
                        extract_from_token(subtoken)
            
            extract_from_token(parsed)
            
        except Exception as e:
            self.logger.warning(f"Failed to parse query for table extraction: {e}")
        
        return list(set(tables))
    
    async def create_connection_pool(self):
        """Create optimized connection pool"""
        try:
            config = self.config['database']
            
            self.connection_pool = await asyncpg.create_pool(
                host=config['host'],
                port=config['port'],
                database=config['database'],
                user=config['user'],
                password=config['password'],
                min_size=config['min_connections'],
                max_size=config['max_connections'],
                command_timeout=30,
                server_settings={
                    'application_name': 'query_optimizer',
                    'log_statement': 'all',
                    'log_min_duration_statement': '0',
                    'auto_explain.log_min_duration': '10ms',
                    'auto_explain.log_analyze': 'true',
                    'auto_explain.log_buffers': 'true',
                    'auto_explain.log_timing': 'true',
                    'auto_explain.log_triggers': 'true',
                    'auto_explain.log_verbose': 'true'
                }
            )
            
            self.logger.info("Query optimizer connection pool created")
            
        except Exception as e:
            self.logger.error(f"Failed to create connection pool: {e}")
            raise
    
    async def analyze_query_plan(self, query: str, params: tuple = None) -> QueryPlan:
        """Analyze query execution plan"""
        try:
            async with self.connection_pool.acquire() as conn:
                # Get execution plan
                explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"
                
                if params:
                    plan_result = await conn.fetch(explain_query, *params)
                else:
                    plan_result = await conn.fetch(explain_query)
                
                plan_data = plan_result[0]['QUERY PLAN'][0]
                
                # Extract plan information
                plan_hash = hashlib.md5(str(plan_data).encode()).hexdigest()
                
                return QueryPlan(
                    plan_hash=plan_hash,
                    plan_text=json.dumps(plan_data, indent=2),
                    cost=plan_data.get('Total Cost', 0),
                    rows=plan_data.get('Plan Rows', 0),
                    width=plan_data.get('Plan Width', 0),
                    startup_cost=plan_data.get('Startup Cost', 0),
                    total_cost=plan_data.get('Total Cost', 0),
                    execution_time_ms=plan_data.get('Actual Total Time', 0),
                    buffer_usage=self._extract_buffer_usage(plan_data),
                    index_usage=self._extract_index_usage(plan_data),
                    table_scans=self._extract_table_scans(plan_data),
                    joins=self._extract_joins(plan_data),
                    sorts=self._extract_sorts(plan_data),
                    aggregations=self._extract_aggregations(plan_data)
                )
                
        except Exception as e:
            self.logger.error(f"Failed to analyze query plan: {e}")
            return None
    
    def _extract_buffer_usage(self, plan_data: Dict) -> Dict[str, int]:
        """Extract buffer usage from plan"""
        buffer_usage = {}
        
        def traverse_plan(node):
            if 'Shared Hit Blocks' in node:
                buffer_usage['shared_hit'] = buffer_usage.get('shared_hit', 0) + node['Shared Hit Blocks']
            if 'Shared Read Blocks' in node:
                buffer_usage['shared_read'] = buffer_usage.get('shared_read', 0) + node['Shared Read Blocks']
            if 'Shared Dirtied Blocks' in node:
                buffer_usage['shared_dirtied'] = buffer_usage.get('shared_dirtied', 0) + node['Shared Dirtied Blocks']
            
            if 'Plans' in node:
                for child in node['Plans']:
                    traverse_plan(child)
        
        traverse_plan(plan_data)
        return buffer_usage
    
    def _extract_index_usage(self, plan_data: Dict) -> List[str]:
        """Extract index usage from plan"""
        indexes = []
        
        def traverse_plan(node):
            node_type = node.get('Node Type', '')
            if 'Index' in node_type:
                index_name = node.get('Index Name', '')
                if index_name:
                    indexes.append(index_name)
            
            if 'Plans' in node:
                for child in node['Plans']:
                    traverse_plan(child)
        
        traverse_plan(plan_data)
        return indexes
    
    def _extract_table_scans(self, plan_data: Dict) -> List[str]:
        """Extract table scans from plan"""
        scans = []
        
        def traverse_plan(node):
            node_type = node.get('Node Type', '')
            if node_type == 'Seq Scan':
                relation = node.get('Relation Name', '')
                if relation:
                    scans.append(relation)
            
            if 'Plans' in node:
                for child in node['Plans']:
                    traverse_plan(child)
        
        traverse_plan(plan_data)
        return scans
    
    def _extract_joins(self, plan_data: Dict) -> List[str]:
        """Extract joins from plan"""
        joins = []
        
        def traverse_plan(node):
            node_type = node.get('Node Type', '')
            if 'Join' in node_type:
                joins.append(node_type)
            
            if 'Plans' in node:
                for child in node['Plans']:
                    traverse_plan(child)
        
        traverse_plan(plan_data)
        return joins
    
    def _extract_sorts(self, plan_data: Dict) -> List[str]:
        """Extract sorts from plan"""
        sorts = []
        
        def traverse_plan(node):
            node_type = node.get('Node Type', '')
            if node_type == 'Sort':
                sort_key = node.get('Sort Key', [])
                sorts.extend(sort_key)
            
            if 'Plans' in node:
                for child in node['Plans']:
                    traverse_plan(child)
        
        traverse_plan(plan_data)
        return sorts
    
    def _extract_aggregations(self, plan_data: Dict) -> List[str]:
        """Extract aggregations from plan"""
        aggregations = []
        
        def traverse_plan(node):
            node_type = node.get('Node Type', '')
            if 'Aggregate' in node_type or 'Group' in node_type:
                aggregations.append(node_type)
            
            if 'Plans' in node:
                for child in node['Plans']:
                    traverse_plan(child)
        
        traverse_plan(plan_data)
        return aggregations
    
    async def execute_optimized_query(self, query: str, params: tuple = None) -> Tuple[Any, QueryMetrics]:
        """Execute query with optimization and monitoring"""
        start_time = time.time()
        query_hash = self._get_query_hash(query)
        query_type = self._classify_query(query)
        tables = self._extract_tables(query)
        
        try:
            # Check query cache
            cache_key = (query_hash, str(params) if params else '')
            if cache_key in self.query_cache:
                cached_result = self.query_cache[cache_key]
                if time.time() - cached_result['timestamp'] < 60:  # 1 minute cache
                    self.cache_hit_rate.labels(cache_type='query').inc()
                    return cached_result['result'], cached_result['metrics']
            
            # Execute query
            async with self.connection_pool.acquire() as conn:
                if params:
                    result = await conn.fetch(query, *params)
                else:
                    result = await conn.fetch(query)
            
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Update metrics
            if query_hash not in self.query_metrics:
                self.query_metrics[query_hash] = {
                    'query_text': query,
                    'execution_count': 0,
                    'total_time_ms': 0,
                    'times': [],
                    'error_count': 0
                }
            
            metrics = self.query_metrics[query_hash]
            metrics['execution_count'] += 1
            metrics['total_time_ms'] += execution_time
            metrics['times'].append(execution_time)
            
            # Keep only recent execution times
            if len(metrics['times']) > 100:
                metrics['times'] = metrics['times'][-100:]
            
            # Check for slow queries
            if execution_time > self.config['optimization']['slow_query_threshold_ms']:
                self.slow_queries.append({
                    'query_hash': query_hash,
                    'query': query,
                    'execution_time': execution_time,
                    'timestamp': datetime.now()
                })
                
                self.slow_query_count.labels(
                    query_type=query_type,
                    table=tables[0] if tables else 'unknown'
                ).inc()
                
                # Analyze plan for slow queries
                if execution_time > self.config['optimization']['auto_explain_threshold_ms']:
                    plan = await self.analyze_query_plan(query, params)
                    if plan:
                        self.plan_cache[query_hash] = plan
            
            # Update Prometheus metrics
            self.query_execution_time.labels(
                query_type=query_type,
                table=tables[0] if tables else 'unknown',
                optimization_level='standard'
            ).observe(execution_time / 1000)
            
            self.query_count.labels(
                query_type=query_type,
                table=tables[0] if tables else 'unknown',
                status='success'
            ).inc()
            
            # Cache result
            query_metrics = QueryMetrics(
                query_hash=query_hash,
                query_text=query,
                execution_count=metrics['execution_count'],
                total_time_ms=metrics['total_time_ms'],
                avg_time_ms=metrics['total_time_ms'] / metrics['execution_count'],
                min_time_ms=min(metrics['times']),
                max_time_ms=max(metrics['times']),
                p95_time_ms=self._percentile(metrics['times'], 0.95),
                p99_time_ms=self._percentile(metrics['times'], 0.99),
                rows_examined=0,  # Would need to extract from plan
                rows_returned=len(result),
                index_usage={},
                plan_hash=self.plan_cache.get(query_hash, {}).get('plan_hash', ''),
                last_execution=datetime.now(),
                error_count=metrics['error_count'],
                cache_hit_rate=0.0
            )
            
            self.query_cache[cache_key] = {
                'result': result,
                'metrics': query_metrics,
                'timestamp': time.time()
            }
            
            return result, query_metrics
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            # Update error metrics
            if query_hash in self.query_metrics:
                self.query_metrics[query_hash]['error_count'] += 1
            
            self.query_count.labels(
                query_type=query_type,
                table=tables[0] if tables else 'unknown',
                status='error'
            ).inc()
            
            self.logger.error(f"Query execution failed: {e}")
            raise
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    async def generate_index_recommendations(self) -> List[IndexRecommendation]:
        """Generate index recommendations based on query patterns"""
        recommendations = []
        
        # Analyze slow queries for index opportunities
        table_column_patterns = defaultdict(lambda: defaultdict(int))
        
        for slow_query in self.slow_queries:
            query_hash = slow_query['query_hash']
            query_text = slow_query['query']
            
            # Extract WHERE clause patterns
            where_patterns = self._extract_where_patterns(query_text)
            tables = self._extract_tables(query_text)
            
            for table in tables:
                for pattern in where_patterns:
                    table_column_patterns[table][pattern] += 1
        
        # Generate recommendations
        for table, column_patterns in table_column_patterns.items():
            for pattern, count in column_patterns.items():
                if count >= self.config['optimization']['index_recommendation_threshold']:
                    columns = self._parse_column_pattern(pattern)
                    if columns:
                        recommendation = IndexRecommendation(
                            table_name=table,
                            columns=columns,
                            index_type='btree',
                            estimated_improvement=self._estimate_improvement(table, columns, count),
                            query_count=count,
                            cost_benefit_ratio=self._calculate_cost_benefit(table, columns, count),
                            priority=self._calculate_priority(count, table)
                        )
                        recommendations.append(recommendation)
        
        return recommendations
    
    def _extract_where_patterns(self, query: str) -> List[str]:
        """Extract WHERE clause patterns from query"""
        patterns = []
        
        # Simple pattern extraction (would need more sophisticated parsing)
        where_match = re.search(r'WHERE\s+(.+?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|\s+LIMIT|\s*$)', query, re.IGNORECASE)
        if where_match:
            where_clause = where_match.group(1)
            
            # Extract column = value patterns
            column_patterns = re.findall(r'(\w+)\s*=\s*', where_clause)
            patterns.extend(column_patterns)
            
            # Extract column IN (...) patterns
            in_patterns = re.findall(r'(\w+)\s+IN\s*\(', where_clause)
            patterns.extend(in_patterns)
        
        return patterns
    
    def _parse_column_pattern(self, pattern: str) -> List[str]:
        """Parse column pattern into column list"""
        return [pattern.strip()] if pattern.strip() else []
    
    def _estimate_improvement(self, table: str, columns: List[str], query_count: int) -> float:
        """Estimate performance improvement from index"""
        # Simple estimation based on query frequency
        base_improvement = min(query_count / 100, 10.0)  # Cap at 10x improvement
        column_factor = 1.0 + (len(columns) - 1) * 0.1  # Multi-column bonus
        return base_improvement * column_factor
    
    def _calculate_cost_benefit(self, table: str, columns: List[str], query_count: int) -> float:
        """Calculate cost-benefit ratio for index"""
        # Simplified calculation
        benefit = query_count * 0.1  # Assume 0.1ms improvement per query
        cost = len(columns) * 10  # Assume 10 units cost per column
        return benefit / max(cost, 1)
    
    def _calculate_priority(self, query_count: int, table: str) -> str:
        """Calculate priority for index recommendation"""
        if query_count > 1000:
            return 'HIGH'
        elif query_count > 500:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    async def optimize_queries(self):
        """Background query optimization"""
        while self.is_running:
            try:
                # Generate index recommendations
                recommendations = await self.generate_index_recommendations()
                
                # Update recommendations
                self.index_recommendations = {
                    rec.table_name + '_' + '_'.join(rec.columns): rec
                    for rec in recommendations
                }
                
                # Update Prometheus metrics
                for rec in recommendations:
                    self.optimization_recommendations.labels(
                        recommendation_type='index',
                        priority=rec.priority
                    ).set(1)
                
                # Log recommendations
                if recommendations:
                    self.logger.info(f"Generated {len(recommendations)} index recommendations")
                    for rec in recommendations[:5]:  # Log top 5
                        self.logger.info(
                            f"Index recommendation: {rec.table_name}({', '.join(rec.columns)}) "
                            f"- Priority: {rec.priority}, Queries: {rec.query_count}"
                        )
                
                await asyncio.sleep(self.config['optimization']['optimization_interval_seconds'])
                
            except Exception as e:
                self.logger.error(f"Optimization error: {e}")
                await asyncio.sleep(60)
    
    async def start_optimizer(self):
        """Start the query optimizer"""
        self.is_running = True
        
        # Create connection pool
        await self.create_connection_pool()
        
        # Start optimization background task
        optimization_task = asyncio.create_task(self.optimize_queries())
        
        self.logger.info("Query optimizer started")
        
        try:
            await optimization_task
        except Exception as e:
            self.logger.error(f"Optimizer error: {e}")
        finally:
            self.stop_optimizer()
    
    def stop_optimizer(self):
        """Stop the query optimizer"""
        self.is_running = False
        
        if self.connection_pool:
            self.connection_pool.close()
        
        if self.analytics_db:
            self.analytics_db.close()
        
        self.optimization_executor.shutdown(wait=True)
        
        self.logger.info("Query optimizer stopped")
    
    def get_performance_report(self) -> Dict:
        """Generate performance report"""
        total_queries = sum(m['execution_count'] for m in self.query_metrics.values())
        total_errors = sum(m['error_count'] for m in self.query_metrics.values())
        
        # Top slow queries
        slow_queries = []
        for query_hash, metrics in self.query_metrics.items():
            if metrics['times']:
                avg_time = sum(metrics['times']) / len(metrics['times'])
                slow_queries.append({
                    'query_hash': query_hash,
                    'query_text': metrics['query_text'][:200],
                    'avg_time_ms': avg_time,
                    'execution_count': metrics['execution_count']
                })
        
        slow_queries.sort(key=lambda x: x['avg_time_ms'], reverse=True)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_queries': total_queries,
            'total_errors': total_errors,
            'error_rate': total_errors / max(total_queries, 1),
            'slow_queries_count': len(self.slow_queries),
            'cache_size': len(self.query_cache),
            'plan_cache_size': len(self.plan_cache),
            'index_recommendations_count': len(self.index_recommendations),
            'top_slow_queries': slow_queries[:10],
            'index_recommendations': [asdict(rec) for rec in self.index_recommendations.values()]
        }

async def main():
    """Main entry point"""
    optimizer = IntelligentQueryOptimizer()
    
    try:
        await optimizer.start_optimizer()
    except KeyboardInterrupt:
        print("\nShutting down query optimizer...")
        optimizer.stop_optimizer()
    except Exception as e:
        print(f"Error: {e}")
        optimizer.stop_optimizer()

if __name__ == "__main__":
    asyncio.run(main())