"""
Connection Pool Manager - Async Redis and Database connection pooling
Optimizes I/O operations with connection reuse and intelligent caching.
"""

import asyncio
import json
import time
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
import weakref
import structlog

logger = structlog.get_logger(__name__)


class ConnectionType(Enum):
    """Types of connections supported."""
    REDIS = "redis"
    DATABASE = "database"
    WEBSOCKET = "websocket"
    HTTP = "http"


@dataclass
class ConnectionConfig:
    """Configuration for connection pools."""
    connection_type: ConnectionType
    host: str
    port: int
    max_connections: int = 20
    min_connections: int = 5
    max_idle_time: int = 300  # 5 minutes
    connection_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    enable_health_checks: bool = True
    health_check_interval: int = 60
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    ssl_enabled: bool = False
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectionStats:
    """Statistics for connection pool."""
    total_connections: int
    active_connections: int
    idle_connections: int
    failed_connections: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    pool_utilization: float


class Connection:
    """Wrapper for individual connections with monitoring."""
    
    def __init__(self, connection_id: str, config: ConnectionConfig):
        self.connection_id = connection_id
        self.config = config
        self.created_at = time.time()
        self.last_used = time.time()
        self.is_healthy = True
        self.use_count = 0
        self.error_count = 0
        
        # Connection-specific attributes
        self._native_connection = None
        self._lock = asyncio.Lock()
        
        logger.debug("Connection created", 
                    connection_id=connection_id,
                    type=config.connection_type.value)
    
    async def connect(self):
        """Establish the actual connection."""
        try:
            if self.config.connection_type == ConnectionType.REDIS:
                await self._connect_redis()
            elif self.config.connection_type == ConnectionType.DATABASE:
                await self._connect_database()
            elif self.config.connection_type == ConnectionType.WEBSOCKET:
                await self._connect_websocket()
            elif self.config.connection_type == ConnectionType.HTTP:
                await self._connect_http()
            
            self.is_healthy = True
            logger.info("Connection established", 
                       connection_id=self.connection_id,
                       type=self.config.connection_type.value)
            
        except Exception as e:
            self.is_healthy = False
            self.error_count += 1
            logger.error("Connection failed", 
                        connection_id=self.connection_id,
                        error=str(e))
            raise
    
    async def _connect_redis(self):
        """Connect to Redis with asyncio."""
        try:
            import aioredis
            
            # Build connection URL
            url = f"redis://{self.config.host}:{self.config.port}"
            if self.config.database:
                url += f"/{self.config.database}"
            
            # Connection parameters
            connection_params = {
                'socket_connect_timeout': self.config.connection_timeout,
                'socket_timeout': self.config.connection_timeout,
                'retry_on_timeout': True,
                'health_check_interval': self.config.health_check_interval,
                **self.config.additional_params
            }
            
            if self.config.username and self.config.password:
                connection_params['username'] = self.config.username
                connection_params['password'] = self.config.password
            
            # Create connection
            self._native_connection = aioredis.from_url(
                url, 
                **connection_params
            )
            
            # Test connection
            await self._native_connection.ping()
            
        except ImportError:
            logger.error("aioredis not installed, using mock connection")
            self._native_connection = MockRedisConnection()
        except Exception as e:
            logger.error("Redis connection failed", error=str(e))
            raise
    
    async def _connect_database(self):
        """Connect to database with asyncio."""
        try:
            import asyncpg
            
            # Build connection string
            connection_params = {
                'host': self.config.host,
                'port': self.config.port,
                'database': self.config.database or 'postgres',
                'user': self.config.username or 'postgres',
                'password': self.config.password or '',
                'command_timeout': self.config.connection_timeout,
                **self.config.additional_params
            }
            
            # Create connection
            self._native_connection = await asyncpg.connect(**connection_params)
            
        except ImportError:
            logger.error("asyncpg not installed, using mock connection")
            self._native_connection = MockDatabaseConnection()
        except Exception as e:
            logger.error("Database connection failed", error=str(e))
            raise
    
    async def _connect_websocket(self):
        """Connect to WebSocket."""
        try:
            import websockets
            
            url = f"ws://{self.config.host}:{self.config.port}"
            if self.config.ssl_enabled:
                url = f"wss://{self.config.host}:{self.config.port}"
            
            self._native_connection = await websockets.connect(
                url,
                timeout=self.config.connection_timeout,
                **self.config.additional_params
            )
            
        except ImportError:
            logger.error("websockets not installed, using mock connection")
            self._native_connection = MockWebSocketConnection()
        except Exception as e:
            logger.error("WebSocket connection failed", error=str(e))
            raise
    
    async def _connect_http(self):
        """Connect to HTTP with session."""
        try:
            import aiohttp
            
            timeout = aiohttp.ClientTimeout(total=self.config.connection_timeout)
            self._native_connection = aiohttp.ClientSession(
                timeout=timeout,
                **self.config.additional_params
            )
            
        except ImportError:
            logger.error("aiohttp not installed, using mock connection")
            self._native_connection = MockHttpConnection()
        except Exception as e:
            logger.error("HTTP connection failed", error=str(e))
            raise
    
    async def execute(self, command: str, *args, **kwargs) -> Any:
        """Execute a command on the connection."""
        async with self._lock:
            self.last_used = time.time()
            self.use_count += 1
            
            try:
                if self.config.connection_type == ConnectionType.REDIS:
                    return await self._execute_redis_command(command, *args, **kwargs)
                elif self.config.connection_type == ConnectionType.DATABASE:
                    return await self._execute_database_command(command, *args, **kwargs)
                elif self.config.connection_type == ConnectionType.WEBSOCKET:
                    return await self._execute_websocket_command(command, *args, **kwargs)
                elif self.config.connection_type == ConnectionType.HTTP:
                    return await self._execute_http_command(command, *args, **kwargs)
                
            except Exception as e:
                self.error_count += 1
                logger.error("Command execution failed", 
                           connection_id=self.connection_id,
                           command=command,
                           error=str(e))
                raise
    
    async def _execute_redis_command(self, command: str, *args, **kwargs) -> Any:
        """Execute Redis command."""
        if hasattr(self._native_connection, command):
            method = getattr(self._native_connection, command)
            return await method(*args, **kwargs)
        else:
            raise ValueError(f"Unknown Redis command: {command}")
    
    async def _execute_database_command(self, command: str, *args, **kwargs) -> Any:
        """Execute database command."""
        if command == "fetch":
            return await self._native_connection.fetch(*args, **kwargs)
        elif command == "fetchrow":
            return await self._native_connection.fetchrow(*args, **kwargs)
        elif command == "execute":
            return await self._native_connection.execute(*args, **kwargs)
        else:
            raise ValueError(f"Unknown database command: {command}")
    
    async def _execute_websocket_command(self, command: str, *args, **kwargs) -> Any:
        """Execute WebSocket command."""
        if command == "send":
            return await self._native_connection.send(*args, **kwargs)
        elif command == "recv":
            return await self._native_connection.recv(*args, **kwargs)
        else:
            raise ValueError(f"Unknown WebSocket command: {command}")
    
    async def _execute_http_command(self, command: str, *args, **kwargs) -> Any:
        """Execute HTTP command."""
        if hasattr(self._native_connection, command):
            method = getattr(self._native_connection, command)
            return await method(*args, **kwargs)
        else:
            raise ValueError(f"Unknown HTTP command: {command}")
    
    async def health_check(self) -> bool:
        """Check connection health."""
        try:
            if self.config.connection_type == ConnectionType.REDIS:
                await self._native_connection.ping()
            elif self.config.connection_type == ConnectionType.DATABASE:
                await self._native_connection.fetchrow("SELECT 1")
            elif self.config.connection_type == ConnectionType.WEBSOCKET:
                await self._native_connection.ping()
            elif self.config.connection_type == ConnectionType.HTTP:
                # HTTP sessions don't have built-in health checks
                pass
            
            self.is_healthy = True
            return True
            
        except Exception as e:
            self.is_healthy = False
            logger.warning("Health check failed", 
                          connection_id=self.connection_id,
                          error=str(e))
            return False
    
    async def close(self):
        """Close the connection."""
        try:
            if self._native_connection:
                if hasattr(self._native_connection, 'close'):
                    await self._native_connection.close()
                elif hasattr(self._native_connection, 'disconnect'):
                    await self._native_connection.disconnect()
                
                self._native_connection = None
                
            logger.debug("Connection closed", connection_id=self.connection_id)
            
        except Exception as e:
            logger.error("Error closing connection", 
                        connection_id=self.connection_id,
                        error=str(e))
    
    def is_expired(self) -> bool:
        """Check if connection has expired."""
        return (time.time() - self.last_used) > self.config.max_idle_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            'connection_id': self.connection_id,
            'created_at': self.created_at,
            'last_used': self.last_used,
            'age_seconds': time.time() - self.created_at,
            'idle_seconds': time.time() - self.last_used,
            'use_count': self.use_count,
            'error_count': self.error_count,
            'is_healthy': self.is_healthy,
            'error_rate': self.error_count / max(1, self.use_count)
        }


class ConnectionPool:
    """Async connection pool with health monitoring and caching."""
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.pool_id = f"{config.connection_type.value}_{config.host}_{config.port}"
        
        # Connection management
        self._connections: Dict[str, Connection] = {}
        self._available_connections: asyncio.Queue = asyncio.Queue()
        self._connection_counter = 0
        self._lock = asyncio.Lock()
        
        # Health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._health_check_active = False
        
        # Statistics
        self._stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Query result caching
        self._query_cache: Dict[str, Tuple[Any, float]] = {}
        self._cache_ttl = 300  # 5 minutes
        
        logger.info("Connection pool created", 
                   pool_id=self.pool_id,
                   max_connections=config.max_connections)
    
    async def start(self):
        """Start the connection pool."""
        # Create initial connections
        for _ in range(self.config.min_connections):
            await self._create_connection()
        
        # Start health check monitoring
        if self.config.enable_health_checks:
            self._health_check_active = True
            self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info("Connection pool started", 
                   pool_id=self.pool_id,
                   initial_connections=self.config.min_connections)
    
    async def stop(self):
        """Stop the connection pool."""
        # Stop health checking
        self._health_check_active = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        async with self._lock:
            for connection in self._connections.values():
                await connection.close()
            self._connections.clear()
            
            # Clear the queue
            while not self._available_connections.empty():
                try:
                    self._available_connections.get_nowait()
                except asyncio.QueueEmpty:
                    break
        
        logger.info("Connection pool stopped", pool_id=self.pool_id)
    
    async def _create_connection(self) -> Connection:
        """Create a new connection."""
        async with self._lock:
            if len(self._connections) >= self.config.max_connections:
                raise RuntimeError("Maximum connections reached")
            
            connection_id = f"{self.pool_id}_{self._connection_counter}"
            self._connection_counter += 1
            
            connection = Connection(connection_id, self.config)
            
            # Connect with retries
            for attempt in range(self.config.retry_attempts):
                try:
                    await connection.connect()
                    break
                except Exception as e:
                    if attempt == self.config.retry_attempts - 1:
                        raise
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
            
            self._connections[connection_id] = connection
            await self._available_connections.put(connection)
            
            logger.debug("New connection created", 
                        connection_id=connection_id,
                        total_connections=len(self._connections))
            
            return connection
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool."""
        connection = None
        try:
            # Try to get an available connection
            try:
                connection = await asyncio.wait_for(
                    self._available_connections.get(), 
                    timeout=self.config.connection_timeout
                )
            except asyncio.TimeoutError:
                # Create new connection if under limit
                if len(self._connections) < self.config.max_connections:
                    connection = await self._create_connection()
                else:
                    raise RuntimeError("No connections available and pool is full")
            
            # Health check
            if not connection.is_healthy:
                await connection.health_check()
                if not connection.is_healthy:
                    await connection.close()
                    connection = await self._create_connection()
            
            yield connection
            
        finally:
            # Return connection to pool
            if connection and connection.is_healthy:
                await self._available_connections.put(connection)
    
    async def execute_with_caching(self, command: str, *args, cache_key: Optional[str] = None, **kwargs) -> Any:
        """Execute command with result caching."""
        start_time = time.time()
        
        try:
            # Check cache first
            if cache_key and cache_key in self._query_cache:
                cached_result, cached_time = self._query_cache[cache_key]
                if (time.time() - cached_time) < self._cache_ttl:
                    self._stats['cache_hits'] += 1
                    return cached_result
                else:
                    # Remove expired cache entry
                    del self._query_cache[cache_key]
            
            # Execute command
            async with self.get_connection() as connection:
                result = await connection.execute(command, *args, **kwargs)
            
            # Cache result
            if cache_key:
                self._query_cache[cache_key] = (result, time.time())
                self._stats['cache_misses'] += 1
            
            # Update statistics
            response_time = (time.time() - start_time) * 1000
            self._stats['response_times'].append(response_time)
            if len(self._stats['response_times']) > 1000:  # Keep last 1000 response times
                self._stats['response_times'] = self._stats['response_times'][-1000:]
            
            self._stats['total_requests'] += 1
            self._stats['successful_requests'] += 1
            
            return result
            
        except Exception as e:
            self._stats['total_requests'] += 1
            self._stats['failed_requests'] += 1
            logger.error("Command execution failed", 
                        pool_id=self.pool_id,
                        command=command,
                        error=str(e))
            raise
    
    async def execute(self, command: str, *args, **kwargs) -> Any:
        """Execute command without caching."""
        return await self.execute_with_caching(command, *args, cache_key=None, **kwargs)
    
    async def _health_check_loop(self):
        """Periodic health check for all connections."""
        while self._health_check_active:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                if not self._health_check_active:
                    break
                
                # Check all connections
                unhealthy_connections = []
                async with self._lock:
                    for connection_id, connection in self._connections.items():
                        if not await connection.health_check():
                            unhealthy_connections.append(connection_id)
                
                # Remove unhealthy connections
                for connection_id in unhealthy_connections:
                    await self._remove_connection(connection_id)
                
                # Remove expired connections
                expired_connections = []
                async with self._lock:
                    for connection_id, connection in self._connections.items():
                        if connection.is_expired():
                            expired_connections.append(connection_id)
                
                for connection_id in expired_connections:
                    await self._remove_connection(connection_id)
                
                # Ensure minimum connections
                current_count = len(self._connections)
                if current_count < self.config.min_connections:
                    for _ in range(self.config.min_connections - current_count):
                        try:
                            await self._create_connection()
                        except Exception as e:
                            logger.error("Failed to create connection during health check", 
                                       error=str(e))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in health check loop", error=str(e))
    
    async def _remove_connection(self, connection_id: str):
        """Remove a connection from the pool."""
        async with self._lock:
            if connection_id in self._connections:
                connection = self._connections[connection_id]
                await connection.close()
                del self._connections[connection_id]
                
                logger.debug("Connection removed", 
                           connection_id=connection_id,
                           remaining_connections=len(self._connections))
    
    def clear_cache(self):
        """Clear query result cache."""
        self._query_cache.clear()
        logger.info("Query cache cleared", pool_id=self.pool_id)
    
    def get_stats(self) -> ConnectionStats:
        """Get pool statistics."""
        total_connections = len(self._connections)
        active_connections = total_connections - self._available_connections.qsize()
        
        # Calculate average response time
        response_times = self._stats['response_times']
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        
        # Calculate cache hit rate
        total_cache_requests = self._stats['cache_hits'] + self._stats['cache_misses']
        cache_hit_rate = (
            self._stats['cache_hits'] / total_cache_requests 
            if total_cache_requests > 0 else 0.0
        )
        
        return ConnectionStats(
            total_connections=total_connections,
            active_connections=active_connections,
            idle_connections=self._available_connections.qsize(),
            failed_connections=sum(1 for conn in self._connections.values() if not conn.is_healthy),
            total_requests=self._stats['total_requests'],
            successful_requests=self._stats['successful_requests'],
            failed_requests=self._stats['failed_requests'],
            avg_response_time_ms=avg_response_time,
            cache_hits=self._stats['cache_hits'],
            cache_misses=self._stats['cache_misses'],
            cache_hit_rate=cache_hit_rate,
            pool_utilization=active_connections / self.config.max_connections
        )


class ConnectionManager:
    """Manages multiple connection pools."""
    
    def __init__(self):
        self._pools: Dict[str, ConnectionPool] = {}
        self._lock = asyncio.Lock()
        
        logger.info("Connection manager initialized")
    
    async def create_pool(self, config: ConnectionConfig) -> ConnectionPool:
        """Create a new connection pool."""
        pool_id = f"{config.connection_type.value}_{config.host}_{config.port}"
        
        async with self._lock:
            if pool_id in self._pools:
                return self._pools[pool_id]
            
            pool = ConnectionPool(config)
            await pool.start()
            self._pools[pool_id] = pool
            
            logger.info("Connection pool created", pool_id=pool_id)
            return pool
    
    async def get_pool(self, connection_type: ConnectionType, host: str, port: int) -> Optional[ConnectionPool]:
        """Get an existing connection pool."""
        pool_id = f"{connection_type.value}_{host}_{port}"
        return self._pools.get(pool_id)
    
    async def remove_pool(self, connection_type: ConnectionType, host: str, port: int):
        """Remove a connection pool."""
        pool_id = f"{connection_type.value}_{host}_{port}"
        
        async with self._lock:
            if pool_id in self._pools:
                pool = self._pools[pool_id]
                await pool.stop()
                del self._pools[pool_id]
                
                logger.info("Connection pool removed", pool_id=pool_id)
    
    async def stop_all(self):
        """Stop all connection pools."""
        async with self._lock:
            for pool in self._pools.values():
                await pool.stop()
            self._pools.clear()
            
            logger.info("All connection pools stopped")
    
    def get_all_stats(self) -> Dict[str, ConnectionStats]:
        """Get statistics for all pools."""
        return {pool_id: pool.get_stats() for pool_id, pool in self._pools.items()}


# Mock connections for testing/fallback
class MockRedisConnection:
    """Mock Redis connection for testing."""
    
    def __init__(self):
        self._data = {}
    
    async def ping(self):
        return True
    
    async def get(self, key):
        return self._data.get(key)
    
    async def set(self, key, value, ex=None):
        self._data[key] = value
        return True
    
    async def delete(self, key):
        return self._data.pop(key, None) is not None
    
    async def close(self):
        pass


class MockDatabaseConnection:
    """Mock database connection for testing."""
    
    async def fetch(self, query, *args):
        return []
    
    async def fetchrow(self, query, *args):
        return None
    
    async def execute(self, query, *args):
        return "SELECT 1"
    
    async def close(self):
        pass


class MockWebSocketConnection:
    """Mock WebSocket connection for testing."""
    
    async def send(self, message):
        pass
    
    async def recv(self):
        return "{}"
    
    async def ping(self):
        return True
    
    async def close(self):
        pass


class MockHttpConnection:
    """Mock HTTP connection for testing."""
    
    async def get(self, url, **kwargs):
        return MockHttpResponse()
    
    async def post(self, url, **kwargs):
        return MockHttpResponse()
    
    async def close(self):
        pass


class MockHttpResponse:
    """Mock HTTP response for testing."""
    
    def __init__(self):
        self.status = 200
        self.headers = {}
    
    async def json(self):
        return {}
    
    async def text(self):
        return ""


# Global connection manager
_global_connection_manager: Optional[ConnectionManager] = None


def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager."""
    global _global_connection_manager
    
    if _global_connection_manager is None:
        _global_connection_manager = ConnectionManager()
    
    return _global_connection_manager


async def create_redis_pool(host: str = "localhost", port: int = 6379, **kwargs) -> ConnectionPool:
    """Create a Redis connection pool."""
    config = ConnectionConfig(
        connection_type=ConnectionType.REDIS,
        host=host,
        port=port,
        **kwargs
    )
    
    return await get_connection_manager().create_pool(config)


async def create_database_pool(host: str = "localhost", port: int = 5432, **kwargs) -> ConnectionPool:
    """Create a database connection pool."""
    config = ConnectionConfig(
        connection_type=ConnectionType.DATABASE,
        host=host,
        port=port,
        **kwargs
    )
    
    return await get_connection_manager().create_pool(config)