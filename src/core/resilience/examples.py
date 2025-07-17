"""
Resilience Framework Integration Examples
========================================

Examples showing how to integrate the resilience framework with
various services and external dependencies.

Examples include:
- Database connections
- API clients
- Broker connections
- Message queues
- Cache systems
- External services
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
import aiohttp
import redis.asyncio as redis
from contextlib import asynccontextmanager

from .resilience_manager import ResilienceManager, ResilienceConfig, create_resilience_manager
from .bulkhead import ResourcePriority

logger = logging.getLogger(__name__)


# Example: Database Integration
class DatabaseClient:
    """Example database client with resilience integration."""
    
    def __init__(self, connection_string: str, resilience_manager: ResilienceManager):
        self.connection_string = connection_string
        self.resilience_manager = resilience_manager
        self.connection = None
    
    async def initialize(self):
        """Initialize database connection."""
        # Register with resilience manager
        await self.resilience_manager.register_service(
            service_name="database",
            service_instance=self,
            service_config={
                'failure_threshold': 3,
                'timeout_seconds': 30,
                'max_retry_attempts': 3,
                'retry_base_delay': 1.0,
                'max_threads': 10,
                'max_connections': 5,
                'max_concurrent': 20
            }
        )
        
        # Simulate database connection
        self.connection = {"connected": True}
        logger.info("Database client initialized")
    
    async def ping(self):
        """Health check method."""
        if not self.connection or not self.connection.get("connected"):
            raise ConnectionError("Database not connected")
        return True
    
    async def deep_health_check(self):
        """Deep health check method."""
        await self.ping()
        # Additional checks could go here
        return True
    
    async def query(self, sql: str, params: tuple = None) -> Dict[str, Any]:
        """Execute database query with resilience protection and parameterized queries."""
        async with self.resilience_manager.resilient_call(
            "database", 
            "query", 
            ResourcePriority.HIGH
        ):
            if not self.connection:
                raise ConnectionError("Database not connected")
            
            # Validate input to prevent SQL injection
            if params:
                # Simulate parameterized query execution
                await asyncio.sleep(0.1)
                return {"result": f"Parameterized query executed: {sql} with params: {params}"}
            else:
                # Simulate query execution
                await asyncio.sleep(0.1)
                return {"result": f"Query executed: {sql}"}
    
    async def transaction(self, queries: list) -> Dict[str, Any]:
        """Execute transaction with resilience protection."""
        async with self.resilience_manager.resilient_call(
            "database", 
            "transaction", 
            ResourcePriority.CRITICAL
        ):
            if not self.connection:
                raise ConnectionError("Database not connected")
            
            # Simulate transaction execution
            await asyncio.sleep(0.2)
            return {"result": f"Transaction executed: {len(queries)} queries"}


# Example: API Client Integration
class APIClient:
    """Example API client with resilience integration."""
    
    def __init__(self, base_url: str, resilience_manager: ResilienceManager):
        self.base_url = base_url
        self.resilience_manager = resilience_manager
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self):
        """Initialize API client."""
        # Register with resilience manager
        await self.resilience_manager.register_service(
            service_name="api_client",
            service_instance=self,
            service_config={
                'failure_threshold': 5,
                'timeout_seconds': 60,
                'max_retry_attempts': 3,
                'retry_base_delay': 2.0,
                'max_threads': 20,
                'max_connections': 10,
                'max_concurrent': 50
            }
        )
        
        # Create HTTP session
        self.session = aiohttp.ClientSession()
        logger.info("API client initialized")
    
    async def ping(self):
        """Health check method."""
        if not self.session:
            raise ConnectionError("API client not initialized")
        
        try:
            async with self.session.get(f"{self.base_url}/health", timeout=5) as response:
                return response.status == 200
        except Exception as e:
            raise ConnectionError(f"API health check failed: {e}")
    
    async def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make GET request with resilience protection."""
        async with self.resilience_manager.resilient_call(
            "api_client", 
            f"GET_{endpoint}", 
            ResourcePriority.MEDIUM
        ):
            if not self.session:
                raise ConnectionError("API client not initialized")
            
            async with self.session.get(
                f"{self.base_url}/{endpoint}",
                params=params,
                timeout=30
            ) as response:
                if response.status >= 400:
                    raise aiohttp.ClientError(f"HTTP {response.status}")
                
                return await response.json()
    
    async def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make POST request with resilience protection."""
        async with self.resilience_manager.resilient_call(
            "api_client", 
            f"POST_{endpoint}", 
            ResourcePriority.HIGH
        ):
            if not self.session:
                raise ConnectionError("API client not initialized")
            
            async with self.session.post(
                f"{self.base_url}/{endpoint}",
                json=data,
                timeout=30
            ) as response:
                if response.status >= 400:
                    raise aiohttp.ClientError(f"HTTP {response.status}")
                
                return await response.json()
    
    async def close(self):
        """Close API client."""
        if self.session:
            await self.session.close()


# Example: Broker Integration
class BrokerClient:
    """Example broker client with resilience integration."""
    
    def __init__(self, broker_config: Dict[str, Any], resilience_manager: ResilienceManager):
        self.broker_config = broker_config
        self.resilience_manager = resilience_manager
        self.connected = False
    
    async def initialize(self):
        """Initialize broker client."""
        # Register with resilience manager
        await self.resilience_manager.register_service(
            service_name="broker",
            service_instance=self,
            service_config={
                'failure_threshold': 2,
                'timeout_seconds': 10,
                'max_retry_attempts': 2,
                'retry_base_delay': 0.5,
                'max_threads': 5,
                'max_connections': 3,
                'max_concurrent': 10
            }
        )
        
        # Simulate broker connection
        self.connected = True
        logger.info("Broker client initialized")
    
    async def ping(self):
        """Health check method."""
        if not self.connected:
            raise ConnectionError("Broker not connected")
        return True
    
    async def submit_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit order with resilience protection."""
        async with self.resilience_manager.resilient_call(
            "broker", 
            "submit_order", 
            ResourcePriority.CRITICAL
        ):
            if not self.connected:
                raise ConnectionError("Broker not connected")
            
            # Simulate order submission
            await asyncio.sleep(0.05)
            return {"order_id": f"ORD_{order_data.get('symbol', 'UNKNOWN')}"}
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel order with resilience protection."""
        async with self.resilience_manager.resilient_call(
            "broker", 
            "cancel_order", 
            ResourcePriority.HIGH
        ):
            if not self.connected:
                raise ConnectionError("Broker not connected")
            
            # Simulate order cancellation
            await asyncio.sleep(0.02)
            return {"status": "cancelled", "order_id": order_id}


# Example: Cache Integration
class CacheClient:
    """Example cache client with resilience integration."""
    
    def __init__(self, redis_url: str, resilience_manager: ResilienceManager):
        self.redis_url = redis_url
        self.resilience_manager = resilience_manager
        self.redis_client: Optional[redis.Redis] = None
    
    async def initialize(self):
        """Initialize cache client."""
        # Register with resilience manager
        await self.resilience_manager.register_service(
            service_name="cache",
            service_instance=self,
            service_config={
                'failure_threshold': 3,
                'timeout_seconds': 5,
                'max_retry_attempts': 2,
                'retry_base_delay': 0.1,
                'max_threads': 10,
                'max_connections': 5,
                'max_concurrent': 100
            }
        )
        
        # Create Redis client
        self.redis_client = redis.from_url(self.redis_url)
        logger.info("Cache client initialized")
    
    async def ping(self):
        """Health check method."""
        if not self.redis_client:
            raise ConnectionError("Cache client not initialized")
        
        await self.redis_client.ping()
        return True
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache with resilience protection."""
        async with self.resilience_manager.resilient_call(
            "cache", 
            "get", 
            ResourcePriority.LOW
        ):
            if not self.redis_client:
                raise ConnectionError("Cache client not initialized")
            
            return await self.redis_client.get(key)
    
    async def set(self, key: str, value: str, ttl: int = 3600) -> bool:
        """Set value in cache with resilience protection."""
        async with self.resilience_manager.resilient_call(
            "cache", 
            "set", 
            ResourcePriority.LOW
        ):
            if not self.redis_client:
                raise ConnectionError("Cache client not initialized")
            
            await self.redis_client.set(key, value, ex=ttl)
            return True
    
    async def close(self):
        """Close cache client."""
        if self.redis_client:
            await self.redis_client.close()


# Example: Complete Application Integration
class TradingApplication:
    """Example trading application with comprehensive resilience."""
    
    def __init__(self):
        self.resilience_manager: Optional[ResilienceManager] = None
        self.database: Optional[DatabaseClient] = None
        self.api_client: Optional[APIClient] = None
        self.broker: Optional[BrokerClient] = None
        self.cache: Optional[CacheClient] = None
    
    async def initialize(self):
        """Initialize trading application."""
        # Create resilience manager
        self.resilience_manager = await create_resilience_manager(
            service_name="trading_app",
            environment="production",
            redis_url="redis://localhost:6379/0",
            circuit_breaker_enabled=True,
            adaptive_circuit_breaker_enabled=True,
            retry_enabled=True,
            health_monitoring_enabled=True,
            bulkhead_enabled=True,
            chaos_engineering_enabled=False  # Disabled in production
        )
        
        # Initialize clients
        self.database = DatabaseClient("postgresql://localhost/trading", self.resilience_manager)
        await self.database.initialize()
        
        self.api_client = APIClient("https://api.marketdata.com", self.resilience_manager)
        await self.api_client.initialize()
        
        self.broker = BrokerClient({"api_key": "test"}, self.resilience_manager)
        await self.broker.initialize()
        
        self.cache = CacheClient("redis://localhost:6379/1", self.resilience_manager)
        await self.cache.initialize()
        
        logger.info("Trading application initialized")
    
    async def execute_trade(self, symbol: str, quantity: int, price: float) -> Dict[str, Any]:
        """Execute a trade with full resilience protection."""
        try:
            # Get market data
            market_data = await self.api_client.get(f"quotes/{symbol}")
            
            # Check cache for recent data
            cache_key = f"trade_history:{symbol}"
            cached_data = await self.cache.get(cache_key)
            
            # Store trade in database
            trade_record = {
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "timestamp": "2025-07-15T10:30:00Z"
            }
            
            db_result = await self.database.query(
                "INSERT INTO trades (symbol, quantity, price) VALUES (?, ?, ?)",
                (symbol, quantity, price)
            )
            
            # Submit order to broker
            order_data = {
                "symbol": symbol,
                "quantity": quantity,
                "price": price,
                "side": "buy"
            }
            
            order_result = await self.broker.submit_order(order_data)
            
            # Update cache
            await self.cache.set(cache_key, f"last_trade:{symbol}:{quantity}", 3600)
            
            return {
                "success": True,
                "order_id": order_result["order_id"],
                "market_data": market_data,
                "db_result": db_result
            }
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        if not self.resilience_manager:
            return {"error": "Not initialized"}
        
        return self.resilience_manager.get_system_status()
    
    async def run_health_checks(self):
        """Run health checks on all services."""
        if not self.resilience_manager:
            return
        
        services = ["database", "api_client", "broker", "cache"]
        
        for service in services:
            try:
                await self.resilience_manager.run_health_check(service)
                logger.info(f"Health check passed for {service}")
            except Exception as e:
                logger.error(f"Health check failed for {service}: {e}")
    
    async def close(self):
        """Close application and all clients."""
        if self.api_client:
            await self.api_client.close()
        
        if self.cache:
            await self.cache.close()
        
        if self.resilience_manager:
            await self.resilience_manager.close()
        
        logger.info("Trading application closed")


# Example usage functions
async def example_basic_usage():
    """Basic usage example."""
    print("=== Basic Resilience Framework Usage ===")
    
    # Create resilience manager
    manager = await create_resilience_manager(
        service_name="example_app",
        environment="development"
    )
    
    try:
        # Create and register a simple service
        database = DatabaseClient("postgresql://localhost/example", manager)
        await database.initialize()
        
        # Use the service
        result = await database.query("SELECT * FROM users")
        print(f"Query result: {result}")
        
        # Check system status
        status = manager.get_system_status()
        print(f"System status: {status}")
        
    finally:
        await manager.close()


async def example_advanced_usage():
    """Advanced usage example with multiple services."""
    print("=== Advanced Resilience Framework Usage ===")
    
    # Create trading application
    app = TradingApplication()
    
    try:
        # Initialize application
        await app.initialize()
        
        # Execute a trade
        trade_result = await app.execute_trade("AAPL", 100, 150.0)
        print(f"Trade result: {trade_result}")
        
        # Run health checks
        await app.run_health_checks()
        
        # Get system status
        status = await app.get_system_status()
        print(f"System status: {status}")
        
    finally:
        await app.close()


async def example_chaos_testing():
    """Chaos engineering example."""
    print("=== Chaos Engineering Example ===")
    
    # Create resilience manager with chaos engineering enabled
    manager = await create_resilience_manager(
        service_name="chaos_test",
        environment="testing",
        chaos_engineering_enabled=True
    )
    
    try:
        # Register a service
        database = DatabaseClient("postgresql://localhost/chaos_test", manager)
        await database.initialize()
        
        # Run chaos experiment
        experiment_result = await manager.run_chaos_experiment(
            "network_delay_test",
            "database"
        )
        print(f"Chaos experiment result: {experiment_result}")
        
    finally:
        await manager.close()


async def example_monitoring():
    """Monitoring and observability example."""
    print("=== Monitoring and Observability Example ===")
    
    manager = await create_resilience_manager(
        service_name="monitoring_example",
        environment="production",
        metrics_enabled=True,
        prometheus_enabled=True
    )
    
    try:
        # Register multiple services
        database = DatabaseClient("postgresql://localhost/monitoring", manager)
        await database.initialize()
        
        api_client = APIClient("https://api.example.com", manager)
        await api_client.initialize()
        
        # Simulate some activity
        for i in range(10):
            try:
                # Validate table name to prevent SQL injection
                table_name = f"table_{i}"
                # Whitelist validation for table names (only allow alphanumeric and underscore)
                if not table_name.replace('_', '').isalnum():
                    raise ValueError(f"Invalid table name: {table_name}")
                await database.query(f"SELECT * FROM {table_name} WHERE id = ?", (i,))
                await api_client.get(f"data/{i}")
            except Exception as e:
                print(f"Operation {i} failed: {e}")
        
        # Get detailed service status
        db_status = manager.get_service_status("database")
        api_status = manager.get_service_status("api_client")
        
        print(f"Database status: {db_status}")
        print(f"API client status: {api_status}")
        
    finally:
        await manager.close()


# Main example runner
async def main():
    """Run all examples."""
    print("Running Resilience Framework Examples...")
    
    try:
        await example_basic_usage()
        print("\n" + "="*50 + "\n")
        
        await example_advanced_usage()
        print("\n" + "="*50 + "\n")
        
        await example_chaos_testing()
        print("\n" + "="*50 + "\n")
        
        await example_monitoring()
        
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run examples
    asyncio.run(main())