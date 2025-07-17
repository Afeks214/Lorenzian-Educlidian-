"""
Database fixture management for test isolation and efficiency.
Agent 4 Mission: Test Data Management & Caching System
"""
import os
import json
import time
import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
import pytest
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

try:
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

import tempfile
from unittest.mock import Mock, patch

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Configuration for database fixtures."""
    host: str = "localhost"
    port: int = 5432
    database: str = "test_db"
    user: str = "test_user"
    password: str = "test_password"
    max_connections: int = 20
    command_timeout: int = 30
    isolation_level: str = "READ_COMMITTED"
    
@dataclass
class RedisConfig:
    """Configuration for Redis fixtures."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 10

class DatabaseFixtureManager:
    """Manages database fixtures for tests."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.active_connections = {}
        self.docker_client = None
        self.containers = {}
        self.schemas = {}
        
        # Initialize Docker client if available
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logger.warning(f"Docker not available: {e}")
    
    def setup_postgres_container(self, container_name: str = "test_postgres") -> Dict[str, Any]:
        """Set up PostgreSQL container for testing."""
        if not self.docker_client:
            raise RuntimeError("Docker not available for container setup")
        
        if not PSYCOPG2_AVAILABLE:
            raise RuntimeError("psycopg2 not available for PostgreSQL operations")
        
        # Remove existing container if exists
        try:
            existing = self.docker_client.containers.get(container_name)
            existing.stop()
            existing.remove()
        except docker.errors.NotFound:
            pass
        
        # Start new container
        container = self.docker_client.containers.run(
            "postgres:13",
            name=container_name,
            environment={
                "POSTGRES_DB": self.config.database,
                "POSTGRES_USER": self.config.user,
                "POSTGRES_PASSWORD": self.config.password,
            },
            ports={"5432/tcp": self.config.port},
            detach=True,
            remove=True
        )
        
        # Wait for container to be ready
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                conn = psycopg2.connect(
                    host=self.config.host,
                    port=self.config.port,
                    database=self.config.database,
                    user=self.config.user,
                    password=self.config.password
                )
                conn.close()
                break
            except psycopg2.OperationalError:
                if attempt == max_attempts - 1:
                    container.stop()
                    raise RuntimeError("PostgreSQL container failed to start")
                time.sleep(1)
        
        self.containers[container_name] = container
        
        return {
            "container": container,
            "host": self.config.host,
            "port": self.config.port,
            "database": self.config.database,
            "user": self.config.user,
            "password": self.config.password
        }
    
    def setup_redis_container(self, container_name: str = "test_redis") -> Dict[str, Any]:
        """Set up Redis container for testing."""
        if not self.docker_client:
            raise RuntimeError("Docker not available for container setup")
        
        # Remove existing container if exists
        try:
            existing = self.docker_client.containers.get(container_name)
            existing.stop()
            existing.remove()
        except docker.errors.NotFound:
            pass
        
        # Start new container
        container = self.docker_client.containers.run(
            "redis:6",
            name=container_name,
            ports={"6379/tcp": 6379},
            detach=True,
            remove=True
        )
        
        # Wait for container to be ready
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                r = redis.Redis(host="localhost", port=6379, db=0)
                r.ping()
                break
            except redis.ConnectionError:
                if attempt == max_attempts - 1:
                    container.stop()
                    raise RuntimeError("Redis container failed to start")
                time.sleep(1)
        
        self.containers[container_name] = container
        
        return {
            "container": container,
            "host": "localhost",
            "port": 6379
        }
    
    @contextmanager
    def get_sync_connection(self, schema: str = None):
        """Get synchronous database connection with optional schema."""
        conn = None
        try:
            conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password
            )
            
            if schema:
                with conn.cursor() as cur:
                    cur.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
            
            yield conn
        finally:
            if conn:
                conn.close()
    
    @asynccontextmanager
    async def get_async_connection(self, schema: str = None):
        """Get asynchronous database connection with optional schema."""
        conn = None
        try:
            conn = await asyncpg.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password
            )
            
            if schema:
                # Use proper identifier quoting to prevent SQL injection
                # Note: asyncpg doesn't use psycopg2's sql.Identifier, so we validate and quote manually
                if not schema.replace('_', '').isalnum():
                    raise ValueError(f"Invalid schema name: {schema}")
                await conn.execute(f"SET search_path TO {schema}")
            
            yield conn
        finally:
            if conn:
                await conn.close()
    
    def create_schema(self, schema_name: str) -> None:
        """Create isolated schema for test."""
        with self.get_sync_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema_name)))
                conn.commit()
        
        self.schemas[schema_name] = datetime.now()
    
    def drop_schema(self, schema_name: str) -> None:
        """Drop schema after test."""
        with self.get_sync_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema_name)))
                conn.commit()
        
        if schema_name in self.schemas:
            del self.schemas[schema_name]
    
    def setup_test_tables(self, schema_name: str) -> None:
        """Set up standard test tables."""
        with self.get_sync_connection(schema_name) as conn:
            with conn.cursor() as cur:
                # Market data table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS market_data (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        open_price DECIMAL(10,2),
                        high_price DECIMAL(10,2),
                        low_price DECIMAL(10,2),
                        close_price DECIMAL(10,2),
                        volume BIGINT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create index for performance
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp 
                    ON market_data(symbol, timestamp)
                """)
                
                # Risk metrics table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS risk_metrics (
                        id SERIAL PRIMARY KEY,
                        portfolio_id VARCHAR(50) NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        var_95 DECIMAL(15,6),
                        var_99 DECIMAL(15,6),
                        expected_shortfall DECIMAL(15,6),
                        max_drawdown DECIMAL(15,6),
                        sharpe_ratio DECIMAL(10,6),
                        volatility DECIMAL(10,6),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Positions table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS positions (
                        id SERIAL PRIMARY KEY,
                        account_id VARCHAR(50) NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        quantity DECIMAL(15,6),
                        entry_price DECIMAL(10,2),
                        current_price DECIMAL(10,2),
                        unrealized_pnl DECIMAL(15,6),
                        timestamp TIMESTAMP NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Orders table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS orders (
                        id SERIAL PRIMARY KEY,
                        order_id VARCHAR(50) UNIQUE NOT NULL,
                        account_id VARCHAR(50) NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        side VARCHAR(10) NOT NULL,
                        quantity DECIMAL(15,6),
                        price DECIMAL(10,2),
                        order_type VARCHAR(20) NOT NULL,
                        status VARCHAR(20) NOT NULL,
                        filled_quantity DECIMAL(15,6) DEFAULT 0,
                        avg_fill_price DECIMAL(10,2),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
    
    def populate_test_data(self, schema_name: str, data_type: str = "market") -> None:
        """Populate test tables with sample data."""
        with self.get_sync_connection(schema_name) as conn:
            with conn.cursor() as cur:
                if data_type == "market":
                    # Insert sample market data
                    cur.execute("""
                        INSERT INTO market_data (symbol, timestamp, open_price, high_price, low_price, close_price, volume)
                        VALUES 
                        ('NQ', '2023-01-01 09:30:00', 12000.00, 12050.00, 11980.00, 12020.00, 1000000),
                        ('NQ', '2023-01-01 09:35:00', 12020.00, 12080.00, 12000.00, 12060.00, 1200000),
                        ('NQ', '2023-01-01 09:40:00', 12060.00, 12100.00, 12040.00, 12080.00, 1100000),
                        ('ES', '2023-01-01 09:30:00', 4000.00, 4020.00, 3990.00, 4010.00, 800000),
                        ('ES', '2023-01-01 09:35:00', 4010.00, 4040.00, 4000.00, 4030.00, 900000),
                        ('ES', '2023-01-01 09:40:00', 4030.00, 4050.00, 4020.00, 4040.00, 850000)
                    """)
                
                elif data_type == "risk":
                    # Insert sample risk metrics
                    cur.execute("""
                        INSERT INTO risk_metrics (portfolio_id, timestamp, var_95, var_99, expected_shortfall, max_drawdown, sharpe_ratio, volatility)
                        VALUES 
                        ('portfolio_1', '2023-01-01 09:30:00', -50000.00, -75000.00, -80000.00, -120000.00, 1.5, 0.15),
                        ('portfolio_1', '2023-01-01 09:35:00', -52000.00, -78000.00, -82000.00, -125000.00, 1.4, 0.16),
                        ('portfolio_2', '2023-01-01 09:30:00', -30000.00, -45000.00, -48000.00, -70000.00, 1.8, 0.12),
                        ('portfolio_2', '2023-01-01 09:35:00', -31000.00, -46000.00, -49000.00, -72000.00, 1.7, 0.13)
                    """)
                
                elif data_type == "positions":
                    # Insert sample positions
                    cur.execute("""
                        INSERT INTO positions (account_id, symbol, quantity, entry_price, current_price, unrealized_pnl, timestamp)
                        VALUES 
                        ('account_1', 'NQ', 10.0, 12000.00, 12020.00, 200.00, '2023-01-01 09:30:00'),
                        ('account_1', 'ES', 5.0, 4000.00, 4010.00, 50.00, '2023-01-01 09:30:00'),
                        ('account_2', 'NQ', -5.0, 12050.00, 12020.00, 150.00, '2023-01-01 09:30:00'),
                        ('account_2', 'ES', 8.0, 4020.00, 4040.00, 160.00, '2023-01-01 09:30:00')
                    """)
                
                elif data_type == "orders":
                    # Insert sample orders
                    cur.execute("""
                        INSERT INTO orders (order_id, account_id, symbol, side, quantity, price, order_type, status, filled_quantity, avg_fill_price)
                        VALUES 
                        ('order_1', 'account_1', 'NQ', 'BUY', 10.0, 12000.00, 'MARKET', 'FILLED', 10.0, 12000.00),
                        ('order_2', 'account_1', 'ES', 'BUY', 5.0, 4000.00, 'LIMIT', 'FILLED', 5.0, 4000.00),
                        ('order_3', 'account_2', 'NQ', 'SELL', 5.0, 12050.00, 'MARKET', 'FILLED', 5.0, 12050.00),
                        ('order_4', 'account_2', 'ES', 'BUY', 8.0, 4020.00, 'LIMIT', 'PARTIALLY_FILLED', 5.0, 4020.00)
                    """)
                
                conn.commit()
    
    def cleanup_containers(self) -> None:
        """Clean up all containers."""
        for container in self.containers.values():
            try:
                container.stop()
                container.remove()
            except Exception as e:
                logger.warning(f"Error cleaning up container: {e}")
        
        self.containers.clear()
    
    def cleanup_schemas(self) -> None:
        """Clean up all test schemas."""
        for schema_name in list(self.schemas.keys()):
            try:
                self.drop_schema(schema_name)
            except Exception as e:
                logger.warning(f"Error cleaning up schema {schema_name}: {e}")

class MockExternalServiceManager:
    """Manager for mock external services."""
    
    def __init__(self):
        self.active_mocks = {}
        self.response_overrides = {}
    
    def setup_broker_mock(self, broker_name: str = "mock_broker") -> Mock:
        """Set up mock broker interface."""
        broker_mock = Mock()
        
        # Default responses
        broker_mock.connect.return_value = True
        broker_mock.disconnect.return_value = True
        broker_mock.submit_order.return_value = {
            "order_id": "mock_order_123",
            "status": "PENDING",
            "timestamp": datetime.now().isoformat()
        }
        broker_mock.cancel_order.return_value = {
            "order_id": "mock_order_123",
            "status": "CANCELLED",
            "timestamp": datetime.now().isoformat()
        }
        broker_mock.get_positions.return_value = [
            {
                "symbol": "NQ",
                "quantity": 10.0,
                "entry_price": 12000.00,
                "current_price": 12020.00,
                "unrealized_pnl": 200.00
            }
        ]
        broker_mock.get_account_balance.return_value = {
            "cash": 100000.00,
            "equity": 120000.00,
            "margin_used": 50000.00,
            "margin_available": 50000.00
        }
        
        self.active_mocks[broker_name] = broker_mock
        return broker_mock
    
    def setup_market_data_mock(self, provider_name: str = "mock_data_provider") -> Mock:
        """Set up mock market data provider."""
        provider_mock = Mock()
        
        # Default responses
        provider_mock.connect.return_value = True
        provider_mock.disconnect.return_value = True
        provider_mock.subscribe.return_value = True
        provider_mock.unsubscribe.return_value = True
        provider_mock.get_quote.return_value = {
            "symbol": "NQ",
            "bid": 12019.00,
            "ask": 12021.00,
            "last": 12020.00,
            "volume": 1000000,
            "timestamp": datetime.now().isoformat()
        }
        provider_mock.get_historical_data.return_value = [
            {
                "timestamp": "2023-01-01T09:30:00",
                "open": 12000.00,
                "high": 12050.00,
                "low": 11980.00,
                "close": 12020.00,
                "volume": 1000000
            }
        ]
        
        self.active_mocks[provider_name] = provider_mock
        return provider_mock
    
    def setup_risk_service_mock(self, service_name: str = "mock_risk_service") -> Mock:
        """Set up mock risk management service."""
        service_mock = Mock()
        
        # Default responses
        service_mock.calculate_var.return_value = {
            "var_95": -50000.00,
            "var_99": -75000.00,
            "confidence_interval": [0.95, 0.99],
            "timestamp": datetime.now().isoformat()
        }
        service_mock.calculate_expected_shortfall.return_value = {
            "expected_shortfall": -80000.00,
            "confidence_level": 0.95,
            "timestamp": datetime.now().isoformat()
        }
        service_mock.assess_portfolio_risk.return_value = {
            "overall_risk": "MEDIUM",
            "risk_score": 6.5,
            "recommendations": ["Reduce position size", "Increase diversification"],
            "timestamp": datetime.now().isoformat()
        }
        
        self.active_mocks[service_name] = service_mock
        return service_mock
    
    def override_response(self, service_name: str, method_name: str, response: Any) -> None:
        """Override response for specific service method."""
        if service_name not in self.response_overrides:
            self.response_overrides[service_name] = {}
        
        self.response_overrides[service_name][method_name] = response
        
        # Apply override to active mock
        if service_name in self.active_mocks:
            mock = self.active_mocks[service_name]
            setattr(mock, method_name, Mock(return_value=response))
    
    def get_mock(self, service_name: str) -> Optional[Mock]:
        """Get active mock by name."""
        return self.active_mocks.get(service_name)
    
    def cleanup_mocks(self) -> None:
        """Clean up all mocks."""
        self.active_mocks.clear()
        self.response_overrides.clear()

# Global instances
database_fixture_manager = DatabaseFixtureManager(DatabaseConfig())
mock_service_manager = MockExternalServiceManager()

# Pytest fixtures
@pytest.fixture(scope="session")
def postgres_container():
    """Set up PostgreSQL container for session."""
    container_info = database_fixture_manager.setup_postgres_container()
    yield container_info
    database_fixture_manager.cleanup_containers()

@pytest.fixture(scope="session")
def redis_container():
    """Set up Redis container for session."""
    container_info = database_fixture_manager.setup_redis_container()
    yield container_info
    database_fixture_manager.cleanup_containers()

@pytest.fixture
def isolated_db_schema():
    """Create isolated database schema for test."""
    schema_name = f"test_schema_{int(time.time() * 1000)}"
    database_fixture_manager.create_schema(schema_name)
    database_fixture_manager.setup_test_tables(schema_name)
    
    yield schema_name
    
    database_fixture_manager.drop_schema(schema_name)

@pytest.fixture
def db_connection():
    """Provide database connection for test."""
    with database_fixture_manager.get_sync_connection() as conn:
        yield conn

@pytest.fixture
async def async_db_connection():
    """Provide async database connection for test."""
    async with database_fixture_manager.get_async_connection() as conn:
        yield conn

@pytest.fixture
def market_data_db(isolated_db_schema):
    """Provide database with market data."""
    database_fixture_manager.populate_test_data(isolated_db_schema, "market")
    yield isolated_db_schema

@pytest.fixture
def risk_metrics_db(isolated_db_schema):
    """Provide database with risk metrics."""
    database_fixture_manager.populate_test_data(isolated_db_schema, "risk")
    yield isolated_db_schema

@pytest.fixture
def positions_db(isolated_db_schema):
    """Provide database with positions."""
    database_fixture_manager.populate_test_data(isolated_db_schema, "positions")
    yield isolated_db_schema

@pytest.fixture
def orders_db(isolated_db_schema):
    """Provide database with orders."""
    database_fixture_manager.populate_test_data(isolated_db_schema, "orders")
    yield isolated_db_schema

@pytest.fixture
def mock_broker():
    """Provide mock broker interface."""
    broker = mock_service_manager.setup_broker_mock()
    yield broker
    mock_service_manager.cleanup_mocks()

@pytest.fixture
def mock_data_provider():
    """Provide mock market data provider."""
    provider = mock_service_manager.setup_market_data_mock()
    yield provider
    mock_service_manager.cleanup_mocks()

@pytest.fixture
def mock_risk_service():
    """Provide mock risk service."""
    service = mock_service_manager.setup_risk_service_mock()
    yield service
    mock_service_manager.cleanup_mocks()

# Test classes
class TestDatabaseFixtureManager:
    """Tests for database fixture manager."""
    
    @pytest.mark.requires_docker
    def test_postgres_container_setup(self):
        """Test PostgreSQL container setup."""
        manager = DatabaseFixtureManager(DatabaseConfig())
        
        container_info = manager.setup_postgres_container("test_postgres_unit")
        
        assert "container" in container_info
        assert "host" in container_info
        assert "port" in container_info
        assert "database" in container_info
        
        # Test connection
        with manager.get_sync_connection() as conn:
            assert conn is not None
        
        # Cleanup
        manager.cleanup_containers()
    
    def test_schema_management(self):
        """Test schema creation and cleanup."""
        manager = DatabaseFixtureManager(DatabaseConfig())
        
        schema_name = "test_schema_unit"
        
        # Create schema
        manager.create_schema(schema_name)
        assert schema_name in manager.schemas
        
        # Drop schema
        manager.drop_schema(schema_name)
        assert schema_name not in manager.schemas
    
    def test_test_table_setup(self):
        """Test test table creation."""
        manager = DatabaseFixtureManager(DatabaseConfig())
        
        schema_name = "test_tables_unit"
        manager.create_schema(schema_name)
        manager.setup_test_tables(schema_name)
        
        # Verify tables exist
        with manager.get_sync_connection(schema_name) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = %s
                """, (schema_name,))
                
                tables = [row[0] for row in cur.fetchall()]
                
                assert "market_data" in tables
                assert "risk_metrics" in tables
                assert "positions" in tables
                assert "orders" in tables
        
        # Cleanup
        manager.drop_schema(schema_name)

class TestMockExternalServiceManager:
    """Tests for mock external service manager."""
    
    def test_broker_mock_setup(self):
        """Test broker mock setup."""
        manager = MockExternalServiceManager()
        
        broker_mock = manager.setup_broker_mock("test_broker")
        
        assert broker_mock is not None
        assert broker_mock.connect() == True
        assert "order_id" in broker_mock.submit_order()
        assert len(broker_mock.get_positions()) > 0
        assert "cash" in broker_mock.get_account_balance()
    
    def test_market_data_mock_setup(self):
        """Test market data provider mock setup."""
        manager = MockExternalServiceManager()
        
        provider_mock = manager.setup_market_data_mock("test_provider")
        
        assert provider_mock is not None
        assert provider_mock.connect() == True
        assert "symbol" in provider_mock.get_quote()
        assert len(provider_mock.get_historical_data()) > 0
    
    def test_risk_service_mock_setup(self):
        """Test risk service mock setup."""
        manager = MockExternalServiceManager()
        
        service_mock = manager.setup_risk_service_mock("test_risk_service")
        
        assert service_mock is not None
        assert "var_95" in service_mock.calculate_var()
        assert "expected_shortfall" in service_mock.calculate_expected_shortfall()
        assert "overall_risk" in service_mock.assess_portfolio_risk()
    
    def test_response_override(self):
        """Test response override functionality."""
        manager = MockExternalServiceManager()
        
        broker_mock = manager.setup_broker_mock("test_broker")
        
        # Override response
        custom_response = {"custom": "response"}
        manager.override_response("test_broker", "submit_order", custom_response)
        
        assert broker_mock.submit_order() == custom_response
    
    def test_mock_cleanup(self):
        """Test mock cleanup."""
        manager = MockExternalServiceManager()
        
        manager.setup_broker_mock("test_broker")
        manager.setup_market_data_mock("test_provider")
        
        assert len(manager.active_mocks) == 2
        
        manager.cleanup_mocks()
        
        assert len(manager.active_mocks) == 0
        assert len(manager.response_overrides) == 0