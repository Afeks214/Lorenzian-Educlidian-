"""
Broker Factory

Factory pattern implementation for creating and managing broker client instances.
Provides unified interface for connecting to multiple broker types.
"""

from typing import Dict, Any, Optional, Type, List
from enum import Enum
import structlog

from .base_broker import BaseBrokerClient, BrokerOrder, BrokerExecution, BrokerPosition
from .interactive_brokers import IBrokerClient
from .alpaca_client import AlpacaClient

logger = structlog.get_logger()


class BrokerType(Enum):
    """Supported broker types"""
    INTERACTIVE_BROKERS = "INTERACTIVE_BROKERS"
    ALPACA = "ALPACA"
    SIMULATED = "SIMULATED"


class SimulatedBrokerClient(BaseBrokerClient):
    """
    Simulated broker client for testing and development.
    
    Provides realistic order execution simulation without actual market connectivity.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.simulation_latency = config.get('simulation_latency_ms', 50)
        self.fill_probability = config.get('fill_probability', 0.98)
        self.slippage_bps = config.get('slippage_bps', 1.0)
    
    async def connect(self) -> bool:
        """Simulate connection"""
        import asyncio
        
        await self._update_connection_status(ConnectionStatus.CONNECTING)
        await asyncio.sleep(0.1)  # Simulate connection time
        await self._update_connection_status(ConnectionStatus.CONNECTED)
        
        # Start heartbeat
        await self._start_heartbeat()
        
        logger.info("Connected to simulated broker")
        return True
    
    async def disconnect(self) -> None:
        """Simulate disconnection"""
        await self._update_connection_status(ConnectionStatus.DISCONNECTED)
        logger.info("Disconnected from simulated broker")
    
    async def submit_order(self, order_data: Dict[str, Any]) -> BrokerOrder:
        """Simulate order submission"""
        import asyncio
        import random
        import uuid
        from datetime import datetime
        
        # Simulate submission latency
        await asyncio.sleep(self.simulation_latency / 1000)
        
        # Create simulated order
        broker_order = BrokerOrder(
            broker_order_id=str(uuid.uuid4()),
            client_order_id=order_data.get('client_order_id', f"SIM_{int(time.time())}"),
            symbol=order_data['symbol'],
            side=order_data['side'],
            quantity=order_data['quantity'],
            order_type=order_data['order_type'],
            price=order_data.get('price'),
            stop_price=order_data.get('stop_price'),
            time_in_force=order_data.get('time_in_force', 'DAY'),
            status=OrderStatus.SUBMITTED
        )
        
        self.orders[broker_order.broker_order_id] = broker_order
        
        # Simulate execution after delay
        asyncio.create_task(self._simulate_execution(broker_order))
        
        logger.info(f"Simulated order submitted: {broker_order.broker_order_id}")
        return broker_order
    
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Simulate order cancellation"""
        if broker_order_id in self.orders:
            order = self.orders[broker_order_id]
            order.status = OrderStatus.CANCELLED
            order.updated_at = datetime.now()
            await self._handle_order_update(order)
            logger.info(f"Simulated order cancelled: {broker_order_id}")
            return True
        return False
    
    async def modify_order(self, broker_order_id: str, modifications: Dict[str, Any]) -> bool:
        """Simulate order modification"""
        if broker_order_id in self.orders:
            order = self.orders[broker_order_id]
            
            if 'quantity' in modifications:
                order.quantity = modifications['quantity']
                order.remaining_quantity = order.quantity - order.filled_quantity
            
            if 'price' in modifications:
                order.price = modifications['price']
            
            order.updated_at = datetime.now()
            await self._handle_order_update(order)
            logger.info(f"Simulated order modified: {broker_order_id}")
            return True
        return False
    
    async def get_order_status(self, broker_order_id: str) -> Optional[BrokerOrder]:
        """Get simulated order status"""
        return self.orders.get(broker_order_id)
    
    async def get_positions(self) -> List[BrokerPosition]:
        """Get simulated positions"""
        return list(self.positions.values())
    
    async def get_account_info(self) -> BrokerAccount:
        """Get simulated account info"""
        if not self.account:
            self.account = BrokerAccount(
                account_id="SIMULATED_ACCOUNT",
                total_value=1000000.0,
                available_cash=500000.0,
                buying_power=1000000.0,
                day_trading_buying_power=2000000.0,
                maintenance_margin=50000.0,
                initial_margin=100000.0
            )
        return self.account
    
    async def _simulate_execution(self, order: BrokerOrder) -> None:
        """Simulate order execution"""
        import asyncio
        import random
        from datetime import datetime
        
        # Wait for execution delay
        execution_delay = random.uniform(0.1, 2.0)  # 100ms to 2s
        await asyncio.sleep(execution_delay)
        
        # Check if order was cancelled
        if order.status == OrderStatus.CANCELLED:
            return
        
        # Simulate fill probability
        if random.random() > self.fill_probability:
            # Order rejected
            order.status = OrderStatus.REJECTED
            order.updated_at = datetime.now()
            await self._handle_order_update(order)
            return
        
        # Calculate fill price with slippage
        base_price = order.price or 150.0  # Default price for market orders
        slippage = random.uniform(-self.slippage_bps/10000, self.slippage_bps/10000)
        
        if order.side == 'BUY':
            fill_price = base_price * (1 + abs(slippage))
        else:
            fill_price = base_price * (1 - abs(slippage))
        
        # Create execution
        execution = BrokerExecution(
            execution_id=f"SIM_EXEC_{int(time.time())}",
            broker_order_id=order.broker_order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            timestamp=datetime.now(),
            commission=0.0,  # Simulated commission-free
            venue='SIMULATED'
        )
        
        # Update order
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.remaining_quantity = 0
        order.average_fill_price = fill_price
        order.updated_at = datetime.now()
        
        # Notify callbacks
        await self._handle_execution_report(execution)
        await self._handle_order_update(order)


class BrokerFactory:
    """
    Factory for creating broker client instances.
    
    Provides centralized broker client creation and configuration management
    with support for multiple broker types and connection pooling.
    """
    
    _broker_classes: Dict[BrokerType, Type[BaseBrokerClient]] = {
        BrokerType.INTERACTIVE_BROKERS: IBrokerClient,
        BrokerType.ALPACA: AlpacaClient,
        BrokerType.SIMULATED: SimulatedBrokerClient
    }
    
    _active_clients: Dict[str, BaseBrokerClient] = {}
    
    @classmethod
    def create_broker_client(
        cls,
        broker_type: BrokerType,
        config: Dict[str, Any],
        client_id: Optional[str] = None
    ) -> BaseBrokerClient:
        """
        Create broker client instance.
        
        Args:
            broker_type: Type of broker to create
            config: Broker configuration
            client_id: Optional client identifier for connection pooling
            
        Returns:
            Configured broker client instance
        """
        
        if broker_type not in cls._broker_classes:
            raise ValueError(f"Unsupported broker type: {broker_type}")
        
        # Use existing client if available and requested
        if client_id and client_id in cls._active_clients:
            existing_client = cls._active_clients[client_id]
            if isinstance(existing_client, cls._broker_classes[broker_type]):
                logger.info(f"Reusing existing broker client: {client_id}")
                return existing_client
        
        # Create new client
        broker_class = cls._broker_classes[broker_type]
        client = broker_class(config)
        
        # Store for reuse if client_id provided
        if client_id:
            cls._active_clients[client_id] = client
        
        logger.info(
            f"Created broker client",
            broker_type=broker_type.value,
            client_id=client_id
        )
        
        return client
    
    @classmethod
    def create_interactive_brokers_client(
        cls,
        config: Dict[str, Any],
        client_id: Optional[str] = None
    ) -> IBrokerClient:
        """Create Interactive Brokers client with type safety"""
        client = cls.create_broker_client(BrokerType.INTERACTIVE_BROKERS, config, client_id)
        return client  # Type checker knows this is IBrokerClient
    
    @classmethod
    def create_alpaca_client(
        cls,
        config: Dict[str, Any],
        client_id: Optional[str] = None
    ) -> AlpacaClient:
        """Create Alpaca client with type safety"""
        client = cls.create_broker_client(BrokerType.ALPACA, config, client_id)
        return client  # Type checker knows this is AlpacaClient
    
    @classmethod
    def create_simulated_client(
        cls,
        config: Dict[str, Any] = None,
        client_id: Optional[str] = None
    ) -> SimulatedBrokerClient:
        """Create simulated client for testing"""
        config = config or {}
        client = cls.create_broker_client(BrokerType.SIMULATED, config, client_id)
        return client  # Type checker knows this is SimulatedBrokerClient
    
    @classmethod
    def create_from_config_file(
        cls,
        config_file: str,
        broker_name: str
    ) -> BaseBrokerClient:
        """
        Create broker client from configuration file.
        
        Args:
            config_file: Path to configuration file
            broker_name: Name of broker configuration section
            
        Returns:
            Configured broker client
        """
        import yaml
        import json
        
        try:
            # Load configuration file
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                with open(config_file, 'r') as f:
                    full_config = yaml.safe_load(f)
            elif config_file.endswith('.json'):
                with open(config_file, 'r') as f:
                    full_config = json.load(f)
            else:
                raise ValueError("Unsupported config file format. Use .yaml, .yml, or .json")
            
            # Extract broker configuration
            broker_config = full_config.get('brokers', {}).get(broker_name)
            if not broker_config:
                raise ValueError(f"Broker configuration '{broker_name}' not found")
            
            # Get broker type
            broker_type_str = broker_config.get('type', '').upper()
            try:
                broker_type = BrokerType(broker_type_str)
            except ValueError:
                raise ValueError(f"Invalid broker type: {broker_type_str}")
            
            # Create client
            return cls.create_broker_client(
                broker_type,
                broker_config.get('config', {}),
                broker_name
            )
            
        except Exception as e:
            logger.error(f"Failed to create broker client from config: {str(e)}")
            raise
    
    @classmethod
    def get_supported_brokers(cls) -> List[str]:
        """Get list of supported broker types"""
        return [broker_type.value for broker_type in cls._broker_classes.keys()]
    
    @classmethod
    def get_active_clients(cls) -> Dict[str, str]:
        """Get information about active clients"""
        return {
            client_id: client.__class__.__name__
            for client_id, client in cls._active_clients.items()
        }
    
    @classmethod
    async def connect_all_clients(cls) -> Dict[str, bool]:
        """Connect all active clients"""
        connection_results = {}
        
        for client_id, client in cls._active_clients.items():
            try:
                success = await client.connect()
                connection_results[client_id] = success
            except Exception as e:
                logger.error(f"Failed to connect client {client_id}: {str(e)}")
                connection_results[client_id] = False
        
        return connection_results
    
    @classmethod
    async def disconnect_all_clients(cls) -> None:
        """Disconnect all active clients"""
        for client_id, client in cls._active_clients.items():
            try:
                await client.disconnect()
            except Exception as e:
                logger.error(f"Failed to disconnect client {client_id}: {str(e)}")
    
    @classmethod
    async def shutdown_all_clients(cls) -> None:
        """Shutdown and cleanup all active clients"""
        for client_id, client in cls._active_clients.items():
            try:
                await client.shutdown()
            except Exception as e:
                logger.error(f"Failed to shutdown client {client_id}: {str(e)}")
        
        cls._active_clients.clear()
        logger.info("All broker clients shutdown")
    
    @classmethod
    def remove_client(cls, client_id: str) -> bool:
        """Remove client from active clients pool"""
        if client_id in cls._active_clients:
            del cls._active_clients[client_id]
            logger.info(f"Removed client: {client_id}")
            return True
        return False
    
    @classmethod
    def get_client_health_status(cls) -> Dict[str, Dict[str, Any]]:
        """Get health status of all active clients"""
        health_status = {}
        
        for client_id, client in cls._active_clients.items():
            try:
                connection_info = client.get_connection_info()
                performance_metrics = client.get_performance_metrics()
                
                health_status[client_id] = {
                    'broker_type': client.__class__.__name__,
                    'connection_status': connection_info.get('status'),
                    'connected_at': connection_info.get('connected_at'),
                    'last_heartbeat': connection_info.get('last_heartbeat'),
                    'connection_latency_ms': connection_info.get('connection_latency_ms'),
                    'total_orders': performance_metrics.get('current_state', {}).get('total_orders', 0),
                    'active_orders': performance_metrics.get('current_state', {}).get('active_orders', 0)
                }
                
            except Exception as e:
                health_status[client_id] = {
                    'broker_type': client.__class__.__name__,
                    'error': str(e),
                    'status': 'ERROR'
                }
        
        return health_status
    
    @classmethod
    def validate_broker_config(cls, broker_type: BrokerType, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate broker configuration.
        
        Returns validation result with required/missing fields.
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'required_fields': [],
            'optional_fields': []
        }
        
        if broker_type == BrokerType.INTERACTIVE_BROKERS:
            required_fields = ['host', 'port', 'client_id']
            optional_fields = ['timeout', 'auto_reconnect']
            
        elif broker_type == BrokerType.ALPACA:
            required_fields = ['api_key', 'api_secret']
            optional_fields = ['base_url', 'data_url', 'websocket_url']
            
        elif broker_type == BrokerType.SIMULATED:
            required_fields = []
            optional_fields = ['simulation_latency_ms', 'fill_probability', 'slippage_bps']
            
        else:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Unknown broker type: {broker_type}")
            return validation_result
        
        # Check required fields
        for field in required_fields:
            if field not in config:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Missing required field: {field}")
        
        # Check for sensitive fields in logs
        sensitive_fields = ['api_secret', 'password', 'private_key']
        for field in sensitive_fields:
            if field in config:
                validation_result['warnings'].append(f"Sensitive field detected: {field}")
        
        validation_result['required_fields'] = required_fields
        validation_result['optional_fields'] = optional_fields
        
        return validation_result