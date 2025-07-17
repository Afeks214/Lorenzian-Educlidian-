"""
Alpaca Trading API Client

Integration with Alpaca Markets API for commission-free equity trading
with real-time market data and execution capabilities.
"""

import asyncio
import aiohttp
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import structlog

from .base_broker import (
    BaseBrokerClient, BrokerOrder, BrokerExecution, BrokerPosition, 
    BrokerAccount, ConnectionStatus, OrderStatus
)

logger = structlog.get_logger()


class AlpacaClient(BaseBrokerClient):
    """
    Alpaca Markets API client for commission-free stock trading.
    
    Provides integration with Alpaca's REST and WebSocket APIs for
    order execution, portfolio management, and real-time market data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Alpaca API configuration
        self.api_key = config.get('api_key', '')
        self.api_secret = config.get('api_secret', '')
        self.base_url = config.get('base_url', 'https://paper-api.alpaca.markets')
        self.data_url = config.get('data_url', 'https://data.alpaca.markets')
        self.websocket_url = config.get('websocket_url', 'wss://stream.data.alpaca.markets/v2')
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # WebSocket connection
        self.websocket: Optional[aiohttp.ClientWebSocketResponse] = None
        
        # Order tracking
        self.alpaca_orders: Dict[str, Dict[str, Any]] = {}
        
        # Market data subscriptions
        self.subscribed_symbols: set = set()
        
        logger.info("AlpacaClient initialized", base_url=self.base_url)
    
    async def connect(self) -> bool:
        """Connect to Alpaca API"""
        try:
            await self._update_connection_status(ConnectionStatus.CONNECTING)
            
            # Create HTTP session
            headers = {
                'APCA-API-KEY-ID': self.api_key,
                'APCA-API-SECRET-KEY': self.api_secret,
                'Content-Type': 'application/json'
            }
            
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout
            )
            
            # Test connection with account info request
            connection_start = time.perf_counter()
            account_info = await self._get_account()
            connection_time = (time.perf_counter() - connection_start) * 1000
            
            if account_info:
                await self._update_connection_status(ConnectionStatus.CONNECTED)
                self.connection.connection_latency_ms = connection_time
                
                # Start background tasks
                await self._start_heartbeat()
                await self._start_websocket_connection()
                
                self._update_performance_metrics('connection', True)
                
                logger.info(
                    "Connected to Alpaca API",
                    connection_time_ms=connection_time,
                    account_id=account_info.get('id', 'unknown')
                )
                
                return True
            else:
                await self._update_connection_status(
                    ConnectionStatus.ERROR,
                    "Failed to retrieve account information"
                )
                self._update_performance_metrics('connection', False)
                return False
                
        except Exception as e:
            error_msg = f"Connection error: {str(e)}"
            await self._update_connection_status(ConnectionStatus.ERROR, error_msg)
            await self._handle_error("connect", e)
            self._update_performance_metrics('connection', False)
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Alpaca API"""
        try:
            # Close WebSocket connection
            if self.websocket and not self.websocket.closed:
                await self.websocket.close()
            
            # Close HTTP session
            if self.session and not self.session.closed:
                await self.session.close()
            
            await self._update_connection_status(ConnectionStatus.DISCONNECTED)
            
            logger.info("Disconnected from Alpaca API")
            
        except Exception as e:
            await self._handle_error("disconnect", e)
    
    async def submit_order(self, order_data: Dict[str, Any]) -> BrokerOrder:
        """Submit order to Alpaca"""
        start_time = time.perf_counter()
        
        try:
            # Prepare order payload
            payload = {
                'symbol': order_data['symbol'],
                'qty': str(order_data['quantity']),
                'side': order_data['side'].lower(),
                'type': self._map_order_type(order_data['order_type']),
                'time_in_force': order_data.get('time_in_force', 'day').lower(),
                'client_order_id': order_data.get('client_order_id')
            }
            
            # Add price fields if applicable
            if order_data.get('price'):
                payload['limit_price'] = str(order_data['price'])
            
            if order_data.get('stop_price'):
                payload['stop_price'] = str(order_data['stop_price'])
            
            # Submit order via REST API
            async with self.session.post(f"{self.base_url}/v2/orders", json=payload) as response:
                if response.status == 201:
                    order_response = await response.json()
                    
                    # Create broker order object
                    broker_order = BrokerOrder(
                        broker_order_id=order_response['id'],
                        client_order_id=order_response.get('client_order_id', ''),
                        symbol=order_response['symbol'],
                        side=order_response['side'].upper(),
                        quantity=int(order_response['qty']),
                        order_type=self._reverse_map_order_type(order_response['order_type']),
                        price=float(order_response.get('limit_price', 0)) if order_response.get('limit_price') else None,
                        stop_price=float(order_response.get('stop_price', 0)) if order_response.get('stop_price') else None,
                        time_in_force=order_response['time_in_force'].upper(),
                        status=self._map_order_status(order_response['status'])
                    )
                    
                    # Store order
                    self.orders[broker_order.broker_order_id] = broker_order
                    self.alpaca_orders[broker_order.broker_order_id] = order_response
                    
                    # Track latency
                    submission_latency = (time.perf_counter() - start_time) * 1000
                    self._track_order_latency(submission_latency)
                    
                    # Update metrics
                    self._update_performance_metrics('order', True)
                    
                    logger.info(
                        "Order submitted to Alpaca",
                        order_id=broker_order.broker_order_id,
                        symbol=order_data['symbol'],
                        latency_ms=submission_latency
                    )
                    
                    return broker_order
                
                else:
                    error_text = await response.text()
                    raise Exception(f"Order submission failed: {response.status} - {error_text}")
                    
        except Exception as e:
            self._update_performance_metrics('order', False)
            await self._handle_error("submit_order", e)
            raise
    
    async def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel order at Alpaca"""
        try:
            async with self.session.delete(f"{self.base_url}/v2/orders/{broker_order_id}") as response:
                if response.status == 204:
                    # Update order status
                    if broker_order_id in self.orders:
                        order = self.orders[broker_order_id]
                        order.status = OrderStatus.CANCELLED
                        order.updated_at = datetime.now()
                        
                        await self._handle_order_update(order)
                    
                    logger.info("Order cancelled at Alpaca", order_id=broker_order_id)
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Order cancellation failed: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            await self._handle_error("cancel_order", e)
            return False
    
    async def modify_order(
        self,
        broker_order_id: str,
        modifications: Dict[str, Any]
    ) -> bool:
        """Modify order at Alpaca"""
        try:
            # Prepare modification payload
            payload = {}
            
            if 'quantity' in modifications:
                payload['qty'] = str(modifications['quantity'])
            
            if 'price' in modifications:
                payload['limit_price'] = str(modifications['price'])
            
            if 'stop_price' in modifications:
                payload['stop_price'] = str(modifications['stop_price'])
            
            if 'time_in_force' in modifications:
                payload['time_in_force'] = modifications['time_in_force'].lower()
            
            async with self.session.patch(
                f"{self.base_url}/v2/orders/{broker_order_id}",
                json=payload
            ) as response:
                if response.status == 200:
                    order_response = await response.json()
                    
                    # Update local order
                    if broker_order_id in self.orders:
                        order = self.orders[broker_order_id]
                        order.quantity = int(order_response['qty'])
                        order.price = float(order_response.get('limit_price', 0)) if order_response.get('limit_price') else None
                        order.stop_price = float(order_response.get('stop_price', 0)) if order_response.get('stop_price') else None
                        order.time_in_force = order_response['time_in_force'].upper()
                        order.updated_at = datetime.now()
                        
                        await self._handle_order_update(order)
                    
                    logger.info("Order modified at Alpaca", order_id=broker_order_id)
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Order modification failed: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            await self._handle_error("modify_order", e)
            return False
    
    async def get_order_status(self, broker_order_id: str) -> Optional[BrokerOrder]:
        """Get order status from Alpaca"""
        try:
            async with self.session.get(f"{self.base_url}/v2/orders/{broker_order_id}") as response:
                if response.status == 200:
                    order_data = await response.json()
                    
                    # Update local order if exists
                    if broker_order_id in self.orders:
                        order = self.orders[broker_order_id]
                        order.status = self._map_order_status(order_data['status'])
                        order.filled_quantity = int(order_data.get('filled_qty', 0))
                        order.remaining_quantity = order.quantity - order.filled_quantity
                        
                        if order_data.get('filled_avg_price'):
                            order.average_fill_price = float(order_data['filled_avg_price'])
                        
                        order.updated_at = datetime.now()
                        
                        return order
                        
            return self.orders.get(broker_order_id)
            
        except Exception as e:
            await self._handle_error("get_order_status", e)
            return None
    
    async def get_positions(self) -> List[BrokerPosition]:
        """Get current positions from Alpaca"""
        try:
            async with self.session.get(f"{self.base_url}/v2/positions") as response:
                if response.status == 200:
                    positions_data = await response.json()
                    
                    positions = []
                    for pos_data in positions_data:
                        position = BrokerPosition(
                            symbol=pos_data['symbol'],
                            quantity=int(pos_data['qty']),
                            average_cost=float(pos_data['avg_entry_price']),
                            market_value=float(pos_data['market_value']),
                            unrealized_pnl=float(pos_data['unrealized_pl']),
                            realized_pnl=0.0,  # Not provided by Alpaca positions endpoint
                            last_updated=datetime.now()
                        )
                        positions.append(position)
                        
                        # Update local cache
                        self.positions[position.symbol] = position
                    
                    return positions
                    
            return []
            
        except Exception as e:
            await self._handle_error("get_positions", e)
            return []
    
    async def get_account_info(self) -> BrokerAccount:
        """Get account information from Alpaca"""
        try:
            account_data = await self._get_account()
            
            if account_data:
                account = BrokerAccount(
                    account_id=account_data['id'],
                    total_value=float(account_data.get('portfolio_value', 0)),
                    available_cash=float(account_data.get('cash', 0)),
                    buying_power=float(account_data.get('buying_power', 0)),
                    day_trading_buying_power=float(account_data.get('daytrading_buying_power', 0)),
                    maintenance_margin=0.0,  # Not provided by Alpaca
                    initial_margin=0.0,  # Not provided by Alpaca
                    currency='USD'
                )
                
                self.account = account
                return account
                
        except Exception as e:
            await self._handle_error("get_account_info", e)
        
        # Return default account on error
        return BrokerAccount(
            account_id="ERROR",
            total_value=0.0,
            available_cash=0.0,
            buying_power=0.0,
            day_trading_buying_power=0.0,
            maintenance_margin=0.0,
            initial_margin=0.0
        )
    
    async def _get_account(self) -> Optional[Dict[str, Any]]:
        """Get account information"""
        try:
            async with self.session.get(f"{self.base_url}/v2/account") as response:
                if response.status == 200:
                    return await response.json()
            return None
        except Exception:
            return None
    
    def _map_order_type(self, order_type: str) -> str:
        """Map order type to Alpaca format"""
        mapping = {
            'MARKET': 'market',
            'LIMIT': 'limit',
            'STOP': 'stop',
            'STOP_LIMIT': 'stop_limit',
            'TRAIL': 'trailing_stop'
        }
        return mapping.get(order_type, 'market')
    
    def _reverse_map_order_type(self, alpaca_order_type: str) -> str:
        """Map Alpaca order type back to standard format"""
        mapping = {
            'market': 'MARKET',
            'limit': 'LIMIT',
            'stop': 'STOP',
            'stop_limit': 'STOP_LIMIT',
            'trailing_stop': 'TRAIL'
        }
        return mapping.get(alpaca_order_type, 'MARKET')
    
    def _map_order_status(self, alpaca_status: str) -> OrderStatus:
        """Map Alpaca order status to standard format"""
        mapping = {
            'new': OrderStatus.SUBMITTED,
            'accepted': OrderStatus.ACKNOWLEDGED,
            'partially_filled': OrderStatus.PARTIALLY_FILLED,
            'filled': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELLED,
            'rejected': OrderStatus.REJECTED,
            'pending_new': OrderStatus.PENDING,
            'pending_cancel': OrderStatus.ACKNOWLEDGED,
            'pending_replace': OrderStatus.ACKNOWLEDGED
        }
        return mapping.get(alpaca_status, OrderStatus.PENDING)
    
    async def _start_websocket_connection(self) -> None:
        """Start WebSocket connection for real-time updates"""
        if self.data_stream_task and not self.data_stream_task.done():
            return
        
        self.data_stream_task = asyncio.create_task(self._websocket_loop())
    
    async def _websocket_loop(self) -> None:
        """WebSocket connection loop"""
        while self.connection.status == ConnectionStatus.CONNECTED:
            try:
                # Connect to WebSocket
                headers = {
                    'APCA-API-KEY-ID': self.api_key,
                    'APCA-API-SECRET-KEY': self.api_secret
                }
                
                async with self.session.ws_connect(
                    self.websocket_url + '/trade',
                    headers=headers
                ) as websocket:
                    self.websocket = websocket
                    
                    # Send authentication message
                    auth_msg = {
                        'action': 'auth',
                        'key': self.api_key,
                        'secret': self.api_secret
                    }
                    await websocket.send_str(json.dumps(auth_msg))
                    
                    # Subscribe to trade updates
                    subscribe_msg = {
                        'action': 'listen',
                        'data': {
                            'streams': ['trade_updates']
                        }
                    }
                    await websocket.send_str(json.dumps(subscribe_msg))
                    
                    logger.info("WebSocket connection established")
                    
                    # Listen for messages
                    async for message in websocket:
                        if message.type == aiohttp.WSMsgType.TEXT:
                            await self._handle_websocket_message(message.data)
                        elif message.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket error: {websocket.exception()}")
                            break
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocket connection error: {str(e)}")
                await asyncio.sleep(5.0)  # Retry after delay
    
    async def _handle_websocket_message(self, message_data: str) -> None:
        """Handle WebSocket message"""
        try:
            message = json.loads(message_data)
            
            if message.get('stream') == 'trade_updates':
                data = message.get('data', {})
                event = data.get('event', '')
                
                if event in ['new', 'partial_fill', 'fill', 'canceled', 'rejected']:
                    await self._handle_trade_update(data)
                    
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {str(e)}")
    
    async def _handle_trade_update(self, update_data: Dict[str, Any]) -> None:
        """Handle trade update from WebSocket"""
        try:
            order_id = update_data.get('order', {}).get('id')
            
            if order_id and order_id in self.orders:
                order = self.orders[order_id]
                
                # Update order status
                event = update_data.get('event', '')
                if event == 'fill':
                    order.status = OrderStatus.FILLED
                elif event == 'partial_fill':
                    order.status = OrderStatus.PARTIALLY_FILLED
                elif event == 'canceled':
                    order.status = OrderStatus.CANCELLED
                elif event == 'rejected':
                    order.status = OrderStatus.REJECTED
                
                # Update fill information
                order_data = update_data.get('order', {})
                order.filled_quantity = int(order_data.get('filled_qty', 0))
                order.remaining_quantity = order.quantity - order.filled_quantity
                
                if order_data.get('filled_avg_price'):
                    order.average_fill_price = float(order_data['filled_avg_price'])
                
                order.updated_at = datetime.now()
                
                await self._handle_order_update(order)
                
                # Handle execution if this is a fill
                if event in ['fill', 'partial_fill']:
                    execution = BrokerExecution(
                        execution_id=f"{order_id}_{int(time.time())}",
                        broker_order_id=order_id,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=int(update_data.get('qty', 0)),
                        price=float(update_data.get('price', 0)),
                        timestamp=datetime.now(),
                        commission=0.0,  # Alpaca is commission-free
                        venue='ALPACA'
                    )
                    
                    await self._handle_execution_report(execution)
                    
        except Exception as e:
            logger.error(f"Error handling trade update: {str(e)}")
    
    async def _perform_heartbeat(self) -> bool:
        """Perform Alpaca-specific heartbeat"""
        try:
            # Simple account request to test connection
            account_data = await self._get_account()
            return account_data is not None
        except Exception as e:
            self.logger.error(f"Alpaca heartbeat failed: {e}")
            return False
    
    async def subscribe_to_market_data(self, symbols: List[str]) -> None:
        """Subscribe to real-time market data"""
        try:
            if self.websocket and not self.websocket.closed:
                subscribe_msg = {
                    'action': 'listen',
                    'data': {
                        'streams': [f'T.{symbol}' for symbol in symbols]  # Trade updates
                    }
                }
                await self.websocket.send_str(json.dumps(subscribe_msg))
                
                self.subscribed_symbols.update(symbols)
                
                logger.info(f"Subscribed to market data for {len(symbols)} symbols")
                
        except Exception as e:
            logger.error(f"Error subscribing to market data: {str(e)}")
    
    def get_broker_specific_info(self) -> Dict[str, Any]:
        """Get Alpaca-specific broker information"""
        return {
            'broker_type': 'Alpaca Markets',
            'connection_details': {
                'base_url': self.base_url,
                'data_url': self.data_url,
                'websocket_url': self.websocket_url
            },
            'api_version': 'v2',
            'supported_order_types': [
                'MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT', 'TRAIL'
            ],
            'supported_time_in_force': [
                'DAY', 'GTC', 'IOC', 'FOK'
            ],
            'features': {
                'commission_free': True,
                'fractional_shares': True,
                'crypto_trading': True,
                'paper_trading': True,
                'real_time_data': True,
                'extended_hours': True,
                'short_selling': False,  # Requires margin account
                'margin_trading': False,  # Requires approval
                'options_trading': False
            },
            'market_data': {
                'subscribed_symbols': list(self.subscribed_symbols),
                'websocket_connected': self.websocket is not None and not self.websocket.closed
            }
        }