# 🔌 GRANDMODEL API DOCUMENTATION
**COMPREHENSIVE API REFERENCE & INTEGRATION GUIDE**

---

## 📋 DOCUMENT OVERVIEW

**Document Purpose**: Complete API documentation for all GrandModel system interfaces  
**Target Audience**: Developers, integrators, and API consumers  
**Classification**: TECHNICAL REFERENCE  
**Version**: 1.0  
**Last Updated**: July 17, 2025  
**Agent**: Documentation & Training Agent (Agent 9)

---

## 🎯 API OVERVIEW

The GrandModel system provides comprehensive APIs for integration with external systems, real-time data access, and system management. All APIs follow RESTful principles with JSON payloads and standard HTTP status codes.

### API Base URLs
- **Production**: `https://api.grandmodel.quantnova.com/v1`
- **Staging**: `https://api-staging.grandmodel.quantnova.com/v1`
- **Development**: `https://api-dev.grandmodel.quantnova.com/v1`

### API Versioning
- **Current Version**: v1
- **Versioning Strategy**: URL-based versioning (`/v1/`, `/v2/`)
- **Backward Compatibility**: Maintained for 12 months

### Authentication
All APIs require JWT-based authentication:
```http
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

---

## 🏗️ CORE SYSTEM APIS

### 1. System Status API

#### GET /api/v1/system/status
**Purpose**: Retrieve overall system health and status

**Request**:
```http
GET /api/v1/system/status
Authorization: Bearer <jwt_token>
```

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-07-17T10:30:00.000Z",
  "version": "1.0.0",
  "uptime": "72h 30m 15s",
  "components": {
    "kernel": {
      "status": "active",
      "health": "healthy",
      "last_check": "2025-07-17T10:29:45.000Z"
    },
    "event_bus": {
      "status": "active",
      "health": "healthy",
      "subscribers": 47,
      "events_processed": 1250000
    },
    "database": {
      "status": "active",
      "health": "healthy",
      "connections": 25,
      "response_time_ms": 12
    }
  },
  "performance": {
    "cpu_usage": 45.2,
    "memory_usage": 62.8,
    "network_io_mbps": 125.3,
    "disk_io_mbps": 45.7
  }
}
```

**Status Codes**:
- `200 OK`: System status retrieved successfully
- `401 Unauthorized`: Invalid or missing authentication
- `503 Service Unavailable`: System is unhealthy

---

### 2. Configuration Management API

#### GET /api/v1/config/environments
**Purpose**: Retrieve available configuration environments

**Request**:
```http
GET /api/v1/config/environments
Authorization: Bearer <jwt_token>
```

**Response**:
```json
{
  "environments": [
    {
      "name": "production",
      "description": "Production environment",
      "active": true,
      "last_updated": "2025-07-17T08:00:00.000Z"
    },
    {
      "name": "staging",
      "description": "Staging environment",
      "active": true,
      "last_updated": "2025-07-17T09:15:00.000Z"
    }
  ]
}
```

#### GET /api/v1/config/{environment}
**Purpose**: Retrieve configuration for specific environment

**Parameters**:
- `environment`: Environment name (production, staging, development)

**Request**:
```http
GET /api/v1/config/production
Authorization: Bearer <jwt_token>
```

**Response**:
```json
{
  "environment": "production",
  "config": {
    "system": {
      "mode": "production",
      "log_level": "INFO",
      "max_memory_gb": 16
    },
    "agents": {
      "strategic_marl": {
        "enabled": true,
        "model_path": "models/strategic_marl.pt",
        "confidence_threshold": 0.7
      },
      "tactical_marl": {
        "enabled": true,
        "model_path": "models/tactical_marl.pt",
        "execution_timeout": 5
      }
    },
    "data_handler": {
      "type": "rithmic",
      "connection_timeout": 30,
      "retry_attempts": 3
    }
  }
}
```

#### PUT /api/v1/config/{environment}
**Purpose**: Update configuration for specific environment

**Request**:
```http
PUT /api/v1/config/production
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "agents": {
    "strategic_marl": {
      "confidence_threshold": 0.75
    }
  }
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Configuration updated successfully",
  "updated_at": "2025-07-17T10:30:00.000Z",
  "validation_result": {
    "valid": true,
    "warnings": [],
    "errors": []
  }
}
```

---

## 🤖 AGENT MANAGEMENT APIS

### 1. Agent Status API

#### GET /api/v1/agents/status
**Purpose**: Retrieve status of all agents

**Request**:
```http
GET /api/v1/agents/status
Authorization: Bearer <jwt_token>
```

**Response**:
```json
{
  "agents": {
    "strategic_marl": {
      "status": "active",
      "health": "healthy",
      "model_loaded": true,
      "last_decision": "2025-07-17T10:29:30.000Z",
      "decisions_today": 847,
      "average_confidence": 0.78,
      "performance_metrics": {
        "success_rate": 0.85,
        "average_response_time_ms": 45
      }
    },
    "tactical_marl": {
      "status": "active",
      "health": "healthy",
      "model_loaded": true,
      "last_execution": "2025-07-17T10:29:45.000Z",
      "executions_today": 156,
      "average_latency_ms": 12,
      "performance_metrics": {
        "success_rate": 0.94,
        "average_slippage_bps": 2.3
      }
    },
    "risk_management": {
      "status": "active",
      "health": "healthy",
      "risk_checks_today": 2341,
      "violations_today": 3,
      "current_var": 0.015,
      "portfolio_exposure": 0.65
    }
  }
}
```

#### GET /api/v1/agents/{agent_name}/status
**Purpose**: Retrieve detailed status for specific agent

**Parameters**:
- `agent_name`: Agent identifier (strategic_marl, tactical_marl, risk_management, execution)

**Request**:
```http
GET /api/v1/agents/strategic_marl/status
Authorization: Bearer <jwt_token>
```

**Response**:
```json
{
  "agent": "strategic_marl",
  "status": "active",
  "health": "healthy",
  "model_info": {
    "model_path": "models/strategic_marl.pt",
    "model_version": "1.2.3",
    "loaded_at": "2025-07-17T08:00:00.000Z",
    "parameters": 1250000,
    "training_date": "2025-07-15T00:00:00.000Z"
  },
  "performance": {
    "decisions_today": 847,
    "average_confidence": 0.78,
    "success_rate": 0.85,
    "average_response_time_ms": 45,
    "last_decision": {
      "timestamp": "2025-07-17T10:29:30.000Z",
      "confidence": 0.82,
      "action": "BUY",
      "symbol": "NQ",
      "quantity": 10
    }
  },
  "configuration": {
    "confidence_threshold": 0.7,
    "max_position_size": 100,
    "risk_limit": 0.02,
    "update_frequency": 30
  }
}
```

---

### 2. Agent Control API

#### POST /api/v1/agents/{agent_name}/start
**Purpose**: Start a specific agent

**Request**:
```http
POST /api/v1/agents/strategic_marl/start
Authorization: Bearer <jwt_token>
```

**Response**:
```json
{
  "status": "success",
  "message": "Agent started successfully",
  "agent": "strategic_marl",
  "started_at": "2025-07-17T10:30:00.000Z"
}
```

#### POST /api/v1/agents/{agent_name}/stop
**Purpose**: Stop a specific agent

**Request**:
```http
POST /api/v1/agents/strategic_marl/stop
Authorization: Bearer <jwt_token>
```

**Response**:
```json
{
  "status": "success",
  "message": "Agent stopped successfully",
  "agent": "strategic_marl",
  "stopped_at": "2025-07-17T10:30:00.000Z"
}
```

#### POST /api/v1/agents/{agent_name}/restart
**Purpose**: Restart a specific agent

**Request**:
```http
POST /api/v1/agents/strategic_marl/restart
Authorization: Bearer <jwt_token>
```

**Response**:
```json
{
  "status": "success",
  "message": "Agent restarted successfully",
  "agent": "strategic_marl",
  "restarted_at": "2025-07-17T10:30:00.000Z"
}
```

#### POST /api/v1/agents/{agent_name}/configure
**Purpose**: Update agent configuration

**Request**:
```http
POST /api/v1/agents/strategic_marl/configure
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "confidence_threshold": 0.8,
  "max_position_size": 150,
  "risk_limit": 0.018
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Agent configuration updated",
  "agent": "strategic_marl",
  "updated_at": "2025-07-17T10:30:00.000Z",
  "applied_config": {
    "confidence_threshold": 0.8,
    "max_position_size": 150,
    "risk_limit": 0.018
  }
}
```

---

## 📊 PERFORMANCE MONITORING APIS

### 1. Performance Metrics API

#### GET /api/v1/performance/metrics
**Purpose**: Retrieve comprehensive performance metrics

**Query Parameters**:
- `timeframe`: Time period (1h, 1d, 1w, 1m)
- `metric_type`: Metric category (returns, risk, execution, system)
- `agent`: Specific agent (optional)

**Request**:
```http
GET /api/v1/performance/metrics?timeframe=1d&metric_type=returns
Authorization: Bearer <jwt_token>
```

**Response**:
```json
{
  "timeframe": "1d",
  "metric_type": "returns",
  "period": {
    "start": "2025-07-16T10:30:00.000Z",
    "end": "2025-07-17T10:30:00.000Z"
  },
  "metrics": {
    "total_return": 0.0245,
    "annualized_return": 0.324,
    "volatility": 0.156,
    "sharpe_ratio": 2.08,
    "max_drawdown": 0.0123,
    "win_rate": 0.67,
    "profit_factor": 1.85,
    "trades_count": 47,
    "average_trade_pnl": 523.45,
    "largest_win": 2145.67,
    "largest_loss": -856.34
  },
  "daily_returns": [
    {
      "date": "2025-07-16",
      "return": 0.0132,
      "trades": 23,
      "pnl": 12456.78
    },
    {
      "date": "2025-07-17",
      "return": 0.0113,
      "trades": 24,
      "pnl": 10678.23
    }
  ]
}
```

#### GET /api/v1/performance/risk-metrics
**Purpose**: Retrieve risk-specific metrics

**Request**:
```http
GET /api/v1/performance/risk-metrics
Authorization: Bearer <jwt_token>
```

**Response**:
```json
{
  "risk_metrics": {
    "var_95": 0.0234,
    "var_99": 0.0456,
    "expected_shortfall": 0.0567,
    "max_drawdown": 0.0123,
    "current_drawdown": 0.0045,
    "portfolio_beta": 1.12,
    "correlation_with_market": 0.78,
    "leverage_ratio": 1.45,
    "concentration_risk": 0.25,
    "liquidity_risk": 0.15
  },
  "position_risk": {
    "total_exposure": 1250000,
    "net_exposure": 875000,
    "gross_exposure": 1625000,
    "largest_position": 125000,
    "position_count": 12,
    "average_position_size": 104166.67
  },
  "risk_limits": {
    "var_limit": 0.03,
    "max_drawdown_limit": 0.05,
    "leverage_limit": 2.0,
    "concentration_limit": 0.3,
    "single_position_limit": 0.1
  }
}
```

---

### 2. Trading Performance API

#### GET /api/v1/performance/trades
**Purpose**: Retrieve trading performance data

**Query Parameters**:
- `start_date`: Start date (YYYY-MM-DD)
- `end_date`: End date (YYYY-MM-DD)
- `symbol`: Trading symbol (optional)
- `agent`: Agent name (optional)
- `limit`: Number of records (default: 100)

**Request**:
```http
GET /api/v1/performance/trades?start_date=2025-07-16&end_date=2025-07-17&limit=10
Authorization: Bearer <jwt_token>
```

**Response**:
```json
{
  "trades": [
    {
      "trade_id": "TRD_20250717_001",
      "symbol": "NQ",
      "side": "BUY",
      "quantity": 10,
      "entry_price": 18450.25,
      "exit_price": 18465.75,
      "entry_time": "2025-07-17T09:15:00.000Z",
      "exit_time": "2025-07-17T09:45:00.000Z",
      "pnl": 1550.00,
      "commission": 12.50,
      "slippage": 0.25,
      "duration_minutes": 30,
      "agent": "strategic_marl",
      "strategy": "momentum_breakout",
      "confidence": 0.85
    },
    {
      "trade_id": "TRD_20250717_002",
      "symbol": "NQ",
      "side": "SELL",
      "quantity": 15,
      "entry_price": 18440.50,
      "exit_price": 18435.25,
      "entry_time": "2025-07-17T10:00:00.000Z",
      "exit_time": "2025-07-17T10:20:00.000Z",
      "pnl": -787.50,
      "commission": 18.75,
      "slippage": 0.50,
      "duration_minutes": 20,
      "agent": "tactical_marl",
      "strategy": "mean_reversion",
      "confidence": 0.72
    }
  ],
  "summary": {
    "total_trades": 47,
    "winning_trades": 31,
    "losing_trades": 16,
    "win_rate": 0.659,
    "total_pnl": 23456.78,
    "average_pnl": 498.87,
    "total_commission": 587.50,
    "average_slippage": 0.35
  }
}
```

---

## 🔍 MARKET DATA APIS

### 1. Market Data API

#### GET /api/v1/market/quotes/{symbol}
**Purpose**: Retrieve real-time market quotes

**Parameters**:
- `symbol`: Trading symbol (e.g., NQ, ES, CL)

**Request**:
```http
GET /api/v1/market/quotes/NQ
Authorization: Bearer <jwt_token>
```

**Response**:
```json
{
  "symbol": "NQ",
  "timestamp": "2025-07-17T10:30:00.000Z",
  "bid": 18445.25,
  "ask": 18445.75,
  "last": 18445.50,
  "volume": 1250000,
  "change": 125.25,
  "change_percent": 0.68,
  "high": 18456.75,
  "low": 18420.00,
  "open": 18430.25,
  "previous_close": 18320.25,
  "market_session": "regular",
  "last_update": "2025-07-17T10:29:59.876Z"
}
```

#### GET /api/v1/market/bars/{symbol}
**Purpose**: Retrieve historical bar data

**Parameters**:
- `symbol`: Trading symbol
- `timeframe`: Bar timeframe (1m, 5m, 15m, 30m, 1h, 1d)
- `start`: Start timestamp
- `end`: End timestamp
- `limit`: Number of bars (default: 100)

**Request**:
```http
GET /api/v1/market/bars/NQ?timeframe=30m&limit=50
Authorization: Bearer <jwt_token>
```

**Response**:
```json
{
  "symbol": "NQ",
  "timeframe": "30m",
  "bars": [
    {
      "timestamp": "2025-07-17T10:00:00.000Z",
      "open": 18440.25,
      "high": 18456.75,
      "low": 18435.50,
      "close": 18445.25,
      "volume": 125000
    },
    {
      "timestamp": "2025-07-17T10:30:00.000Z",
      "open": 18445.50,
      "high": 18452.25,
      "low": 18441.75,
      "close": 18448.00,
      "volume": 118000
    }
  ],
  "count": 50,
  "has_more": true
}
```

---

### 2. Indicators API

#### GET /api/v1/indicators/{symbol}
**Purpose**: Retrieve technical indicators

**Parameters**:
- `symbol`: Trading symbol
- `indicators`: Comma-separated list of indicators (mlmi, fvg, nwrqk, lvn, mmd)
- `timeframe`: Analysis timeframe (5m, 30m, 1h, 1d)

**Request**:
```http
GET /api/v1/indicators/NQ?indicators=mlmi,fvg&timeframe=30m
Authorization: Bearer <jwt_token>
```

**Response**:
```json
{
  "symbol": "NQ",
  "timeframe": "30m",
  "timestamp": "2025-07-17T10:30:00.000Z",
  "indicators": {
    "mlmi": {
      "value": 0.745,
      "signal": "bullish",
      "strength": 0.82,
      "last_update": "2025-07-17T10:30:00.000Z"
    },
    "fvg": {
      "zones": [
        {
          "type": "bullish",
          "top": 18455.50,
          "bottom": 18450.25,
          "created_at": "2025-07-17T09:30:00.000Z",
          "status": "active",
          "strength": 0.68
        }
      ],
      "active_zones": 3,
      "last_update": "2025-07-17T10:30:00.000Z"
    }
  }
}
```

---

## 🎯 TRADING APIS

### 1. Order Management API

#### POST /api/v1/orders
**Purpose**: Submit a new order

**Request**:
```http
POST /api/v1/orders
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "symbol": "NQ",
  "side": "BUY",
  "quantity": 10,
  "order_type": "MARKET",
  "time_in_force": "DAY",
  "agent": "strategic_marl",
  "strategy": "momentum_breakout",
  "risk_parameters": {
    "stop_loss": 18400.00,
    "take_profit": 18500.00,
    "max_slippage": 0.5
  }
}
```

**Response**:
```json
{
  "order_id": "ORD_20250717_001",
  "status": "submitted",
  "submitted_at": "2025-07-17T10:30:00.000Z",
  "symbol": "NQ",
  "side": "BUY",
  "quantity": 10,
  "order_type": "MARKET",
  "estimated_price": 18445.50,
  "agent": "strategic_marl",
  "strategy": "momentum_breakout"
}
```

#### GET /api/v1/orders/{order_id}
**Purpose**: Retrieve order status

**Request**:
```http
GET /api/v1/orders/ORD_20250717_001
Authorization: Bearer <jwt_token>
```

**Response**:
```json
{
  "order_id": "ORD_20250717_001",
  "status": "filled",
  "symbol": "NQ",
  "side": "BUY",
  "quantity": 10,
  "filled_quantity": 10,
  "average_price": 18445.75,
  "submitted_at": "2025-07-17T10:30:00.000Z",
  "filled_at": "2025-07-17T10:30:02.345Z",
  "agent": "strategic_marl",
  "strategy": "momentum_breakout",
  "commission": 12.50,
  "slippage": 0.25,
  "execution_quality": {
    "implementation_shortfall": 0.15,
    "market_impact": 0.08,
    "timing_cost": 0.07
  }
}
```

#### DELETE /api/v1/orders/{order_id}
**Purpose**: Cancel an order

**Request**:
```http
DELETE /api/v1/orders/ORD_20250717_001
Authorization: Bearer <jwt_token>
```

**Response**:
```json
{
  "order_id": "ORD_20250717_001",
  "status": "cancelled",
  "cancelled_at": "2025-07-17T10:30:00.000Z",
  "message": "Order cancelled successfully"
}
```

---

### 2. Position Management API

#### GET /api/v1/positions
**Purpose**: Retrieve current positions

**Request**:
```http
GET /api/v1/positions
Authorization: Bearer <jwt_token>
```

**Response**:
```json
{
  "positions": [
    {
      "symbol": "NQ",
      "quantity": 25,
      "side": "LONG",
      "average_price": 18445.25,
      "current_price": 18450.75,
      "unrealized_pnl": 1375.00,
      "realized_pnl": 0.00,
      "total_pnl": 1375.00,
      "position_value": 461250.00,
      "margin_used": 92250.00,
      "open_time": "2025-07-17T09:15:00.000Z",
      "agent": "strategic_marl",
      "strategy": "momentum_breakout"
    }
  ],
  "summary": {
    "total_positions": 3,
    "total_value": 1250000.00,
    "total_pnl": 23456.78,
    "total_margin_used": 250000.00,
    "available_margin": 750000.00
  }
}
```

#### POST /api/v1/positions/{symbol}/close
**Purpose**: Close a position

**Request**:
```http
POST /api/v1/positions/NQ/close
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "quantity": 10,
  "order_type": "MARKET",
  "reason": "risk_management"
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Position close order submitted",
  "order_id": "ORD_20250717_002",
  "symbol": "NQ",
  "quantity": 10,
  "submitted_at": "2025-07-17T10:30:00.000Z"
}
```

---

## 🔔 REAL-TIME APIS (WEBSOCKET)

### 1. WebSocket Connection
**Endpoint**: `wss://api.grandmodel.quantnova.com/v1/ws`

**Authentication**:
```javascript
const ws = new WebSocket('wss://api.grandmodel.quantnova.com/v1/ws');
ws.onopen = function() {
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'your_jwt_token'
  }));
};
```

### 2. Market Data Stream
**Purpose**: Real-time market data feed

**Subscribe**:
```javascript
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'market-data',
  symbols: ['NQ', 'ES', 'CL']
}));
```

**Data Format**:
```json
{
  "channel": "market-data",
  "type": "tick",
  "data": {
    "symbol": "NQ",
    "timestamp": "2025-07-17T10:30:00.123Z",
    "bid": 18445.25,
    "ask": 18445.75,
    "last": 18445.50,
    "volume": 125
  }
}
```

### 3. Trade Updates Stream
**Purpose**: Real-time trade execution updates

**Subscribe**:
```javascript
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'trade-updates'
}));
```

**Data Format**:
```json
{
  "channel": "trade-updates",
  "type": "execution",
  "data": {
    "order_id": "ORD_20250717_001",
    "symbol": "NQ",
    "side": "BUY",
    "quantity": 10,
    "price": 18445.75,
    "timestamp": "2025-07-17T10:30:00.345Z",
    "status": "filled"
  }
}
```

### 4. Risk Alerts Stream
**Purpose**: Real-time risk management alerts

**Subscribe**:
```javascript
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'risk-alerts'
}));
```

**Data Format**:
```json
{
  "channel": "risk-alerts",
  "type": "alert",
  "data": {
    "alert_id": "RISK_20250717_001",
    "severity": "warning",
    "type": "position_limit",
    "message": "Position size approaching limit",
    "symbol": "NQ",
    "current_position": 95,
    "limit": 100,
    "timestamp": "2025-07-17T10:30:00.000Z"
  }
}
```

---

## 📊 ANALYTICS APIS

### 1. Backtesting API

#### POST /api/v1/backtest
**Purpose**: Run backtesting analysis

**Request**:
```http
POST /api/v1/backtest
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "strategy": "momentum_breakout",
  "symbols": ["NQ"],
  "start_date": "2025-01-01",
  "end_date": "2025-06-30",
  "initial_capital": 100000,
  "parameters": {
    "confidence_threshold": 0.75,
    "max_position_size": 50,
    "risk_limit": 0.02
  }
}
```

**Response**:
```json
{
  "backtest_id": "BT_20250717_001",
  "status": "running",
  "started_at": "2025-07-17T10:30:00.000Z",
  "estimated_completion": "2025-07-17T10:45:00.000Z"
}
```

#### GET /api/v1/backtest/{backtest_id}
**Purpose**: Retrieve backtest results

**Request**:
```http
GET /api/v1/backtest/BT_20250717_001
Authorization: Bearer <jwt_token>
```

**Response**:
```json
{
  "backtest_id": "BT_20250717_001",
  "status": "completed",
  "results": {
    "total_return": 0.245,
    "annualized_return": 0.324,
    "volatility": 0.156,
    "sharpe_ratio": 2.08,
    "max_drawdown": 0.123,
    "win_rate": 0.67,
    "profit_factor": 1.85,
    "total_trades": 247,
    "winning_trades": 165,
    "losing_trades": 82
  },
  "daily_returns": [
    {
      "date": "2025-01-01",
      "return": 0.012,
      "pnl": 1200.00
    }
  ]
}
```

---

## 🔐 SECURITY APIS

### 1. Authentication API

#### POST /api/v1/auth/login
**Purpose**: Authenticate user and obtain JWT token

**Request**:
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "trader_001",
  "password": "secure_password",
  "mfa_token": "123456"
}
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "id": "user_001",
    "username": "trader_001",
    "roles": ["trader", "analyst"],
    "permissions": ["read_data", "execute_trades", "view_performance"]
  }
}
```

#### POST /api/v1/auth/refresh
**Purpose**: Refresh JWT token

**Request**:
```http
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

---

## 📋 ERROR HANDLING

### HTTP Status Codes
- `200 OK`: Request successful
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation errors
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

### Error Response Format
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "quantity",
      "reason": "must be greater than 0"
    },
    "timestamp": "2025-07-17T10:30:00.000Z",
    "request_id": "req_123456"
  }
}
```

### Common Error Codes
- `AUTHENTICATION_ERROR`: Invalid or expired token
- `VALIDATION_ERROR`: Request validation failed
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `RESOURCE_NOT_FOUND`: Requested resource not found
- `INSUFFICIENT_PERMISSIONS`: User lacks required permissions
- `SYSTEM_ERROR`: Internal system error
- `MARKET_CLOSED`: Market is closed for trading
- `INSUFFICIENT_MARGIN`: Insufficient margin for trade

---

## 🚀 INTEGRATION EXAMPLES

### Python SDK Example
```python
import requests
import json

class GrandModelAPI:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
    
    def get_system_status(self):
        response = requests.get(
            f'{self.base_url}/system/status',
            headers=self.headers
        )
        return response.json()
    
    def submit_order(self, symbol, side, quantity, order_type='MARKET'):
        order_data = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'order_type': order_type
        }
        response = requests.post(
            f'{self.base_url}/orders',
            headers=self.headers,
            json=order_data
        )
        return response.json()
    
    def get_performance_metrics(self, timeframe='1d'):
        response = requests.get(
            f'{self.base_url}/performance/metrics',
            headers=self.headers,
            params={'timeframe': timeframe}
        )
        return response.json()

# Usage
api = GrandModelAPI('https://api.grandmodel.quantnova.com/v1', 'your_token')
status = api.get_system_status()
print(f"System status: {status['status']}")
```

### JavaScript SDK Example
```javascript
class GrandModelAPI {
    constructor(baseUrl, token) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
        };
    }
    
    async getSystemStatus() {
        const response = await fetch(`${this.baseUrl}/system/status`, {
            headers: this.headers
        });
        return await response.json();
    }
    
    async submitOrder(symbol, side, quantity, orderType = 'MARKET') {
        const orderData = {
            symbol,
            side,
            quantity,
            order_type: orderType
        };
        
        const response = await fetch(`${this.baseUrl}/orders`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(orderData)
        });
        
        return await response.json();
    }
    
    async getPerformanceMetrics(timeframe = '1d') {
        const response = await fetch(`${this.baseUrl}/performance/metrics?timeframe=${timeframe}`, {
            headers: this.headers
        });
        return await response.json();
    }
}

// Usage
const api = new GrandModelAPI('https://api.grandmodel.quantnova.com/v1', 'your_token');
const status = await api.getSystemStatus();
console.log(`System status: ${status.status}`);
```

---

## 📊 RATE LIMITING

### Rate Limits
- **Authentication**: 10 requests per minute
- **Market Data**: 1000 requests per minute
- **Trading**: 100 requests per minute
- **Analytics**: 50 requests per minute
- **System Management**: 20 requests per minute

### Rate Limit Headers
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1626525600
```

### Rate Limit Exceeded Response
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "retry_after": 60,
    "limit": 1000,
    "remaining": 0,
    "reset": 1626525600
  }
}
```

---

## 🎯 CONCLUSION

This comprehensive API documentation provides complete integration guidance for the GrandModel system. The APIs are designed to be:

- **RESTful**: Follow standard REST principles
- **Secure**: JWT authentication and authorization
- **Scalable**: Handle high-frequency trading requirements
- **Comprehensive**: Cover all system functionality
- **Real-time**: WebSocket support for live data
- **Well-documented**: Complete examples and error handling

Regular updates ensure the APIs remain current with system enhancements and new features.

---

**Document Version**: 1.0  
**Last Updated**: July 17, 2025  
**Next Review**: July 24, 2025  
**Owner**: Documentation & Training Agent (Agent 9)  
**Classification**: TECHNICAL REFERENCE  

---

*This document serves as the definitive API reference for the GrandModel system, providing essential guidance for developers and integrators.*