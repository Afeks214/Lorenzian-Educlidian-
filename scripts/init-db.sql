-- GrandModel Database Initialization Script
-- Creates necessary tables and initial data for production deployment

-- Create database (if not exists)
CREATE DATABASE IF NOT EXISTS grandmodel;
USE grandmodel;

-- Create users table for authentication
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Create sessions table for user sessions
CREATE TABLE IF NOT EXISTS sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create trading_strategies table
CREATE TABLE IF NOT EXISTS trading_strategies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    strategy_type VARCHAR(100) NOT NULL,
    timeframe VARCHAR(50) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    configuration JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Create trading_positions table
CREATE TABLE IF NOT EXISTS trading_positions (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES trading_strategies(id),
    symbol VARCHAR(50) NOT NULL,
    position_type VARCHAR(20) NOT NULL, -- 'long' or 'short'
    quantity DECIMAL(20,8) NOT NULL,
    entry_price DECIMAL(20,8) NOT NULL,
    current_price DECIMAL(20,8),
    unrealized_pnl DECIMAL(20,8),
    realized_pnl DECIMAL(20,8) DEFAULT 0,
    status VARCHAR(20) DEFAULT 'open', -- 'open', 'closed', 'partial'
    opened_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP NULL,
    metadata JSON
);

-- Create trading_orders table
CREATE TABLE IF NOT EXISTS trading_orders (
    id SERIAL PRIMARY KEY,
    position_id INTEGER REFERENCES trading_positions(id),
    order_type VARCHAR(20) NOT NULL, -- 'market', 'limit', 'stop', 'stop_limit'
    side VARCHAR(10) NOT NULL, -- 'buy', 'sell'
    symbol VARCHAR(50) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    price DECIMAL(20,8),
    stop_price DECIMAL(20,8),
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'filled', 'cancelled', 'rejected'
    filled_quantity DECIMAL(20,8) DEFAULT 0,
    average_fill_price DECIMAL(20,8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    metadata JSON
);

-- Create risk_metrics table
CREATE TABLE IF NOT EXISTS risk_metrics (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES trading_strategies(id),
    metric_type VARCHAR(100) NOT NULL, -- 'var', 'sharpe', 'max_drawdown', etc.
    metric_value DECIMAL(20,8) NOT NULL,
    confidence_level DECIMAL(5,4), -- for VaR calculations
    time_horizon VARCHAR(20), -- '1d', '5d', '1m', etc.
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON
);

-- Create performance_metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES trading_strategies(id),
    date DATE NOT NULL,
    total_return DECIMAL(20,8),
    daily_return DECIMAL(20,8),
    cumulative_return DECIMAL(20,8),
    volatility DECIMAL(20,8),
    sharpe_ratio DECIMAL(20,8),
    max_drawdown DECIMAL(20,8),
    win_rate DECIMAL(5,4),
    profit_factor DECIMAL(20,8),
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    metadata JSON,
    UNIQUE(strategy_id, date)
);

-- Create system_events table for audit log
CREATE TABLE IF NOT EXISTS system_events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    event_source VARCHAR(100) NOT NULL,
    user_id INTEGER REFERENCES users(id),
    severity VARCHAR(20) NOT NULL, -- 'info', 'warning', 'error', 'critical'
    message TEXT NOT NULL,
    event_data JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create configuration table
CREATE TABLE IF NOT EXISTS system_configuration (
    id SERIAL PRIMARY KEY,
    config_key VARCHAR(255) UNIQUE NOT NULL,
    config_value JSON,
    is_sensitive BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON trading_positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_strategy ON trading_positions(strategy_id);
CREATE INDEX IF NOT EXISTS idx_positions_status ON trading_positions(status);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON trading_orders(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_status ON trading_orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_created ON trading_orders(created_at);
CREATE INDEX IF NOT EXISTS idx_risk_metrics_strategy ON risk_metrics(strategy_id);
CREATE INDEX IF NOT EXISTS idx_risk_metrics_type ON risk_metrics(metric_type);
CREATE INDEX IF NOT EXISTS idx_performance_date ON performance_metrics(date);
CREATE INDEX IF NOT EXISTS idx_performance_strategy ON performance_metrics(strategy_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON system_events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_source ON system_events(event_source);
CREATE INDEX IF NOT EXISTS idx_events_severity ON system_events(severity);
CREATE INDEX IF NOT EXISTS idx_events_created ON system_events(created_at);

-- Insert default admin user (password: admin123 - CHANGE IN PRODUCTION!)
INSERT INTO users (username, email, password_hash, is_admin) VALUES
('admin', 'admin@grandmodel.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LsKjmWOz9o3VLCmH6', TRUE)
ON DUPLICATE KEY UPDATE username = username;

-- Insert default system configuration
INSERT INTO system_configuration (config_key, config_value) VALUES
('system.initialized', 'true'),
('trading.max_positions', '10'),
('trading.default_risk_per_trade', '0.02'),
('risk.var_confidence_level', '0.95'),
('risk.max_drawdown_limit', '0.15'),
('performance.benchmark_symbol', 'SPY'),
('alerts.enabled', 'true'),
('maintenance.mode', 'false')
ON DUPLICATE KEY UPDATE config_value = VALUES(config_value);

-- Create initial trading strategies
INSERT INTO trading_strategies (name, strategy_type, timeframe, configuration) VALUES
('Strategic MARL 30m', 'marl', '30m', '{"agents": ["mlmi", "nwrqk", "fvg"], "risk_tolerance": 0.1}'),
('Tactical MARL 5m', 'marl', '5m', '{"agents": ["momentum", "mean_reversion"], "risk_tolerance": 0.05}'),
('Risk Management', 'risk', 'adaptive', '{"var_method": "historical", "correlation_tracking": true}')
ON DUPLICATE KEY UPDATE name = VALUES(name);

-- Log initialization
INSERT INTO system_events (event_type, event_source, severity, message) VALUES
('system.init', 'database', 'info', 'Database initialized successfully');

-- Grant permissions (adjust for your specific database setup)
-- GRANT ALL PRIVILEGES ON grandmodel.* TO 'grandmodel_user'@'%';
-- FLUSH PRIVILEGES;

COMMIT;