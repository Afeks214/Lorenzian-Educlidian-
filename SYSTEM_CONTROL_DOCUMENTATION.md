# GrandModel System Control Web Interface and API Documentation

## Overview

This documentation describes the comprehensive system control interface and API endpoints built for the GrandModel trading system. The implementation provides secure, authenticated, and real-time control capabilities with enterprise-grade security features.

## Architecture Overview

The system consists of several key components:

1. **REST API Endpoints** - Core system control operations
2. **Web Dashboard** - Interactive control interface
3. **Status Monitoring** - Real-time system monitoring
4. **Authentication System** - Enhanced security with MFA
5. **Rate Limiting & Audit** - Security and compliance
6. **Real-time Updates** - WebSocket-based notifications

## Components

### 1. System Control API (`src/api/system_control_api.py`)

Core REST API for system control operations.

#### Key Features:
- System ON/OFF controls
- Emergency stop functionality
- Status monitoring
- Health checks
- Activity logging
- JWT-based authentication
- Rate limiting with slowapi

#### API Endpoints:

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/api/system/status` | Get system status | Yes |
| POST | `/api/system/on` | Turn system ON | Yes |
| POST | `/api/system/off` | Turn system OFF | Yes |
| POST | `/api/system/emergency` | Emergency stop | Yes |
| GET | `/api/system/health` | Health check | Yes |
| GET | `/api/system/logs` | Activity logs | Yes |

#### Example Usage:

```bash
# Get system status
curl -H "Authorization: Bearer <token>" http://localhost:8001/api/system/status

# Turn system ON
curl -X POST -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"action": "on", "reason": "Routine startup"}' \
  http://localhost:8001/api/system/on

# Emergency stop
curl -X POST -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"action": "emergency", "reason": "Security incident"}' \
  http://localhost:8001/api/system/emergency
```

### 2. Status Monitoring API (`src/api/system_status_endpoints.py`)

Comprehensive monitoring endpoints for system health and performance.

#### Key Features:
- Real-time system metrics
- Component health monitoring
- Performance analytics
- Alert management
- Historical data tracking
- WebSocket support

#### API Endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status/overview` | System overview |
| GET | `/api/status/components` | Component health |
| GET | `/api/status/performance` | Performance metrics |
| GET | `/api/status/alerts` | Active alerts |
| POST | `/api/status/alerts/{id}/acknowledge` | Acknowledge alert |
| GET | `/api/status/historical` | Historical data |
| WS | `/ws/status` | Real-time updates |

### 3. Web Dashboard (`src/dashboard/system_control_dashboard.py`)

Interactive web-based control interface built with Dash.

#### Key Features:
- Large ON/OFF toggle switch
- Real-time status indicators
- Component health dashboard
- Recent activity log
- Performance metrics display
- Emergency stop button
- Authentication interface
- Mobile-responsive design

#### Dashboard Components:

- **System Control Panel**: Main ON/OFF switch with emergency stop
- **Status Overview**: System uptime, health score, alerts
- **Component Health**: Individual component status grid
- **Performance Metrics**: CPU, memory, disk usage gauges
- **Active Alerts**: Real-time alert notifications
- **Activity Log**: Recent system operations

### 4. Enhanced Authentication (`src/api/system_control_auth.py`)

Enterprise-grade authentication with enhanced security features.

#### Key Features:
- Multi-factor authentication (MFA)
- IP whitelisting
- Session management
- Hardware token support
- Suspicious activity detection
- Role-based access control (RBAC)

#### Security Permissions:

| Permission | Description |
|------------|-------------|
| SYSTEM_START | Turn system ON |
| SYSTEM_STOP | Turn system OFF |
| EMERGENCY_STOP | Emergency stop |
| SYSTEM_STATUS | View system status |
| COMPONENT_CONTROL | Control components |
| MAINTENANCE_MODE | Maintenance operations |
| AUDIT_ACCESS | Access audit logs |

### 5. Rate Limiting & Audit (`src/api/rate_limiting_audit.py`)

Advanced rate limiting and comprehensive audit logging.

#### Key Features:
- Multi-tier rate limiting (user, IP, endpoint)
- Adaptive rate limiting based on system load
- Comprehensive audit logging
- Real-time monitoring
- Emergency bypass
- Detailed analytics

#### Rate Limits:

| Type | Limit | Description |
|------|-------|-------------|
| User | 60/min, 1000/hour | Per-user requests |
| IP | 100/min, 5000/hour | Per-IP requests |
| Endpoint | Variable | Per-endpoint limits |

### 6. Real-time Updates (`src/api/realtime_updates.py`)

WebSocket-based real-time communication system.

#### Key Features:
- WebSocket connection management
- Real-time status broadcasting
- Component health monitoring
- Performance metrics streaming
- Alert notifications
- Connection recovery

#### WebSocket Message Types:

| Type | Description |
|------|-------------|
| SYSTEM_STATUS | System status updates |
| COMPONENT_HEALTH | Component health changes |
| PERFORMANCE_METRICS | Performance metrics |
| ALERT_NOTIFICATION | New alerts |
| HEARTBEAT | Connection heartbeat |

## Security Features

### Authentication & Authorization

1. **JWT-based Authentication**: Secure token-based authentication
2. **Role-based Access Control**: Granular permissions by user role
3. **Multi-factor Authentication**: TOTP, SMS, email, hardware tokens
4. **Session Management**: Secure session handling with Redis
5. **IP Whitelisting**: Restrict access to authorized IP ranges

### Security Monitoring

1. **Suspicious Activity Detection**: Automatic detection of unusual patterns
2. **Failed Login Tracking**: Account lockout after failed attempts
3. **Audit Logging**: Comprehensive logging of all operations
4. **Rate Limiting**: Protection against abuse and DoS attacks
5. **Security Alerts**: Real-time notifications of security events

## Installation & Setup

### Prerequisites

- Python 3.8+
- Redis server
- PostgreSQL database
- Node.js (for dashboard components)

### Installation Steps

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Setup Environment Variables**:
```bash
export DATABASE_URL="postgresql://user:password@localhost/grandmodel"
export REDIS_URL="redis://localhost:6379"
export JWT_SECRET_KEY="your-jwt-secret-key"
export SYSTEM_CONTROL_IP_WHITELIST="127.0.0.1/32,10.0.0.0/8"
```

3. **Initialize Database**:
```bash
python -c "from src.api.rate_limiting_audit import Base, engine; Base.metadata.create_all(engine)"
```

4. **Start Services**:
```bash
# Start System Control API
python -m src.api.system_control_api

# Start Status Monitoring API
python -m src.api.system_status_endpoints

# Start Dashboard
python -m src.dashboard.system_control_dashboard
```

## Configuration

### System Control Configuration

```python
# Rate limiting configuration
RATE_LIMIT_CONFIG = {
    "user_requests_per_minute": 60,
    "ip_requests_per_minute": 100,
    "endpoint_requests_per_minute": {
        "/api/system/on": 5,
        "/api/system/off": 5,
        "/api/system/emergency": 10
    }
}

# Security configuration
SECURITY_CONFIG = {
    "mfa_required": True,
    "ip_whitelist_enabled": True,
    "hardware_token_required": False,
    "session_timeout": 3600
}
```

### Dashboard Configuration

```python
# Dashboard configuration
DASHBOARD_CONFIG = {
    "api_base_url": "http://localhost:8001",
    "status_api_url": "http://localhost:8002",
    "refresh_interval": 5,
    "theme": "dark"
}
```

## API Reference

### System Control API

#### Turn System ON

```http
POST /api/system/on
Content-Type: application/json
Authorization: Bearer <token>

{
  "action": "on",
  "reason": "Scheduled startup",
  "force": false,
  "timeout_seconds": 30
}
```

#### Emergency Stop

```http
POST /api/system/emergency
Content-Type: application/json
Authorization: Bearer <token>

{
  "action": "emergency",
  "reason": "Security incident detected"
}
```

#### Get System Status

```http
GET /api/system/status
Authorization: Bearer <token>
```

Response:
```json
{
  "status": "ON",
  "timestamp": "2024-01-01T12:00:00Z",
  "uptime_seconds": 3600,
  "version": "1.0.0",
  "components": {
    "marl_engine": {"status": "running", "health": "healthy"},
    "risk_manager": {"status": "running", "health": "healthy"}
  },
  "performance_metrics": {
    "cpu_usage_percent": 45.2,
    "memory_usage_percent": 67.8
  }
}
```

### Status Monitoring API

#### Get Component Health

```http
GET /api/status/components
Authorization: Bearer <token>
```

Response:
```json
[
  {
    "component_id": "marl_engine",
    "component_type": "engine",
    "status": "running",
    "health_score": 0.95,
    "last_check": "2024-01-01T12:00:00Z",
    "metrics": [...],
    "uptime_seconds": 3600
  }
]
```

#### Get Alerts

```http
GET /api/status/alerts?status=active&severity=high
Authorization: Bearer <token>
```

## WebSocket API

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8002/ws/status?token=<jwt-token>');

ws.onmessage = function(event) {
  const message = JSON.parse(event.data);
  console.log('Received:', message);
};

// Subscribe to system status updates
ws.send(JSON.stringify({
  type: 'subscription_ack',
  subscription_type: 'system_status',
  action: 'subscribe'
}));
```

### Message Format

```json
{
  "type": "system_status",
  "data": {
    "status": "ON",
    "timestamp": "2024-01-01T12:00:00Z",
    "overall_health": 0.95
  },
  "timestamp": "2024-01-01T12:00:00Z",
  "message_id": "msg_123456"
}
```

## Mobile Responsive Design

The dashboard is fully mobile-responsive with:

- Touch-friendly controls (minimum 44px touch targets)
- Responsive grid layout
- Mobile-optimized navigation
- Adaptive typography
- Touch gestures support
- Offline capability indicators

### CSS Features (`src/dashboard/static/mobile-responsive.css`)

- Mobile-first responsive design
- Dark theme support
- Accessibility features
- Performance optimizations
- Touch-friendly interface elements

## Monitoring & Observability

### Metrics

The system exposes Prometheus-compatible metrics:

- Request rates and latencies
- Error rates by endpoint
- Authentication success/failure rates
- System resource usage
- Component health scores

### Logging

Structured logging with:

- JSON format for machine readability
- Correlation IDs for request tracing
- Security event logging
- Performance metrics logging
- Error tracking and alerting

### Alerting

Configurable alerts for:

- System component failures
- High error rates
- Security violations
- Performance degradation
- Resource exhaustion

## Security Best Practices

### Authentication

1. Use strong JWT secrets
2. Implement token rotation
3. Enable MFA for all users
4. Use hardware tokens for critical operations
5. Implement IP whitelisting

### Authorization

1. Follow principle of least privilege
2. Regular permission audits
3. Role-based access control
4. Time-limited access tokens
5. Emergency access procedures

### Network Security

1. Use TLS/SSL for all communications
2. Implement rate limiting
3. Network segmentation
4. Firewall rules
5. VPN access requirements

## Troubleshooting

### Common Issues

1. **Authentication Failures**
   - Check JWT secret configuration
   - Verify user permissions
   - Check IP whitelist settings

2. **Rate Limiting Issues**
   - Review rate limit configuration
   - Check Redis connectivity
   - Monitor error logs

3. **WebSocket Connection Problems**
   - Verify token in query parameters
   - Check network connectivity
   - Review firewall settings

### Debug Mode

Enable debug mode for detailed logging:

```python
DASHBOARD_CONFIG = {
    "debug": True,
    "log_level": "DEBUG"
}
```

## Performance Optimization

### Caching

- Redis caching for session data
- Response caching for status endpoints
- Static asset caching
- Database query optimization

### Database

- Connection pooling
- Query optimization
- Index optimization
- Partitioning for large tables

### Network

- HTTP/2 support
- Compression (gzip)
- CDN for static assets
- Load balancing

## Compliance & Auditing

### Audit Trail

All operations are logged with:

- User identification
- Timestamp
- Action performed
- IP address
- User agent
- Request details
- Response status

### Compliance Features

- SOX compliance logging
- GDPR data handling
- PCI DSS security controls
- ISO 27001 alignment
- Regulatory reporting

## Future Enhancements

### Planned Features

1. **Advanced Analytics**
   - Machine learning for anomaly detection
   - Predictive maintenance alerts
   - Performance trend analysis

2. **Integration Capabilities**
   - LDAP/Active Directory integration
   - SIEM integration
   - Ticketing system integration

3. **Enhanced Security**
   - Behavioral analytics
   - Zero-trust architecture
   - Hardware security module (HSM) support

4. **Scalability Improvements**
   - Microservices architecture
   - Container orchestration
   - Auto-scaling capabilities

## Support & Maintenance

### Regular Maintenance

- Security updates
- Performance monitoring
- Database maintenance
- Log rotation
- Certificate renewal

### Support Channels

- Documentation updates
- Bug reports
- Feature requests
- Security vulnerabilities
- Performance issues

## Conclusion

The GrandModel System Control Web Interface and API provides a comprehensive, secure, and scalable solution for system control operations. With enterprise-grade security features, real-time monitoring, and mobile-responsive design, it ensures reliable and secure system management across all deployment scenarios.

For additional support or questions, please refer to the system documentation or contact the development team.