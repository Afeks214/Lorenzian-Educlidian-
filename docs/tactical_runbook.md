# Tactical MARL System Operational Runbook

## Table of Contents
- [System Overview](#system-overview)
- [Key Performance Metrics](#key-performance-metrics)
- [Operational Procedures](#operational-procedures)
- [Alert Response Protocols](#alert-response-protocols)
- [Troubleshooting Guide](#troubleshooting-guide)
- [Performance Optimization](#performance-optimization)
- [Disaster Recovery](#disaster-recovery)

## System Overview

The Tactical MARL System is a high-frequency, low-latency trading system that processes 5-minute market data matrices and makes tactical trading decisions in under 100ms. The system consists of:

- **Tactical Service**: CPU-optimized Docker container with TorchScript JIT compilation
- **Redis Streams**: Guaranteed message delivery for SYNERGY_DETECTED events
- **Multi-Agent Architecture**: FVG Agent, Momentum Agent, and Entry Agent
- **Decision Aggregation**: Weighted voting with 65% execution threshold
- **Monitoring Stack**: Prometheus metrics with ultra-fine latency buckets

### Architecture Components

```
[Strategic System] → [Redis Streams] → [Tactical Controller] → [3 Agents] → [Aggregator] → [Execution]
                                    ↓
                              [Monitoring & Alerting]
```

### Critical Performance Targets

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| P99 Latency | < 100ms | 100ms |
| P50 Latency | < 50ms | 50ms |
| Event Processing Rate | > 100 events/sec | 50 events/sec |
| Model Staleness | < 15 minutes | 15 minutes |
| Consumer Lag | < 2 seconds | 2 seconds |

## Key Performance Metrics

### Prometheus Metrics Overview

**Latency Metrics (tactical_request_duration_seconds)**
- Buckets: [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0]
- Labels: operation, agent_type, synergy_type
- Critical: P99 < 100ms

**Business Metrics**
- `tactical_active_positions`: Current position count
- `tactical_model_confidence`: Average model confidence (0-1)
- `tactical_agent_accuracy`: Agent prediction accuracy
- `tactical_execution_rate`: Percentage of decisions executed

**System Health Metrics**
- `tactical_redis_consumer_lag_seconds`: Redis Stream consumer lag
- `tactical_model_staleness_seconds`: Time since last model update
- `tactical_error_rate`: Error rate by operation type
- **NEW**: `tactical_health_state`: Current health state (0=healthy, 1=degraded, 2=critical, 3=failed)
- **NEW**: `tactical_circuit_breaker_state`: Circuit breaker state (0=closed, 1=open, 2=half-open)
- **NEW**: `tactical_duplicate_events_total`: Total duplicate events detected
- **NEW**: `tactical_state_transitions_total`: Total health state transitions

### Grafana Dashboard Queries

**P99 Latency by Operation**
```promql
histogram_quantile(0.99, rate(tactical_request_duration_seconds_bucket[5m]))
```

**Event Processing Rate**
```promql
rate(tactical_events_processed_total[5m])
```

**Consumer Lag**
```promql
tactical_redis_consumer_lag_seconds
```

**NEW Queries for Enhanced Monitoring**

**Health State Over Time**
```promql
tactical_health_state
```

**Circuit Breaker Status**
```promql
tactical_circuit_breaker_state
```

**Duplicate Event Detection Rate**
```promql
rate(tactical_duplicate_events_total[5m])
```

**State Transition Frequency**
```promql
rate(tactical_state_transitions_total[1h])
```

## Operational Procedures

### Daily Operations Checklist

**Morning Startup (30 minutes before market open)**
1. Verify tactical service health: `docker-compose ps tactical-marl`
2. Check Redis connectivity: `redis-cli -h redis -p 6379 -n 2 ping`
3. Validate model compilation: Check logs for "JIT compilation successful"
4. Review overnight performance metrics in Grafana
5. Run health check: `curl http://localhost:8001/health`
6. **NEW**: Check advanced health state: `curl http://localhost:8001/health/state`
7. **NEW**: Verify idempotency system: Check Redis for stale processing keys
8. **NEW**: Review circuit breaker status: Check restart count and alert history

**During Market Hours (Every 15 minutes)**
1. Monitor P99 latency dashboard
2. Check consumer lag metrics
3. Verify active positions count
4. Review error rate trends
5. Confirm model confidence levels
6. **NEW**: Monitor health state transitions
7. **NEW**: Check for duplicate event processing alerts
8. **NEW**: Verify circuit breaker status

**End of Day (After market close)**
1. Archive performance logs
2. Generate daily performance report
3. Check for any degradation patterns
4. Update model if needed
5. Verify backup processes completed
6. **NEW**: Review health state machine transitions
7. **NEW**: Clean up old idempotency keys (if needed)
8. **NEW**: Reset circuit breaker counters if appropriate

### Service Management

**Start Tactical Service**
```bash
# Start full stack
docker-compose up -d

# Start tactical service only
docker-compose up -d tactical-marl redis

# Check service status
docker-compose ps tactical-marl
```

**Restart Tactical Service**
```bash
# Graceful restart
docker-compose restart tactical-marl

# Force restart with rebuild
docker-compose down tactical-marl
docker-compose build tactical-marl
docker-compose up -d tactical-marl
```

**View Service Logs**
```bash
# Real-time logs
docker-compose logs -f tactical-marl

# Last 100 lines
docker-compose logs --tail=100 tactical-marl

# Filter for errors
docker-compose logs tactical-marl | grep ERROR
```

### Model Management

**JIT Compilation Status**
```bash
# Check compilation logs
docker-compose exec tactical-marl cat /tmp/jit_compilation.log

# Verify compiled models exist
docker-compose exec tactical-marl ls -la /app/models/jit/
```

**Model Update Process**
1. Stop tactical service: `docker-compose stop tactical-marl`
2. Update model files in `models/tactical/`
3. Rebuild container: `docker-compose build tactical-marl`
4. Start service: `docker-compose up -d tactical-marl`
5. Verify JIT compilation successful in logs
6. Run performance validation: `pytest tests/performance/test_e2e_latency.py`

## Alert Response Protocols

### Critical Alerts (P0 - Immediate Response)

**P99 Latency > 100ms**
- **Impact**: Trading decisions delayed, potential missed opportunities
- **Response Time**: < 2 minutes
- **Actions**:
  1. Check CPU utilization: `docker stats tactical-marl`
  2. Verify Redis connectivity: `redis-cli -h redis -p 6379 -n 2 ping`
  3. Review error logs: `docker-compose logs tactical-marl | grep ERROR`
  4. If persistent, restart service: `docker-compose restart tactical-marl`
  5. Escalate to on-call engineer if not resolved in 5 minutes

**Consumer Lag > 5 seconds**
- **Impact**: Stale event processing, outdated decisions
- **Response Time**: < 1 minute
- **Actions**:
  1. Check Redis Stream status: `redis-cli -h redis -p 6379 -n 2 XINFO STREAM synergy_events`
  2. Verify consumer group: `redis-cli -h redis -p 6379 -n 2 XINFO GROUPS synergy_events`
  3. Check service processing rate in Grafana
  4. Restart tactical service if lag increasing
  5. Consider scaling if persistent

**Model Staleness > 15 minutes**
- **Impact**: Trading with outdated models, reduced accuracy
- **Response Time**: < 5 minutes
- **Actions**:
  1. Check model update pipeline
  2. Verify JIT compilation process
  3. Review model training logs
  4. Restart service to trigger recompilation
  5. Contact ML team if models unavailable

### Warning Alerts (P1 - Response within 15 minutes)

**P50 Latency > 50ms**
- Monitor for trend escalation
- Check system resource usage
- Review recent changes
- Prepare for potential restart

**Error Rate > 5%**
- Investigate error patterns in logs
- Check data quality issues
- Verify external service connectivity
- Monitor for escalation

**Model Confidence < 0.6**
- Review market conditions
- Check data feed quality
- Verify model inputs
- Consider model refresh

### Escalation Matrix

| Alert Type | Primary | Secondary | Executive |
|------------|---------|-----------|-----------|
| P0 Critical | On-Call Engineer | Lead Engineer | CTO |
| P1 Warning | On-Call Engineer | Lead Engineer | - |
| P2 Info | Team Lead | - | - |

## Troubleshooting Guide

### Common Issues and Solutions

**Issue: Circuit Breaker Opened**

*Symptoms*:
- Service restart limit exceeded
- P0 alerts triggered
- Automatic recovery failed

*Diagnosis*:
```bash
# Check circuit breaker status
curl http://localhost:8001/health/state | jq '.health_checks'

# Check restart count
docker-compose ps tactical-marl

# Check alert history
redis-cli -h redis -p 6379 -n 2 XREAD COUNT 10 STREAMS tactical_alerts 0
```

*Solutions*:
1. Identify root cause from logs
2. Fix underlying issue
3. Reset circuit breaker manually if needed
4. Consider scaling resources
5. Review restart policies

**Issue: Duplicate Event Processing**

*Symptoms*:
- Multiple execution commands for same event
- Idempotency warnings in logs
- Duplicate trade alerts

*Diagnosis*:
```bash
# Check processing keys
redis-cli -h redis -p 6379 -n 2 KEYS tactical:processing:*

# Check completed keys
redis-cli -h redis -p 6379 -n 2 KEYS tactical:completed:*

# Check for correlation ID patterns
grep "duplicate_event" /app/logs/tactical.log
```

*Solutions*:
1. Clean up stale processing keys
2. Verify correlation ID generation
3. Check Redis TTL settings
4. Review event producer logic
5. Validate idempotency implementation

**Issue: Health State Machine Stuck**

*Symptoms*:
- Health state not transitioning
- Continuous degraded/critical state
- Recovery actions not working

*Diagnosis*:
```bash
# Check health state
curl http://localhost:8001/health/state

# Check health check results
curl http://localhost:8001/health/state | jq '.health_checks'

# Check state transition history
curl http://localhost:8001/health/state | jq '.recent_transitions'
```

*Solutions*:
1. Review health check thresholds
2. Check health check implementations
3. Force state transition if needed
4. Restart health monitoring
5. Verify Redis connectivity

**Issue: High Latency (P99 > 100ms)**

*Symptoms*:
- Increased response times
- Timeout errors
- Delayed executions

*Diagnosis*:
```bash
# Check CPU usage
docker stats tactical-marl

# Check memory usage
docker-compose exec tactical-marl free -h

# Check Redis latency
redis-cli -h redis -p 6379 -n 2 --latency-history

# Check network latency
docker-compose exec tactical-marl ping redis
```

*Solutions*:
1. Restart tactical service
2. Check for memory leaks
3. Verify JIT compilation
4. Scale resources if needed
5. Check Redis performance

**Issue: Consumer Lag Increasing**

*Symptoms*:
- Growing consumer lag metrics
- Delayed event processing
- Stale decisions

*Diagnosis*:
```bash
# Check consumer group status
redis-cli -h redis -p 6379 -n 2 XINFO GROUPS synergy_events

# Check pending messages
redis-cli -h redis -p 6379 -n 2 XPENDING synergy_events tactical_group

# Check processing rate
# (Use Grafana dashboard)
```

*Solutions*:
1. Restart consumer
2. Check processing bottlenecks
3. Increase parallelism
4. Clear stuck messages
5. Scale consumer instances

**Issue: Model Compilation Failures**

*Symptoms*:
- JIT compilation errors in logs
- Model loading failures
- Degraded performance

*Diagnosis*:
```bash
# Check compilation logs
docker-compose exec tactical-marl cat /tmp/jit_compilation.log

# Verify model files
docker-compose exec tactical-marl ls -la /app/models/

# Check PyTorch version
docker-compose exec tactical-marl python -c "import torch; print(torch.__version__)"
```

*Solutions*:
1. Rebuild container
2. Verify model compatibility
3. Check PyTorch installation
4. Revert to previous model version
5. Contact ML team

### Performance Debugging

**CPU Profiling**
```bash
# Install profiling tools
docker-compose exec tactical-marl pip install py-spy

# Profile running process
docker-compose exec tactical-marl py-spy record -o /tmp/profile.svg -d 60 -p 1
```

**Memory Profiling**
```bash
# Check memory usage over time
docker-compose exec tactical-marl free -h

# Check for memory leaks
docker-compose exec tactical-marl cat /proc/meminfo
```

**Redis Monitoring**
```bash
# Monitor Redis performance
redis-cli -h redis -p 6379 INFO

# Check slow queries
redis-cli -h redis -p 6379 SLOWLOG GET 10
```

## Performance Optimization

### System Tuning

**CPU Optimization**
- Verify 16 CPU cores allocated
- Check CPU affinity settings
- Monitor CPU utilization patterns
- Optimize thread pool sizes

**Memory Optimization**
- Verify 32GB RAM allocated
- Monitor memory usage patterns
- Check for memory leaks
- Optimize batch sizes

**Redis Optimization**
- Use Redis database 2 for isolation
- Monitor memory usage
- Optimize stream retention
- Configure appropriate timeouts

### Model Optimization

**JIT Compilation**
- Verify TorchScript compilation
- Check compilation performance
- Monitor model loading times
- Optimize model architecture

**Batch Processing**
- Optimize batch sizes
- Use vectorized operations
- Minimize data transfers
- Cache frequently used data

### Monitoring Optimization

**Metrics Collection**
- Use appropriate sampling rates
- Minimize metric cardinality
- Optimize query performance
- Archive historical data

**Alerting**
- Set appropriate thresholds
- Minimize false positives
- Optimize alert routing
- Use smart aggregation

## Disaster Recovery

### Backup Procedures

**Daily Backups**
- Model artifacts
- Configuration files
- Performance metrics
- Alert definitions

**Weekly Backups**
- Complete system state
- Historical performance data
- Operational logs
- Documentation updates

### Recovery Procedures

**Service Recovery**
1. Verify backup integrity
2. Restore configuration files
3. Rebuild Docker containers
4. Restore model artifacts
5. Verify system functionality

**Data Recovery**
1. Stop tactical service
2. Restore Redis data
3. Verify data consistency
4. Restart services
5. Validate operations

**Full System Recovery**
1. Provision new infrastructure
2. Restore from backups
3. Verify network connectivity
4. Test all components
5. Resume operations

### Business Continuity

**Failover Procedures**
- Maintain hot standby system
- Automated failover triggers
- Manual failover process
- Service restoration steps

**Communication Plan**
- Stakeholder notification
- Status page updates
- Customer communication
- Regulatory reporting

## Appendices

### A. Configuration Files
- `docker-compose.yml`
- `tactical.Dockerfile`
- `prometheus.yml`
- `grafana-dashboards.json`

### B. Log Formats
- Application logs
- Error logs
- Performance logs
- Audit logs

### C. Contact Information
- On-call rotation
- Escalation contacts
- Vendor support
- Emergency procedures

### D. Compliance Requirements
- Regulatory guidelines
- Data retention policies
- Audit requirements
- Security protocols

---

**Document Version**: 1.0  
**Last Updated**: 2025-07-11  
**Next Review**: 2025-08-11  
**Owner**: Systems & MLOps Team