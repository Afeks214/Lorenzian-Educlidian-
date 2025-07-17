# GrandModel Alerts & Incident Response Playbook

## Alert Severity Levels

| Level | Response Time | Description | Examples |
|-------|--------------|-------------|----------|
| SEV-1 | < 5 minutes | Critical: Service down or major functionality broken | API returning 500s, Redis down |
| SEV-2 | < 30 minutes | High: Degraded performance or partial outage | Latency > 200ms, Memory > 80% |
| SEV-3 | < 2 hours | Medium: Non-critical issues affecting UX | Slow queries, High error rate |
| SEV-4 | < 24 hours | Low: Minor issues or improvements needed | Log warnings, Config updates |

## Critical Alerts (SEV-1)

### 🚨 API Service Down
**Alert**: `grandmodel_health_status{component="api"} != 1`

**Symptoms**:
- Health endpoint returns non-200 status
- No metrics being collected
- Connection refused errors

**Immediate Actions**:
1. Check container status:
   ```bash
   docker ps -a | grep grandmodel
   ```
2. Check logs for crash reason:
   ```bash
   docker logs --tail=100 grandmodel-app
   ```
3. Restart service:
   ```bash
   docker-compose -f docker-compose.prod.yml restart grandmodel
   ```

**Root Cause Analysis**:
- Check for OOM kills: `dmesg | grep -i "killed process"`
- Review error logs: `docker logs grandmodel-app | jq 'select(.level=="ERROR")'`
- Check recent deployments

**Escalation**: If not resolved in 10 minutes, page on-call engineer

**Dashboard**: [API Health Dashboard](http://localhost:3000/d/api-health)

---

### 🚨 Redis Connection Lost
**Alert**: `grandmodel_health_status{component="redis"} != 1`

**Symptoms**:
- "Connection refused" errors in logs
- API requests failing with 503
- Cache misses spike to 100%

**Immediate Actions**:
1. Check Redis container:
   ```bash
   docker-compose -f docker-compose.prod.yml ps redis
   ```
2. Test Redis connectivity:
   ```bash
   docker-compose exec redis redis-cli ping
   ```
3. Check Redis memory:
   ```bash
   docker-compose exec redis redis-cli info memory
   ```

**Recovery Steps**:
1. If OOM, increase memory limit
2. If crashed, restart: `docker-compose restart redis`
3. If data corrupted, restore from backup

**Dashboard**: [Redis Metrics](http://localhost:3000/d/redis-metrics)

---

### 🚨 Inference Latency Critical
**Alert**: `histogram_quantile(0.99, grandmodel_inference_latency_seconds) > 0.1`

**Symptoms**:
- P99 latency > 100ms
- Timeouts in decision endpoint
- Queue buildup

**Immediate Actions**:
1. Check current load:
   ```bash
   curl -s localhost:8000/metrics | grep http_requests_total
   ```
2. Check model memory usage:
   ```bash
   docker stats grandmodel-app --no-stream
   ```
3. Enable circuit breaker if needed:
   ```bash
   # Disable ML inference temporarily
   curl -X POST localhost:8000/admin/circuit-breaker/open
   ```

**Investigation**:
- Check for model loading issues
- Review recent model updates
- Analyze request patterns

**Dashboard**: [Inference Performance](http://localhost:3000/d/inference-perf)

## High Priority Alerts (SEV-2)

### ⚠️ High Memory Usage
**Alert**: `grandmodel_process_memory_mb > 400`

**Threshold**: > 400MB (80% of 512MB limit)

**Immediate Actions**:
1. Check memory breakdown:
   ```bash
   docker exec grandmodel-app ps aux --sort=-%mem | head
   ```
2. Check for memory leaks:
   ```bash
   # Review memory growth over time
   curl -s localhost:9090/api/v1/query_range?query=grandmodel_process_memory_mb
   ```
3. Force garbage collection (if Python):
   ```bash
   docker exec grandmodel-app python -c "import gc; gc.collect()"
   ```

**Mitigation**:
- Restart service during low traffic
- Scale horizontally if consistent high usage
- Review recent code changes for leaks

**Dashboard**: [Resource Usage](http://localhost:3000/d/resources)

---

### ⚠️ High Error Rate
**Alert**: `rate(grandmodel_errors_total[5m]) > 10`

**Threshold**: > 10 errors per minute

**Immediate Actions**:
1. Identify error types:
   ```bash
   curl -s localhost:8000/metrics | grep errors_total
   ```
2. Check recent logs:
   ```bash
   docker logs grandmodel-app --since 5m | jq 'select(.level=="ERROR")' | head -20
   ```
3. Identify affected endpoints:
   ```bash
   # Group errors by endpoint
   docker logs grandmodel-app --since 5m | \
     jq -r 'select(.level=="ERROR") | .path' | sort | uniq -c
   ```

**Common Causes**:
- Invalid input data
- External service failures
- Database connection issues
- Rate limiting

**Dashboard**: [Error Analysis](http://localhost:3000/d/errors)

---

### ⚠️ Rate Limit Exceeded
**Alert**: `rate(grandmodel_rate_limit_exceeded_total[5m]) > 50`

**Threshold**: > 50 rate limit hits per 5 minutes

**Immediate Actions**:
1. Identify source:
   ```bash
   # Check rate limit metrics by IP
   curl -s localhost:8000/metrics | grep rate_limit | grep -v "^#"
   ```
2. Review NGINX logs:
   ```bash
   docker logs nginx --since 5m | grep "429"
   ```
3. Temporary increase limits if legitimate:
   ```bash
   # Update rate limit configuration
   vim configs/nginx/nginx.conf
   docker-compose restart nginx
   ```

**Investigation**:
- Check for DDoS patterns
- Identify top users/IPs
- Review API key usage

**Dashboard**: [Rate Limiting](http://localhost:3000/d/rate-limits)

## Medium Priority Alerts (SEV-3)

### 📊 Model Confidence Low
**Alert**: `avg(grandmodel_model_confidence_score) < 0.6`

**Threshold**: Average confidence < 60%

**Actions**:
1. Check model performance by type:
   ```bash
   curl -s localhost:9090/api/v1/query?query=grandmodel_model_confidence_score
   ```
2. Review recent market conditions
3. Check for data quality issues

**Investigation Steps**:
- Analyze synergy patterns
- Review feature distributions
- Check for market regime changes

**Dashboard**: [Model Performance](http://localhost:3000/d/model-perf)

---

### 📊 High Active Positions
**Alert**: `grandmodel_active_positions_count > 8`

**Threshold**: > 8 concurrent positions (80% of max)

**Actions**:
1. Review current positions:
   ```bash
   curl -s localhost:8000/api/positions | jq .
   ```
2. Check risk metrics:
   ```bash
   curl -s localhost:8000/metrics | grep trade_pnl
   ```
3. Consider position limits adjustment

**Risk Management**:
- Review position sizing
- Check correlation between positions
- Evaluate total exposure

**Dashboard**: [Trading Dashboard](http://localhost:3000/d/trading)

## Low Priority Alerts (SEV-4)

### 💡 Certificate Expiry Warning
**Alert**: `probe_ssl_earliest_cert_expiry - time() < 7 * 86400`

**Threshold**: Certificate expires in < 7 days

**Actions**:
1. Check certificate:
   ```bash
   openssl s_client -connect grandmodel.app:443 -servername grandmodel.app | \
     openssl x509 -noout -dates
   ```
2. Renew certificate
3. Update and restart NGINX

---

### 💡 Log Volume High
**Alert**: `rate(grandmodel_logs_written_total[5m]) > 1000`

**Threshold**: > 1000 logs per minute

**Actions**:
1. Check log levels:
   ```bash
   docker logs grandmodel-app --since 5m | \
     jq -r .level | sort | uniq -c
   ```
2. Identify noisy components
3. Adjust log levels if needed

## Metric Reference

### Key Metrics to Monitor

| Metric | Description | Normal Range | Alert Threshold |
|--------|-------------|--------------|-----------------|
| `grandmodel_health_status` | Component health (0/1) | 1 | != 1 |
| `grandmodel_inference_latency_seconds` | Model inference time | < 5ms | > 5ms (P99) |
| `grandmodel_http_request_duration_seconds` | API response time | < 100ms | > 100ms (P95) |
| `grandmodel_active_connections` | Current connections | < 50 | > 80 |
| `grandmodel_errors_total` | Error counter | < 1/min | > 10/min |
| `grandmodel_process_memory_mb` | Memory usage | < 400MB | > 400MB |
| `grandmodel_trade_pnl_dollars` | Trading P&L | Varies | < -1000 |
| `grandmodel_model_confidence_score` | Model confidence | > 0.7 | < 0.6 |

### Useful Prometheus Queries

```promql
# Request rate by endpoint
sum(rate(grandmodel_http_requests_total[5m])) by (path)

# Error rate percentage
sum(rate(grandmodel_http_requests_total{status=~"5.."}[5m])) / 
sum(rate(grandmodel_http_requests_total[5m])) * 100

# Memory usage trend
grandmodel_process_memory_mb[1h]

# Latency by percentile
histogram_quantile(0.95, 
  sum(rate(grandmodel_http_request_duration_seconds_bucket[5m])) by (le))

# Active positions vs P&L correlation
grandmodel_active_positions_count + on() group_left 
sum(grandmodel_trade_pnl_dollars)
```

## Incident Response Process

### 1. Acknowledge
- Acknowledge alert within SLA
- Join incident channel
- Assign incident commander

### 2. Assess
- Determine severity
- Check impact scope
- Review recent changes

### 3. Communicate
- Update status page
- Notify stakeholders
- Post in #incidents channel

### 4. Mitigate
- Apply immediate fixes
- Enable circuit breakers
- Scale resources if needed

### 5. Resolve
- Verify fix effectiveness
- Monitor for recurrence
- Update documentation

### 6. Post-Mortem
- Schedule within 48 hours
- Document root cause
- Create action items
- Share learnings

## Emergency Contacts

| Role | Contact | When to Call |
|------|---------|--------------|
| On-Call Engineer | Check PagerDuty | First responder for all alerts |
| Team Lead | #team-lead | Escalation for SEV-1/2 |
| Infrastructure | #infra-team | Hardware/network issues |
| Security | #security-team | Security incidents |
| Database Admin | #db-team | Database issues |

## Grafana Dashboard Links

- [System Overview](http://localhost:3000/d/system-overview)
- [API Performance](http://localhost:3000/d/api-performance)
- [Trading Metrics](http://localhost:3000/d/trading-metrics)
- [Infrastructure Health](http://localhost:3000/d/infrastructure)
- [Error Analysis](http://localhost:3000/d/error-analysis)
- [Custom Queries](http://localhost:3000/explore)

---

**Last Updated**: 2024-01-11
**Version**: 1.0.0
**Next Review**: 2024-02-11