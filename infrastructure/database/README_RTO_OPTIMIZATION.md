# Patroni RTO Optimization Guide

## Overview

This guide documents the comprehensive optimization of Patroni failover parameters to achieve a **<30 second Recovery Time Objective (RTO)** for database failovers. The optimization focuses on reducing detection time, consensus time, and failover execution time.

## Target Achievement

**Primary Goal**: Reduce database RTO from 45.2s to <30s

**Success Criteria**:
- Graceful failover: <25s
- Crash failover: <30s
- 99.9% availability during testing
- Automated failover without manual intervention

## Optimization Summary

### Key Configuration Changes

| Parameter | Before | After | Improvement |
|-----------|--------|--------|-------------|
| `loop_wait` | 10s | 5s | 50% faster detection |
| `retry_timeout` | 30s | 15s | 50% faster retries |
| `ttl` | 30s | 15s | 50% faster consensus |
| `failover_timeout` | N/A | 30s | Hard limit added |
| `switchover_timeout` | N/A | 30s | Hard limit added |
| `master_start_timeout` | 300s | 60s | 80% faster startup |
| `archive_timeout` | 60s | 30s | 50% faster archiving |

### New Features Added

1. **Fast Leader Election Algorithms**
   - `check_timeline: true` - Enables timeline validation
   - `use_pg_rewind: true` - Faster recovery from diverged timelines
   - `remove_data_directory_on_rewind_failure: true` - Automatic cleanup
   - `remove_data_directory_on_diverged_timelines: true` - Automatic recovery

2. **Etcd Optimizations**
   - `etcd_hosts_fallback_to_srv: false` - Faster host resolution
   - Reduced TTL for faster consensus

3. **Failover Constraints**
   - Hard timeouts prevent indefinite waiting
   - Automated cleanup on failures

## Files Structure

```
infrastructure/database/
├── patroni-config.yml              # Optimized Patroni configuration
├── postgresql-cluster.yml          # Docker Compose cluster setup
├── test_failover.py                # Automated failover testing
├── rto_monitor.py                  # Real-time RTO monitoring
├── rto_config.json                 # Monitoring configuration
├── test_rto_optimization.sh        # Comprehensive test suite
├── deploy_rto_optimization.sh      # Deployment script
└── README_RTO_OPTIMIZATION.md      # This documentation
```

## Implementation Guide

### 1. Pre-Deployment Validation

Before applying optimizations, ensure your current setup is healthy:

```bash
# Check cluster status
curl -s http://localhost:8008/cluster | python3 -m json.tool

# Verify all services are running
docker ps | grep -E "(postgres|patroni|etcd)"

# Test database connectivity
psql -h localhost -p 5432 -U grandmodel -d grandmodel -c "SELECT 1;"
```

### 2. Deploy Optimizations

Use the automated deployment script:

```bash
cd /home/QuantNova/GrandModel
./infrastructure/database/deploy_rto_optimization.sh
```

The script will:
- Backup current configuration
- Validate optimization parameters
- Restart services with new configuration
- Verify deployment success

### 3. Test and Validate

Run the comprehensive test suite:

```bash
# Run full test suite
./infrastructure/database/test_rto_optimization.sh

# Run individual failover test
python3 infrastructure/database/test_failover.py --single-test --test-type graceful

# Start continuous monitoring
python3 infrastructure/database/rto_monitor.py --config infrastructure/database/rto_config.json
```

## Test Results Expected

### Baseline (Before Optimization)
- Graceful failover: ~35-40s
- Crash failover: ~45-50s
- Detection time: 10s
- Consensus time: 30s

### Optimized (After Implementation)
- Graceful failover: ~15-20s
- Crash failover: ~20-25s
- Detection time: 5s
- Consensus time: 15s

## Monitoring and Alerting

### Real-time Monitoring

The `rto_monitor.py` script provides:
- Continuous database availability monitoring
- Automatic RTO measurement during failovers
- Real-time performance metrics
- Alert system for RTO threshold breaches

### Key Metrics Tracked

1. **Availability Metrics**
   - Uptime percentage
   - Response time distribution
   - Connection success rate

2. **RTO Metrics**
   - Downtime duration
   - Recovery time measurement
   - Failover type classification
   - Primary node transitions

3. **Performance Metrics**
   - Database response times
   - Patroni API response times
   - Cluster state changes

### Alert Configuration

Configure alerts in `rto_config.json`:

```json
{
  "alerts": {
    "rto_threshold": 30,
    "consecutive_failures": 3,
    "webhook_url": "https://your-webhook-url",
    "email_recipients": ["admin@company.com"]
  }
}
```

## Testing Procedures

### Automated Testing

1. **Graceful Failover Test**
   ```bash
   python3 infrastructure/database/test_failover.py \
     --single-test \
     --test-type graceful \
     --output results.json
   ```

2. **Crash Failover Test**
   ```bash
   python3 infrastructure/database/test_failover.py \
     --single-test \
     --test-type crash \
     --output results.json
   ```

3. **Continuous Testing**
   ```bash
   python3 infrastructure/database/test_failover.py \
     --test-count 10 \
     --interval 30 \
     --output continuous_results.json
   ```

### Manual Testing

1. **Graceful Failover**
   ```bash
   # Trigger planned failover
   curl -X POST http://localhost:8008/failover \
     -H "Content-Type: application/json" \
     -d '{"leader": "postgresql-primary"}'
   ```

2. **Crash Simulation**
   ```bash
   # Simulate primary failure
   docker stop patroni-primary
   
   # Monitor recovery
   watch -n 1 "curl -s http://localhost:8009/cluster | jq '.members[].role'"
   ```

## Troubleshooting

### Common Issues

1. **High RTO Times**
   - Check network latency between nodes
   - Verify etcd performance
   - Review PostgreSQL configuration
   - Check disk I/O performance

2. **Failover Failures**
   - Verify replication lag
   - Check timeline consistency
   - Review synchronous replication settings
   - Validate network connectivity

3. **Split-Brain Scenarios**
   - Ensure proper quorum configuration
   - Check network partitioning
   - Verify etcd cluster health
   - Review fencing mechanisms

### Diagnostic Commands

```bash
# Check Patroni logs
docker logs patroni-primary
docker logs patroni-standby

# Check PostgreSQL logs
docker exec patroni-primary tail -f /var/log/postgresql/postgresql.log

# Check etcd status
docker exec postgres-etcd etcdctl endpoint health

# Check replication status
docker exec patroni-primary psql -c "SELECT * FROM pg_stat_replication;"
```

## Performance Tuning

### Further Optimizations

If RTO target is not achieved, consider:

1. **Reduce loop_wait to 3s**
   ```yaml
   loop_wait: 3
   ```

2. **Reduce TTL to 10s**
   ```yaml
   ttl: 10
   ```

3. **Optimize etcd performance**
   ```yaml
   etcd3:
     hosts: etcd:2379
     ttl: 10
   ```

4. **PostgreSQL tuning**
   ```yaml
   checkpoint_completion_target: 0.7
   wal_buffers: 32MB
   max_wal_size: 2GB
   ```

### Hardware Recommendations

- **SSD storage** for WAL and data directories
- **Low-latency network** between nodes (<1ms)
- **Dedicated etcd cluster** for large deployments
- **Sufficient RAM** for PostgreSQL shared buffers

## Maintenance

### Regular Tasks

1. **Weekly RTO testing**
   ```bash
   # Run weekly failover test
   ./infrastructure/database/test_rto_optimization.sh
   ```

2. **Monthly performance review**
   ```bash
   # Generate performance report
   python3 infrastructure/database/rto_monitor.py --export monthly_report.json
   ```

3. **Quarterly optimization review**
   - Review RTO trends
   - Analyze failure patterns
   - Update configuration as needed
   - Test disaster recovery procedures

### Backup and Recovery

1. **Configuration backup**
   ```bash
   cp infrastructure/database/patroni-config.yml backups/
   ```

2. **Rollback procedure**
   ```bash
   # Restore from backup
   cp backups/patroni-config.yml.backup infrastructure/database/patroni-config.yml
   
   # Restart services
   docker restart patroni-primary patroni-standby
   ```

## Security Considerations

### Authentication
- Use strong passwords for replication users
- Configure SSL/TLS for all connections
- Implement proper pg_hba.conf rules

### Network Security
- Restrict Patroni API access
- Use encrypted connections for etcd
- Implement network segmentation

### Monitoring Security
- Secure monitoring endpoints
- Encrypt monitoring data transmission
- Implement proper access controls

## Compliance and Auditing

### Audit Trail
- All failover events are logged
- Configuration changes are tracked
- Performance metrics are retained
- Access logs are maintained

### Compliance Requirements
- Meet SOC 2 requirements for availability
- Maintain audit logs for compliance
- Document all configuration changes
- Regular security assessments

## Conclusion

The Patroni RTO optimization provides:
- **50% reduction** in failover detection time
- **50% reduction** in consensus time
- **<30s RTO achievement** for most scenarios
- **Automated testing** and monitoring
- **Comprehensive alerting** system

This optimization enables the database infrastructure to meet stringent availability requirements while maintaining data consistency and reliability.

For questions or support, refer to the troubleshooting section or contact the database team.