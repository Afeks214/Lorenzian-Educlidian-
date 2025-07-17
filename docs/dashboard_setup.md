# GrandModel Grafana Dashboard Setup

## Overview
This document provides the Grafana dashboard configuration for monitoring the Strategic MARL 30m System. Import the provided JSON to instantly set up comprehensive monitoring.

## Dashboard Features

### 1. System Overview Dashboard
- Overall system health status
- Component health indicators
- Active alerts summary
- Key performance metrics

### 2. API Performance Dashboard
- Request rate by endpoint
- Response time percentiles (P50, P95, P99)
- Error rates and types
- Active connections
- Rate limiting metrics

### 3. Trading Metrics Dashboard
- Active positions count
- P&L tracking
- Model confidence scores
- Synergy detection rates
- Decision distribution

### 4. Infrastructure Dashboard
- CPU usage
- Memory consumption
- Disk I/O
- Network traffic
- Container health

## Import Instructions

1. **Access Grafana**
   ```
   URL: http://localhost:3000
   Username: admin
   Password: ${GRAFANA_PASSWORD}
   ```

2. **Import Dashboard**
   - Click "+" → "Import"
   - Paste the JSON below
   - Select Prometheus datasource
   - Click "Import"

## Dashboard JSON Configuration

```json
{
  "annotations": {
    "list": [
      {
        "builtIn": 1,
        "datasource": "-- Grafana --",
        "enable": true,
        "hide": true,
        "iconColor": "rgba(0, 211, 255, 1)",
        "name": "Annotations & Alerts",
        "type": "dashboard"
      }
    ]
  },
  "editable": true,
  "gnetId": null,
  "graphTooltip": 1,
  "id": null,
  "links": [],
  "panels": [
    {
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "thresholds"
          },
          "mappings": [
            {
              "options": {
                "0": {
                  "color": "red",
                  "index": 0,
                  "text": "Unhealthy"
                },
                "1": {
                  "color": "green",
                  "index": 1,
                  "text": "Healthy"
                }
              },
              "type": "value"
            }
          ],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "red",
                "value": null
              },
              {
                "color": "green",
                "value": 1
              }
            ]
          },
          "unit": "none"
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 0
      },
      "id": 1,
      "options": {
        "orientation": "auto",
        "reduceOptions": {
          "calcs": [
            "lastNotNull"
          ],
          "fields": "",
          "values": false
        },
        "showThresholdLabels": false,
        "showThresholdMarkers": true,
        "text": {}
      },
      "pluginVersion": "8.0.0",
      "targets": [
        {
          "expr": "grandmodel_health_status{component=\"api\"}",
          "refId": "A"
        }
      ],
      "title": "API Health Status",
      "type": "stat"
    },
    {
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "custom": {
            "axisLabel": "",
            "axisPlacement": "auto",
            "barAlignment": 0,
            "drawStyle": "line",
            "fillOpacity": 10,
            "gradientMode": "none",
            "hideFrom": {
              "tooltip": false,
              "viz": false,
              "legend": false
            },
            "lineInterpolation": "linear",
            "lineWidth": 1,
            "pointSize": 5,
            "scaleDistribution": {
              "type": "linear"
            },
            "showPoints": "never",
            "spanNulls": false,
            "stacking": {
              "group": "A",
              "mode": "none"
            },
            "thresholdsStyle": {
              "mode": "off"
            }
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green",
                "value": null
              }
            ]
          },
          "unit": "ms"
        },
        "overrides": [
          {
            "matcher": {
              "id": "byName",
              "options": "P99"
            },
            "properties": [
              {
                "id": "color",
                "value": {
                  "fixedColor": "red",
                  "mode": "fixed"
                }
              }
            ]
          }
        ]
      },
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 12,
        "y": 0
      },
      "id": 2,
      "options": {
        "tooltip": {
          "mode": "single"
        },
        "legend": {
          "calcs": [],
          "displayMode": "list",
          "placement": "bottom"
        }
      },
      "pluginVersion": "8.0.0",
      "targets": [
        {
          "expr": "histogram_quantile(0.50, sum(rate(grandmodel_inference_latency_seconds_bucket[5m])) by (le)) * 1000",
          "legendFormat": "P50",
          "refId": "A"
        },
        {
          "expr": "histogram_quantile(0.95, sum(rate(grandmodel_inference_latency_seconds_bucket[5m])) by (le)) * 1000",
          "legendFormat": "P95",
          "refId": "B"
        },
        {
          "expr": "histogram_quantile(0.99, sum(rate(grandmodel_inference_latency_seconds_bucket[5m])) by (le)) * 1000",
          "legendFormat": "P99",
          "refId": "C"
        }
      ],
      "title": "Inference Latency Percentiles",
      "type": "timeseries"
    },
    {
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "thresholds"
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green",
                "value": null
              },
              {
                "color": "yellow",
                "value": 100
              },
              {
                "color": "red",
                "value": 200
              }
            ]
          },
          "unit": "reqps"
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 6,
        "x": 0,
        "y": 8
      },
      "id": 3,
      "options": {
        "orientation": "auto",
        "reduceOptions": {
          "calcs": [
            "lastNotNull"
          ],
          "fields": "",
          "values": false
        },
        "showThresholdLabels": false,
        "showThresholdMarkers": true,
        "text": {}
      },
      "pluginVersion": "8.0.0",
      "targets": [
        {
          "expr": "sum(rate(grandmodel_http_requests_total[5m]))",
          "refId": "A"
        }
      ],
      "title": "Request Rate",
      "type": "gauge"
    },
    {
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "thresholds"
          },
          "mappings": [],
          "max": 1,
          "min": 0,
          "thresholds": {
            "mode": "percentage",
            "steps": [
              {
                "color": "green",
                "value": null
              },
              {
                "color": "yellow",
                "value": 70
              },
              {
                "color": "red",
                "value": 90
              }
            ]
          },
          "unit": "percentunit"
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 6,
        "x": 6,
        "y": 8
      },
      "id": 4,
      "options": {
        "orientation": "auto",
        "reduceOptions": {
          "calcs": [
            "lastNotNull"
          ],
          "fields": "",
          "values": false
        },
        "showThresholdLabels": false,
        "showThresholdMarkers": true,
        "text": {}
      },
      "pluginVersion": "8.0.0",
      "targets": [
        {
          "expr": "sum(rate(grandmodel_http_requests_total{status=~\"5..\"}[5m])) / sum(rate(grandmodel_http_requests_total[5m]))",
          "refId": "A"
        }
      ],
      "title": "Error Rate",
      "type": "gauge"
    },
    {
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "custom": {
            "hideFrom": {
              "tooltip": false,
              "viz": false,
              "legend": false
            }
          },
          "mappings": [],
          "unit": "short"
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 6,
        "x": 12,
        "y": 8
      },
      "id": 5,
      "options": {
        "legend": {
          "displayMode": "list",
          "placement": "right"
        },
        "pieType": "pie",
        "reduceOptions": {
          "calcs": [
            "lastNotNull"
          ],
          "fields": "",
          "values": false
        },
        "tooltip": {
          "mode": "single"
        }
      },
      "pluginVersion": "8.0.0",
      "targets": [
        {
          "expr": "sum(increase(grandmodel_http_requests_total[1h])) by (path)",
          "refId": "A"
        }
      ],
      "title": "Requests by Endpoint",
      "type": "piechart"
    },
    {
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "thresholds"
          },
          "mappings": [],
          "max": 10,
          "min": 0,
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green",
                "value": null
              },
              {
                "color": "yellow",
                "value": 6
              },
              {
                "color": "red",
                "value": 8
              }
            ]
          },
          "unit": "short"
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 6,
        "x": 18,
        "y": 8
      },
      "id": 6,
      "options": {
        "orientation": "auto",
        "reduceOptions": {
          "calcs": [
            "lastNotNull"
          ],
          "fields": "",
          "values": false
        },
        "showThresholdLabels": true,
        "showThresholdMarkers": true,
        "text": {}
      },
      "pluginVersion": "8.0.0",
      "targets": [
        {
          "expr": "grandmodel_active_positions_count",
          "refId": "A"
        }
      ],
      "title": "Active Positions",
      "type": "gauge"
    },
    {
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "palette-classic"
          },
          "custom": {
            "axisLabel": "",
            "axisPlacement": "auto",
            "barAlignment": 0,
            "drawStyle": "line",
            "fillOpacity": 10,
            "gradientMode": "none",
            "hideFrom": {
              "tooltip": false,
              "viz": false,
              "legend": false
            },
            "lineInterpolation": "linear",
            "lineWidth": 2,
            "pointSize": 5,
            "scaleDistribution": {
              "type": "linear"
            },
            "showPoints": "never",
            "spanNulls": false,
            "stacking": {
              "group": "A",
              "mode": "none"
            },
            "thresholdsStyle": {
              "mode": "off"
            }
          },
          "mappings": [],
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green",
                "value": null
              }
            ]
          },
          "unit": "currencyUSD"
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 16
      },
      "id": 7,
      "options": {
        "tooltip": {
          "mode": "single"
        },
        "legend": {
          "calcs": ["sum"],
          "displayMode": "list",
          "placement": "bottom"
        }
      },
      "pluginVersion": "8.0.0",
      "targets": [
        {
          "expr": "sum(grandmodel_trade_pnl_dollars) by (symbol)",
          "legendFormat": "{{symbol}}",
          "refId": "A"
        }
      ],
      "title": "P&L by Symbol",
      "type": "timeseries"
    },
    {
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "color": {
            "mode": "continuous-GrYlRd"
          },
          "custom": {
            "fillOpacity": 70,
            "lineWidth": 0
          },
          "mappings": [],
          "max": 1,
          "min": 0,
          "thresholds": {
            "mode": "absolute",
            "steps": [
              {
                "color": "green",
                "value": null
              }
            ]
          },
          "unit": "percentunit"
        },
        "overrides": []
      },
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 12,
        "y": 16
      },
      "id": 8,
      "options": {
        "colWidth": 0.9,
        "legend": {
          "displayMode": "list",
          "placement": "bottom"
        },
        "rowHeight": 0.9,
        "showValue": "auto",
        "tooltip": {
          "mode": "single"
        }
      },
      "pluginVersion": "8.0.0",
      "targets": [
        {
          "expr": "grandmodel_model_confidence_score",
          "format": "time_series",
          "refId": "A"
        }
      ],
      "title": "Model Confidence Heatmap",
      "type": "status-history"
    }
  ],
  "refresh": "5s",
  "schemaVersion": 30,
  "style": "dark",
  "tags": ["grandmodel", "production"],
  "templating": {
    "list": []
  },
  "time": {
    "from": "now-1h",
    "to": "now"
  },
  "timepicker": {},
  "timezone": "",
  "title": "GrandModel Strategic MARL Dashboard",
  "uid": "grandmodel-main",
  "version": 1
}
```

## Custom Queries

### Useful Prometheus Queries for Custom Panels

```promql
# Request latency heatmap
sum(rate(grandmodel_http_request_duration_seconds_bucket[5m])) by (le)

# Synergy detection rate by type
sum(rate(grandmodel_synergy_response_total[5m])) by (synergy_type)

# Memory usage over time
grandmodel_process_memory_mb

# Correlation between confidence and P&L
grandmodel_model_confidence_score * on() group_left grandmodel_trade_pnl_dollars

# Rate limit hits
sum(rate(grandmodel_rate_limit_exceeded_total[5m])) by (endpoint)
```

## Alert Configuration

Add these alert rules to Grafana:

```yaml
groups:
  - name: grandmodel_alerts
    interval: 30s
    rules:
      - alert: APIDown
        expr: grandmodel_health_status{component="api"} != 1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "API service is down"
          
      - alert: HighLatency
        expr: histogram_quantile(0.99, sum(rate(grandmodel_inference_latency_seconds_bucket[5m])) by (le)) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Inference latency P99 > 100ms"
          
      - alert: HighErrorRate
        expr: sum(rate(grandmodel_errors_total[5m])) > 10
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Error rate > 10/min"
```

## Dashboard Best Practices

1. **Refresh Rate**: Set to 5-10 seconds for real-time monitoring
2. **Time Range**: Default to last 1 hour, adjust as needed
3. **Variables**: Add template variables for environment, service
4. **Annotations**: Enable alert annotations to see incidents on graphs
5. **Thresholds**: Set visual thresholds based on SLAs

## Troubleshooting

### Dashboard Not Loading
- Check Prometheus datasource configuration
- Verify Prometheus is collecting metrics: `curl localhost:9090/api/v1/query?query=up`
- Check Grafana logs: `docker logs grafana`

### Missing Metrics
- Verify application is exposing metrics: `curl localhost:8000/metrics`
- Check Prometheus targets: http://localhost:9090/targets
- Review scrape configuration in `prometheus.yml`

### Performance Issues
- Reduce query complexity
- Increase dashboard refresh interval
- Use recording rules for complex queries
- Limit time range for heavy queries

---

**Last Updated**: 2024-01-11
**Dashboard Version**: 1.0.0