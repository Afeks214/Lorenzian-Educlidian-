# Monitoring and Observability Infrastructure - Agent 20 Implementation
# Comprehensive monitoring setup with Prometheus, Grafana, and CloudWatch

# Prometheus Helm Chart
resource "helm_release" "prometheus" {
  name             = "prometheus"
  repository       = "https://prometheus-community.github.io/helm-charts"
  chart            = "kube-prometheus-stack"
  version          = "54.0.0"
  namespace        = "monitoring"
  create_namespace = true
  
  values = [
    <<-EOF
    prometheus:
      prometheusSpec:
        retention: 15d
        retentionSize: 50GB
        storageSpec:
          volumeClaimTemplate:
            spec:
              storageClassName: gp3
              accessModes: ["ReadWriteOnce"]
              resources:
                requests:
                  storage: 100Gi
        resources:
          requests:
            cpu: 1000m
            memory: 2Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        additionalScrapeConfigs:
          - job_name: 'grandmodel-strategic'
            static_configs:
              - targets: ['strategic-service.grandmodel.svc.cluster.local:9090']
            scrape_interval: 15s
            metrics_path: /metrics
          - job_name: 'grandmodel-tactical'
            static_configs:
              - targets: ['tactical-service.grandmodel.svc.cluster.local:9090']
            scrape_interval: 10s
            metrics_path: /metrics
          - job_name: 'grandmodel-risk'
            static_configs:
              - targets: ['risk-service.grandmodel.svc.cluster.local:9090']
            scrape_interval: 5s
            metrics_path: /metrics
        
        ruleFiles:
          - "/etc/prometheus/rules/*.yaml"
        
        ruleSelectorNilUsesHelmValues: false
        ruleSelector:
          matchLabels:
            app: grandmodel
    
    grafana:
      adminPassword: ${random_password.grafana_password.result}
      persistence:
        enabled: true
        storageClassName: gp3
        size: 20Gi
      
      resources:
        requests:
          cpu: 500m
          memory: 1Gi
        limits:
          cpu: 1000m
          memory: 2Gi
      
      sidecar:
        dashboards:
          enabled: true
          label: grafana_dashboard
        datasources:
          enabled: true
          label: grafana_datasource
      
      env:
        GF_SECURITY_ADMIN_PASSWORD: ${random_password.grafana_password.result}
        GF_USERS_ALLOW_SIGN_UP: false
        GF_USERS_ALLOW_ORG_CREATE: false
        GF_AUTH_ANONYMOUS_ENABLED: false
        GF_SECURITY_DISABLE_GRAVATAR: true
        GF_SNAPSHOTS_EXTERNAL_ENABLED: false
    
    alertmanager:
      alertmanagerSpec:
        storage:
          volumeClaimTemplate:
            spec:
              storageClassName: gp3
              accessModes: ["ReadWriteOnce"]
              resources:
                requests:
                  storage: 10Gi
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 200m
            memory: 256Mi
        
        config:
          global:
            smtp_smarthost: 'smtp.gmail.com:587'
            smtp_from: 'alerts@grandmodel.com'
          
          route:
            group_by: ['alertname']
            group_wait: 30s
            group_interval: 5m
            repeat_interval: 12h
            receiver: 'web.hook'
            routes:
            - match:
                severity: critical
              receiver: 'critical-alerts'
              continue: true
            - match:
                component: risk
              receiver: 'risk-alerts'
              continue: true
          
          receivers:
          - name: 'web.hook'
            webhook_configs:
            - url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
              send_resolved: true
          
          - name: 'critical-alerts'
            email_configs:
            - to: 'oncall@grandmodel.com'
              subject: 'CRITICAL: {{ .GroupLabels.alertname }}'
              body: |
                {{ range .Alerts }}
                Alert: {{ .Annotations.summary }}
                Description: {{ .Annotations.description }}
                Labels: {{ .Labels }}
                {{ end }}
          
          - name: 'risk-alerts'
            email_configs:
            - to: 'risk-team@grandmodel.com'
              subject: 'RISK ALERT: {{ .GroupLabels.alertname }}'
              body: |
                {{ range .Alerts }}
                Alert: {{ .Annotations.summary }}
                Description: {{ .Annotations.description }}
                Labels: {{ .Labels }}
                {{ end }}
    
    nodeExporter:
      enabled: true
    
    kubeStateMetrics:
      enabled: true
    
    defaultRules:
      create: true
      rules:
        alertmanager: true
        etcd: true
        configReloaders: true
        general: true
        k8s: true
        kubeApiserverAvailability: true
        kubeApiserverBurnrate: true
        kubeApiserverHistogram: true
        kubeApiserverSlos: true
        kubelet: true
        kubeProxy: true
        kubePrometheusGeneral: true
        kubePrometheusNodeRecording: true
        kubernetesApps: true
        kubernetesResources: true
        kubernetesStorage: true
        kubernetesSystem: true
        network: true
        node: true
        nodeExporterAlerting: true
        nodeExporterRecording: true
        prometheus: true
        prometheusOperator: true
    EOF
  ]
  
  depends_on = [module.eks]
}

resource "random_password" "grafana_password" {
  length  = 32
  special = true
}

# Jaeger for distributed tracing
resource "helm_release" "jaeger" {
  name             = "jaeger"
  repository       = "https://jaegertracing.github.io/helm-charts"
  chart            = "jaeger"
  version          = "0.71.0"
  namespace        = "monitoring"
  create_namespace = true
  
  values = [
    <<-EOF
    provisionDataStore:
      cassandra: false
      elasticsearch: true
    
    elasticsearch:
      deploy: true
      replicas: 3
      minimumMasterNodes: 2
      resources:
        requests:
          cpu: 500m
          memory: 1Gi
        limits:
          cpu: 1000m
          memory: 2Gi
      volumeClaimTemplate:
        accessModes: ["ReadWriteOnce"]
        storageClassName: gp3
        resources:
          requests:
            storage: 50Gi
    
    collector:
      replicaCount: 3
      resources:
        requests:
          cpu: 100m
          memory: 128Mi
        limits:
          cpu: 1000m
          memory: 1Gi
      service:
        type: ClusterIP
        grpc:
          port: 14250
        http:
          port: 14268
    
    query:
      replicaCount: 2
      resources:
        requests:
          cpu: 100m
          memory: 128Mi
        limits:
          cpu: 500m
          memory: 512Mi
      service:
        type: ClusterIP
        targetPort: 16686
        port: 16686
    
    agent:
      daemonset:
        useHostNetwork: true
      resources:
        requests:
          cpu: 50m
          memory: 64Mi
        limits:
          cpu: 200m
          memory: 256Mi
    EOF
  ]
  
  depends_on = [module.eks]
}

# Fluentd for log aggregation
resource "helm_release" "fluentd" {
  name             = "fluentd"
  repository       = "https://fluent.github.io/helm-charts"
  chart            = "fluentd"
  version          = "0.4.0"
  namespace        = "monitoring"
  create_namespace = true
  
  values = [
    <<-EOF
    replicaCount: 3
    
    image:
      repository: fluent/fluentd-kubernetes-daemonset
      tag: v1.16-debian-elasticsearch7-1
    
    resources:
      requests:
        cpu: 100m
        memory: 200Mi
      limits:
        cpu: 500m
        memory: 512Mi
    
    env:
      - name: FLUENT_ELASTICSEARCH_HOST
        value: "elasticsearch-master.monitoring.svc.cluster.local"
      - name: FLUENT_ELASTICSEARCH_PORT
        value: "9200"
      - name: FLUENT_ELASTICSEARCH_SCHEME
        value: "http"
      - name: FLUENT_ELASTICSEARCH_SSL_VERIFY
        value: "false"
      - name: FLUENT_ELASTICSEARCH_LOGSTASH_PREFIX
        value: "grandmodel"
      - name: FLUENT_ELASTICSEARCH_LOGSTASH_FORMAT
        value: "true"
      - name: FLUENT_ELASTICSEARCH_BUFFER_CHUNK_LIMIT_SIZE
        value: "2M"
      - name: FLUENT_ELASTICSEARCH_BUFFER_QUEUE_LIMIT_LENGTH
        value: "8"
      - name: FLUENT_ELASTICSEARCH_BUFFER_FLUSH_INTERVAL
        value: "5s"
      - name: FLUENT_ELASTICSEARCH_BUFFER_RETRY_LIMIT
        value: "17"
      - name: FLUENT_ELASTICSEARCH_BUFFER_RETRY_WAIT
        value: "1.0"
      - name: FLUENT_ELASTICSEARCH_BUFFER_MAX_RETRY_WAIT
        value: "30"
      - name: FLUENT_ELASTICSEARCH_RELOAD_CONNECTIONS
        value: "false"
      - name: FLUENT_ELASTICSEARCH_RECONNECT_ON_ERROR
        value: "true"
      - name: FLUENT_ELASTICSEARCH_RELOAD_ON_FAILURE
        value: "true"
    
    configMaps:
      general.conf: |
        <system>
          root_dir /tmp/fluentd-buffers/
        </system>
        
        <source>
          @type tail
          @id in_tail_container_logs
          path /var/log/containers/*.log
          pos_file /var/log/fluentd-containers.log.pos
          tag kubernetes.*
          read_from_head true
          <parse>
            @type json
            time_format %Y-%m-%dT%H:%M:%S.%NZ
          </parse>
        </source>
        
        <filter kubernetes.**>
          @type kubernetes_metadata
          @id filter_kube_metadata
          kubernetes_url "#{ENV['FLUENT_FILTER_KUBERNETES_URL'] || 'https://' + ENV.fetch('KUBERNETES_SERVICE_HOST') + ':' + ENV.fetch('KUBERNETES_SERVICE_PORT') + '/api'}"
          verify_ssl "#{ENV['KUBERNETES_VERIFY_SSL'] || true}"
          ca_file "#{ENV['KUBERNETES_CA_FILE']}"
          skip_labels "#{ENV['FLUENT_KUBERNETES_METADATA_SKIP_LABELS'] || 'false'}"
          skip_container_metadata "#{ENV['FLUENT_KUBERNETES_METADATA_SKIP_CONTAINER_METADATA'] || 'false'}"
          skip_master_url "#{ENV['FLUENT_KUBERNETES_METADATA_SKIP_MASTER_URL'] || 'false'}"
          skip_namespace_metadata "#{ENV['FLUENT_KUBERNETES_METADATA_SKIP_NAMESPACE_METADATA'] || 'false'}"
        </filter>
        
        <match **>
          @type elasticsearch
          @id out_es
          @log_level info
          include_tag_key true
          host "#{ENV['FLUENT_ELASTICSEARCH_HOST']}"
          port "#{ENV['FLUENT_ELASTICSEARCH_PORT']}"
          path "#{ENV['FLUENT_ELASTICSEARCH_PATH']}"
          scheme "#{ENV['FLUENT_ELASTICSEARCH_SCHEME'] || 'http'}"
          ssl_verify "#{ENV['FLUENT_ELASTICSEARCH_SSL_VERIFY'] || 'true'}"
          ssl_version "#{ENV['FLUENT_ELASTICSEARCH_SSL_VERSION'] || 'TLSv1_2'}"
          user "#{ENV['FLUENT_ELASTICSEARCH_USER'] || use_default}"
          password "#{ENV['FLUENT_ELASTICSEARCH_PASSWORD'] || use_default}"
          reload_connections "#{ENV['FLUENT_ELASTICSEARCH_RELOAD_CONNECTIONS'] || 'false'}"
          reconnect_on_error "#{ENV['FLUENT_ELASTICSEARCH_RECONNECT_ON_ERROR'] || 'true'}"
          reload_on_failure "#{ENV['FLUENT_ELASTICSEARCH_RELOAD_ON_FAILURE'] || 'true'}"
          log_es_400_reason "#{ENV['FLUENT_ELASTICSEARCH_LOG_ES_400_REASON'] || 'false'}"
          logstash_prefix "#{ENV['FLUENT_ELASTICSEARCH_LOGSTASH_PREFIX'] || 'logstash'}"
          logstash_format "#{ENV['FLUENT_ELASTICSEARCH_LOGSTASH_FORMAT'] || 'true'}"
          index_name "#{ENV['FLUENT_ELASTICSEARCH_LOGSTASH_INDEX_NAME'] || 'logstash'}"
          type_name "#{ENV['FLUENT_ELASTICSEARCH_LOGSTASH_TYPE_NAME'] || 'fluentd'}"
          <buffer>
            flush_thread_count "#{ENV['FLUENT_ELASTICSEARCH_BUFFER_FLUSH_THREAD_COUNT'] || '8'}"
            flush_interval "#{ENV['FLUENT_ELASTICSEARCH_BUFFER_FLUSH_INTERVAL'] || '5s'}"
            chunk_limit_size "#{ENV['FLUENT_ELASTICSEARCH_BUFFER_CHUNK_LIMIT_SIZE'] || '2M'}"
            queue_limit_length "#{ENV['FLUENT_ELASTICSEARCH_BUFFER_QUEUE_LIMIT_LENGTH'] || '8'}"
            retry_limit "#{ENV['FLUENT_ELASTICSEARCH_BUFFER_RETRY_LIMIT'] || '17'}"
            retry_wait "#{ENV['FLUENT_ELASTICSEARCH_BUFFER_RETRY_WAIT'] || '1.0'}"
            max_retry_wait "#{ENV['FLUENT_ELASTICSEARCH_BUFFER_MAX_RETRY_WAIT'] || '30'}"
            overflow_action "#{ENV['FLUENT_ELASTICSEARCH_BUFFER_OVERFLOW_ACTION'] || 'block'}"
          </buffer>
        </match>
    
    tolerations:
      - key: node-role.kubernetes.io/control-plane
        operator: Exists
        effect: NoSchedule
      - key: node-role.kubernetes.io/master
        operator: Exists
        effect: NoSchedule
    
    updateStrategy:
      type: RollingUpdate
      rollingUpdate:
        maxUnavailable: 1
    EOF
  ]
  
  depends_on = [module.eks]
}

# CloudWatch Container Insights
resource "aws_cloudwatch_log_group" "container_insights" {
  name              = "/aws/containerinsights/${module.eks.cluster_name}/application"
  retention_in_days = var.monitoring_config.log_retention_days
  
  tags = local.tags
}

resource "aws_cloudwatch_log_group" "container_insights_dataplane" {
  name              = "/aws/containerinsights/${module.eks.cluster_name}/dataplane"
  retention_in_days = var.monitoring_config.log_retention_days
  
  tags = local.tags
}

resource "aws_cloudwatch_log_group" "container_insights_host" {
  name              = "/aws/containerinsights/${module.eks.cluster_name}/host"
  retention_in_days = var.monitoring_config.log_retention_days
  
  tags = local.tags
}

resource "aws_cloudwatch_log_group" "container_insights_performance" {
  name              = "/aws/containerinsights/${module.eks.cluster_name}/performance"
  retention_in_days = var.monitoring_config.log_retention_days
  
  tags = local.tags
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "cluster_cpu_high" {
  alarm_name          = "${local.name}-cluster-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EKS"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors cluster cpu utilization"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  
  dimensions = {
    ClusterName = module.eks.cluster_name
  }
  
  tags = local.tags
}

resource "aws_cloudwatch_metric_alarm" "cluster_memory_high" {
  alarm_name          = "${local.name}-cluster-memory-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "MemoryUtilization"
  namespace           = "AWS/EKS"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors cluster memory utilization"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  
  dimensions = {
    ClusterName = module.eks.cluster_name
  }
  
  tags = local.tags
}

resource "aws_cloudwatch_metric_alarm" "rds_cpu_high" {
  alarm_name          = "${local.name}-rds-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors RDS cpu utilization"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  
  dimensions = {
    DBInstanceIdentifier = aws_db_instance.main.id
  }
  
  tags = local.tags
}

# SNS Topic for alerts
resource "aws_sns_topic" "alerts" {
  name = "${local.name}-alerts"
  
  tags = local.tags
}

resource "aws_sns_topic_subscription" "email_alerts" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = "alerts@grandmodel.com"
}

# Custom CloudWatch Dashboard
resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "${local.name}-dashboard"
  
  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        
        properties = {
          metrics = [
            ["AWS/EKS", "cluster_failed_request_count", "ClusterName", module.eks.cluster_name],
            [".", "cluster_request_total", ".", "."],
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "EKS Cluster Metrics"
          period  = 300
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        
        properties = {
          metrics = [
            ["AWS/RDS", "CPUUtilization", "DBInstanceIdentifier", aws_db_instance.main.id],
            [".", "DatabaseConnections", ".", "."],
            [".", "ReadLatency", ".", "."],
            [".", "WriteLatency", ".", "."],
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "RDS Metrics"
          period  = 300
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 24
        height = 6
        
        properties = {
          metrics = [
            ["AWS/ElastiCache", "CPUUtilization", "CacheClusterId", aws_elasticache_replication_group.main.id],
            [".", "NetworkBytesIn", ".", "."],
            [".", "NetworkBytesOut", ".", "."],
            [".", "CurrConnections", ".", "."],
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "ElastiCache Metrics"
          period  = 300
        }
      }
    ]
  })
}