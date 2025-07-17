# Variables for GrandModel Production Infrastructure - Agent 20 Implementation

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
  
  validation {
    condition     = contains(["production", "staging", "development"], var.environment)
    error_message = "Environment must be one of: production, staging, development."
  }
}

variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = "grandmodel.production.local"
}

variable "cluster_version" {
  description = "Kubernetes version for the EKS cluster"
  type        = string
  default     = "1.28"
}

variable "node_group_instance_types" {
  description = "Instance types for EKS node groups"
  type = object({
    strategic = list(string)
    tactical  = list(string)
    risk      = list(string)
    system    = list(string)
  })
  default = {
    strategic = ["m5.xlarge", "m5.2xlarge"]
    tactical  = ["m5.large", "m5.xlarge"]
    risk      = ["m5.large"]
    system    = ["m5.large"]
  }
}

variable "node_group_scaling" {
  description = "Scaling configuration for EKS node groups"
  type = object({
    strategic = object({
      min_size     = number
      max_size     = number
      desired_size = number
    })
    tactical = object({
      min_size     = number
      max_size     = number
      desired_size = number
    })
    risk = object({
      min_size     = number
      max_size     = number
      desired_size = number
    })
    system = object({
      min_size     = number
      max_size     = number
      desired_size = number
    })
  })
  default = {
    strategic = {
      min_size     = 3
      max_size     = 10
      desired_size = 3
    }
    tactical = {
      min_size     = 5
      max_size     = 20
      desired_size = 5
    }
    risk = {
      min_size     = 3
      max_size     = 8
      desired_size = 3
    }
    system = {
      min_size     = 2
      max_size     = 6
      desired_size = 2
    }
  }
}

variable "database_config" {
  description = "Database configuration"
  type = object({
    instance_class        = string
    allocated_storage     = number
    max_allocated_storage = number
    backup_retention_days = number
    multi_az             = bool
    deletion_protection  = bool
  })
  default = {
    instance_class        = "db.r5.xlarge"
    allocated_storage     = 100
    max_allocated_storage = 1000
    backup_retention_days = 30
    multi_az             = true
    deletion_protection  = true
  }
}

variable "redis_config" {
  description = "Redis configuration"
  type = object({
    node_type              = string
    num_cache_clusters     = number
    automatic_failover     = bool
    multi_az              = bool
    snapshot_retention    = number
  })
  default = {
    node_type              = "cache.r6g.large"
    num_cache_clusters     = 3
    automatic_failover     = true
    multi_az              = true
    snapshot_retention    = 7
  }
}

variable "monitoring_config" {
  description = "Monitoring configuration"
  type = object({
    log_retention_days     = number
    enable_flow_logs      = bool
    enable_performance_insights = bool
    monitoring_interval   = number
  })
  default = {
    log_retention_days     = 14
    enable_flow_logs      = true
    enable_performance_insights = true
    monitoring_interval   = 60
  }
}

variable "security_config" {
  description = "Security configuration"
  type = object({
    enable_encryption          = bool
    enable_key_rotation       = bool
    deletion_window_days      = number
    enable_secrets_manager    = bool
  })
  default = {
    enable_encryption          = true
    enable_key_rotation       = true
    deletion_window_days      = 7
    enable_secrets_manager    = true
  }
}

variable "backup_config" {
  description = "Backup configuration"
  type = object({
    s3_lifecycle_days     = number
    backup_window        = string
    maintenance_window   = string
    snapshot_window      = string
  })
  default = {
    s3_lifecycle_days     = 90
    backup_window        = "03:00-04:00"
    maintenance_window   = "sun:04:00-sun:05:00"
    snapshot_window      = "03:00-05:00"
  }
}

variable "network_config" {
  description = "Network configuration"
  type = object({
    vpc_cidr                = string
    enable_nat_gateway     = bool
    single_nat_gateway     = bool
    enable_dns_hostnames   = bool
    enable_dns_support     = bool
  })
  default = {
    vpc_cidr                = "10.0.0.0/16"
    enable_nat_gateway     = true
    single_nat_gateway     = false
    enable_dns_hostnames   = true
    enable_dns_support     = true
  }
}

variable "tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}

variable "enable_disaster_recovery" {
  description = "Enable disaster recovery features"
  type        = bool
  default     = true
}

variable "disaster_recovery_region" {
  description = "Secondary region for disaster recovery"
  type        = string
  default     = "us-west-2"
}

variable "compliance_requirements" {
  description = "Compliance requirements configuration"
  type = object({
    enable_compliance_logging = bool
    enable_access_logging    = bool
    enable_audit_logging     = bool
    log_retention_years      = number
  })
  default = {
    enable_compliance_logging = true
    enable_access_logging    = true
    enable_audit_logging     = true
    log_retention_years      = 7
  }
}

variable "cost_optimization" {
  description = "Cost optimization configuration"
  type = object({
    enable_spot_instances    = bool
    spot_instance_percentage = number
    enable_scheduled_scaling = bool
  })
  default = {
    enable_spot_instances    = false
    spot_instance_percentage = 0
    enable_scheduled_scaling = true
  }
}

variable "performance_config" {
  description = "Performance configuration"
  type = object({
    enable_enhanced_networking = bool
    enable_sr_iov             = bool
    enable_placement_groups    = bool
  })
  default = {
    enable_enhanced_networking = true
    enable_sr_iov             = true
    enable_placement_groups    = true
  }
}