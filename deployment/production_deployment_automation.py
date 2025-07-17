#!/usr/bin/env python3
"""
GrandModel Production Deployment Automation System - Agent 20 Implementation
Enterprise-grade deployment automation with validation, monitoring, and rollback capabilities
"""

import asyncio
import json
import time
import logging
import subprocess
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import yaml
import requests
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import prometheus_client
from prometheus_client.parser import text_string_to_metric_families

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeploymentStatus(Enum):
    """Deployment status enumeration"""
    PENDING = "pending"
    VALIDATING = "validating"
    DEPLOYING = "deploying"
    MONITORING = "monitoring"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"

class DeploymentStrategy(Enum):
    """Deployment strategy enumeration"""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING_UPDATE = "rolling_update"
    RECREATE = "recreate"

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    name: str
    namespace: str
    strategy: DeploymentStrategy
    image: str
    replicas: int
    health_check_timeout: int = 300
    success_threshold: float = 0.99
    latency_threshold: float = 0.01
    rollback_on_failure: bool = True
    validation_steps: List[str] = field(default_factory=list)
    monitoring_duration: int = 600  # 10 minutes
    
@dataclass
class DeploymentResult:
    """Deployment result"""
    name: str
    status: DeploymentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    validation_results: Dict[str, bool] = field(default_factory=dict)

class HealthChecker:
    """Health checking utilities"""
    
    @staticmethod
    async def check_pod_health(k8s_client, namespace: str, label_selector: str) -> bool:
        """Check if pods are healthy"""
        try:
            v1 = client.CoreV1Api(k8s_client)
            pods = v1.list_namespaced_pod(
                namespace=namespace,
                label_selector=label_selector
            )
            
            healthy_count = 0
            total_count = len(pods.items)
            
            for pod in pods.items:
                if pod.status.phase == "Running":
                    # Check readiness
                    if pod.status.conditions:
                        for condition in pod.status.conditions:
                            if condition.type == "Ready" and condition.status == "True":
                                healthy_count += 1
                                break
            
            return healthy_count == total_count and total_count > 0
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    @staticmethod
    async def check_service_health(service_url: str) -> bool:
        """Check if service is responding"""
        try:
            response = requests.get(f"{service_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Service health check failed: {e}")
            return False

class MetricsCollector:
    """Metrics collection and validation"""
    
    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url
    
    async def get_metric_value(self, query: str) -> Optional[float]:
        """Get metric value from Prometheus"""
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data["data"]["result"]:
                    return float(data["data"]["result"][0]["value"][1])
            return None
        except Exception as e:
            logger.error(f"Metric collection failed: {e}")
            return None
    
    async def validate_deployment_metrics(self, deployment_name: str, config: DeploymentConfig) -> Dict[str, bool]:
        """Validate deployment metrics"""
        results = {}
        
        # Success rate validation
        success_rate_query = f'sum(rate({deployment_name}_requests_total{{status!~"5.."}}[5m])) / sum(rate({deployment_name}_requests_total[5m]))'
        success_rate = await self.get_metric_value(success_rate_query)
        results["success_rate"] = success_rate is not None and success_rate >= config.success_threshold
        
        # Latency validation
        latency_query = f'histogram_quantile(0.95, sum(rate({deployment_name}_latency_seconds_bucket[5m])) by (le))'
        latency = await self.get_metric_value(latency_query)
        results["latency"] = latency is not None and latency <= config.latency_threshold
        
        # Error rate validation
        error_rate_query = f'sum(rate({deployment_name}_errors_total[5m])) / sum(rate({deployment_name}_requests_total[5m]))'
        error_rate = await self.get_metric_value(error_rate_query)
        results["error_rate"] = error_rate is not None and error_rate <= 0.01
        
        return results

class RollbackManager:
    """Rollback management"""
    
    def __init__(self, k8s_client):
        self.k8s_client = k8s_client
    
    async def rollback_deployment(self, deployment_name: str, namespace: str) -> bool:
        """Rollback deployment to previous version"""
        try:
            apps_v1 = client.AppsV1Api(self.k8s_client)
            
            # Get current deployment
            deployment = apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            
            # Get previous revision
            current_revision = deployment.metadata.annotations.get("deployment.kubernetes.io/revision", "1")
            previous_revision = str(int(current_revision) - 1)
            
            # Perform rollback
            subprocess.run([
                "kubectl", "rollout", "undo", f"deployment/{deployment_name}",
                f"--namespace={namespace}",
                f"--to-revision={previous_revision}"
            ], check=True)
            
            # Wait for rollback to complete
            subprocess.run([
                "kubectl", "rollout", "status", f"deployment/{deployment_name}",
                f"--namespace={namespace}",
                "--timeout=300s"
            ], check=True)
            
            return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

class ProductionDeploymentManager:
    """Main production deployment manager"""
    
    def __init__(self, kubeconfig_path: Optional[str] = None, prometheus_url: str = "http://prometheus:9090"):
        # Load Kubernetes config
        if kubeconfig_path:
            config.load_kube_config(config_file=kubeconfig_path)
        else:
            config.load_incluster_config()
        
        self.k8s_client = client.ApiClient()
        self.health_checker = HealthChecker()
        self.metrics_collector = MetricsCollector(prometheus_url)
        self.rollback_manager = RollbackManager(self.k8s_client)
        
        # Deployment tracking
        self.active_deployments: Dict[str, DeploymentResult] = {}
    
    async def validate_pre_deployment(self, config: DeploymentConfig) -> Tuple[bool, str]:
        """Validate pre-deployment conditions"""
        logger.info(f"Validating pre-deployment conditions for {config.name}")
        
        # Check namespace exists
        try:
            v1 = client.CoreV1Api(self.k8s_client)
            v1.read_namespace(name=config.namespace)
        except ApiException as e:
            if e.status == 404:
                return False, f"Namespace {config.namespace} does not exist"
            return False, f"Failed to check namespace: {e}"
        
        # Check image exists
        try:
            # This would typically check a container registry
            # For now, we'll assume image validation is handled by the registry
            pass
        except Exception as e:
            return False, f"Image validation failed: {e}"
        
        # Check resource availability
        try:
            # Check if sufficient resources are available
            # This is a simplified check
            pass
        except Exception as e:
            return False, f"Resource validation failed: {e}"
        
        return True, "Pre-deployment validation passed"
    
    async def deploy_blue_green(self, config: DeploymentConfig) -> DeploymentResult:
        """Deploy using blue-green strategy"""
        logger.info(f"Starting blue-green deployment for {config.name}")
        
        result = DeploymentResult(
            name=config.name,
            status=DeploymentStatus.DEPLOYING,
            start_time=datetime.now()
        )
        
        try:
            # Apply blue-green rollout
            subprocess.run([
                "kubectl", "apply", "-f", "/app/k8s/deployment-strategies.yaml",
                f"--namespace={config.namespace}"
            ], check=True)
            
            # Wait for preview deployment
            await asyncio.sleep(30)
            
            # Validate preview deployment
            is_healthy = await self.health_checker.check_pod_health(
                self.k8s_client,
                config.namespace,
                f"app=grandmodel,component={config.name.replace('-deployment', '')}"
            )
            
            if not is_healthy:
                result.status = DeploymentStatus.FAILED
                result.error_message = "Preview deployment health check failed"
                return result
            
            # Promote to active
            subprocess.run([
                "kubectl", "argo", "rollouts", "promote", f"{config.name.replace('-deployment', '-rollout')}",
                f"--namespace={config.namespace}"
            ], check=True)
            
            result.status = DeploymentStatus.COMPLETED
            result.end_time = datetime.now()
            
        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.error_message = str(e)
            logger.error(f"Blue-green deployment failed: {e}")
        
        return result
    
    async def deploy_canary(self, config: DeploymentConfig) -> DeploymentResult:
        """Deploy using canary strategy"""
        logger.info(f"Starting canary deployment for {config.name}")
        
        result = DeploymentResult(
            name=config.name,
            status=DeploymentStatus.DEPLOYING,
            start_time=datetime.now()
        )
        
        try:
            # Apply canary rollout
            subprocess.run([
                "kubectl", "apply", "-f", "/app/k8s/deployment-strategies.yaml",
                f"--namespace={config.namespace}"
            ], check=True)
            
            # Monitor canary phases
            rollout_name = config.name.replace('-deployment', '-rollout')
            
            # Wait for each canary phase
            phases = [20, 40, 60, 80, 100]
            for phase in phases:
                logger.info(f"Monitoring canary phase {phase}%")
                await asyncio.sleep(60)  # Wait between phases
                
                # Check metrics
                metrics = await self.metrics_collector.validate_deployment_metrics(
                    config.name.replace('-deployment', ''),
                    config
                )
                
                if not all(metrics.values()):
                    logger.error(f"Canary metrics validation failed: {metrics}")
                    # Abort canary
                    subprocess.run([
                        "kubectl", "argo", "rollouts", "abort", rollout_name,
                        f"--namespace={config.namespace}"
                    ], check=True)
                    
                    result.status = DeploymentStatus.FAILED
                    result.error_message = f"Canary metrics validation failed: {metrics}"
                    return result
                
                result.validation_results.update(metrics)
            
            result.status = DeploymentStatus.COMPLETED
            result.end_time = datetime.now()
            
        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.error_message = str(e)
            logger.error(f"Canary deployment failed: {e}")
        
        return result
    
    async def monitor_deployment(self, config: DeploymentConfig, result: DeploymentResult) -> None:
        """Monitor deployment post-deployment"""
        logger.info(f"Monitoring deployment {config.name} for {config.monitoring_duration} seconds")
        
        start_time = time.time()
        while time.time() - start_time < config.monitoring_duration:
            try:
                # Check health
                is_healthy = await self.health_checker.check_pod_health(
                    self.k8s_client,
                    config.namespace,
                    f"app=grandmodel,component={config.name.replace('-deployment', '')}"
                )
                
                if not is_healthy:
                    logger.warning(f"Health check failed for {config.name}")
                    if config.rollback_on_failure:
                        await self.rollback_deployment(config, result)
                        return
                
                # Check metrics
                metrics = await self.metrics_collector.validate_deployment_metrics(
                    config.name.replace('-deployment', ''),
                    config
                )
                
                if not all(metrics.values()):
                    logger.warning(f"Metrics validation failed for {config.name}: {metrics}")
                    if config.rollback_on_failure:
                        await self.rollback_deployment(config, result)
                        return
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def rollback_deployment(self, config: DeploymentConfig, result: DeploymentResult) -> None:
        """Rollback failed deployment"""
        logger.info(f"Rolling back deployment {config.name}")
        
        result.status = DeploymentStatus.ROLLING_BACK
        
        success = await self.rollback_manager.rollback_deployment(
            config.name,
            config.namespace
        )
        
        if success:
            result.status = DeploymentStatus.ROLLED_BACK
            logger.info(f"Rollback successful for {config.name}")
        else:
            result.status = DeploymentStatus.FAILED
            result.error_message = "Rollback failed"
            logger.error(f"Rollback failed for {config.name}")
    
    async def deploy(self, config: DeploymentConfig) -> DeploymentResult:
        """Main deployment method"""
        logger.info(f"Starting deployment {config.name} using {config.strategy.value} strategy")
        
        # Pre-deployment validation
        is_valid, message = await self.validate_pre_deployment(config)
        if not is_valid:
            return DeploymentResult(
                name=config.name,
                status=DeploymentStatus.FAILED,
                start_time=datetime.now(),
                end_time=datetime.now(),
                error_message=message
            )
        
        # Track deployment
        self.active_deployments[config.name] = DeploymentResult(
            name=config.name,
            status=DeploymentStatus.PENDING,
            start_time=datetime.now()
        )
        
        # Execute deployment based on strategy
        if config.strategy == DeploymentStrategy.BLUE_GREEN:
            result = await self.deploy_blue_green(config)
        elif config.strategy == DeploymentStrategy.CANARY:
            result = await self.deploy_canary(config)
        else:
            result = DeploymentResult(
                name=config.name,
                status=DeploymentStatus.FAILED,
                start_time=datetime.now(),
                end_time=datetime.now(),
                error_message=f"Unsupported strategy: {config.strategy}"
            )
        
        # Update tracking
        self.active_deployments[config.name] = result
        
        # Start monitoring if deployment succeeded
        if result.status == DeploymentStatus.COMPLETED:
            asyncio.create_task(self.monitor_deployment(config, result))
        
        return result
    
    async def deploy_multiple(self, configs: List[DeploymentConfig]) -> List[DeploymentResult]:
        """Deploy multiple components"""
        tasks = []
        for config in configs:
            task = asyncio.create_task(self.deploy(config))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(DeploymentResult(
                    name=configs[i].name,
                    status=DeploymentStatus.FAILED,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    error_message=str(result)
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    def get_deployment_status(self, deployment_name: str) -> Optional[DeploymentResult]:
        """Get deployment status"""
        return self.active_deployments.get(deployment_name)
    
    def get_all_deployments(self) -> Dict[str, DeploymentResult]:
        """Get all active deployments"""
        return self.active_deployments.copy()

class DeploymentOrchestrator:
    """High-level deployment orchestrator"""
    
    def __init__(self, config_path: str = "/app/config/deployment-config.yaml"):
        self.config_path = config_path
        self.manager = ProductionDeploymentManager()
    
    def load_deployment_configs(self) -> List[DeploymentConfig]:
        """Load deployment configurations"""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            configs = []
            for deployment in config_data.get("deployments", []):
                configs.append(DeploymentConfig(
                    name=deployment["name"],
                    namespace=deployment.get("namespace", "grandmodel"),
                    strategy=DeploymentStrategy(deployment.get("strategy", "rolling_update")),
                    image=deployment["image"],
                    replicas=deployment.get("replicas", 1),
                    health_check_timeout=deployment.get("health_check_timeout", 300),
                    success_threshold=deployment.get("success_threshold", 0.99),
                    latency_threshold=deployment.get("latency_threshold", 0.01),
                    rollback_on_failure=deployment.get("rollback_on_failure", True),
                    validation_steps=deployment.get("validation_steps", []),
                    monitoring_duration=deployment.get("monitoring_duration", 600)
                ))
            
            return configs
        except Exception as e:
            logger.error(f"Failed to load deployment configs: {e}")
            return []
    
    async def deploy_all(self) -> List[DeploymentResult]:
        """Deploy all configured components"""
        configs = self.load_deployment_configs()
        if not configs:
            logger.error("No deployment configurations found")
            return []
        
        return await self.manager.deploy_multiple(configs)
    
    async def deploy_component(self, component_name: str) -> Optional[DeploymentResult]:
        """Deploy specific component"""
        configs = self.load_deployment_configs()
        
        for config in configs:
            if config.name == component_name:
                return await self.manager.deploy(config)
        
        logger.error(f"Component {component_name} not found in configuration")
        return None

# Example usage and testing
async def main():
    """Main function for testing"""
    orchestrator = DeploymentOrchestrator()
    
    # Test deployment configuration
    test_config = DeploymentConfig(
        name="strategic-deployment",
        namespace="grandmodel",
        strategy=DeploymentStrategy.BLUE_GREEN,
        image="grandmodel/strategic-agent:v1.0.0",
        replicas=3,
        health_check_timeout=300,
        success_threshold=0.99,
        latency_threshold=0.002,
        rollback_on_failure=True,
        validation_steps=["health_check", "metrics_validation"],
        monitoring_duration=600
    )
    
    # Deploy single component
    result = await orchestrator.manager.deploy(test_config)
    
    print(f"Deployment Status: {result.status}")
    print(f"Start Time: {result.start_time}")
    print(f"End Time: {result.end_time}")
    print(f"Validation Results: {result.validation_results}")
    
    if result.error_message:
        print(f"Error: {result.error_message}")

if __name__ == "__main__":
    asyncio.run(main())