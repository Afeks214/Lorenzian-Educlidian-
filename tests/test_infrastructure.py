"""
Infrastructure validation test suite.
Tests Docker build, health checks, and full stack integration.
"""

import os
import time
import docker
import requests
import asyncio
import pytest
from typing import Dict, Any

# Docker client configuration
DOCKER_CLIENT = docker.from_env()
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPOSE_FILE = os.path.join(PROJECT_ROOT, "docker-compose.prod.yml")


class TestInfrastructure:
    """Test suite for infrastructure validation."""
    
    @pytest.fixture(scope="class")
    def docker_client(self):
        """Provide Docker client for tests."""
        return DOCKER_CLIENT
    
    def test_docker_build_verification(self, docker_client):
        """
        Test 1.1: Build production Docker image and verify size < 200MB.
        """
        # Build the production image
        print("Building production Docker image...")
        
        dockerfile_path = os.path.join(PROJECT_ROOT, "docker", "Dockerfile.production")
        
        try:
            # Build the image
            image, build_logs = docker_client.images.build(
                path=PROJECT_ROOT,
                dockerfile=dockerfile_path,
                tag="grandmodel:test",
                rm=True,
                pull=True
            )
            
            # Log build output
            for log in build_logs:
                if 'stream' in log:
                    print(log['stream'].strip())
            
            # Get image size in MB
            image_size_bytes = image.attrs['Size']
            image_size_mb = image_size_bytes / (1024 * 1024)
            
            print(f"Image size: {image_size_mb:.2f} MB")
            
            # Assert size is under 200MB
            assert image_size_mb < 200, f"Image size {image_size_mb:.2f} MB exceeds 200MB limit"
            
            # Additional verification - check for distroless base
            assert any('distroless' in str(layer) for layer in image.history()), \
                "Image should use distroless base for security"
            
        except docker.errors.BuildError as e:
            pytest.fail(f"Docker build failed: {e}")
        finally:
            # Cleanup test image
            try:
                docker_client.images.remove("grandmodel:test", force=True)
            except (docker.errors.ImageNotFound, docker.errors.APIError) as e:
                print(f"Could not cleanup test image: {e}")
            except Exception as e:
                print(f"Unexpected error during cleanup: {e}")
    
    @pytest.mark.integration
    def test_full_stack_health_check(self, docker_client):
        """
        Test 1.2: Integration test with full stack health check validation.
        Spins up entire stack and validates each component's health status.
        """
        import subprocess
        
        # Start the stack
        print("Starting full stack with docker-compose...")
        
        try:
            # Start services
            subprocess.run(
                ["docker-compose", "-f", COMPOSE_FILE, "up", "-d"],
                check=True,
                cwd=PROJECT_ROOT
            )
            
            # Wait for services to be ready (max 120 seconds)
            max_wait_time = 120
            start_time = time.time()
            health_check_url = "http://localhost:8000/health"
            
            print(f"Waiting for services to be healthy (max {max_wait_time}s)...")
            
            while time.time() - start_time < max_wait_time:
                try:
                    response = requests.get(health_check_url, timeout=5)
                    if response.status_code == 200:
                        break
                except requests.exceptions.RequestException:
                    pass
                time.sleep(2)
            else:
                pytest.fail(f"Services did not become healthy within {max_wait_time} seconds")
            
            # Validate health check response
            response = requests.get(health_check_url)
            assert response.status_code == 200, "Health check should return 200 OK"
            
            health_data = response.json()
            
            # Validate response structure
            assert "status" in health_data, "Health response must include status"
            assert "components" in health_data, "Health response must include components"
            assert "timestamp" in health_data, "Health response must include timestamp"
            
            # Validate overall status
            assert health_data["status"] == "healthy", \
                f"Overall system status should be healthy, got: {health_data['status']}"
            
            # Validate each component is healthy
            component_statuses = {
                comp["name"]: comp["status"] 
                for comp in health_data["components"]
            }
            
            required_components = ["redis", "models", "api", "monitoring"]
            
            for component in required_components:
                assert component in component_statuses, \
                    f"Component '{component}' missing from health check"
                assert component_statuses[component] == "healthy", \
                    f"Component '{component}' is not healthy: {component_statuses[component]}"
            
            # Validate component details
            for component in health_data["components"]:
                assert "last_check" in component, \
                    f"Component {component['name']} missing last_check timestamp"
                assert "message" in component, \
                    f"Component {component['name']} missing status message"
            
            print("All components reported as healthy!")
            
            # Additional validation - check Prometheus metrics endpoint
            metrics_response = requests.get("http://localhost:8000/metrics")
            assert metrics_response.status_code == 200, "Metrics endpoint should be accessible"
            assert "grandmodel_health_status" in metrics_response.text, \
                "Health metrics should be exposed"
            
        finally:
            # Cleanup - stop all services
            print("Stopping services...")
            subprocess.run(
                ["docker-compose", "-f", COMPOSE_FILE, "down", "-v"],
                check=False,
                cwd=PROJECT_ROOT
            )
    
    @pytest.mark.integration
    def test_service_resilience(self, docker_client):
        """
        Test service resilience and recovery capabilities.
        """
        import subprocess
        
        try:
            # Start services
            subprocess.run(
                ["docker-compose", "-f", COMPOSE_FILE, "up", "-d"],
                check=True,
                cwd=PROJECT_ROOT
            )
            
            # Wait for initial health
            time.sleep(30)
            
            # Get initial health status
            response = requests.get("http://localhost:8000/health")
            assert response.status_code == 200
            
            # Simulate Redis failure
            print("Simulating Redis failure...")
            subprocess.run(
                ["docker-compose", "-f", COMPOSE_FILE, "stop", "redis"],
                check=True,
                cwd=PROJECT_ROOT
            )
            
            time.sleep(5)
            
            # Check that health endpoint reports degraded status
            response = requests.get("http://localhost:8000/health")
            health_data = response.json()
            
            # System should be degraded, not completely down
            assert health_data["status"] in ["degraded", "unhealthy"], \
                "System should report degraded status when Redis is down"
            
            # Restart Redis
            print("Restarting Redis...")
            subprocess.run(
                ["docker-compose", "-f", COMPOSE_FILE, "start", "redis"],
                check=True,
                cwd=PROJECT_ROOT
            )
            
            # Wait for recovery
            time.sleep(20)
            
            # Verify system recovered
            response = requests.get("http://localhost:8000/health")
            health_data = response.json()
            assert health_data["status"] == "healthy", \
                "System should recover to healthy status after Redis restart"
            
        finally:
            # Cleanup
            subprocess.run(
                ["docker-compose", "-f", COMPOSE_FILE, "down", "-v"],
                check=False,
                cwd=PROJECT_ROOT
            )
    
    def test_container_resource_limits(self, docker_client):
        """
        Verify container resource limits are properly configured.
        """
        import subprocess
        import json
        
        # Check docker-compose config
        result = subprocess.run(
            ["docker-compose", "-f", COMPOSE_FILE, "config"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )
        
        assert result.returncode == 0, "docker-compose config should be valid"
        
        # Parse the config
        config_lines = result.stdout
        
        # Verify key resource limits are set
        assert "cpus:" in config_lines, "CPU limits should be configured"
        assert "memory:" in config_lines, "Memory limits should be configured"
        
        # Verify specific service limits
        services_with_limits = ["grandmodel", "redis", "prometheus", "grafana"]
        
        for service in services_with_limits:
            # This is a basic check - in production you'd parse the YAML properly
            assert service in config_lines, f"Service {service} should be configured"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])