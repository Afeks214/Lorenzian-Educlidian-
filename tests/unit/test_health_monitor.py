"""
Unit tests for the HealthMonitor component.
Tests health checks, resource monitoring, and recommendations.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import psutil

from src.monitoring.health_monitor import (
    HealthMonitor, HealthStatus, ComponentHealth,
    ResourceMonitor, get_health_monitor
)


class TestHealthMonitor:
    """Test suite for HealthMonitor."""
    
    @pytest.fixture
    def health_monitor(self):
        """Create a HealthMonitor instance for testing."""
        monitor = HealthMonitor(check_interval=1)  # Fast interval for tests
        return monitor
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        mock = AsyncMock()
        mock.ping.return_value = True
        mock.info.return_value = {"used_memory": "100MB"}
        return mock
    
    def test_initialization(self, health_monitor):
        """Test HealthMonitor initialization."""
        assert health_monitor.check_interval == 1
        assert health_monitor._running is False
        assert isinstance(health_monitor.component_status, dict)
        assert isinstance(health_monitor.check_intervals, dict)
    
    @pytest.mark.asyncio
    async def test_check_redis_healthy(self, health_monitor, mock_redis):
        """Test Redis health check when healthy."""
        health_monitor.redis_client = mock_redis
        
        health = await health_monitor._check_redis()
        
        assert health.name == "redis"
        assert health.status == HealthStatus.HEALTHY
        assert "Connection: OK" in health.message
        mock_redis.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_redis_unhealthy(self, health_monitor):
        """Test Redis health check when unhealthy."""
        mock_redis = AsyncMock()
        mock_redis.ping.side_effect = Exception("Connection refused")
        health_monitor.redis_client = mock_redis
        
        health = await health_monitor._check_redis()
        
        assert health.name == "redis"
        assert health.status == HealthStatus.UNHEALTHY
        assert "Redis health check failed" in health.message
    
    @pytest.mark.asyncio
    async def test_check_models_healthy(self, health_monitor):
        """Test model health check."""
        # Mock model loading
        with patch.object(health_monitor, '_validate_model_inference', return_value=True):
            health = await health_monitor._check_models()
            
            assert health.name == "models"
            assert health.status == HealthStatus.HEALTHY
            assert "loaded" in health.message
    
    @pytest.mark.asyncio
    async def test_check_api_healthy(self, health_monitor):
        """Test API health check."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        
        with patch('httpx.AsyncClient.get', return_value=mock_response):
            health = await health_monitor._check_api()
            
            assert health.name == "api"
            assert health.status == HealthStatus.HEALTHY
            assert "responding" in health.message
    
    @pytest.mark.asyncio
    async def test_check_api_degraded(self, health_monitor):
        """Test API health check when degraded."""
        # Mock slow API response
        async def slow_response(*args, **kwargs):
            await asyncio.sleep(0.2)  # Simulate slow response
            mock = Mock()
            mock.status_code = 200
            return mock
        
        with patch('httpx.AsyncClient.get', side_effect=slow_response):
            health = await health_monitor._check_api()
            
            assert health.name == "api"
            assert health.status == HealthStatus.DEGRADED
            assert "slow" in health.message.lower()
    
    def test_check_system_resources_healthy(self, health_monitor):
        """Test system resource check when healthy."""
        # Mock resource usage
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory', return_value=Mock(percent=60.0)), \
             patch('psutil.disk_usage', return_value=Mock(percent=70.0)):
            
            health = health_monitor._check_system_resources()
            
            assert health.name == "system"
            assert health.status == HealthStatus.HEALTHY
            assert health.details["cpu_percent"] == 50.0
            assert health.details["memory_percent"] == 60.0
            assert health.details["disk_percent"] == 70.0
    
    def test_check_system_resources_degraded(self, health_monitor):
        """Test system resource check when degraded."""
        # Mock high resource usage
        with patch('psutil.cpu_percent', return_value=85.0), \
             patch('psutil.virtual_memory', return_value=Mock(percent=60.0)), \
             patch('psutil.disk_usage', return_value=Mock(percent=70.0)):
            
            health = health_monitor._check_system_resources()
            
            assert health.name == "system"
            assert health.status == HealthStatus.DEGRADED
            assert "CPU usage high" in health.message
    
    def test_check_system_resources_unhealthy(self, health_monitor):
        """Test system resource check when unhealthy."""
        # Mock critical resource usage
        with patch('psutil.cpu_percent', return_value=95.0), \
             patch('psutil.virtual_memory', return_value=Mock(percent=92.0)), \
             patch('psutil.disk_usage', return_value=Mock(percent=96.0)):
            
            health = health_monitor._check_system_resources()
            
            assert health.name == "system"
            assert health.status == HealthStatus.UNHEALTHY
            assert "critical" in health.message.lower()
    
    @pytest.mark.asyncio
    async def test_get_health_status(self, health_monitor):
        """Test getting overall health status."""
        # Mock component statuses
        health_monitor.component_status = {
            "redis": ComponentHealth(
                name="redis",
                status=HealthStatus.HEALTHY,
                message="OK",
                last_check=datetime.utcnow()
            ),
            "models": ComponentHealth(
                name="models",
                status=HealthStatus.HEALTHY,
                message="OK",
                last_check=datetime.utcnow()
            ),
            "system": ComponentHealth(
                name="system",
                status=HealthStatus.HEALTHY,
                message="OK",
                last_check=datetime.utcnow()
            )
        }
        
        status = await health_monitor.get_health_status()
        
        assert status == HealthStatus.HEALTHY
    
    @pytest.mark.asyncio
    async def test_get_health_status_degraded(self, health_monitor):
        """Test overall health when one component is degraded."""
        health_monitor.component_status = {
            "redis": ComponentHealth(
                name="redis",
                status=HealthStatus.HEALTHY,
                message="OK",
                last_check=datetime.utcnow()
            ),
            "models": ComponentHealth(
                name="models",
                status=HealthStatus.DEGRADED,
                message="Slow",
                last_check=datetime.utcnow()
            )
        }
        
        status = await health_monitor.get_health_status()
        
        assert status == HealthStatus.DEGRADED
    
    @pytest.mark.asyncio
    async def test_get_health_status_unhealthy(self, health_monitor):
        """Test overall health when one component is unhealthy."""
        health_monitor.component_status = {
            "redis": ComponentHealth(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message="Down",
                last_check=datetime.utcnow()
            ),
            "models": ComponentHealth(
                name="models",
                status=HealthStatus.HEALTHY,
                message="OK",
                last_check=datetime.utcnow()
            )
        }
        
        status = await health_monitor.get_health_status()
        
        assert status == HealthStatus.UNHEALTHY
    
    def test_generate_recommendations_healthy(self, health_monitor):
        """Test recommendation generation when healthy."""
        health_monitor.component_status = {
            "system": ComponentHealth(
                name="system",
                status=HealthStatus.HEALTHY,
                message="OK",
                details={"cpu_percent": 50.0},
                last_check=datetime.utcnow()
            )
        }
        
        recommendations = health_monitor._generate_recommendations()
        
        assert len(recommendations) == 0
    
    def test_generate_recommendations_degraded(self, health_monitor):
        """Test recommendation generation when degraded."""
        health_monitor.component_status = {
            "system": ComponentHealth(
                name="system",
                status=HealthStatus.DEGRADED,
                message="CPU high",
                details={"cpu_percent": 85.0},
                last_check=datetime.utcnow()
            )
        }
        
        recommendations = health_monitor._generate_recommendations()
        
        assert len(recommendations) > 0
        assert any("CPU" in rec for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_get_detailed_health(self, health_monitor):
        """Test getting detailed health information."""
        # Mock component status
        health_monitor.component_status = {
            "redis": ComponentHealth(
                name="redis",
                status=HealthStatus.HEALTHY,
                message="OK",
                last_check=datetime.utcnow()
            )
        }
        
        detailed = await health_monitor.get_detailed_health()
        
        assert "status" in detailed
        assert "timestamp" in detailed
        assert "components" in detailed
        assert "recommendations" in detailed
        assert "check_intervals" in detailed
        assert "version" in detailed
    
    @pytest.mark.asyncio
    async def test_run_health_checks(self, health_monitor):
        """Test running all health checks."""
        # Mock individual checks
        with patch.object(health_monitor, '_check_redis', 
                         return_value=ComponentHealth(
                             name="redis",
                             status=HealthStatus.HEALTHY,
                             message="OK",
                             last_check=datetime.utcnow()
                         )), \
             patch.object(health_monitor, '_check_models',
                         return_value=ComponentHealth(
                             name="models",
                             status=HealthStatus.HEALTHY,
                             message="OK",
                             last_check=datetime.utcnow()
                         )):
            
            await health_monitor._run_health_checks()
            
            assert "redis" in health_monitor.component_status
            assert "models" in health_monitor.component_status


class TestResourceMonitor:
    """Test suite for ResourceMonitor."""
    
    @pytest.fixture
    def resource_monitor(self):
        """Create ResourceMonitor instance."""
        return ResourceMonitor()
    
    def test_get_cpu_usage(self, resource_monitor):
        """Test CPU usage monitoring."""
        with patch('psutil.cpu_percent', return_value=55.5):
            usage = resource_monitor.get_cpu_usage()
            assert usage == 55.5
    
    def test_get_memory_usage(self, resource_monitor):
        """Test memory usage monitoring."""
        mock_memory = Mock()
        mock_memory.percent = 65.5
        mock_memory.used = 1024 * 1024 * 1024  # 1GB
        mock_memory.total = 2 * 1024 * 1024 * 1024  # 2GB
        
        with patch('psutil.virtual_memory', return_value=mock_memory):
            usage = resource_monitor.get_memory_usage()
            
            assert usage["percent"] == 65.5
            assert usage["used_mb"] == 1024.0
            assert usage["total_mb"] == 2048.0
    
    def test_get_disk_usage(self, resource_monitor):
        """Test disk usage monitoring."""
        mock_disk = Mock()
        mock_disk.percent = 75.0
        mock_disk.used = 50 * 1024 * 1024 * 1024  # 50GB
        mock_disk.free = 20 * 1024 * 1024 * 1024  # 20GB
        
        with patch('psutil.disk_usage', return_value=mock_disk):
            usage = resource_monitor.get_disk_usage()
            
            assert usage["percent"] == 75.0
            assert usage["used_gb"] == 50.0
            assert usage["free_gb"] == 20.0
    
    def test_get_process_info(self, resource_monitor):
        """Test process information gathering."""
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 512 * 1024 * 1024  # 512MB
        mock_process.cpu_percent.return_value = 25.0
        mock_process.num_threads.return_value = 10
        mock_process.connections.return_value = [1, 2, 3]  # 3 connections
        
        with patch('psutil.Process', return_value=mock_process):
            info = resource_monitor.get_process_info()
            
            assert info["memory_mb"] == 512.0
            assert info["cpu_percent"] == 25.0
            assert info["threads"] == 10
            assert info["connections"] == 3
    
    def test_check_resource_thresholds_ok(self, resource_monitor):
        """Test resource threshold checking when all OK."""
        with patch.object(resource_monitor, 'get_cpu_usage', return_value=50.0), \
             patch.object(resource_monitor, 'get_memory_usage', 
                         return_value={"percent": 60.0}), \
             patch.object(resource_monitor, 'get_disk_usage',
                         return_value={"percent": 70.0}):
            
            status, message = resource_monitor.check_resource_thresholds()
            
            assert status == "healthy"
            assert message == "All resources within normal limits"
    
    def test_check_resource_thresholds_warning(self, resource_monitor):
        """Test resource threshold checking with warnings."""
        with patch.object(resource_monitor, 'get_cpu_usage', return_value=85.0), \
             patch.object(resource_monitor, 'get_memory_usage',
                         return_value={"percent": 60.0}):
            
            status, message = resource_monitor.check_resource_thresholds()
            
            assert status == "degraded"
            assert "CPU" in message


class TestHealthMonitorIntegration:
    """Integration tests for HealthMonitor with mocked dependencies."""
    
    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        """Test starting and stopping the health monitor."""
        monitor = HealthMonitor(check_interval=0.1)
        
        # Start monitor
        await monitor.start()
        assert monitor._running is True
        assert monitor._monitor_task is not None
        
        # Let it run briefly
        await asyncio.sleep(0.2)
        
        # Stop monitor
        await monitor.stop()
        assert monitor._running is False
    
    @pytest.mark.asyncio
    async def test_periodic_health_checks(self):
        """Test that health checks run periodically."""
        monitor = HealthMonitor(check_interval=0.1)
        check_count = 0
        
        async def mock_check():
            nonlocal check_count
            check_count += 1
        
        with patch.object(monitor, '_run_health_checks', side_effect=mock_check):
            await monitor.start()
            await asyncio.sleep(0.35)  # Should run ~3 times
            await monitor.stop()
            
            assert check_count >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])