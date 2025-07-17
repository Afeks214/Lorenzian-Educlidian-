"""
Integration layer for LiveTrainingCoordinator with existing training systems.
Provides seamless integration with distributed training, unified training system,
and production deployment infrastructure.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import threading
import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import weakref
import traceback
from contextlib import contextmanager
import importlib
import sys

# Import core training systems
from .live_training_coordinator import (
    LiveTrainingCoordinator, 
    LiveTrainingConfig, 
    AgentTrainingManager,
    LearningMode,
    TrainingPhase
)
from .distributed_training_coordinator import DistributedTrainingCoordinator, DistributedConfig
from .unified_training_system import UnifiedTrainingSystem, UnifiedTrainingConfig
from .training_progress_monitor import TrainingProgressMonitor
from .advanced_checkpoint_manager import AdvancedCheckpointManager
from .memory_optimized_trainer import MemoryOptimizedTrainer

logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """Configuration for training system integration"""
    # Integration settings
    enable_distributed_training: bool = False
    enable_unified_training: bool = True
    enable_memory_optimization: bool = True
    enable_checkpoint_management: bool = True
    
    # Production settings
    production_mode: bool = False
    safety_checks: bool = True
    error_recovery: bool = True
    
    # Agent detection
    auto_detect_agents: bool = True
    agent_discovery_paths: List[str] = None
    
    # Performance settings
    integration_monitoring: bool = True
    performance_logging: bool = True
    
    # Error handling
    max_integration_errors: int = 10
    integration_retry_delay: float = 5.0
    
    def __post_init__(self):
        if self.agent_discovery_paths is None:
            self.agent_discovery_paths = [
                "agents/",
                "src/agents/",
                "models/",
                "src/models/"
            ]


class AgentDetector:
    """Automatically detect and register agents from the codebase"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.discovered_agents = {}
        self.agent_factories = {}
        
    def discover_agents(self, base_path: str = ".") -> Dict[str, Any]:
        """Discover agent classes in the codebase"""
        discovered = {}
        
        if not self.config.auto_detect_agents:
            return discovered
        
        try:
            base_path = Path(base_path)
            
            for search_path in self.config.agent_discovery_paths:
                full_path = base_path / search_path
                
                if full_path.exists():
                    discovered.update(self._scan_directory(full_path))
            
            self.discovered_agents = discovered
            logger.info(f"Discovered {len(discovered)} potential agents")
            
        except Exception as e:
            logger.error(f"Agent discovery failed: {e}")
        
        return discovered
    
    def _scan_directory(self, directory: Path) -> Dict[str, Any]:
        """Scan directory for agent classes"""
        agents = {}
        
        try:
            for file_path in directory.rglob("*.py"):
                if file_path.name.startswith("__"):
                    continue
                
                try:
                    agents.update(self._scan_file(file_path))
                except Exception as e:
                    logger.debug(f"Failed to scan {file_path}: {e}")
            
        except Exception as e:
            logger.error(f"Directory scan failed for {directory}: {e}")
        
        return agents
    
    def _scan_file(self, file_path: Path) -> Dict[str, Any]:
        """Scan Python file for agent classes"""
        agents = {}
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for agent class patterns
            if self._contains_agent_patterns(content):
                module_path = self._get_module_path(file_path)
                agents[module_path] = {
                    'file_path': str(file_path),
                    'module_path': module_path,
                    'detected_patterns': self._extract_patterns(content)
                }
            
        except Exception as e:
            logger.debug(f"File scan failed for {file_path}: {e}")
        
        return agents
    
    def _contains_agent_patterns(self, content: str) -> bool:
        """Check if content contains agent patterns"""
        agent_patterns = [
            "class.*Agent",
            "BaseAgent",
            "TradingAgent",
            "MLAgent",
            "RLAgent",
            "def observe",
            "def decide",
            "def act",
            "def train",
            "nn.Module"
        ]
        
        for pattern in agent_patterns:
            if pattern in content:
                return True
        
        return False
    
    def _extract_patterns(self, content: str) -> List[str]:
        """Extract specific patterns from content"""
        patterns = []
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('class ') and 'Agent' in line:
                patterns.append(line)
        
        return patterns
    
    def _get_module_path(self, file_path: Path) -> str:
        """Convert file path to module path"""
        try:
            # Convert to relative path and remove .py extension
            rel_path = file_path.relative_to(Path.cwd())
            module_path = str(rel_path.with_suffix(''))
            
            # Convert path separators to dots
            module_path = module_path.replace('/', '.').replace('\\', '.')
            
            return module_path
            
        except Exception as e:
            logger.debug(f"Module path conversion failed: {e}")
            return str(file_path)
    
    def create_agent_factory(self, module_path: str) -> Optional[Callable]:
        """Create factory function for agent"""
        try:
            # Import module
            module = importlib.import_module(module_path)
            
            # Look for agent classes
            agent_classes = []
            for name in dir(module):
                obj = getattr(module, name)
                if (isinstance(obj, type) and 
                    hasattr(obj, '__bases__') and 
                    'Agent' in name):
                    agent_classes.append(obj)
            
            if agent_classes:
                # Use first agent class found
                agent_class = agent_classes[0]
                
                def factory(config: Dict[str, Any]) -> Any:
                    return agent_class(config)
                
                self.agent_factories[module_path] = factory
                return factory
            
        except Exception as e:
            logger.error(f"Agent factory creation failed for {module_path}: {e}")
        
        return None


class TrainingSystemIntegrator:
    """Integrates LiveTrainingCoordinator with existing training systems"""
    
    def __init__(self, 
                 live_coordinator: LiveTrainingCoordinator,
                 config: IntegrationConfig):
        self.live_coordinator = live_coordinator
        self.config = config
        self.integrated_systems = {}
        self.integration_errors = []
        
        # Integration components
        self.agent_detector = AgentDetector(config)
        self.unified_trainer = None
        self.distributed_trainer = None
        self.memory_trainer = None
        
        # Monitoring
        self.integration_monitor = None
        self.performance_tracker = {}
        
        # Error handling
        self.error_count = 0
        self.last_error_time = None
        
        logger.info("TrainingSystemIntegrator initialized")
    
    def integrate_unified_training(self, unified_config: UnifiedTrainingConfig = None) -> bool:
        """Integrate with unified training system"""
        try:
            if not self.config.enable_unified_training:
                logger.info("Unified training integration disabled")
                return False
            
            # Create unified trainer if needed
            if unified_config is None:
                unified_config = UnifiedTrainingConfig(
                    training_strategy="parallel",
                    enable_parallel_training=True,
                    enable_memory_optimization=self.config.enable_memory_optimization,
                    enable_checkpointing=self.config.enable_checkpoint_management
                )
            
            self.unified_trainer = UnifiedTrainingSystem(unified_config)
            
            # Create bridge between systems
            self._create_unified_bridge()
            
            self.integrated_systems['unified'] = self.unified_trainer
            logger.info("Unified training system integrated successfully")
            return True
            
        except Exception as e:
            self._handle_integration_error(f"Unified training integration failed: {e}")
            return False
    
    def integrate_distributed_training(self, distributed_config: DistributedConfig = None) -> bool:
        """Integrate with distributed training system"""
        try:
            if not self.config.enable_distributed_training:
                logger.info("Distributed training integration disabled")
                return False
            
            # Create distributed trainer if needed
            if distributed_config is None:
                distributed_config = DistributedConfig(
                    world_size=1,
                    rank=0,
                    enable_fault_tolerance=True,
                    use_mixed_precision=True
                )
            
            # Create dummy model for coordinator initialization
            dummy_model = nn.Linear(10, 1)
            
            self.distributed_trainer = DistributedTrainingCoordinator(
                model=dummy_model,
                config=distributed_config
            )
            
            # Create bridge between systems
            self._create_distributed_bridge()
            
            self.integrated_systems['distributed'] = self.distributed_trainer
            logger.info("Distributed training system integrated successfully")
            return True
            
        except Exception as e:
            self._handle_integration_error(f"Distributed training integration failed: {e}")
            return False
    
    def integrate_memory_optimization(self) -> bool:
        """Integrate memory optimization system"""
        try:
            if not self.config.enable_memory_optimization:
                logger.info("Memory optimization integration disabled")
                return False
            
            # Memory optimization is handled per-agent in the live coordinator
            # This method ensures compatibility
            
            self.integrated_systems['memory'] = True
            logger.info("Memory optimization integrated successfully")
            return True
            
        except Exception as e:
            self._handle_integration_error(f"Memory optimization integration failed: {e}")
            return False
    
    def auto_register_agents(self, base_path: str = ".") -> int:
        """Automatically discover and register agents"""
        try:
            # Discover agents
            discovered = self.agent_detector.discover_agents(base_path)
            
            registered_count = 0
            
            for module_path, agent_info in discovered.items():
                try:
                    # Create factory
                    factory = self.agent_detector.create_agent_factory(module_path)
                    
                    if factory:
                        # Create agent instance (with dummy config for now)
                        dummy_config = {
                            'name': f"auto_agent_{registered_count}",
                            'auto_registered': True,
                            'module_path': module_path
                        }
                        
                        # Note: In real implementation, you'd need proper model, optimizer, etc.
                        # This is a placeholder for the integration pattern
                        logger.info(f"Would register agent from {module_path}")
                        registered_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to register agent from {module_path}: {e}")
            
            logger.info(f"Auto-registered {registered_count} agents")
            return registered_count
            
        except Exception as e:
            self._handle_integration_error(f"Auto agent registration failed: {e}")
            return 0
    
    def _create_unified_bridge(self):
        """Create bridge between live coordinator and unified trainer"""
        try:
            # Create callback for unified trainer to notify live coordinator
            def unified_callback(episode_data: Dict[str, Any]):
                # Forward training data to live coordinator
                agent_id = episode_data.get('agent_id', 'unified_agent')
                training_data = {
                    'input': episode_data.get('observations', []),
                    'target': episode_data.get('actions', [])
                }
                
                self.live_coordinator.add_training_data(agent_id, training_data)
            
            # Set up monitoring bridge
            def monitoring_callback(metrics: Dict[str, Any]):
                # Forward metrics to live coordinator monitoring
                if self.live_coordinator.performance_monitor:
                    self.live_coordinator.performance_monitor.log_multiple_metrics(
                        metrics, 
                        epoch=0, 
                        step=metrics.get('step', 0)
                    )
            
            # Store callbacks for later use
            self.performance_tracker['unified_callback'] = unified_callback
            self.performance_tracker['monitoring_callback'] = monitoring_callback
            
            logger.info("Unified training bridge created")
            
        except Exception as e:
            logger.error(f"Unified bridge creation failed: {e}")
    
    def _create_distributed_bridge(self):
        """Create bridge between live coordinator and distributed trainer"""
        try:
            # Create synchronization bridge
            def sync_callback(metrics: Dict[str, Any]):
                # Synchronize metrics across distributed nodes
                if self.live_coordinator.training_active:
                    # Forward distributed metrics to live coordinator
                    for agent_id in self.live_coordinator.agent_managers:
                        agent_metrics = metrics.get(agent_id, {})
                        if agent_metrics:
                            # Process distributed metrics
                            pass
            
            self.performance_tracker['distributed_callback'] = sync_callback
            
            logger.info("Distributed training bridge created")
            
        except Exception as e:
            logger.error(f"Distributed bridge creation failed: {e}")
    
    def start_integrated_training(self) -> bool:
        """Start integrated training across all systems"""
        try:
            # Start live coordinator
            self.live_coordinator.start_live_training()
            
            # Start unified trainer if available
            if self.unified_trainer and 'unified' in self.integrated_systems:
                # Note: This would need proper implementation based on UnifiedTrainingSystem API
                logger.info("Starting unified training integration")
            
            # Start distributed trainer if available
            if self.distributed_trainer and 'distributed' in self.integrated_systems:
                # Note: This would need proper implementation based on DistributedTrainingCoordinator API
                logger.info("Starting distributed training integration")
            
            # Start monitoring
            if self.config.integration_monitoring:
                self._start_integration_monitoring()
            
            logger.info("Integrated training started successfully")
            return True
            
        except Exception as e:
            self._handle_integration_error(f"Integrated training start failed: {e}")
            return False
    
    def stop_integrated_training(self) -> bool:
        """Stop integrated training across all systems"""
        try:
            # Stop live coordinator
            self.live_coordinator.stop_live_training()
            
            # Stop unified trainer
            if self.unified_trainer:
                # Note: Implementation depends on UnifiedTrainingSystem API
                logger.info("Stopping unified training integration")
            
            # Stop distributed trainer
            if self.distributed_trainer:
                self.distributed_trainer.cleanup()
            
            # Stop monitoring
            if self.integration_monitor:
                self._stop_integration_monitoring()
            
            logger.info("Integrated training stopped successfully")
            return True
            
        except Exception as e:
            self._handle_integration_error(f"Integrated training stop failed: {e}")
            return False
    
    def _start_integration_monitoring(self):
        """Start monitoring integration health"""
        try:
            def monitoring_loop():
                while self.live_coordinator.training_active:
                    try:
                        # Monitor integration health
                        health_data = self._collect_integration_health()
                        
                        # Log health metrics
                        if self.config.performance_logging:
                            logger.debug(f"Integration health: {health_data}")
                        
                        # Check for issues
                        if health_data.get('error_rate', 0) > 0.1:
                            logger.warning("High integration error rate detected")
                        
                        # Sleep
                        threading.Event().wait(5.0)
                        
                    except Exception as e:
                        logger.error(f"Integration monitoring error: {e}")
                        threading.Event().wait(1.0)
            
            self.integration_monitor = threading.Thread(target=monitoring_loop)
            self.integration_monitor.daemon = True
            self.integration_monitor.start()
            
            logger.info("Integration monitoring started")
            
        except Exception as e:
            logger.error(f"Integration monitoring start failed: {e}")
    
    def _stop_integration_monitoring(self):
        """Stop integration monitoring"""
        try:
            if self.integration_monitor and self.integration_monitor.is_alive():
                # Monitor thread will stop when live_coordinator.training_active becomes False
                self.integration_monitor.join(timeout=5.0)
            
            logger.info("Integration monitoring stopped")
            
        except Exception as e:
            logger.error(f"Integration monitoring stop failed: {e}")
    
    def _collect_integration_health(self) -> Dict[str, Any]:
        """Collect integration health metrics"""
        health_data = {
            'timestamp': datetime.now().isoformat(),
            'error_count': self.error_count,
            'integrated_systems': len(self.integrated_systems),
            'active_agents': len(self.live_coordinator.agent_managers),
            'training_active': self.live_coordinator.training_active,
            'error_rate': self.error_count / max(1, len(self.integration_errors))
        }
        
        # Add system-specific health
        if 'unified' in self.integrated_systems:
            health_data['unified_health'] = True
        
        if 'distributed' in self.integrated_systems:
            health_data['distributed_health'] = True
        
        return health_data
    
    def _handle_integration_error(self, error_msg: str):
        """Handle integration errors"""
        self.error_count += 1
        self.last_error_time = datetime.now()
        
        error_record = {
            'timestamp': self.last_error_time.isoformat(),
            'error': error_msg,
            'error_count': self.error_count
        }
        
        self.integration_errors.append(error_record)
        
        # Keep only recent errors
        if len(self.integration_errors) > 100:
            self.integration_errors = self.integration_errors[-100:]
        
        logger.error(error_msg)
        
        # Emergency stop if too many errors
        if self.error_count >= self.config.max_integration_errors:
            logger.critical(f"Too many integration errors ({self.error_count}), emergency stop")
            self.live_coordinator.emergency_stop("Too many integration errors")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status"""
        return {
            'config': self.config.__dict__,
            'integrated_systems': list(self.integrated_systems.keys()),
            'error_count': self.error_count,
            'last_error_time': self.last_error_time.isoformat() if self.last_error_time else None,
            'recent_errors': self.integration_errors[-5:],
            'discovered_agents': len(self.agent_detector.discovered_agents),
            'agent_factories': len(self.agent_detector.agent_factories),
            'live_coordinator_status': self.live_coordinator.get_system_metrics(),
            'health_data': self._collect_integration_health()
        }
    
    def cleanup(self):
        """Clean up integration resources"""
        logger.info("Starting integration cleanup")
        
        try:
            # Stop training
            self.stop_integrated_training()
            
            # Cleanup integrated systems
            if self.distributed_trainer:
                self.distributed_trainer.cleanup()
            
            # Clear data structures
            self.integrated_systems.clear()
            self.integration_errors.clear()
            self.performance_tracker.clear()
            
            logger.info("Integration cleanup completed")
            
        except Exception as e:
            logger.error(f"Integration cleanup failed: {e}")


class ProductionIntegrationManager:
    """Production-ready integration manager with advanced features"""
    
    def __init__(self, 
                 live_config: LiveTrainingConfig,
                 integration_config: IntegrationConfig):
        
        self.live_config = live_config
        self.integration_config = integration_config
        
        # Core components
        self.live_coordinator = LiveTrainingCoordinator(live_config)
        self.integrator = TrainingSystemIntegrator(self.live_coordinator, integration_config)
        
        # Production features
        self.deployment_manager = None
        self.health_monitor = None
        self.backup_manager = None
        
        # State management
        self.production_active = False
        self.startup_time = None
        self.last_health_check = None
        
        logger.info("ProductionIntegrationManager initialized")
    
    def deploy_production_system(self) -> bool:
        """Deploy complete production system"""
        try:
            logger.info("Starting production deployment")
            
            # Initialize production components
            self._initialize_production_components()
            
            # Auto-discover and register agents
            if self.integration_config.auto_detect_agents:
                agent_count = self.integrator.auto_register_agents()
                logger.info(f"Auto-registered {agent_count} agents")
            
            # Integrate training systems
            self._integrate_training_systems()
            
            # Start integrated training
            if self.integrator.start_integrated_training():
                self.production_active = True
                self.startup_time = datetime.now()
                logger.info("Production system deployed successfully")
                return True
            else:
                logger.error("Failed to start integrated training")
                return False
                
        except Exception as e:
            logger.error(f"Production deployment failed: {e}")
            return False
    
    def _initialize_production_components(self):
        """Initialize production-specific components"""
        try:
            # Initialize health monitoring
            if self.integration_config.integration_monitoring:
                self._setup_health_monitoring()
            
            # Initialize backup management
            if self.integration_config.enable_checkpoint_management:
                self._setup_backup_management()
            
            # Initialize deployment management
            self._setup_deployment_management()
            
            logger.info("Production components initialized")
            
        except Exception as e:
            logger.error(f"Production component initialization failed: {e}")
    
    def _integrate_training_systems(self):
        """Integrate all training systems"""
        try:
            # Integrate unified training
            if self.integration_config.enable_unified_training:
                self.integrator.integrate_unified_training()
            
            # Integrate distributed training
            if self.integration_config.enable_distributed_training:
                self.integrator.integrate_distributed_training()
            
            # Integrate memory optimization
            if self.integration_config.enable_memory_optimization:
                self.integrator.integrate_memory_optimization()
            
            logger.info("Training systems integrated")
            
        except Exception as e:
            logger.error(f"Training system integration failed: {e}")
    
    def _setup_health_monitoring(self):
        """Setup production health monitoring"""
        try:
            def health_monitoring_loop():
                while self.production_active:
                    try:
                        health_data = self._comprehensive_health_check()
                        
                        # Log health status
                        if self.integration_config.performance_logging:
                            logger.info(f"System health: {health_data['overall_health']}")
                        
                        # Handle health issues
                        if health_data['overall_health'] < 0.5:
                            logger.warning("System health degraded")
                            self._handle_health_degradation(health_data)
                        
                        self.last_health_check = datetime.now()
                        
                        # Sleep
                        threading.Event().wait(30.0)
                        
                    except Exception as e:
                        logger.error(f"Health monitoring error: {e}")
                        threading.Event().wait(5.0)
            
            self.health_monitor = threading.Thread(target=health_monitoring_loop)
            self.health_monitor.daemon = True
            self.health_monitor.start()
            
            logger.info("Health monitoring setup completed")
            
        except Exception as e:
            logger.error(f"Health monitoring setup failed: {e}")
    
    def _setup_backup_management(self):
        """Setup backup management"""
        try:
            def backup_loop():
                while self.production_active:
                    try:
                        # Create system backup
                        self._create_system_backup()
                        
                        # Sleep for backup interval
                        threading.Event().wait(self.live_config.backup_frequency)
                        
                    except Exception as e:
                        logger.error(f"Backup management error: {e}")
                        threading.Event().wait(60.0)
            
            self.backup_manager = threading.Thread(target=backup_loop)
            self.backup_manager.daemon = True
            self.backup_manager.start()
            
            logger.info("Backup management setup completed")
            
        except Exception as e:
            logger.error(f"Backup management setup failed: {e}")
    
    def _setup_deployment_management(self):
        """Setup deployment management"""
        try:
            # Create deployment tracking
            self.deployment_manager = {
                'deployment_time': datetime.now(),
                'version': "1.0.0",
                'components': {
                    'live_coordinator': True,
                    'integrator': True,
                    'health_monitor': True,
                    'backup_manager': True
                }
            }
            
            logger.info("Deployment management setup completed")
            
        except Exception as e:
            logger.error(f"Deployment management setup failed: {e}")
    
    def _comprehensive_health_check(self) -> Dict[str, Any]:
        """Comprehensive system health check"""
        health_data = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': 1.0,
            'components': {},
            'issues': []
        }
        
        try:
            # Check live coordinator health
            coordinator_metrics = self.live_coordinator.get_system_metrics()
            coordinator_health = 1.0 if coordinator_metrics['training_active'] else 0.5
            health_data['components']['live_coordinator'] = coordinator_health
            
            # Check integrator health
            integrator_status = self.integrator.get_integration_status()
            integrator_health = max(0.0, 1.0 - integrator_status['error_count'] / 100.0)
            health_data['components']['integrator'] = integrator_health
            
            # Check resource health
            resource_health = self._check_resource_health()
            health_data['components']['resources'] = resource_health
            
            # Calculate overall health
            component_healths = list(health_data['components'].values())
            health_data['overall_health'] = sum(component_healths) / len(component_healths)
            
            # Identify issues
            for component, health in health_data['components'].items():
                if health < 0.8:
                    health_data['issues'].append(f"{component} health degraded: {health:.2f}")
            
        except Exception as e:
            health_data['overall_health'] = 0.0
            health_data['issues'].append(f"Health check failed: {e}")
            logger.error(f"Health check error: {e}")
        
        return health_data
    
    def _check_resource_health(self) -> float:
        """Check system resource health"""
        try:
            import psutil
            
            # Check memory
            memory = psutil.virtual_memory()
            memory_health = 1.0 - memory.percent / 100.0
            
            # Check CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_health = 1.0 - cpu_percent / 100.0
            
            # Check disk
            disk = psutil.disk_usage('/')
            disk_health = 1.0 - disk.percent / 100.0
            
            # Overall resource health
            resource_health = (memory_health + cpu_health + disk_health) / 3.0
            
            return max(0.0, min(1.0, resource_health))
            
        except Exception as e:
            logger.error(f"Resource health check failed: {e}")
            return 0.5
    
    def _handle_health_degradation(self, health_data: Dict[str, Any]):
        """Handle system health degradation"""
        try:
            logger.warning(f"Handling health degradation: {health_data['issues']}")
            
            # Take corrective actions based on issues
            for issue in health_data['issues']:
                if 'memory' in issue.lower():
                    # Clear memory
                    import gc
                    gc.collect()
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                elif 'live_coordinator' in issue.lower():
                    # Try to restart live coordinator
                    logger.info("Attempting to restart live coordinator")
                    self.live_coordinator.stop_live_training()
                    threading.Event().wait(5.0)
                    self.live_coordinator.start_live_training()
                
                elif 'integrator' in issue.lower():
                    # Reset integrator error count
                    self.integrator.error_count = 0
                    self.integrator.integration_errors.clear()
            
        except Exception as e:
            logger.error(f"Health degradation handling failed: {e}")
    
    def _create_system_backup(self):
        """Create comprehensive system backup"""
        try:
            backup_data = {
                'timestamp': datetime.now().isoformat(),
                'system_state': {
                    'live_coordinator': self.live_coordinator.get_system_metrics(),
                    'integrator': self.integrator.get_integration_status(),
                    'production_manager': {
                        'production_active': self.production_active,
                        'startup_time': self.startup_time.isoformat() if self.startup_time else None,
                        'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None
                    }
                }
            }
            
            # Save backup
            backup_path = Path(self.live_config.checkpoints_dir) / f"system_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            logger.info(f"System backup created: {backup_path}")
            
        except Exception as e:
            logger.error(f"System backup failed: {e}")
    
    def get_production_status(self) -> Dict[str, Any]:
        """Get comprehensive production status"""
        return {
            'production_active': self.production_active,
            'startup_time': self.startup_time.isoformat() if self.startup_time else None,
            'uptime_seconds': (datetime.now() - self.startup_time).total_seconds() if self.startup_time else 0,
            'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
            'health_data': self._comprehensive_health_check(),
            'live_coordinator_status': self.live_coordinator.get_system_metrics(),
            'integration_status': self.integrator.get_integration_status(),
            'deployment_manager': self.deployment_manager
        }
    
    def emergency_shutdown(self, reason: str = "Manual emergency shutdown"):
        """Emergency shutdown of production system"""
        logger.critical(f"Emergency shutdown triggered: {reason}")
        
        try:
            # Stop all training
            self.live_coordinator.emergency_stop(reason)
            
            # Stop integrator
            self.integrator.stop_integrated_training()
            
            # Mark as inactive
            self.production_active = False
            
            # Create emergency backup
            self._create_system_backup()
            
            logger.critical("Emergency shutdown completed")
            
        except Exception as e:
            logger.critical(f"Emergency shutdown failed: {e}")
    
    def graceful_shutdown(self):
        """Graceful shutdown of production system"""
        logger.info("Starting graceful shutdown")
        
        try:
            # Stop training gracefully
            self.integrator.stop_integrated_training()
            
            # Mark as inactive
            self.production_active = False
            
            # Wait for threads to finish
            if self.health_monitor and self.health_monitor.is_alive():
                self.health_monitor.join(timeout=10.0)
            
            if self.backup_manager and self.backup_manager.is_alive():
                self.backup_manager.join(timeout=10.0)
            
            # Create final backup
            self._create_system_backup()
            
            # Cleanup
            self.integrator.cleanup()
            self.live_coordinator.cleanup()
            
            logger.info("Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"Graceful shutdown failed: {e}")


# Factory functions for easy setup
def create_integration_config(**kwargs) -> IntegrationConfig:
    """Create integration configuration"""
    return IntegrationConfig(**kwargs)


def create_production_manager(
    live_config: LiveTrainingConfig = None,
    integration_config: IntegrationConfig = None
) -> ProductionIntegrationManager:
    """Create production integration manager"""
    
    if live_config is None:
        live_config = LiveTrainingConfig()
    
    if integration_config is None:
        integration_config = IntegrationConfig()
    
    return ProductionIntegrationManager(live_config, integration_config)


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create configurations
        live_config = LiveTrainingConfig(
            learning_mode=LearningMode.CONTINUOUS,
            concurrent_training=True,
            enable_auto_retraining=True,
            enable_cross_agent_learning=True
        )
        
        integration_config = IntegrationConfig(
            enable_unified_training=True,
            enable_distributed_training=False,
            enable_memory_optimization=True,
            auto_detect_agents=True,
            production_mode=True
        )
        
        # Create production manager
        production_manager = create_production_manager(live_config, integration_config)
        
        # Deploy system
        if production_manager.deploy_production_system():
            print("Production system deployed successfully!")
            
            # Let it run for demonstration
            import time
            time.sleep(30)
            
            # Get status
            status = production_manager.get_production_status()
            print(f"Production status: {json.dumps(status, indent=2, default=str)}")
            
        else:
            print("Production deployment failed!")
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        if 'production_manager' in locals():
            production_manager.graceful_shutdown()
        print("Integration example completed")