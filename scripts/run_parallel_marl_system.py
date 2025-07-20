#!/usr/bin/env python3
"""
Launch Script for Parallel MARL System

Starts the complete integrated system:
- 30m Matrix Assembler with MMD integration
- 5m Matrix Assembler for tactical data
- 8 Parallel Agents (4 Strategic + 4 Tactical)
- Matrix delivery validation
- Real-time monitoring

Usage:
    python scripts/run_parallel_marl_system.py [--config CONFIG_PATH]
"""

import asyncio
import argparse
import yaml
import logging
import signal
import sys
from pathlib import Path
from datetime import datetime
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.event_bus import EventBus
from src.core.minimal_dependencies import Event, EventType
from src.matrix.assembler_30m import MatrixAssembler30m
from src.matrix.assembler_5m import MatrixAssembler5m
from src.agents.parallel_marl_system import ParallelMARLSystem
from src.utils.logger import get_logger
import numpy as np


class ParallelMARLLauncher:
    """Main launcher for the Parallel MARL System."""
    
    def __init__(self, config_path: str = None):
        self.logger = get_logger(self.__class__.__name__)
        self.config_path = config_path or "config/parallel_marl_config.yaml"
        self.config = self._load_config()
        
        # Core components
        self.event_bus = None
        self.assembler_30m = None
        self.assembler_5m = None
        self.marl_system = None
        
        # State
        self.is_running = False
        self.start_time = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info(f"Parallel MARL Launcher initialized with config: {self.config_path}")
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            self.logger.info(f"Configuration loaded successfully from {self.config_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        if self.is_running:
            asyncio.create_task(self.shutdown())
    
    async def initialize(self):
        """Initialize all system components."""
        try:
            self.logger.info("Initializing Parallel MARL System components...")
            
            # Create event bus
            self.event_bus = EventBus()
            self.logger.info("Event bus created")
            
            # Create mock kernel for assemblers
            mock_kernel = type('MockKernel', (), {
                'get_event_bus': lambda: self.event_bus
            })()
            
            # Initialize 30m Matrix Assembler with MMD
            assembler_30m_config = self.config['matrix_assemblers']['30m'].copy()
            assembler_30m_config['kernel'] = mock_kernel
            
            self.assembler_30m = MatrixAssembler30m(assembler_30m_config)
            self.logger.info("30m Matrix Assembler initialized with MMD integration")
            
            # Initialize 5m Matrix Assembler
            assembler_5m_config = self.config['matrix_assemblers']['5m'].copy()
            assembler_5m_config['kernel'] = mock_kernel
            
            self.assembler_5m = MatrixAssembler5m(assembler_5m_config)
            self.logger.info("5m Matrix Assembler initialized")
            
            # Initialize Parallel MARL System
            self.marl_system = ParallelMARLSystem(self.event_bus)
            self.logger.info("Parallel MARL System initialized with 8 agents")
            
            # Subscribe to system events for monitoring
            await self.event_bus.subscribe(EventType.STRATEGIC_DECISION, self._log_strategic_decision)
            await self.event_bus.subscribe(EventType.TACTICAL_DECISION, self._log_tactical_decision)
            await self.event_bus.subscribe(EventType.AGENT_ACKNOWLEDGMENT, self._log_acknowledgment)
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def start(self):
        """Start the complete system."""
        try:
            if self.is_running:
                self.logger.warning("System is already running")
                return
            
            self.logger.info("Starting Parallel MARL System...")
            self.start_time = datetime.now()
            self.is_running = True
            
            # Start MARL system (agents)
            await self.marl_system.start()
            self.logger.info("8 parallel agents started successfully")
            
            # Start data simulation
            asyncio.create_task(self._simulate_market_data())
            self.logger.info("Market data simulation started")
            
            # Start monitoring
            asyncio.create_task(self._monitor_system())
            self.logger.info("System monitoring started")
            
            self.logger.info(
                "🚀 Parallel MARL System is now RUNNING with maximum velocity!\n"
                "📊 4 Strategic Agents processing 30m matrices with MMD integration\n"
                "⚡ 4 Tactical Agents processing 5m matrices\n"
                "🔒 300% trustworthy matrix delivery validation active\n"
                "📈 Real-time performance monitoring enabled"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to start system: {e}")
            self.is_running = False
            raise
    
    async def _simulate_market_data(self):
        """Simulate market data feeds for testing."""
        self.logger.info("Starting market data simulation...")
        
        # Simulation parameters
        base_price = 1.0500
        price_30m = base_price
        price_5m = base_price
        update_count = 0
        
        try:
            while self.is_running:
                # Generate 30m data every 30 seconds (fast simulation)
                if update_count % 6 == 0:  # Every 30 seconds in simulation
                    await self._generate_30m_data(price_30m, update_count)
                    price_30m += np.random.normal(0, 0.0001)  # Random walk
                
                # Generate 5m data every 5 seconds (fast simulation)
                await self._generate_5m_data(price_5m, update_count)
                price_5m += np.random.normal(0, 0.00005)  # Smaller moves for 5m
                
                update_count += 1
                await asyncio.sleep(5)  # 5 second intervals
                
        except Exception as e:
            self.logger.error(f"Market data simulation error: {e}")
    
    async def _generate_30m_data(self, price: float, update_id: int):
        """Generate 30m market data and indicators."""
        try:
            # Generate realistic market data
            high = price + np.random.uniform(0, 0.0010)
            low = price - np.random.uniform(0, 0.0010)
            close = price + np.random.normal(0, 0.0003)
            volume = np.random.uniform(800, 1200)
            
            # Generate technical indicators
            indicators = {
                'timestamp': datetime.now(),
                'current_price': close,
                'open': price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'spread': 0.0001,
                'bid_volume': volume * 0.6,
                'ask_volume': volume * 0.4,
                
                # Technical indicators
                'mlmi_value': 50 + np.random.normal(0, 10),
                'mlmi_signal': np.random.choice([-1, 0, 1]),
                'nwrqk_value': close + np.random.normal(0, 0.0005),
                'nwrqk_slope': np.random.normal(0, 0.1),
                'lvn_distance_points': np.random.uniform(0, 10),
                'lvn_nearest_strength': np.random.uniform(60, 95),
                'time_hour_sin': np.sin(2 * np.pi * datetime.now().hour / 24),
                'time_hour_cos': np.cos(2 * np.pi * datetime.now().hour / 24),
                
                # Additional market microstructure data for MMD
                'rsi': np.random.uniform(30, 70),
                'bb_position': np.random.uniform(0.2, 0.8),
                'macd_signal': np.random.normal(0, 0.0001),
                'volume_sma_ratio': np.random.uniform(0.8, 1.5),
                'tick_direction': np.random.choice([-1, 1]),
                'trade_size_avg': np.random.uniform(50, 200),
                'trade_frequency': np.random.uniform(10, 50)
            }
            
            # Publish INDICATORS_READY event
            event = Event(
                type=EventType.INDICATORS_READY,
                payload=indicators,
                source='market_data_simulator_30m'
            )
            
            await self.event_bus.publish(event)
            self.logger.debug(f"Published 30m indicators (update #{update_id})")
            
        except Exception as e:
            self.logger.error(f"Error generating 30m data: {e}")
    
    async def _generate_5m_data(self, price: float, update_id: int):
        """Generate 5m market data and indicators."""
        try:
            # Generate 5m specific indicators
            indicators = {
                'timestamp': datetime.now(),
                'current_price': price,
                'current_volume': np.random.uniform(150, 250),
                
                # FVG indicators
                'fvg_bullish_active': np.random.choice([0, 1]),
                'fvg_bearish_active': np.random.choice([0, 1]),
                'fvg_nearest_level': price + np.random.normal(0, 0.0002),
                'fvg_age': np.random.randint(0, 20),
                'fvg_mitigation_signal': np.random.choice([0, 1]),
                
                # Short-term momentum
                'price_momentum_5': np.random.normal(0, 0.01),
                'volume_ratio': np.random.uniform(0.5, 2.0)
            }
            
            # Create separate event for 5m data
            # For simplicity, we'll simulate the 5m assembler processing directly
            # In a real system, there would be separate indicator engines
            
            # Simulate matrix publication
            if hasattr(self.assembler_5m, '_on_indicators_ready'):
                event = Event(
                    type=EventType.INDICATORS_READY,
                    payload=indicators,
                    source='market_data_simulator_5m'
                )
                self.assembler_5m._on_indicators_ready(event)
            
            self.logger.debug(f"Generated 5m indicators (update #{update_id})")
            
        except Exception as e:
            self.logger.error(f"Error generating 5m data: {e}")
    
    async def _monitor_system(self):
        """Monitor system performance and health."""
        self.logger.info("Starting system monitoring...")
        
        try:
            while self.is_running:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                # Get system metrics
                metrics = self.marl_system.get_system_metrics()
                trustworthiness = self.marl_system.get_trustworthiness_score()
                
                # Log key metrics
                uptime = metrics.get('system_uptime_seconds', 0)
                total_decisions = metrics.get('total_decisions', 0)
                error_rate = metrics.get('error_rate', 0)
                
                self.logger.info(
                    f"📊 System Status: {metrics.get('system_status', 'unknown')} | "
                    f"⏰ Uptime: {uptime:.0f}s | "
                    f"🎯 Decisions: {total_decisions} | "
                    f"🔒 Trustworthiness: {trustworthiness:.2f}/3.0 ({trustworthiness/3*100:.1f}%) | "
                    f"❌ Error Rate: {error_rate:.3f}"
                )
                
                # Check for alerts
                if trustworthiness < 2.5:
                    self.logger.warning(f"⚠️  Trustworthiness below threshold: {trustworthiness:.2f}/3.0")
                
                if error_rate > 0.05:
                    self.logger.warning(f"⚠️  High error rate detected: {error_rate:.3f}")
                
                # Get MMD statistics periodically
                if hasattr(self.assembler_30m, 'get_mmd_statistics'):
                    mmd_stats = self.assembler_30m.get_mmd_statistics()
                    if 'performance' in mmd_stats:
                        avg_time = mmd_stats['performance'].get('avg_processing_time_ms', 0)
                        self.logger.debug(f"🧠 MMD Processing Time: {avg_time:.2f}ms avg")
                
        except Exception as e:
            self.logger.error(f"System monitoring error: {e}")
    
    async def _log_strategic_decision(self, event: Event):
        """Log strategic decisions."""
        decision = event.payload.get('decision', {})
        agent_id = decision.get('agent_id', 'unknown')
        action = decision.get('action', 'unknown')
        confidence = decision.get('confidence', 0)
        
        self.logger.info(f"🎯 Strategic Decision: {agent_id} → {action} (confidence: {confidence:.3f})")
    
    async def _log_tactical_decision(self, event: Event):
        """Log tactical decisions."""
        decision = event.payload.get('decision', {})
        agent_id = decision.get('agent_id', 'unknown')
        action = decision.get('action', 'unknown')
        urgency = decision.get('urgency', 'unknown')
        
        self.logger.info(f"⚡ Tactical Decision: {agent_id} → {action} (urgency: {urgency})")
    
    async def _log_acknowledgment(self, event: Event):
        """Log agent acknowledgments."""
        payload = event.payload
        agent_id = payload.get('agent_id', 'unknown')
        matrix_type = payload.get('matrix_type', 'unknown')
        
        self.logger.debug(f"✅ Agent Acknowledgment: {agent_id} received {matrix_type} matrix")
    
    async def shutdown(self):
        """Shutdown the system gracefully."""
        try:
            self.logger.info("Initiating graceful shutdown...")
            self.is_running = False
            
            if self.marl_system:
                await self.marl_system.stop()
                self.logger.info("MARL system stopped")
            
            # Get final statistics
            if self.marl_system:
                final_metrics = self.marl_system.get_system_metrics()
                final_trustworthiness = self.marl_system.get_trustworthiness_score()
                
                uptime = final_metrics.get('system_uptime_seconds', 0)
                total_decisions = final_metrics.get('total_decisions', 0)
                
                self.logger.info(
                    f"📈 Final Statistics:\n"
                    f"   💾 Total Runtime: {uptime:.0f} seconds\n"
                    f"   🎯 Total Decisions: {total_decisions}\n"
                    f"   🔒 Final Trustworthiness: {final_trustworthiness:.2f}/3.0 ({final_trustworthiness/3*100:.1f}%)\n"
                    f"   ⚡ Average Decision Rate: {total_decisions/(uptime/60):.1f} decisions/minute"
                )
            
            self.logger.info("🛑 Parallel MARL System shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def run(self):
        """Main run loop."""
        try:
            await self.initialize()
            await self.start()
            
            # Keep running until shutdown
            while self.is_running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Runtime error: {e}")
        finally:
            await self.shutdown()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Launch Parallel MARL System")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/parallel_marl_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)8s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create and run launcher
    launcher = ParallelMARLLauncher(config_path=args.config)
    await launcher.run()


if __name__ == "__main__":
    asyncio.run(main())