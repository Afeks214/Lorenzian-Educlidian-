"""
Real-time XAI Pipeline Demonstration

Agent Beta: Real-time streaming specialist
Mission: Demonstrate the complete real-time explanation pipeline

This demonstration script showcases the complete Agent Beta real-time explanation
pipeline, from Strategic MARL decision capture to WebSocket streaming delivery.

Features Demonstrated:
- Zero-latency decision capture from Strategic MARL
- Real-time context processing and feature extraction
- LLM-powered explanation generation
- WebSocket streaming to multiple clients
- Performance monitoring and health checks
- Graceful degradation and error handling

Author: Agent Beta - Real-time Streaming Specialist
Version: 1.0 - Real-time Pipeline Demonstration
"""

import asyncio
import logging
import time
import json
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from core.events import EventType, Event, EventBus
from xai.pipeline.marl_integration import XAIPipelineIntegration


# Demo configuration
DEMO_CONFIG = {
    'health_check_interval_seconds': 5,
    'enable_graceful_degradation': True,
    
    'websocket_manager': {
        'host': 'localhost',
        'port': 8765,
        'max_connections': 100,
        'authentication': {'enabled': False}  # Simplified for demo
    },
    
    'decision_capture': {
        'max_capture_latency_ns': 100_000,  # 100 microseconds
        'queue_size': 10000,
        'redis': {'enabled': False}  # Use in-memory for demo
    },
    
    'context_processor': {
        'queue_size': 5000,
        'embedding_dim': 384,
        'cache_size': 2000,
        'feature_vector_dim': 128
    },
    
    'streaming_engine': {
        'target_explanation_latency_ms': 200,
        'explanation_queue_size': 5000,
        'llm': {
            'model': 'llama3.2:3b',
            'timeout_seconds': 10,
            'max_tokens': 500,
            'temperature': 0.3
        }
    }
}


class MockKernel:
    """Mock kernel for demonstration"""
    def __init__(self):
        self.event_bus = EventBus()
        self.config = {}


class MarketDataSimulator:
    """Simulates realistic market data and Strategic MARL decisions"""
    
    def __init__(self):
        self.symbol = "NQ"
        self.current_price = 15000.0
        self.volatility = 0.02
        self.trend = 0.0
        self.volume_base = 1000000
        
        # Market regime states
        self.regimes = ['trending', 'ranging', 'volatile', 'transitional']
        self.current_regime = 'trending'
        
        # Agent performance tracking
        self.agent_accuracy = {
            'MLMI': 0.75,
            'NWRQK': 0.72,
            'Regime': 0.68
        }
    
    def generate_market_decision(self) -> Dict[str, Any]:
        """Generate realistic Strategic MARL decision"""
        
        # Simulate market movement
        price_change = np.random.normal(self.trend, self.volatility)
        self.current_price *= (1 + price_change)
        
        # Determine action based on market conditions
        momentum_short = np.random.normal(0, 0.02)
        momentum_long = np.random.normal(0, 0.015)
        
        # Generate action probabilities
        if momentum_short > 0.01 and momentum_long > 0.01:
            action = 'buy'
            probabilities = [0.1, 0.15, 0.75]
        elif momentum_short < -0.01 and momentum_long < -0.01:
            action = 'sell'
            probabilities = [0.75, 0.15, 0.1]
        else:
            action = 'hold'
            probabilities = [0.2, 0.6, 0.2]
        
        # Agent contributions with some randomness
        base_contributions = {
            'MLMI': 0.4 + np.random.normal(0, 0.05),
            'NWRQK': 0.35 + np.random.normal(0, 0.05),
            'Regime': 0.25 + np.random.normal(0, 0.05)
        }
        
        # Normalize contributions
        total = sum(base_contributions.values())
        agent_contributions = {k: v/total for k, v in base_contributions.items()}
        
        # Calculate confidence based on consensus
        consensus_strength = 1.0 - np.std(list(agent_contributions.values()))
        confidence = 0.5 + consensus_strength * 0.4 + np.random.normal(0, 0.05)
        confidence = np.clip(confidence, 0.1, 0.95)
        
        # Generate gating weights (similar to contributions but with slight variation)
        gating_weights = [
            agent_contributions['MLMI'] * (1 + np.random.normal(0, 0.1)),
            agent_contributions['NWRQK'] * (1 + np.random.normal(0, 0.1)),
            agent_contributions['Regime'] * (1 + np.random.normal(0, 0.1))
        ]
        
        # Normalize gating weights
        total_weight = sum(gating_weights)
        gating_weights = [w/total_weight for w in gating_weights]
        
        # Update regime occasionally
        if np.random.random() < 0.1:
            self.current_regime = np.random.choice(self.regimes)
        
        # Generate decision
        decision = {
            'action': action,
            'confidence': float(confidence),
            'uncertainty': float(1.0 - confidence),
            'should_proceed': confidence > 0.6,
            'reasoning': self._generate_reasoning(action, confidence, momentum_short, momentum_long),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            
            'agent_contributions': agent_contributions,
            
            'performance_metrics': {
                'ensemble_probabilities': probabilities,
                'dynamic_weights': gating_weights,
                'gating_confidence': float(0.7 + np.random.normal(0, 0.1)),
                'inference_time_ms': float(2.0 + np.random.exponential(1.0)),
                'max_confidence': float(max(agent_contributions.values())),
                'min_confidence': float(min(agent_contributions.values())),
                'total_weight': 1.0
            },
            
            # Market context
            'market_context': {
                'symbol': self.symbol,
                'current_price': self.current_price,
                'volatility': self.volatility,
                'volume_ratio': 0.8 + np.random.exponential(0.5),
                'momentum_short': momentum_short,
                'momentum_long': momentum_long,
                'regime': self.current_regime,
                'session': self._get_market_session()
            }
        }
        
        return decision
    
    def _generate_reasoning(self, action: str, confidence: float, momentum_short: float, momentum_long: float) -> str:
        """Generate realistic reasoning text"""
        reasons = []
        
        if confidence > 0.8:
            reasons.append("Strong consensus across all agents")
        elif confidence > 0.6:
            reasons.append("Moderate agreement among agents")
        else:
            reasons.append("Mixed signals with low consensus")
        
        if abs(momentum_short) > 0.015:
            direction = "bullish" if momentum_short > 0 else "bearish"
            reasons.append(f"Strong {direction} short-term momentum")
        
        if abs(momentum_long) > 0.01:
            direction = "bullish" if momentum_long > 0 else "bearish"
            reasons.append(f"{direction.title()} long-term trend")
        
        reasons.append(f"Market regime: {self.current_regime}")
        
        if self.volatility > 0.03:
            reasons.append("Elevated volatility requiring careful position sizing")
        
        return ". ".join(reasons) + "."
    
    def _get_market_session(self) -> str:
        """Determine current market session"""
        hour = datetime.now().hour
        if 9 <= hour < 16:
            return 'regular'
        elif 4 <= hour < 9 or 16 <= hour < 20:
            return 'extended'
        else:
            return 'closed'


class DemoController:
    """Controls the demonstration flow"""
    
    def __init__(self):
        self.kernel = MockKernel()
        self.integration: XAIPipelineIntegration = None
        self.market_simulator = MarketDataSimulator()
        self.demo_active = False
        self.stats = {
            'decisions_generated': 0,
            'explanations_created': 0,
            'start_time': None,
            'errors': 0
        }
    
    async def start_demo(self):
        """Start the demonstration"""
        print("🚀 Starting Agent Beta Real-time XAI Pipeline Demonstration")
        print("=" * 70)
        
        try:
            # Initialize pipeline
            print("\n🔧 Initializing XAI Pipeline...")
            self.integration = XAIPipelineIntegration(self.kernel, DEMO_CONFIG)
            await self.integration.initialize()
            
            # Check initialization status
            status = self.integration.get_integration_status()
            print(f"✅ Pipeline initialized successfully!")
            print(f"   Components: {status['components_initialized']}/{status['total_components']}")
            print(f"   Health: {status['health']['status']}")
            
            # Display component status
            print(f"\n📊 Component Status:")
            for component, ready in status['component_status'].items():
                status_icon = "✅" if ready else "❌"
                print(f"   {status_icon} {component.replace('_', ' ').title()}")
            
            # Start demonstration
            self.demo_active = True
            self.stats['start_time'] = time.time()
            
            print(f"\n🎯 Starting real-time decision simulation...")
            print(f"   WebSocket server: ws://localhost:{DEMO_CONFIG['websocket_manager']['port']}")
            print(f"   Target latency: {DEMO_CONFIG['streaming_engine']['target_explanation_latency_ms']}ms")
            print(f"   Press Ctrl+C to stop\n")
            
            # Run demonstration
            await self._run_decision_simulation()
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Demo stopped by user")
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            self.stats['errors'] += 1
        finally:
            await self._cleanup()
    
    async def _run_decision_simulation(self):
        """Run the main decision simulation loop"""
        
        while self.demo_active:
            try:
                # Generate market decision
                decision = self.market_simulator.generate_market_decision()
                self.stats['decisions_generated'] += 1
                
                # Create and publish event
                event = self.kernel.event_bus.create_event(
                    EventType.STRATEGIC_DECISION,
                    decision,
                    source="demo_strategic_marl"
                )
                
                # Measure decision processing time
                start_time = time.perf_counter_ns()
                self.kernel.event_bus.publish(event)
                processing_time_ns = time.perf_counter_ns() - start_time
                
                # Display decision info
                confidence_icon = "🟢" if decision['confidence'] > 0.8 else "🟡" if decision['confidence'] > 0.6 else "🔴"
                print(f"{confidence_icon} Decision #{self.stats['decisions_generated']}: "
                      f"{decision['action'].upper()} "
                      f"(confidence: {decision['confidence']:.1%}, "
                      f"latency: {processing_time_ns/1000:.1f}μs)")
                
                # Show performance stats every 10 decisions
                if self.stats['decisions_generated'] % 10 == 0:
                    await self._display_performance_stats()
                
                # Variable delay to simulate realistic decision frequency
                delay = np.random.exponential(2.0) + 0.5  # 0.5-10 seconds typically
                await asyncio.sleep(min(delay, 10.0))
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error in decision simulation: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(1.0)
    
    async def _display_performance_stats(self):
        """Display current performance statistics"""
        try:
            status = self.integration.get_integration_status()
            runtime = time.time() - self.stats['start_time']
            
            print(f"\n📈 Performance Stats (Runtime: {runtime:.1f}s)")
            print(f"   Decisions: {self.stats['decisions_generated']}")
            print(f"   Rate: {self.stats['decisions_generated']/runtime:.1f} decisions/sec")
            print(f"   Errors: {self.stats['errors']}")
            print(f"   Health: {status['health']['status']}")
            
            # Component metrics if available
            if 'component_metrics' in status:
                metrics = status['component_metrics']
                
                # Decision capture metrics
                if 'decision_capture' in metrics:
                    dc = metrics['decision_capture']
                    print(f"   📡 Capture: {dc.get('total_decisions_captured', 0)} decisions, "
                          f"{dc.get('avg_capture_latency_ns', 0)/1000:.1f}μs avg")
                
                # Context processor metrics
                if 'context_processor' in metrics:
                    cp = metrics['context_processor']
                    print(f"   🔄 Processing: {cp.get('total_contexts_processed', 0)} contexts, "
                          f"{cp.get('avg_processing_time_ms', 0):.1f}ms avg")
                
                # Streaming engine metrics
                if 'streaming_engine' in metrics:
                    se = metrics['streaming_engine']
                    print(f"   📤 Streaming: {se.get('total_explanations_generated', 0)} explanations, "
                          f"{se.get('avg_generation_time_ms', 0):.1f}ms avg")
                
                # WebSocket metrics
                if 'websocket_manager' in metrics:
                    ws = metrics['websocket_manager']
                    print(f"   🌐 WebSocket: {ws.get('active_connections', 0)} connections, "
                          f"{ws.get('total_messages_sent', 0)} messages sent")
            
            print()
            
        except Exception as e:
            print(f"❌ Error displaying stats: {e}")
    
    async def _cleanup(self):
        """Clean up resources"""
        self.demo_active = False
        
        print("\n🧹 Cleaning up...")
        
        if self.integration:
            try:
                await self.integration.shutdown()
                print("✅ Pipeline shutdown complete")
            except Exception as e:
                print(f"⚠️  Shutdown error: {e}")
        
        # Final stats
        if self.stats['start_time']:
            runtime = time.time() - self.stats['start_time']
            print(f"\n📊 Final Statistics:")
            print(f"   Runtime: {runtime:.1f} seconds")
            print(f"   Total decisions: {self.stats['decisions_generated']}")
            print(f"   Average rate: {self.stats['decisions_generated']/runtime:.1f} decisions/sec")
            print(f"   Errors: {self.stats['errors']}")
            
            if self.stats['errors'] == 0:
                print("🎉 Demo completed successfully!")
            else:
                print(f"⚠️  Demo completed with {self.stats['errors']} errors")


# WebSocket client example for testing
class WebSocketTestClient:
    """Simple WebSocket client for testing the pipeline"""
    
    def __init__(self, uri: str = "ws://localhost:8765"):
        self.uri = uri
        self.connected = False
    
    async def connect_and_listen(self):
        """Connect to WebSocket and listen for explanations"""
        try:
            import websockets
            
            print(f"\n🔌 Connecting to WebSocket: {self.uri}")
            
            async with websockets.connect(self.uri) as websocket:
                self.connected = True
                print("✅ WebSocket connected, listening for explanations...")
                
                # Send subscription message
                subscription = {
                    'type': 'subscription',
                    'action': 'subscribe',
                    'topics': ['explanations_trader', 'explanations_risk_manager']
                }
                await websocket.send(json.dumps(subscription))
                
                # Listen for messages
                message_count = 0
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        message_count += 1
                        
                        if data.get('type') == 'explanation':
                            print(f"\n💬 Explanation #{message_count}:")
                            payload = data.get('payload', {})
                            print(f"   Decision: {payload.get('decision_id', 'unknown')}")
                            print(f"   Summary: {payload.get('summary', 'N/A')}")
                            print(f"   Quality: {payload.get('quality_score', 0):.2f}")
                        
                    except json.JSONDecodeError:
                        print(f"⚠️  Invalid JSON received: {message[:100]}...")
                    except KeyboardInterrupt:
                        break
                        
        except ImportError:
            print("⚠️  websockets library not available for client testing")
        except Exception as e:
            print(f"❌ WebSocket client error: {e}")
        finally:
            self.connected = False


async def main():
    """Main demonstration function"""
    print("🔥 Agent Beta - Real-time XAI Pipeline Demonstration")
    print("📡 Building the nervous system for trading explanations")
    print()
    
    # Create demo controller
    demo = DemoController()
    
    # Start demo
    await demo.start_demo()


def run_websocket_client():
    """Run WebSocket client for testing"""
    client = WebSocketTestClient()
    asyncio.run(client.connect_and_listen())


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--client":
        # Run WebSocket test client
        run_websocket_client()
    else:
        # Run main demo
        asyncio.run(main())