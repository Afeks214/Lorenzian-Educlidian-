"""
Agent Coordination Integration Tests
Tests the coordination between MARL agents in the trading system
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Mock agent classes for testing
class MockStrategicAgent:
    """Mock strategic MARL agent."""
    
    def __init__(self, agent_id: str = "strategic_agent"):
        self.agent_id = agent_id
        self.state = "initialized"
        self.last_action = None
        self.last_observation = None
        self.performance_metrics = {
            'total_reward': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0
        }
        self.coordination_messages = []
    
    async def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategic action."""
        self.last_observation = observation
        
        # Simulate strategic decision making
        await asyncio.sleep(0.001)  # 1ms decision time
        
        action = {
            'position_direction': np.random.choice(['long', 'short', 'neutral']),
            'position_size': np.random.uniform(0.01, 0.1),
            'confidence': np.random.uniform(0.5, 1.0),
            'time_horizon': np.random.randint(30, 300),  # 30-300 minutes
            'timestamp': time.time()
        }
        
        self.last_action = action
        return action
    
    def learn(self, reward: float, next_observation: Dict[str, Any]):
        """Update agent based on reward."""
        self.performance_metrics['total_reward'] += reward
    
    def send_coordination_message(self, recipient: str, message: Dict[str, Any]):
        """Send coordination message to another agent."""
        self.coordination_messages.append({
            'recipient': recipient,
            'message': message,
            'timestamp': time.time()
        })
    
    def receive_coordination_message(self, sender: str, message: Dict[str, Any]):
        """Receive coordination message from another agent."""
        # Process coordination message
        if message.get('type') == 'position_update':
            # Adjust strategy based on other agent's position
            pass
        elif message.get('type') == 'risk_warning':
            # Adjust risk based on warning
            pass


class MockTacticalAgent:
    """Mock tactical MARL agent."""
    
    def __init__(self, agent_id: str = "tactical_agent"):
        self.agent_id = agent_id
        self.state = "initialized"
        self.last_action = None
        self.last_observation = None
        self.performance_metrics = {
            'total_reward': 0.0,
            'execution_quality': 0.0,
            'slippage': 0.0
        }
        self.coordination_messages = []
    
    async def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate tactical action."""
        self.last_observation = observation
        
        # Simulate tactical decision making
        await asyncio.sleep(0.0005)  # 0.5ms decision time
        
        action = {
            'entry_signal': np.random.choice([0, 1, -1]),  # Buy, Hold, Sell
            'exit_signal': np.random.choice([0, 1]),  # Hold, Exit
            'order_type': np.random.choice(['market', 'limit', 'stop']),
            'urgency': np.random.uniform(0.1, 1.0),
            'timestamp': time.time()
        }
        
        self.last_action = action
        return action
    
    def learn(self, reward: float, next_observation: Dict[str, Any]):
        """Update agent based on reward."""
        self.performance_metrics['total_reward'] += reward
    
    def send_coordination_message(self, recipient: str, message: Dict[str, Any]):
        """Send coordination message to another agent."""
        self.coordination_messages.append({
            'recipient': recipient,
            'message': message,
            'timestamp': time.time()
        })
    
    def receive_coordination_message(self, sender: str, message: Dict[str, Any]):
        """Receive coordination message from another agent."""
        # Process coordination message
        if message.get('type') == 'strategic_direction':
            # Align tactical actions with strategic direction
            pass


class MockRiskAgent:
    """Mock risk management agent."""
    
    def __init__(self, agent_id: str = "risk_agent"):
        self.agent_id = agent_id
        self.state = "initialized"
        self.last_action = None
        self.last_observation = None
        self.performance_metrics = {
            'total_reward': 0.0,
            'risk_adjusted_return': 0.0,
            'max_drawdown': 0.0
        }
        self.coordination_messages = []
        self.risk_limits = {
            'max_position_size': 0.1,
            'max_portfolio_var': 0.05,
            'max_correlation_exposure': 0.3
        }
    
    async def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk management action."""
        self.last_observation = observation
        
        # Simulate risk assessment
        await asyncio.sleep(0.002)  # 2ms decision time
        
        action = {
            'position_size_limit': np.random.uniform(0.01, 0.1),
            'stop_loss_level': np.random.uniform(0.01, 0.05),
            'take_profit_level': np.random.uniform(0.02, 0.1),
            'risk_score': np.random.uniform(0.0, 1.0),
            'timestamp': time.time()
        }
        
        self.last_action = action
        return action
    
    def learn(self, reward: float, next_observation: Dict[str, Any]):
        """Update agent based on reward."""
        self.performance_metrics['total_reward'] += reward
    
    def check_risk_limits(self, proposed_action: Dict[str, Any]) -> bool:
        """Check if proposed action violates risk limits."""
        if proposed_action.get('position_size', 0) > self.risk_limits['max_position_size']:
            return False
        return True
    
    def send_coordination_message(self, recipient: str, message: Dict[str, Any]):
        """Send coordination message to another agent."""
        self.coordination_messages.append({
            'recipient': recipient,
            'message': message,
            'timestamp': time.time()
        })
    
    def receive_coordination_message(self, sender: str, message: Dict[str, Any]):
        """Receive coordination message from another agent."""
        # Process coordination message
        if message.get('type') == 'position_request':
            # Validate position request against risk limits
            pass


class AgentCoordinator:
    """Coordinates communication between MARL agents."""
    
    def __init__(self):
        self.agents = {}
        self.message_history = []
        self.coordination_rules = {
            'strategic_to_tactical': True,
            'tactical_to_risk': True,
            'risk_to_strategic': True,
            'risk_to_tactical': True
        }
    
    def register_agent(self, agent):
        """Register an agent with the coordinator."""
        self.agents[agent.agent_id] = agent
    
    def send_message(self, sender_id: str, recipient_id: str, message: Dict[str, Any]):
        """Send message between agents."""
        if sender_id not in self.agents or recipient_id not in self.agents:
            raise ValueError(f"Agent not found: {sender_id} or {recipient_id}")
        
        # Check coordination rules
        rule_key = f"{sender_id}_to_{recipient_id}"
        if rule_key in self.coordination_rules and not self.coordination_rules[rule_key]:
            raise ValueError(f"Communication not allowed: {rule_key}")
        
        # Record message
        message_record = {
            'sender': sender_id,
            'recipient': recipient_id,
            'message': message,
            'timestamp': time.time()
        }
        self.message_history.append(message_record)
        
        # Deliver message
        sender = self.agents[sender_id]
        recipient = self.agents[recipient_id]
        
        sender.send_coordination_message(recipient_id, message)
        recipient.receive_coordination_message(sender_id, message)
    
    def broadcast_message(self, sender_id: str, message: Dict[str, Any]):
        """Broadcast message to all other agents."""
        sender = self.agents[sender_id]
        for agent_id, agent in self.agents.items():
            if agent_id != sender_id:
                self.send_message(sender_id, agent_id, message)
    
    def get_message_history(self) -> List[Dict[str, Any]]:
        """Get message history."""
        return self.message_history.copy()


@pytest.mark.integration
class TestAgentCoordination:
    """Test agent coordination functionality."""
    
    def test_agent_registration(self):
        """Test agent registration with coordinator."""
        coordinator = AgentCoordinator()
        
        # Create agents
        strategic_agent = MockStrategicAgent()
        tactical_agent = MockTacticalAgent()
        risk_agent = MockRiskAgent()
        
        # Register agents
        coordinator.register_agent(strategic_agent)
        coordinator.register_agent(tactical_agent)
        coordinator.register_agent(risk_agent)
        
        # Verify registration
        assert len(coordinator.agents) == 3
        assert 'strategic_agent' in coordinator.agents
        assert 'tactical_agent' in coordinator.agents
        assert 'risk_agent' in coordinator.agents
    
    def test_message_passing(self):
        """Test message passing between agents."""
        coordinator = AgentCoordinator()
        
        # Create and register agents
        strategic_agent = MockStrategicAgent()
        tactical_agent = MockTacticalAgent()
        coordinator.register_agent(strategic_agent)
        coordinator.register_agent(tactical_agent)
        
        # Send message
        message = {
            'type': 'strategic_direction',
            'direction': 'long',
            'confidence': 0.8,
            'time_horizon': 60
        }
        
        coordinator.send_message('strategic_agent', 'tactical_agent', message)
        
        # Verify message delivery
        assert len(coordinator.message_history) == 1
        assert len(strategic_agent.coordination_messages) == 1
        assert strategic_agent.coordination_messages[0]['recipient'] == 'tactical_agent'
        assert strategic_agent.coordination_messages[0]['message'] == message
    
    def test_broadcast_messaging(self):
        """Test broadcast messaging to all agents."""
        coordinator = AgentCoordinator()
        
        # Create and register agents
        strategic_agent = MockStrategicAgent()
        tactical_agent = MockTacticalAgent()
        risk_agent = MockRiskAgent()
        coordinator.register_agent(strategic_agent)
        coordinator.register_agent(tactical_agent)
        coordinator.register_agent(risk_agent)
        
        # Broadcast message
        message = {
            'type': 'market_update',
            'price': 100.5,
            'volume': 1000,
            'volatility': 0.2
        }
        
        coordinator.broadcast_message('strategic_agent', message)
        
        # Verify broadcast
        assert len(coordinator.message_history) == 2  # To tactical and risk agents
        assert len(strategic_agent.coordination_messages) == 2
    
    def test_coordination_rules(self):
        """Test coordination rules enforcement."""
        coordinator = AgentCoordinator()
        
        # Create and register agents
        strategic_agent = MockStrategicAgent()
        tactical_agent = MockTacticalAgent()
        coordinator.register_agent(strategic_agent)
        coordinator.register_agent(tactical_agent)
        
        # Disable communication
        coordinator.coordination_rules['strategic_to_tactical'] = False
        
        # Try to send message
        message = {'type': 'test'}
        
        with pytest.raises(ValueError):
            coordinator.send_message('strategic_agent', 'tactical_agent', message)
    
    @pytest.mark.asyncio
    async def test_async_coordination(self):
        """Test asynchronous coordination between agents."""
        coordinator = AgentCoordinator()
        
        # Create and register agents
        strategic_agent = MockStrategicAgent()
        tactical_agent = MockTacticalAgent()
        risk_agent = MockRiskAgent()
        coordinator.register_agent(strategic_agent)
        coordinator.register_agent(tactical_agent)
        coordinator.register_agent(risk_agent)
        
        # Simulate market observation
        observation = {
            'price': 100.0,
            'volume': 1000,
            'volatility': 0.2,
            'timestamp': time.time()
        }
        
        # Get actions from all agents concurrently
        strategic_task = asyncio.create_task(strategic_agent.act(observation))
        tactical_task = asyncio.create_task(tactical_agent.act(observation))
        risk_task = asyncio.create_task(risk_agent.act(observation))
        
        # Wait for all agents to act
        strategic_action, tactical_action, risk_action = await asyncio.gather(
            strategic_task, tactical_task, risk_task
        )
        
        # Verify actions
        assert strategic_action is not None
        assert tactical_action is not None
        assert risk_action is not None
        assert 'timestamp' in strategic_action
        assert 'timestamp' in tactical_action
        assert 'timestamp' in risk_action


@pytest.mark.integration
class TestCoordinationScenarios:
    """Test specific coordination scenarios."""
    
    def test_strategic_tactical_coordination(self):
        """Test coordination between strategic and tactical agents."""
        coordinator = AgentCoordinator()
        
        # Create agents
        strategic_agent = MockStrategicAgent()
        tactical_agent = MockTacticalAgent()
        coordinator.register_agent(strategic_agent)
        coordinator.register_agent(tactical_agent)
        
        # Strategic agent sends direction to tactical agent
        strategic_message = {
            'type': 'strategic_direction',
            'direction': 'long',
            'target_position': 0.05,
            'confidence': 0.8,
            'time_horizon': 120
        }
        
        coordinator.send_message('strategic_agent', 'tactical_agent', strategic_message)
        
        # Tactical agent acknowledges and provides execution feedback
        tactical_response = {
            'type': 'execution_feedback',
            'status': 'acknowledged',
            'estimated_slippage': 0.001,
            'execution_time': 30
        }
        
        coordinator.send_message('tactical_agent', 'strategic_agent', tactical_response)
        
        # Verify coordination
        assert len(coordinator.message_history) == 2
        assert len(strategic_agent.coordination_messages) == 1
        assert len(tactical_agent.coordination_messages) == 1
    
    def test_risk_management_coordination(self):
        """Test risk management coordination."""
        coordinator = AgentCoordinator()
        
        # Create agents
        strategic_agent = MockStrategicAgent()
        risk_agent = MockRiskAgent()
        coordinator.register_agent(strategic_agent)
        coordinator.register_agent(risk_agent)
        
        # Strategic agent requests position
        position_request = {
            'type': 'position_request',
            'symbol': 'AAPL',
            'direction': 'long',
            'size': 0.08,
            'confidence': 0.9
        }
        
        coordinator.send_message('strategic_agent', 'risk_agent', position_request)
        
        # Risk agent evaluates and responds
        risk_response = {
            'type': 'risk_assessment',
            'approved': True,
            'max_position_size': 0.06,  # Reduced from requested 0.08
            'stop_loss': 0.02,
            'risk_score': 0.3
        }
        
        coordinator.send_message('risk_agent', 'strategic_agent', risk_response)
        
        # Verify risk coordination
        assert len(coordinator.message_history) == 2
        assert risk_response['max_position_size'] < position_request['size']
    
    def test_emergency_coordination(self):
        """Test emergency coordination scenarios."""
        coordinator = AgentCoordinator()
        
        # Create agents
        strategic_agent = MockStrategicAgent()
        tactical_agent = MockTacticalAgent()
        risk_agent = MockRiskAgent()
        coordinator.register_agent(strategic_agent)
        coordinator.register_agent(tactical_agent)
        coordinator.register_agent(risk_agent)
        
        # Risk agent detects emergency
        emergency_message = {
            'type': 'emergency_alert',
            'severity': 'high',
            'reason': 'portfolio_var_exceeded',
            'action': 'reduce_positions',
            'urgency': 1.0
        }
        
        # Broadcast emergency to all agents
        coordinator.broadcast_message('risk_agent', emergency_message)
        
        # Verify emergency coordination
        assert len(coordinator.message_history) == 2  # To strategic and tactical
        assert len(risk_agent.coordination_messages) == 2
        
        # All agents should have received the emergency message
        strategic_received = any(
            msg['message']['type'] == 'emergency_alert' 
            for msg in coordinator.message_history 
            if msg['recipient'] == 'strategic_agent'
        )
        tactical_received = any(
            msg['message']['type'] == 'emergency_alert' 
            for msg in coordinator.message_history 
            if msg['recipient'] == 'tactical_agent'
        )
        
        assert strategic_received
        assert tactical_received
    
    @pytest.mark.asyncio
    async def test_coordination_under_load(self):
        """Test coordination under high message load."""
        coordinator = AgentCoordinator()
        
        # Create agents
        strategic_agent = MockStrategicAgent()
        tactical_agent = MockTacticalAgent()
        risk_agent = MockRiskAgent()
        coordinator.register_agent(strategic_agent)
        coordinator.register_agent(tactical_agent)
        coordinator.register_agent(risk_agent)
        
        # Generate high message load
        async def send_messages():
            for i in range(100):
                message = {
                    'type': 'market_update',
                    'sequence': i,
                    'price': 100 + i * 0.1,
                    'timestamp': time.time()
                }
                
                # Send messages between all agents
                coordinator.send_message('strategic_agent', 'tactical_agent', message)
                coordinator.send_message('tactical_agent', 'risk_agent', message)
                coordinator.send_message('risk_agent', 'strategic_agent', message)
                
                await asyncio.sleep(0.001)  # 1ms between messages
        
        # Measure coordination performance
        start_time = time.time()
        await send_messages()
        end_time = time.time()
        
        # Verify performance
        total_time = end_time - start_time
        messages_per_second = 300 / total_time  # 100 messages * 3 agent pairs
        
        assert len(coordinator.message_history) == 300
        assert messages_per_second > 1000, f"Message rate {messages_per_second:.0f}/s too low"
    
    def test_coordination_failure_handling(self):
        """Test handling of coordination failures."""
        coordinator = AgentCoordinator()
        
        # Create agents
        strategic_agent = MockStrategicAgent()
        tactical_agent = MockTacticalAgent()
        coordinator.register_agent(strategic_agent)
        coordinator.register_agent(tactical_agent)
        
        # Test message to non-existent agent
        message = {'type': 'test'}
        
        with pytest.raises(ValueError):
            coordinator.send_message('strategic_agent', 'non_existent_agent', message)
        
        # Test message from non-existent agent
        with pytest.raises(ValueError):
            coordinator.send_message('non_existent_agent', 'tactical_agent', message)


@pytest.mark.integration
class TestCoordinationPerformance:
    """Test coordination performance requirements."""
    
    def test_message_latency(self):
        """Test message delivery latency."""
        coordinator = AgentCoordinator()
        
        # Create agents
        strategic_agent = MockStrategicAgent()
        tactical_agent = MockTacticalAgent()
        coordinator.register_agent(strategic_agent)
        coordinator.register_agent(tactical_agent)
        
        # Measure message latency
        latencies = []
        for i in range(1000):
            message = {
                'type': 'latency_test',
                'sequence': i,
                'timestamp': time.time()
            }
            
            start_time = time.time()
            coordinator.send_message('strategic_agent', 'tactical_agent', message)
            end_time = time.time()
            
            latencies.append(end_time - start_time)
        
        # Verify latency requirements
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        assert avg_latency < 0.001, f"Average latency {avg_latency:.6f}s too high"
        assert p95_latency < 0.002, f"95th percentile latency {p95_latency:.6f}s too high"
        assert max_latency < 0.01, f"Maximum latency {max_latency:.6f}s too high"
    
    def test_coordination_throughput(self):
        """Test coordination throughput under load."""
        coordinator = AgentCoordinator()
        
        # Create multiple agents
        agents = []
        for i in range(10):
            agent = MockStrategicAgent(f"agent_{i}")
            coordinator.register_agent(agent)
            agents.append(agent)
        
        # Generate high throughput messages
        start_time = time.time()
        
        for i in range(1000):
            sender_idx = i % len(agents)
            recipient_idx = (i + 1) % len(agents)
            
            message = {
                'type': 'throughput_test',
                'sequence': i,
                'timestamp': time.time()
            }
            
            coordinator.send_message(
                agents[sender_idx].agent_id,
                agents[recipient_idx].agent_id,
                message
            )
        
        end_time = time.time()
        
        # Verify throughput
        total_time = end_time - start_time
        messages_per_second = 1000 / total_time
        
        assert messages_per_second > 5000, f"Throughput {messages_per_second:.0f}/s too low"
        assert len(coordinator.message_history) == 1000
    
    def test_concurrent_coordination(self):
        """Test concurrent coordination scenarios."""
        coordinator = AgentCoordinator()
        
        # Create agents
        strategic_agent = MockStrategicAgent()
        tactical_agent = MockTacticalAgent()
        risk_agent = MockRiskAgent()
        coordinator.register_agent(strategic_agent)
        coordinator.register_agent(tactical_agent)
        coordinator.register_agent(risk_agent)
        
        # Function to send messages concurrently
        def send_concurrent_messages():
            for i in range(100):
                message = {
                    'type': 'concurrent_test',
                    'sequence': i,
                    'thread_id': threading.current_thread().ident,
                    'timestamp': time.time()
                }
                coordinator.send_message('strategic_agent', 'tactical_agent', message)
        
        # Start multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=send_concurrent_messages)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify concurrent coordination
        assert len(coordinator.message_history) == 1000  # 10 threads * 100 messages
        assert len(strategic_agent.coordination_messages) == 1000
        
        # Verify message ordering and integrity
        sequences = [msg['message']['sequence'] for msg in coordinator.message_history]
        assert len(set(sequences)) <= 100  # Should have at most 100 unique sequences
        
        # Check for thread safety
        thread_ids = {msg['message']['thread_id'] for msg in coordinator.message_history}
        assert len(thread_ids) == 10  # Should have messages from all 10 threads


def generate_coordination_test_report():
    """Generate agent coordination test report."""
    
    report = {
        'timestamp': time.time(),
        'coordination_tests': {
            'agent_registration': 'Tests agent registration and discovery',
            'message_passing': 'Tests message delivery between agents',
            'coordination_rules': 'Tests coordination rule enforcement',
            'emergency_coordination': 'Tests emergency coordination scenarios',
            'performance_requirements': 'Tests latency and throughput requirements'
        },
        'coordination_requirements': [
            'Sub-millisecond message latency',
            'Support for 10+ concurrent agents',
            'Message throughput >5000 messages/second',
            'Emergency coordination within 1ms',
            'Thread-safe concurrent coordination',
            'Configurable coordination rules'
        ],
        'recommendations': [
            'Implement message priority queues',
            'Add message acknowledgment system',
            'Implement heartbeat monitoring',
            'Add coordination performance metrics',
            'Implement message replay for recovery',
            'Add coordination audit logging'
        ]
    }
    
    return report


if __name__ == "__main__":
    # Generate coordination test report
    report = generate_coordination_test_report()
    print("Agent Coordination Test Report Generated")
    print(f"Coordination Test Categories: {len(report['coordination_tests'])}")
    print("Coordination Requirements:")
    for req in report['coordination_requirements']:
        print(f"  - {req}")
    print("Recommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")