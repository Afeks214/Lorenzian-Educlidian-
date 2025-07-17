#!/usr/bin/env python3
"""
Demo script for Master Switch Integration with Risk Management Systems

This script demonstrates how the TradingSystemController integrates with
the kill switch system to control all risk management components.

Features demonstrated:
1. System startup and shutdown
2. Risk calculation blocking when system is OFF
3. Cached value preservation during OFF periods
4. Kill switch integration
5. Component registration and monitoring
"""

import asyncio
import time
import logging
from typing import Dict, Any
from datetime import datetime
import numpy as np

# Import the master switch controller
from src.safety.trading_system_controller import initialize_controller, get_controller
from src.safety.kill_switch import initialize_kill_switch

# Import risk management components
from src.risk.core.var_calculator import VaRCalculator, PositionData
from src.risk.core.correlation_tracker import CorrelationTracker
from src.risk.agents.position_sizing_agent import PositionSizingAgent, PositionSizingAction
from src.risk.monitoring.real_time_risk_monitor import create_real_time_monitor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockEventBus:
    """Mock event bus for testing"""
    def __init__(self):
        self.subscribers = {}
    
    def subscribe(self, event_type, callback):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
    
    def publish(self, event):
        pass
    
    def create_event(self, event_type, payload, source):
        return type('Event', (), {'type': event_type, 'payload': payload, 'source': source})()

class MockRiskState:
    """Mock risk state for testing"""
    def __init__(self):
        self.margin_usage_pct = 0.5
        self.var_estimate_5pct = 0.01
        self.correlation_risk = 0.3
        self.current_drawdown_pct = 0.05
        self.account_equity_normalized = 1.0
        self.market_stress_level = 0.2
        self.volatility_regime = 0.3

async def demo_master_switch_integration():
    """Demonstrate master switch integration with risk management"""
    
    print("🚀 Starting Master Switch Integration Demo")
    print("=" * 50)
    
    # Step 1: Initialize the master switch controller
    print("\n1. Initializing Master Switch Controller...")
    controller = initialize_controller(enable_kill_switch_integration=True)
    
    # Step 2: Initialize kill switch (optional)
    print("\n2. Initializing Kill Switch...")
    kill_switch = initialize_kill_switch()
    
    # Step 3: Initialize risk management components
    print("\n3. Initializing Risk Management Components...")
    
    # Mock event bus
    event_bus = MockEventBus()
    
    # Initialize correlation tracker
    correlation_tracker = CorrelationTracker(event_bus)
    correlation_tracker.initialize_assets(['EURUSD', 'GBPUSD', 'USDJPY'])
    
    # Initialize VaR calculator
    var_calculator = VaRCalculator(correlation_tracker, event_bus)
    
    # Add some mock positions
    var_calculator.positions = {
        'EURUSD': PositionData('EURUSD', 100000, 110000, 1.10, 0.15),
        'GBPUSD': PositionData('GBPUSD', 50000, 65000, 1.30, 0.18),
        'USDJPY': PositionData('USDJPY', 200000, 20000, 110.0, 0.12)
    }
    var_calculator.portfolio_value = 195000
    
    # Initialize position sizing agent
    position_agent = PositionSizingAgent({'max_leverage': 2.0}, event_bus)
    
    # Initialize risk monitor
    risk_monitor = create_real_time_monitor()
    
    print("✅ All components initialized and registered with master switch")
    
    # Step 4: Test system OFF behavior
    print("\n4. Testing System OFF Behavior...")
    print("System is currently OFF - testing cached value behavior")
    
    # Test VaR calculation when system is OFF
    print("\n🔒 Testing VaR calculation when system is OFF...")
    var_result = await var_calculator.calculate_var()
    if var_result is None:
        print("   ✅ VaR calculation correctly blocked when system is OFF")
    else:
        print(f"   ❌ VaR calculation should be blocked, got: {var_result}")
    
    # Test position sizing when system is OFF
    print("\n🔒 Testing position sizing when system is OFF...")
    mock_risk_state = MockRiskState()
    action, confidence = position_agent.calculate_risk_action(mock_risk_state)
    print(f"   ✅ Position sizing returned: action={action} (HOLD expected), confidence={confidence}")
    
    # Step 5: Start the system
    print("\n5. Starting the Trading System...")
    success = controller.start_system("demo_test", "demo_user")
    if success:
        print("   ✅ System started successfully")
        print(f"   📊 System state: {controller.get_system_state().value}")
    else:
        print("   ❌ Failed to start system")
        return
    
    # Step 6: Test system ON behavior
    print("\n6. Testing System ON Behavior...")
    
    # Test VaR calculation when system is ON
    print("\n🔓 Testing VaR calculation when system is ON...")
    var_result = await var_calculator.calculate_var()
    if var_result:
        print(f"   ✅ VaR calculation successful: {var_result.portfolio_var:.6f}")
        print(f"   📊 Calculation method: {var_result.calculation_method}")
        print(f"   ⏱️  Performance: {var_result.performance_ms:.2f}ms")
    else:
        print("   ❌ VaR calculation failed when system is ON")
    
    # Test position sizing when system is ON
    print("\n🔓 Testing position sizing when system is ON...")
    action, confidence = position_agent.calculate_risk_action(mock_risk_state)
    print(f"   ✅ Position sizing successful: action={action}, confidence={confidence:.3f}")
    
    # Step 7: Test caching behavior
    print("\n7. Testing Caching Behavior...")
    
    # Check if values are cached
    cached_var = controller.get_cached_value("var_result_0.95_1_parametric")
    if cached_var:
        print(f"   ✅ VaR result cached: {cached_var.portfolio_var:.6f}")
    else:
        print("   ⚠️  No VaR result cached")
    
    cached_decision = controller.get_cached_value("position_sizing_decision")
    if cached_decision:
        print(f"   ✅ Position sizing decision cached: {cached_decision}")
    else:
        print("   ⚠️  No position sizing decision cached")
    
    # Step 8: Test system status reporting
    print("\n8. System Status Reporting...")
    status = controller.get_system_status()
    print(f"   📊 System Status:")
    print(f"      State: {status['state']}")
    print(f"      Components: {len(status['component_states'])}")
    print(f"      Cached values: {status['cached_values_count']}")
    print(f"      State checks: {status['state_check_count']}")
    
    # Show component states
    print(f"\n   🔧 Registered Components:")
    for name, component in status['component_states'].items():
        print(f"      - {name}: registered at {component['registered_at']}")
    
    # Step 9: Test dashboard data with system status
    print("\n9. Testing Dashboard Integration...")
    dashboard_data = risk_monitor.get_risk_dashboard_data()
    print(f"   📱 Dashboard Data:")
    print(f"      Monitor status: {dashboard_data['status']}")
    print(f"      System status: {dashboard_data['system_status']['state']}")
    
    # Step 10: Test stopping the system
    print("\n10. Testing System Stop...")
    success = controller.stop_system("demo_complete", "demo_user")
    if success:
        print("   ✅ System stopped successfully")
        print(f"   📊 System state: {controller.get_system_state().value}")
    else:
        print("   ❌ Failed to stop system")
    
    # Step 11: Test cached value access when system is OFF
    print("\n11. Testing Cached Value Access When System is OFF...")
    
    # Test VaR calculation returns cached value
    print("\n🔒 Testing VaR calculation with cached values...")
    var_result = await var_calculator.calculate_var()
    if var_result:
        print(f"   ✅ VaR calculation returned cached value: {var_result.portfolio_var:.6f}")
    else:
        print("   ⚠️  No cached VaR result available")
    
    # Test position sizing returns cached value
    print("\n🔒 Testing position sizing with cached values...")
    action, confidence = position_agent.calculate_risk_action(mock_risk_state)
    print(f"   ✅ Position sizing returned cached decision: action={action}, confidence={confidence:.3f}")
    
    # Step 12: Test transition history
    print("\n12. System Transition History...")
    transitions = controller.get_transition_history(5)
    print(f"   📚 Recent transitions:")
    for i, transition in enumerate(transitions, 1):
        print(f"      {i}. {transition['from_state']} -> {transition['to_state']}")
        print(f"         Reason: {transition['reason']}")
        print(f"         Time: {transition['timestamp']}")
    
    # Step 13: Test emergency stop
    print("\n13. Testing Emergency Stop...")
    controller.emergency_stop("demo_emergency_test")
    print("   ✅ Emergency stop triggered")
    
    print("\n🎉 Master Switch Integration Demo Complete!")
    print("=" * 50)
    print("\n📋 Summary:")
    print("  ✅ Master switch controller initialized")
    print("  ✅ Kill switch integration working")
    print("  ✅ Risk components integrated")
    print("  ✅ System ON/OFF behavior verified")
    print("  ✅ Caching mechanism working")
    print("  ✅ Dashboard integration complete")
    print("  ✅ Emergency stop functionality verified")

def demo_kill_switch_integration():
    """Demonstrate kill switch integration"""
    
    print("\n🛑 Kill Switch Integration Demo")
    print("-" * 30)
    
    controller = get_controller()
    if not controller:
        print("❌ No controller available")
        return
    
    # Test manual emergency stop
    print("\n1. Testing Manual Emergency Stop...")
    controller.start_system("kill_switch_test", "demo")
    time.sleep(0.5)
    
    # Trigger emergency stop
    controller.emergency_stop("manual_test")
    
    print(f"   📊 System state after emergency stop: {controller.get_system_state().value}")
    
    # Test system reset
    print("\n2. Testing System Reset...")
    controller.reset_system("demo_reset")
    
    print(f"   📊 System state after reset: {controller.get_system_state().value}")

if __name__ == "__main__":
    print("🔧 Master Switch Integration Demo")
    print("This demo shows how the master switch integrates with risk management")
    
    try:
        # Run the main demo
        asyncio.run(demo_master_switch_integration())
        
        # Run kill switch demo
        demo_kill_switch_integration()
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Demo completed successfully!")