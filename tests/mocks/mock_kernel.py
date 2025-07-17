"""
Mock AlgoSpace Kernel for testing

Provides a minimal kernel implementation for unit testing components
in isolation without full AlgoSpace infrastructure.
"""

from typing import Dict, Any, Optional
from src.core.minimal_dependencies import MinimalKernel, MinimalEventBus


class MockAlgoSpaceKernel(MinimalKernel):
    """
    Mock kernel for testing
    
    Provides test fixtures and controlled behavior for unit tests.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Default test configuration
        default_config = {
            "synergy": {
                "mlmi_threshold": 0.5,
                "nwrqk_threshold": 0.5,
                "fvg_threshold": 0.5,
                "time_window_bars": 10,
                "bar_duration_minutes": 5,
                "cooldown_bars": 50
            },
            "matrix": {
                "shape": [48, 13],
                "feature_names": [
                    "sma_20", "sma_50", "rsi", "macd", "macd_signal",
                    "bb_upper", "bb_lower", "volume", "atr", "mlmi",
                    "mmd_1", "mmd_2", "mmd_3"
                ]
            },
            "environment": {
                "max_timesteps": 100,
                "feature_indices": {
                    "mlmi_expert": [0, 1, 9, 10],
                    "nwrqk_expert": [2, 3, 4, 5],
                    "regime_expert": [10, 11, 12]
                }
            }
        }
        
        # Merge with provided config
        if config:
            default_config.update(config)
            
        super().__init__(default_config)
        
        # Test fixtures
        self.published_events = []
        self.component_calls = {}
        
        # Override event bus to track events
        self.event_bus = MockEventBus(self)
        
    def track_call(self, component_name: str, method_name: str, *args, **kwargs):
        """Track component method calls for testing"""
        if component_name not in self.component_calls:
            self.component_calls[component_name] = []
            
        self.component_calls[component_name].append({
            "method": method_name,
            "args": args,
            "kwargs": kwargs
        })
        
    def get_calls(self, component_name: str) -> list:
        """Get tracked calls for a component"""
        return self.component_calls.get(component_name, [])
        
    def clear_calls(self):
        """Clear tracked calls"""
        self.component_calls.clear()
        self.published_events.clear()


class MockEventBus(MinimalEventBus):
    """Mock event bus that tracks published events"""
    
    def __init__(self, kernel: MockAlgoSpaceKernel):
        super().__init__()
        self.kernel = kernel
        
    def publish(self, event):
        """Override to track events"""
        self.kernel.published_events.append(event)
        super().publish(event)