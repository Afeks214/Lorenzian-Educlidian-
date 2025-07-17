"""
Mock Synergy Detector for testing

Provides deterministic synergy detection for unit testing components
that depend on synergy patterns.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from src.core.minimal_dependencies import MinimalComponentBase, Event


class MockSynergyDetector(MinimalComponentBase):
    """
    Mock synergy detector with controllable outputs
    
    Allows tests to simulate different synergy scenarios without
    complex market data setup.
    """
    
    def __init__(self, name: str = "mock_synergy_detector", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        
        # Current synergy state
        self.current_synergy: Optional[Dict[str, Any]] = None
        self.scenario = "none"
        self.detection_history = []
        
    def set_scenario(self, scenario: str) -> None:
        """
        Set synergy scenario
        
        Available scenarios:
        - 'none': No synergy detected
        - 'type1_bullish': TYPE_1 bullish synergy
        - 'type1_bearish': TYPE_1 bearish synergy
        - 'type2_bullish': TYPE_2 bullish synergy
        - 'type2_bearish': TYPE_2 bearish synergy
        - 'type3_neutral': TYPE_3 neutral synergy
        - 'type4_high_confidence': TYPE_4 with high confidence
        - 'random': Random synergy patterns
        """
        self.scenario = scenario
        self._update_synergy()
        
    def _update_synergy(self) -> None:
        """Update current synergy based on scenario"""
        if self.scenario == "none":
            self.current_synergy = None
            
        elif self.scenario == "type1_bullish":
            self.current_synergy = {
                "type": "TYPE_1",
                "direction": 1,
                "confidence": 0.85,
                "timestamp": datetime.now(),
                "metadata": {
                    "signals": ["mlmi", "nwrqk", "fvg"],
                    "strength": 0.8
                }
            }
            
        elif self.scenario == "type1_bearish":
            self.current_synergy = {
                "type": "TYPE_1",
                "direction": -1,
                "confidence": 0.80,
                "timestamp": datetime.now(),
                "metadata": {
                    "signals": ["mlmi", "nwrqk", "fvg"],
                    "strength": 0.75
                }
            }
            
        elif self.scenario == "type2_bullish":
            self.current_synergy = {
                "type": "TYPE_2",
                "direction": 1,
                "confidence": 0.75,
                "timestamp": datetime.now(),
                "metadata": {
                    "signals": ["mlmi", "nwrqk"],
                    "strength": 0.7
                }
            }
            
        elif self.scenario == "type2_bearish":
            self.current_synergy = {
                "type": "TYPE_2",
                "direction": -1,
                "confidence": 0.70,
                "timestamp": datetime.now(),
                "metadata": {
                    "signals": ["mlmi", "nwrqk"],
                    "strength": 0.65
                }
            }
            
        elif self.scenario == "type3_neutral":
            self.current_synergy = {
                "type": "TYPE_3",
                "direction": 0,
                "confidence": 0.60,
                "timestamp": datetime.now(),
                "metadata": {
                    "signals": ["mlmi", "fvg"],
                    "strength": 0.5
                }
            }
            
        elif self.scenario == "type4_high_confidence":
            self.current_synergy = {
                "type": "TYPE_4",
                "direction": 1,
                "confidence": 0.95,
                "timestamp": datetime.now(),
                "metadata": {
                    "signals": ["nwrqk", "fvg"],
                    "strength": 0.9
                }
            }
            
        elif self.scenario == "random":
            import random
            types = ["TYPE_1", "TYPE_2", "TYPE_3", "TYPE_4"]
            self.current_synergy = {
                "type": random.choice(types),
                "direction": random.choice([-1, 0, 1]),
                "confidence": random.uniform(0.5, 0.95),
                "timestamp": datetime.now(),
                "metadata": {
                    "signals": random.sample(["mlmi", "nwrqk", "fvg"], k=2),
                    "strength": random.uniform(0.4, 0.9)
                }
            }
            
    def detect_synergy(self, feature_store: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Detect synergy (mock implementation)
        
        Returns predetermined synergy based on scenario.
        """
        synergy = self.current_synergy
        
        if synergy:
            # Record detection
            self.detection_history.append({
                "timestamp": datetime.now(),
                "synergy": synergy.copy(),
                "feature_store": feature_store.copy() if feature_store else {}
            })
            
        return synergy
        
    def get_current_synergy(self) -> Optional[Dict[str, Any]]:
        """Get current active synergy"""
        return self.current_synergy
        
    def clear_synergy(self) -> None:
        """Clear current synergy"""
        self.current_synergy = None
        
    def get_detection_history(self) -> List[Dict[str, Any]]:
        """Get synergy detection history"""
        return self.detection_history
        
    def reset(self) -> None:
        """Reset detector state"""
        self.current_synergy = None
        self.scenario = "none"
        self.detection_history.clear()
        
    def publish_synergy_event(self, event_bus) -> None:
        """Publish synergy event to event bus"""
        if self.current_synergy:
            event = Event(
                event_type="SYNERGY_DETECTED",
                data=self.current_synergy,
                source=self.name
            )
            event_bus.publish(event)