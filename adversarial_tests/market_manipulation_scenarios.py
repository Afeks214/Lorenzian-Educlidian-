#!/usr/bin/env python3
"""
🚨 RED TEAM ADVERSARIAL TESTING SUITE
Agent 3 Mission: Sophisticated Market Manipulation Attack Scenarios

This module creates deceptive market scenarios designed to fool Strategic MARL agents:
- Bull traps (fake breakouts to lure bulls into losses)
- Bear traps (fake breakdowns to trigger panic selling)
- Whipsaw patterns (rapid direction changes)
- Fake volume spikes
- Coordinated manipulation patterns

MISSION: Prove the Strategic MARL system can detect and resist sophisticated market manipulation.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

class MarketManipulationGenerator:
    """
    Generates sophisticated market manipulation scenarios for testing 
    Strategic MARL agent resilience.
    """
    
    def __init__(self, base_price: float = 15000.0, volatility: float = 0.02):
        self.base_price = base_price
        self.volatility = volatility
        self.manipulation_patterns = []
        
    def generate_bull_trap_scenario(self, bars: int = 100) -> pd.DataFrame:
        """
        🎯 SOPHISTICATED BULL TRAP ATTACK
        
        Creates a fake bullish breakout designed to lure algorithmic traders
        into long positions before executing a sharp reversal.
        
        Pattern:
        1. Gradual uptrend to build confidence
        2. Fake volume spike to simulate institutional buying
        3. False breakout above resistance
        4. Sudden sharp reversal to trigger stop losses
        """
        print("🚨 GENERATING BULL TRAP ATTACK SCENARIO...")
        
        dates = pd.date_range(start='2024-01-01', periods=bars, freq='30min')
        data = []
        
        current_price = self.base_price
        manipulation_phase = 0  # 0: setup, 1: trap_set, 2: execution
        
        for i in range(bars):
            if i < bars * 0.6:  # Setup phase - gradual uptrend
                manipulation_phase = 0
                trend = 0.0005  # Gentle upward bias
                volume_multiplier = 1.0
            elif i < bars * 0.8:  # Trap setting phase
                manipulation_phase = 1
                trend = 0.002   # Stronger upward momentum
                volume_multiplier = 2.5  # FAKE VOLUME SPIKE
            else:  # Execution phase - the trap is sprung
                manipulation_phase = 2
                trend = -0.004  # Sharp reversal
                volume_multiplier = 3.0  # Panic selling volume
            
            # Generate OHLC with manipulation bias
            daily_return = np.random.normal(trend, self.volatility)
            
            if manipulation_phase == 1:  # Add fake breakout pattern
                daily_return += 0.001 * np.sin(i * 0.5)  # Oscillating momentum
            elif manipulation_phase == 2:  # Sharp reversal pattern
                daily_return -= 0.002 * (i - bars * 0.8) / (bars * 0.2)
            
            # Calculate price movements
            open_price = current_price
            close_price = open_price * (1 + daily_return)
            
            # Generate deceptive high/low
            if manipulation_phase == 1:  # Fake breakout highs
                high_price = close_price * (1 + abs(daily_return) * 1.5)
                low_price = min(open_price, close_price) * (1 - abs(daily_return) * 0.5)
            else:
                high_price = max(open_price, close_price) * (1 + abs(daily_return) * 0.8)
                low_price = min(open_price, close_price) * (1 - abs(daily_return) * 0.8)
            
            # Generate deceptive volume
            base_volume = 1000
            volume = int(base_volume * volume_multiplier * (1 + np.random.uniform(0.5, 1.5)))
            
            data.append({
                'Date': dates[i],
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume,
                'manipulation_phase': manipulation_phase,
                'attack_type': 'bull_trap'
            })
            
            current_price = close_price
        
        return pd.DataFrame(data)
    
    def generate_whipsaw_attack(self, bars: int = 100) -> pd.DataFrame:
        """
        🌪️ WHIPSAW PATTERN ATTACK
        
        Creates rapid directional changes designed to trigger maximum 
        stop losses and confuse momentum-based algorithms.
        """
        print("🚨 GENERATING WHIPSAW ATTACK SCENARIO...")
        
        dates = pd.date_range(start='2024-01-01', periods=bars, freq='30min')
        data = []
        
        current_price = self.base_price
        direction = 1  # 1 for up, -1 for down
        bars_in_direction = 0
        switch_frequency = 5  # Switch every 5 bars for maximum chaos
        
        for i in range(bars):
            # Rapid direction switching logic
            if bars_in_direction >= switch_frequency:
                direction *= -1  # Reverse direction
                bars_in_direction = 0
                switch_frequency = np.random.randint(3, 8)  # Randomize next switch
            
            # Generate deceptive price movement
            base_move = direction * 0.001  # Base directional movement
            noise = np.random.normal(0, self.volatility)
            
            # Add momentum deception
            if bars_in_direction < 2:  # First bars in new direction - fake strength
                momentum_boost = direction * 0.0015
            else:  # Later bars - weakening signal
                momentum_boost = direction * 0.0005
            
            daily_return = base_move + momentum_boost + noise
            
            # Calculate prices
            open_price = current_price
            close_price = open_price * (1 + daily_return)
            
            # Generate extreme high/low to trigger stops
            extreme_factor = 1.5 if bars_in_direction < 2 else 1.0
            high_price = max(open_price, close_price) * (1 + abs(daily_return) * extreme_factor)
            low_price = min(open_price, close_price) * (1 - abs(daily_return) * extreme_factor)
            
            # Variable volume based on directional change
            volume_spike = 2.0 if bars_in_direction == 0 else 1.0
            volume = int(1000 * volume_spike * (1 + np.random.uniform(0.5, 1.5)))
            
            data.append({
                'Date': dates[i],
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume,
                'direction': direction,
                'bars_in_direction': bars_in_direction,
                'attack_type': 'whipsaw'
            })
            
            current_price = close_price
            bars_in_direction += 1
        
        return pd.DataFrame(data)
    
    def generate_fake_breakout_scenario(self, bars: int = 100) -> pd.DataFrame:
        """
        📈 FAKE BREAKOUT ATTACK
        
        Simulates institutional-level manipulation with fake support/resistance breaks
        designed to trigger algorithmic trading systems into poor decisions.
        """
        print("🚨 GENERATING FAKE BREAKOUT ATTACK SCENARIO...")
        
        dates = pd.date_range(start='2024-01-01', periods=bars, freq='30min')
        data = []
        
        current_price = self.base_price
        resistance_level = self.base_price * 1.02  # 2% above
        support_level = self.base_price * 0.98     # 2% below
        
        manipulation_stage = 0  # 0: range, 1: fake_break, 2: reversal
        
        for i in range(bars):
            if i < bars * 0.7:  # Range-bound phase
                manipulation_stage = 0
                # Keep price in range with slight bias toward resistance
                target_price = np.random.uniform(support_level * 1.005, resistance_level * 0.995)
                move_toward_target = (target_price - current_price) / current_price * 0.3
                daily_return = move_toward_target + np.random.normal(0, self.volatility * 0.5)
                
            elif i < bars * 0.85:  # Fake breakout phase
                manipulation_stage = 1
                # Force breakout above resistance with fake volume
                if current_price < resistance_level * 1.01:
                    daily_return = 0.005  # Strong upward move
                else:
                    daily_return = np.random.normal(0.001, self.volatility * 0.8)
                    
            else:  # Reversal phase - trap is sprung
                manipulation_stage = 2
                # Sharp reversal below original support
                daily_return = -0.008 + np.random.normal(0, self.volatility)
            
            # Calculate prices
            open_price = current_price
            close_price = open_price * (1 + daily_return)
            
            # Generate volume patterns that look institutional
            if manipulation_stage == 0:  # Normal range volume
                volume = int(1000 * (1 + np.random.uniform(0.3, 0.8)))
            elif manipulation_stage == 1:  # Fake institutional volume
                volume = int(1000 * (3 + np.random.uniform(0.5, 2.0)))
            else:  # Panic selling volume
                volume = int(1000 * (4 + np.random.uniform(1.0, 3.0)))
            
            # Calculate high/low based on manipulation stage
            if manipulation_stage == 1:  # Fake breakout - exaggerated highs
                high_price = max(open_price, close_price) * (1 + abs(daily_return) * 1.8)
                low_price = min(open_price, close_price) * (1 - abs(daily_return) * 0.3)
            else:
                high_price = max(open_price, close_price) * (1 + abs(daily_return) * 1.2)
                low_price = min(open_price, close_price) * (1 - abs(daily_return) * 1.2)
            
            data.append({
                'Date': dates[i],
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume,
                'manipulation_stage': manipulation_stage,
                'resistance_level': resistance_level,
                'support_level': support_level,
                'attack_type': 'fake_breakout'
            })
            
            current_price = close_price
        
        return pd.DataFrame(data)
    
    def generate_coordinated_attack_scenario(self, bars: int = 100) -> pd.DataFrame:
        """
        🎯 COORDINATED MANIPULATION ATTACK
        
        Simulates a sophisticated coordinated attack combining multiple manipulation
        techniques to maximize confusion for algorithmic systems.
        """
        print("🚨 GENERATING COORDINATED MANIPULATION ATTACK...")
        
        dates = pd.date_range(start='2024-01-01', periods=bars, freq='30min')
        data = []
        
        current_price = self.base_price
        attack_phases = [
            'accumulation',  # Quiet accumulation
            'markup',        # Price inflation
            'distribution',  # Selling to retail
            'decline'        # Sharp decline
        ]
        
        phase_lengths = [0.3, 0.25, 0.25, 0.2]  # Relative lengths
        phase_boundaries = np.cumsum([0] + [int(bars * length) for length in phase_lengths])
        
        for i in range(bars):
            # Determine current phase
            current_phase = 0
            for j, boundary in enumerate(phase_boundaries[1:]):
                if i < boundary:
                    current_phase = j
                    break
            
            phase_name = attack_phases[current_phase]
            
            # Phase-specific manipulation
            if phase_name == 'accumulation':
                # Quiet accumulation with price suppression
                daily_return = np.random.normal(-0.0002, self.volatility * 0.6)
                volume_multiplier = 0.7  # Lower volume to hide accumulation
                
            elif phase_name == 'markup':
                # Aggressive price inflation with fake indicators
                daily_return = np.random.normal(0.003, self.volatility * 0.8)
                volume_multiplier = 1.8  # Moderate volume increase
                
            elif phase_name == 'distribution':
                # Maintaining high prices while secretly selling
                daily_return = np.random.normal(0.0005, self.volatility * 1.2)
                volume_multiplier = 2.5  # High volume from selling
                
            else:  # decline
                # Sharp decline to complete the manipulation
                daily_return = np.random.normal(-0.006, self.volatility * 1.5)
                volume_multiplier = 3.0  # Panic selling volume
            
            # Add phase transition signals
            phase_progress = (i - phase_boundaries[current_phase]) / (phase_boundaries[current_phase + 1] - phase_boundaries[current_phase])
            
            if phase_progress > 0.8 and current_phase < len(attack_phases) - 1:
                # Signal transition to next phase
                daily_return += 0.001 * (current_phase % 2 * 2 - 1)  # Alternating signals
            
            # Calculate prices
            open_price = current_price
            close_price = open_price * (1 + daily_return)
            
            # Generate deceptive high/low
            high_price = max(open_price, close_price) * (1 + abs(daily_return) * 1.3)
            low_price = min(open_price, close_price) * (1 - abs(daily_return) * 1.3)
            
            # Calculate volume with phase-based multiplier
            base_volume = 1000
            volume = int(base_volume * volume_multiplier * (1 + np.random.uniform(0.3, 1.2)))
            
            data.append({
                'Date': dates[i],
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume,
                'manipulation_phase': phase_name,
                'phase_progress': phase_progress,
                'attack_type': 'coordinated'
            })
            
            current_price = close_price
        
        return pd.DataFrame(data)
    
    def generate_all_attack_scenarios(self, output_dir: str = "adversarial_tests/data") -> Dict[str, str]:
        """
        Generate all manipulation scenarios and save to files for testing.
        """
        print("🚨 GENERATING COMPLETE ADVERSARIAL ATTACK SUITE...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        scenarios = {
            'bull_trap': self.generate_bull_trap_scenario,
            'whipsaw': self.generate_whipsaw_attack,
            'fake_breakout': self.generate_fake_breakout_scenario,
            'coordinated': self.generate_coordinated_attack_scenario
        }
        
        file_paths = {}
        
        for scenario_name, generator_func in scenarios.items():
            print(f"\n🎯 Generating {scenario_name} attack scenario...")
            data = generator_func()
            
            file_path = os.path.join(output_dir, f"attack_{scenario_name}.csv")
            data.to_csv(file_path, index=False)
            file_paths[scenario_name] = file_path
            
            print(f"✅ {scenario_name} attack scenario saved to {file_path}")
            print(f"   📊 Bars: {len(data)}")
            print(f"   💰 Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
            print(f"   📈 Total return: {((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100:.2f}%")
        
        return file_paths

def main():
    """
    Execute the complete market manipulation attack generation suite.
    """
    print("🚨" * 20)
    print("RED TEAM ADVERSARIAL TESTING - MARKET MANIPULATION SCENARIOS")
    print("🚨" * 20)
    
    generator = MarketManipulationGenerator()
    file_paths = generator.generate_all_attack_scenarios()
    
    print("\n" + "="*80)
    print("🎯 ATTACK SCENARIOS GENERATION COMPLETE")
    print("="*80)
    
    for scenario, path in file_paths.items():
        print(f"📁 {scenario.upper()}: {path}")
    
    print("\n🚨 READY FOR STRATEGIC MARL AGENT TESTING")
    print("These scenarios are designed to test agent resilience against:")
    print("- Deceptive market signals")
    print("- Fake volume patterns") 
    print("- Coordinated manipulation")
    print("- Rapid directional changes")
    print("- False breakout patterns")
    
    return file_paths

if __name__ == "__main__":
    main()