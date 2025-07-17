import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

class TradingMAEnv(AECEnv):
    """Multi-Agent Trading Environment"""
    
    metadata = {"name": "grandmodel_trading_v1"}
    
    def __init__(self):
        super().__init__()
        self.agents = ["strategic", "tactical", "risk"]
        self.possible_agents = self.agents[:]
        
        self._action_spaces = {
            "strategic": spaces.Discrete(3),
            "tactical": spaces.Discrete(3),
            "risk": spaces.Box(0, 1, shape=(1,))
        }
        
        self._observation_spaces = {
            "strategic": spaces.Box(-np.inf, np.inf, (48, 13)),
            "tactical": spaces.Box(-np.inf, np.inf, (60, 7)),
            "risk": spaces.Box(-np.inf, np.inf, (10,))
        }
    
    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        return {agent: np.zeros(self._observation_spaces[agent].shape) for agent in self.agents}
    
    def step(self, action):
        pass