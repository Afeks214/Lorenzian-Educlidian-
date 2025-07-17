"""
Strategic MAPPO Trainer for Google Colab - FIXED VERSION
Fixed BatchNorm issue for single-sample inference
Optimized for both GPU training and CPU inference
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from collections import deque
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategicActorNetwork(nn.Module):
    """Actor network for strategic MAPPO agents - FIXED with LayerNorm"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        super(StrategicActorNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # Changed from BatchNorm1d to LayerNorm
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # Changed from BatchNorm1d to LayerNorm
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, state):
        return self.net(state)

class StrategicCriticNetwork(nn.Module):
    """Critic network for strategic value estimation - FIXED with LayerNorm"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 512):
        super(StrategicCriticNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # Changed from BatchNorm1d to LayerNorm
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # Changed from BatchNorm1d to LayerNorm
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, state):
        return self.net(state)

class StrategicMAPPOTrainer:
    """
    Strategic MAPPO Trainer for Long-term Trading Strategy - FIXED VERSION
    Uses 30-minute data for strategic decision making
    Fixed BatchNorm issues for single-sample inference
    """
    
    def __init__(self, 
                 state_dim: int = 13,  # 30min matrix features including MMD
                 action_dim: int = 7,  # [HOLD, BUY_CONSERVATIVE, BUY_AGGRESSIVE, SELL_CONSERVATIVE, SELL_AGGRESSIVE, REDUCE_RISK, INCREASE_RISK]
                 n_agents: int = 3,    # strategic_agent, portfolio_manager, regime_detector
                 lr_actor: float = 1e-4,
                 lr_critic: float = 3e-4,
                 gamma: float = 0.995,  # Higher gamma for strategic decisions
                 eps_clip: float = 0.1,  # Smaller clip for stability
                 k_epochs: int = 8,     # More epochs for strategic learning
                 device: str = None):
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Strategic Trainer using device: {self.device}")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        # Initialize networks for each agent
        self.actors = []
        self.critics = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        
        for i in range(n_agents):
            actor = StrategicActorNetwork(state_dim, action_dim).to(self.device)
            critic = StrategicCriticNetwork(state_dim).to(self.device)
            
            self.actors.append(actor)
            self.critics.append(critic)
            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=lr_actor, weight_decay=1e-5))
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=lr_critic, weight_decay=1e-5))
        
        # Strategic training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []
        self.portfolio_values = []
        self.sharpe_ratios = []
        self.max_drawdowns = []
        
        self.training_stats = {
            'episodes': 0,
            'total_steps': 0,
            'best_reward': float('-inf'),
            'best_sharpe': 0.0,
            'avg_reward_50': 0.0,
            'current_regime': 'UNKNOWN'
        }
        
        # Strategic experience buffer (larger for longer-term patterns)
        self.buffer_size = 50000
        self.clear_buffers()
        
        # Market regime tracking
        self.regime_history = deque(maxlen=100)
        self.volatility_history = deque(maxlen=50)
        
    def clear_buffers(self):
        """Clear experience buffers"""
        self.states = [[] for _ in range(self.n_agents)]
        self.actions = [[] for _ in range(self.n_agents)]
        self.rewards = [[] for _ in range(self.n_agents)]
        self.log_probs = [[] for _ in range(self.n_agents)]
        self.values = [[] for _ in range(self.n_agents)]
        self.dones = [[] for _ in range(self.n_agents)]
        
    def detect_market_regime(self, data: pd.DataFrame, lookback: int = 20) -> str:
        """Detect current market regime"""
        if len(data) < lookback:
            return 'UNKNOWN'
            
        # Calculate volatility
        returns = data['Close'].pct_change().dropna()
        volatility = returns.rolling(lookback).std().iloc[-1]
        
        # Calculate trend strength
        sma_short = data['Close'].rolling(5).mean().iloc[-1]
        sma_long = data['Close'].rolling(20).mean().iloc[-1]
        trend_strength = (sma_short - sma_long) / sma_long
        
        # Volume analysis
        volume_avg = data['Volume'].rolling(20).mean().iloc[-1]
        volume_current = data['Volume'].iloc[-1]
        volume_ratio = volume_current / volume_avg
        
        # Regime classification
        if volatility > 0.03:  # High volatility
            if volume_ratio > 1.5:
                regime = 'CRISIS'
            else:
                regime = 'VOLATILE'
        elif abs(trend_strength) > 0.02:  # Strong trend
            if trend_strength > 0:
                regime = 'BULL_TREND'
            else:
                regime = 'BEAR_TREND'
        elif volatility < 0.01:  # Low volatility
            regime = 'CONSOLIDATION'
        else:
            regime = 'NORMAL'
            
        self.regime_history.append(regime)
        self.volatility_history.append(volatility)
        self.training_stats['current_regime'] = regime
        
        return regime
    
    def calculate_strategic_features(self, data: pd.DataFrame, lookback: int = 48) -> np.ndarray:
        """Calculate strategic features for 30min data"""
        if len(data) < lookback:
            return np.zeros(self.state_dim)
            
        # Price-based features
        closes = data['Close'].values[-lookback:]
        highs = data['High'].values[-lookback:]
        lows = data['Low'].values[-lookback:]
        volumes = data['Volume'].values[-lookback:]
        
        # Trend features
        price_change_1d = (closes[-1] - closes[-8]) / closes[-8]  # 4 hours (8 * 30min)
        price_change_3d = (closes[-1] - closes[-24]) / closes[-24]  # 12 hours
        price_change_7d = (closes[-1] - closes[-48]) / closes[-48] if len(closes) >= 48 else 0
        
        # Volatility features
        returns = np.diff(closes) / closes[:-1]
        volatility_short = np.std(returns[-16:])  # 8 hours
        volatility_long = np.std(returns[-32:]) if len(returns) >= 32 else volatility_short
        
        # Volume features
        volume_ratio = volumes[-1] / np.mean(volumes[-20:])
        volume_trend = np.polyfit(range(20), volumes[-20:], 1)[0]
        
        # Technical indicators
        rsi = self._calculate_rsi(closes, 14)
        bb_position = self._calculate_bollinger_position(closes, 20)
        macd_signal = self._calculate_macd_signal(closes)
        
        # Market structure
        higher_highs = np.sum(highs[-10:] > np.max(highs[-20:-10])) / 10
        lower_lows = np.sum(lows[-10:] < np.min(lows[-20:-10])) / 10
        
        # MMD (Market Microstructure Dynamics) approximation
        mmd = self._calculate_mmd_proxy(data)
        
        features = np.array([
            price_change_1d,
            price_change_3d, 
            price_change_7d,
            volatility_short,
            volatility_long,
            volume_ratio,
            volume_trend / 100000,  # Normalize
            rsi / 100,
            bb_position,
            macd_signal,
            higher_highs,
            lower_lows,
            mmd
        ])
        
        # Ensure no NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return features
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_position(self, prices: np.ndarray, period: int = 20) -> float:
        """Calculate position within Bollinger Bands"""
        if len(prices) < period:
            return 0.5
            
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        
        if upper_band == lower_band:
            return 0.5
            
        position = (prices[-1] - lower_band) / (upper_band - lower_band)
        return np.clip(position, 0, 1)
    
    def _calculate_macd_signal(self, prices: np.ndarray) -> float:
        """Calculate MACD signal"""
        if len(prices) < 26:
            return 0.0
            
        ema12 = self._ema(prices, 12)
        ema26 = self._ema(prices, 26)
        macd_line = ema12[-1] - ema26[-1]
        
        if len(prices) < 35:
            return macd_line / prices[-1]
            
        macd_values = ema12[-9:] - ema26[-9:]
        signal_line = self._ema(macd_values, 9)[-1]
        
        return (macd_line - signal_line) / prices[-1]
    
    def _ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
            
        return ema
    
    def _calculate_mmd_proxy(self, data: pd.DataFrame) -> float:
        """Calculate Market Microstructure Dynamics proxy"""
        if len(data) < 20:
            return 0.0
            
        # Price impact approximation
        price_changes = data['Close'].pct_change().values[-20:]
        volume_changes = data['Volume'].pct_change().values[-20:]
        
        # Remove NaN values
        valid_idx = ~(np.isnan(price_changes) | np.isnan(volume_changes))
        price_changes = price_changes[valid_idx]
        volume_changes = volume_changes[valid_idx]
        
        if len(price_changes) < 5:
            return 0.0
            
        # Simple correlation as MMD proxy
        correlation = np.corrcoef(np.abs(price_changes), volume_changes)[0, 1]
        
        return 0.0 if np.isnan(correlation) else correlation
    
    def get_action(self, states: List[np.ndarray], agent_idx: int = None, deterministic: bool = False):
        """Get actions for agents given states - FIXED for single sample inference"""
        # Set models to evaluation mode for inference
        for actor in self.actors:
            actor.eval()
        for critic in self.critics:
            critic.eval()
            
        try:
            if agent_idx is not None:
                # Single agent action
                state_tensor = torch.FloatTensor(states).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action_probs = self.actors[agent_idx](state_tensor)
                    value = self.critics[agent_idx](state_tensor)
                    
                if deterministic:
                    action = torch.argmax(action_probs, dim=-1)
                else:
                    action = torch.multinomial(action_probs, 1)
                    
                log_prob = torch.log(action_probs.gather(1, action))
                
                return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0]
            else:
                # Multi-agent actions
                actions = []
                log_probs = []
                values = []
                
                for i in range(self.n_agents):
                    state_tensor = torch.FloatTensor(states[i]).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        action_probs = self.actors[i](state_tensor)
                        value = self.critics[i](state_tensor)
                        
                    if deterministic:
                        action = torch.argmax(action_probs, dim=-1)
                    else:
                        action = torch.multinomial(action_probs, 1)
                        
                    log_prob = torch.log(action_probs.gather(1, action))
                    
                    actions.append(action.cpu().numpy()[0])
                    log_probs.append(log_prob.cpu().numpy()[0])
                    values.append(value.cpu().numpy()[0])
                    
                return actions, log_probs, values
        finally:
            # Set models back to training mode
            for actor in self.actors:
                actor.train()
            for critic in self.critics:
                critic.train()
    
    def store_transition(self, 
                        states: List[np.ndarray], 
                        actions: List[int], 
                        rewards: List[float], 
                        log_probs: List[float], 
                        values: List[float], 
                        dones: List[bool]):
        """Store transition in experience buffers"""
        for i in range(self.n_agents):
            self.states[i].append(states[i])
            self.actions[i].append(actions[i])
            self.rewards[i].append(rewards[i])
            self.log_probs[i].append(log_probs[i])
            self.values[i].append(values[i])
            self.dones[i].append(dones[i])
    
    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool], 
                   next_value: float = 0.0, gae_lambda: float = 0.98):
        """Compute Generalized Advantage Estimation with higher lambda for strategic decisions"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[i]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[i+1]
                next_val = values[i+1]
                
            delta = rewards[i] + self.gamma * next_val * next_non_terminal - values[i]
            gae = delta + self.gamma * gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            
        return advantages
    
    def update_networks(self):
        """Update actor and critic networks using PPO with strategic enhancements"""
        if len(self.states[0]) < 32:  # Minimum batch size
            return
            
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        
        for agent_idx in range(self.n_agents):
            # Convert to tensors
            states = torch.FloatTensor(np.array(self.states[agent_idx])).to(self.device)
            actions = torch.LongTensor(self.actions[agent_idx]).to(self.device)
            old_log_probs = torch.FloatTensor(self.log_probs[agent_idx]).to(self.device)
            rewards = self.rewards[agent_idx]
            values = self.values[agent_idx]
            dones = self.dones[agent_idx]
            
            # Compute advantages
            advantages = self.compute_gae(rewards, values, dones)
            advantages = torch.FloatTensor(advantages).to(self.device)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Compute returns
            returns = advantages + torch.FloatTensor(values).to(self.device)
            
            # PPO update with strategic modifications
            for epoch in range(self.k_epochs):
                # Get current policy
                action_probs = self.actors[agent_idx](states)
                current_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1))).squeeze()
                current_values = self.critics[agent_idx](states).squeeze()
                
                # Compute ratio
                ratio = torch.exp(current_log_probs - old_log_probs)
                
                # Compute surrogate loss with entropy bonus for exploration
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1)
                
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()
                
                # Critic loss with value clipping
                values_tensor = torch.FloatTensor(values).to(self.device)
                value_pred_clipped = values_tensor + torch.clamp(
                    current_values - values_tensor,
                    -self.eps_clip, self.eps_clip
                )
                value_loss_1 = (current_values - returns).pow(2)
                value_loss_2 = (value_pred_clipped - returns).pow(2)
                critic_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()
                
                # Update networks
                self.actor_optimizers[agent_idx].zero_grad()
                actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), 0.5)
                self.actor_optimizers[agent_idx].step()
                
                self.critic_optimizers[agent_idx].zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critics[agent_idx].parameters(), 0.5)
                self.critic_optimizers[agent_idx].step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
        
        # Store losses
        self.actor_losses.append(total_actor_loss / (self.n_agents * self.k_epochs))
        self.critic_losses.append(total_critic_loss / (self.n_agents * self.k_epochs))
        
        # Clear buffers
        self.clear_buffers()
    
    def train_episode(self, data: pd.DataFrame, start_idx: int = 0, episode_length: int = 500):
        """Train one strategic episode"""
        episode_reward = 0.0
        episode_step = 0
        
        # Initialize strategic positions
        portfolio_value = 1000000.0  # $1M starting portfolio
        positions = [0.0] * self.n_agents  # Strategic positions
        cash_allocation = [1.0 / self.n_agents] * self.n_agents  # Equal allocation
        
        portfolio_history = []
        
        for step in range(episode_length):
            if start_idx + step + 48 >= len(data):  # Need 48 bars for strategic analysis
                break
                
            # Get current market regime
            current_data = data.iloc[start_idx + step:start_idx + step + 48]
            regime = self.detect_market_regime(current_data)
            
            # Prepare strategic states
            states = []
            for agent_idx in range(self.n_agents):
                base_features = self.calculate_strategic_features(current_data)
                
                # Add agent-specific features to reach state_dim=13
                if len(base_features) >= self.state_dim:
                    state = base_features[:self.state_dim]
                else:
                    # Pad with agent-specific features if needed
                    padding_needed = self.state_dim - len(base_features)
                    if agent_idx == 0:  # Strategic agent
                        agent_features = [
                            positions[agent_idx] / 10.0,  # Normalized position
                            cash_allocation[agent_idx]
                        ][:padding_needed]
                    elif agent_idx == 1:  # Portfolio manager
                        total_exposure = sum(abs(p) for p in positions)
                        agent_features = [
                            total_exposure / len(positions),
                            portfolio_value / 1000000.0
                        ][:padding_needed]
                    else:  # Regime detector
                        regime_stability = len(set(list(self.regime_history)[-5:])) / 5.0 if len(self.regime_history) >= 5 else 1.0
                        agent_features = [
                            regime_stability,
                            self.volatility_history[-1] if self.volatility_history else 0.0
                        ][:padding_needed]
                    
                    # Pad agent_features to exact length needed
                    while len(agent_features) < padding_needed:
                        agent_features.append(0.0)
                    
                    state = np.concatenate([base_features, agent_features[:padding_needed]])
                    
                states.append(state)
            
            # Get strategic actions
            actions, log_probs, values = self.get_action(states)
            
            # Execute strategic actions and calculate rewards
            rewards = []
            current_price = current_data['Close'].iloc[-1]
            
            for agent_idx in range(self.n_agents):
                action = actions[agent_idx]
                reward = 0.0
                
                # Strategic action mapping
                if action == 1:  # BUY_CONSERVATIVE
                    if cash_allocation[agent_idx] >= 0.1:
                        positions[agent_idx] += 0.5
                        cash_allocation[agent_idx] -= 0.1
                        reward = 0.1
                elif action == 2:  # BUY_AGGRESSIVE
                    if cash_allocation[agent_idx] >= 0.2:
                        positions[agent_idx] += 1.0
                        cash_allocation[agent_idx] -= 0.2
                        reward = 0.2
                elif action == 3:  # SELL_CONSERVATIVE
                    if positions[agent_idx] >= 0.5:
                        positions[agent_idx] -= 0.5
                        cash_allocation[agent_idx] += 0.1
                        reward = 0.1
                elif action == 4:  # SELL_AGGRESSIVE
                    if positions[agent_idx] >= 1.0:
                        positions[agent_idx] -= 1.0
                        cash_allocation[agent_idx] += 0.2
                        reward = 0.2
                elif action == 5:  # REDUCE_RISK
                    positions[agent_idx] *= 0.8
                    reward = 0.05
                elif action == 6:  # INCREASE_RISK
                    if regime in ['BULL_TREND', 'NORMAL']:
                        positions[agent_idx] *= 1.2
                        reward = 0.05
                    else:
                        reward = -0.1  # Penalty for increasing risk in bad regimes
                
                # Strategic P&L calculation (longer timeframe)
                if step >= 8:  # Look back 4 hours (8 * 30min)
                    prev_price = data.iloc[start_idx + step - 8]['Close']
                    strategic_pnl = positions[agent_idx] * (current_price - prev_price)
                    reward += strategic_pnl / 10000.0  # Scale for strategic timeframe
                
                # Regime-based reward adjustments
                if regime == 'CRISIS' and action == 0:  # HOLD during crisis
                    reward += 0.2
                elif regime == 'BULL_TREND' and action in [1, 2]:  # Buy in bull market
                    reward += 0.1
                elif regime == 'BEAR_TREND' and action in [3, 4]:  # Sell in bear market
                    reward += 0.1
                
                # Risk management penalties
                total_exposure = sum(abs(p) for p in positions)
                if total_exposure > 10.0:  # Too much leverage
                    reward -= 0.3
                
                rewards.append(reward)
            
            # Update portfolio value
            total_position_value = sum(positions) * current_price
            total_cash_value = sum(cash_allocation) * portfolio_value
            new_portfolio_value = total_cash_value + total_position_value
            
            portfolio_history.append(new_portfolio_value)
            portfolio_value = new_portfolio_value
            
            # Store transition
            dones = [False] * self.n_agents
            self.store_transition(states, actions, rewards, log_probs, values, dones)
            
            episode_reward += sum(rewards)
            episode_step += 1
            
            # Update networks periodically (less frequent for strategic decisions)
            if len(self.states[0]) >= 64:
                self.update_networks()
        
        # Final update
        if len(self.states[0]) > 0:
            self.update_networks()
        
        # Calculate strategic metrics
        if len(portfolio_history) > 1:
            returns = np.diff(portfolio_history) / portfolio_history[:-1]
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 48)  # Annualized
            
            # Calculate max drawdown
            peak = np.maximum.accumulate(portfolio_history)
            drawdown = (portfolio_history - peak) / peak
            max_drawdown = np.min(drawdown)
            
            self.sharpe_ratios.append(sharpe_ratio)
            self.max_drawdowns.append(max_drawdown)
            self.portfolio_values.append(portfolio_history[-1])
        
        # Update training stats
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_step)
        self.training_stats['episodes'] += 1
        self.training_stats['total_steps'] += episode_step
        
        if episode_reward > self.training_stats['best_reward']:
            self.training_stats['best_reward'] = episode_reward
        
        if self.sharpe_ratios and self.sharpe_ratios[-1] > self.training_stats['best_sharpe']:
            self.training_stats['best_sharpe'] = self.sharpe_ratios[-1]
        
        # Calculate average reward over last 50 episodes
        if len(self.episode_rewards) >= 50:
            self.training_stats['avg_reward_50'] = np.mean(self.episode_rewards[-50:])
        
        return episode_reward, episode_step, portfolio_history[-1] if portfolio_history else 0
    
    def get_strategic_stats(self) -> Dict:
        """Get current strategic training statistics"""
        return {
            'episodes': self.training_stats['episodes'],
            'total_steps': self.training_stats['total_steps'],
            'best_reward': self.training_stats['best_reward'],
            'best_sharpe': self.training_stats['best_sharpe'],
            'avg_reward_50': self.training_stats['avg_reward_50'],
            'current_regime': self.training_stats['current_regime'],
            'latest_reward': self.episode_rewards[-1] if self.episode_rewards else 0.0,
            'latest_portfolio_value': self.portfolio_values[-1] if self.portfolio_values else 0.0,
            'latest_sharpe': self.sharpe_ratios[-1] if self.sharpe_ratios else 0.0,
            'latest_drawdown': self.max_drawdowns[-1] if self.max_drawdowns else 0.0,
            'actor_loss': self.actor_losses[-1] if self.actor_losses else 0.0,
            'critic_loss': self.critic_losses[-1] if self.critic_losses else 0.0
        }