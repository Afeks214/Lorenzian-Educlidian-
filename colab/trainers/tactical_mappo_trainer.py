"""
Tactical MAPPO Trainer for Google Colab
Optimized for GPU training with tactical decision making
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

class ActorNetwork(nn.Module):
    """Actor network for MAPPO tactical agent"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, state):
        return self.net(state)

class CriticNetwork(nn.Module):
    """Critic network for MAPPO value estimation"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super(CriticNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        return self.net(state)

class TacticalMAPPOTrainer:
    """
    MAPPO Trainer for Tactical Trading Agents
    Optimized for Google Colab environment with GPU acceleration
    """
    
    def __init__(self, 
                 state_dim: int = 7,  # 5min matrix features
                 action_dim: int = 5,  # [HOLD, BUY_SMALL, BUY_LARGE, SELL_SMALL, SELL_LARGE]
                 n_agents: int = 3,    # tactical_agent, risk_agent, execution_agent
                 lr_actor: float = 3e-4,
                 lr_critic: float = 1e-3,
                 gamma: float = 0.99,
                 eps_clip: float = 0.2,
                 k_epochs: int = 4,
                 device: str = None):
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
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
            actor = ActorNetwork(state_dim, action_dim).to(self.device)
            critic = CriticNetwork(state_dim).to(self.device)
            
            self.actors.append(actor)
            self.critics.append(critic)
            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=lr_actor))
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=lr_critic))
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []
        self.training_stats = {
            'episodes': 0,
            'total_steps': 0,
            'best_reward': float('-inf'),
            'avg_reward_100': 0.0
        }
        
        # Experience buffer
        self.buffer_size = 10000
        self.clear_buffers()
        
    def clear_buffers(self):
        """Clear experience buffers"""
        self.states = [[] for _ in range(self.n_agents)]
        self.actions = [[] for _ in range(self.n_agents)]
        self.rewards = [[] for _ in range(self.n_agents)]
        self.log_probs = [[] for _ in range(self.n_agents)]
        self.values = [[] for _ in range(self.n_agents)]
        self.dones = [[] for _ in range(self.n_agents)]
        
    def get_action(self, states: List[np.ndarray], agent_idx: int = None, deterministic: bool = False):
        """Get actions for agents given states"""
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
                   next_value: float = 0.0, gae_lambda: float = 0.95):
        """Compute Generalized Advantage Estimation"""
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
        """Update actor and critic networks using PPO"""
        if len(self.states[0]) < 64:  # Minimum batch size
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
            
            # PPO update
            for _ in range(self.k_epochs):
                # Get current policy
                action_probs = self.actors[agent_idx](states)
                current_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1))).squeeze()
                current_values = self.critics[agent_idx](states).squeeze()
                
                # Compute ratio
                ratio = torch.exp(current_log_probs - old_log_probs)
                
                # Compute surrogate loss
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss
                critic_loss = nn.MSELoss()(current_values, returns)
                
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
    
    def train_episode(self, data: pd.DataFrame, start_idx: int = 0, episode_length: int = 1000):
        """Train one episode"""
        episode_reward = 0.0
        episode_step = 0
        
        # Initialize positions and cash
        positions = [0.0] * self.n_agents  # Position size for each agent
        cash = [100000.0] * self.n_agents  # Starting cash
        
        for step in range(episode_length):
            if start_idx + step + 60 >= len(data):  # Need 60 bars for 5min matrix
                break
                
            # Prepare states (simplified 5min features)
            current_data = data.iloc[start_idx + step:start_idx + step + 60]
            states = []
            
            for agent_idx in range(self.n_agents):
                # Simple features: OHLCV + basic indicators
                close_prices = current_data['Close'].values
                volumes = current_data['Volume'].values
                
                # Calculate simple features
                price_change = (close_prices[-1] - close_prices[0]) / close_prices[0]
                volatility = np.std(close_prices[-20:]) / np.mean(close_prices[-20:])
                volume_avg = np.mean(volumes[-10:])
                price_momentum = (close_prices[-1] - close_prices[-5]) / close_prices[-5]
                rsi = self._calculate_rsi(close_prices, 14)
                sma_ratio = close_prices[-1] / np.mean(close_prices[-20:])
                position_ratio = positions[agent_idx] / 10.0  # Normalize position
                
                state = np.array([price_change, volatility, volume_avg/100000, 
                                price_momentum, rsi/100, sma_ratio, position_ratio])
                states.append(state)
            
            # Get actions
            actions, log_probs, values = self.get_action(states)
            
            # Execute actions and calculate rewards
            rewards = []
            current_price = data.iloc[start_idx + step]['Close']
            
            for agent_idx in range(self.n_agents):
                action = actions[agent_idx]
                reward = 0.0
                
                # Action mapping: 0=HOLD, 1=BUY_SMALL, 2=BUY_LARGE, 3=SELL_SMALL, 4=SELL_LARGE
                if action == 1:  # BUY_SMALL
                    if cash[agent_idx] >= current_price * 0.1:
                        positions[agent_idx] += 0.1
                        cash[agent_idx] -= current_price * 0.1
                        reward = 0.01  # Small positive reward for action
                elif action == 2:  # BUY_LARGE
                    if cash[agent_idx] >= current_price * 0.5:
                        positions[agent_idx] += 0.5
                        cash[agent_idx] -= current_price * 0.5
                        reward = 0.02
                elif action == 3:  # SELL_SMALL
                    if positions[agent_idx] >= 0.1:
                        positions[agent_idx] -= 0.1
                        cash[agent_idx] += current_price * 0.1
                        reward = 0.01
                elif action == 4:  # SELL_LARGE
                    if positions[agent_idx] >= 0.5:
                        positions[agent_idx] -= 0.5
                        cash[agent_idx] += current_price * 0.5
                        reward = 0.02
                
                # Calculate P&L reward
                if step > 0:
                    prev_price = data.iloc[start_idx + step - 1]['Close']
                    pnl = positions[agent_idx] * (current_price - prev_price)
                    reward += pnl / 1000.0  # Scale reward
                
                # Risk penalty
                if abs(positions[agent_idx]) > 5.0:
                    reward -= 0.1
                
                rewards.append(reward)
            
            # Store transition
            dones = [False] * self.n_agents
            self.store_transition(states, actions, rewards, log_probs, values, dones)
            
            episode_reward += sum(rewards)
            episode_step += 1
            
            # Update networks periodically
            if len(self.states[0]) >= 128:
                self.update_networks()
        
        # Final update
        if len(self.states[0]) > 0:
            self.update_networks()
        
        # Update training stats
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_step)
        self.training_stats['episodes'] += 1
        self.training_stats['total_steps'] += episode_step
        
        if episode_reward > self.training_stats['best_reward']:
            self.training_stats['best_reward'] = episode_reward
        
        # Calculate average reward over last 100 episodes
        if len(self.episode_rewards) >= 100:
            self.training_stats['avg_reward_100'] = np.mean(self.episode_rewards[-100:])
        
        return episode_reward, episode_step
    
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
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint"""
        checkpoint = {
            'training_stats': self.training_stats,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
            'actor_optimizers': [opt.state_dict() for opt in self.actor_optimizers],
            'critic_optimizers': [opt.state_dict() for opt in self.critic_optimizers],
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.training_stats = checkpoint['training_stats']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        self.actor_losses = checkpoint['actor_losses']
        self.critic_losses = checkpoint['critic_losses']
        
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(checkpoint['actors'][i])
        
        for i, critic in enumerate(self.critics):
            critic.load_state_dict(checkpoint['critics'][i])
            
        for i, opt in enumerate(self.actor_optimizers):
            opt.load_state_dict(checkpoint['actor_optimizers'][i])
            
        for i, opt in enumerate(self.critic_optimizers):
            opt.load_state_dict(checkpoint['critic_optimizers'][i])
        
        logger.info(f"Checkpoint loaded from {filepath}")
    
    def plot_training_progress(self, save_path: str = None):
        """Plot training progress"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        ax1.plot(self.episode_rewards)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True)
        
        # Moving average
        if len(self.episode_rewards) >= 50:
            moving_avg = np.convolve(self.episode_rewards, np.ones(50)/50, mode='valid')
            ax1.plot(range(49, len(self.episode_rewards)), moving_avg, 'r-', label='MA50')
            ax1.legend()
        
        # Episode lengths
        ax2.plot(self.episode_lengths)
        ax2.set_title('Episode Lengths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.grid(True)
        
        # Actor losses
        if self.actor_losses:
            ax3.plot(self.actor_losses)
            ax3.set_title('Actor Losses')
            ax3.set_xlabel('Update')
            ax3.set_ylabel('Loss')
            ax3.grid(True)
        
        # Critic losses
        if self.critic_losses:
            ax4.plot(self.critic_losses)
            ax4.set_title('Critic Losses')
            ax4.set_xlabel('Update')
            ax4.set_ylabel('Loss')
            ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_training_stats(self) -> Dict:
        """Get current training statistics"""
        return {
            'episodes': self.training_stats['episodes'],
            'total_steps': self.training_stats['total_steps'],
            'best_reward': self.training_stats['best_reward'],
            'avg_reward_100': self.training_stats['avg_reward_100'],
            'latest_reward': self.episode_rewards[-1] if self.episode_rewards else 0.0,
            'actor_loss': self.actor_losses[-1] if self.actor_losses else 0.0,
            'critic_loss': self.critic_losses[-1] if self.critic_losses else 0.0
        }