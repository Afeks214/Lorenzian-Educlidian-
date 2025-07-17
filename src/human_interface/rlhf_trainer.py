"""
Reinforcement Learning from Human Feedback (RLHF) Training System

This module implements the RLHF training loop that incorporates expert preferences
into the MARL agent training process to align AI decisions with human expertise.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import deque
import sqlite3
import json
import structlog
from pathlib import Path

from .feedback_api import ExpertChoice, DecisionPoint, TradingStrategy
from ..core.event_bus import EventBus, Event, EventType

logger = structlog.get_logger()


@dataclass
class PreferenceRecord:
    """A single preference record for training"""
    decision_id: str
    expert_id: str
    chosen_strategy: TradingStrategy
    rejected_strategy: TradingStrategy
    context_features: np.ndarray
    expert_confidence: float
    timestamp: datetime
    market_outcome: Optional[float] = None  # Actual PnL if available


@dataclass
class RLHFBatch:
    """Batch of preference data for training"""
    chosen_states: torch.Tensor
    chosen_actions: torch.Tensor
    rejected_states: torch.Tensor
    rejected_actions: torch.Tensor
    preferences: torch.Tensor  # 1 if chosen > rejected, 0 otherwise
    weights: torch.Tensor  # Expert confidence weights


class PreferenceDatabase:
    """Database for storing and retrieving expert preferences"""
    
    def __init__(self, db_path: str = "expert_preferences.db"):
        self.db_path = Path(db_path)
        self._initialize_database()
        logger.info("Preference database initialized", db_path=db_path)

    def _initialize_database(self):
        """Initialize SQLite database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Expert choices table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS expert_choices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id TEXT NOT NULL,
                    expert_id TEXT NOT NULL,
                    chosen_strategy_id TEXT NOT NULL,
                    chosen_strategy_data BLOB NOT NULL,
                    alternative_strategy_data BLOB,
                    context_features BLOB NOT NULL,
                    expert_confidence REAL NOT NULL,
                    reasoning TEXT,
                    market_view TEXT,
                    risk_assessment TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    market_outcome REAL,
                    validated BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Decision contexts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS decision_contexts (
                    decision_id TEXT PRIMARY KEY,
                    context_data BLOB NOT NULL,
                    complexity TEXT NOT NULL,
                    strategies_data BLOB NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Expert performance tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS expert_performance (
                    expert_id TEXT,
                    decision_id TEXT,
                    predicted_outcome REAL,
                    actual_outcome REAL,
                    timestamp DATETIME,
                    PRIMARY KEY (expert_id, decision_id)
                )
            """)
            
            conn.commit()

    def store_expert_choice(
        self, 
        expert_choice: ExpertChoice, 
        decision_point: DecisionPoint
    ) -> bool:
        """Store expert choice with full context"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Find chosen and alternative strategies
                chosen_strategy = None
                alternative_strategies = []
                
                for strategy in decision_point.strategies:
                    if strategy.strategy_id == expert_choice.chosen_strategy_id:
                        chosen_strategy = strategy
                    else:
                        alternative_strategies.append(strategy)
                
                if not chosen_strategy:
                    logger.error("Chosen strategy not found", decision_id=expert_choice.decision_id)
                    return False
                
                # Extract context features
                context_features = self._extract_context_features(decision_point.context)
                
                # Store choice
                cursor.execute("""
                    INSERT INTO expert_choices (
                        decision_id, expert_id, chosen_strategy_id, chosen_strategy_data,
                        alternative_strategy_data, context_features, expert_confidence,
                        reasoning, market_view, risk_assessment
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    expert_choice.decision_id,
                    expert_choice.expert_id,
                    expert_choice.chosen_strategy_id,
                    json.dumps(asdict(chosen_strategy)),
                    json.dumps([asdict(s) for s in alternative_strategies]),
                    json.dumps(context_features),
                    expert_choice.confidence,
                    expert_choice.reasoning,
                    expert_choice.market_view,
                    expert_choice.risk_assessment
                ))
                
                # Store decision context
                cursor.execute("""
                    INSERT OR REPLACE INTO decision_contexts (
                        decision_id, context_data, complexity, strategies_data
                    ) VALUES (?, ?, ?, ?)
                """, (
                    decision_point.decision_id,
                    json.dumps(asdict(decision_point.context)),
                    decision_point.complexity.value,
                    json.dumps([asdict(s) for s in decision_point.strategies])
                ))
                
                conn.commit()
                logger.info("Expert choice stored", decision_id=expert_choice.decision_id)
                return True
                
        except Exception as e:
            logger.error("Failed to store expert choice", error=str(e))
            return False

    def _extract_context_features(self, context) -> np.ndarray:
        """Extract numerical features from market context"""
        features = np.array([
            context.price,
            context.volatility,
            context.volume,
            context.trend_strength,
            context.support_level,
            context.resistance_level,
            1.0 if context.correlation_shock else 0.0,
            # Add time-based features
            datetime.now().hour / 24.0,  # Normalized hour
            datetime.now().weekday() / 6.0,  # Normalized weekday
        ])
        return features

    def get_preference_records(
        self, 
        expert_id: Optional[str] = None,
        limit: Optional[int] = None,
        min_confidence: float = 0.5
    ) -> List[PreferenceRecord]:
        """Retrieve preference records for training"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT decision_id, expert_id, chosen_strategy_data, 
                           alternative_strategy_data, context_features, 
                           expert_confidence, timestamp, market_outcome
                    FROM expert_choices 
                    WHERE expert_confidence >= ?
                """
                params = [min_confidence]
                
                if expert_id:
                    query += " AND expert_id = ?"
                    params.append(expert_id)
                
                query += " ORDER BY timestamp DESC"
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                records = []
                for row in rows:
                    decision_id, expert_id, chosen_data, alt_data, features, confidence, timestamp, outcome = row
                    
                    chosen_strategy = json.loads(chosen_data)
                    alternative_strategies = json.loads(alt_data) if alt_data else []
                    context_features = json.loads(features)
                    
                    # Create preference records for each alternative
                    for alt_strategy in alternative_strategies:
                        record = PreferenceRecord(
                            decision_id=decision_id,
                            expert_id=expert_id,
                            chosen_strategy=chosen_strategy,
                            rejected_strategy=alt_strategy,
                            context_features=context_features,
                            expert_confidence=confidence,
                            timestamp=datetime.fromisoformat(timestamp),
                            market_outcome=outcome
                        )
                        records.append(record)
                
                return records
                
        except Exception as e:
            logger.error("Failed to retrieve preference records", error=str(e))
            return []

    def update_market_outcome(self, decision_id: str, outcome: float) -> bool:
        """Update the market outcome for a decision"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE expert_choices 
                    SET market_outcome = ?, validated = TRUE
                    WHERE decision_id = ?
                """, (outcome, decision_id))
                conn.commit()
                return True
        except Exception as e:
            logger.error("Failed to update market outcome", error=str(e))
            return False


class RewardModel(nn.Module):
    """Neural network reward model for RLHF"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 4),
            nn.ReLU()
        )
        
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim // 2 + hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute reward score"""
        state_encoded = self.state_encoder(state)
        action_encoded = self.action_encoder(action)
        
        combined = torch.cat([state_encoded, action_encoded], dim=-1)
        reward = self.reward_head(combined)
        
        return reward


class RLHFTrainer:
    """RLHF trainer for aligning MARL agents with expert preferences"""
    
    def __init__(
        self, 
        event_bus: EventBus,
        preference_db: PreferenceDatabase,
        state_dim: int = 9,  # Context features dimension
        action_dim: int = 7,  # Strategy parameters dimension
        learning_rate: float = 1e-4,
        batch_size: int = 32
    ):
        self.event_bus = event_bus
        self.preference_db = preference_db
        self.batch_size = batch_size
        
        # Initialize reward model
        self.reward_model = RewardModel(state_dim, action_dim)
        self.optimizer = optim.Adam(self.reward_model.parameters(), lr=learning_rate)
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        # Training metrics
        self.training_history = []
        self.validation_accuracy = deque(maxlen=100)
        
        # Model checkpointing
        self.best_accuracy = 0.0
        self.checkpoint_dir = Path("rlhf_checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Event subscriptions
        self._setup_event_listeners()
        
        logger.info("RLHF Trainer initialized")

    def _setup_event_listeners(self):
        """Setup event listeners for real-time preference learning"""
        self.event_bus.subscribe(EventType.STRATEGIC_DECISION, self._handle_expert_feedback)

    def _handle_expert_feedback(self, event: Event):
        """Handle new expert feedback events"""
        if event.payload.get("type") == "expert_feedback":
            expert_choice = ExpertChoice(**event.payload["expert_choice"])
            decision_context = event.payload["decision_context"]
            
            # Store in database
            self.preference_db.store_expert_choice(expert_choice, decision_context)
            
            # Trigger incremental training if enough new data
            if self._should_trigger_training():
                self._schedule_training()

    def _should_trigger_training(self) -> bool:
        """Determine if training should be triggered"""
        # Train after every 10 new preferences or daily
        recent_count = len(self.preference_db.get_preference_records(limit=10))
        return recent_count >= 10

    def _schedule_training(self):
        """Schedule asynchronous training"""
        # In production, this would use a job queue
        logger.info("Scheduling RLHF training update")
        self.train_reward_model()

    def _strategy_to_action_vector(self, strategy: TradingStrategy) -> np.ndarray:
        """Convert trading strategy to action vector"""
        return np.array([
            strategy.position_size / 1000.0,  # Normalized position size
            strategy.stop_loss / strategy.entry_price - 1.0,  # Stop loss %
            strategy.take_profit / strategy.entry_price - 1.0,  # Take profit %
            strategy.time_horizon / 60.0,  # Normalized time horizon
            strategy.risk_reward_ratio / 5.0,  # Normalized RR ratio
            strategy.confidence_score,
            float(strategy.strategy_type.value == "aggressive")  # Strategy type encoding
        ])

    def _prepare_training_batch(self, preference_records: List[PreferenceRecord]) -> RLHFBatch:
        """Prepare batch for training"""
        chosen_states = []
        chosen_actions = []
        rejected_states = []
        rejected_actions = []
        preferences = []
        weights = []
        
        for record in preference_records:
            # State (context features)
            state = torch.tensor(record.context_features, dtype=torch.float32)
            
            # Actions (strategy vectors)
            chosen_action = torch.tensor(
                self._strategy_to_action_vector(record.chosen_strategy), 
                dtype=torch.float32
            )
            rejected_action = torch.tensor(
                self._strategy_to_action_vector(record.rejected_strategy), 
                dtype=torch.float32
            )
            
            chosen_states.append(state)
            chosen_actions.append(chosen_action)
            rejected_states.append(state)  # Same state for both
            rejected_actions.append(rejected_action)
            preferences.append(1.0)  # Chosen > rejected
            weights.append(record.expert_confidence)
        
        return RLHFBatch(
            chosen_states=torch.stack(chosen_states),
            chosen_actions=torch.stack(chosen_actions),
            rejected_states=torch.stack(rejected_states),
            rejected_actions=torch.stack(rejected_actions),
            preferences=torch.tensor(preferences, dtype=torch.float32),
            weights=torch.tensor(weights, dtype=torch.float32)
        )

    def train_reward_model(self, epochs: int = 10) -> Dict[str, float]:
        """Train the reward model on expert preferences"""
        logger.info("Starting RLHF reward model training")
        
        # Get training data
        preference_records = self.preference_db.get_preference_records(min_confidence=0.5)
        
        if len(preference_records) < 10:
            logger.warning("Insufficient preference data for training", count=len(preference_records))
            return {"error": "insufficient_data"}
        
        # Split into train/validation
        split_idx = int(len(preference_records) * 0.8)
        train_records = preference_records[:split_idx]
        val_records = preference_records[split_idx:]
        
        train_losses = []
        
        for epoch in range(epochs):
            self.reward_model.train()
            epoch_losses = []
            
            # Process in batches
            for i in range(0, len(train_records), self.batch_size):
                batch_records = train_records[i:i + self.batch_size]
                batch = self._prepare_training_batch(batch_records)
                
                # Forward pass
                chosen_rewards = self.reward_model(batch.chosen_states, batch.chosen_actions)
                rejected_rewards = self.reward_model(batch.rejected_states, batch.rejected_actions)
                
                # Preference loss: P(chosen > rejected)
                reward_diff = chosen_rewards - rejected_rewards
                probs = torch.sigmoid(reward_diff.squeeze())
                
                # Weighted loss based on expert confidence
                loss = self.loss_fn(reward_diff.squeeze(), batch.preferences)
                weighted_loss = (loss * batch.weights).mean()
                
                # Backward pass
                self.optimizer.zero_grad()
                weighted_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_losses.append(weighted_loss.item())
            
            avg_loss = np.mean(epoch_losses)
            train_losses.append(avg_loss)
            
            # Validation
            if val_records:
                val_accuracy = self._validate_model(val_records)
                self.validation_accuracy.append(val_accuracy)
                
                # Save best model
                if val_accuracy > self.best_accuracy:
                    self.best_accuracy = val_accuracy
                    self._save_checkpoint("best_model.pt")
                
                logger.info(
                    "RLHF training epoch completed",
                    epoch=epoch + 1,
                    loss=avg_loss,
                    val_accuracy=val_accuracy
                )
            else:
                logger.info("RLHF training epoch completed", epoch=epoch + 1, loss=avg_loss)
        
        # Save training history
        training_metrics = {
            "avg_loss": np.mean(train_losses),
            "final_loss": train_losses[-1],
            "val_accuracy": self.validation_accuracy[-1] if self.validation_accuracy else 0.0,
            "training_samples": len(train_records),
            "timestamp": datetime.now().isoformat()
        }
        
        self.training_history.append(training_metrics)
        
        logger.info("RLHF training completed", metrics=training_metrics)
        return training_metrics

    def _validate_model(self, val_records: List[PreferenceRecord]) -> float:
        """Validate reward model accuracy"""
        self.reward_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_batch = self._prepare_training_batch(val_records)
            
            chosen_rewards = self.reward_model(val_batch.chosen_states, val_batch.chosen_actions)
            rejected_rewards = self.reward_model(val_batch.rejected_states, val_batch.rejected_actions)
            
            # Check if chosen strategies have higher rewards
            predictions = (chosen_rewards > rejected_rewards).float()
            correct = (predictions.squeeze() == val_batch.preferences).sum().item()
            total = len(val_records)
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        torch.save({
            'model_state_dict': self.reward_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.best_accuracy,
            'training_history': self.training_history
        }, checkpoint_path)

    def load_checkpoint(self, filename: str) -> bool:
        """Load model checkpoint"""
        try:
            checkpoint_path = self.checkpoint_dir / filename
            checkpoint = torch.load(checkpoint_path)
            
            self.reward_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_accuracy = checkpoint['best_accuracy']
            self.training_history = checkpoint['training_history']
            
            logger.info("RLHF checkpoint loaded", file=filename)
            return True
            
        except Exception as e:
            logger.error("Failed to load checkpoint", error=str(e))
            return False

    def get_strategy_reward(self, context_features: np.ndarray, strategy: TradingStrategy) -> float:
        """Get reward score for a strategy in given context"""
        self.reward_model.eval()
        
        with torch.no_grad():
            state = torch.tensor(context_features, dtype=torch.float32).unsqueeze(0)
            action = torch.tensor(
                self._strategy_to_action_vector(strategy), 
                dtype=torch.float32
            ).unsqueeze(0)
            
            reward = self.reward_model(state, action)
            return reward.item()

    def rank_strategies(
        self, 
        context_features: np.ndarray, 
        strategies: List[TradingStrategy]
    ) -> List[Tuple[TradingStrategy, float]]:
        """Rank strategies by human-preference-aligned reward"""
        strategy_rewards = []
        
        for strategy in strategies:
            reward = self.get_strategy_reward(context_features, strategy)
            strategy_rewards.append((strategy, reward))
        
        # Sort by reward (highest first)
        strategy_rewards.sort(key=lambda x: x[1], reverse=True)
        
        return strategy_rewards

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status and metrics"""
        return {
            "total_preferences": len(self.preference_db.get_preference_records()),
            "validation_accuracy": self.validation_accuracy[-1] if self.validation_accuracy else 0.0,
            "best_accuracy": self.best_accuracy,
            "training_history_length": len(self.training_history),
            "last_training": self.training_history[-1] if self.training_history else None,
            "model_ready": self.best_accuracy > 0.6  # Model is considered ready above 60% accuracy
        }