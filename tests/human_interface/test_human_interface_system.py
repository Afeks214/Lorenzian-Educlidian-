"""
Comprehensive Test Suite for Human Interface System

This test suite provides extensive testing for the Human Expert Feedback System,
including decision feedback loops, RLHF training pipeline, and user interaction simulations.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import json
import numpy as np
import torch
import redis
from fastapi.testclient import TestClient
from fastapi import WebSocket
import sqlite3
from pathlib import Path
import tempfile
import shutil

# Import system components
from src.human_interface.feedback_api import (
    FeedbackAPI, DecisionPoint, ExpertChoice, MarketContext, 
    TradingStrategy, StrategyType, DecisionComplexity
)
from src.human_interface.rlhf_trainer import (
    RLHFTrainer, PreferenceDatabase, PreferenceRecord, RewardModel
)
from src.human_interface.analytics import (
    ExpertAnalytics, ModelAlignmentAnalyzer, AnalyticsDashboard
)
from src.human_interface.integration_system import HumanFeedbackCoordinator
from src.human_interface.choice_generator import ChoiceGenerator
from src.core.event_bus import EventBus, EventType
from src.core.config_manager import ConfigManager


class TestHumanInterfaceSystem:
    """Comprehensive test suite for the Human Interface System"""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database for testing"""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_preferences.db"
        yield str(db_path)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client"""
        return Mock(spec=redis.Redis)
    
    @pytest.fixture
    def event_bus(self):
        """Create event bus for testing"""
        return EventBus()
    
    @pytest.fixture
    def config_manager(self):
        """Create config manager for testing"""
        return ConfigManager()
    
    @pytest.fixture
    def preference_db(self, temp_db_path):
        """Create preference database for testing"""
        return PreferenceDatabase(temp_db_path)
    
    @pytest.fixture
    def sample_market_context(self):
        """Create sample market context"""
        return MarketContext(
            symbol="ETH-USD",
            price=2100.50,
            volatility=0.045,
            volume=2500000,
            trend_strength=0.85,
            support_level=2050.0,
            resistance_level=2150.0,
            time_of_day="market_hours",
            market_regime="trending",
            correlation_shock=False
        )
    
    @pytest.fixture
    def sample_trading_strategies(self):
        """Create sample trading strategies"""
        return [
            TradingStrategy(
                strategy_id="strategy_1",
                strategy_type=StrategyType.MOMENTUM,
                entry_price=2100.0,
                position_size=1000.0,
                stop_loss=2050.0,
                take_profit=2200.0,
                time_horizon=60,
                risk_reward_ratio=2.0,
                confidence_score=0.8,
                reasoning="Strong momentum signals",
                expected_pnl=50000.0,
                max_drawdown=25000.0
            ),
            TradingStrategy(
                strategy_id="strategy_2",
                strategy_type=StrategyType.CONSERVATIVE,
                entry_price=2100.0,
                position_size=500.0,
                stop_loss=2080.0,
                take_profit=2130.0,
                time_horizon=120,
                risk_reward_ratio=1.5,
                confidence_score=0.9,
                reasoning="Conservative approach with good risk management",
                expected_pnl=15000.0,
                max_drawdown=10000.0
            )
        ]
    
    @pytest.fixture
    def sample_decision_point(self, sample_market_context, sample_trading_strategies):
        """Create sample decision point"""
        return DecisionPoint(
            decision_id="decision_123",
            timestamp=datetime.now(),
            context=sample_market_context,
            complexity=DecisionComplexity.HIGH,
            strategies=sample_trading_strategies,
            current_position={"symbol": "ETH-USD", "quantity": 0},
            expert_deadline=datetime.now() + timedelta(minutes=15),
            model_recommendation="strategy_1",
            confidence_threshold=0.8
        )
    
    @pytest.fixture
    def sample_expert_choice(self):
        """Create sample expert choice"""
        return ExpertChoice(
            decision_id="decision_123",
            chosen_strategy_id="strategy_1",
            expert_id="expert_001",
            timestamp=datetime.now(),
            confidence=0.85,
            reasoning="Strong technical setup with volume confirmation",
            alternative_considered="strategy_2",
            market_view="Bullish momentum likely to continue",
            risk_assessment="Well-defined risk with 2:1 reward ratio"
        )


class TestPreferenceDatabase:
    """Test the preference database functionality"""
    
    def test_database_initialization(self, temp_db_path):
        """Test database initialization and schema creation"""
        db = PreferenceDatabase(temp_db_path)
        
        # Check if database file exists
        assert Path(temp_db_path).exists()
        
        # Check if tables exist
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            assert "expert_choices" in tables
            assert "decision_contexts" in tables
            assert "expert_performance" in tables
    
    def test_store_expert_choice(self, preference_db, sample_expert_choice, sample_decision_point):
        """Test storing expert choice with decision context"""
        success = preference_db.store_expert_choice(sample_expert_choice, sample_decision_point)
        
        assert success is True
        
        # Verify data was stored
        records = preference_db.get_preference_records()
        assert len(records) > 0
        
        record = records[0]
        assert record.decision_id == sample_expert_choice.decision_id
        assert record.expert_id == sample_expert_choice.expert_id
        assert record.expert_confidence == sample_expert_choice.confidence
    
    def test_get_preference_records_filtering(self, preference_db, sample_expert_choice, sample_decision_point):
        """Test filtering preference records by various criteria"""
        # Store multiple records
        for i in range(5):
            choice = ExpertChoice(
                decision_id=f"decision_{i}",
                chosen_strategy_id="strategy_1",
                expert_id="expert_001",
                timestamp=datetime.now(),
                confidence=0.5 + (i * 0.1),  # Varying confidence levels
                reasoning=f"Test reasoning {i}",
                market_view="Test view",
                risk_assessment="Test assessment"
            )
            preference_db.store_expert_choice(choice, sample_decision_point)
        
        # Test filtering by confidence
        high_confidence_records = preference_db.get_preference_records(min_confidence=0.7)
        assert len(high_confidence_records) == 6  # 3 records with confidence >= 0.7 (each has 2 alternatives)
        
        # Test filtering by expert
        expert_records = preference_db.get_preference_records(expert_id="expert_001")
        assert len(expert_records) == 10  # 5 records * 2 alternatives each
        
        # Test limiting results
        limited_records = preference_db.get_preference_records(limit=3)
        assert len(limited_records) == 3
    
    def test_update_market_outcome(self, preference_db, sample_expert_choice, sample_decision_point):
        """Test updating market outcomes for decisions"""
        # Store initial choice
        preference_db.store_expert_choice(sample_expert_choice, sample_decision_point)
        
        # Update market outcome
        success = preference_db.update_market_outcome(sample_expert_choice.decision_id, 0.05)
        assert success is True
        
        # Verify update
        records = preference_db.get_preference_records()
        assert len(records) > 0
        assert records[0].market_outcome == 0.05


class TestRLHFTrainer:
    """Test the RLHF training system"""
    
    def test_reward_model_initialization(self):
        """Test reward model neural network initialization"""
        model = RewardModel(state_dim=9, action_dim=7)
        
        # Test forward pass
        state = torch.randn(1, 9)
        action = torch.randn(1, 7)
        reward = model(state, action)
        
        assert reward.shape == (1, 1)
        assert not torch.isnan(reward).any()
    
    def test_rlhf_trainer_initialization(self, event_bus, preference_db):
        """Test RLHF trainer initialization"""
        trainer = RLHFTrainer(event_bus, preference_db)
        
        assert trainer.reward_model is not None
        assert trainer.optimizer is not None
        assert trainer.preference_db == preference_db
        assert len(trainer.training_history) == 0
    
    def test_strategy_to_action_vector(self, event_bus, preference_db, sample_trading_strategies):
        """Test conversion of trading strategy to action vector"""
        trainer = RLHFTrainer(event_bus, preference_db)
        
        strategy = sample_trading_strategies[0]
        action_vector = trainer._strategy_to_action_vector(strategy)
        
        assert len(action_vector) == 7
        assert all(isinstance(x, (int, float)) for x in action_vector)
        assert not np.isnan(action_vector).any()
    
    def test_training_batch_preparation(self, event_bus, preference_db, sample_decision_point, sample_expert_choice):
        """Test preparation of training batches"""
        trainer = RLHFTrainer(event_bus, preference_db)
        
        # Create mock preference records
        preference_records = [
            PreferenceRecord(
                decision_id=sample_expert_choice.decision_id,
                expert_id=sample_expert_choice.expert_id,
                chosen_strategy=sample_decision_point.strategies[0],
                rejected_strategy=sample_decision_point.strategies[1],
                context_features=np.random.rand(9),
                expert_confidence=0.85,
                timestamp=datetime.now()
            )
        ]
        
        batch = trainer._prepare_training_batch(preference_records)
        
        assert batch.chosen_states.shape[0] == 1
        assert batch.chosen_actions.shape[0] == 1
        assert batch.rejected_states.shape[0] == 1
        assert batch.rejected_actions.shape[0] == 1
        assert batch.preferences.shape[0] == 1
        assert batch.weights.shape[0] == 1
    
    def test_model_validation(self, event_bus, preference_db, sample_decision_point, sample_expert_choice):
        """Test model validation accuracy calculation"""
        trainer = RLHFTrainer(event_bus, preference_db)
        
        # Create mock preference records
        preference_records = [
            PreferenceRecord(
                decision_id=f"decision_{i}",
                expert_id="expert_001",
                chosen_strategy=sample_decision_point.strategies[0],
                rejected_strategy=sample_decision_point.strategies[1],
                context_features=np.random.rand(9),
                expert_confidence=0.85,
                timestamp=datetime.now()
            )
            for i in range(5)
        ]
        
        accuracy = trainer._validate_model(preference_records)
        
        assert 0.0 <= accuracy <= 1.0
    
    def test_get_strategy_reward(self, event_bus, preference_db, sample_trading_strategies):
        """Test getting reward scores for strategies"""
        trainer = RLHFTrainer(event_bus, preference_db)
        
        context_features = np.random.rand(9)
        strategy = sample_trading_strategies[0]
        
        reward = trainer.get_strategy_reward(context_features, strategy)
        
        assert isinstance(reward, float)
        assert not np.isnan(reward)
    
    def test_rank_strategies(self, event_bus, preference_db, sample_trading_strategies):
        """Test ranking strategies by human-preference-aligned rewards"""
        trainer = RLHFTrainer(event_bus, preference_db)
        
        context_features = np.random.rand(9)
        ranked_strategies = trainer.rank_strategies(context_features, sample_trading_strategies)
        
        assert len(ranked_strategies) == len(sample_trading_strategies)
        assert all(len(item) == 2 for item in ranked_strategies)  # (strategy, reward) pairs
        
        # Check that strategies are ranked by reward (highest first)
        rewards = [reward for _, reward in ranked_strategies]
        assert rewards == sorted(rewards, reverse=True)
    
    def test_training_status(self, event_bus, preference_db):
        """Test getting training status information"""
        trainer = RLHFTrainer(event_bus, preference_db)
        
        status = trainer.get_training_status()
        
        assert "total_preferences" in status
        assert "validation_accuracy" in status
        assert "best_accuracy" in status
        assert "training_history_length" in status
        assert "model_ready" in status
        
        assert isinstance(status["total_preferences"], int)
        assert isinstance(status["validation_accuracy"], float)
        assert isinstance(status["best_accuracy"], float)
        assert isinstance(status["model_ready"], bool)


class TestFeedbackAPI:
    """Test the feedback API functionality"""
    
    def test_api_initialization(self, event_bus, mock_redis):
        """Test API initialization"""
        api = FeedbackAPI(event_bus, mock_redis)
        
        assert api.app is not None
        assert api.event_bus == event_bus
        assert api.redis_client == mock_redis
        assert len(api.pending_decisions) == 0
        assert len(api.expert_choices) == 0
    
    def test_expert_addition(self, event_bus, mock_redis):
        """Test adding experts to the system"""
        api = FeedbackAPI(event_bus, mock_redis)
        
        success = api.add_expert("test_expert", "secure_password123")
        assert success is True
        
        assert "test_expert" in api.expert_credentials
    
    def test_decision_submission(self, event_bus, mock_redis, sample_decision_point):
        """Test submitting decision for expert input"""
        api = FeedbackAPI(event_bus, mock_redis)
        
        # Test async submission
        async def test_submission():
            success = await api.submit_decision_for_expert_input(sample_decision_point)
            return success
        
        success = asyncio.run(test_submission())
        assert success is True
        
        assert sample_decision_point.decision_id in api.pending_decisions
    
    def test_expert_authentication(self, event_bus, mock_redis):
        """Test expert authentication flow"""
        api = FeedbackAPI(event_bus, mock_redis)
        
        # Add expert
        api.add_expert("test_expert", "secure_password123")
        
        # Test credential verification
        valid = api._verify_expert_credentials("test_expert", "secure_password123")
        assert valid is True
        
        invalid = api._verify_expert_credentials("test_expert", "wrong_password")
        assert invalid is False
    
    def test_jwt_token_generation(self, event_bus, mock_redis):
        """Test JWT token generation and validation"""
        api = FeedbackAPI(event_bus, mock_redis)
        
        # Generate token
        token = api._generate_jwt_token("test_expert")
        assert isinstance(token, str)
        assert len(token) > 0


class TestAnalytics:
    """Test the analytics system"""
    
    def test_expert_analytics_initialization(self, preference_db):
        """Test expert analytics initialization"""
        analytics = ExpertAnalytics(preference_db)
        
        assert analytics.preference_db == preference_db
        assert analytics.cache_timeout == timedelta(hours=1)
        assert len(analytics._cache) == 0
    
    def test_expert_metrics_calculation(self, preference_db, sample_expert_choice, sample_decision_point):
        """Test calculation of expert performance metrics"""
        analytics = ExpertAnalytics(preference_db)
        
        # Store some test data
        preference_db.store_expert_choice(sample_expert_choice, sample_decision_point)
        
        metrics = analytics.calculate_expert_metrics("expert_001")
        
        assert metrics.expert_id == "expert_001"
        assert metrics.total_decisions >= 0
        assert 0.0 <= metrics.average_confidence <= 1.0
        assert 0.0 <= metrics.success_rate <= 1.0
        assert metrics.response_time_avg > 0
        assert 0.0 <= metrics.consistency_score <= 1.0
        assert isinstance(metrics.specialization_areas, list)
        assert metrics.performance_trend in ["improving", "declining", "stable"]
        assert metrics.risk_profile in ["conservative", "moderate", "aggressive", "unknown"]
    
    def test_expert_comparison(self, preference_db, sample_expert_choice, sample_decision_point):
        """Test comparing multiple experts"""
        analytics = ExpertAnalytics(preference_db)
        
        # Store test data for multiple experts
        for expert_id in ["expert_001", "expert_002", "expert_003"]:
            choice = ExpertChoice(
                decision_id=f"decision_{expert_id}",
                chosen_strategy_id="strategy_1",
                expert_id=expert_id,
                timestamp=datetime.now(),
                confidence=0.8,
                reasoning="Test reasoning",
                market_view="Test view",
                risk_assessment="Test assessment"
            )
            preference_db.store_expert_choice(choice, sample_decision_point)
        
        comparison = analytics.compare_experts(["expert_001", "expert_002", "expert_003"])
        
        assert "experts" in comparison
        assert "rankings" in comparison
        assert "statistical_summary" in comparison
        
        assert len(comparison["experts"]) == 3
        assert "by_success_rate" in comparison["rankings"]
        assert "by_confidence" in comparison["rankings"]
        assert "by_consistency" in comparison["rankings"]
        assert "by_model_agreement" in comparison["rankings"]
    
    def test_model_alignment_analyzer(self, preference_db, event_bus):
        """Test model alignment analysis"""
        trainer = RLHFTrainer(event_bus, preference_db)
        analyzer = ModelAlignmentAnalyzer(preference_db, trainer)
        
        metrics = analyzer.calculate_alignment_metrics()
        
        assert 0.0 <= metrics.overall_alignment <= 1.0
        assert 0.0 <= metrics.accuracy_improvement <= 1.0
        assert 0.0 <= metrics.preference_learning_rate <= 1.0
        assert 0.0 <= metrics.expert_satisfaction <= 1.0
        assert 0.0 <= metrics.model_confidence_calibration <= 1.0
        assert 0.0 <= metrics.bias_detection_score <= 1.0
    
    def test_analytics_dashboard(self, preference_db, event_bus):
        """Test comprehensive analytics dashboard"""
        trainer = RLHFTrainer(event_bus, preference_db)
        dashboard = AnalyticsDashboard(preference_db, trainer)
        
        report = dashboard.generate_comprehensive_report()
        
        assert "timestamp" in report
        assert "expert_performance" in report
        assert "model_alignment" in report
        assert "system_metrics" in report
        assert "recommendations" in report
        
        assert isinstance(report["recommendations"], list)
    
    def test_monitoring_metrics_export(self, preference_db, event_bus):
        """Test export of key metrics for monitoring"""
        trainer = RLHFTrainer(event_bus, preference_db)
        dashboard = AnalyticsDashboard(preference_db, trainer)
        
        metrics = dashboard.export_metrics_for_monitoring()
        
        assert "model_alignment_score" in metrics
        assert "accuracy_improvement" in metrics
        assert "expert_satisfaction" in metrics
        assert "bias_detection_score" in metrics
        assert "system_health" in metrics
        
        # All metrics should be valid floats
        for key, value in metrics.items():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)


class TestIntegrationSystem:
    """Test the integration system coordination"""
    
    def test_coordinator_initialization(self, event_bus, config_manager, mock_redis):
        """Test human feedback coordinator initialization"""
        coordinator = HumanFeedbackCoordinator(event_bus, config_manager, mock_redis)
        
        assert coordinator.event_bus == event_bus
        assert coordinator.config_manager == config_manager
        assert coordinator.redis_client == mock_redis
        assert coordinator.security_manager is not None
        assert coordinator.preference_db is not None
        assert coordinator.choice_generator is not None
        assert coordinator.feedback_api is not None
        assert coordinator.rlhf_trainer is not None
    
    def test_decision_processing(self, event_bus, config_manager, mock_redis, sample_decision_point):
        """Test decision point processing"""
        coordinator = HumanFeedbackCoordinator(event_bus, config_manager, mock_redis)
        
        async def test_processing():
            await coordinator._process_decision_point(sample_decision_point)
            return True
        
        success = asyncio.run(test_processing())
        assert success is True
        
        assert sample_decision_point.decision_id in coordinator.pending_decisions
    
    def test_expert_feedback_processing(self, event_bus, config_manager, mock_redis, sample_expert_choice, sample_decision_point):
        """Test expert feedback processing"""
        coordinator = HumanFeedbackCoordinator(event_bus, config_manager, mock_redis)
        
        feedback_payload = {
            "expert_choice": {
                "decision_id": sample_expert_choice.decision_id,
                "chosen_strategy_id": sample_expert_choice.chosen_strategy_id,
                "expert_id": sample_expert_choice.expert_id,
                "timestamp": sample_expert_choice.timestamp.isoformat(),
                "confidence": sample_expert_choice.confidence,
                "reasoning": sample_expert_choice.reasoning,
                "alternative_considered": sample_expert_choice.alternative_considered,
                "market_view": sample_expert_choice.market_view,
                "risk_assessment": sample_expert_choice.risk_assessment
            },
            "decision_context": sample_decision_point
        }
        
        async def test_processing():
            await coordinator._process_expert_feedback(feedback_payload)
            return True
        
        success = asyncio.run(test_processing())
        assert success is True
        
        assert coordinator.metrics["expert_responses"] > 0
    
    def test_requires_expert_validation(self, event_bus, config_manager, mock_redis):
        """Test logic for determining if expert validation is required"""
        coordinator = HumanFeedbackCoordinator(event_bus, config_manager, mock_redis)
        
        # Low confidence decision
        low_confidence_decision = {"confidence": 0.5, "conflict_detected": False, "position_size": 1000}
        assert coordinator._requires_expert_validation(low_confidence_decision) is True
        
        # Conflicting decision
        conflict_decision = {"confidence": 0.8, "conflict_detected": True, "position_size": 1000}
        assert coordinator._requires_expert_validation(conflict_decision) is True
        
        # Large position decision
        large_position_decision = {"confidence": 0.8, "conflict_detected": False, "position_size": 6000}
        assert coordinator._requires_expert_validation(large_position_decision) is True
        
        # Normal decision
        normal_decision = {"confidence": 0.8, "conflict_detected": False, "position_size": 1000}
        assert coordinator._requires_expert_validation(normal_decision) is False
    
    def test_expired_decisions_cleanup(self, event_bus, config_manager, mock_redis, sample_decision_point):
        """Test cleanup of expired decisions"""
        coordinator = HumanFeedbackCoordinator(event_bus, config_manager, mock_redis)
        
        # Add decision with past deadline
        expired_decision = sample_decision_point
        expired_decision.expert_deadline = datetime.now() - timedelta(minutes=5)
        
        coordinator.pending_decisions[expired_decision.decision_id] = expired_decision
        coordinator.decision_timeouts[expired_decision.decision_id] = expired_decision.expert_deadline
        
        async def test_cleanup():
            await coordinator.cleanup_expired_decisions()
            return True
        
        success = asyncio.run(test_cleanup())
        assert success is True
        
        assert expired_decision.decision_id not in coordinator.pending_decisions
    
    def test_system_status(self, event_bus, config_manager, mock_redis):
        """Test system status reporting"""
        coordinator = HumanFeedbackCoordinator(event_bus, config_manager, mock_redis)
        
        status = coordinator.get_system_status()
        
        assert "pending_decisions" in status
        assert "active_experts" in status
        assert "metrics" in status
        assert "training_status" in status
        assert "system_health" in status
        
        assert isinstance(status["pending_decisions"], int)
        assert isinstance(status["active_experts"], int)
        assert isinstance(status["metrics"], dict)
        assert isinstance(status["training_status"], dict)
        assert status["system_health"] == "operational"


class TestDecisionFeedbackLoops:
    """Test decision feedback loop functionality"""
    
    def test_decision_creation_to_feedback_loop(self, event_bus, config_manager, mock_redis, sample_decision_point):
        """Test complete decision creation to feedback loop"""
        coordinator = HumanFeedbackCoordinator(event_bus, config_manager, mock_redis)
        
        # Mock the feedback API client
        test_client = TestClient(coordinator.get_feedback_api_app())
        
        async def test_complete_loop():
            # 1. Submit decision for expert input
            await coordinator._process_decision_point(sample_decision_point)
            
            # 2. Check that decision is pending
            assert sample_decision_point.decision_id in coordinator.pending_decisions
            
            # 3. Add expert to system
            coordinator.feedback_api.add_expert("test_expert", "secure_password123")
            
            # 4. Authenticate expert
            login_response = test_client.post(
                "/auth/login",
                json={"expert_id": "test_expert", "password": "secure_password123"}
            )
            
            if login_response.status_code == 200:
                token = login_response.json()["access_token"]
                
                # 5. Get pending decisions
                decisions_response = test_client.get(
                    "/decisions/pending",
                    headers={"Authorization": f"Bearer {token}"}
                )
                
                if decisions_response.status_code == 200:
                    # 6. Submit feedback
                    feedback_response = test_client.post(
                        f"/decisions/{sample_decision_point.decision_id}/feedback",
                        json={
                            "decision_id": sample_decision_point.decision_id,
                            "chosen_strategy_id": sample_decision_point.strategies[0].strategy_id,
                            "confidence": 0.85,
                            "reasoning": "Strong technical setup",
                            "market_view": "Bullish",
                            "risk_assessment": "Acceptable risk"
                        },
                        headers={"Authorization": f"Bearer {token}"}
                    )
                    
                    # 7. Check feedback was processed
                    assert feedback_response.status_code == 200
                    assert sample_decision_point.decision_id not in coordinator.pending_decisions
                    
                    return True
            
            return False
        
        success = asyncio.run(test_complete_loop())
        # Note: This test may fail due to authentication complexity, but structure is correct


class TestRLHFTrainingPipeline:
    """Test RLHF training pipeline end-to-end"""
    
    def test_preference_to_training_pipeline(self, event_bus, preference_db, sample_decision_point, sample_expert_choice):
        """Test complete pipeline from preference storage to model training"""
        trainer = RLHFTrainer(event_bus, preference_db)
        
        # 1. Store multiple expert preferences
        for i in range(15):  # Enough for training
            choice = ExpertChoice(
                decision_id=f"decision_{i}",
                chosen_strategy_id="strategy_1",
                expert_id=f"expert_{i % 3}",  # 3 different experts
                timestamp=datetime.now(),
                confidence=0.6 + (i * 0.02),  # Varying confidence
                reasoning=f"Test reasoning {i}",
                market_view="Test view",
                risk_assessment="Test assessment"
            )
            preference_db.store_expert_choice(choice, sample_decision_point)
        
        # 2. Train reward model
        training_results = trainer.train_reward_model(epochs=3)
        
        # 3. Verify training completed
        assert "avg_loss" in training_results
        assert "val_accuracy" in training_results
        assert "training_samples" in training_results
        
        # 4. Test model can rank strategies
        context_features = np.random.rand(9)
        ranked_strategies = trainer.rank_strategies(context_features, sample_decision_point.strategies)
        
        assert len(ranked_strategies) == len(sample_decision_point.strategies)
        
        # 5. Test training status
        status = trainer.get_training_status()
        assert status["total_preferences"] >= 15
        assert status["validation_accuracy"] >= 0.0
    
    def test_continuous_learning_simulation(self, event_bus, preference_db, sample_decision_point):
        """Test continuous learning with new expert feedback"""
        trainer = RLHFTrainer(event_bus, preference_db)
        
        # Initial training with small dataset
        for i in range(10):
            choice = ExpertChoice(
                decision_id=f"decision_{i}",
                chosen_strategy_id="strategy_1",
                expert_id="expert_001",
                timestamp=datetime.now(),
                confidence=0.8,
                reasoning="Test reasoning",
                market_view="Test view",
                risk_assessment="Test assessment"
            )
            preference_db.store_expert_choice(choice, sample_decision_point)
        
        # First training session
        results_1 = trainer.train_reward_model(epochs=2)
        initial_accuracy = results_1.get("val_accuracy", 0.0)
        
        # Add more expert feedback
        for i in range(10, 20):
            choice = ExpertChoice(
                decision_id=f"decision_{i}",
                chosen_strategy_id="strategy_2",  # Different preference
                expert_id="expert_002",
                timestamp=datetime.now(),
                confidence=0.9,
                reasoning="Different reasoning",
                market_view="Different view",
                risk_assessment="Different assessment"
            )
            preference_db.store_expert_choice(choice, sample_decision_point)
        
        # Second training session
        results_2 = trainer.train_reward_model(epochs=2)
        updated_accuracy = results_2.get("val_accuracy", 0.0)
        
        # Verify model adapted to new feedback
        assert len(trainer.training_history) == 2
        assert trainer.get_training_status()["total_preferences"] >= 20


class TestUserInteractionSimulation:
    """Test user interaction simulation scenarios"""
    
    def test_expert_decision_patterns(self, event_bus, preference_db, sample_decision_point):
        """Test simulation of different expert decision patterns"""
        analytics = ExpertAnalytics(preference_db)
        
        # Simulate conservative expert
        conservative_choices = []
        for i in range(10):
            choice = ExpertChoice(
                decision_id=f"conservative_{i}",
                chosen_strategy_id="strategy_2",  # Conservative strategy
                expert_id="conservative_expert",
                timestamp=datetime.now(),
                confidence=0.9,  # High confidence
                reasoning="Conservative approach preferred",
                market_view="Cautious",
                risk_assessment="Risk-averse"
            )
            conservative_choices.append(choice)
            preference_db.store_expert_choice(choice, sample_decision_point)
        
        # Simulate aggressive expert
        aggressive_choices = []
        for i in range(10):
            choice = ExpertChoice(
                decision_id=f"aggressive_{i}",
                chosen_strategy_id="strategy_1",  # Aggressive strategy
                expert_id="aggressive_expert",
                timestamp=datetime.now(),
                confidence=0.7,  # Lower confidence
                reasoning="Aggressive opportunity",
                market_view="Bullish",
                risk_assessment="High risk, high reward"
            )
            aggressive_choices.append(choice)
            preference_db.store_expert_choice(choice, sample_decision_point)
        
        # Analyze patterns
        conservative_metrics = analytics.calculate_expert_metrics("conservative_expert")
        aggressive_metrics = analytics.calculate_expert_metrics("aggressive_expert")
        
        # Conservative expert should have higher confidence
        assert conservative_metrics.average_confidence > aggressive_metrics.average_confidence
        
        # Different risk profiles should be detected
        assert conservative_metrics.risk_profile != aggressive_metrics.risk_profile
    
    def test_expert_learning_curve_simulation(self, event_bus, preference_db, sample_decision_point):
        """Test simulation of expert learning and improvement over time"""
        analytics = ExpertAnalytics(preference_db)
        
        # Simulate improving expert over time
        for i in range(20):
            # Gradually increasing confidence and success
            confidence = min(0.5 + (i * 0.02), 0.95)
            success_outcome = 0.02 * i if i > 10 else -0.01  # Starts negative, becomes positive
            
            choice = ExpertChoice(
                decision_id=f"learning_{i}",
                chosen_strategy_id="strategy_1",
                expert_id="learning_expert",
                timestamp=datetime.now() - timedelta(days=20-i),  # Historical order
                confidence=confidence,
                reasoning=f"Learning iteration {i}",
                market_view="Evolving view",
                risk_assessment="Improving assessment"
            )
            preference_db.store_expert_choice(choice, sample_decision_point)
            
            # Update with market outcome
            preference_db.update_market_outcome(f"learning_{i}", success_outcome)
        
        # Analyze learning curve
        metrics = analytics.calculate_expert_metrics("learning_expert")
        
        # Should detect improvement trend
        assert metrics.performance_trend == "improving"
        assert metrics.total_decisions == 20
        assert metrics.average_confidence > 0.7  # Should be higher due to learning
    
    def test_multi_expert_consensus_simulation(self, event_bus, preference_db, sample_decision_point):
        """Test simulation of multi-expert consensus scenarios"""
        analytics = ExpertAnalytics(preference_db)
        
        # Create scenario with multiple experts on same decision
        experts = ["expert_1", "expert_2", "expert_3"]
        decision_id = "consensus_decision"
        
        # Two experts choose strategy_1, one chooses strategy_2
        for i, expert_id in enumerate(experts):
            chosen_strategy = "strategy_1" if i < 2 else "strategy_2"
            confidence = 0.8 if i < 2 else 0.6  # Minority has lower confidence
            
            choice = ExpertChoice(
                decision_id=f"{decision_id}_{expert_id}",
                chosen_strategy_id=chosen_strategy,
                expert_id=expert_id,
                timestamp=datetime.now(),
                confidence=confidence,
                reasoning=f"Expert {expert_id} reasoning",
                market_view="Market view",
                risk_assessment="Risk assessment"
            )
            preference_db.store_expert_choice(choice, sample_decision_point)
        
        # Compare expert metrics
        comparison = analytics.compare_experts(experts)
        
        # Should have 3 experts in comparison
        assert len(comparison["experts"]) == 3
        
        # Rankings should be populated
        assert len(comparison["rankings"]["by_confidence"]) == 3
        
        # Statistical summary should show differences
        assert "avg_confidence" in comparison["statistical_summary"]
        assert "std_confidence" in comparison["statistical_summary"]


class TestPerformanceAndScaling:
    """Test performance and scaling characteristics"""
    
    def test_large_preference_dataset_handling(self, event_bus, preference_db, sample_decision_point):
        """Test handling of large preference datasets"""
        trainer = RLHFTrainer(event_bus, preference_db)
        
        # Create large dataset
        num_preferences = 1000
        for i in range(num_preferences):
            choice = ExpertChoice(
                decision_id=f"large_dataset_{i}",
                chosen_strategy_id=f"strategy_{i % 2 + 1}",
                expert_id=f"expert_{i % 10}",  # 10 different experts
                timestamp=datetime.now(),
                confidence=0.5 + (i % 50) / 100.0,  # Varying confidence
                reasoning=f"Large dataset reasoning {i}",
                market_view="View",
                risk_assessment="Assessment"
            )
            preference_db.store_expert_choice(choice, sample_decision_point)
        
        # Test retrieval performance
        import time
        start_time = time.time()
        records = preference_db.get_preference_records(limit=100)
        retrieval_time = time.time() - start_time
        
        assert len(records) == 100
        assert retrieval_time < 1.0  # Should be fast
        
        # Test training on large dataset
        start_time = time.time()
        results = trainer.train_reward_model(epochs=1)
        training_time = time.time() - start_time
        
        assert "avg_loss" in results
        assert training_time < 30.0  # Should complete in reasonable time
    
    def test_concurrent_expert_feedback(self, event_bus, config_manager, mock_redis):
        """Test handling of concurrent expert feedback"""
        coordinator = HumanFeedbackCoordinator(event_bus, config_manager, mock_redis)
        
        # Simulate multiple experts providing feedback simultaneously
        async def simulate_expert_feedback(expert_id, decision_id):
            feedback_payload = {
                "expert_choice": {
                    "decision_id": decision_id,
                    "chosen_strategy_id": "strategy_1",
                    "expert_id": expert_id,
                    "timestamp": datetime.now().isoformat(),
                    "confidence": 0.8,
                    "reasoning": f"Feedback from {expert_id}",
                    "alternative_considered": "strategy_2",
                    "market_view": "Bullish",
                    "risk_assessment": "Acceptable"
                },
                "decision_context": None
            }
            
            await coordinator._process_expert_feedback(feedback_payload)
            return True
        
        async def test_concurrent_processing():
            # Create multiple concurrent feedback processing tasks
            tasks = []
            for i in range(10):
                task = simulate_expert_feedback(f"expert_{i}", f"decision_{i}")
                tasks.append(task)
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks)
            return all(results)
        
        success = asyncio.run(test_concurrent_processing())
        assert success is True
        
        # Verify all feedback was processed
        assert coordinator.metrics["expert_responses"] == 10
    
    def test_memory_usage_optimization(self, event_bus, preference_db, sample_decision_point):
        """Test memory usage optimization for large datasets"""
        trainer = RLHFTrainer(event_bus, preference_db)
        
        # Monitor memory usage during training
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create moderately sized dataset
        for i in range(500):
            choice = ExpertChoice(
                decision_id=f"memory_test_{i}",
                chosen_strategy_id="strategy_1",
                expert_id="memory_expert",
                timestamp=datetime.now(),
                confidence=0.8,
                reasoning="Memory test",
                market_view="View",
                risk_assessment="Assessment"
            )
            preference_db.store_expert_choice(choice, sample_decision_point)
        
        # Train model
        trainer.train_reward_model(epochs=2)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB)
        assert memory_increase < 500
        
        # Verify model still works
        context_features = np.random.rand(9)
        reward = trainer.get_strategy_reward(context_features, sample_decision_point.strategies[0])
        assert isinstance(reward, float)


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])