"""
Extreme Market Conditions Stress Testing for XAI Trading System
Agent Epsilon - Production Resilience Validation

Simulates extreme market scenarios to test system resilience:
1. Flash crash scenarios
2. High volatility periods  
3. Market circuit breaker events
4. News event spikes
5. System overload conditions
6. Network partition scenarios
"""

import asyncio
import time
import json
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
import aiohttp
import concurrent.futures
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketCondition(Enum):
    """Types of extreme market conditions"""
    FLASH_CRASH = "flash_crash"
    HIGH_VOLATILITY = "high_volatility"
    CIRCUIT_BREAKER = "circuit_breaker"
    NEWS_SPIKE = "news_spike"
    SYSTEM_OVERLOAD = "system_overload"
    NETWORK_PARTITION = "network_partition"
    DATA_CORRUPTION = "data_corruption"
    LIQUIDITY_CRISIS = "liquidity_crisis"


@dataclass
class StressTestResult:
    """Result of a stress test scenario"""
    scenario: str
    condition: MarketCondition
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    max_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    errors: List[str]
    system_recovery_time_ms: Optional[float]
    explanation_quality_degradation: float


class ExtremeMarketStressTester:
    """Main stress testing engine for extreme market conditions"""
    
    def __init__(self, base_url: str, max_concurrent: int = 100):
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.session = None
        self.results: List[StressTestResult] = []
        
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent * 2,
            limit_per_host=self.max_concurrent,
            keepalive_timeout=30
        )
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def run_all_stress_tests(self) -> List[StressTestResult]:
        """Run comprehensive stress test suite"""
        logger.info("🔥 Starting Extreme Market Conditions Stress Testing")
        logger.info("=" * 60)
        
        test_scenarios = [
            ("Flash Crash Simulation", self.test_flash_crash),
            ("High Volatility Stress", self.test_high_volatility),
            ("Circuit Breaker Event", self.test_circuit_breaker),
            ("News Event Spike", self.test_news_spike),
            ("System Overload", self.test_system_overload),
            ("Network Partition", self.test_network_partition),
            ("Data Corruption Handling", self.test_data_corruption),
            ("Liquidity Crisis", self.test_liquidity_crisis)
        ]
        
        for scenario_name, test_func in test_scenarios:
            logger.info(f"\n🧪 Running: {scenario_name}")
            logger.info("-" * 40)
            
            try:
                result = await test_func()
                self.results.append(result)
                self.log_test_result(result)
                
                # Recovery pause between tests
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"❌ Test {scenario_name} failed: {e}")
                continue
        
        logger.info(f"\n🏁 Stress Testing Complete - {len(self.results)} scenarios executed")
        return self.results
    
    async def test_flash_crash(self) -> StressTestResult:
        """Simulate flash crash conditions"""
        start_time = time.time()
        latencies = []
        errors = []
        successful = 0
        failed = 0
        
        # Simulate flash crash: rapid price drops with massive volume
        crash_scenarios = [
            {"price_drop": -0.10, "volume_spike": 50},  # 10% drop, 50x volume
            {"price_drop": -0.05, "volume_spike": 30},  # 5% drop, 30x volume
            {"price_drop": -0.15, "volume_spike": 100}, # 15% drop, 100x volume
        ]
        
        tasks = []
        for i in range(500):  # High request volume during crash
            scenario = random.choice(crash_scenarios)
            task = self.generate_crash_explanation_request(scenario, latencies, errors)
            tasks.append(task)
        
        # Execute in batches to avoid overwhelming the system
        batch_size = 50
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    failed += 1
                    errors.append(str(result))
                else:
                    successful += 1
            
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        duration = time.time() - start_time
        
        return StressTestResult(
            scenario="Flash Crash Simulation",
            condition=MarketCondition.FLASH_CRASH,
            duration_seconds=duration,
            total_requests=len(tasks),
            successful_requests=successful,
            failed_requests=failed,
            avg_latency_ms=np.mean(latencies) if latencies else 0,
            max_latency_ms=np.max(latencies) if latencies else 0,
            p95_latency_ms=np.percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=np.percentile(latencies, 99) if latencies else 0,
            errors=errors[:10],  # Keep first 10 errors
            system_recovery_time_ms=None,
            explanation_quality_degradation=self.calculate_quality_degradation(latencies)
        )
    
    async def test_high_volatility(self) -> StressTestResult:
        """Test system under extreme volatility conditions"""
        start_time = time.time()
        latencies = []
        errors = []
        successful = 0
        failed = 0
        
        # Simulate extreme volatility with rapid price swings
        tasks = []
        for i in range(300):
            volatility = random.uniform(0.05, 0.25)  # 5-25% volatility
            task = self.generate_volatility_explanation_request(volatility, latencies, errors)
            tasks.append(task)
        
        # Execute with high concurrency
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                failed += 1
                errors.append(str(result))
            else:
                successful += 1
        
        duration = time.time() - start_time
        
        return StressTestResult(
            scenario="High Volatility Stress",
            condition=MarketCondition.HIGH_VOLATILITY,
            duration_seconds=duration,
            total_requests=len(tasks),
            successful_requests=successful,
            failed_requests=failed,
            avg_latency_ms=np.mean(latencies) if latencies else 0,
            max_latency_ms=np.max(latencies) if latencies else 0,
            p95_latency_ms=np.percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=np.percentile(latencies, 99) if latencies else 0,
            errors=errors[:10],
            system_recovery_time_ms=None,
            explanation_quality_degradation=self.calculate_quality_degradation(latencies)
        )
    
    async def test_circuit_breaker(self) -> StressTestResult:
        """Test circuit breaker event handling"""
        start_time = time.time()
        latencies = []
        errors = []
        successful = 0
        failed = 0
        
        # Simulate circuit breaker conditions
        circuit_breaker_data = {
            "market_halt": True,
            "halt_reason": "LIMIT_UP_DOWN",
            "price_movement": -0.20,  # 20% drop triggers breaker
            "trading_suspended": True,
            "expected_resume_time": (datetime.utcnow() + timedelta(minutes=15)).isoformat()
        }
        
        tasks = []
        for i in range(200):
            task = self.generate_circuit_breaker_request(circuit_breaker_data, latencies, errors)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                failed += 1
                errors.append(str(result))
            else:
                successful += 1
        
        duration = time.time() - start_time
        
        return StressTestResult(
            scenario="Circuit Breaker Event",
            condition=MarketCondition.CIRCUIT_BREAKER,
            duration_seconds=duration,
            total_requests=len(tasks),
            successful_requests=successful,
            failed_requests=failed,
            avg_latency_ms=np.mean(latencies) if latencies else 0,
            max_latency_ms=np.max(latencies) if latencies else 0,
            p95_latency_ms=np.percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=np.percentile(latencies, 99) if latencies else 0,
            errors=errors[:10],
            system_recovery_time_ms=None,
            explanation_quality_degradation=self.calculate_quality_degradation(latencies)
        )
    
    async def test_news_spike(self) -> StressTestResult:
        """Test system during major news events"""
        start_time = time.time()
        latencies = []
        errors = []
        successful = 0
        failed = 0
        
        # Simulate major news event impact
        news_scenarios = [
            {"event": "FOMC_DECISION", "impact": 0.03, "volume_spike": 20},
            {"event": "GEOPOLITICAL_CRISIS", "impact": -0.08, "volume_spike": 80},
            {"event": "EARNINGS_SURPRISE", "impact": 0.15, "volume_spike": 40},
            {"event": "REGULATORY_ANNOUNCEMENT", "impact": -0.12, "volume_spike": 60}
        ]
        
        tasks = []
        for i in range(400):  # High volume during news
            scenario = random.choice(news_scenarios)
            task = self.generate_news_spike_request(scenario, latencies, errors)
            tasks.append(task)
        
        # Simulate burst pattern typical of news events
        batch_size = 100
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    failed += 1
                    errors.append(str(result))
                else:
                    successful += 1
            
            # Rapid bursts then brief pause
            await asyncio.sleep(0.05)
        
        duration = time.time() - start_time
        
        return StressTestResult(
            scenario="News Event Spike",
            condition=MarketCondition.NEWS_SPIKE,
            duration_seconds=duration,
            total_requests=len(tasks),
            successful_requests=successful,
            failed_requests=failed,
            avg_latency_ms=np.mean(latencies) if latencies else 0,
            max_latency_ms=np.max(latencies) if latencies else 0,
            p95_latency_ms=np.percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=np.percentile(latencies, 99) if latencies else 0,
            errors=errors[:10],
            system_recovery_time_ms=None,
            explanation_quality_degradation=self.calculate_quality_degradation(latencies)
        )
    
    async def test_system_overload(self) -> StressTestResult:
        """Test system under extreme load beyond normal capacity"""
        start_time = time.time()
        latencies = []
        errors = []
        successful = 0
        failed = 0
        
        # Create massive request overload
        tasks = []
        for i in range(1000):  # Very high load
            task = self.generate_overload_request(latencies, errors)
            tasks.append(task)
        
        # Execute all at once to test overload handling
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                failed += 1
                errors.append(str(result))
            else:
                successful += 1
        
        duration = time.time() - start_time
        
        return StressTestResult(
            scenario="System Overload",
            condition=MarketCondition.SYSTEM_OVERLOAD,
            duration_seconds=duration,
            total_requests=len(tasks),
            successful_requests=successful,
            failed_requests=failed,
            avg_latency_ms=np.mean(latencies) if latencies else 0,
            max_latency_ms=np.max(latencies) if latencies else 0,
            p95_latency_ms=np.percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=np.percentile(latencies, 99) if latencies else 0,
            errors=errors[:10],
            system_recovery_time_ms=None,
            explanation_quality_degradation=self.calculate_quality_degradation(latencies)
        )
    
    async def test_network_partition(self) -> StressTestResult:
        """Test network partition scenarios"""
        start_time = time.time()
        latencies = []
        errors = []
        successful = 0
        failed = 0
        
        # Simulate network issues with timeouts and connection errors
        tasks = []
        for i in range(200):
            # Some requests will timeout to simulate network issues
            task = self.generate_network_partition_request(latencies, errors)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                failed += 1
                errors.append(str(result))
            else:
                successful += 1
        
        duration = time.time() - start_time
        
        return StressTestResult(
            scenario="Network Partition",
            condition=MarketCondition.NETWORK_PARTITION,
            duration_seconds=duration,
            total_requests=len(tasks),
            successful_requests=successful,
            failed_requests=failed,
            avg_latency_ms=np.mean(latencies) if latencies else 0,
            max_latency_ms=np.max(latencies) if latencies else 0,
            p95_latency_ms=np.percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=np.percentile(latencies, 99) if latencies else 0,
            errors=errors[:10],
            system_recovery_time_ms=None,
            explanation_quality_degradation=self.calculate_quality_degradation(latencies)
        )
    
    async def test_data_corruption(self) -> StressTestResult:
        """Test data corruption handling"""
        start_time = time.time()
        latencies = []
        errors = []
        successful = 0
        failed = 0
        
        # Generate requests with corrupted/invalid data
        tasks = []
        for i in range(150):
            task = self.generate_corrupted_data_request(latencies, errors)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                failed += 1
                errors.append(str(result))
            else:
                successful += 1
        
        duration = time.time() - start_time
        
        return StressTestResult(
            scenario="Data Corruption Handling",
            condition=MarketCondition.DATA_CORRUPTION,
            duration_seconds=duration,
            total_requests=len(tasks),
            successful_requests=successful,
            failed_requests=failed,
            avg_latency_ms=np.mean(latencies) if latencies else 0,
            max_latency_ms=np.max(latencies) if latencies else 0,
            p95_latency_ms=np.percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=np.percentile(latencies, 99) if latencies else 0,
            errors=errors[:10],
            system_recovery_time_ms=None,
            explanation_quality_degradation=self.calculate_quality_degradation(latencies)
        )
    
    async def test_liquidity_crisis(self) -> StressTestResult:
        """Test liquidity crisis conditions"""
        start_time = time.time()
        latencies = []
        errors = []
        successful = 0
        failed = 0
        
        # Simulate liquidity crisis with wide spreads and low volume
        crisis_data = {
            "bid_ask_spread": 0.05,  # 5% spread
            "market_depth": 0.1,     # Very low depth
            "volume": 0.05,          # 5% of normal volume
            "liquidity_providers": 2  # Limited providers
        }
        
        tasks = []
        for i in range(250):
            task = self.generate_liquidity_crisis_request(crisis_data, latencies, errors)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                failed += 1
                errors.append(str(result))
            else:
                successful += 1
        
        duration = time.time() - start_time
        
        return StressTestResult(
            scenario="Liquidity Crisis",
            condition=MarketCondition.LIQUIDITY_CRISIS,
            duration_seconds=duration,
            total_requests=len(tasks),
            successful_requests=successful,
            failed_requests=failed,
            avg_latency_ms=np.mean(latencies) if latencies else 0,
            max_latency_ms=np.max(latencies) if latencies else 0,
            p95_latency_ms=np.percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=np.percentile(latencies, 99) if latencies else 0,
            errors=errors[:10],
            system_recovery_time_ms=None,
            explanation_quality_degradation=self.calculate_quality_degradation(latencies)
        )
    
    async def generate_crash_explanation_request(self, scenario: Dict[str, Any], latencies: List[float], errors: List[str]):
        """Generate explanation request during flash crash"""
        request_data = {
            "symbol": "NQ",
            "action": "EMERGENCY_SELL",
            "confidence": 0.95,
            "market_features": [
                scenario["price_drop"],  # Negative price movement
                scenario["volume_spike"],  # Volume spike
                0.95,  # Very high volatility
                -0.8,  # Negative momentum
                0.1,   # Low liquidity
                0.9,   # High urgency
                -0.7,  # Negative sentiment
                0.8,   # High risk
                0.95,  # Market stress indicator
                -0.9,  # Correlation breakdown
                0.85,  # Fear index
                0.1,   # Market depth
                0.95,  # Volatility spike
                -0.8,  # Technical breakdown
                0.9    # Emergency condition
            ],
            "feature_names": [
                "price_movement", "volume_spike", "volatility", "momentum",
                "liquidity", "urgency", "sentiment", "risk_level",
                "market_stress", "correlation", "fear_index", "market_depth",
                "volatility_spike", "technical_signal", "emergency_indicator"
            ],
            "market_conditions": {
                "crash_detected": True,
                "volatility": 0.95,
                "market_stress": 0.9,
                "emergency_mode": True
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self.make_explanation_request(request_data, latencies, errors)
    
    async def generate_volatility_explanation_request(self, volatility: float, latencies: List[float], errors: List[str]):
        """Generate explanation request during high volatility"""
        request_data = {
            "symbol": random.choice(["NQ", "ES", "BTC", "ETH"]),
            "action": random.choice(["REDUCE_POSITION", "HOLD", "HEDGE"]),
            "confidence": random.uniform(0.5, 0.8),
            "market_features": [
                random.uniform(-0.1, 0.1),  # Price movement
                random.uniform(1, 10),      # Volume ratio
                volatility,                 # High volatility
                random.uniform(-0.5, 0.5),  # Momentum
                random.uniform(0.1, 0.8),   # Liquidity
                random.uniform(0.6, 0.9),   # Uncertainty
                random.uniform(-0.3, 0.3),  # Sentiment
                volatility * 0.8,           # Risk level
                volatility,                 # Market stress
                random.uniform(0.2, 0.8),   # Correlation
                volatility * 0.9,           # Fear index
                random.uniform(0.2, 0.7),   # Market depth
                volatility,                 # Volatility spike
                random.uniform(-0.4, 0.4),  # Technical signal
                volatility * 0.7            # Stress indicator
            ],
            "feature_names": [
                "price_movement", "volume_ratio", "volatility", "momentum",
                "liquidity", "uncertainty", "sentiment", "risk_level",
                "market_stress", "correlation", "fear_index", "market_depth",
                "volatility_spike", "technical_signal", "stress_indicator"
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self.make_explanation_request(request_data, latencies, errors)
    
    async def generate_circuit_breaker_request(self, circuit_data: Dict[str, Any], latencies: List[float], errors: List[str]):
        """Generate explanation request during circuit breaker event"""
        request_data = {
            "symbol": "SPY",
            "action": "TRADING_HALT",
            "confidence": 1.0,
            "market_features": [
                circuit_data["price_movement"],  # Large price move
                0.0,    # No volume during halt
                0.5,    # Moderate volatility during halt
                -0.9,   # Strong negative momentum before halt
                0.0,    # No liquidity during halt
                1.0,    # Maximum uncertainty
                -0.8,   # Negative sentiment
                0.9,    # High risk
                1.0,    # Maximum market stress
                0.1,    # Low correlation during crisis
                0.95,   # High fear
                0.0,    # No market depth
                0.8,    # Previous volatility spike
                -1.0,   # Circuit breaker signal
                1.0     # Emergency condition
            ],
            "feature_names": [
                "price_movement", "volume", "volatility", "momentum",
                "liquidity", "uncertainty", "sentiment", "risk_level",
                "market_stress", "correlation", "fear_index", "market_depth",
                "volatility_spike", "circuit_breaker_signal", "emergency_condition"
            ],
            "market_conditions": circuit_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self.make_explanation_request(request_data, latencies, errors)
    
    async def generate_news_spike_request(self, news_scenario: Dict[str, Any], latencies: List[float], errors: List[str]):
        """Generate explanation request during news event"""
        request_data = {
            "symbol": random.choice(["NQ", "ES", "YM", "RTY"]),
            "action": "REBALANCE" if news_scenario["impact"] > 0 else "DEFENSIVE",
            "confidence": random.uniform(0.7, 0.9),
            "market_features": [
                news_scenario["impact"],              # News impact
                news_scenario["volume_spike"],        # Volume spike
                abs(news_scenario["impact"]) * 2,     # Volatility
                news_scenario["impact"] * 0.8,        # Momentum
                1.0 / news_scenario["volume_spike"],  # Liquidity impact
                0.8,                                  # High uncertainty
                news_scenario["impact"] * 0.6,        # Sentiment
                abs(news_scenario["impact"]) * 0.7,   # Risk level
                0.8,                                  # Market stress
                0.6,                                  # Correlation
                abs(news_scenario["impact"]) * 0.9,   # News reaction
                0.3,                                  # Market depth
                abs(news_scenario["impact"]) * 1.5,   # Volatility spike
                news_scenario["impact"] * 0.5,        # Technical signal
                0.9                                   # News importance
            ],
            "feature_names": [
                "news_impact", "volume_spike", "volatility", "momentum",
                "liquidity", "uncertainty", "sentiment", "risk_level",
                "market_stress", "correlation", "news_reaction", "market_depth",
                "volatility_spike", "technical_signal", "news_importance"
            ],
            "news_event": news_scenario,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self.make_explanation_request(request_data, latencies, errors)
    
    async def generate_overload_request(self, latencies: List[float], errors: List[str]):
        """Generate high-frequency overload request"""
        request_data = {
            "symbol": "NQ",
            "action": "RAPID_DECISION",
            "confidence": 0.75,
            "market_features": np.random.normal(0, 1, 15).tolist(),
            "feature_names": [f"feature_{i}" for i in range(15)],
            "high_frequency": True,
            "priority": "HIGH",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self.make_explanation_request(request_data, latencies, errors)
    
    async def generate_network_partition_request(self, latencies: List[float], errors: List[str]):
        """Generate request that may experience network issues"""
        request_data = {
            "symbol": "ES",
            "action": "NETWORK_TEST",
            "confidence": 0.6,
            "market_features": [0.1] * 15,
            "feature_names": [f"network_feature_{i}" for i in range(15)],
            "network_test": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Randomly simulate network timeouts
        if random.random() < 0.3:  # 30% chance of timeout
            await asyncio.sleep(random.uniform(5, 10))  # Simulate timeout
        
        return await self.make_explanation_request(request_data, latencies, errors)
    
    async def generate_corrupted_data_request(self, latencies: List[float], errors: List[str]):
        """Generate request with corrupted data"""
        # Create various types of corrupted data
        corruption_types = [
            "invalid_json",
            "missing_fields",
            "wrong_types",
            "extreme_values",
            "null_values",
            "malformed_arrays"
        ]
        
        corruption_type = random.choice(corruption_types)
        
        if corruption_type == "invalid_json":
            # This will cause JSON parsing error
            return await self.make_raw_request("INVALID JSON DATA", latencies, errors)
        
        elif corruption_type == "missing_fields":
            request_data = {
                "symbol": "BTC",
                # Missing required fields
                "timestamp": datetime.utcnow().isoformat()
            }
        
        elif corruption_type == "wrong_types":
            request_data = {
                "symbol": 12345,  # Should be string
                "action": ["INVALID_ACTION"],  # Should be string
                "confidence": "high",  # Should be float
                "market_features": "not_an_array",  # Should be array
                "feature_names": 42,  # Should be array
                "timestamp": datetime.utcnow().isoformat()
            }
        
        elif corruption_type == "extreme_values":
            request_data = {
                "symbol": "ETH",
                "action": "TEST",
                "confidence": float('inf'),  # Infinite value
                "market_features": [float('nan')] * 15,  # NaN values
                "feature_names": [f"extreme_feature_{i}" for i in range(15)],
                "timestamp": datetime.utcnow().isoformat()
            }
        
        elif corruption_type == "null_values":
            request_data = {
                "symbol": None,
                "action": None,
                "confidence": None,
                "market_features": None,
                "feature_names": None,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        else:  # malformed_arrays
            request_data = {
                "symbol": "NQ",
                "action": "TEST",
                "confidence": 0.5,
                "market_features": [1, 2, [3, 4], {"nested": "object"}],  # Mixed types
                "feature_names": ["f1", 2, None, {"invalid": "name"}],  # Mixed types
                "timestamp": datetime.utcnow().isoformat()
            }
        
        return await self.make_explanation_request(request_data, latencies, errors)
    
    async def generate_liquidity_crisis_request(self, crisis_data: Dict[str, Any], latencies: List[float], errors: List[str]):
        """Generate explanation request during liquidity crisis"""
        request_data = {
            "symbol": "RTY",
            "action": "LIQUIDITY_ADJUSTMENT",
            "confidence": 0.6,
            "market_features": [
                -0.02,                              # Small price movement
                0.1,                                # Low volume
                0.3,                                # Moderate volatility
                0.05,                               # Weak momentum
                crisis_data["market_depth"],        # Very low liquidity
                0.9,                                # High uncertainty
                -0.4,                               # Negative sentiment
                0.7,                                # High risk due to illiquidity
                0.8,                                # Market stress
                0.2,                                # Low correlation
                0.7,                                # Fear index
                crisis_data["market_depth"],        # Market depth
                0.3,                                # Volatility
                -0.3,                               # Technical signal
                crisis_data["bid_ask_spread"]       # Wide spreads indicator
            ],
            "feature_names": [
                "price_movement", "volume", "volatility", "momentum",
                "liquidity", "uncertainty", "sentiment", "risk_level",
                "market_stress", "correlation", "fear_index", "market_depth",
                "volatility_level", "technical_signal", "spread_indicator"
            ],
            "liquidity_conditions": crisis_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self.make_explanation_request(request_data, latencies, errors)
    
    async def make_explanation_request(self, request_data: Dict[str, Any], latencies: List[float], errors: List[str]):
        """Make explanation request and track performance"""
        start_time = time.time()
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/v1/explanations/generate",
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)
                
                if response.status == 200:
                    await response.json()  # Consume response
                    return True
                else:
                    error_text = await response.text()
                    errors.append(f"HTTP {response.status}: {error_text[:100]}")
                    return False
                    
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)
            errors.append(f"Request error: {str(e)[:100]}")
            return False
    
    async def make_raw_request(self, raw_data: str, latencies: List[float], errors: List[str]):
        """Make raw request for testing invalid data"""
        start_time = time.time()
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/v1/explanations/generate",
                data=raw_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)
                
                error_text = await response.text()
                errors.append(f"Raw request HTTP {response.status}: {error_text[:100]}")
                return False
                    
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)
            errors.append(f"Raw request error: {str(e)[:100]}")
            return False
    
    def calculate_quality_degradation(self, latencies: List[float]) -> float:
        """Calculate explanation quality degradation during stress"""
        if not latencies:
            return 0.0
        
        # Calculate quality degradation based on latency increase
        avg_latency = np.mean(latencies)
        baseline_latency = 50  # Expected baseline latency in ms
        
        if avg_latency <= baseline_latency:
            return 0.0
        
        # Quality degrades as latency increases
        degradation = min(1.0, (avg_latency - baseline_latency) / baseline_latency)
        return degradation
    
    def log_test_result(self, result: StressTestResult):
        """Log individual test result"""
        success_rate = (result.successful_requests / result.total_requests * 100) if result.total_requests > 0 else 0
        
        logger.info(f"   Scenario: {result.scenario}")
        logger.info(f"   Duration: {result.duration_seconds:.2f}s")
        logger.info(f"   Requests: {result.total_requests} (Success: {success_rate:.1f}%)")
        logger.info(f"   Latency: Avg={result.avg_latency_ms:.1f}ms, P95={result.p95_latency_ms:.1f}ms, Max={result.max_latency_ms:.1f}ms")
        
        if result.p95_latency_ms <= 100:
            logger.info(f"   ✅ PASSED: P95 latency {result.p95_latency_ms:.1f}ms <= 100ms")
        else:
            logger.info(f"   ⚠️  WARNING: P95 latency {result.p95_latency_ms:.1f}ms > 100ms")
        
        if result.explanation_quality_degradation > 0.3:
            logger.info(f"   ⚠️  Quality degradation: {result.explanation_quality_degradation:.1%}")
        
        if result.errors:
            logger.info(f"   Errors: {len(result.errors)} unique errors")
    
    def generate_stress_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive stress test report"""
        if not self.results:
            return {"error": "No test results available"}
        
        report = {
            "test_summary": {
                "total_scenarios": len(self.results),
                "total_requests": sum(r.total_requests for r in self.results),
                "total_successful": sum(r.successful_requests for r in self.results),
                "total_failed": sum(r.failed_requests for r in self.results),
                "overall_success_rate": 0,
                "test_timestamp": datetime.utcnow().isoformat()
            },
            "performance_metrics": {
                "avg_latency_ms": 0,
                "max_latency_ms": 0,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0,
                "scenarios_meeting_sla": 0
            },
            "scenario_results": [],
            "recommendations": []
        }
        
        # Calculate overall metrics
        total_requests = report["test_summary"]["total_requests"]
        total_successful = report["test_summary"]["total_successful"]
        
        if total_requests > 0:
            report["test_summary"]["overall_success_rate"] = total_successful / total_requests * 100
        
        # Aggregate performance metrics
        all_avg_latencies = [r.avg_latency_ms for r in self.results if r.avg_latency_ms > 0]
        all_max_latencies = [r.max_latency_ms for r in self.results if r.max_latency_ms > 0]
        all_p95_latencies = [r.p95_latency_ms for r in self.results if r.p95_latency_ms > 0]
        all_p99_latencies = [r.p99_latency_ms for r in self.results if r.p99_latency_ms > 0]
        
        if all_avg_latencies:
            report["performance_metrics"]["avg_latency_ms"] = np.mean(all_avg_latencies)
            report["performance_metrics"]["max_latency_ms"] = max(all_max_latencies)
            report["performance_metrics"]["p95_latency_ms"] = np.mean(all_p95_latencies)
            report["performance_metrics"]["p99_latency_ms"] = np.mean(all_p99_latencies)
        
        # Count scenarios meeting SLA
        scenarios_meeting_sla = sum(1 for r in self.results if r.p95_latency_ms <= 100)
        report["performance_metrics"]["scenarios_meeting_sla"] = scenarios_meeting_sla
        
        # Add individual scenario results
        for result in self.results:
            scenario_data = {
                "scenario": result.scenario,
                "condition": result.condition.value,
                "duration_seconds": result.duration_seconds,
                "requests": result.total_requests,
                "success_rate": (result.successful_requests / result.total_requests * 100) if result.total_requests > 0 else 0,
                "avg_latency_ms": result.avg_latency_ms,
                "p95_latency_ms": result.p95_latency_ms,
                "meets_sla": result.p95_latency_ms <= 100,
                "quality_degradation": result.explanation_quality_degradation,
                "error_count": len(result.errors)
            }
            report["scenario_results"].append(scenario_data)
        
        # Generate recommendations
        report["recommendations"] = self.generate_recommendations()
        
        return report
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check latency performance
        high_latency_scenarios = [r for r in self.results if r.p95_latency_ms > 100]
        if high_latency_scenarios:
            recommendations.append(
                f"🔧 Optimize performance: {len(high_latency_scenarios)} scenarios exceeded 100ms P95 latency"
            )
        
        # Check error rates
        high_error_scenarios = [r for r in self.results if (r.failed_requests / r.total_requests) > 0.05]
        if high_error_scenarios:
            recommendations.append(
                f"🛠️ Improve error handling: {len(high_error_scenarios)} scenarios had >5% error rate"
            )
        
        # Check quality degradation
        quality_issues = [r for r in self.results if r.explanation_quality_degradation > 0.2]
        if quality_issues:
            recommendations.append(
                f"📊 Address quality degradation: {len(quality_issues)} scenarios showed significant quality loss"
            )
        
        # System-specific recommendations
        if any(r.condition == MarketCondition.SYSTEM_OVERLOAD for r in self.results):
            recommendations.append("⚡ Consider horizontal scaling for peak load handling")
        
        if any(r.condition == MarketCondition.NETWORK_PARTITION for r in self.results):
            recommendations.append("🌐 Implement better network resilience and timeout handling")
        
        if any(r.condition == MarketCondition.DATA_CORRUPTION for r in self.results):
            recommendations.append("🔍 Strengthen input validation and error recovery")
        
        if not recommendations:
            recommendations.append("✅ All stress tests passed - system shows excellent resilience")
        
        return recommendations


async def main():
    """Main stress testing execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="XAI Trading System Extreme Market Stress Testing")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL for XAI system")
    parser.add_argument("--output", default="./stress_test_results", help="Output directory for results")
    parser.add_argument("--concurrent", type=int, default=100, help="Max concurrent requests")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Run stress tests
    async with ExtremeMarketStressTester(args.url, args.concurrent) as tester:
        results = await tester.run_all_stress_tests()
        
        # Generate and save report
        report = tester.generate_stress_test_report()
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"stress_test_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n📋 Stress test report saved: {report_file}")
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("🔥 EXTREME MARKET CONDITIONS STRESS TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total scenarios: {report['test_summary']['total_scenarios']}")
        logger.info(f"Total requests: {report['test_summary']['total_requests']}")
        logger.info(f"Success rate: {report['test_summary']['overall_success_rate']:.1f}%")
        logger.info(f"Avg P95 latency: {report['performance_metrics']['p95_latency_ms']:.1f}ms")
        logger.info(f"Scenarios meeting SLA: {report['performance_metrics']['scenarios_meeting_sla']}/{len(results)}")
        
        logger.info("\n📋 Recommendations:")
        for rec in report['recommendations']:
            logger.info(f"   {rec}")
        
        logger.info("\n🏁 Stress testing complete!")


if __name__ == "__main__":
    asyncio.run(main())