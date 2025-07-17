"""
Comprehensive Load Testing Suite for XAI Trading System
Agent Epsilon - Production Performance Validation

Tests:
1. Explanation latency under load (<100ms requirement)
2. Query response times (<2s requirement)
3. Concurrent user simulation
4. Market data stress testing
5. WebSocket connection handling
"""

import logging


import time
import json
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np

from locust import HttpUser, task, between, events
from locust.contrib.fasthttp import FastHttpUser
import websocket
import threading


class XAITradingLoadTest(FastHttpUser):
    """Main load test class for XAI Trading System"""
    
    wait_time = between(0.1, 2.0)  # 0.1-2 seconds between requests
    
    def on_start(self):
        """Initialize test user"""
        self.user_id = str(uuid.uuid4())
        self.auth_token = None
        self.trading_symbols = ["NQ", "ES", "YM", "RTY", "BTC", "ETH"]
        self.explanation_metrics = []
        self.query_metrics = []
        
        # Authenticate user
        self.authenticate()
        
        # Setup WebSocket connection
        self.setup_websocket()
    
    def authenticate(self):
        """Authenticate test user"""
        auth_data = {
            "username": f"test_user_{self.user_id[:8]}",
            "password": "test_password_123"
        }
        
        with self.client.post("/auth/login", json=auth_data, catch_response=True) as response:
            if response.status_code == 200:
                self.auth_token = response.json().get("access_token")
                response.success()
            else:
                response.failure(f"Authentication failed: {response.status_code}")
    
    def setup_websocket(self):
        """Setup WebSocket connection for real-time updates"""
        try:
            self.ws = websocket.create_connection(
                f"ws://{self.host.replace('http://', '').replace('https://', '')}/ws/trading",
                timeout=10
            )
            self.ws_connected = True
        except Exception as e:
            print(f"WebSocket connection failed: {e}")
            self.ws_connected = False
    
    @task(40)
    def request_explanation(self):
        """Test explanation generation latency (highest priority - 40% of requests)"""
        symbol = random.choice(self.trading_symbols)
        
        # Create realistic trading decision context
        decision_data = {
            "symbol": symbol,
            "action": random.choice(["BUY", "SELL", "HOLD"]),
            "confidence": random.uniform(0.6, 0.95),
            "market_features": np.random.normal(0, 1, 15).tolist(),
            "feature_names": [
                "price_momentum", "volume_ratio", "volatility", "fvg_signal",
                "ma_cross", "rsi", "trend_strength", "support_resistance",
                "order_flow", "bid_ask_spread", "time_of_day", "vix",
                "correlation", "momentum_divergence", "liquidity"
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
        
        start_time = time.time()
        
        with self.client.post(
            "/api/v1/explanations/generate",
            json=decision_data,
            headers=headers,
            catch_response=True,
            name="explanation_generation"
        ) as response:
            
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            if response.status_code == 200:
                # Validate explanation quality
                explanation = response.json()
                
                if self.validate_explanation(explanation, latency):
                    response.success()
                    self.explanation_metrics.append(latency)
                    
                    # Log latency violation
                    if latency > 100:
                        events.request_failure.fire(
                            request_type="explanation_latency_violation",
                            name="explanation_generation",
                            response_time=latency,
                            response_length=len(response.text),
                            exception=f"Latency {latency:.2f}ms exceeds 100ms requirement"
                        )
                else:
                    response.failure("Invalid explanation response")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(25)
    def query_performance_analytics(self):
        """Test complex query response times (25% of requests)"""
        query_types = [
            "daily_performance",
            "decision_attribution",
            "risk_analysis",
            "model_performance",
            "consensus_history"
        ]
        
        query_type = random.choice(query_types)
        
        query_data = {
            "query_type": query_type,
            "time_range": {
                "start": (datetime.utcnow() - timedelta(hours=24)).isoformat(),
                "end": datetime.utcnow().isoformat()
            },
            "filters": {
                "symbols": random.sample(self.trading_symbols, random.randint(1, 3)),
                "confidence_threshold": random.uniform(0.5, 0.9)
            }
        }
        
        headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
        
        start_time = time.time()
        
        with self.client.post(
            "/api/v1/analytics/query",
            json=query_data,
            headers=headers,
            catch_response=True,
            name=f"query_{query_type}"
        ) as response:
            
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                if response_time <= 2000:  # 2 second requirement
                    response.success()
                    self.query_metrics.append(response_time)
                else:
                    response.failure(f"Query timeout: {response_time:.2f}ms > 2000ms")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(20)
    def real_time_market_data(self):
        """Test real-time market data processing (20% of requests)"""
        symbol = random.choice(self.trading_symbols)
        
        market_data = {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "price": random.uniform(4000, 4200),
            "volume": random.uniform(1000, 5000),
            "bid": random.uniform(4000, 4100),
            "ask": random.uniform(4100, 4200),
            "volatility": random.uniform(0.01, 0.05)
        }
        
        headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
        
        with self.client.post(
            "/api/v1/market-data/update",
            json=market_data,
            headers=headers,
            catch_response=True,
            name="market_data_update"
        ) as response:
            
            if response.status_code in [200, 201]:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(10)
    def trading_decision_feedback(self):
        """Test trading decision feedback loop (10% of requests)"""
        feedback_data = {
            "decision_id": str(uuid.uuid4()),
            "symbol": random.choice(self.trading_symbols),
            "outcome": random.choice(["profitable", "loss", "neutral"]),
            "pnl": random.uniform(-100, 200),
            "execution_quality": random.uniform(0.7, 1.0),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
        
        with self.client.post(
            "/api/v1/decisions/feedback",
            json=feedback_data,
            headers=headers,
            catch_response=True,
            name="decision_feedback"
        ) as response:
            
            if response.status_code in [200, 201]:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(5)
    def websocket_test(self):
        """Test WebSocket connection stability (5% of requests)"""
        if not self.ws_connected:
            return
        
        try:
            # Send ping message
            ping_data = {
                "type": "ping",
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": self.user_id
            }
            
            self.ws.send(json.dumps(ping_data))
            
            # Wait for response with timeout
            self.ws.settimeout(5.0)
            response = self.ws.recv()
            response_data = json.loads(response)
            
            if response_data.get("type") == "pong":
                events.request_success.fire(
                    request_type="websocket",
                    name="ping_pong",
                    response_time=100,  # Approximate
                    response_length=len(response)
                )
            else:
                events.request_failure.fire(
                    request_type="websocket",
                    name="ping_pong",
                    response_time=5000,
                    response_length=0,
                    exception="Invalid WebSocket response"
                )
                
        except Exception as e:
            events.request_failure.fire(
                request_type="websocket",
                name="ping_pong",
                response_time=5000,
                response_length=0,
                exception=str(e)
            )
    
    def validate_explanation(self, explanation: Dict[str, Any], latency: float) -> bool:
        """Validate explanation response quality"""
        required_fields = [
            "explanation_type",
            "decision_reasoning",
            "feature_importance",
            "confidence_intervals",
            "generation_time_ms"
        ]
        
        # Check required fields
        for field in required_fields:
            if field not in explanation:
                return False
        
        # Check explanation quality
        if len(explanation.get("decision_reasoning", "")) < 50:
            return False
        
        # Check feature importance
        feature_importance = explanation.get("feature_importance", {})
        if not feature_importance or len(feature_importance) < 5:
            return False
        
        # Check latency consistency
        reported_time = explanation.get("generation_time_ms", 0)
        if abs(reported_time - latency) > 50:  # 50ms tolerance
            return False
        
        return True
    
    def on_stop(self):
        """Cleanup when test stops"""
        if hasattr(self, 'ws') and self.ws_connected:
            try:
                self.ws.close()
            except (ConnectionError, OSError, TimeoutError) as e:
                logger.error(f'Error occurred: {e}')
        
        # Log performance statistics
        if self.explanation_metrics:
            avg_latency = np.mean(self.explanation_metrics)
            p95_latency = np.percentile(self.explanation_metrics, 95)
            p99_latency = np.percentile(self.explanation_metrics, 99)
            
            print(f"\nUser {self.user_id[:8]} Explanation Metrics:")
            print(f"  Average latency: {avg_latency:.2f}ms")
            print(f"  95th percentile: {p95_latency:.2f}ms")
            print(f"  99th percentile: {p99_latency:.2f}ms")
            print(f"  Total explanations: {len(self.explanation_metrics)}")
        
        if self.query_metrics:
            avg_query_time = np.mean(self.query_metrics)
            p95_query_time = np.percentile(self.query_metrics, 95)
            
            print(f"\nUser {self.user_id[:8]} Query Metrics:")
            print(f"  Average response time: {avg_query_time:.2f}ms")
            print(f"  95th percentile: {p95_query_time:.2f}ms")
            print(f"  Total queries: {len(self.query_metrics)}")


class HighFrequencyExplanationTest(FastHttpUser):
    """High-frequency explanation request test"""
    
    wait_time = between(0.01, 0.1)  # Very short wait times
    weight = 1
    
    @task
    def rapid_explanations(self):
        """Generate explanations at high frequency"""
        decision_data = {
            "symbol": "NQ",
            "action": "BUY",
            "confidence": 0.85,
            "market_features": [0.5] * 15,
            "feature_names": ["feature_" + str(i) for i in range(15)],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        start_time = time.time()
        
        with self.client.post(
            "/api/v1/explanations/generate",
            json=decision_data,
            catch_response=True,
            name="rapid_explanation"
        ) as response:
            
            latency = (time.time() - start_time) * 1000
            
            if response.status_code == 200 and latency <= 100:
                response.success()
            else:
                response.failure(f"Latency: {latency:.2f}ms")


class ConcurrentQueryTest(FastHttpUser):
    """Test concurrent complex queries"""
    
    wait_time = between(0.5, 2.0)
    weight = 1
    
    @task
    def complex_analytics_query(self):
        """Execute complex analytics queries"""
        query_data = {
            "query_type": "comprehensive_analysis",
            "time_range": {
                "start": (datetime.utcnow() - timedelta(hours=12)).isoformat(),
                "end": datetime.utcnow().isoformat()
            },
            "complexity": "high",
            "include_explanations": True,
            "symbols": ["NQ", "ES", "BTC", "ETH"]
        }
        
        start_time = time.time()
        
        with self.client.post(
            "/api/v1/analytics/complex-query",
            json=query_data,
            catch_response=True,
            name="complex_query"
        ) as response:
            
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200 and response_time <= 2000:
                response.success()
            else:
                response.failure(f"Response time: {response_time:.2f}ms")


# Event handlers for custom metrics
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize test environment"""
    print("🚀 Starting XAI Trading System Load Test")
    print("=" * 50)
    print("Performance Requirements:")
    print("  - Explanation latency: <100ms (95th percentile)")
    print("  - Query response time: <2 seconds (95th percentile)")
    print("  - System availability: >99.9%")
    print("=" * 50)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Test completion statistics"""
    print("\n" + "=" * 50)
    print("🏁 XAI Trading System Load Test Complete")
    print("=" * 50)
    
    stats = environment.stats
    
    # Explanation latency analysis
    explanation_stats = stats.get("/api/v1/explanations/generate", "POST")
    if explanation_stats:
        print(f"\n📊 Explanation Performance:")
        print(f"  Total requests: {explanation_stats.num_requests}")
        print(f"  Failures: {explanation_stats.num_failures}")
        print(f"  Average response time: {explanation_stats.avg_response_time:.2f}ms")
        print(f"  95th percentile: {explanation_stats.get_response_time_percentile(0.95):.2f}ms")
        print(f"  99th percentile: {explanation_stats.get_response_time_percentile(0.99):.2f}ms")
        print(f"  Max response time: {explanation_stats.max_response_time:.2f}ms")
        
        # Check if requirements are met
        p95_latency = explanation_stats.get_response_time_percentile(0.95)
        if p95_latency <= 100:
            print(f"  ✅ PASSED: 95th percentile latency {p95_latency:.2f}ms <= 100ms")
        else:
            print(f"  ❌ FAILED: 95th percentile latency {p95_latency:.2f}ms > 100ms")
    
    # Query performance analysis
    query_stats = stats.get("/api/v1/analytics/query", "POST")
    if query_stats:
        print(f"\n📈 Query Performance:")
        print(f"  Total requests: {query_stats.num_requests}")
        print(f"  Failures: {query_stats.num_failures}")
        print(f"  Average response time: {query_stats.avg_response_time:.2f}ms")
        print(f"  95th percentile: {query_stats.get_response_time_percentile(0.95):.2f}ms")
        
        # Check if requirements are met
        p95_query_time = query_stats.get_response_time_percentile(0.95)
        if p95_query_time <= 2000:
            print(f"  ✅ PASSED: 95th percentile query time {p95_query_time:.2f}ms <= 2000ms")
        else:
            print(f"  ❌ FAILED: 95th percentile query time {p95_query_time:.2f}ms > 2000ms")
    
    # Overall system health
    total_requests = stats.total.num_requests
    total_failures = stats.total.num_failures
    failure_rate = (total_failures / total_requests * 100) if total_requests > 0 else 0
    availability = 100 - failure_rate
    
    print(f"\n🏥 System Health:")
    print(f"  Total requests: {total_requests}")
    print(f"  Total failures: {total_failures}")
    print(f"  Failure rate: {failure_rate:.2f}%")
    print(f"  Availability: {availability:.2f}%")
    
    if availability >= 99.9:
        print(f"  ✅ PASSED: Availability {availability:.2f}% >= 99.9%")
    else:
        print(f"  ❌ FAILED: Availability {availability:.2f}% < 99.9%")
    
    print("\n" + "=" * 50)