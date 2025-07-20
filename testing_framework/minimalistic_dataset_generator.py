#!/usr/bin/env python3
"""
COMPREHENSIVE TESTING FRAMEWORK - MINIMALISTIC DATASET GENERATOR
================================================================

This module generates lightweight, minimalistic datasets for rapid testing
of both Terminal 1 and Terminal 2 components. 

Key Features:
- Strategic Testing Data (30-minute timeframe): 100 samples (48×13 matrices)
- Tactical Testing Data (5-minute timeframe): 500 samples (60×7 matrices) 
- Risk Management Testing Data: Portfolio scenarios and stress test data
- Execution Engine Testing Data: Order flow simulation with latency testing
- Real-time simulation capabilities
- Validation criteria and quality checks
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

class MinimalisticDatasetGenerator:
    """
    Generates minimal but realistic datasets for comprehensive testing
    across all system components.
    """
    
    def __init__(self, base_path: str = "/home/QuantNova/GrandModel/testing_framework/test_data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Dataset configuration
        self.strategic_config = {
            "samples": 100,
            "matrix_shape": (48, 13),  # 48 time points, 13 features
            "timeframe": "30min",
            "features": ["open", "high", "low", "close", "volume", "rsi", "ema", 
                        "macd", "bollinger_upper", "bollinger_lower", "atr", "vwap", "momentum"]
        }
        
        self.tactical_config = {
            "samples": 500,
            "matrix_shape": (60, 7),   # 60 time points, 7 features
            "timeframe": "5min", 
            "features": ["open", "high", "low", "close", "volume", "price_change", "volume_change"]
        }
        
        self.risk_config = {
            "portfolio_scenarios": 50,
            "stress_test_scenarios": 20,
            "instruments": ["CL", "NQ", "ES", "GC", "EUR", "GBP", "JPY"]
        }
        
        self.execution_config = {
            "order_flow_samples": 1000,
            "latency_samples": 10000,
            "market_scenarios": ["normal", "volatile", "trending", "choppy"]
        }

    def generate_strategic_testing_data(self) -> Dict:
        """
        Generate strategic testing data for 30-minute timeframe analysis.
        
        Returns:
            Dict containing matrices, metadata, and validation info
        """
        print("🔄 Generating Strategic Testing Data (30-minute timeframe)...")
        
        # Generate base price data
        np.random.seed(42)  # Reproducible results
        base_price = 100.0
        
        strategic_data = {
            "matrices": [],
            "labels": [],
            "metadata": {},
            "validation_criteria": {}
        }
        
        for i in range(self.strategic_config["samples"]):
            # Generate realistic OHLCV data
            matrix = np.zeros(self.strategic_config["matrix_shape"])
            
            # Generate price movements with trend and volatility
            trend = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
            volatility = np.random.uniform(0.01, 0.05)
            
            prices = []
            current_price = base_price + np.random.normal(0, 2)
            
            for t in range(48):  # 48 time points
                # Price evolution with trend
                price_change = np.random.normal(trend * 0.001, volatility)
                current_price *= (1 + price_change)
                
                # OHLC generation
                high = current_price * (1 + np.random.uniform(0, 0.02))
                low = current_price * (1 - np.random.uniform(0, 0.02))
                open_price = current_price + np.random.normal(0, 0.5)
                close_price = current_price
                volume = np.random.lognormal(10, 1)
                
                # Technical indicators (simplified)
                rsi = 50 + np.random.normal(0, 20)
                rsi = np.clip(rsi, 0, 100)
                ema = current_price * (1 + np.random.normal(0, 0.01))
                macd = np.random.normal(0, 0.5)
                bb_upper = current_price * 1.02
                bb_lower = current_price * 0.98
                atr = volatility * current_price
                vwap = current_price * (1 + np.random.normal(0, 0.005))
                momentum = np.random.normal(0, 1)
                
                matrix[t] = [open_price, high, low, close_price, volume, rsi, ema,
                           macd, bb_upper, bb_lower, atr, vwap, momentum]
                
                prices.append(current_price)
            
            strategic_data["matrices"].append(matrix)
            
            # Generate labels (0: sell, 1: hold, 2: buy)
            final_return = (prices[-1] - prices[0]) / prices[0]
            if final_return > 0.02:
                label = 2  # buy
            elif final_return < -0.02:
                label = 0  # sell
            else:
                label = 1  # hold
            
            strategic_data["labels"].append(label)
        
        # Add metadata
        strategic_data["metadata"] = {
            "generation_time": datetime.now().isoformat(),
            "samples": self.strategic_config["samples"],
            "matrix_shape": self.strategic_config["matrix_shape"],
            "features": self.strategic_config["features"],
            "timeframe": self.strategic_config["timeframe"],
            "base_price": base_price
        }
        
        # Validation criteria
        strategic_data["validation_criteria"] = {
            "expected_shape": (self.strategic_config["samples"],) + self.strategic_config["matrix_shape"],
            "price_range": [50, 200],
            "volume_positive": True,
            "rsi_range": [0, 100],
            "no_nan_values": True,
            "label_distribution": {"0": 0.3, "1": 0.4, "2": 0.3}  # Expected proportions
        }
        
        return strategic_data

    def generate_tactical_testing_data(self) -> Dict:
        """
        Generate tactical testing data for 5-minute timeframe analysis.
        
        Returns:
            Dict containing matrices, metadata, and validation info
        """
        print("🔄 Generating Tactical Testing Data (5-minute timeframe)...")
        
        np.random.seed(43)  # Different seed for tactical data
        base_price = 100.0
        
        tactical_data = {
            "matrices": [],
            "labels": [],
            "metadata": {},
            "validation_criteria": {}
        }
        
        for i in range(self.tactical_config["samples"]):
            matrix = np.zeros(self.tactical_config["matrix_shape"])
            
            # High-frequency price movements
            current_price = base_price + np.random.normal(0, 1)
            
            for t in range(60):  # 60 time points (5 hours of 5-min data)
                # Faster price movements for tactical trading
                price_change = np.random.normal(0, 0.002)  # Smaller movements
                current_price *= (1 + price_change)
                
                # OHLC for 5-min bar
                high = current_price * (1 + np.random.uniform(0, 0.005))
                low = current_price * (1 - np.random.uniform(0, 0.005))
                open_price = current_price + np.random.normal(0, 0.1)
                close_price = current_price
                volume = np.random.lognormal(8, 0.5)  # Lower volume for 5-min
                
                # Price and volume changes
                if t > 0:
                    price_change_pct = (close_price - matrix[t-1, 3]) / matrix[t-1, 3] if matrix[t-1, 3] != 0 else 0
                    volume_change_pct = (volume - matrix[t-1, 4]) / matrix[t-1, 4] if matrix[t-1, 4] != 0 else 0
                else:
                    price_change_pct = 0
                    volume_change_pct = 0
                
                matrix[t] = [open_price, high, low, close_price, volume, 
                           price_change_pct, volume_change_pct]
            
            tactical_data["matrices"].append(matrix)
            
            # Generate tactical labels (more frequent signals)
            recent_returns = []
            for t in range(1, min(10, 60)):  # Look at last 10 bars
                if matrix[t-1, 3] != 0:
                    ret = (matrix[t, 3] - matrix[t-1, 3]) / matrix[t-1, 3]
                    recent_returns.append(ret)
            
            avg_return = np.mean(recent_returns) if recent_returns else 0
            
            if avg_return > 0.001:
                label = 2  # buy
            elif avg_return < -0.001:
                label = 0  # sell
            else:
                label = 1  # hold
            
            tactical_data["labels"].append(label)
        
        # Add metadata
        tactical_data["metadata"] = {
            "generation_time": datetime.now().isoformat(),
            "samples": self.tactical_config["samples"],
            "matrix_shape": self.tactical_config["matrix_shape"],
            "features": self.tactical_config["features"],
            "timeframe": self.tactical_config["timeframe"],
            "base_price": base_price
        }
        
        # Validation criteria
        tactical_data["validation_criteria"] = {
            "expected_shape": (self.tactical_config["samples"],) + self.tactical_config["matrix_shape"],
            "price_range": [95, 105],  # Tighter range for 5-min data
            "volume_positive": True,
            "no_nan_values": True,
            "price_change_range": [-0.1, 0.1],
            "label_distribution": {"0": 0.35, "1": 0.3, "2": 0.35}
        }
        
        return tactical_data

    def generate_risk_management_testing_data(self) -> Dict:
        """
        Generate risk management testing data including portfolio scenarios
        and stress test data.
        """
        print("🔄 Generating Risk Management Testing Data...")
        
        np.random.seed(44)
        
        risk_data = {
            "portfolio_scenarios": [],
            "stress_test_scenarios": [],
            "var_scenarios": [],
            "correlation_matrices": [],
            "metadata": {},
            "validation_criteria": {}
        }
        
        # Generate portfolio scenarios
        for i in range(self.risk_config["portfolio_scenarios"]):
            scenario = {
                "portfolio_id": f"portfolio_{i:03d}",
                "positions": {},
                "total_value": np.random.uniform(100000, 10000000),
                "leverage": np.random.uniform(1.0, 5.0),
                "risk_metrics": {}
            }
            
            # Generate positions for each instrument
            for instrument in self.risk_config["instruments"]:
                position_size = np.random.uniform(-1000000, 1000000)
                current_price = np.random.uniform(50, 200)
                
                scenario["positions"][instrument] = {
                    "size": position_size,
                    "price": current_price,
                    "value": position_size * current_price,
                    "volatility": np.random.uniform(0.1, 0.5),
                    "beta": np.random.uniform(0.5, 2.0)
                }
            
            # Calculate risk metrics
            portfolio_volatility = np.random.uniform(0.15, 0.35)
            var_95 = scenario["total_value"] * portfolio_volatility * 1.65  # 95% VaR
            expected_shortfall = var_95 * 1.3
            
            scenario["risk_metrics"] = {
                "portfolio_volatility": portfolio_volatility,
                "var_95": var_95,
                "expected_shortfall": expected_shortfall,
                "sharpe_ratio": np.random.uniform(0.5, 2.0),
                "max_drawdown": np.random.uniform(0.05, 0.25)
            }
            
            risk_data["portfolio_scenarios"].append(scenario)
        
        # Generate stress test scenarios
        stress_scenarios = [
            {"name": "market_crash", "returns": np.random.normal(-0.1, 0.05, 252).tolist()},
            {"name": "volatility_spike", "volatility_multiplier": float(np.random.uniform(2, 5))},
            {"name": "liquidity_crisis", "liquidity_factor": float(np.random.uniform(0.1, 0.5))},
            {"name": "correlation_breakdown", "correlation_shock": float(np.random.uniform(-0.5, 0.5))}
        ]
        
        for scenario in stress_scenarios:
            for i in range(self.risk_config["stress_test_scenarios"] // 4):
                stress_test = scenario.copy()
                stress_test["scenario_id"] = f"{scenario['name']}_{i:02d}"
                stress_test["severity"] = np.random.choice(["mild", "moderate", "severe"])
                risk_data["stress_test_scenarios"].append(stress_test)
        
        # Generate correlation matrices
        n_instruments = len(self.risk_config["instruments"])
        for i in range(20):  # 20 correlation scenarios
            # Generate random correlation matrix
            A = np.random.randn(n_instruments, n_instruments)
            corr_matrix = np.corrcoef(A)
            
            risk_data["correlation_matrices"].append({
                "matrix_id": f"corr_{i:02d}",
                "matrix": corr_matrix.tolist(),
                "instruments": self.risk_config["instruments"],
                "regime": np.random.choice(["normal", "crisis", "recovery"])
            })
        
        # Add metadata
        risk_data["metadata"] = {
            "generation_time": datetime.now().isoformat(),
            "portfolio_scenarios": self.risk_config["portfolio_scenarios"],
            "stress_scenarios": len(risk_data["stress_test_scenarios"]),
            "instruments": self.risk_config["instruments"]
        }
        
        # Validation criteria
        risk_data["validation_criteria"] = {
            "portfolio_count": self.risk_config["portfolio_scenarios"],
            "stress_test_count": self.risk_config["stress_test_scenarios"],
            "var_positive": True,
            "correlation_valid_range": [-1, 1],
            "portfolio_value_positive": True
        }
        
        return risk_data

    def generate_execution_engine_testing_data(self) -> Dict:
        """
        Generate execution engine testing data including order flow simulation
        and latency testing data.
        """
        print("🔄 Generating Execution Engine Testing Data...")
        
        np.random.seed(45)
        
        execution_data = {
            "order_flow": [],
            "latency_measurements": [],
            "market_impact_scenarios": [],
            "execution_quality_metrics": [],
            "metadata": {},
            "validation_criteria": {}
        }
        
        # Generate order flow data
        for i in range(self.execution_config["order_flow_samples"]):
            order = {
                "order_id": f"order_{i:06d}",
                "timestamp": datetime.now() + timedelta(microseconds=i*100),
                "instrument": np.random.choice(self.risk_config["instruments"]),
                "side": np.random.choice(["buy", "sell"]),
                "quantity": np.random.randint(1, 1000),
                "order_type": np.random.choice(["market", "limit", "stop"]),
                "price": np.random.uniform(50, 200),
                "urgency": np.random.choice(["low", "medium", "high"]),
                "execution_time": np.random.uniform(0.1, 50.0),  # milliseconds
                "slippage": np.random.normal(0, 0.002),
                "market_scenario": np.random.choice(self.execution_config["market_scenarios"])
            }
            
            execution_data["order_flow"].append(order)
        
        # Generate latency measurements for sub-millisecond testing
        for i in range(self.execution_config["latency_samples"]):
            latency_sample = {
                "measurement_id": f"latency_{i:06d}",
                "component": np.random.choice(["order_validation", "risk_check", "routing", "execution"]),
                "latency_microseconds": np.random.exponential(200),  # Exponential distribution
                "timestamp": datetime.now() + timedelta(microseconds=i),
                "cpu_usage": np.random.uniform(0.1, 0.9),
                "memory_usage": np.random.uniform(0.2, 0.8),
                "network_latency": np.random.exponential(50)
            }
            
            execution_data["latency_measurements"].append(latency_sample)
        
        # Generate market impact scenarios
        for scenario in self.execution_config["market_scenarios"]:
            for i in range(25):  # 25 scenarios per market type
                impact_scenario = {
                    "scenario_id": f"{scenario}_impact_{i:02d}",
                    "market_scenario": scenario,
                    "order_size": np.random.uniform(100, 10000),
                    "market_depth": np.random.uniform(10000, 100000),
                    "volatility": np.random.uniform(0.1, 0.5),
                    "expected_impact": np.random.uniform(0.001, 0.01),
                    "actual_impact": np.random.uniform(0.001, 0.01),
                    "execution_shortfall": np.random.normal(0, 0.002)
                }
                
                execution_data["market_impact_scenarios"].append(impact_scenario)
        
        # Generate execution quality metrics
        for i in range(200):
            quality_metric = {
                "metric_id": f"quality_{i:03d}",
                "arrival_price": np.random.uniform(100, 200),
                "execution_price": np.random.uniform(100, 200),
                "benchmark_price": np.random.uniform(100, 200),
                "implementation_shortfall": np.random.normal(0, 0.001),
                "market_impact": np.random.uniform(0, 0.005),
                "timing_cost": np.random.normal(0, 0.001),
                "commission": np.random.uniform(0.001, 0.01),
                "total_cost": np.random.uniform(0.005, 0.02)
            }
            
            execution_data["execution_quality_metrics"].append(quality_metric)
        
        # Add metadata
        execution_data["metadata"] = {
            "generation_time": datetime.now().isoformat(),
            "order_flow_samples": self.execution_config["order_flow_samples"],
            "latency_samples": self.execution_config["latency_samples"],
            "market_scenarios": self.execution_config["market_scenarios"]
        }
        
        # Validation criteria
        execution_data["validation_criteria"] = {
            "order_count": self.execution_config["order_flow_samples"],
            "latency_count": self.execution_config["latency_samples"],
            "latency_target_microseconds": 500,  # Sub-millisecond target
            "slippage_range": [-0.01, 0.01],
            "execution_time_range": [0.1, 100],  # milliseconds
            "no_negative_quantities": True
        }
        
        return execution_data

    def validate_dataset_quality(self, data: Dict, data_type: str) -> Dict:
        """
        Validate the quality of generated datasets against validation criteria.
        
        Args:
            data: Generated dataset
            data_type: Type of dataset (strategic, tactical, risk, execution)
            
        Returns:
            Validation report
        """
        print(f"🔍 Validating {data_type} dataset quality...")
        
        validation_report = {
            "data_type": data_type,
            "validation_time": datetime.now().isoformat(),
            "passed": True,
            "checks": {},
            "summary": {}
        }
        
        criteria = data.get("validation_criteria", {})
        
        if data_type == "strategic":
            matrices = np.array(data["matrices"])
            labels = np.array(data["labels"])
            
            # Shape validation
            expected_shape = criteria.get("expected_shape")
            actual_shape = matrices.shape
            validation_report["checks"]["shape_check"] = {
                "expected": expected_shape,
                "actual": actual_shape,
                "passed": actual_shape == expected_shape
            }
            
            # NaN validation
            has_nan = np.isnan(matrices).any()
            validation_report["checks"]["nan_check"] = {
                "has_nan": has_nan,
                "passed": not has_nan
            }
            
            # Price range validation
            price_range = criteria.get("price_range", [0, 1000])
            price_cols = [0, 1, 2, 3]  # OHLC columns
            prices = matrices[:, :, price_cols]
            in_range = np.all((prices >= price_range[0]) & (prices <= price_range[1]))
            validation_report["checks"]["price_range_check"] = {
                "range": price_range,
                "in_range": in_range,
                "passed": in_range
            }
            
        elif data_type == "tactical":
            matrices = np.array(data["matrices"])
            
            # Shape validation
            expected_shape = criteria.get("expected_shape")
            actual_shape = matrices.shape
            validation_report["checks"]["shape_check"] = {
                "expected": expected_shape,
                "actual": actual_shape,
                "passed": actual_shape == expected_shape
            }
            
            # Price change validation
            price_change_range = criteria.get("price_change_range", [-1, 1])
            price_changes = matrices[:, :, 5]  # Price change column
            in_range = np.all((price_changes >= price_change_range[0]) & 
                            (price_changes <= price_change_range[1]))
            validation_report["checks"]["price_change_check"] = {
                "range": price_change_range,
                "in_range": in_range,
                "passed": in_range
            }
            
        elif data_type == "risk":
            portfolio_count = len(data.get("portfolio_scenarios", []))
            expected_count = criteria.get("portfolio_count", 0)
            
            validation_report["checks"]["portfolio_count_check"] = {
                "expected": expected_count,
                "actual": portfolio_count,
                "passed": portfolio_count == expected_count
            }
            
            # VaR positivity check
            var_values = [scenario["risk_metrics"]["var_95"] 
                         for scenario in data.get("portfolio_scenarios", [])]
            all_positive = all(var > 0 for var in var_values)
            validation_report["checks"]["var_positive_check"] = {
                "all_positive": all_positive,
                "passed": all_positive
            }
            
        elif data_type == "execution":
            order_count = len(data.get("order_flow", []))
            expected_count = criteria.get("order_count", 0)
            
            validation_report["checks"]["order_count_check"] = {
                "expected": expected_count,
                "actual": order_count,
                "passed": order_count == expected_count
            }
            
            # Latency validation
            latencies = [sample["latency_microseconds"] 
                        for sample in data.get("latency_measurements", [])]
            target_latency = criteria.get("latency_target_microseconds", 500)
            within_target = sum(1 for lat in latencies if lat <= target_latency)
            target_percentage = within_target / len(latencies) if latencies else 0
            
            validation_report["checks"]["latency_check"] = {
                "target_microseconds": target_latency,
                "within_target_percentage": target_percentage,
                "passed": target_percentage >= 0.8  # 80% should be within target
            }
        
        # Overall validation result
        all_checks_passed = all(check["passed"] for check in validation_report["checks"].values())
        validation_report["passed"] = all_checks_passed
        
        # Summary
        validation_report["summary"] = {
            "total_checks": len(validation_report["checks"]),
            "passed_checks": sum(1 for check in validation_report["checks"].values() if check["passed"]),
            "overall_passed": all_checks_passed
        }
        
        return validation_report

    def save_datasets(self, strategic_data: Dict, tactical_data: Dict, 
                     risk_data: Dict, execution_data: Dict) -> None:
        """
        Save all generated datasets to files.
        """
        print("💾 Saving datasets to files...")
        
        # Create subdirectories
        (self.base_path / "strategic").mkdir(exist_ok=True)
        (self.base_path / "tactical").mkdir(exist_ok=True)
        (self.base_path / "risk_management").mkdir(exist_ok=True)
        (self.base_path / "execution_engine").mkdir(exist_ok=True)
        
        # Save strategic data
        np.save(self.base_path / "strategic" / "matrices.npy", strategic_data["matrices"])
        np.save(self.base_path / "strategic" / "labels.npy", strategic_data["labels"])
        with open(self.base_path / "strategic" / "metadata.json", "w") as f:
            json.dump(strategic_data["metadata"], f, indent=2, cls=NumpyEncoder)
        with open(self.base_path / "strategic" / "validation_criteria.json", "w") as f:
            json.dump(strategic_data["validation_criteria"], f, indent=2, cls=NumpyEncoder)
        
        # Save tactical data
        np.save(self.base_path / "tactical" / "matrices.npy", tactical_data["matrices"])
        np.save(self.base_path / "tactical" / "labels.npy", tactical_data["labels"])
        with open(self.base_path / "tactical" / "metadata.json", "w") as f:
            json.dump(tactical_data["metadata"], f, indent=2, cls=NumpyEncoder)
        with open(self.base_path / "tactical" / "validation_criteria.json", "w") as f:
            json.dump(tactical_data["validation_criteria"], f, indent=2, cls=NumpyEncoder)
        
        # Save risk management data
        with open(self.base_path / "risk_management" / "portfolio_scenarios.json", "w") as f:
            json.dump(risk_data["portfolio_scenarios"], f, indent=2, cls=NumpyEncoder)
        with open(self.base_path / "risk_management" / "stress_test_scenarios.json", "w") as f:
            json.dump(risk_data["stress_test_scenarios"], f, indent=2, cls=NumpyEncoder)
        with open(self.base_path / "risk_management" / "correlation_matrices.json", "w") as f:
            json.dump(risk_data["correlation_matrices"], f, indent=2, cls=NumpyEncoder)
        with open(self.base_path / "risk_management" / "metadata.json", "w") as f:
            json.dump(risk_data["metadata"], f, indent=2, cls=NumpyEncoder)
        
        # Save execution engine data
        with open(self.base_path / "execution_engine" / "order_flow.json", "w") as f:
            json.dump(execution_data["order_flow"], f, indent=2, cls=NumpyEncoder, default=str)
        with open(self.base_path / "execution_engine" / "latency_measurements.json", "w") as f:
            json.dump(execution_data["latency_measurements"], f, indent=2, cls=NumpyEncoder, default=str)
        with open(self.base_path / "execution_engine" / "market_impact_scenarios.json", "w") as f:
            json.dump(execution_data["market_impact_scenarios"], f, indent=2, cls=NumpyEncoder)
        with open(self.base_path / "execution_engine" / "metadata.json", "w") as f:
            json.dump(execution_data["metadata"], f, indent=2, cls=NumpyEncoder)

    def generate_all_datasets(self) -> Dict:
        """
        Generate all testing datasets and return comprehensive report.
        """
        print("🚀 Starting Comprehensive Dataset Generation...")
        print("=" * 60)
        
        # Generate all datasets
        strategic_data = self.generate_strategic_testing_data()
        tactical_data = self.generate_tactical_testing_data()
        risk_data = self.generate_risk_management_testing_data()
        execution_data = self.generate_execution_engine_testing_data()
        
        # Validate datasets
        strategic_validation = self.validate_dataset_quality(strategic_data, "strategic")
        tactical_validation = self.validate_dataset_quality(tactical_data, "tactical")
        risk_validation = self.validate_dataset_quality(risk_data, "risk")
        execution_validation = self.validate_dataset_quality(execution_data, "execution")
        
        # Save datasets
        self.save_datasets(strategic_data, tactical_data, risk_data, execution_data)
        
        # Generate comprehensive report
        report = {
            "generation_time": datetime.now().isoformat(),
            "datasets": {
                "strategic": {
                    "samples": len(strategic_data["matrices"]),
                    "shape": strategic_data["metadata"]["matrix_shape"],
                    "features": strategic_data["metadata"]["features"],
                    "validation": strategic_validation
                },
                "tactical": {
                    "samples": len(tactical_data["matrices"]),
                    "shape": tactical_data["metadata"]["matrix_shape"],
                    "features": tactical_data["metadata"]["features"],
                    "validation": tactical_validation
                },
                "risk_management": {
                    "portfolio_scenarios": len(risk_data["portfolio_scenarios"]),
                    "stress_tests": len(risk_data["stress_test_scenarios"]),
                    "correlation_matrices": len(risk_data["correlation_matrices"]),
                    "validation": risk_validation
                },
                "execution_engine": {
                    "order_flow_samples": len(execution_data["order_flow"]),
                    "latency_samples": len(execution_data["latency_measurements"]),
                    "market_scenarios": len(execution_data["market_impact_scenarios"]),
                    "validation": execution_validation
                }
            },
            "file_locations": {
                "strategic": str(self.base_path / "strategic"),
                "tactical": str(self.base_path / "tactical"),
                "risk_management": str(self.base_path / "risk_management"),
                "execution_engine": str(self.base_path / "execution_engine")
            },
            "overall_validation": {
                "all_passed": all([
                    strategic_validation["passed"],
                    tactical_validation["passed"],
                    risk_validation["passed"],
                    execution_validation["passed"]
                ])
            }
        }
        
        # Save comprehensive report
        with open(self.base_path / "generation_report.json", "w") as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        
        print("✅ Dataset generation completed successfully!")
        print(f"📁 All files saved to: {self.base_path}")
        print(f"📊 Report saved to: {self.base_path}/generation_report.json")
        
        return report

def main():
    """Main function to generate all testing datasets."""
    generator = MinimalisticDatasetGenerator()
    report = generator.generate_all_datasets()
    
    print("\n" + "=" * 60)
    print("📈 DATASET GENERATION SUMMARY")
    print("=" * 60)
    
    for dataset_name, dataset_info in report["datasets"].items():
        validation_status = "✅ PASSED" if dataset_info["validation"]["passed"] else "❌ FAILED"
        print(f"{dataset_name.upper()}: {validation_status}")
        
        if "samples" in dataset_info:
            print(f"  • Samples: {dataset_info['samples']}")
        if "shape" in dataset_info:
            print(f"  • Shape: {dataset_info['shape']}")
        if "portfolio_scenarios" in dataset_info:
            print(f"  • Portfolio Scenarios: {dataset_info['portfolio_scenarios']}")
            print(f"  • Stress Tests: {dataset_info['stress_tests']}")
        if "order_flow_samples" in dataset_info:
            print(f"  • Order Flow Samples: {dataset_info['order_flow_samples']}")
            print(f"  • Latency Samples: {dataset_info['latency_samples']}")
        print()
    
    overall_status = "✅ ALL PASSED" if report["overall_validation"]["all_passed"] else "❌ SOME FAILED"
    print(f"OVERALL VALIDATION: {overall_status}")

if __name__ == "__main__":
    main()