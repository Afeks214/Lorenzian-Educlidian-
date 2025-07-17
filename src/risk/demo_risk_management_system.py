"""
Risk Management System Demonstration
====================================

This script demonstrates the comprehensive risk management system with:
- Position sizing optimization
- Portfolio risk assessment
- Real-time monitoring
- Stress testing
- Model validation

Author: Risk Management System
Date: 2025-07-17
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the risk management system
from risk_management_system import (
    ComprehensiveRiskManagementSystem,
    SystemConfig,
    quick_risk_assessment,
    quick_position_sizing,
    quick_stress_test
)

from position_sizing.advanced_position_sizing import (
    TradeOpportunity,
    PortfolioState,
    MarketCondition,
    PositionSizingMethod
)


def generate_sample_data():
    """Generate sample data for demonstration"""
    
    # Generate sample returns
    np.random.seed(42)
    n_days = 252
    n_assets = 5
    
    returns = np.random.multivariate_normal(
        mean=[0.0005] * n_assets,
        cov=np.eye(n_assets) * 0.0004 + np.ones((n_assets, n_assets)) * 0.0001,
        size=n_days
    )
    
    # Create sample positions
    positions = {
        'AAPL': {'size': 0.20, 'weight': 0.20, 'entry_price': 150.0, 'entry_time': datetime.now() - timedelta(days=30)},
        'GOOGL': {'size': 0.15, 'weight': 0.15, 'entry_price': 2800.0, 'entry_time': datetime.now() - timedelta(days=20)},
        'MSFT': {'size': 0.25, 'weight': 0.25, 'entry_price': 300.0, 'entry_time': datetime.now() - timedelta(days=15)},
        'AMZN': {'size': 0.20, 'weight': 0.20, 'entry_price': 3200.0, 'entry_time': datetime.now() - timedelta(days=10)},
        'TSLA': {'size': 0.20, 'weight': 0.20, 'entry_price': 800.0, 'entry_time': datetime.now() - timedelta(days=5)}
    }
    
    # Create market data
    market_data = {
        'AAPL': {'price': 155.0, 'volatility': 0.25, 'beta': 1.2, 'liquidity_risk': 0.1},
        'GOOGL': {'price': 2900.0, 'volatility': 0.30, 'beta': 1.1, 'liquidity_risk': 0.1},
        'MSFT': {'price': 310.0, 'volatility': 0.22, 'beta': 1.0, 'liquidity_risk': 0.1},
        'AMZN': {'price': 3300.0, 'volatility': 0.35, 'beta': 1.3, 'liquidity_risk': 0.1},
        'TSLA': {'price': 850.0, 'volatility': 0.45, 'beta': 1.5, 'liquidity_risk': 0.2}
    }
    
    return returns, positions, market_data


def create_sample_trade_opportunity():
    """Create sample trade opportunity"""
    
    return TradeOpportunity(
        symbol='NVDA',
        signal_confidence=0.85,
        expected_return=0.15,
        expected_volatility=0.35,
        stop_loss_distance=0.05,
        take_profit_distance=0.10,
        win_probability=0.65,
        historical_performance={
            'wins': 13,
            'losses': 7,
            'avg_win': 0.12,
            'avg_loss': 0.04
        },
        sector='Technology',
        market_cap=2000000000000,
        liquidity_score=0.9
    )


def create_sample_portfolio_state():
    """Create sample portfolio state"""
    
    return PortfolioState(
        total_value=1000000.0,
        cash_available=200000.0,
        positions={'AAPL': 0.20, 'GOOGL': 0.15, 'MSFT': 0.25, 'AMZN': 0.20, 'TSLA': 0.20},
        sector_exposures={'Technology': 0.80, 'Consumer': 0.20},
        current_leverage=1.2,
        current_heat=0.12,
        correlation_matrix=np.eye(5) * 0.7 + np.ones((5, 5)) * 0.3,
        sector_correlations={'Technology': 0.6, 'Consumer': 0.4},
        recent_performance={'win_rate': 0.60, 'avg_return': 0.08, 'sharpe_ratio': 1.2}
    )


def create_sample_market_condition():
    """Create sample market condition"""
    
    return MarketCondition(
        volatility=0.25,
        trend_strength=0.7,
        market_regime='trending',
        correlation_level=0.45,
        liquidity_score=0.8
    )


async def demonstrate_position_sizing():
    """Demonstrate position sizing capabilities"""
    
    print("\n" + "="*60)
    print("POSITION SIZING DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    opportunity = create_sample_trade_opportunity()
    portfolio_state = create_sample_portfolio_state()
    market_condition = create_sample_market_condition()
    
    print(f"Trade Opportunity: {opportunity.symbol}")
    print(f"Signal Confidence: {opportunity.signal_confidence:.2%}")
    print(f"Expected Return: {opportunity.expected_return:.2%}")
    print(f"Expected Volatility: {opportunity.expected_volatility:.2%}")
    print(f"Win Probability: {opportunity.win_probability:.2%}")
    
    # Use quick position sizing
    result = await quick_position_sizing(opportunity, portfolio_state, market_condition)
    
    print("\nPosition Sizing Results:")
    print(f"Recommended Position Size: {result['position_sizing_result']['position_size']:.4f}")
    print(f"Risk Amount: ${result['position_sizing_result']['risk_amount']:,.2f}")
    print(f"Confidence Score: {result['position_sizing_result']['confidence_score']:.2%}")
    print(f"Method Used: {result['position_sizing_result']['method_used']}")
    
    print("\nRisk Metrics:")
    for metric, value in result['risk_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nRisk Checks:")
    for check, passed in result['risk_checks'].items():
        status = "PASS" if passed else "FAIL"
        print(f"  {check}: {status}")


async def demonstrate_portfolio_risk_assessment():
    """Demonstrate portfolio risk assessment"""
    
    print("\n" + "="*60)
    print("PORTFOLIO RISK ASSESSMENT DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    returns, positions, market_data = generate_sample_data()
    portfolio_value = 1000000.0
    
    print(f"Portfolio Value: ${portfolio_value:,.2f}")
    print(f"Number of Positions: {len(positions)}")
    print(f"Historical Data Points: {len(returns)}")
    
    # Use quick risk assessment
    assessment = await quick_risk_assessment(positions, market_data, portfolio_value)
    
    print("\nPortfolio Risk Summary:")
    portfolio_risk = assessment['portfolio_risk']
    print(f"Total Risk: {portfolio_risk['total_risk']:.4f}")
    print(f"VaR (95%): {portfolio_risk['var_95']:.4f}")
    print(f"Current Drawdown: {portfolio_risk['current_drawdown']:.2%}")
    print(f"Leverage: {portfolio_risk['leverage']:.2f}x")
    
    print("\nPortfolio Heat Analysis:")
    portfolio_heat = assessment['portfolio_heat']
    print(f"Total Heat: {portfolio_heat['total_heat']:.4f}")
    print(f"Correlation Heat: {portfolio_heat['correlation_heat']:.4f}")
    print(f"Concentration Heat: {portfolio_heat['concentration_heat']:.4f}")
    
    print("\nCorrelation Analysis:")
    correlation_analysis = assessment['correlation_analysis']
    print(f"Average Correlation: {correlation_analysis['average_correlation']:.2%}")
    print(f"Max Correlation: {correlation_analysis['max_correlation']:.2%}")
    print(f"Correlation Regime: {correlation_analysis['correlation_regime']}")
    
    print(f"\nSystem Health: {assessment['system_health']:.2%}")


async def demonstrate_stress_testing():
    """Demonstrate stress testing capabilities"""
    
    print("\n" + "="*60)
    print("STRESS TESTING DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    returns, positions, market_data = generate_sample_data()
    
    scenarios = ['market_crash', 'volatility_spike', 'correlation_breakdown']
    
    for scenario in scenarios:
        print(f"\nRunning Stress Test: {scenario}")
        print("-" * 40)
        
        result = await quick_stress_test(scenario, positions, market_data)
        
        stress_result = result['stress_test_result']
        print(f"Portfolio Loss: {stress_result['portfolio_loss']:.4f}")
        print(f"Recovery Time: {stress_result['recovery_time']:.1f} days")
        print(f"Correlation Impact: {stress_result['correlation_impact']:.2%}")
        print(f"Liquidity Impact: {stress_result['liquidity_impact']:.2%}")


async def demonstrate_comprehensive_system():
    """Demonstrate the comprehensive system"""
    
    print("\n" + "="*60)
    print("COMPREHENSIVE SYSTEM DEMONSTRATION")
    print("="*60)
    
    # Create system
    config = SystemConfig(
        auto_start_monitoring=True,
        enable_real_time_alerts=True,
        enable_validation=True
    )
    
    system = ComprehensiveRiskManagementSystem(config)
    
    # Start system
    print("Starting risk management system...")
    await system.start_system()
    
    # Get system status
    status = system.get_system_status()
    print(f"\nSystem Status:")
    print(f"System Health: {status.system_health:.2%}")
    print(f"Active Alerts: {status.active_alerts}")
    print(f"Uptime: {status.uptime.total_seconds():.1f} seconds")
    
    # Test position sizing
    print("\nTesting position sizing...")
    opportunity = create_sample_trade_opportunity()
    portfolio_state = create_sample_portfolio_state()
    market_condition = create_sample_market_condition()
    
    position_result = await system.calculate_position_size(
        opportunity, portfolio_state, market_condition
    )
    
    print(f"Position Size: {position_result['position_sizing_result']['position_size']:.4f}")
    print(f"Decision: {position_result['decision']['recommended_action']}")
    
    # Test portfolio risk update
    print("\nUpdating portfolio risk...")
    returns, positions, market_data = generate_sample_data()
    
    risk_update = await system.update_portfolio_risk(
        positions, market_data, 1000000.0
    )
    
    print(f"Portfolio VaR: {risk_update['portfolio_risk']['var_95']:.4f}")
    print(f"Portfolio Heat: {risk_update['portfolio_heat']['total_heat']:.4f}")
    
    # Test stress testing
    print("\nRunning stress test...")
    stress_result = await system.run_stress_test('market_crash', positions, market_data)
    
    print(f"Stress Test Loss: {stress_result['stress_test_result']['portfolio_loss']:.4f}")
    
    # Test model validation
    print("\nValidating risk models...")
    returns_1d = np.random.randn(100) * 0.02
    var_forecasts = {
        'historical': np.random.rand(100) * 0.05,
        'parametric': np.random.rand(100) * 0.05
    }
    
    validation_result = await system.validate_risk_models(returns_1d, var_forecasts)
    print(f"Validation Score: {validation_result['validation_report']['overall_score']:.2f}")
    
    # Get comprehensive dashboard
    print("\nGenerating comprehensive dashboard...")
    dashboard = system.get_comprehensive_dashboard()
    
    print(f"Dashboard Components: {len(dashboard)}")
    print(f"System Components: {len(dashboard['system_status']['components_status'])}")
    print(f"Recent Decisions: {len(dashboard['recent_decisions'])}")
    
    # Stop system
    print("\nStopping risk management system...")
    await system.stop_system()
    
    print("System demonstration completed successfully!")


async def demonstrate_performance_benchmarks():
    """Demonstrate performance benchmarks"""
    
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARKS")
    print("="*60)
    
    # Create system
    system = ComprehensiveRiskManagementSystem(SystemConfig())
    
    # Benchmark Numba engine
    print("Benchmarking Numba JIT optimization...")
    benchmark_results = system.numba_engine.benchmark_performance(
        data_size=10000,
        n_assets=100,
        iterations=100
    )
    
    print(f"VaR Calculation: {benchmark_results['var_time_ms']:.2f}ms")
    print(f"CVaR Calculation: {benchmark_results['cvar_time_ms']:.2f}ms")
    print(f"Correlation Matrix: {benchmark_results['correlation_time_ms']:.2f}ms")
    print(f"Portfolio Risk: {benchmark_results['portfolio_risk_time_ms']:.2f}ms")
    
    # Performance stats
    print("\nPerformance Statistics:")
    perf_stats = system.numba_engine.get_performance_stats()
    
    for metric, stats in perf_stats.items():
        if isinstance(stats, dict) and 'avg_time_ms' in stats:
            print(f"  {metric}: {stats['avg_time_ms']:.2f}ms avg")


async def main():
    """Main demonstration function"""
    
    print("COMPREHENSIVE RISK MANAGEMENT SYSTEM DEMONSTRATION")
    print("=" * 80)
    print("This demonstration showcases the institutional-grade risk management system")
    print("with advanced position sizing, real-time monitoring, and comprehensive analytics.")
    print("=" * 80)
    
    try:
        # Run all demonstrations
        await demonstrate_position_sizing()
        await demonstrate_portfolio_risk_assessment()
        await demonstrate_stress_testing()
        await demonstrate_comprehensive_system()
        await demonstrate_performance_benchmarks()
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("The risk management system has been successfully demonstrated with:")
        print("✓ Advanced position sizing with Kelly Criterion optimization")
        print("✓ Comprehensive portfolio risk assessment")
        print("✓ Real-time monitoring and alerting")
        print("✓ Stress testing and scenario analysis")
        print("✓ Model validation and compliance checking")
        print("✓ Numba JIT optimized calculations")
        print("✓ Portfolio heat and correlation controls")
        print("✓ Institutional-grade performance")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"ERROR: {e}")


if __name__ == "__main__":
    asyncio.run(main())