"""
AI-Powered Test Generation Framework

This module provides intelligent test generation capabilities using:
- Hypothesis for property-based testing
- GPT-based test case generation
- Self-healing test maintenance
- Intelligent test repair mechanisms
"""

from .hypothesis_integration import (
    MarketDataStrategy,
    TradingSignalStrategy,
    RiskMetricsStrategy,
    PropertyBasedTestEngine,
    trading_invariants,
    risk_invariants,
    performance_invariants
)

from .gpt_generator import (
    GPTTestGenerator,
    TestCaseTemplate,
    TestGenerationRequest,
    GeneratedTestCase,
    TestComplexity,
    TestCategory
)

from .self_healing import (
    SelfHealingTestFramework,
    TestFailureAnalyzer,
    TestRepairEngine,
    TestMaintenanceScheduler,
    RepairStrategy,
    FailurePattern
)

from .intelligent_repair import (
    IntelligentTestRepair,
    TestCodeAnalyzer,
    RepairSuggestion,
    CodeChangeDetector,
    TestImpactAnalyzer
)

from .fixtures import (
    property_based_test_setup,
    generated_test_runner,
    self_healing_test_context,
    ai_test_generation_session
)

__all__ = [
    # Hypothesis Integration
    "MarketDataStrategy",
    "TradingSignalStrategy", 
    "RiskMetricsStrategy",
    "PropertyBasedTestEngine",
    "trading_invariants",
    "risk_invariants",
    "performance_invariants",
    
    # GPT Generation
    "GPTTestGenerator",
    "TestCaseTemplate",
    "TestGenerationRequest",
    "GeneratedTestCase",
    "TestComplexity",
    "TestCategory",
    
    # Self-Healing
    "SelfHealingTestFramework",
    "TestFailureAnalyzer",
    "TestRepairEngine",
    "TestMaintenanceScheduler",
    "RepairStrategy",
    "FailurePattern",
    
    # Intelligent Repair
    "IntelligentTestRepair",
    "TestCodeAnalyzer",
    "RepairSuggestion",
    "CodeChangeDetector",
    "TestImpactAnalyzer",
    
    # Fixtures
    "property_based_test_setup",
    "generated_test_runner",
    "self_healing_test_context",
    "ai_test_generation_session"
]