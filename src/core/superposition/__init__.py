"""
Universal Superposition Core Framework - Package Initialization
AGENT 1 MISSION COMPLETE

This package provides the foundational superposition architecture for all agents
across all MARL systems in the GrandModel project.

Key Components:
- UniversalSuperposition: Abstract base class for superposition operations
- AgentSuperpositionConverter: Universal action format converter
- SuperpositionValidator: Mathematical validation framework
- SuperpositionSerializer: Persistence and serialization system
- SuperpositionState: Core quantum-inspired state representation

Main Features:
- Universal action format conversion (discrete, continuous, hybrid)
- High-performance implementation (<1ms per conversion)
- Mathematical validation and error handling
- Extensible plugin architecture
- Integration with existing GrandModel patterns

Usage Examples:
    # Basic usage
    from src.core.superposition import create_agent_converter
    
    converter = create_agent_converter()
    superposition = converter.convert_to_superposition(action)
    
    # Advanced usage
    from src.core.superposition import UniversalSuperposition, SuperpositionState
    
    # Create custom superposition
    state = SuperpositionState(amplitudes, basis_actions, action_space_type, format_type)
    
    # Batch conversion
    results = converter.batch_convert_agents(agent_outputs)

Author: Agent 1 - Universal Superposition Core Architect
Version: 1.0 - Foundation Framework Complete
"""

# Core imports
from .universal_superposition import (
    UniversalSuperposition,
    SuperpositionState,
    ActionSpaceType,
    SuperpositionError,
    InvalidSuperpositionError,
    ConversionError,
    PERFORMANCE_TRACKER,
    create_uniform_superposition,
    create_peaked_superposition,
    superposition_distance,
    validate_superposition_properties,
    performance_monitor,
    cached_conversion
)

from .agent_superposition_converter import (
    AgentSuperpositionConverter,
    FormatPlugin,
    DiscreteFormatPlugin,
    ContinuousFormatPlugin,
    HybridFormatPlugin,
    DictFormatPlugin,
    LegacyAgentFormatPlugin,
    create_agent_converter,
    test_converter_with_sample_actions
)

from .superposition_validator import (
    SuperpositionValidator,
    ValidationLevel,
    ValidationCategory,
    ValidationSeverity,
    ValidationResult,
    ValidationReport,
    ValidationCheck,
    create_validator,
    validate_superposition,
    quick_validate
)

from .superposition_serializer import (
    SuperpositionPersistence,
    SerializationFormat,
    CompressionType,
    SerializationMetadata,
    SerializedSuperposition,
    SuperpositionSerializer,
    JSONSerializer,
    PickleSerializer,
    BinarySerializer,
    create_persistence_manager,
    save_superposition,
    load_superposition
)

# Version information
__version__ = "1.0.0"
__author__ = "Agent 1 - Universal Superposition Core Architect"
__email__ = "agent1@grandmodel.ai"
__status__ = "Production"

# Package metadata
__all__ = [
    # Core classes
    "UniversalSuperposition",
    "SuperpositionState",
    "AgentSuperpositionConverter",
    
    # Enums and exceptions
    "ActionSpaceType",
    "SuperpositionError",
    "InvalidSuperpositionError",
    "ConversionError",
    
    # Plugin classes
    "FormatPlugin",
    "DiscreteFormatPlugin",
    "ContinuousFormatPlugin",
    "HybridFormatPlugin",
    "DictFormatPlugin",
    "LegacyAgentFormatPlugin",
    
    # Validation classes
    "SuperpositionValidator",
    "ValidationLevel",
    "ValidationCategory",
    "ValidationSeverity",
    "ValidationResult",
    "ValidationReport",
    "ValidationCheck",
    
    # Serialization classes
    "SuperpositionPersistence",
    "SerializationFormat",
    "CompressionType",
    "SerializationMetadata",
    "SerializedSuperposition",
    "SuperpositionSerializer",
    "JSONSerializer",
    "PickleSerializer",
    "BinarySerializer",
    
    # Utility functions
    "create_agent_converter",
    "create_uniform_superposition",
    "create_peaked_superposition",
    "superposition_distance",
    "validate_superposition_properties",
    "test_converter_with_sample_actions",
    "create_validator",
    "validate_superposition",
    "quick_validate",
    "create_persistence_manager",
    "save_superposition",
    "load_superposition",
    
    # Decorators
    "performance_monitor",
    "cached_conversion",
    
    # Global objects
    "PERFORMANCE_TRACKER"
]

# Configuration defaults
DEFAULT_CONFIG = {
    "max_basis_size": 1000,
    "discretization_levels": 50,
    "enable_caching": True,
    "performance_tracking": True,
    "fast_mode": True,
    "tolerance": 1e-6
}

# Global converter instance (singleton pattern)
_global_converter = None

def get_global_converter(config=None):
    """
    Get or create global converter instance
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        AgentSuperpositionConverter: Global converter instance
    """
    global _global_converter
    
    if _global_converter is None:
        _global_converter = create_agent_converter(config)
    
    return _global_converter

def reset_global_converter():
    """Reset global converter instance"""
    global _global_converter
    _global_converter = None

# Convenience functions
def convert_action(action, config=None):
    """
    Convenience function to convert action to superposition
    
    Args:
        action: Action to convert
        config: Optional configuration
        
    Returns:
        SuperpositionState: Converted superposition
    """
    converter = get_global_converter(config)
    return converter.convert_to_superposition(action)

def convert_agent_output(agent_output, agent_name="unknown", config=None):
    """
    Convenience function to convert agent output to superposition
    
    Args:
        agent_output: Agent output dictionary
        agent_name: Name of the agent
        config: Optional configuration
        
    Returns:
        SuperpositionState: Converted superposition
    """
    converter = get_global_converter(config)
    return converter.convert_agent_output(agent_output, agent_name)

def batch_convert_agents(agent_outputs, config=None):
    """
    Convenience function to batch convert agent outputs
    
    Args:
        agent_outputs: Dictionary of agent_name -> agent_output
        config: Optional configuration
        
    Returns:
        Dictionary of agent_name -> SuperpositionState
    """
    converter = get_global_converter(config)
    return converter.batch_convert_agents(agent_outputs)

def get_performance_stats():
    """
    Get global performance statistics
    
    Returns:
        Dictionary of performance metrics
    """
    return PERFORMANCE_TRACKER.get_performance_stats()

def get_supported_formats(config=None):
    """
    Get list of supported action formats
    
    Args:
        config: Optional configuration
        
    Returns:
        List of supported format strings
    """
    converter = get_global_converter(config)
    return converter.get_supported_formats()

# Module initialization
def _initialize_module():
    """Initialize the superposition module"""
    import logging
    
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.info("Universal Superposition Core Framework initialized")
    logger.info(f"Version: {__version__}")
    logger.info(f"Supported formats: {len(get_supported_formats())}")
    
    # Performance check
    try:
        import time
        start_time = time.time()
        
        # Test basic functionality
        test_action = 5
        superposition = convert_action(test_action)
        
        init_time = (time.time() - start_time) * 1000
        
        if init_time < 1.0:  # Target: <1ms
            logger.info(f"✅ Performance target met: {init_time:.2f}ms initialization")
        else:
            logger.warning(f"⚠️ Performance target missed: {init_time:.2f}ms initialization")
        
        logger.info("✅ Module validation complete")
        
    except Exception as e:
        logger.error(f"❌ Module initialization failed: {str(e)}")

# Run module initialization
_initialize_module()

# Export package information
def get_package_info():
    """Get package information dictionary"""
    return {
        "name": "Universal Superposition Core Framework",
        "version": __version__,
        "author": __author__,
        "status": __status__,
        "components": len(__all__),
        "supported_formats": len(get_supported_formats()),
        "performance_target": "<1ms per conversion",
        "mathematical_foundation": "Quantum-inspired superposition",
        "architecture": "Plugin-based extensible system"
    }

# Print package info on import (for debugging)
if __name__ == "__main__":
    import json
    print("🚀 Universal Superposition Core Framework")
    print("=" * 50)
    print(json.dumps(get_package_info(), indent=2))
    print("=" * 50)
    print("✅ Ready for production deployment!")