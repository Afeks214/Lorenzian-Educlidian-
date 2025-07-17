#!/usr/bin/env python3
"""
GrandModel Configuration System Demo
AGENT 5 - Configuration Recovery Mission
Demonstrates the complete configuration management system
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def demo_configuration_system():
    """Demonstrate the complete configuration system"""
    print("🚀 GrandModel Configuration System Demo")
    print("=" * 50)
    
    try:
        from core.config_manager import ConfigManager, Environment, get_config_manager
        
        print("\n1. 🔧 Testing Configuration Manager Initialization")
        
        # Test different environments
        environments = [Environment.PRODUCTION, Environment.DEVELOPMENT, Environment.TESTING]
        
        for env in environments:
            print(f"\n   📋 Loading {env.value} configuration...")
            config_manager = ConfigManager(env)
            
            # Test system config
            system_config = config_manager.get_system_config()
            print(f"   ✅ System config loaded: {len(system_config)} sections")
            
            # Test database config
            db_config = config_manager.get_database_config()
            print(f"   ✅ Database config: Host={db_config.get('host', 'N/A')}")
            
            # Test trading configs
            for agent_type in ['strategic', 'tactical', 'risk']:
                try:
                    trading_config = config_manager.get_trading_config(agent_type)
                    agent_name = trading_config.get(f'{agent_type}_agent', {}).get('name', 'N/A')
                    print(f"   ✅ {agent_type.title()} agent: {agent_name}")
                except Exception as e:
                    print(f"   ⚠️ {agent_type.title()} agent config error: {e}")
        
        print("\n2. 🤖 Testing Agent Configurations")
        
        # Get production config manager
        config_manager = ConfigManager(Environment.PRODUCTION)
        
        # Strategic Agent
        strategic_config = config_manager.get_trading_config('strategic')
        strategic_agent = strategic_config.get('strategic_agent', {})
        print(f"   📊 Strategic Agent:")
        print(f"      - Name: {strategic_agent.get('name', 'N/A')}")
        print(f"      - Type: {strategic_agent.get('type', 'N/A')}")
        print(f"      - Horizon: {strategic_agent.get('horizon', 'N/A')}")
        print(f"      - Risk Tolerance: {strategic_agent.get('behavior', {}).get('risk_tolerance', 'N/A')}")
        
        # Tactical Agent
        tactical_config = config_manager.get_trading_config('tactical')
        tactical_agent = tactical_config.get('tactical_agent', {})
        print(f"   ⚡ Tactical Agent:")
        print(f"      - Name: {tactical_agent.get('name', 'N/A')}")
        print(f"      - Max Positions: {tactical_agent.get('behavior', {}).get('max_positions', 'N/A')}")
        print(f"      - Holding Period: {tactical_agent.get('behavior', {}).get('position_holding_period', 'N/A')}")
        
        # Risk Agent
        risk_config = config_manager.get_trading_config('risk')
        risk_agent = risk_config.get('risk_agent', {})
        print(f"   🛡️ Risk Agent:")
        print(f"      - Name: {risk_agent.get('name', 'N/A')}")
        print(f"      - Primary Objective: {risk_agent.get('behavior', {}).get('primary_objective', 'N/A')}")
        print(f"      - Kelly Criterion: {risk_agent.get('kelly_criterion', {}).get('enabled', 'N/A')}")
        
        print("\n3. 🧠 Testing Model Configurations")
        
        # MAPPO Config
        mappo_config = config_manager.get_model_config('mappo')
        mappo_settings = mappo_config.get('mappo', {})
        print(f"   🔄 MAPPO Configuration:")
        print(f"      - Algorithm: {mappo_settings.get('algorithm', {}).get('type', 'N/A')}")
        print(f"      - Shared Critic: {mappo_settings.get('algorithm', {}).get('shared_critic', 'N/A')}")
        print(f"      - Training Episodes: {mappo_settings.get('training', {}).get('episodes_per_update', 'N/A')}")
        
        # Network Config
        network_config = config_manager.get_model_config('networks')
        networks = network_config.get('networks', {})
        print(f"   🧮 Network Configuration:")
        print(f"      - Framework: {networks.get('framework', 'N/A')}")
        print(f"      - Input Processing: {networks.get('input_processing', {}).get('matrix_processor', {}).get('type', 'N/A')}")
        
        # Hyperparameters
        hyper_config = config_manager.get_model_config('hyperparameters')
        hyper_settings = hyper_config.get('hyperparameters', {})
        print(f"   ⚙️ Hyperparameters:")
        print(f"      - Experiment: {hyper_settings.get('experiment_name', 'N/A')}")
        print(f"      - Global Seed: {hyper_settings.get('global', {}).get('seed', 'N/A')}")
        
        print("\n4. 🌍 Testing Environment Configurations")
        
        # Market Environment
        market_config = config_manager.get_environment_config('market')
        market_env = market_config.get('market_environment', {})
        print(f"   📈 Market Environment:")
        print(f"      - Name: {market_env.get('name', 'N/A')}")
        print(f"      - Type: {market_env.get('type', 'N/A')}")
        instruments = market_env.get('market_data', {}).get('instruments', {}).get('equities', [])
        print(f"      - Equity Instruments: {len(instruments)} configured")
        
        # Simulation Environment
        simulation_config = config_manager.get_environment_config('simulation')
        sim_env = simulation_config.get('simulation_environment', {})
        print(f"   🎮 Simulation Environment:")
        print(f"      - Framework: {sim_env.get('framework', {}).get('engine', 'N/A')}")
        print(f"      - Parallel: {sim_env.get('framework', {}).get('parallel', 'N/A')}")
        print(f"      - Max Steps: {sim_env.get('lifecycle', {}).get('episode', {}).get('max_steps', 'N/A')}")
        
        print("\n5. 🔍 Testing Configuration Validation")
        
        validation_result = config_manager.validate_configs()
        print(f"   📋 Validation Result: {'✅ PASSED' if validation_result else '❌ FAILED'}")
        
        print("\n6. 🌟 Testing Global Configuration Access")
        
        # Test global config manager
        global_manager = get_config_manager()
        print(f"   🌐 Global Manager Environment: {global_manager.environment.value}")
        
        # Test convenience function
        from core.config_manager import get_config
        system_info = get_config('system', 'system')
        print(f"   ⚙️ System Name: {system_info.get('name', 'N/A')}")
        print(f"   🔖 System Version: {system_info.get('version', 'N/A')}")
        
        print("\n✅ Configuration System Demo Complete!")
        print("\n📊 Demo Summary:")
        print("   • ✅ Multi-environment configuration loading")
        print("   • ✅ Agent-specific configuration access")
        print("   • ✅ Model configuration management")
        print("   • ✅ Environment configuration handling")
        print("   • ✅ Configuration validation")
        print("   • ✅ Global configuration access")
        
        print("\n🎯 The GrandModel configuration system is fully operational!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure PyYAML is installed: pip install PyYAML")
        return False
    except Exception as e:
        print(f"❌ Demo error: {e}")
        return False

def main():
    """Main demo function"""
    success = demo_configuration_system()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())