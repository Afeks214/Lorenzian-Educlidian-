# src/core/kernel.py
"""
The System Kernel & Orchestration class. This is the master conductor.
"""
import logging
from typing import Dict, Any, Optional

from .minimal_config import load_config
from .event_bus import EventBus
from .component_base import ComponentBase

# Component imports - these will be replaced with actual imports as they are developed
try:
    from ..components.data_handler import LiveDataHandler, BacktestDataHandler
except ImportError:
    LiveDataHandler = BacktestDataHandler = None

try:
    from ..components.bar_generator import BarGenerator
except ImportError:
    BarGenerator = None

try:
    from ..components.indicator_engine import IndicatorEngine
except ImportError:
    IndicatorEngine = None

try:
    from ..matrix.assembler_30m import MatrixAssembler30m
    from ..matrix.assembler_5m import MatrixAssembler5m
    from ..matrix.assembler_regime import MatrixAssemblerRegime
except ImportError:
    MatrixAssembler30m = MatrixAssembler5m = MatrixAssemblerRegime = None

# 4 Specialized MARL Systems - Strategic, Tactical, Risk, Execution
try:
    from ..agents.strategic_marl_component import StrategicMARLComponent
except ImportError:
    StrategicMARLComponent = None

try:
    from ..tactical.controller import TacticalMARLController
except ImportError:
    TacticalMARLController = None

try:
    from ..risk.marl.agent_coordinator import RiskMARLCoordinator
except ImportError:
    RiskMARLCoordinator = None

# Temporarily disable execution MARL due to dependency issues
UnifiedExecutionMARLSystem = None
# try:
#     from ..execution.unified_execution_marl_system import UnifiedExecutionMARLSystem
# except ImportError:
#     UnifiedExecutionMARLSystem = None

try:
    from ..agents.main_core import MainMARLCoreComponent
except ImportError:
    MainMARLCoreComponent = None

try:
    from ..agents.synergy.detector import SynergyDetector
except ImportError:
    SynergyDetector = None

try:
    from ..agents.strategic_marl_component import StrategicMARLComponent
except ImportError:
    StrategicMARLComponent = None

try:
    from ..components.execution_handler import (
        LiveExecutionHandler,
        BacktestExecutionHandler,
    )
except ImportError:
    LiveExecutionHandler = BacktestExecutionHandler = None

logger = logging.getLogger(__name__)


class AlgoSpaceKernel:
    """
    The main system kernel that orchestrates all components of the AlgoSpace trading system.

    This class is responsible for:
    - Loading configuration
    - Instantiating all system components
    - Wiring components together via the event bus
    - Managing the system lifecycle
    """

    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Initializes the Kernel, but does not start it.

        Args:
            config_path: Path to the system configuration file.
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.event_bus = EventBus()
        self.components: Dict[str, Any] = {}
        self.running = False

        # Configure logging
        self._setup_logging()

        logger.info(f"AlgoSpace Kernel initialized with config path: {config_path}")

    def _setup_logging(self) -> None:
        """Configure logging for the system."""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("logs/algospace.log", mode="a"),
            ],
        )

    def initialize(self) -> None:
        """
        Initializes and wires all system components in the correct dependency order.

        Raises:
            Exception: If initialization fails at any stage.
        """
        try:
            logger.info("=== AlgoSpace System Initialization Starting ===")

            # Load configuration
            self.config = load_config(self.config_path)
            logger.info("Configuration loaded successfully")

            # Phase 1: Component Instantiation
            logger.info("\n--- Phase 1: Component Instantiation ---")
            self._instantiate_components()

            # Phase 2: Event Wiring
            logger.info("\n--- Phase 2: Event Wiring ---")
            self._wire_events()

            # Phase 3: Component Initialization
            logger.info("\n--- Phase 3: Component Initialization ---")
            self._initialize_components()

            logger.info("\n=== AlgoSpace Initialization Complete. System is READY. ===")

        except Exception as e:
            logger.error(f"Kernel initialization failed: {e}", exc_info=True)
            self.shutdown()
            raise

    def _instantiate_components(self) -> None:
        """Instantiates all system components based on configuration."""
        # Data Pipeline
        data_type = self.config["data_handler"]["type"]
        logger.info(f"Instantiating components for data handler type: {data_type}")

        if data_type in ["rithmic", "ib"]:
            if LiveDataHandler:
                self.components["data_handler"] = LiveDataHandler(
                    self.config, self.event_bus
                )
                logger.info("LiveDataHandler instantiated")
            else:
                logger.warning("LiveDataHandler not available")
        else:  # backtest
            if BacktestDataHandler:
                self.components["data_handler"] = BacktestDataHandler(
                    self.config, self.event_bus
                )
                logger.info("BacktestDataHandler instantiated")
            else:
                logger.warning("BacktestDataHandler not available")

        # Bar Generation
        if BarGenerator:
            self.components["bar_generator"] = BarGenerator(self.config, self.event_bus)
            logger.info("BarGenerator instantiated")

        # Indicator Engine
        if IndicatorEngine:
            self.components["indicator_engine"] = IndicatorEngine(
                self.config, self.event_bus
            )
            logger.info("IndicatorEngine instantiated")

        # Feature Preparation - Matrix Assemblers
        # Get matrix assemblers configuration section
        matrix_config = self.config.get("matrix_assemblers", {})

        if MatrixAssembler30m:
            # Prepare configuration for 30m assembler
            config_30m = matrix_config.get(
                "30m",
                {
                    "window_size": 48,
                    "features": [
                        "mlmi_value",
                        "mlmi_signal",
                        "nwrqk_value",
                        "nwrqk_slope",
                        "lvn_distance_points",
                        "lvn_nearest_strength",
                        "time_hour_sin",
                        "time_hour_cos",
                    ],
                },
            )
            config_30m["name"] = "MatrixAssembler30m"
            config_30m["kernel"] = self

            self.components["matrix_30m"] = MatrixAssembler30m(config_30m)
            logger.info("MatrixAssembler30m instantiated")

        if MatrixAssembler5m:
            # Prepare configuration for 5m assembler
            config_5m = matrix_config.get(
                "5m",
                {
                    "window_size": 60,
                    "features": [
                        "fvg_bullish_active",
                        "fvg_bearish_active",
                        "fvg_nearest_level",
                        "fvg_age",
                        "fvg_mitigation_signal",
                        "price_momentum_5",
                        "volume_ratio",
                    ],
                },
            )
            config_5m["name"] = "MatrixAssembler5m"
            config_5m["kernel"] = self

            self.components["matrix_5m"] = MatrixAssembler5m(config_5m)
            logger.info("MatrixAssembler5m instantiated")

        if MatrixAssemblerRegime:
            # Prepare configuration for regime assembler
            config_regime = matrix_config.get(
                "regime",
                {
                    "window_size": 96,
                    "features": [
                        "mmd_features",
                        "volatility_30",
                        "volume_profile_skew",
                        "price_acceleration",
                    ],
                },
            )
            config_regime["name"] = "MatrixAssemblerRegime"
            config_regime["kernel"] = self

            self.components["matrix_regime"] = MatrixAssemblerRegime(config_regime)
            logger.info("MatrixAssemblerRegime instantiated")

        # Intelligence Layer
        if SynergyDetector:
            # Pass the kernel reference with name
            self.components["synergy_detector"] = SynergyDetector(
                "SynergyDetector", self
            )
            logger.info("SynergyDetector instantiated")

        # Strategic MARL Component
        strategic_marl_config = self.config.get("strategic_marl", {})
        if strategic_marl_config.get("enabled", False) and StrategicMARLComponent:
            self.components["strategic_marl"] = StrategicMARLComponent(
                "StrategicMARLComponent", self
            )
            logger.info("StrategicMARLComponent instantiated")
        elif strategic_marl_config.get("enabled", False):
            logger.warning(
                "Strategic MARL enabled in config but StrategicMARLComponent not available"
            )

        # 4 Specialized MARL Systems
        
        # 1. Strategic MARL (30-minute)
        if StrategicMARLComponent:
            strategic_config = self.config.get("strategic_marl", {})
            if strategic_config.get("enabled", True):
                self.components["strategic_marl"] = StrategicMARLComponent(self)
                logger.info("Strategic MARL Component instantiated")
        
        # 2. Tactical MARL (5-minute)
        if TacticalMARLController:
            tactical_config = self.config.get("tactical_marl", {})
            if tactical_config.get("enabled", True):
                self.components["tactical_marl"] = TacticalMARLController(tactical_config)
                logger.info("Tactical MARL Controller instantiated")
        
        # 3. Risk Management MARL
        if RiskMARLCoordinator:
            risk_config = self.config.get("risk_marl", {})
            if risk_config.get("enabled", True):
                self.components["risk_marl"] = RiskMARLCoordinator(risk_config)
                logger.info("Risk MARL Coordinator instantiated")
        
        # 4. Execution MARL
        if UnifiedExecutionMARLSystem:
            execution_config = self.config.get("execution_marl", {})
            if execution_config.get("enabled", True):
                self.components["execution_marl"] = UnifiedExecutionMARLSystem(execution_config)
                logger.info("Execution MARL System instantiated")

        # Main MARL Core
        if MainMARLCoreComponent:
            # Get main core configuration
            main_core_config = self.config.get("main_core", {})
            self.components["main_marl_core"] = MainMARLCoreComponent(
                main_core_config, self.components
            )
            logger.info("MainMARLCoreComponent instantiated")

        # Execution Layer - determine based on data handler type
        data_type = self.config["data_handler"]["type"]
        if data_type in ["rithmic", "ib"]:
            if LiveExecutionHandler:
                self.components["execution_handler"] = LiveExecutionHandler(
                    self.config, self.event_bus
                )
                logger.info("LiveExecutionHandler instantiated")
        else:  # backtest
            if BacktestExecutionHandler:
                self.components["execution_handler"] = BacktestExecutionHandler(
                    self.config, self.event_bus
                )
                logger.info("BacktestExecutionHandler instantiated")

        logger.info(f"Total components instantiated: {len(self.components)}")

    def _wire_events(self) -> None:
        """Connects all components via event subscriptions."""
        from .events import EventType
        
        # Data Flow Events
        if "bar_generator" in self.components:
            self.event_bus.subscribe(
                EventType.NEW_TICK, self.components["bar_generator"].on_new_tick
            )
            logger.info("Wired: NEW_TICK -> BarGenerator")

        if "indicator_engine" in self.components:
            self.event_bus.subscribe(
                EventType.NEW_5MIN_BAR, self.components["indicator_engine"].on_new_bar
            )
            self.event_bus.subscribe(
                EventType.NEW_30MIN_BAR, self.components["indicator_engine"].on_new_bar
            )
            logger.info("Wired: NEW_*_BAR -> IndicatorEngine")

        # Matrix Assembly Events
        for matrix_name in ["matrix_30m", "matrix_5m", "matrix_regime"]:
            if matrix_name in self.components:
                self.event_bus.subscribe(
                    EventType.INDICATORS_READY, self.components[matrix_name].on_indicators_ready
                )
                logger.info(f"Wired: INDICATORS_READY -> {matrix_name}")

        # Decision Flow Events
        if "synergy_detector" in self.components:
            self.event_bus.subscribe(
                EventType.INDICATORS_READY,
                self.components["synergy_detector"]._handle_indicators_ready,
            )
            logger.info("Wired: INDICATORS_READY -> SynergyDetector")

        # Strategic MARL Events
        if "strategic_marl" in self.components:
            self.event_bus.subscribe(
                EventType.SYNERGY_DETECTED,
                self.components["strategic_marl"]._handle_synergy_detected,
            )
            self.event_bus.subscribe(
                EventType.NEW_30MIN_BAR, self.components["strategic_marl"]._handle_new_30min_bar
            )
            logger.info("Wired: SYNERGY_DETECTED -> StrategicMARLComponent")
            logger.info("Wired: NEW_30MIN_BAR -> StrategicMARLComponent")

        if "main_marl_core" in self.components:
            self.event_bus.subscribe(
                EventType.SYNERGY_DETECTED,
                self.components["main_marl_core"].initiate_qualification,
            )
            logger.info("Wired: SYNERGY_DETECTED -> MainMARLCore")

        if "execution_handler" in self.components:
            self.event_bus.subscribe(
                EventType.EXECUTE_TRADE, self.components["execution_handler"].execute_trade
            )
            logger.info("Wired: EXECUTE_TRADE -> ExecutionHandler")

        # Feedback Loop Events
        if "main_marl_core" in self.components:
            self.event_bus.subscribe(
                EventType.TRADE_CLOSED, self.components["main_marl_core"].record_outcome
            )
            logger.info("Wired: TRADE_CLOSED -> MainMARLCore")

        # System Events
        self.event_bus.subscribe(EventType.SYSTEM_ERROR, self._handle_system_error)
        self.event_bus.subscribe(EventType.SHUTDOWN_REQUEST, lambda _: self.shutdown())

        logger.info("Event wiring completed")

    def _initialize_components(self) -> None:
        """Initialize components that require post-instantiation setup."""
        # Load pre-trained models for 4 MARL systems
        models_config = self.config.get("models", {})
        
        # Strategic MARL model loading
        if "strategic_marl" in self.components:
            strategic_model_path = models_config.get("strategic_marl_path")
            if strategic_model_path and hasattr(self.components["strategic_marl"], "load_model"):
                try:
                    self.components["strategic_marl"].load_model(strategic_model_path)
                    logger.info(f"Strategic MARL model loaded from: {strategic_model_path}")
                except Exception as e:
                    logger.error(f"Failed to load Strategic MARL model: {e}")
        
        # Tactical MARL model loading
        if "tactical_marl" in self.components:
            tactical_model_path = models_config.get("tactical_marl_path")
            if tactical_model_path and hasattr(self.components["tactical_marl"], "load_model"):
                try:
                    self.components["tactical_marl"].load_model(tactical_model_path)
                    logger.info(f"Tactical MARL model loaded from: {tactical_model_path}")
                except Exception as e:
                    logger.error(f"Failed to load Tactical MARL model: {e}")
        
        # Risk MARL model loading
        if "risk_marl" in self.components:
            risk_model_path = models_config.get("risk_marl_path")
            if risk_model_path and hasattr(self.components["risk_marl"], "load_model"):
                try:
                    self.components["risk_marl"].load_model(risk_model_path)
                    logger.info(f"Risk MARL model loaded from: {risk_model_path}")
                except Exception as e:
                    logger.error(f"Failed to load Risk MARL model: {e}")
        
        # Execution MARL model loading
        if "execution_marl" in self.components:
            execution_model_path = models_config.get("execution_marl_path")
            if execution_model_path and hasattr(self.components["execution_marl"], "load_model"):
                try:
                    self.components["execution_marl"].load_model(execution_model_path)
                    logger.info(f"Execution MARL model loaded from: {execution_model_path}")
                except Exception as e:
                    logger.error(f"Failed to load Execution MARL model: {e}")

        # Strategic MARL Component Initialization
        if "strategic_marl" in self.components:
            if hasattr(self.components["strategic_marl"], "initialize"):
                try:
                    success = self.components["strategic_marl"].initialize()
                    if success:
                        logger.info("Strategic MARL Component initialized successfully")
                    else:
                        logger.error("Strategic MARL Component initialization failed")
                except Exception as e:
                    logger.error(f"Failed to initialize Strategic MARL Component: {e}")

        if "main_marl_core" in self.components:
            if hasattr(self.components["main_marl_core"], "load_models"):
                try:
                    self.components["main_marl_core"].load_models()
                    logger.info("MARL models loaded")
                except Exception as e:
                    logger.error(f"Failed to load MARL models: {e}")

    def _handle_system_error(self, error_info: Dict[str, Any]) -> None:
        """
        Handles system-wide errors.

        Args:
            error_info: Dictionary containing error details.
        """
        logger.error(f"System error: {error_info}")

        # Determine if error is critical
        if error_info.get("critical", False):
            logger.critical("Critical error detected. Initiating shutdown.")
            self.shutdown()

    def run(self) -> None:
        """Starts the main system loop."""
        if not self.components:
            raise RuntimeError("Kernel not initialized. Call initialize() first.")

        self.running = True
        logger.info("\n=== AlgoSpace System Running ===")

        try:
            # Start data stream
            if "data_handler" in self.components:
                logger.info("Starting data stream...")
                if hasattr(self.components["data_handler"], "start_stream"):
                    self.components["data_handler"].start_stream()

            # Run the event loop
            logger.info("Starting event dispatcher...")
            self.event_bus.dispatch_forever()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            self.shutdown()
        except Exception as e:
            logger.error(f"Critical error in main loop: {e}", exc_info=True)
            self.shutdown()

    def shutdown(self) -> None:
        """Initiates a graceful shutdown of the system."""
        if not self.running:
            return

        logger.info("\n=== Graceful Shutdown Initiated ===")
        self.running = False

        try:
            # Stop data streams
            if "data_handler" in self.components:
                if hasattr(self.components["data_handler"], "stop_stream"):
                    self.components["data_handler"].stop_stream()
                    logger.info("Data stream stopped")

            # Close all positions
            if "execution_handler" in self.components:
                if hasattr(self.components["execution_handler"], "close_all_positions"):
                    self.components["execution_handler"].close_all_positions()
                    logger.info("All positions closed")

            # Save component states
            for name, component in self.components.items():
                if hasattr(component, "save_state"):
                    try:
                        component.save_state()
                        logger.info(f"State saved for: {name}")
                    except Exception as e:
                        logger.error(f"Failed to save state for {name}: {e}")

            # Stop event bus
            self.event_bus.stop()
            logger.info("Event bus stopped")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)

        logger.info("=== System Shutdown Complete ===")

    def get_component(self, name: str) -> Optional[Any]:
        """
        Retrieves a component by name.

        Args:
            name: The component name.

        Returns:
            The component instance or None if not found.
        """
        return self.components.get(name)

    def get_status(self) -> Dict[str, Any]:
        """
        Returns the current system status.

        Returns:
            Dictionary containing system status information.
        """
        return {
            "running": self.running,
            "mode": self.config.get("data_handler", {}).get("type", "unknown"),
            "components": list(self.components.keys()),
            "subscribers": len(self.event_bus._subscribers),
        }

    def get_event_bus(self) -> EventBus:
        """
        Returns the system event bus.

        Returns:
            The EventBus instance.
        """
        return self.event_bus
