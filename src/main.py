import logging
from pathlib import Path

from src.core.kernel import AlgoSpaceKernel

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main GrandModel application entry point with Strategic MARL integration"""
    logger.info("🚀 GrandModel starting...")

    try:
        # Initialize kernel with strategic MARL configuration
        config_path = Path("configs/strategic_marl_config.yaml")
        if not config_path.exists():
            logger.warning(
                f"Strategic MARL config not found at {config_path}, using default config"
            )
            config_path = "config/settings.yaml"  # Fallback to default

        # Create and initialize kernel
        kernel = AlgoSpaceKernel(config_path=str(config_path))
        kernel.initialize()

        logger.info("✅ GrandModel initialized with Strategic MARL integration")

        # Start the kernel (this will run indefinitely)
        kernel.run()

    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ GrandModel startup failed: {e}", exc_info=True)
        raise
    finally:
        logger.info("🏁 GrandModel shutdown complete")


if __name__ == "__main__":
    main()
