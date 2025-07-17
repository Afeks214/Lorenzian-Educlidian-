"""Minimal configuration loader for production kernel"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file with minimal dependencies.
    Falls back to safe defaults if file not found.
    """
    # Safe production defaults
    default_config = {
        "environment": "production",
        "debug": False,
        "data_handler": {"type": "backtest"},
        "events": {"async_processing": True, "max_queue_size": 1000},
        "strategic_marl": {"enabled": True, "device": "cpu", "deterministic": True},
        "logging": {"level": "INFO", "format": "json"},
        "performance": {
            "max_inference_latency_ms": 10.0,
            "max_memory_usage_mb": 512,
            "batch_size": 32
        }
    }
    
    if not config_path:
        logger.info("No config path provided, using defaults")
        return default_config
    
    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return default_config
    
    try:
        with open(config_file) as f:
            if config_file.suffix == '.json':
                file_config = json.load(f)
            elif config_file.suffix in ['.yaml', '.yml']:
                file_config = yaml.safe_load(f)
            else:
                logger.warning(f"Unknown config format: {config_file.suffix}")
                return default_config
        
        # Simple merge
        default_config.update(file_config)
        logger.info(f"Configuration loaded from {config_path}")
        return default_config
        
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        logger.info("Using default configuration")
        return default_config