"""
Logging Configuration for Lorentzian Trading Strategy
Provides centralized logging setup with file rotation and structured output.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

from ..config.config import get_config


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        # Format the message
        formatted = super().format(record)
        
        # Reset levelname for other handlers
        record.levelname = levelname
        
        return formatted


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for production logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return str(log_entry)


def setup_logging(config=None, logger_name: str = "lorentzian_strategy") -> logging.Logger:
    """Setup comprehensive logging configuration"""
    config = config or get_config()
    
    # Create logs directory
    log_dir = Path(config.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get or create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, config.logging.level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(module)s:%(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    colored_formatter = ColoredFormatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    if config.logging.console_handler:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Use colored formatter for console if supported
        if sys.stdout.isatty():  # Check if terminal supports colors
            console_handler.setFormatter(colored_formatter)
        else:
            console_handler.setFormatter(simple_formatter)
        
        logger.addHandler(console_handler)
    
    # File handlers
    if config.logging.file_handler:
        # Main log file with rotation
        main_log_file = log_dir / f"{logger_name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            filename=main_log_file,
            maxBytes=config.logging.max_bytes,
            backupCount=config.logging.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # Error log file
        error_log_file = log_dir / f"{logger_name}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            filename=error_log_file,
            maxBytes=config.logging.max_bytes,
            backupCount=config.logging.backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)
        
        # Performance log file for timing and metrics
        perf_log_file = log_dir / f"{logger_name}_performance.log"
        perf_handler = logging.handlers.RotatingFileHandler(
            filename=perf_log_file,
            maxBytes=config.logging.max_bytes,
            backupCount=config.logging.backup_count,
            encoding='utf-8'
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(detailed_formatter)
        
        # Add filter to only capture performance logs
        class PerformanceFilter(logging.Filter):
            def filter(self, record):
                return hasattr(record, 'performance') and record.performance
        
        perf_handler.addFilter(PerformanceFilter())
        logger.addHandler(perf_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


class PerformanceLogger:
    """Performance timing and metrics logger"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'operation_name'):
            duration = (datetime.now() - self.start_time).total_seconds()
            # Create log record with performance flag
            record = logging.LogRecord(
                name=self.logger.name,
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=f"Performance: {self.operation_name} completed in {duration:.4f}s",
                args=(),
                exc_info=None
            )
            record.performance = True
            self.logger.handle(record)
    
    def log_operation(self, operation_name: str):
        """Set operation name for performance logging"""
        self.operation_name = operation_name
        return self


class TimingDecorator:
    """Decorator for automatic function timing"""
    
    def __init__(self, logger: logging.Logger, log_args: bool = False):
        self.logger = logger
        self.log_args = log_args
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            
            # Log function entry
            if self.log_args:
                self.logger.debug(f"Entering {func.__name__} with args={args[:3]} kwargs={list(kwargs.keys())}")
            else:
                self.logger.debug(f"Entering {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                
                # Create performance log record
                record = logging.LogRecord(
                    name=self.logger.name,
                    level=logging.INFO,
                    pathname="",
                    lineno=0,
                    msg=f"Performance: {func.__name__} completed in {duration:.4f}s",
                    args=(),
                    exc_info=None
                )
                record.performance = True
                self.logger.handle(record)
                
                return result
                
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                self.logger.error(f"Function {func.__name__} failed after {duration:.4f}s: {e}")
                raise
        
        return wrapper


def get_logger(name: str = None, config=None) -> logging.Logger:
    """Get configured logger instance"""
    name = name or "lorentzian_strategy"
    logger = logging.getLogger(name)
    
    # Setup logging if not already configured
    if not logger.handlers:
        logger = setup_logging(config, name)
    
    return logger


def log_function_call(logger: logging.Logger, log_args: bool = False):
    """Decorator factory for logging function calls"""
    return TimingDecorator(logger, log_args)


# Pre-configured loggers for different components
def get_data_logger():
    """Get logger for data operations"""
    return get_logger("lorentzian_strategy.data")

def get_features_logger():
    """Get logger for feature engineering"""
    return get_logger("lorentzian_strategy.features")

def get_backtest_logger():
    """Get logger for backtesting"""
    return get_logger("lorentzian_strategy.backtest")

def get_core_logger():
    """Get logger for core algorithms"""
    return get_logger("lorentzian_strategy.core")