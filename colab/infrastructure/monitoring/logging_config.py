#!/usr/bin/env python3
"""
Comprehensive Logging Configuration System
Provides structured logging for training infrastructure
"""

import os
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json
import traceback

class TrainingLogger:
    """Comprehensive logging system for training infrastructure"""
    
    def __init__(self, log_dir: str = "/home/QuantNova/GrandModel/colab/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for subdir in ['training', 'validation', 'errors', 'performance', 'system', 'debug', 'audit']:
            (self.log_dir / subdir).mkdir(exist_ok=True)
        
        self.loggers = {}
        self.setup_loggers()
    
    def setup_loggers(self):
        """Setup all logging categories"""
        
        # Training logger
        self.loggers['training'] = self._create_logger(
            'training',
            self.log_dir / 'training' / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO
        )
        
        # Validation logger
        self.loggers['validation'] = self._create_logger(
            'validation',
            self.log_dir / 'validation' / f'validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO
        )
        
        # Error logger
        self.loggers['error'] = self._create_logger(
            'error',
            self.log_dir / 'errors' / f'errors_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.ERROR
        )
        
        # Performance logger
        self.loggers['performance'] = self._create_logger(
            'performance',
            self.log_dir / 'performance' / f'performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO
        )
        
        # System logger
        self.loggers['system'] = self._create_logger(
            'system',
            self.log_dir / 'system' / f'system_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO
        )
        
        # Debug logger
        self.loggers['debug'] = self._create_logger(
            'debug',
            self.log_dir / 'debug' / f'debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.DEBUG
        )
        
        # Audit logger
        self.loggers['audit'] = self._create_logger(
            'audit',
            self.log_dir / 'audit' / f'audit_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO
        )
    
    def _create_logger(self, name: str, log_file: Path, level: int = logging.INFO) -> logging.Logger:
        """Create a logger with file and console handlers"""
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler for errors and warnings
        if level <= logging.WARNING:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(logging.WARNING)
            logger.addHandler(console_handler)
        
        return logger
    
    def get_logger(self, category: str) -> logging.Logger:
        """Get logger by category"""
        return self.loggers.get(category, self.loggers['system'])
    
    def log_training_step(self, epoch: int, step: int, loss: float, lr: float, **kwargs):
        """Log training step information"""
        logger = self.get_logger('training')
        message = f"Epoch {epoch:03d}, Step {step:05d}: Loss={loss:.6f}, LR={lr:.8f}"
        
        if kwargs:
            extra_info = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
            message += f", {extra_info}"
        
        logger.info(message)
    
    def log_validation_results(self, epoch: int, metrics: Dict[str, Any]):
        """Log validation results"""
        logger = self.get_logger('validation')
        message = f"Epoch {epoch:03d} Validation: "
        message += ", ".join([f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" 
                             for k, v in metrics.items()])
        logger.info(message)
    
    def log_error(self, error: Exception, context: Optional[str] = None):
        """Log error with full traceback"""
        logger = self.get_logger('error')
        
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or 'Unknown'
        }
        
        logger.error(f"ERROR in {context}: {error_info}")
        
        # Also log to audit for critical errors
        if isinstance(error, (SystemError, MemoryError, KeyboardInterrupt)):
            self.log_audit_event('critical_error', error_info)
    
    def log_performance_metric(self, metric_name: str, value: float, unit: str = '', context: Optional[str] = None):
        """Log performance metric"""
        logger = self.get_logger('performance')
        message = f"{metric_name}: {value:.6f}{unit}"
        if context:
            message += f" ({context})"
        logger.info(message)
    
    def log_system_event(self, event: str, details: Optional[Dict[str, Any]] = None):
        """Log system event"""
        logger = self.get_logger('system')
        message = f"SYSTEM EVENT: {event}"
        if details:
            message += f" - {json.dumps(details)}"
        logger.info(message)
    
    def log_audit_event(self, event_type: str, details: Dict[str, Any]):
        """Log audit event"""
        logger = self.get_logger('audit')
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details
        }
        logger.info(json.dumps(audit_entry))
    
    def log_debug(self, message: str, **kwargs):
        """Log debug information"""
        logger = self.get_logger('debug')
        if kwargs:
            debug_info = json.dumps(kwargs)
            message += f" - {debug_info}"
        logger.debug(message)
    
    def flush_all(self):
        """Flush all loggers"""
        for logger in self.loggers.values():
            for handler in logger.handlers:
                handler.flush()
    
    def close_all(self):
        """Close all loggers"""
        for logger in self.loggers.values():
            for handler in logger.handlers:
                handler.close()
    
    def get_log_summary(self) -> Dict[str, Any]:
        """Get summary of all log files"""
        summary = {}
        
        for category, logger in self.loggers.items():
            log_files = []
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    log_file = Path(handler.baseFilename)
                    if log_file.exists():
                        log_files.append({
                            'path': str(log_file),
                            'size_mb': log_file.stat().st_size / (1024 * 1024),
                            'modified': datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                        })
            
            summary[category] = {
                'log_files': log_files,
                'total_size_mb': sum(f['size_mb'] for f in log_files)
            }
        
        return summary

# Global logger instance
_global_logger = None

def get_training_logger() -> TrainingLogger:
    """Get global training logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = TrainingLogger()
    return _global_logger

def setup_training_logging(log_dir: str = "/home/QuantNova/GrandModel/colab/logs") -> TrainingLogger:
    """Setup training logging system"""
    global _global_logger
    _global_logger = TrainingLogger(log_dir)
    return _global_logger

# Convenience functions
def log_training_step(epoch: int, step: int, loss: float, lr: float, **kwargs):
    """Convenience function for logging training steps"""
    get_training_logger().log_training_step(epoch, step, loss, lr, **kwargs)

def log_validation_results(epoch: int, metrics: Dict[str, Any]):
    """Convenience function for logging validation results"""
    get_training_logger().log_validation_results(epoch, metrics)

def log_error(error: Exception, context: Optional[str] = None):
    """Convenience function for logging errors"""
    get_training_logger().log_error(error, context)

def log_performance_metric(metric_name: str, value: float, unit: str = '', context: Optional[str] = None):
    """Convenience function for logging performance metrics"""
    get_training_logger().log_performance_metric(metric_name, value, unit, context)

def log_system_event(event: str, details: Optional[Dict[str, Any]] = None):
    """Convenience function for logging system events"""
    get_training_logger().log_system_event(event, details)

# Example usage
if __name__ == "__main__":
    # Setup logging
    logger = setup_training_logging()
    
    # Example usage
    logger.log_system_event("Training started", {"model": "MAPPO", "config": "tactical"})
    
    # Simulate training loop
    for epoch in range(3):
        for step in range(10):
            loss = 1.0 / (step + 1)
            lr = 0.001 * (0.9 ** epoch)
            
            logger.log_training_step(epoch, step, loss, lr, batch_size=32)
            logger.log_performance_metric("step_time", 0.5, "s", f"epoch_{epoch}_step_{step}")
        
        # Validation
        val_metrics = {"val_loss": 0.8 / (epoch + 1), "val_acc": 0.9 + 0.05 * epoch}
        logger.log_validation_results(epoch, val_metrics)
    
    # Log summary
    summary = logger.get_log_summary()
    print(f"Log Summary: {json.dumps(summary, indent=2)}")
    
    logger.log_system_event("Training completed")
    logger.flush_all()