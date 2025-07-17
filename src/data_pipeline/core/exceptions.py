"""Data pipeline exceptions and error handling"""

class DataPipelineException(Exception):
    """Base exception for data pipeline errors"""
    pass

class DataLoadingException(DataPipelineException):
    """Exception raised during data loading operations"""
    pass

class DataStreamingException(DataPipelineException):
    """Exception raised during data streaming operations"""
    pass

class DataValidationException(DataPipelineException):
    """Exception raised during data validation"""
    pass

class DataCachingException(DataPipelineException):
    """Exception raised during caching operations"""
    pass

class DataPreprocessingException(DataPipelineException):
    """Exception raised during data preprocessing"""
    pass

class PerformanceException(DataPipelineException):
    """Exception raised for performance-related issues"""
    pass