"""
XAI API Package
AGENT DELTA MISSION: Production XAI API Backend

This package provides a production-grade FastAPI backend for Explainable AI (XAI)
with seamless Strategic MARL integration, natural language query processing,
and real-time WebSocket communication.

Author: Agent Delta - Integration Specialist
Version: 1.0 - Production XAI API Backend
"""

from .main import app
from .models import *
from .integration import StrategicMARLIntegrator
from .query_engine import NaturalLanguageQueryEngine
from .websocket_handlers import WebSocketConnectionManager

__version__ = "1.0.0"
__author__ = "Agent Delta"
__description__ = "Production XAI API Backend with Strategic MARL Integration"

__all__ = [
    "app",
    "StrategicMARLIntegrator", 
    "NaturalLanguageQueryEngine",
    "WebSocketConnectionManager",
    # Models
    "ExplanationRequest",
    "ExplanationResponse", 
    "QueryRequest",
    "QueryResponse",
    "DecisionHistoryResponse",
    "PerformanceAnalyticsResponse",
    "ComplianceReportRequest",
    "ComplianceReportResponse",
    "HealthCheckResponse",
    "ErrorResponse"
]