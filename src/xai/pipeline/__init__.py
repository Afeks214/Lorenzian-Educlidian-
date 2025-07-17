"""
XAI Real-time Pipeline Module

Implements the real-time explanation pipeline infrastructure for trading decisions,
including decision capture, WebSocket streaming, context processing, and explanation delivery.
"""

from .streaming_engine import StreamingEngine
from .decision_capture import DecisionCapture
from .websocket_manager import WebSocketManager
from .context_processor import ContextProcessor

__all__ = [
    "StreamingEngine",
    "DecisionCapture",
    "WebSocketManager", 
    "ContextProcessor"
]