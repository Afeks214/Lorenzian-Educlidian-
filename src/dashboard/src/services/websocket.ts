/**
 * WebSocket service for real-time dashboard updates
 */

import { io, Socket } from 'socket.io-client';
import { WebSocketMessage } from '../types';

class WebSocketService {
  private socket: Socket | null = null;
  private isConnected = false;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectInterval = 1000;
  private listeners: Map<string, ((data: any) => void)[]> = new Map();

  connect(token: string): void {
    if (this.socket) {
      this.disconnect();
    }

    const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8001';
    
    this.socket = io(wsUrl, {
      auth: {
        token: token
      },
      transports: ['websocket'],
      upgrade: false,
      rememberUpgrade: false
    });

    this.setupEventHandlers();
  }

  private setupEventHandlers(): void {
    if (!this.socket) return;

    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      this.isConnected = true;
      this.reconnectAttempts = 0;
      this.emit('connection_status', { connected: true });
    });

    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      this.isConnected = false;
      this.emit('connection_status', { connected: false, reason });
      
      if (reason === 'io server disconnect') {
        // Server initiated disconnect, try to reconnect
        this.handleReconnect();
      }
    });

    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      this.isConnected = false;
      this.emit('connection_error', { error: error.message });
      this.handleReconnect();
    });

    // Handle incoming messages
    this.socket.on('message', (message: WebSocketMessage) => {
      this.handleMessage(message);
    });

    // Handle specific event types
    this.socket.on('dashboard_update', (data) => {
      this.emit('dashboard_update', data);
    });

    this.socket.on('trade_flagged', (data) => {
      this.emit('trade_flagged', data);
    });

    this.socket.on('crisis_alert', (data) => {
      this.emit('crisis_alert', data);
    });

    this.socket.on('decision_made', (data) => {
      this.emit('decision_made', data);
    });

    this.socket.on('initial_data', (data) => {
      this.emit('initial_data', data);
    });

    // Handle ping/pong for keepalive
    this.socket.on('ping', () => {
      this.socket?.emit('pong');
    });
  }

  private handleMessage(message: WebSocketMessage): void {
    console.log('Received WebSocket message:', message);
    this.emit(message.type, message.data);
  }

  private handleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      this.emit('max_reconnect_attempts_reached', {});
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectInterval * Math.pow(2, this.reconnectAttempts - 1);
    
    console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts})`);
    
    setTimeout(() => {
      if (this.socket && !this.isConnected) {
        this.socket.connect();
      }
    }, delay);
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    this.isConnected = false;
    this.listeners.clear();
  }

  on(eventType: string, callback: (data: any) => void): void {
    if (!this.listeners.has(eventType)) {
      this.listeners.set(eventType, []);
    }
    this.listeners.get(eventType)!.push(callback);
  }

  off(eventType: string, callback: (data: any) => void): void {
    const callbacks = this.listeners.get(eventType);
    if (callbacks) {
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  private emit(eventType: string, data: any): void {
    const callbacks = this.listeners.get(eventType);
    if (callbacks) {
      callbacks.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error('Error in WebSocket event callback:', error);
        }
      });
    }
  }

  sendMessage(type: string, data: any): void {
    if (this.socket && this.isConnected) {
      this.socket.emit('message', { type, data, timestamp: new Date().toISOString() });
    } else {
      console.warn('Cannot send message: WebSocket not connected');
    }
  }

  getConnectionStatus(): boolean {
    return this.isConnected;
  }

  getReconnectAttempts(): number {
    return this.reconnectAttempts;
  }
}

export const webSocketService = new WebSocketService();
export default webSocketService;