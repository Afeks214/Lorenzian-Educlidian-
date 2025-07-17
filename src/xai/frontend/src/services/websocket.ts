/**
 * WebSocket Service for Real-time XAI Trading Explanations
 * 
 * Features:
 * - Automatic reconnection with exponential backoff
 * - Message queuing during disconnections
 * - Heartbeat/ping-pong for connection health
 * - Event-driven architecture with TypeScript support
 * - Connection status monitoring
 * - Multiple subscription management
 */

import { 
  WebSocketMessage, 
  WSMessageType, 
  DecisionUpdate, 
  ExplanationData,
  PerformanceMetrics,
  TradingDecision,
  SystemStatusResponse 
} from '../types';

export interface WebSocketConfig {
  url: string;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
  messageQueueSize?: number;
  autoConnect?: boolean;
}

export interface WebSocketEventHandlers {
  onDecisionUpdate?: (update: DecisionUpdate) => void;
  onExplanationReady?: (explanation: ExplanationData) => void;
  onPerformanceUpdate?: (metrics: PerformanceMetrics) => void;
  onRiskAlert?: (alert: any) => void;
  onSystemStatus?: (status: SystemStatusResponse) => void;
  onChatResponse?: (response: any) => void;
  onAgentStatus?: (status: any) => void;
  onConnect?: () => void;
  onDisconnect?: (reason: string) => void;
  onError?: (error: Error) => void;
  onReconnecting?: (attempt: number) => void;
}

export enum ConnectionStatus {
  DISCONNECTED = 'DISCONNECTED',
  CONNECTING = 'CONNECTING',
  CONNECTED = 'CONNECTED',
  RECONNECTING = 'RECONNECTING',
  ERROR = 'ERROR'
}

export interface ConnectionInfo {
  status: ConnectionStatus;
  uptime: number;
  reconnectCount: number;
  lastError?: string;
  latency?: number;
  messagesSent: number;
  messagesReceived: number;
}

export interface Subscription {
  id: string;
  type: WSMessageType | string;
  filters?: Record<string, any>;
  handler: (data: any) => void;
}

class WebSocketService {
  private ws: WebSocket | null = null;
  private config: WebSocketConfig;
  private eventHandlers: WebSocketEventHandlers = {};
  private subscriptions: Map<string, Subscription> = new Map();
  private messageQueue: WebSocketMessage[] = [];
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private connectionInfo: ConnectionInfo = {
    status: ConnectionStatus.DISCONNECTED,
    uptime: 0,
    reconnectCount: 0,
    messagesSent: 0,
    messagesReceived: 0
  };
  private connectTime: number = 0;
  private lastPingTime: number = 0;
  private isDestroyed = false;

  constructor(config: WebSocketConfig) {
    this.config = {
      reconnectInterval: 5000,
      maxReconnectAttempts: 10,
      heartbeatInterval: 30000,
      messageQueueSize: 100,
      autoConnect: true,
      ...config
    };

    if (this.config.autoConnect) {
      this.connect();
    }
  }

  /**
   * Connect to WebSocket server
   */
  public connect(): Promise<void> {
    if (this.isDestroyed) {
      throw new Error('WebSocketService has been destroyed');
    }

    return new Promise((resolve, reject) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }

      this.setStatus(ConnectionStatus.CONNECTING);
      this.connectTime = Date.now();

      try {
        this.ws = new WebSocket(this.config.url);
        
        this.ws.onopen = () => {
          this.handleConnect();
          resolve();
        };

        this.ws.onmessage = (event) => {
          this.handleMessage(event);
        };

        this.ws.onclose = (event) => {
          this.handleDisconnect(event.reason || 'Connection closed');
        };

        this.ws.onerror = (event) => {
          const error = new Error('WebSocket connection error');
          this.handleError(error);
          reject(error);
        };

      } catch (error) {
        const wsError = error instanceof Error ? error : new Error('Unknown WebSocket error');
        this.handleError(wsError);
        reject(wsError);
      }
    });
  }

  /**
   * Disconnect from WebSocket server
   */
  public disconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }

    if (this.ws) {
      this.ws.onopen = null;
      this.ws.onmessage = null;
      this.ws.onclose = null;
      this.ws.onerror = null;
      
      if (this.ws.readyState === WebSocket.OPEN) {
        this.ws.close(1000, 'Client disconnect');
      }
      this.ws = null;
    }

    this.setStatus(ConnectionStatus.DISCONNECTED);
  }

  /**
   * Destroy the WebSocket service
   */
  public destroy(): void {
    this.isDestroyed = true;
    this.disconnect();
    this.subscriptions.clear();
    this.messageQueue = [];
    this.eventHandlers = {};
  }

  /**
   * Send message to server
   */
  public send(message: WebSocketMessage): boolean {
    if (this.ws?.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(JSON.stringify(message));
        this.connectionInfo.messagesSent++;
        return true;
      } catch (error) {
        console.error('Failed to send WebSocket message:', error);
        return false;
      }
    } else {
      // Queue message if disconnected (within limits)
      if (this.messageQueue.length < (this.config.messageQueueSize || 100)) {
        this.messageQueue.push(message);
      } else {
        console.warn('Message queue is full, dropping message');
      }
      return false;
    }
  }

  /**
   * Subscribe to specific message types
   */
  public subscribe(
    type: WSMessageType | string,
    handler: (data: any) => void,
    filters?: Record<string, any>
  ): string {
    const id = `sub_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const subscription: Subscription = {
      id,
      type,
      filters,
      handler
    };

    this.subscriptions.set(id, subscription);

    // Send subscription message to server
    this.send({
      type: 'SUBSCRIBE' as WSMessageType,
      payload: {
        subscriptionId: id,
        messageType: type,
        filters
      },
      timestamp: new Date().toISOString()
    });

    return id;
  }

  /**
   * Unsubscribe from messages
   */
  public unsubscribe(subscriptionId: string): boolean {
    if (this.subscriptions.has(subscriptionId)) {
      this.subscriptions.delete(subscriptionId);
      
      // Send unsubscription message to server
      this.send({
        type: 'UNSUBSCRIBE' as WSMessageType,
        payload: {
          subscriptionId
        },
        timestamp: new Date().toISOString()
      });

      return true;
    }
    return false;
  }

  /**
   * Set event handlers
   */
  public setEventHandlers(handlers: WebSocketEventHandlers): void {
    this.eventHandlers = { ...this.eventHandlers, ...handlers };
  }

  /**
   * Get connection information
   */
  public getConnectionInfo(): ConnectionInfo {
    const now = Date.now();
    return {
      ...this.connectionInfo,
      uptime: this.connectTime > 0 ? now - this.connectTime : 0,
      latency: this.lastPingTime > 0 ? now - this.lastPingTime : undefined
    };
  }

  /**
   * Get list of active subscriptions
   */
  public getSubscriptions(): Subscription[] {
    return Array.from(this.subscriptions.values());
  }

  /**
   * Send ping to measure latency
   */
  public ping(): void {
    this.lastPingTime = Date.now();
    this.send({
      type: 'PING' as WSMessageType,
      payload: { timestamp: this.lastPingTime },
      timestamp: new Date().toISOString()
    });
  }

  // Private methods

  private handleConnect(): void {
    this.setStatus(ConnectionStatus.CONNECTED);
    this.connectionInfo.reconnectCount = 0;
    this.connectTime = Date.now();

    // Process queued messages
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      if (message) {
        this.send(message);
      }
    }

    // Start heartbeat
    this.startHeartbeat();

    // Resubscribe to all active subscriptions
    this.resubscribeAll();

    this.eventHandlers.onConnect?.();
  }

  private handleDisconnect(reason: string): void {
    this.setStatus(ConnectionStatus.DISCONNECTED);
    this.stopHeartbeat();

    this.eventHandlers.onDisconnect?.(reason);

    // Attempt reconnection if not manually disconnected
    if (!this.isDestroyed && reason !== 'Client disconnect') {
      this.attemptReconnect();
    }
  }

  private handleError(error: Error): void {
    this.connectionInfo.lastError = error.message;
    this.setStatus(ConnectionStatus.ERROR);
    this.eventHandlers.onError?.(error);
  }

  private handleMessage(event: MessageEvent): void {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);
      this.connectionInfo.messagesReceived++;

      // Handle ping/pong for latency measurement
      if (message.type === 'PONG' as WSMessageType) {
        this.connectionInfo.latency = Date.now() - this.lastPingTime;
        return;
      }

      // Route message to appropriate handler
      this.routeMessage(message);

    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
    }
  }

  private routeMessage(message: WebSocketMessage): void {
    // Check subscriptions first
    for (const subscription of this.subscriptions.values()) {
      if (subscription.type === message.type) {
        // Apply filters if specified
        if (this.matchesFilters(message.payload, subscription.filters)) {
          subscription.handler(message.payload);
        }
      }
    }

    // Route to specific event handlers
    switch (message.type) {
      case WSMessageType.DECISION_UPDATE:
        this.eventHandlers.onDecisionUpdate?.(message.payload as DecisionUpdate);
        break;
      
      case WSMessageType.EXPLANATION_READY:
        this.eventHandlers.onExplanationReady?.(message.payload as ExplanationData);
        break;
      
      case WSMessageType.PERFORMANCE_UPDATE:
        this.eventHandlers.onPerformanceUpdate?.(message.payload as PerformanceMetrics);
        break;
      
      case WSMessageType.RISK_ALERT:
        this.eventHandlers.onRiskAlert?.(message.payload);
        break;
      
      case WSMessageType.SYSTEM_STATUS:
        this.eventHandlers.onSystemStatus?.(message.payload as SystemStatusResponse);
        break;
      
      case WSMessageType.CHAT_RESPONSE:
        this.eventHandlers.onChatResponse?.(message.payload);
        break;
      
      case WSMessageType.AGENT_STATUS:
        this.eventHandlers.onAgentStatus?.(message.payload);
        break;
    }
  }

  private matchesFilters(payload: any, filters?: Record<string, any>): boolean {
    if (!filters) return true;

    for (const [key, value] of Object.entries(filters)) {
      if (payload[key] !== value) {
        return false;
      }
    }
    return true;
  }

  private setStatus(status: ConnectionStatus): void {
    this.connectionInfo.status = status;
  }

  private attemptReconnect(): void {
    if (this.connectionInfo.reconnectCount >= (this.config.maxReconnectAttempts || 10)) {
      console.error('Max reconnection attempts reached');
      this.setStatus(ConnectionStatus.ERROR);
      return;
    }

    this.setStatus(ConnectionStatus.RECONNECTING);
    this.connectionInfo.reconnectCount++;

    const delay = Math.min(
      (this.config.reconnectInterval || 5000) * Math.pow(2, this.connectionInfo.reconnectCount - 1),
      30000 // Max 30 seconds
    );

    this.eventHandlers.onReconnecting?.(this.connectionInfo.reconnectCount);

    this.reconnectTimer = setTimeout(() => {
      this.connect().catch(error => {
        console.error('Reconnection failed:', error);
        this.attemptReconnect();
      });
    }, delay);
  }

  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      this.ping();
    }, this.config.heartbeatInterval || 30000);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private resubscribeAll(): void {
    for (const subscription of this.subscriptions.values()) {
      this.send({
        type: 'SUBSCRIBE' as WSMessageType,
        payload: {
          subscriptionId: subscription.id,
          messageType: subscription.type,
          filters: subscription.filters
        },
        timestamp: new Date().toISOString()
      });
    }
  }
}

// Create and export a singleton instance
let wsService: WebSocketService | null = null;

export const createWebSocketService = (config: WebSocketConfig): WebSocketService => {
  if (wsService) {
    wsService.destroy();
  }
  wsService = new WebSocketService(config);
  return wsService;
};

export const getWebSocketService = (): WebSocketService | null => {
  return wsService;
};

export const destroyWebSocketService = (): void => {
  if (wsService) {
    wsService.destroy();
    wsService = null;
  }
};

export { WebSocketService, ConnectionStatus };
export type { WebSocketConfig, WebSocketEventHandlers, ConnectionInfo, Subscription };