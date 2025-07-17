import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { io, Socket } from 'socket.io-client';
import { useAuth } from './AuthContext';

interface WebSocketContextType {
  socket: Socket | null;
  isConnected: boolean;
  lastMessage: MessageEvent | null;
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

interface WebSocketProviderProps {
  children: ReactNode;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children }) => {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<MessageEvent | null>(null);
  const { user, isAuthenticated } = useAuth();

  useEffect(() => {
    if (isAuthenticated && user) {
      // Create WebSocket connection
      const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';
      const newSocket = io(wsUrl, {
        auth: {
          token: localStorage.getItem('auth_token')
        }
      });

      newSocket.on('connect', () => {
        console.log('WebSocket connected');
        setIsConnected(true);
      });

      newSocket.on('disconnect', () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
      });

      newSocket.on('message', (data: any) => {
        // Create a MessageEvent-like object
        const messageEvent = {
          data: JSON.stringify(data)
        } as MessageEvent;
        setLastMessage(messageEvent);
      });

      newSocket.on('new_decision', (data: any) => {
        const messageEvent = {
          data: JSON.stringify({ type: 'new_decision', ...data })
        } as MessageEvent;
        setLastMessage(messageEvent);
      });

      newSocket.on('decision_update', (data: any) => {
        const messageEvent = {
          data: JSON.stringify({ type: 'decision_update', ...data })
        } as MessageEvent;
        setLastMessage(messageEvent);
      });

      setSocket(newSocket);

      return () => {
        newSocket.close();
        setSocket(null);
        setIsConnected(false);
      };
    }
  }, [isAuthenticated, user]);

  const value: WebSocketContextType = {
    socket,
    isConnected,
    lastMessage
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};

export const useWebSocket = (): WebSocketContextType => {
  const context = useContext(WebSocketContext);
  if (context === undefined) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};