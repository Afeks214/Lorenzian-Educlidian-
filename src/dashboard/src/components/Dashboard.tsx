/**
 * Main Dashboard Component
 * Human-in-the-Loop Risk Validation Dashboard
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Alert,
  Snackbar,
  CircularProgress,
  Chip,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  WifiOff as WifiOffIcon
} from '@mui/icons-material';

import { DashboardData, WebSocketMessage, CrisisAlert, FlaggedTrade } from '../types';
import { apiService } from '../services/api';
import { webSocketService } from '../services/websocket';
import SystemStatusPanel from './SystemStatusPanel';
import RiskOverviewPanel from './RiskOverviewPanel';
import PendingDecisionsPanel from './PendingDecisionsPanel';
import CrisisAlertsPanel from './CrisisAlertsPanel';
import DecisionHistoryPanel from './DecisionHistoryPanel';
import PerformanceMetricsPanel from './PerformanceMetricsPanel';

interface DashboardState {
  data: DashboardData | null;
  loading: boolean;
  error: string | null;
  wsConnected: boolean;
  lastUpdate: Date | null;
}

const Dashboard: React.FC = () => {
  const [state, setState] = useState<DashboardState>({
    data: null,
    loading: true,
    error: null,
    wsConnected: false,
    lastUpdate: null
  });

  const [notifications, setNotifications] = useState<{
    open: boolean;
    message: string;
    severity: 'info' | 'warning' | 'error' | 'success';
  }>({
    open: false,
    message: '',
    severity: 'info'
  });

  const showNotification = useCallback((message: string, severity: 'info' | 'warning' | 'error' | 'success' = 'info') => {
    setNotifications({ open: true, message, severity });
  }, []);

  const handleCloseNotification = () => {
    setNotifications(prev => ({ ...prev, open: false }));
  };

  // Load initial dashboard data
  const loadDashboardData = useCallback(async () => {
    try {
      setState(prev => ({ ...prev, loading: true, error: null }));
      const data = await apiService.getDashboardData();
      setState(prev => ({
        ...prev,
        data,
        loading: false,
        lastUpdate: new Date()
      }));
    } catch (error: any) {
      setState(prev => ({
        ...prev,
        loading: false,
        error: error.message
      }));
      showNotification('Failed to load dashboard data', 'error');
    }
  }, [showNotification]);

  // Handle WebSocket events
  const handleWebSocketMessage = useCallback((type: string, data: any) => {
    switch (type) {
      case 'dashboard_update':
        setState(prev => ({
          ...prev,
          data: data.data,
          lastUpdate: new Date()
        }));
        break;

      case 'trade_flagged':
        const tradeData = data.data as FlaggedTrade;
        showNotification(
          `New trade flagged for review: ${tradeData.symbol} ${tradeData.direction}`,
          'warning'
        );
        // Play notification sound if enabled
        if ('serviceWorker' in navigator) {
          navigator.serviceWorker.ready.then(registration => {
            registration.showNotification('Trade Flagged', {
              body: `${tradeData.symbol} ${tradeData.direction} requires human review`,
              icon: '/notification-icon.png',
              badge: '/badge-icon.png',
              tag: 'trade-flagged',
              requireInteraction: true
            });
          });
        }
        break;

      case 'crisis_alert':
        const alertData = data.data as CrisisAlert;
        showNotification(
          `Crisis Alert: ${alertData.message}`,
          alertData.severity === 'CRITICAL' ? 'error' : 'warning'
        );
        // Show browser notification for crisis alerts
        if ('serviceWorker' in navigator) {
          navigator.serviceWorker.ready.then(registration => {
            registration.showNotification('Crisis Alert', {
              body: alertData.message,
              icon: '/crisis-icon.png',
              badge: '/badge-icon.png',
              tag: 'crisis-alert',
              requireInteraction: true,
              vibrate: [200, 100, 200]
            });
          });
        }
        break;

      case 'decision_made':
        showNotification(
          `Decision processed: ${data.data.decision} for trade ${data.data.trade_id}`,
          'success'
        );
        break;

      case 'connection_status':
        setState(prev => ({
          ...prev,
          wsConnected: data.connected
        }));
        if (data.connected) {
          showNotification('Real-time connection established', 'success');
        } else {
          showNotification('Real-time connection lost', 'warning');
        }
        break;

      case 'connection_error':
        showNotification(`Connection error: ${data.error}`, 'error');
        break;

      default:
        console.log('Unhandled WebSocket message type:', type);
    }
  }, [showNotification]);

  // Initialize dashboard
  useEffect(() => {
    const initializeDashboard = async () => {
      // Load initial data
      await loadDashboardData();

      // Setup WebSocket connection
      const token = localStorage.getItem('access_token');
      if (token) {
        webSocketService.connect(token);

        // Setup WebSocket event listeners
        webSocketService.on('dashboard_update', (data) => handleWebSocketMessage('dashboard_update', { data }));
        webSocketService.on('trade_flagged', (data) => handleWebSocketMessage('trade_flagged', { data }));
        webSocketService.on('crisis_alert', (data) => handleWebSocketMessage('crisis_alert', { data }));
        webSocketService.on('decision_made', (data) => handleWebSocketMessage('decision_made', { data }));
        webSocketService.on('connection_status', (data) => handleWebSocketMessage('connection_status', data));
        webSocketService.on('connection_error', (data) => handleWebSocketMessage('connection_error', data));
      }
    };

    initializeDashboard();

    // Cleanup on unmount
    return () => {
      webSocketService.disconnect();
    };
  }, [loadDashboardData, handleWebSocketMessage]);

  // Auto-refresh fallback (in case WebSocket fails)
  useEffect(() => {
    const interval = setInterval(() => {
      if (!state.wsConnected) {
        loadDashboardData();
      }
    }, 5000); // 5 second fallback refresh

    return () => clearInterval(interval);
  }, [state.wsConnected, loadDashboardData]);

  // Handle decision approval/rejection
  const handleDecision = async (tradeId: string, decision: 'APPROVE' | 'REJECT', reasoning: string) => {
    try {
      const user = apiService.getCurrentUser();
      if (!user) {
        throw new Error('User not authenticated');
      }

      await apiService.makeDecision({
        trade_id: tradeId,
        decision,
        reasoning,
        user_id: user.user_id,
        timestamp: new Date().toISOString()
      });

      showNotification(`Trade ${decision.toLowerCase()}d successfully`, 'success');
      
      // Refresh dashboard data
      await loadDashboardData();
    } catch (error: any) {
      showNotification(`Failed to ${decision.toLowerCase()} trade: ${error.message}`, 'error');
    }
  };

  if (state.loading && !state.data) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="80vh">
        <CircularProgress size={60} />
        <Typography variant="h6" sx={{ ml: 2 }}>
          Loading dashboard...
        </Typography>
      </Box>
    );
  }

  if (state.error && !state.data) {
    return (
      <Box p={3}>
        <Alert severity="error" action={
          <IconButton onClick={loadDashboardData} color="inherit" size="small">
            <RefreshIcon />
          </IconButton>
        }>
          {state.error}
        </Alert>
      </Box>
    );
  }

  return (
    <Box p={3}>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          Risk Management Dashboard
        </Typography>
        
        <Box display="flex" alignItems="center" gap={2}>
          {/* Connection Status */}
          <Tooltip title={state.wsConnected ? 'Real-time connection active' : 'Real-time connection inactive'}>
            <Chip
              icon={state.wsConnected ? <CheckCircleIcon /> : <WifiOffIcon />}
              label={state.wsConnected ? 'Live' : 'Offline'}
              color={state.wsConnected ? 'success' : 'warning'}
              variant="outlined"
            />
          </Tooltip>

          {/* Last Update */}
          {state.lastUpdate && (
            <Typography variant="body2" color="text.secondary">
              Last update: {state.lastUpdate.toLocaleTimeString()}
            </Typography>
          )}

          {/* Manual Refresh */}
          <Tooltip title="Refresh dashboard">
            <IconButton onClick={loadDashboardData} disabled={state.loading}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Dashboard Grid */}
      <Grid container spacing={3}>
        {/* System Status Panel */}
        <Grid item xs={12} md={6}>
          <SystemStatusPanel 
            agentStatuses={state.data?.agent_statuses || []}
            wsConnected={state.wsConnected}
          />
        </Grid>

        {/* Risk Overview Panel */}
        <Grid item xs={12} md={6}>
          <RiskOverviewPanel 
            riskMetrics={state.data?.risk_metrics}
          />
        </Grid>

        {/* Pending Decisions Panel */}
        <Grid item xs={12} lg={8}>
          <PendingDecisionsPanel 
            flaggedTrades={state.data?.flagged_trades || []}
            onDecision={handleDecision}
          />
        </Grid>

        {/* Crisis Alerts Panel */}
        <Grid item xs={12} lg={4}>
          <CrisisAlertsPanel 
            crisisAlerts={state.data?.crisis_alerts || []}
          />
        </Grid>

        {/* Performance Metrics Panel */}
        <Grid item xs={12} md={6}>
          <PerformanceMetricsPanel />
        </Grid>

        {/* Decision History Panel */}
        <Grid item xs={12} md={6}>
          <DecisionHistoryPanel />
        </Grid>
      </Grid>

      {/* Notifications */}
      <Snackbar
        open={notifications.open}
        autoHideDuration={6000}
        onClose={handleCloseNotification}
        anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
      >
        <Alert 
          onClose={handleCloseNotification} 
          severity={notifications.severity}
          variant="filled"
        >
          {notifications.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default Dashboard;