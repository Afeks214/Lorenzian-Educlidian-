/**
 * System Status Panel
 * Shows MARL agent statuses and system health
 */

import React from 'react';
import {
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  Grid,
  Chip,
  LinearProgress,
  Avatar,
  Tooltip,
  Alert
} from '@mui/material';
import {
  SmartToy as SmartToyIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Wifi as WifiIcon,
  WifiOff as WifiOffIcon
} from '@mui/icons-material';
import { AgentStatus } from '../types';

interface SystemStatusPanelProps {
  agentStatuses: AgentStatus[];
  wsConnected: boolean;
}

const SystemStatusPanel: React.FC<SystemStatusPanelProps> = ({ agentStatuses, wsConnected }) => {
  const getStatusColor = (status: string): 'success' | 'warning' | 'error' => {
    switch (status.toLowerCase()) {
      case 'active':
        return 'success';
      case 'inactive':
        return 'warning';
      case 'error':
        return 'error';
      default:
        return 'warning';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case 'active':
        return <CheckCircleIcon sx={{ color: '#4caf50' }} />;
      case 'inactive':
        return <WarningIcon sx={{ color: '#ff9800' }} />;
      case 'error':
        return <ErrorIcon sx={{ color: '#f44336' }} />;
      default:
        return <WarningIcon sx={{ color: '#ff9800' }} />;
    }
  };

  const getAgentIcon = (agentName: string) => {
    const color = agentName.includes('strategic') ? '#2196f3' : 
                  agentName.includes('tactical') ? '#ff5722' :
                  agentName.includes('risk') ? '#f44336' :
                  agentName.includes('portfolio') ? '#9c27b0' : '#607d8b';
    
    return (
      <Avatar sx={{ bgcolor: color, width: 32, height: 32 }}>
        <SmartToyIcon fontSize="small" />
      </Avatar>
    );
  };

  const getRecommendationColor = (recommendation: string): string => {
    switch (recommendation.toLowerCase()) {
      case 'long':
      case 'buy':
        return '#4caf50';
      case 'short':
      case 'sell':
        return '#f44336';
      case 'hold':
        return '#ff9800';
      case 'reduce_exposure':
        return '#f44336';
      case 'rebalance':
        return '#2196f3';
      default:
        return '#607d8b';
    }
  };

  const activeAgents = agentStatuses.filter(agent => agent.status === 'active').length;
  const totalAgents = agentStatuses.length;
  const systemHealth = totalAgents > 0 ? (activeAgents / totalAgents) * 100 : 0;

  return (
    <Paper elevation={3} sx={{ height: '100%' }}>
      <Box p={3}>
        <Typography variant="h6" component="h2" gutterBottom>
          System Status
        </Typography>

        {/* System Health Summary */}
        <Box mb={3}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
            <Typography variant="body2" color="text.secondary">
              System Health
            </Typography>
            <Typography variant="body2" fontWeight="bold">
              {systemHealth.toFixed(0)}%
            </Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={systemHealth}
            sx={{
              height: 8,
              borderRadius: 4,
              bgcolor: 'grey.200',
              '& .MuiLinearProgress-bar': {
                bgcolor: systemHealth >= 80 ? '#4caf50' : systemHealth >= 60 ? '#ff9800' : '#f44336'
              }
            }}
          />
          <Box display="flex" justifyContent="space-between" alignItems="center" mt={1}>
            <Typography variant="caption" color="text.secondary">
              {activeAgents}/{totalAgents} agents active
            </Typography>
            <Chip
              icon={wsConnected ? <WifiIcon /> : <WifiOffIcon />}
              label={wsConnected ? 'Connected' : 'Disconnected'}
              color={wsConnected ? 'success' : 'error'}
              size="small"
              variant="outlined"
            />
          </Box>
        </Box>

        {/* Agent Status Cards */}
        {agentStatuses.length === 0 ? (
          <Alert severity="warning">
            No agent status data available
          </Alert>
        ) : (
          <Grid container spacing={2}>
            {agentStatuses.map((agent) => (
              <Grid item xs={12} key={agent.agent_name}>
                <Card 
                  variant="outlined" 
                  sx={{ 
                    borderColor: agent.status === 'active' ? '#4caf50' : 
                                 agent.status === 'inactive' ? '#ff9800' : '#f44336',
                    borderWidth: 2
                  }}
                >
                  <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                    <Box display="flex" alignItems="center" gap={2}>
                      {/* Agent Icon */}
                      {getAgentIcon(agent.agent_name)}
                      
                      {/* Agent Info */}
                      <Box flex={1}>
                        <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                          <Typography variant="subtitle2" fontWeight="bold">
                            {agent.agent_name.replace(/_/g, ' ').toUpperCase()}
                          </Typography>
                          <Tooltip title={`Status: ${agent.status}`}>
                            {getStatusIcon(agent.status)}
                          </Tooltip>
                        </Box>
                        
                        {/* Performance Score */}
                        <Box mb={1}>
                          <Box display="flex" justifyContent="space-between" alignItems="center">
                            <Typography variant="caption" color="text.secondary">
                              Performance
                            </Typography>
                            <Typography variant="caption" fontWeight="bold">
                              {(agent.performance_score * 100).toFixed(1)}%
                            </Typography>
                          </Box>
                          <LinearProgress
                            variant="determinate"
                            value={agent.performance_score * 100}
                            sx={{
                              height: 4,
                              borderRadius: 2,
                              bgcolor: 'grey.200',
                              '& .MuiLinearProgress-bar': {
                                bgcolor: agent.performance_score >= 0.8 ? '#4caf50' : 
                                         agent.performance_score >= 0.6 ? '#ff9800' : '#f44336'
                              }
                            }}
                          />
                        </Box>
                        
                        {/* Current Recommendation */}
                        <Box display="flex" justifyContent="space-between" alignItems="center">
                          <Typography variant="caption" color="text.secondary">
                            Recommendation:
                          </Typography>
                          <Chip
                            label={agent.current_recommendation}
                            size="small"
                            sx={{
                              bgcolor: getRecommendationColor(agent.current_recommendation),
                              color: 'white',
                              fontWeight: 'bold',
                              fontSize: '0.7rem'
                            }}
                          />
                        </Box>
                        
                        {/* Confidence */}
                        <Box display="flex" justifyContent="space-between" alignItems="center" mt={0.5}>
                          <Typography variant="caption" color="text.secondary">
                            Confidence:
                          </Typography>
                          <Typography variant="caption" fontWeight="bold">
                            {(agent.confidence * 100).toFixed(1)}%
                          </Typography>
                        </Box>
                        
                        {/* Last Update */}
                        <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                          Last update: {new Date(agent.last_update).toLocaleTimeString()}
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        )}

        {/* System Alerts */}
        {systemHealth < 80 && (
          <Alert severity="warning" sx={{ mt: 2 }}>
            System performance is degraded. {totalAgents - activeAgents} agent(s) not responding.
          </Alert>
        )}
        
        {!wsConnected && (
          <Alert severity="error" sx={{ mt: 2 }}>
            Real-time connection lost. Dashboard may show stale data.
          </Alert>
        )}
      </Box>
    </Paper>
  );
};

export default SystemStatusPanel;