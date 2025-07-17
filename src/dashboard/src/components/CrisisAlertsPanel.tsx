/**
 * Crisis Alerts Panel
 * Shows crisis detection alerts from Meta-Learning Agent
 */

import React, { useState } from 'react';
import {
  Paper,
  Typography,
  Box,
  Alert,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  IconButton,
  Collapse,
  Card,
  CardContent,
  Grid,
  Tooltip,
  Badge
} from '@mui/material';
import {
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  ReportProblem as ReportProblemIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  TrendingDown as TrendingDownIcon,
  Speed as SpeedIcon,
  Timeline as TimelineIcon
} from '@mui/icons-material';
import { CrisisAlert } from '../types';

interface CrisisAlertsPanelProps {
  crisisAlerts: CrisisAlert[];
}

const CrisisAlertsPanel: React.FC<CrisisAlertsPanelProps> = ({ crisisAlerts }) => {
  const [expandedAlerts, setExpandedAlerts] = useState<Set<string>>(new Set());

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'CRITICAL':
        return <ErrorIcon sx={{ color: '#f44336' }} />;
      case 'HIGH':
        return <ReportProblemIcon sx={{ color: '#ff5722' }} />;
      case 'MEDIUM':
        return <WarningIcon sx={{ color: '#ff9800' }} />;
      case 'LOW':
        return <InfoIcon sx={{ color: '#2196f3' }} />;
      default:
        return <InfoIcon sx={{ color: '#607d8b' }} />;
    }
  };

  const getSeverityColor = (severity: string): 'error' | 'warning' | 'info' | 'success' => {
    switch (severity) {
      case 'CRITICAL':
        return 'error';
      case 'HIGH':
        return 'error';
      case 'MEDIUM':
        return 'warning';
      case 'LOW':
        return 'info';
      default:
        return 'info';
    }
  };

  const getAlertTypeIcon = (alertType: string) => {
    switch (alertType.toLowerCase()) {
      case 'market_crash':
      case 'volatility_spike':
        return <TrendingDownIcon />;
      case 'liquidity_crisis':
        return <SpeedIcon />;
      case 'correlation_breakdown':
        return <TimelineIcon />;
      default:
        return <WarningIcon />;
    }
  };

  const toggleExpanded = (alertId: string) => {
    const newExpanded = new Set(expandedAlerts);
    if (newExpanded.has(alertId)) {
      newExpanded.delete(alertId);
    } else {
      newExpanded.add(alertId);
    }
    setExpandedAlerts(newExpanded);
  };

  const getTimeAgo = (timestamp: string): string => {
    const now = new Date();
    const alertTime = new Date(timestamp);
    const diffMs = now.getTime() - alertTime.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return alertTime.toLocaleDateString();
  };

  const criticalAlerts = crisisAlerts.filter(alert => alert.severity === 'CRITICAL').length;
  const highAlerts = crisisAlerts.filter(alert => alert.severity === 'HIGH').length;
  const mediumAlerts = crisisAlerts.filter(alert => alert.severity === 'MEDIUM').length;

  return (
    <Paper elevation={3} sx={{ height: '100%' }}>
      <Box p={3}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6" component="h2">
            Crisis Alerts
          </Typography>
          <Box display="flex" gap={1}>
            {criticalAlerts > 0 && (
              <Badge badgeContent={criticalAlerts} color="error">
                <Chip label="CRITICAL" color="error" size="small" />
              </Badge>
            )}
            {highAlerts > 0 && (
              <Badge badgeContent={highAlerts} color="error">
                <Chip label="HIGH" color="warning" size="small" />
              </Badge>
            )}
            {mediumAlerts > 0 && (
              <Badge badgeContent={mediumAlerts} color="warning">
                <Chip label="MEDIUM" color="info" size="small" />
              </Badge>
            )}
          </Box>
        </Box>

        {crisisAlerts.length === 0 ? (
          <Alert severity="success">
            No crisis alerts. All systems operating normally.
          </Alert>
        ) : (
          <Box sx={{ maxHeight: '60vh', overflowY: 'auto' }}>
            {crisisAlerts
              .sort((a, b) => {
                // Sort by severity (CRITICAL first) then by time (newest first)
                const severityOrder = { 'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3 };
                const severityDiff = (severityOrder[a.severity as keyof typeof severityOrder] || 4) - 
                                   (severityOrder[b.severity as keyof typeof severityOrder] || 4);
                if (severityDiff !== 0) return severityDiff;
                return new Date(b.detected_at).getTime() - new Date(a.detected_at).getTime();
              })
              .map((alert) => {
                const isExpanded = expandedAlerts.has(alert.alert_id);
                
                return (
                  <Card 
                    key={alert.alert_id} 
                    sx={{ 
                      mb: 2,
                      border: `2px solid ${
                        alert.severity === 'CRITICAL' ? '#f44336' :
                        alert.severity === 'HIGH' ? '#ff5722' :
                        alert.severity === 'MEDIUM' ? '#ff9800' : '#2196f3'
                      }`,
                      bgcolor: alert.severity === 'CRITICAL' ? '#ffebee' : 'inherit'
                    }}
                  >
                    <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                      <Box display="flex" alignItems="flex-start" gap={2}>
                        <Box sx={{ mt: 0.5 }}>
                          {getSeverityIcon(alert.severity)}
                        </Box>
                        
                        <Box flex={1}>
                          <Box display="flex" justifyContent="between" alignItems="flex-start" mb={1}>
                            <Box flex={1}>
                              <Typography variant="subtitle1" fontWeight="bold">
                                {alert.alert_type.replace(/_/g, ' ').toUpperCase()}
                              </Typography>
                              <Typography variant="body2" color="text.secondary" gutterBottom>
                                {alert.message}
                              </Typography>
                            </Box>
                            
                            <Box display="flex" alignItems="center" gap={1}>
                              <Chip
                                label={alert.severity}
                                color={getSeverityColor(alert.severity)}
                                size="small"
                              />
                              <Tooltip title="Show details">
                                <IconButton
                                  onClick={() => toggleExpanded(alert.alert_id)}
                                  size="small"
                                >
                                  {isExpanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                                </IconButton>
                              </Tooltip>
                            </Box>
                          </Box>
                          
                          <Typography variant="caption" color="text.secondary">
                            Detected {getTimeAgo(alert.detected_at)}
                          </Typography>
                        </Box>
                      </Box>

                      <Collapse in={isExpanded}>
                        <Box mt={2} p={2} bgcolor="grey.50" borderRadius={1}>
                          {/* Metrics */}
                          {Object.keys(alert.metrics).length > 0 && (
                            <Box mb={2}>
                              <Typography variant="subtitle2" gutterBottom>
                                Key Metrics:
                              </Typography>
                              <Grid container spacing={1}>
                                {Object.entries(alert.metrics).map(([key, value]) => (
                                  <Grid item xs={6} key={key}>
                                    <Typography variant="body2">
                                      <strong>{key.replace(/_/g, ' ')}:</strong> {
                                        typeof value === 'number' ? 
                                          (value < 1 ? (value * 100).toFixed(2) + '%' : value.toFixed(4)) :
                                          value
                                      }
                                    </Typography>
                                  </Grid>
                                ))}
                              </Grid>
                            </Box>
                          )}
                          
                          {/* Recommended Actions */}
                          {alert.recommended_actions.length > 0 && (
                            <Box>
                              <Typography variant="subtitle2" gutterBottom>
                                Recommended Actions:
                              </Typography>
                              <List dense>
                                {alert.recommended_actions.map((action, index) => (
                                  <ListItem key={index} sx={{ pl: 0 }}>
                                    <ListItemIcon sx={{ minWidth: 32 }}>
                                      {getAlertTypeIcon(alert.alert_type)}
                                    </ListItemIcon>
                                    <ListItemText 
                                      primary={action}
                                      primaryTypographyProps={{ variant: 'body2' }}
                                    />
                                  </ListItem>
                                ))}
                              </List>
                            </Box>
                          )}
                          
                          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                            Alert ID: {alert.alert_id}
                          </Typography>
                        </Box>
                      </Collapse>
                    </CardContent>
                  </Card>
                );
              })}
          </Box>
        )}

        {/* Summary Alert for Critical Situations */}
        {criticalAlerts > 0 && (
          <Alert severity="error" sx={{ mt: 2 }}>
            <Typography variant="body2" fontWeight="bold">
              {criticalAlerts} CRITICAL alert{criticalAlerts > 1 ? 's' : ''} require immediate attention!
            </Typography>
          </Alert>
        )}
      </Box>
    </Paper>
  );
};

export default CrisisAlertsPanel;