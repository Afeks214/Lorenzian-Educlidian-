/**
 * Performance Metrics Panel
 * Shows decision processing performance and system metrics
 */

import React, { useState, useEffect } from 'react';
import {
  Paper,
  Typography,
  Box,
  Grid,
  Card,
  CardContent,
  LinearProgress,
  Chip,
  Alert,
  CircularProgress,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  Speed as SpeedIcon,
  Timeline as TimelineIcon,
  CheckCircle as CheckCircleIcon,
  ThumbUp as ThumbUpIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';
import { PerformanceMetrics } from '../types';
import { apiService } from '../services/api';

const PerformanceMetricsPanel: React.FC = () => {
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadPerformanceMetrics = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiService.getPerformanceMetrics();
      setMetrics(data);
    } catch (error: any) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadPerformanceMetrics();
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(loadPerformanceMetrics, 30000);
    return () => clearInterval(interval);
  }, []);

  const formatNumber = (value: number): string => {
    return value.toLocaleString();
  };

  const formatTime = (timeMs: number): string => {
    if (timeMs < 1000) {
      return `${timeMs.toFixed(0)}ms`;
    }
    return `${(timeMs / 1000).toFixed(2)}s`;
  };

  const formatPercentage = (value: number): string => {
    return `${(value * 100).toFixed(1)}%`;
  };

  const getPerformanceColor = (value: number, thresholds: { good: number; warning: number }): 'success' | 'warning' | 'error' => {
    if (value <= thresholds.good) return 'success';
    if (value <= thresholds.warning) return 'warning';
    return 'error';
  };

  const getApprovalRateColor = (rate: number): 'success' | 'warning' | 'error' => {
    if (rate >= 0.7) return 'success';
    if (rate >= 0.4) return 'warning';
    return 'error';
  };

  if (loading) {
    return (
      <Paper elevation={3} sx={{ height: '100%', p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Performance Metrics
        </Typography>
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
          <CircularProgress />
        </Box>
      </Paper>
    );
  }

  if (error) {
    return (
      <Paper elevation={3} sx={{ height: '100%', p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Performance Metrics
        </Typography>
        <Alert severity="error">
          {error}
        </Alert>
      </Paper>
    );
  }

  if (!metrics) {
    return (
      <Paper elevation={3} sx={{ height: '100%', p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Performance Metrics
        </Typography>
        <Alert severity="info">
          No performance data available
        </Alert>
      </Paper>
    );
  }

  const performanceCards = [
    {
      title: 'Total Decisions',
      value: metrics.total_decisions,
      formatter: formatNumber,
      icon: <CheckCircleIcon />,
      color: '#2196f3',
      description: 'Lifetime decisions processed'
    },
    {
      title: 'Pending Decisions',
      value: metrics.pending_decisions,
      formatter: formatNumber,
      icon: <TimelineIcon />,
      color: metrics.pending_decisions > 5 ? '#ff9800' : '#4caf50',
      description: 'Currently awaiting review',
      alert: metrics.pending_decisions > 10 ? 'High backlog detected' : undefined
    },
    {
      title: 'Avg Processing Time',
      value: metrics.average_processing_time_ms,
      formatter: formatTime,
      icon: <SpeedIcon />,
      color: metrics.average_processing_time_ms > 5000 ? '#f44336' : 
             metrics.average_processing_time_ms > 2000 ? '#ff9800' : '#4caf50',
      description: 'Average human decision time',
      target: '< 2s for optimal user experience'
    },
    {
      title: 'Approval Rate',
      value: metrics.approval_rate,
      formatter: formatPercentage,
      icon: <ThumbUpIcon />,
      color: metrics.approval_rate >= 0.7 ? '#4caf50' : 
             metrics.approval_rate >= 0.4 ? '#ff9800' : '#f44336',
      description: 'Percentage of approved trades',
      progress: metrics.approval_rate * 100
    }
  ];

  // Calculate system efficiency score
  const efficiencyScore = Math.min(
    (1 / Math.max(metrics.average_processing_time_ms / 1000, 1)) * 25 + // Processing speed (max 25 points)
    (metrics.approval_rate * 30) + // Approval rate (max 30 points)
    (Math.min(metrics.total_decisions / 100, 1) * 20) + // Experience (max 20 points)
    (Math.max(1 - metrics.pending_decisions / 10, 0) * 25), // Backlog management (max 25 points)
    100
  );

  return (
    <Paper elevation={3} sx={{ height: '100%' }}>
      <Box p={3}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6" component="h2">
            Performance Metrics
          </Typography>
          <Tooltip title="Refresh metrics">
            <IconButton onClick={loadPerformanceMetrics} disabled={loading}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>

        <Grid container spacing={2}>
          {performanceCards.map((card) => (
            <Grid item xs={12} sm={6} key={card.title}>
              <Card variant="outlined" sx={{ height: '100%' }}>
                <CardContent sx={{ p: 2 }}>
                  <Box display="flex" alignItems="center" gap={2} mb={1}>
                    <Box sx={{ color: card.color }}>
                      {card.icon}
                    </Box>
                    <Typography variant="subtitle2" color="text.secondary">
                      {card.title}
                    </Typography>
                  </Box>
                  
                  <Typography variant="h5" component="div" gutterBottom sx={{ color: card.color }}>
                    {card.formatter(card.value)}
                  </Typography>
                  
                  {card.progress !== undefined && (
                    <LinearProgress
                      variant="determinate"
                      value={card.progress}
                      sx={{
                        height: 6,
                        borderRadius: 3,
                        bgcolor: 'grey.200',
                        '& .MuiLinearProgress-bar': {
                          bgcolor: card.color
                        }
                      }}
                    />
                  )}
                  
                  <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                    {card.description}
                  </Typography>
                  
                  {card.target && (
                    <Typography variant="caption" sx={{ mt: 0.5, display: 'block', fontStyle: 'italic' }}>
                      Target: {card.target}
                    </Typography>
                  )}
                  
                  {card.alert && (
                    <Alert severity="warning" sx={{ mt: 1, py: 0 }}>
                      <Typography variant="caption">
                        {card.alert}
                      </Typography>
                    </Alert>
                  )}
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>

        {/* System Efficiency Score */}
        <Box mt={3}>
          <Typography variant="subtitle2" gutterBottom>
            System Efficiency Score
          </Typography>
          <Box display="flex" alignItems="center" gap={2}>
            <LinearProgress
              variant="determinate"
              value={efficiencyScore}
              sx={{
                flex: 1,
                height: 12,
                borderRadius: 6,
                bgcolor: 'grey.200',
                '& .MuiLinearProgress-bar': {
                  bgcolor: efficiencyScore >= 80 ? '#4caf50' : 
                           efficiencyScore >= 60 ? '#ff9800' : '#f44336'
                }
              }}
            />
            <Typography variant="h6" fontWeight="bold" sx={{
              color: efficiencyScore >= 80 ? '#4caf50' : 
                     efficiencyScore >= 60 ? '#ff9800' : '#f44336'
            }}>
              {efficiencyScore.toFixed(0)}%
            </Typography>
          </Box>
          <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
            Based on processing speed, approval rate, experience, and backlog management
          </Typography>
        </Box>

        {/* Performance Indicators */}
        <Box mt={2} p={2} bgcolor="grey.50" borderRadius={1}>
          <Typography variant="subtitle2" gutterBottom>
            Performance Indicators
          </Typography>
          <Box display="flex" gap={1} flexWrap="wrap">
            <Chip
              label={metrics.average_processing_time_ms < 2000 ? 'Fast Response' : 'Slow Response'}
              color={metrics.average_processing_time_ms < 2000 ? 'success' : 'warning'}
              size="small"
            />
            <Chip
              label={metrics.pending_decisions === 0 ? 'No Backlog' : `${metrics.pending_decisions} Pending`}
              color={metrics.pending_decisions === 0 ? 'success' : metrics.pending_decisions > 5 ? 'error' : 'warning'}
              size="small"
            />
            <Chip
              label={metrics.total_decisions > 100 ? 'Experienced' : 'Building Experience'}
              color={metrics.total_decisions > 100 ? 'success' : 'info'}
              size="small"
            />
            <Chip
              label={`${formatPercentage(metrics.approval_rate)} Approval`}
              color={getApprovalRateColor(metrics.approval_rate)}
              size="small"
            />
          </Box>
        </Box>

        {/* Recommendations */}
        {(metrics.pending_decisions > 5 || metrics.average_processing_time_ms > 5000) && (
          <Alert severity="warning" sx={{ mt: 2 }}>
            <Typography variant="body2">
              <strong>Performance Recommendations:</strong>
              {metrics.pending_decisions > 5 && " Reduce decision backlog. "}
              {metrics.average_processing_time_ms > 5000 && " Improve decision response time. "}
              Consider additional operators during peak hours.
            </Typography>
          </Alert>
        )}
      </Box>
    </Paper>
  );
};

export default PerformanceMetricsPanel;