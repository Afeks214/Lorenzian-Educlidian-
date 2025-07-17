/**
 * Risk Overview Panel
 * Displays current portfolio risk metrics with real-time updates
 */

import React from 'react';
import {
  Paper,
  Typography,
  Box,
  Grid,
  Card,
  CardContent,
  LinearProgress,
  Chip,
  Tooltip
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon
} from '@mui/icons-material';
import { RiskMetrics } from '../types';

interface RiskOverviewPanelProps {
  riskMetrics?: RiskMetrics;
}

const RiskOverviewPanel: React.FC<RiskOverviewPanelProps> = ({ riskMetrics }) => {
  if (!riskMetrics) {
    return (
      <Paper elevation={3} sx={{ height: '100%', p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Risk Overview
        </Typography>
        <Typography color="text.secondary">
          No risk data available
        </Typography>
      </Paper>
    );
  }

  const getRiskLevel = (value: number, thresholds: number[]): { level: string; color: string; icon: React.ReactNode } => {
    if (value <= thresholds[0]) {
      return { level: 'Low', color: '#4caf50', icon: <CheckCircleIcon sx={{ color: '#4caf50' }} /> };
    } else if (value <= thresholds[1]) {
      return { level: 'Medium', color: '#ff9800', icon: <WarningIcon sx={{ color: '#ff9800' }} /> };
    } else {
      return { level: 'High', color: '#f44336', icon: <ErrorIcon sx={{ color: '#f44336' }} /> };
    }
  };

  const formatPercentage = (value: number): string => {
    return `${(value * 100).toFixed(2)}%`;
  };

  const formatRatio = (value: number): string => {
    return value.toFixed(2);
  };

  const riskItems = [
    {
      label: 'Portfolio VaR',
      value: riskMetrics.portfolio_var,
      formatter: formatPercentage,
      thresholds: [0.02, 0.05],
      description: 'Value at Risk (95% confidence)',
      progress: Math.min(riskMetrics.portfolio_var / 0.1, 1) * 100
    },
    {
      label: 'Correlation Shock',
      value: riskMetrics.correlation_shock_level,
      formatter: formatPercentage,
      thresholds: [0.3, 0.6],
      description: 'Current correlation shock level',
      progress: riskMetrics.correlation_shock_level * 100
    },
    {
      label: 'Max Drawdown',
      value: riskMetrics.max_drawdown,
      formatter: formatPercentage,
      thresholds: [0.03, 0.08],
      description: 'Maximum portfolio drawdown',
      progress: Math.min(riskMetrics.max_drawdown / 0.15, 1) * 100
    },
    {
      label: 'Portfolio Volatility',
      value: riskMetrics.volatility,
      formatter: formatPercentage,
      thresholds: [0.1, 0.2],
      description: 'Annualized portfolio volatility',
      progress: Math.min(riskMetrics.volatility / 0.3, 1) * 100
    },
    {
      label: 'Current Leverage',
      value: riskMetrics.leverage,
      formatter: formatRatio,
      thresholds: [2.0, 4.0],
      description: 'Current portfolio leverage ratio',
      progress: Math.min(riskMetrics.leverage / 5.0, 1) * 100
    },
    {
      label: 'Liquidity Risk',
      value: riskMetrics.liquidity_risk,
      formatter: formatPercentage,
      thresholds: [0.15, 0.35],
      description: 'Portfolio liquidity risk score',
      progress: riskMetrics.liquidity_risk * 100
    }
  ];

  const performanceItems = [
    {
      label: 'Sharpe Ratio',
      value: riskMetrics.sharpe_ratio,
      formatter: formatRatio,
      thresholds: [1.5, 1.0], // Reverse thresholds for Sharpe ratio (higher is better)
      description: 'Risk-adjusted return measure',
      isReverse: true
    }
  ];

  return (
    <Paper elevation={3} sx={{ height: '100%' }}>
      <Box p={3}>
        <Typography variant="h6" component="h2" gutterBottom>
          Risk Overview
        </Typography>

        <Grid container spacing={2}>
          {riskItems.map((item) => {
            const risk = getRiskLevel(item.value, item.thresholds);
            
            return (
              <Grid item xs={12} sm={6} key={item.label}>
                <Card variant="outlined" sx={{ height: '100%' }}>
                  <CardContent sx={{ p: 2 }}>
                    <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={1}>
                      <Typography variant="body2" color="text.secondary">
                        {item.label}
                      </Typography>
                      <Tooltip title={risk.level + ' Risk'}>
                        {risk.icon}
                      </Tooltip>
                    </Box>
                    
                    <Typography variant="h6" component="div" gutterBottom>
                      {item.formatter(item.value)}
                    </Typography>
                    
                    <LinearProgress
                      variant="determinate"
                      value={item.progress}
                      sx={{
                        height: 6,
                        borderRadius: 3,
                        bgcolor: 'grey.200',
                        '& .MuiLinearProgress-bar': {
                          bgcolor: risk.color
                        }
                      }}
                    />
                    
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                      {item.description}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            );
          })}

          {/* Sharpe Ratio - Special handling for performance metric */}
          {performanceItems.map((item) => {
            const getPerformanceLevel = (value: number) => {
              if (value >= item.thresholds[0]) {
                return { level: 'Excellent', color: '#4caf50', icon: <CheckCircleIcon sx={{ color: '#4caf50' }} /> };
              } else if (value >= item.thresholds[1]) {
                return { level: 'Good', color: '#ff9800', icon: <TrendingUpIcon sx={{ color: '#ff9800' }} /> };
              } else {
                return { level: 'Poor', color: '#f44336', icon: <TrendingDownIcon sx={{ color: '#f44336' }} /> };
              }
            };

            const performance = getPerformanceLevel(item.value);
            const progress = Math.min(Math.max(item.value / 2.0, 0), 1) * 100;

            return (
              <Grid item xs={12} sm={6} key={item.label}>
                <Card variant="outlined" sx={{ height: '100%' }}>
                  <CardContent sx={{ p: 2 }}>
                    <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={1}>
                      <Typography variant="body2" color="text.secondary">
                        {item.label}
                      </Typography>
                      <Tooltip title={performance.level + ' Performance'}>
                        {performance.icon}
                      </Tooltip>
                    </Box>
                    
                    <Typography variant="h6" component="div" gutterBottom>
                      {item.formatter(item.value)}
                    </Typography>
                    
                    <LinearProgress
                      variant="determinate"
                      value={progress}
                      sx={{
                        height: 6,
                        borderRadius: 3,
                        bgcolor: 'grey.200',
                        '& .MuiLinearProgress-bar': {
                          bgcolor: performance.color
                        }
                      }}
                    />
                    
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                      {item.description}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            );
          })}
        </Grid>

        {/* Summary Status */}
        <Box mt={3} p={2} bgcolor="grey.50" borderRadius={1}>
          <Typography variant="subtitle2" gutterBottom>
            Risk Summary
          </Typography>
          <Box display="flex" gap={1} flexWrap="wrap">
            <Chip
              label={`VaR: ${formatPercentage(riskMetrics.portfolio_var)}`}
              color={getRiskLevel(riskMetrics.portfolio_var, [0.02, 0.05]).level === 'Low' ? 'success' : 
                     getRiskLevel(riskMetrics.portfolio_var, [0.02, 0.05]).level === 'Medium' ? 'warning' : 'error'}
              size="small"
            />
            <Chip
              label={`Leverage: ${formatRatio(riskMetrics.leverage)}x`}
              color={getRiskLevel(riskMetrics.leverage, [2.0, 4.0]).level === 'Low' ? 'success' : 
                     getRiskLevel(riskMetrics.leverage, [2.0, 4.0]).level === 'Medium' ? 'warning' : 'error'}
              size="small"
            />
            <Chip
              label={`Sharpe: ${formatRatio(riskMetrics.sharpe_ratio)}`}
              color={riskMetrics.sharpe_ratio >= 1.5 ? 'success' : riskMetrics.sharpe_ratio >= 1.0 ? 'warning' : 'error'}
              size="small"
            />
          </Box>
        </Box>
      </Box>
    </Paper>
  );
};

export default RiskOverviewPanel;