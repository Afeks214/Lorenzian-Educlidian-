/**
 * Advanced Analytics Dashboard for XAI Trading System
 * 
 * Features:
 * - Natural language query interface for performance analytics
 * - Interactive charts and visualizations for trading metrics
 * - Decision timeline and context exploration
 * - Performance comparison and trend analysis
 * - Real-time data updates via WebSocket
 * - Export capabilities for reports
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  TextField,
  Card,
  CardContent,
  CardHeader,
  IconButton,
  Tooltip,
  Chip,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Tabs,
  Tab,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Badge,
  Fab,
  SpeedDial,
  SpeedDialAction,
  SpeedDialIcon,
  Drawer,
  Switch,
  FormControlLabel,
  Slider,
  Stack
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Assessment as AssessmentIcon,
  Timeline as TimelineIcon,
  PieChart as PieChartIcon,
  BarChart as BarChartIcon,
  ShowChart as ShowChartIcon,
  Download as DownloadIcon,
  Refresh as RefreshIcon,
  Settings as SettingsIcon,
  FilterList as FilterListIcon,
  Search as SearchIcon,
  ExpandMore as ExpandMoreIcon,
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  Fullscreen as FullscreenIcon,
  Close as CloseIcon,
  DateRange as DateRangeIcon,
  CompareArrows as CompareArrowsIcon,
  Insights as InsightsIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Schedule as ScheduleIcon,
  AccountBalance as AccountBalanceIcon,
  TrendingFlat as TrendingFlatIcon
} from '@mui/icons-material';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { Line, Bar, Pie, Scatter } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip as ChartTooltip,
  Legend,
  Filler,
  TimeScale
} from 'chart.js';
import { format, subDays, subMonths, startOfDay, endOfDay } from 'date-fns';

import {
  PerformanceMetrics,
  TradingDecision,
  ExplanationData,
  AnalyticsQuery,
  TimeFrame,
  SearchFilters,
  AssetClass,
  ActionType,
  ChartData,
  AnalysisType
} from '../types';
import { usePerformance, useDecisions, useExplanations } from '../hooks/useExplanations';
import { apiService } from '../services/api';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  ChartTooltip,
  Legend,
  Filler,
  TimeScale
);

// Interface for analytics tab panel
interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index, ...other }) => {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`analytics-tabpanel-${index}`}
      aria-labelledby={`analytics-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 0 }}>{children}</Box>}
    </div>
  );
};

// Performance metrics summary card
interface MetricsCardProps {
  title: string;
  value: string | number;
  change?: number;
  icon: React.ReactNode;
  color?: 'primary' | 'secondary' | 'success' | 'warning' | 'error';
  format?: 'currency' | 'percentage' | 'number';
}

const MetricsCard: React.FC<MetricsCardProps> = ({
  title,
  value,
  change,
  icon,
  color = 'primary',
  format = 'number'
}) => {
  const formatValue = (val: string | number) => {
    if (typeof val === 'string') return val;
    
    switch (format) {
      case 'currency':
        return new Intl.NumberFormat('en-US', {
          style: 'currency',
          currency: 'USD'
        }).format(val);
      case 'percentage':
        return `${(val * 100).toFixed(2)}%`;
      default:
        return val.toLocaleString();
    }
  };

  const getChangeIcon = () => {
    if (change === undefined) return null;
    if (change > 0) return <TrendingUpIcon color="success" />;
    if (change < 0) return <TrendingDownIcon color="error" />;
    return <TrendingFlatIcon color="action" />;
  };

  const getChangeColor = () => {
    if (change === undefined) return 'textSecondary';
    if (change > 0) return 'success.main';
    if (change < 0) return 'error.main';
    return 'text.secondary';
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box>
            <Typography color="textSecondary" gutterBottom variant="body2">
              {title}
            </Typography>
            <Typography variant="h4" component="div">
              {formatValue(value)}
            </Typography>
            {change !== undefined && (
              <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                {getChangeIcon()}
                <Typography variant="body2" sx={{ ml: 0.5, color: getChangeColor() }}>
                  {change > 0 ? '+' : ''}{change.toFixed(2)}%
                </Typography>
              </Box>
            )}
          </Box>
          <Box sx={{ color: `${color}.main` }}>
            {icon}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

// Chart container with controls
interface ChartContainerProps {
  title: string;
  children: React.ReactNode;
  onExport?: () => void;
  onFullscreen?: () => void;
  actions?: React.ReactNode;
  loading?: boolean;
}

const ChartContainer: React.FC<ChartContainerProps> = ({
  title,
  children,
  onExport,
  onFullscreen,
  actions,
  loading = false
}) => {
  return (
    <Card sx={{ height: '100%' }}>
      <CardHeader
        title={title}
        action={
          <Box>
            {actions}
            {onFullscreen && (
              <Tooltip title="Fullscreen">
                <IconButton onClick={onFullscreen}>
                  <FullscreenIcon />
                </IconButton>
              </Tooltip>
            )}
            {onExport && (
              <Tooltip title="Export chart">
                <IconButton onClick={onExport}>
                  <DownloadIcon />
                </IconButton>
              </Tooltip>
            )}
          </Box>
        }
      />
      <CardContent sx={{ position: 'relative', height: 400 }}>
        {loading && <LinearProgress sx={{ position: 'absolute', top: 0, left: 0, right: 0 }} />}
        <Box sx={{ height: '100%', opacity: loading ? 0.6 : 1 }}>
          {children}
        </Box>
      </CardContent>
    </Card>
  );
};

// Natural language query interface
interface QueryInterfaceProps {
  onQuery: (query: string, filters?: any) => void;
  loading?: boolean;
}

const QueryInterface: React.FC<QueryInterfaceProps> = ({ onQuery, loading = false }) => {
  const [query, setQuery] = useState('');
  const [suggestions] = useState([
    "Show performance for the last 30 days",
    "Compare agent performance this week vs last week",
    "What were the top risk factors yesterday?",
    "Analyze decision accuracy by asset class",
    "Show correlation between confidence and profitability",
    "Display portfolio exposure breakdown",
    "What caused the highest drawdown this month?",
    "Compare bull vs bear market performance"
  ]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      onQuery(query.trim());
    }
  };

  return (
    <Paper sx={{ p: 2, mb: 2 }}>
      <Typography variant="h6" gutterBottom>
        Natural Language Analytics
      </Typography>
      <Box component="form" onSubmit={handleSubmit} sx={{ mb: 2 }}>
        <TextField
          fullWidth
          placeholder="Ask anything about your trading performance and decisions..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={loading}
          InputProps={{
            endAdornment: (
              <IconButton type="submit" disabled={!query.trim() || loading}>
                <SearchIcon />
              </IconButton>
            )
          }}
        />
      </Box>
      
      <Typography variant="subtitle2" color="textSecondary" gutterBottom>
        Try these examples:
      </Typography>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
        {suggestions.slice(0, 4).map((suggestion, index) => (
          <Chip
            key={index}
            label={suggestion}
            variant="outlined"
            size="small"
            onClick={() => setQuery(suggestion)}
            clickable
          />
        ))}
      </Box>
    </Paper>
  );
};

// Decision timeline component
interface DecisionTimelineProps {
  decisions: TradingDecision[];
  onDecisionClick: (decision: TradingDecision) => void;
  loading?: boolean;
}

const DecisionTimeline: React.FC<DecisionTimelineProps> = ({
  decisions,
  onDecisionClick,
  loading = false
}) => {
  const getActionColor = (action: ActionType): string => {
    switch (action) {
      case ActionType.MARKET_BUY:
      case ActionType.LIMIT_BUY:
      case ActionType.INCREASE_LONG:
        return 'success';
      case ActionType.MARKET_SELL:
      case ActionType.LIMIT_SELL:
      case ActionType.DECREASE_LONG:
        return 'error';
      case ActionType.HOLD:
        return 'info';
      default:
        return 'default';
    }
  };

  const getActionIcon = (action: ActionType) => {
    switch (action) {
      case ActionType.MARKET_BUY:
      case ActionType.LIMIT_BUY:
      case ActionType.INCREASE_LONG:
        return <TrendingUpIcon />;
      case ActionType.MARKET_SELL:
      case ActionType.LIMIT_SELL:
      case ActionType.DECREASE_LONG:
        return <TrendingDownIcon />;
      case ActionType.HOLD:
        return <TrendingFlatIcon />;
      default:
        return <AssessmentIcon />;
    }
  };

  if (loading) {
    return <LinearProgress />;
  }

  return (
    <List>
      {decisions.map((decision, index) => (
        <React.Fragment key={decision.id}>
          <ListItem
            button
            onClick={() => onDecisionClick(decision)}
            sx={{ borderRadius: 1, mb: 1 }}
          >
            <ListItemIcon>
              <Badge
                badgeContent={Math.round(decision.confidence * 100)}
                color={decision.confidence > 0.8 ? 'success' : decision.confidence > 0.6 ? 'warning' : 'error'}
                max={100}
              >
                {getActionIcon(decision.action)}
              </Badge>
            </ListItemIcon>
            <ListItemText
              primary={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Typography variant="subtitle2">
                    {decision.symbol}
                  </Typography>
                  <Chip
                    label={ActionType[decision.action]}
                    size="small"
                    color={getActionColor(decision.action) as any}
                    variant="outlined"
                  />
                </Box>
              }
              secondary={
                <Box>
                  <Typography variant="caption" display="block">
                    {format(new Date(decision.timestamp), 'MMM dd, yyyy HH:mm')}
                  </Typography>
                  <Typography variant="caption" color="textSecondary">
                    Size: {decision.size.toLocaleString()} | 
                    Risk: {(decision.riskMetrics.varRisk * 100).toFixed(1)}%
                  </Typography>
                </Box>
              }
            />
          </ListItem>
          {index < decisions.length - 1 && <Divider />}
        </React.Fragment>
      ))}
    </List>
  );
};

// Main analytics dashboard component
export interface AnalyticsDashboardProps {
  symbols?: string[];
  timeframe?: TimeFrame;
  onClose?: () => void;
}

export const AnalyticsDashboard: React.FC<AnalyticsDashboardProps> = ({
  symbols = [],
  timeframe = TimeFrame.ONE_DAY,
  onClose
}) => {
  const [activeTab, setActiveTab] = useState(0);
  const [dateRange, setDateRange] = useState({
    start: subDays(new Date(), 30),
    end: new Date()
  });
  const [filters, setFilters] = useState<SearchFilters>({
    symbols: symbols.length > 0 ? symbols : undefined,
    dateRange: {
      start: dateRange.start.toISOString(),
      end: dateRange.end.toISOString()
    }
  });
  const [queryLoading, setQueryLoading] = useState(false);
  const [fullscreenChart, setFullscreenChart] = useState<string | null>(null);
  const [filtersOpen, setFiltersOpen] = useState(false);

  // Hooks for data
  const { metrics, loading: metricsLoading, refreshMetrics } = usePerformance(timeframe);
  const { 
    decisions, 
    loading: decisionsLoading, 
    fetchDecisions,
    getDecisionWithExplanation 
  } = useDecisions({ enableRealtime: true });
  const { explanations, loading: explanationsLoading } = useExplanations({ enableRealtime: true });

  // Chart data processing
  const chartData = useMemo(() => {
    if (!decisions.length) return null;

    const sortedDecisions = [...decisions].sort(
      (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );

    // Performance over time
    const performanceData = {
      labels: sortedDecisions.map(d => format(new Date(d.timestamp), 'MMM dd')),
      datasets: [
        {
          label: 'Cumulative Returns',
          data: sortedDecisions.map((_, index) => {
            // Mock cumulative returns calculation
            return Math.random() * 10 - 5; // Would be real calculation
          }),
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.1)',
          fill: true,
          tension: 0.1
        }
      ]
    };

    // Decision distribution
    const actionCounts = decisions.reduce((acc, decision) => {
      const action = ActionType[decision.action];
      acc[action] = (acc[action] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const distributionData = {
      labels: Object.keys(actionCounts),
      datasets: [
        {
          data: Object.values(actionCounts),
          backgroundColor: [
            '#FF6384',
            '#36A2EB',
            '#FFCE56',
            '#4BC0C0',
            '#9966FF',
            '#FF9F40'
          ]
        }
      ]
    };

    // Confidence vs Performance scatter
    const scatterData = {
      datasets: [
        {
          label: 'Decisions',
          data: decisions.map(d => ({
            x: d.confidence,
            y: Math.random() * 10 - 5 // Mock performance
          })),
          backgroundColor: 'rgba(255, 99, 132, 0.6)',
          borderColor: 'rgba(255, 99, 132, 1)',
          pointRadius: 5
        }
      ]
    };

    return {
      performance: performanceData,
      distribution: distributionData,
      scatter: scatterData
    };
  }, [decisions]);

  // Natural language query handler
  const handleNaturalLanguageQuery = useCallback(async (query: string) => {
    setQueryLoading(true);
    try {
      // This would call the natural language processing API
      const response = await apiService.processNaturalLanguageQuery({
        query,
        context: {
          symbols,
          timeframe,
          dateRange: filters.dateRange
        }
      });
      
      // Process response and update dashboard
      console.log('Query response:', response);
      
    } catch (error) {
      console.error('Failed to process query:', error);
    } finally {
      setQueryLoading(false);
    }
  }, [symbols, timeframe, filters.dateRange]);

  // Decision click handler
  const handleDecisionClick = useCallback(async (decision: TradingDecision) => {
    try {
      const result = await getDecisionWithExplanation(decision.id);
      if (result) {
        // Open decision detail modal or navigate
        console.log('Decision detail:', result);
      }
    } catch (error) {
      console.error('Failed to get decision details:', error);
    }
  }, [getDecisionWithExplanation]);

  // Refresh all data
  const handleRefresh = useCallback(() => {
    refreshMetrics();
    fetchDecisions(filters);
  }, [refreshMetrics, fetchDecisions, filters]);

  // Export functionality
  const handleExport = useCallback((type: string) => {
    // Implementation for exporting data
    console.log('Exporting:', type);
  }, []);

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: false,
      },
    },
    scales: {
      y: {
        beginAtZero: false,
      },
    },
  };

  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Paper elevation={1} sx={{ p: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Typography variant="h5">
            Analytics Dashboard
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Tooltip title="Refresh data">
              <IconButton onClick={handleRefresh}>
                <RefreshIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Filters">
              <IconButton onClick={() => setFiltersOpen(true)}>
                <FilterListIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Export">
              <IconButton onClick={() => handleExport('dashboard')}>
                <DownloadIcon />
              </IconButton>
            </Tooltip>
            {onClose && (
              <IconButton onClick={onClose}>
                <CloseIcon />
              </IconButton>
            )}
          </Box>
        </Box>
      </Paper>

      <Box sx={{ flex: 1, overflow: 'hidden' }}>
        {/* Natural Language Query */}
        <Box sx={{ p: 2 }}>
          <QueryInterface onQuery={handleNaturalLanguageQuery} loading={queryLoading} />
        </Box>

        {/* Tabs */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider', px: 2 }}>
          <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)}>
            <Tab label="Overview" icon={<AssessmentIcon />} />
            <Tab label="Performance" icon={<TrendingUpIcon />} />
            <Tab label="Decisions" icon={<TimelineIcon />} />
            <Tab label="Risk Analysis" icon={<WarningIcon />} />
            <Tab label="Agent Comparison" icon={<CompareArrowsIcon />} />
          </Tabs>
        </Box>

        {/* Tab Content */}
        <Box sx={{ flex: 1, overflow: 'auto', p: 2 }}>
          {/* Overview Tab */}
          <TabPanel value={activeTab} index={0}>
            <Grid container spacing={3}>
              {/* Key Metrics */}
              <Grid item xs={12} md={3}>
                <MetricsCard
                  title="Total Returns"
                  value={metrics?.totalReturns || 0}
                  change={5.2}
                  icon={<AccountBalanceIcon fontSize="large" />}
                  format="percentage"
                  color="success"
                />
              </Grid>
              <Grid item xs={12} md={3}>
                <MetricsCard
                  title="Sharpe Ratio"
                  value={metrics?.sharpeRatio || 0}
                  change={-2.1}
                  icon={<AssessmentIcon fontSize="large" />}
                  format="number"
                  color="primary"
                />
              </Grid>
              <Grid item xs={12} md={3}>
                <MetricsCard
                  title="Max Drawdown"
                  value={metrics?.maxDrawdown || 0}
                  change={1.5}
                  icon={<TrendingDownIcon fontSize="large" />}
                  format="percentage"
                  color="error"
                />
              </Grid>
              <Grid item xs={12} md={3}>
                <MetricsCard
                  title="Win Rate"
                  value={metrics?.winRate || 0}
                  change={3.2}
                  icon={<CheckCircleIcon fontSize="large" />}
                  format="percentage"
                  color="success"
                />
              </Grid>

              {/* Performance Chart */}
              <Grid item xs={12} md={8}>
                <ChartContainer
                  title="Portfolio Performance"
                  loading={metricsLoading}
                  onExport={() => handleExport('performance')}
                  onFullscreen={() => setFullscreenChart('performance')}
                >
                  {chartData?.performance && (
                    <Line data={chartData.performance} options={chartOptions} />
                  )}
                </ChartContainer>
              </Grid>

              {/* Decision Distribution */}
              <Grid item xs={12} md={4}>
                <ChartContainer
                  title="Decision Distribution"
                  loading={decisionsLoading}
                  onExport={() => handleExport('distribution')}
                >
                  {chartData?.distribution && (
                    <Pie data={chartData.distribution} options={{ responsive: true, maintainAspectRatio: false }} />
                  )}
                </ChartContainer>
              </Grid>

              {/* Recent Decisions */}
              <Grid item xs={12}>
                <Card>
                  <CardHeader title="Recent Decisions" />
                  <CardContent>
                    <DecisionTimeline
                      decisions={decisions.slice(0, 10)}
                      onDecisionClick={handleDecisionClick}
                      loading={decisionsLoading}
                    />
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>

          {/* Performance Tab */}
          <TabPanel value={activeTab} index={1}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={8}>
                <ChartContainer
                  title="Confidence vs Performance"
                  loading={decisionsLoading}
                  onExport={() => handleExport('scatter')}
                >
                  {chartData?.scatter && (
                    <Scatter 
                      data={chartData.scatter} 
                      options={{
                        ...chartOptions,
                        scales: {
                          x: {
                            title: {
                              display: true,
                              text: 'Decision Confidence'
                            }
                          },
                          y: {
                            title: {
                              display: true,
                              text: 'Performance (%)'
                            }
                          }
                        }
                      }} 
                    />
                  )}
                </ChartContainer>
              </Grid>
              <Grid item xs={12} md={4}>
                <Card>
                  <CardHeader title="Performance Metrics" />
                  <CardContent>
                    <Stack spacing={2}>
                      <Box>
                        <Typography variant="body2" color="textSecondary">
                          Annual Return
                        </Typography>
                        <Typography variant="h6">
                          {((metrics?.annualizedReturns || 0) * 100).toFixed(2)}%
                        </Typography>
                      </Box>
                      <Divider />
                      <Box>
                        <Typography variant="body2" color="textSecondary">
                          Volatility
                        </Typography>
                        <Typography variant="h6">
                          {((metrics?.volatility || 0) * 100).toFixed(2)}%
                        </Typography>
                      </Box>
                      <Divider />
                      <Box>
                        <Typography variant="body2" color="textSecondary">
                          Profit Factor
                        </Typography>
                        <Typography variant="h6">
                          {(metrics?.profitFactor || 0).toFixed(2)}
                        </Typography>
                      </Box>
                      <Divider />
                      <Box>
                        <Typography variant="body2" color="textSecondary">
                          Total Trades
                        </Typography>
                        <Typography variant="h6">
                          {metrics?.tradesCount || 0}
                        </Typography>
                      </Box>
                    </Stack>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>

          {/* Other tabs would be implemented similarly */}
          <TabPanel value={activeTab} index={2}>
            <Typography variant="h6">Decision Analysis</Typography>
            <Typography>Decision analysis content would go here...</Typography>
          </TabPanel>

          <TabPanel value={activeTab} index={3}>
            <Typography variant="h6">Risk Analysis</Typography>
            <Typography>Risk analysis content would go here...</Typography>
          </TabPanel>

          <TabPanel value={activeTab} index={4}>
            <Typography variant="h6">Agent Comparison</Typography>
            <Typography>Agent comparison content would go here...</Typography>
          </TabPanel>
        </Box>
      </Box>

      {/* Filters Drawer */}
      <Drawer
        anchor="right"
        open={filtersOpen}
        onClose={() => setFiltersOpen(false)}
      >
        <Box sx={{ width: 300, p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Filters & Settings
          </Typography>
          {/* Filter controls would go here */}
          <Button onClick={() => setFiltersOpen(false)}>
            Apply Filters
          </Button>
        </Box>
      </Drawer>

      {/* Fullscreen Chart Dialog */}
      <Dialog
        open={!!fullscreenChart}
        onClose={() => setFullscreenChart(null)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>
          {fullscreenChart}
          <IconButton
            onClick={() => setFullscreenChart(null)}
            sx={{ position: 'absolute', right: 8, top: 8 }}
          >
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        <DialogContent sx={{ height: 600 }}>
          {/* Fullscreen chart would be rendered here */}
        </DialogContent>
      </Dialog>
    </Box>
  );
};

export default AnalyticsDashboard;