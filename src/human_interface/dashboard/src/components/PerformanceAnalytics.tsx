import React, { useState, useEffect } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  Box,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Alert,
  CircularProgress,
  Divider
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { TrendingUp, TrendingDown, Assessment, CheckCircle } from '@mui/icons-material';

import { apiService } from '../services/api';
import { useAuth } from '../contexts/AuthContext';

interface AnalyticsData {
  decisions_count: number;
  average_confidence: number;
  success_rate: number;
  recent_decisions: Array<{
    decision_id: string;
    timestamp: string;
    confidence: number;
    strategy: string;
  }>;
}

interface Props {
  onNotification: (message: string, severity?: 'success' | 'error' | 'warning' | 'info') => void;
}

const PerformanceAnalytics: React.FC<Props> = ({ onNotification }) => {
  const { user } = useAuth();
  const [analyticsData, setAnalyticsData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedPeriod, setSelectedPeriod] = useState('30');

  // Mock data for charts
  const confidenceTrendData = [
    { date: '2024-01-01', confidence: 0.65, decisions: 5 },
    { date: '2024-01-02', confidence: 0.72, decisions: 8 },
    { date: '2024-01-03', confidence: 0.68, decisions: 6 },
    { date: '2024-01-04', confidence: 0.75, decisions: 12 },
    { date: '2024-01-05', confidence: 0.80, decisions: 9 },
    { date: '2024-01-06', confidence: 0.78, decisions: 7 },
    { date: '2024-01-07', confidence: 0.82, decisions: 11 },
  ];

  const strategyDistributionData = [
    { name: 'Momentum', value: 35, color: '#8884d8' },
    { name: 'Mean Reversion', value: 25, color: '#82ca9d' },
    { name: 'Conservative', value: 20, color: '#ffc658' },
    { name: 'Aggressive', value: 15, color: '#ff7300' },
    { name: 'Breakout', value: 5, color: '#00ff00' },
  ];

  const performanceMetricsData = [
    { metric: 'Accuracy', value: 78, benchmark: 70 },
    { metric: 'Speed', value: 85, benchmark: 80 },
    { metric: 'Consistency', value: 72, benchmark: 75 },
    { metric: 'Risk Awareness', value: 88, benchmark: 80 },
  ];

  useEffect(() => {
    fetchAnalyticsData();
  }, [selectedPeriod, user]);

  const fetchAnalyticsData = async () => {
    if (!user) return;

    try {
      setLoading(true);
      const data = await apiService.getExpertAnalytics(user.expert_id);
      setAnalyticsData(data);
    } catch (error: any) {
      onNotification(error.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  const formatStrategyType = (strategy: string) => {
    return strategy.replace('_', ' ').toUpperCase();
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'error';
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (!analyticsData) {
    return (
      <Alert severity="error">
        Failed to load analytics data.
      </Alert>
    );
  }

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" gutterBottom>
          Performance Analytics
        </Typography>
        <FormControl size="small" sx={{ minWidth: 120 }}>
          <InputLabel>Period</InputLabel>
          <Select
            value={selectedPeriod}
            label="Period"
            onChange={(e) => setSelectedPeriod(e.target.value)}
          >
            <MenuItem value="7">Last 7 days</MenuItem>
            <MenuItem value="30">Last 30 days</MenuItem>
            <MenuItem value="90">Last 90 days</MenuItem>
          </Select>
        </FormControl>
      </Box>

      {/* Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Total Decisions
                  </Typography>
                  <Typography variant="h4">
                    {analyticsData.decisions_count}
                  </Typography>
                </Box>
                <Assessment color="primary" />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Avg. Confidence
                  </Typography>
                  <Typography variant="h4">
                    {(analyticsData.average_confidence * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <CheckCircle color={getConfidenceColor(analyticsData.average_confidence) as any} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Success Rate
                  </Typography>
                  <Typography variant="h4">
                    {(analyticsData.success_rate * 100).toFixed(1)}%
                  </Typography>
                </Box>
                {analyticsData.success_rate > 0.7 ? (
                  <TrendingUp color="success" />
                ) : (
                  <TrendingDown color="error" />
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Model Alignment
                  </Typography>
                  <Typography variant="h4">
                    85%
                  </Typography>
                </Box>
                <TrendingUp color="success" />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* Confidence Trend Chart */}
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Confidence Trend Over Time
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={confidenceTrendData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis domain={[0, 1]} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                <Tooltip formatter={(value: any) => [`${(value * 100).toFixed(1)}%`, 'Confidence']} />
                <Line 
                  type="monotone" 
                  dataKey="confidence" 
                  stroke="#8884d8" 
                  strokeWidth={2}
                  dot={{ fill: '#8884d8' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Strategy Distribution */}
        <Grid item xs={12} lg={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Strategy Preferences
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={strategyDistributionData}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {strategyDistributionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Performance Metrics */}
        <Grid item xs={12} lg={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Performance vs Benchmark
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={performanceMetricsData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="metric" />
                <YAxis domain={[0, 100]} />
                <Tooltip />
                <Bar dataKey="value" fill="#8884d8" name="Your Score" />
                <Bar dataKey="benchmark" fill="#82ca9d" name="Benchmark" />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Recent Decisions */}
        <Grid item xs={12} lg={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Recent Decisions
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Date</TableCell>
                    <TableCell>Strategy</TableCell>
                    <TableCell align="right">Confidence</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {analyticsData.recent_decisions.map((decision) => (
                    <TableRow key={decision.decision_id}>
                      <TableCell>
                        {new Date(decision.timestamp).toLocaleDateString()}
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={formatStrategyType(decision.strategy)}
                          size="small"
                          variant="outlined"
                        />
                      </TableCell>
                      <TableCell align="right">
                        <Chip
                          label={`${(decision.confidence * 100).toFixed(1)}%`}
                          size="small"
                          color={getConfidenceColor(decision.confidence) as any}
                        />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>

        {/* RLHF Impact */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              RLHF Training Impact
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <Grid container spacing={3}>
              <Grid item xs={12} md={4}>
                <Box textAlign="center">
                  <Typography variant="h3" color="primary">92%</Typography>
                  <Typography variant="body2" color="textSecondary">
                    Model Accuracy Improvement
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={12} md={4}>
                <Box textAlign="center">
                  <Typography variant="h3" color="success.main">156</Typography>
                  <Typography variant="body2" color="textSecondary">
                    Feedback Points Contributed
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={12} md={4}>
                <Box textAlign="center">
                  <Typography variant="h3" color="warning.main">78%</Typography>
                  <Typography variant="body2" color="textSecondary">
                    Agreement with Final Model
                  </Typography>
                </Box>
              </Grid>
            </Grid>
            <Alert severity="info" sx={{ mt: 2 }}>
              Your expert feedback is helping improve the AI model's decision-making capabilities. 
              The more feedback you provide, the better the model becomes at understanding expert preferences.
            </Alert>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default PerformanceAnalytics;