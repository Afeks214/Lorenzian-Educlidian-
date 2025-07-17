import React, { useState, useEffect } from 'react';
import {
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  Box,
  Chip,
  Divider,
  TextField,
  Slider,
  Alert,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  LinearProgress
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  ShowChart,
  Warning,
  CheckCircle,
  TimerOutlined
} from '@mui/icons-material';
import { useParams, useNavigate } from 'react-router-dom';

import { apiService } from '../services/api';

interface TradingStrategy {
  strategy_id: string;
  strategy_type: string;
  entry_price: number;
  position_size: number;
  stop_loss: number;
  take_profit: number;
  time_horizon: number;
  risk_reward_ratio: number;
  confidence_score: number;
  reasoning: string;
  expected_pnl: number;
  max_drawdown: number;
}

interface MarketContext {
  symbol: string;
  price: number;
  volatility: number;
  volume: number;
  trend_strength: number;
  support_level: number;
  resistance_level: number;
  time_of_day: string;
  market_regime: string;
  correlation_shock: boolean;
}

interface DecisionDetails {
  decision_id: string;
  timestamp: string;
  context: MarketContext;
  complexity: string;
  strategies: TradingStrategy[];
  current_position: any;
  expert_deadline: string;
  model_recommendation: string;
  confidence_threshold: number;
}

interface Props {
  onNotification: (message: string, severity?: 'success' | 'error' | 'warning' | 'info') => void;
}

const DecisionEvaluator: React.FC<Props> = ({ onNotification }) => {
  const { decisionId } = useParams<{ decisionId: string }>();
  const navigate = useNavigate();

  const [decision, setDecision] = useState<DecisionDetails | null>(null);
  const [selectedStrategy, setSelectedStrategy] = useState<string>('');
  const [confidence, setConfidence] = useState<number>(70);
  const [reasoning, setReasoning] = useState<string>('');
  const [marketView, setMarketView] = useState<string>('');
  const [riskAssessment, setRiskAssessment] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [timeRemaining, setTimeRemaining] = useState<number>(0);
  const [confirmDialogOpen, setConfirmDialogOpen] = useState(false);

  useEffect(() => {
    if (decisionId) {
      fetchDecisionDetails();
    }
  }, [decisionId]);

  useEffect(() => {
    if (decision) {
      const timer = setInterval(() => {
        const now = new Date().getTime();
        const deadline = new Date(decision.expert_deadline).getTime();
        const remaining = Math.max(0, deadline - now);
        setTimeRemaining(remaining);
        
        if (remaining === 0) {
          onNotification('Decision deadline has expired', 'warning');
        }
      }, 1000);

      return () => clearInterval(timer);
    }
  }, [decision, onNotification]);

  const fetchDecisionDetails = async () => {
    try {
      setLoading(true);
      const response = await apiService.getDecisionDetails(decisionId!);
      setDecision(response.decision);
      
      // Pre-select the model recommendation
      if (response.decision.strategies.length > 0) {
        setSelectedStrategy(response.decision.model_recommendation || response.decision.strategies[0].strategy_id);
      }
    } catch (error: any) {
      onNotification(error.message, 'error');
      navigate('/dashboard');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmitFeedback = async () => {
    if (!selectedStrategy || !reasoning.trim() || !marketView.trim() || !riskAssessment.trim()) {
      onNotification('Please fill in all required fields', 'warning');
      return;
    }

    setSubmitting(true);

    try {
      await apiService.submitFeedback(decisionId!, {
        decision_id: decisionId!,
        chosen_strategy_id: selectedStrategy,
        confidence: confidence / 100,
        reasoning: reasoning.trim(),
        market_view: marketView.trim(),
        risk_assessment: riskAssessment.trim()
      });

      onNotification('Feedback submitted successfully!', 'success');
      navigate('/dashboard');
    } catch (error: any) {
      onNotification(error.message, 'error');
    } finally {
      setSubmitting(false);
      setConfirmDialogOpen(false);
    }
  };

  const formatTimeRemaining = (ms: number) => {
    const minutes = Math.floor(ms / (1000 * 60));
    const seconds = Math.floor((ms % (1000 * 60)) / 1000);
    return `${minutes}m ${seconds}s`;
  };

  const getStrategyTypeColor = (type: string) => {
    const colors: { [key: string]: 'primary' | 'secondary' | 'error' | 'warning' | 'info' | 'success' } = {
      'aggressive': 'error',
      'conservative': 'success',
      'momentum': 'primary',
      'mean_reversion': 'secondary',
      'breakout': 'warning',
      'scalping': 'info'
    };
    return colors[type] || 'default';
  };

  const getComplexityColor = (complexity: string) => {
    switch (complexity.toLowerCase()) {
      case 'critical': return 'error';
      case 'high': return 'warning';
      case 'medium': return 'info';
      default: return 'default';
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (!decision) {
    return (
      <Alert severity="error">
        Decision not found or has expired.
      </Alert>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={6}>
            <Typography variant="h4" gutterBottom>
              Decision Evaluation
            </Typography>
            <Box display="flex" alignItems="center" gap={2}>
              <Typography variant="h6">
                {decision.context.symbol}
              </Typography>
              <Chip
                label={decision.complexity}
                color={getComplexityColor(decision.complexity) as any}
              />
              <Chip
                label={`${decision.strategies.length} strategies`}
                variant="outlined"
              />
            </Box>
          </Grid>
          <Grid item xs={12} md={6}>
            <Box textAlign="right">
              <Typography variant="body2" color="textSecondary">
                Time Remaining
              </Typography>
              <Box display="flex" alignItems="center" justifyContent="flex-end" gap={1}>
                <TimerOutlined color={timeRemaining < 300000 ? 'error' : 'primary'} />
                <Typography
                  variant="h6"
                  color={timeRemaining < 300000 ? 'error' : 'primary'}
                >
                  {formatTimeRemaining(timeRemaining)}
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={Math.max(0, timeRemaining / (15 * 60 * 1000) * 100)}
                color={timeRemaining < 300000 ? 'error' : 'primary'}
                sx={{ mt: 1 }}
              />
            </Box>
          </Grid>
        </Grid>
      </Paper>

      <Grid container spacing={3}>
        {/* Market Context */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Market Context
            </Typography>
            <Divider sx={{ mb: 2 }} />
            
            <TableContainer>
              <Table size="small">
                <TableBody>
                  <TableRow>
                    <TableCell>Price</TableCell>
                    <TableCell align="right">${decision.context.price.toFixed(4)}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Volatility</TableCell>
                    <TableCell align="right">{(decision.context.volatility * 100).toFixed(2)}%</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Volume</TableCell>
                    <TableCell align="right">{decision.context.volume.toFixed(2)}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Trend Strength</TableCell>
                    <TableCell align="right">{(decision.context.trend_strength * 100).toFixed(1)}%</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Support</TableCell>
                    <TableCell align="right">${decision.context.support_level.toFixed(4)}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Resistance</TableCell>
                    <TableCell align="right">${decision.context.resistance_level.toFixed(4)}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Market Regime</TableCell>
                    <TableCell align="right">{decision.context.market_regime}</TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </TableContainer>

            {decision.context.correlation_shock && (
              <Alert severity="warning" sx={{ mt: 2 }}>
                <strong>Correlation Shock Detected</strong>
                <br />
                Exercise heightened caution in decision making.
              </Alert>
            )}
          </Paper>
        </Grid>

        {/* Strategy Selection */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Strategy Options
            </Typography>
            <Divider sx={{ mb: 2 }} />

            <Grid container spacing={2}>
              {decision.strategies.map((strategy) => (
                <Grid item xs={12} key={strategy.strategy_id}>
                  <Card
                    sx={{
                      border: selectedStrategy === strategy.strategy_id ? 2 : 1,
                      borderColor: selectedStrategy === strategy.strategy_id ? 'primary.main' : 'divider',
                      cursor: 'pointer',
                      '&:hover': {
                        borderColor: 'primary.main',
                      }
                    }}
                    onClick={() => setSelectedStrategy(strategy.strategy_id)}
                  >
                    <CardContent>
                      <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
                        <Box>
                          <Typography variant="h6" gutterBottom>
                            {strategy.strategy_type.replace('_', ' ').toUpperCase()}
                          </Typography>
                          <Box display="flex" gap={1} mb={1}>
                            <Chip
                              label={strategy.strategy_type}
                              color={getStrategyTypeColor(strategy.strategy_type)}
                              size="small"
                            />
                            <Chip
                              label={`Confidence: ${(strategy.confidence_score * 100).toFixed(1)}%`}
                              variant="outlined"
                              size="small"
                            />
                            {decision.model_recommendation === strategy.strategy_id && (
                              <Chip
                                label="Model Recommended"
                                color="primary"
                                size="small"
                                icon={<CheckCircle />}
                              />
                            )}
                          </Box>
                        </Box>
                        <Typography variant="h6" color="primary">
                          R/R: {strategy.risk_reward_ratio.toFixed(2)}
                        </Typography>
                      </Box>

                      <Grid container spacing={2}>
                        <Grid item xs={6} md={3}>
                          <Typography variant="body2" color="textSecondary">Entry Price</Typography>
                          <Typography variant="body1">${strategy.entry_price.toFixed(4)}</Typography>
                        </Grid>
                        <Grid item xs={6} md={3}>
                          <Typography variant="body2" color="textSecondary">Position Size</Typography>
                          <Typography variant="body1">{strategy.position_size.toFixed(0)}</Typography>
                        </Grid>
                        <Grid item xs={6} md={3}>
                          <Typography variant="body2" color="textSecondary">Stop Loss</Typography>
                          <Typography variant="body1" color="error">${strategy.stop_loss.toFixed(4)}</Typography>
                        </Grid>
                        <Grid item xs={6} md={3}>
                          <Typography variant="body2" color="textSecondary">Take Profit</Typography>
                          <Typography variant="body1" color="success">${strategy.take_profit.toFixed(4)}</Typography>
                        </Grid>
                        <Grid item xs={6} md={3}>
                          <Typography variant="body2" color="textSecondary">Time Horizon</Typography>
                          <Typography variant="body1">{strategy.time_horizon}min</Typography>
                        </Grid>
                        <Grid item xs={6} md={3}>
                          <Typography variant="body2" color="textSecondary">Expected P&L</Typography>
                          <Typography variant="body1" color={strategy.expected_pnl >= 0 ? 'success' : 'error'}>
                            ${strategy.expected_pnl.toFixed(2)}
                          </Typography>
                        </Grid>
                        <Grid item xs={6} md={3}>
                          <Typography variant="body2" color="textSecondary">Max Drawdown</Typography>
                          <Typography variant="body1" color="error">${strategy.max_drawdown.toFixed(2)}</Typography>
                        </Grid>
                      </Grid>

                      <Box mt={2}>
                        <Typography variant="body2" color="textSecondary" gutterBottom>
                          Reasoning:
                        </Typography>
                        <Typography variant="body2">
                          {strategy.reasoning}
                        </Typography>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Paper>
        </Grid>

        {/* Feedback Form */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Your Expert Feedback
            </Typography>
            <Divider sx={{ mb: 3 }} />

            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography gutterBottom>
                  Confidence Level: {confidence}%
                </Typography>
                <Slider
                  value={confidence}
                  onChange={(_, value) => setConfidence(value as number)}
                  min={1}
                  max={100}
                  marks={[
                    { value: 25, label: '25%' },
                    { value: 50, label: '50%' },
                    { value: 75, label: '75%' },
                    { value: 100, label: '100%' }
                  ]}
                  sx={{ mb: 3 }}
                />
              </Grid>
            </Grid>

            <Grid container spacing={3}>
              <Grid item xs={12} md={4}>
                <TextField
                  label="Detailed Reasoning *"
                  multiline
                  rows={4}
                  fullWidth
                  value={reasoning}
                  onChange={(e) => setReasoning(e.target.value)}
                  placeholder="Explain why you chose this strategy..."
                  required
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <TextField
                  label="Market View *"
                  multiline
                  rows={4}
                  fullWidth
                  value={marketView}
                  onChange={(e) => setMarketView(e.target.value)}
                  placeholder="Your overall view of current market conditions..."
                  required
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <TextField
                  label="Risk Assessment *"
                  multiline
                  rows={4}
                  fullWidth
                  value={riskAssessment}
                  onChange={(e) => setRiskAssessment(e.target.value)}
                  placeholder="Your assessment of the risks involved..."
                  required
                />
              </Grid>
            </Grid>

            <Box display="flex" justifyContent="space-between" mt={4}>
              <Button
                variant="outlined"
                onClick={() => navigate('/dashboard')}
                disabled={submitting}
              >
                Cancel
              </Button>
              <Button
                variant="contained"
                color="primary"
                onClick={() => setConfirmDialogOpen(true)}
                disabled={!selectedStrategy || !reasoning.trim() || !marketView.trim() || !riskAssessment.trim() || submitting || timeRemaining === 0}
                startIcon={submitting ? <CircularProgress size={20} /> : <CheckCircle />}
              >
                {submitting ? 'Submitting...' : 'Submit Feedback'}
              </Button>
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* Confirmation Dialog */}
      <Dialog open={confirmDialogOpen} onClose={() => setConfirmDialogOpen(false)}>
        <DialogTitle>Confirm Your Decision</DialogTitle>
        <DialogContent>
          <Typography gutterBottom>
            You are about to submit feedback for:
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Symbol: <strong>{decision.context.symbol}</strong>
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Selected Strategy: <strong>
              {decision.strategies.find(s => s.strategy_id === selectedStrategy)?.strategy_type.replace('_', ' ').toUpperCase()}
            </strong>
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Confidence: <strong>{confidence}%</strong>
          </Typography>
          <Alert severity="info" sx={{ mt: 2 }}>
            This feedback will be used to improve the AI model's decision-making capabilities.
          </Alert>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleSubmitFeedback} variant="contained" autoFocus>
            Confirm Submission
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default DecisionEvaluator;