/**
 * Pending Decisions Panel
 * Shows trades flagged for human review with approve/reject actions
 */

import React, { useState } from 'react';
import {
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  CardActions,
  Button,
  Chip,
  Grid,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Alert,
  Tooltip,
  IconButton,
  Collapse
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Warning as WarningIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Schedule as ScheduleIcon,
  Assessment as AssessmentIcon
} from '@mui/icons-material';
import { FlaggedTrade, Permission } from '../types';
import { apiService } from '../services/api';

interface PendingDecisionsPanelProps {
  flaggedTrades: FlaggedTrade[];
  onDecision: (tradeId: string, decision: 'APPROVE' | 'REJECT', reasoning: string) => Promise<void>;
}

interface DecisionDialogState {
  open: boolean;
  trade: FlaggedTrade | null;
  decision: 'APPROVE' | 'REJECT' | null;
  reasoning: string;
  submitting: boolean;
}

const PendingDecisionsPanel: React.FC<PendingDecisionsPanelProps> = ({
  flaggedTrades,
  onDecision
}) => {
  const [dialogState, setDialogState] = useState<DecisionDialogState>({
    open: false,
    trade: null,
    decision: null,
    reasoning: '',
    submitting: false
  });

  const [expandedTrades, setExpandedTrades] = useState<Set<string>>(new Set());

  const user = apiService.getCurrentUser();
  const canApprove = apiService.hasPermission(Permission.TRADE_APPROVE);
  const canReject = apiService.hasPermission(Permission.TRADE_REJECT);
  const canHighRiskApprove = apiService.hasPermission(Permission.HIGH_RISK_APPROVE);

  const handleDecisionClick = (trade: FlaggedTrade, decision: 'APPROVE' | 'REJECT') => {
    // Check permissions for high-risk trades
    if (decision === 'APPROVE' && trade.risk_score > 0.7 && !canHighRiskApprove) {
      alert('High-risk approval requires additional permissions');
      return;
    }

    setDialogState({
      open: true,
      trade,
      decision,
      reasoning: '',
      submitting: false
    });
  };

  const handleDialogClose = () => {
    setDialogState({
      open: false,
      trade: null,
      decision: null,
      reasoning: '',
      submitting: false
    });
  };

  const handleSubmitDecision = async () => {
    if (!dialogState.trade || !dialogState.decision || !dialogState.reasoning.trim()) {
      return;
    }

    if (dialogState.reasoning.trim().length < 10) {
      alert('Please provide at least 10 characters of reasoning');
      return;
    }

    setDialogState(prev => ({ ...prev, submitting: true }));

    try {
      await onDecision(dialogState.trade.trade_id, dialogState.decision, dialogState.reasoning.trim());
      handleDialogClose();
    } catch (error) {
      setDialogState(prev => ({ ...prev, submitting: false }));
    }
  };

  const toggleExpanded = (tradeId: string) => {
    const newExpanded = new Set(expandedTrades);
    if (newExpanded.has(tradeId)) {
      newExpanded.delete(tradeId);
    } else {
      newExpanded.add(tradeId);
    }
    setExpandedTrades(newExpanded);
  };

  const getTimeRemaining = (expiresAt: string): string => {
    const now = new Date();
    const expiry = new Date(expiresAt);
    const diff = expiry.getTime() - now.getTime();
    
    if (diff <= 0) return 'EXPIRED';
    
    const minutes = Math.floor(diff / 60000);
    const seconds = Math.floor((diff % 60000) / 1000);
    
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const getRiskColor = (riskScore: number): 'success' | 'warning' | 'error' => {
    if (riskScore < 0.3) return 'success';
    if (riskScore < 0.7) return 'warning';
    return 'error';
  };

  const getPriorityColor = (riskScore: number): 'default' | 'primary' | 'secondary' | 'error' => {
    if (riskScore < 0.3) return 'default';
    if (riskScore < 0.5) return 'primary';
    if (riskScore < 0.7) return 'secondary';
    return 'error';
  };

  return (
    <Paper elevation={3} sx={{ height: '100%' }}>
      <Box p={3}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6" component="h2">
            Pending Decisions
          </Typography>
          <Chip 
            label={flaggedTrades.length}
            color={flaggedTrades.length > 0 ? 'warning' : 'default'}
            variant="outlined"
          />
        </Box>

        {flaggedTrades.length === 0 ? (
          <Alert severity="info">
            No trades currently flagged for human review
          </Alert>
        ) : (
          <Box sx={{ maxHeight: '70vh', overflowY: 'auto' }}>
            {flaggedTrades.map((trade) => {
              const isExpanded = expandedTrades.has(trade.trade_id);
              const timeRemaining = getTimeRemaining(trade.expires_at);
              const isExpired = timeRemaining === 'EXPIRED';

              return (
                <Card 
                  key={trade.trade_id} 
                  sx={{ 
                    mb: 2, 
                    border: isExpired ? '2px solid #f44336' : '1px solid #e0e0e0',
                    opacity: isExpired ? 0.7 : 1
                  }}
                >
                  <CardContent>
                    <Grid container spacing={2} alignItems="center">
                      <Grid item xs={12} md={8}>
                        <Box display="flex" alignItems="center" gap={1} mb={1}>
                          <Typography variant="h6">
                            {trade.symbol}
                          </Typography>
                          <Chip
                            icon={trade.direction === 'LONG' ? <TrendingUpIcon /> : <TrendingDownIcon />}
                            label={trade.direction}
                            color={trade.direction === 'LONG' ? 'success' : 'error'}
                            size="small"
                          />
                          <Chip
                            label={`Risk: ${(trade.risk_score * 100).toFixed(1)}%`}
                            color={getRiskColor(trade.risk_score)}
                            size="small"
                          />
                          <Chip
                            icon={<ScheduleIcon />}
                            label={timeRemaining}
                            color={isExpired ? 'error' : 'default'}
                            size="small"
                          />
                        </Box>

                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          Quantity: {trade.quantity.toLocaleString()} @ ${trade.entry_price.toFixed(4)}
                        </Typography>

                        <Typography variant="body2" color="text.secondary">
                          Flagged: {trade.flagged_reason}
                        </Typography>

                        {/* Risk Metrics */}
                        <Box mt={1}>
                          <Typography variant="caption" color="text.secondary">
                            Failure Probability: {(trade.failure_probability * 100).toFixed(1)}%
                          </Typography>
                          <LinearProgress
                            variant="determinate"
                            value={trade.failure_probability * 100}
                            color={getRiskColor(trade.failure_probability)}
                            sx={{ mt: 0.5, height: 6, borderRadius: 3 }}
                          />
                        </Box>
                      </Grid>

                      <Grid item xs={12} md={4}>
                        <Box display="flex" flexDirection="column" gap={1}>
                          <Button
                            variant="contained"
                            color="success"
                            size="small"
                            disabled={!canApprove || isExpired || (trade.risk_score > 0.7 && !canHighRiskApprove)}
                            onClick={() => handleDecisionClick(trade, 'APPROVE')}
                            fullWidth
                          >
                            APPROVE
                          </Button>
                          <Button
                            variant="contained"
                            color="error"
                            size="small"
                            disabled={!canReject || isExpired}
                            onClick={() => handleDecisionClick(trade, 'REJECT')}
                            fullWidth
                          >
                            REJECT
                          </Button>
                        </Box>
                      </Grid>
                    </Grid>

                    {/* Expandable Details */}
                    <Box display="flex" justifyContent="space-between" alignItems="center" mt={1}>
                      <Typography variant="caption" color="text.secondary">
                        ID: {trade.trade_id}
                      </Typography>
                      <Tooltip title="Show details">
                        <IconButton
                          onClick={() => toggleExpanded(trade.trade_id)}
                          size="small"
                        >
                          {isExpanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                        </IconButton>
                      </Tooltip>
                    </Box>

                    <Collapse in={isExpanded}>
                      <Box mt={2} p={2} bgcolor="grey.50" borderRadius={1}>
                        <Typography variant="subtitle2" gutterBottom>
                          Agent Recommendations:
                        </Typography>
                        {trade.agent_recommendations.map((rec, index) => (
                          <Box key={index} mb={1}>
                            <Typography variant="body2">
                              <strong>{rec.agent_name}:</strong> {rec.recommendation} 
                              (confidence: {(rec.confidence * 100).toFixed(1)}%)
                            </Typography>
                            {rec.reasoning && Object.keys(rec.reasoning).length > 0 && (
                              <Typography variant="caption" color="text.secondary">
                                Reasoning: {JSON.stringify(rec.reasoning)}
                              </Typography>
                            )}
                          </Box>
                        ))}
                      </Box>
                    </Collapse>
                  </CardContent>
                </Card>
              );
            })}
          </Box>
        )}
      </Box>

      {/* Decision Dialog */}
      <Dialog
        open={dialogState.open}
        onClose={handleDialogClose}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          {dialogState.decision} Trade: {dialogState.trade?.symbol} {dialogState.trade?.direction}
        </DialogTitle>
        
        <DialogContent>
          {dialogState.trade && (
            <>
              <Alert severity={dialogState.decision === 'APPROVE' ? 'success' : 'warning'} sx={{ mb: 2 }}>
                You are about to {dialogState.decision?.toLowerCase()} this trade.
                {dialogState.decision === 'APPROVE' && dialogState.trade.risk_score > 0.7 && (
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    <WarningIcon sx={{ fontSize: 16, mr: 0.5 }} />
                    This is a high-risk trade requiring additional authorization.
                  </Typography>
                )}
              </Alert>

              <Box mb={2}>
                <Typography variant="body2" color="text.secondary">
                  Trade Details:
                </Typography>
                <Typography variant="body2">
                  {dialogState.trade.quantity.toLocaleString()} shares of {dialogState.trade.symbol}
                </Typography>
                <Typography variant="body2">
                  Entry Price: ${dialogState.trade.entry_price.toFixed(4)}
                </Typography>
                <Typography variant="body2">
                  Risk Score: {(dialogState.trade.risk_score * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2">
                  Failure Probability: {(dialogState.trade.failure_probability * 100).toFixed(1)}%
                </Typography>
              </Box>

              <TextField
                autoFocus
                label="Decision Reasoning"
                placeholder="Please provide reasoning for your decision (minimum 10 characters)..."
                multiline
                rows={4}
                value={dialogState.reasoning}
                onChange={(e) => setDialogState(prev => ({ ...prev, reasoning: e.target.value }))}
                fullWidth
                required
                error={dialogState.reasoning.length > 0 && dialogState.reasoning.length < 10}
                helperText={
                  dialogState.reasoning.length > 0 && dialogState.reasoning.length < 10
                    ? 'Reasoning must be at least 10 characters'
                    : `${dialogState.reasoning.length} characters`
                }
              />
            </>
          )}
        </DialogContent>

        <DialogActions>
          <Button onClick={handleDialogClose} disabled={dialogState.submitting}>
            Cancel
          </Button>
          <Button
            onClick={handleSubmitDecision}
            variant="contained"
            color={dialogState.decision === 'APPROVE' ? 'success' : 'error'}
            disabled={dialogState.submitting || dialogState.reasoning.trim().length < 10}
          >
            {dialogState.submitting ? 'Processing...' : `Confirm ${dialogState.decision}`}
          </Button>
        </DialogActions>
      </Dialog>
    </Paper>
  );
};

export default PendingDecisionsPanel;