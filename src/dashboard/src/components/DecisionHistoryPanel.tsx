/**
 * Decision History Panel
 * Shows audit trail of recent human decisions
 */

import React, { useState, useEffect } from 'react';
import {
  Paper,
  Typography,
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  Tooltip,
  Alert,
  CircularProgress,
  Pagination,
  Select,
  MenuItem,
  FormControl,
  InputLabel
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
  Refresh as RefreshIcon,
  Visibility as VisibilityIcon
} from '@mui/icons-material';
import { DecisionHistoryEntry } from '../types';
import { apiService } from '../services/api';

const DecisionHistoryPanel: React.FC = () => {
  const [decisions, setDecisions] = useState<DecisionHistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [userFilter, setUserFilter] = useState<string>('');

  const loadDecisionHistory = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiService.getDecisionHistory(50); // Load last 50 decisions
      setDecisions(data);
    } catch (error: any) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDecisionHistory();
  }, []);

  const getDecisionIcon = (decision: string) => {
    return decision === 'APPROVE' ? 
      <CheckCircleIcon sx={{ color: '#4caf50' }} /> : 
      <CancelIcon sx={{ color: '#f44336' }} />;
  };

  const getDecisionColor = (decision: string): 'success' | 'error' => {
    return decision === 'APPROVE' ? 'success' : 'error';
  };

  const formatTimestamp = (timestamp: string): string => {
    return new Date(timestamp).toLocaleString();
  };

  const formatProcessingTime = (timeMs: number): string => {
    if (timeMs < 1000) {
      return `${timeMs.toFixed(0)}ms`;
    }
    return `${(timeMs / 1000).toFixed(1)}s`;
  };

  const getUserInitials = (userId: string): string => {
    return userId.split('_').map(part => part[0]).join('').toUpperCase();
  };

  // Filter and paginate decisions
  const filteredDecisions = decisions.filter(decision => 
    userFilter === '' || decision.user_id === userFilter
  );

  const paginatedDecisions = filteredDecisions.slice(
    (page - 1) * pageSize,
    page * pageSize
  );

  const totalPages = Math.ceil(filteredDecisions.length / pageSize);

  // Get unique users for filter
  const uniqueUsers = Array.from(new Set(decisions.map(d => d.user_id)));

  if (loading) {
    return (
      <Paper elevation={3} sx={{ height: '100%', p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Decision History
        </Typography>
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
          <CircularProgress />
        </Box>
      </Paper>
    );
  }

  return (
    <Paper elevation={3} sx={{ height: '100%' }}>
      <Box p={3}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6" component="h2">
            Decision History
          </Typography>
          <Box display="flex" gap={1} alignItems="center">
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel>User Filter</InputLabel>
              <Select
                value={userFilter}
                label="User Filter"
                onChange={(e) => setUserFilter(e.target.value)}
              >
                <MenuItem value="">All Users</MenuItem>
                {uniqueUsers.map(userId => (
                  <MenuItem key={userId} value={userId}>
                    {userId}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <Tooltip title="Refresh history">
              <IconButton onClick={loadDecisionHistory} disabled={loading}>
                <RefreshIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {error ? (
          <Alert severity="error">
            {error}
          </Alert>
        ) : decisions.length === 0 ? (
          <Alert severity="info">
            No decision history available
          </Alert>
        ) : (
          <>
            <TableContainer sx={{ maxHeight: '400px' }}>
              <Table stickyHeader size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Decision</TableCell>
                    <TableCell>Trade ID</TableCell>
                    <TableCell>User</TableCell>
                    <TableCell>Time</TableCell>
                    <TableCell>Processing</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {paginatedDecisions.map((decision) => (
                    <TableRow 
                      key={`${decision.decision_id}-${decision.timestamp}`}
                      hover
                    >
                      <TableCell>
                        <Box display="flex" alignItems="center" gap={1}>
                          {getDecisionIcon(decision.decision)}
                          <Chip
                            label={decision.decision}
                            color={getDecisionColor(decision.decision)}
                            size="small"
                          />
                        </Box>
                      </TableCell>
                      
                      <TableCell>
                        <Typography variant="body2" fontFamily="monospace">
                          {decision.trade_id.substring(0, 8)}...
                        </Typography>
                      </TableCell>
                      
                      <TableCell>
                        <Tooltip title={`${decision.user_id} (${decision.user_role})`}>
                          <Chip
                            label={getUserInitials(decision.user_id)}
                            size="small"
                            variant="outlined"
                          />
                        </Tooltip>
                      </TableCell>
                      
                      <TableCell>
                        <Typography variant="body2">
                          {formatTimestamp(decision.timestamp)}
                        </Typography>
                      </TableCell>
                      
                      <TableCell>
                        <Typography 
                          variant="body2" 
                          color={decision.processing_time_ms > 1000 ? 'warning.main' : 'text.secondary'}
                        >
                          {formatProcessingTime(decision.processing_time_ms)}
                        </Typography>
                      </TableCell>
                      
                      <TableCell>
                        <Chip
                          label={decision.execution_confirmed ? 'Executed' : 'Failed'}
                          color={decision.execution_confirmed ? 'success' : 'error'}
                          size="small"
                          variant="outlined"
                        />
                      </TableCell>
                      
                      <TableCell>
                        <Tooltip title="View reasoning">
                          <IconButton 
                            size="small"
                            onClick={() => {
                              alert(`Reasoning: ${decision.reasoning}`);
                            }}
                          >
                            <VisibilityIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>

            {/* Pagination */}
            <Box display="flex" justifyContent="space-between" alignItems="center" mt={2}>
              <Typography variant="body2" color="text.secondary">
                Showing {Math.min((page - 1) * pageSize + 1, filteredDecisions.length)}-
                {Math.min(page * pageSize, filteredDecisions.length)} of {filteredDecisions.length} decisions
              </Typography>
              
              <Box display="flex" gap={2} alignItems="center">
                <FormControl size="small">
                  <Select
                    value={pageSize}
                    onChange={(e) => {
                      setPageSize(Number(e.target.value));
                      setPage(1);
                    }}
                  >
                    <MenuItem value={5}>5 per page</MenuItem>
                    <MenuItem value={10}>10 per page</MenuItem>
                    <MenuItem value={25}>25 per page</MenuItem>
                  </Select>
                </FormControl>
                
                <Pagination
                  count={totalPages}
                  page={page}
                  onChange={(_, newPage) => setPage(newPage)}
                  size="small"
                />
              </Box>
            </Box>

            {/* Summary Stats */}
            <Box mt={2} p={2} bgcolor="grey.50" borderRadius={1}>
              <Typography variant="subtitle2" gutterBottom>
                Summary (Last 24h)
              </Typography>
              {(() => {
                const last24h = decisions.filter(d => 
                  new Date(d.timestamp).getTime() > Date.now() - 24 * 60 * 60 * 1000
                );
                const approvals = last24h.filter(d => d.decision === 'APPROVE').length;
                const rejections = last24h.filter(d => d.decision === 'REJECT').length;
                const avgProcessingTime = last24h.length > 0 ? 
                  last24h.reduce((sum, d) => sum + d.processing_time_ms, 0) / last24h.length : 0;

                return (
                  <Box display="flex" gap={2} flexWrap="wrap">
                    <Chip label={`${last24h.length} total decisions`} size="small" />
                    <Chip 
                      label={`${approvals} approved`} 
                      color="success" 
                      size="small" 
                      variant="outlined" 
                    />
                    <Chip 
                      label={`${rejections} rejected`} 
                      color="error" 
                      size="small" 
                      variant="outlined" 
                    />
                    <Chip 
                      label={`${formatProcessingTime(avgProcessingTime)} avg time`} 
                      size="small" 
                      variant="outlined" 
                    />
                  </Box>
                );
              })()}
            </Box>
          </>
        )}
      </Box>
    </Paper>
  );
};

export default DecisionHistoryPanel;