import React, { useState, useEffect } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  CardActions,
  Button,
  Chip,
  Box,
  Alert,
  CircularProgress,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Badge,
  Tooltip
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Warning,
  CheckCircle,
  AccessTime,
  ShowChart,
  Assessment,
  Notifications
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

import { apiService } from '../services/api';
import { useWebSocket } from '../contexts/WebSocketContext';
import { useAuth } from '../contexts/AuthContext';

interface PendingDecision {
  decision_id: string;
  timestamp: string;
  complexity: string;
  symbol: string;
  deadline: string;
  strategies_count: number;
}

interface DashboardStats {
  total_decisions: number;
  pending_decisions: number;
  completed_today: number;
  average_confidence: number;
  success_rate: number;
}

interface Props {
  onNotification: (message: string, severity?: 'success' | 'error' | 'warning' | 'info') => void;
}

const ExpertDashboard: React.FC<Props> = ({ onNotification }) => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const { lastMessage } = useWebSocket();
  
  const [pendingDecisions, setPendingDecisions] = useState<PendingDecision[]>([]);
  const [dashboardStats, setDashboardStats] = useState<DashboardStats>({
    total_decisions: 0,
    pending_decisions: 0,
    completed_today: 0,
    average_confidence: 0,
    success_rate: 0
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      
      // Fetch pending decisions
      const pendingResponse = await apiService.getPendingDecisions();
      setPendingDecisions(pendingResponse.decisions);
      
      // Fetch expert analytics if user is available
      if (user) {
        const analyticsResponse = await apiService.getExpertAnalytics(user.expert_id);
        setDashboardStats({
          total_decisions: analyticsResponse.decisions_count,
          pending_decisions: pendingResponse.count,
          completed_today: Math.floor(analyticsResponse.decisions_count * 0.1), // Mock
          average_confidence: analyticsResponse.average_confidence,
          success_rate: analyticsResponse.success_rate
        });
      }
      
      setError(null);
    } catch (err: any) {
      setError(err.message);
      onNotification('Failed to load dashboard data', 'error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
    
    // Refresh every 30 seconds
    const interval = setInterval(fetchDashboardData, 30000);
    
    return () => clearInterval(interval);
  }, [user]);

  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      const data = JSON.parse(lastMessage.data);
      
      if (data.type === 'new_decision') {
        onNotification(`New decision requires your input: ${data.symbol}`, 'info');
        fetchDashboardData(); // Refresh data
      } else if (data.type === 'decision_update') {
        fetchDashboardData(); // Refresh data
      }
    }
  }, [lastMessage, onNotification]);

  const getComplexityColor = (complexity: string) => {
    switch (complexity.toLowerCase()) {
      case 'critical': return 'error';
      case 'high': return 'warning';
      case 'medium': return 'info';
      default: return 'default';
    }
  };

  const getTimeRemaining = (deadline: string) => {
    const now = new Date();
    const deadlineDate = new Date(deadline);
    const diffMs = deadlineDate.getTime() - now.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    
    if (diffMins <= 0) return 'Expired';
    if (diffMins < 60) return `${diffMins}m`;
    return `${Math.floor(diffMins / 60)}h ${diffMins % 60}m`;
  };

  const getUrgencyIcon = (deadline: string) => {
    const timeRemaining = getTimeRemaining(deadline);
    if (timeRemaining === 'Expired') return <Warning color="error" />;
    if (timeRemaining.includes('m') && !timeRemaining.includes('h')) return <Warning color="warning" />;
    return <AccessTime color="info" />;
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
        Expert Trading Dashboard
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Stats Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Pending Decisions
                  </Typography>
                  <Typography variant="h4">
                    {dashboardStats.pending_decisions}
                  </Typography>
                </Box>
                <Badge badgeContent={dashboardStats.pending_decisions} color="error">
                  <Notifications color="action" />
                </Badge>
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
                    Total Decisions
                  </Typography>
                  <Typography variant="h4">
                    {dashboardStats.total_decisions}
                  </Typography>
                </Box>
                <Assessment color="action" />
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
                    {(dashboardStats.average_confidence * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <ShowChart color="action" />
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
                    {(dashboardStats.success_rate * 100).toFixed(1)}%
                  </Typography>
                </Box>
                {dashboardStats.success_rate > 0.7 ? (
                  <TrendingUp color="success" />
                ) : (
                  <TrendingDown color="error" />
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Pending Decisions */}
      <Grid container spacing={3}>
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Pending Decisions Requiring Your Input
            </Typography>
            <Divider sx={{ mb: 2 }} />

            {pendingDecisions.length === 0 ? (
              <Box textAlign="center" py={4}>
                <CheckCircle color="success" sx={{ fontSize: 48, mb: 2 }} />
                <Typography variant="h6" color="textSecondary">
                  No pending decisions
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  All decisions have been processed or are within model confidence thresholds
                </Typography>
              </Box>
            ) : (
              <List>
                {pendingDecisions.map((decision, index) => (
                  <React.Fragment key={decision.decision_id}>
                    {index > 0 && <Divider />}
                    <ListItem
                      sx={{
                        border: '1px solid',
                        borderColor: 'divider',
                        borderRadius: 1,
                        mb: 1,
                        '&:hover': {
                          backgroundColor: 'action.hover',
                        }
                      }}
                    >
                      <ListItemIcon>
                        {getUrgencyIcon(decision.deadline)}
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Box display="flex" alignItems="center" gap={1}>
                            <Typography variant="subtitle1" fontWeight="bold">
                              {decision.symbol}
                            </Typography>
                            <Chip
                              label={decision.complexity}
                              color={getComplexityColor(decision.complexity) as any}
                              size="small"
                            />
                            <Chip
                              label={`${decision.strategies_count} strategies`}
                              variant="outlined"
                              size="small"
                            />
                          </Box>
                        }
                        secondary={
                          <Box>
                            <Typography variant="body2" color="textSecondary">
                              Submitted: {new Date(decision.timestamp).toLocaleString()}
                            </Typography>
                            <Typography variant="body2" color="textSecondary">
                              Time remaining: {getTimeRemaining(decision.deadline)}
                            </Typography>
                          </Box>
                        }
                      />
                      <Box>
                        <Button
                          variant="contained"
                          color="primary"
                          onClick={() => navigate(`/decision/${decision.decision_id}`)}
                          disabled={getTimeRemaining(decision.deadline) === 'Expired'}
                        >
                          Evaluate
                        </Button>
                      </Box>
                    </ListItem>
                  </React.Fragment>
                ))}
              </List>
            )}
          </Paper>
        </Grid>

        <Grid item xs={12} lg={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Quick Actions
            </Typography>
            <Divider sx={{ mb: 2 }} />
            
            <Box display="flex" flexDirection="column" gap={2}>
              <Button
                variant="outlined"
                fullWidth
                onClick={() => navigate('/analytics')}
                startIcon={<Assessment />}
              >
                View Performance Analytics
              </Button>
              
              <Button
                variant="outlined"
                fullWidth
                onClick={fetchDashboardData}
                startIcon={<CheckCircle />}
              >
                Refresh Dashboard
              </Button>
              
              <Tooltip title="Real-time updates via WebSocket">
                <Button
                  variant="outlined"
                  fullWidth
                  disabled
                  startIcon={<Notifications />}
                >
                  Real-time Alerts Active
                </Button>
              </Tooltip>
            </Box>
          </Paper>

          {/* Recent Activity */}
          <Paper sx={{ p: 3, mt: 3 }}>
            <Typography variant="h6" gutterBottom>
              System Status
            </Typography>
            <Divider sx={{ mb: 2 }} />
            
            <List dense>
              <ListItem>
                <ListItemIcon>
                  <CheckCircle color="success" />
                </ListItemIcon>
                <ListItemText primary="API Connection" secondary="Connected" />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <CheckCircle color="success" />
                </ListItemIcon>
                <ListItemText primary="WebSocket" secondary="Active" />
              </ListItem>
              <ListItem>
                <ListItemIcon>
                  <CheckCircle color="success" />
                </ListItemIcon>
                <ListItemText primary="RLHF Training" secondary="Running" />
              </ListItem>
            </List>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ExpertDashboard;