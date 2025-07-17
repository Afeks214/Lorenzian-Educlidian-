import React, { useState } from 'react';
import {
  Paper,
  TextField,
  Button,
  Typography,
  Box,
  Alert,
  CircularProgress,
  Container,
  Card,
  CardContent,
  Divider
} from '@mui/material';
import { Lock, Person } from '@mui/icons-material';

import { useAuth } from '../contexts/AuthContext';

interface Props {
  onNotification: (message: string, severity?: 'success' | 'error' | 'warning' | 'info') => void;
}

const LoginForm: React.FC<Props> = ({ onNotification }) => {
  const [expertId, setExpertId] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const { login } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!expertId.trim() || !password.trim()) {
      setError('Please fill in all fields');
      return;
    }

    setLoading(true);
    setError('');

    try {
      await login(expertId.trim(), password);
      onNotification('Login successful!', 'success');
    } catch (err: any) {
      setError(err.message);
      onNotification(err.message, 'error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="sm">
      <Box
        display="flex"
        flexDirection="column"
        alignItems="center"
        justifyContent="center"
        minHeight="100vh"
      >
        <Card sx={{ width: '100%', maxWidth: 400 }}>
          <CardContent sx={{ p: 4 }}>
            <Box textAlign="center" mb={3}>
              <Typography variant="h4" gutterBottom>
                🤖 Expert Login
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Trading Feedback System
              </Typography>
            </Box>

            <Divider sx={{ mb: 3 }} />

            {error && (
              <Alert severity="error" sx={{ mb: 3 }}>
                {error}
              </Alert>
            )}

            <form onSubmit={handleSubmit}>
              <TextField
                fullWidth
                label="Expert ID"
                variant="outlined"
                value={expertId}
                onChange={(e) => setExpertId(e.target.value)}
                disabled={loading}
                InputProps={{
                  startAdornment: <Person sx={{ mr: 1, color: 'action.active' }} />
                }}
                sx={{ mb: 3 }}
                autoComplete="username"
              />

              <TextField
                fullWidth
                label="Password"
                type="password"
                variant="outlined"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={loading}
                InputProps={{
                  startAdornment: <Lock sx={{ mr: 1, color: 'action.active' }} />
                }}
                sx={{ mb: 3 }}
                autoComplete="current-password"
              />

              <Button
                type="submit"
                fullWidth
                variant="contained"
                size="large"
                disabled={loading || !expertId.trim() || !password.trim()}
                startIcon={loading ? <CircularProgress size={20} /> : <Lock />}
                sx={{ mb: 2 }}
              >
                {loading ? 'Signing In...' : 'Sign In'}
              </Button>
            </form>

            <Divider sx={{ my: 3 }} />

            <Box>
              <Typography variant="body2" color="textSecondary" textAlign="center" gutterBottom>
                Demo Accounts
              </Typography>
              <Box display="flex" flexDirection="column" gap={1}>
                <Button
                  variant="outlined"
                  size="small"
                  onClick={() => {
                    setExpertId('trader001');
                    setPassword('SecurePass123!');
                  }}
                  disabled={loading}
                >
                  Trader Demo
                </Button>
                <Button
                  variant="outlined"
                  size="small"
                  onClick={() => {
                    setExpertId('senior001');
                    setPassword('SecurePass456!');
                  }}
                  disabled={loading}
                >
                  Senior Trader Demo
                </Button>
                <Button
                  variant="outlined"
                  size="small"
                  onClick={() => {
                    setExpertId('pm001');
                    setPassword('SecurePass789!');
                  }}
                  disabled={loading}
                >
                  Portfolio Manager Demo
                </Button>
              </Box>
            </Box>

            <Box mt={3} textAlign="center">
              <Typography variant="caption" color="textSecondary">
                Secure authentication with JWT tokens
                <br />
                All sessions are monitored and logged
              </Typography>
            </Box>
          </CardContent>
        </Card>
      </Box>
    </Container>
  );
};

export default LoginForm;