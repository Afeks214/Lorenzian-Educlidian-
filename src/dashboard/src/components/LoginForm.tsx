/**
 * Login Form Component
 * Handles user authentication with MFA support
 */

import React, { useState } from 'react';
import {
  Box,
  Paper,
  TextField,
  Button,
  Typography,
  Alert,
  CircularProgress,
  InputAdornment,
  IconButton,
  Chip
} from '@mui/material';
import {
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
  Security as SecurityIcon,
  VpnKey as VpnKeyIcon
} from '@mui/icons-material';
import { User } from '../types';
import { apiService } from '../services/api';

interface LoginFormProps {
  onLogin: (user: User) => void;
  onError: (message: string) => void;
}

const LoginForm: React.FC<LoginFormProps> = ({ onLogin, onError }) => {
  const [formData, setFormData] = useState({
    username: '',
    password: '',
    mfaToken: ''
  });
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [requiresMfa, setRequiresMfa] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleInputChange = (field: string) => (event: React.ChangeEvent<HTMLInputElement>) => {
    setFormData(prev => ({
      ...prev,
      [field]: event.target.value
    }));
    setError(null);
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    
    if (!formData.username || !formData.password) {
      setError('Username and password are required');
      return;
    }

    if (requiresMfa && !formData.mfaToken) {
      setError('MFA token is required');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await apiService.login({
        username: formData.username,
        password: formData.password,
        mfa_token: formData.mfaToken || undefined
      });

      onLogin(response.user_info);
    } catch (error: any) {
      if (error.message.includes('MFA token required')) {
        setRequiresMfa(true);
        setError('Please enter your MFA token');
      } else {
        setError(error.message);
        onError(error.message);
      }
    } finally {
      setLoading(false);
    }
  };

  const demoCredentials = [
    { username: 'admin', password: 'admin123!', role: 'System Admin', mfa: false },
    { username: 'risk_manager', password: 'risk123!', role: 'Risk Manager', mfa: true },
    { username: 'operator', password: 'operator123!', role: 'Risk Operator', mfa: false },
    { username: 'compliance', password: 'compliance123!', role: 'Compliance Officer', mfa: false }
  ];

  const fillDemoCredentials = (username: string, password: string) => {
    setFormData(prev => ({ ...prev, username, password }));
    setRequiresMfa(false);
    setError(null);
  };

  return (
    <Box
      display="flex"
      justifyContent="center"
      alignItems="center"
      minHeight="100vh"
      p={3}
      sx={{
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
      }}
    >
      <Paper elevation={8} sx={{ p: 4, maxWidth: 400, width: '100%' }}>
        <Box display="flex" alignItems="center" justifyContent="center" mb={3}>
          <SecurityIcon sx={{ fontSize: 40, mr: 2, color: 'primary.main' }} />
          <Typography variant="h4" component="h1" fontWeight="bold">
            Risk Dashboard
          </Typography>
        </Box>

        <Typography variant="body2" color="text.secondary" textAlign="center" mb={3}>
          Human-in-the-Loop Validation System
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        <form onSubmit={handleSubmit}>
          <TextField
            fullWidth
            label="Username"
            value={formData.username}
            onChange={handleInputChange('username')}
            margin="normal"
            required
            disabled={loading}
            autoComplete="username"
          />

          <TextField
            fullWidth
            label="Password"
            type={showPassword ? 'text' : 'password'}
            value={formData.password}
            onChange={handleInputChange('password')}
            margin="normal"
            required
            disabled={loading}
            autoComplete="current-password"
            InputProps={{
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton
                    onClick={() => setShowPassword(!showPassword)}
                    edge="end"
                  >
                    {showPassword ? <VisibilityOffIcon /> : <VisibilityIcon />}
                  </IconButton>
                </InputAdornment>
              ),
            }}
          />

          {requiresMfa && (
            <TextField
              fullWidth
              label="MFA Token"
              value={formData.mfaToken}
              onChange={handleInputChange('mfaToken')}
              margin="normal"
              required
              disabled={loading}
              placeholder="123456"
              inputProps={{ maxLength: 6 }}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <VpnKeyIcon />
                  </InputAdornment>
                ),
              }}
              helperText="Enter the 6-digit code from your authenticator app"
            />
          )}

          <Button
            type="submit"
            fullWidth
            variant="contained"
            size="large"
            disabled={loading}
            sx={{ mt: 3, mb: 2, py: 1.5 }}
          >
            {loading ? (
              <CircularProgress size={24} color="inherit" />
            ) : (
              'Sign In'
            )}
          </Button>
        </form>

        {/* Demo Credentials */}
        <Box mt={3}>
          <Typography variant="subtitle2" gutterBottom color="text.secondary">
            Demo Credentials:
          </Typography>
          <Box display="flex" flexDirection="column" gap={1}>
            {demoCredentials.map((cred) => (
              <Box key={cred.username} display="flex" alignItems="center" gap={1}>
                <Button
                  size="small"
                  variant="outlined"
                  onClick={() => fillDemoCredentials(cred.username, cred.password)}
                  disabled={loading}
                  sx={{ minWidth: 100 }}
                >
                  {cred.username}
                </Button>
                <Typography variant="caption" color="text.secondary" flex={1}>
                  {cred.role}
                </Typography>
                {cred.mfa && (
                  <Chip label="MFA" size="small" color="warning" />
                )}
              </Box>
            ))}
          </Box>
          
          {requiresMfa && (
            <Alert severity="info" sx={{ mt: 2 }}>
              <Typography variant="body2">
                For demo MFA accounts, use token: <strong>123456</strong>
              </Typography>
            </Alert>
          )}
        </Box>

        {/* Security Notice */}
        <Alert severity="info" sx={{ mt: 3 }}>
          <Typography variant="caption">
            This is a secure trading system. All actions are logged and audited.
          </Typography>
        </Alert>
      </Paper>
    </Box>
  );
};

export default LoginForm;