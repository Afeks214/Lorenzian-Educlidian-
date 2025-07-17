import React, { useState, useEffect } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Paper,
  Grid,
  Alert,
  Snackbar,
  ThemeProvider,
  createTheme,
  CssBaseline,
  Box
} from '@mui/material';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';

import LoginForm from './components/LoginForm';
import ExpertDashboard from './components/ExpertDashboard';
import DecisionEvaluator from './components/DecisionEvaluator';
import PerformanceAnalytics from './components/PerformanceAnalytics';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { WebSocketProvider } from './contexts/WebSocketContext';
import { apiService } from './services/api';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00bcd4',
    },
    secondary: {
      main: '#ff9800',
    },
    background: {
      default: '#0a0a0a',
      paper: '#1a1a1a',
    },
  },
  typography: {
    fontFamily: '"Roboto Mono", "Roboto", "Arial", sans-serif',
  },
});

const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isAuthenticated } = useAuth();
  return isAuthenticated ? <>{children}</> : <Navigate to="/login" />;
};

const AppContent: React.FC = () => {
  const { isAuthenticated, user, logout } = useAuth();
  const [notification, setNotification] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error' | 'warning' | 'info';
  }>({
    open: false,
    message: '',
    severity: 'info'
  });

  useEffect(() => {
    // Global error handler
    const handleError = (error: any) => {
      setNotification({
        open: true,
        message: error.message || 'An unexpected error occurred',
        severity: 'error'
      });
    };

    // Setup global error listener
    window.addEventListener('unhandledrejection', handleError);
    
    return () => {
      window.removeEventListener('unhandledrejection', handleError);
    };
  }, []);

  const handleCloseNotification = () => {
    setNotification(prev => ({ ...prev, open: false }));
  };

  const showNotification = (message: string, severity: 'success' | 'error' | 'warning' | 'info' = 'info') => {
    setNotification({
      open: true,
      message,
      severity
    });
  };

  return (
    <Router>
      <Box sx={{ flexGrow: 1, minHeight: '100vh' }}>
        {isAuthenticated && (
          <AppBar position="static" sx={{ mb: 3 }}>
            <Toolbar>
              <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                🤖 Expert Trading Feedback System
              </Typography>
              {user && (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Typography variant="body2">
                    Welcome, {user.expert_id}
                  </Typography>
                  <Typography
                    variant="body2"
                    sx={{ cursor: 'pointer', textDecoration: 'underline' }}
                    onClick={logout}
                  >
                    Logout
                  </Typography>
                </Box>
              )}
            </Toolbar>
          </AppBar>
        )}

        <Container maxWidth="xl">
          <Routes>
            <Route
              path="/login"
              element={
                isAuthenticated ? (
                  <Navigate to="/dashboard" />
                ) : (
                  <LoginForm onNotification={showNotification} />
                )
              }
            />
            <Route
              path="/dashboard"
              element={
                <ProtectedRoute>
                  <ExpertDashboard onNotification={showNotification} />
                </ProtectedRoute>
              }
            />
            <Route
              path="/decision/:decisionId"
              element={
                <ProtectedRoute>
                  <DecisionEvaluator onNotification={showNotification} />
                </ProtectedRoute>
              }
            />
            <Route
              path="/analytics"
              element={
                <ProtectedRoute>
                  <PerformanceAnalytics onNotification={showNotification} />
                </ProtectedRoute>
              }
            />
            <Route
              path="/"
              element={<Navigate to={isAuthenticated ? "/dashboard" : "/login"} />}
            />
          </Routes>
        </Container>

        <Snackbar
          open={notification.open}
          autoHideDuration={6000}
          onClose={handleCloseNotification}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        >
          <Alert
            onClose={handleCloseNotification}
            severity={notification.severity}
            sx={{ width: '100%' }}
          >
            {notification.message}
          </Alert>
        </Snackbar>
      </Box>
    </Router>
  );
};

const App: React.FC = () => {
  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <AuthProvider>
        <WebSocketProvider>
          <AppContent />
        </WebSocketProvider>
      </AuthProvider>
    </ThemeProvider>
  );
};

export default App;