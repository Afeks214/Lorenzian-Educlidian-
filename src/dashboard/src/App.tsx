/**
 * Main App Component for Human-in-the-Loop Risk Dashboard
 */

import React, { useState, useEffect } from 'react';
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  Box,
  AppBar,
  Toolbar,
  Typography,
  Button,
  IconButton,
  Menu,
  MenuItem,
  Avatar,
  Tooltip,
  Chip,
  Alert,
  Snackbar
} from '@mui/material';
import {
  AccountCircle as AccountCircleIcon,
  Logout as LogoutIcon,
  DarkMode as DarkModeIcon,
  LightMode as LightModeIcon,
  Security as SecurityIcon
} from '@mui/icons-material';
import { SnackbarProvider } from 'notistack';

import Dashboard from './components/Dashboard';
import LoginForm from './components/LoginForm';
import { apiService } from './services/api';
import { User, Permission } from './types';

const App: React.FC = () => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [darkMode, setDarkMode] = useState(false);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [notification, setNotification] = useState<{
    open: boolean;
    message: string;
    severity: 'info' | 'warning' | 'error' | 'success';
  }>({
    open: false,
    message: '',
    severity: 'info'
  });

  // Create theme
  const theme = createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: {
        main: '#1976d2',
      },
      secondary: {
        main: '#dc004e',
      },
      background: {
        default: darkMode ? '#121212' : '#f5f5f5',
        paper: darkMode ? '#1e1e1e' : '#ffffff',
      },
    },
    components: {
      MuiPaper: {
        styleOverrides: {
          root: {
            backgroundImage: 'none',
          },
        },
      },
    },
  });

  useEffect(() => {
    // Check if user is already authenticated
    const checkAuth = async () => {
      if (apiService.isAuthenticated()) {
        const currentUser = apiService.getCurrentUser();
        if (currentUser) {
          setUser(currentUser);
        } else {
          // Token exists but user data is missing, clear auth
          apiService.clearToken();
        }
      }
      setLoading(false);
    };

    checkAuth();

    // Load theme preference
    const savedTheme = localStorage.getItem('theme_mode');
    if (savedTheme === 'dark') {
      setDarkMode(true);
    }
  }, []);

  const handleLogin = (userData: User) => {
    setUser(userData);
    showNotification(`Welcome back, ${userData.username}!`, 'success');
  };

  const handleLogout = async () => {
    try {
      await apiService.logout();
      setUser(null);
      setAnchorEl(null);
      showNotification('Logged out successfully', 'info');
    } catch (error) {
      showNotification('Logout failed', 'error');
    }
  };

  const toggleTheme = () => {
    const newMode = !darkMode;
    setDarkMode(newMode);
    localStorage.setItem('theme_mode', newMode ? 'dark' : 'light');
  };

  const showNotification = (message: string, severity: 'info' | 'warning' | 'error' | 'success' = 'info') => {
    setNotification({ open: true, message, severity });
  };

  const handleCloseNotification = () => {
    setNotification(prev => ({ ...prev, open: false }));
  };

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const getRoleColor = (role: string): 'default' | 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning' => {
    switch (role) {
      case 'system_admin':
        return 'error';
      case 'risk_manager':
        return 'primary';
      case 'risk_operator':
        return 'success';
      case 'compliance_officer':
        return 'info';
      default:
        return 'default';
    }
  };

  const getPermissionCount = (permissions: Permission[]): { high: number; total: number } => {
    const highPermissions = [
      Permission.TRADE_APPROVE,
      Permission.HIGH_RISK_APPROVE,
      Permission.SYSTEM_INTEGRATION,
      Permission.USER_MANAGEMENT
    ];
    
    const highCount = permissions.filter(p => highPermissions.includes(p)).length;
    return { high: highCount, total: permissions.length };
  };

  if (loading) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
          <Typography variant="h6">Loading...</Typography>
        </Box>
      </ThemeProvider>
    );
  }

  if (!user) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <SnackbarProvider maxSnack={3}>
          <Box minHeight="100vh" bgcolor="background.default">
            <LoginForm onLogin={handleLogin} onError={(msg) => showNotification(msg, 'error')} />
            
            <Snackbar
              open={notification.open}
              autoHideDuration={6000}
              onClose={handleCloseNotification}
              anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
            >
              <Alert onClose={handleCloseNotification} severity={notification.severity}>
                {notification.message}
              </Alert>
            </Snackbar>
          </Box>
        </SnackbarProvider>
      </ThemeProvider>
    );
  }

  const permissionCounts = getPermissionCount(user.permissions);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <SnackbarProvider maxSnack={3}>
        <Box minHeight="100vh" bgcolor="background.default">
          {/* App Bar */}
          <AppBar position="static" elevation={2}>
            <Toolbar>
              <SecurityIcon sx={{ mr: 2 }} />
              <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                GrandModel Risk Dashboard
              </Typography>

              {/* User Info */}
              <Box display="flex" alignItems="center" gap={2}>
                <Chip
                  label={user.role.replace(/_/g, ' ').toUpperCase()}
                  color={getRoleColor(user.role)}
                  size="small"
                  variant="outlined"
                />
                
                <Tooltip title={`${permissionCounts.high} high-level permissions`}>
                  <Chip
                    label={`${permissionCounts.total} perms`}
                    color={permissionCounts.high > 0 ? 'warning' : 'default'}
                    size="small"
                  />
                </Tooltip>

                <Tooltip title="Toggle theme">
                  <IconButton onClick={toggleTheme} color="inherit">
                    {darkMode ? <LightModeIcon /> : <DarkModeIcon />}
                  </IconButton>
                </Tooltip>

                <Tooltip title="User menu">
                  <IconButton onClick={handleMenuOpen} color="inherit">
                    <Avatar sx={{ width: 32, height: 32, bgcolor: 'secondary.main' }}>
                      {user.username.charAt(0).toUpperCase()}
                    </Avatar>
                  </IconButton>
                </Tooltip>
              </Box>
            </Toolbar>
          </AppBar>

          {/* User Menu */}
          <Menu
            anchorEl={anchorEl}
            open={Boolean(anchorEl)}
            onClose={handleMenuClose}
            anchorOrigin={{
              vertical: 'bottom',
              horizontal: 'right',
            }}
            transformOrigin={{
              vertical: 'top',
              horizontal: 'right',
            }}
          >
            <MenuItem disabled>
              <Box>
                <Typography variant="subtitle2">{user.username}</Typography>
                <Typography variant="caption" color="text.secondary">
                  {user.email}
                </Typography>
              </Box>
            </MenuItem>
            <MenuItem disabled>
              <Typography variant="caption" color="text.secondary">
                Session: {new Date(user.login_time).toLocaleString()}
              </Typography>
            </MenuItem>
            <MenuItem onClick={handleLogout}>
              <LogoutIcon sx={{ mr: 1 }} />
              Logout
            </MenuItem>
          </Menu>

          {/* Main Dashboard */}
          <Dashboard />

          {/* Global Notifications */}
          <Snackbar
            open={notification.open}
            autoHideDuration={6000}
            onClose={handleCloseNotification}
            anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
          >
            <Alert onClose={handleCloseNotification} severity={notification.severity}>
              {notification.message}
            </Alert>
          </Snackbar>
        </Box>
      </SnackbarProvider>
    </ThemeProvider>
  );
};

export default App;