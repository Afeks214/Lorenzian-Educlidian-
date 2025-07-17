/**
 * Main XAI Trading Frontend Application
 * 
 * Provides a modern, ChatGPT-like interface for AI trading explanations
 * with comprehensive analytics and real-time capabilities.
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  Box,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
  Avatar,
  Tooltip,
  Menu,
  MenuItem,
  Badge,
  Alert,
  Snackbar,
  Fab,
  SpeedDial,
  SpeedDialAction,
  SpeedDialIcon,
  useMediaQuery,
  Container
} from '@mui/material';
import {
  Menu as MenuIcon,
  Brightness4 as DarkModeIcon,
  Brightness7 as LightModeIcon,
  AccountCircle as AccountIcon,
  Logout as LogoutIcon,
  Settings as SettingsIcon,
  Chat as ChatIcon,
  Assessment as AnalyticsIcon,
  Timeline as TimelineIcon,
  Notifications as NotificationsIcon,
  Help as HelpIcon,
  Fullscreen as FullscreenIcon,
  FullscreenExit as FullscreenExitIcon,
  Close as CloseIcon
} from '@mui/icons-material';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';

import ChatInterface from './components/ChatInterface';
import AnalyticsDashboard from './components/AnalyticsDashboard';
import { createWebSocketService, getWebSocketService } from './services/websocket';
import { apiService } from './services/api';
import { User, TimeFrame, Notification } from './types';

// Application views
enum AppView {
  CHAT = 'chat',
  ANALYTICS = 'analytics',
  TIMELINE = 'timeline'
}

interface AppProps {
  // Props for embedding in larger applications
  embedded?: boolean;
  initialView?: AppView;
  symbols?: string[];
  timeframe?: TimeFrame;
  onClose?: () => void;
}

const XAIApp: React.FC<AppProps> = ({
  embedded = false,
  initialView = AppView.CHAT,
  symbols = [],
  timeframe = TimeFrame.ONE_DAY,
  onClose
}) => {
  // Theme and UI state
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('xai_theme_mode');
    return saved === 'dark';
  });
  const [drawerOpen, setDrawerOpen] = useState(!embedded);
  const [currentView, setCurrentView] = useState<AppView>(initialView);
  const [fullscreen, setFullscreen] = useState(false);
  
  // User and authentication state
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [userMenuAnchor, setUserMenuAnchor] = useState<null | HTMLElement>(null);
  
  // Notifications
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [snackbar, setSnackbar] = useState<{
    open: boolean;
    message: string;
    severity: 'info' | 'warning' | 'error' | 'success';
  }>({
    open: false,
    message: '',
    severity: 'info'
  });

  // WebSocket connection state
  const [wsConnected, setWsConnected] = useState(false);

  // Responsive design
  const isMobile = useMediaQuery('(max-width:768px)');
  const isTablet = useMediaQuery('(max-width:1024px)');

  // Create theme
  const theme = useMemo(() => createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: {
        main: '#1976d2',
        dark: '#115293',
        light: '#42a5f5'
      },
      secondary: {
        main: '#dc004e',
        dark: '#9a0036',
        light: '#e33371'
      },
      background: {
        default: darkMode ? '#0a0e27' : '#f8fafc',
        paper: darkMode ? '#1a1d35' : '#ffffff'
      },
      success: {
        main: '#2e7d32'
      },
      error: {
        main: '#d32f2f'
      },
      warning: {
        main: '#ed6c02'
      }
    },
    typography: {
      fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
      h4: {
        fontWeight: 600
      },
      h5: {
        fontWeight: 600
      },
      h6: {
        fontWeight: 600
      }
    },
    shape: {
      borderRadius: 12
    },
    components: {
      MuiPaper: {
        styleOverrides: {
          root: {
            backgroundImage: 'none'
          }
        }
      },
      MuiAppBar: {
        styleOverrides: {
          root: {
            backgroundColor: darkMode ? '#1a1d35' : '#ffffff',
            color: darkMode ? '#ffffff' : '#000000',
            boxShadow: 'none',
            borderBottom: `1px solid ${darkMode ? '#2a2d45' : '#e0e0e0'}`
          }
        }
      },
      MuiDrawer: {
        styleOverrides: {
          paper: {
            backgroundColor: darkMode ? '#1a1d35' : '#ffffff',
            borderRight: `1px solid ${darkMode ? '#2a2d45' : '#e0e0e0'}`
          }
        }
      }
    }
  }), [darkMode]);

  // Navigation items
  const navigationItems = [
    {
      id: AppView.CHAT,
      label: 'AI Chat',
      icon: <ChatIcon />,
      description: 'Chat with AI about trading decisions'
    },
    {
      id: AppView.ANALYTICS,
      label: 'Analytics',
      icon: <AnalyticsIcon />,
      description: 'Performance and risk analytics'
    },
    {
      id: AppView.TIMELINE,
      label: 'Timeline',
      icon: <TimelineIcon />,
      description: 'Decision history and timeline'
    }
  ];

  // Initialize application
  useEffect(() => {
    const initializeApp = async () => {
      try {
        setLoading(true);

        // Check authentication if not embedded
        if (!embedded && apiService.isAuthenticated()) {
          const userData = apiService.getCurrentUser();
          if (userData) {
            setUser(userData);
          }
        }

        // Initialize WebSocket connection
        if (!embedded) {
          const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8001/ws';
          const wsService = createWebSocketService({
            url: wsUrl,
            autoConnect: true,
            reconnectInterval: 5000,
            maxReconnectAttempts: 5
          });

          wsService.setEventHandlers({
            onConnect: () => {
              setWsConnected(true);
              showNotification('Connected to real-time updates', 'success');
            },
            onDisconnect: () => {
              setWsConnected(false);
              showNotification('Disconnected from real-time updates', 'warning');
            },
            onError: (error) => {
              setWsConnected(false);
              showNotification(`Connection error: ${error.message}`, 'error');
            },
            onReconnecting: (attempt) => {
              showNotification(`Reconnecting... (attempt ${attempt})`, 'info');
            }
          });
        }

      } catch (error) {
        console.error('Failed to initialize app:', error);
        showNotification('Failed to initialize application', 'error');
      } finally {
        setLoading(false);
      }
    };

    initializeApp();

    // Cleanup on unmount
    return () => {
      const wsService = getWebSocketService();
      if (wsService) {
        wsService.destroy();
      }
    };
  }, [embedded]);

  // Handle theme toggle
  const toggleTheme = () => {
    const newMode = !darkMode;
    setDarkMode(newMode);
    localStorage.setItem('xai_theme_mode', newMode ? 'dark' : 'light');
  };

  // Handle fullscreen toggle
  const toggleFullscreen = () => {
    if (!fullscreen) {
      document.documentElement.requestFullscreen?.();
    } else {
      document.exitFullscreen?.();
    }
    setFullscreen(!fullscreen);
  };

  // Show notification
  const showNotification = (message: string, severity: 'info' | 'warning' | 'error' | 'success' = 'info') => {
    setSnackbar({ open: true, message, severity });
  };

  // Handle logout
  const handleLogout = async () => {
    try {
      await apiService.logout();
      setUser(null);
      setUserMenuAnchor(null);
      showNotification('Logged out successfully', 'info');
    } catch (error) {
      showNotification('Logout failed', 'error');
    }
  };

  // Render main content based on current view
  const renderMainContent = () => {
    switch (currentView) {
      case AppView.CHAT:
        return (
          <ChatInterface
            onClose={embedded ? onClose : undefined}
            symbols={symbols}
            timeframe={timeframe}
          />
        );
      
      case AppView.ANALYTICS:
        return (
          <AnalyticsDashboard
            symbols={symbols}
            timeframe={timeframe}
            onClose={embedded ? onClose : undefined}
          />
        );
      
      case AppView.TIMELINE:
        return (
          <Box sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom>
              Decision Timeline
            </Typography>
            <Typography variant="body1" color="textSecondary">
              Timeline view coming soon...
            </Typography>
          </Box>
        );
      
      default:
        return null;
    }
  };

  // Render sidebar
  const renderSidebar = () => (
    <Box sx={{ width: 280, height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Logo/Title */}
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Typography variant="h6" component="div" sx={{ fontWeight: 'bold' }}>
          XAI Trading
        </Typography>
        <Typography variant="caption" color="textSecondary">
          AI-Powered Explanations
        </Typography>
      </Box>

      {/* Navigation */}
      <List sx={{ flex: 1, py: 1 }}>
        {navigationItems.map((item) => (
          <ListItem key={item.id} disablePadding sx={{ px: 1, mb: 0.5 }}>
            <ListItemButton
              onClick={() => setCurrentView(item.id)}
              selected={currentView === item.id}
              sx={{
                borderRadius: 2,
                '&.Mui-selected': {
                  backgroundColor: 'primary.main',
                  color: 'primary.contrastText',
                  '&:hover': {
                    backgroundColor: 'primary.dark'
                  }
                }
              }}
            >
              <ListItemIcon sx={{ color: 'inherit' }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText
                primary={item.label}
                secondary={currentView === item.id ? item.description : null}
                secondaryTypographyProps={{ 
                  color: 'inherit',
                  sx: { opacity: 0.8 }
                }}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>

      {/* Connection Status */}
      {!embedded && (
        <Box sx={{ p: 2, borderTop: 1, borderColor: 'divider' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Box
              sx={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                backgroundColor: wsConnected ? 'success.main' : 'error.main'
              }}
            />
            <Typography variant="caption" color="textSecondary">
              {wsConnected ? 'Connected' : 'Disconnected'}
            </Typography>
          </Box>
        </Box>
      )}
    </Box>
  );

  if (loading) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Box
          sx={{
            height: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          <Typography variant="h6">Loading XAI Trading...</Typography>
        </Box>
      </ThemeProvider>
    );
  }

  return (
    <ThemeProvider theme={theme}>
      <LocalizationProvider dateAdapter={AdapterDateFns}>
        <CssBaseline />
        
        <Box sx={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
          {/* Sidebar */}
          {!embedded && (
            <>
              {isMobile ? (
                <Drawer
                  variant="temporary"
                  open={drawerOpen}
                  onClose={() => setDrawerOpen(false)}
                  ModalProps={{ keepMounted: true }}
                >
                  {renderSidebar()}
                </Drawer>
              ) : (
                <Drawer
                  variant="persistent"
                  open={drawerOpen}
                  sx={{
                    '& .MuiDrawer-paper': {
                      position: 'relative',
                      transition: 'none'
                    }
                  }}
                >
                  {renderSidebar()}
                </Drawer>
              )}
            </>
          )}

          {/* Main Content */}
          <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            {/* App Bar */}
            {!embedded && (
              <AppBar position="static" elevation={0}>
                <Toolbar>
                  <IconButton
                    edge="start"
                    color="inherit"
                    onClick={() => setDrawerOpen(!drawerOpen)}
                    sx={{ mr: 2 }}
                  >
                    <MenuIcon />
                  </IconButton>

                  <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                    {navigationItems.find(item => item.id === currentView)?.label}
                  </Typography>

                  {/* Symbols display */}
                  {symbols.length > 0 && (
                    <Box sx={{ mr: 2 }}>
                      {symbols.map(symbol => (
                        <Typography
                          key={symbol}
                          variant="body2"
                          component="span"
                          sx={{
                            px: 1,
                            py: 0.5,
                            mr: 1,
                            backgroundColor: 'primary.main',
                            color: 'primary.contrastText',
                            borderRadius: 1,
                            fontSize: '0.75rem'
                          }}
                        >
                          {symbol}
                        </Typography>
                      ))}
                    </Box>
                  )}

                  {/* Connection status */}
                  <Tooltip title={wsConnected ? 'Real-time connected' : 'Real-time disconnected'}>
                    <Badge color={wsConnected ? 'success' : 'error'} variant="dot">
                      <NotificationsIcon />
                    </Badge>
                  </Tooltip>

                  {/* Theme toggle */}
                  <Tooltip title="Toggle theme">
                    <IconButton onClick={toggleTheme} color="inherit" sx={{ ml: 1 }}>
                      {darkMode ? <LightModeIcon /> : <DarkModeIcon />}
                    </IconButton>
                  </Tooltip>

                  {/* Fullscreen toggle */}
                  <Tooltip title="Toggle fullscreen">
                    <IconButton onClick={toggleFullscreen} color="inherit">
                      {fullscreen ? <FullscreenExitIcon /> : <FullscreenIcon />}
                    </IconButton>
                  </Tooltip>

                  {/* User menu */}
                  {user && (
                    <>
                      <Tooltip title="User menu">
                        <IconButton
                          onClick={(e) => setUserMenuAnchor(e.currentTarget)}
                          color="inherit"
                          sx={{ ml: 1 }}
                        >
                          <Avatar sx={{ width: 32, height: 32 }}>
                            {user.username.charAt(0).toUpperCase()}
                          </Avatar>
                        </IconButton>
                      </Tooltip>

                      <Menu
                        anchorEl={userMenuAnchor}
                        open={Boolean(userMenuAnchor)}
                        onClose={() => setUserMenuAnchor(null)}
                      >
                        <MenuItem disabled>
                          <Typography variant="subtitle2">{user.username}</Typography>
                        </MenuItem>
                        <Divider />
                        <MenuItem onClick={() => setUserMenuAnchor(null)}>
                          <SettingsIcon sx={{ mr: 1 }} />
                          Settings
                        </MenuItem>
                        <MenuItem onClick={handleLogout}>
                          <LogoutIcon sx={{ mr: 1 }} />
                          Logout
                        </MenuItem>
                      </Menu>
                    </>
                  )}

                  {/* Close button for embedded mode */}
                  {embedded && onClose && (
                    <IconButton onClick={onClose} color="inherit" sx={{ ml: 1 }}>
                      <CloseIcon />
                    </IconButton>
                  )}
                </Toolbar>
              </AppBar>
            )}

            {/* Main Content Area */}
            <Box sx={{ flex: 1, overflow: 'hidden' }}>
              {renderMainContent()}
            </Box>
          </Box>
        </Box>

        {/* Speed Dial for mobile */}
        {embedded && isMobile && (
          <SpeedDial
            ariaLabel="XAI Actions"
            sx={{ position: 'fixed', bottom: 16, right: 16 }}
            icon={<SpeedDialIcon />}
          >
            <SpeedDialAction
              icon={<ChatIcon />}
              tooltipTitle="Chat"
              onClick={() => setCurrentView(AppView.CHAT)}
            />
            <SpeedDialAction
              icon={<AnalyticsIcon />}
              tooltipTitle="Analytics"
              onClick={() => setCurrentView(AppView.ANALYTICS)}
            />
            <SpeedDialAction
              icon={<TimelineIcon />}
              tooltipTitle="Timeline"
              onClick={() => setCurrentView(AppView.TIMELINE)}
            />
          </SpeedDial>
        )}

        {/* Global Snackbar */}
        <Snackbar
          open={snackbar.open}
          autoHideDuration={6000}
          onClose={() => setSnackbar(prev => ({ ...prev, open: false }))}
          anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
        >
          <Alert
            onClose={() => setSnackbar(prev => ({ ...prev, open: false }))}
            severity={snackbar.severity}
            variant="filled"
          >
            {snackbar.message}
          </Alert>
        </Snackbar>
      </LocalizationProvider>
    </ThemeProvider>
  );
};

export default XAIApp;