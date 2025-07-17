/**
 * ChatGPT-like Interface for XAI Trading Explanations
 * 
 * Features:
 * - Real-time chat with trading AI explanations
 * - Message history with conversation threads
 * - Rich formatting for trading data and charts
 * - Voice input/output capabilities
 * - File upload for data analysis
 * - Export conversation functionality
 */

import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import {
  Box,
  Paper,
  TextField,
  IconButton,
  Typography,
  Avatar,
  Chip,
  Fab,
  Tooltip,
  Menu,
  MenuItem,
  Divider,
  LinearProgress,
  Alert,
  Card,
  CardContent,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Grid,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  ListItemAvatar,
  Badge,
  Zoom,
  Slide,
  CircularProgress,
  Switch,
  FormControlLabel
} from '@mui/material';
import {
  Send as SendIcon,
  Mic as MicIcon,
  MicOff as MicOffIcon,
  AttachFile as AttachFileIcon,
  MoreVert as MoreVertIcon,
  Download as DownloadIcon,
  Clear as ClearIcon,
  SmartToy as AIIcon,
  Person as PersonIcon,
  TrendingUp as TrendingUpIcon,
  Timeline as TimelineIcon,
  Assessment as AssessmentIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Schedule as ScheduleIcon,
  Refresh as RefreshIcon,
  Settings as SettingsIcon,
  VolumeUp as VolumeUpIcon,
  VolumeOff as VolumeOffIcon
} from '@mui/icons-material';
import { useHotkeys } from 'react-hotkeys-hook';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { format, formatDistanceToNow } from 'date-fns';

import { 
  ChatMessage, 
  ConversationThread, 
  ChatAttachment,
  TradingDecision,
  ExplanationData,
  AnalysisType,
  TimeFrame
} from '../types';
import { useConversations } from '../hooks/useExplanations';
import { apiService } from '../services/api';

// Chat message component with rich formatting
interface ChatMessageProps {
  message: ChatMessage;
  isUser: boolean;
  onRetry?: () => void;
  onExport?: () => void;
}

const ChatMessageComponent: React.FC<ChatMessageProps> = ({ 
  message, 
  isUser, 
  onRetry, 
  onExport 
}) => {
  const [expanded, setExpanded] = useState(false);
  const [speaking, setSpeaking] = useState(false);

  const handleSpeak = useCallback(() => {
    if ('speechSynthesis' in window) {
      if (speaking) {
        window.speechSynthesis.cancel();
        setSpeaking(false);
      } else {
        const utterance = new SpeechSynthesisUtterance(message.content);
        utterance.onend = () => setSpeaking(false);
        window.speechSynthesis.speak(utterance);
        setSpeaking(true);
      }
    }
  }, [message.content, speaking]);

  const renderAttachment = (attachment: ChatAttachment) => {
    switch (attachment.type) {
      case 'chart':
        return (
          <Card sx={{ mt: 1, maxWidth: 600 }}>
            <CardContent>
              <Typography variant="subtitle2" gutterBottom>
                {attachment.title}
              </Typography>
              {attachment.data && (
                <Box sx={{ height: 300, width: '100%' }}>
                  {/* Chart component would go here */}
                  <Box 
                    sx={{ 
                      height: '100%', 
                      display: 'flex', 
                      alignItems: 'center', 
                      justifyContent: 'center',
                      backgroundColor: 'grey.100',
                      borderRadius: 1
                    }}
                  >
                    <Typography color="textSecondary">
                      Chart: {attachment.title}
                    </Typography>
                  </Box>
                </Box>
              )}
            </CardContent>
          </Card>
        );
      
      case 'table':
        return (
          <Card sx={{ mt: 1, maxWidth: 600 }}>
            <CardContent>
              <Typography variant="subtitle2" gutterBottom>
                {attachment.title}
              </Typography>
              {/* Table component would go here */}
              <Typography variant="body2" color="textSecondary">
                Data table: {attachment.description}
              </Typography>
            </CardContent>
          </Card>
        );
      
      default:
        return (
          <Chip
            label={attachment.title}
            size="small"
            sx={{ mt: 1, mr: 1 }}
            onClick={() => attachment.url && window.open(attachment.url, '_blank')}
          />
        );
    }
  };

  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: isUser ? 'flex-end' : 'flex-start',
        mb: 2,
        animation: 'fadeIn 0.3s ease-in'
      }}
    >
      <Box sx={{ display: 'flex', maxWidth: '80%', alignItems: 'flex-start' }}>
        {!isUser && (
          <Avatar 
            sx={{ 
              bgcolor: 'primary.main', 
              mr: 1,
              width: 32,
              height: 32
            }}
          >
            <AIIcon fontSize="small" />
          </Avatar>
        )}
        
        <Paper
          elevation={1}
          sx={{
            p: 2,
            bgcolor: isUser ? 'primary.main' : 'background.paper',
            color: isUser ? 'primary.contrastText' : 'text.primary',
            borderRadius: 2,
            position: 'relative',
            flex: 1
          }}
        >
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <Box sx={{ flex: 1 }}>
              <ReactMarkdown
                components={{
                  code({ node, inline, className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || '');
                    return !inline && match ? (
                      <SyntaxHighlighter
                        style={oneDark}
                        language={match[1]}
                        PreTag="div"
                        {...props}
                      >
                        {String(children).replace(/\n$/, '')}
                      </SyntaxHighlighter>
                    ) : (
                      <code className={className} {...props}>
                        {children}
                      </code>
                    );
                  }
                }}
              >
                {message.content}
              </ReactMarkdown>
              
              {/* Render attachments */}
              {message.attachments?.map((attachment, index) => (
                <Box key={index}>
                  {renderAttachment(attachment)}
                </Box>
              ))}
              
              {/* Message metadata */}
              {message.metadata && (
                <Box sx={{ mt: 1, pt: 1, borderTop: '1px solid', borderColor: 'divider' }}>
                  <Grid container spacing={1}>
                    {message.metadata.confidence && (
                      <Grid item>
                        <Chip
                          label={`Confidence: ${(message.metadata.confidence * 100).toFixed(0)}%`}
                          size="small"
                          color={message.metadata.confidence > 0.8 ? 'success' : 'default'}
                        />
                      </Grid>
                    )}
                    {message.metadata.processingTime && (
                      <Grid item>
                        <Chip
                          label={`${message.metadata.processingTime}ms`}
                          size="small"
                          icon={<ScheduleIcon />}
                        />
                      </Grid>
                    )}
                    {message.metadata.symbols && (
                      <Grid item>
                        <Chip
                          label={message.metadata.symbols.join(', ')}
                          size="small"
                          icon={<TrendingUpIcon />}
                        />
                      </Grid>
                    )}
                  </Grid>
                </Box>
              )}
            </Box>
            
            {/* Message actions */}
            <Box sx={{ ml: 1, display: 'flex', flexDirection: 'column' }}>
              <Tooltip title={speaking ? "Stop speaking" : "Read aloud"}>
                <IconButton size="small" onClick={handleSpeak}>
                  {speaking ? <VolumeOffIcon fontSize="small" /> : <VolumeUpIcon fontSize="small" />}
                </IconButton>
              </Tooltip>
              
              {!isUser && onRetry && (
                <Tooltip title="Regenerate response">
                  <IconButton size="small" onClick={onRetry}>
                    <RefreshIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              )}
              
              {onExport && (
                <Tooltip title="Export message">
                  <IconButton size="small" onClick={onExport}>
                    <DownloadIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              )}
            </Box>
          </Box>
          
          <Typography 
            variant="caption" 
            sx={{ 
              display: 'block', 
              mt: 1, 
              opacity: 0.7,
              textAlign: 'right'
            }}
          >
            {formatDistanceToNow(new Date(message.timestamp), { addSuffix: true })}
          </Typography>
        </Paper>
        
        {isUser && (
          <Avatar 
            sx={{ 
              bgcolor: 'secondary.main', 
              ml: 1,
              width: 32,
              height: 32
            }}
          >
            <PersonIcon fontSize="small" />
          </Avatar>
        )}
      </Box>
    </Box>
  );
};

// Conversation sidebar component
interface ConversationSidebarProps {
  conversations: ConversationThread[];
  currentConversation: ConversationThread | null;
  onSelectConversation: (id: string) => void;
  onNewConversation: () => void;
  onDeleteConversation: (id: string) => void;
  loading: boolean;
}

const ConversationSidebar: React.FC<ConversationSidebarProps> = ({
  conversations,
  currentConversation,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
  loading
}) => {
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [conversationToDelete, setConversationToDelete] = useState<string | null>(null);

  const handleDeleteClick = (id: string, event: React.MouseEvent) => {
    event.stopPropagation();
    setConversationToDelete(id);
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirm = () => {
    if (conversationToDelete) {
      onDeleteConversation(conversationToDelete);
      setConversationToDelete(null);
    }
    setDeleteDialogOpen(false);
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ p: 2, borderBottom: '1px solid', borderColor: 'divider' }}>
        <Button
          variant="contained"
          fullWidth
          onClick={onNewConversation}
          startIcon={<AIIcon />}
          disabled={loading}
        >
          New Conversation
        </Button>
      </Box>
      
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        <List>
          {conversations.map((conversation) => (
            <ListItem key={conversation.id} disablePadding>
              <ListItemButton
                selected={currentConversation?.id === conversation.id}
                onClick={() => onSelectConversation(conversation.id)}
                sx={{ pr: 1 }}
              >
                <ListItemAvatar>
                  <Avatar sx={{ bgcolor: 'primary.light' }}>
                    <TimelineIcon />
                  </Avatar>
                </ListItemAvatar>
                <ListItemText
                  primary={conversation.title || 'Untitled Conversation'}
                  secondary={
                    <Box>
                      <Typography variant="caption" display="block">
                        {conversation.messages.length} messages
                      </Typography>
                      <Typography variant="caption" color="textSecondary">
                        {formatDistanceToNow(new Date(conversation.updatedAt), { addSuffix: true })}
                      </Typography>
                    </Box>
                  }
                  primaryTypographyProps={{
                    noWrap: true,
                    variant: 'body2'
                  }}
                />
                <IconButton
                  size="small"
                  onClick={(e) => handleDeleteClick(conversation.id, e)}
                  sx={{ ml: 1 }}
                >
                  <ClearIcon fontSize="small" />
                </IconButton>
              </ListItemButton>
            </ListItem>
          ))}
        </List>
      </Box>
      
      {/* Delete confirmation dialog */}
      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle>Delete Conversation</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete this conversation? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleDeleteConfirm} color="error" variant="contained">
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

// Main chat interface component
export interface ChatInterfaceProps {
  onClose?: () => void;
  initialQuery?: string;
  symbols?: string[];
  timeframe?: TimeFrame;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  onClose,
  initialQuery,
  symbols,
  timeframe
}) => {
  const {
    conversations,
    currentConversation,
    loading,
    error,
    createConversation,
    selectConversation,
    sendMessage,
    deleteConversation
  } = useConversations();

  const [inputMessage, setInputMessage] = useState(initialQuery || '');
  const [sending, setSending] = useState(false);
  const [listening, setListening] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const recognition = useRef<any>(null);

  // Scroll to bottom when new messages arrive
  const scrollToBottom = useCallback(() => {
    if (autoScroll && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [autoScroll]);

  useEffect(() => {
    scrollToBottom();
  }, [currentConversation?.messages, scrollToBottom]);

  // Speech recognition setup
  useEffect(() => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition;
      recognition.current = new SpeechRecognition();
      recognition.current.continuous = false;
      recognition.current.interimResults = false;
      recognition.current.lang = 'en-US';

      recognition.current.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        setInputMessage(transcript);
        setListening(false);
      };

      recognition.current.onerror = () => {
        setListening(false);
      };

      recognition.current.onend = () => {
        setListening(false);
      };
    }
  }, []);

  // Keyboard shortcuts
  useHotkeys('cmd+enter', () => handleSendMessage(), { enableOnFormTags: true });
  useHotkeys('cmd+k', () => handleNewConversation(), { preventDefault: true });
  useHotkeys('escape', () => onClose?.(), { preventDefault: true });

  const handleSendMessage = useCallback(async () => {
    if (!inputMessage.trim() || sending) return;

    setSending(true);
    const messageText = inputMessage.trim();
    setInputMessage('');

    try {
      // Create conversation if none exists
      let conversationId = currentConversation?.id;
      if (!conversationId) {
        conversationId = await createConversation();
      }

      // Add user message to current conversation immediately
      const userMessage: ChatMessage = {
        id: `temp_${Date.now()}`,
        type: 'user',
        content: messageText,
        timestamp: new Date().toISOString(),
        metadata: {
          symbols,
          timeframe
        }
      };

      // Send message and get response
      await sendMessage(messageText, conversationId);

    } catch (error) {
      console.error('Failed to send message:', error);
      // Could show error notification here
    } finally {
      setSending(false);
      inputRef.current?.focus();
    }
  }, [inputMessage, sending, currentConversation, createConversation, sendMessage, symbols, timeframe]);

  const handleNewConversation = useCallback(async () => {
    try {
      await createConversation();
    } catch (error) {
      console.error('Failed to create conversation:', error);
    }
  }, [createConversation]);

  const handleVoiceInput = useCallback(() => {
    if (recognition.current) {
      if (listening) {
        recognition.current.stop();
        setListening(false);
      } else {
        recognition.current.start();
        setListening(true);
      }
    }
  }, [listening]);

  const handleRetryMessage = useCallback(async (messageId: string) => {
    // Implementation for retrying a message
    console.log('Retry message:', messageId);
  }, []);

  const handleExportConversation = useCallback(() => {
    if (currentConversation) {
      const dataStr = JSON.stringify(currentConversation, null, 2);
      const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
      
      const exportFileDefaultName = `conversation_${currentConversation.id}.json`;
      
      const linkElement = document.createElement('a');
      linkElement.setAttribute('href', dataUri);
      linkElement.setAttribute('download', exportFileDefaultName);
      linkElement.click();
    }
  }, [currentConversation]);

  // Suggested questions based on context
  const suggestedQuestions = useMemo(() => {
    const base = [
      "Explain the latest trading decision",
      "What's the current portfolio performance?",
      "Show me risk metrics for today",
      "Why did the AI choose this action?"
    ];

    if (symbols?.length) {
      base.push(`Analyze ${symbols.join(', ')} performance`);
      base.push(`What factors drove ${symbols[0]} decision?`);
    }

    return base;
  }, [symbols]);

  return (
    <Box sx={{ height: '100vh', display: 'flex', bgcolor: 'background.default' }}>
      {/* Conversation Sidebar */}
      <Slide direction="right" in={sidebarOpen} mountOnEnter unmountOnExit>
        <Paper
          elevation={2}
          sx={{
            width: 320,
            height: '100%',
            borderRadius: 0,
            borderRight: '1px solid',
            borderColor: 'divider'
          }}
        >
          <ConversationSidebar
            conversations={conversations}
            currentConversation={currentConversation}
            onSelectConversation={selectConversation}
            onNewConversation={handleNewConversation}
            onDeleteConversation={deleteConversation}
            loading={loading}
          />
        </Paper>
      </Slide>

      {/* Main Chat Area */}
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <Paper elevation={1} sx={{ p: 2, borderRadius: 0 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <IconButton onClick={() => setSidebarOpen(!sidebarOpen)}>
                <TimelineIcon />
              </IconButton>
              <Typography variant="h6" sx={{ ml: 1 }}>
                {currentConversation?.title || 'XAI Trading Assistant'}
              </Typography>
              {symbols && (
                <Box sx={{ ml: 2 }}>
                  {symbols.map(symbol => (
                    <Chip
                      key={symbol}
                      label={symbol}
                      size="small"
                      sx={{ mr: 1 }}
                      color="primary"
                    />
                  ))}
                </Box>
              )}
            </Box>
            
            <Box>
              <IconButton onClick={() => setSettingsOpen(true)}>
                <SettingsIcon />
              </IconButton>
              <IconButton onClick={handleExportConversation}>
                <DownloadIcon />
              </IconButton>
              {onClose && (
                <IconButton onClick={onClose}>
                  <ClearIcon />
                </IconButton>
              )}
            </Box>
          </Box>
          
          {sending && <LinearProgress sx={{ mt: 1 }} />}
        </Paper>

        {/* Messages Area */}
        <Box
          sx={{
            flex: 1,
            overflow: 'auto',
            p: 2,
            display: 'flex',
            flexDirection: 'column'
          }}
        >
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {currentConversation?.messages.length === 0 && !loading && (
            <Box
              sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                flex: 1,
                textAlign: 'center'
              }}
            >
              <AIIcon sx={{ fontSize: 64, color: 'primary.main', mb: 2 }} />
              <Typography variant="h5" gutterBottom>
                XAI Trading Assistant
              </Typography>
              <Typography variant="body1" color="textSecondary" sx={{ mb: 4 }}>
                Ask me anything about your trading decisions, performance, or market analysis.
              </Typography>
              
              <Grid container spacing={1} sx={{ maxWidth: 600 }}>
                {suggestedQuestions.map((question, index) => (
                  <Grid item xs={12} sm={6} key={index}>
                    <Button
                      variant="outlined"
                      fullWidth
                      onClick={() => setInputMessage(question)}
                      sx={{ textAlign: 'left', justifyContent: 'flex-start' }}
                    >
                      {question}
                    </Button>
                  </Grid>
                ))}
              </Grid>
            </Box>
          )}

          {currentConversation?.messages.map((message) => (
            <ChatMessageComponent
              key={message.id}
              message={message}
              isUser={message.type === 'user'}
              onRetry={() => handleRetryMessage(message.id)}
              onExport={() => handleExportConversation()}
            />
          ))}

          {loading && (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
              <CircularProgress size={24} />
              <Typography variant="body2" sx={{ ml: 2 }}>
                Thinking...
              </Typography>
            </Box>
          )}

          <div ref={messagesEndRef} />
        </Box>

        {/* Input Area */}
        <Paper elevation={3} sx={{ p: 2, borderRadius: 0 }}>
          <Box sx={{ display: 'flex', alignItems: 'flex-end', gap: 1 }}>
            <TextField
              ref={inputRef}
              fullWidth
              multiline
              maxRows={4}
              placeholder="Ask about trading decisions, performance, or market analysis..."
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSendMessage();
                }
              }}
              disabled={sending}
              variant="outlined"
              size="small"
            />
            
            <Tooltip title="Voice input">
              <IconButton
                onClick={handleVoiceInput}
                color={listening ? 'secondary' : 'default'}
                disabled={!recognition.current}
              >
                {listening ? <MicOffIcon /> : <MicIcon />}
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Send message (Cmd+Enter)">
              <IconButton
                onClick={handleSendMessage}
                disabled={!inputMessage.trim() || sending}
                color="primary"
              >
                {sending ? <CircularProgress size={24} /> : <SendIcon />}
              </IconButton>
            </Tooltip>
          </Box>
        </Paper>
      </Box>

      {/* Settings Dialog */}
      <Dialog open={settingsOpen} onClose={() => setSettingsOpen(false)}>
        <DialogTitle>Chat Settings</DialogTitle>
        <DialogContent>
          <Box sx={{ py: 2 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={voiceEnabled}
                  onChange={(e) => setVoiceEnabled(e.target.checked)}
                />
              }
              label="Enable voice responses"
            />
          </Box>
          <Box sx={{ py: 2 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={autoScroll}
                  onChange={(e) => setAutoScroll(e.target.checked)}
                />
              }
              label="Auto-scroll to new messages"
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSettingsOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ChatInterface;