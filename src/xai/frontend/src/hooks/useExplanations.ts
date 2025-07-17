/**
 * React Hooks for XAI Explanations Data Management
 * 
 * Provides hooks for:
 * - Managing explanation data with caching and pagination
 * - Real-time explanation updates via WebSocket
 * - Decision history and filtering
 * - Performance metrics and analytics
 */

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { 
  TradingDecision, 
  ExplanationData, 
  PerformanceMetrics, 
  AnalyticsQuery,
  SearchFilters,
  SortOptions,
  ConversationThread,
  ChatMessage,
  WSMessageType,
  DecisionUpdate,
  TimeFrame
} from '../types';
import { getWebSocketService } from '../services/websocket';
import { apiService } from '../services/api';

// Hook for managing explanations data
export interface UseExplanationsOptions {
  autoRefresh?: boolean;
  refreshInterval?: number;
  cacheSize?: number;
  enableRealtime?: boolean;
}

export interface UseExplanationsResult {
  explanations: ExplanationData[];
  loading: boolean;
  error: string | null;
  totalCount: number;
  hasMore: boolean;
  
  // Actions
  fetchExplanations: (filters?: SearchFilters, sort?: SortOptions) => Promise<void>;
  loadMore: () => Promise<void>;
  refreshExplanations: () => Promise<void>;
  getExplanationById: (id: string) => ExplanationData | null;
  clearCache: () => void;
}

export const useExplanations = (options: UseExplanationsOptions = {}): UseExplanationsResult => {
  const {
    autoRefresh = false,
    refreshInterval = 30000,
    cacheSize = 1000,
    enableRealtime = true
  } = options;

  const [explanations, setExplanations] = useState<ExplanationData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [totalCount, setTotalCount] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const [currentFilters, setCurrentFilters] = useState<SearchFilters>({});
  const [currentSort, setCurrentSort] = useState<SortOptions>({ field: 'timestamp', direction: 'desc' });

  const explanationCache = useRef<Map<string, ExplanationData>>(new Map());
  const refreshTimer = useRef<NodeJS.Timeout | null>(null);
  const wsSubscription = useRef<string | null>(null);

  // Cache management
  const addToCache = useCallback((explanation: ExplanationData) => {
    const cache = explanationCache.current;
    
    // Remove oldest entries if cache is full
    if (cache.size >= cacheSize) {
      const firstKey = cache.keys().next().value;
      cache.delete(firstKey);
    }
    
    cache.set(explanation.id, explanation);
  }, [cacheSize]);

  const getFromCache = useCallback((id: string): ExplanationData | null => {
    return explanationCache.current.get(id) || null;
  }, []);

  const clearCache = useCallback(() => {
    explanationCache.current.clear();
    setExplanations([]);
    setCurrentPage(1);
    setHasMore(true);
  }, []);

  // API calls
  const fetchExplanations = useCallback(async (
    filters: SearchFilters = {},
    sort: SortOptions = { field: 'timestamp', direction: 'desc' },
    page: number = 1,
    append: boolean = false
  ) => {
    setLoading(true);
    setError(null);

    try {
      const response = await apiService.getExplanations({
        ...filters,
        page,
        pageSize: 20,
        sortBy: sort.field,
        sortOrder: sort.direction
      });

      const newExplanations = response.data.explanations;
      
      // Add to cache
      newExplanations.forEach(addToCache);

      if (append) {
        setExplanations(prev => [...prev, ...newExplanations]);
      } else {
        setExplanations(newExplanations);
      }

      setTotalCount(response.data.pagination.totalItems);
      setHasMore(response.data.pagination.hasNext);
      setCurrentPage(page);
      setCurrentFilters(filters);
      setCurrentSort(sort);

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch explanations';
      setError(errorMessage);
      console.error('Failed to fetch explanations:', err);
    } finally {
      setLoading(false);
    }
  }, [addToCache]);

  const loadMore = useCallback(async () => {
    if (!hasMore || loading) return;
    
    await fetchExplanations(currentFilters, currentSort, currentPage + 1, true);
  }, [hasMore, loading, currentFilters, currentSort, currentPage, fetchExplanations]);

  const refreshExplanations = useCallback(async () => {
    await fetchExplanations(currentFilters, currentSort, 1, false);
  }, [currentFilters, currentSort, fetchExplanations]);

  const getExplanationById = useCallback((id: string): ExplanationData | null => {
    return getFromCache(id);
  }, [getFromCache]);

  // Real-time updates
  useEffect(() => {
    if (!enableRealtime) return;

    const wsService = getWebSocketService();
    if (!wsService) return;

    const handleExplanationUpdate = (explanation: ExplanationData) => {
      addToCache(explanation);
      
      // Update the explanations list if the new explanation matches current filters
      setExplanations(prev => {
        const index = prev.findIndex(exp => exp.id === explanation.id);
        if (index >= 0) {
          // Update existing
          const updated = [...prev];
          updated[index] = explanation;
          return updated;
        } else {
          // Add new (if it matches filters)
          return [explanation, ...prev].slice(0, 20); // Keep list manageable
        }
      });
    };

    wsSubscription.current = wsService.subscribe(
      WSMessageType.EXPLANATION_READY,
      handleExplanationUpdate
    );

    return () => {
      if (wsSubscription.current) {
        wsService.unsubscribe(wsSubscription.current);
        wsSubscription.current = null;
      }
    };
  }, [enableRealtime, addToCache]);

  // Auto-refresh
  useEffect(() => {
    if (!autoRefresh) return;

    refreshTimer.current = setInterval(() => {
      refreshExplanations();
    }, refreshInterval);

    return () => {
      if (refreshTimer.current) {
        clearInterval(refreshTimer.current);
        refreshTimer.current = null;
      }
    };
  }, [autoRefresh, refreshInterval, refreshExplanations]);

  // Initial load
  useEffect(() => {
    fetchExplanations();
  }, []);

  return {
    explanations,
    loading,
    error,
    totalCount,
    hasMore,
    fetchExplanations: (filters, sort) => fetchExplanations(filters, sort, 1, false),
    loadMore,
    refreshExplanations,
    getExplanationById,
    clearCache
  };
};

// Hook for managing trading decisions
export interface UseDecisionsResult {
  decisions: TradingDecision[];
  loading: boolean;
  error: string | null;
  totalCount: number;
  hasMore: boolean;
  
  // Actions
  fetchDecisions: (filters?: SearchFilters, sort?: SortOptions) => Promise<void>;
  loadMore: () => Promise<void>;
  refreshDecisions: () => Promise<void>;
  getDecisionById: (id: string) => TradingDecision | null;
  getDecisionWithExplanation: (id: string) => Promise<{ decision: TradingDecision; explanation: ExplanationData } | null>;
}

export const useDecisions = (options: UseExplanationsOptions = {}): UseDecisionsResult => {
  const [decisions, setDecisions] = useState<TradingDecision[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [totalCount, setTotalCount] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const [currentFilters, setCurrentFilters] = useState<SearchFilters>({});
  const [currentSort, setCurrentSort] = useState<SortOptions>({ field: 'timestamp', direction: 'desc' });

  const decisionCache = useRef<Map<string, TradingDecision>>(new Map());
  const wsSubscription = useRef<string | null>(null);

  const fetchDecisions = useCallback(async (
    filters: SearchFilters = {},
    sort: SortOptions = { field: 'timestamp', direction: 'desc' },
    page: number = 1,
    append: boolean = false
  ) => {
    setLoading(true);
    setError(null);

    try {
      const response = await apiService.getDecisions({
        ...filters,
        page,
        pageSize: 20,
        sortBy: sort.field,
        sortOrder: sort.direction
      });

      const newDecisions = response.data.decisions;
      
      // Add to cache
      newDecisions.forEach(decision => {
        decisionCache.current.set(decision.id, decision);
      });

      if (append) {
        setDecisions(prev => [...prev, ...newDecisions]);
      } else {
        setDecisions(newDecisions);
      }

      setTotalCount(response.data.pagination.totalItems);
      setHasMore(response.data.pagination.hasNext);
      setCurrentPage(page);
      setCurrentFilters(filters);
      setCurrentSort(sort);

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch decisions';
      setError(errorMessage);
      console.error('Failed to fetch decisions:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const loadMore = useCallback(async () => {
    if (!hasMore || loading) return;
    await fetchDecisions(currentFilters, currentSort, currentPage + 1, true);
  }, [hasMore, loading, currentFilters, currentSort, currentPage, fetchDecisions]);

  const refreshDecisions = useCallback(async () => {
    await fetchDecisions(currentFilters, currentSort, 1, false);
  }, [currentFilters, currentSort, fetchDecisions]);

  const getDecisionById = useCallback((id: string): TradingDecision | null => {
    return decisionCache.current.get(id) || null;
  }, []);

  const getDecisionWithExplanation = useCallback(async (id: string) => {
    try {
      const response = await apiService.getDecisionDetail(id);
      return {
        decision: response.data.decision,
        explanation: response.data.explanation
      };
    } catch (err) {
      console.error('Failed to fetch decision with explanation:', err);
      return null;
    }
  }, []);

  // Real-time decision updates
  useEffect(() => {
    if (!options.enableRealtime) return;

    const wsService = getWebSocketService();
    if (!wsService) return;

    const handleDecisionUpdate = (update: DecisionUpdate) => {
      const decision = update.decision;
      decisionCache.current.set(decision.id, decision);
      
      setDecisions(prev => {
        const index = prev.findIndex(d => d.id === decision.id);
        if (index >= 0) {
          const updated = [...prev];
          updated[index] = decision;
          return updated;
        } else {
          return [decision, ...prev].slice(0, 20);
        }
      });
    };

    wsSubscription.current = wsService.subscribe(
      WSMessageType.DECISION_UPDATE,
      handleDecisionUpdate
    );

    return () => {
      if (wsSubscription.current) {
        wsService.unsubscribe(wsSubscription.current);
        wsSubscription.current = null;
      }
    };
  }, [options.enableRealtime]);

  // Initial load
  useEffect(() => {
    fetchDecisions();
  }, []);

  return {
    decisions,
    loading,
    error,
    totalCount,
    hasMore,
    fetchDecisions: (filters, sort) => fetchDecisions(filters, sort, 1, false),
    loadMore,
    refreshDecisions,
    getDecisionById,
    getDecisionWithExplanation
  };
};

// Hook for performance metrics
export interface UsePerformanceResult {
  metrics: PerformanceMetrics | null;
  loading: boolean;
  error: string | null;
  refreshMetrics: (timeframe?: TimeFrame) => Promise<void>;
}

export const usePerformance = (timeframe: TimeFrame = TimeFrame.ONE_DAY): UsePerformanceResult => {
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshMetrics = useCallback(async (tf: TimeFrame = timeframe) => {
    setLoading(true);
    setError(null);

    try {
      const response = await apiService.getPerformanceMetrics(tf);
      setMetrics(response.data);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch performance metrics';
      setError(errorMessage);
      console.error('Failed to fetch performance metrics:', err);
    } finally {
      setLoading(false);
    }
  }, [timeframe]);

  // Real-time performance updates
  useEffect(() => {
    const wsService = getWebSocketService();
    if (!wsService) return;

    const subscription = wsService.subscribe(
      WSMessageType.PERFORMANCE_UPDATE,
      (updatedMetrics: PerformanceMetrics) => {
        setMetrics(updatedMetrics);
      }
    );

    return () => {
      wsService.unsubscribe(subscription);
    };
  }, []);

  // Initial load
  useEffect(() => {
    refreshMetrics();
  }, [refreshMetrics]);

  return {
    metrics,
    loading,
    error,
    refreshMetrics
  };
};

// Hook for conversation management
export interface UseConversationsResult {
  conversations: ConversationThread[];
  currentConversation: ConversationThread | null;
  loading: boolean;
  error: string | null;
  
  // Actions
  fetchConversations: () => Promise<void>;
  createConversation: (title?: string) => Promise<string>;
  selectConversation: (id: string) => Promise<void>;
  sendMessage: (message: string, conversationId?: string) => Promise<void>;
  deleteConversation: (id: string) => Promise<void>;
}

export const useConversations = (): UseConversationsResult => {
  const [conversations, setConversations] = useState<ConversationThread[]>([]);
  const [currentConversation, setCurrentConversation] = useState<ConversationThread | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchConversations = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await apiService.getConversations();
      setConversations(response.data.conversations);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch conversations';
      setError(errorMessage);
      console.error('Failed to fetch conversations:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const createConversation = useCallback(async (title?: string): Promise<string> => {
    try {
      const response = await apiService.createConversation({ title });
      const newConversation = response.data.conversation;
      setConversations(prev => [newConversation, ...prev]);
      return newConversation.id;
    } catch (err) {
      console.error('Failed to create conversation:', err);
      throw err;
    }
  }, []);

  const selectConversation = useCallback(async (id: string) => {
    setLoading(true);
    setError(null);

    try {
      const response = await apiService.getConversationDetail(id);
      setCurrentConversation(response.data.conversation);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load conversation';
      setError(errorMessage);
      console.error('Failed to load conversation:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const sendMessage = useCallback(async (message: string, conversationId?: string) => {
    try {
      const response = await apiService.sendChatMessage({
        message,
        conversationId: conversationId || currentConversation?.id
      });

      // Update current conversation with new message
      if (currentConversation) {
        const assistantMessage: ChatMessage = {
          id: response.data.id,
          type: 'assistant',
          content: response.data.response,
          timestamp: new Date().toISOString(),
          metadata: {
            confidence: response.data.confidence,
            processingTime: response.data.processingTime
          },
          attachments: response.data.attachments
        };

        setCurrentConversation(prev => prev ? {
          ...prev,
          messages: [...prev.messages, assistantMessage],
          updatedAt: new Date().toISOString()
        } : null);
      }
    } catch (err) {
      console.error('Failed to send message:', err);
      throw err;
    }
  }, [currentConversation]);

  const deleteConversation = useCallback(async (id: string) => {
    try {
      await apiService.deleteConversation(id);
      setConversations(prev => prev.filter(conv => conv.id !== id));
      
      if (currentConversation?.id === id) {
        setCurrentConversation(null);
      }
    } catch (err) {
      console.error('Failed to delete conversation:', err);
      throw err;
    }
  }, [currentConversation]);

  // Initial load
  useEffect(() => {
    fetchConversations();
  }, []);

  return {
    conversations,
    currentConversation,
    loading,
    error,
    fetchConversations,
    createConversation,
    selectConversation,
    sendMessage,
    deleteConversation
  };
};