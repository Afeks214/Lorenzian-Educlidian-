/**
 * API-specific types for XAI Trading Frontend
 */

import { TradingDecision, ExplanationData, PerformanceMetrics, AnalyticsQuery, TimeFrame, AssetClass } from './index';

// Base API types
export interface ApiConfig {
  baseURL: string;
  timeout: number;
  retryAttempts: number;
  retryDelay: number;
}

export interface ApiError {
  code: string;
  message: string;
  details?: any;
  timestamp: string;
}

// Authentication API types
export interface LoginRequest {
  username: string;
  password: string;
}

export interface LoginResponse {
  token: string;
  user: {
    id: string;
    username: string;
    email: string;
    role: string;
    permissions: string[];
  };
  expiresAt: string;
}

export interface TokenRefreshResponse {
  token: string;
  expiresAt: string;
}

// Decisions API types
export interface DecisionsListRequest {
  page?: number;
  pageSize?: number;
  symbols?: string[];
  startDate?: string;
  endDate?: string;
  assetClasses?: AssetClass[];
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

export interface DecisionsListResponse {
  decisions: TradingDecision[];
  pagination: {
    page: number;
    pageSize: number;
    totalItems: number;
    totalPages: number;
  };
}

export interface DecisionDetailResponse {
  decision: TradingDecision;
  explanation: ExplanationData;
  relatedDecisions: TradingDecision[];
}

// Explanations API types
export interface ExplanationRequest {
  decisionId: string;
  audience?: string;
  includeAlternatives?: boolean;
  includeFeatureDetails?: boolean;
}

export interface ExplanationResponse {
  explanation: ExplanationData;
  processingTime: number;
}

export interface BulkExplanationRequest {
  decisionIds: string[];
  audience?: string;
  format?: 'individual' | 'summary' | 'comparison';
}

export interface BulkExplanationResponse {
  explanations: ExplanationData[];
  summary?: {
    commonFactors: string[];
    averageConfidence: number;
    consensusLevel: number;
  };
}

// Analytics API types
export interface AnalyticsRequest extends AnalyticsQuery {
  includeCharts?: boolean;
  includeMetrics?: boolean;
  customMetrics?: string[];
}

export interface AnalyticsResponse {
  metrics: PerformanceMetrics;
  charts?: {
    performanceChart: ChartDataResponse;
    drawdownChart: ChartDataResponse;
    returnsDistribution: ChartDataResponse;
    factorContribution: ChartDataResponse;
  };
  insights: AnalyticsInsight[];
}

export interface ChartDataResponse {
  type: 'line' | 'bar' | 'scatter' | 'pie' | 'candlestick';
  data: {
    labels: string[];
    datasets: Array<{
      label: string;
      data: number[];
      metadata?: any;
    }>;
  };
  options: any;
}

export interface AnalyticsInsight {
  type: 'performance' | 'risk' | 'behavior' | 'opportunity';
  title: string;
  description: string;
  severity: 'info' | 'warning' | 'critical';
  actionable: boolean;
  recommendations?: string[];
}

// Chat API types
export interface ChatRequest {
  message: string;
  conversationId?: string;
  context?: {
    symbols?: string[];
    timeframe?: TimeFrame;
    decisionIds?: string[];
  };
  responseFormat?: 'text' | 'structured' | 'chart';
}

export interface ChatResponse {
  id: string;
  response: string;
  conversationId: string;
  attachments?: ChatAttachmentResponse[];
  suggestedQuestions?: string[];
  processingTime: number;
  confidence: number;
}

export interface ChatAttachmentResponse {
  type: 'chart' | 'table' | 'analysis' | 'recommendation';
  title: string;
  data: any;
  format: string;
}

export interface ConversationListResponse {
  conversations: Array<{
    id: string;
    title: string;
    lastMessage: string;
    timestamp: string;
    messageCount: number;
  }>;
}

export interface ConversationDetailResponse {
  conversation: {
    id: string;
    title: string;
    createdAt: string;
    messages: Array<{
      id: string;
      type: 'user' | 'assistant';
      content: string;
      timestamp: string;
      attachments?: ChatAttachmentResponse[];
    }>;
  };
}

// Real-time data API types
export interface LiveDecisionStream {
  symbol: string;
  subscribe: boolean;
}

export interface PerformanceMetricsStream {
  timeframe: TimeFrame;
  metrics: string[];
  updateInterval: number;
}

export interface RiskAlertsStream {
  thresholds: {
    varLimit: number;
    drawdownLimit: number;
    volatilityLimit: number;
  };
}

// System status API types
export interface SystemStatusResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  uptime: number;
  services: ServiceStatus[];
  performance: {
    avgResponseTime: number;
    throughput: number;
    errorRate: number;
  };
  lastUpdate: string;
}

export interface ServiceStatus {
  name: string;
  status: 'running' | 'stopped' | 'error';
  health: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  lastCheck: string;
  metrics?: {
    cpu: number;
    memory: number;
    latency: number;
  };
}

// Configuration API types
export interface UserPreferencesRequest {
  defaultTimeframe?: TimeFrame;
  explanationAudience?: string;
  chartSettings?: any;
  notificationSettings?: any;
}

export interface UserPreferencesResponse {
  preferences: {
    defaultTimeframe: TimeFrame;
    explanationAudience: string;
    chartSettings: any;
    notificationSettings: any;
  };
}

// Export reports API types
export interface ExportRequest {
  type: 'decisions' | 'explanations' | 'analytics' | 'performance';
  format: 'csv' | 'xlsx' | 'pdf' | 'json';
  dateRange: {
    start: string;
    end: string;
  };
  filters?: any;
  includeCharts?: boolean;
}

export interface ExportResponse {
  exportId: string;
  status: 'processing' | 'completed' | 'failed';
  downloadUrl?: string;
  expiresAt?: string;
  estimatedCompletion?: string;
}

// Search API types
export interface SearchRequest {
  query: string;
  type?: 'decisions' | 'explanations' | 'conversations' | 'all';
  filters?: {
    dateRange?: { start: string; end: string };
    symbols?: string[];
    confidence?: { min: number; max: number };
  };
  limit?: number;
  offset?: number;
}

export interface SearchResponse {
  results: SearchResult[];
  totalCount: number;
  searchTime: number;
  suggestions?: string[];
}

export interface SearchResult {
  type: 'decision' | 'explanation' | 'conversation';
  id: string;
  title: string;
  snippet: string;
  relevance: number;
  timestamp: string;
  metadata: any;
}

// Aggregated types for complex operations
export interface DashboardDataRequest {
  timeframe: TimeFrame;
  symbols?: string[];
  includeRecentDecisions?: boolean;
  includePerformanceMetrics?: boolean;
  includeRiskMetrics?: boolean;
  includeSystemStatus?: boolean;
}

export interface DashboardDataResponse {
  recentDecisions?: TradingDecision[];
  performanceMetrics?: PerformanceMetrics;
  riskMetrics?: {
    currentVaR: number;
    portfolioExposure: number;
    correlationRisk: number;
    stressTestResults: any;
  };
  systemStatus?: SystemStatusResponse;
  marketOverview?: {
    symbols: Array<{
      symbol: string;
      price: number;
      change: number;
      volume: number;
    }>;
  };
}