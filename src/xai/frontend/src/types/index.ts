/**
 * TypeScript definitions for XAI Trading Frontend
 */

// Trading-related types
export interface TradingSymbol {
  symbol: string;
  name: string;
  assetClass: AssetClass;
  exchange: string;
  currentPrice?: number;
  priceChange?: number;
  priceChangePercent?: number;
}

export enum AssetClass {
  EQUITIES = 'EQUITIES',
  FUTURES = 'FUTURES',
  FOREX = 'FOREX',
  CRYPTO = 'CRYPTO',
  BONDS = 'BONDS',
  COMMODITIES = 'COMMODITIES'
}

export enum ActionType {
  HOLD = 0,
  DECREASE_LONG = 1,
  INCREASE_LONG = 2,
  MARKET_BUY = 3,
  MARKET_SELL = 4,
  LIMIT_BUY = 5,
  LIMIT_SELL = 6,
  INCREASE_SHORT = 7,
  DECREASE_SHORT = 8
}

export interface TradingDecision {
  id: string;
  timestamp: string;
  symbol: string;
  assetClass: AssetClass;
  action: ActionType;
  confidence: number;
  size: number;
  price?: number;
  reasoning: string;
  agentContributions: AgentContribution[];
  riskMetrics: RiskMetrics;
  marketContext: MarketContext;
  executionStatus: ExecutionStatus;
}

export interface AgentContribution {
  agentId: string;
  agentName: string;
  action: ActionType;
  confidence: number;
  weight: number;
  keyFactors: string[];
  shapValues?: number[];
}

export interface RiskMetrics {
  varRisk: number;
  positionRisk: number;
  correlationRisk: number;
  volatility: number;
  sharpeRatio?: number;
  maxDrawdown?: number;
  beta?: number;
}

export interface MarketContext {
  volatility: number;
  trend: 'BULLISH' | 'BEARISH' | 'SIDEWAYS';
  volume: number;
  liquidity: number;
  technicalIndicators: Record<string, number>;
  fundamentalFactors: Record<string, number>;
}

export enum ExecutionStatus {
  PENDING = 'PENDING',
  EXECUTING = 'EXECUTING',
  EXECUTED = 'EXECUTED',
  FAILED = 'FAILED',
  CANCELLED = 'CANCELLED'
}

// Chat/Explanation types
export interface ChatMessage {
  id: string;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  metadata?: ChatMessageMetadata;
  attachments?: ChatAttachment[];
}

export interface ChatMessageMetadata {
  decisionId?: string;
  symbols?: string[];
  timeframe?: string;
  analysisType?: AnalysisType;
  confidence?: number;
  processingTime?: number;
}

export interface ChatAttachment {
  type: 'chart' | 'table' | 'image' | 'document';
  url?: string;
  data?: any;
  title: string;
  description?: string;
}

export enum AnalysisType {
  DECISION_EXPLANATION = 'DECISION_EXPLANATION',
  PERFORMANCE_ANALYSIS = 'PERFORMANCE_ANALYSIS',
  RISK_ASSESSMENT = 'RISK_ASSESSMENT',
  MARKET_ANALYSIS = 'MARKET_ANALYSIS',
  AGENT_COMPARISON = 'AGENT_COMPARISON',
  FEATURE_IMPORTANCE = 'FEATURE_IMPORTANCE'
}

export interface ConversationThread {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessage[];
  tags?: string[];
  archived?: boolean;
}

// Analytics types
export interface PerformanceMetrics {
  totalReturns: number;
  annualizedReturns: number;
  volatility: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  profitFactor: number;
  tradesCount: number;
  avgTradeSize: number;
  periodStart: string;
  periodEnd: string;
}

export interface AnalyticsQuery {
  type: AnalysisType;
  symbols?: string[];
  timeframe: TimeFrame;
  startDate: string;
  endDate: string;
  filters?: Record<string, any>;
  aggregation?: 'hourly' | 'daily' | 'weekly' | 'monthly';
}

export enum TimeFrame {
  ONE_HOUR = '1h',
  FOUR_HOURS = '4h',
  ONE_DAY = '1d',
  ONE_WEEK = '1w',
  ONE_MONTH = '1M',
  THREE_MONTHS = '3M',
  SIX_MONTHS = '6M',
  ONE_YEAR = '1Y',
  ALL_TIME = 'ALL'
}

export interface ChartData {
  labels: string[];
  datasets: ChartDataset[];
  options?: any;
}

export interface ChartDataset {
  label: string;
  data: number[];
  backgroundColor?: string | string[];
  borderColor?: string;
  borderWidth?: number;
  fill?: boolean;
  tension?: number;
  type?: 'line' | 'bar' | 'scatter' | 'pie' | 'doughnut';
}

// WebSocket types
export interface WebSocketMessage {
  type: WSMessageType;
  payload: any;
  timestamp: string;
  id?: string;
}

export enum WSMessageType {
  DECISION_UPDATE = 'DECISION_UPDATE',
  EXPLANATION_READY = 'EXPLANATION_READY',
  PERFORMANCE_UPDATE = 'PERFORMANCE_UPDATE',
  RISK_ALERT = 'RISK_ALERT',
  SYSTEM_STATUS = 'SYSTEM_STATUS',
  CHAT_RESPONSE = 'CHAT_RESPONSE',
  AGENT_STATUS = 'AGENT_STATUS'
}

export interface DecisionUpdate {
  decision: TradingDecision;
  explanation: ExplanationData;
  priority: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
}

export interface ExplanationData {
  id: string;
  decisionId: string;
  type: AnalysisType;
  audience: ExplanationAudience;
  summary: string;
  detailedReasoning: string;
  featureImportance: Record<string, number>;
  topFactors: Array<{ name: string; importance: number; direction: 'positive' | 'negative' }>;
  alternativeScenarios?: AlternativeScenario[];
  confidence: number;
  generationTime: number;
}

export enum ExplanationAudience {
  TRADER = 'TRADER',
  RISK_MANAGER = 'RISK_MANAGER',
  REGULATOR = 'REGULATOR',
  CLIENT = 'CLIENT',
  TECHNICAL = 'TECHNICAL'
}

export interface AlternativeScenario {
  description: string;
  action: ActionType;
  confidence: number;
  expectedOutcome: string;
  riskAssessment: string;
}

// UI State types
export interface AppState {
  user: User | null;
  theme: 'light' | 'dark';
  sidebarOpen: boolean;
  notifications: Notification[];
  loading: boolean;
  error: string | null;
}

export interface User {
  id: string;
  username: string;
  email: string;
  role: UserRole;
  permissions: Permission[];
  preferences: UserPreferences;
}

export enum UserRole {
  TRADER = 'TRADER',
  RISK_MANAGER = 'RISK_MANAGER',
  ANALYST = 'ANALYST',
  ADMIN = 'ADMIN'
}

export enum Permission {
  VIEW_DECISIONS = 'VIEW_DECISIONS',
  VIEW_EXPLANATIONS = 'VIEW_EXPLANATIONS',
  VIEW_ANALYTICS = 'VIEW_ANALYTICS',
  MANAGE_SETTINGS = 'MANAGE_SETTINGS',
  EXPORT_DATA = 'EXPORT_DATA'
}

export interface UserPreferences {
  defaultTimeframe: TimeFrame;
  chartSettings: ChartSettings;
  notificationSettings: NotificationSettings;
  explanationAudience: ExplanationAudience;
}

export interface ChartSettings {
  defaultType: 'line' | 'candlestick' | 'bar';
  showVolume: boolean;
  showTechnicalIndicators: boolean;
  colorScheme: 'default' | 'colorblind' | 'high_contrast';
}

export interface NotificationSettings {
  enabled: boolean;
  decisionAlerts: boolean;
  riskAlerts: boolean;
  performanceAlerts: boolean;
  sound: boolean;
}

export interface Notification {
  id: string;
  type: 'info' | 'warning' | 'error' | 'success';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  actions?: NotificationAction[];
}

export interface NotificationAction {
  label: string;
  action: () => void;
  style?: 'primary' | 'secondary' | 'danger';
}

// API Response types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  metadata?: {
    pagination?: PaginationInfo;
    processingTime?: number;
    version?: string;
  };
}

export interface PaginationInfo {
  page: number;
  pageSize: number;
  totalItems: number;
  totalPages: number;
  hasNext: boolean;
  hasPrevious: boolean;
}

// Search and Filter types
export interface SearchFilters {
  symbols?: string[];
  assetClasses?: AssetClass[];
  actions?: ActionType[];
  dateRange?: {
    start: string;
    end: string;
  };
  confidenceRange?: {
    min: number;
    max: number;
  };
  agents?: string[];
  executionStatus?: ExecutionStatus[];
}

export interface SortOptions {
  field: string;
  direction: 'asc' | 'desc';
}

// Export all types
export * from './api';
export * from './charts';