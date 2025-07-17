/**
 * TypeScript type definitions for the Risk Dashboard
 */

export interface RiskMetrics {
  timestamp: string;
  portfolio_var: number;
  correlation_shock_level: number;
  max_drawdown: number;
  sharpe_ratio: number;
  volatility: number;
  leverage: number;
  liquidity_risk: number;
}

export interface AgentStatus {
  agent_name: string;
  status: 'active' | 'inactive' | 'error';
  last_update: string;
  performance_score: number;
  current_recommendation: string;
  confidence: number;
}

export interface FlaggedTrade {
  trade_id: string;
  symbol: string;
  direction: 'LONG' | 'SHORT';
  quantity: number;
  entry_price: number;
  risk_score: number;
  failure_probability: number;
  agent_recommendations: AgentRecommendation[];
  flagged_at: string;
  flagged_reason: string;
  expires_at: string;
}

export interface AgentRecommendation {
  agent_name: string;
  recommendation: string;
  confidence: number;
  reasoning: Record<string, any>;
}

export interface CrisisAlert {
  alert_id: string;
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  alert_type: string;
  message: string;
  detected_at: string;
  metrics: Record<string, number>;
  recommended_actions: string[];
}

export interface DashboardData {
  risk_metrics: RiskMetrics;
  agent_statuses: AgentStatus[];
  flagged_trades: FlaggedTrade[];
  crisis_alerts: CrisisAlert[];
  last_updated: string;
}

export interface HumanDecision {
  trade_id: string;
  decision: 'APPROVE' | 'REJECT';
  reasoning: string;
  user_id: string;
  timestamp: string;
}

export interface User {
  user_id: string;
  username: string;
  email: string;
  role: UserRole;
  permissions: Permission[];
  session_id: string;
  login_time: string;
  last_activity: string;
  mfa_enabled: boolean;
}

export enum UserRole {
  RISK_OPERATOR = 'risk_operator',
  RISK_MANAGER = 'risk_manager',
  SYSTEM_ADMIN = 'system_admin',
  COMPLIANCE_OFFICER = 'compliance_officer',
  VIEWER = 'viewer'
}

export enum Permission {
  DASHBOARD_READ = 'dashboard_read',
  DASHBOARD_ADMIN = 'dashboard_admin',
  TRADE_REVIEW = 'trade_review',
  TRADE_APPROVE = 'trade_approve',
  TRADE_REJECT = 'trade_reject',
  HIGH_RISK_APPROVE = 'high_risk_approve',
  SYSTEM_INTEGRATION = 'system_integration',
  USER_MANAGEMENT = 'user_management',
  AUDIT_READ = 'audit_read',
  CONFIG_MANAGE = 'config_manage',
  COMPLIANCE_READ = 'compliance_read',
  COMPLIANCE_REPORT = 'compliance_report'
}

export interface LoginRequest {
  username: string;
  password: string;
  mfa_token?: string;
}

export interface LoginResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  user_info: User;
}

export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
}

export interface ThemeConfig {
  mode: 'light' | 'dark';
  primary: string;
  secondary: string;
}

export interface NotificationSettings {
  sound_enabled: boolean;
  push_enabled: boolean;
  severity_filter: string[];
}

export interface UserPreferences {
  theme: ThemeConfig;
  notifications: NotificationSettings;
  auto_refresh: boolean;
  refresh_interval: number;
}

export interface ApiResponse<T> {
  data?: T;
  error?: string;
  message?: string;
  status: number;
}

export interface DecisionHistoryEntry {
  decision_id: string;
  trade_id: string;
  decision: 'APPROVE' | 'REJECT';
  reasoning: string;
  user_id: string;
  user_role: string;
  timestamp: string;
  processing_time_ms: number;
  execution_confirmed: boolean;
}

export interface PerformanceMetrics {
  total_decisions: number;
  pending_decisions: number;
  average_processing_time_ms: number;
  approval_rate: number;
  decision_history_size: number;
}