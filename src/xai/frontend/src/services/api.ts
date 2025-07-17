/**
 * API Service for XAI Trading Frontend
 * 
 * Provides comprehensive API client for:
 * - Authentication and authorization
 * - Trading decisions and explanations
 * - Performance analytics and metrics
 * - Chat/conversation management
 * - Real-time data subscriptions
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import {
  ApiResponse,
  LoginRequest,
  LoginResponse,
  DecisionsListRequest,
  DecisionsListResponse,
  DecisionDetailResponse,
  ExplanationRequest,
  ExplanationResponse,
  BulkExplanationRequest,
  BulkExplanationResponse,
  AnalyticsRequest,
  AnalyticsResponse,
  ChatRequest,
  ChatResponse,
  ConversationListResponse,
  ConversationDetailResponse,
  SystemStatusResponse,
  UserPreferencesRequest,
  UserPreferencesResponse,
  ExportRequest,
  ExportResponse,
  SearchRequest,
  SearchResponse,
  DashboardDataRequest,
  DashboardDataResponse,
  ApiError,
  TimeFrame,
  PerformanceMetrics,
  TradingDecision,
  ExplanationData
} from '../types/api';

export class ApiService {
  private client: AxiosInstance;
  private authToken: string | null = null;

  constructor(baseURL: string = process.env.REACT_APP_API_URL || 'http://localhost:8001') {
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor for authentication
    this.client.interceptors.request.use(
      (config) => {
        if (this.authToken) {
          config.headers.Authorization = `Bearer ${this.authToken}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      async (error) => {
        if (error.response?.status === 401) {
          // Token expired or invalid
          this.clearToken();
          // Optionally redirect to login
          window.location.href = '/login';
        }
        
        // Transform error to our ApiError format
        const apiError: ApiError = {
          code: error.response?.data?.code || 'UNKNOWN_ERROR',
          message: error.response?.data?.message || error.message || 'An unknown error occurred',
          details: error.response?.data?.details,
          timestamp: new Date().toISOString()
        };
        
        return Promise.reject(apiError);
      }
    );

    // Load token from localStorage if available
    this.loadToken();
  }

  // Authentication methods
  private loadToken(): void {
    const token = localStorage.getItem('auth_token');
    if (token) {
      this.authToken = token;
    }
  }

  private saveToken(token: string): void {
    this.authToken = token;
    localStorage.setItem('auth_token', token);
  }

  public clearToken(): void {
    this.authToken = null;
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user_data');
  }

  public isAuthenticated(): boolean {
    return !!this.authToken;
  }

  public getCurrentUser(): any {
    const userData = localStorage.getItem('user_data');
    return userData ? JSON.parse(userData) : null;
  }

  // Authentication API
  public async login(credentials: LoginRequest): Promise<ApiResponse<LoginResponse>> {
    const response = await this.client.post<ApiResponse<LoginResponse>>('/auth/login', credentials);
    
    if (response.data.success && response.data.data) {
      this.saveToken(response.data.data.token);
      localStorage.setItem('user_data', JSON.stringify(response.data.data.user));
    }
    
    return response.data;
  }

  public async logout(): Promise<void> {
    try {
      await this.client.post('/auth/logout');
    } catch (error) {
      // Continue with logout even if API call fails
      console.warn('Logout API call failed:', error);
    } finally {
      this.clearToken();
    }
  }

  public async refreshToken(): Promise<void> {
    const response = await this.client.post<ApiResponse<{ token: string }>>('/auth/refresh');
    
    if (response.data.success && response.data.data) {
      this.saveToken(response.data.data.token);
    }
  }

  // Trading Decisions API
  public async getDecisions(params: DecisionsListRequest): Promise<ApiResponse<DecisionsListResponse>> {
    const response = await this.client.get<ApiResponse<DecisionsListResponse>>('/decisions', { params });
    return response.data;
  }

  public async getDecisionDetail(id: string): Promise<ApiResponse<DecisionDetailResponse>> {
    const response = await this.client.get<ApiResponse<DecisionDetailResponse>>(`/decisions/${id}`);
    return response.data;
  }

  public async searchDecisions(params: SearchRequest): Promise<ApiResponse<SearchResponse>> {
    const response = await this.client.post<ApiResponse<SearchResponse>>('/decisions/search', params);
    return response.data;
  }

  // Explanations API
  public async getExplanations(params: any): Promise<ApiResponse<{ explanations: ExplanationData[]; pagination: any }>> {
    const response = await this.client.get<ApiResponse<{ explanations: ExplanationData[]; pagination: any }>>('/explanations', { params });
    return response.data;
  }

  public async getExplanation(params: ExplanationRequest): Promise<ApiResponse<ExplanationResponse>> {
    const response = await this.client.post<ApiResponse<ExplanationResponse>>('/explanations/generate', params);
    return response.data;
  }

  public async getBulkExplanations(params: BulkExplanationRequest): Promise<ApiResponse<BulkExplanationResponse>> {
    const response = await this.client.post<ApiResponse<BulkExplanationResponse>>('/explanations/bulk', params);
    return response.data;
  }

  // Analytics API
  public async getAnalytics(params: AnalyticsRequest): Promise<ApiResponse<AnalyticsResponse>> {
    const response = await this.client.post<ApiResponse<AnalyticsResponse>>('/analytics/query', params);
    return response.data;
  }

  public async getPerformanceMetrics(timeframe: TimeFrame): Promise<ApiResponse<PerformanceMetrics>> {
    const response = await this.client.get<ApiResponse<PerformanceMetrics>>(`/analytics/performance`, {
      params: { timeframe }
    });
    return response.data;
  }

  public async getDashboardData(params: DashboardDataRequest): Promise<ApiResponse<DashboardDataResponse>> {
    const response = await this.client.post<ApiResponse<DashboardDataResponse>>('/analytics/dashboard', params);
    return response.data;
  }

  // Natural Language Processing
  public async processNaturalLanguageQuery(params: { query: string; context?: any }): Promise<ApiResponse<any>> {
    const response = await this.client.post<ApiResponse<any>>('/nlp/query', params);
    return response.data;
  }

  // Chat/Conversation API
  public async getConversations(): Promise<ApiResponse<ConversationListResponse>> {
    const response = await this.client.get<ApiResponse<ConversationListResponse>>('/conversations');
    return response.data;
  }

  public async getConversationDetail(id: string): Promise<ApiResponse<ConversationDetailResponse>> {
    const response = await this.client.get<ApiResponse<ConversationDetailResponse>>(`/conversations/${id}`);
    return response.data;
  }

  public async createConversation(params: { title?: string }): Promise<ApiResponse<{ conversation: any }>> {
    const response = await this.client.post<ApiResponse<{ conversation: any }>>('/conversations', params);
    return response.data;
  }

  public async deleteConversation(id: string): Promise<ApiResponse<void>> {
    const response = await this.client.delete<ApiResponse<void>>(`/conversations/${id}`);
    return response.data;
  }

  public async sendChatMessage(params: ChatRequest): Promise<ApiResponse<ChatResponse>> {
    const response = await this.client.post<ApiResponse<ChatResponse>>('/chat/message', params);
    return response.data;
  }

  // System Status API
  public async getSystemStatus(): Promise<ApiResponse<SystemStatusResponse>> {
    const response = await this.client.get<ApiResponse<SystemStatusResponse>>('/system/status');
    return response.data;
  }

  public async getSystemHealth(): Promise<ApiResponse<any>> {
    const response = await this.client.get<ApiResponse<any>>('/system/health');
    return response.data;
  }

  // User Preferences API
  public async getUserPreferences(): Promise<ApiResponse<UserPreferencesResponse>> {
    const response = await this.client.get<ApiResponse<UserPreferencesResponse>>('/user/preferences');
    return response.data;
  }

  public async updateUserPreferences(params: UserPreferencesRequest): Promise<ApiResponse<UserPreferencesResponse>> {
    const response = await this.client.put<ApiResponse<UserPreferencesResponse>>('/user/preferences', params);
    return response.data;
  }

  // Export API
  public async requestExport(params: ExportRequest): Promise<ApiResponse<ExportResponse>> {
    const response = await this.client.post<ApiResponse<ExportResponse>>('/export/request', params);
    return response.data;
  }

  public async getExportStatus(exportId: string): Promise<ApiResponse<ExportResponse>> {
    const response = await this.client.get<ApiResponse<ExportResponse>>(`/export/status/${exportId}`);
    return response.data;
  }

  public async downloadExport(exportId: string): Promise<Blob> {
    const response = await this.client.get(`/export/download/${exportId}`, {
      responseType: 'blob'
    });
    return response.data;
  }

  // File Upload API
  public async uploadFile(file: File, purpose: string = 'analysis'): Promise<ApiResponse<{ fileId: string; url: string }>> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('purpose', purpose);

    const response = await this.client.post<ApiResponse<{ fileId: string; url: string }>>('/files/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  }

  // Advanced Search API
  public async advancedSearch(params: SearchRequest): Promise<ApiResponse<SearchResponse>> {
    const response = await this.client.post<ApiResponse<SearchResponse>>('/search/advanced', params);
    return response.data;
  }

  // Real-time Subscriptions API (for WebSocket subscription management)
  public async subscribeToStream(streamType: string, params: any): Promise<ApiResponse<{ subscriptionId: string }>> {
    const response = await this.client.post<ApiResponse<{ subscriptionId: string }>>('/streams/subscribe', {
      streamType,
      params
    });
    return response.data;
  }

  public async unsubscribeFromStream(subscriptionId: string): Promise<ApiResponse<void>> {
    const response = await this.client.delete<ApiResponse<void>>(`/streams/unsubscribe/${subscriptionId}`);
    return response.data;
  }

  // Utility methods
  public async ping(): Promise<number> {
    const start = Date.now();
    await this.client.get('/ping');
    return Date.now() - start;
  }

  public async getApiVersion(): Promise<ApiResponse<{ version: string; build: string }>> {
    const response = await this.client.get<ApiResponse<{ version: string; build: string }>>('/version');
    return response.data;
  }

  // Batch operations
  public async batchRequest(requests: Array<{ method: string; url: string; data?: any }>): Promise<ApiResponse<any[]>> {
    const response = await this.client.post<ApiResponse<any[]>>('/batch', { requests });
    return response.data;
  }

  // Configuration
  public setBaseURL(baseURL: string): void {
    this.client.defaults.baseURL = baseURL;
  }

  public setTimeout(timeout: number): void {
    this.client.defaults.timeout = timeout;
  }

  public setRequestInterceptor(interceptor: (config: AxiosRequestConfig) => AxiosRequestConfig): void {
    this.client.interceptors.request.use(interceptor);
  }

  public setResponseInterceptor(
    onFulfilled: (response: AxiosResponse) => AxiosResponse,
    onRejected: (error: any) => any
  ): void {
    this.client.interceptors.response.use(onFulfilled, onRejected);
  }
}

// Create and export singleton instance
export const apiService = new ApiService();

// Export the class for testing or custom instances
export default ApiService;