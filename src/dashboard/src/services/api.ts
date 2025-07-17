/**
 * API service for communication with the Risk Dashboard backend
 */

import axios, { AxiosInstance, AxiosResponse } from 'axios';
import { 
  DashboardData, 
  RiskMetrics, 
  FlaggedTrade, 
  HumanDecision, 
  LoginRequest, 
  LoginResponse,
  ApiResponse,
  DecisionHistoryEntry,
  PerformanceMetrics,
  User
} from '../types';

class ApiService {
  private client: AxiosInstance;
  private token: string | null = null;

  constructor() {
    this.client = axios.create({
      baseURL: process.env.REACT_APP_API_BASE_URL || 'http://localhost:8001',
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor to add auth token
    this.client.interceptors.request.use(
      (config) => {
        if (this.token) {
          config.headers.Authorization = `Bearer ${this.token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          this.clearToken();
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );

    // Load token from localStorage
    const savedToken = localStorage.getItem('access_token');
    if (savedToken) {
      this.setToken(savedToken);
    }
  }

  setToken(token: string): void {
    this.token = token;
    localStorage.setItem('access_token', token);
  }

  clearToken(): void {
    this.token = null;
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user_info');
  }

  isAuthenticated(): boolean {
    return this.token !== null;
  }

  async login(credentials: LoginRequest): Promise<LoginResponse> {
    try {
      const response: AxiosResponse<LoginResponse> = await this.client.post('/auth/login', credentials);
      const { access_token, user_info } = response.data;
      
      this.setToken(access_token);
      localStorage.setItem('user_info', JSON.stringify(user_info));
      
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Login failed');
    }
  }

  async logout(): Promise<void> {
    try {
      await this.client.post('/auth/logout');
    } catch (error) {
      // Continue with logout even if API call fails
    } finally {
      this.clearToken();
    }
  }

  async getDashboardData(): Promise<DashboardData> {
    try {
      const response: AxiosResponse<DashboardData> = await this.client.get('/api/dashboard/data');
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch dashboard data');
    }
  }

  async getRiskMetrics(): Promise<RiskMetrics> {
    try {
      const response: AxiosResponse<RiskMetrics> = await this.client.get('/api/dashboard/risk-metrics');
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch risk metrics');
    }
  }

  async getFlaggedTrades(): Promise<FlaggedTrade[]> {
    try {
      const response: AxiosResponse<FlaggedTrade[]> = await this.client.get('/api/dashboard/flagged-trades');
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch flagged trades');
    }
  }

  async makeDecision(decision: HumanDecision): Promise<ApiResponse<string>> {
    try {
      const response: AxiosResponse<ApiResponse<string>> = await this.client.post('/api/dashboard/decide', decision);
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to process decision');
    }
  }

  async getDecisionHistory(limit?: number, userId?: string): Promise<DecisionHistoryEntry[]> {
    try {
      const params = new URLSearchParams();
      if (limit) params.append('limit', limit.toString());
      if (userId) params.append('user_id', userId);
      
      const response: AxiosResponse<DecisionHistoryEntry[]> = await this.client.get(
        `/api/dashboard/decision-history?${params.toString()}`
      );
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch decision history');
    }
  }

  async getPerformanceMetrics(): Promise<PerformanceMetrics> {
    try {
      const response: AxiosResponse<PerformanceMetrics> = await this.client.get('/api/dashboard/performance-metrics');
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch performance metrics');
    }
  }

  async getHealth(): Promise<Record<string, any>> {
    try {
      const response: AxiosResponse<Record<string, any>> = await this.client.get('/api/dashboard/health');
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.message || 'Failed to fetch health status');
    }
  }

  getCurrentUser(): User | null {
    const userInfo = localStorage.getItem('user_info');
    return userInfo ? JSON.parse(userInfo) : null;
  }

  hasPermission(permission: string): boolean {
    const user = this.getCurrentUser();
    return user?.permissions?.includes(permission as any) || false;
  }
}

export const apiService = new ApiService();
export default apiService;