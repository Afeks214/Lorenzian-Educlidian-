import axios, { AxiosInstance, AxiosResponse } from 'axios';

interface LoginRequest {
  expert_id: string;
  password: string;
}

interface LoginResponse {
  access_token: string;
  token_type: string;
}

interface PendingDecisionsResponse {
  decisions: Array<{
    decision_id: string;
    timestamp: string;
    complexity: string;
    symbol: string;
    deadline: string;
    strategies_count: number;
  }>;
  count: number;
}

interface DecisionDetailsResponse {
  decision: {
    decision_id: string;
    timestamp: string;
    context: {
      symbol: string;
      price: number;
      volatility: number;
      volume: number;
      trend_strength: number;
      support_level: number;
      resistance_level: number;
      time_of_day: string;
      market_regime: string;
      correlation_shock: boolean;
    };
    complexity: string;
    strategies: Array<{
      strategy_id: string;
      strategy_type: string;
      entry_price: number;
      position_size: number;
      stop_loss: number;
      take_profit: number;
      time_horizon: number;
      risk_reward_ratio: number;
      confidence_score: number;
      reasoning: string;
      expected_pnl: number;
      max_drawdown: number;
    }>;
    current_position: any;
    expert_deadline: string;
    model_recommendation: string;
    confidence_threshold: number;
  };
  time_remaining: number;
}

interface FeedbackRequest {
  decision_id: string;
  chosen_strategy_id: string;
  confidence: number;
  reasoning: string;
  market_view: string;
  risk_assessment: string;
}

interface ExpertAnalyticsResponse {
  decisions_count: number;
  average_confidence: number;
  success_rate: number;
  recent_decisions: Array<{
    decision_id: string;
    timestamp: string;
    confidence: number;
    strategy: string;
  }>;
}

class ApiService {
  private api: AxiosInstance;

  constructor() {
    this.api = axios.create({
      baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor to add auth token
    this.api.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('auth_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor for error handling
    this.api.interceptors.response.use(
      (response: AxiosResponse) => {
        return response;
      },
      (error) => {
        if (error.response?.status === 401) {
          // Token expired or invalid
          localStorage.removeItem('auth_token');
          window.location.href = '/login';
        }
        
        const message = error.response?.data?.detail || 
                       error.response?.data?.message || 
                       error.message || 
                       'An unexpected error occurred';
        
        return Promise.reject(new Error(message));
      }
    );
  }

  async login(expertId: string, password: string): Promise<LoginResponse> {
    const response = await this.api.post<LoginResponse>('/auth/login', {
      expert_id: expertId,
      password: password,
    });
    return response.data;
  }

  async getPendingDecisions(): Promise<PendingDecisionsResponse> {
    const response = await this.api.get<PendingDecisionsResponse>('/decisions/pending');
    return response.data;
  }

  async getDecisionDetails(decisionId: string): Promise<DecisionDetailsResponse> {
    const response = await this.api.get<DecisionDetailsResponse>(`/decisions/${decisionId}`);
    return response.data;
  }

  async submitFeedback(decisionId: string, feedback: FeedbackRequest): Promise<{ status: string; message: string }> {
    const response = await this.api.post(`/decisions/${decisionId}/feedback`, feedback);
    return response.data;
  }

  async getExpertAnalytics(expertId: string): Promise<ExpertAnalyticsResponse> {
    const response = await this.api.get<ExpertAnalyticsResponse>(`/analytics/expert/${expertId}`);
    return response.data;
  }

  // Health check endpoint
  async healthCheck(): Promise<{ status: string }> {
    const response = await this.api.get('/health');
    return response.data;
  }
}

export const apiService = new ApiService();
export default apiService;