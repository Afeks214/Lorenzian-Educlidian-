import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { apiService } from '../services/api';

interface User {
  expert_id: string;
  name?: string;
  role?: string;
  permissions?: string[];
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  login: (expertId: string, password: string) => Promise<void>;
  logout: () => void;
  loading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  // Check for existing token on mount
  useEffect(() => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      // Validate token by making a test API call
      apiService.getPendingDecisions()
        .then(() => {
          // Token is valid, extract user info from token
          try {
            const payload = JSON.parse(atob(token.split('.')[1]));
            setUser({
              expert_id: payload.expert_id,
              name: payload.name,
              role: payload.role,
              permissions: payload.permissions
            });
          } catch (error) {
            // Invalid token
            localStorage.removeItem('auth_token');
          }
        })
        .catch(() => {
          // Token is invalid
          localStorage.removeItem('auth_token');
        })
        .finally(() => {
          setLoading(false);
        });
    } else {
      setLoading(false);
    }
  }, []);

  const login = async (expertId: string, password: string): Promise<void> => {
    try {
      const response = await apiService.login(expertId, password);
      
      // Store token
      localStorage.setItem('auth_token', response.access_token);
      
      // Extract user info from token
      const payload = JSON.parse(atob(response.access_token.split('.')[1]));
      setUser({
        expert_id: payload.expert_id,
        name: payload.name,
        role: payload.role,
        permissions: payload.permissions
      });
      
    } catch (error) {
      throw error;
    }
  };

  const logout = () => {
    localStorage.removeItem('auth_token');
    setUser(null);
  };

  const value: AuthContextType = {
    user,
    isAuthenticated: !!user,
    login,
    logout,
    loading
  };

  if (loading) {
    return <div>Loading...</div>; // You could use a proper loading component here
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};