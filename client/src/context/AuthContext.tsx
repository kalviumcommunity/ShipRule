'use client';

import React, { createContext, useContext, useState, useEffect } from 'react';
import { apiService } from '@/services/api';
import { AuthResponse } from '@/types/api';

interface UserSession {
  email: string;
  role: 'user' | 'admin';
}

interface AuthContextType {
  user: UserSession | null;
  token: string | null;
  role: 'user' | 'admin' | null;
  loading: boolean;
  login: (email: string, pass: string) => Promise<AuthResponse>;
  signup: (email: string, pass: string, fullName?: string) => Promise<AuthResponse>;
  masterLogin: (key: string) => Promise<AuthResponse>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [token, setToken] = useState<string | null>(null);
  const [role, setRole] = useState<'user' | 'admin' | null>(null);
  const [user, setUser] = useState<UserSession | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const savedToken = localStorage.getItem('shiprule_token');
    const savedRole = localStorage.getItem('shiprule_role') as 'user' | 'admin' | null;
    const savedEmail = localStorage.getItem('shiprule_email');

    if (savedToken && savedRole && savedEmail) {
      setToken(savedToken);
      setRole(savedRole);
      setUser({ email: savedEmail, role: savedRole });
    }
    setLoading(false);
  }, []);

  const signup = async (email: string, pass: string, fullName?: string): Promise<AuthResponse> => {
    const res = await apiService.signup(email, pass, fullName);
    setToken(res.access_token);
    setRole(res.role);
    setUser({ email: res.email, role: res.role });

    localStorage.setItem('shiprule_token', res.access_token);
    localStorage.setItem('shiprule_role', res.role);
    localStorage.setItem('shiprule_email', res.email);
    return res;
  };

  const login = async (email: string, pass: string): Promise<AuthResponse> => {
    const res = await apiService.login(email, pass);
    setToken(res.access_token);
    setRole(res.role);
    setUser({ email: res.email, role: res.role });

    localStorage.setItem('shiprule_token', res.access_token);
    localStorage.setItem('shiprule_role', res.role);
    localStorage.setItem('shiprule_email', res.email);
    return res;
  };

  const masterLogin = async (key: string): Promise<AuthResponse> => {
    const res = await apiService.masterLogin(key);
    setToken(res.access_token);
    setRole(res.role);
    setUser({ email: res.email, role: res.role });

    localStorage.setItem('shiprule_token', res.access_token);
    localStorage.setItem('shiprule_role', res.role);
    localStorage.setItem('shiprule_email', res.email);
    return res;
  };

  const logout = () => {
    setToken(null);
    setRole(null);
    setUser(null);

    localStorage.removeItem('shiprule_token');
    localStorage.removeItem('shiprule_role');
    localStorage.removeItem('shiprule_email');
  };

  return (
    <AuthContext.Provider
      value={{ user, token, role, loading, login, signup, masterLogin, logout }}
    >
      {children}
    </AuthContext.Provider>
  );
}


export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
