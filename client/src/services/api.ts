import { API_BASE_URL } from '@/config/env';
import {
  QueryRequest,
  QueryResponse,
  DocumentUploadResponse,
  DocumentItem,
  HealthResponse,
  AuthResponse,
  AdminSettings,
} from '@/types/api';

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

function getStoredToken(): string | null {
  if (typeof window !== 'undefined') {
    return localStorage.getItem('shiprule_token');
  }
  return null;
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `HTTP Error ${response.status}: ${response.statusText}`;
    try {
      const errorJson = await response.json();
      if (errorJson.detail) {
        detail = typeof errorJson.detail === 'string'
          ? errorJson.detail
          : JSON.stringify(errorJson.detail);
      }
    } catch {
      // Use fallback status text
    }
    throw new ApiError(response.status, detail);
  }
  return response.json();
}

function getHeaders(customHeaders: Record<string, string> = {}): Record<string, string> {
  const headers: Record<string, string> = { ...customHeaders };
  const token = getStoredToken();
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
}

export const apiService = {
  /**
   * User signup with email, password, and optional full name.
   */
  async signup(email: string, password: string, full_name?: string): Promise<AuthResponse> {
    try {
      const res = await fetch(`${API_BASE_URL}/auth/signup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password, full_name }),
      });
      return await handleResponse<AuthResponse>(res);
    } catch (err: any) {
      if (err instanceof ApiError) throw err;
      throw new ApiError(500, err.message || 'Signup failed.');
    }
  },

  /**
   * User login with email and password.
   */
  async login(email: string, password: string): Promise<AuthResponse> {
    try {
      const res = await fetch(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });
      return await handleResponse<AuthResponse>(res);
    } catch (err: any) {
      if (err instanceof ApiError) throw err;
      throw new ApiError(500, err.message || 'Login failed.');
    }
  },


  /**
   * Master Admin login with access key (krishna7).
   */
  async masterLogin(access_key: string): Promise<AuthResponse> {
    try {
      const res = await fetch(`${API_BASE_URL}/auth/master-login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ access_key }),
      });
      return await handleResponse<AuthResponse>(res);
    } catch (err: any) {
      if (err instanceof ApiError) throw err;
      throw new ApiError(500, err.message || 'Master Admin authentication failed.');
    }
  },

  /**
   * Updates user password.
   */
  async changePassword(old_password: string, new_password: string): Promise<{ status: string; message: string }> {
    try {
      const res = await fetch(`${API_BASE_URL}/auth/change-password`, {
        method: 'POST',
        headers: getHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ old_password, new_password }),
      });
      return await handleResponse<{ status: string; message: string }>(res);
    } catch (err: any) {
      if (err instanceof ApiError) throw err;
      throw new ApiError(500, err.message || 'Failed to update password.');
    }
  },

  /**
   * Fetches backend health status.
   */
  async checkHealth(): Promise<HealthResponse> {
    try {
      const res = await fetch(`${API_BASE_URL}/health`, { cache: 'no-store' });
      return await handleResponse<HealthResponse>(res);
    } catch (err: any) {
      if (err instanceof ApiError) throw err;
      throw new ApiError(500, `Cannot connect to ShipRule backend at ${API_BASE_URL}.`);
    }
  },

  /**
   * Submits query to RAG pipeline.
   */
  async submitQuery(payload: QueryRequest): Promise<QueryResponse> {
    try {
      const res = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: getHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify(payload),
      });
      return await handleResponse<QueryResponse>(res);
    } catch (err: any) {
      if (err instanceof ApiError) throw err;
      throw new ApiError(500, err.message || 'Failed to submit query.');
    }
  },

  /**
   * Admin-protected: Uploads document.
   */
  async uploadDocument(file: File): Promise<DocumentUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE_URL}/documents`, {
        method: 'POST',
        headers: getHeaders(),
        body: formData,
      });
      return await handleResponse<DocumentUploadResponse>(res);
    } catch (err: any) {
      if (err instanceof ApiError) throw err;
      throw new ApiError(500, err.message || 'Failed to upload document.');
    }
  },

  /**
   * Admin-protected: Deletes uploaded document.
   */
  async deleteDocument(storedFilename: string): Promise<{ status: string; removed_chunks: number }> {
    try {
      const res = await fetch(`${API_BASE_URL}/admin/documents/${encodeURIComponent(storedFilename)}`, {
        method: 'DELETE',
        headers: getHeaders(),
      });
      return await handleResponse<{ status: string; removed_chunks: number }>(res);
    } catch (err: any) {
      if (err instanceof ApiError) throw err;
      throw new ApiError(500, err.message || 'Failed to delete document.');
    }
  },

  /**
   * Admin-protected: Fetches RAG admin settings (Top-K).
   */
  async getAdminSettings(): Promise<AdminSettings> {
    try {
      const res = await fetch(`${API_BASE_URL}/admin/settings`, {
        headers: getHeaders(),
        cache: 'no-store',
      });
      return await handleResponse<AdminSettings>(res);
    } catch (err: any) {
      if (err instanceof ApiError) throw err;
      throw new ApiError(500, err.message || 'Failed to fetch admin settings.');
    }
  },

  /**
   * Admin-protected: Updates Top-K setting.
   */
  async updateAdminSettings(default_top_k: number): Promise<{ status: string; settings: { default_top_k: number } }> {
    try {
      const res = await fetch(`${API_BASE_URL}/admin/settings`, {
        method: 'POST',
        headers: getHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ default_top_k }),
      });
      return await handleResponse<{ status: string; settings: { default_top_k: number } }>(res);
    } catch (err: any) {
      if (err instanceof ApiError) throw err;
      throw new ApiError(500, err.message || 'Failed to update Top-K settings.');
    }
  },

  /**
   * Admin-protected: Fetches registered users list.
   */
  async getAdminUsers(): Promise<{ users: Array<import('../types/api').AdminUser> }> {
    try {
      const res = await fetch(`${API_BASE_URL}/admin/users`, {
        headers: getHeaders(),
        cache: 'no-store',
      });
      return await handleResponse<{ users: Array<import('../types/api').AdminUser> }>(res);
    } catch (err) {
      console.warn('Failed to fetch admin users:', err);
      return { users: [] };
    }
  },

  /**
   * Admin-protected: Fetches user analytics and token metrics for a single user.
   */
  async getAdminUserDetail(userEmail: string): Promise<import('../types/api').UserAnalytics> {
    const encoded = encodeURIComponent(userEmail);
    const res = await fetch(`${API_BASE_URL}/admin/users/detail/${encoded}`, {
      headers: getHeaders(),
      cache: 'no-store',
    });
    return await handleResponse<import('../types/api').UserAnalytics>(res);
  },

  /**
   * Admin-protected: Fetches query activity logs for user monitoring.
   */
  async getAdminLogs(limit: number = 50): Promise<{ logs: Array<{ timestamp: string; user_email: string; question: string; status: string; latency_ms: number }> }> {
    try {
      const res = await fetch(`${API_BASE_URL}/admin/logs?limit=${limit}`, {
        headers: getHeaders(),
        cache: 'no-store',
      });
      return await handleResponse<{ logs: Array<{ timestamp: string; user_email: string; question: string; status: string; latency_ms: number }> }>(res);
    } catch (err) {
      console.warn('Failed to fetch admin logs:', err);
      return { logs: [] };
    }
  },

  /**
   * Fetches uploaded document list.
   */
  async getDocuments(): Promise<DocumentItem[]> {
    try {
      const res = await fetch(`${API_BASE_URL}/documents`, { cache: 'no-store' });
      return await handleResponse<DocumentItem[]>(res);
    } catch {
      return [];
    }
  },
};
