'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';

import {
  Key,
  ShieldCheck,
  UploadCloud,
  FileText,
  Trash2,
  Sliders,
  CheckCircle2,
  AlertCircle,
  Clock,
  HardDrive,
  RefreshCw,
  Activity,
  Cpu,
  Save,
  Lock,
  Users,
  Search,
  UserCheck,
  BarChart3,
  ChevronRight
} from 'lucide-react';
import { useAuth } from '@/context/AuthContext';
import { apiService, ApiError } from '@/services/api';
import { DocumentItem, DocumentUploadResponse, AdminSettings, HealthResponse, AdminUser } from '@/types/api';
import { formatBytes } from '@/lib/utils';

export default function MasterAdminPage() {
  const { role, masterLogin } = useAuth();
  const [accessKeyInput, setAccessKeyInput] = useState('');
  const [authError, setAuthError] = useState<string | null>(null);
  const [authLoading, setAuthLoading] = useState(false);

  // Active Admin Tab
  const [activeTab, setActiveTab] = useState<'logs' | 'documents' | 'settings' | 'status'>('logs');

  // User Monitoring & Logs state
  const [registeredUsers, setRegisteredUsers] = useState<AdminUser[]>([]);

  const [queryLogs, setQueryLogs] = useState<Array<{ timestamp: string; user_email: string; question: string; status: string; latency_ms: number }>>([]);
  const [logsLoading, setLogsLoading] = useState(false);
  const [logSearchQuery, setLogSearchQuery] = useState('');

  // Documents state
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [docsLoading, setDocsLoading] = useState(true);

  // File Upload state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState<DocumentUploadResponse | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [deleteStatus, setDeleteStatus] = useState<string | null>(null);

  // Settings state
  const [adminSettings, setAdminSettings] = useState<AdminSettings | null>(null);
  const [topKInput, setTopKInput] = useState(3);
  const [savingSettings, setSavingSettings] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState<string | null>(null);

  // Health state
  const [health, setHealth] = useState<HealthResponse | null>(null);

  // Fetch Admin Data
  const loadAdminData = async () => {
    setDocsLoading(true);
    setLogsLoading(true);
    try {
      const [docs, settingsRes, healthRes, usersRes, logsRes] = await Promise.all([
        apiService.getDocuments(),
        apiService.getAdminSettings(),
        apiService.checkHealth(),
        apiService.getAdminUsers(),
        apiService.getAdminLogs(50),
      ]);
      setDocuments(docs);
      setAdminSettings(settingsRes);
      setTopKInput(settingsRes.default_top_k);
      setHealth(healthRes);
      setRegisteredUsers(usersRes.users || []);
      setQueryLogs(logsRes.logs || []);
    } catch {
      // Catch network or authorization error gracefully
    } finally {
      setDocsLoading(false);
      setLogsLoading(false);
    }
  };

  useEffect(() => {
    if (role === 'admin') {
      loadAdminData();
    }
  }, [role]);

  // Handle direct Access Key unlock on /master page
  const handleKeyAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthLoading(true);
    setAuthError(null);

    try {
      await masterLogin(accessKeyInput);
    } catch (err: any) {
      if (err instanceof ApiError) {
        setAuthError(err.detail);
      } else {
        setAuthError('Invalid Master Admin access key.');
      }
    } finally {
      setAuthLoading(false);
    }
  };

  // Upload handler
  const handleUpload = async () => {
    if (!selectedFile) return;

    setUploading(true);
    setUploadError(null);
    setUploadSuccess(null);

    try {
      const res = await apiService.uploadDocument(selectedFile);
      setUploadSuccess(res);
      setSelectedFile(null);
      loadAdminData();
    } catch (err: any) {
      if (err instanceof ApiError) {
        setUploadError(err.detail);
      } else {
        setUploadError(err.message || 'Failed to upload document.');
      }
    } finally {
      setUploading(false);
    }
  };

  // Delete handler
  const handleDelete = async (storedFilename: string) => {
    if (!confirm(`Are you sure you want to delete "${storedFilename}" from vector storage?`)) return;

    setDeleteStatus(null);
    try {
      const res = await apiService.deleteDocument(storedFilename);
      setDeleteStatus(`Successfully deleted "${storedFilename}" (${res.removed_chunks} chunks removed).`);
      loadAdminData();
    } catch (err: any) {
      alert(err instanceof ApiError ? err.detail : 'Failed to delete document');
    }
  };

  // Save Settings handler
  const handleSaveSettings = async () => {
    setSavingSettings(true);
    setSaveSuccess(null);

    try {
      const res = await apiService.updateAdminSettings(topKInput);
      setSaveSuccess(`Top-K configuration updated to ${res.settings.default_top_k}`);
      loadAdminData();
    } catch (err: any) {
      alert(err instanceof ApiError ? err.detail : 'Failed to update Top-K');
    } finally {
      setSavingSettings(false);
    }
  };

  // Filter logs by search query
  const filteredLogs = queryLogs.filter((log) => {
    if (!logSearchQuery.trim()) return true;
    const q = logSearchQuery.toLowerCase();
    return (
      (log.question || '').toLowerCase().includes(q) ||
      (log.user_email || '').toLowerCase().includes(q) ||
      (log.status || '').toLowerCase().includes(q)
    );
  });

  // Protected View Check: If not logged in as Admin
  if (role !== 'admin') {
    return (
      <div className="w-full min-h-[calc(100vh-5rem)] bg-[#0B192C] flex items-center justify-center p-6 text-slate-100">
        <div className="w-full max-w-md space-y-6">
          
          {/* Header Branding */}
          <div className="text-center space-y-2 flex flex-col items-center">
            <div className="p-3 bg-[#0F2537] border border-[#D4AF37] text-[#D4AF37] rounded-xl shadow-xl mb-1">
              <Lock className="h-6 w-6" />
            </div>
            <h1 className="text-xl font-black uppercase tracking-tight text-white font-sans">
              Master Admin Authentication
            </h1>
            <p className="text-xs text-slate-300 max-w-xs">
              Enter master key to access user monitoring logs, vector indexes, and Top-K settings.
            </p>
          </div>

          {authError && (
            <div className="flex items-center gap-2.5 border border-rose-800 bg-[#1e1015] p-3 text-xs text-rose-300 rounded-xl">
              <AlertCircle className="h-4 w-4 text-rose-400 flex-shrink-0" />
              <span>{authError}</span>
            </div>
          )}

          <form onSubmit={handleKeyAuth} className="border border-[#D4AF37]/30 bg-[#0F2537] p-6 space-y-4 shadow-xl rounded-xl">
            <div className="space-y-1.5">
              <label className="text-xs font-bold uppercase tracking-wider text-slate-300">Master Access Key</label>
              <div className="relative">
                <Key className="absolute left-3.5 top-3 h-4 w-4 text-[#D4AF37]" />
                <input
                  type="password"
                  required
                  value={accessKeyInput}
                  onChange={(e) => setAccessKeyInput(e.target.value)}
                  placeholder="Enter master key"
                  className="w-full border border-slate-700 bg-[#0B192C] py-2.5 pl-10 pr-4 text-xs text-white placeholder-slate-500 focus:border-[#D4AF37] focus:outline-none rounded-xl font-mono"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={authLoading || !accessKeyInput.trim()}
              className="w-full flex items-center justify-center gap-2 bg-[#D4AF37] hover:bg-[#C59E2B] py-3 text-xs font-extrabold uppercase tracking-wider text-[#0B192C] transition-all shadow-md disabled:opacity-50 border border-[#D4AF37] rounded-xl"
            >
              {authLoading ? 'Verifying Key...' : 'Unlock Admin Panel'} <ShieldCheck className="h-4 w-4" />
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full min-h-screen bg-[#0B192C] py-12 px-6 lg:px-12 text-slate-100">
      <div className="max-w-6xl mx-auto space-y-8">
        
        {/* Header Banner */}
        <div className="border border-[#D4AF37]/30 bg-[#0F2537] p-8 shadow-xl rounded-xl space-y-6">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-[#D4AF37]/20 pb-4">
            <div>
              <div className="inline-flex items-center gap-2 border border-[#D4AF37] bg-[#0B192C] px-3 py-1 text-[11px] font-extrabold uppercase tracking-widest text-[#D4AF37] mb-2 rounded-xl">
                <Key className="h-3.5 w-3.5" /> MASTER ADMIN CONTROL PANEL
              </div>
              <h1 className="text-3xl font-black uppercase text-white tracking-tight font-sans">
                User Activity Monitoring & System Administration
              </h1>
              <p className="mt-1 text-xs text-slate-300">
                Monitor user accounts, inspect live query execution logs, manage document ingestion, and tune RAG parameters.
              </p>
            </div>
          </div>

          {/* Tab Navigation */}
          <div className="flex flex-wrap items-center gap-3 text-xs font-bold uppercase tracking-wider">
            <button
              onClick={() => setActiveTab('logs')}
              className={`flex items-center gap-2 px-4 py-2.5 transition rounded-xl ${
                activeTab === 'logs'
                  ? 'bg-[#D4AF37] text-[#0B192C] font-extrabold shadow-md'
                  : 'border border-slate-700 bg-[#0B192C] text-slate-300 hover:border-[#D4AF37] hover:text-[#D4AF37]'
              }`}
            >
              <Users className="h-4 w-4" /> Users & Activity Logs
            </button>
            <button
              onClick={() => setActiveTab('documents')}
              className={`flex items-center gap-2 px-4 py-2.5 transition rounded-xl ${
                activeTab === 'documents'
                  ? 'bg-[#D4AF37] text-[#0B192C] font-extrabold shadow-md'
                  : 'border border-slate-700 bg-[#0B192C] text-slate-300 hover:border-[#D4AF37] hover:text-[#D4AF37]'
              }`}
            >
              <FileText className="h-4 w-4" /> Document Ingestion ({documents.length})
            </button>
            <button
              onClick={() => setActiveTab('settings')}
              className={`flex items-center gap-2 px-4 py-2.5 transition rounded-xl ${
                activeTab === 'settings'
                  ? 'bg-[#D4AF37] text-[#0B192C] font-extrabold shadow-md'
                  : 'border border-slate-700 bg-[#0B192C] text-slate-300 hover:border-[#D4AF37] hover:text-[#D4AF37]'
              }`}
            >
              <Sliders className="h-4 w-4" /> Top-K & RAG Settings
            </button>
            <button
              onClick={() => setActiveTab('status')}
              className={`flex items-center gap-2 px-4 py-2.5 transition rounded-xl ${
                activeTab === 'status'
                  ? 'bg-[#D4AF37] text-[#0B192C] font-extrabold shadow-md'
                  : 'border border-slate-700 bg-[#0B192C] text-slate-300 hover:border-[#D4AF37] hover:text-[#D4AF37]'
              }`}
            >
              <Cpu className="h-4 w-4" /> System Status
            </button>
          </div>
        </div>

        {/* TAB 1: USERS & QUERY LOGS MONITORING */}
        {activeTab === 'logs' && (
          <div className="space-y-6">
            
            {/* Quick Metrics */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
              <div className="border border-[#D4AF37]/30 bg-[#0F2537] p-6 space-y-2 rounded-xl shadow-md">
                <div className="flex items-center justify-between text-slate-300">
                  <span className="text-xs font-bold uppercase tracking-wider">Registered Users</span>
                  <UserCheck className="h-4 w-4 text-[#D4AF37]" />
                </div>
                <div className="text-3xl font-black text-white font-sans">
                  {registeredUsers.length} <span className="text-xs text-slate-400 font-normal font-mono">accounts</span>
                </div>
                <p className="text-[11px] text-slate-400">Stored in MongoDB authentication database</p>
              </div>

              <div className="border border-[#D4AF37]/30 bg-[#0F2537] p-6 space-y-2 rounded-xl shadow-md">
                <div className="flex items-center justify-between text-slate-300">
                  <span className="text-xs font-bold uppercase tracking-wider font-sans">Total Queries Executed</span>
                  <Activity className="h-4 w-4 text-[#D4AF37]" />
                </div>
                <div className="text-3xl font-black text-white font-sans">
                  {queryLogs.length} <span className="text-xs text-slate-400 font-normal font-mono">logged</span>
                </div>
                <p className="text-[11px] text-slate-400">Real-time user search & RAG activity</p>
              </div>

              <div className="border border-[#D4AF37]/30 bg-[#0F2537] p-6 space-y-2 rounded-xl shadow-md">
                <div className="flex items-center justify-between text-slate-300">
                  <span className="text-xs font-bold uppercase tracking-wider font-sans">Active Top-K Depth</span>
                  <Sliders className="h-4 w-4 text-[#D4AF37]" />
                </div>
                <div className="text-3xl font-black text-white font-sans">
                  {adminSettings?.default_top_k || topKInput} <span className="text-xs text-slate-400 font-normal font-mono">chunks</span>
                </div>
                <p className="text-[11px] text-slate-400">Retrieval candidate context depth</p>
              </div>
            </div>

            {/* Registered Users Table */}
            <div className="border border-[#D4AF37]/30 bg-[#0F2537] p-6 space-y-4 rounded-xl shadow-md">
              <div className="flex items-center justify-between border-b border-[#D4AF37]/20 pb-3">
                <h3 className="font-extrabold text-white uppercase text-xs tracking-wider flex items-center gap-2 font-sans">
                  <Users className="h-4 w-4 text-[#D4AF37]" />
                  Registered User Accounts ({registeredUsers.length})
                </h3>
                <button onClick={loadAdminData} className="text-xs text-slate-300 hover:text-[#D4AF37] flex items-center gap-1 font-bold uppercase">
                  <RefreshCw className="h-3.5 w-3.5" /> Refresh
                </button>
              </div>

              {registeredUsers.length === 0 ? (
                <p className="text-xs text-slate-400 italic py-4 text-center">No user accounts registered yet.</p>
              ) : (
                <div className="divide-y divide-slate-800">
                  {registeredUsers.map((u, idx) => (
                    <div key={idx} className="flex flex-col sm:flex-row sm:items-center justify-between py-3.5 gap-3 text-xs hover:bg-[#0B192C]/50 px-3 rounded-xl transition">
                      <div className="flex items-center gap-3">
                        <div className="h-9 w-9 rounded-xl border border-[#D4AF37]/40 bg-[#0B192C] flex items-center justify-center font-bold text-[#D4AF37] uppercase text-xs shadow-inner">
                          {u.full_name ? u.full_name[0] : u.email[0]}
                        </div>
                        <div>
                          <div className="flex items-center gap-2">
                            <p className="font-bold text-white font-mono">{u.email}</p>
                            <span className="px-2 py-0.5 bg-[#0B192C] border border-[#D4AF37]/30 text-[#D4AF37] font-bold text-[9px] uppercase rounded-lg">
                              {u.role}
                            </span>
                          </div>
                          <p className="text-[11px] text-slate-400">{u.full_name || 'Registered User'}</p>
                        </div>
                      </div>

                      <div className="flex items-center gap-4">
                        <div className="text-right hidden sm:block">
                          <p className="text-[11px] font-bold text-slate-200">
                            {u.total_queries ?? 0} <span className="text-[10px] text-slate-400 font-normal">queries</span>
                          </p>
                          <p className="text-[10px] text-[#D4AF37] font-mono">
                            {(u.total_tokens ?? 0).toLocaleString()} tokens spent
                          </p>
                        </div>

                        <Link
                          href={`/master/${encodeURIComponent(u.email)}`}
                          className="flex items-center gap-1.5 px-3 py-1.5 bg-[#0B192C] border border-[#D4AF37]/40 hover:bg-[#D4AF37] text-[#D4AF37] hover:text-[#0B192C] font-bold text-xs rounded-xl transition shadow-sm"
                        >
                          <BarChart3 className="h-3.5 w-3.5" />
                          <span>Monitor Details</span>
                          <ChevronRight className="h-3.5 w-3.5" />
                        </Link>
                      </div>
                    </div>
                  ))}
                </div>

              )}
            </div>

            {/* User Query Activity Logs Table */}
            <div className="border border-[#D4AF37]/30 bg-[#0F2537] p-6 space-y-4 rounded-xl shadow-md">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[#D4AF37]/20 pb-3">
                <h3 className="font-extrabold text-white uppercase text-xs tracking-wider flex items-center gap-2 font-sans">
                  <Activity className="h-4 w-4 text-[#D4AF37]" />
                  User Query Execution Logs ({filteredLogs.length})
                </h3>

                {/* Filter Search Input */}
                <div className="relative max-w-xs w-full">
                  <Search className="absolute left-3 top-2.5 h-3.5 w-3.5 text-slate-400" />
                  <input
                    type="text"
                    value={logSearchQuery}
                    onChange={(e) => setLogSearchQuery(e.target.value)}
                    placeholder="Search queries or status..."
                    className="w-full bg-[#0B192C] border border-slate-700 rounded-xl py-1.5 pl-9 pr-3 text-xs text-white placeholder-slate-500 focus:border-[#D4AF37] focus:outline-none"
                  />
                </div>
              </div>

              {logsLoading ? (
                <p className="text-xs text-slate-400 font-mono py-6 text-center">Loading activity logs...</p>
              ) : filteredLogs.length === 0 ? (
                <p className="text-xs text-slate-400 italic py-6 text-center">No user query logs recorded yet.</p>
              ) : (
                <div className="divide-y divide-slate-800">
                  {filteredLogs.map((log, idx) => (
                    <div key={idx} className="py-3 space-y-1 text-xs">
                      <div className="flex flex-wrap items-center justify-between gap-2 font-mono">
                        <span className="text-slate-400 text-[11px] flex items-center gap-1">
                          <Clock className="h-3 w-3 text-[#D4AF37]" /> {log.timestamp ? new Date(log.timestamp).toLocaleString() : 'Recent'}
                        </span>
                        
                        <span className={`px-2.5 py-0.5 text-[10px] font-bold uppercase rounded-xl border ${
                          log.status === 'SUPPORTED'
                            ? 'bg-emerald-950 border-emerald-700 text-emerald-400'
                            : log.status === 'OUT_OF_SCOPE'
                            ? 'bg-amber-950 border-amber-700 text-amber-400'
                            : 'bg-rose-950 border-rose-700 text-rose-400'
                        }`}>
                          {log.status}
                        </span>
                      </div>

                      <p className="text-slate-100 font-medium font-sans">&ldquo;{log.question}&rdquo;</p>
                      
                      <div className="flex items-center gap-4 text-[11px] text-slate-400 font-mono">
                        <span>User: <strong className="text-slate-300">{log.user_email || 'Registered User'}</strong></span>
                        {log.latency_ms > 0 && (
                          <span>Latency: <strong className="text-[#D4AF37]">{log.latency_ms} ms</strong></span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

          </div>
        )}

        {/* TAB 2: DOCUMENTS MANAGEMENT */}
        {activeTab === 'documents' && (
          <div className="space-y-6">
            {/* Upload Area */}
            <div className="border border-dashed border-[#D4AF37]/40 bg-[#0F2537] p-8 text-center space-y-4 rounded-xl shadow-md">
              <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-[#0B192C] border border-[#D4AF37] text-[#D4AF37] shadow-md">
                <UploadCloud className="h-6 w-6" />
              </div>
              <div>
                <h3 className="font-bold text-white uppercase text-sm font-sans">Upload New Document to Corpus</h3>
                <p className="text-xs text-slate-300 mt-1">
                  Supported formats: <strong className="text-white">.txt, .md, .pdf</strong> (Max file size: 10MB)
                </p>
              </div>

              <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
                <label className="cursor-pointer rounded-xl border border-slate-700 bg-[#0B192C] px-4 py-2.5 text-xs font-bold text-slate-200 hover:border-[#D4AF37] transition uppercase">
                  Browse File
                  <input
                    type="file"
                    accept=".txt,.md,.pdf"
                    onChange={(e) => {
                      if (e.target.files && e.target.files[0]) {
                        setSelectedFile(e.target.files[0]);
                        setUploadError(null);
                        setUploadSuccess(null);
                      }
                    }}
                    className="hidden"
                  />
                </label>

                {selectedFile && (
                  <span className="text-xs text-[#D4AF37] font-mono bg-[#0B192C] px-3 py-2 rounded-xl border border-[#D4AF37]/40">
                    {selectedFile.name} ({formatBytes(selectedFile.size)})
                  </span>
                )}

                <button
                  onClick={handleUpload}
                  disabled={!selectedFile || uploading}
                  className="flex items-center gap-2 rounded-xl bg-[#D4AF37] hover:bg-[#C59E2B] px-5 py-2.5 text-xs font-extrabold uppercase tracking-wider text-[#0B192C] disabled:opacity-50 shadow-md border border-[#D4AF37]"
                >
                  {uploading ? 'Processing & Indexing...' : 'Upload & Index'}
                </button>
              </div>

              {uploadSuccess && (
                <div className="p-4 rounded-xl bg-emerald-950/60 border border-emerald-700 text-xs text-emerald-300 text-left font-mono space-y-1">
                  <p className="font-bold text-emerald-200 flex items-center gap-1.5 uppercase">
                    <CheckCircle2 className="h-4 w-4" /> Document Indexed Successfully
                  </p>
                  <p>Path: {uploadSuccess.summary.document}</p>
                  <p>Chunks: {uploadSuccess.summary.chunks} | Embedded: {uploadSuccess.summary.indexed}</p>
                </div>
              )}

              {uploadError && (
                <div className="p-3 rounded-xl bg-rose-950/60 border border-rose-800 text-xs text-rose-300 text-left">
                  {uploadError}
                </div>
              )}
            </div>

            {/* Delete Status Alert */}
            {deleteStatus && (
              <div className="p-3 rounded-xl bg-[#0F2537] border border-[#D4AF37]/50 text-xs text-[#D4AF37] font-mono">
                {deleteStatus}
              </div>
            )}

            {/* Documents Table */}
            <div className="border border-[#D4AF37]/30 bg-[#0F2537] p-6 space-y-4 rounded-xl shadow-md">
              <div className="flex items-center justify-between border-b border-[#D4AF37]/20 pb-3">
                <h3 className="font-extrabold text-white uppercase text-xs tracking-wider flex items-center gap-2 font-sans">
                  <HardDrive className="h-4 w-4 text-[#D4AF37]" />
                  Manage Stored Documents ({documents.length})
                </h3>
                <button onClick={loadAdminData} className="text-xs text-slate-300 hover:text-[#D4AF37] flex items-center gap-1 font-bold uppercase">
                  <RefreshCw className="h-3.5 w-3.5" /> Refresh List
                </button>
              </div>

              {docsLoading ? (
                <div className="py-6 text-center text-xs text-slate-400 font-mono">Loading corpus documents...</div>
              ) : documents.length === 0 ? (
                <div className="py-6 text-center text-xs text-slate-400">No documents found in uploads corpus directory.</div>
              ) : (
                <div className="divide-y divide-slate-800">
                  {documents.map((doc, idx) => (
                    <div key={idx} className="flex items-center justify-between py-3 text-xs">
                      <div className="flex items-center gap-3">
                        <FileText className="h-4 w-4 text-[#D4AF37] flex-shrink-0" />
                        <div>
                          <h5 className="font-bold text-slate-100">{doc.filename}</h5>
                          <span className="text-[11px] text-slate-400 font-mono">{doc.stored_filename}</span>
                        </div>
                      </div>

                      <div className="flex items-center gap-4">
                        <span className="rounded-xl border border-slate-700 bg-[#0B192C] px-2.5 py-0.5 text-[10px] font-mono text-slate-300 uppercase">
                          .{doc.document_type}
                        </span>
                        <span className="text-slate-400 font-mono">{formatBytes(doc.size_bytes)}</span>
                        <button
                          onClick={() => handleDelete(doc.stored_filename)}
                          className="flex items-center gap-1 rounded-xl bg-rose-950/80 border border-rose-800 px-3 py-1 text-[11px] font-bold text-rose-300 hover:bg-rose-900 transition uppercase"
                        >
                          <Trash2 className="h-3.5 w-3.5" /> Delete
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 3: RAG SETTINGS (TOP-K) */}
        {activeTab === 'settings' && (
          <div className="border border-[#D4AF37]/30 bg-[#0F2537] p-6 space-y-6 rounded-xl shadow-md">
            <div className="border-b border-[#D4AF37]/20 pb-3">
              <h3 className="font-extrabold text-white uppercase text-sm tracking-wider flex items-center gap-2 font-sans">
                <Sliders className="h-4 w-4 text-[#D4AF37]" />
                Configure RAG Retrieval Top-K Parameter
              </h3>
              <p className="text-xs text-slate-300 mt-1">
                Adjust the default number of Top-K context document chunks retrieved for user queries.
              </p>
            </div>

            {saveSuccess && (
              <div className="p-3 rounded-xl bg-emerald-950/60 border border-emerald-700 text-xs text-emerald-300 flex items-center gap-2 font-mono">
                <CheckCircle2 className="h-4 w-4 text-emerald-400" /> {saveSuccess}
              </div>
            )}

            <div className="rounded-xl border border-slate-700 bg-[#0B192C] p-6 space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-200">Default Top-K Chunks</span>
                <span className="text-lg font-black text-[#D4AF37] font-mono">{topKInput}</span>
              </div>

              <input
                type="range"
                min={1}
                max={10}
                value={topKInput}
                onChange={(e) => setTopKInput(Number(e.target.value))}
                className="w-full accent-[#D4AF37] cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                <span>1 (Fastest / Concise)</span>
                <span>5 (Standard)</span>
                <span>10 (Maximum Context)</span>
              </div>

              <button
                onClick={handleSaveSettings}
                disabled={savingSettings}
                className="flex items-center gap-2 rounded-xl bg-[#D4AF37] hover:bg-[#C59E2B] px-5 py-2.5 text-xs font-extrabold uppercase tracking-wider text-[#0B192C] transition-all shadow-md border border-[#D4AF37] disabled:opacity-50"
              >
                <Save className="h-4 w-4" /> Save Top-K Parameter
              </button>
            </div>
          </div>
        )}

        {/* TAB 4: SYSTEM STATUS */}
        {activeTab === 'status' && (
          <div className="border border-[#D4AF37]/30 bg-[#0F2537] p-6 space-y-6 rounded-xl shadow-md">
            <div className="border-b border-[#D4AF37]/20 pb-3">
              <h3 className="font-extrabold text-white uppercase text-sm tracking-wider flex items-center gap-2 font-sans">
                <Cpu className="h-4 w-4 text-[#D4AF37]" />
                Backend Operational Status & Environment
              </h3>
            </div>

            {health ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
                <div className="p-4 rounded-xl border border-slate-700 bg-[#0B192C] space-y-2">
                  <span className="text-[#D4AF37] text-[10px] font-sans font-extrabold uppercase tracking-wider">API Service</span>
                  <div className="text-slate-200">Name: <strong className="text-white">{health.service}</strong></div>
                  <div className="text-slate-200">Status: <strong className="text-emerald-400 font-bold">{health.status}</strong></div>
                </div>

                <div className="p-4 rounded-xl border border-slate-700 bg-[#0B192C] space-y-2">
                  <span className="text-[#D4AF37] text-[10px] font-sans font-extrabold uppercase tracking-wider">Vector Store</span>
                  <div className="text-slate-200">Path: <strong className="text-slate-300">{health.environment.vector_db_path}</strong></div>
                  <div className="text-slate-200">Collection: <strong className="text-slate-300">{health.environment.collection_name}</strong></div>
                </div>
              </div>
            ) : (
              <div className="text-xs text-rose-400 font-mono">Backend status check failed.</div>
            )}
          </div>
        )}

      </div>
    </div>
  );
}
