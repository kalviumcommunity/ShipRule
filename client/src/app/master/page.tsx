'use client';

import React, { useEffect, useState } from 'react';
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
  FileType,
  Activity,
  Cpu,
  Layers,
  Save,
  Lock
} from 'lucide-react';
import { useAuth } from '@/context/AuthContext';
import { apiService, ApiError } from '@/services/api';
import { DocumentItem, DocumentUploadResponse, AdminSettings, HealthResponse } from '@/types/api';
import { formatBytes } from '@/lib/utils';

export default function MasterAdminPage() {
  const { role, masterLogin } = useAuth();
  const [accessKeyInput, setAccessKeyInput] = useState('');
  const [authError, setAuthError] = useState<string | null>(null);
  const [authLoading, setAuthLoading] = useState(false);

  // Active Admin Tab
  const [activeTab, setActiveTab] = useState<'overview' | 'documents' | 'settings' | 'status'>('overview');

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
    try {
      const [docs, settingsRes, healthRes] = await Promise.all([
        apiService.getDocuments(),
        apiService.getAdminSettings(),
        apiService.checkHealth(),
      ]);
      setDocuments(docs);
      setAdminSettings(settingsRes);
      setTopKInput(settingsRes.default_top_k);
      setHealth(healthRes);
    } catch (err: any) {
      // If error occurs
    } finally {
      setDocsLoading(false);
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
        setAuthError('Invalid Master Admin access key');
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

  // Protected View Check: If not logged in as Admin
  if (role !== 'admin') {
    return (
      <div className="flex min-h-[calc(100vh-8rem)] items-center justify-center p-4">
        <div className="w-full max-w-md space-y-6">
          <div className="text-center space-y-2">
            <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-amber-950/80 border border-amber-800/80 text-amber-400 shadow-xl shadow-amber-500/10">
              <Lock className="h-7 w-7" />
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-white">
              Protected Master Admin Panel
            </h1>
            <p className="text-xs text-slate-400">
              Access to <code className="text-amber-400 font-mono">/master</code> requires Master Admin Access Key authentication.
            </p>
          </div>

          {authError && (
            <div className="flex items-center gap-2.5 rounded-xl border border-rose-900/50 bg-rose-950/30 p-3 text-xs text-rose-300">
              <AlertCircle className="h-4 w-4 text-rose-400 flex-shrink-0" />
              <span>{authError}</span>
            </div>
          )}

          <form onSubmit={handleKeyAuth} className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6 space-y-4 shadow-xl">
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-slate-300">Enter Admin Access Key</label>
              <div className="relative">
                <Key className="absolute left-3.5 top-3 h-4 w-4 text-amber-400" />
                <input
                  type="password"
                  required
                  value={accessKeyInput}
                  onChange={(e) => setAccessKeyInput(e.target.value)}
                  placeholder="Enter access key"
                  className="w-full border border-slate-700 bg-[#0B192C] py-2.5 pl-10 pr-4 text-xs text-white placeholder-slate-500 focus:border-[#D4AF37] focus:outline-none font-mono"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={authLoading || !accessKeyInput.trim()}
              className="w-full flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-amber-500 to-orange-600 py-2.5 text-xs font-semibold text-white hover:from-amber-400 hover:to-orange-500 transition shadow-lg shadow-amber-500/20 disabled:opacity-50"
            >
              {authLoading ? 'Verifying Key...' : 'Authenticate Admin Session'} <ShieldCheck className="h-4 w-4" />
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl space-y-8">
      {/* Header Banner */}
      <div className="rounded-2xl border border-amber-500/30 bg-gradient-to-r from-amber-950/30 via-slate-900 to-slate-950 p-6 shadow-xl">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-amber-500/40 bg-amber-500/10 px-3 py-1 text-xs font-bold text-amber-400 mb-2">
              <Key className="h-3.5 w-3.5" /> MASTER ADMIN CONTROL PANEL
            </div>
            <h1 className="text-2xl font-extrabold text-white">
              ShipRule Administration & RAG Configuration
            </h1>
            <p className="text-xs text-slate-400 mt-1">
              Upload corpus documents, delete files, configure Top-K retrieval parameters, and inspect vector indexes.
            </p>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="mt-6 flex flex-wrap items-center gap-2 border-t border-slate-800/80 pt-4 text-xs font-medium">
          <button
            onClick={() => setActiveTab('overview')}
            className={`flex items-center gap-2 rounded-lg px-4 py-2 transition ${
              activeTab === 'overview'
                ? 'bg-amber-500/20 text-amber-400 border border-amber-500/40'
                : 'text-slate-400 hover:bg-slate-800 hover:text-white'
            }`}
          >
            <Activity className="h-4 w-4" /> Overview & Metrics
          </button>
          <button
            onClick={() => setActiveTab('documents')}
            className={`flex items-center gap-2 rounded-lg px-4 py-2 transition ${
              activeTab === 'documents'
                ? 'bg-amber-500/20 text-amber-400 border border-amber-500/40'
                : 'text-slate-400 hover:bg-slate-800 hover:text-white'
            }`}
          >
            <FileText className="h-4 w-4" /> Document Ingestion ({documents.length})
          </button>
          <button
            onClick={() => setActiveTab('settings')}
            className={`flex items-center gap-2 rounded-lg px-4 py-2 transition ${
              activeTab === 'settings'
                ? 'bg-amber-500/20 text-amber-400 border border-amber-500/40'
                : 'text-slate-400 hover:bg-slate-800 hover:text-white'
            }`}
          >
            <Sliders className="h-4 w-4" /> Top-K & RAG Settings
          </button>
          <button
            onClick={() => setActiveTab('status')}
            className={`flex items-center gap-2 rounded-lg px-4 py-2 transition ${
              activeTab === 'status'
                ? 'bg-amber-500/20 text-amber-400 border border-amber-500/40'
                : 'text-slate-400 hover:bg-slate-800 hover:text-white'
            }`}
          >
            <Cpu className="h-4 w-4" /> System Status
          </button>
        </div>
      </div>

      {/* TAB 1: OVERVIEW */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5 space-y-2">
              <div className="flex items-center justify-between text-slate-400">
                <span className="text-xs font-medium">Configured Top-K</span>
                <Sliders className="h-4 w-4 text-amber-400" />
              </div>
              <div className="text-3xl font-bold text-white">
                {adminSettings?.default_top_k || topKInput} <span className="text-xs text-slate-400 font-normal">chunks</span>
              </div>
              <p className="text-[11px] text-slate-500">Default chunks retrieved for queries</p>
            </div>

            <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5 space-y-2">
              <div className="flex items-center justify-between text-slate-400">
                <span className="text-xs font-medium">Corpus Documents</span>
                <FileText className="h-4 w-4 text-cyan-400" />
              </div>
              <div className="text-3xl font-bold text-white">
                {documents.length} <span className="text-xs text-slate-400 font-normal">files</span>
              </div>
              <p className="text-[11px] text-slate-500">Stored in server/uploads/</p>
            </div>

            <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5 space-y-2">
              <div className="flex items-center justify-between text-slate-400">
                <span className="text-xs font-medium">Max Upload Limit</span>
                <HardDrive className="h-4 w-4 text-blue-400" />
              </div>
              <div className="text-3xl font-bold text-white">
                {adminSettings?.max_upload_size_mb || 10} <span className="text-xs text-slate-400 font-normal">MB</span>
              </div>
              <p className="text-[11px] text-slate-500">Supported: .txt, .md, .pdf</p>
            </div>
          </div>

          <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6 space-y-4">
            <h3 className="font-semibold text-white flex items-center gap-2">
              <Layers className="h-4 w-4 text-amber-400" /> Master Admin Capabilities
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-300">
              <div className="p-4 rounded-xl border border-slate-800 bg-slate-950 space-y-1">
                <h4 className="font-semibold text-cyan-300">1. Document Ingestion & Deletion</h4>
                <p className="text-slate-400 leading-relaxed">
                  Upload new maritime regulations or delete obsolete policy documents directly from vector store index.
                </p>
              </div>
              <div className="p-4 rounded-xl border border-slate-800 bg-slate-950 space-y-1">
                <h4 className="font-semibold text-amber-300">2. Top-K Retrieval Tuning</h4>
                <p className="text-slate-400 leading-relaxed">
                  Tune the default candidate context depth (Top-K) passed to context assembler and LLM generator.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: DOCUMENTS MANAGEMENT */}
      {activeTab === 'documents' && (
        <div className="space-y-6">
          {/* Upload Area */}
          <div className="rounded-2xl border border-dashed border-slate-700 bg-slate-900/50 p-6 text-center space-y-4">
            <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-amber-950/60 border border-amber-800 text-amber-400 shadow-md">
              <UploadCloud className="h-6 w-6" />
            </div>
            <div>
              <h3 className="font-semibold text-white">Upload New Document to Corpus</h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Supported file types: <strong className="text-slate-300">.txt, .md, .pdf</strong> (Max limit: 10MB)
              </p>
            </div>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
              <label className="cursor-pointer rounded-xl border border-slate-700 bg-slate-800 px-4 py-2 text-xs font-semibold text-slate-200 hover:bg-slate-700">
                Browse File
                <input type="file" accept=".txt,.md,.pdf" onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    setSelectedFile(e.target.files[0]);
                    setUploadError(null);
                    setUploadSuccess(null);
                  }
                }} className="hidden" />
              </label>

              {selectedFile && (
                <span className="text-xs text-cyan-300 font-mono bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800">
                  {selectedFile.name} ({formatBytes(selectedFile.size)})
                </span>
              )}

              <button
                onClick={handleUpload}
                disabled={!selectedFile || uploading}
                className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-amber-500 to-orange-600 px-4 py-2 text-xs font-semibold text-white hover:from-amber-400 hover:to-orange-500 disabled:opacity-50 shadow-md"
              >
                {uploading ? 'Processing...' : 'Upload & Index'}
              </button>
            </div>

            {uploadSuccess && (
              <div className="p-3 rounded-xl bg-emerald-950/40 border border-emerald-800 text-xs text-emerald-300 text-left font-mono space-y-1">
                <p className="font-bold text-emerald-200 flex items-center gap-1.5">
                  <CheckCircle2 className="h-4 w-4" /> Document Indexed Successfully
                </p>
                <p>Path: {uploadSuccess.summary.document}</p>
                <p>Chunks: {uploadSuccess.summary.chunks} | Embedded: {uploadSuccess.summary.indexed}</p>
              </div>
            )}

            {uploadError && (
              <div className="p-3 rounded-xl bg-rose-950/40 border border-rose-900 text-xs text-rose-300 text-left">
                {uploadError}
              </div>
            )}
          </div>

          {/* Delete Status Alert */}
          {deleteStatus && (
            <div className="p-3 rounded-xl bg-slate-900 border border-slate-700 text-xs text-cyan-300">
              {deleteStatus}
            </div>
          )}

          {/* Documents Table */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-white flex items-center gap-2">
                <HardDrive className="h-4 w-4 text-amber-400" />
                Manage Stored Documents ({documents.length})
              </h3>
              <button onClick={loadAdminData} className="text-xs text-slate-400 hover:text-white flex items-center gap-1">
                <RefreshCw className="h-3.5 w-3.5" /> Refresh
              </button>
            </div>

            {docsLoading ? (
              <div className="py-6 text-center text-xs text-slate-500">Loading documents...</div>
            ) : documents.length === 0 ? (
              <div className="py-6 text-center text-xs text-slate-500">No documents found in corpus uploads/ folder.</div>
            ) : (
              <div className="divide-y divide-slate-800">
                {documents.map((doc, idx) => (
                  <div key={idx} className="flex items-center justify-between py-3 text-xs">
                    <div className="flex items-center gap-3">
                      <FileText className="h-4 w-4 text-cyan-400 flex-shrink-0" />
                      <div>
                        <h5 className="font-medium text-slate-200">{doc.filename}</h5>
                        <span className="text-[11px] text-slate-500 font-mono">{doc.stored_filename}</span>
                      </div>
                    </div>

                    <div className="flex items-center gap-4">
                      <span className="rounded bg-slate-800 px-2 py-0.5 text-[10px] font-mono text-slate-300 uppercase">
                        .{doc.document_type}
                      </span>
                      <span className="text-slate-400">{formatBytes(doc.size_bytes)}</span>
                      <button
                        onClick={() => handleDelete(doc.stored_filename)}
                        className="flex items-center gap-1 rounded bg-rose-950/60 border border-rose-800/80 px-2.5 py-1 text-[11px] font-medium text-rose-300 hover:bg-rose-900 transition"
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
        <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6 space-y-6">
          <div>
            <h3 className="font-semibold text-white flex items-center gap-2">
              <Sliders className="h-5 w-5 text-amber-400" />
              Configure RAG Retrieval Top-K Parameter
            </h3>
            <p className="text-xs text-slate-400 mt-1">
              Adjust the default number of Top-K context chunks retrieved for user questions.
            </p>
          </div>

          {saveSuccess && (
            <div className="p-3 rounded-xl bg-emerald-950/40 border border-emerald-800 text-xs text-emerald-300 flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-emerald-400" /> {saveSuccess}
            </div>
          )}

          <div className="rounded-xl border border-slate-800 bg-slate-950 p-6 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-200">Default Top-K Chunks</span>
              <span className="text-base font-bold text-amber-400 font-mono">{topKInput}</span>
            </div>

            <input
              type="range"
              min={1}
              max={10}
              value={topKInput}
              onChange={(e) => setTopKInput(Number(e.target.value))}
              className="w-full accent-amber-500 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500">
              <span>1 (Fastest / Concise)</span>
              <span>5 (Standard)</span>
              <span>10 (Maximum Context)</span>
            </div>

            <button
              onClick={handleSaveSettings}
              disabled={savingSettings}
              className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-amber-500 to-orange-600 px-5 py-2 text-xs font-semibold text-white hover:from-amber-400 hover:to-orange-500 transition shadow-md shadow-amber-500/20 disabled:opacity-50"
            >
              <Save className="h-4 w-4" /> Save Top-K Parameter
            </button>
          </div>
        </div>
      )}

      {/* TAB 4: SYSTEM STATUS */}
      {activeTab === 'status' && (
        <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6 space-y-6">
          <h3 className="font-semibold text-white flex items-center gap-2">
            <Cpu className="h-5 w-5 text-cyan-400" />
            Backend Operational Status & Environment
          </h3>

          {health ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
              <div className="p-4 rounded-xl border border-slate-800 bg-slate-950 space-y-2">
                <span className="text-slate-500 text-[10px] font-sans font-semibold uppercase">API Service</span>
                <div className="text-slate-200">Name: <strong className="text-cyan-300">{health.service}</strong></div>
                <div className="text-slate-200">Status: <strong className="text-emerald-400">{health.status}</strong></div>
              </div>

              <div className="p-4 rounded-xl border border-slate-800 bg-slate-950 space-y-2">
                <span className="text-slate-500 text-[10px] font-sans font-semibold uppercase">Vector Store</span>
                <div className="text-slate-200">Path: <strong className="text-amber-300">{health.environment.vector_db_path}</strong></div>
                <div className="text-slate-200">Collection: <strong className="text-slate-300">{health.environment.collection_name}</strong></div>
              </div>
            </div>
          ) : (
            <div className="text-xs text-rose-400">Backend status check failed.</div>
          )}
        </div>
      )}
    </div>
  );
}
