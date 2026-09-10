'use client';

import React, { useEffect, useState } from 'react';
import { Settings, Cpu, HardDrive, Shield, Server, CheckCircle2, XCircle, RefreshCw } from 'lucide-react';
import { apiService } from '@/services/api';
import { HealthResponse } from '@/types/api';
import { API_BASE_URL } from '@/config/env';

export default function SettingsPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchHealth = async () => {
    setLoading(true);
    try {
      const res = await apiService.checkHealth();
      setHealth(res);
    } catch {
      setHealth(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
  }, []);

  return (
    <div className="max-w-4xl space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-2">
          <Settings className="h-6 w-6 text-cyan-400" />
          System Settings & RAG Pipeline Inspector
        </h1>
        <p className="mt-1 text-sm text-slate-400">
          Inspect backend configuration parameters, model bindings, vector database paths, and API endpoints.
        </p>
      </div>

      {/* Backend Status Box */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6 space-y-6">
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div className="flex items-center gap-3">
            <Server className="h-5 w-5 text-cyan-400" />
            <div>
              <h3 className="font-semibold text-white">FastAPI Backend Connection</h3>
              <p className="text-xs text-slate-400 font-mono">{API_BASE_URL}</p>
            </div>
          </div>

          <button
            onClick={fetchHealth}
            disabled={loading}
            className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs text-slate-200 hover:bg-slate-700 transition"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
            Re-test API
          </button>
        </div>

        {health ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
            <div className="rounded-xl border border-slate-800 bg-slate-950 p-4 space-y-2">
              <span className="text-slate-500 uppercase text-[10px] font-sans font-semibold">Service Info</span>
              <div className="text-slate-200">Name: <strong className="text-cyan-300">{health.service}</strong></div>
              <div className="text-slate-200">State: <strong className="text-emerald-400">{health.status}</strong></div>
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-950 p-4 space-y-2">
              <span className="text-slate-500 uppercase text-[10px] font-sans font-semibold">Max File Upload</span>
              <div className="text-slate-200">Limit: <strong className="text-amber-300">{health.environment.max_upload_size_mb} MB</strong></div>
              <div className="text-slate-200">Formats: <strong className="text-slate-400">.txt, .md, .pdf</strong></div>
            </div>
          </div>
        ) : (
          <div className="flex items-center gap-2 text-xs text-rose-400 p-4 rounded-xl bg-rose-950/20 border border-rose-900/40">
            <XCircle className="h-4 w-4" />
            FastAPI Server Offline. Start server using `cd server && python main.py`.
          </div>
        )}
      </div>

      {/* Model & Storage Configurations */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6 space-y-6">
        <h3 className="font-semibold text-white flex items-center gap-2">
          <Cpu className="h-5 w-5 text-cyan-400" />
          Active Model Configurations
        </h3>

        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 rounded-xl border border-slate-800 bg-slate-950 text-xs">
            <div>
              <h4 className="font-semibold text-slate-200">Embedding Model</h4>
              <p className="text-slate-400">OpenAI compatible embeddings endpoint</p>
            </div>
            <span className="rounded-md bg-cyan-950 border border-cyan-800 px-3 py-1 font-mono text-cyan-300">
              text-embedding-3-small
            </span>
          </div>

          <div className="flex items-center justify-between p-4 rounded-xl border border-slate-800 bg-slate-950 text-xs">
            <div>
              <h4 className="font-semibold text-slate-200">LLM Chat Completion</h4>
              <p className="text-slate-400">Groq high-speed completion model</p>
            </div>
            <span className="rounded-md bg-amber-950 border border-amber-800 px-3 py-1 font-mono text-amber-300">
              groq/compound-mini
            </span>
          </div>

          <div className="flex items-center justify-between p-4 rounded-xl border border-slate-800 bg-slate-950 text-xs">
            <div>
              <h4 className="font-semibold text-slate-200">Vector Collection Name</h4>
              <p className="text-slate-400">Chroma / Indexing Collection</p>
            </div>
            <span className="rounded-md bg-slate-900 border border-slate-700 px-3 py-1 font-mono text-slate-300">
              rag_chunks
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
