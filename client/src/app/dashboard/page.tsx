'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import {
  Anchor,
  FileText,
  MessageSquare,
  Activity,
  ShieldCheck,
  Zap,
  ArrowRight,
  Database,
  Cpu,
  Layers,
  Ship
} from 'lucide-react';
import { apiService } from '@/services/api';
import { HealthResponse, DocumentItem } from '@/types/api';

export default function DashboardPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const [h, docs] = await Promise.all([
          apiService.checkHealth(),
          apiService.getDocuments(),
        ]);
        setHealth(h);
        setDocuments(docs);
      } catch {
        setHealth(null);
        setDocuments([]);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  return (
    <div className="w-full bg-[#0B192C] min-h-screen py-12 px-6 lg:px-12 text-slate-100 space-y-8">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Welcome Banner */}
        <div className="border border-[#D4AF37]/30 bg-[#0F2537] p-8 shadow-xl space-y-4">
          <div className="inline-flex items-center gap-2 border border-[#D4AF37] bg-[#0B192C] px-3 py-1 text-xs font-black uppercase tracking-widest text-[#D4AF37]">
            <Ship className="h-4 w-4 text-[#D4AF37]" /> GFS | ShipRule CDLP Engine v1.0.0
          </div>
          <h1 className="text-3xl font-black tracking-tight text-white uppercase sm:text-4xl font-sans">
            Customs Duty & Shipping Documentation Platform
          </h1>
          <p className="text-sm text-slate-300 leading-relaxed max-w-3xl">
            Enterprise RAG engine delivering anti-hallucination grounded answers, cross-encoder re-ranking, and citation verification for international customs clearance.
          </p>

          <div className="pt-2 flex flex-wrap items-center gap-4">
            <Link
              href="/query"
              className="bg-[#D4AF37] hover:bg-[#C59E2B] text-[#0B192C] px-6 py-3 text-xs font-black uppercase tracking-wider transition-all border border-[#D4AF37] shadow-lg flex items-center gap-2"
            >
              <MessageSquare className="h-4 w-4" /> Ask ShipRule Now <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              href="/settings"
              className="border border-[#D4AF37]/50 bg-[#0B192C] hover:bg-[#162A45] text-white px-6 py-3 text-xs font-bold uppercase tracking-wider transition-all flex items-center gap-2"
            >
              <Layers className="h-4 w-4 text-[#D4AF37]" /> System Metrics
            </Link>
          </div>
        </div>

        {/* Metric Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="border border-slate-700 bg-[#0F2537] p-5 space-y-2">
            <div className="flex items-center justify-between text-slate-400">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-300">Service Status</span>
              <Activity className="h-4 w-4 text-[#D4AF37]" />
            </div>
            <div className="text-xl font-black text-white">
              {loading ? (
                <span className="text-[#D4AF37] text-sm font-mono">Checking...</span>
              ) : health?.status === 'healthy' ? (
                <span className="text-emerald-400 flex items-center gap-1.5 text-lg font-mono">
                  <ShieldCheck className="h-5 w-5 text-emerald-400" /> Operational
                </span>
              ) : (
                <span className="text-rose-400 text-lg font-mono">Offline</span>
              )}
            </div>
            <p className="text-[11px] text-slate-400 font-mono">FastAPI http://localhost:8000</p>
          </div>

          <div className="border border-slate-700 bg-[#0F2537] p-5 space-y-2">
            <div className="flex items-center justify-between text-slate-400">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-300">Uploaded Documents</span>
              <FileText className="h-4 w-4 text-[#D4AF37]" />
            </div>
            <div className="text-2xl font-black text-white font-mono">
              {documents.length} <span className="text-xs text-slate-400 font-normal">files</span>
            </div>
            <p className="text-[11px] text-slate-400 font-mono">Active corpus in uploads/</p>
          </div>

          <div className="border border-slate-700 bg-[#0F2537] p-5 space-y-2">
            <div className="flex items-center justify-between text-slate-400">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-300">Embedding Model</span>
              <Cpu className="h-4 w-4 text-[#D4AF37]" />
            </div>
            <div className="text-sm font-bold text-[#D4AF37] truncate font-mono">
              {health?.environment.embedding_model_configured ? 'text-embedding-3-small' : 'Default (384-dim)'}
            </div>
            <p className="text-[11px] text-slate-400 font-mono">Dense vector representations</p>
          </div>

          <div className="border border-slate-700 bg-[#0F2537] p-5 space-y-2">
            <div className="flex items-center justify-between text-slate-400">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-300">LLM Provider</span>
              <Zap className="h-4 w-4 text-[#D4AF37]" />
            </div>
            <div className="text-sm font-bold text-[#D4AF37] truncate font-mono">
              groq/compound-mini
            </div>
            <p className="text-[11px] text-slate-400 font-mono">Grounded answer synthesis</p>
          </div>
        </div>

        {/* Feature Highlight Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="border border-slate-700 bg-[#0F2537] p-6 space-y-3">
            <div className="h-8 w-8 bg-[#D4AF37] text-[#0B192C] flex items-center justify-center font-bold">
              <Database className="h-4 w-4" />
            </div>
            <h3 className="font-bold text-white uppercase text-sm font-sans">Hybrid Retrieval Engine</h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              Combines dense vector similarity with keyword BM25 scoring for exact tariff codes and document titles.
            </p>
          </div>

          <div className="border border-slate-700 bg-[#0F2537] p-6 space-y-3">
            <div className="h-8 w-8 bg-[#D4AF37] text-[#0B192C] flex items-center justify-center font-bold">
              <Layers className="h-4 w-4" />
            </div>
            <h3 className="font-bold text-white uppercase text-sm font-sans">Cross-Encoder Re-Ranking</h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              Applies cross-encoder re-ranking stage to refine top candidate chunks before context assembly.
            </p>
          </div>

          <div className="border border-slate-700 bg-[#0F2537] p-6 space-y-3">
            <div className="h-8 w-8 bg-[#D4AF37] text-[#0B192C] flex items-center justify-center font-bold">
              <ShieldCheck className="h-4 w-4" />
            </div>
            <h3 className="font-bold text-white uppercase text-sm font-sans">Anti-Hallucination Guard</h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              Enforces strict grounding validation and scope checks to prevent hallucinated compliance rules.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
