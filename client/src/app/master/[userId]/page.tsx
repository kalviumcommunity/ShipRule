'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { useParams, useRouter } from 'next/navigation';
import {
  ArrowLeft,
  Activity,
  Zap,
  Clock,
  CheckCircle2,
  AlertCircle,
  Search,
  RefreshCw,
  User,
  ShieldCheck,
  Cpu,
  BarChart3,
  ChevronDown,
  ChevronUp,
  FileText
} from 'lucide-react';
import { useAuth } from '@/context/AuthContext';
import { apiService } from '@/services/api';
import { UserAnalytics, UserQueryLog } from '@/types/api';

export default function UserDetailMonitoringPage() {
  const params = useParams();
  const router = useRouter();
  const { role } = useAuth();

  const userIdParam = params?.userId as string;
  const decodedEmail = userIdParam ? decodeURIComponent(userIdParam) : '';

  const [analytics, setAnalytics] = useState<UserAnalytics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('ALL');
  const [expandedLogIndex, setExpandedLogIndex] = useState<number | null>(null);

  const fetchUserData = async () => {
    if (!decodedEmail) return;
    setLoading(true);
    setError(null);
    try {
      const data = await apiService.getAdminUserDetail(decodedEmail);
      setAnalytics(data);
    } catch (err: any) {
      console.error('Failed to load user detail:', err);
      setError(err.message || 'Failed to load user monitoring data.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (decodedEmail) {
      fetchUserData();
    }
  }, [decodedEmail]);

  const user = analytics?.user;
  const metrics = analytics?.metrics;
  const logs = analytics?.logs || [];

  // Filter logs
  const filteredLogs = logs.filter((log) => {
    const matchesSearch =
      log.question.toLowerCase().includes(searchQuery.toLowerCase()) ||
      log.status.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus = statusFilter === 'ALL' || log.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  const promptPercent =
    metrics && metrics.total_tokens > 0
      ? Math.round((metrics.prompt_tokens / metrics.total_tokens) * 100)
      : 50;

  const completionPercent = 100 - promptPercent;

  return (
    <div className="min-h-screen bg-[#0B192C] text-slate-100 font-sans p-4 sm:p-8 space-y-8">
      {/* HEADER & BREADCRUMB */}
      <div className="max-w-7xl mx-auto space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[#D4AF37]/20 pb-6">
          <div className="space-y-1">
            <div className="flex items-center gap-2 text-xs font-mono text-[#D4AF37]">
              <Link href="/master" className="hover:underline flex items-center gap-1">
                <ArrowLeft className="h-3.5 w-3.5" /> Master Admin
              </Link>
              <span>/</span>
              <span>User Monitoring</span>
              <span>/</span>
              <span className="text-white font-bold">{decodedEmail}</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-black text-white tracking-tight flex items-center gap-3">
              <User className="h-7 w-7 text-[#D4AF37]" />
              User Activity & Token Analytics
            </h1>
            <p className="text-xs text-slate-400">
              Detailed query history, LLM token consumption metrics, and performance analytics for user account.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={fetchUserData}
              disabled={loading}
              className="flex items-center gap-2 px-4 py-2 bg-[#0F2537] border border-[#D4AF37]/40 hover:border-[#D4AF37] text-white text-xs font-bold rounded-xl transition shadow-md"
            >
              <RefreshCw className={`h-4 w-4 text-[#D4AF37] ${loading ? 'animate-spin' : ''}`} />
              Refresh Data
            </button>
            <Link
              href="/master"
              className="px-4 py-2 bg-[#D4AF37] text-[#0B192C] font-extrabold text-xs rounded-xl shadow-md hover:bg-yellow-400 transition"
            >
              Back to Master Admin
            </Link>
          </div>
        </div>

        {/* USER PROFILE HEADER BANNER */}
        {user && (
          <div className="border border-[#D4AF37]/30 bg-[#0F2537] p-6 rounded-2xl shadow-lg flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className="h-12 w-12 rounded-2xl border border-[#D4AF37]/50 bg-[#0B192C] flex items-center justify-center font-black text-[#D4AF37] text-lg uppercase shadow-inner">
                {user.full_name ? user.full_name[0] : user.email[0]}
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h2 className="text-lg font-bold text-white font-mono">{user.email}</h2>
                  <span className="px-2.5 py-0.5 bg-[#0B192C] border border-[#D4AF37]/40 text-[#D4AF37] font-bold text-[10px] uppercase rounded-xl">
                    {user.role}
                  </span>
                </div>
                <p className="text-xs text-slate-400">{user.full_name || 'Registered Account'}</p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-4 text-xs font-mono text-slate-300">
              <div className="bg-[#0B192C] px-3 py-1.5 rounded-xl border border-slate-800">
                <span className="text-slate-400 text-[10px] block">Status</span>
                <span className="text-emerald-400 font-bold flex items-center gap-1">
                  <ShieldCheck className="h-3.5 w-3.5" /> Verified User
                </span>
              </div>
              <div className="bg-[#0B192C] px-3 py-1.5 rounded-xl border border-slate-800">
                <span className="text-slate-400 text-[10px] block">Last Activity</span>
                <span className="text-white font-bold">
                  {metrics?.last_active ? new Date(metrics.last_active).toLocaleString() : 'No recent activity'}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>

      {loading ? (
        <div className="max-w-7xl mx-auto border border-[#D4AF37]/20 bg-[#0F2537] p-12 rounded-2xl text-center space-y-3">
          <RefreshCw className="h-8 w-8 text-[#D4AF37] animate-spin mx-auto" />
          <p className="text-sm text-slate-300 font-bold font-mono">Loading user analytics & token logs...</p>
        </div>
      ) : error ? (
        <div className="max-w-7xl mx-auto border border-red-500/40 bg-red-950/30 p-8 rounded-2xl text-center space-y-3">
          <AlertCircle className="h-8 w-8 text-red-400 mx-auto" />
          <p className="text-sm font-bold text-red-300">{error}</p>
          <button
            onClick={fetchUserData}
            className="px-4 py-2 bg-red-600 text-white font-bold text-xs rounded-xl"
          >
            Retry
          </button>
        </div>
      ) : metrics ? (
        <div className="max-w-7xl mx-auto space-y-8">
          {/* METRICS CARDS GRID */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {/* Card 1: Total Tokens */}
            <div className="border border-[#D4AF37]/30 bg-[#0F2537] p-6 space-y-3 rounded-2xl shadow-lg relative overflow-hidden">
              <div className="flex items-center justify-between text-slate-300">
                <span className="text-xs font-extrabold uppercase tracking-wider">Total LLM Tokens</span>
                <Zap className="h-5 w-5 text-[#D4AF37]" />
              </div>
              <div className="text-3xl font-black text-white font-mono">
                {metrics.total_tokens.toLocaleString()}{' '}
                <span className="text-xs text-slate-400 font-normal">tokens</span>
              </div>
              <p className="text-[11px] text-slate-400">
                Prompt + Completion tokens consumed
              </p>
            </div>

            {/* Card 2: Total Queries */}
            <div className="border border-[#D4AF37]/30 bg-[#0F2537] p-6 space-y-3 rounded-2xl shadow-lg">
              <div className="flex items-center justify-between text-slate-300">
                <span className="text-xs font-extrabold uppercase tracking-wider">Queries Executed</span>
                <Activity className="h-5 w-5 text-[#D4AF37]" />
              </div>
              <div className="text-3xl font-black text-white font-mono">
                {metrics.total_queries}{' '}
                <span className="text-xs text-slate-400 font-normal">queries</span>
              </div>
              <p className="text-[11px] text-slate-400">
                Total RAG search requests submitted
              </p>
            </div>

            {/* Card 3: Avg Latency */}
            <div className="border border-[#D4AF37]/30 bg-[#0F2537] p-6 space-y-3 rounded-2xl shadow-lg">
              <div className="flex items-center justify-between text-slate-300">
                <span className="text-xs font-extrabold uppercase tracking-wider">Avg Latency</span>
                <Clock className="h-5 w-5 text-[#D4AF37]" />
              </div>
              <div className="text-3xl font-black text-white font-mono">
                {metrics.avg_latency_ms}{' '}
                <span className="text-xs text-slate-400 font-normal">ms</span>
              </div>
              <p className="text-[11px] text-slate-400">
                Average pipeline execution speed
              </p>
            </div>

            {/* Card 4: Success Rate */}
            <div className="border border-[#D4AF37]/30 bg-[#0F2537] p-6 space-y-3 rounded-2xl shadow-lg">
              <div className="flex items-center justify-between text-slate-300">
                <span className="text-xs font-extrabold uppercase tracking-wider">Supported Answers</span>
                <CheckCircle2 className="h-5 w-5 text-emerald-400" />
              </div>
              <div className="text-3xl font-black text-white font-mono">
                {metrics.successful_queries} / {metrics.total_queries}
              </div>
              <p className="text-[11px] text-slate-400">
                Grounded answer success rate
              </p>
            </div>
          </div>

          {/* TOKEN USAGE BREAKDOWN BAR */}
          <div className="border border-[#D4AF37]/30 bg-[#0F2537] p-6 space-y-4 rounded-2xl shadow-lg">
            <div className="flex items-center justify-between border-b border-[#D4AF37]/20 pb-3">
              <h3 className="font-extrabold text-white uppercase text-xs tracking-wider flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-[#D4AF37]" />
                Token Breakdown (Prompt vs. Completion)
              </h3>
              <span className="text-xs font-mono text-slate-300">
                Total: <strong className="text-[#D4AF37]">{metrics.total_tokens.toLocaleString()}</strong> tokens
              </span>
            </div>

            <div className="space-y-2">
              <div className="h-4 w-full bg-[#0B192C] rounded-full overflow-hidden flex border border-slate-700">
                <div
                  style={{ width: `${promptPercent}%` }}
                  className="bg-[#D4AF37] h-full transition-all duration-500 relative group cursor-pointer"
                  title={`Prompt Tokens: ${metrics.prompt_tokens.toLocaleString()}`}
                />
                <div
                  style={{ width: `${completionPercent}%` }}
                  className="bg-sky-500 h-full transition-all duration-500 relative group cursor-pointer"
                  title={`Completion Tokens: ${metrics.completion_tokens.toLocaleString()}`}
                />
              </div>

              <div className="flex justify-between items-center text-xs font-mono">
                <div className="flex items-center gap-2">
                  <span className="h-3 w-3 rounded-full bg-[#D4AF37]" />
                  <span className="text-slate-300">
                    Prompt (Input): <strong>{metrics.prompt_tokens.toLocaleString()}</strong> ({promptPercent}%)
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="h-3 w-3 rounded-full bg-sky-500" />
                  <span className="text-slate-300">
                    Completion (Output): <strong>{metrics.completion_tokens.toLocaleString()}</strong> ({completionPercent}%)
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* USER QUERY LOG HISTORY */}
          <div className="border border-[#D4AF37]/30 bg-[#0F2537] p-6 space-y-6 rounded-2xl shadow-lg">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[#D4AF37]/20 pb-4">
              <div>
                <h3 className="font-extrabold text-white uppercase text-xs tracking-wider flex items-center gap-2">
                  <FileText className="h-4 w-4 text-[#D4AF37]" />
                  Query Execution History ({filteredLogs.length})
                </h3>
                <p className="text-[11px] text-slate-400">
                  Search and inspect specific questions asked by {decodedEmail}.
                </p>
              </div>

              <div className="flex items-center gap-3">
                {/* Search Input */}
                <div className="relative max-w-xs w-full">
                  <Search className="absolute left-3 top-2.5 h-3.5 w-3.5 text-slate-400" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search queries..."
                    className="w-full bg-[#0B192C] border border-slate-700 rounded-xl py-1.5 pl-9 pr-3 text-xs text-white placeholder-slate-500 focus:border-[#D4AF37] focus:outline-none"
                  />
                </div>

                {/* Status Filter Dropdown */}
                <select
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                  className="bg-[#0B192C] border border-slate-700 text-xs text-slate-300 py-1.5 px-3 rounded-xl focus:border-[#D4AF37] focus:outline-none"
                >
                  <option value="ALL">All Statuses</option>
                  <option value="SUPPORTED">SUPPORTED</option>
                  <option value="INSUFFICIENT_CONTEXT">INSUFFICIENT_CONTEXT</option>
                  <option value="SECURITY_BLOCKED">SECURITY_BLOCKED</option>
                  <option value="OUT_OF_SCOPE">OUT_OF_SCOPE</option>
                </select>
              </div>
            </div>

            {filteredLogs.length === 0 ? (
              <div className="text-center py-10 space-y-2">
                <FileText className="h-8 w-8 text-slate-600 mx-auto" />
                <p className="text-xs text-slate-400 italic">No query logs match the selected filter.</p>
              </div>
            ) : (
              <div className="space-y-3">
                {filteredLogs.map((log, idx) => {
                  const isExpanded = expandedLogIndex === idx;
                  return (
                    <div
                      key={idx}
                      className="border border-slate-800 bg-[#0B192C] p-4 rounded-xl space-y-3 hover:border-[#D4AF37]/40 transition"
                    >
                      <div
                        onClick={() => setExpandedLogIndex(isExpanded ? null : idx)}
                        className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 cursor-pointer"
                      >
                        <div className="flex items-start gap-3">
                          <span
                            className={`px-2.5 py-1 text-[10px] font-bold uppercase rounded-lg border mt-0.5 ${
                              log.status === 'SUPPORTED'
                                ? 'bg-emerald-950/60 border-emerald-500/40 text-emerald-300'
                                : log.status === 'SECURITY_BLOCKED'
                                ? 'bg-red-950/60 border-red-500/40 text-red-300'
                                : 'bg-amber-950/60 border-amber-500/40 text-amber-300'
                            }`}
                          >
                            {log.status}
                          </span>
                          <p className="text-xs font-bold text-white leading-relaxed">
                            "{log.question}"
                          </p>
                        </div>

                        <div className="flex items-center gap-4 text-xs font-mono text-slate-400 shrink-0">
                          <span className="text-[#D4AF37] font-bold">
                            {(log.total_tokens || 0).toLocaleString()} tokens
                          </span>
                          <span>{log.latency_ms} ms</span>
                          <span className="text-[11px]">
                            {log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : ''}
                          </span>
                          {isExpanded ? (
                            <ChevronUp className="h-4 w-4 text-[#D4AF37]" />
                          ) : (
                            <ChevronDown className="h-4 w-4 text-slate-500" />
                          )}
                        </div>
                      </div>

                      {/* EXPANDABLE DETAIL */}
                      {isExpanded && (
                        <div className="pt-3 border-t border-slate-800 text-xs space-y-2 font-mono text-slate-300 bg-[#0F2537]/50 p-3 rounded-lg">
                          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-[11px]">
                            <div>
                              <span className="text-slate-400 block">Prompt Tokens:</span>
                              <strong className="text-white">{log.prompt_tokens || 0}</strong>
                            </div>
                            <div>
                              <span className="text-slate-400 block">Completion Tokens:</span>
                              <strong className="text-white">{log.completion_tokens || 0}</strong>
                            </div>
                            <div>
                              <span className="text-slate-400 block">Timestamp:</span>
                              <strong className="text-white">
                                {log.timestamp ? new Date(log.timestamp).toLocaleString() : 'N/A'}
                              </strong>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
}
