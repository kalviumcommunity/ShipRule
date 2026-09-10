'use client';

import React, { useState } from 'react';
import {
  Search,
  Sparkles,
  BookOpen,
  AlertCircle,
  Clock,
  ShieldAlert,
  ShieldCheck,
  Zap,
  FileCheck,
  Lock,
  X,
  User,
  LogIn
} from 'lucide-react';
import { apiService, ApiError } from '@/services/api';
import { QueryResponse, Source } from '@/types/api';
import { useAuth } from '@/context/AuthContext';

const SAMPLE_QUESTIONS = [
  'What shipping documents are required for international customs clearance?',
  'What mandatory statutory registrations and compliance licenses are required for importing IT hardware in India?',
  'What are the CIF and FOB shipping terms obligations?',
  'What is the speed of light in vacuum and how is it measured?'
];

export function QueryInterface() {
  const { user, login, signup } = useAuth();
  
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Authentication Modal & Blur State
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [pendingQuery, setPendingQuery] = useState('');
  const [authMode, setAuthMode] = useState<'signin' | 'signup'>('signin');
  const [authEmail, setAuthEmail] = useState('');
  const [authPassword, setAuthPassword] = useState('');
  const [authFullName, setAuthFullName] = useState('');
  const [authError, setAuthError] = useState<string | null>(null);
  const [authLoading, setAuthLoading] = useState(false);

  const executeSubmittedQuery = async (queryText: string) => {
    setLoading(true);
    setError(null);

    try {
      const res = await apiService.submitQuery({
        question: queryText,
        use_reranking: true,
      });
      setResponse(res);
    } catch (err: any) {
      if (err instanceof ApiError) {
        setError(err.detail);
      } else {
        setError(err.message || 'An unexpected error occurred while querying ShipRule.');
      }
      setResponse(null);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (qToSubmit?: string) => {
    const queryText = (qToSubmit || question).trim();
    if (!queryText) return;

    // Blur page and require authentication if user is not logged in
    if (!user) {
      setPendingQuery(queryText);
      setShowAuthModal(true);
      return;
    }

    await executeSubmittedQuery(queryText);
  };

  const handleAuthSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!authEmail.trim() || !authPassword.trim()) {
      setAuthError('Please fill in all required fields.');
      return;
    }

    setAuthLoading(true);
    setAuthError(null);

    try {
      if (authMode === 'signin') {
        await login(authEmail, authPassword);
      } else {
        await signup(authEmail, authPassword, authFullName);
      }
      setShowAuthModal(false);
      
      // Auto-run the pending query after authentication
      const qToRun = pendingQuery || question;
      if (qToRun.trim()) {
        await executeSubmittedQuery(qToRun.trim());
      }
    } catch (err: any) {
      if (err instanceof ApiError) {
        setAuthError(err.detail);
      } else {
        setAuthError(err.message || 'Authentication failed. Please check credentials.');
      }
    } finally {
      setAuthLoading(false);
    }
  };

  return (
    <div className="relative w-full space-y-6">
      {/* Background container blurred when auth modal is open */}
      <div className={`w-full space-y-6 transition-all duration-300 ${showAuthModal ? 'filter blur-md pointer-events-none select-none' : ''}`}>
        
        {/* Top Search Card (12px rounded corners) */}
        <div className="border border-[#D4AF37]/30 bg-[#0F2537] p-8 shadow-xl rounded-xl">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-[#D4AF37]/20 pb-4">
            <div>
              <div className="inline-block border border-[#D4AF37] bg-[#0B192C] px-3 py-1 text-[11px] font-extrabold uppercase tracking-widest text-[#D4AF37] mb-2 rounded-xl">
                CDLP Grounded Intelligence Engine
              </div>
              <h2 className="text-2xl font-black text-white uppercase tracking-tight font-sans flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-[#D4AF37]" />
                Ask ShipRule RAG Engine
              </h2>
              <p className="mt-1 text-xs text-slate-300">
                Query official customs duty policies, tariff codes, import regulations, and maritime shipping documentation.
              </p>
            </div>
          </div>

          {/* Input Textarea & Submit Button */}
          <div className="mt-6 space-y-4">
            <div className="relative">
              <textarea
                rows={3}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="e.g. What shipping documents are required for international customs clearance?"
                className="w-full border border-slate-700 bg-[#0B192C] p-4 text-sm text-white placeholder-slate-500 focus:border-[#D4AF37] focus:outline-none focus:ring-1 focus:ring-[#D4AF37] font-sans rounded-xl"
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                    handleSubmit();
                  }
                }}
              />
              <button
                onClick={() => handleSubmit()}
                disabled={loading || !question.trim()}
                className="absolute bottom-4 right-4 flex items-center gap-2 bg-[#D4AF37] hover:bg-[#C59E2B] px-5 py-2.5 text-xs font-black uppercase tracking-wider text-[#0B192C] transition-all disabled:opacity-50 disabled:cursor-not-allowed border border-[#D4AF37] rounded-xl shadow-md"
              >
                {loading ? (
                  <>
                    <div className="h-4 w-4 animate-spin border-2 border-[#0B192C] border-t-transparent rounded-full" />
                    Querying Engine...
                  </>
                ) : (
                  <>
                    <Search className="h-4 w-4" />
                    Submit Query
                  </>
                )}
              </button>
            </div>

            <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono">
              <span>Press Ctrl + Enter to submit question</span>
              <span className="text-[#D4AF37]">Grounded Answers & Citation Verification Active</span>
            </div>
          </div>

          {/* Preset Sample Buttons */}
          <div className="mt-6 border-t border-[#D4AF37]/20 pt-4">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-300 block mb-2">Try Asking:</span>
            <div className="flex flex-wrap items-center gap-2">
              {SAMPLE_QUESTIONS.map((sample, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    setQuestion(sample);
                    handleSubmit(sample);
                  }}
                  className="border border-slate-700 bg-[#0B192C] px-3.5 py-2 text-xs font-medium text-slate-200 transition hover:border-[#D4AF37] hover:text-[#D4AF37] hover:bg-[#162A45] rounded-xl"
                >
                  {sample.length > 55 ? sample.slice(0, 55) + '...' : sample}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="flex items-start gap-3 border border-rose-800 bg-[#1e1015] p-4 text-rose-300 rounded-xl">
            <AlertCircle className="h-5 w-5 text-rose-400 flex-shrink-0 mt-0.5" />
            <div>
              <h4 className="font-bold text-rose-200 uppercase text-xs">Execution Error</h4>
              <p className="text-xs mt-1">{error}</p>
            </div>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="border border-[#D4AF37]/30 bg-[#0F2537] p-6 space-y-4 animate-pulse rounded-xl">
            <div className="h-4 w-1/4 bg-slate-800 rounded-xl" />
            <div className="space-y-2">
              <div className="h-3 w-full bg-slate-800 rounded-xl" />
              <div className="h-3 w-5/6 bg-slate-800 rounded-xl" />
              <div className="h-3 w-4/6 bg-slate-800 rounded-xl" />
            </div>
          </div>
        )}

        {/* RAG Answer & Source Results */}
        {response && !loading && (
          <div className="space-y-6">
            <div className="border border-[#D4AF37]/40 bg-[#0F2537] p-6 shadow-xl space-y-6 rounded-xl">
              {/* Header Badge Row */}
              <div className="flex items-center justify-between border-b border-[#D4AF37]/20 pb-4">
                <div className="flex items-center gap-2">
                  <FileCheck className="h-5 w-5 text-[#D4AF37]" />
                  <h3 className="font-extrabold text-white uppercase text-sm tracking-wider font-sans">Grounded Response</h3>
                </div>

                <div className="flex items-center gap-2">
                  {(() => {
                    const s = (response.status || '').toUpperCase();
                    const ansLower = (response.answer || '').toLowerCase();
                    if (s === 'SUPPORTED' || s === 'ANSWERED') {
                      return (
                        <span className="flex items-center gap-1.5 border border-emerald-700 bg-emerald-950 px-3 py-1 text-xs font-bold text-emerald-400 uppercase tracking-wider rounded-xl">
                          <ShieldCheck className="h-3.5 w-3.5" /> Grounded Response
                        </span>
                      );
                    }
                    if (s === 'INSUFFICIENT_CONTEXT' || (s === 'REFUSED' && (ansLower.includes('insufficient') || ansLower.includes('understand your question')))) {
                      return (
                        <span className="flex items-center gap-1.5 border border-[#D4AF37] bg-[#1a1708] px-3 py-1 text-xs font-bold text-[#D4AF37] uppercase tracking-wider rounded-xl">
                          <AlertCircle className="h-3.5 w-3.5 text-[#D4AF37]" /> Insufficient Evidence
                        </span>
                      );
                    }
                    if (s === 'SECURITY_BLOCKED' || (s === 'REFUSED' && ansLower.includes('administrative credentials'))) {
                      return (
                        <span className="flex items-center gap-1.5 border border-rose-700 bg-rose-950 px-3 py-1 text-xs font-bold text-rose-400 uppercase tracking-wider rounded-xl">
                          <ShieldAlert className="h-3.5 w-3.5 text-rose-400" /> Request Not Allowed
                        </span>
                      );
                    }
                    return (
                      <span className="flex items-center gap-1.5 border border-amber-700 bg-amber-950 px-3 py-1 text-xs font-bold text-amber-400 uppercase tracking-wider rounded-xl">
                        <ShieldAlert className="h-3.5 w-3.5" /> Outside Supported Domain
                      </span>
                    );
                  })()}
                </div>
              </div>

              {/* Answer Body */}
              <div className="prose prose-invert max-w-none text-slate-100 leading-relaxed text-sm whitespace-pre-wrap font-sans">
                {response.answer}
              </div>

              {/* Metrics Footer */}
              {response.metadata && (
                <div className="flex flex-wrap items-center gap-6 border border-[#28693C]/40 bg-[#0B192C] p-3 text-xs text-slate-300 font-mono rounded-xl">
                  {response.metadata.timing?.total_pipeline_time_ms && (
                    <span className="flex items-center gap-1.5">
                      <Clock className="h-3.5 w-3.5 text-[#D4AF37]" />
                      Latency: <strong className="text-white">{response.metadata.timing.total_pipeline_time_ms.toFixed(1)} ms</strong>
                    </span>
                  )}
                  {response.metadata.token_usage?.total_tokens && (
                    <span className="flex items-center gap-1.5">
                      <Zap className="h-3.5 w-3.5 text-[#D4AF37]" />
                      Token Usage: <strong className="text-white">{response.metadata.token_usage.total_tokens} tokens</strong>
                    </span>
                  )}
                  {response.metadata.guardrail_decision && (
                    <span className="flex items-center gap-1">
                      Guardrail: <strong className="text-[#D4AF37]">{response.metadata.guardrail_decision}</strong>
                    </span>
                  )}
                </div>
              )}
            </div>

            {/* Sources Registry */}
            <div className="border border-[#D4AF37]/30 bg-[#0F2537] p-6 space-y-4 rounded-xl">
              <h4 className="font-extrabold text-white uppercase text-xs tracking-wider flex items-center gap-2 font-sans">
                <BookOpen className="h-4 w-4 text-[#D4AF37]" />
                Cited Source Documents ({response.sources.length})
              </h4>

              {response.sources.length === 0 ? (
                <p className="text-xs text-slate-400 italic">
                  Zero hallucination safeguard active: No source references cited for refused query.
                </p>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {response.sources.map((src: Source, idx: number) => (
                    <div
                      key={idx}
                      className="flex flex-col justify-between border border-slate-700 bg-[#0B192C] p-4 space-y-2 hover:border-[#D4AF37] transition rounded-xl"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <span className="font-bold text-xs text-[#D4AF37] truncate font-mono">
                          [{idx + 1}] {src.source}
                        </span>
                        {src.score !== undefined && (
                          <span className="border border-[#D4AF37]/50 bg-[#0F2537] px-2 py-0.5 text-[10px] font-mono text-slate-200 rounded-xl">
                            Score: {src.score}
                          </span>
                        )}
                      </div>
                      {src.chunk_id && (
                        <span className="text-[11px] text-slate-400 font-mono">
                          ID: {src.chunk_id}
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Auth Modal & Blurred Backdrop */}
      {showAuthModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-[#0B192C]/80 backdrop-blur-md transition-all duration-300">
          <div className="w-full max-w-md bg-[#0F2537] border border-[#D4AF37] rounded-xl p-8 shadow-2xl space-y-6 relative text-slate-100 animate-in fade-in zoom-in duration-200">
            
            {/* Close Modal Button */}
            <button
              onClick={() => setShowAuthModal(false)}
              className="absolute top-4 right-4 p-1 text-slate-400 hover:text-white hover:bg-slate-800 rounded-xl transition"
            >
              <X className="h-5 w-5" />
            </button>

            {/* Header Branding & Lock Notice */}
            <div className="text-center space-y-2 flex flex-col items-center">
              <div className="p-3 bg-[#0B192C] border border-[#D4AF37]/40 rounded-xl mb-1">
                <img src="/logo.png" alt="ShipRule Logo" className="h-10 w-auto object-contain" />
              </div>
              <div className="inline-flex items-center gap-1.5 px-3 py-1 bg-[#D4AF37]/10 border border-[#D4AF37]/40 text-[#D4AF37] text-xs font-bold uppercase tracking-wider rounded-xl">
                <Lock className="h-3.5 w-3.5" /> Authentication Required
              </div>
              <p className="text-xs text-slate-300 max-w-xs">
                You must be logged in to query the ShipRule RAG Customs Engine. Sign in or register below to proceed.
              </p>
            </div>

            {/* Mode Switcher Tabs */}
            <div className="grid grid-cols-2 bg-[#0B192C] p-1 border border-slate-700 text-xs font-bold uppercase tracking-wider rounded-xl">
              <button
                type="button"
                onClick={() => { setAuthMode('signin'); setAuthError(null); }}
                className={`py-2 text-center transition rounded-xl ${authMode === 'signin' ? 'bg-[#D4AF37] text-[#0B192C]' : 'text-slate-400 hover:text-white'}`}
              >
                Sign In
              </button>
              <button
                type="button"
                onClick={() => { setAuthMode('signup'); setAuthError(null); }}
                className={`py-2 text-center transition rounded-xl ${authMode === 'signup' ? 'bg-[#D4AF37] text-[#0B192C]' : 'text-slate-400 hover:text-white'}`}
              >
                Register
              </button>
            </div>

            {/* Auth Form */}
            <form onSubmit={handleAuthSubmit} className="space-y-4">
              {authError && (
                <div className="p-3 border border-rose-800 bg-[#1e1015] text-rose-300 text-xs font-medium rounded-xl">
                  {authError}
                </div>
              )}

              {authMode === 'signup' && (
                <div className="space-y-1">
                  <label className="text-xs font-bold text-slate-300 uppercase tracking-wider block">Full Name</label>
                  <input
                    type="text"
                    required
                    value={authFullName}
                    onChange={(e) => setAuthFullName(e.target.value)}
                    placeholder="John Doe"
                    className="w-full border border-slate-700 bg-[#0B192C] px-3.5 py-2.5 text-xs text-white placeholder-slate-500 focus:border-[#D4AF37] focus:outline-none rounded-xl"
                  />
                </div>
              )}

              <div className="space-y-1">
                <label className="text-xs font-bold text-slate-300 uppercase tracking-wider block">Email Address</label>
                <input
                  type="email"
                  required
                  value={authEmail}
                  onChange={(e) => setAuthEmail(e.target.value)}
                  placeholder="user@example.com"
                  className="w-full border border-slate-700 bg-[#0B192C] px-3.5 py-2.5 text-xs text-white placeholder-slate-500 focus:border-[#D4AF37] focus:outline-none rounded-xl"
                />
              </div>

              <div className="space-y-1">
                <label className="text-xs font-bold text-slate-300 uppercase tracking-wider block">Password</label>
                <input
                  type="password"
                  required
                  value={authPassword}
                  onChange={(e) => setAuthPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full border border-slate-700 bg-[#0B192C] px-3.5 py-2.5 text-xs text-white placeholder-slate-500 focus:border-[#D4AF37] focus:outline-none rounded-xl"
                />
              </div>

              <button
                type="submit"
                disabled={authLoading}
                className="w-full flex items-center justify-center gap-2 bg-[#D4AF37] hover:bg-[#C59E2B] py-3 text-xs font-extrabold uppercase tracking-wider text-[#0B192C] transition-all disabled:opacity-50 rounded-xl shadow-lg border border-[#D4AF37]"
              >
                {authLoading ? (
                  <div className="h-4 w-4 animate-spin border-2 border-[#0B192C] border-t-transparent rounded-full" />
                ) : authMode === 'signin' ? (
                  <>
                    <LogIn className="h-4 w-4" /> Sign In & Run Query
                  </>
                ) : (
                  <>
                    <User className="h-4 w-4" /> Register & Run Query
                  </>
                )}
              </button>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
