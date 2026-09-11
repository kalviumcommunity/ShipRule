'use client';

import React, { useState, useEffect, useRef } from 'react';
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
  LogIn,
  Globe2,
  RotateCcw,
  Bot
} from 'lucide-react';
import { apiService, ApiError } from '@/services/api';
import { QueryResponse, Source } from '@/types/api';
import { useAuth } from '@/context/AuthContext';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  response?: QueryResponse;
  error?: string;
  timestamp: string;
}

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
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [activeQueryText, setActiveQueryText] = useState('');
  const [searchStep, setSearchStep] = useState(0);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  // Interval for cycling dynamic search loading status
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (loading) {
      setSearchStep(0);
      interval = setInterval(() => {
        setSearchStep((prev) => (prev + 1) % 3);
      }, 1400);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [loading]);

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
    setActiveQueryText(queryText);
    setLoading(true);

    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: queryText,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setQuestion('');

    try {
      const res = await apiService.submitQuery({
        question: queryText,
        use_reranking: true,
      });

      const assistantMsg: ChatMessage = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: res.answer,
        response: res,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err: any) {
      let errDetail = 'An unexpected error occurred while querying ShipRule.';
      if (err instanceof ApiError) {
        errDetail = err.detail;
      } else if (err.message) {
        errDetail = err.message;
      }

      const errorMsg: ChatMessage = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: 'Execution Error',
        error: errDetail,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (qToSubmit?: string) => {
    const queryText = (qToSubmit || question).trim();
    if (!queryText) return;

    if (!user) {
      setPendingQuery(queryText);
      setShowAuthModal(true);
      return;
    }

    await executeSubmittedQuery(queryText);
  };

  const handleClearHistory = () => {
    setMessages([]);
    setActiveQueryText('');
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
        
        {/* Top Header & Platform Description */}
        <div className="border border-[#D4AF37]/30 bg-[#0F2537] p-6 shadow-xl rounded-xl flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <div className="inline-block border border-[#D4AF37] bg-[#0B192C] px-3 py-1 text-[11px] font-extrabold uppercase tracking-widest text-[#D4AF37] mb-2 rounded-xl">
              CDLP Grounded Intelligence Engine
            </div>
            <h2 className="text-xl sm:text-2xl font-black text-white uppercase tracking-tight font-sans flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-[#D4AF37]" />
              ShipRule Customs AI Assistant
            </h2>
            <p className="mt-1 text-xs text-slate-300">
              Query official customs duty policies, tariff codes, import regulations, and maritime shipping documentation.
            </p>
          </div>

          {messages.length > 0 && (
            <button
              onClick={handleClearHistory}
              className="inline-flex items-center gap-2 border border-slate-700 hover:border-[#D4AF37] bg-[#0B192C] hover:bg-[#162A45] px-4 py-2 text-xs font-bold text-slate-300 hover:text-[#D4AF37] transition rounded-xl"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              New Conversation Thread
            </button>
          )}
        </div>

        {/* Conversation Thread Stream (History Keeps All Previous Queries & Answers Visible) */}
        {messages.length > 0 && (
          <div className="space-y-6">
            {messages.map((msg) => (
              <div key={msg.id} className="space-y-4 animate-fade-in">
                
                {/* User Prompt Message Bubble */}
                {msg.role === 'user' && (
                  <div className="flex justify-end my-2">
                    <div className="bg-[#162A45] border border-[#D4AF37]/50 text-slate-100 px-5 py-3.5 rounded-2xl max-w-2xl text-xs sm:text-sm font-medium shadow-md font-sans flex items-start gap-3">
                      <span className="flex-1 whitespace-pre-wrap leading-relaxed">{msg.content}</span>
                      <div className="flex flex-col items-end gap-1 flex-shrink-0">
                        <div className="h-6 w-6 rounded-full bg-[#D4AF37]/20 border border-[#D4AF37] flex items-center justify-center text-[#D4AF37]">
                          <User className="h-3.5 w-3.5" />
                        </div>
                        <span className="text-[9px] text-slate-400 font-mono">{msg.timestamp}</span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Assistant Answer Message Card */}
                {msg.role === 'assistant' && (
                  <div className="space-y-4">
                    {msg.error ? (
                      /* Error Display Card */
                      <div className="flex items-start gap-3 border border-rose-800 bg-[#1e1015] p-4 text-rose-300 rounded-xl">
                        <AlertCircle className="h-5 w-5 text-rose-400 flex-shrink-0 mt-0.5" />
                        <div>
                          <h4 className="font-bold text-rose-200 uppercase text-xs">Execution Error</h4>
                          <p className="text-xs mt-1">{msg.error}</p>
                        </div>
                      </div>
                    ) : msg.response ? (
                      /* Full RAG Answer & Source Results */
                      <div className="space-y-6">
                        <div className="border border-[#D4AF37]/40 bg-[#0F2537] p-6 shadow-xl space-y-6 rounded-xl">
                          {/* Header Badge Row */}
                          <div className="flex items-center justify-between border-b border-[#D4AF37]/20 pb-4">
                            <div className="flex items-center gap-2">
                              <div className="h-7 w-7 rounded-lg bg-[#0B192C] border border-[#D4AF37]/50 flex items-center justify-center text-[#D4AF37]">
                                <Bot className="h-4 w-4" />
                              </div>
                              <div>
                                <h3 className="font-extrabold text-white uppercase text-xs sm:text-sm tracking-wider font-sans">ShipRule CDLP Response</h3>
                                <span className="text-[10px] text-slate-400 font-mono">{msg.timestamp}</span>
                              </div>
                            </div>

                            <div className="flex items-center gap-2">
                              {(() => {
                                const s = (msg.response.status || '').toUpperCase();
                                const ansLower = (msg.response.answer || '').toLowerCase();
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
                            {msg.response.answer}
                          </div>

                          {/* Metrics Footer */}
                          {msg.response.metadata && (
                            <div className="flex flex-wrap items-center gap-6 border border-[#28693C]/40 bg-[#0B192C] p-3 text-xs text-slate-300 font-mono rounded-xl">
                              {msg.response.metadata.timing?.total_pipeline_time_ms && (
                                <span className="flex items-center gap-1.5">
                                  <Clock className="h-3.5 w-3.5 text-[#D4AF37]" />
                                  Latency: <strong className="text-white">{msg.response.metadata.timing.total_pipeline_time_ms.toFixed(1)} ms</strong>
                                </span>
                              )}
                              {msg.response.metadata.token_usage?.total_tokens && (
                                <span className="flex items-center gap-1.5">
                                  <Zap className="h-3.5 w-3.5 text-[#D4AF37]" />
                                  Token Usage: <strong className="text-white">{msg.response.metadata.token_usage.total_tokens} tokens</strong>
                                </span>
                              )}
                              {msg.response.metadata.guardrail_decision && (
                                <span className="flex items-center gap-1">
                                  Guardrail: <strong className="text-[#D4AF37]">{msg.response.metadata.guardrail_decision}</strong>
                                </span>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    ) : null}
                  </div>
                )}

              </div>
            ))}
          </div>
        )}

        {/* Searching Document Name Loading Indicator (Matching Image 2) */}
        {loading && (
          <div className="flex items-center gap-3.5 p-4 sm:p-5 bg-[#0F2537] border border-[#38BDF8]/40 rounded-xl my-4 text-slate-200 text-xs sm:text-sm font-sans animate-fade-in shadow-xl">
            <div className="relative flex items-center justify-center flex-shrink-0">
              <Globe2 className="h-5 w-5 text-[#38BDF8] animate-spin" style={{ animationDuration: '3.5s' }} />
              <span className="absolute -inset-1 rounded-full bg-[#38BDF8]/20 animate-ping" />
            </div>

            <div className="flex flex-col min-w-0 flex-1">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="font-semibold text-slate-100 tracking-wide">
                  {searchStep === 0 && `Searching customs regulations & tariff documentation for "${activeQueryText}"`}
                  {searchStep === 1 && `Searching document index: ShipRule_Customs_Duty_Tariff_Schedule_1.txt & international_shipping_guide.pdf`}
                  {searchStep === 2 && `Retrieving grounded vector passages & calculating compliance citations...`}
                </span>
                <span className="inline-flex gap-1 items-center">
                  <span className="h-1.5 w-1.5 bg-[#D4AF37] rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <span className="h-1.5 w-1.5 bg-[#38BDF8] rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <span className="h-1.5 w-1.5 bg-[#D4AF37] rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </span>
              </div>

              <span className="text-[10px] text-[#D4AF37] font-mono mt-1 uppercase tracking-wider">
                Vector DB Index Lookup · Grounded RAG Query Protocol
              </span>
            </div>
          </div>
        )}

        {/* Scroll anchor */}
        <div ref={messagesEndRef} />

        {/* Bottom Input Area Card */}
        <div className="border border-[#D4AF37]/30 bg-[#0F2537] p-6 shadow-xl rounded-xl space-y-4">
          <div className="relative">
            <textarea
              rows={3}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question... e.g. What shipping documents are required for international customs clearance?"
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
              className="absolute bottom-4 right-4 flex items-center gap-2 bg-[#D4AF37] hover:bg-[#C59E2B] px-5 py-2.5 text-xs font-black uppercase tracking-wider text-[#0B192C] transition-all disabled:opacity-50 disabled:cursor-not-allowed border border-[#D4AF37] rounded-xl shadow-md cursor-pointer"
            >
              {loading ? (
                <>
                  <div className="h-4 w-4 animate-spin border-2 border-[#0B192C] border-t-transparent rounded-full" />
                  Searching...
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
            <span className="text-[#D4AF37]">Grounded Answers &amp; Citation Verification Active</span>
          </div>

          {/* Preset Sample Buttons */}
          <div className="border-t border-[#D4AF37]/20 pt-4">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-300 block mb-2">Try Asking:</span>
            <div className="flex flex-wrap items-center gap-2">
              {SAMPLE_QUESTIONS.map((sample, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    setQuestion(sample);
                    handleSubmit(sample);
                  }}
                  className="border border-slate-700 bg-[#0B192C] px-3.5 py-2 text-xs font-medium text-slate-200 transition hover:border-[#D4AF37] hover:text-[#D4AF37] hover:bg-[#162A45] rounded-xl cursor-pointer"
                >
                  {sample.length > 55 ? sample.slice(0, 55) + '...' : sample}
                </button>
              ))}
            </div>
          </div>
        </div>

      </div>

      {/* Auth Modal & Blurred Backdrop */}
      {showAuthModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-[#0B192C]/80 backdrop-blur-md transition-all duration-300">
          <div className="w-full max-w-md bg-[#0F2537] border border-[#D4AF37] rounded-xl p-8 shadow-2xl space-y-6 relative text-slate-100 animate-in fade-in zoom-in duration-200">
            
            {/* Close Modal Button */}
            <button
              onClick={() => setShowAuthModal(false)}
              className="absolute top-4 right-4 p-1 text-slate-400 hover:text-white hover:bg-slate-800 rounded-xl transition cursor-pointer"
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
                className={`py-2 text-center transition rounded-xl cursor-pointer ${authMode === 'signin' ? 'bg-[#D4AF37] text-[#0B192C]' : 'text-slate-400 hover:text-white'}`}
              >
                Sign In
              </button>
              <button
                type="button"
                onClick={() => { setAuthMode('signup'); setAuthError(null); }}
                className={`py-2 text-center transition rounded-xl cursor-pointer ${authMode === 'signup' ? 'bg-[#D4AF37] text-[#0B192C]' : 'text-slate-400 hover:text-white'}`}
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
                className="w-full flex items-center justify-center gap-2 bg-[#D4AF37] hover:bg-[#C59E2B] py-3 text-xs font-extrabold uppercase tracking-wider text-[#0B192C] transition-all disabled:opacity-50 rounded-xl shadow-lg border border-[#D4AF37] cursor-pointer"
              >
                {authLoading ? (
                  <div className="h-4 w-4 animate-spin border-2 border-[#0B192C] border-t-transparent rounded-full" />
                ) : authMode === 'signin' ? (
                  <>
                    <LogIn className="h-4 w-4" /> Sign In &amp; Run Query
                  </>
                ) : (
                  <>
                    <User className="h-4 w-4" /> Register &amp; Run Query
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
