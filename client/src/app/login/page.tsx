'use client';

import React, { useState } from 'react';
import Image from 'next/image';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/context/AuthContext';
import { ApiError } from '@/services/api';

export default function LoginPage() {
  const router = useRouter();
  const { login, signup } = useAuth();
  const [activeTab, setActiveTab] = useState<'signin' | 'signup'>('signin');

  // Form states
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim() || !password.trim()) {
      setError('Please fill in all required fields.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      if (activeTab === 'signin') {
        await login(email, password);
      } else {
        await signup(email, password, fullName || 'User');
      }
      router.push('/query');
    } catch (err: any) {
      if (err instanceof ApiError) {
        setError(err.detail);
      } else {
        setError(err.message || 'Authentication failed. Please check your credentials.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full min-h-screen bg-[#0B192C] text-white flex flex-col lg:flex-row font-sans selection:bg-[#D4AF37] selection:text-[#0B192C] overflow-x-hidden">
      
      {/* LEFT COLUMN: Form Card Container */}
      <div className="w-full lg:w-[55%] min-h-screen flex flex-col justify-center items-center p-6 sm:p-10 lg:p-12 z-10">
        
        {/* CARD CONTAINER */}
        <div className="w-full max-w-[540px] bg-[#0F2537] border border-[#D4AF37]/30 rounded-2xl p-8 sm:p-10 shadow-2xl space-y-6">
          
          {/* LOGO & HEADING */}
          <div className="space-y-3 text-center sm:text-left">
            <div className="flex items-center gap-3 justify-center sm:justify-start mb-2">
              <img src="/logo.png" alt="ShipRule Logo" className="h-10 w-auto object-contain" />
            </div>
            <h1 className="text-[28px] sm:text-[34px] font-black text-white uppercase tracking-tight font-sans leading-tight">
              {activeTab === 'signin' ? 'Welcome Back' : 'Create Account'}
            </h1>
            <p className="text-slate-300 text-xs sm:text-sm leading-relaxed">
              Log in or register your profile to query official customs duty policies, tariff rates, and shipping documentation.
            </p>
          </div>

          {/* TAB SWITCHER */}
          <div className="grid grid-cols-2 bg-[#0B192C] p-1 border border-[#D4AF37]/20 text-xs font-bold uppercase tracking-wider rounded-xl">
            <button
              type="button"
              onClick={() => { setActiveTab('signin'); setError(null); }}
              className={`py-2.5 text-center transition rounded-xl ${activeTab === 'signin' ? 'bg-[#D4AF37] text-[#0B192C] font-extrabold' : 'text-slate-300 hover:text-white'}`}
            >
              Sign In
            </button>
            <button
              type="button"
              onClick={() => { setActiveTab('signup'); setError(null); }}
              className={`py-2.5 text-center transition rounded-xl ${activeTab === 'signup' ? 'bg-[#D4AF37] text-[#0B192C] font-extrabold' : 'text-slate-300 hover:text-white'}`}
            >
              Register
            </button>
          </div>

          {/* FORM */}
          <form onSubmit={handleSubmit} className="space-y-4 pt-1">
            {error && (
              <div className="p-3.5 bg-rose-950/90 border border-rose-800 text-rose-300 text-xs font-medium rounded-xl animate-fadeIn">
                {error}
              </div>
            )}

            {activeTab === 'signup' && (
              <div className="space-y-1.5">
                <label className="block text-slate-200 text-xs font-bold uppercase tracking-wider">Full Name</label>
                <input
                  type="text"
                  required
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  placeholder="e.g. John Doe"
                  className="w-full h-[44px] bg-[#0B192C] border border-slate-700 focus:border-[#D4AF37] rounded-xl px-4 text-xs text-white placeholder-slate-500 focus:outline-none transition-colors font-sans"
                />
              </div>
            )}

            <div className="space-y-1.5">
              <label className="block text-slate-200 text-xs font-bold uppercase tracking-wider">Email Address</label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="yourmail@yourdomain.com"
                className="w-full h-[44px] bg-[#0B192C] border border-slate-700 focus:border-[#D4AF37] rounded-xl px-4 text-xs text-white placeholder-slate-500 focus:outline-none transition-colors font-sans"
              />
            </div>

            <div className="space-y-1.5">
              <label className="block text-slate-200 text-xs font-bold uppercase tracking-wider">Password</label>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full h-[44px] bg-[#0B192C] border border-slate-700 focus:border-[#D4AF37] rounded-xl px-4 text-xs text-white placeholder-slate-500 focus:outline-none transition-colors font-sans"
              />
            </div>

            {/* SUBMIT BUTTON */}
            <div className="pt-2">
              <button
                type="submit"
                disabled={loading}
                className="w-full h-[48px] bg-[#D4AF37] hover:bg-[#C59E2B] active:bg-[#B58E1B] text-[#0B192C] font-extrabold uppercase text-xs tracking-wider rounded-xl transition-all duration-200 flex items-center justify-center cursor-pointer shadow-lg disabled:opacity-60 border border-[#D4AF37]"
              >
                {loading ? (
                  <div className="h-5 w-5 animate-spin border-2 border-[#0B192C] border-t-transparent rounded-full" />
                ) : activeTab === 'signin' ? (
                  'Sign In'
                ) : (
                  'Create Account'
                )}
              </button>
            </div>
          </form>

          {/* BOTTOM TEXT */}
          <div className="text-center pt-2 text-xs">
            {activeTab === 'signin' ? (
              <p className="text-slate-400">
                Don&apos;t have an account?{' '}
                <button
                  type="button"
                  onClick={() => { setActiveTab('signup'); setError(null); }}
                  className="text-[#D4AF37] font-bold hover:underline cursor-pointer bg-transparent border-none p-0 inline ml-1"
                >
                  Register now
                </button>
              </p>
            ) : (
              <p className="text-slate-400">
                Already registered?{' '}
                <button
                  type="button"
                  onClick={() => { setActiveTab('signin'); setError(null); }}
                  className="text-[#D4AF37] font-bold hover:underline cursor-pointer bg-transparent border-none p-0 inline ml-1"
                >
                  Sign in
                </button>
              </p>
            )}
          </div>
        </div>
      </div>

      {/* RIGHT HERO PANEL WITH SHIP HERO IMAGE */}
      <div className="hidden lg:flex w-full lg:w-[45%] p-6 lg:p-8 flex-col justify-center items-center">
        <div className="relative w-full max-w-[580px] h-[calc(100vh-64px)] min-h-[600px] rounded-2xl overflow-hidden shadow-2xl border-2 border-[#D4AF37]/30 bg-[#0F2537]">
          
          {/* SHIP HERO IMAGE */}
          <Image
            src="/ship_hero.png"
            alt="Global Freight Solutions Cargo Ship"
            fill
            priority
            className="object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-[#0B192C] via-[#0B192C]/40 to-transparent" />

          {/* FLOATING OVERLAY */}
          <div className="relative z-10 m-6 mt-auto bg-[#0B192C]/90 backdrop-blur-md border border-[#D4AF37]/40 rounded-xl p-6 space-y-4 text-white shadow-2xl">
            <div className="flex items-center gap-2">
              <span className="px-3 py-1 bg-[#0F2537] text-[#D4AF37] border border-[#D4AF37]/40 text-[11px] font-extrabold uppercase tracking-wider rounded-xl">
                Global Freight Solutions
              </span>
              <span className="px-3 py-1 bg-[#0F2537] text-slate-200 border border-slate-700 text-[11px] font-bold uppercase tracking-wider rounded-xl">
                CDLP Platform
              </span>
            </div>

            <p className="text-sm sm:text-base font-semibold text-slate-100 leading-relaxed">
              &ldquo;ShipRule CDLP provides instant, anti-hallucination customs clearance intelligence across international maritime trade routes.&rdquo;
            </p>

            <div className="pt-1 border-t border-slate-800 flex items-center justify-between text-xs">
              <div>
                <p className="font-extrabold text-white uppercase">Grounded RAG Intelligence</p>
                <p className="text-slate-400 text-[11px]">Zero-Hallucination Tariff Engine</p>
              </div>
              <div className="flex items-center gap-1 text-[#D4AF37] font-bold">
                <span>4.8 ★★★★★</span>
              </div>
            </div>
          </div>

        </div>
      </div>

    </div>
  );
}
