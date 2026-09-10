'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Mail, Lock, User, AlertCircle, ArrowRight, UserPlus, LogIn, Ship } from 'lucide-react';
import { useAuth } from '@/context/AuthContext';
import { ApiError } from '@/services/api';

export default function LoginPage() {
  const router = useRouter();
  const { login, signup } = useAuth();
  const [activeTab, setActiveTab] = useState<'signin' | 'signup'>('signin');

  // Form states (no pre-filled hardcoded credentials)
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleUserLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      await login(email, password);
      router.push('/query');
    } catch (err: any) {
      if (err instanceof ApiError) {
        setError(err.detail);
      } else {
        setError(err.message || 'Login failed. Please check your credentials.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleUserSignup = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      await signup(email, password, fullName);
      router.push('/query');
    } catch (err: any) {
      if (err instanceof ApiError) {
        setError(err.detail);
      } else {
        setError(err.message || 'Signup failed. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-[calc(100vh-8rem)] items-center justify-center p-6 bg-[#0B192C] text-slate-100">
      <div className="w-full max-w-md space-y-6">
        
        {/* Header Branding */}
        <div className="text-center space-y-2 flex flex-col items-center">
          <img src="/logo.png" alt="ShipRule Logo" className="h-12 w-auto object-contain mb-1" />
          <h1 className="text-xl font-extrabold uppercase tracking-tight text-white font-sans">
            Authentication Portal
          </h1>
          <p className="text-xs text-slate-300">
            Sign in to your account or register a new user profile
          </p>
        </div>

        {/* Tab Switcher (Sign In vs Sign Up - No Admin Info) */}
        <div className="grid grid-cols-2 bg-[#0F2537] border border-[#D4AF37]/30 text-xs font-bold uppercase tracking-wider rounded-xl p-1">
          <button
            onClick={() => { setActiveTab('signin'); setError(null); }}
            className={`flex items-center justify-center gap-2 py-3 transition rounded-xl ${
              activeTab === 'signin'
                ? 'bg-[#D4AF37] text-[#0B192C] font-black'
                : 'text-slate-300 hover:text-[#D4AF37]'
            }`}
          >
            <LogIn className="h-3.5 w-3.5" /> Sign In
          </button>
          <button
            onClick={() => { setActiveTab('signup'); setError(null); }}
            className={`flex items-center justify-center gap-2 py-3 transition rounded-xl ${
              activeTab === 'signup'
                ? 'bg-[#D4AF37] text-[#0B192C] font-black'
                : 'text-slate-300 hover:text-[#D4AF37]'
            }`}
          >
            <UserPlus className="h-3.5 w-3.5" /> Sign Up
          </button>
        </div>

        {/* Error Notification */}
        {error && (
          <div className="flex items-center gap-2.5 border border-rose-800 bg-[#1e1015] p-3 text-xs text-rose-300 rounded-xl">
            <AlertCircle className="h-4 w-4 text-rose-400 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Sign In Form */}
        {activeTab === 'signin' && (
          <form onSubmit={handleUserLogin} className="border border-[#D4AF37]/30 bg-[#0F2537] p-6 space-y-4 shadow-xl rounded-xl">
            <div className="space-y-1.5">
              <label className="text-xs font-bold uppercase text-slate-300">Email Address</label>
              <div className="relative">
                <Mail className="absolute left-3.5 top-3 h-4 w-4 text-slate-400" />
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="user@company.com"
                  className="w-full border border-slate-700 bg-[#0B192C] py-2.5 pl-10 pr-4 text-xs text-white placeholder-slate-500 focus:border-[#D4AF37] focus:outline-none rounded-xl"
                />
              </div>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-bold uppercase text-slate-300">Password</label>
              <div className="relative">
                <Lock className="absolute left-3.5 top-3 h-4 w-4 text-slate-400" />
                <input
                  type="password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full border border-slate-700 bg-[#0B192C] py-2.5 pl-10 pr-4 text-xs text-white placeholder-slate-500 focus:border-[#D4AF37] focus:outline-none rounded-xl"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 bg-[#D4AF37] hover:bg-[#C59E2B] py-3 text-xs font-black uppercase tracking-wider text-[#0B192C] transition shadow-md disabled:opacity-50 border border-[#D4AF37] rounded-xl"
            >
              {loading ? 'Authenticating...' : 'Sign In'} <ArrowRight className="h-3.5 w-3.5" />
            </button>
          </form>
        )}

        {/* Sign Up Form (MongoDB User Registration) */}
        {activeTab === 'signup' && (
          <form onSubmit={handleUserSignup} className="border border-[#D4AF37]/30 bg-[#0F2537] p-6 space-y-4 shadow-xl rounded-xl">
            <div className="space-y-1.5">
              <label className="text-xs font-bold uppercase text-slate-300">Full Name</label>
              <div className="relative">
                <User className="absolute left-3.5 top-3 h-4 w-4 text-slate-400" />
                <input
                  type="text"
                  required
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  placeholder="Full Name"
                  className="w-full border border-slate-700 bg-[#0B192C] py-2.5 pl-10 pr-4 text-xs text-white placeholder-slate-500 focus:border-[#D4AF37] focus:outline-none rounded-xl"
                />
              </div>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-bold uppercase text-slate-300">Email Address</label>
              <div className="relative">
                <Mail className="absolute left-3.5 top-3 h-4 w-4 text-slate-400" />
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="user@company.com"
                  className="w-full border border-slate-700 bg-[#0B192C] py-2.5 pl-10 pr-4 text-xs text-white placeholder-slate-500 focus:border-[#D4AF37] focus:outline-none rounded-xl"
                />
              </div>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-bold uppercase text-slate-300">Password</label>
              <div className="relative">
                <Lock className="absolute left-3.5 top-3 h-4 w-4 text-slate-400" />
                <input
                  type="password"
                  required
                  minLength={4}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full border border-slate-700 bg-[#0B192C] py-2.5 pl-10 pr-4 text-xs text-white placeholder-slate-500 focus:border-[#D4AF37] focus:outline-none rounded-xl"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 bg-[#D4AF37] hover:bg-[#C59E2B] py-3 text-xs font-black uppercase tracking-wider text-[#0B192C] transition shadow-md disabled:opacity-50 border border-[#D4AF37] rounded-xl"
            >
              {loading ? 'Creating User Profile...' : 'Register Account'} <UserPlus className="h-3.5 w-3.5" />
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
