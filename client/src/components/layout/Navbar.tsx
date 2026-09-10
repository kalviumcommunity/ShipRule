'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter, usePathname } from 'next/navigation';
import { Anchor, Activity, CheckCircle, XCircle, User, Shield, LogOut, LogIn, Menu } from 'lucide-react';
import { apiService } from '@/services/api';
import { useAuth } from '@/context/AuthContext';
import { HealthResponse } from '@/types/api';

export function Navbar() {
  const { user, role, logout } = useAuth();
  const router = useRouter();
  const pathname = usePathname();
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function check() {
      try {
        const res = await apiService.checkHealth();
        setHealth(res);
      } catch {
        setHealth(null);
      } finally {
        setLoading(false);
      }
    }
    check();
    const interval = setInterval(check, 15000);
    return () => clearInterval(interval);
  }, []);

  const handleLogout = () => {
    logout();
    router.push('/login');
  };

  const navLinks = [
    { name: 'Home', href: '/' },
    { name: 'Services', href: '/#services' },
    { name: 'Track & Query', href: '/query' },
    { name: 'Scope & Overview', href: '/#overview' },
    { name: 'About', href: '/#about' },
    { name: 'Contact', href: '/#contact' },
  ];

  return (
    <header className="sticky top-0 z-50 flex h-20 w-full items-center justify-between border-b border-[#D4AF37]/30 bg-[#0B192C] px-8 text-white shadow-xl">
      {/* Brand / Logo */}
      <Link href="/" className="flex items-center gap-3 group">
        <img
          src="/logo.png"
          alt="ShipRule Logo"
          className="h-10 w-auto object-contain"
        />
        <span className="hidden lg:inline-block border border-[#D4AF37]/40 bg-[#0F2537] px-2 py-0.5 text-[10px] font-semibold tracking-wider text-[#D4AF37] uppercase rounded-xl">
          CDLP Platform
        </span>
      </Link>

      {/* Center Navigation Links (Image 1 Style) */}
      <nav className="hidden md:flex items-center gap-8 text-sm font-medium text-slate-300">
        {navLinks.map((link) => {
          const isActive = pathname === link.href;
          return (
            <Link
              key={link.name}
              href={link.href}
              className={`transition-colors duration-150 py-1 border-b-2 uppercase tracking-wider text-xs font-bold ${isActive
                  ? 'border-[#D4AF37] text-[#D4AF37]'
                  : 'border-transparent text-slate-300 hover:text-[#D4AF37] hover:border-[#D4AF37]/50'
                }`}
            >
              {link.name}
            </Link>
          );
        })}
      </nav>

      {/* Right Actions & Auth */}
      <div className="flex items-center gap-4">
        {/* API Status Badge */}
        <div className="hidden xl:flex items-center gap-2 border border-[#28693C]/60 bg-[#0F2537] px-3 py-1 text-xs rounded-xl">
          <Activity className="h-3.5 w-3.5 text-[#D4AF37]" />
          <span className="text-slate-400">API Status:</span>
          {loading ? (
            <span className="text-[#D4AF37] animate-pulse">Checking...</span>
          ) : health?.status === 'healthy' ? (
            <span className="flex items-center gap-1 font-semibold text-emerald-400">
              <CheckCircle className="h-3 w-3" /> Online
            </span>
          ) : (
            <span className="flex items-center gap-1 font-semibold text-rose-400">
              <XCircle className="h-3 w-3" /> Offline
            </span>
          )}
        </div>

        {/* Auth User Info & Actions */}
        {user ? (
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 border border-[#D4AF37]/30 bg-[#0F2537] px-3 py-1.5 text-xs rounded-xl">
              {role === 'admin' ? (
                <span className="flex items-center gap-1.5 font-bold text-[#D4AF37]">
                  <Shield className="h-3.5 w-3.5 text-[#D4AF37]" /> MASTER ADMIN
                </span>
              ) : (
                <span className="flex items-center gap-1.5 font-medium text-slate-200">
                  <User className="h-3.5 w-3.5 text-[#D4AF37]" /> {user.email}
                </span>
              )}
            </div>

            <button
              onClick={handleLogout}
              className="flex items-center gap-1 border border-slate-700 bg-[#0F2537] px-3 py-1.5 text-xs font-semibold text-slate-300 hover:bg-slate-800 hover:text-white transition rounded-xl"
              title="Log out"
            >
              <LogOut className="h-3.5 w-3.5" /> Logout
            </button>
          </div>
        ) : (
          <Link
            href="/login"
            className="flex items-center gap-1.5 border border-slate-700 bg-[#0F2537] px-4 py-2 text-xs font-bold text-slate-200 hover:border-[#D4AF37] hover:text-[#D4AF37] transition rounded-xl"
          >
            <LogIn className="h-3.5 w-3.5" /> Sign In
          </Link>
        )}

        {/* Primary CTA Button (Image 1 "Get a Quote" / "Submit Query") */}
        <Link
          href="/query"
          className="bg-[#D4AF37] hover:bg-[#C59E2B] text-[#0B192C] px-5 py-2.5 text-xs font-extrabold uppercase tracking-wider transition-all shadow-md flex items-center gap-2 border border-[#D4AF37] rounded-xl"
        >
          Get a Quote
        </Link>
      </div>
    </header>
  );
}

