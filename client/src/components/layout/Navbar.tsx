'use client';

import React, { useEffect, useState, useRef } from 'react';
import Link from 'next/link';
import { useRouter, usePathname } from 'next/navigation';
import { Anchor, Activity, CheckCircle, XCircle, User, Shield, LogOut, LogIn, Menu, Settings, ChevronDown } from 'lucide-react';
import { apiService } from '@/services/api';
import { useAuth } from '@/context/AuthContext';
import { HealthResponse } from '@/types/api';

export function Navbar() {
  const { user, role, logout } = useAuth();
  const router = useRouter();
  const pathname = usePathname();
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [profileOpen, setProfileOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement | null>(null);

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

  // Close profile dropdown on click outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setProfileOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Close profile dropdown on route change
  useEffect(() => {
    setProfileOpen(false);
  }, [pathname]);

  const handleLogout = () => {
    setProfileOpen(false);
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

        {/* Profile Icon Dropdown containing Email, Settings, & Logout */}
        {user ? (
          <div className="relative" ref={dropdownRef}>
            <button
              onClick={() => setProfileOpen(!profileOpen)}
              className="flex items-center gap-2 border border-[#D4AF37]/50 bg-[#0F2537] hover:border-[#D4AF37] p-2 sm:px-3 sm:py-1.5 text-xs font-semibold text-slate-200 transition rounded-xl shadow-md cursor-pointer group"
              title="Account Menu"
            >
              <div className="h-7 w-7 rounded-full bg-[#0B192C] border border-[#D4AF37] flex items-center justify-center text-[#D4AF37] group-hover:scale-105 transition">
                <User className="h-4 w-4" />
              </div>
              <ChevronDown className={`h-3.5 w-3.5 text-[#D4AF37] transition-transform duration-200 ${profileOpen ? 'rotate-180' : ''}`} />
            </button>

            {/* Dropdown Menu */}
            {profileOpen && (
              <div className="absolute right-0 top-12 w-64 bg-[#0F2537] border border-[#D4AF37]/40 shadow-2xl rounded-2xl p-4 space-y-3 z-50 animate-in fade-in zoom-in duration-150">
                
                {/* User Info Header with Email */}
                <div className="border-b border-[#D4AF37]/20 pb-3 space-y-1">
                  <div className="flex items-center gap-2">
                    <User className="h-4 w-4 text-[#D4AF37] flex-shrink-0" />
                    <p className="text-xs font-bold text-white truncate font-mono">{user.email}</p>
                  </div>
                  <span className="inline-block px-2 py-0.5 bg-[#0B192C] border border-[#D4AF37]/30 text-[#D4AF37] font-bold text-[9px] uppercase tracking-wider rounded-lg">
                    {role === 'admin' ? 'Master Admin' : 'User Account'}
                  </span>
                </div>

                {/* Dropdown Navigation Links */}
                <div className="space-y-1 text-xs font-medium">
                  <Link
                    href="/settings"
                    onClick={() => setProfileOpen(false)}
                    className="flex items-center gap-2.5 px-3 py-2 text-slate-200 hover:text-[#D4AF37] hover:bg-[#0B192C] rounded-xl transition font-sans"
                  >
                    <Settings className="h-4 w-4 text-[#D4AF37]" />
                    <span>Settings &amp; Change Password</span>
                  </Link>

                  {role === 'admin' && (
                    <Link
                      href="/master"
                      onClick={() => setProfileOpen(false)}
                      className="flex items-center gap-2.5 px-3 py-2 text-[#D4AF37] hover:bg-[#0B192C] rounded-xl transition font-sans font-bold"
                    >
                      <Shield className="h-4 w-4 text-[#D4AF37]" />
                      <span>Master Admin Dashboard</span>
                    </Link>
                  )}
                </div>

                {/* Logout Button */}
                <div className="border-t border-slate-700/60 pt-2">
                  <button
                    onClick={handleLogout}
                    className="w-full flex items-center gap-2.5 px-3 py-2 text-rose-300 hover:text-white hover:bg-rose-950/50 text-xs font-bold rounded-xl transition cursor-pointer"
                  >
                    <LogOut className="h-4 w-4 text-rose-400" />
                    <span>Log Out Session</span>
                  </button>
                </div>

              </div>
            )}
          </div>
        ) : (
          <Link
            href="/login"
            className="flex items-center gap-1.5 border border-slate-700 bg-[#0F2537] px-4 py-2 text-xs font-bold text-slate-200 hover:border-[#D4AF37] hover:text-[#D4AF37] transition rounded-xl"
          >
            <LogIn className="h-3.5 w-3.5" /> Sign In
          </Link>
        )}

        {/* Primary CTA Button */}
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
