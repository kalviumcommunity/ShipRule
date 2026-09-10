'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { LayoutDashboard, MessageSquare, FileText, Settings, Shield, ShieldCheck, Key } from 'lucide-react';
import { useAuth } from '@/context/AuthContext';
import { cn } from '@/lib/utils';

export function Sidebar() {
  const pathname = usePathname();
  const { role } = useAuth();

  const userNavigation = [
    { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
    { name: 'Ask ShipRule', href: '/query', icon: MessageSquare },
    { name: 'System Info', href: '/settings', icon: Settings },
  ];

  return (
    <aside className="w-64 flex-shrink-0 border-r border-slate-800 bg-slate-950 p-4 min-h-[calc(100vh-4rem)] flex flex-col justify-between">
      <div className="space-y-6">
        <div className="px-3 text-xs font-semibold uppercase tracking-wider text-slate-500">
          User Navigation
        </div>
        <nav className="space-y-1.5">
          {userNavigation.map((item) => {
            const isActive = pathname === item.href || (pathname === '/' && item.href === '/dashboard');
            const Icon = item.icon;
            return (
              <Link
                key={item.name}
                href={item.href}
                className={cn(
                  'flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all duration-150',
                  isActive
                    ? 'bg-gradient-to-r from-cyan-500/10 to-blue-500/10 text-cyan-400 border border-cyan-500/30 shadow-sm shadow-cyan-500/10'
                    : 'text-slate-400 hover:bg-slate-900 hover:text-slate-200'
                )}
              >
                <Icon className={cn('h-4 w-4', isActive ? 'text-cyan-400' : 'text-slate-500')} />
                {item.name}
              </Link>
            );
          })}
        </nav>

        {/* Master Admin Panel Link Section - ONLY shown if authenticated as Admin */}
        {role === 'admin' && (
          <div className="pt-4 border-t border-slate-800 space-y-2">
            <div className="px-3 text-xs font-semibold uppercase tracking-wider text-slate-500 flex items-center justify-between">
              <span>Administration</span>
              <span className="text-[10px] text-amber-400 font-mono font-bold">ADMIN</span>
            </div>

            <Link
              href="/master"
              className={cn(
                'flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all duration-150',
                pathname === '/master'
                  ? 'bg-gradient-to-r from-amber-500/10 to-orange-500/10 text-amber-400 border border-amber-500/30 shadow-sm shadow-amber-500/10'
                  : 'text-slate-400 hover:bg-slate-900 hover:text-amber-300'
              )}
            >
              <Key className={cn('h-4 w-4', pathname === '/master' ? 'text-amber-400' : 'text-slate-500')} />
              Master Admin Panel
            </Link>
          </div>
        )}
      </div>

      <div className="rounded-xl border border-slate-800/80 bg-slate-900/40 p-4 space-y-2">
        <div className="flex items-center gap-2 text-xs font-medium text-cyan-400">
          <ShieldCheck className="h-4 w-4" /> Grounded RAG Active
        </div>
        <p className="text-xs text-slate-400 leading-relaxed">
          Grounded answer validation & citation auditing enabled on all queries.
        </p>
      </div>
    </aside>
  );
}
