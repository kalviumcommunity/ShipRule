'use client';

import React, { useState } from 'react';
import { Settings, Lock, Key, AlertCircle, CheckCircle2 } from 'lucide-react';
import { apiService, ApiError } from '@/services/api';
import { useAuth } from '@/context/AuthContext';

export default function SettingsPage() {
  const { user } = useAuth();

  // Change password states
  const [oldPassword, setOldPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [pwdLoading, setPwdLoading] = useState(false);
  const [pwdSuccess, setPwdSuccess] = useState<string | null>(null);
  const [pwdError, setPwdError] = useState<string | null>(null);

  const handlePasswordChange = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!oldPassword || !newPassword || !confirmPassword) {
      setPwdError('Please fill in all password fields.');
      return;
    }
    if (newPassword !== confirmPassword) {
      setPwdError('New password and confirm password do not match.');
      return;
    }
    if (newPassword.length < 4) {
      setPwdError('New password must be at least 4 characters long.');
      return;
    }

    setPwdLoading(true);
    setPwdError(null);
    setPwdSuccess(null);

    try {
      const res = await apiService.changePassword(oldPassword, newPassword);
      setPwdSuccess(res.message || 'Password updated successfully.');
      setOldPassword('');
      setNewPassword('');
      setConfirmPassword('');
    } catch (err: any) {
      if (err instanceof ApiError) {
        setPwdError(err.detail);
      } else {
        setPwdError(err.message || 'Failed to update password.');
      }
    } finally {
      setPwdLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8 py-8 px-4 text-slate-100">
      <div>
        <h1 className="text-2xl sm:text-3xl font-black text-white flex items-center gap-3 font-sans uppercase">
          <Settings className="h-7 w-7 text-[#D4AF37]" />
          Account &amp; Security Settings
        </h1>
        <p className="mt-1 text-xs sm:text-sm text-slate-300">
          Manage your account credentials and security preferences.
        </p>
      </div>

      {/* Account & Change Password Section */}
      <div className="rounded-2xl border border-[#D4AF37]/30 bg-[#0F2537] p-6 space-y-6 shadow-xl">
        <div className="flex items-center justify-between border-b border-[#D4AF37]/20 pb-4">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-[#0B192C] border border-[#D4AF37]/40 flex items-center justify-center text-[#D4AF37]">
              <Lock className="h-5 w-5" />
            </div>
            <div>
              <h3 className="font-extrabold text-white text-base uppercase font-sans">Account Security &amp; Password</h3>
              <p className="text-xs text-slate-300">{user?.email || 'Logged in user'}</p>
            </div>
          </div>
        </div>

        {pwdSuccess && (
          <div className="flex items-center gap-2 p-3.5 rounded-xl bg-emerald-950 border border-emerald-700 text-xs font-bold text-emerald-400 font-mono">
            <CheckCircle2 className="h-4 w-4 text-emerald-400" />
            <span>{pwdSuccess}</span>
          </div>
        )}

        {pwdError && (
          <div className="flex items-center gap-2 p-3.5 rounded-xl bg-rose-950 border border-rose-800 text-xs font-bold text-rose-300 font-mono">
            <AlertCircle className="h-4 w-4 text-rose-400" />
            <span>{pwdError}</span>
          </div>
        )}

        <form onSubmit={handlePasswordChange} className="space-y-4 max-w-md">
          <div className="space-y-1.5">
            <label className="text-xs font-bold uppercase tracking-wider text-slate-300">Current Password</label>
            <div className="relative">
              <Key className="absolute left-3.5 top-3 h-4 w-4 text-[#D4AF37]" />
              <input
                type="password"
                required
                value={oldPassword}
                onChange={(e) => setOldPassword(e.target.value)}
                placeholder="Enter current password"
                className="w-full border border-slate-700 bg-[#0B192C] py-2.5 pl-10 pr-4 text-xs text-white placeholder-slate-500 focus:border-[#D4AF37] focus:outline-none rounded-xl"
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-bold uppercase tracking-wider text-slate-300">New Password</label>
            <div className="relative">
              <Lock className="absolute left-3.5 top-3 h-4 w-4 text-[#D4AF37]" />
              <input
                type="password"
                required
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                placeholder="Enter new password (min 4 characters)"
                className="w-full border border-slate-700 bg-[#0B192C] py-2.5 pl-10 pr-4 text-xs text-white placeholder-slate-500 focus:border-[#D4AF37] focus:outline-none rounded-xl"
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-bold uppercase tracking-wider text-slate-300">Confirm New Password</label>
            <div className="relative">
              <Lock className="absolute left-3.5 top-3 h-4 w-4 text-[#D4AF37]" />
              <input
                type="password"
                required
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder="Confirm new password"
                className="w-full border border-slate-700 bg-[#0B192C] py-2.5 pl-10 pr-4 text-xs text-white placeholder-slate-500 focus:border-[#D4AF37] focus:outline-none rounded-xl"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={pwdLoading}
            className="flex items-center gap-2 bg-[#D4AF37] hover:bg-[#C59E2B] px-6 py-2.5 text-xs font-extrabold uppercase tracking-wider text-[#0B192C] transition-all disabled:opacity-50 border border-[#D4AF37] rounded-xl shadow-md cursor-pointer"
          >
            {pwdLoading ? 'Updating Password...' : 'Update Password'}
          </button>
        </form>
      </div>
    </div>
  );
}
