'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/context/AuthContext';

export default function DocumentsPage() {
  const router = useRouter();
  const { role, loading } = useAuth();

  useEffect(() => {
    if (!loading) {
      if (role === 'admin') {
        router.replace('/master');
      } else {
        router.replace('/query');
      }
    }
  }, [role, loading, router]);

  return (
    <div className="flex h-64 items-center justify-center text-xs text-slate-500 font-mono">
      Redirecting...
    </div>
  );
}
