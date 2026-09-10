import { QueryInterface } from '@/components/chat/QueryInterface';

export default function QueryPage() {
  return (
    <div className="w-full bg-[#0B192C] min-h-screen py-12 px-6 lg:px-12 text-slate-100">
      <div className="max-w-6xl mx-auto space-y-6">
        <div className="border-b border-[#D4AF37]/30 pb-4">
          <span className="border border-[#D4AF37] bg-[#0F2537] px-3 py-1 text-xs font-black uppercase tracking-widest text-[#D4AF37]">
            RAG Query Engine
          </span>
          <h1 className="text-3xl font-black uppercase text-white tracking-tight mt-2 font-sans">
            Customs & Shipping Rules Lookup
          </h1>
        </div>
        <QueryInterface />
      </div>
    </div>
  );
}
