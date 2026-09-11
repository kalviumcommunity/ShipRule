'use client';

import React from 'react';
import Image from 'next/image';
import Link from 'next/link';
import {
  Anchor,
  Search,
  Sparkles,
  ShieldCheck,
  BookOpen,
  ArrowRight,
  CheckCircle2,
  Globe2,
  Ship,
  FileText,
  Lock,
  Cpu,
  Layers,
  PhoneCall,
  Mail,
  MapPin,
  Star,
  Activity
} from 'lucide-react';
import { QueryInterface } from '@/components/chat/QueryInterface';

export default function Home() {
  return (
    <div className="w-full bg-[#0B192C] text-slate-100 min-h-screen">
      {/* ==================================================================== */}
      {/* 1. HERO BANNER SECTION (Image 1 Layout + Image 2 Container Ship) */}
      {/* ==================================================================== */}
      <section className="relative w-full border-b border-[#D4AF37]/30 bg-[#0B192C] pt-8 pb-16 px-6 lg:px-12 overflow-hidden">
        <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">

          {/* Left Hero Content */}
          <div className="lg:col-span-6 space-y-6 z-10">
            <div className="inline-flex items-center gap-2 border border-[#D4AF37] bg-[#0F2537] px-3 py-1 text-xs font-black uppercase tracking-widest text-[#D4AF37]">
              <Ship className="h-4 w-4 text-[#D4AF37]" /> Global Freight Solutions & CDLP Platform
            </div>

            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-black tracking-tight text-white uppercase font-sans leading-none">
              Global <br />
              <span className="text-white">Freight</span> <br />
              <span className="text-[#D4AF37]">Solutions</span>
            </h1>

            <p className="text-sm sm:text-base text-slate-300 leading-relaxed font-sans max-w-xl">
              Global Freight Solutions is a trusted provider of fast, reliable, anti-hallucination customs clearance, tariff compliance, and container transportation services across international markets.
            </p>

            {/* Action Buttons (12px Border Radius) */}
            <div className="pt-2 flex flex-wrap items-center gap-4">
              <Link
                href="/query"
                className="bg-[#D4AF37] hover:bg-[#C59E2B] text-[#0B192C] px-8 py-3.5 text-xs font-black uppercase tracking-wider transition-all border border-[#D4AF37] shadow-lg flex items-center gap-2 rounded-xl"
              >
                Get a Quote <ArrowRight className="h-4 w-4" />
              </Link>
              <a
                href="#services"
                className="border border-[#D4AF37]/50 bg-[#0F2537] hover:bg-[#162A45] hover:border-[#D4AF37] text-white px-8 py-3.5 text-xs font-bold uppercase tracking-wider transition-all rounded-xl"
              >
                Track Shipment / Scope
              </a>
            </div>

            {/* Mini Unloading Banner */}
            <div className="mt-8 border border-[#D4AF37]/30 bg-[#0F2537] p-4 max-w-sm flex items-center justify-between rounded-xl">
              <div>
                <span className="text-[10px] font-extrabold uppercase tracking-widest text-[#D4AF37] block">UNLOADING</span>
                <h4 className="text-sm font-bold text-white uppercase mt-0.5">Book Container Unloading Today</h4>
                <a href="#query-engine" className="text-xs text-[#D4AF37] hover:underline font-bold mt-1 inline-block">
                  Book Now &rarr;
                </a>
              </div>
              <div className="h-10 w-10 border border-[#D4AF37]/40 bg-[#0B192C] flex items-center justify-center text-[#D4AF37] rounded-xl">
                <Ship className="h-5 w-5" />
              </div>
            </div>
          </div>

          {/* Right Hero Image (Image 2 Container Ship Asset) */}
          <div className="lg:col-span-6 relative z-10">
            <div className="relative border-2 border-[#D4AF37]/40 shadow-2xl bg-[#0F2537] rounded-xl overflow-hidden">
              {/* Image 2 Asset: Cargo Container Ship at Sea */}
              <div className="relative w-full h-[380px] sm:h-[450px]">
                <Image
                  src="/ship_hero.png"
                  alt="Global Freight Solutions Container Ship"
                  fill
                  sizes="(max-width: 1200px) 100vw, 50vw"
                  priority
                  className="object-cover"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-[#0B192C] via-transparent to-transparent opacity-40" />
              </div>

              {/* Floating Rating Widget */}
              <div className="absolute top-6 right-6 border border-slate-700 bg-[#0B192C]/95 p-4 max-w-xs shadow-xl backdrop-blur-md rounded-xl">
                <div className="flex items-center gap-2 mb-2">
                  <div className="flex -space-x-2">
                    <div className="h-7 w-7 border border-[#D4AF37] bg-[#0F2537] text-[10px] font-bold text-[#D4AF37] flex items-center justify-center rounded-full">US</div>
                    <div className="h-7 w-7 border border-[#D4AF37] bg-[#0F2537] text-[10px] font-bold text-[#D4AF37] flex items-center justify-center rounded-full">EU</div>
                    <div className="h-7 w-7 border border-[#D4AF37] bg-[#0F2537] text-[10px] font-bold text-[#D4AF37] flex items-center justify-center rounded-full">IN</div>
                  </div>
                  <span className="text-[11px] font-bold text-[#D4AF37] underline cursor-pointer">Leave a Review</span>
                </div>
                <p className="text-[11px] text-slate-300 leading-tight">
                  Trusted by businesses of all sizes. See why enterprise logistics trust our grounded answer services.
                </p>
                <div className="mt-2 flex items-baseline gap-1">
                  <span className="text-2xl font-black text-white">4.8</span>
                  <span className="text-[#D4AF37]">★</span>
                </div>
              </div>

              {/* Floating Route Tracker Widget */}
              <div className="absolute bottom-6 left-6 right-6 border border-[#D4AF37]/40 bg-[#0B192C]/95 p-4 shadow-xl backdrop-blur-md rounded-xl">
                <div className="flex items-center justify-between text-xs font-mono border-b border-slate-700 pb-2 mb-2">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-white">CN SHG</span>
                    <span className="text-[#D4AF37]">&bull;</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-[#D4AF37]">&bull;</span>
                    <span className="font-bold text-white">US OAK</span>
                  </div>
                </div>
                <div className="flex items-center justify-between text-[11px] font-mono text-slate-400">
                  <div>
                    <span className="block text-[9px] uppercase tracking-wider text-slate-500">ATD</span>
                    <span className="font-bold text-slate-200">May 3 22:57</span>
                  </div>
                  <div className="text-center font-bold text-[#D4AF37]">
                    &rarr; IN TRANSIT &rarr;
                  </div>
                  <div className="text-right">
                    <span className="block text-[9px] uppercase tracking-wider text-slate-500">ETA</span>
                    <span className="font-bold text-slate-200">May 5 09:00</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

        </div>
      </section>

      {/* ==================================================================== */}
      {/* 2. GROUNDED RAG QUERY ENGINE SECTION */}
      {/* ==================================================================== */}
      <section id="query-engine" className="w-full border-b border-[#D4AF37]/30 bg-[#0B192C] py-16 px-6 lg:px-12">
        <div className="max-w-6xl mx-auto space-y-8">
          <div className="text-center max-w-3xl mx-auto space-y-3">
            <span className="border border-[#D4AF37] bg-[#0F2537] px-3 py-1 text-xs font-black uppercase tracking-widest text-[#D4AF37] inline-block">
              Interactive Grounded Lookup
            </span>
            <h2 className="text-3xl sm:text-4xl font-black uppercase tracking-tight text-white font-sans">
              Customs Duty & Shipping Documentation Query Engine
            </h2>
            <p className="text-sm text-slate-300 leading-relaxed font-sans">
              Enter any query regarding import tariff rates, mandatory shipping documents, Incoterms, statutory licenses, or compliance regulations.
            </p>
          </div>

          {/* RAG Query Interface Component */}
          <QueryInterface />
        </div>
      </section>

      {/* ==================================================================== */}
      {/* 3. SCOPE & SERVICES OVERVIEW SECTION */}
      {/* ==================================================================== */}
      <section id="services" className="w-full border-b border-[#D4AF37]/30 bg-[#0F2537] py-16 px-6 lg:px-12">
        <div className="max-w-6xl mx-auto space-y-12">
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 border-b border-[#D4AF37]/20 pb-6">
            <div>
              <span className="text-xs font-extrabold uppercase tracking-widest text-[#D4AF37] block mb-2">
                Supported Domain & Services
              </span>
              <h2 className="text-3xl font-black uppercase tracking-tight text-white font-sans">
                Comprehensive Logistics Scope & Services
              </h2>
            </div>
            <p className="text-xs text-slate-300 max-w-md">
              ShipRule CDLP processes multi-departmental customs regulations and shipping rules with zero tolerance for out-of-domain hallucinations.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="border border-slate-700 bg-[#0B192C] p-6 space-y-4 hover:border-[#D4AF37] transition">
              <div className="h-10 w-10 border border-[#D4AF37] bg-[#0F2537] text-[#D4AF37] flex items-center justify-center font-bold">
                01
              </div>
              <h3 className="text-lg font-bold text-white uppercase font-sans">Customs Duty & Tariff Schedule</h3>
              <p className="text-xs text-slate-300 leading-relaxed">
                Retrieve precise tariff schedules, effective basic customs duties (BCD), integrated GST (IGST), social welfare surcharges, and anti-dumping duties.
              </p>
              <ul className="text-xs text-slate-400 space-y-1 font-mono pt-2 border-t border-slate-800">
                <li className="flex items-center gap-1.5"><CheckCircle2 className="h-3.5 w-3.5 text-[#D4AF37]" /> HS Code Classification</li>
                <li className="flex items-center gap-1.5"><CheckCircle2 className="h-3.5 w-3.5 text-[#D4AF37]" /> Tariff Duty Calculator</li>
              </ul>
            </div>

            <div className="border border-slate-700 bg-[#0B192C] p-6 space-y-4 hover:border-[#D4AF37] transition">
              <div className="h-10 w-10 border border-[#D4AF37] bg-[#0F2537] text-[#D4AF37] flex items-center justify-center font-bold">
                02
              </div>
              <h3 className="text-lg font-bold text-white uppercase font-sans">Mandatory Shipping Documentation</h3>
              <p className="text-xs text-slate-300 leading-relaxed">
                Complete compliance requirements for Bills of Lading (B/L), Commercial Invoices, Packing Lists, Certificates of Origin, and Bill of Entry (BOE).
              </p>
              <ul className="text-xs text-slate-400 space-y-1 font-mono pt-2 border-t border-slate-800">
                <li className="flex items-center gap-1.5"><CheckCircle2 className="h-3.5 w-3.5 text-[#D4AF37]" /> Import Clearance Documents</li>
                <li className="flex items-center gap-1.5"><CheckCircle2 className="h-3.5 w-3.5 text-[#D4AF37]" /> Statutory Registrations</li>
              </ul>
            </div>

            <div className="border border-slate-700 bg-[#0B192C] p-6 space-y-4 hover:border-[#D4AF37] transition">
              <div className="h-10 w-10 border border-[#D4AF37] bg-[#0F2537] text-[#D4AF37] flex items-center justify-center font-bold">
                03
              </div>
              <h3 className="text-lg font-bold text-white uppercase font-sans">Incoterms & Risk Allocation</h3>
              <p className="text-xs text-slate-300 leading-relaxed">
                Clear legal definitions of obligations under CIF (Cost, Insurance, and Freight), FOB (Free on Board), EXW (Ex Works), and DDP (Delivered Duty Paid).
              </p>
              <ul className="text-xs text-slate-400 space-y-1 font-mono pt-2 border-t border-slate-800">
                <li className="flex items-center gap-1.5"><CheckCircle2 className="h-3.5 w-3.5 text-[#D4AF37]" /> Freight Risk Liability</li>
                <li className="flex items-center gap-1.5"><CheckCircle2 className="h-3.5 w-3.5 text-[#D4AF37]" /> Port Clearance Protocols</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* ==================================================================== */}
      {/* 4. OVERVIEW & SYSTEM ARCHITECTURE SECTION */}
      {/* ==================================================================== */}
      <section id="overview" className="w-full border-b border-[#D4AF37]/30 bg-[#0B192C] py-16 px-6 lg:px-12">
        <div className="max-w-6xl mx-auto space-y-12">
          <div className="text-center max-w-3xl mx-auto space-y-3">
            <span className="border border-[#D4AF37] bg-[#0F2537] px-3 py-1 text-xs font-black uppercase tracking-widest text-[#D4AF37] inline-block">
              Anti-Hallucination Guardrails
            </span>
            <h2 className="text-3xl font-black uppercase tracking-tight text-white font-sans">
              System Architecture & Grounded Safeguards
            </h2>
            <p className="text-sm text-slate-300 leading-relaxed font-sans">
              Built on dense semantic vector embeddings, cross-encoder re-ranking, and formal decision guardrails.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="border border-slate-800 bg-[#0F2537] p-5 space-y-3">
              <div className="h-8 w-8 bg-[#D4AF37] text-[#0B192C] flex items-center justify-center font-bold">
                <Globe2 className="h-4 w-4" />
              </div>
              <h4 className="font-bold text-white uppercase text-sm font-sans">Universal Semantic Similarity</h4>
              <p className="text-xs text-slate-300 leading-relaxed">
                Normalizes user terminology, abbreviations, and informal phrasing without relying on hardcoded synonym lists.
              </p>
            </div>

            <div className="border border-slate-800 bg-[#0F2537] p-5 space-y-3">
              <div className="h-8 w-8 bg-[#D4AF37] text-[#0B192C] flex items-center justify-center font-bold">
                <Layers className="h-4 w-4" />
              </div>
              <h4 className="font-bold text-white uppercase text-sm font-sans">Cross-Encoder Re-Ranking</h4>
              <p className="text-xs text-slate-300 leading-relaxed">
                Re-ranks top document chunks before context assembly to maximize precise policy retrieval.
              </p>
            </div>

            <div className="border border-slate-800 bg-[#0F2537] p-5 space-y-3">
              <div className="h-8 w-8 bg-[#D4AF37] text-[#0B192C] flex items-center justify-center font-bold">
                <Lock className="h-4 w-4" />
              </div>
              <h4 className="font-bold text-white uppercase text-sm font-sans">Domain Guardrails</h4>
              <p className="text-xs text-slate-300 leading-relaxed">
                Intersects out-of-scope non-logistics questions and returns polite domain redirection without wasting retrieval resources.
              </p>
            </div>

            <div className="border border-slate-800 bg-[#0F2537] p-5 space-y-3">
              <div className="h-8 w-8 bg-[#D4AF37] text-[#0B192C] flex items-center justify-center font-bold">
                <ShieldCheck className="h-4 w-4" />
              </div>
              <h4 className="font-bold text-white uppercase text-sm font-sans">Citation Verification</h4>
              <p className="text-xs text-slate-300 leading-relaxed">
                Audits every generated claim against official source document chunk IDs to guarantee zero hallucinated compliance claims.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ==================================================================== */}
      {/* 5. ABOUT SECTION */}
      {/* ==================================================================== */}
      <section id="about" className="w-full border-b border-[#D4AF37]/30 bg-[#0F2537] py-16 px-6 lg:px-12">
        <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
          <div className="lg:col-span-6 space-y-6">
            <span className="border border-[#D4AF37] bg-[#0B192C] px-3 py-1 text-xs font-black uppercase tracking-widest text-[#D4AF37] inline-block rounded-xl">
              About ShipRule CDLP
            </span>
            <h2 className="text-3xl font-black uppercase tracking-tight text-white font-sans">
              Global Freight & Customs Intelligence Platform
            </h2>
            <p className="text-sm text-slate-300 leading-relaxed font-sans">
              ShipRule CDLP (Customs Duty & Shipping Documentation Lookup Platform) is a dedicated logistics hub engineered to streamline international trade compliance for importers, exporters, supply chain managers, and freight forwarders.
            </p>
            <p className="text-sm text-slate-300 leading-relaxed font-sans">
              Our platform centralizes complex tariff schedules, departmental customs regulations, mandatory shipping documentation requirements, and Incoterm guidelines into a unified, grounded lookup system designed to eliminate port clearance delays.
            </p>
            <div className="pt-2 flex items-center gap-6 text-xs font-mono text-[#D4AF37]">
              <div>
                <span className="block text-2xl font-black text-white font-sans">24 / 7</span>
                <span>Customs Compliance Access</span>
              </div>
              <div className="border-l border-slate-700 pl-6">
                <span className="block text-2xl font-black text-white font-sans">100%</span>
                <span>Verified Source Citations</span>
              </div>
            </div>
          </div>

          <div className="lg:col-span-6 border border-[#D4AF37]/40 bg-[#0B192C] p-8 space-y-6 rounded-xl">
            <h3 className="text-xl font-bold text-white uppercase tracking-wider font-sans border-b border-slate-800 pb-3">
              Core Platform Capabilities
            </h3>
            <div className="space-y-4 text-xs text-slate-300">
              <div className="flex items-start gap-3">
                <CheckCircle2 className="h-5 w-5 text-[#D4AF37] flex-shrink-0 mt-0.5" />
                <div>
                  <strong className="text-white uppercase font-sans">Instant Tariff & Duty Calculation</strong>
                  <p className="mt-0.5">Quickly retrieve applicable basic customs duties (BCD), integrated GST, and social welfare surcharges for imported goods.</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <CheckCircle2 className="h-5 w-5 text-[#D4AF37] flex-shrink-0 mt-0.5" />
                <div>
                  <strong className="text-white uppercase font-sans">Shipping Documentation Guidance</strong>
                  <p className="mt-0.5">Identify mandatory documentation requirements including Bills of Lading, Commercial Invoices, Packing Lists, and Bills of Entry.</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <CheckCircle2 className="h-5 w-5 text-[#D4AF37] flex-shrink-0 mt-0.5" />
                <div>
                  <strong className="text-white uppercase font-sans">Incoterms & Risk Allocation</strong>
                  <p className="mt-0.5">Clear legal breakdowns of buyer and seller obligations under CIF, FOB, EXW, and DDP shipping terms.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ==================================================================== */}
      {/* 6. CONTACT SECTION */}
      {/* ==================================================================== */}
      <section id="contact" className="w-full border-b border-[#D4AF37]/30 bg-[#0B192C] py-16 px-6 lg:px-12">
        <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-12">
          <div className="lg:col-span-5 space-y-6">
            <span className="border border-[#D4AF37] bg-[#0F2537] px-3 py-1 text-xs font-black uppercase tracking-widest text-[#D4AF37] inline-block">
              Contact & Support
            </span>
            <h2 className="text-3xl font-black uppercase tracking-tight text-white font-sans">
              Get in Touch with GFS Team
            </h2>
            <p className="text-sm text-slate-300 leading-relaxed font-sans">
              Have questions regarding API integration, enterprise RAG deployment, or custom customs document indexing? Reach out to our technical team.
            </p>

            <div className="space-y-4 text-xs font-mono text-slate-300 pt-4 border-t border-slate-800">
              <div className="flex items-center gap-3">
                <Mail className="h-4 w-4 text-[#D4AF37]" />
                <span>support@shiprule.gfs.com</span>
              </div>
              <div className="flex items-center gap-3">
                <PhoneCall className="h-4 w-4 text-[#D4AF37]" />
                <span>+1 (800) 555-SHIP-RULE</span>
              </div>
              <div className="flex items-center gap-3">
                <MapPin className="h-4 w-4 text-[#D4AF37]" />
                <span>Global Freight Terminal, Port Logistics HQ</span>
              </div>
            </div>
          </div>

          <div className="lg:col-span-7 border border-[#D4AF37]/30 bg-[#0F2537] p-8 space-y-4">
            <h3 className="text-lg font-bold text-white uppercase tracking-wider font-sans border-b border-slate-700 pb-3">
              Send an Inquiry / Request API Access
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-bold uppercase text-slate-300 mb-1">Full Name</label>
                <input
                  type="text"
                  placeholder="e.g. Abhi Kollepara"
                  className="w-full border border-slate-700 bg-[#0B192C] p-3 text-xs text-white placeholder-slate-500 focus:border-[#D4AF37] focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-xs font-bold uppercase text-slate-300 mb-1">Work Email</label>
                <input
                  type="email"
                  placeholder="e.g. user@company.com"
                  className="w-full border border-slate-700 bg-[#0B192C] p-3 text-xs text-white placeholder-slate-500 focus:border-[#D4AF37] focus:outline-none"
                />
              </div>
            </div>
            <div>
              <label className="block text-xs font-bold uppercase text-slate-300 mb-1">Message / Requirements</label>
              <textarea
                rows={3}
                placeholder="Describe your logistics documentation or custom indexing needs..."
                className="w-full border border-slate-700 bg-[#0B192C] p-3 text-xs text-white placeholder-slate-500 focus:border-[#D4AF37] focus:outline-none"
              />
            </div>
            <button className="bg-[#D4AF37] hover:bg-[#C59E2B] text-[#0B192C] px-6 py-3 text-xs font-black uppercase tracking-wider transition-all border border-[#D4AF37]">
              Submit Inquiry
            </button>
          </div>
        </div>
      </section>

      {/* ==================================================================== */}
      {/* 7. FOOTER SECTION (Image 1 Style - Deep Navy & Gold Leaf Footer) */}
      {/* ==================================================================== */}
      <footer className="w-full border-t border-[#D4AF37]/30 bg-[#07101C] py-12 px-6 lg:px-12 text-slate-400">
        <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-4 gap-8 mb-12">
          {/* Col 1: Brand & Description */}
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <img src="/logo.png" alt="ShipRule Logo" className="h-9 w-auto object-contain" />
            </div>
            <p className="text-xs text-slate-400 leading-relaxed font-sans">
              Global Freight Solutions & ShipRule CDLP Platform provides anti-hallucination grounded customs intelligence across international sea, air, and land trade routes.
            </p>
          </div>

          {/* Col 2: Quick Links */}
          <div className="space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-white border-b border-slate-800 pb-2 font-sans">
              Navigation
            </h4>
            <ul className="space-y-2 text-xs">
              <li><a href="#" className="hover:text-[#D4AF37] transition">Home</a></li>
              <li><a href="#services" className="hover:text-[#D4AF37] transition">Services & Scope</a></li>
              <li><a href="#query-engine" className="hover:text-[#D4AF37] transition">Grounded RAG Lookup</a></li>
              <li><a href="#about" className="hover:text-[#D4AF37] transition">About Platform</a></li>
              <li><a href="#contact" className="hover:text-[#D4AF37] transition">Contact & Support</a></li>
            </ul>
          </div>

          {/* Col 3: Supported Topics */}
          <div className="space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-white border-b border-slate-800 pb-2 font-sans">
              Customs Topics
            </h4>
            <ul className="space-y-2 text-xs">
              <li><a href="#query-engine" className="hover:text-[#D4AF37] transition">Import & Export Duties</a></li>
              <li><a href="#query-engine" className="hover:text-[#D4AF37] transition">Bill of Lading & Packing Lists</a></li>
              <li><a href="#query-engine" className="hover:text-[#D4AF37] transition">Incoterms (CIF, FOB, DDP)</a></li>
              <li><a href="#query-engine" className="hover:text-[#D4AF37] transition">Statutory Registrations</a></li>
              <li><a href="#query-engine" className="hover:text-[#D4AF37] transition">HS Code Tariff Schedule</a></li>
            </ul>
          </div>

          {/* Col 4: System Status & Credentials */}
          <div className="space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider text-white border-b border-slate-800 pb-2 font-sans">
              System Audit
            </h4>
            <div className="border border-slate-800 bg-[#0F2537] p-3 space-y-2 text-[11px] font-mono">
              <div className="flex items-center justify-between">
                <span>RAG Pipeline:</span>
                <span className="text-emerald-400 font-bold">ONLINE</span>
              </div>
              <div className="flex items-center justify-between">
                <span>Re-Ranker:</span>
                <span className="text-[#D4AF37] font-bold">ACTIVE</span>
              </div>
              <div className="flex items-center justify-between">
                <span>Citation Guard:</span>
                <span className="text-emerald-400 font-bold">STRICT</span>
              </div>
            </div>
          </div>
        </div>

        <div className="max-w-7xl mx-auto border-t border-slate-800 pt-6 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs font-mono">
          <div>
            &copy; {new Date().getFullYear()} Global Freight Solutions (GFS) & ShipRule CDLP. All rights reserved.
          </div>
          <div className="flex items-center gap-4 text-[#D4AF37]">
            <span>Deep Navy (#0B192C)</span>
            <span>&bull;</span>
            <span>Gold Leaf (#D4AF37)</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
