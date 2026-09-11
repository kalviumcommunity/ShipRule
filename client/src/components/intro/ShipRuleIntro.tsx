'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import Image from 'next/image';

interface ShipRuleIntroProps {
  onComplete?: () => void;
  autoSkipDelay?: number;
}

// World landmass particle cluster points (normalized 0..1 scale)
const CONTINENT_POINTS: Array<[number, number]> = [
  // North America
  [0.18, 0.28], [0.22, 0.25], [0.25, 0.22], [0.20, 0.35], [0.26, 0.33], [0.29, 0.30], [0.15, 0.22], [0.24, 0.40],
  // South America
  [0.32, 0.60], [0.34, 0.68], [0.36, 0.76], [0.33, 0.52], [0.30, 0.55],
  // Europe
  [0.48, 0.25], [0.52, 0.22], [0.55, 0.26], [0.50, 0.30], [0.54, 0.32],
  // Africa
  [0.50, 0.48], [0.53, 0.56], [0.56, 0.65], [0.52, 0.42], [0.58, 0.52],
  // Asia
  [0.65, 0.25], [0.72, 0.22], [0.78, 0.26], [0.82, 0.32], [0.70, 0.35], [0.75, 0.40], [0.85, 0.38], [0.68, 0.45],
  // Oceania
  [0.84, 0.72], [0.88, 0.76], [0.82, 0.78], [0.86, 0.68]
];

// Major logistics hubs (normalized coordinates 0..1)
const HUBS = [
  { name: 'US-WEST', x: 0.20, y: 0.35 },
  { name: 'US-EAST', x: 0.29, y: 0.32 },
  { name: 'EU-ROTTERDAM', x: 0.51, y: 0.26 },
  { name: 'ME-DUBAI', x: 0.62, y: 0.42 },
  { name: 'ASIA-SHANGHAI', x: 0.79, y: 0.38 },
  { name: 'ASIA-SINGAPORE', x: 0.75, y: 0.52 },
  { name: 'AU-SYDNEY', x: 0.86, y: 0.74 }
];

// Active trade connections between hubs
const TRADE_ROUTES = [
  { from: 0, to: 4, color: '#D4AF37' }, // US-WEST <-> SHANGHAI
  { from: 1, to: 2, color: '#38BDF8' }, // US-EAST <-> ROTTERDAM
  { from: 2, to: 3, color: '#D4AF37' }, // ROTTERDAM <-> DUBAI
  { from: 3, to: 5, color: '#06B6D4' }, // DUBAI <-> SINGAPORE
  { from: 5, to: 4, color: '#D4AF37' }, // SINGAPORE <-> SHANGHAI
  { from: 4, to: 6, color: '#38BDF8' }, // SHANGHAI <-> SYDNEY
];

export const ShipRuleIntro: React.FC<ShipRuleIntroProps> = ({ onComplete }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [stage, setStage] = useState<number>(1);
  const [progress, setProgress] = useState<number>(0);
  const [isFadingOut, setIsFadingOut] = useState<boolean>(false);
  const [skipVisible, setSkipVisible] = useState<boolean>(false);
  const animationFrameRef = useRef<number | null>(null);
  const startTimeRef = useRef<number>(Date.now());
  const completedRef = useRef<boolean>(false);

  // Handle stage completion
  const handleFinish = useCallback(() => {
    if (completedRef.current) return;
    completedRef.current = true;
    setIsFadingOut(true);
    setTimeout(() => {
      if (onComplete) onComplete();
    }, 700);
  }, [onComplete]);

  // Reduced motion preference check & Keyboard ESC listener
  useEffect(() => {
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) {
      setProgress(100);
      handleFinish();
      return;
    }

    // Show skip button after 1.5s
    const skipTimer = setTimeout(() => setSkipVisible(true), 1500);

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        handleFinish();
      }
    };
    window.addEventListener('keydown', handleKeyDown);

    return () => {
      clearTimeout(skipTimer);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleFinish]);

  // Main Animation Timeline & Canvas Renderer
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    const handleResize = () => {
      if (!canvas) return;
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    };

    window.addEventListener('resize', handleResize);

    // Dynamic trade particles
    interface RouteParticle {
      routeIndex: number;
      progress: number;
      speed: number;
      size: number;
      color: string;
    }

    const particles: RouteParticle[] = [];
    for (let i = 0; i < 35; i++) {
      particles.push({
        routeIndex: i % TRADE_ROUTES.length,
        progress: Math.random(),
        speed: 0.002 + Math.random() * 0.003,
        size: 1.5 + Math.random() * 2,
        color: TRADE_ROUTES[i % TRADE_ROUTES.length].color,
      });
    }

    // Converging particles for Stage 3
    interface ImplodeParticle {
      x: number;
      y: number;
      targetX: number;
      targetY: number;
      speed: number;
      color: string;
      size: number;
      alpha: number;
    }
    const implodeParticles: ImplodeParticle[] = [];

    const TOTAL_DURATION = 7200; // ms

    const render = () => {
      const elapsed = Date.now() - startTimeRef.current;
      const currentProgress = Math.min(100, Math.floor((elapsed / TOTAL_DURATION) * 100));
      setProgress(currentProgress);

      // Timeline Stage Controller
      if (elapsed < 1600) {
        setStage(1);
      } else if (elapsed < 3800) {
        setStage(2);
      } else if (elapsed < 5200) {
        setStage(3);
      } else if (elapsed < 6800) {
        setStage(4);
      } else {
        setStage(5);
      }

      if (elapsed >= TOTAL_DURATION) {
        handleFinish();
        return;
      }

      // Clear canvas
      ctx.clearRect(0, 0, width, height);

      // Center coordinates
      const cx = width / 2;
      const cy = height / 2;

      // STAGE 2+: Render Map & Trade Routes
      if (elapsed >= 1200) {
        const mapOpacity = elapsed < 1600 
          ? (elapsed - 1200) / 400 
          : elapsed > 4500 
            ? Math.max(0, 1 - (elapsed - 4500) / 1000) 
            : 0.65;

        ctx.save();
        ctx.globalAlpha = mapOpacity;

        // Render dot landmasses
        ctx.fillStyle = 'rgba(212, 175, 55, 0.15)';

        CONTINENT_POINTS.forEach(([px, py]) => {
          const baseX = px * width;
          const baseY = py * height;
          for (let dx = -3; dx <= 3; dx++) {
            for (let dy = -3; dy <= 3; dy++) {
              if (Math.random() > 0.4) continue;
              const dotX = baseX + dx * 14;
              const dotY = baseY + dy * 10;
              ctx.beginPath();
              ctx.arc(dotX, dotY, 1.2, 0, Math.PI * 2);
              ctx.fill();
            }
          }
        });

        // Draw Trade Route Arcs
        TRADE_ROUTES.forEach((route) => {
          const h1 = HUBS[route.from];
          const h2 = HUBS[route.to];
          const x1 = h1.x * width;
          const y1 = h1.y * height;
          const x2 = h2.x * width;
          const y2 = h2.y * height;

          const controlX = (x1 + x2) / 2;
          const controlY = Math.min(y1, y2) - 60;

          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.quadraticCurveTo(controlX, controlY, x2, y2);
          ctx.strokeStyle = route.color === '#D4AF37' ? 'rgba(212, 175, 55, 0.25)' : 'rgba(56, 189, 248, 0.25)';
          ctx.lineWidth = 1.2;
          ctx.setLineDash([4, 4]);
          ctx.stroke();
          ctx.setLineDash([]);
        });

        // Draw flowing trade route particles
        particles.forEach((p) => {
          p.progress = (p.progress + p.speed) % 1;
          const route = TRADE_ROUTES[p.routeIndex];
          const h1 = HUBS[route.from];
          const h2 = HUBS[route.to];
          const x1 = h1.x * width;
          const y1 = h1.y * height;
          const x2 = h2.x * width;
          const y2 = h2.y * height;
          const controlX = (x1 + x2) / 2;
          const controlY = Math.min(y1, y2) - 60;

          const t = p.progress;
          const px = (1 - t) * (1 - t) * x1 + 2 * (1 - t) * t * controlX + t * t * x2;
          const py = (1 - t) * (1 - t) * y1 + 2 * (1 - t) * t * controlY + t * t * y2;

          ctx.beginPath();
          ctx.arc(px, py, p.size, 0, Math.PI * 2);
          ctx.fillStyle = p.color;
          ctx.shadowBlur = 8;
          ctx.shadowColor = p.color;
          ctx.fill();
          ctx.shadowBlur = 0;
        });

        // Draw Hub Pulses
        HUBS.forEach((hub) => {
          const hx = hub.x * width;
          const hy = hub.y * height;
          ctx.beginPath();
          ctx.arc(hx, hy, 3, 0, Math.PI * 2);
          ctx.fillStyle = '#D4AF37';
          ctx.fill();

          ctx.beginPath();
          ctx.arc(hx, hy, 6 + Math.sin(Date.now() / 300) * 3, 0, Math.PI * 2);
          ctx.strokeStyle = 'rgba(212, 175, 55, 0.4)';
          ctx.lineWidth = 1;
          ctx.stroke();
        });

        ctx.restore();
      }

      // STAGE 3: Implosion / Convergence of particles to Center
      if (elapsed >= 3600 && elapsed < 4800) {
        if (implodeParticles.length === 0) {
          for (let i = 0; i < 90; i++) {
            const angle = Math.random() * Math.PI * 2;
            const dist = 250 + Math.random() * 300;
            implodeParticles.push({
              x: cx + Math.cos(angle) * dist,
              y: cy + Math.sin(angle) * dist,
              targetX: cx,
              targetY: cy,
              speed: 0.04 + Math.random() * 0.05,
              color: i % 2 === 0 ? '#D4AF37' : '#38BDF8',
              size: 2 + Math.random() * 2.5,
              alpha: 1,
            });
          }
        }

        ctx.save();
        implodeParticles.forEach((p) => {
          p.x += (p.targetX - p.x) * p.speed;
          p.y += (p.targetY - p.y) * p.speed;
          const distLeft = Math.hypot(p.targetX - p.x, p.targetY - p.y);
          p.alpha = Math.min(1, distLeft / 150);

          ctx.beginPath();
          ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
          ctx.fillStyle = p.color;
          ctx.globalAlpha = p.alpha;
          ctx.shadowBlur = 10;
          ctx.shadowColor = p.color;
          ctx.fill();
        });
        ctx.restore();
      }

      animationFrameRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      window.removeEventListener('resize', handleResize);
    };
  }, [handleFinish]);

  return (
    <div
      className={`fixed inset-0 z-[9999] flex flex-col items-center justify-center bg-[#070E17] text-white overflow-hidden transition-all duration-700 ease-in-out ${
        isFadingOut ? 'opacity-0 scale-105 pointer-events-none' : 'opacity-100 scale-100'
      }`}
    >
      {/* Background Interactive Canvas */}
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />

      {/* Radial Glass Accent Glow */}
      <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-[#0F2537]/60 via-[#070E17]/80 to-[#070E17]" />

      {/* Skip Button */}
      {skipVisible && (
        <button
          onClick={handleFinish}
          className="absolute top-6 right-6 z-50 text-[11px] font-bold uppercase tracking-widest text-slate-400 hover:text-[#D4AF37] border border-slate-800 hover:border-[#D4AF37]/50 bg-[#0B192C]/80 px-4 py-2 rounded-xl backdrop-blur-md transition-all duration-300 shadow-lg cursor-pointer"
        >
          Skip Intro <span className="text-[9px] opacity-60 ml-1">(ESC)</span>
        </button>
      )}

      {/* Central Content Container */}
      <div className="relative z-10 flex flex-col items-center justify-center text-center px-6 max-w-3xl mx-auto space-y-6">

        {/* STAGE 1: Dark Introduction — Center Beam & Teaser Text */}
        {stage === 1 && (
          <div className="flex flex-col items-center space-y-4 animate-fade-in">
            {/* Center glowing line */}
            <div className="h-[2px] w-24 sm:w-40 bg-gradient-to-r from-transparent via-[#D4AF37] to-transparent shadow-[0_0_15px_#D4AF37] animate-pulse" />

            <div className="space-y-1">
              <h2 className="text-xs sm:text-sm font-extrabold uppercase tracking-[0.4em] text-[#D4AF37] drop-shadow-md">
                GLOBAL TRADE
              </h2>
              <h1 className="text-xl sm:text-2xl font-black uppercase tracking-[0.3em] text-white drop-shadow-[0_0_20px_rgba(255,255,255,0.4)]">
                SIMPLIFIED
              </h1>
            </div>
          </div>
        )}

        {/* STAGE 2: Global Trade Visualization Floating Badges */}
        {stage === 2 && (
          <div className="flex flex-wrap items-center justify-center gap-3 animate-fade-in">
            <span className="border border-[#D4AF37]/40 bg-[#0F2537]/80 px-3.5 py-1.5 text-[11px] font-bold uppercase tracking-widest text-[#D4AF37] backdrop-blur-md rounded-xl shadow-md">
              Global Trade
            </span>
            <span className="border border-[#38BDF8]/40 bg-[#0F2537]/80 px-3.5 py-1.5 text-[11px] font-bold uppercase tracking-widest text-[#38BDF8] backdrop-blur-md rounded-xl shadow-md">
              Shipping
            </span>
            <span className="border border-[#D4AF37]/40 bg-[#0F2537]/80 px-3.5 py-1.5 text-[11px] font-bold uppercase tracking-widest text-[#D4AF37] backdrop-blur-md rounded-xl shadow-md">
              Customs
            </span>
            <span className="border border-[#06B6D4]/40 bg-[#0F2537]/80 px-3.5 py-1.5 text-[11px] font-bold uppercase tracking-widest text-[#06B6D4] backdrop-blur-md rounded-xl shadow-md">
              Compliance
            </span>
          </div>
        )}

        {/* STAGE 3, 4 & 5: Logo & Tagline Reveal */}
        {stage >= 3 && (
          <div className="flex flex-col items-center space-y-6 transition-all duration-700 animate-fade-in">
            
            {/* Preserved ShipRule Logo Asset with Soft Radial Backlight Glow */}
            <div className="relative group">
              {/* Radial backlight glow behind logo */}
              <div className="absolute -inset-4 rounded-full bg-gradient-to-r from-[#D4AF37]/30 via-[#38BDF8]/20 to-[#D4AF37]/30 blur-2xl opacity-75 animate-pulse" />

              <div className="relative border border-[#D4AF37]/30 bg-[#0B192C]/90 p-4 sm:p-6 rounded-2xl backdrop-blur-xl shadow-2xl">
                <Image
                  src="/logo.png"
                  alt="ShipRule Logo"
                  width={220}
                  height={55}
                  priority
                  className="h-10 sm:h-14 w-auto object-contain drop-shadow-[0_0_15px_rgba(212,175,55,0.4)]"
                />
              </div>
            </div>

            {/* STAGE 4: Product Statement Typography */}
            {stage >= 4 && (
              <div className="space-y-2 max-w-lg transition-all duration-500 animate-fade-in">
                <h3 className="text-sm sm:text-base font-semibold tracking-wide text-slate-200 uppercase font-sans">
                  Customs Duty &amp; Documentation Lookup Platform
                </h3>
                <p className="text-xs sm:text-sm font-medium tracking-widest text-[#D4AF37] uppercase">
                  Smart. Grounded. Reliable.
                </p>
              </div>
            )}

            {/* STAGE 5: Sleek Gold/Cyan Loading Beam */}
            {stage >= 3 && (
              <div className="w-64 sm:w-80 space-y-2 pt-4 transition-all duration-300">
                <div className="relative w-full h-1.5 bg-[#0F2537] rounded-full overflow-hidden border border-[#D4AF37]/30 shadow-inner">
                  <div
                    className="h-full bg-gradient-to-r from-[#D4AF37] via-[#38BDF8] to-[#D4AF37] transition-all duration-150 ease-out rounded-full shadow-[0_0_10px_#D4AF37]"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <div className="flex justify-between items-center text-[10px] font-bold text-slate-400 tracking-wider">
                  <span className="uppercase text-[#D4AF37]">Initializing CDLP Engine</span>
                  <span className="text-slate-300">{progress}%</span>
                </div>
              </div>
            )}
          </div>
        )}

      </div>
    </div>
  );
};

export default ShipRuleIntro;
