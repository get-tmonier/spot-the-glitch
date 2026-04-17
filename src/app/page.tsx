"use client";

import Link from "next/link";
import { motion, type Easing } from "framer-motion";

const fade = (delay = 0) => ({
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0 },
  transition: { delay, duration: 0.5, ease: "easeOut" as Easing },
});

function SurpriseDiagram() {
  const normal = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3];
  const glitched = [3, 3, 3, 3, 3, 3, 38, 4, 3, 3];
  const w = 160;
  const h = 48;
  const px = 6;
  const py = 4;
  const uw = w - px * 2;
  const uh = h - py * 2;
  const max = 40;

  const toPath = (pts: number[]) =>
    pts
      .map((v, i) => {
        const x = px + (i / (pts.length - 1)) * uw;
        const y = h - py - (v / max) * uh;
        return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(" ");

  return (
    <div className="mt-3">
      <svg width={w} height={h} className="overflow-visible">
        <path d={toPath(normal)} stroke="#3b82f6" strokeWidth="1.5" fill="none" opacity="0.7" />
        <path d={toPath(glitched)} stroke="#f97316" strokeWidth="1.5" fill="none" />
        <line
          x1={(px + (6 / 9) * uw).toFixed(1)}
          y1={py}
          x2={(px + (6 / 9) * uw).toFixed(1)}
          y2={h - py}
          stroke="#f97316"
          strokeWidth="1"
          strokeDasharray="2,3"
          opacity="0.5"
        />
      </svg>
      <div className="mt-1.5 flex gap-4 text-[10px] text-white/35">
        <span className="flex items-center gap-1.5">
          <span className="h-px w-4 bg-[#f97316]" /> Glitched clip
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-px w-4 bg-[#3b82f6] opacity-70" /> Normal clip
        </span>
      </div>
    </div>
  );
}

export default function HomePage() {
  return (
    <main className="min-h-screen px-5 py-16 md:py-24">
      <div className="mx-auto max-w-2xl">
        {/* Hero */}
        <motion.div className="text-center" {...fade(0)}>
          <p className="mb-3 text-[10px] uppercase tracking-[0.3em] text-white/35">
            AI World Model Challenge
          </p>
          <h1 className="text-5xl font-black tracking-tight md:text-7xl">
            Spot the <span className="glitch-text text-[var(--accent)]">Glitch</span>
          </h1>
          <p className="mt-4 text-base text-white/50 md:text-lg">
            Can you beat an AI that learned physics by watching robots?
          </p>
        </motion.div>

        {/* Cards */}
        <div className="mt-12 grid gap-4 md:grid-cols-2">
          {/* The AI */}
          <motion.div className="rounded-2xl border border-white/8 bg-white/3 p-6" {...fade(0.15)}>
            <div className="mb-4 flex items-center gap-2">
              <span className="rounded-full bg-[var(--accent)]/15 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider text-[var(--accent)]">
                The AI
              </span>
            </div>
            <h2 className="text-lg font-bold leading-snug">LeWorldModel</h2>
            <p className="mt-2 text-sm leading-relaxed text-white/50">
              A world model trained on thousands of robotic manipulation trajectories. It never sees
              physics equations — it learns them implicitly, by predicting what comes next frame by
              frame.
            </p>
            <p className="mt-3 text-sm leading-relaxed text-white/50">
              When physics is violated, its prediction error — the{" "}
              <span className="text-white/80 font-medium">surprise signal</span> — spikes. That
              spike is how it detects glitches.
            </p>
            <SurpriseDiagram />
            <div className="mt-4 flex flex-wrap gap-2">
              <a
                href="https://arxiv.org/abs/2603.19312"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 rounded-lg border border-white/10 px-3 py-1.5 text-xs text-white/40 transition-colors hover:border-white/20 hover:text-white/60"
              >
                <span className="text-[10px]">📄</span> arXiv:2603.19312
              </a>
              <a
                href="https://github.com/lucas-maes/le-wm"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 rounded-lg border border-white/10 px-3 py-1.5 text-xs text-white/40 transition-colors hover:border-white/20 hover:text-white/60"
              >
                <span className="text-[10px]">⚙</span> lucas-maes/le-wm
              </a>
            </div>
          </motion.div>

          {/* The Rules */}
          <motion.div className="rounded-2xl border border-white/8 bg-white/3 p-6" {...fade(0.25)}>
            <div className="mb-4 flex items-center gap-2">
              <span className="rounded-full bg-white/8 px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider text-white/50">
                The Rules
              </span>
            </div>
            <ol className="space-y-4">
              {[
                {
                  n: "1",
                  title: "Watch two clips",
                  body: "Two 3-second loops play side by side. One follows normal physics, the other doesn't.",
                },
                {
                  n: "2",
                  title: "Spot the violation",
                  body: "Tap the clip you think has a physics glitch before time runs out.",
                },
                {
                  n: "3",
                  title: "See what the AI found",
                  body: "After each pick, the AI's surprise signal is revealed — did it agree with you?",
                },
              ].map(({ n, title, body }) => (
                <li key={n} className="flex gap-3">
                  <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-white/8 text-[10px] font-bold text-white/40">
                    {n}
                  </span>
                  <div>
                    <p className="text-sm font-semibold text-white/80">{title}</p>
                    <p className="mt-0.5 text-xs leading-relaxed text-white/40">{body}</p>
                  </div>
                </li>
              ))}
            </ol>
            <div className="mt-5 border-t border-white/6 pt-4">
              <p className="mb-2 text-[10px] uppercase tracking-wider text-white/25">
                Glitch types
              </p>
              <div className="flex flex-wrap gap-2">
                {[
                  { label: "Teleportation", color: "#f97316" },
                  { label: "Time Reversal", color: "#a855f7" },
                  { label: "Gotcha — no glitch", color: "#22c55e" },
                ].map(({ label, color }) => (
                  <span
                    key={label}
                    className="rounded-full px-2.5 py-1 text-[10px] font-medium"
                    style={{ background: color + "18", color }}
                  >
                    {label}
                  </span>
                ))}
              </div>
            </div>
          </motion.div>
        </div>

        {/* CTA */}
        <motion.div className="mt-8 text-center" {...fade(0.4)}>
          <Link
            href="/play"
            className="inline-block rounded-xl bg-[var(--accent)] px-10 py-3.5 text-sm font-bold text-white transition-opacity hover:opacity-90 active:opacity-80"
          >
            Start Playing →
          </Link>
          <p className="mt-3 text-xs text-white/25">10 rounds · 3 s clips · real AI scores</p>
        </motion.div>
      </div>
    </main>
  );
}
