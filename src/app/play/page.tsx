"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { AreaChart, Area, ResponsiveContainer, Tooltip } from "recharts";
import { TOTAL_QUESTIONS } from "@/shared/config";
import type { QuizQuestion, GlitchFamily } from "@/shared/types/quiz";

const GLITCH_INFO: Record<GlitchFamily, { label: string; color: string; explanation: string }> = {
  teleport: {
    label: "Teleportation",
    color: "#f97316",
    explanation:
      "The object jumped position instantaneously — violating conservation of momentum. The AI's world model predicted smooth motion but detected a sudden spatial discontinuity.",
  },
  "time-reversal": {
    label: "Time Reversal",
    color: "#a855f7",
    explanation:
      "Entropy briefly decreased — the system moved backward in time. The AI learned that physical processes are irreversible and flagged this causality violation.",
  },
  none: {
    label: "No Glitch",
    color: "#22c55e",
    explanation:
      "Both clips followed normal physics. This was a gotcha — the AI had similar surprise scores on both clips. Spotting the absence of a glitch is the hardest challenge.",
  },
};

const TIER_LABEL: Record<string, string> = {
  easy: "Easy",
  medium: "Medium",
  hard: "Hard",
  gotcha: "Gotcha",
};

function shuffle<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

type Phase = "loading" | "playing" | "revealing";

export default function PlayPage() {
  const router = useRouter();
  const [questions, setQuestions] = useState<QuizQuestion[]>([]);
  const [round, setRound] = useState(0);
  const scoreRef = useRef(0);
  const [phase, setPhase] = useState<Phase>("loading");
  const [chosen, setChosen] = useState<"A" | "B" | null>(null);
  const [correct, setCorrect] = useState(false);

  useEffect(() => {
    fetch("/quiz-data.json")
      .then((r) => r.json())
      .then((data: { pool: QuizQuestion[] }) => {
        scoreRef.current = 0;
        setQuestions(shuffle(data.pool).slice(0, TOTAL_QUESTIONS));
        setPhase("playing");
      });
  }, []);

  const question = questions[round];

  const handleAnswer = useCallback(
    (choice: "A" | "B") => {
      if (phase !== "playing" || !question) return;
      const isCorrect = choice === "A" ? question.clipA.isGlitched : question.clipB.isGlitched;
      if (isCorrect) scoreRef.current += 1;
      setChosen(choice);
      setCorrect(isCorrect);
      setPhase("revealing");
    },
    [phase, question],
  );

  const handleNext = useCallback(() => {
    if (round + 1 >= TOTAL_QUESTIONS) {
      router.push(`/result?score=${scoreRef.current}`);
    } else {
      setRound((r) => r + 1);
      setChosen(null);
      setPhase("playing");
    }
  }, [round, router]);

  if (phase === "loading" || !question) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <span className="text-xs uppercase tracking-[0.3em] text-white/30">Loading…</span>
      </div>
    );
  }

  const glitchInfo = GLITCH_INFO[question.glitchFamily];
  const chartData = question.clipA.surpriseScore.map((val, i) => ({
    t: i,
    a: val,
    b: question.clipB.surpriseScore[i],
  }));

  return (
    <div className="relative flex h-screen flex-col overflow-hidden bg-black">
      {/* Top bar */}
      <div className="relative z-10 flex items-center justify-between px-5 py-4">
        <span className="text-[10px] font-bold uppercase tracking-[0.25em] text-white/40">
          Spot the Glitch
        </span>
        <div className="flex items-center gap-1.5">
          {Array.from({ length: TOTAL_QUESTIONS }).map((_, i) => (
            <div
              key={i}
              className={`h-1 rounded-full transition-all duration-300 ${
                i < round
                  ? "w-5 bg-[var(--accent)]"
                  : i === round
                    ? "w-5 bg-white"
                    : "w-3 bg-white/15"
              }`}
            />
          ))}
        </div>
        <span className="text-xs font-medium tabular-nums text-white/40">
          {scoreRef.current} pts
        </span>
      </div>

      {/* Prompt */}
      <p className="relative z-10 pb-3 text-center text-sm font-semibold tracking-wide text-white/80">
        Which clip contains the physics glitch?
      </p>

      {/* Video panels */}
      <div className="relative flex flex-1 gap-px overflow-hidden sm:flex-row flex-col">
        {(["A", "B"] as const).map((clip) => {
          const clipData = clip === "A" ? question.clipA : question.clipB;
          const isChosen = chosen === clip;
          const showGlitched = phase === "revealing" && clipData.isGlitched;
          const _showWrong = phase === "revealing" && isChosen && !clipData.isGlitched;

          return (
            <button
              key={clip}
              className={`group relative overflow-hidden sm:flex-1 ${
                phase === "playing" ? "cursor-pointer" : "pointer-events-none"
              }`}
              onClick={() => handleAnswer(clip)}
            >
              <video
                className="h-full w-full object-contain bg-[#0a0a0a]"
                src={clipData.src}
                autoPlay
                loop
                muted
                playsInline
              />

              {/* Gradient */}
              <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-transparent to-transparent" />

              {/* Hover glow */}
              {phase === "playing" && (
                <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-150 ring-inset ring-2 ring-white/30" />
              )}

              {/* Reveal: dim non-chosen clips */}
              {phase === "revealing" && !isChosen && !showGlitched && (
                <div className="absolute inset-0 bg-black/60" />
              )}

              {/* Reveal: centered verdict icon on chosen clip */}
              {phase === "revealing" && isChosen && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/30">
                  <div
                    className={`flex h-20 w-20 items-center justify-center rounded-full text-4xl font-black shadow-2xl ${
                      correct ? "bg-green-500" : "bg-red-500"
                    }`}
                  >
                    {correct ? "✓" : "✗"}
                  </div>
                </div>
              )}

              {/* Reveal: glitch stamp on the glitched clip */}
              {showGlitched && (
                <div className="absolute inset-x-0 top-4 flex justify-center">
                  <div className="rounded-full bg-orange-500 px-4 py-1.5 text-xs font-bold uppercase tracking-widest text-white shadow-lg">
                    ⚡ Glitched
                  </div>
                </div>
              )}

              {/* Label */}
              {phase === "playing" && (
                <div className="absolute bottom-4 left-1/2 -translate-x-1/2">
                  <div className="rounded-full bg-black/40 px-4 py-1.5 text-xs font-bold uppercase tracking-widest text-white/70 backdrop-blur-sm group-hover:bg-white/15 group-hover:text-white transition-all duration-150">
                    Clip {clip}
                  </div>
                </div>
              )}
            </button>
          );
        })}

        {/* VS badge */}
        {phase === "playing" && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center z-10">
            <div className="rounded-full border border-white/15 bg-black/50 px-2.5 py-1 text-[10px] font-bold tracking-wider text-white/40 backdrop-blur-sm">
              VS
            </div>
          </div>
        )}
      </div>

      {/* AI Analysis bottom sheet */}
      <AnimatePresence>
        {phase === "revealing" && (
          <motion.div
            className="absolute inset-x-0 bottom-0 z-40 rounded-t-2xl border-t border-white/10 bg-[#111] px-5 pt-5 pb-6"
            initial={{ y: "100%" }}
            animate={{ y: 0 }}
            transition={{ type: "spring", damping: 28, stiffness: 220 }}
          >
            <div className="flex gap-4">
              {/* Text analysis */}
              <div className="flex-1 min-w-0">
                <div className="flex flex-wrap items-center gap-2 mb-2">
                  <span
                    className={`text-sm font-bold ${correct ? "text-green-400" : "text-red-400"}`}
                  >
                    {correct ? "✓ Correct" : "✗ Missed it"}
                  </span>
                  <span
                    className="rounded px-2 py-0.5 text-xs font-semibold"
                    style={{ background: glitchInfo.color + "22", color: glitchInfo.color }}
                  >
                    {glitchInfo.label}
                  </span>
                  <span className="rounded px-2 py-0.5 text-xs text-white/30 bg-white/5">
                    {TIER_LABEL[question.tier]}
                  </span>
                </div>
                <p className="text-xs leading-relaxed text-white/50">{glitchInfo.explanation}</p>
              </div>

              {/* Surprise chart */}
              <div className="shrink-0 w-36">
                <p className="text-[9px] uppercase tracking-wider text-white/30 mb-1 text-center">
                  AI Surprise Signal
                </p>
                <ResponsiveContainer width="100%" height={56}>
                  <AreaChart data={chartData} margin={{ top: 2, right: 2, bottom: 0, left: 2 }}>
                    <Area
                      type="monotone"
                      dataKey="a"
                      name="Clip A"
                      stroke={question.clipA.isGlitched ? "#f97316" : "#3b82f6"}
                      fill={question.clipA.isGlitched ? "#f9731618" : "#3b82f618"}
                      strokeWidth={1.5}
                      dot={false}
                    />
                    <Area
                      type="monotone"
                      dataKey="b"
                      name="Clip B"
                      stroke={question.clipB.isGlitched ? "#f97316" : "#3b82f6"}
                      fill={question.clipB.isGlitched ? "#f9731618" : "#3b82f618"}
                      strokeWidth={1.5}
                      dot={false}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "#1a1a1a",
                        border: "none",
                        borderRadius: 6,
                        fontSize: 10,
                        padding: "4px 8px",
                      }}
                      formatter={(v) => (typeof v === "number" ? v.toFixed(3) : v)}
                    />
                  </AreaChart>
                </ResponsiveContainer>
                <div className="mt-1 flex justify-center gap-3">
                  {(["a", "b"] as const).map((k) => {
                    const isGlitched =
                      k === "a" ? question.clipA.isGlitched : question.clipB.isGlitched;
                    return (
                      <span key={k} className="flex items-center gap-1 text-[9px] text-white/35">
                        <span
                          className="h-1.5 w-1.5 rounded-full"
                          style={{ background: isGlitched ? "#f97316" : "#3b82f6" }}
                        />
                        Clip {k.toUpperCase()}
                      </span>
                    );
                  })}
                </div>
              </div>
            </div>

            <button
              onClick={handleNext}
              className="mt-4 w-full rounded-xl bg-[var(--accent)] py-3 text-sm font-semibold text-white transition-opacity hover:opacity-90 active:opacity-80"
            >
              {round + 1 >= TOTAL_QUESTIONS ? "See My Score →" : "Next Round →"}
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
