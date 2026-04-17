"use client";

import { Suspense, useCallback } from "react";
import { useSearchParams } from "next/navigation";
import { motion } from "framer-motion";
import Link from "next/link";
import { AI_SCORE, TOTAL_QUESTIONS } from "@/shared/config";

function ResultContent() {
  const params = useSearchParams();
  const score = Math.min(Math.max(Number(params.get("score") ?? 0), 0), TOTAL_QUESTIONS);
  const pct = Math.round((score / TOTAL_QUESTIONS) * 100);

  const message =
    score === TOTAL_QUESTIONS
      ? "Perfect score. You matched the AI world model."
      : score >= 8
        ? "Impressive — the AI barely edges you out."
        : score >= 5
          ? "Decent. The AI world model still wins."
          : "The AI dominates. Physics is hard.";

  const shareText = `I scored ${score}/${TOTAL_QUESTIONS} on Spot the Glitch — can you beat the AI world model? 🤖⚡`;

  const handleShare = useCallback(() => {
    if (navigator.share) {
      void navigator.share({ text: shareText });
    } else {
      void navigator.clipboard.writeText(shareText);
    }
  }, [shareText]);

  return (
    <div className="flex min-h-screen flex-col items-center justify-center px-6 text-center">
      <motion.p
        className="mb-8 text-[10px] uppercase tracking-[0.3em] text-white/30"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.4 }}
      >
        Game Over
      </motion.p>

      {/* Score vs AI */}
      <div className="flex items-end gap-10 mb-6">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15, type: "spring", damping: 18 }}
        >
          <div className="text-7xl font-black tabular-nums">{score}</div>
          <p className="mt-1 text-xs uppercase tracking-wider text-white/40">You</p>
        </motion.div>

        <motion.div
          className="pb-3 text-lg text-white/20"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          vs
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, type: "spring", damping: 18 }}
        >
          <div className="text-7xl font-black tabular-nums text-[var(--accent)]">{AI_SCORE}</div>
          <p className="mt-1 text-xs uppercase tracking-wider text-white/40">AI</p>
        </motion.div>
      </div>

      {/* Progress bar */}
      <motion.div
        className="mb-6 w-full max-w-xs"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
      >
        <div className="h-1.5 w-full rounded-full bg-white/10">
          <motion.div
            className="h-full rounded-full bg-[var(--accent)]"
            initial={{ width: 0 }}
            animate={{ width: `${pct}%` }}
            transition={{ delay: 0.6, duration: 0.8, ease: "easeOut" }}
          />
        </div>
        <p className="mt-2 text-xs text-white/30">{pct}% accuracy</p>
      </motion.div>

      <motion.p
        className="mb-8 max-w-xs text-sm text-white/50"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.7 }}
      >
        {message}
      </motion.p>

      <motion.div
        className="flex w-full max-w-xs flex-col gap-3"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.9 }}
      >
        <button
          onClick={handleShare}
          className="rounded-xl bg-[var(--accent)] py-3 text-sm font-semibold text-white transition-opacity hover:opacity-90 active:opacity-80"
        >
          Share Score
        </button>
        <Link
          href="/play"
          className="rounded-xl border border-white/15 py-3 text-sm font-semibold text-white/60 transition-colors hover:bg-white/5"
        >
          Play Again
        </Link>
      </motion.div>
    </div>
  );
}

export default function ResultPage() {
  return (
    <Suspense>
      <ResultContent />
    </Suspense>
  );
}
