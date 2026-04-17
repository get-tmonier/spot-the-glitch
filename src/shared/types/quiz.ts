export type GlitchFamily = "teleport" | "time-reversal" | "none";
export type DifficultyTier = "easy" | "medium" | "hard" | "gotcha";

export interface QuizQuestion {
  id: string;
  tier: DifficultyTier;
  glitchFamily: GlitchFamily;
  clipA: { src: string; isGlitched: boolean; surpriseScore: number[] };
  clipB: { src: string; isGlitched: boolean; surpriseScore: number[] };
}

export interface UserAnswer {
  questionId: string;
  chosenClip: "A" | "B";
  correct: boolean;
  timeToAnswer: number; // ms
}

export interface GameResult {
  answers: UserAnswer[];
  score: number; // 0-10
  aiScore: number; // always 10 for now
}
