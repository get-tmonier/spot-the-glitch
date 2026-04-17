import Link from "next/link";
import { Container, H1, Text, Button } from "@/shared/ui";

export default function HomePage() {
  return (
    <Container className="flex min-h-screen flex-col items-center justify-center text-center">
      <p className="mb-3 text-xs uppercase tracking-[0.3em] text-white/40">
        AI World Model Challenge
      </p>
      <H1>
        Spot the <span className="glitch-text text-[var(--accent)]">Glitch</span>
      </H1>
      <Text className="mt-4 max-w-md text-lg leading-relaxed">
        An AI trained on physics watches the same clips you do. Can you beat it at spotting
        impossible motion?
      </Text>
      <div className="mt-3 flex flex-wrap justify-center gap-2">
        {["10 rounds", "3s clips", "real AI scores"].map((tag) => (
          <span
            key={tag}
            className="rounded-full border border-white/10 px-3 py-1 text-xs text-white/40"
          >
            {tag}
          </span>
        ))}
      </div>
      <Link href="/play" className="mt-8">
        <Button>Start Playing</Button>
      </Link>
    </Container>
  );
}
