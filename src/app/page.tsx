import Link from "next/link";
import { Container, H1, Text, Button } from "@/shared/ui";

export default function HomePage() {
  return (
    <Container className="flex min-h-screen flex-col items-center justify-center text-center">
      <H1>
        Spot the <span className="text-[var(--accent)]">Glitch</span>
      </H1>
      <Text className="mt-4 max-w-lg text-lg">
        Can you beat an AI that learned physics from scratch? Watch two clips, find the one with a
        physics violation. 10 rounds. Good luck.
      </Text>
      <Link href="/play" className="mt-8">
        <Button>Start Playing</Button>
      </Link>
    </Container>
  );
}
