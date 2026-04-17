import { Card, Container, H2, Text } from "@/shared/ui";

export default function PlayPage() {
  return (
    <Container className="flex min-h-screen flex-col items-center justify-center text-center">
      <H2>Round 1 / 10</H2>
      <Text className="mt-4">Which clip contains the physics glitch?</Text>
      <div className="mt-8 grid grid-cols-2 gap-4">
        <Card className="aspect-video" />
        <Card className="aspect-video" />
      </div>
    </Container>
  );
}
