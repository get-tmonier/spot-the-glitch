import { Container, H1, Text } from "@/shared/ui";

export default function ResultPage() {
  return (
    <Container className="flex min-h-screen flex-col items-center justify-center text-center">
      <H1>Results</H1>
      <Text className="mt-4">Your score will appear here.</Text>
    </Container>
  );
}
