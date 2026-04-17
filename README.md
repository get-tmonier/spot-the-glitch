# Spot the Glitch

> Can you beat an AI that learned physics from scratch?

A viral web game where you watch pairs of 3-second video clips and try to spot the one with a subtle physics violation. After 10 rounds, see how you compare to an AI world model that scores 10/10 every time — based on the [LeWorldModel (LeWM)](https://arxiv.org/abs/2603.19312) paper.

## Getting started

```bash
bun install
bun dev
```

Open [http://localhost:3000](http://localhost:3000).

## Architecture

Built with Next.js 16, TypeScript strict, and [Feature-Sliced Design](https://feature-sliced.design/) (FSD):

```
src/
├── app/          # Next.js App Router + providers, global setup
├── views/        # Route-level page compositions
├── widgets/      # Complex UI blocks (quiz player, surprise chart)
├── features/     # User scenarios (answer, score, share)
├── entities/     # Domain models (quiz, user)
└── shared/       # UI primitives, types, utilities
```

## License

[MIT](./LICENSE)
