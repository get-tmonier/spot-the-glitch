# Spot the Glitch

## Project overview

Viral web game based on the LeWorldModel paper (arXiv:2603.19312). Users spot physics glitches in video clip pairs, then compare their score to an AI world model.

## Tech stack

- Next.js 16 (App Router, Turbopack) + TypeScript strict + ESM only
- Tailwind CSS v4
- Bun as package manager and runtime
- tsgo (`@typescript/native-preview`) for type checking
- oxlint for linting, knip for dead code/dependency detection, dependency-cruiser for FSD boundary enforcement
- Framer Motion, Recharts, @vercel/og

## Architecture

Feature-Sliced Design (FSD) with all source under `src/`:

- `src/app/` — Next.js App Router pages + global providers
- `src/views/` — Route-level page compositions (named `views` to avoid Next.js `pages` conflict)
- `src/widgets/` — Complex UI blocks (quiz-player, score-display, surprise-chart)
- `src/features/` — User scenarios (answer-question, track-score, share-result)
- `src/entities/` — Domain models (quiz, user)
- `src/shared/` — UI primitives, types, lib, config

**FSD import rules** (enforced by `bun run fsd`):
- Layers can only import from layers below: shared < entities < features < widgets < views < app
- No cross-slice imports within the same layer (e.g. feature A cannot import from feature B)

## Commands

```bash
bun dev            # Start dev server (localhost:3000)
bun run build      # Production build
bun run verify     # Run ALL checks (type-check, format, lint, fsd, knip, test)
bun run verify:fix # Auto-fix what's fixable, then verify

# Individual checks
bun run type-check # tsgo --noEmit (must pass with 0 errors)
bun run format     # oxfmt --check (formatting)
bun run format:fix # oxfmt --write (auto-format)
bun run lint       # oxlint (0 warnings)
bun run lint:fix   # oxlint --fix
bun run fsd        # FSD layer boundary enforcement (dependency-cruiser)
bun run knip       # Dead code & unused dependency detection
bun run knip:fix   # Auto-remove unused exports/deps
bun test           # bun:test runner
bun run deps:check # Check for outdated deps (ncu, 3-day cooldown)
bun run deps:update # Update deps to latest pinned versions
```

## Path aliases

`@/app/*`, `@/views/*`, `@/widgets/*`, `@/features/*`, `@/entities/*`, `@/shared/*` — all resolve to `src/<layer>/*`.

## Design system

- Background: `#0a0a0a` (--bg), text: `#ededed` (--fg), accent: `#3b82f6` (--accent)
- Mobile-first responsive design
- Shared UI primitives in `src/shared/ui/`: Button, Card, Container, H1, H2, Text

## Conventions

- All dependencies use pinned versions (no ^ or ~)
- ESM only (`"type": "module"`)
- TypeScript strict mode
