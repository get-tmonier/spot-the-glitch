# Spot the Glitch

> Can you beat an AI that learned physics from scratch?

A minimal web game exploring world model capabilities through interactive physics violation detection, based on the [LeWorldModel (LeWM)](https://arxiv.org/abs/2603.19312) paper.

[![Demo](https://img.shields.io/badge/demo-live-blue.svg)](https://spot-the-glitch.vercel.app)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue.svg)](https://www.typescriptlang.org/)
[![Next.js](https://img.shields.io/badge/Next.js-16-black.svg)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What is this?

You're shown pairs of 3-second video clips. One is real, one contains a subtle physics violation. Your job: identify the glitched clip.

The twist? An AI world model, trained only on pixel observations, achieves perfect accuracy by detecting "surprise" in its predictions. Humans typically score 4-6/10.

**The pedagogical goal**: demonstrate that modern world models capture intuitive physics understanding that emerges without explicit supervision.

*After each answer, see exactly where the AI detected the violation through its surprise signal.*

## How it works

### The AI side

1. **World Model Training**: A lightweight JEPA (Joint-Embedding Predictive Architecture) learns environment dynamics from raw pixels using the LeWorldModel approach
2. **Violation Detection**: Physical glitches (object teleportation, impossible movements) spike the model's prediction error
3. **Surprise Quantification**: The error magnitude becomes a physics violation detector

### The human side

- 10 carefully calibrated question pairs
- Real-time scoring with AI comparison
- Shareable results via generated Open Graph images
- Interactive surprise curve visualization post-answer

## Technical architecture

**LeWM Model** (Python/PyTorch) → **Clip Generator** (MP4 + JSON) → **Web Frontend** (Next.js/React)

- ViT Encoder, Transformer, SIGReg loss → Normal clips, Glitched clips, Surprise data → Quiz logic, Surprise viz, Result sharing

### Core technologies

- **Frontend**: Next.js 16 + TypeScript + Tailwind CSS
- **Runtime**: Bun for fast package management and development
- **Animations**: Framer Motion for smooth transitions
- **Charts**: Recharts for surprise curve visualization  
- **ML Pipeline**: LeWorldModel (PyTorch) for physics understanding
- **Deployment**: Vercel with edge caching for global performance

## Getting started

### Prerequisites

- Bun 1.0+
- Git

### Local development

```bash
# Clone the repository
git clone https://github.com/get-tmonier/spot-the-glitch.git
cd spot-the-glitch

# Install dependencies
bun install

# Start development server
bun dev
```

Open [http://localhost:3000](http://localhost:3000) to play the game.

### Project structure

```
spot-the-glitch/
├── app/                    # Next.js App Router pages
│   ├── play/              # Quiz interface
│   ├── result/            # Score & sharing screen
│   └── api/og/            # Open Graph image generation
├── components/
│   ├── ui/                # Reusable UI primitives
│   └── quiz/              # Game-specific components
├── lib/                   # Types and utilities
├── public/
│   ├── clips/             # Pre-generated MP4 files
│   └── quiz-data.json     # Question pairs & surprise data
└── scripts/               # ML pipeline (Python)
```

## Research foundation

This project builds on [LeWorldModel](https://arxiv.org/abs/2603.19312) by Maes et al. (2026):

> *LeWorldModel (LeWM), the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer enforcing Gaussian-distributed latent embeddings.*

**Key insight**: World models that learn purely from observational data develop emergent capabilities to detect violations of physical laws, without any explicit physics supervision.

### Citation

```bibtex
@article{maes2026leworldmodel,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint arXiv:2603.19312},
  year={2026}
}
```

## Contributing

This is an experimental project exploring the intersection of world models and interactive learning. Contributions are welcome:

- **Bug reports**: Found an issue? Open an [issue](https://github.com/get-tmonier/spot-the-glitch/issues)
- **Feature ideas**: Suggest improvements or new game modes
- **Code contributions**: See [CONTRIBUTING.md](./CONTRIBUTING.md) for development guidelines

### Development priorities

1. **Mobile optimization**: Ensure smooth playback on all devices
2. **Difficulty calibration**: Fine-tune glitch subtlety for optimal challenge
3. **Educational content**: Expand explanations of world model concepts
4. **Accessibility**: Screen reader support and keyboard navigation

## Inspiration

Inspired by successful viral AI demos like:
- [Which Face is Real?](https://whichfaceisreal.com) - Human vs. AI discrimination
- [Human or Not?](https://humanornot.ai) - Reverse Turing test

The format of "humans vs. AI perception" creates an engaging way to explore AI capabilities beyond traditional benchmarks.

## License

[MIT](./LICENSE) - feel free to build upon this work.

## Author

**Damien Meur** · [Website](https://tmonier.com) · [Twitter](https://twitter.com/damienmeur) · [GitHub](https://github.com/damienmeur)

*Built during exploration of world model applications. Part of ongoing research into AI agent supervision at [Tmonier](https://tmonier.com).*

---

*💡 Interested in world models for production systems? Check out [Vigie](https://github.com/get-tmonier/vigie), my local-first agent supervision toolkit.*
