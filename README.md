# ARCprize2025_HIN: Hybrid Intelligence Network 🧠

A hybrid approach to ARC-AGI combining Deep Variational Autoencoders (D-VAE) and Meta-Reinforcement Learning.

![Hybrid ARC System](arc_hybrid_system_visual.png)

## Overview
This project explores the intersection of neural and symbolic reasoning for the Abstraction and Reasoning Corpus (ARC). The **Hybrid Intelligence Network (HIN)** aims to bridge the gap between implicit latent representations and explicit program synthesis.

### Core Architecture
- **D-VAE (Visual Encoding):** Learns a compact, plausible latent space for grid configurations.
- **Meta-RL Agent:** A Transformer-based policy that synthesizes symbolic programs to solve tasks.
- **Program Executor:** A robust, custom interpreter for executing and verifying generated code.

## Key Research Components
- **Evolutionary Test-Time Compute (ETC):** Dynamic population-based refinement of solution functions.
- **Self-Supervised Case Memory:** Learning from solved tasks to guide future reasoning steps.
- **Curriculum Learning:** Systematic progression from simple patterns to complex abstract rules.

## Project Structure
- `baselines/`: Collection of baseline models (VAE, RL, etc.).
- `data/`: Curated HIN datasets and augmented training samples.
- `gan_trainer/`: Generative Adversarial Network implementation for grid generation.
- `docs/`: Project plans, research notes, and architectural advice.

## Installation & Usage
```bash
# Set up environment
pip install -r requirements.txt

# Run GAN trainer
python gantrain.py
```

---
*Part of the ARC Prize 2025 research.*

