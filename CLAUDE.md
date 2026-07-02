# Project: Clinical Note Extraction Pipeline

## Output style

- Keep terminal/build output concise. Avoid verbose explanations of every internal step while running commands — show results, not narration.
- When explaining code changes, keep it to a brief summary (what changed, why) rather than a full walkthrough, unless I ask for more detail.
- Prefer showing me the diff/output over describing it in prose when both are available.

## Environment

- Conda (Miniforge) for environment management
- API key loaded from .env — never hardcode or commit it

## Conventions

- Favor clear, readable code with explanatory comments
- Colorblind-safe palette: blue/amber/gray, never red/green or blue/purple
- Never encode meaning with color alone — always add labels
- Cache API results; only call the API on input/prompt changes

## Architecture

- 5-stage pipeline: generate → extract → normalize → evaluate → display
- Modular files, each independently testable
