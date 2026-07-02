# Project: Clinical Note Extraction Pipeline

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
