# Clinical Note Extraction Pipeline

LLM-based **structured extraction** from unstructured clinical text. Messy
synthetic clinical notes go in; clean structured fields come out, with
extraction accuracy measured against known ground truth and per-run token/cost
tracking.

> Standalone companion to `clinical-data-pipeline` — conceptually the
> unstructured-text front-end that would feed a structured classifier, but it
> stands on its own.

## Pipeline (5 stages)

```
1. GENERATE  synthetic clinical notes (unstructured text + hidden ground truth)
2. EXTRACT   structured fields from each note via LLM (with caching)
3. NORMALIZE extracted values into consistent formats
4. EVALUATE  extraction vs. ground truth (accuracy per field)
5. DISPLAY   results + token/cost tracking in a Streamlit dashboard
```

## Fields extracted

`chief_complaint`, `duration`, `medical_history`, `blood_pressure`,
`heart_rate`, `medications` — some deliberately missing per note, so extraction
must return null rather than hallucinate.

## Setup

```bash
# 1. Create and activate the environment (Miniforge/Conda)
conda env create -f environment.yml
conda activate clinical-note-extraction

# 2. Add your API key
cp .env.example .env      # then edit .env and paste your real key
```

## Running

_Build in progress — run instructions land as each stage is built (Steps 2–6)._

## Design notes

_Filled in at Step 7._
