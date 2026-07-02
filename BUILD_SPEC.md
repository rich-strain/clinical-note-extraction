# Clinical Note Extraction Pipeline — Build Spec

## For Claude Code

This is a **learning-focused** build. Do NOT generate the entire project at once.
Build it **one component at a time**, in the order below. After each component:

1. Explain what the code does and why, in plain language
2. Stop and let me test it before moving to the next component
3. Wait for my confirmation before continuing

I am transitioning into data science / AI engineering from a full-stack background.
I understand code well but am newer to ML/LLM concepts, so favor clear explanations
of the _why_ over dense implementation. Teach as you build.

---

## Project Overview

A standalone Python project that demonstrates **LLM-based structured extraction**
from unstructured clinical text — a core problem in health-data curation. It takes
messy synthetic clinical notes, extracts structured fields via the Anthropic API,
normalizes them, and measures extraction accuracy against known ground truth.

This is a **separate repo** from my existing `clinflow-data-pipeline` project.
Conceptually it is the unstructured-text front-end that would feed clinflow's
structured classifier, but it stands on its own.

---

## The Pipeline (5 stages)

```
1. GENERATE  synthetic clinical notes (unstructured text + hidden ground truth)
2. EXTRACT   structured fields from each note via LLM (with caching)
3. NORMALIZE extracted values into consistent formats
4. EVALUATE  extraction vs. ground truth (accuracy per field)
5. DISPLAY   results + token/cost tracking in a Streamlit dashboard
```

---

## Target Fields (6)

Extract these six fields from each note:

1. `chief_complaint` — the main reason for the visit (e.g., "chest pain")
2. `duration` — how long the symptom has been present (e.g., "2 days")
3. `medical_history` — relevant prior conditions (e.g., "hypertension")
4. `blood_pressure` — systolic/diastolic (e.g., "150/90")
5. `heart_rate` — beats per minute (e.g., "88")
6. `medications` — current medications mentioned (e.g., "lisinopril")

Some notes should deliberately have MISSING fields (not every note mentions every
field), so extraction has to correctly return null/empty for absent data rather
than hallucinating a value. This is important for demonstrating robustness.

---

## Build Order (one component per step)

### STEP 1 — Project scaffolding

- Create the repo structure, a Conda-compatible `environment.yml` (I use Miniforge),
  a `requirements.txt`, a `.gitignore` (ignore the cache folder, .env, secrets),
  and a `README.md` stub.
- Set up a `.env` file pattern for the Anthropic API key (NEVER hardcode the key;
  load it from the environment). Add `.env` to `.gitignore`.
- Explain the folder structure and why each piece exists.
- STOP for testing.

### STEP 2 — Note generator (`note_generator.py`)

- Template-based synthetic clinical note generation.
- Each generated note returns TWO things: the messy text string AND a ground-truth
  dict of the correct values embedded in it.
- Include deliberate messiness: medical abbreviations (c/o, hx, SOB, HTN), varied
  formats for the same value (BP as "150/90" or "150 over 90"), and randomly
  omitted fields (so some ground-truth values are null).
- Include a function to generate N notes at once.
- Explain how having ground truth built in enables evaluation later.
- STOP for testing.

### STEP 3 — Extractor (`extractor.py`)

- A function that takes a note's text and calls the Anthropic API (use the
  `anthropic` Python SDK) with a structured-extraction prompt that requests
  JSON output for the 6 target fields.
- Use **Claude Haiku** (cheapest model, well-suited to extraction) — model string
  should be easy to change in one place.
- Parse the JSON response safely (handle the case where the model wraps it in
  markdown code fences or adds stray text).
- Capture the `usage` object (input_tokens, output_tokens) from each response.
- Return the extracted dict PLUS the token usage.
- **Caching layer**: before calling the API, check a local file cache.
  - Cache key = hash of (note_text + prompt_version_string).
  - On cache hit, load the saved result and skip the API call.
  - On cache miss, call the API and save the result.
  - Include a `force_refresh` parameter to bypass the cache.
  - Store cache files as JSON in a `cache/` folder (gitignored).
- Explain the caching logic, especially why the prompt version is in the cache key
  (so improving the prompt correctly invalidates stale cached results).
- Explain the cost implications and roughly what a run costs.
- STOP for testing.

### STEP 4 — Normalizer (`normalizer.py`)

- Takes raw extracted values and standardizes formats:
  - Blood pressure: "150 over 90", "150-90", "150/90" all become "150/90"
  - Duration: normalize to a consistent phrasing where reasonable
  - Strip whitespace, lowercase where appropriate, handle None/missing cleanly
- Use regex and plain Python (pandas optional if it helps).
- Explain that this mirrors the "Transform" step in an ETL pipeline and the
  real-world problem of normalizing inconsistent source formats (FHIR-style
  standardization).
- STOP for testing.

### STEP 5 — Evaluator (`evaluator.py`)

- Compares normalized extracted values against ground truth, per field, across
  all notes.
- Calculates per-field accuracy (what % of notes had this field extracted
  correctly).
- Handles the missing-field case correctly (extracting null when ground truth
  is null counts as correct).
- Produces a summary: overall accuracy + per-field breakdown, and flags the
  weakest field(s).
- Explain how this is the same evaluation discipline as measuring a classifier —
  comparing predictions against ground truth — just applied to extraction.
- STOP for testing.

### STEP 6 — Streamlit dashboard (`app.py`)

- Runs the full pipeline end to end.
- Shows a before/after view: the messy note text next to the clean structured
  output.
- Displays per-field accuracy as a bar chart using Plotly.
- **IMPORTANT — colorblind-safe palette (red-green colorblind / deuteranopia-protanopia).**
  Follow these rules strictly for every chart, legend, and colored UI element:
  - **Safe primary contrast**: blue vs. amber/orange. This is the most reliable
    distinguishable pair — use it as the default for any two-way comparison.
  - **Safe secondary contrast**: teal vs. coral, or blue vs. gray.
  - **Neutral**: gray.
  - **NEVER use these pairs (indistinguishable or hard):**
    - red vs. green (the core problem — never use together)
    - red vs. brown (hard depending on shade)
    - blue vs. purple (hard when adjacent)
    - green vs. brown, green vs. amber (green is unreliable against warm tones)
  - **Minimize green and red overall.** If a status color is unavoidable, do NOT
    rely on hue alone — pair it with a text label, icon, or shape so the meaning
    is never carried by color by itself.
  - **Don't encode meaning with color alone anywhere.** Always add a second cue:
    direct text labels on bars, distinct patterns/shapes, or clear legends with
    text. A colorblind viewer should be able to read every chart with the color
    stripped out entirely.
  - When in doubt, default to the blue/amber/gray family and label directly.
- Displays token consumption (total input/output tokens) and estimated cost for
  the run (tokens x per-token rate).
- Includes a "Force refresh (ignore cache)" checkbox to demo cached vs. live paths.
- Explain how this maps to the core value proposition of clinical data curation
  (messy in, clean structured out, with measurable quality).
- STOP for testing.

### STEP 7 — README + polish

- Write a clear README explaining the project, the pipeline, the design decisions
  (labeled synthetic data for measurable accuracy, caching for cost control,
  Haiku for cost efficiency), how to run it, and how it relates conceptually to
  real health-data extraction.
- Note the deliberate simplifications and what production would require (real
  unlabeled data, human review to build validation sets, PHI/compliance handling).
- STOP for testing.

### STEP 8 (optional) — Model comparison benchmark

- Add the ability to run the SAME extraction across two models (Claude Haiku and
  Claude Sonnet) and compare them on BOTH accuracy and cost.
- The model string is already isolated in one place (from Step 3), so this step
  builds a small benchmark harness that runs the full note set through each model,
  evaluates each against ground truth, and records: per-field accuracy, overall
  accuracy, total tokens used, and estimated cost — for each model.
- Produce a simple side-by-side comparison (table or chart) showing the
  accuracy-vs-cost tradeoff between the two models.
- IMPORTANT: this will make real API calls against BOTH models, so it costs more
  than a normal cached run. Keep the note set small for the benchmark (e.g. 20-50
  notes) and rely on the cache so re-runs are free. Warn me before running it live
  and show the estimated cost first.
- Explain how this mirrors the "simple baseline vs. more capable option, measured
  against metrics" reasoning used in classical ML model selection (e.g. Logistic
  Regression vs. Random Forest) — here applied to choosing an LLM for a task based
  on a measured cost-quality tradeoff, not just a guess.
- This step produces strong portfolio/interview material: a defensible,
  data-driven model choice ("Haiku was X% accurate at 1/5 the cost, so the
  tradeoff favored Haiku") rather than an arbitrary one.

---

## Key Design Decisions (context for why the project is built this way)

- **Labeled synthetic data**: I generate the notes myself so I have ground truth,
  which lets me MEASURE extraction accuracy rather than just eyeballing output.
  In production the data would be unlabeled real notes — a harder problem requiring
  human review to build a validation set. The synthetic approach is a deliberate
  simplification that makes results measurable.
- **Caching**: keeps API costs near-zero during iterative development; only calls
  the API when the input or prompt actually changes.
- **Haiku model**: cheapest current model, well-suited to extraction/classification
  tasks — a deliberate cost-conscious choice.
- **Token/cost tracking**: demonstrates awareness of the unit economics of running
  an LLM feature (cost per note, cost per thousand records).
- **Modular files**: each pipeline stage is independently testable and mirrors an
  ETL structure (extract → transform → load/evaluate).
- **Swappable model + optional benchmark**: the extraction model is isolated so it
  can be changed in one place. An optional final step benchmarks Haiku vs. Sonnet
  on accuracy AND cost, turning "I picked a model" into "I measured and justified a
  model choice" — the same baseline-vs-more-capable reasoning used in classical ML
  model selection, applied to LLM selection.

---

## Constraints & Preferences

- I use **Conda (Miniforge)** for environment management.
- Load the API key from an environment variable / `.env` file — NEVER hardcode it,
  NEVER commit it.
- Keep secrets and the `cache/` folder out of git.
- Favor clear, readable code with explanatory comments over clever/dense code.
- Explain each component as you build it — I am learning, not just shipping.
- Build incrementally and STOP between components for testing.
- Ask if a commit has been pushed before moving to the next step.
