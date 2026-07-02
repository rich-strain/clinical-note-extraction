"""
LLM-based structured extraction from clinical note text.

Takes a note's messy text and asks Claude to pull out the 6 target fields as
JSON. This is the "Extract" half of the pipeline: raw text in, a Python dict
out, plus the token usage so we can track cost.

A local file cache sits in front of the API call. Every note in a batch gets
re-processed each time you iterate on the pipeline (tune the prompt, add a
field, fix a bug in normalization) — without a cache you'd pay for the same
API call over and over during development. The cache key includes a
"prompt version" string alongside the note text specifically so that
changing the prompt invalidates old cached results: if we didn't do this,
improving the prompt would silently keep returning stale extractions from
the old prompt version.
"""

import hashlib
import json
import os
import re
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

# The model is isolated here in one place so Step 8 (benchmarking Haiku vs.
# Sonnet) only has to change this one line, or pass a different value in.
MODEL = "claude-haiku-4-5"

# Bump this string whenever the extraction prompt changes. It's folded into
# the cache key, so an old cached result (produced by the old prompt) won't
# be mistaken for a result from the new prompt — the cache miss forces a
# fresh API call instead of silently serving stale data.
PROMPT_VERSION = "v4"

CACHE_DIR = Path(__file__).parent / "cache"

TARGET_FIELDS = [
    "chief_complaint",
    "duration",
    "medical_history",
    "blood_pressure",
    "heart_rate",
    "medications",
]

# The synthetic notes are generated from a small, fixed vocabulary (see
# note_generator.py's CHIEF_COMPLAINTS / MEDICAL_HISTORY pools) — real
# clinical terminology normalization would map free text onto a large
# standard ontology (e.g. SNOMED CT), but here the full universe of correct
# answers is exactly this closed list. Giving the model that list and asking
# it to normalize onto it turns "difficulty breathing" -> "shortness of
# breath" instead of returning the literal phrase from the note, which is
# what ground truth expects.
CHIEF_COMPLAINT_TERMS = [
    "chest pain",
    "shortness of breath",
    "abdominal pain",
    "headache",
    "dizziness",
    "back pain",
]
MEDICAL_HISTORY_TERMS = [
    "hypertension",
    "type 2 diabetes",
    "asthma",
    "hyperlipidemia",
]

EXTRACTION_PROMPT_TEMPLATE = """You are extracting structured data from a clinical note for a data curation pipeline.

Extract exactly these 6 fields from the note below. Each value must be a
single JSON string (or null) — never a list, and never a bare JSON number
even for numeric fields like heart_rate (write "105", not 105). If the
note mentions more than one item for a field, pick the single most
clinically significant one.
- chief_complaint: the main reason for the visit
- duration: how long the symptom has been present
- medical_history: relevant prior conditions
- blood_pressure: systolic/diastolic reading
- heart_rate: beats per minute
- medications: current medications mentioned

Normalization:
- For chief_complaint, map the note's wording to the closest matching standard term from this list: {chief_complaint_terms}. For example, "difficulty breathing" or "SOB" should be normalized to "shortness of breath".
- For medical_history, map the note's wording to the closest matching standard term from this list: {medical_history_terms}. For example, "HTN" should be normalized to "hypertension".
- If the note's wording doesn't clearly match any term in the list, use your best standard clinical terminology instead of the literal phrase.

Rules:
- If a field is not mentioned in the note, use null for that field. Do NOT guess or invent a value.
- Return ONLY a JSON object with exactly these 6 keys: chief_complaint, duration, medical_history, blood_pressure, heart_rate, medications.
- Do not include any text before or after the JSON object.

Note:
{note_text}
"""

_client = anthropic.Anthropic()


def _cache_key(note_text: str) -> str:
    """Hash of (note_text + prompt_version) — see module docstring for why
    the prompt version is included."""
    raw = f"{PROMPT_VERSION}:{note_text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_path(cache_key: str) -> Path:
    return CACHE_DIR / f"{cache_key}.json"


def _parse_json_response(text: str) -> dict:
    """
    Parse the model's JSON response safely.

    Models sometimes wrap JSON in markdown code fences (```json ... ```) or
    add a stray sentence before/after the object even when explicitly told
    not to. Strip fences first, then fall back to extracting the outermost
    {...} block if a direct parse fails.
    """
    cleaned = text.strip()

    # Strip markdown code fences if present (```json ... ``` or ``` ... ```)
    fence_match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fall back to grabbing the first {...} block anywhere in the text.
    brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if brace_match:
        return json.loads(brace_match.group(0))

    raise ValueError(f"Could not parse JSON from model response: {text!r}")


def extract_fields(note_text: str, force_refresh: bool = False) -> dict:
    """
    Extract the 6 target fields from a single note's text.

    Returns a dict with two keys:
      - "fields": the extracted dict (each of the 6 target fields, or None
        if the model didn't find it in the note)
      - "usage": {"input_tokens": int, "output_tokens": int} — zero for a
        cache hit, since no API call was made

    Caching: before calling the API, check for a cached result keyed on
    hash(note_text + PROMPT_VERSION). On a hit, load it and skip the API
    call entirely. On a miss, call the API and save the result. Pass
    force_refresh=True to bypass the cache and always call the API (e.g.
    when you suspect the cached result is bad, independent of a prompt
    version bump).
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_key = _cache_key(note_text)
    cache_file = _cache_path(cache_key)

    if not force_refresh and cache_file.exists():
        with open(cache_file, "r") as f:
            cached = json.load(f)
        return {"fields": cached["fields"], "usage": {"input_tokens": 0, "output_tokens": 0}}

    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        chief_complaint_terms=", ".join(CHIEF_COMPLAINT_TERMS),
        medical_history_terms=", ".join(MEDICAL_HISTORY_TERMS),
        note_text=note_text,
    )
    response = _client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = next(b.text for b in response.content if b.type == "text")
    fields = _parse_json_response(response_text)

    # Guarantee exactly the 6 expected keys, even if the model added extras
    # or dropped one — downstream code (normalizer, evaluator) can then rely
    # on the shape without defensive checks at every call site.
    fields = {key: fields.get(key) for key in TARGET_FIELDS}

    # Belt-and-suspenders: the prompt asks for single, quoted-string values,
    # but the model doesn't always comply — it occasionally returns a list
    # (e.g. multiple history items) or a bare JSON number for a numeric
    # field like heart_rate. Force every non-null value to a plain str here
    # so downstream normalization/evaluation can always assume a string (or
    # None) and never has to type-check before comparing.
    for key, value in fields.items():
        if isinstance(value, list):
            value = value[0] if value else None
        if value is not None and not isinstance(value, str):
            value = str(value)
        fields[key] = value

    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }

    with open(cache_file, "w") as f:
        json.dump({"fields": fields, "usage": usage}, f, indent=2)

    return {"fields": fields, "usage": usage}


if __name__ == "__main__":
    # Quick manual smoke test against one real note.
    from note_generator import generate_notes

    note = generate_notes(1, seed=1)[0]
    print("--- Note text ---")
    print(note["text"])
    print()
    print("--- Ground truth ---")
    print(note["ground_truth"])
    print()

    result = extract_fields(note["text"])
    print("--- Extracted (API call) ---")
    print(result["fields"])
    print("Usage:", result["usage"])
    print()

    # Second call with the same text should hit the cache (zero token usage).
    result2 = extract_fields(note["text"])
    print("--- Extracted (cache hit) ---")
    print(result2["fields"])
    print("Usage:", result2["usage"])
