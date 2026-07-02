"""
Normalize raw extracted field values into consistent formats.

The extractor (Step 3) already asks the LLM to normalize chief_complaint and
medical_history to a closed clinical vocabulary, but it does NOT ask it to
normalize the *formatting* of duration, blood pressure, or heart rate — so
those fields can come back however the source note happened to phrase them
("150 over 90" vs "150/90", "x2 days" vs "for the past 2 days"). This module
is the "Transform" step of the pipeline: extract -> normalize -> evaluate,
the same shape as an ETL pipeline's extract -> transform -> load. It's also
the same real-world problem health-data systems solve when standardizing
values into a fixed schema like FHIR — different source systems and
clinicians write the same fact in different formats, and downstream
consumers (billing, analytics, clinical decision support) need one
consistent representation to compare against.

Every function here is pure and total: given any input (including None),
it returns a normalized value or None, and never raises on malformed input
— a value the normalizer can't confidently parse is returned stripped but
otherwise unchanged, rather than dropped, so evaluation can still see it.
"""

import re

# --- Blood pressure -----------------------------------------------------

# Matches "150/90", "150-90", "150 over 90", "150 / 90", with or without
# a leading "BP" / "BP:" label that may have survived from the note text.
_BP_PATTERN = re.compile(
    r"(?:bp\s*:?\s*)?(\d{2,3})\s*(?:/|-|over)\s*(\d{2,3})",
    re.IGNORECASE,
)


def normalize_blood_pressure(value: str | None) -> str | None:
    """Standardize any blood pressure phrasing to "systolic/diastolic"."""
    if value is None:
        return None
    match = _BP_PATTERN.search(value)
    if match:
        systolic, diastolic = match.group(1), match.group(2)
        return f"{systolic}/{diastolic}"
    return value.strip()


# --- Duration -------------------------------------------------------------

# Maps unit abbreviations/variants to a canonical unit name. Order matters
# only in that longer/more-specific keys should not be shadowed by shorter
# ones, which isn't an issue here since each entry is checked by regex
# alternation, not prefix matching.
_DURATION_UNIT_CANONICAL = {
    "day": "day", "days": "day",
    "hour": "hour", "hours": "hour", "hr": "hour", "hrs": "hour",
    "week": "week", "weeks": "week", "wk": "week", "wks": "week",
    "minute": "minute", "minutes": "minute", "min": "minute", "mins": "minute",
}

_DURATION_PATTERN = re.compile(
    r"(\d+)\s*(days?|hours?|hrs?|weeks?|wks?|minutes?|mins?)\b",
    re.IGNORECASE,
)


def normalize_duration(value: str | None) -> str | None:
    """
    Standardize duration phrasing to "N unit" (singular for N == 1), e.g.
    "for the past 2 days" -> "2 days", "x30 min" -> "30 minutes",
    "started 3 hrs ago" -> "3 hours", "started half an hour ago" -> "30 minutes".
    """
    if value is None:
        return None
    if re.search(r"half\s+an?\s+hour", value, re.IGNORECASE):
        return "30 minutes"
    match = _DURATION_PATTERN.search(value)
    if not match:
        return value.strip()
    quantity = int(match.group(1))
    unit = _DURATION_UNIT_CANONICAL[match.group(2).lower()]
    if quantity != 1:
        unit += "s"
    return f"{quantity} {unit}"


# --- Heart rate -------------------------------------------------------------


def normalize_heart_rate(value: str | None) -> str | None:
    """Strip any label/unit text down to the bare numeric string, e.g.
    "HR 105" or "105 bpm" -> "105"."""
    if value is None:
        return None
    match = re.search(r"\d+", value)
    if match:
        return match.group(0)
    return value.strip()


# --- Text fields (chief_complaint, medical_history, medications) ----------


def normalize_text_field(value: str | None) -> str | None:
    """Strip whitespace and lowercase for consistent, case-insensitive
    comparison. The extractor already maps chief_complaint/medical_history
    onto a closed vocabulary, so this is mostly a safety net against stray
    whitespace or inconsistent casing rather than heavy lifting."""
    if value is None:
        return None
    cleaned = value.strip().lower()
    return cleaned if cleaned else None


# --- Top-level entry point --------------------------------------------------

_FIELD_NORMALIZERS = {
    "chief_complaint": normalize_text_field,
    "duration": normalize_duration,
    "medical_history": normalize_text_field,
    "blood_pressure": normalize_blood_pressure,
    "heart_rate": normalize_heart_rate,
    "medications": normalize_text_field,
}


def normalize_fields(fields: dict) -> dict:
    """Apply the appropriate normalizer to each of the 6 target fields."""
    return {
        key: _FIELD_NORMALIZERS[key](fields.get(key)) for key in _FIELD_NORMALIZERS
    }


if __name__ == "__main__":
    # Quick manual smoke test covering the messy formats each normalizer
    # needs to handle.
    sample = {
        "chief_complaint": "  Shortness Of Breath ",
        "duration": "for the past 2 days",
        "medical_history": "HYPERTENSION",
        "blood_pressure": "150 over 90",
        "heart_rate": "HR 105",
        "medications": " Lisinopril ",
    }
    print("--- Before ---")
    print(sample)
    print("--- After ---")
    print(normalize_fields(sample))

    print()
    print("--- None passthrough ---")
    print(normalize_fields({k: None for k in _FIELD_NORMALIZERS}))
