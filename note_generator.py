"""
Synthetic clinical note generator.

Real clinical notes are unlabeled: nobody hands you the "correct" chief
complaint or blood pressure alongside the free-text note. That means you
can't directly measure how well an extraction pipeline is doing against
real data without a human manually labeling a validation set first.

This module sidesteps that by generating notes FROM known values. Each
note is built by picking a chief complaint, duration, history, blood
pressure, heart rate, and medication list from fixed pools, then rendering
them into messy, doctor-shorthand text. Because we chose the values
ourselves, we already know the "ground truth" — so every note comes back
as a (messy_text, ground_truth_dict) pair. That pairing is what makes
Step 5 (evaluation) possible: we can compare what the LLM extracted
against what we know is actually correct, field by field.

Notes are generated in one of two documentation styles (see
LABELED_NOTE_PROBABILITY below): labeled/structured, where vitals carry
explicit labels ("HR 100, BP 137/100") — the realistic majority, testing
whether the extractor uses labels rather than assuming numbers are unique;
and bare/messy, where vitals appear as unlabeled numbers — the harder
minority, testing whether the extractor can still cope once the context
that would normally disambiguate a value is gone.
"""

import random

# --- Value pools -----------------------------------------------------------
# Each pool pairs a "canonical" ground-truth value with the note phrasing(s)
# that might express it. Keeping these separate is what lets normalization
# (Step 4) and evaluation (Step 5) check against one consistent answer even
# though the generated text varies.

CHIEF_COMPLAINTS = [
    ("chest pain", ["chest pain", "c/o chest pain", "CP"]),
    ("shortness of breath", ["shortness of breath", "SOB", "difficulty breathing"]),
    ("abdominal pain", ["abdominal pain", "abd pain", "c/o abd pain"]),
    ("headache", ["headache", "HA", "c/o headache", "migraine"]),
    ("dizziness", ["dizziness", "lightheadedness", "c/o feeling dizzy"]),
    ("back pain", ["back pain", "lower back pain", "c/o back pain"]),
]

DURATIONS = [
    ("2 days", ["2 days", "x2 days", "for the past 2 days"]),
    ("3 hours", ["3 hours", "x3 hrs", "started 3 hours ago"]),
    ("1 week", ["1 week", "x1 wk", "for about a week"]),
    ("30 minutes", ["30 minutes", "x30 min", "started half an hour ago"]),
    ("5 days", ["5 days", "x5 days", "for the last 5 days"]),
]

MEDICAL_HISTORY = [
    ("hypertension", ["hypertension", "HTN", "hx of HTN"]),
    ("type 2 diabetes", ["type 2 diabetes", "T2DM", "hx of diabetes"]),
    ("asthma", ["asthma", "hx of asthma"]),
    ("hyperlipidemia", ["hyperlipidemia", "high cholesterol", "hx of high cholesterol"]),
]

MEDICATIONS = [
    ("lisinopril", ["lisinopril", "takes lisinopril daily"]),
    ("metformin", ["metformin", "on metformin"]),
    ("albuterol", ["albuterol", "uses albuterol inhaler prn"]),
    ("atorvastatin", ["atorvastatin", "takes atorvastatin"]),
]

# Blood pressure and heart rate are generated as numbers rather than picked
# from a fixed pool, then rendered into one of several messy formats.
BP_SYSTOLIC_RANGE = (100, 170)
BP_DIASTOLIC_RANGE = (60, 100)
HEART_RATE_RANGE = (55, 110)

# Real notes split roughly into two documentation styles, and vitals are
# rendered differently depending on which style a note is in:
#
#   - Labeled/structured (the realistic majority): vitals carry explicit
#     labels, e.g. "HR 100, BP 137/100" or "Pulse: 88. BP: 120/80." The
#     label disambiguates the number, so it's fine for HR and BP to
#     coincidentally share a value — a reader (or the extractor) can tell
#     them apart from the label alone. This is the common real-world case
#     and tests whether the extractor actually uses labels rather than
#     assuming numbers are unique.
#   - Bare/messy (the harder minority): vitals appear as unlabeled numbers
#     floating in a sentence, e.g. "59." or "137-100." — the kind of
#     fragment you get from a hurried note or a dropped OCR label. Here a
#     repeated number IS genuinely ambiguous (is "100" restating diastolic,
#     or is it the heart rate?), so the collision guard in generate_note()
#     re-rolls heart rate until it can't be confused with the BP values.
#     This style tests whether the extractor can cope when the context
#     that would normally disambiguate a value is simply missing — the
#     harder tail where a real curation system earns its value.
#
# One constant controls the mix so it's easy to tune later.
LABELED_NOTE_PROBABILITY = 0.7

BP_LABELED_FORMATS = [
    "BP {sys}/{dia}",
    "BP: {sys}/{dia}",
    "BP {sys} over {dia}",
]
BP_BARE_FORMATS = [
    "{sys}/{dia}",
    "{sys} over {dia}",
    "{sys}-{dia}",
]

HR_LABELED_FORMATS = [
    "HR {hr}",
    "HR: {hr}",
    "HR {hr} bpm",
    "Pulse: {hr}",
    "Pulse {hr} bpm",
]
HR_BARE_FORMATS = [
    "{hr}",
]

# Probability that each OPTIONAL field is actually mentioned in the note.
# chief_complaint is not in here — it's always present, since a visit
# without any stated reason isn't a realistic note.
FIELD_INCLUSION_PROBABILITY = {
    "duration": 0.75,
    "medical_history": 0.6,
    "blood_pressure": 0.8,
    "heart_rate": 0.8,
    "medications": 0.55,
}


def _pick(pool: list[tuple[str, list[str]]], rng: random.Random) -> tuple[str, str]:
    """Pick a (canonical_value, rendered_phrase) pair from a value pool."""
    canonical, phrasings = rng.choice(pool)
    phrase = rng.choice(phrasings)
    return canonical, phrase


def generate_note(rng: random.Random | None = None) -> dict:
    """
    Generate a single synthetic clinical note.

    Returns a dict with two keys:
      - "text": the messy, unstructured note text (what the LLM will see)
      - "ground_truth": the correct value for each of the 6 target fields,
        with None for any field the note doesn't mention

    Accepting an optional `rng` (rather than always using the global random
    module) lets generate_notes() reuse one seeded Random instance across
    many notes, so a whole batch is reproducible from a single seed.
    """
    rng = rng or random.Random()

    ground_truth = {
        "chief_complaint": None,
        "duration": None,
        "medical_history": None,
        "blood_pressure": None,
        "heart_rate": None,
        "medications": None,
    }

    # Chief complaint is always present — every note needs a stated reason
    # for the visit.
    cc_canonical, cc_phrase = _pick(CHIEF_COMPLAINTS, rng)
    ground_truth["chief_complaint"] = cc_canonical
    parts = [f"Pt presents with {cc_phrase}."]

    if rng.random() < FIELD_INCLUSION_PROBABILITY["duration"]:
        dur_canonical, dur_phrase = _pick(DURATIONS, rng)
        ground_truth["duration"] = dur_canonical
        parts.append(f"Duration: {dur_phrase}.")

    if rng.random() < FIELD_INCLUSION_PROBABILITY["medical_history"]:
        hx_canonical, hx_phrase = _pick(MEDICAL_HISTORY, rng)
        ground_truth["medical_history"] = hx_canonical
        parts.append(f"PMH: {hx_phrase}.")

    # Decide this note's documentation style up front, since it governs how
    # both vitals get rendered below.
    is_labeled = rng.random() < LABELED_NOTE_PROBABILITY

    sys = dia = hr = None

    if rng.random() < FIELD_INCLUSION_PROBABILITY["blood_pressure"]:
        sys = rng.randint(*BP_SYSTOLIC_RANGE)
        dia = rng.randint(*BP_DIASTOLIC_RANGE)
        ground_truth["blood_pressure"] = f"{sys}/{dia}"

    if rng.random() < FIELD_INCLUSION_PROBABILITY["heart_rate"]:
        hr = rng.randint(*HEART_RATE_RANGE)
        if not is_labeled:
            # Bare notes have no label to disambiguate a repeated number,
            # so re-roll until heart rate can't be confused with either BP
            # value. Labeled notes skip this entirely — "HR 100, BP
            # 137/100" is unambiguous because of the labels.
            used_numbers = {v for v in (sys, dia) if v is not None}
            while hr in used_numbers:
                hr = rng.randint(*HEART_RATE_RANGE)
        ground_truth["heart_rate"] = str(hr)

    if is_labeled and sys is not None and dia is not None and hr is not None and rng.random() < 0.5:
        # Occasionally render both vitals as a single labeled block, e.g.
        # "HR 100, BP 137/100." — mirrors how vitals are often grouped
        # together in real charting rather than split into separate lines.
        bp_phrase = rng.choice(BP_LABELED_FORMATS).format(sys=sys, dia=dia)
        hr_phrase = rng.choice(HR_LABELED_FORMATS).format(hr=hr)
        parts.append(f"{hr_phrase}, {bp_phrase}.")
    else:
        if sys is not None and dia is not None:
            bp_format = rng.choice(BP_LABELED_FORMATS if is_labeled else BP_BARE_FORMATS)
            parts.append(bp_format.format(sys=sys, dia=dia) + ".")
        if hr is not None:
            hr_format = rng.choice(HR_LABELED_FORMATS if is_labeled else HR_BARE_FORMATS)
            parts.append(hr_format.format(hr=hr) + ".")

    if rng.random() < FIELD_INCLUSION_PROBABILITY["medications"]:
        med_canonical, med_phrase = _pick(MEDICATIONS, rng)
        ground_truth["medications"] = med_canonical
        parts.append(f"Meds: {med_phrase}.")

    # Shuffle sentence order (after the opening complaint sentence) so the
    # model can't rely on field position — real notes don't follow a fixed
    # template either.
    opening, rest = parts[0], parts[1:]
    rng.shuffle(rest)
    text = " ".join([opening] + rest)

    return {"text": text, "ground_truth": ground_truth}


def generate_notes(n: int, seed: int | None = None) -> list[dict]:
    """
    Generate `n` synthetic notes at once.

    Passing the same `seed` reproduces the exact same batch of notes later
    — useful for keeping a stable test set across pipeline runs while
    iterating on the extractor or normalizer.
    """
    rng = random.Random(seed)
    return [generate_note(rng) for _ in range(n)]


if __name__ == "__main__":
    # Quick manual smoke test: print a few notes side by side with their
    # ground truth so you can eyeball the messiness and confirm missing
    # fields show up as None.
    for i, note in enumerate(generate_notes(5, seed=42), start=1):
        print(f"--- Note {i} ---")
        print(note["text"])
        print(note["ground_truth"])
        print()
