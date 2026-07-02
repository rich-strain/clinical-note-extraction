"""
Evaluate normalized extractions against ground truth.

This is the same evaluation discipline used to measure a classifier — take
a set of predictions, compare each one against a known-correct label, and
compute accuracy — just applied to extraction instead of classification.
For a classifier you'd ask "did it predict the right class?" per example;
here we ask "did it extract the right value?" per field, per note. Having
ground truth built into every generated note (Step 2) is what makes this
possible at all: without it there'd be no known-correct answer to compare
against, only a subjective impression of whether the output "looks right."

A field counts as correct when its normalized extracted value exactly
equals the ground-truth value — including the case where both are None
(the note didn't mention the field, and the pipeline correctly didn't
invent one). That symmetry matters: an extractor that hallucinates a value
for an absent field should score exactly as wrong as one that misses a
field that was actually present.
"""

from collections import Counter

# The only place scikit-learn enters this pipeline: chief_complaint and
# medical_history are the two fields mapped onto a closed vocabulary (see
# extractor.py's CHIEF_COMPLAINT_TERMS / MEDICAL_HISTORY_TERMS), which makes
# them genuinely classification-shaped — the other 4 fields are open-ended
# extraction and don't have a fixed label set to build a confusion matrix
# from.
from sklearn.metrics import confusion_matrix

from extractor import CHIEF_COMPLAINT_TERMS, MEDICAL_HISTORY_TERMS

FIELD_NAMES = [
    "chief_complaint",
    "duration",
    "medical_history",
    "blood_pressure",
    "heart_rate",
    "medications",
]


def is_correct(ground_truth_value: str | None, normalized_value: str | None) -> bool:
    """A field is correct on exact match, including None == None (correctly
    identifying that the note doesn't mention this field)."""
    return ground_truth_value == normalized_value


def evaluate_note(ground_truth: dict, normalized: dict) -> dict[str, bool]:
    """Compare one note's ground truth against its normalized extraction,
    field by field."""
    return {
        field: is_correct(ground_truth.get(field), normalized.get(field))
        for field in FIELD_NAMES
    }


def evaluate_batch(pairs: list[tuple[dict, dict]]) -> dict:
    """
    Evaluate a batch of (ground_truth, normalized_extraction) pairs.

    Returns a summary dict:
      - "n_notes": how many notes were evaluated
      - "overall_accuracy": correct field-checks / total field-checks, across
        every note and every field
      - "per_field_accuracy": {field_name: accuracy} for each of the 6 fields
      - "per_field_counts": {field_name: {"correct": int, "total": int}} —
        the raw counts behind each accuracy, useful for display
      - "weakest_fields": the field(s) with the lowest accuracy (ties
        included), so a quick glance shows where the pipeline is weakest
    """
    n_notes = len(pairs)
    correct_counts = Counter()
    total_counts = Counter()

    for ground_truth, normalized in pairs:
        results = evaluate_note(ground_truth, normalized)
        for field, correct in results.items():
            total_counts[field] += 1
            if correct:
                correct_counts[field] += 1

    per_field_accuracy = {
        field: correct_counts[field] / total_counts[field] for field in FIELD_NAMES
    }
    per_field_counts = {
        field: {"correct": correct_counts[field], "total": total_counts[field]}
        for field in FIELD_NAMES
    }

    total_correct = sum(correct_counts.values())
    total_checks = sum(total_counts.values())
    overall_accuracy = total_correct / total_checks if total_checks else 0.0

    lowest_accuracy = min(per_field_accuracy.values()) if per_field_accuracy else 0.0
    weakest_fields = [
        field for field, acc in per_field_accuracy.items() if acc == lowest_accuracy
    ]

    return {
        "n_notes": n_notes,
        "overall_accuracy": overall_accuracy,
        "per_field_accuracy": per_field_accuracy,
        "per_field_counts": per_field_counts,
        "weakest_fields": weakest_fields,
    }


def _bucket_label(value: str | None, canonical_terms: list[str]) -> str:
    """Map a value onto the closed-vocabulary axis for a confusion matrix.

    A value outside the canonical terms (missing, or something the
    extractor invented that isn't in the vocabulary) still needs a bucket —
    silently excluding it from the matrix would hide exactly the kind of
    failure a confusion matrix is meant to surface.
    """
    if value is None:
        return "(missing)"
    if value not in canonical_terms:
        return "(other)"
    return value


def confusion_matrix_for_field(
    pairs: list[tuple[dict, dict]], field: str, canonical_terms: list[str]
) -> tuple[list[list[int]], list[str]]:
    """
    Build a confusion matrix for one closed-vocabulary field across a batch.

    Rows = ground truth, columns = predicted (normalized extraction) —
    sklearn's confusion_matrix convention. Labels are the field's canonical
    terms plus two catch-all buckets, "(other)" (an extracted value outside
    the vocabulary) and "(missing)" (None), so every note is accounted for
    rather than silently dropped from the matrix.

    Returns (matrix, labels) — the ordered label list is required to read
    the matrix's rows/columns correctly.
    """
    labels = list(canonical_terms) + ["(other)", "(missing)"]
    y_true = [_bucket_label(gt.get(field), canonical_terms) for gt, _ in pairs]
    y_pred = [_bucket_label(pred.get(field), canonical_terms) for _, pred in pairs]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return matrix.tolist(), labels


def chief_complaint_confusion_matrix(
    pairs: list[tuple[dict, dict]],
) -> tuple[list[list[int]], list[str]]:
    """Confusion matrix for chief_complaint against its closed vocabulary."""
    return confusion_matrix_for_field(pairs, "chief_complaint", CHIEF_COMPLAINT_TERMS)


def medical_history_confusion_matrix(
    pairs: list[tuple[dict, dict]],
) -> tuple[list[list[int]], list[str]]:
    """Confusion matrix for medical_history against its closed vocabulary."""
    return confusion_matrix_for_field(pairs, "medical_history", MEDICAL_HISTORY_TERMS)


def print_confusion_matrix(matrix: list[list[int]], labels: list[str], title: str) -> None:
    """Pretty-print a confusion matrix as a labeled grid (rows = ground
    truth, columns = predicted), with a legend for any truncated column
    headers — readable in a terminal, not a raw array dump."""
    row_label_width = max(len(label) for label in labels) + 2
    col_width = 10

    def short(label: str) -> str:
        return label if len(label) <= col_width - 1 else label[: col_width - 2] + "…"

    print(title)
    print("(rows = ground truth, columns = predicted)")
    header = " " * row_label_width + "".join(f"{short(l):>{col_width}}" for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row_str = row_label.ljust(row_label_width) + "".join(
            f"{matrix[i][j]:>{col_width}}" for j in range(len(labels))
        )
        print(row_str)

    truncated = [l for l in labels if short(l) != l]
    if truncated:
        print("Legend (truncated column headers):")
        for l in truncated:
            print(f"  {short(l)} = {l}")


def print_summary(summary: dict) -> None:
    """Pretty-print an evaluate_batch() summary to the console."""
    print(f"Notes evaluated: {summary['n_notes']}")
    print(f"Overall accuracy: {summary['overall_accuracy']:.1%}")
    print()
    print("Per-field accuracy:")
    for field in FIELD_NAMES:
        acc = summary["per_field_accuracy"][field]
        counts = summary["per_field_counts"][field]
        marker = " <-- weakest" if field in summary["weakest_fields"] else ""
        print(f"  {field:<18} {acc:6.1%}  ({counts['correct']}/{counts['total']}){marker}")


if __name__ == "__main__":
    # End-to-end smoke test: generate notes, extract, normalize, evaluate.
    from note_generator import generate_notes
    from extractor import extract_fields
    from normalizer import normalize_fields

    notes = generate_notes(20, seed=7)
    pairs = []
    for note in notes:
        raw = extract_fields(note["text"])["fields"]
        normalized = normalize_fields(raw)
        pairs.append((note["ground_truth"], normalized))

    summary = evaluate_batch(pairs)
    print_summary(summary)
    print()

    matrix, labels = chief_complaint_confusion_matrix(pairs)
    print_confusion_matrix(matrix, labels, title="chief_complaint confusion matrix")
