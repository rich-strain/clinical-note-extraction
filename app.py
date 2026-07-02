"""
Streamlit dashboard: run the full clinical note extraction pipeline and
show the results.

This ties every prior step together into the thing the project is actually
demonstrating: messy, unstructured clinical text goes in; clean, structured,
*measurably accurate* data comes out. That last word is the point — "looks
right" isn't a claim you can act on in a data-curation pipeline, but "94.2%
field-level accuracy, weakest on duration" is. That's the whole value
proposition of this project in one sentence, and this page is where it
becomes visible: generate -> extract -> normalize -> evaluate, with the
messy input and clean output shown side by side, and the accuracy numbers
that tell you whether to trust them.
"""

import hashlib

import plotly.graph_objects as go
import streamlit as st

from evaluator import FIELD_NAMES, evaluate_batch, evaluate_note
from extractor import MODEL, extract_fields
from normalizer import normalize_fields
from note_generator import generate_notes

# --- Colorblind-safe palette -------------------------------------------
# Validated with the dataviz skill's CVD-separation script (protanopia /
# deuteranopia simulation): blue vs. amber is the safe primary contrast
# pair. Never red/green anywhere in this file.
COLOR_NORMAL = "#2563EB"  # blue — every field's bar, by default
COLOR_WEAKEST = "#D97706"  # amber — the weakest field(s), highlighted
COLOR_NEUTRAL = "#9CA3AF"  # gray — neutral UI chrome (not used for meaning)

# Haiku 4.5 pricing, per million tokens. Kept next to MODEL (imported from
# extractor.py) so if the model ever changes there, this rate is the one
# thing left to update by hand.
INPUT_RATE_PER_MILLION = 1.00
OUTPUT_RATE_PER_MILLION = 5.00


def run_pipeline(n_notes: int, seed: int, force_refresh: bool) -> dict:
    """
    Run generate -> extract -> normalize -> evaluate for n_notes notes.

    Returns a dict with everything the UI needs to render: the notes
    themselves, their normalized extractions, per-note per-field
    correctness, the aggregate evaluation summary, and total token usage.
    """
    notes = generate_notes(n_notes, seed=seed)

    normalized_predictions = []
    per_note_correctness = []
    total_input_tokens = 0
    total_output_tokens = 0

    for note in notes:
        result = extract_fields(note["text"], force_refresh=force_refresh)
        total_input_tokens += result["usage"]["input_tokens"]
        total_output_tokens += result["usage"]["output_tokens"]

        normalized = normalize_fields(result["fields"])
        normalized_predictions.append(normalized)
        per_note_correctness.append(evaluate_note(note["ground_truth"], normalized))

    pairs = [
        (note["ground_truth"], normalized)
        for note, normalized in zip(notes, normalized_predictions)
    ]
    summary = evaluate_batch(pairs)

    return {
        "notes": notes,
        "normalized_predictions": normalized_predictions,
        "per_note_correctness": per_note_correctness,
        "summary": summary,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }


def render_metrics(run: dict) -> None:
    summary = run["summary"]
    input_tokens = run["total_input_tokens"]
    output_tokens = run["total_output_tokens"]
    cost = (
        input_tokens / 1_000_000 * INPUT_RATE_PER_MILLION
        + output_tokens / 1_000_000 * OUTPUT_RATE_PER_MILLION
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Notes processed", summary["n_notes"])
    col2.metric("Overall accuracy", f"{summary['overall_accuracy']:.1%}")
    col3.metric("Tokens (in / out)", f"{input_tokens:,} / {output_tokens:,}")
    col4.metric("Estimated cost", f"${cost:.4f}")


def render_accuracy_chart(summary: dict) -> None:
    st.subheader("Per-field accuracy")

    weakest = set(summary["weakest_fields"])
    accuracies = [summary["per_field_accuracy"][f] for f in FIELD_NAMES]
    colors = [COLOR_WEAKEST if f in weakest else COLOR_NORMAL for f in FIELD_NAMES]

    # Never encode "weakest" by color alone: every bar carries its exact
    # percentage as a direct text label, and the weakest bar's x-axis tick
    # additionally says so in plain text — a colorblind viewer (or the
    # color stripped out entirely) can still read the chart correctly.
    tick_labels = [
        f"{f.replace('_', ' ')}<br>(weakest)" if f in weakest else f.replace("_", " ")
        for f in FIELD_NAMES
    ]

    fig = go.Figure(
        go.Bar(
            x=FIELD_NAMES,
            y=accuracies,
            marker_color=colors,
            text=[f"{a:.0%}" for a in accuracies],
            textposition="outside",
        )
    )
    fig.update_layout(
        yaxis=dict(title="Accuracy", tickformat=".0%", range=[0, 1.1]),
        xaxis=dict(
            title=None,
            tickmode="array",
            tickvals=FIELD_NAMES,
            ticktext=tick_labels,
        ),
        showlegend=False,
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Blue = normal, amber = weakest field(s) (also labeled directly on the "
        "x-axis, not color alone)."
    )


def render_before_after(run: dict) -> None:
    st.subheader("Before / after")

    for note, normalized, correctness in zip(
        run["notes"], run["normalized_predictions"], run["per_note_correctness"]
    ):
        with st.expander(note["text"][:80] + ("..." if len(note["text"]) > 80 else "")):
            col_before, col_after = st.columns(2)
            with col_before:
                st.markdown("**Messy note text**")
                st.text(note["text"])
            with col_after:
                st.markdown("**Extracted (normalized)**")
                rows = []
                for field in FIELD_NAMES:
                    is_match = correctness[field]
                    rows.append(
                        {
                            "field": field,
                            "extracted": normalized[field] or "—",
                            "ground truth": note["ground_truth"][field] or "—",
                            # Symbol carries meaning, not color — readable
                            # even with color stripped out entirely.
                            "match": "correct" if is_match else "wrong",
                        }
                    )
                st.dataframe(rows, hide_index=True, use_container_width=True)


def check_password() -> bool:
    """
    Gate the app behind a single shared password.

    This is not real multi-user auth — one password for everyone, no
    usernames, no per-user anything. It exists to stop a public deployment
    from letting anyone trigger paid Anthropic API calls, not to protect
    sensitive data (the notes are synthetic). The password itself is never
    stored anywhere; only its SHA-256 hash lives in Streamlit Secrets, so a
    leaked secrets file still doesn't reveal the plaintext password.
    """
    if st.session_state.get("authenticated"):
        return True

    stored_hash = st.secrets.get("APP_PASSWORD_HASH")
    if not stored_hash:
        st.error(
            "No APP_PASSWORD_HASH configured in Streamlit Secrets — the app "
            "can't verify a password. See README for setup."
        )
        return False

    password = st.text_input("Password", type="password")
    if not password:
        return False

    entered_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
    if entered_hash == stored_hash:
        st.session_state["authenticated"] = True
        return True

    st.error("Incorrect password.")
    return False


def main() -> None:
    st.set_page_config(page_title="Clinical Note Extraction", layout="wide")

    if not check_password():
        st.stop()

    st.title("Clinical Note Extraction Pipeline")
    st.markdown(
        "Messy, unstructured clinical text in; clean, structured data out — "
        "with a measured accuracy score, not just a plausible-looking guess. "
        f"Extraction runs on **{MODEL}** with a local cache, so a re-run of "
        "the same notes at the same prompt version costs nothing after the "
        "first pass."
    )

    with st.sidebar:
        st.header("Run settings")
        n_notes = st.slider("Number of notes", min_value=5, max_value=50, value=15)
        seed = st.number_input("Random seed", value=42, step=1)
        force_refresh = st.checkbox(
            "Force refresh (ignore cache)",
            value=False,
            help="Bypass the cache and call the API for every note, even if "
            "a cached result already exists. Use this to demo the "
            "cached-vs-live cost difference.",
        )
        run_clicked = st.button("Run pipeline", type="primary")

    if run_clicked:
        with st.spinner("Running generate -> extract -> normalize -> evaluate..."):
            st.session_state["run"] = run_pipeline(n_notes, int(seed), force_refresh)

    run = st.session_state.get("run")
    if run is None:
        st.info("Set your run settings in the sidebar and click **Run pipeline**.")
        return

    render_metrics(run)
    render_accuracy_chart(run["summary"])
    render_before_after(run)


if __name__ == "__main__":
    main()
