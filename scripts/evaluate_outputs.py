#!/usr/bin/env python3
"""
Objective evaluation for base-versus-fine-tuned SLM story outputs.

Behavior target: given 1–3 phonics focuses, the model should generate a
decodable story that emphasizes requested patterns while staying at the
requested progression level (minimizing above-level spelling patterns),
forming a complete narrative, and avoiding phonics-term leakage.

Primary objective metric: estimated above-level / decodability compliance
(rule-based spelling-pattern classifier). Target satisfaction and full-spec
pass remain secondary, because an advanced story can hit requested patterns
accidentally.

These metrics are estimated rule-based measures from phonics_profiler_threshold4
detection logic — not a perfect measure of true linguistic decodability.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from phonics_profiler_threshold4 import (
    classify_word,
    extract_words,
)

# Repo-root-relative defaults (scripts/ is one level below repo root).
REPO_ROOT = Path(__file__).resolve().parents[1]
INPUTS_DIR = REPO_ROOT / "inputs"
OBJECTIVE_DIR = REPO_ROOT / "results" / "objective"

# ---------------------------------------------------------------------------
# Display names (as stored in target_phonics) <-> internal profiler labels
# ---------------------------------------------------------------------------
DISPLAY_TO_INTERNAL = {
    "short-vowel words": "short_vowel",
    "consonant blends": "blend",
    "consonant digraphs": "digraph",
    "final-e words": "final_e",
    "vowel teams": "vowel_team",
    "r-controlled vowels": "r_controlled",
    "diphthongs": "diphthong",
    "multisyllabic words": "multisyllabic",
}

INTERNAL_TO_DISPLAY = {v: k for k, v in DISPLAY_TO_INTERNAL.items()}

# Evaluation hierarchy (higher number = more advanced).
# Blends and digraphs share one stage; l_controlled rides with r_controlled
# because the profiler places both at the same instructional level.
STAGE_RANK = {
    "short_vowel": 1,
    "blend": 2,
    "digraph": 2,
    "final_e": 3,
    "vowel_team": 4,
    "r_controlled": 5,
    "l_controlled": 5,
    "diphthong": 6,
    "multisyllabic": 7,
}

# Patterns reported in the row-level profile (user-facing set).
PROFILE_PATTERNS = [
    "short_vowel",
    "blend",
    "digraph",
    "final_e",
    "vowel_team",
    "r_controlled",
    "diphthong",
    "multisyllabic",
]

MIN_DISTINCT_FOR_PASS = 4

# Phonics-term leakage phrases (matched case-insensitively as substrings).
LEAKAGE_TERMS = (
    "phonics",
    "short vowel",
    "final e",
    "magic e",
    "vowel team",
    "digraph",
    "consonant blend",
    "r-controlled",
    "r controlled",
    "diphthong",
    "multisyllabic",
)

REQUIRED_COLUMNS = (
    "model",
    "prompt_id",
    "generation_id",
    "seed",
    "target_phonics",
    "prompt",
    "story",
)

TITLE_PREFIX_RE = re.compile(
    r"^\s*(?:title\s*[:\-–—]\s*)+",
    re.IGNORECASE,
)
MARKDOWN_BOLD_RE = re.compile(r"^\s*\*\*(.+?)\*\*\s*$")
MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S+")


def parse_target_phonics(raw: str) -> list[str]:
    """Parse a comma-separated target_phonics cell into display-name list."""
    if raw is None:
        return []
    parts = [p.strip() for p in str(raw).split(",")]
    return [p for p in parts if p]


def validate_targets(targets: list[str], row_index: int) -> list[str]:
    """Ensure every requested target maps to a known pattern."""
    unknown = [t for t in targets if t not in DISPLAY_TO_INTERNAL]
    if unknown:
        raise ValueError(
            f"Row {row_index}: unknown target_phonics value(s): {unknown}. "
            f"Expected one of: {sorted(DISPLAY_TO_INTERNAL)}"
        )
    return targets


def highest_stage_for_word(patterns: set[str]) -> int:
    """
    Assign a word to its highest detected hierarchy stage.

    short_vowel is derived from the profiler's basic_or_unclassified bucket
    (no Level 3–8 target pattern, and not an irregular/heart word).
    Words with only irregular / no hierarchy label receive stage 0.
    """
    stages = [STAGE_RANK[p] for p in patterns if p in STAGE_RANK]
    return max(stages) if stages else 0


def derive_short_vowel(patterns: set[str]) -> bool:
    """
    Short-vowel words in this pipeline are the profiler's basic bucket:
    no advanced target pattern and not irregular.
    """
    advanced = {
        "digraph",
        "blend",
        "final_e",
        "vowel_team",
        "r_controlled",
        "l_controlled",
        "diphthong",
        "multisyllabic",
    }
    return (not patterns.intersection(advanced)) and ("irregular" not in patterns)


def profile_story_patterns(story: str) -> dict:
    """
    Profile a story with the existing classify_word logic.

    Returns distinct words / counts per pattern, token list, and per-word
    highest stage (for estimated above-level / decodability metrics).
    """
    tokens = extract_words(story)
    token_counts = Counter(tokens)
    unique_words = sorted(token_counts)

    pattern_words: dict[str, list[str]] = defaultdict(list)
    word_highest_stage: dict[str, int] = {}

    for word in unique_words:
        result = classify_word(word)
        patterns = set(result["patterns"])

        # Derive short-vowel from existing classification, not a new regex.
        if derive_short_vowel(patterns):
            patterns.add("short_vowel")

        for pattern in patterns:
            if pattern in PROFILE_PATTERNS or pattern == "l_controlled":
                pattern_words[pattern].append(word)

        word_highest_stage[word] = highest_stage_for_word(patterns)

    profile = {
        "tokens": tokens,
        "token_counts": token_counts,
        "unique_words": unique_words,
        "word_count": len(tokens),
        "unique_word_count": len(unique_words),
        "word_highest_stage": word_highest_stage,
        "pattern_words": {},
        "pattern_distinct_counts": {},
    }

    for pattern in PROFILE_PATTERNS:
        words = sorted(set(pattern_words.get(pattern, [])))
        profile["pattern_words"][pattern] = words
        profile["pattern_distinct_counts"][pattern] = len(words)

    return profile


def permitted_ceiling(targets: list[str]) -> int:
    """Highest hierarchy stage among requested target patterns."""
    if not targets:
        return 0
    return max(STAGE_RANK[DISPLAY_TO_INTERNAL[t]] for t in targets)


def compute_level_metrics(profile: dict, ceiling: int) -> dict:
    """
    Estimated rule-based above-level / decodability metrics.

    Each word is assigned only to its highest detected stage so the same word
    is not double-counted across patterns. Progression severity uses how many
    hierarchy stages above the permitted ceiling an above-level word sits.
    """
    token_counts: Counter = profile["token_counts"]
    word_highest_stage: dict[str, int] = profile["word_highest_stage"]
    total = profile["word_count"]
    unique_total = profile["unique_word_count"]

    above_words: list[str] = []
    distance_occurrence_sum = 0.0
    above_occurrence_distances: list[int] = []
    on_or_below_occurrences = 0

    if ceiling <= 0:
        # No usable ceiling: do not mark vocabulary as above-level.
        on_or_below_occurrences = total
    else:
        for word, stage in word_highest_stage.items():
            freq = token_counts[word]
            if stage > ceiling:
                distance = stage - ceiling
                above_words.append(word)
                distance_occurrence_sum += distance * freq
                above_occurrence_distances.extend([distance] * freq)
            else:
                # Includes stage 0 (irregular / unlabeled) as not above-level.
                on_or_below_occurrences += freq

    above_words = sorted(above_words)
    above_occurrences = sum(token_counts[w] for w in above_words)
    distinct_above = len(above_words)

    above_level_rate = (above_occurrences / total) if total else 0.0
    distinct_above_level_rate = (
        (distinct_above / unique_total) if unique_total else 0.0
    )
    on_or_below_level_rate = (
        (on_or_below_occurrences / total) if total else 0.0
    )
    weighted_above_level_rate = (
        (distance_occurrence_sum / total) if total else 0.0
    )

    if above_occurrence_distances:
        mean_above_level_distance = (
            sum(above_occurrence_distances) / len(above_occurrence_distances)
        )
        max_above_level_distance = max(above_occurrence_distances)
    else:
        mean_above_level_distance = 0.0
        max_above_level_distance = 0

    decodability_compliance = 1.0 - above_level_rate
    distinct_decodability_compliance = 1.0 - distinct_above_level_rate

    return {
        "above_level_words": ", ".join(above_words),
        "above_level_word_occurrences": above_occurrences,
        "above_level_distinct_words": distinct_above,
        # Primary objective family (estimated rule-based measures)
        "above_level_rate": round(above_level_rate, 4),
        "distinct_above_level_rate": round(distinct_above_level_rate, 4),
        "decodability_compliance": round(decodability_compliance, 4),
        "distinct_decodability_compliance": round(
            distinct_decodability_compliance, 4
        ),
        "on_or_below_level_rate": round(on_or_below_level_rate, 4),
        "mean_above_level_distance": round(mean_above_level_distance, 4),
        "max_above_level_distance": int(max_above_level_distance),
        "weighted_above_level_rate": round(weighted_above_level_rate, 4),
    }


def detect_phonics_leakage(story: str) -> tuple[bool, str]:
    """Return whether the story mentions phonics metalanguage, plus matches."""
    text = story or ""
    lower = text.lower()
    hits = [term for term in LEAKAGE_TERMS if term in lower]
    deduped = sorted(set(hits))
    return (len(deduped) > 0, ", ".join(deduped))


def detect_title(story: str) -> tuple[bool, str]:
    """
    Heuristic title detection for generated stories.

    Looks at the opening lines for common title forms used in the outputs
    (markdown bold, 'Title:', heading, or a short first line before a blank).
    """
    if not story or not story.strip():
        return False, ""

    lines = story.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return False, ""

    first = lines[0].strip()

    bold = MARKDOWN_BOLD_RE.match(first)
    if bold:
        return True, bold.group(1).strip()

    if MARKDOWN_HEADING_RE.match(first):
        return True, re.sub(r"^\s{0,3}#{1,6}\s*", "", first).strip()

    if TITLE_PREFIX_RE.match(first):
        title = TITLE_PREFIX_RE.sub("", first).strip().strip("*").strip()
        return True, title

    if (
        len(lines) >= 3
        and first
        and not lines[1].strip()
        and lines[2].strip()
        and len(first.split()) <= 12
        and not first.endswith((".", "?", "!"))
    ):
        return True, first

    return False, ""


def evaluate_row(row: dict, row_index: int) -> dict:
    """Build the full objective-eval record for one CSV row."""
    for col in REQUIRED_COLUMNS:
        if col not in row:
            raise KeyError(f"Row {row_index}: missing required column '{col}'")

    story = row.get("story") or ""
    targets = validate_targets(
        parse_target_phonics(row.get("target_phonics", "")), row_index
    )
    profile = profile_story_patterns(story)

    out = {
        "model": row["model"],
        "prompt_id": row["prompt_id"],
        "generation_id": row["generation_id"],
        "seed": row["seed"],
        "target_phonics": row["target_phonics"],
        "word_count": profile["word_count"],
        "unique_word_count": profile["unique_word_count"],
    }

    for pattern in PROFILE_PATTERNS:
        display = INTERNAL_TO_DISPLAY[pattern]
        words = profile["pattern_words"][pattern]
        count = profile["pattern_distinct_counts"][pattern]
        out[f"{pattern}_words"] = ", ".join(words)
        out[f"{pattern}_distinct_count"] = count
        safe = display.replace(" ", "_").replace("-", "_")
        out[f"{safe}_distinct_count"] = count

    # Secondary metrics: target pattern evidence (>= 4 distinct matches)
    targets_requested = len(targets)
    passed = []
    failed = []
    for display_name in targets:
        internal = DISPLAY_TO_INTERNAL[display_name]
        distinct = profile["pattern_distinct_counts"].get(internal, 0)
        ok = distinct >= MIN_DISTINCT_FOR_PASS
        out[f"target_pass__{internal}"] = int(ok)
        out[f"target_distinct__{internal}"] = distinct
        if ok:
            passed.append(display_name)
        else:
            failed.append(display_name)

    targets_passed = len(passed)
    satisfaction = (
        round(targets_passed / targets_requested, 4) if targets_requested else 0.0
    )
    full_spec_pass = int(
        targets_requested > 0 and targets_passed == targets_requested
    )

    out["targets_requested"] = targets_requested
    out["targets_passed"] = targets_passed
    out["targets_passed_list"] = ", ".join(passed)
    out["targets_failed_list"] = ", ".join(failed)
    out["target_satisfaction_rate"] = satisfaction
    out["full_spec_pass"] = full_spec_pass

    ceiling = permitted_ceiling(targets)
    out["permitted_ceiling_stage"] = ceiling
    out["permitted_ceiling_label"] = next(
        (
            name
            for name, internal in DISPLAY_TO_INTERNAL.items()
            if STAGE_RANK[internal] == ceiling and name in targets
        ),
        "",
    )
    out.update(compute_level_metrics(profile, ceiling))

    leaked, leak_terms = detect_phonics_leakage(story)
    out["phonics_leakage"] = int(leaked)
    out["phonics_leakage_terms"] = leak_terms

    has_title, title_text = detect_title(story)
    out["has_title"] = int(has_title)
    out["detected_title"] = title_text

    return out


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_group(rows: list[dict], group_keys: dict) -> dict:
    """Shared aggregation used by model-level and prompt-level summaries."""
    n = len(rows)
    return {
        **group_keys,
        "n": n,
        "mean_decodability_compliance": round(
            _mean([r["decodability_compliance"] for r in rows]), 4
        ),
        "mean_distinct_decodability_compliance": round(
            _mean([r["distinct_decodability_compliance"] for r in rows]), 4
        ),
        "mean_weighted_above_level_rate": round(
            _mean([r["weighted_above_level_rate"] for r in rows]), 4
        ),
        "mean_above_level_distance": round(
            _mean([r["mean_above_level_distance"] for r in rows]), 4
        ),
        "mean_above_level_rate": round(
            _mean([r["above_level_rate"] for r in rows]), 4
        ),
        "full_spec_pass_rate": round(
            _mean([r["full_spec_pass"] for r in rows]), 4
        ),
        "mean_target_satisfaction_rate": round(
            _mean([r["target_satisfaction_rate"] for r in rows]), 4
        ),
        "phonics_leakage_rate": round(
            _mean([r["phonics_leakage"] for r in rows]), 4
        ),
        "title_rate": round(_mean([r["has_title"] for r in rows]), 4),
        "mean_word_count": round(_mean([r["word_count"] for r in rows]), 2),
    }


def summarize_by_model(rows: list[dict]) -> list[dict]:
    """Aggregate row-level metrics into a model-level summary."""
    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    return [
        summarize_group(by_model[model], {"model": model})
        for model in sorted(by_model)
    ]


def summarize_by_model_prompt(rows: list[dict]) -> list[dict]:
    """Aggregate metrics by model x prompt_id for condition-level diagnosis."""
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_key[(row["model"], row["prompt_id"])].append(row)

    summary = []
    for model, prompt_id in sorted(by_key):
        group = by_key[(model, prompt_id)]
        target_phonics = group[0].get("target_phonics", "")
        summary.append(
            summarize_group(
                group,
                {
                    "model": model,
                    "prompt_id": prompt_id,
                    "target_phonics": target_phonics,
                },
            )
        )
    return summary


def build_paired_comparison(rows: list[dict]) -> list[dict]:
    """
    Pair base vs tuned on (prompt_id, generation_id, seed).

    compliance_delta = tuned - base  (positive => tuned improved)
    weighted_rate_delta = tuned - base  (negative => tuned improved)
    satisfaction_delta = tuned - base
    """
    indexed: dict[tuple[str, str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = (
            str(row["prompt_id"]),
            str(row["generation_id"]),
            str(row["seed"]),
        )
        indexed[key][row["model"]] = row

    pairs = []
    unmatched = 0
    for key, models in sorted(indexed.items()):
        base = models.get("base")
        tuned = models.get("tuned")
        if base is None or tuned is None:
            unmatched += 1
            continue

        compliance_delta = (
            tuned["decodability_compliance"] - base["decodability_compliance"]
        )
        weighted_rate_delta = (
            tuned["weighted_above_level_rate"] - base["weighted_above_level_rate"]
        )
        satisfaction_delta = (
            tuned["target_satisfaction_rate"] - base["target_satisfaction_rate"]
        )

        pairs.append(
            {
                "prompt_id": key[0],
                "generation_id": key[1],
                "seed": key[2],
                "target_phonics": base.get("target_phonics", ""),
                "base_decodability_compliance": base["decodability_compliance"],
                "tuned_decodability_compliance": tuned["decodability_compliance"],
                "compliance_delta": round(compliance_delta, 4),
                "base_weighted_above_level_rate": base["weighted_above_level_rate"],
                "tuned_weighted_above_level_rate": tuned["weighted_above_level_rate"],
                "weighted_rate_delta": round(weighted_rate_delta, 4),
                "base_target_satisfaction_rate": base["target_satisfaction_rate"],
                "tuned_target_satisfaction_rate": tuned["target_satisfaction_rate"],
                "satisfaction_delta": round(satisfaction_delta, 4),
                "base_word_count": base["word_count"],
                "tuned_word_count": tuned["word_count"],
            }
        )

    return pairs, unmatched


def pct(rate: float) -> str:
    """Format a 0–1 rate as a percentage string."""
    return f"{100.0 * rate:.1f}%"


def print_model_summary(summary_rows: list[dict]) -> None:
    """Print model-level summary with percentage-friendly primary metrics."""
    if not summary_rows:
        print("No summary rows to display.")
        return

    headers = [
        ("model", "model", False),
        ("n", "n", False),
        ("mean_decodability_compliance", "decod_comp", True),
        ("mean_distinct_decodability_compliance", "dist_decod", True),
        ("mean_weighted_above_level_rate", "wt_above", False),
        ("mean_above_level_distance", "mean_dist", False),
        ("full_spec_pass_rate", "full_spec", True),
        ("mean_target_satisfaction_rate", "tgt_sat", True),
        ("phonics_leakage_rate", "leakage", True),
        ("title_rate", "title", True),
        ("mean_word_count", "mean_words", False),
    ]

    display_rows = []
    for row in summary_rows:
        display = {}
        for key, label, as_pct in headers:
            value = row[key]
            display[label] = pct(value) if as_pct else str(value)
        display_rows.append(display)

    labels = [label for _, label, _ in headers]
    widths = []
    for label in labels:
        width = len(label)
        for display in display_rows:
            width = max(width, len(display[label]))
        widths.append(width)

    def fmt_row(values: list[str]) -> str:
        return "  ".join(str(v).ljust(w) for v, w in zip(values, widths))

    print(fmt_row(labels))
    print(fmt_row(["-" * w for w in widths]))
    for display in display_rows:
        print(fmt_row([display[label] for label in labels]))


def print_prompt_summary(summary_rows: list[dict]) -> None:
    """Print a compact model x prompt summary."""
    if not summary_rows:
        print("No prompt-level rows to display.")
        return

    headers = [
        ("model", "model", False),
        ("prompt_id", "prompt_id", False),
        ("n", "n", False),
        ("mean_decodability_compliance", "decod_comp", True),
        ("mean_weighted_above_level_rate", "wt_above", False),
        ("full_spec_pass_rate", "full_spec", True),
        ("mean_target_satisfaction_rate", "tgt_sat", True),
        ("mean_word_count", "words", False),
    ]

    display_rows = []
    for row in summary_rows:
        display = {}
        for key, label, as_pct in headers:
            value = row[key]
            display[label] = pct(value) if as_pct else str(value)
        display_rows.append(display)

    labels = [label for _, label, _ in headers]
    widths = []
    for label in labels:
        width = len(label)
        for display in display_rows:
            width = max(width, len(display[label]))
        widths.append(width)

    def fmt_row(values: list[str]) -> str:
        return "  ".join(str(v).ljust(w) for v, w in zip(values, widths))

    print(fmt_row(labels))
    print(fmt_row(["-" * w for w in widths]))
    for display in display_rows:
        print(fmt_row([display[label] for label in labels]))


def print_paired_means(pairs: list[dict]) -> None:
    """Print mean paired deltas (tuned - base)."""
    if not pairs:
        print("No matched base/tuned pairs found.")
        return

    mean_compliance_delta = _mean([p["compliance_delta"] for p in pairs])
    mean_weighted_delta = _mean([p["weighted_rate_delta"] for p in pairs])
    mean_satisfaction_delta = _mean([p["satisfaction_delta"] for p in pairs])

    print(f"Matched pairs: {len(pairs)}")
    print(
        f"mean compliance_delta (tuned - base; + = tuned better): "
        f"{mean_compliance_delta:+.4f} ({pct(mean_compliance_delta)} pts)"
    )
    print(
        f"mean weighted_rate_delta (tuned - base; - = tuned better): "
        f"{mean_weighted_delta:+.4f}"
    )
    print(
        f"mean satisfaction_delta (tuned - base): "
        f"{mean_satisfaction_delta:+.4f} ({pct(mean_satisfaction_delta)} pts)"
    )


def load_outputs(path: Path) -> list[dict]:
    """Load all_outputs.csv with basic validation."""
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header row: {path}")
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"CSV missing required columns {missing}. "
                f"Found: {reader.fieldnames}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write rows to CSV with a stable column order."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Objective base-vs-tuned SLM evaluation using "
            "phonics_profiler_threshold4 detection logic. "
            "Primary metric: estimated decodability compliance / above-level rate."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUTS_DIR / "all_outputs.csv",
        help="Path to all_outputs.csv",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=OBJECTIVE_DIR / "objective_eval_results.csv",
        help="Row-level results CSV path",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=OBJECTIVE_DIR / "objective_eval_summary.csv",
        help="Model-level summary CSV path",
    )
    parser.add_argument(
        "--by-prompt",
        type=Path,
        default=OBJECTIVE_DIR / "objective_eval_by_prompt.csv",
        help="Model x prompt_id summary CSV path",
    )
    parser.add_argument(
        "--paired",
        type=Path,
        default=OBJECTIVE_DIR / "paired_eval_comparison.csv",
        help="Paired base-vs-tuned comparison CSV path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    OBJECTIVE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        raw_rows = load_outputs(args.input)
    except (OSError, ValueError) as exc:
        print(f"ERROR loading input: {exc}", file=sys.stderr)
        return 1

    results: list[dict] = []
    try:
        for i, row in enumerate(raw_rows, start=2):  # header is line 1
            results.append(evaluate_row(row, row_index=i))
    except (KeyError, ValueError) as exc:
        print(f"ERROR evaluating rows: {exc}", file=sys.stderr)
        return 1

    # Stable column order: identifiers, primary decodability metrics, secondary
    # target metrics, then pattern evidence.
    front = [
        "model",
        "prompt_id",
        "generation_id",
        "seed",
        "target_phonics",
        "word_count",
        "unique_word_count",
        "permitted_ceiling_stage",
        "permitted_ceiling_label",
        # Primary objective family
        "above_level_rate",
        "decodability_compliance",
        "distinct_above_level_rate",
        "distinct_decodability_compliance",
        "on_or_below_level_rate",
        "above_level_word_occurrences",
        "above_level_distinct_words",
        "mean_above_level_distance",
        "max_above_level_distance",
        "weighted_above_level_rate",
        "above_level_words",
        # Secondary target-satisfaction family
        "targets_requested",
        "targets_passed",
        "target_satisfaction_rate",
        "full_spec_pass",
        "targets_passed_list",
        "targets_failed_list",
        # Other checks
        "phonics_leakage",
        "phonics_leakage_terms",
        "has_title",
        "detected_title",
    ]
    pattern_cols = []
    for pattern in PROFILE_PATTERNS:
        pattern_cols.extend([f"{pattern}_distinct_count", f"{pattern}_words"])
    extra = [
        c for c in results[0].keys() if c not in front and c not in pattern_cols
    ]
    fieldnames = front + pattern_cols + sorted(extra)

    summary_fields = [
        "model",
        "n",
        "mean_decodability_compliance",
        "mean_distinct_decodability_compliance",
        "mean_weighted_above_level_rate",
        "mean_above_level_distance",
        "full_spec_pass_rate",
        "mean_target_satisfaction_rate",
        "phonics_leakage_rate",
        "title_rate",
        "mean_word_count",
        # retained for continuity / debugging
        "mean_above_level_rate",
    ]
    by_prompt_fields = [
        "model",
        "prompt_id",
        "target_phonics",
        "n",
        "mean_decodability_compliance",
        "mean_distinct_decodability_compliance",
        "mean_weighted_above_level_rate",
        "mean_above_level_distance",
        "full_spec_pass_rate",
        "mean_target_satisfaction_rate",
        "phonics_leakage_rate",
        "title_rate",
        "mean_word_count",
        "mean_above_level_rate",
    ]
    paired_fields = [
        "prompt_id",
        "generation_id",
        "seed",
        "target_phonics",
        "base_decodability_compliance",
        "tuned_decodability_compliance",
        "compliance_delta",
        "base_weighted_above_level_rate",
        "tuned_weighted_above_level_rate",
        "weighted_rate_delta",
        "base_target_satisfaction_rate",
        "tuned_target_satisfaction_rate",
        "satisfaction_delta",
        "base_word_count",
        "tuned_word_count",
    ]

    try:
        write_csv(args.results, results, fieldnames)

        summary_rows = summarize_by_model(results)
        write_csv(args.summary, summary_rows, summary_fields)

        by_prompt_rows = summarize_by_model_prompt(results)
        write_csv(args.by_prompt, by_prompt_rows, by_prompt_fields)

        pairs, unmatched = build_paired_comparison(results)
        write_csv(args.paired, pairs, paired_fields)
    except OSError as exc:
        print(f"ERROR writing output CSVs: {exc}", file=sys.stderr)
        return 1

    print(f"Evaluated {len(results)} stories from {args.input}")
    print(f"Saved row-level results: {args.results}")
    print(f"Saved model summary:     {args.summary}")
    print(f"Saved prompt summary:    {args.by_prompt}")
    print(f"Saved paired comparison: {args.paired}")
    if unmatched:
        print(
            f"WARNING: {unmatched} match-key group(s) lacked both base and tuned."
        )
    print()
    print(
        "Model-level summary "
        "(estimated rule-based decodability measures; not perfect linguistic decodability)"
    )
    print("-" * 72)
    print_model_summary(summary_rows)
    print()
    print("Prompt-level summary (model x prompt_id)")
    print("-" * 72)
    print_prompt_summary(by_prompt_rows)
    print()
    print("Paired deltas (matched on prompt_id, generation_id, seed)")
    print("-" * 72)
    print_paired_means(pairs)
    print()
    print(
        "Note: all above-level / compliance metrics are estimated rule-based "
        "spelling-pattern measures from phonics_profiler_threshold4.py, not a "
        "perfect measure of true linguistic decodability. Target satisfaction "
        "and full-spec pass are secondary because advanced stories can satisfy "
        "requested patterns accidentally."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
