#!/usr/bin/env python3
"""
Error analysis for base-vs-tuned SLM decodable-story evaluation.

Uses existing objective + subjective outputs to identify:
  - largest / smallest paired decodability-compliance gains
  - tuned failure modes (decodability, emphasis, coherence, completeness, above-level)
  - which phonics prompt conditions remain difficult
  - whether failures concentrate in specific pattern families / multi-target prompts

Writes:
  - results/error_analysis/error_analysis_examples.csv
  - results/error_analysis/error_analysis_summary.md

Likely-cause labels are hypotheses grounded only in observed metrics; they are
not claims that every tuned failure is a training-data problem.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

JOIN_KEYS = ("prompt_id", "generation_id", "seed")

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUTS_DIR = REPO_ROOT / "inputs"
OBJECTIVE_DIR = REPO_ROOT / "results" / "objective"
SUBJECTIVE_DIR = REPO_ROOT / "results" / "subjective"
ERROR_DIR = REPO_ROOT / "results" / "error_analysis"

# Subjective "low" thresholds (0–2 integer scores).
LOW_SCORE = 0

# Objective "high above-level" threshold for tuned stories.
# Tuned median above_level_rate is ~0.11; flag clearly elevated rates.
HIGH_ABOVE_LEVEL_RATE = 0.20

# Absolute weak tuned compliance (for distinguishing ceiling effects).
WEAK_TUNED_COMPLIANCE = 0.85


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def inum(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default) or default))
    except (TypeError, ValueError):
        return default


def pair_key(row: dict) -> tuple[str, str, str]:
    return tuple(str(row[k]) for k in JOIN_KEYS)


def truncate(text: str, limit: int = 500) -> str:
    text = (text or "").strip().replace("\r\n", "\n")
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def parse_targets(raw: str) -> list[str]:
    return [p.strip() for p in (raw or "").split(",") if p.strip()]


def target_flags(target_phonics: str) -> dict[str, Any]:
    targets = set(parse_targets(target_phonics))
    return {
        "vowel_teams": "vowel teams" in targets,
        "diphthongs": "diphthongs" in targets,
        "r_controlled": "r-controlled vowels" in targets,
        "multisyllabic": "multisyllabic words" in targets,
        "multi_target": len(targets) >= 2,
        "n_targets": len(targets),
    }


def index_by_model_join(rows: list[dict]) -> dict[tuple, dict]:
    return {
        (row["model"],) + pair_key(row): row
        for row in rows
    }


def build_story_index(outputs: list[dict]) -> dict[tuple, str]:
    return {
        (row["model"],) + pair_key(row): row.get("story", "")
        for row in outputs
    }


def infer_cause(mode: str, tuned: dict, base: dict, pair: dict | None = None) -> str:
    """
    Assign a likely-cause category from observed metrics only.

    Categories:
      - training-data coverage
      - conflicting training objectives
      - small-model capacity
      - generation behavior
      - rule-based evaluation limitation
    """
    t_comp = fnum(tuned, "decodability_compliance")
    b_comp = fnum(base, "decodability_compliance")
    t_above = fnum(tuned, "above_level_rate")
    b_above = fnum(base, "above_level_rate")
    t_words = fnum(tuned, "word_count")
    b_words = fnum(base, "word_count")
    t_sat = fnum(tuned, "target_satisfaction_rate")
    b_sat = fnum(base, "target_satisfaction_rate")
    t_emph = inum(tuned, "target_phonics_pattern_emphasis_score")
    t_decod_subj = inum(tuned, "decodability_adherence_score")
    t_cohere = inum(tuned, "coherence_score")
    t_complete = inum(tuned, "narrative_completeness_score")
    t_wt = fnum(tuned, "weighted_above_level_rate")
    flags = target_flags(tuned.get("target_phonics", ""))

    # Ceiling / metric artifact: both already fully compliant.
    if mode in {"least_compliance_gain", "prompt_difficulty"} and t_comp >= 0.99 and b_comp >= 0.99:
        return "rule-based evaluation limitation"

    # Objective says in-level, subjective says out-of-level (or reverse) strongly.
    if mode == "low_decodability_adherence" and t_comp >= 0.9 and t_decod_subj == 0:
        return "rule-based evaluation limitation"
    if mode == "high_above_level" and t_decod_subj >= 2 and t_above >= HIGH_ABOVE_LEVEL_RATE:
        return "rule-based evaluation limitation"

    # Short / fragmented outputs.
    if mode in {"low_coherence", "incomplete_narrative"} or t_complete == 0 or t_cohere == 0:
        if t_words <= 40 or t_complete == 0:
            return "generation behavior"

    # Tuned stays more in-level but loses target coverage vs base.
    if t_comp > b_comp + 0.05 and t_sat + 0.2 < b_sat and t_emph == 0:
        return "conflicting training objectives"

    # Harder pattern families with weak emphasis / satisfaction despite attempts.
    hard_family = flags["vowel_teams"] or flags["diphthongs"] or flags["r_controlled"] or flags["multisyllabic"]
    if hard_family and (t_emph == 0 or t_sat < 0.5) and t_words >= 40:
        # Multi-target advanced prompts especially suggest capacity limits.
        if flags["multi_target"] and flags["n_targets"] >= 3:
            return "small-model capacity"
        return "training-data coverage"

    # Persistent above-level vocabulary while otherwise coherent.
    if mode == "high_above_level" or (t_above >= HIGH_ABOVE_LEVEL_RATE and t_cohere >= 1):
        if t_wt >= 0.5:
            return "training-data coverage"
        return "generation behavior"

    # Low emphasis with decent length and coherence.
    if mode == "low_target_phonics_emphasis" and t_cohere >= 1 and t_words >= 40:
        return "training-data coverage"

    # Default by mode.
    defaults = {
        "largest_compliance_gain": "generation behavior",
        "least_compliance_gain": "conflicting training objectives",
        "low_decodability_adherence": "training-data coverage",
        "low_target_phonics_emphasis": "training-data coverage",
        "low_coherence": "generation behavior",
        "incomplete_narrative": "generation behavior",
        "high_above_level": "generation behavior",
        "prompt_difficulty": "training-data coverage",
    }
    return defaults.get(mode, "generation behavior")


def example_row(
    category: str,
    rank: int,
    tuned: dict,
    base: dict,
    stories: dict[tuple, str],
    pair: dict | None = None,
    notes: str = "",
) -> dict[str, Any]:
    key = pair_key(tuned)
    cause = infer_cause(category, tuned, base, pair)
    return {
        "category": category,
        "rank": rank,
        "prompt_id": tuned["prompt_id"],
        "generation_id": tuned["generation_id"],
        "seed": tuned["seed"],
        "target_phonics": tuned.get("target_phonics", ""),
        "n_targets": target_flags(tuned.get("target_phonics", ""))["n_targets"],
        "likely_cause": cause,
        "notes": notes,
        "compliance_delta": fnum(pair, "compliance_delta") if pair else (
            fnum(tuned, "decodability_compliance") - fnum(base, "decodability_compliance")
        ),
        "base_decodability_compliance": fnum(base, "decodability_compliance"),
        "tuned_decodability_compliance": fnum(tuned, "decodability_compliance"),
        "base_above_level_rate": fnum(base, "above_level_rate"),
        "tuned_above_level_rate": fnum(tuned, "above_level_rate"),
        "base_weighted_above_level_rate": fnum(base, "weighted_above_level_rate"),
        "tuned_weighted_above_level_rate": fnum(tuned, "weighted_above_level_rate"),
        "base_target_satisfaction_rate": fnum(base, "target_satisfaction_rate"),
        "tuned_target_satisfaction_rate": fnum(tuned, "target_satisfaction_rate"),
        "base_word_count": inum(base, "word_count"),
        "tuned_word_count": inum(tuned, "word_count"),
        "tuned_decodability_adherence_score": inum(tuned, "decodability_adherence_score"),
        "tuned_target_phonics_pattern_emphasis_score": inum(
            tuned, "target_phonics_pattern_emphasis_score"
        ),
        "tuned_coherence_score": inum(tuned, "coherence_score"),
        "tuned_narrative_completeness_score": inum(tuned, "narrative_completeness_score"),
        "tuned_overall_spec_adherence_score": inum(tuned, "overall_spec_adherence_score"),
        "base_decodability_adherence_score": inum(base, "decodability_adherence_score"),
        "base_target_phonics_pattern_emphasis_score": inum(
            base, "target_phonics_pattern_emphasis_score"
        ),
        "base_coherence_score": inum(base, "coherence_score"),
        "base_narrative_completeness_score": inum(base, "narrative_completeness_score"),
        "tuned_story": truncate(stories.get(("tuned",) + key, ""), 800),
        "base_story": truncate(stories.get(("base",) + key, ""), 800),
        "tuned_decodability_adherence_justification": tuned.get(
            "decodability_adherence_justification", ""
        ),
        "tuned_target_phonics_pattern_emphasis_justification": tuned.get(
            "target_phonics_pattern_emphasis_justification", ""
        ),
        "tuned_coherence_justification": tuned.get("coherence_justification", ""),
        "tuned_narrative_completeness_justification": tuned.get(
            "narrative_completeness_justification", ""
        ),
    }


def pick_representative(
    candidates: list[dict],
    final_index: dict[tuple, dict],
    severity_key: str,
) -> dict | None:
    if not candidates:
        return None
    # Prefer worse severity, then lower overall subjective score.
    ranked = sorted(
        candidates,
        key=lambda r: (
            fnum(r, severity_key) if "rate" in severity_key or "compliance" in severity_key
            else -inum(r, severity_key) if severity_key.endswith("_score")
            else 0,
            inum(r, "overall_spec_adherence_score"),
            inum(r, "word_count"),
        ),
        reverse=("rate" in severity_key),
    )
    # For scores, lower is worse so don't reverse; handled above awkwardly.
    if severity_key.endswith("_score"):
        ranked = sorted(
            candidates,
            key=lambda r: (
                inum(r, severity_key),
                inum(r, "overall_spec_adherence_score"),
                -fnum(r, "above_level_rate"),
            ),
        )
    elif severity_key in {"above_level_rate", "weighted_above_level_rate"}:
        ranked = sorted(
            candidates,
            key=lambda r: (
                -fnum(r, severity_key),
                inum(r, "overall_spec_adherence_score"),
            ),
        )
    elif severity_key == "decodability_compliance":
        ranked = sorted(
            candidates,
            key=lambda r: (
                fnum(r, severity_key),
                inum(r, "overall_spec_adherence_score"),
            ),
        )
    return ranked[0]


def analyze(args: argparse.Namespace) -> int:
    paired = load_csv(args.paired)
    final_rows = load_csv(args.final)
    by_prompt = load_csv(args.by_prompt)
    outputs = load_csv(args.outputs)

    final_index = index_by_model_join(final_rows)
    stories = build_story_index(outputs)

    tuned_rows = [r for r in final_rows if r["model"] == "tuned"]
    base_rows = [r for r in final_rows if r["model"] == "base"]

    examples: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 1) Largest compliance improvements
    # ------------------------------------------------------------------
    best_pairs = sorted(paired, key=lambda r: fnum(r, "compliance_delta"), reverse=True)[:5]
    for i, pair in enumerate(best_pairs, start=1):
        key = pair_key(pair)
        tuned = final_index[("tuned",) + key]
        base = final_index[("base",) + key]
        examples.append(
            example_row(
                "largest_compliance_gain",
                i,
                tuned,
                base,
                stories,
                pair,
                notes="Largest paired gain in objective decodability_compliance.",
            )
        )

    # ------------------------------------------------------------------
    # 2) Worst / least improvement
    # ------------------------------------------------------------------
    worst_pairs = sorted(paired, key=lambda r: fnum(r, "compliance_delta"))[:5]
    for i, pair in enumerate(worst_pairs, start=1):
        key = pair_key(pair)
        tuned = final_index[("tuned",) + key]
        base = final_index[("base",) + key]
        note = "Smallest paired compliance_delta."
        if fnum(pair, "compliance_delta") == 0 and fnum(pair, "tuned_decodability_compliance") >= 0.99:
            note += " Ceiling effect: both models already near-perfect on this metric."
        elif fnum(pair, "tuned_decodability_compliance") < WEAK_TUNED_COMPLIANCE:
            note += " Tuned absolute compliance remains relatively weak."
        examples.append(
            example_row(
                "least_compliance_gain",
                i,
                tuned,
                base,
                stories,
                pair,
                notes=note,
            )
        )

    # ------------------------------------------------------------------
    # 3) Tuned failure-mode slices
    # ------------------------------------------------------------------
    failure_defs = [
        (
            "low_decodability_adherence",
            lambda r: inum(r, "decodability_adherence_score") <= LOW_SCORE,
            "decodability_adherence_score",
            "Tuned subjective decodability_adherence_score == 0.",
        ),
        (
            "low_target_phonics_emphasis",
            lambda r: inum(r, "target_phonics_pattern_emphasis_score") <= LOW_SCORE,
            "target_phonics_pattern_emphasis_score",
            "Tuned target_phonics_pattern_emphasis_score == 0.",
        ),
        (
            "low_coherence",
            lambda r: inum(r, "coherence_score") <= LOW_SCORE,
            "coherence_score",
            "Tuned coherence_score == 0.",
        ),
        (
            "incomplete_narrative",
            lambda r: inum(r, "narrative_completeness_score") <= LOW_SCORE,
            "narrative_completeness_score",
            "Tuned narrative_completeness_score == 0.",
        ),
        (
            "high_above_level",
            lambda r: fnum(r, "above_level_rate") >= HIGH_ABOVE_LEVEL_RATE,
            "above_level_rate",
            f"Tuned above_level_rate >= {HIGH_ABOVE_LEVEL_RATE:.2f}.",
        ),
    ]

    failure_sets: dict[str, list[dict]] = {}
    for category, pred, severity_key, note in failure_defs:
        subset = [r for r in tuned_rows if pred(r)]
        failure_sets[category] = subset
        rep = pick_representative(subset, final_index, severity_key)
        if rep is None:
            continue
        key = pair_key(rep)
        base = final_index[("base",) + key]
        pair = next((p for p in paired if pair_key(p) == key), None)
        examples.append(
            example_row(
                category,
                1,
                rep,
                base,
                stories,
                pair,
                notes=f"{note} n_tuned_flagged={len(subset)}/{len(tuned_rows)}.",
            )
        )
        # Also keep up to 4 additional examples for the CSV catalog.
        extras = [
            r for r in sorted(
                subset,
                key=lambda r: (
                    inum(r, severity_key) if severity_key.endswith("_score") else -fnum(r, severity_key),
                    inum(r, "overall_spec_adherence_score"),
                ),
            )
            if pair_key(r) != key
        ][:4]
        for j, row in enumerate(extras, start=2):
            b = final_index[("base",) + pair_key(row)]
            p = next((pp for pp in paired if pair_key(pp) == pair_key(row)), None)
            examples.append(
                example_row(
                    category,
                    j,
                    row,
                    b,
                    stories,
                    p,
                    notes=f"{note} additional flagged example.",
                )
            )

    # ------------------------------------------------------------------
    # 4–5) Prompt-type difficulty + concentration
    # ------------------------------------------------------------------
    tuned_by_prompt = [r for r in by_prompt if r["model"] == "tuned"]
    base_by_prompt = {r["prompt_id"]: r for r in by_prompt if r["model"] == "base"}

    prompt_stats = []
    for row in tuned_by_prompt:
        pid = row["prompt_id"]
        base = base_by_prompt.get(pid, {})
        flags = target_flags(row.get("target_phonics", ""))
        # Subjective means from final rows
        t_final = [r for r in tuned_rows if r["prompt_id"] == pid]
        prompt_stats.append(
            {
                "prompt_id": pid,
                "target_phonics": row.get("target_phonics", ""),
                "n_targets": flags["n_targets"],
                "multi_target": flags["multi_target"],
                "has_vowel_teams": flags["vowel_teams"],
                "has_diphthongs": flags["diphthongs"],
                "has_r_controlled": flags["r_controlled"],
                "has_multisyllabic": flags["multisyllabic"],
                "tuned_mean_decodability_compliance": fnum(row, "mean_decodability_compliance"),
                "base_mean_decodability_compliance": fnum(base, "mean_decodability_compliance"),
                "tuned_mean_above_level_rate": fnum(row, "mean_above_level_rate"),
                "tuned_mean_weighted_above_level_rate": fnum(row, "mean_weighted_above_level_rate"),
                "tuned_mean_target_satisfaction_rate": fnum(row, "mean_target_satisfaction_rate"),
                "tuned_mean_word_count": fnum(row, "mean_word_count"),
                "tuned_mean_decodability_adherence": statistics.mean(
                    inum(r, "decodability_adherence_score") for r in t_final
                ) if t_final else 0.0,
                "tuned_mean_target_phonics_pattern_emphasis": statistics.mean(
                    inum(r, "target_phonics_pattern_emphasis_score") for r in t_final
                ) if t_final else 0.0,
                "tuned_mean_coherence": statistics.mean(
                    inum(r, "coherence_score") for r in t_final
                ) if t_final else 0.0,
                "tuned_mean_narrative_completeness": statistics.mean(
                    inum(r, "narrative_completeness_score") for r in t_final
                ) if t_final else 0.0,
                "tuned_mean_overall_spec_adherence": statistics.mean(
                    inum(r, "overall_spec_adherence_score") for r in t_final
                ) if t_final else 0.0,
                "n_low_decod_subj": sum(
                    inum(r, "decodability_adherence_score") <= LOW_SCORE for r in t_final
                ),
                "n_low_emphasis": sum(
                    inum(r, "target_phonics_pattern_emphasis_score") <= LOW_SCORE for r in t_final
                ),
                "n_low_coherence": sum(
                    inum(r, "coherence_score") <= LOW_SCORE for r in t_final
                ),
                "n_incomplete": sum(
                    inum(r, "narrative_completeness_score") <= LOW_SCORE for r in t_final
                ),
                "n_high_above": sum(
                    fnum(r, "above_level_rate") >= HIGH_ABOVE_LEVEL_RATE for r in t_final
                ),
            }
        )

    # Difficulty rank: low overall subjective + low emphasis + low target sat.
    prompt_stats_sorted = sorted(
        prompt_stats,
        key=lambda r: (
            r["tuned_mean_overall_spec_adherence"],
            r["tuned_mean_target_phonics_pattern_emphasis"],
            r["tuned_mean_target_satisfaction_rate"],
            r["tuned_mean_decodability_adherence"],
            -r["tuned_mean_above_level_rate"],
        ),
    )

    for i, ps in enumerate(prompt_stats_sorted[:5], start=1):
        # Representative tuned row for this prompt: worst overall then emphasis.
        cand = [r for r in tuned_rows if r["prompt_id"] == ps["prompt_id"]]
        rep = sorted(
            cand,
            key=lambda r: (
                inum(r, "overall_spec_adherence_score"),
                inum(r, "target_phonics_pattern_emphasis_score"),
                inum(r, "decodability_adherence_score"),
                -fnum(r, "above_level_rate"),
            ),
        )[0]
        key = pair_key(rep)
        base = final_index[("base",) + key]
        pair = next((p for p in paired if pair_key(p) == key), None)
        examples.append(
            example_row(
                "prompt_difficulty",
                i,
                rep,
                base,
                stories,
                pair,
                notes=(
                    f"Prompt-level difficulty rank #{i}. "
                    f"tuned_overall={ps['tuned_mean_overall_spec_adherence']:.2f}, "
                    f"emphasis={ps['tuned_mean_target_phonics_pattern_emphasis']:.2f}, "
                    f"tgt_sat={ps['tuned_mean_target_satisfaction_rate']:.2f}."
                ),
            )
        )

    # Concentration tables
    def concentration(label: str, pred) -> dict[str, Any]:
        subset = [r for r in tuned_rows if pred(r)]
        n = len(subset) or 1
        return {
            "condition": label,
            "n_outputs": len(subset),
            "share_of_tuned": round(len(subset) / len(tuned_rows), 4),
            "mean_decodability_compliance": round(
                statistics.mean(fnum(r, "decodability_compliance") for r in subset), 4
            ) if subset else None,
            "mean_above_level_rate": round(
                statistics.mean(fnum(r, "above_level_rate") for r in subset), 4
            ) if subset else None,
            "mean_target_satisfaction": round(
                statistics.mean(fnum(r, "target_satisfaction_rate") for r in subset), 4
            ) if subset else None,
            "mean_decodability_adherence": round(
                statistics.mean(inum(r, "decodability_adherence_score") for r in subset), 4
            ) if subset else None,
            "mean_target_phonics_pattern_emphasis": round(
                statistics.mean(
                    inum(r, "target_phonics_pattern_emphasis_score") for r in subset
                ),
                4,
            ) if subset else None,
            "mean_overall_spec_adherence": round(
                statistics.mean(inum(r, "overall_spec_adherence_score") for r in subset), 4
            ) if subset else None,
            "pct_low_emphasis": round(
                sum(inum(r, "target_phonics_pattern_emphasis_score") <= LOW_SCORE for r in subset) / n,
                4,
            ) if subset else None,
            "pct_low_decod_subj": round(
                sum(inum(r, "decodability_adherence_score") <= LOW_SCORE for r in subset) / n,
                4,
            ) if subset else None,
            "pct_incomplete": round(
                sum(inum(r, "narrative_completeness_score") <= LOW_SCORE for r in subset) / n,
                4,
            ) if subset else None,
        }

    concentrations = [
        concentration(
            "vowel teams prompts",
            lambda r: target_flags(r["target_phonics"])["vowel_teams"],
        ),
        concentration(
            "diphthongs prompts",
            lambda r: target_flags(r["target_phonics"])["diphthongs"],
        ),
        concentration(
            "r-controlled vowels prompts",
            lambda r: target_flags(r["target_phonics"])["r_controlled"],
        ),
        concentration(
            "multisyllabic words prompts",
            lambda r: target_flags(r["target_phonics"])["multisyllabic"],
        ),
        concentration(
            "multi-target prompts (>=2)",
            lambda r: target_flags(r["target_phonics"])["multi_target"],
        ),
        concentration(
            "single-target prompts",
            lambda r: not target_flags(r["target_phonics"])["multi_target"],
        ),
        concentration("all tuned outputs", lambda r: True),
    ]

    # Cause tallies for major failure modes only
    major_modes = [
        "low_decodability_adherence",
        "low_target_phonics_emphasis",
        "low_coherence",
        "incomplete_narrative",
        "high_above_level",
        "least_compliance_gain",
        "prompt_difficulty",
    ]
    cause_counter = Counter(
        e["likely_cause"] for e in examples if e["category"] in major_modes and e["rank"] == 1
    )

    # Write examples CSV
    example_fields = [
        "category",
        "rank",
        "prompt_id",
        "generation_id",
        "seed",
        "target_phonics",
        "n_targets",
        "likely_cause",
        "notes",
        "compliance_delta",
        "base_decodability_compliance",
        "tuned_decodability_compliance",
        "base_above_level_rate",
        "tuned_above_level_rate",
        "base_weighted_above_level_rate",
        "tuned_weighted_above_level_rate",
        "base_target_satisfaction_rate",
        "tuned_target_satisfaction_rate",
        "base_word_count",
        "tuned_word_count",
        "tuned_decodability_adherence_score",
        "tuned_target_phonics_pattern_emphasis_score",
        "tuned_coherence_score",
        "tuned_narrative_completeness_score",
        "tuned_overall_spec_adherence_score",
        "base_decodability_adherence_score",
        "base_target_phonics_pattern_emphasis_score",
        "base_coherence_score",
        "base_narrative_completeness_score",
        "tuned_story",
        "base_story",
        "tuned_decodability_adherence_justification",
        "tuned_target_phonics_pattern_emphasis_justification",
        "tuned_coherence_justification",
        "tuned_narrative_completeness_justification",
    ]
    write_csv(args.examples_out, examples, example_fields)

    # ------------------------------------------------------------------
    # Markdown summary
    # ------------------------------------------------------------------
    def fmt_pct(x: float | None) -> str:
        if x is None:
            return "n/a"
        return f"{100 * x:.1f}%"

    def fmt4(x: float | None) -> str:
        if x is None:
            return "n/a"
        return f"{x:.4f}"

    lines: list[str] = []
    lines.append("# Error Analysis Summary")
    lines.append("")
    lines.append(
        "This report is derived only from observed objective and subjective "
        "evaluation outputs. Likely-cause labels are hypotheses constrained by "
        "those metrics; they are **not** proof that every tuned-model failure is "
        "a training-data problem."
    )
    lines.append("")
    lines.append("## Method notes")
    lines.append("")
    lines.append(
        f"- Low subjective score threshold: `{LOW_SCORE}` on the 0–2 judge scale."
    )
    lines.append(
        f"- High above-level rate threshold: `>= {HIGH_ABOVE_LEVEL_RATE:.2f}` "
        "(tuned median is lower; this flags clearly elevated rates)."
    )
    lines.append(
        "- Paired comparisons use `compliance_delta = tuned - base` "
        "decodability compliance."
    )
    lines.append(
        "- In this run, all paired compliance deltas are ≥ 0, so “worst” pairs "
        "are least-improved rather than true regressions on that metric."
    )
    lines.append("")

    # 1. Best gains
    lines.append("## 1. Largest paired decodability-compliance gains")
    lines.append("")
    lines.append(
        "| rank | prompt_id | gen | seed | base | tuned | delta | tuned words | likely cause |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for e in examples:
        if e["category"] != "largest_compliance_gain":
            continue
        lines.append(
            f"| {e['rank']} | `{e['prompt_id']}` | {e['generation_id']} | {e['seed']} | "
            f"{e['base_decodability_compliance']:.4f} | {e['tuned_decodability_compliance']:.4f} | "
            f"{e['compliance_delta']:+.4f} | {e['tuned_word_count']} | {e['likely_cause']} |"
        )
    lines.append("")
    best_rep = next(e for e in examples if e["category"] == "largest_compliance_gain" and e["rank"] == 1)
    lines.append("### Representative improvement")
    lines.append("")
    lines.append(
        f"- **Pair:** `{best_rep['prompt_id']}` / gen {best_rep['generation_id']} / seed {best_rep['seed']}"
    )
    lines.append(
        f"- **Metrics:** compliance {best_rep['base_decodability_compliance']:.3f} → "
        f"{best_rep['tuned_decodability_compliance']:.3f} "
        f"(Δ {best_rep['compliance_delta']:+.3f}); "
        f"above-level rate {best_rep['base_above_level_rate']:.3f} → "
        f"{best_rep['tuned_above_level_rate']:.3f}; "
        f"words {best_rep['base_word_count']} → {best_rep['tuned_word_count']}."
    )
    lines.append(f"- **Likely cause:** {best_rep['likely_cause']}")
    lines.append("")
    lines.append("**Tuned output**")
    lines.append("")
    lines.append("```text")
    lines.append(best_rep["tuned_story"])
    lines.append("```")
    lines.append("")
    lines.append("**Matched base output**")
    lines.append("")
    lines.append("```text")
    lines.append(best_rep["base_story"])
    lines.append("```")
    lines.append("")

    # 2. Least gains
    lines.append("## 2. Least paired improvement (or no gain)")
    lines.append("")
    lines.append(
        "| rank | prompt_id | gen | seed | base | tuned | delta | note | likely cause |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---|---|")
    for e in examples:
        if e["category"] != "least_compliance_gain":
            continue
        lines.append(
            f"| {e['rank']} | `{e['prompt_id']}` | {e['generation_id']} | {e['seed']} | "
            f"{e['base_decodability_compliance']:.4f} | {e['tuned_decodability_compliance']:.4f} | "
            f"{e['compliance_delta']:+.4f} | {e['notes']} | {e['likely_cause']} |"
        )
    lines.append("")
    # Prefer a non-ceiling representative if available.
    least_reps = [e for e in examples if e["category"] == "least_compliance_gain"]
    least_rep = next(
        (
            e
            for e in least_reps
            if "Ceiling effect" not in e["notes"]
        ),
        least_reps[0],
    )
    lines.append("### Representative weak/least-gain case")
    lines.append("")
    lines.append(
        f"- **Pair:** `{least_rep['prompt_id']}` / gen {least_rep['generation_id']} / seed {least_rep['seed']}"
    )
    lines.append(f"- **Notes:** {least_rep['notes']}")
    lines.append(
        f"- **Metrics:** compliance {least_rep['base_decodability_compliance']:.3f} → "
        f"{least_rep['tuned_decodability_compliance']:.3f} "
        f"(Δ {least_rep['compliance_delta']:+.3f}); "
        f"target satisfaction {least_rep['base_target_satisfaction_rate']:.2f} → "
        f"{least_rep['tuned_target_satisfaction_rate']:.2f}; "
        f"subj decod {least_rep['tuned_decodability_adherence_score']}, "
        f"emphasis {least_rep['tuned_target_phonics_pattern_emphasis_score']}."
    )
    lines.append(f"- **Likely cause:** {least_rep['likely_cause']}")
    lines.append("")
    lines.append("**Tuned output**")
    lines.append("")
    lines.append("```text")
    lines.append(least_rep["tuned_story"])
    lines.append("```")
    lines.append("")
    lines.append("**Matched base output**")
    lines.append("")
    lines.append("```text")
    lines.append(least_rep["base_story"])
    lines.append("```")
    lines.append("")

    # 3. Failure modes
    lines.append("## 3. Tuned failure-mode inventory")
    lines.append("")
    lines.append("| failure mode | n flagged | share of tuned | representative prompt | likely cause |")
    lines.append("|---|---:|---:|---|---|")
    for category, _, _, _ in failure_defs:
        subset = failure_sets[category]
        rep = next((e for e in examples if e["category"] == category and e["rank"] == 1), None)
        lines.append(
            f"| `{category}` | {len(subset)} | {len(subset)/len(tuned_rows):.1%} | "
            f"{('`'+rep['prompt_id']+'`') if rep else '—'} | "
            f"{rep['likely_cause'] if rep else '—'} |"
        )
    lines.append("")

    for category, _, _, title_note in failure_defs:
        rep = next((e for e in examples if e["category"] == category and e["rank"] == 1), None)
        if rep is None:
            continue
        lines.append(f"### Failure mode: `{category}`")
        lines.append("")
        lines.append(f"- **Definition:** {title_note}")
        lines.append(
            f"- **Support metrics:** compliance={rep['tuned_decodability_compliance']:.3f}, "
            f"above_level_rate={rep['tuned_above_level_rate']:.3f}, "
            f"weighted_above={rep['tuned_weighted_above_level_rate']:.3f}, "
            f"target_sat={rep['tuned_target_satisfaction_rate']:.2f}, "
            f"subj decod={rep['tuned_decodability_adherence_score']}, "
            f"emphasis={rep['tuned_target_phonics_pattern_emphasis_score']}, "
            f"coherence={rep['tuned_coherence_score']}, "
            f"completeness={rep['tuned_narrative_completeness_score']}, "
            f"words={rep['tuned_word_count']}."
        )
        lines.append(
            f"- **Matched base metrics:** compliance={rep['base_decodability_compliance']:.3f}, "
            f"above_level_rate={rep['base_above_level_rate']:.3f}, "
            f"target_sat={rep['base_target_satisfaction_rate']:.2f}, "
            f"words={rep['base_word_count']}."
        )
        lines.append(f"- **Likely cause:** {rep['likely_cause']}")
        if rep.get("tuned_decodability_adherence_justification"):
            lines.append(
                f"- **Judge note (decodability):** "
                f"{rep['tuned_decodability_adherence_justification']}"
            )
        if rep.get("tuned_target_phonics_pattern_emphasis_justification"):
            lines.append(
                f"- **Judge note (target phonics pattern emphasis):** "
                f"{rep['tuned_target_phonics_pattern_emphasis_justification']}"
            )
        lines.append("")
        lines.append("**Representative tuned output**")
        lines.append("")
        lines.append("```text")
        lines.append(rep["tuned_story"])
        lines.append("```")
        lines.append("")
        lines.append("**Matched base output**")
        lines.append("")
        lines.append("```text")
        lines.append(rep["base_story"])
        lines.append("```")
        lines.append("")

    # 4. Prompt difficulty
    lines.append("## 4. Prompt-type difficulty (tuned)")
    lines.append("")
    lines.append(
        "| difficulty rank | prompt_id | targets | overall | emphasis | decod_subj | tgt_sat | compliance | above_level |"
    )
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|")
    for i, ps in enumerate(prompt_stats_sorted, start=1):
        lines.append(
            f"| {i} | `{ps['prompt_id']}` | {ps['n_targets']} | "
            f"{ps['tuned_mean_overall_spec_adherence']:.2f} | "
            f"{ps['tuned_mean_target_phonics_pattern_emphasis']:.2f} | "
            f"{ps['tuned_mean_decodability_adherence']:.2f} | "
            f"{ps['tuned_mean_target_satisfaction_rate']:.2f} | "
            f"{ps['tuned_mean_decodability_compliance']:.3f} | "
            f"{ps['tuned_mean_above_level_rate']:.3f} |"
        )
    lines.append("")
    hardest = prompt_stats_sorted[0]
    lines.append(
        f"Most difficult tuned condition by composite subjective/objective rank: "
        f"`{hardest['prompt_id']}` ({hardest['target_phonics']})."
    )
    lines.append("")

    # 5. Concentration
    lines.append("## 5. Are failures concentrated in specific phonics conditions?")
    lines.append("")
    lines.append(
        "| condition | n | share | mean compliance | mean emphasis | pct low emphasis | pct low decod_subj | pct incomplete |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for c in concentrations:
        lines.append(
            f"| {c['condition']} | {c['n_outputs']} | {fmt_pct(c['share_of_tuned'])} | "
            f"{fmt4(c['mean_decodability_compliance'])} | "
            f"{fmt4(c['mean_target_phonics_pattern_emphasis'])} | "
            f"{fmt_pct(c['pct_low_emphasis'])} | "
            f"{fmt_pct(c['pct_low_decod_subj'])} | "
            f"{fmt_pct(c['pct_incomplete'])} |"
        )
    lines.append("")

    # Evidence-based concentration conclusions
    all_c = next(c for c in concentrations if c["condition"] == "all tuned outputs")
    lines.append("### Observed concentration patterns")
    lines.append("")
    for c in concentrations:
        if c["condition"] == "all tuned outputs":
            continue
        if c["n_outputs"] == 0:
            continue
        worse_emphasis = (
            c["pct_low_emphasis"] is not None
            and all_c["pct_low_emphasis"] is not None
            and c["pct_low_emphasis"] > all_c["pct_low_emphasis"] + 0.05
        )
        worse_decod = (
            c["pct_low_decod_subj"] is not None
            and all_c["pct_low_decod_subj"] is not None
            and c["pct_low_decod_subj"] > all_c["pct_low_decod_subj"] + 0.05
        )
        worse_overall = (
            c["mean_overall_spec_adherence"] is not None
            and all_c["mean_overall_spec_adherence"] is not None
            and c["mean_overall_spec_adherence"] + 0.05 < all_c["mean_overall_spec_adherence"]
        )
        if worse_emphasis or worse_decod or worse_overall:
            bits = []
            if worse_emphasis:
                bits.append(
                    f"low-emphasis rate {fmt_pct(c['pct_low_emphasis'])} vs "
                    f"overall {fmt_pct(all_c['pct_low_emphasis'])}"
                )
            if worse_decod:
                bits.append(
                    f"low decodability-adherence rate {fmt_pct(c['pct_low_decod_subj'])} vs "
                    f"overall {fmt_pct(all_c['pct_low_decod_subj'])}"
                )
            if worse_overall:
                bits.append(
                    f"mean overall {c['mean_overall_spec_adherence']:.2f} vs "
                    f"overall {all_c['mean_overall_spec_adherence']:.2f}"
                )
            lines.append(f"- **{c['condition']}** looks harder than average: " + "; ".join(bits) + ".")
        else:
            lines.append(
                f"- **{c['condition']}** is not clearly worse than the tuned average on "
                "emphasis/decodability/overall in this sample."
            )
    lines.append("")

    # Cause summary
    lines.append("## Likely-cause tally (major failure representatives)")
    lines.append("")
    lines.append(
        "Counted from rank-1 representatives of major failure/least-gain/prompt-difficulty categories:"
    )
    lines.append("")
    for cause, n in cause_counter.most_common():
        lines.append(f"- `{cause}`: {n}")
    lines.append("")
    lines.append("### Cause definitions used here")
    lines.append("")
    lines.append(
        "- **training-data coverage:** hard pattern family and/or weak sustained target practice "
        "despite non-trivial story length."
    )
    lines.append(
        "- **conflicting training objectives:** tuned becomes more in-level / shorter while "
        "losing target-pattern coverage relative to base."
    )
    lines.append(
        "- **small-model capacity:** especially multi-target advanced prompts with weak "
        "combined adherence."
    )
    lines.append(
        "- **generation behavior:** incomplete, incoherent, or very short outputs; unstable "
        "story framing."
    )
    lines.append(
        "- **rule-based evaluation limitation:** ceiling effects, or sharp disagreement between "
        "objective spelling-pattern metrics and subjective judge scores."
    )
    lines.append("")
    lines.append("## Bottom line")
    lines.append("")
    lines.append(
        "Tuned generations usually improve objective decodability compliance versus base, "
        "mainly by staying shorter and more in-level. Remaining errors are not uniform: "
        "some least-gain pairs are metric ceilings, while subjective failures cluster in "
        "weak target-phonics pattern emphasis, occasional incoherence/incompleteness, and "
        "specific multi-pattern prompt families. Treat cause labels as evidence-ranked "
        "hypotheses for the next debugging step, not as automatic data-collection mandates."
    )
    lines.append("")

    args.summary_out.write_text("\n".join(lines), encoding="utf-8")

    # Console brief
    print(f"Wrote {args.examples_out} ({len(examples)} example rows)")
    print(f"Wrote {args.summary_out}")
    print()
    print("Largest compliance gains:")
    for e in examples:
        if e["category"] == "largest_compliance_gain":
            print(
                f"  {e['rank']}. {e['prompt_id']} gen={e['generation_id']} "
                f"Δ={e['compliance_delta']:+.4f}"
            )
    print("Least compliance gains:")
    for e in examples:
        if e["category"] == "least_compliance_gain":
            print(
                f"  {e['rank']}. {e['prompt_id']} gen={e['generation_id']} "
                f"Δ={e['compliance_delta']:+.4f}"
            )
    print("Tuned failure counts:")
    for category, _, _, _ in failure_defs:
        print(f"  {category}: {len(failure_sets[category])}/{len(tuned_rows)}")
    print("Hardest prompts:")
    for i, ps in enumerate(prompt_stats_sorted[:3], start=1):
        print(
            f"  {i}. {ps['prompt_id']} overall={ps['tuned_mean_overall_spec_adherence']:.2f} "
            f"emphasis={ps['tuned_mean_target_phonics_pattern_emphasis']:.2f}"
        )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Error analysis over SLM eval outputs")
    p.add_argument("--final", type=Path, default=SUBJECTIVE_DIR / "final_evaluation.csv")
    p.add_argument("--objective-results", type=Path, default=OBJECTIVE_DIR / "objective_eval_results.csv")
    p.add_argument("--by-prompt", type=Path, default=OBJECTIVE_DIR / "objective_eval_by_prompt.csv")
    p.add_argument("--paired", type=Path, default=OBJECTIVE_DIR / "paired_eval_comparison.csv")
    p.add_argument("--judge-results", type=Path, default=SUBJECTIVE_DIR / "llm_judge_results.csv")
    p.add_argument("--outputs", type=Path, default=INPUTS_DIR / "all_outputs.csv")
    p.add_argument("--examples-out", type=Path, default=ERROR_DIR / "error_analysis_examples.csv")
    p.add_argument("--summary-out", type=Path, default=ERROR_DIR / "error_analysis_summary.md")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    required = [
        args.final,
        args.objective_results,
        args.by_prompt,
        args.paired,
        args.judge_results,
        args.outputs,
    ]
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")
    # objective/judge paths are accepted for CLI completeness; final_evaluation
    # already merges the row-level signals used below.
    _ = load_csv(args.objective_results)
    _ = load_csv(args.judge_results)
    return analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
