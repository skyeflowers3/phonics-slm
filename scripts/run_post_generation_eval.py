#!/usr/bin/env python3
"""
Post-generation evaluation for phonics-slm.

Runs AFTER all_outputs.csv already exists. Does not generate stories.

Produces:
  - results/scored_outputs.csv
  - results/evaluation_summary.csv
  - results/error_analysis/error_analysis.csv

Objective checks reuse phonics_profiler_threshold4 via evaluate_outputs helpers.
LLM judge uses a decodable-story rubric (spec_adherence, robustness, task_quality)
with prompt-level consistency scored once across the three matched generations.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_outputs import (  # noqa: E402
    DISPLAY_TO_INTERNAL,
    MIN_DISTINCT_FOR_PASS,
    compute_level_metrics,
    detect_phonics_leakage,
    detect_title,
    parse_target_phonics,
    permitted_ceiling,
    profile_story_patterns,
    validate_targets,
)
from llm_judge import JudgeClient, extract_json_object, load_dotenv_files  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUTS_DIR = REPO_ROOT / "inputs"
RESULTS_DIR = REPO_ROOT / "results"

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
INCOMPLETE_TRAILERS = (
    "...",
    "…",
    " -",
    " —",
    " –",
    '"',
    "'",
    ",",
    ";",
    ":",
    "(",
    "[",
    "{",
)

# Per-story judge dimensions (consistency is scored separately at prompt level).
STORY_JUDGE_DIMENSIONS = [
    (
        "spec_adherence",
        "Spec adherence",
        "PRIMARY: Does this story work as a decodable text that students at the "
        "requested phonics level can actually read and learn from? "
        "The most important requirement is staying within the permitted phonics "
        "progression so the passage is readable for that learner. "
        "Secondary: natural title, complete enough narrative, no phonics metalanguage. "
        "Do NOT penalize mainly for missing some target-pattern repetitions if the "
        "text is appropriately decodable. "
        "Do NOT reward advanced vocabulary, literary style, or length. "
        "0 = not usable as decodable student text (too hard / off-level / unreadable "
        "for the requested level); "
        "1 = partly usable but noticeable above-level vocabulary or other spec issues; "
        "2 = clearly usable decodable practice text for the requested level.",
    ),
    (
        "robustness",
        "Robustness",
        "How stable/usable is this story as instructional decodable text? Penalize "
        "truncation, repetition loops, gibberish, contradictory plot, or formatting "
        "collapse. 0 = unusable / broken; 1 = mostly usable with clear defects; "
        "2 = clean, reliable instructional text. Do not reward length.",
    ),
    (
        "task_quality",
        "Task quality",
        "Given that the text is meant for students to read and learn from: how "
        "effective is it as practice at this level? Prefer clear, child-readable "
        "decodable prose that also gives some useful practice of the requested "
        "patterns. Coverage helps, but readability at level matters more than "
        "stuffing in every pattern. "
        "0 = poor learning text; 1 = mixed; 2 = strong student-facing practice text.",
    ),
]

STORY_DIM_KEYS = [k for k, _, _ in STORY_JUDGE_DIMENSIONS]
ALL_JUDGE_KEYS = STORY_DIM_KEYS + ["consistency"]

SCORED_FIELDS = [
    "model",
    "prompt_id",
    "generation_id",
    "seed",
    "target_phonics",
    "target_phonics_coverage",
    "off_target_phonics_rate",
    "on_level_rate",
    "weighted_above_level_rate",
    "title_present",
    "sentence_count",
    "no_duplicate_sentences",
    "has_duplicate_sentences",
    "complete_output",
    "incomplete_output",
    "no_phonics_leakage",
    "phonics_leakage",
    "word_count",
    "spec_adherence_score",
    "robustness_score",
    "task_quality_score",
    "consistency_score",
    "spec_adherence_justification",
    "robustness_justification",
    "task_quality_justification",
    "consistency_justification",
    "detected_title",
    "duplicate_sentence_norms",
    "incomplete_reasons",
    "phonics_leakage_terms",
    "above_level_words",
    "permitted_ceiling_stage",
    "raw_story_judge_json",
    "raw_consistency_judge_json",
    "prompt",
    "story",
]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    return rows


def story_body(story: str) -> str:
    """Strip a detected title line so sentence checks focus on narrative body."""
    has_title, title = detect_title(story or "")
    text = (story or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not has_title or not title:
        return text
    lines = text.split("\n")
    # Drop leading blank / title-ish first line(s).
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines:
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines.pop(0)
    return "\n".join(lines).strip()


def split_sentences(text: str) -> list[str]:
    chunks = [c.strip() for c in SENTENCE_SPLIT_RE.split(text) if c and c.strip()]
    # Keep fragments that look like sentences / dialogue turns.
    return [c for c in chunks if re.search(r"[A-Za-z]", c)]


def normalize_sentence(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def detect_incomplete(story: str, body: str, sentences: list[str]) -> tuple[int, str]:
    """Heuristic incomplete / truncated detection. 1 = incomplete."""
    reasons: list[str] = []
    raw = (story or "").strip()
    if not raw:
        return 1, "empty_story"
    if not body:
        reasons.append("title_only_or_empty_body")
    if len(sentences) <= 1 and len(body.split()) < 20:
        reasons.append("too_short")
    stripped = body.rstrip() or raw.rstrip()
    if stripped.endswith(INCOMPLETE_TRAILERS):
        reasons.append("bad_ending_punctuation")
    if re.search(r"\b(and|but|or|the|a|an|to|of|with|for)\s*$", stripped, re.I):
        reasons.append("ends_mid_phrase")
    if re.search(r"[A-Za-z]$", stripped) and not re.search(r"[.!?\"']\s*$", stripped):
        # Ends on a letter with no terminal punctuation.
        reasons.append("no_terminal_punctuation")
    if re.search(r"\b(Title|Write a|Emphasize these|phonics patterns)\b", raw, re.I):
        # Prompt echo / unfinished template bleed.
        if "phonics" in raw.lower() and "title:" not in raw.lower()[:40]:
            pass
    open_q = raw.count('"') % 2 == 1
    if open_q:
        reasons.append("unbalanced_quotes")
    return (1 if reasons else 0), ",".join(reasons)


def objective_checks(row: dict, row_index: int) -> dict[str, Any]:
    story = row.get("story") or ""
    targets = validate_targets(
        parse_target_phonics(row.get("target_phonics", "")), row_index
    )
    profile = profile_story_patterns(story)
    ceiling = permitted_ceiling(targets)
    level = compute_level_metrics(profile, ceiling)

    credits: list[float] = []
    for display_name in targets:
        internal = DISPLAY_TO_INTERNAL[display_name]
        distinct = profile["pattern_distinct_counts"].get(internal, 0)
        credits.append(min(distinct / MIN_DISTINCT_FOR_PASS, 1.0))
    target_coverage = round(_mean(credits), 4) if credits else 0.0

    has_title, title_text = detect_title(story)
    leaked, leak_terms = detect_phonics_leakage(story)
    body = story_body(story)
    sentences = split_sentences(body)
    sentence_count = len(sentences)

    norms = [normalize_sentence(s) for s in sentences if normalize_sentence(s)]
    dup_counts = Counter(norms)
    duplicate_sentences = sorted({s for s, n in dup_counts.items() if n > 1 and s})
    has_duplicate_sentences = int(bool(duplicate_sentences))

    incomplete, incomplete_reasons = detect_incomplete(story, body, sentences)

    # Higher-is-better objective scores for summary averaging.
    return {
        "model": row["model"],
        "prompt_id": row["prompt_id"],
        "generation_id": row["generation_id"],
        "seed": row.get("seed", ""),
        "target_phonics": row.get("target_phonics", ""),
        "prompt": row.get("prompt", ""),
        "story": story,
        # Continuous / rate metrics
        "target_phonics_coverage": target_coverage,
        "off_target_phonics_rate": level["above_level_rate"],
        "on_level_rate": level["decodability_compliance"],
        "weighted_above_level_rate": level["weighted_above_level_rate"],
        "word_count": profile["word_count"],
        "sentence_count": sentence_count,
        # Binary pass metrics (1 = good)
        "title_present": int(has_title),
        "no_duplicate_sentences": int(not has_duplicate_sentences),
        "complete_output": int(not incomplete),
        "no_phonics_leakage": int(not leaked),
        # Diagnostics
        "detected_title": title_text,
        "has_duplicate_sentences": has_duplicate_sentences,
        "duplicate_sentence_norms": " | ".join(duplicate_sentences[:5]),
        "incomplete_output": incomplete,
        "incomplete_reasons": incomplete_reasons,
        "phonics_leakage": int(leaked),
        "phonics_leakage_terms": leak_terms,
        "above_level_words": level["above_level_words"],
        "permitted_ceiling_stage": ceiling,
    }


def validate_story_judge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, _, _ in STORY_JUDGE_DIMENSIONS:
        if key not in payload:
            raise ValueError(f"Missing judge dimension: {key}")
        item = payload[key]
        if not isinstance(item, dict):
            raise ValueError(f"{key} must be an object with score/justification")
        if "score" not in item or "justification" not in item:
            raise ValueError(f"{key} missing score or justification")
        score = item["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"{key} score must be numeric")
        score_int = int(score)
        if score_int != score or score_int not in (0, 1, 2):
            raise ValueError(f"{key} score must be integer 0–2, got {score!r}")
        justification = str(item["justification"]).strip()
        if not justification:
            raise ValueError(f"{key} justification empty")
        first = re.split(r"(?<=[.!?])\s+", justification, maxsplit=1)[0].strip()
        out[f"{key}_score"] = score_int
        out[f"{key}_justification"] = first or justification
    out["raw_story_judge_json"] = json.dumps(
        {
            k: {
                "score": out[f"{k}_score"],
                "justification": out[f"{k}_justification"],
            }
            for k in STORY_DIM_KEYS
        },
        ensure_ascii=False,
    )
    return out


def validate_consistency_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "consistency" not in payload:
        raise ValueError("Missing consistency in judge response")
    item = payload["consistency"]
    if not isinstance(item, dict) or "score" not in item or "justification" not in item:
        raise ValueError("consistency must be an object with score/justification")
    score = item["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("consistency score must be numeric")
    score_int = int(score)
    if score_int != score or score_int not in (0, 1, 2):
        raise ValueError(f"consistency score must be integer 0–2, got {score!r}")
    justification = str(item["justification"]).strip()
    if not justification:
        raise ValueError("consistency justification empty")
    first = re.split(r"(?<=[.!?])\s+", justification, maxsplit=1)[0].strip()
    return {
        "consistency_score": score_int,
        "consistency_justification": first or justification,
        "raw_consistency_judge_json": json.dumps(
            {"consistency": {"score": score_int, "justification": first or justification}},
            ensure_ascii=False,
        ),
    }


def build_story_judge_messages(target_phonics: str, prompt: str, story: str) -> list[dict]:
    schema = {
        key: {"score": 0, "justification": "..."} for key, _, _ in STORY_JUDGE_DIMENSIONS
    }
    dim_text = "\n".join(
        f"- {key} ({title}): {desc}" for key, title, desc in STORY_JUDGE_DIMENSIONS
    )
    system = (
        "You are an expert early-literacy evaluator of instructional decodable text. "
        "Score ONLY the story against the requested phonics targets and behavior spec. "
        "The #1 requirement is that the story is a decodable text students at that "
        "level can read and learn from — staying in-level beats literary quality, "
        "length, and even imperfect target-pattern coverage. "
        "Never infer or mention which model produced the story. "
        "Return valid JSON only matching the schema."
    )
    user = f"""Behavior specification (priority order):
1. MOST IMPORTANT: Produce a decodable story students at the requested phonics
   progression can read and learn from. Minimize above-level vocabulary/patterns.
2. Emphasize the requested phonics patterns enough to be useful practice, without
   sacrificing decodability.
3. Include a natural title; form a coherent beginning/middle/end; do not mention
   phonics terms or pattern names.

Do NOT reward advanced vocabulary, sophistication, or longer stories.
An advanced story that happens to contain the target patterns is NOT on-spec if
students at the requested level could not decode it.

Target phonics: {target_phonics}

Generation prompt (context only):
{prompt}

Story to evaluate:
{story}

Score these dimensions on an integer 0–2 scale:
{dim_text}

CRITICAL: Keep each justification to ONE short sentence (max ~20 words).
Return compact JSON only, exactly like:
{json.dumps(schema, indent=2)}
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_consistency_messages(
    target_phonics: str,
    prompt: str,
    stories: list[tuple[str, str]],
) -> list[dict]:
    """stories: list of (generation_id, story_text)."""
    blocks = []
    for gen_id, story in stories:
        blocks.append(f"[generation_id={gen_id}]\n{story}")
    joined = "\n\n---\n\n".join(blocks)
    schema = {"consistency": {"score": 0, "justification": "..."}}
    system = (
        "You evaluate consistency across multiple decodable-story generations for "
        "the SAME prompt. Prioritize whether they stably produce readable "
        "in-level decodable text students can learn from. Do not reward literary "
        "variety. Return JSON only."
    )
    user = f"""Target phonics: {target_phonics}

Generation prompt (context only):
{prompt}

Three generations for the same model+prompt (identity of the model is withheld):

{joined}

Score consistency 0–2:
0 = highly inconsistent (decodability/readability for the level swings wildly)
1 = mixed consistency
2 = consistently usable as in-level decodable student text across generations

Return JSON exactly like:
{json.dumps(schema, indent=2)}
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def judge_with_retries(
    client: JudgeClient,
    messages: list[dict],
    validator,
    max_retries: int = 4,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            raw = client.judge(messages)
            payload = extract_json_object(raw)
            return validator(payload)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"Judge failed after retries: {last_error}")


def resolve_provider_and_model(args: argparse.Namespace) -> tuple[str, str]:
    load_dotenv_files()
    provider = args.provider
    if provider is None:
        if os_env("TFY_API_KEY"):
            provider = "truefoundry"
        elif os_env("OPENAI_API_KEY"):
            provider = "openai"
        elif os_env("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        else:
            raise SystemExit(
                "No API credentials found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, "
                "or TFY_API_KEY."
            )
    defaults = {
        "openai": "gpt-4.1",
        "anthropic": "claude-sonnet-4-6",
        "truefoundry": "openai-group/gpt-4.1",
    }
    model = args.model or defaults[provider]
    return provider, model


def os_env(key: str) -> str:
    import os

    return os.environ.get(key, "")


def build_summary(scored: list[dict]) -> list[dict]:
    objective_metrics = [
        "target_phonics_coverage",
        "off_target_phonics_rate",
        "on_level_rate",
        "title_present",
        "no_duplicate_sentences",
        "complete_output",
        "no_phonics_leakage",
        "word_count",
        "sentence_count",
    ]
    judge_metrics = [f"{k}_score" for k in ALL_JUDGE_KEYS]

    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in scored:
        by_model[row["model"]].append(row)

    means: dict[str, dict[str, float]] = {}
    for model, rows in by_model.items():
        means[model] = {}
        for metric in objective_metrics + judge_metrics:
            means[model][metric] = round(_mean([float(r[metric]) for r in rows]), 4)

    # Long-form summary table: one row per metric.
    out = []
    for metric in objective_metrics + judge_metrics:
        base = means.get("base", {}).get(metric)
        tuned = means.get("tuned", {}).get(metric)
        delta = None
        if base is not None and tuned is not None:
            delta = round(tuned - base, 4)
        out.append(
            {
                "metric": metric,
                "metric_family": (
                    "judge" if metric.endswith("_score") or metric in {
                        f"{k}_score" for k in ALL_JUDGE_KEYS
                    }
                    else "objective"
                ),
                "base_mean": base if base is not None else "",
                "tuned_mean": tuned if tuned is not None else "",
                "tuned_minus_base": delta if delta is not None else "",
                "n_base": len(by_model.get("base", [])),
                "n_tuned": len(by_model.get("tuned", [])),
            }
        )
    return out


def build_error_analysis(scored: list[dict], top_n: int = 8) -> list[dict]:
    tuned = [r for r in scored if r["model"] == "tuned"]
    has_judge = any(str(r.get("spec_adherence_score", "")) != "" for r in tuned)
    failure_defs = [
        ("low_target_phonics_coverage", lambda r: float(r["target_phonics_coverage"]) < 0.5),
        ("high_off_target_phonics", lambda r: float(r["off_target_phonics_rate"]) >= 0.25),
        ("missing_title", lambda r: int(r["title_present"]) == 0),
        ("duplicate_sentences", lambda r: int(r["has_duplicate_sentences"]) == 1),
        ("incomplete_output", lambda r: int(r["incomplete_output"]) == 1),
        ("phonics_leakage", lambda r: int(r["phonics_leakage"]) == 1),
    ]
    if has_judge:
        failure_defs.extend(
            [
                ("low_spec_adherence", lambda r: int(r["spec_adherence_score"]) == 0),
                ("low_robustness", lambda r: int(r["robustness_score"]) == 0),
                ("low_task_quality", lambda r: int(r["task_quality_score"]) == 0),
                ("low_consistency", lambda r: int(r["consistency_score"]) == 0),
            ]
        )

    rows_out: list[dict] = []
    for failure_type, pred in failure_defs:
        flagged = [r for r in tuned if pred(r)]
        if not flagged:
            continue
        # Prefer worst examples by relevant judge/objective signals.
        flagged_sorted = sorted(
            flagged,
            key=lambda r: (
                int(r["spec_adherence_score"]),
                int(r["task_quality_score"]),
                int(r["robustness_score"]),
                float(r["target_phonics_coverage"]),
                -float(r["off_target_phonics_rate"]),
            ),
        )
        for rank, ex in enumerate(flagged_sorted[:top_n], start=1):
            rows_out.append(
                {
                    "failure_type": failure_type,
                    "rank": rank,
                    "n_tuned_flagged": len(flagged),
                    "pct_tuned_flagged": round(len(flagged) / max(len(tuned), 1), 4),
                    "model": ex["model"],
                    "prompt_id": ex["prompt_id"],
                    "generation_id": ex["generation_id"],
                    "seed": ex["seed"],
                    "target_phonics": ex["target_phonics"],
                    "target_phonics_coverage": ex["target_phonics_coverage"],
                    "off_target_phonics_rate": ex["off_target_phonics_rate"],
                    "title_present": ex["title_present"],
                    "sentence_count": ex["sentence_count"],
                    "incomplete_output": ex["incomplete_output"],
                    "spec_adherence_score": ex["spec_adherence_score"],
                    "robustness_score": ex["robustness_score"],
                    "task_quality_score": ex["task_quality_score"],
                    "consistency_score": ex["consistency_score"],
                    "story": ex["story"],
                    "spec_adherence_justification": ex.get("spec_adherence_justification", ""),
                    "robustness_justification": ex.get("robustness_justification", ""),
                    "task_quality_justification": ex.get("task_quality_justification", ""),
                    "consistency_justification": ex.get("consistency_justification", ""),
                }
            )
    return rows_out


def run_evaluation(args: argparse.Namespace) -> int:
    rows = load_csv(args.input)
    required = {"model", "prompt_id", "generation_id", "seed", "target_phonics", "prompt", "story"}
    missing = required - set(rows[0].keys())
    if missing:
        raise SystemExit(f"all_outputs.csv missing columns: {sorted(missing)}")

    # Verify matching base/tuned pairs (preserve existing pipeline contract).
    base_keys = {
        (r["prompt_id"], str(r["generation_id"]), str(r["seed"]))
        for r in rows
        if r["model"] == "base"
    }
    tuned_keys = {
        (r["prompt_id"], str(r["generation_id"]), str(r["seed"]))
        for r in rows
        if r["model"] == "tuned"
    }
    if base_keys != tuned_keys:
        only_base = sorted(base_keys - tuned_keys)
        only_tuned = sorted(tuned_keys - base_keys)
        raise SystemExit(
            "Base/tuned outputs are not matched by prompt_id+generation_id+seed. "
            f"only_base={only_base[:5]} only_tuned={only_tuned[:5]}"
        )

    print(f"Loaded {len(rows)} rows from {args.input}")
    objective_rows = [objective_checks(r, i) for i, r in enumerate(rows)]

    if args.objective_only:
        scored = []
        for obj in objective_rows:
            row = dict(obj)
            for key in ALL_JUDGE_KEYS:
                row[f"{key}_score"] = ""
                row[f"{key}_justification"] = ""
            row["raw_story_judge_json"] = ""
            row["raw_consistency_judge_json"] = ""
            scored.append(row)
        return write_outputs(scored, args)

    provider, model = resolve_provider_and_model(args)
    client = JudgeClient(
        provider=provider,
        model=model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        base_url=args.base_url,
    )
    print(
        f"Judge provider={provider} model={model} | "
        "model identity withheld | consistency scored at prompt level"
    )

    # Optional resume of per-story judgments.
    existing_story: dict[tuple, dict] = {}
    if args.resume and args.scored.exists():
        for row in load_csv(args.scored):
            key = (row["model"], row["prompt_id"], str(row["generation_id"]))
            if row.get("spec_adherence_score", "") != "":
                existing_story[key] = row
        print(f"Resume: {len(existing_story)} story judgments already present")

    scored_by_key: dict[tuple, dict] = {}
    to_judge = []
    for obj in objective_rows:
        key = (obj["model"], obj["prompt_id"], str(obj["generation_id"]))
        if key in existing_story:
            merged = dict(obj)
            prev = existing_story[key]
            for dim in STORY_DIM_KEYS:
                merged[f"{dim}_score"] = int(prev[f"{dim}_score"])
                merged[f"{dim}_justification"] = prev.get(f"{dim}_justification", "")
            merged["raw_story_judge_json"] = prev.get("raw_story_judge_json", "")
            scored_by_key[key] = merged
        else:
            to_judge.append(obj)

    if args.limit is not None:
        to_judge = to_judge[: args.limit]

    for i, obj in enumerate(to_judge, start=1):
        messages = build_story_judge_messages(
            obj["target_phonics"], obj["prompt"], obj["story"]
        )
        judged = judge_with_retries(
            client, messages, validate_story_judge_payload, max_retries=args.max_retries
        )
        merged = dict(obj)
        merged.update(judged)
        key = (obj["model"], obj["prompt_id"], str(obj["generation_id"]))
        scored_by_key[key] = merged
        print(
            f"[story {i}/{len(to_judge)}] {obj['model']} {obj['prompt_id']} "
            f"gen={obj['generation_id']} "
            f"spec={merged['spec_adherence_score']} "
            f"robust={merged['robustness_score']} "
            f"quality={merged['task_quality_score']}"
        )
        # Checkpoint scored rows so --resume can recover after API/truncation failures.
        checkpoint_rows = []
        for o in objective_rows:
            k = (o["model"], o["prompt_id"], str(o["generation_id"]))
            if k in scored_by_key:
                row = dict(scored_by_key[k])
            else:
                row = dict(o)
                for dim in STORY_DIM_KEYS:
                    row[f"{dim}_score"] = ""
                    row[f"{dim}_justification"] = ""
                row["raw_story_judge_json"] = ""
            row.setdefault("consistency_score", "")
            row.setdefault("consistency_justification", "")
            row.setdefault("raw_consistency_judge_json", "")
            checkpoint_rows.append(row)
        write_csv(args.scored, checkpoint_rows, SCORED_FIELDS)
        if args.sleep > 0:
            time.sleep(args.sleep)

    # Ensure all objective rows are present (limit mode may leave some unjudged).
    for obj in objective_rows:
        key = (obj["model"], obj["prompt_id"], str(obj["generation_id"]))
        if key not in scored_by_key:
            merged = dict(obj)
            for dim in STORY_DIM_KEYS:
                merged[f"{dim}_score"] = ""
                merged[f"{dim}_justification"] = ""
            merged["raw_story_judge_json"] = ""
            scored_by_key[key] = merged

    # Prompt-level consistency: one score per (model, prompt_id), broadcast to gens.
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in scored_by_key.values():
        if row.get("spec_adherence_score", "") == "":
            continue
        groups[(row["model"], row["prompt_id"])].append(row)

    consistency_cache: dict[tuple[str, str], dict] = {}
    # Resume consistency if present.
    if args.resume and args.scored.exists():
        for row in load_csv(args.scored):
            gkey = (row["model"], row["prompt_id"])
            if row.get("consistency_score", "") != "" and gkey not in consistency_cache:
                consistency_cache[gkey] = {
                    "consistency_score": int(row["consistency_score"]),
                    "consistency_justification": row.get("consistency_justification", ""),
                    "raw_consistency_judge_json": row.get("raw_consistency_judge_json", ""),
                }

    group_items = sorted(groups.items())
    for gi, ((model, prompt_id), members) in enumerate(group_items, start=1):
        gkey = (model, prompt_id)
        if gkey in consistency_cache:
            cons = consistency_cache[gkey]
        else:
            members_sorted = sorted(members, key=lambda r: str(r["generation_id"]))
            stories = [(str(r["generation_id"]), r["story"]) for r in members_sorted]
            messages = build_consistency_messages(
                members_sorted[0]["target_phonics"],
                members_sorted[0]["prompt"],
                stories,
            )
            cons = judge_with_retries(
                client,
                messages,
                validate_consistency_payload,
                max_retries=args.max_retries,
            )
            consistency_cache[gkey] = cons
            print(
                f"[consistency {gi}/{len(group_items)}] {model} {prompt_id} "
                f"-> {cons['consistency_score']}"
            )
            if args.sleep > 0:
                time.sleep(args.sleep)

        for member in members:
            key = (member["model"], member["prompt_id"], str(member["generation_id"]))
            scored_by_key[key].update(cons)

    # Fill consistency blanks for unjudged rows.
    for key, row in scored_by_key.items():
        if "consistency_score" not in row or row["consistency_score"] == "":
            row["consistency_score"] = ""
            row["consistency_justification"] = ""
            row["raw_consistency_judge_json"] = ""

    scored = [
        scored_by_key[(o["model"], o["prompt_id"], str(o["generation_id"]))]
        for o in objective_rows
    ]
    return write_outputs(scored, args)


def write_outputs(scored: list[dict], args: argparse.Namespace) -> int:
    write_csv(args.scored, scored, SCORED_FIELDS)

    # Summary requires numeric judge scores; skip empty if objective_only.
    scored_for_summary = [
        r for r in scored if str(r.get("spec_adherence_score", "")) != ""
    ]
    if scored_for_summary:
        summary = build_summary(scored_for_summary)
        error_rows = build_error_analysis(scored_for_summary)
    else:
        summary = build_summary(
            [{**r, **{f"{k}_score": 0 for k in ALL_JUDGE_KEYS}} for r in scored]
        )
        summary = [s for s in summary if s["metric_family"] == "objective"]
        error_rows = build_error_analysis(scored)

    summary_fields = [
        "metric",
        "metric_family",
        "base_mean",
        "tuned_mean",
        "tuned_minus_base",
        "n_base",
        "n_tuned",
    ]
    write_csv(args.summary, summary, summary_fields)

    error_fields = [
        "failure_type",
        "rank",
        "n_tuned_flagged",
        "pct_tuned_flagged",
        "model",
        "prompt_id",
        "generation_id",
        "seed",
        "target_phonics",
        "target_phonics_coverage",
        "off_target_phonics_rate",
        "title_present",
        "sentence_count",
        "incomplete_output",
        "spec_adherence_score",
        "robustness_score",
        "task_quality_score",
        "consistency_score",
        "story",
        "spec_adherence_justification",
        "robustness_justification",
        "task_quality_justification",
        "consistency_justification",
    ]
    write_csv(args.error_analysis, error_rows, error_fields)

    print()
    print("evaluation_summary.csv")
    print("metric                              base    tuned   delta")
    print("----------------------------------  ------  ------  ------")
    for row in summary:
        print(
            f"{row['metric'][:34]:34}  "
            f"{str(row['base_mean']):6}  "
            f"{str(row['tuned_mean']):6}  "
            f"{str(row['tuned_minus_base']):6}"
        )
    print()
    print(f"Saved: {args.scored}")
    print(f"Saved: {args.summary}")
    print(f"Saved: {args.error_analysis}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Post-generation objective + LLM-judge evaluation for all_outputs.csv"
    )
    p.add_argument("--input", type=Path, default=INPUTS_DIR / "all_outputs.csv")
    p.add_argument("--scored", type=Path, default=RESULTS_DIR / "scored_outputs.csv")
    p.add_argument("--summary", type=Path, default=RESULTS_DIR / "evaluation_summary.csv")
    p.add_argument(
        "--error-analysis",
        type=Path,
        default=RESULTS_DIR / "error_analysis" / "error_analysis.csv",
    )
    p.add_argument("--provider", choices=["openai", "anthropic", "truefoundry"], default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=2000)
    p.add_argument("--sleep", type=float, default=0.2)
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="Cap story judgments (smoke test)")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--objective-only",
        action="store_true",
        help="Skip LLM judge; write objective columns only",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        return run_evaluation(args)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
