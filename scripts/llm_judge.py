#!/usr/bin/env python3
"""
Blind LLM-as-a-judge evaluation for base-vs-tuned SLM decodable stories.

The judge never sees the `model` column. Scoring is based only on:
  - target_phonics
  - the generation prompt (optional context)
  - the story text

Rubric evaluates behavior-spec adherence for decodable reading passages:
narrative completeness, coherence, decodability adherence, target phonics
emphasis, and overall adherence — without rewarding literary sophistication.

Uses a frontier model via OpenAI, Anthropic, or a TrueFoundry OpenAI-compatible
gateway (TFY_API_KEY).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Rubric / behavior context shared with the judge
# ---------------------------------------------------------------------------
BEHAVIOR_SPEC = """
Given 1–3 target phonics focuses, the model generates a decodable story that
adheres to the requested phonics level and forms a complete narrative with a
clear beginning, middle, and end. The model minimizes phonics patterns above
the requested decodability level, avoids unnecessary out-of-scope vocabulary,
and does not produce incomplete or incoherent stories.
""".strip()

PHONICS_HIERARCHY = """
Phonics progression (lower = more basic; higher = more advanced):
1. short-vowel words
2. consonant blends and consonant digraphs
3. final-e words
4. vowel teams
5. r-controlled vowels
6. diphthongs
7. multisyllabic words

The permitted ceiling for a story is the highest requested target pattern.
Vocabulary or spelling patterns above that ceiling should lower the
decodability-adherence score. Target-phonics emphasis is separate: a story can
stay within level yet still fail to practice the requested patterns enough.
""".strip()

JUDGING_PRIORITIES = """
Before scoring, remember:
- Evaluate the story only against the behavior specification.
- Do NOT reward advanced vocabulary.
- Do NOT reward literary sophistication.
- Do NOT reward longer stories.
- Do NOT reward more descriptive writing.
- Judge whether the story would function as an appropriate decodable reading
  passage for the requested phonics progression.

Score each dimension independently. Avoid double-counting the same issue across
dimensions when possible (e.g., above-level vocabulary belongs under
decodability adherence, not coherence or completeness).
""".strip()

SCORE_DIMENSIONS = [
    (
        "narrative_completeness",
        "Narrative completeness",
        "Score only whether the story has a beginning, middle, and ending. "
        "0 = incomplete or obviously cut off; "
        "1 = mostly complete but weak transitions or ending; "
        "2 = clear beginning, middle, and ending. "
        "Do not reward length or complexity.",
    ),
    (
        "coherence",
        "Coherence",
        "Judge whether the story is understandable and internally consistent. "
        "0 = confusing or contradictory; "
        "1 = mostly coherent with minor issues; "
        "2 = easy to follow and internally consistent. "
        "Do not reward literary quality or creativity.",
    ),
    (
        "decodability_adherence",
        "Decodability adherence",
        "Judge whether the vocabulary stays within the requested phonics progression. "
        "0 = frequent unnecessary vocabulary or spelling patterns above the requested progression; "
        "1 = mostly appropriate but contains noticeable unnecessary above-level vocabulary; "
        "2 = vocabulary overwhelmingly stays within the requested progression with only "
        "occasional unavoidable exceptions.",
    ),
    (
        "target_phonics_pattern_emphasis",
        "Target phonics pattern emphasis",
        "Judge whether the requested phonics patterns are practiced throughout the story. "
        "0 = requested patterns are rarely present; "
        "1 = requested patterns appear several times; "
        "2 = requested patterns are naturally repeated throughout the story. "
        "Do not require every sentence to contain the target pattern.",
    ),
    (
        "overall_spec_adherence",
        "Overall behavior specification adherence",
        "Considering all of the above, how well does the story satisfy the complete "
        "behavior specification? "
        "0 = poor adherence; "
        "1 = partial adherence; "
        "2 = strong adherence.",
    ),
]

DIMENSION_KEYS = [key for key, _, _ in SCORE_DIMENSIONS]

JOIN_KEYS = ("prompt_id", "generation_id", "seed")

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUTS_DIR = REPO_ROOT / "inputs"
OBJECTIVE_DIR = REPO_ROOT / "results" / "objective"
SUBJECTIVE_DIR = REPO_ROOT / "results" / "subjective"


def load_dotenv_files() -> None:
    """Load simple KEY=VALUE / export KEY=VALUE lines from common .env locations."""
    candidates = [
        Path.cwd() / ".env",
        Path.home() / ".env",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                # Do not overwrite explicitly exported shell vars.
                if key and key not in os.environ:
                    os.environ[key] = value
        except OSError:
            continue


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pct(rate: float) -> str:
    return f"{100.0 * rate:.1f}%"


def build_judge_messages(target_phonics: str, prompt: str, story: str) -> list[dict[str, str]]:
    """
    Build the blind judge prompt.

    Intentionally omits any base/tuned model identity.
    """
    dimension_block = "\n".join(
        f"- {key}: {title}\n  Scale: {scale}"
        for key, title, scale in SCORE_DIMENSIONS
    )

    schema_example = {
        "narrative_completeness": {"score": 0, "justification": "..."},
        "coherence": {"score": 0, "justification": "..."},
        "decodability_adherence": {"score": 0, "justification": "..."},
        "target_phonics_pattern_emphasis": {"score": 0, "justification": "..."},
        "overall_spec_adherence": {"score": 0, "justification": "..."},
    }

    system = (
        "You are an expert early-literacy evaluator of decodable reading passages.\n"
        "Evaluate each story only against the behavior specification for the "
        "requested phonics progression.\n"
        "Do not reward advanced vocabulary, literary sophistication, longer "
        "stories, or more descriptive writing.\n"
        "Do not infer quality from authorship or model identity; none is provided.\n"
        "Return structured JSON only. No markdown fences, no extra keys."
    )

    user = f"""Behavior specification:
{BEHAVIOR_SPEC}

{PHONICS_HIERARCHY}

{JUDGING_PRIORITIES}

Score these five dimensions independently on an integer 0–2 scale.
For every dimension, provide exactly one concise justification sentence.

Dimensions:
{dimension_block}

Requested target phonics:
{target_phonics}

Generation instructions given to the story model:
\"\"\"{prompt}\"\"\"

Story to evaluate:
\"\"\"{story}\"\"\"

Return JSON exactly in this shape:
{json.dumps(schema_example, indent=2)}
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from a model response, tolerating accidental fences."""
    if not text or not text.strip():
        raise ValueError("Empty judge response")

    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"No JSON object found in response: {cleaned[:200]!r}")
        cleaned = cleaned[start : end + 1]

    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("Judge JSON root must be an object")
    return data


def validate_scores(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize and validate the six scored dimensions."""
    out: dict[str, Any] = {}
    for key, title, _ in SCORE_DIMENSIONS:
        if key not in payload:
            raise ValueError(f"Missing dimension in judge response: {key}")
        item = payload[key]
        if not isinstance(item, dict):
            raise ValueError(f"Dimension {key} must be an object with score/justification")
        if "score" not in item or "justification" not in item:
            raise ValueError(f"Dimension {key} missing score or justification")
        score = item["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"Dimension {key} score must be numeric, got {score!r}")
        score_int = int(score)
        if score_int != score or score_int not in (0, 1, 2):
            raise ValueError(f"Dimension {key} score must be integer 0–2, got {score!r}")
        justification = str(item["justification"]).strip()
        if not justification:
            raise ValueError(f"Dimension {key} justification is empty")
        # Keep to one sentence when the model returns more.
        first = re.split(r"(?<=[.!?])\s+", justification, maxsplit=1)[0].strip()
        out[f"{key}_score"] = score_int
        out[f"{key}_justification"] = first or justification
    out["raw_judge_json"] = json.dumps(
        {
            k: {
                "score": out[f"{k}_score"],
                "justification": out[f"{k}_justification"],
            }
            for k in DIMENSION_KEYS
        },
        ensure_ascii=False,
    )
    return out


class JudgeClient:
    """Thin wrapper over OpenAI / Anthropic / TrueFoundry chat APIs."""

    def __init__(
        self,
        provider: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1200,
        base_url: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self._openai = None
        self._anthropic = None

        if provider in {"openai", "truefoundry"}:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise SystemExit(
                    "openai package required. Install with: pip install openai"
                ) from exc
            if provider == "openai":
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise SystemExit("OPENAI_API_KEY is not set.")
                kwargs: dict[str, Any] = {"api_key": api_key}
                if base_url:
                    kwargs["base_url"] = base_url
                elif os.environ.get("OPENAI_BASE_URL"):
                    kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
            else:
                api_key = os.environ.get("TFY_API_KEY") or os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise SystemExit(
                        "TFY_API_KEY (or OPENAI_API_KEY) is required for --provider truefoundry."
                    )
                kwargs = {
                    "api_key": api_key,
                    "base_url": base_url
                    or os.environ.get("OPENAI_BASE_URL")
                    or os.environ.get("TFY_BASE_URL")
                    or "https://gateway.truefoundry.ai",
                }
            self._openai = OpenAI(**kwargs)

        elif provider == "anthropic":
            try:
                import anthropic
            except ImportError as exc:
                raise SystemExit(
                    "anthropic package required. Install with: pip install anthropic"
                ) from exc
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                # TrueFoundry Claude-Code style setups sometimes only have TFY key.
                api_key = os.environ.get("TFY_API_KEY")
            if not api_key:
                raise SystemExit("ANTHROPIC_API_KEY is not set.")
            client_kwargs: dict[str, Any] = {"api_key": api_key}
            anth_base = base_url or os.environ.get("ANTHROPIC_BASE_URL")
            if anth_base:
                client_kwargs["base_url"] = anth_base
            default_headers = {}
            custom = os.environ.get("ANTHROPIC_CUSTOM_HEADERS", "")
            # Support "Header: value" or "Header: $TFY_API_KEY"
            for part in re.split(r"[\n,]", custom):
                part = part.strip()
                if not part or ":" not in part:
                    continue
                h_key, h_val = part.split(":", 1)
                h_val = h_val.strip()
                if h_val.startswith("$"):
                    h_val = os.environ.get(h_val[1:], "")
                if h_val:
                    default_headers[h_key.strip()] = h_val
            if default_headers:
                client_kwargs["default_headers"] = default_headers
            self._anthropic = anthropic.Anthropic(**client_kwargs)
        else:
            raise SystemExit(f"Unsupported provider: {provider}")

    def judge(self, messages: list[dict[str, str]]) -> str:
        if self.provider in {"openai", "truefoundry"}:
            assert self._openai is not None
            response = self._openai.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                messages=messages,
            )
            content = response.choices[0].message.content or ""
            return content

        assert self._anthropic is not None
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_messages = [m for m in messages if m["role"] != "system"]
        response = self._anthropic.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=user_messages,
        )
        parts = []
        for block in response.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts)


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_subjective(rows: list[dict], group_keys: list[str]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped[key].append(row)

    out = []
    for key in sorted(grouped):
        group = grouped[key]
        record = {k: v for k, v in zip(group_keys, key)}
        record["n"] = len(group)
        for dim in DIMENSION_KEYS:
            record[f"mean_{dim}"] = round(
                _mean([float(r[f"{dim}_score"]) for r in group]), 4
            )
        out.append(record)
    return out


def merge_final(
    subjective_rows: list[dict],
    objective_rows: list[dict],
) -> list[dict]:
    obj_index = {
        tuple(str(r[k]) for k in ("model",) + JOIN_KEYS): r for r in objective_rows
    }
    merged = []
    for subj in subjective_rows:
        key = tuple(str(subj[k]) for k in ("model",) + JOIN_KEYS)
        obj = obj_index.get(key, {})
        combined = dict(obj)
        combined.update(subj)
        # Prefer subjective copies of shared metadata if present.
        for field in ("model", "prompt_id", "generation_id", "seed", "target_phonics"):
            if field in subj:
                combined[field] = subj[field]
        merged.append(combined)
    return merged


def print_table(rows: list[dict], columns: list[tuple[str, str]], title: str) -> None:
    print(title)
    print("-" * len(title))
    if not rows:
        print("(no rows)")
        print()
        return
    labels = [label for _, label in columns]
    widths = []
    for key, label in columns:
        width = len(label)
        for row in rows:
            width = max(width, len(str(row.get(key, ""))))
        widths.append(width)

    def fmt(vals: list[str]) -> str:
        return "  ".join(str(v).ljust(w) for v, w in zip(vals, widths))

    print(fmt(labels))
    print(fmt(["-" * w for w in widths]))
    for row in rows:
        print(fmt([str(row.get(key, "")) for key, _ in columns]))
    print()


def interpret_results(
    obj_summary: list[dict],
    subj_summary: list[dict],
    by_prompt: list[dict],
) -> str:
    """Short automatic interpretation of tuned vs base."""
    obj = {r["model"]: r for r in obj_summary}
    subj = {r["model"]: r for r in subj_summary}
    if "base" not in obj or "tuned" not in obj or "base" not in subj or "tuned" not in subj:
        return "Could not compute interpretation because base/tuned summaries are incomplete."

    lines = ["Interpretation (estimated; judge is blind to model identity)"]
    lines.append("")

    # Objective improvements / regressions
    obj_metrics = [
        ("mean_decodability_compliance", "decodability compliance", True),
        ("mean_weighted_above_level_rate", "weighted above-level rate", False),
        ("mean_target_satisfaction_rate", "target satisfaction", True),
        ("full_spec_pass_rate", "full-spec pass rate", True),
        ("phonics_leakage_rate", "phonics leakage rate", False),
    ]
    obj_improved = []
    obj_regressed = []
    for key, label, higher_better in obj_metrics:
        b = float(obj["base"][key])
        t = float(obj["tuned"][key])
        delta = t - b
        better = delta > 0.01 if higher_better else delta < -0.01
        worse = delta < -0.01 if higher_better else delta > 0.01
        if better:
            obj_improved.append(f"{label} ({b:.3f} → {t:.3f})")
        elif worse:
            obj_regressed.append(f"{label} ({b:.3f} → {t:.3f})")

    lines.append("Where the tuned model improved (objective):")
    lines.extend([f"  - {x}" for x in obj_improved] or ["  - none clear"])
    lines.append("Where the tuned model regressed (objective):")
    lines.extend([f"  - {x}" for x in obj_regressed] or ["  - none clear"])
    lines.append("")

    # Subjective
    subj_improved = []
    subj_regressed = []
    for dim in DIMENSION_KEYS:
        key = f"mean_{dim}"
        b = float(subj["base"][key])
        t = float(subj["tuned"][key])
        delta = t - b
        label = dim.replace("_", " ")
        if delta > 0.05:
            subj_improved.append(f"{label} ({b:.2f} → {t:.2f})")
        elif delta < -0.05:
            subj_regressed.append(f"{label} ({b:.2f} → {t:.2f})")

    lines.append("Where the tuned model improved (subjective judge):")
    lines.extend([f"  - {x}" for x in subj_improved] or ["  - none clear"])
    lines.append("Where the tuned model regressed (subjective judge):")
    lines.extend([f"  - {x}" for x in subj_regressed] or ["  - none clear"])
    lines.append("")

    # Difficult prompt types: lowest tuned overall + decodability
    tuned_prompt = [r for r in by_prompt if r["model"] == "tuned"]
    if tuned_prompt:
        ranked = sorted(
            tuned_prompt,
            key=lambda r: (
                float(r["mean_overall_spec_adherence"]),
                float(r["mean_decodability_adherence"]),
                float(r["mean_target_phonics_pattern_emphasis"]),
                float(r["mean_narrative_completeness"]),
            ),
        )
        lines.append("Prompt types that remain most difficult for the tuned model:")
        for row in ranked[:3]:
            lines.append(
                "  - "
                f"{row['prompt_id']}: overall={float(row['mean_overall_spec_adherence']):.2f}, "
                f"decodability={float(row['mean_decodability_adherence']):.2f}, "
                f"target phonics pattern emphasis={float(row['mean_target_phonics_pattern_emphasis']):.2f}, "
                f"completeness={float(row['mean_narrative_completeness']):.2f}"
            )
    lines.append("")
    lines.append(
        "Reminder: subjective scores are LLM judgments of the story text only; "
        "objective decodability metrics are estimated rule-based spelling-pattern "
        "measures, not perfect linguistic decodability."
    )
    return "\n".join(lines)


def judge_all(
    rows: list[dict[str, str]],
    client: JudgeClient,
    sleep_s: float,
    max_retries: int,
    limit: int | None,
) -> list[dict]:
    results = []
    total = len(rows) if limit is None else min(limit, len(rows))
    for idx, row in enumerate(rows[:total], start=1):
        # Blind payload: never send model identity to the judge.
        target_phonics = row.get("target_phonics", "")
        prompt = row.get("prompt", "")
        story = row.get("story", "")
        messages = build_judge_messages(target_phonics, prompt, story)

        last_error = None
        scored = None
        raw_text = ""
        for attempt in range(1, max_retries + 1):
            try:
                raw_text = client.judge(messages)
                payload = extract_json_object(raw_text)
                scored = validate_scores(payload)
                break
            except Exception as exc:  # noqa: BLE001 - collect and retry transient failures
                last_error = exc
                wait = min(2 ** attempt, 20)
                print(
                    f"[{idx}/{total}] judge attempt {attempt}/{max_retries} failed: {exc}",
                    file=sys.stderr,
                )
                time.sleep(wait)

        if scored is None:
            raise RuntimeError(
                f"Judge failed for prompt_id={row.get('prompt_id')} "
                f"generation_id={row.get('generation_id')} seed={row.get('seed')}: {last_error}"
            )

        record = {
            "model": row["model"],  # retained only for local aggregation / merge
            "prompt_id": row["prompt_id"],
            "generation_id": row["generation_id"],
            "seed": row["seed"],
            "target_phonics": target_phonics,
            "judge_provider": client.provider,
            "judge_model": client.model,
            **{k: scored[k] for k in scored if k.endswith("_score") or k.endswith("_justification")},
            "raw_judge_json": scored["raw_judge_json"],
        }
        # Convenience total (not a separate judged dimension)
        record["subjective_total"] = sum(int(record[f"{k}_score"]) for k in DIMENSION_KEYS)
        results.append(record)
        print(
            f"[{idx}/{total}] scored prompt={row['prompt_id']} gen={row['generation_id']} "
            f"seed={row['seed']} total={record['subjective_total']}/{2 * len(DIMENSION_KEYS)} "
            f"(model hidden from judge)",
            flush=True,
        )
        if sleep_s > 0 and idx < total:
            time.sleep(sleep_s)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Blind LLM-as-a-judge evaluation for decodable SLM stories. "
            "The judge never receives the model column."
        )
    )
    parser.add_argument("--input", type=Path, default=INPUTS_DIR / "all_outputs.csv")
    parser.add_argument(
        "--objective-results",
        type=Path,
        default=OBJECTIVE_DIR / "objective_eval_results.csv",
        help="Row-level objective results to merge into final_evaluation.csv",
    )
    parser.add_argument(
        "--objective-summary",
        type=Path,
        default=OBJECTIVE_DIR / "objective_eval_summary.csv",
        help="Model-level objective summary to print",
    )
    parser.add_argument("--results", type=Path, default=SUBJECTIVE_DIR / "llm_judge_results.csv")
    parser.add_argument("--summary", type=Path, default=SUBJECTIVE_DIR / "llm_judge_summary.csv")
    parser.add_argument("--by-prompt", type=Path, default=SUBJECTIVE_DIR / "llm_judge_by_prompt.csv")
    parser.add_argument("--final", type=Path, default=SUBJECTIVE_DIR / "final_evaluation.csv")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "truefoundry"],
        default=None,
        help="API provider. Default: truefoundry if TFY_API_KEY is set, else openai, else anthropic.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Judge model id (provider-specific). Defaults depend on provider.",
    )
    parser.add_argument("--base-url", default=None, help="Optional custom API base URL")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--sleep", type=float, default=0.4, help="Pause between calls")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of stories (for smoke tests)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip stories already present in --results (matched on model+join keys).",
    )
    return parser


def resolve_provider_and_model(args: argparse.Namespace) -> tuple[str, str]:
    provider = args.provider
    if provider is None:
        if os.environ.get("TFY_API_KEY"):
            provider = "truefoundry"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        else:
            raise SystemExit(
                "No API credentials found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, "
                "or TFY_API_KEY (TrueFoundry)."
            )

    defaults = {
        "openai": "gpt-4.1",
        "anthropic": "claude-sonnet-4-6",
        "truefoundry": "openai-group/gpt-4.1",
    }
    model = args.model or defaults[provider]
    return provider, model


def main(argv: list[str] | None = None) -> int:
    load_dotenv_files()
    args = build_arg_parser().parse_args(argv)
    SUBJECTIVE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        provider, model = resolve_provider_and_model(args)
        client = JudgeClient(
            provider=provider,
            model=model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            base_url=args.base_url,
        )
        raw_rows = load_csv(args.input)
    except (OSError, ValueError, SystemExit) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    required = {
        "model",
        "prompt_id",
        "generation_id",
        "seed",
        "target_phonics",
        "prompt",
        "story",
    }
    missing = required - set(raw_rows[0].keys())
    if missing:
        print(f"ERROR: input CSV missing columns: {sorted(missing)}", file=sys.stderr)
        return 1

    existing: dict[tuple, dict] = {}
    if args.resume and args.results.exists():
        for row in load_csv(args.results):
            key = tuple(str(row[k]) for k in ("model",) + JOIN_KEYS)
            existing[key] = row
        print(f"Resume mode: loaded {len(existing)} existing judged rows")

    to_judge = []
    carried = []
    for row in raw_rows:
        key = tuple(str(row[k]) for k in ("model",) + JOIN_KEYS)
        if key in existing:
            carried.append(existing[key])
        else:
            to_judge.append(row)

    if args.limit is not None:
        to_judge = to_judge[: max(0, args.limit - len(carried))]

    print(
        f"Judge provider={provider} model={model} | "
        f"to_judge={len(to_judge)} carried={len(carried)} | "
        f"model identity withheld from judge"
    )

    try:
        judged = judge_all(
            to_judge,
            client=client,
            sleep_s=args.sleep,
            max_retries=args.max_retries,
            limit=None,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR during judging: {exc}", file=sys.stderr)
        return 1

    subjective_rows = carried + judged
    # Stable ordering by original input order
    order_index = {
        tuple(str(r[k]) for k in ("model",) + JOIN_KEYS): i
        for i, r in enumerate(raw_rows)
    }
    subjective_rows.sort(
        key=lambda r: order_index.get(
            tuple(str(r[k]) for k in ("model",) + JOIN_KEYS), 10**9
        )
    )

    result_fields = [
        "model",
        "prompt_id",
        "generation_id",
        "seed",
        "target_phonics",
        "judge_provider",
        "judge_model",
        "subjective_total",
    ]
    for dim in DIMENSION_KEYS:
        result_fields.extend([f"{dim}_score", f"{dim}_justification"])
    result_fields.append("raw_judge_json")

    summary_rows = summarize_subjective(subjective_rows, ["model"])
    by_prompt_rows = summarize_subjective(subjective_rows, ["model", "prompt_id"])

    summary_fields = ["model", "n"] + [f"mean_{d}" for d in DIMENSION_KEYS]
    by_prompt_fields = ["model", "prompt_id", "n"] + [
        f"mean_{d}" for d in DIMENSION_KEYS
    ]

    try:
        write_csv(args.results, subjective_rows, result_fields)
        write_csv(args.summary, summary_rows, summary_fields)
        write_csv(args.by_prompt, by_prompt_rows, by_prompt_fields)

        objective_rows = load_csv(args.objective_results)
        final_rows = merge_final(subjective_rows, objective_rows)
        # Put identifiers + key objective/subjective metrics first.
        final_front = [
            "model",
            "prompt_id",
            "generation_id",
            "seed",
            "target_phonics",
            "decodability_compliance",
            "weighted_above_level_rate",
            "target_satisfaction_rate",
            "full_spec_pass",
            "phonics_leakage",
            "word_count",
            "subjective_total",
        ]
        for dim in DIMENSION_KEYS:
            final_front.append(f"{dim}_score")
        final_extra = [c for c in final_rows[0].keys() if c not in final_front]
        write_csv(args.final, final_rows, final_front + final_extra)
    except (OSError, ValueError) as exc:
        print(f"ERROR writing outputs: {exc}", file=sys.stderr)
        return 1

    # Load objective summary for printing if available.
    try:
        obj_summary = load_csv(args.objective_summary)
    except (OSError, ValueError):
        obj_summary = []

    print()
    print_table(
        obj_summary,
        [
            ("model", "model"),
            ("n", "n"),
            ("mean_decodability_compliance", "decod_comp"),
            ("mean_weighted_above_level_rate", "wt_above"),
            ("full_spec_pass_rate", "full_spec"),
            ("mean_target_satisfaction_rate", "tgt_sat"),
            ("phonics_leakage_rate", "leakage"),
            ("mean_word_count", "words"),
        ],
        "1) Objective summary (estimated rule-based measures)",
    )

    print_table(
        summary_rows,
        [("model", "model"), ("n", "n")]
        + [(f"mean_{d}", d.replace("_", " ")[:18]) for d in DIMENSION_KEYS],
        "2) Subjective summary (blind LLM judge, 0–2 means)",
    )

    # Combined summary: side-by-side key metrics
    combined = []
    subj_map = {r["model"]: r for r in summary_rows}
    obj_map = {r["model"]: r for r in obj_summary}
    for model in sorted(set(subj_map) | set(obj_map)):
        o = obj_map.get(model, {})
        s = subj_map.get(model, {})
        combined.append(
            {
                "model": model,
                "decod_comp": o.get("mean_decodability_compliance", ""),
                "wt_above": o.get("mean_weighted_above_level_rate", ""),
                "tgt_sat": o.get("mean_target_satisfaction_rate", ""),
                "overall": s.get("mean_overall_spec_adherence", ""),
                "decod_subj": s.get("mean_decodability_adherence", ""),
                "emphasis": s.get("mean_target_phonics_pattern_emphasis", ""),
                "coherence": s.get("mean_coherence", ""),
                "complete": s.get("mean_narrative_completeness", ""),
            }
        )
    print_table(
        combined,
        [
            ("model", "model"),
            ("decod_comp", "obj_decod"),
            ("wt_above", "obj_wt_above"),
            ("tgt_sat", "obj_tgt_sat"),
            ("overall", "subj_overall"),
            ("decod_subj", "subj_decod"),
            ("emphasis", "subj_tgt_phon_emph"),
            ("coherence", "subj_cohere"),
            ("complete", "subj_complete"),
        ],
        "3) Combined summary",
    )

    print(interpret_results(obj_summary, summary_rows, by_prompt_rows))
    print()
    print(f"Saved: {args.results}")
    print(f"Saved: {args.summary}")
    print(f"Saved: {args.by_prompt}")
    print(f"Saved: {args.final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
