# Phonics SLM Evaluation

Base vs fine-tuned small language model evaluation for **decodable phonics stories**.

Given 1–3 target phonics patterns, each model generates a short story that should stay at the requested decodability level so students can read and learn from it, emphasize the target patterns, and form a complete narrative.

## Repo layout

| Path | Contents |
|------|----------|
| `data/` | Train / validation / test splits used to fine-tune the SLM |
| `inputs/` | Eval prompts and base/tuned generations (`all_outputs.csv`) |
| `scripts/` | Post-generation evaluation (`run_post_generation_eval.py`) |
| `results/` | Saved evaluation outputs |

## How to read the results

Canonical run (all from the same scoring pass):

1. **`results/evaluation_summary.csv`** — complete base/tuned means + deltas (objective + judge)
2. **`results/objective/objective_eval_summary.csv`** — objective-only wide summary (same numbers)
3. **`results/subjective/llm_judge_summary.csv`** — judge-only wide summary (same numbers)
4. **`results/error_analysis/error_analysis.csv`** — common tuned failures + examples

Also:

- `results/scored_outputs.csv` — per-story objective + judge scores
- `results/objective/` — row-level / by-prompt objective views
- `results/subjective/` — row-level / by-prompt judge views + `final_evaluation.csv`

## Evaluation design

Primary behavior goal: **decodable text students at the requested level can read and learn from.**

- **Objective:** rule-based phonics metrics (`decodability_compliance`, `above_level_rate`, `weighted_above_level_rate`, `target_phonics_coverage`, leakage, title, completeness, duplicates). Not a perfect linguistic decodability measure.
- **Subjective (blind LLM judge):** `spec_adherence`, `robustness`, `task_quality`, plus prompt-level `consistency` across the three matched generations. Spec adherence prioritizes in-level readable decodable text.

## Reproducing

```bash
pip install -r requirements.txt

# After all_outputs.csv exists
python scripts/run_post_generation_eval.py

# Objective only (no API key)
python scripts/run_post_generation_eval.py --objective-only
```
