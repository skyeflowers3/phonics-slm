# Phonics SLM Evaluation

Base vs fine-tuned small language model evaluation for **decodable phonics stories**.

Given 1–3 target phonics patterns, each model generates a short story that should stay at the requested decodability level so students can read and learn from it, emphasize the target patterns, and form a complete narrative.

## Model, dataset, and inference

- **Model:** [skyeflo/qwen3-decodable-story-sft](https://huggingface.co/skyeflo/qwen3-decodable-story-sft)
- **Dataset:** [skyeflo/decodable-story-dataset](https://huggingface.co/datasets/skyeflo/decodable-story-dataset)
- **Run inference:** Open this [Colab notebook](https://colab.research.google.com/drive/18j2d5T1Te0vRFnE1bCYtHKEI3QBvZWCk?usp=sharing), click **Run all**, scroll to the bottom, and click the Gradio link the cell produces.

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

- **Objective (primary):** rule-based phonics metrics (`decodability_compliance`, `above_level_rate`, `weighted_above_level_rate`, leakage, title, completeness, duplicates). Not a perfect linguistic decodability measure.
- **Objective (secondary diagnostic):** `target_phonics_coverage` — reported with a caveat that advanced/above-level stories can inflate it; not a primary success metric.
- **Subjective (blind LLM judge):** `spec_adherence`, `robustness`, `task_quality`, plus prompt-level `consistency` across the three matched generations. Spec adherence prioritizes in-level readable decodable text.

## Reproducing

```bash
pip install -r requirements.txt

# After all_outputs.csv exists
python scripts/run_post_generation_eval.py

# Objective only (no API key)
python scripts/run_post_generation_eval.py --objective-only
```
