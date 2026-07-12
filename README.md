# Phonics SLM Evaluation

Base vs fine-tuned small language model evaluation for **decodable phonics stories**.

Given 1–3 target phonics patterns, each model generates a short story that should stay at the requested decodability level, emphasize the target patterns, and form a complete narrative.

## Repo layout

| Path | Contents |
|------|----------|
| `data/` | Train / validation / test splits used to fine-tune the SLM |
| `inputs/` | Eval prompts and base/tuned generations (`all_outputs.csv`) |
| `scripts/` | Objective eval, blind LLM judge, and error analysis |
| `results/` | Saved evaluation outputs |

## How to read the results

Start here:

1. **`results/error_analysis/error_analysis_summary.md`** — narrative write-up of gains, failures, and hard prompt types
2. **`results/objective/objective_eval_summary.csv`** — rule-based phonics / decodability metrics by model
3. **`results/subjective/llm_judge_summary.csv`** — blind LLM-as-a-judge scores by model (0–2 scale)

Supporting detail:

- `results/objective/` — row-level, by-prompt, and paired base-vs-tuned CSVs
- `results/subjective/` — per-story judge scores, by-prompt means, and `final_evaluation.csv` (objective + subjective merged)
- `results/error_analysis/error_analysis_examples.csv` — example failures cited in the summary

## Evaluation design

- **Objective** (`scripts/evaluate_outputs.py`): estimated rule-based spelling-pattern metrics. Primary: decodability compliance / above-level rate. Secondary: target satisfaction with partial credit (`min(distinct_matches / 4, 1)` per requested pattern, then averaged). Also tracks phonics-term leakage. Not a perfect measure of linguistic decodability.
- **Subjective** (`scripts/llm_judge.py`): blind LLM judge. The judge never sees the `model` column; it scores story text against the prompt/target phonics only.
- **Error analysis** (`scripts/error_analysis.py`): combines both to surface where the tuned model improved, where it still fails, and which prompt conditions remain hard.

## Reproducing (optional)

```bash
pip install -r requirements.txt

# After all_outputs.csv exists: objective checks + new LLM-judge rubric
python scripts/run_post_generation_eval.py

# Objective only (no API key)
python scripts/run_post_generation_eval.py --objective-only
```

Outputs:

- `results/scored_outputs.csv`
- `results/evaluation_summary.csv`
- `results/error_analysis/error_analysis.csv`

Legacy scripts (`evaluate_outputs.py`, `llm_judge.py`, `error_analysis.py`) remain available under `scripts/` and write to `results/objective/` / `results/subjective/`.
