import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUTS_DIR = REPO_ROOT / "inputs"

PHONICS_EXAMPLES = {
    "short-vowel words": ["cat", "sit", "red", "hop"],
    "consonant blends": ["stop", "frog", "black", "plant"],
    "consonant digraphs": ["ship", "fish", "that", "chop"],
    "final-e words": ["cake", "bike", "home", "smile"],
    "vowel teams": ["rain", "seed", "boat", "play"],
    "r-controlled vowels": ["farm", "bird", "corn", "turn"],
    "diphthongs": ["coin", "out", "toy", "cloud"],
    "multisyllabic words": ["rabbit", "family", "dinosaur", "computer"],
}


def build_prompt(target_phonics):
    patterns = "\n".join(
        f"- {pattern}"
        for pattern in target_phonics
    )

    examples = "\n".join(
        f"- {pattern}: {', '.join(PHONICS_EXAMPLES[pattern])}"
        for pattern in target_phonics
    )

    return (
        "Write a decodable story.\n\n"
        "Emphasize these phonics patterns:\n"
        f"{patterns}\n\n"
        "Here are some example words containing these patterns:\n"
        f"{examples}\n\n"
        "Use the example words only as guidance. You may use other words "
        "that contain the same patterns.\n"
        "Write a coherent narrative with a clear beginning, middle, and ending.\n"
        "Include a natural story title.\n"
        "Do not mention phonics terms or pattern names in the title or story."
    )


with (INPUTS_DIR / "eval_prompts.json").open("r", encoding="utf-8") as f:
    eval_cases = json.load(f)

for case in eval_cases:
    case["prompt"] = build_prompt(case["target_phonics"])

out_path = INPUTS_DIR / "eval_prompts_ready.json"
with out_path.open("w", encoding="utf-8") as f:
    json.dump(eval_cases, f, indent=2, ensure_ascii=False)

print(f"Created {len(eval_cases)} evaluation prompts -> {out_path}")
print()
print(eval_cases[0]["prompt"])
