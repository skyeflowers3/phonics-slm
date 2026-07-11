import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

try:
    import cmudict
    CMU = cmudict.dict()
except Exception:
    CMU = {}

# -----------------------------
# EDITABLE PHONICS DEFINITIONS
# -----------------------------
DIGRAPHS = ("sh", "ch", "th", "wh", "ph", "ck", "ng")

INITIAL_BLENDS = (
    "bl", "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "pl", "pr", "sc", "sk", "sl", "sm", "sn", "sp", "st", "sw", "tr", "tw",
    "scr", "shr", "spl", "spr", "squ", "str", "thr"
)
FINAL_BLENDS = (
    "ft", "ld", "lf", "lk", "lp", "lt", "mp", "nd", "nt", "pt", "sk", "sp", "st", "ct"
)

VOWEL_TEAMS = ("ai", "ay", "ee", "ea", "oa", "oe", "ie", "igh", "ue", "ui", "oo", "ew")
DIPHTHONGS = ("oi", "oy", "ou", "ow", "au", "aw")
R_CONTROLLED = ("ar", "er", "ir", "or", "ur")
L_CONTROLLED = ("all", "alk", "al")

# Words that look like a regular pattern but are commonly irregular.
# Keep this list visible and editable rather than silently forcing a label.
IRREGULAR_WORDS = {
    "a", "again", "against", "any", "are", "because", "been", "both", "buy", "come", "could", "do", "does", "done",
    "eye", "friend", "from", "give", "gone", "good", "great", "have", "here", "house", "into", "live", "love", "many",
    "move", "of", "one", "once", "only", "other", "our", "people", "put", "said", "says", "school", "should", "some",
    "the", "their", "there", "these", "they", "though", "thought", "through", "to", "two", "very", "walk", "want", "was",
    "water", "were", "what", "where", "who", "why", "would", "you", "your"
}

# Common OW spellings pronounced as long-o rather than a diphthong.
OW_LONG_O = {
    "blow", "bowl", "crow", "flow", "glow", "grow", "know", "low", "mow", "own", "row", "show", "slow", "snow", "throw",
    "window", "yellow"
}

# Common final-e exceptions that should be reviewed rather than counted as regular magic-e.
FINAL_E_EXCEPTIONS = {
    "are", "come", "done", "give", "gone", "have", "live", "love", "move", "one", "some", "there", "these", "were"
}


# -----------------------------
# TARGET-PHONICS SELECTION
# -----------------------------
MIN_DISTINCT_TARGET_WORDS = 4
MAX_TARGET_PHONICS = 3

TARGET_DISPLAY_NAMES = {
    "digraph": "consonant digraphs",
    "blend": "consonant blends",
    "final_e": "final-e words",
    "vowel_team": "vowel teams",
    "r_controlled": "r-controlled vowels",
    "l_controlled": "l-controlled vowels",
    "diphthong": "diphthongs",
    "multisyllabic": "multisyllabic words",
}

TARGET_TIE_PRIORITY = {
    "multisyllabic": 8,
    "diphthong": 7,
    "r_controlled": 6,
    "l_controlled": 6,
    "vowel_team": 5,
    "final_e": 4,
    "digraph": 3,
    "blend": 3,
}

def select_target_phonics(profile):
    candidates = []
    for internal_name, display_name in TARGET_DISPLAY_NAMES.items():
        words_text = profile.get(f"{internal_name}_words", "") or ""
        distinct_words = {
            word.strip()
            for word in words_text.split(",")
            if word.strip()
        }
        count = len(distinct_words)
        if count >= MIN_DISTINCT_TARGET_WORDS:
            candidates.append(
                (display_name, count, TARGET_TIE_PRIORITY[internal_name])
            )

    candidates.sort(key=lambda item: (-item[1], -item[2], item[0]))
    selected = [name for name, _, _ in candidates[:MAX_TARGET_PHONICS]]
    return ", ".join(selected) if selected else "short-vowel words"

WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
VOWELS = set("aeiouy")


def normalize_word(word: str) -> str:
    word = word.lower().replace("’", "'")
    # Reduce possessives/contractions to the base form when possible.
    if word.endswith("'s") and len(word) > 2:
        word = word[:-2]
    elif word.endswith("n't") and len(word) > 3:
        word = word[:-3]
    elif "'" in word:
        word = word.split("'", 1)[0]
    return word


def extract_words(text: str):
    return [normalize_word(w) for w in WORD_RE.findall(text or "") if normalize_word(w)]


def syllable_count(word: str) -> int:
    """Use CMU pronunciations when available; otherwise use a conservative spelling heuristic."""
    pronunciations = CMU.get(word)
    if pronunciations:
        counts = []
        for pronunciation in pronunciations:
            counts.append(sum(1 for phoneme in pronunciation if phoneme[-1:].isdigit()))
        return min(c for c in counts if c > 0) if any(c > 0 for c in counts) else 1

    w = re.sub(r"[^a-z]", "", word.lower())
    if len(w) <= 3:
        return 1
    groups = re.findall(r"[aeiouy]+", w)
    count = len(groups)
    if w.endswith("e") and not w.endswith(("le", "ye")) and count > 1:
        count -= 1
    if w.endswith("le") and len(w) > 2 and w[-3] not in VOWELS:
        count += 1
    return max(1, count)


def has_magic_e(word: str) -> bool:
    if word in FINAL_E_EXCEPTIONS or len(word) < 4 or not word.endswith("e"):
        return False
    stem = word[:-1]
    # Final e after a consonant, with a vowel earlier in the final syllable-like spelling.
    return bool(re.search(r"[aeiou][^aeiou]$", stem))


def classify_word(word: str):
    patterns = []
    notes = []

    syllables = syllable_count(word)
    if syllables >= 2:
        patterns.append("multisyllabic")

    is_irregular = word in IRREGULAR_WORDS
    if is_irregular:
        patterns.append("irregular")

    # Detect advanced spelling patterns independently. Irregular/heart words are
    # kept separate so words such as "the" do not falsely count as digraph practice.
    diph_matches = [] if is_irregular else [p for p in DIPHTHONGS if p in word]
    if "ow" in diph_matches and word in OW_LONG_O:
        diph_matches.remove("ow")
        notes.append("ow may represent long-o")
    if diph_matches:
        patterns.append("diphthong")

    if not is_irregular and any(p in word for p in R_CONTROLLED):
        patterns.append("r_controlled")

    if not is_irregular and any(p in word for p in L_CONTROLLED):
        patterns.append("l_controlled")

    if not is_irregular and any(p in word for p in VOWEL_TEAMS):
        patterns.append("vowel_team")

    if not is_irregular and has_magic_e(word):
        patterns.append("final_e")

    starts_with_blend = (not is_irregular) and any(word.startswith(p) for p in INITIAL_BLENDS)
    ends_with_blend = (not is_irregular) and any(word.endswith(p) for p in FINAL_BLENDS)
    if starts_with_blend or ends_with_blend:
        patterns.append("blend")

    if not is_irregular and any(p in word for p in DIGRAPHS):
        patterns.append("digraph")

    # A simple/basic bucket means no Level 3-8 target pattern was found.
    advanced = {"digraph", "blend", "final_e", "vowel_team", "r_controlled", "l_controlled", "diphthong", "multisyllabic"}
    if not advanced.intersection(patterns):
        patterns.append("basic_or_unclassified")

    # Flag words that deserve manual attention.
    if len(set(patterns).intersection(advanced)) >= 2:
        notes.append("contains multiple target patterns")
    if word in FINAL_E_EXCEPTIONS:
        notes.append("final-e spelling exception")
    if word not in CMU:
        notes.append("not found in CMU pronunciation dictionary")

    return {
        "word": word,
        "syllables": syllables,
        "patterns": sorted(set(patterns)),
        "notes": sorted(set(notes)),
    }


def profile_story(text: str):
    tokens = extract_words(text)
    token_counts = Counter(tokens)
    unique_words = sorted(token_counts)

    pattern_token_counts = Counter()
    pattern_words = defaultdict(list)
    flagged = []

    for word in unique_words:
        result = classify_word(word)
        frequency = token_counts[word]
        for pattern in result["patterns"]:
            pattern_token_counts[pattern] += frequency
            pattern_words[pattern].append(word)
        if result["notes"]:
            flagged.append({
                "word": word,
                "count": frequency,
                "patterns": result["patterns"],
                "notes": result["notes"],
            })

    total = len(tokens) or 1
    target_patterns = [
        "digraph", "blend", "final_e", "vowel_team", "r_controlled", "l_controlled", "diphthong", "multisyllabic", "irregular"
    ]

    profile = {
        "analyzed_word_count": len(tokens),
        "unique_word_count": len(unique_words),
        "pattern_token_counts": dict(pattern_token_counts),
        "pattern_words": {k: v for k, v in sorted(pattern_words.items())},
        "flagged_words": flagged,
    }

    flat = {
        "analyzed_word_count": len(tokens),
        "unique_word_count": len(unique_words),
    }
    for pattern in target_patterns:
        count = pattern_token_counts.get(pattern, 0)
        flat[f"{pattern}_count"] = count
        flat[f"{pattern}_pct"] = round(100 * count / total, 1)
        flat[f"{pattern}_words"] = ", ".join(pattern_words.get(pattern, []))

    # Evaluate the existing label without pretending one accidental word determines the story level.
    level_to_targets = {
        2: set(),
        3: {"digraph", "blend"},
        4: {"final_e"},
        5: {"vowel_team"},
        6: {"r_controlled", "l_controlled"},
        7: {"diphthong"},
        8: {"multisyllabic"},
    }
    pattern_level = {
        "digraph": 3, "blend": 3, "final_e": 4, "vowel_team": 5,
        "r_controlled": 6, "l_controlled": 6, "diphthong": 7, "multisyllabic": 8,
    }

    flat["present_patterns"] = ", ".join(
        p for p in pattern_level if pattern_token_counts.get(p, 0) > 0
    )

    # assigned_level is passed in later by profile_csv; placeholders are filled there.
    flat["target_pattern_count"] = ""
    flat["target_pattern_pct"] = ""
    flat["above_level_count"] = ""
    flat["above_level_pct"] = ""
    flat["review_required"] = ""
    flat["review_reason"] = ""
    flat["flagged_words_json"] = json.dumps(flagged, ensure_ascii=False)
    flat["phonics_profile_json"] = json.dumps(profile, ensure_ascii=False)
    return flat


def profile_csv(input_path: str, output_path: str):
    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        original_fields = reader.fieldnames or []

    profiled_rows = []
    new_fields = None
    for row in rows:
        profile = profile_story(row.get("story", ""))

        try:
            assigned_level = int(float(row.get("menon_hiebert_level", "")))
        except (TypeError, ValueError):
            assigned_level = None

        pattern_level = {
            "digraph": 3, "blend": 3, "final_e": 4, "vowel_team": 5,
            "r_controlled": 6, "l_controlled": 6, "diphthong": 7, "multisyllabic": 8,
        }
        level_targets = {
            2: set(), 3: {"digraph", "blend"}, 4: {"final_e"}, 5: {"vowel_team"},
            6: {"r_controlled", "l_controlled"}, 7: {"diphthong"}, 8: {"multisyllabic"},
        }
        total = int(profile.get("analyzed_word_count", 0)) or 1
        if assigned_level in level_targets:
            targets = level_targets[assigned_level]
            target_count = sum(int(profile.get(f"{p}_count", 0)) for p in targets)
            above_count = sum(
                int(profile.get(f"{p}_count", 0))
                for p, lev in pattern_level.items() if lev > assigned_level
            )
            reasons = []
            if assigned_level >= 3 and target_count < 2:
                reasons.append("fewer than 2 target-pattern tokens")
            if above_count / total > 0.05:
                reasons.append("more than 5% above-level pattern tokens")
            profile["target_pattern_count"] = target_count
            profile["target_pattern_pct"] = round(100 * target_count / total, 1)
            profile["above_level_count"] = above_count
            profile["above_level_pct"] = round(100 * above_count / total, 1)
            profile["review_required"] = "YES" if reasons else "NO"
            profile["review_reason"] = "; ".join(reasons)
        else:
            profile["review_required"] = "YES"
            profile["review_reason"] = "missing or unsupported assigned level"

        profile["target_phonics"] = select_target_phonics(profile)
        if profile["target_phonics"] == "short-vowel words":
            profile["present_phonics_summary"] = "short-vowel words"

        combined = dict(row)
        combined.update(profile)
        profiled_rows.append(combined)
        if new_fields is None:
            new_fields = list(profile.keys())

    fields = original_fields + [f for f in (new_fields or []) if f not in original_fields]
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(profiled_rows)

    print(f"Profiled {len(profiled_rows)} stories")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    profile_csv(
        "/mnt/data/cl_profiled_df.csv",
        "/mnt/data/cl_profiled_df_threshold4.csv",
    )
