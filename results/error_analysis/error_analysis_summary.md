# Error Analysis Summary

This report is derived only from observed objective and subjective evaluation outputs. Likely-cause labels are hypotheses constrained by those metrics; they are **not** proof that every tuned-model failure is a training-data problem.

## Method notes

- Low subjective score threshold: `0` on the 0–2 judge scale.
- High above-level rate threshold: `>= 0.20` (tuned median is lower; this flags clearly elevated rates).
- Paired comparisons use `compliance_delta = tuned - base` decodability compliance.
- In this run, all paired compliance deltas are ≥ 0, so “worst” pairs are least-improved rather than true regressions on that metric.

## 1. Largest paired decodability-compliance gains

| rank | prompt_id | gen | seed | base | tuned | delta | tuned words | likely cause |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `short_vowels_1` | 2 | 1002 | 0.3869 | 0.7692 | +0.3823 | 65 | training-data coverage |
| 2 | `blends_digraphs_1` | 1 | 1001 | 0.4977 | 0.8571 | +0.3594 | 28 | generation behavior |
| 3 | `short_vowels_1` | 3 | 1003 | 0.4313 | 0.7708 | +0.3395 | 48 | training-data coverage |
| 4 | `blends_digraphs_1` | 3 | 1003 | 0.5024 | 0.8207 | +0.3183 | 184 | generation behavior |
| 5 | `final_e_blends_1` | 2 | 1002 | 0.5659 | 0.8696 | +0.3037 | 46 | conflicting training objectives |

### Representative improvement

- **Pair:** `short_vowels_1` / gen 2 / seed 1002
- **Metrics:** compliance 0.387 → 0.769 (Δ +0.382); above-level rate 0.613 → 0.231; words 199 → 65.
- **Likely cause:** training-data coverage

**Tuned output**

```text
Title: The Cat and the Dog

The dog was playing when he saw the cat. He jumped up on his bed to see if he could catch it. “What do you think?” asked the cat. “I like this spot!” said the dog. “You can sit on my lap,” said the cat. Then they both sat in a big pile of straw! They were so happy.
```

**Matched base output**

```text
**The Happy Forest of Squirrels**

In the heart of the forest lived a small group of squirrels. They had long been friends, sharing their favorite places—honeycombs, nests, and trees. The main tree was called Maple Tree, which had many holes filled with seeds from all the fruits grown by the squirrels. 

One day, a new squirrel came to join them. She was named Sparky. She loved singing songs, playing games, and exploring new places. Soon after, more squirrels joined the group, each bringing their own unique skills and ideas. There were no longer just the usual ones—they were all part of a big family.

One morning, the group went on a journey. They traveled through different kinds of woods, meeting birds, animals, and even a river. Along the way, they learned about nature and made new frie…
```

## 2. Least paired improvement (or no gain)

| rank | prompt_id | gen | seed | base | tuned | delta | note | likely cause |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | `diphthongs_digraphs_1` | 2 | 1002 | 0.8294 | 0.8205 | -0.0089 | Smallest paired compliance_delta. Tuned absolute compliance remains relatively weak. | conflicting training objectives |
| 2 | `multisyllabic_vowel_teams_r_controlled_1` | 1 | 1001 | 1.0000 | 1.0000 | +0.0000 | Smallest paired compliance_delta. Ceiling effect: both models already near-perfect on this metric. | rule-based evaluation limitation |
| 3 | `multisyllabic_vowel_teams_r_controlled_1` | 2 | 1002 | 1.0000 | 1.0000 | +0.0000 | Smallest paired compliance_delta. Ceiling effect: both models already near-perfect on this metric. | rule-based evaluation limitation |
| 4 | `multisyllabic_vowel_teams_r_controlled_1` | 3 | 1003 | 1.0000 | 1.0000 | +0.0000 | Smallest paired compliance_delta. Ceiling effect: both models already near-perfect on this metric. | rule-based evaluation limitation |
| 5 | `diphthongs_digraphs_1` | 1 | 1001 | 0.7981 | 0.8163 | +0.0182 | Smallest paired compliance_delta. Tuned absolute compliance remains relatively weak. | generation behavior |

### Representative weak/least-gain case

- **Pair:** `diphthongs_digraphs_1` / gen 2 / seed 1002
- **Notes:** Smallest paired compliance_delta. Tuned absolute compliance remains relatively weak.
- **Metrics:** compliance 0.829 → 0.821 (Δ -0.009); target satisfaction 1.00 → 0.50; subj decod 0, emphasis 0.
- **Likely cause:** conflicting training objectives

**Tuned output**

```text
Title: The Fun Game

Sara and Sam have fun playing games together.  
They play ball and they play hopscotch too.  
Sam has two cups of lemonade to drink.  
The game ends when the kids say “No!” from the table.
```

**Matched base output**

```text
**The Sunny Farm and Its Friends**

One day, there was a new friend named Mr. Piggy who came to live on the farm. The old farmer had a garden full of flowers, but now Mr. Pig was growing vegetables from seeds. He brought in a big pot of tomatoes, carrots, and a bunch of strawberries.

The farmers were all happy to see him come. They planted more crops together and made new toys with their own hands. One day, they found a special toy called "the Fish." It was a small boat, and it could go far. The water was blue.

Then one morning, the group decided to play games. They played hopscotch, chased each other around the farm, and even built a new bridge. They learned how to move faster than the others.

After a long time, the friends left the farm for another day. They went home, and Mr. Pig st…
```

## 3. Tuned failure-mode inventory

| failure mode | n flagged | share of tuned | representative prompt | likely cause |
|---|---:|---:|---|---|
| `low_decodability_adherence` | 8 | 38.1% | `blends_digraphs_1` | generation behavior |
| `low_target_phonics_emphasis` | 8 | 38.1% | `blends_digraphs_1` | generation behavior |
| `low_coherence` | 5 | 23.8% | `blends_digraphs_1` | generation behavior |
| `incomplete_narrative` | 5 | 23.8% | `blends_digraphs_1` | generation behavior |
| `high_above_level` | 5 | 23.8% | `blends_digraphs_1` | generation behavior |

### Failure mode: `low_decodability_adherence`

- **Definition:** Tuned subjective decodability_adherence_score == 0.
- **Support metrics:** compliance=0.657, above_level_rate=0.343, weighted_above=1.114, target_sat=0.88, subj decod=0, emphasis=0, coherence=0, completeness=0, words=35.
- **Matched base metrics:** compliance=0.567, above_level_rate=0.433, target_sat=1.00, words=171.
- **Likely cause:** generation behavior
- **Judge note (decodability):** The story uses above-level words like 'together', 'happening', and 'jump', which are not limited to consonant blends and digraphs.
- **Judge note (target phonics pattern emphasis):** There is almost no use of consonant blends or digraphs, with only 'grass' and 'jump' as possible examples, and these are not emphasized.

**Representative tuned output**

```text
Title: The Fun Game

Sara and Sam have fun playing games together.  
They play ball and jump rope.  
A big game is happening at the park!  
We run to get on the grass for fun!
```

**Matched base output**

```text
**The Forest and the Lizard**

One day, a little fox found a small tree full of shiny leaves. The leaves were green, but they had a strange shape—like the letters *stop*. The fox was curious about them and thought it might be a kind of animal. 

Then one morning, a big lizard came to visit the forest. He was tall, covered in long fur, and his eyes were bright blue. He noticed that there were many plants with big ears and small heads. Those plants were different from the ones on the leaves. They looked like *ship*, which made a sound when moving.

The next day, the forest began to grow more beautiful. Trees bloomed, flowers grew, and a new creature appeared—the owl. The owl said something sweet and magical to everyone in the forest. And so the forest became home for all things.

Now, if yo…
```

### Failure mode: `low_target_phonics_emphasis`

- **Definition:** Tuned target_phonics_pattern_emphasis_score == 0.
- **Support metrics:** compliance=0.657, above_level_rate=0.343, weighted_above=1.114, target_sat=0.88, subj decod=0, emphasis=0, coherence=0, completeness=0, words=35.
- **Matched base metrics:** compliance=0.567, above_level_rate=0.433, target_sat=1.00, words=171.
- **Likely cause:** generation behavior
- **Judge note (decodability):** The story uses above-level words like 'together', 'happening', and 'jump', which are not limited to consonant blends and digraphs.
- **Judge note (target phonics pattern emphasis):** There is almost no use of consonant blends or digraphs, with only 'grass' and 'jump' as possible examples, and these are not emphasized.

**Representative tuned output**

```text
Title: The Fun Game

Sara and Sam have fun playing games together.  
They play ball and jump rope.  
A big game is happening at the park!  
We run to get on the grass for fun!
```

**Matched base output**

```text
**The Forest and the Lizard**

One day, a little fox found a small tree full of shiny leaves. The leaves were green, but they had a strange shape—like the letters *stop*. The fox was curious about them and thought it might be a kind of animal. 

Then one morning, a big lizard came to visit the forest. He was tall, covered in long fur, and his eyes were bright blue. He noticed that there were many plants with big ears and small heads. Those plants were different from the ones on the leaves. They looked like *ship*, which made a sound when moving.

The next day, the forest began to grow more beautiful. Trees bloomed, flowers grew, and a new creature appeared—the owl. The owl said something sweet and magical to everyone in the forest. And so the forest became home for all things.

Now, if yo…
```

### Failure mode: `low_coherence`

- **Definition:** Tuned coherence_score == 0.
- **Support metrics:** compliance=0.657, above_level_rate=0.343, weighted_above=1.114, target_sat=0.88, subj decod=0, emphasis=0, coherence=0, completeness=0, words=35.
- **Matched base metrics:** compliance=0.567, above_level_rate=0.433, target_sat=1.00, words=171.
- **Likely cause:** generation behavior
- **Judge note (decodability):** The story uses above-level words like 'together', 'happening', and 'jump', which are not limited to consonant blends and digraphs.
- **Judge note (target phonics pattern emphasis):** There is almost no use of consonant blends or digraphs, with only 'grass' and 'jump' as possible examples, and these are not emphasized.

**Representative tuned output**

```text
Title: The Fun Game

Sara and Sam have fun playing games together.  
They play ball and jump rope.  
A big game is happening at the park!  
We run to get on the grass for fun!
```

**Matched base output**

```text
**The Forest and the Lizard**

One day, a little fox found a small tree full of shiny leaves. The leaves were green, but they had a strange shape—like the letters *stop*. The fox was curious about them and thought it might be a kind of animal. 

Then one morning, a big lizard came to visit the forest. He was tall, covered in long fur, and his eyes were bright blue. He noticed that there were many plants with big ears and small heads. Those plants were different from the ones on the leaves. They looked like *ship*, which made a sound when moving.

The next day, the forest began to grow more beautiful. Trees bloomed, flowers grew, and a new creature appeared—the owl. The owl said something sweet and magical to everyone in the forest. And so the forest became home for all things.

Now, if yo…
```

### Failure mode: `incomplete_narrative`

- **Definition:** Tuned narrative_completeness_score == 0.
- **Support metrics:** compliance=0.657, above_level_rate=0.343, weighted_above=1.114, target_sat=0.88, subj decod=0, emphasis=0, coherence=0, completeness=0, words=35.
- **Matched base metrics:** compliance=0.567, above_level_rate=0.433, target_sat=1.00, words=171.
- **Likely cause:** generation behavior
- **Judge note (decodability):** The story uses above-level words like 'together', 'happening', and 'jump', which are not limited to consonant blends and digraphs.
- **Judge note (target phonics pattern emphasis):** There is almost no use of consonant blends or digraphs, with only 'grass' and 'jump' as possible examples, and these are not emphasized.

**Representative tuned output**

```text
Title: The Fun Game

Sara and Sam have fun playing games together.  
They play ball and jump rope.  
A big game is happening at the park!  
We run to get on the grass for fun!
```

**Matched base output**

```text
**The Forest and the Lizard**

One day, a little fox found a small tree full of shiny leaves. The leaves were green, but they had a strange shape—like the letters *stop*. The fox was curious about them and thought it might be a kind of animal. 

Then one morning, a big lizard came to visit the forest. He was tall, covered in long fur, and his eyes were bright blue. He noticed that there were many plants with big ears and small heads. Those plants were different from the ones on the leaves. They looked like *ship*, which made a sound when moving.

The next day, the forest began to grow more beautiful. Trees bloomed, flowers grew, and a new creature appeared—the owl. The owl said something sweet and magical to everyone in the forest. And so the forest became home for all things.

Now, if yo…
```

### Failure mode: `high_above_level`

- **Definition:** Tuned above_level_rate >= 0.20.
- **Support metrics:** compliance=0.657, above_level_rate=0.343, weighted_above=1.114, target_sat=0.88, subj decod=0, emphasis=0, coherence=0, completeness=0, words=35.
- **Matched base metrics:** compliance=0.567, above_level_rate=0.433, target_sat=1.00, words=171.
- **Likely cause:** generation behavior
- **Judge note (decodability):** The story uses above-level words like 'together', 'happening', and 'jump', which are not limited to consonant blends and digraphs.
- **Judge note (target phonics pattern emphasis):** There is almost no use of consonant blends or digraphs, with only 'grass' and 'jump' as possible examples, and these are not emphasized.

**Representative tuned output**

```text
Title: The Fun Game

Sara and Sam have fun playing games together.  
They play ball and jump rope.  
A big game is happening at the park!  
We run to get on the grass for fun!
```

**Matched base output**

```text
**The Forest and the Lizard**

One day, a little fox found a small tree full of shiny leaves. The leaves were green, but they had a strange shape—like the letters *stop*. The fox was curious about them and thought it might be a kind of animal. 

Then one morning, a big lizard came to visit the forest. He was tall, covered in long fur, and his eyes were bright blue. He noticed that there were many plants with big ears and small heads. Those plants were different from the ones on the leaves. They looked like *ship*, which made a sound when moving.

The next day, the forest began to grow more beautiful. Trees bloomed, flowers grew, and a new creature appeared—the owl. The owl said something sweet and magical to everyone in the forest. And so the forest became home for all things.

Now, if yo…
```

## 4. Prompt-type difficulty (tuned)

| difficulty rank | prompt_id | targets | overall | emphasis | decod_subj | tgt_sat | compliance | above_level |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `diphthongs_digraphs_1` | 2 | 0.00 | 0.00 | 0.00 | 0.38 | 0.848 | 0.152 |
| 2 | `multisyllabic_vowel_teams_r_controlled_1` | 3 | 0.33 | 0.33 | 1.67 | 0.92 | 1.000 | 0.000 |
| 3 | `blends_digraphs_1` | 2 | 0.33 | 0.67 | 0.67 | 0.79 | 0.778 | 0.222 |
| 4 | `vowel_teams_digraphs_1` | 2 | 0.67 | 0.67 | 0.67 | 0.62 | 0.866 | 0.134 |
| 5 | `final_e_blends_1` | 2 | 0.67 | 0.67 | 1.67 | 0.71 | 0.829 | 0.171 |
| 6 | `short_vowels_1` | 1 | 0.67 | 1.00 | 0.67 | 1.00 | 0.751 | 0.249 |
| 7 | `r_controlled_blends_1` | 2 | 0.67 | 1.33 | 0.67 | 0.96 | 0.872 | 0.129 |

Most difficult tuned condition by composite subjective/objective rank: `diphthongs_digraphs_1` (diphthongs, consonant digraphs).

## 5. Are failures concentrated in specific phonics conditions?

| condition | n | share | mean compliance | mean emphasis | pct low emphasis | pct low decod_subj | pct incomplete |
|---|---:|---:|---:|---:|---:|---:|---:|
| vowel teams prompts | 6 | 28.6% | 0.9329 | 0.5000 | 50.0% | 16.7% | 16.7% |
| diphthongs prompts | 3 | 14.3% | 0.8479 | 0.0000 | 100.0% | 100.0% | 66.7% |
| r-controlled vowels prompts | 6 | 28.6% | 0.9357 | 0.8333 | 33.3% | 16.7% | 0.0% |
| multisyllabic words prompts | 3 | 14.3% | 1.0000 | 0.3333 | 66.7% | 0.0% | 0.0% |
| multi-target prompts (>=2) | 18 | 85.7% | 0.8654 | 0.6111 | 44.4% | 38.9% | 22.2% |
| single-target prompts | 3 | 14.3% | 0.7514 | 1.0000 | 0.0% | 33.3% | 33.3% |
| all tuned outputs | 21 | 100.0% | 0.8491 | 0.6667 | 38.1% | 38.1% | 23.8% |

### Observed concentration patterns

- **vowel teams prompts** looks harder than average: low-emphasis rate 50.0% vs overall 38.1%.
- **diphthongs prompts** looks harder than average: low-emphasis rate 100.0% vs overall 38.1%; low decodability-adherence rate 100.0% vs overall 38.1%; mean overall 0.00 vs overall 0.48.
- **r-controlled vowels prompts** is not clearly worse than the tuned average on emphasis/decodability/overall in this sample.
- **multisyllabic words prompts** looks harder than average: low-emphasis rate 66.7% vs overall 38.1%; mean overall 0.33 vs overall 0.48.
- **multi-target prompts (>=2)** looks harder than average: low-emphasis rate 44.4% vs overall 38.1%.
- **single-target prompts** is not clearly worse than the tuned average on emphasis/decodability/overall in this sample.

## Likely-cause tally (major failure representatives)

Counted from rank-1 representatives of major failure/least-gain/prompt-difficulty categories:

- `generation behavior`: 6
- `conflicting training objectives`: 1

### Cause definitions used here

- **training-data coverage:** hard pattern family and/or weak sustained target practice despite non-trivial story length.
- **conflicting training objectives:** tuned becomes more in-level / shorter while losing target-pattern coverage relative to base.
- **small-model capacity:** especially multi-target advanced prompts with weak combined adherence.
- **generation behavior:** incomplete, incoherent, or very short outputs; unstable story framing.
- **rule-based evaluation limitation:** ceiling effects, or sharp disagreement between objective spelling-pattern metrics and subjective judge scores.

## Bottom line

Tuned generations usually improve objective decodability compliance versus base, mainly by staying shorter and more in-level. Remaining errors are not uniform: some least-gain pairs are metric ceilings, while subjective failures cluster in weak target-phonics pattern emphasis, occasional incoherence/incompleteness, and specific multi-pattern prompt families. Treat cause labels as evidence-ranked hypotheses for the next debugging step, not as automatic data-collection mandates.
