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
| 1 | `blends_digraphs_1` | 1 | 1001 | 0.4977 | 0.9259 | +0.4282 | 27 | generation behavior |
| 2 | `final_e_blends_1` | 1 | 1001 | 0.6202 | 0.9286 | +0.3084 | 28 | generation behavior |
| 3 | `vowel_teams_digraphs_1` | 1 | 1001 | 0.6571 | 0.9596 | +0.3025 | 99 | conflicting training objectives |
| 4 | `short_vowels_1` | 2 | 1002 | 0.3869 | 0.6825 | +0.2956 | 63 | generation behavior |
| 5 | `short_vowels_1` | 1 | 1001 | 0.4842 | 0.7500 | +0.2658 | 28 | training-data coverage |

### Representative improvement

- **Pair:** `blends_digraphs_1` / gen 1 / seed 1001
- **Metrics:** compliance 0.498 → 0.926 (Δ +0.428); above-level rate 0.502 → 0.074; words 215 → 27.
- **Likely cause:** generation behavior

**Tuned output**

```text
Title: The Frog

The frog is in the pond.  
The frog jumps to the moon.  
The sun shines on the frog.  
The dog runs past the pond.
```

**Matched base output**

```text
**The Happy Forest Adventure**

One day, a little red squirrel named Mira noticed something unusual in her favorite forest. She found a small tree with big leaves and a bright red apple on top of it. As she looked around, she saw a new animal—a curious yellow bear who was always looking for adventure.

The bear had been following Mira all over the forest, but now they were together! They both wanted to explore more. After a while, Mira helped the bear find a safe place to rest by building a tiny nest made from sticks and leaves. The bear said, “I can take you somewhere,” and they both started walking through the forest.

As they walked, they met many interesting creatures: a wise old owl, a clever fox, and a funny eagle. They learned about different animals and how they lived. Mira was es…
```

## 2. Least paired improvement (or no gain)

| rank | prompt_id | gen | seed | base | tuned | delta | note | likely cause |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | `multisyllabic_vowel_teams_r_controlled_1` | 1 | 1001 | 1.0000 | 1.0000 | +0.0000 | Smallest paired compliance_delta. Ceiling effect: both models already near-perfect on this metric. | rule-based evaluation limitation |
| 2 | `multisyllabic_vowel_teams_r_controlled_1` | 2 | 1002 | 1.0000 | 1.0000 | +0.0000 | Smallest paired compliance_delta. Ceiling effect: both models already near-perfect on this metric. | rule-based evaluation limitation |
| 3 | `multisyllabic_vowel_teams_r_controlled_1` | 3 | 1003 | 1.0000 | 1.0000 | +0.0000 | Smallest paired compliance_delta. Ceiling effect: both models already near-perfect on this metric. | rule-based evaluation limitation |
| 4 | `final_e_blends_1` | 3 | 1003 | 0.6259 | 0.7073 | +0.0814 | Smallest paired compliance_delta. Tuned absolute compliance remains relatively weak. | conflicting training objectives |
| 5 | `r_controlled_blends_1` | 3 | 1003 | 0.7186 | 0.8305 | +0.1119 | Smallest paired compliance_delta. Tuned absolute compliance remains relatively weak. | conflicting training objectives |

### Representative weak/least-gain case

- **Pair:** `final_e_blends_1` / gen 3 / seed 1003
- **Notes:** Smallest paired compliance_delta. Tuned absolute compliance remains relatively weak.
- **Metrics:** compliance 0.626 → 0.707 (Δ +0.081); target satisfaction 1.00 → 0.50; subj decod 0, emphasis 0.
- **Likely cause:** conflicting training objectives

**Tuned output**

```text
Title: A Trip to the Farmyard

The family has just left their house for a trip to the farmyard.  
They will ride on their horses to see the animals there!  
As they ride by the barn, they spot many small birds on the grass.  
They run into a big red cow who looks very sad.  
The cows are happy to be at the farm, so the cows give them a big hug.  
Then they look back at the farm and say goodbye!
```

**Matched base output**

```text
**The Happy Forest Adventure**

In a small forest where the trees were all tall and green, there lived a young fox named Max. One day, Max heard a sound so loud it made his heart jump. He ran to the edge of the forest and saw a big white bear standing on a high hill. The bear was full of energy and had a shiny red fur.

Max thought, “I wonder if I can help this bear.” He decided to go with the bear and helped him find food for himself. Together they found a big pile of apples, bananas, and nuts.

Now, when Max looked back, he saw the bear still happy and the forest full of wild animals. They shared their favorite foods and stayed together long after the bear left.

That's how the forest continued to grow and prosper, filled with joy and friendship. 🌟
```

## 3. Tuned failure-mode inventory

| failure mode | n flagged | share of tuned | representative prompt | likely cause |
|---|---:|---:|---|---|
| `low_decodability_adherence` | 11 | 52.4% | `short_vowels_1` | training-data coverage |
| `low_target_phonics_emphasis` | 13 | 61.9% | `short_vowels_1` | training-data coverage |
| `low_coherence` | 9 | 42.9% | `short_vowels_1` | generation behavior |
| `incomplete_narrative` | 8 | 38.1% | `short_vowels_1` | generation behavior |
| `high_above_level` | 6 | 28.6% | `short_vowels_1` | training-data coverage |

### Failure mode: `low_decodability_adherence`

- **Definition:** Tuned subjective decodability_adherence_score == 0.
- **Support metrics:** compliance=0.679, above_level_rate=0.321, weighted_above=1.210, target_sat=1.00, subj decod=0, emphasis=0, coherence=1, completeness=1, words=81.
- **Matched base metrics:** compliance=0.431, above_level_rate=0.569, target_sat=1.00, words=211.
- **Likely cause:** training-data coverage
- **Judge note (decodability):** The story includes several above-level words and patterns such as 'watching,' 'movie,' 'reading,' 'lunch,' 'kitchen,' 'find,' 'soup,' and 'outside,' which are not short-vowel words.
- **Judge note (target phonics pattern emphasis):** Short-vowel words are rarely emphasized, with most content relying on above-level vocabulary.

**Representative tuned output**

```text
Title: A Hot Day at Home

Dad is sitting on the couch watching a movie while Mom is reading a book.  
I am getting ready to go for lunch with Dad.  
Mom says I can take my hat if it’s hot outside.  
I get to the kitchen and find two pots of soup in the sink.  
The pot has a lid, so it will be good to heat up the soup.  
When I put the lid back, the pot is warm!
```

**Matched base output**

```text
**The Little Red Rabbit's Adventure**

One day, the little red rabbit found a small box full of shiny toys. Inside, there was a picture of a big, happy elephant. The elephant had a special tool, a tiny stick and a bunch of colorful buttons. The rabbit was curious and asked what the elephant did with its tools.

The elephant said, "I like to play games with my friends!" Then it pointed to a toy hop on the floor. The rabbit saw the toy hop and thought, "Oh! That's a short vowel word." She giggled and played with her own hop. Soon after, she found another toy—a red one. The red color made her smile, and she realized that the colors could be a short vowel pair.

Next, she found a cat. The cat was cute, so she named her friend a cat. And then, the rabbit found a sit. It was a simple word, and…
```

### Failure mode: `low_target_phonics_emphasis`

- **Definition:** Tuned target_phonics_pattern_emphasis_score == 0.
- **Support metrics:** compliance=0.679, above_level_rate=0.321, weighted_above=1.210, target_sat=1.00, subj decod=0, emphasis=0, coherence=1, completeness=1, words=81.
- **Matched base metrics:** compliance=0.431, above_level_rate=0.569, target_sat=1.00, words=211.
- **Likely cause:** training-data coverage
- **Judge note (decodability):** The story includes several above-level words and patterns such as 'watching,' 'movie,' 'reading,' 'lunch,' 'kitchen,' 'find,' 'soup,' and 'outside,' which are not short-vowel words.
- **Judge note (target phonics pattern emphasis):** Short-vowel words are rarely emphasized, with most content relying on above-level vocabulary.

**Representative tuned output**

```text
Title: A Hot Day at Home

Dad is sitting on the couch watching a movie while Mom is reading a book.  
I am getting ready to go for lunch with Dad.  
Mom says I can take my hat if it’s hot outside.  
I get to the kitchen and find two pots of soup in the sink.  
The pot has a lid, so it will be good to heat up the soup.  
When I put the lid back, the pot is warm!
```

**Matched base output**

```text
**The Little Red Rabbit's Adventure**

One day, the little red rabbit found a small box full of shiny toys. Inside, there was a picture of a big, happy elephant. The elephant had a special tool, a tiny stick and a bunch of colorful buttons. The rabbit was curious and asked what the elephant did with its tools.

The elephant said, "I like to play games with my friends!" Then it pointed to a toy hop on the floor. The rabbit saw the toy hop and thought, "Oh! That's a short vowel word." She giggled and played with her own hop. Soon after, she found another toy—a red one. The red color made her smile, and she realized that the colors could be a short vowel pair.

Next, she found a cat. The cat was cute, so she named her friend a cat. And then, the rabbit found a sit. It was a simple word, and…
```

### Failure mode: `low_coherence`

- **Definition:** Tuned coherence_score == 0.
- **Support metrics:** compliance=0.682, above_level_rate=0.318, weighted_above=0.968, target_sat=1.00, subj decod=0, emphasis=0, coherence=0, completeness=0, words=63.
- **Matched base metrics:** compliance=0.387, above_level_rate=0.613, target_sat=1.00, words=199.
- **Likely cause:** generation behavior
- **Judge note (decodability):** The story includes above-level words such as 'playing,' 'outside,' 'jumps,' 'onto,' 'back,' 'home,' and 'happy,' which are not short-vowel words.
- **Judge note (target phonics pattern emphasis):** Short-vowel words are rarely present, and most words do not practice the target pattern.

**Representative tuned output**

```text
Title: The Cat and the Dog

The dog is playing outside when it sees the cat.
The cat jumps onto the dog’s back to play.
When they get home, the dog will run fast for the cat.
“Can you help me get the cat,” said the dog.
“Yes!” ran the cat up high and down.
The cat was so happy to see them!
```

**Matched base output**

```text
**The Happy Forest of Squirrels**

In the heart of the forest lived a small group of squirrels. They had long been friends, sharing their favorite places—honeycombs, nests, and trees. The main tree was called Maple Tree, which had many holes filled with seeds from all the fruits grown by the squirrels. 

One day, a new squirrel came to join them. She was named Sparky. She loved singing songs, playing games, and exploring new places. Soon after, more squirrels joined the group, each bringing their own unique skills and ideas. There were no longer just the usual ones—they were all part of a big family.

One morning, the group went on a journey. They traveled through different kinds of woods, meeting birds, animals, and even a river. Along the way, they learned about nature and made new frie…
```

### Failure mode: `incomplete_narrative`

- **Definition:** Tuned narrative_completeness_score == 0.
- **Support metrics:** compliance=0.682, above_level_rate=0.318, weighted_above=0.968, target_sat=1.00, subj decod=0, emphasis=0, coherence=0, completeness=0, words=63.
- **Matched base metrics:** compliance=0.387, above_level_rate=0.613, target_sat=1.00, words=199.
- **Likely cause:** generation behavior
- **Judge note (decodability):** The story includes above-level words such as 'playing,' 'outside,' 'jumps,' 'onto,' 'back,' 'home,' and 'happy,' which are not short-vowel words.
- **Judge note (target phonics pattern emphasis):** Short-vowel words are rarely present, and most words do not practice the target pattern.

**Representative tuned output**

```text
Title: The Cat and the Dog

The dog is playing outside when it sees the cat.
The cat jumps onto the dog’s back to play.
When they get home, the dog will run fast for the cat.
“Can you help me get the cat,” said the dog.
“Yes!” ran the cat up high and down.
The cat was so happy to see them!
```

**Matched base output**

```text
**The Happy Forest of Squirrels**

In the heart of the forest lived a small group of squirrels. They had long been friends, sharing their favorite places—honeycombs, nests, and trees. The main tree was called Maple Tree, which had many holes filled with seeds from all the fruits grown by the squirrels. 

One day, a new squirrel came to join them. She was named Sparky. She loved singing songs, playing games, and exploring new places. Soon after, more squirrels joined the group, each bringing their own unique skills and ideas. There were no longer just the usual ones—they were all part of a big family.

One morning, the group went on a journey. They traveled through different kinds of woods, meeting birds, animals, and even a river. Along the way, they learned about nature and made new frie…
```

### Failure mode: `high_above_level`

- **Definition:** Tuned above_level_rate >= 0.20.
- **Support metrics:** compliance=0.679, above_level_rate=0.321, weighted_above=1.210, target_sat=1.00, subj decod=0, emphasis=0, coherence=1, completeness=1, words=81.
- **Matched base metrics:** compliance=0.431, above_level_rate=0.569, target_sat=1.00, words=211.
- **Likely cause:** training-data coverage
- **Judge note (decodability):** The story includes several above-level words and patterns such as 'watching,' 'movie,' 'reading,' 'lunch,' 'kitchen,' 'find,' 'soup,' and 'outside,' which are not short-vowel words.
- **Judge note (target phonics pattern emphasis):** Short-vowel words are rarely emphasized, with most content relying on above-level vocabulary.

**Representative tuned output**

```text
Title: A Hot Day at Home

Dad is sitting on the couch watching a movie while Mom is reading a book.  
I am getting ready to go for lunch with Dad.  
Mom says I can take my hat if it’s hot outside.  
I get to the kitchen and find two pots of soup in the sink.  
The pot has a lid, so it will be good to heat up the soup.  
When I put the lid back, the pot is warm!
```

**Matched base output**

```text
**The Little Red Rabbit's Adventure**

One day, the little red rabbit found a small box full of shiny toys. Inside, there was a picture of a big, happy elephant. The elephant had a special tool, a tiny stick and a bunch of colorful buttons. The rabbit was curious and asked what the elephant did with its tools.

The elephant said, "I like to play games with my friends!" Then it pointed to a toy hop on the floor. The rabbit saw the toy hop and thought, "Oh! That's a short vowel word." She giggled and played with her own hop. Soon after, she found another toy—a red one. The red color made her smile, and she realized that the colors could be a short vowel pair.

Next, she found a cat. The cat was cute, so she named her friend a cat. And then, the rabbit found a sit. It was a simple word, and…
```

## 4. Prompt-type difficulty (tuned)

| difficulty rank | prompt_id | targets | overall | emphasis | decod_subj | full_spec | compliance | above_level |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `diphthongs_digraphs_1` | 2 | 0.00 | 0.00 | 0.67 | 0.00 | 0.949 | 0.051 |
| 2 | `vowel_teams_digraphs_1` | 2 | 0.00 | 0.33 | 0.33 | 0.00 | 0.934 | 0.066 |
| 3 | `final_e_blends_1` | 2 | 0.00 | 0.33 | 0.67 | 0.33 | 0.785 | 0.215 |
| 4 | `r_controlled_blends_1` | 2 | 0.33 | 0.33 | 0.33 | 0.67 | 0.854 | 0.146 |
| 5 | `short_vowels_1` | 1 | 0.33 | 0.33 | 0.33 | 1.00 | 0.704 | 0.296 |
| 6 | `multisyllabic_vowel_teams_r_controlled_1` | 3 | 0.67 | 0.33 | 1.33 | 0.67 | 1.000 | 0.000 |
| 7 | `blends_digraphs_1` | 2 | 0.67 | 1.00 | 1.00 | 0.33 | 0.811 | 0.189 |

Most difficult tuned condition by composite subjective/objective rank: `diphthongs_digraphs_1` (diphthongs, consonant digraphs).

## 5. Are failures concentrated in specific phonics conditions?

| condition | n | share | mean compliance | mean emphasis | pct low emphasis | pct low decod_subj | pct incomplete |
|---|---:|---:|---:|---:|---:|---:|---:|
| vowel teams prompts | 6 | 28.6% | 0.9672 | 0.3333 | 66.7% | 33.3% | 50.0% |
| diphthongs prompts | 3 | 14.3% | 0.9494 | 0.0000 | 100.0% | 66.7% | 33.3% |
| r-controlled vowels prompts | 6 | 28.6% | 0.9270 | 0.3333 | 66.7% | 33.3% | 33.3% |
| multisyllabic words prompts | 3 | 14.3% | 1.0000 | 0.3333 | 66.7% | 0.0% | 33.3% |
| multi-target prompts (>=2) | 18 | 85.7% | 0.8889 | 0.3889 | 61.1% | 50.0% | 38.9% |
| single-target prompts | 3 | 14.3% | 0.7038 | 0.3333 | 66.7% | 66.7% | 33.3% |
| all tuned outputs | 21 | 100.0% | 0.8624 | 0.3810 | 61.9% | 52.4% | 38.1% |

### Observed concentration patterns

- **vowel teams prompts** is not clearly worse than the tuned average on emphasis/decodability/overall in this sample.
- **diphthongs prompts** looks harder than average: low-emphasis rate 100.0% vs overall 61.9%; low decodability-adherence rate 66.7% vs overall 52.4%; mean overall 0.00 vs overall 0.29.
- **r-controlled vowels prompts** is not clearly worse than the tuned average on emphasis/decodability/overall in this sample.
- **multisyllabic words prompts** is not clearly worse than the tuned average on emphasis/decodability/overall in this sample.
- **multi-target prompts (>=2)** is not clearly worse than the tuned average on emphasis/decodability/overall in this sample.
- **single-target prompts** looks harder than average: low decodability-adherence rate 66.7% vs overall 52.4%.

## Likely-cause tally (major failure representatives)

Counted from rank-1 representatives of major failure/least-gain/prompt-difficulty categories:

- `training-data coverage`: 3
- `generation behavior`: 2
- `rule-based evaluation limitation`: 1
- `conflicting training objectives`: 1

### Cause definitions used here

- **training-data coverage:** hard pattern family and/or weak sustained target practice despite non-trivial story length.
- **conflicting training objectives:** tuned becomes more in-level / shorter while losing target-pattern coverage relative to base.
- **small-model capacity:** especially multi-target advanced prompts with weak combined adherence.
- **generation behavior:** incomplete, incoherent, or very short outputs; unstable story framing.
- **rule-based evaluation limitation:** ceiling effects, or sharp disagreement between objective spelling-pattern metrics and subjective judge scores.

## Bottom line

Tuned generations usually improve objective decodability compliance versus base, mainly by staying shorter and more in-level. Remaining errors are not uniform: some least-gain pairs are metric ceilings, while subjective failures cluster in weak target-phonics pattern emphasis, occasional incoherence/incompleteness, and specific multi-pattern prompt families. Treat cause labels as evidence-ranked hypotheses for the next debugging step, not as automatic data-collection mandates.
