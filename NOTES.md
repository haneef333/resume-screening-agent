# Design Decisions & Tradeoffs

This document explains the reasoning behind the key technical choices, and is
deliberately honest about what does and doesn't work well — per the
challenge's own scoring guidance, clearly stating limitations is worth more
than hiding them.

## 1. Semantic similarity: a hybrid engine (embeddings with automatic TF-IDF fallback)

**What I chose:** a `SimilarityEngine` (in `scorer.py`) that tries to load a
sentence-embedding model (`all-MiniLM-L6-v2`, via `sentence-transformers`)
once per run. If that succeeds, all similarity scoring uses embeddings +
cosine similarity. If it fails for any reason — package not installed, or
no internet access to download the model on first use — the engine
automatically and transparently falls back to `scikit-learn`'s
`TfidfVectorizer` + cosine similarity instead. Every scored candidate
records which method actually ran (`similarity_method`), so results stay
auditable either way.

**How I got here:** my first implementation used sentence embeddings only.
While building, I hit a real failure — the pipeline errored out with a
connection failure to huggingface.co, because the environment I was
building in had no route to that domain. A pure "embeddings-only" design
would mean the entire agent breaks for any reviewer on a similarly
restricted network. Rather than just downgrading to TF-IDF-only (my second
implementation), I upgraded the design once more into a hybrid: try the
richer method, degrade gracefully to the cheaper method, log clearly when
that happens, and expose which one ran. This is a more realistic reflection
of how I'd actually want a production system to behave — richer capability
when the environment allows it, no hard failure when it doesn't.

**What I considered and rejected outright:**
- **LLM API call per resume** (e.g. asking Claude/GPT "how well does this
  resume match this JD, 0–100") — rejected because it requires an API key
  from whoever runs the project, costs money per run, is non-deterministic
  (same input can give slightly different scores across runs), and is
  slower for batches of resumes. This applies regardless of network
  access, so it wasn't just downgraded — it was dropped entirely.

**The real cost that remains:** on a network-restricted machine, the agent
runs on TF-IDF, which is a lexical (word-overlap) method, not a semantic
one — it cannot recognize that "led a cross-functional team" and
"leadership experience" mean similar things if they share no words. This is
partially mitigated because the skill-match component (35% of the score) is
taxonomy/alias-based and catches many of these cases independently (e.g.
"ML" and "machine learning" both map to the same canonical skill). When
embeddings are available, this gap mostly closes.

**To force a specific backend** (e.g. for testing): install or uninstall
`sentence-transformers` in the environment — `SimilarityEngine` probes for it
at startup and picks accordingly. No code change needed either way.

## 2. Skill extraction: curated taxonomy instead of open-ended NER

**What I chose:** a hand-curated list of ~35 skills with common aliases
(e.g. "ML" → "machine learning", "sklearn" → "scikit-learn"), matched via
regex with word boundaries.

**Why:** this is fast, fully explainable (you can see exactly why a skill was
or wasn't matched), and needs no training data or model. It directly
supports the "reasoning" output the rubric rewards.

**The real cost:** any skill not in the taxonomy (or an alias I didn't
anticipate) won't be recognized, even if it's genuinely present in the
resume. This is a closed-vocabulary approach, not open-ended extraction. The
taxonomy is easy to extend (it's a plain dict in `parser.py`), but it will
never be exhaustive.

## 3. Experience-year extraction: regex heuristics, not guaranteed-accurate

**What I chose:** look for explicit "X years of experience" statements
first; if none found, sum date ranges (e.g. "2020–2023") found in the text.

**The real cost:** this can undercount candidates who list experience in
unusual formats (e.g. only month/year without a range, or a narrative
paragraph with no explicit dates). It is a heuristic, not a guarantee, on
arbitrary real-world resumes.

**Bug caught during review, fixed:** an earlier version of the "explicit
years" regex had the `experience`/`exp` keyword marked optional, which meant
*any* "N years" mention in the resume text — a candidate's age, a company's
founding year, an unrelated project duration — was incorrectly counted as
work experience. A test resume containing only "I turned 30 years old this
year" scored `experience_match_pct: 100.0` with zero real qualifications.
Fixed by making that keyword required (`experience|exp` is no longer
optional in the pattern). Two regression tests
(`test_ignores_years_unrelated_to_experience`,
`test_still_matches_real_experience_mentions_after_fix`) now lock this
behavior in. This bug existed despite 47 passing tests at the time, because
none of them happened to include a "years" mention unrelated to work
experience — a good reminder that passing tests only prove what they
actually check for, not correctness in general.

## 4. No OCR for scanned/image-only PDFs

**What I chose:** `pdfplumber` extracts text directly from PDFs that have a
text layer. If a PDF is a scanned image with no text layer, extraction
returns empty text and the resume is flagged with a parse error rather than
silently scored as zero.

**Why not add OCR:** OCR (e.g. Tesseract) adds a real dependency and
meaningfully more processing time per resume, for a case (scanned resumes)
that's now uncommon in real hiring pipelines dominated by ATS-exported PDFs
and DOCX files. Given the 24-hour window, I prioritized reliability on the
common case over coverage of an increasingly rare one.

## 5. Education is extracted but deliberately not scored

**What I chose:** `parser.py` extracts an education level (PhD / Master's /
Bachelor's / Diploma / High School) and it's shown in every output record
for reviewer context, but it carries **zero weight** in the final score.

**Why:** the sample JD (like most real JDs for this kind of role) doesn't
state a hard degree requirement — it lists skills and experience, not
"Bachelor's from a top-tier university required." Scoring resumes partly on
degree level (or worse, on which institution it's from) would reward
credential/pedigree signals that are only loosely correlated with the
skills the JD actually asks for, and would bias the ranking against
otherwise strong candidates from less traditional backgrounds. Since the
whole point of this agent is to evaluate *demonstrated* skills and
experience objectively, folding in an unrequested pedigree signal would
undermine that goal. If a specific role genuinely required a degree
(e.g. a regulated field), that would belong in the JD's required-skills
text and could be captured as an explicit requirement — not hardcoded into
the scoring model as a blanket assumption.

## 6. Scoring weights (50% semantic / 35% skills / 15% experience)

These weights are a deliberate judgment call, not derived from data:
- **Skills (35%) and semantic similarity (50%) are weighted highest** because
  they're the most direct signal of role fit and are explicitly what the
  rubric asks the agent to compute ("relevance score against the Job
  Description using NLP similarity").
- **Experience (15%) is weighted lower** and uses partial credit (not a hard
  cutoff) because years-of-experience is an imperfect proxy for capability,
  and a hard cutoff would unfairly zero out otherwise strong candidates who
  are slightly under a stated minimum.

These weights are configurable in one place (`WEIGHTS` dict in `scorer.py`)
and could be tuned further with labeled outcome data (e.g. actual hiring
decisions) if this were a real production system — which it is not; this is
a challenge submission, not a validated production model.

## 7. What I'd improve with more time

- Expand the skill taxonomy considerably, or replace it with a lightweight
  NER model trained/fine-tuned on resume data.
- Add a lightweight OCR fallback (Tesseract) for scanned PDFs, gated behind
  a flag so it doesn't slow down the common case.
- Validate scoring weights against real labeled outcomes instead of judgment.
- Blend embeddings and TF-IDF scores when both are available, rather than
  using embeddings exclusively whenever they're loadable — could smooth out
  edge cases where one method's failure mode shows up in the other's blind
  spot.
- Add a config file (YAML/JSON) for scoring weights and skill taxonomy
  instead of editing Python source, so non-engineers could tune the agent.
