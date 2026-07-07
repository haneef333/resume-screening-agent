# Resume Screening Agent

An AI agent that ranks a folder of resumes against a job description using
NLP-based similarity scoring, skill-requirement matching, and experience-fit
scoring — producing a ranked shortlist with a human-readable explanation for
every candidate's score.

Built for the Rooman Technologies **24-Hour AI Agent Challenge** (Junior AI
Research Associate — Selection Round).

---

## What it does

Given:
- a **job description** (plain text), and
- a **folder of resumes** (PDF, DOCX, or TXT — any mix),

the agent outputs a **ranked, scored shortlist** with reasoning for each
candidate, as both JSON and CSV.

Example output (top of the ranked list):

| rank | candidate_name | final_score | semantic_similarity | skill_match_pct | experience_match_pct |
|------|-----------------|-------------|----------------------|-------------------|------------------------|
| 1    | Aarav Mehta     | 70.96       | 45.9                 | 100.0             | 100.0                  |
| 2    | Isha Choudhary  | 65.31       | 39.2                 | 85.7              | 100.0                  |

---

## How it works (pipeline)

```
Resumes (PDF/DOCX/TXT) ──┐
                         ├──> Parser ──> structured fields (name, skills, experience, education)
Job Description (TXT) ───┘                        │
                                                   ▼
                                              Scorer
                              (TF-IDF similarity + skill overlap + experience match)
                                                   │
                                                   ▼
                                  Ranked shortlist (JSON + CSV) with reasoning
```

1. **Parsing** (`src/parser.py`) — extracts raw text from each resume
   (`pdfplumber` for PDF, `python-docx` for DOCX, plain read for TXT), then
   extracts structured fields:
   - **Skills** — matched against a curated ~35-skill taxonomy (with aliases,
     e.g. "ML" → "machine learning")
   - **Experience (years)** — from explicit "X years" mentions, falling back
     to summing date ranges in work history
   - **Education level** — highest degree mentioned (PhD / Master's /
     Bachelor's / Diploma / High School)
   - **Candidate name** — best-effort heuristic from the first lines of text

2. **Scoring** (`src/scorer.py`) — combines three signals into one weighted
   score (0–100):

   | Component | Weight | What it measures |
   |---|---|---|
   | Semantic similarity | 50% | Embedding or TF-IDF cosine similarity between full resume text and full JD text (see below) |
   | Skill match | 35% | % of JD-required skills found in the candidate's extracted skills |
   | Experience match | 15% | Candidate's years vs. the JD's stated minimum (if any) |

   Each candidate also gets a **plain-English reasoning string** listing
   matched/missing skills and the experience comparison — not just a number.

   **Semantic similarity uses a hybrid engine with automatic fallback:** it
   first tries transformer sentence embeddings (`all-MiniLM-L6-v2`, via
   `sentence-transformers`) for deeper paraphrase-level similarity. If that
   model can't be loaded — not installed, or no internet access to download
   it on first run — the engine automatically falls back to TF-IDF cosine
   similarity, logs a one-line notice to stderr, and continues without
   interrupting the run. Every scored candidate records which method was
   actually used (`similarity_method: "embeddings"` or `"tfidf"`), so results
   stay transparent. Note: education level is extracted and shown for
   context but deliberately **not** scored — see `NOTES.md` for why.

3. **Ranking** (`src/main.py`) — runs the above over every resume in the
   folder, sorts by final score, and writes results to `output/`.

---

## Setup

### Requirements
- Python 3.9+
- No API keys needed (see "Why a hybrid engine" below)

### Install

```bash
git clone <this-repo-url>
cd resume-screening-agent
pip install -r requirements.txt
```

This installs everything needed to run the agent (TF-IDF-based similarity,
guaranteed to work with no network access). If you also want the optional
higher-quality embedding-based similarity and/or to run the test suite:

```bash
pip install -r requirements-dev.txt
```

If `sentence-transformers` is installed but the model can't be downloaded
(no internet), the agent detects this automatically and falls back to
TF-IDF — no crash, no manual intervention needed.

### Run

```bash
python src/main.py --jd data/job_description.txt --resumes data/resumes --output output
```

This will:
- Parse every `.pdf`, `.docx`, and `.txt` file in `data/resumes/`
- Score each one against `data/job_description.txt`
- Write `output/ranked_candidates.json` and `output/ranked_candidates.csv`
- Print the top 5 candidates to the console

### Run on your own data

```bash
python src/main.py --jd path/to/your_job_description.txt --resumes path/to/your_resumes_folder --output path/to/output_folder
```

Any mix of PDF/DOCX/TXT resumes is supported in the same folder.

---

## Project structure

```
resume-screening-agent/
├── README.md
├── requirements.txt              # runtime dependencies (TF-IDF path)
├── requirements-dev.txt          # + pytest, + optional sentence-transformers
├── NOTES.md                      # tradeoffs, design decisions, limitations
├── data/
│   ├── job_description.txt      # sample JD (Data Scientist role)
│   └── resumes/                 # 12 sample resumes (10 TXT + 1 PDF + 1 DOCX)
├── output/
│   ├── ranked_candidates.json
│   └── ranked_candidates.csv
├── src/
│   ├── parser.py                 # text extraction + structured field extraction
│   ├── scorer.py                  # hybrid similarity + skill/experience scoring
│   └── main.py                    # CLI entry point
└── tests/
    ├── test_parser.py             # unit tests for parsing/extraction logic
    └── test_scorer.py             # unit + integration tests for scoring/ranking
```

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

47 tests covering skill extraction (including regression tests for
regex-escaping edge cases like `c++` and single-letter aliases like `R`),
experience/education parsing, and end-to-end ranking sanity checks (e.g. a
senior data scientist must outrank a marketing manager against a Data
Scientist JD).

---

## Sample data included

`data/resumes/` ships with **12 sample resumes** across all three supported
formats (10 `.txt`, 1 `.pdf`, 1 `.docx`), deliberately spanning a range of
fits against the sample JD (Data Scientist role) — from strong matches
(senior data scientists, ML engineers) to weak matches (a marketing manager,
a frontend developer) — so the ranking output is meaningful to review, not
just a list of near-identical scores.

---

## Why a hybrid engine (embeddings-with-fallback), not an LLM API

This is the most important design decision in the project, so it's worth
explaining directly (see `NOTES.md` for the fuller tradeoff writeup):

- **No API key required.** An LLM-based similarity score (OpenAI/Claude) would
  need an API key configured by whoever runs this, adding setup friction and
  a paid, non-deterministic dependency for something classical NLP already
  does well.
- **Embeddings when available, TF-IDF when not.** The agent first tries
  transformer sentence embeddings (`all-MiniLM-L6-v2`) for the richer,
  paraphrase-aware similarity they provide. If the model can't be downloaded
  (restricted/offline network — something I hit directly while building
  this), the agent detects the failure and **automatically falls back to
  TF-IDF cosine similarity** instead of crashing. Either way it keeps
  running, and it records which method actually produced each score.
- **Fully reproducible either way.** TF-IDF is exactly deterministic; the
  embedding path is effectively deterministic for a fixed model version.
  Neither depends on an external API's response variability.
- **The tradeoff:** TF-IDF (the fallback path) matches on literal word
  overlap weighted by term distinctiveness, so on its own it won't catch
  pure paraphrases with zero shared vocabulary (e.g. "led a team of five" vs.
  "leadership experience"). Running with embeddings available closes most of
  that gap. See `NOTES.md` for the full writeup, including why I built it
  this way after hitting the network limitation firsthand.

---

## Command-line options

| Flag | Required | Description |
|---|---|---|
| `--jd` | Yes | Path to the job description text file |
| `--resumes` | Yes | Path to a folder containing resumes (PDF/DOCX/TXT) |
| `--output` | No (default: `output`) | Folder to write `ranked_candidates.json` / `.csv` |

---

## Limitations

See `NOTES.md` for the full list — briefly: scanned/image-only PDFs aren't
supported (no OCR), skill extraction is taxonomy-based (won't catch skills
outside the ~35-skill list unless added), and experience-year extraction is
heuristic (best-effort regex, not guaranteed exact for unusually formatted
resumes).
