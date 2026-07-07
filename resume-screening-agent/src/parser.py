"""
parser.py
Extracts raw text and structured fields (skills, experience, education) from
resumes in PDF, DOCX, or TXT format.

Design notes (see NOTES.md for full tradeoff discussion):
- Skill extraction uses a curated taxonomy + keyword/alias matching (fast,
  deterministic, explainable) rather than an LLM call per resume. This keeps
  the pipeline free, fast, and fully reproducible for reviewers.
- Experience extraction uses regex over explicit "X years" mentions and
  date-range patterns in work history sections. This is heuristic, not
  perfect (see tradeoffs), but works well for typical resume formats.
"""

import re
import os
from datetime import datetime
from typing import Dict, List, Optional

import pdfplumber
import docx


# ---------------------------------------------------------------------------
# Skill taxonomy — extend this list to widen recognizable skills.
# Keys are canonical skill names; values are alias/variant strings to match.
# ---------------------------------------------------------------------------
SKILL_TAXONOMY: Dict[str, List[str]] = {
    "python": ["python"],
    "r": ["r"],
    "sql": ["sql", "mysql", "postgresql", "postgres", "sqlite", "nosql"],
    "java": ["java"],
    "c++": ["c++", "cpp"],
    "javascript": ["javascript", "js", "typescript"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "dl"],
    "nlp": ["nlp", "natural language processing"],
    "computer vision": ["computer vision", "cv", "opencv"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "tensorflow": ["tensorflow", "tf"],
    "pytorch": ["pytorch", "torch"],
    "keras": ["keras"],
    "huggingface": ["huggingface", "hugging face", "transformers library"],
    "data visualization": ["data visualization", "matplotlib", "seaborn", "tableau", "power bi", "powerbi"],
    "statistics": ["statistics", "statistical analysis", "hypothesis testing"],
    "data analysis": ["data analysis", "data analytics"],
    "aws": ["aws", "amazon web services"],
    "gcp": ["gcp", "google cloud"],
    "azure": ["azure"],
    "docker": ["docker", "containerization"],
    "kubernetes": ["kubernetes", "k8s"],
    "git": ["git", "github", "gitlab", "version control"],
    "flask": ["flask"],
    "django": ["django"],
    "fastapi": ["fastapi"],
    "rest api": ["rest api", "restful api", "rest apis"],
    "spark": ["spark", "pyspark", "apache spark"],
    "hadoop": ["hadoop"],
    "excel": ["excel", "ms excel", "microsoft excel"],
    "communication": ["communication skills", "presentation skills"],
    "leadership": ["leadership", "team lead", "team management"],
    "linux": ["linux", "unix", "bash", "shell scripting"],
    "data engineering": ["data engineering", "etl", "data pipeline"],
    "llm": ["llm", "large language model", "gpt", "openai", "anthropic", "claude"],
    "agile": ["agile", "scrum", "kanban"],
}

EDUCATION_PATTERNS = [
    (r"\bph\.?d\.?\b|\bdoctorate\b", "PhD"),
    (r"\bm\.?tech\b|\bm\.?e\.?\b|\bmaster'?s?\b|\bmsc\b|\bm\.?s\.?\b(?!c)", "Master's"),
    (r"\bb\.?tech\b|\bb\.?e\.?\b|\bbachelor'?s?\b|\bbsc\b|\bb\.?s\.?\b(?!c)", "Bachelor's"),
    (r"\bdiploma\b", "Diploma"),
    (r"\b12th\b|\bhigh school\b|\bhsc\b", "High School"),
]


def extract_text(file_path: str) -> str:
    """Extract raw text from a PDF, DOCX, or TXT file."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts)

    elif ext == ".docx":
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    else:
        raise ValueError(f"Unsupported file type: {ext} (expected .pdf, .docx, or .txt)")


def extract_skills(text: str) -> List[str]:
    """
    Match resume text against the skill taxonomy. Returns canonical skill names.

    Each alias is regex-escaped (so literal characters like '+' in "c++" are
    matched literally, not interpreted as regex operators). Aliases made up
    entirely of word characters get word boundaries so short tokens like "r"
    or "java" don't match inside unrelated words (e.g. "r" inside "recruiter").
    Aliases containing symbols (like "c++") are matched as escaped substrings
    without boundaries, since '+' is not a word character.
    """
    text_lower = text.lower()
    found = []
    for canonical, aliases in SKILL_TAXONOMY.items():
        for alias in aliases:
            escaped = re.escape(alias)
            if re.fullmatch(r"[\w\s]+", alias):
                pattern = rf"\b{escaped}\b"
            else:
                pattern = escaped
            if re.search(pattern, text_lower):
                found.append(canonical)
                break
    return sorted(found)


def extract_experience_years(text: str) -> float:
    """
    Estimate total years of experience.

    Strategy (in priority order):
    1. Look for explicit statements like "5 years of experience" / "3+ years".
    2. Fall back to summing date ranges found in a work-history-like section
       (e.g. "Jan 2020 - Mar 2023").
    3. If nothing found, return 0.0 (treated as entry-level / unspecified).
    """
    text_lower = text.lower()

    # Strategy 1: explicit "X years" mentions
    explicit_matches = re.findall(
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s*(?:of\s*)?(?:experience|exp)",
        text_lower,
    )
    if explicit_matches:
        # Use the maximum explicit figure mentioned (usually the headline total)
        try:
            return max(float(m) for m in explicit_matches)
        except ValueError:
            pass

    # Strategy 2: date ranges like "2020 - 2023" or "Jan 2019 - Present"
    range_pattern = re.findall(
        r"(20\d{2}|19\d{2})\s*[-–to]+\s*(20\d{2}|19\d{2}|present|current)",
        text_lower,
    )
    total_months = 0
    current_year = datetime.now().year
    for start, end in range_pattern:
        start_year = int(start)
        end_year = current_year if end in ("present", "current") else int(end)
        if end_year >= start_year:
            total_months += (end_year - start_year) * 12

    if total_months > 0:
        return round(total_months / 12, 1)

    return 0.0


def extract_education(text: str) -> str:
    """Return the highest education level mentioned, based on pattern priority."""
    text_lower = text.lower()
    for pattern, label in EDUCATION_PATTERNS:
        if re.search(pattern, text_lower):
            return label
    return "Not specified"


def extract_name(text: str) -> str:
    """
    Best-effort guess at candidate name: first non-empty line that looks like
    a name (short, no digits, no common resume section headers).
    """
    section_headers = {"resume", "curriculum vitae", "cv", "profile", "summary", "objective"}
    for line in text.strip().split("\n")[:5]:
        clean = line.strip()
        if not clean:
            continue
        if any(ch.isdigit() for ch in clean):
            continue
        if clean.lower() in section_headers:
            continue
        if len(clean.split()) <= 5 and len(clean) < 60:
            return clean
    return "Unknown Candidate"


def parse_resume(file_path: str) -> Dict:
    """Parse a single resume file into a structured dict."""
    text = extract_text(file_path)
    if not text.strip():
        raise ValueError(f"No extractable text found in {file_path} (possibly a scanned/image PDF)")

    return {
        "file_name": os.path.basename(file_path),
        "candidate_name": extract_name(text),
        "raw_text": text,
        "skills": extract_skills(text),
        "experience_years": extract_experience_years(text),
        "education": extract_education(text),
    }


def parse_resumes_from_folder(folder_path: str) -> List[Dict]:
    """
    Parse every supported resume file in a folder.

    Files with unsupported extensions (e.g. .doc, .rtf, .xyz) are not parsed,
    but are still reported in the results as a "skipped" entry rather than
    silently ignored — so a reviewer who drops an unsupported file in the
    folder gets a clear signal it wasn't processed, instead of it just
    quietly not appearing anywhere.
    """
    supported_ext = (".pdf", ".docx", ".txt")
    results = []
    for fname in sorted(os.listdir(folder_path)):
        fpath = os.path.join(folder_path, fname)
        if not os.path.isfile(fpath):
            continue  # skip subdirectories
        if fname.lower().endswith(supported_ext):
            try:
                results.append(parse_resume(fpath))
            except Exception as e:
                results.append({
                    "file_name": fname,
                    "error": str(e),
                })
        else:
            results.append({
                "file_name": fname,
                "error": f"Skipped: unsupported file type (supported: {', '.join(supported_ext)})",
            })
    return results


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) != 2:
        print("Usage: python parser.py <path_to_resume_or_folder>")
        sys.exit(1)

    path = sys.argv[1]
    if os.path.isdir(path):
        output = parse_resumes_from_folder(path)
    else:
        output = parse_resume(path)

    print(json.dumps(output, indent=2, default=str))
