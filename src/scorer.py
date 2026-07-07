"""
scorer.py
Scores parsed resumes against a job description using a hybrid approach:

  1. Semantic similarity — see SimilarityEngine below. Tries transformer
     sentence embeddings first (captures paraphrase-level semantic fit, e.g.
     "led a team of five" ~ "leadership experience"); if the model can't be
     loaded (no internet, restricted network, first-run download blocked),
     automatically and silently falls back to TF-IDF cosine similarity, which
     needs no network access and is fully deterministic. Either way, the
     output records which method was actually used, so scores stay
     transparent and comparable within a single run.
  2. Skill overlap — fraction of JD-required skills explicitly found in the
     candidate's extracted skill list. Captures precise, explainable
     requirement matching.
  3. Experience match — how the candidate's years of experience compares to
     the JD's stated requirement (if any).

Final score = weighted combination (see WEIGHTS below), producing both a
single ranking number and a human-readable reasoning string per candidate.

NOTE ON EDUCATION: education level is extracted (see parser.py) and shown in
the output for reviewer context, but is deliberately NOT part of the weighted
score. The JD doesn't state a hard education requirement, and scoring on
institution/degree tier risks rewarding pedigree over actual demonstrated
skill and experience — which is exactly what this agent is designed to
evaluate instead. See NOTES.md for the full reasoning.
"""

import re
import sys
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from parser import SKILL_TAXONOMY, extract_skills, extract_experience_years

# ---------------------------------------------------------------------------
# Scoring weights — must sum to 1.0. Tunable; see NOTES.md for rationale.
# ---------------------------------------------------------------------------
WEIGHTS = {
    "semantic_similarity": 0.5,
    "skill_match": 0.35,
    "experience_match": 0.15,
}


class SimilarityEngine:
    """
    Provides semantic similarity scoring with automatic method selection:
    - Preferred: sentence-transformer embeddings (all-MiniLM-L6-v2), which
      capture deeper semantic/paraphrase similarity.
    - Fallback: TF-IDF cosine similarity, used automatically if the embedding
      model can't be loaded (e.g. no network access to download it).

    The engine tries to load the embedding model exactly once per run (not
    once per resume) and caches the result. `self.method` records which
    backend ended up being used, so it can be surfaced in output/reasoning.
    """

    def __init__(self):
        self.method = None
        self._embed_model = None
        self._try_load_embedding_model()

    def _try_load_embedding_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.method = "embeddings"
        except Exception as e:
            # Covers: package not installed, no network to download the model,
            # or any other load failure. Fall back to TF-IDF automatically.
            self.method = "tfidf"
            print(
                f"[scorer] Semantic embedding model unavailable ({type(e).__name__}); "
                f"falling back to TF-IDF similarity.",
                file=sys.stderr,
            )

    def score(self, resume_text: str, jd_text: str) -> float:
        if self.method == "embeddings":
            return self._embedding_similarity(resume_text, jd_text)
        return self._tfidf_similarity(resume_text, jd_text)

    def _embedding_similarity(self, resume_text: str, jd_text: str) -> float:
        embeddings = self._embed_model.encode([resume_text, jd_text])
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(max(0.0, min(1.0, sim)))

    @staticmethod
    def _tfidf_similarity(resume_text: str, jd_text: str) -> float:
        """
        TF-IDF cosine similarity between resume and JD text, scaled to 0-1.
        Vectorizer is fit fresh on the (resume, JD) pair each call so the
        score is self-contained and doesn't depend on a shared corpus
        vocabulary.
        """
        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
        except ValueError:
            return 0.0  # e.g. empty text after stopword removal
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(max(0.0, min(1.0, sim)))


# Module-level singleton so the embedding model (if available) is loaded once
# per process, not once per resume.
_engine = None


def get_similarity_engine() -> SimilarityEngine:
    global _engine
    if _engine is None:
        _engine = SimilarityEngine()
    return _engine


def extract_required_experience_years(jd_text: str) -> float:
    """Extract the minimum years of experience required, from JD text."""
    matches = re.findall(
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s*(?:of\s*)?(?:experience|exp)",
        jd_text.lower(),
    )
    if matches:
        return min(float(m) for m in matches)  # minimum stated requirement
    return 0.0


def semantic_similarity_score(resume_text: str, jd_text: str) -> float:
    """Semantic similarity via the shared SimilarityEngine (embeddings or TF-IDF)."""
    return get_similarity_engine().score(resume_text, jd_text)


def skill_match_score(candidate_skills: List[str], jd_skills: List[str]) -> Dict:
    """
    Fraction of JD-required skills present in the candidate's skill list.
    Returns the score plus matched/missing skill lists for transparency.
    """
    if not jd_skills:
        return {"score": 0.0, "matched": [], "missing": []}

    candidate_set = set(candidate_skills)
    jd_set = set(jd_skills)
    matched = sorted(candidate_set & jd_set)
    missing = sorted(jd_set - candidate_set)
    score = len(matched) / len(jd_set)
    return {"score": score, "matched": matched, "missing": missing}


def experience_match_score(candidate_years: float, required_years: float) -> float:
    """
    Score how candidate experience compares to the JD requirement.
    - Meets or exceeds requirement -> 1.0
    - Below requirement -> linear partial credit (proportional shortfall)
    - No requirement stated -> neutral 0.75 (don't penalize or reward blindly)
    """
    if required_years == 0:
        return 0.75
    if candidate_years >= required_years:
        return 1.0
    return max(0.0, candidate_years / required_years)


def build_reasoning(
    semantic_score: float,
    skill_result: Dict,
    exp_score: float,
    candidate_years: float,
    required_years: float,
    similarity_method: str,
) -> str:
    """Generate a short, human-readable explanation for the final score."""
    parts = []

    method_label = "embedding-based" if similarity_method == "embeddings" else "TF-IDF"
    parts.append(f"Semantic fit with JD ({method_label}): {semantic_score * 100:.0f}%.")

    if skill_result["matched"]:
        parts.append(
            f"Matched {len(skill_result['matched'])}/{len(skill_result['matched']) + len(skill_result['missing'])} "
            f"required skills ({', '.join(skill_result['matched'])})."
        )
    else:
        parts.append("No required skills matched.")

    if skill_result["missing"]:
        parts.append(f"Missing: {', '.join(skill_result['missing'])}.")

    if required_years > 0:
        parts.append(
            f"Experience: {candidate_years} yrs vs {required_years} yrs required."
        )
    else:
        parts.append(f"Experience: {candidate_years} yrs (no minimum stated in JD).")

    return " ".join(parts)


def score_candidate(parsed_resume: Dict, jd_text: str, jd_skills: List[str], required_years: float) -> Dict:
    """Compute the full weighted score + reasoning for one parsed resume."""
    engine = get_similarity_engine()
    semantic_score = engine.score(parsed_resume["raw_text"], jd_text)
    skill_result = skill_match_score(parsed_resume["skills"], jd_skills)
    exp_score = experience_match_score(parsed_resume["experience_years"], required_years)

    final_score = (
        WEIGHTS["semantic_similarity"] * semantic_score
        + WEIGHTS["skill_match"] * skill_result["score"]
        + WEIGHTS["experience_match"] * exp_score
    )

    reasoning = build_reasoning(
        semantic_score, skill_result, exp_score,
        parsed_resume["experience_years"], required_years,
        engine.method,
    )

    return {
        "file_name": parsed_resume["file_name"],
        "candidate_name": parsed_resume.get("candidate_name", "Unknown"),
        "final_score": round(final_score * 100, 2),  # 0-100 scale for readability
        "semantic_similarity": round(semantic_score * 100, 2),
        "similarity_method": engine.method,
        "skill_match_pct": round(skill_result["score"] * 100, 2),
        "experience_match_pct": round(exp_score * 100, 2),
        "matched_skills": skill_result["matched"],
        "missing_skills": skill_result["missing"],
        "candidate_experience_years": parsed_resume["experience_years"],
        "candidate_education": parsed_resume.get("education", "Not specified"),
        "reasoning": reasoning,
    }


def rank_candidates(parsed_resumes: List[Dict], jd_text: str) -> List[Dict]:
    """Score and rank all parsed resumes against a job description."""
    jd_skills = extract_skills(jd_text)
    required_years = extract_required_experience_years(jd_text)

    scored = []
    for resume in parsed_resumes:
        if "error" in resume:
            is_skipped = resume["error"].startswith("Skipped:")
            scored.append({
                "file_name": resume["file_name"],
                "candidate_name": "SKIPPED (unsupported format)" if is_skipped else "PARSE ERROR",
                "final_score": 0.0,
                "reasoning": resume["error"] if is_skipped else f"Could not be scored: {resume['error']}",
            })
            continue
        scored.append(score_candidate(resume, jd_text, jd_skills, required_years))

    scored.sort(key=lambda c: c["final_score"], reverse=True)
    for i, c in enumerate(scored, start=1):
        c["rank"] = i

    return scored


if __name__ == "__main__":
    import sys
    import json
    from parser import parse_resumes_from_folder

    if len(sys.argv) != 3:
        print("Usage: python scorer.py <jd_text_file> <resumes_folder>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        jd_text = f.read()

    resumes = parse_resumes_from_folder(sys.argv[2])
    ranked = rank_candidates(resumes, jd_text)
    print(json.dumps(ranked, indent=2, default=str))
