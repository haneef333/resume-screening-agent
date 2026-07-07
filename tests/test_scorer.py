"""
Unit tests for scorer.py.
Run from project root with: pytest tests/
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from scorer import (
    extract_required_experience_years,
    skill_match_score,
    experience_match_score,
    rank_candidates,
    get_similarity_engine,
)
from parser import parse_resumes_from_folder


class TestExtractRequiredExperienceYears:
    def test_extracts_minimum_years_required(self):
        jd = "We require at least 2 years of experience in data science."
        assert extract_required_experience_years(jd) == 2.0

    def test_no_requirement_returns_zero(self):
        jd = "We are looking for a passionate data scientist."
        assert extract_required_experience_years(jd) == 0.0

    def test_takes_minimum_when_multiple_mentions(self):
        jd = "5 years of experience preferred, 2 years of experience minimum required."
        assert extract_required_experience_years(jd) == 2.0


class TestSkillMatchScore:
    def test_full_match_scores_one(self):
        result = skill_match_score(["python", "sql", "aws"], ["python", "sql"])
        assert result["score"] == 1.0
        assert result["matched"] == ["python", "sql"]
        assert result["missing"] == []

    def test_partial_match(self):
        result = skill_match_score(["python"], ["python", "sql"])
        assert result["score"] == 0.5
        assert result["matched"] == ["python"]
        assert result["missing"] == ["sql"]

    def test_no_match(self):
        result = skill_match_score(["excel"], ["python", "sql"])
        assert result["score"] == 0.0

    def test_no_jd_skills_returns_zero_not_error(self):
        result = skill_match_score(["python"], [])
        assert result["score"] == 0.0


class TestExperienceMatchScore:
    def test_meets_requirement_scores_full(self):
        assert experience_match_score(candidate_years=5, required_years=3) == 1.0

    def test_exactly_meets_requirement(self):
        assert experience_match_score(candidate_years=3, required_years=3) == 1.0

    def test_below_requirement_gives_partial_credit(self):
        score = experience_match_score(candidate_years=1.5, required_years=3)
        assert 0.0 < score < 1.0
        assert score == pytest.approx(0.5)

    def test_no_requirement_stated_is_neutral(self):
        assert experience_match_score(candidate_years=0, required_years=0) == 0.75

    def test_zero_experience_below_requirement(self):
        assert experience_match_score(candidate_years=0, required_years=2) == 0.0


class TestRankCandidatesEndToEnd:
    """
    Integration-style tests against the real sample data shipped in
    data/resumes/ and data/job_description.txt.
    """

    SAMPLE_RESUMES = os.path.join(os.path.dirname(__file__), "..", "data", "resumes")
    SAMPLE_JD = os.path.join(os.path.dirname(__file__), "..", "data", "job_description.txt")

    @pytest.fixture(scope="class")
    @classmethod
    def ranked(cls):
        with open(cls.SAMPLE_JD, "r", encoding="utf-8") as f:
            jd_text = f.read()
        resumes = parse_resumes_from_folder(cls.SAMPLE_RESUMES)
        return rank_candidates(resumes, jd_text)

    def test_all_resumes_scored(self, ranked):
        assert len(ranked) == 12

    def test_ranks_are_sequential(self, ranked):
        ranks = [c["rank"] for c in ranked]
        assert ranks == list(range(1, len(ranked) + 1))

    def test_sorted_descending_by_score(self, ranked):
        scores = [c["final_score"] for c in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_strong_fit_outranks_weak_fit(self, ranked):
        """
        Sanity check the ranking is actually meaningful: a senior data
        scientist should clearly outrank a marketing manager against a
        Data Scientist JD.
        """
        by_name = {c["candidate_name"]: c["final_score"] for c in ranked}
        assert by_name["Aarav Mehta"] > by_name["Arjun Kapoor"]
        assert by_name["Aarav Mehta"] > by_name["Karan Shah"]

    def test_every_candidate_has_reasoning(self, ranked):
        assert all(c.get("reasoning") for c in ranked)

    def test_similarity_method_is_recorded(self, ranked):
        assert all(c.get("similarity_method") in ("embeddings", "tfidf") for c in ranked)


class TestSimilarityEngine:
    def test_engine_selects_a_valid_method(self):
        engine = get_similarity_engine()
        assert engine.method in ("embeddings", "tfidf")

    def test_identical_text_scores_near_one(self):
        engine = get_similarity_engine()
        text = "Python developer with machine learning experience"
        score = engine.score(text, text)
        assert score > 0.9

    def test_unrelated_text_scores_low(self):
        engine = get_similarity_engine()
        score = engine.score(
            "Professional chef specializing in pastry and baking",
            "Senior Java backend engineer with Kubernetes experience",
        )
        assert score < 0.3
