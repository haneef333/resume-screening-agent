"""
Unit tests for parser.py.
Run from project root with: pytest tests/
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from parser import (
    extract_skills,
    extract_experience_years,
    extract_education,
    extract_name,
    parse_resume,
)


class TestExtractSkills:
    def test_matches_known_skills(self):
        text = "Experienced in Python, SQL, and Machine Learning."
        skills = extract_skills(text)
        assert "python" in skills
        assert "sql" in skills
        assert "machine learning" in skills

    def test_case_insensitive(self):
        text = "PYTHON and pandas and NumPy"
        skills = extract_skills(text)
        assert "python" in skills
        assert "pandas" in skills
        assert "numpy" in skills

    def test_cpp_does_not_false_positive_on_bare_c(self):
        """Regression test: 'c++' alias must not match every 'c' character."""
        text = "I have experience in customer service and communication."
        skills = extract_skills(text)
        assert "c++" not in skills

    def test_cpp_matches_when_actually_present(self):
        text = "Proficient in C++ and Java."
        skills = extract_skills(text)
        assert "c++" in skills
        assert "java" in skills

    def test_r_does_not_match_inside_other_words(self):
        """Regression test: single-letter alias 'r' must not match inside 'recruiter'."""
        text = "Worked closely with the recruiter and hiring manager."
        skills = extract_skills(text)
        assert "r" not in skills

    def test_r_matches_as_standalone_language(self):
        text = "Statistical modeling using R and Python."
        skills = extract_skills(text)
        assert "r" in skills

    def test_alias_matches_canonical_skill(self):
        text = "Built models using sklearn and TF."
        skills = extract_skills(text)
        assert "scikit-learn" in skills
        assert "tensorflow" in skills

    def test_no_skills_found_returns_empty_list(self):
        text = "This text mentions nothing technical at all."
        skills = extract_skills(text)
        assert skills == []


class TestExtractExperienceYears:
    def test_explicit_years_statement(self):
        text = "Data scientist with 5 years of experience in ML."
        assert extract_experience_years(text) == 5.0

    def test_explicit_years_with_plus(self):
        text = "3+ years of experience building models."
        assert extract_experience_years(text) == 3.0

    def test_takes_max_of_multiple_explicit_mentions(self):
        text = "5 years of experience overall, including 2 years of experience in NLP."
        assert extract_experience_years(text) == 5.0

    def test_falls_back_to_date_range_when_no_explicit_statement(self):
        text = "Data Scientist, TechCorp — 2020 - 2023"
        years = extract_experience_years(text)
        assert years == 3.0

    def test_present_is_treated_as_current_year(self):
        text = "Software Engineer — 2020 - Present"
        years = extract_experience_years(text)
        assert years > 0  # exact value depends on current year, just check it's computed

    def test_no_experience_info_returns_zero(self):
        text = "A resume with no dates or years mentioned anywhere."
        assert extract_experience_years(text) == 0.0

    def test_ignores_years_unrelated_to_experience(self):
        """
        Regression test: a prior version of this regex had the
        'experience'/'exp' keyword as optional, so ANY "N years" mention
        (age, company founding date, revenue growth period, etc.) was
        incorrectly counted as work experience. This must return 0.0.
        """
        assert extract_experience_years("I turned 25 years old last week.") == 0.0
        assert extract_experience_years("The company was founded 10 years ago.") == 0.0
        assert extract_experience_years("Grew revenue over a 4 years period.") == 0.0

    def test_still_matches_real_experience_mentions_after_fix(self):
        assert extract_experience_years("5 years of experience in ML.") == 5.0
        assert extract_experience_years("3+ years of experience building models.") == 3.0


class TestExtractEducation:
    def test_detects_phd(self):
        assert extract_education("PhD in Statistics, 2023") == "PhD"

    def test_detects_masters(self):
        assert extract_education("M.Tech in Artificial Intelligence") == "Master's"

    def test_detects_bachelors(self):
        assert extract_education("B.Tech in Computer Science, 2020") == "Bachelor's"

    def test_prioritizes_highest_degree_when_multiple_present(self):
        text = "B.Tech in Computer Science, 2018. PhD in Statistics, 2023."
        assert extract_education(text) == "PhD"

    def test_no_education_found(self):
        text = "No degree information mentioned here."
        assert extract_education(text) == "Not specified"


class TestExtractName:
    def test_picks_first_plausible_line(self):
        text = "Aarav Mehta\nData Scientist\n\nSummary\n..."
        assert extract_name(text) == "Aarav Mehta"

    def test_skips_section_headers(self):
        text = "Resume\nPriya Nair\nMachine Learning Engineer"
        assert extract_name(text) == "Priya Nair"

    def test_falls_back_when_nothing_plausible(self):
        text = "12345\n9999999999\n"
        assert extract_name(text) == "Unknown Candidate"


class TestParseResume:
    SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "resumes")

    def test_parses_txt_resume_end_to_end(self):
        path = os.path.join(self.SAMPLE_DIR, "resume_01_aarav_mehta.txt")
        result = parse_resume(path)
        assert result["candidate_name"] == "Aarav Mehta"
        assert "python" in result["skills"]
        assert result["experience_years"] > 0
        assert result["education"] in ("Bachelor's", "Master's", "PhD", "Diploma", "High School", "Not specified")

    def test_parses_pdf_resume(self):
        path = os.path.join(self.SAMPLE_DIR, "resume_11_rohan_bhatt.pdf")
        result = parse_resume(path)
        assert result["file_name"] == "resume_11_rohan_bhatt.pdf"
        assert len(result["skills"]) > 0

    def test_parses_docx_resume(self):
        path = os.path.join(self.SAMPLE_DIR, "resume_12_isha_choudhary.docx")
        result = parse_resume(path)
        assert result["file_name"] == "resume_12_isha_choudhary.docx"
        assert len(result["skills"]) > 0

    def test_unsupported_file_type_raises(self):
        with pytest.raises(ValueError):
            parse_resume("somefile.xyz")
