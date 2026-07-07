"""
main.py
CLI entry point for the Resume Screening Agent.

Usage:
    python src/main.py --jd data/job_description.txt --resumes data/resumes --output output

Produces:
    output/ranked_candidates.json
    output/ranked_candidates.csv
"""

import argparse
import json
import os
import sys

import pandas as pd

from parser import parse_resumes_from_folder
from scorer import rank_candidates


def run(jd_path: str, resumes_dir: str, output_dir: str) -> None:
    if not os.path.isfile(jd_path):
        print(f"Error: job description file not found: {jd_path}")
        sys.exit(1)
    if not os.path.isdir(resumes_dir):
        print(f"Error: resumes folder not found: {resumes_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read()

    print(f"Loading resumes from '{resumes_dir}'...")
    parsed_resumes = parse_resumes_from_folder(resumes_dir)

    skipped = [r for r in parsed_resumes if r.get("error", "").startswith("Skipped:")]
    failed = [r for r in parsed_resumes if "error" in r and not r["error"].startswith("Skipped:")]
    succeeded = [r for r in parsed_resumes if "error" not in r]

    print(f"Found {len(parsed_resumes)} file(s) in folder: "
          f"{len(succeeded)} parsed, {len(failed)} failed to parse, {len(skipped)} skipped (unsupported format).")
    if skipped:
        print(f"  Skipped: {', '.join(r['file_name'] for r in skipped)}")
    if failed:
        print(f"  Failed:  {', '.join(r['file_name'] for r in failed)}")

    print("Scoring candidates against job description...")
    ranked = rank_candidates(parsed_resumes, jd_text)

    json_path = os.path.join(output_dir, "ranked_candidates.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ranked, f, indent=2, default=str)

    csv_path = os.path.join(output_dir, "ranked_candidates.csv")
    df = pd.DataFrame(ranked)
    column_order = [
        "rank", "candidate_name", "file_name", "final_score",
        "semantic_similarity", "skill_match_pct", "experience_match_pct",
        "candidate_experience_years", "candidate_education",
        "matched_skills", "missing_skills", "reasoning",
    ]
    column_order = [c for c in column_order if c in df.columns]
    df = df[column_order]
    df.to_csv(csv_path, index=False)

    print(f"\nDone. Results written to:\n  {json_path}\n  {csv_path}\n")
    print("Top candidates:")
    for c in ranked[:5]:
        print(f"  #{c.get('rank', '-')}  {c.get('candidate_name', '?'):25s}  score={c.get('final_score', 0)}")


def main():
    parser_arg = argparse.ArgumentParser(description="Resume Screening Agent")
    parser_arg.add_argument("--jd", required=True, help="Path to job description text file")
    parser_arg.add_argument("--resumes", required=True, help="Path to folder of resumes (PDF/DOCX/TXT)")
    parser_arg.add_argument("--output", default="output", help="Output folder (default: output)")
    args = parser_arg.parse_args()

    run(args.jd, args.resumes, args.output)


if __name__ == "__main__":
    main()
