#!/usr/bin/env python3
"""
Milestone 2: Job Matching & Ranking Engine (job_matcher.py)
Ingests unstructured Job Descriptions, vectorizes queries, executes dense-sparse hybrid search,
applies constraint filtering, computes normalized 0-100 scores, and generates structured JSON output with rationales.
"""

import os
import sys
import json
import argparse
import time
from typing import Dict, Any, Optional

from src.vector_store import VectorStoreManager
from src.retriever import HybridRetriever
from src.scorer import MatchRanker

def parse_args():
    parser = argparse.ArgumentParser(description="Milestone 2: Job Matching & Ranking Engine")
    parser.add_argument("--jd_path", type=str, default="data/job_descriptions/jd1_senior_ml_engineer.txt", help="Path to Job Description text file")
    parser.add_argument("--jd_text", type=str, default=None, help="Raw Job Description string")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top candidate matches to retrieve (default: 10)")
    parser.add_argument("--db_dir", type=str, default="data/chroma_db", help="Path to ChromaDB persistent storage")
    parser.add_argument("--min_exp", type=float, default=None, help="Minimum required experience years filter")
    parser.add_argument("--output_json", type=str, default="output_match.json", help="File path to save output JSON payload")
    return parser.parse_args()

def run_job_matcher(
    jd_text: str,
    top_k: int = 10,
    db_dir: str = "data/chroma_db",
    min_exp: Optional[float] = None,
    output_json_path: Optional[str] = "output_match.json"
) -> Dict[str, Any]:
    """
    Executes the job matching, ranking, and explainability pipeline.
    Returns structured JSON payload.
    """
    start_time = time.time()

    print(f"\n========================================================")
    print(f"[INFO] Starting Job Matching & Ranking Engine (Milestone 2)")
    print(f"========================================================")
    print(f"Vector Database Path : {os.path.abspath(db_dir)}")
    print(f"Top-K Candidates Requested : {top_k}")
    if min_exp is not None:
        print(f"Minimum Experience Filter : {min_exp} years")

    vector_store = VectorStoreManager(db_path=db_dir)
    if vector_store.count() == 0:
        print("[WARNING] Vector database is empty! Auto-triggering resume ingestion first...")
        from resume_rag import run_ingestion_pipeline
        run_ingestion_pipeline(data_dir="data/resumes", db_dir=db_dir)

    # 1. Initialize Retriever & Scorer
    retriever = HybridRetriever(vector_store=vector_store, alpha=0.7)
    scorer = MatchRanker(target_jd_text=jd_text)

    # 2. Hybrid Search Retrieval
    raw_candidates = retriever.retrieve_candidates(
        jd_text=jd_text,
        top_k=top_k * 2,
        min_experience_years=min_exp
    )

    # 3. Ranking, 0-100 Scoring & Explainable Reasoning Generation
    top_matches = scorer.process_and_rank_candidates(
        candidates=raw_candidates,
        top_k=top_k
    )

    # 4. JSON Payload Formatting
    payload = scorer.format_output_payload(
        job_description_text=jd_text,
        top_matches=top_matches
    )

    elapsed_time = round(time.time() - start_time, 2)

    # Print Results Summary
    print(f"\n========================================================")
    print(f"[SUCCESS] Candidate Matching Complete! (Query Latency: {elapsed_time}s)")
    print(f"========================================================")
    print(f"Top {len(top_matches)} Matched Candidates:\n")

    for rank, m in enumerate(top_matches, 1):
        print(f"  Rank #{rank:02d} | Candidate: {m['candidate_name']:<20} | Score: {m['match_score']}/100")
        print(f"          Resume Path : {m['resume_path']}")
        print(f"          Matched Skills: {', '.join(m['matched_skills'][:5])}")
        print(f"          Reasoning   : {m['reasoning']}")
        print(f"          ---------------------------------------------------")

    # Write output JSON file
    if output_json_path:
        out_dir = os.path.dirname(output_json_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[INFO] Saved structured JSON response to: {os.path.abspath(output_json_path)}\n")

    return payload

def main():
    args = parse_args()
    
    # Read JD text from path or string arg
    if args.jd_text:
        jd_text = args.jd_text
    elif os.path.exists(args.jd_path):
        with open(args.jd_path, "r", encoding="utf-8", errors="ignore") as f:
            jd_text = f.read()
    else:
        print(f"[ERROR] Job Description file not found at '{args.jd_path}'.")
        sys.exit(1)

    run_job_matcher(
        jd_text=jd_text,
        top_k=args.top_k,
        db_dir=args.db_dir,
        min_exp=args.min_exp,
        output_json_path=args.output_json
    )

if __name__ == "__main__":
    main()
