#!/usr/bin/env python3
"""
Milestone 1: RAG System Setup (resume_rag.py)
Ingests varied resume formats (.pdf, .docx, .txt), executes section-aware chunking,
extracts structured candidate metadata, and indexes vector embeddings in ChromaDB.
"""

import os
import sys
import argparse
import time
from typing import List, Dict, Any

from src.chunker import ResumeChunker
from src.metadata_extractor import MetadataExtractor
from src.vector_store import VectorStoreManager

def parse_args():
    parser = argparse.ArgumentParser(description="Milestone 1: Resume RAG Ingestion & Vector Indexing Engine")
    parser.add_argument("--data_dir", type=str, default="data/resumes", help="Path to resume corpus directory")
    parser.add_argument("--db_dir", type=str, default="data/chroma_db", help="Path to ChromaDB persistent storage")
    parser.add_argument("--reset", action="store_true", help="Reset existing vector database before indexing")
    return parser.parse_args()

def run_ingestion_pipeline(data_dir: str, db_dir: str, reset: bool = False) -> Dict[str, Any]:
    """
    Executes the end-to-end ingestion and indexing pipeline over all resumes in data_dir.
    """
    start_time = time.time()
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Resume directory '{data_dir}' does not exist.")

    print(f"\n========================================================")
    print(f"[INFO] Starting RAG Ingestion Pipeline (Milestone 1)")
    print(f"========================================================")
    print(f"Resume Source Directory : {os.path.abspath(data_dir)}")
    print(f"Vector Database Path     : {os.path.abspath(db_dir)}")

    chunker = ResumeChunker()
    metadata_extractor = MetadataExtractor()
    vector_store = VectorStoreManager(db_path=db_dir)

    if reset:
        print("[INFO] Resetting existing vector collection...")
        vector_store.reset_collection()

    supported_extensions = {".pdf", ".docx", ".txt"}
    files = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if os.path.splitext(f)[1].lower() in supported_extensions
    ]

    print(f"[INFO] Found {len(files)} resume documents for processing.\n")

    total_chunks = 0
    indexed_resumes = 0
    all_extracted_skills = set()

    for idx, file_path in enumerate(files, 1):
        filename = os.path.basename(file_path)
        try:
            raw_text = chunker.extract_text_from_file(file_path)
            metadata = metadata_extractor.extract_metadata(raw_text, file_path)
            chunks = chunker.chunk_resume(file_path)

            vector_store.add_resume_chunks(chunks=chunks, metadata=metadata)

            total_chunks += len(chunks)
            indexed_resumes += 1
            all_extracted_skills.update(metadata.get("skills", []))

            print(f"  [{idx:02d}/{len(files):02d}] Indexed '{filename}' | Candidate: {metadata['candidate_name']} ({metadata['total_experience_years']} yrs exp) | Chunks: {len(chunks)}")
        except Exception as e:
            print(f"  [ERROR] Failed to process '{filename}': {e}")

    elapsed_time = time.time() - start_time
    total_indexed_vectors = vector_store.count()

    stats = {
        "status": "success",
        "resumes_processed": indexed_resumes,
        "total_chunks_created": total_chunks,
        "total_indexed_vectors": total_indexed_vectors,
        "unique_skills_extracted": len(all_extracted_skills),
        "elapsed_seconds": round(elapsed_time, 2)
    }

    print(f"\n========================================================")
    print(f"[SUCCESS] Ingestion & Indexing Pipeline Complete!")
    print(f"========================================================")
    print(f"Resumes Ingested     : {stats['resumes_processed']} / {len(files)}")
    print(f"Chunks Generated     : {stats['total_chunks_created']}")
    print(f"Total Vector Items   : {stats['total_indexed_vectors']}")
    print(f"Unique Skills Tagged : {stats['unique_skills_extracted']}")
    print(f"Pipeline Execution   : {stats['elapsed_seconds']}s")
    print(f"========================================================\n")

    return stats

def main():
    args = parse_args()
    run_ingestion_pipeline(data_dir=args.data_dir, db_dir=args.db_dir, reset=args.reset)

if __name__ == "__main__":
    main()
