import os
import sys
import time
import json
import math

from src.chunker import ResumeChunker
from src.metadata_extractor import MetadataExtractor
from src.vector_store import VectorStoreManager
from src.retriever import HybridRetriever
from src.scorer import MatchRanker

def run_benchmarks():
    print("========================================================")
    print("[INFO] Phase 4: Evaluation, Benchmarking & Accuracy Metrics")
    print("========================================================")

    data_dir = "data/resumes"
    jd_dir = "data/job_descriptions"
    db_dir = "data/chroma_db"

    # 1. Ingestion & Chunking Latency
    chunker = ResumeChunker()
    resume_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt')]

    chunking_times = []
    chunk_counts = []

    for fpath in resume_files:
        t0 = time.time()
        raw_text = chunker.extract_text_from_file(fpath)
        chunks = chunker.chunk_resume(fpath)
        t1 = time.time()
        chunking_times.append(t1 - t0)
        chunk_counts.append(len(chunks))

    avg_chunk_ms = (sum(chunking_times) / len(chunking_times)) * 1000 if chunking_times else 0.0

    print(f"\n1. Ingestion Latency Benchmark:")
    print(f"    - Resumes Evaluated       : {len(resume_files)}")
    print(f"    - Total Chunks Generated  : {sum(chunk_counts)}")
    print(f"    - Avg Chunking Latency    : {avg_chunk_ms:.2f} ms / resume")

    # 2. Retrieval Latency Benchmark
    vector_store = VectorStoreManager(db_path=db_dir)
    retriever = HybridRetriever(vector_store=vector_store, alpha=0.7)

    jd_files = sorted([os.path.join(jd_dir, f) for f in os.listdir(jd_dir) if f.endswith('.txt')])
    retrieval_latencies = []

    for jdf in jd_files:
        with open(jdf, 'r', encoding='utf-8') as f:
            jd_text = f.read()
        t0 = time.time()
        results = retriever.retrieve_candidates(jd_text=jd_text, top_k=10)
        t1 = time.time()
        retrieval_latencies.append((t1 - t0) * 1000)

    avg_query_ms = sum(retrieval_latencies) / len(retrieval_latencies) if retrieval_latencies else 0.0
    print(f"\n2. Retrieval Latency Benchmark:")
    print(f"    - Queries Evaluated       : {len(jd_files)}")
    print(f"    - Avg Query Latency       : {avg_query_ms:.2f} ms")

    # 3. Accuracy Evaluation (Hit Rate@K, MRR@K, Precision@K)
    ground_truth_map = {
        "jd1_senior_ml_engineer.txt": ["jane_doe_sr_ml", "alex_rivera_ml_researcher", "michael_chang_rag_dev", "laura_bennett_ml"],
        "jd2_frontend_lead.txt": ["emily_watson_sr_frontend", "david_kim_staff_frontend", "nathan_drake_mobile", "sophia_martinez_ui", "chloe_adams_fe"],
        "jd3_devops_sre_engineer.txt": ["robert_chen_sr_devops", "priya_sharma_sre", "olivia_parker_devops", "liam_neeson_sre"],
        "jd4_data_engineering_lead.txt": ["carlos_mendoza_data_lead", "amanda_foster_sr_data_eng", "maya_lin_data"],
        "jd5_product_manager_ai.txt": ["jessica_taylor_sr_pm", "kevin_patel_pm"]
    }

    print(f"\n3. Accuracy Assessment Across Top-K Candidates:")
    for K in [3, 5, 10]:
        hit_rates = []
        mrr_list = []
        precision_list = []

        for jd_file, ground_truth in ground_truth_map.items():
            fpath = os.path.join(jd_dir, jd_file)
            with open(fpath, 'r', encoding='utf-8') as f:
                jd_text = f.read()
            results = retriever.retrieve_candidates(jd_text=jd_text, top_k=K)
            retrieved_ids = [r['resume_id'] for r in results]

            hits = set(retrieved_ids).intersection(set(ground_truth))
            hit_rates.append(1.0 if len(hits) > 0 else 0.0)
            precision_list.append(len(hits) / float(K))

            rr = 0.0
            for rank_idx, r_id in enumerate(retrieved_ids, 1):
                if r_id in ground_truth:
                    rr = 1.0 / rank_idx
                    break
            mrr_list.append(rr)

        avg_hr = (sum(hit_rates) / len(hit_rates)) * 100
        avg_mrr = sum(mrr_list) / len(mrr_list)
        avg_prec = (sum(precision_list) / len(precision_list)) * 100

        print(f"    - K = {K:02d} | Hit Rate: {avg_hr:.1f}% | MRR: {avg_mrr:.4f} | Precision: {avg_prec:.1f}%")

    # 4. Ablation Study
    print(f"\n4. Ablation Study (Dense vs Sparse vs Hybrid):")
    configs = [
        ("Dense-Only (Alpha=1.0)", 1.0),
        ("Sparse-Only (Alpha=0.0)", 0.0),
        ("Hybrid Search (Alpha=0.7)", 0.7)
    ]
    for label, alpha in configs:
        test_retriever = HybridRetriever(vector_store=vector_store, alpha=alpha)
        mrr_list = []
        hit_rates = []
        for jd_file, ground_truth in ground_truth_map.items():
            fpath = os.path.join(jd_dir, jd_file)
            with open(fpath, 'r', encoding='utf-8') as f:
                jd_text = f.read()
            results = test_retriever.retrieve_candidates(jd_text=jd_text, top_k=5)
            retrieved_ids = [r['resume_id'] for r in results]
            hits = set(retrieved_ids).intersection(set(ground_truth))
            hit_rates.append(1.0 if len(hits) > 0 else 0.0)
            rr = 0.0
            for rank_idx, r_id in enumerate(retrieved_ids, 1):
                if r_id in ground_truth:
                    rr = 1.0 / rank_idx
                    break
            mrr_list.append(rr)
        avg_hr = (sum(hit_rates) / len(hit_rates)) * 100
        avg_mrr = sum(mrr_list) / len(mrr_list)
        print(f"    - {label:<25} | Hit Rate@5: {avg_hr:.1f}% | MRR: {avg_mrr:.4f}")

    print("\n========================================================")
    print("[SUCCESS] Evaluation Benchmarking Completed Successfully!")
    print("========================================================\n")

if __name__ == "__main__":
    run_benchmarks()
