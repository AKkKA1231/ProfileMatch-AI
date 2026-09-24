import re
import math
from typing import List, Dict, Any, Optional
from src.vector_store import VectorStoreManager

class HybridRetriever:
    """
    Hybrid Search Strategy combining Dense Semantic Vector Search with Sparse BM25 Keyword Search.
    Fuses similarity scores using weighted scoring fusion.
    """

    def __init__(self, vector_store: VectorStoreManager, alpha: float = 0.7):
        self.vector_store = vector_store
        self.alpha = alpha  # Weight for dense vector similarity (1 - alpha for sparse BM25)

    def retrieve_candidates(self, jd_text: str, top_k: int = 10, min_experience_years: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Executes hybrid search across vector store and ranks candidate chunks.
        Applies metadata filters (e.g. minimum experience years).
        """
        # Build where filter if experience constraint is specified
        where_filter = None
        if min_experience_years is not None and min_experience_years > 0:
            where_filter = {"total_experience_years": {"$gte": float(min_experience_years)}}

        # 1. Dense Semantic Vector Retrieval
        dense_results = self.vector_store.query_similarity(query_text=jd_text, top_k=top_k * 3, where_filter=where_filter)

        if not dense_results:
            return []

        # 2. Extract JD keywords for Sparse BM25 Matching
        jd_keywords = self._extract_keywords(jd_text)

        # 3. Aggregate Chunks by Candidate / Resume ID and Compute Hybrid Fusion Score
        candidate_map = {}

        for item in dense_results:
            meta = item["metadata"]
            resume_id = meta.get("resume_id", "unknown")
            cand_name = meta.get("candidate_name", "Unknown")
            content = item.get("content", "")
            dense_score = item.get("similarity_score", 0.0)

            # Calculate Sparse Keyword Score (BM25 / Jaccard keyword overlap)
            chunk_keywords = set(self._extract_keywords(content))
            matched_keywords = list(set(jd_keywords).intersection(chunk_keywords))
            
            sparse_score = 0.0
            if jd_keywords:
                sparse_score = len(matched_keywords) / math.sqrt(len(set(jd_keywords)))
                sparse_score = min(1.0, sparse_score)

            # Compute Combined Hybrid Score
            hybrid_score = (self.alpha * dense_score) + ((1.0 - self.alpha) * sparse_score)

            if resume_id not in candidate_map:
                candidate_map[resume_id] = {
                    "resume_id": resume_id,
                    "candidate_name": cand_name,
                    "resume_path": meta.get("resume_path", ""),
                    "total_experience_years": meta.get("total_experience_years", 0.0),
                    "highest_degree": meta.get("highest_degree", "Bachelor's"),
                    "skills_str": meta.get("skills_str", ""),
                    "best_hybrid_score": hybrid_score,
                    "best_dense_score": dense_score,
                    "best_sparse_score": sparse_score,
                    "chunks": [],
                    "matched_keywords": set()
                }

            # Update highest score for candidate
            if hybrid_score > candidate_map[resume_id]["best_hybrid_score"]:
                candidate_map[resume_id]["best_hybrid_score"] = hybrid_score
                candidate_map[resume_id]["best_dense_score"] = dense_score
                candidate_map[resume_id]["best_sparse_score"] = sparse_score

            candidate_map[resume_id]["chunks"].append({
                "section_name": meta.get("section_name", "GENERAL"),
                "content": content,
                "score": hybrid_score
            })
            candidate_map[resume_id]["matched_keywords"].update(matched_keywords)

        # Sort candidate profiles by hybrid score
        ranked_candidates = list(candidate_map.values())
        ranked_candidates.sort(key=lambda x: x["best_hybrid_score"], reverse=True)

        for c in ranked_candidates:
            c["matched_keywords"] = sorted(list(c["matched_keywords"]))

        return ranked_candidates[:top_k]

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful technical terms and keywords from text."""
        words = re.findall(r"\b[A-Za-z0-9+#.-]{2,30}\b", text.lower())
        stopwords = {"and", "the", "with", "for", "in", "to", "of", "a", "an", "is", "are", "on", "at", "by", "from", "be", "or", "as", "our", "we", "you", "your", "required", "experience", "looking", "role", "overview", "responsibilities"}
        return [w for w in words if w not in stopwords and not w.isdigit()]
