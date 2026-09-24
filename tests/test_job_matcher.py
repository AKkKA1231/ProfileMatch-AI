import os
import shutil
import tempfile
import unittest
from src.vector_store import VectorStoreManager
from src.retriever import HybridRetriever
from src.scorer import MatchRanker

class TestJobMatcher(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_dir = os.path.join(self.test_dir, "chroma_db")
        self.vector_store = VectorStoreManager(db_path=self.db_dir)

        # Index sample resume candidates
        candidates = [
            {
                "metadata": {
                    "resume_id": "jane_doe",
                    "resume_path": "data/resumes/jane_doe_sr_ml.txt",
                    "candidate_name": "Jane Doe",
                    "total_experience_years": 6.0,
                    "highest_degree": "Master's",
                    "skills_str": "Python, PyTorch, Retrieval-Augmented Generation, Transformers, ChromaDB, Kubernetes"
                },
                "chunks": [
                    {"section_name": "SUMMARY", "content": "Senior ML Engineer with 6 years experience building RAG pipelines and PyTorch models."},
                    {"section_name": "WORK EXPERIENCE", "content": "Led deployment of enterprise RAG pipelines reducing query latency using ChromaDB and PyTorch on Kubernetes."}
                ]
            },
            {
                "metadata": {
                    "resume_id": "david_kim",
                    "resume_path": "data/resumes/david_kim_staff_frontend.txt",
                    "candidate_name": "David Kim",
                    "total_experience_years": 7.0,
                    "highest_degree": "Bachelor's",
                    "skills_str": "React, TypeScript, Next.js, Redux, Tailwind CSS"
                },
                "chunks": [
                    {"section_name": "SUMMARY", "content": "Staff Frontend Engineer with 7 years expertise in React and TypeScript."},
                    {"section_name": "WORK EXPERIENCE", "content": "Architected core Next.js web application reducing page load time."}
                ]
            }
        ]

        for cand in candidates:
            self.vector_store.add_resume_chunks(chunks=cand["chunks"], metadata=cand["metadata"])

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_hybrid_retrieval(self):
        retriever = HybridRetriever(vector_store=self.vector_store, alpha=0.7)
        jd_text = "Looking for a Senior ML Engineer with PyTorch, RAG pipelines, and vector database experience."

        candidates = retriever.retrieve_candidates(jd_text=jd_text, top_k=2)
        self.assertGreater(len(candidates), 0)
        top_cand = candidates[0]
        self.assertEqual(top_cand["candidate_name"], "Jane Doe")

    def test_match_scorer_and_json_schema(self):
        jd_text = "Senior Machine Learning Engineer specializing in RAG, PyTorch, and distributed training. Minimum 5+ years experience required."
        retriever = HybridRetriever(vector_store=self.vector_store, alpha=0.7)
        candidates = retriever.retrieve_candidates(jd_text=jd_text, top_k=2)

        scorer = MatchRanker(target_jd_text=jd_text)
        top_matches = scorer.process_and_rank_candidates(candidates=candidates, top_k=2)

        self.assertEqual(len(top_matches), 2)
        top = top_matches[0]

        # Verify output keys
        self.assertIn("candidate_name", top)
        self.assertIn("resume_path", top)
        self.assertIn("match_score", top)
        self.assertIn("matched_skills", top)
        self.assertIn("relevant_excerpts", top)
        self.assertIn("reasoning", top)

        # Verify score normalization (0-100)
        self.assertGreaterEqual(top["match_score"], 0)
        self.assertLessEqual(top["match_score"], 100)

        # Verify structured JSON payload schema
        payload = scorer.format_output_payload(job_description_text=jd_text, top_matches=top_matches)
        self.assertIn("job_description", payload)
        self.assertIn("top_matches", payload)
        self.assertIsInstance(payload["top_matches"], list)

if __name__ == "__main__":
    unittest.main()
