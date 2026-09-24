import os
import json
import shutil
import tempfile
import unittest

from resume_rag import run_ingestion_pipeline
from job_matcher import run_job_matcher

class TestEndToEndPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.test_dir, "resumes")
        self.db_dir = os.path.join(self.test_dir, "chroma_db")
        self.output_json = os.path.join(self.test_dir, "output.json")
        os.makedirs(self.data_dir, exist_ok=True)

        # Create 2 test resumes
        resume1 = """Name: Jane Doe
Title: Senior ML Engineer

=== SUMMARY ===
6 years experience in Python, PyTorch, RAG, and ChromaDB.

=== WORK EXPERIENCE ===
Senior ML Engineer (2020 - Present)
Led RAG pipeline development using PyTorch and ChromaDB.
"""
        resume2 = """Name: Bob Smith
Title: Frontend Engineer

=== SUMMARY ===
4 years experience in React, TypeScript, and CSS.

=== WORK EXPERIENCE ===
Frontend Developer (2020 - Present)
Built web apps using React and TypeScript.
"""
        with open(os.path.join(self.data_dir, "jane_doe.txt"), "w", encoding="utf-8") as f:
            f.write(resume1)
        with open(os.path.join(self.data_dir, "bob_smith.txt"), "w", encoding="utf-8") as f:
            f.write(resume2)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_end_to_end_pipeline(self):
        # 1. Milestone 1: Ingestion Pipeline
        stats = run_ingestion_pipeline(data_dir=self.data_dir, db_dir=self.db_dir, reset=True)
        self.assertEqual(stats["resumes_processed"], 2)
        self.assertGreater(stats["total_chunks_created"], 0)

        # 2. Milestone 2: Job Matching Query
        jd_text = "Looking for a Senior ML Engineer with 5+ years experience in PyTorch, Python, and RAG architectures."
        payload = run_job_matcher(
            jd_text=jd_text,
            top_k=2,
            db_dir=self.db_dir,
            output_json_path=self.output_json
        )

        # 3. JSON Schema Validation
        self.assertIn("job_description", payload)
        self.assertIn("top_matches", payload)
        self.assertEqual(len(payload["top_matches"]), 2)

        top = payload["top_matches"][0]
        self.assertEqual(top["candidate_name"], "Jane Doe")
        self.assertGreaterEqual(top["match_score"], 0)
        self.assertLessEqual(top["match_score"], 100)
        self.assertIn("PyTorch", top["matched_skills"])
        self.assertGreater(len(top["relevant_excerpts"]), 0)
        self.assertIn("match", top["reasoning"].lower())

        # Verify output JSON file exists and is valid
        self.assertTrue(os.path.exists(self.output_json))
        with open(self.output_json, "r", encoding="utf-8") as f:
            saved_json = json.load(f)
        self.assertEqual(saved_json["top_matches"][0]["candidate_name"], "Jane Doe")

if __name__ == "__main__":
    unittest.main()
