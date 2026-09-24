import os
import shutil
import tempfile
import unittest
from src.chunker import ResumeChunker
from src.metadata_extractor import MetadataExtractor
from src.vector_store import VectorStoreManager

class TestRAGSetup(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_resume_chunker_section_preservation(self):
        sample_text = """Name: John Smith
Title: Senior Software Engineer

=== SUMMARY ===
Experienced software engineer with 5 years building scalable Python systems.

=== WORK EXPERIENCE ===
Software Engineer at TechCorp (2020 - Present)
- Developed REST microservices using Python and FastAPI.
- Implemented ChromaDB search vector embeddings.

=== EDUCATION ===
B.S. in Computer Science, MIT (2016 - 2020)

=== SKILLS ===
Python, FastAPI, Docker, ChromaDB, SQL
"""
        resume_file = os.path.join(self.test_dir, "sample_resume.txt")
        with open(resume_file, "w", encoding="utf-8") as f:
            f.write(sample_text)

        chunker = ResumeChunker()
        chunks = chunker.chunk_resume(resume_file)

        self.assertGreaterEqual(len(chunks), 4)
        section_names = [c["section_name"] for c in chunks]
        self.assertIn("SUMMARY", section_names)
        self.assertIn("WORK EXPERIENCE", section_names)
        self.assertIn("EDUCATION", section_names)
        self.assertIn("SKILLS", section_names)

    def test_metadata_extractor(self):
        sample_text = """Candidate Name: Alice Johnson
Total Experience: 6 years

=== WORK EXPERIENCE ===
Senior ML Engineer (2018 - 2024)
Extensive hands-on experience with PyTorch, Transformers, RAG, and Kubernetes.

=== EDUCATION ===
Master of Science in Artificial Intelligence
"""
        extractor = MetadataExtractor()
        meta = extractor.extract_metadata(sample_text, "alice_johnson_ml.txt")

        self.assertEqual(meta["candidate_name"], "Alice Johnson")
        self.assertEqual(meta["total_experience_years"], 6.0)
        self.assertIn("PyTorch", meta["skills"])
        self.assertIn("Kubernetes", meta["skills"])
        self.assertEqual(meta["highest_degree"], "Master's")

    def test_vector_store_indexing(self):
        db_dir = os.path.join(self.test_dir, "chroma_test_db")
        vm = VectorStoreManager(db_path=db_dir)

        metadata = {
            "resume_id": "test_candidate_1",
            "resume_path": "test_path.txt",
            "candidate_name": "Test Candidate",
            "total_experience_years": 4.0,
            "highest_degree": "Bachelor's",
            "skills_str": "Python, Docker, SQL"
        }

        chunks = [
            {"section_name": "WORK EXPERIENCE", "content": "Built Python apps using Docker and SQL."},
            {"section_name": "SKILLS", "content": "Python, Docker, SQL"}
        ]

        vm.add_resume_chunks(chunks=chunks, metadata=metadata)
        self.assertEqual(vm.count(), 2)

        results = vm.query_similarity(query_text="Python Docker microservices", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertIn("content", results[0])

if __name__ == "__main__":
    unittest.main()
