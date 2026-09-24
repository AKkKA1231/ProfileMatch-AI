# Problem Statement: Intelligent Resume-to-Job Matching Engine using Retrieval-Augmented Generation (RAG)

## 1. Executive Summary
Traditional applicant tracking systems (ATS) largely rely on rigid keyword searches and surface-level parsing, often failing to capture candidate nuances, transferable skills, and semantic relevance to complex job descriptions (JDs). This project involves building an automated, context-aware **Resume-to-Job Matching Engine** utilizing a Retrieval-Augmented Generation (RAG) architecture. The engine semantically parses unstructured resumes, generates vector embeddings, persists contextual metadata, and applies hybrid search algorithms to surface and explain candidate suitability with high accuracy.

---

## 2. Background & Problem Context
Recruitment teams process hundreds of incoming resumes across diverse document layouts and formats. Standard keyword matching frequently introduces two distinct failure modes:
* **False Negatives:** Qualified candidates are filtered out simply due to phrasing variations (e.g., using "NLP specialist" instead of "LLM engineer").
* **False Positives:** Candidates are matched purely on keyword density despite lacking real-world role context or requisite depth of experience.

To solve this, the proposed system leverages semantic chunking, vector embeddings, hybrid retrieval (dense semantic vectors combined with sparse keyword matching), and dynamic metadata filtering.

---

## 3. Project Objectives & Core Requirements

### Milestone 1: RAG System Setup (`resume_rag.py`) — 50%
* **Document Processing & Section Preservation:** Ingest varied resume formats (`.pdf`, `.docx`, `.txt`) using local file system tooling. Implement structural chunking designed to preserve critical logical sections (e.g., Summary, Work Experience, Education, Skills, Certifications) rather than naive fixed-size character splitting.
* **Embeddings & Vector Store:** Generate high-dimensional vector representations using production embedding models (OpenAI, Cohere, or Hugging Face) and index them in a scalable vector store (ChromaDB, Pinecone, or Weaviate).
* **Automated Metadata Extraction:** Extract structured metadata entities—such as `candidate_name`, `skills`, `experience_years`, and `education`—and store them alongside document chunks to enable precise filtering at query time.

### Milestone 2: Job Matching & Ranking Engine (`job_matcher.py`) — 50%
* **Query Vectorization & Retrieval:** Ingest unstructured job descriptions, vectorize the context, and retrieve the top-$K$ candidate resumes ($K = 10$).
* **Hybrid Search Strategy:** Integrate dense semantic vector search with sparse keyword validation for non-negotiable hard skills and specific tooling requirements.
* **Ranking, Filtering & Explainability:** 
  * Apply strict pre- or post-retrieval filters for must-have constraints (e.g., minimum years of experience, mandatory programming languages).
  * Compute a normalized alignment score on a scale of `0–100`.
  * Output actionable rationale explaining *why* the candidate matched, citing explicit sections and skill overlaps.

---

## 4. Input & Output Specifications

### 4.1 System Input
* A textual or file-based Job Description (JD) specifying role responsibilities, experience levels, and required toolsets.

### 4.2 System Output
The system must generate a structured JSON payload adhering to the following schema:

```json
{
  "job_description": "Senior Machine Learning Engineer specializing in LLMs and distributed training...",
  "top_matches": [
    {
      "candidate_name": "Jane Doe",
      "resume_path": "resumes/jane_doe.pdf",
      "match_score": 92,
      "matched_skills": ["Python", "PyTorch", "Retrieval-Augmented Generation", "Transformers"],
      "relevant_excerpts": [
        "Led deployment of enterprise RAG pipelines reducing query latency by 40%...",
        "5+ years scaling distributed training workloads on Kubernetes."
      ],
      "reasoning": "Strong match: exceeds 5+ years requirement in Python and distributed ML frameworks; exhibits direct production experience with RAG architectures."
    }
  ]
}


---

## 5. Scope & Dataset Requirements
* **Resume Corpus:** Minimum of 30 diverse resumes spanning distinct disciplines, experience bands, and document layouts.
* **Job Descriptions:** Minimum of 5 comprehensive job descriptions targeting different technical and domain-specific roles.
* **Evaluation & Experimentation Notebook:** A dedicated Jupyter Notebook containing:
  * Chunking and embedding latency benchmarks.
  * Retrieval accuracy assessments (e.g., Hit Rate, MRR, or Precision@K).
  * Impact analysis of hybrid search versus semantic-only search.

---

## 6. Deliverables & Submission Requirements
* **Complete RAG Implementation:** Production-ready scripts including `resume_rag.py` and `job_matcher.py`.
* **Curated Dataset:** Directory containing 30+ diverse resumes and 5+ job descriptions.
* **Jupyter Notebook:** Full experimentation, benchmarking, latency, and retrieval accuracy analysis.
* **GitHub Repository:** Clean code repository containing all source code, dataset directories, and a clear `README.md`.
* **Demo Video:** 3–4 minute recorded walkthrough demonstrating end-to-end ingestion, retrieval, and JSON output generation.

---

## 7. Deliverables & Evaluation Matrix

| Deliverable | Description | Weight / Priority |
| :--- | :--- | :--- |
| `resume_rag.py` | Ingestion pipeline, section-aware chunker, metadata extractor, and vector database indexer. | 50% (Part A) |
| `job_matcher.py` | Hybrid search engine, metadata filter, 0–100 ranking logic, and reasoning generator. | 50% (Part B) |
| **Analysis Notebook** | Benchmarks, latency metrics, retrieval performance comparisons, and edge-case handling. | Mandatory Deliverable |
| **Dataset Assets** | 30+ resumes and 5+ job descriptions organized within a `/data` directory. | Mandatory Deliverable |
| **GitHub Repository** | Clean, modular codebase with setup instructions, documentation, and `requirements.txt`. | Submission Link |
| **Demo Video** | 3–4 minute walkthrough demonstrating ingestion, querying, and structured JSON output. | Submission Video |