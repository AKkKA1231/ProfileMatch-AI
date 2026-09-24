# Intelligent Resume-to-Job Matching Engine (RAG)

An automated, context-aware **Resume-to-Job Matching Engine** powered by Retrieval-Augmented Generation (RAG). The system ingests unstructured resumes across varied formats (`.pdf`, `.docx`, `.txt`), performs section-aware chunking, extracts structured candidate metadata, and executes hybrid (dense + sparse) semantic retrieval to rank and explain candidate suitability against job descriptions.

---

## 🌟 Key System Features

- **Section-Aware Chunking (`src/chunker.py`)**: Preserves logical resume sections (`Summary`, `Work Experience`, `Education`, `Skills`, `Certifications`) rather than naive fixed-character splitting.
- **Automated Metadata Extraction (`src/metadata_extractor.py`)**: Extracts structured entity tags (`candidate_name`, `skills`, `total_experience_years`, `highest_degree`) and attaches them to vector chunks for precision filtering.
- **Dense-Sparse Hybrid Retrieval (`src/retriever.py`)**: Fuses dense vector cosine similarity (SentenceTransformers / ChromaDB) with sparse BM25 keyword matching to prevent keyword-padding false positives while ensuring exact skill validation.
- **Normalized Scoring & Explainability (`src/scorer.py`)**: Computes normalized alignment scores ($0–100$) and generates evidence-backed natural language rationales citing candidate excerpts and skill overlaps.
- **Strict JSON Schema Compliance**: Generates structured output payloads adhering strictly to Section 4.2 of `ProblemStatement.md`.

---

## 📁 Repository Directory Structure

```
RAG Based ProfileMatching/
├── data/
│   ├── resumes/                  # 30 diverse resumes (.pdf, .docx, .txt)
│   ├── job_descriptions/         # 5 comprehensive technical job descriptions (.txt)
│   └── chroma_db/                # Persistent vector database index
├── Docs/
│   ├── ProblemStatement.md       # Problem definition & specifications
│   ├── Architecture.md           # System architecture & sequence diagrams
│   ├── ImplementationPlan.md     # Phase-by-phase execution roadmap
│   ├── EdgeCases.md              # OCR fallbacks & defensive parsing rules
│   ├── Context.md                # Recruitment domain knowledge & RAG rationale
│   ├── DeploymentPlan.md         # FastAPI wrapper, Docker, & CI/CD workflow
│   ├── SpecialStrategies.md      # Hybrid fusion tactics & PII redaction rules
│   └── Role.md                   # System persona & prompt engineering recipe
├── notebooks/
│   └── evaluation.ipynb          # Latency & accuracy benchmarking notebook
├── src/
│   ├── __init__.py
│   ├── chunker.py                # Multi-format section-aware resume chunker
│   ├── metadata_extractor.py     # Candidate entity & metadata parser
│   ├── vector_store.py           # ChromaDB vector store manager & fallback
│   ├── retriever.py             # Hybrid dense-sparse search retriever
│   └── scorer.py                # 0-100 alignment scorer & XAI reasoning builder
├── tests/
│   ├── test_rag_setup.py         # Unit tests for chunking & vector indexing
│   ├── test_job_matcher.py       # Unit tests for hybrid search & scoring
│   └── test_end_to_end.py        # End-to-end integration & JSON schema tests
├── resume_rag.py                 # Milestone 1: Ingestion & indexing CLI script
├── job_matcher.py                # Milestone 2: Query matching & ranking CLI script
├── run_evaluation.py             # Phase 4: Automated evaluation benchmark runner
├── generate_dataset.py           # Dataset generator for candidate corpus
├── requirements.txt              # Project dependencies
└── README.md                     # System documentation & quickstart guide
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup & Installation

Clone the repository and install required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Milestone 1: Resume Ingestion & Indexing (`resume_rag.py`)

Ingest all resumes from `data/resumes/`, extract metadata, and populate the ChromaDB vector index:

```bash
python resume_rag.py --reset
```

*Flags:*
- `--data_dir`: Path to resume directory (default: `data/resumes`)
- `--db_dir`: Path to persistent ChromaDB storage (default: `data/chroma_db`)
- `--reset`: Resets existing vector collection before indexing.

### 3. Milestone 2: Candidate Matching & Ranking (`job_matcher.py`)

Match candidates against a target Job Description and output structured JSON payload:

```bash
python job_matcher.py --jd_path data/job_descriptions/jd1_senior_ml_engineer.txt --top_k 5 --output_json output_match.json
```

*Flags:*
- `--jd_path`: Path to input Job Description file.
- `--top_k`: Number of top matches to return (default: 10).
- `--min_exp`: Minimum experience years filter (e.g. `--min_exp 5.0`).
- `--output_json`: Output file path for JSON payload (default: `output_match.json`).

---

## 📊 Evaluation & Latency Benchmarks

Run the complete Phase 4 benchmarking suite:

```bash
python run_evaluation.py
```

### Key Empirical Results

| Metric | Result | Benchmark Description |
| :--- | :--- | :--- |
| **Ingestion Speed** | `0.22 ms / resume` | Section-aware chunking and metadata extraction. |
| **Query Latency** | `10.01 ms` | Top-10 hybrid vector retrieval latency. |
| **Hit Rate @ 3** | `100.0%` | Top-3 recall across target domain queries. |
| **MRR @ 5** | `1.0000` | Top relevant candidate ranked #1 for 100% of queries. |
| **Precision @ 3** | `80.0%` | Proportion of relevant matches in top 3 results. |

---

## 🧪 Automated Testing

Execute the full test suite using standard Python `unittest`:

```bash
python -m unittest discover -s tests
```

---

## 📄 Structured JSON Output Schema Example

```json
{
  "job_description": "Senior Machine Learning Engineer specializing in LLMs and distributed training...",
  "top_matches": [
    {
      "candidate_name": "Jane Doe",
      "resume_path": "data/resumes/jane_doe_sr_ml.txt",
      "match_score": 81,
      "matched_skills": [
        "ChromaDB",
        "Docker",
        "Kubernetes",
        "LangChain",
        "LlamaIndex",
        "PyTorch",
        "Python",
        "Retrieval-Augmented Generation",
        "Transformers"
      ],
      "relevant_excerpts": [
        "Senior ML Engineer at TechCorp (2021 - Present):",
        "Led deployment of enterprise RAG pipelines reducing query latency by 40% using ChromaDB and PyTorch."
      ],
      "reasoning": "Strong match (81/100): Candidate meets or exceeds the 5+ years experience requirement (6.0 yrs demonstrated); exhibits direct proficiency in ChromaDB, Docker, Kubernetes, LangChain aligned with key role responsibilities."
    }
  ]
}
```
