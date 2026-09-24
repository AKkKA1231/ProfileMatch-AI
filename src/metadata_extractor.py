import os
import re
from typing import List, Dict, Any, Optional

class MetadataExtractor:
    """
    Automated Metadata Extraction Module.
    Extracts structured entities (candidate_name, skills, experience_years, education)
    from resume text and chunks for persistent index metadata storage.
    """
    
    COMMON_SKILLS_TAXONOMY = {
        "python", "pytorch", "tensorflow", "transformers", "rag", "retrieval-augmented generation",
        "langchain", "llamaindex", "chromadb", "pinecone", "weaviate", "qdrant", "deep-speed", "cuda",
        "scikit-learn", "pandas", "numpy", "sql", "postgresql", "mysql", "oracle", "mongodb", "redis",
        "react", "typescript", "javascript", "next.js", "node.js", "redux", "tailwind", "css3", "html5",
        "cypress", "jest", "playwright", "graphql", "rest apis", "fastapi", "flask", "django",
        "docker", "kubernetes", "k8s", "terraform", "ansible", "aws", "gcp", "azure", "ci/cd",
        "prometheus", "grafana", "datadog", "linux", "bash", "git", "istio", "helm",
        "apache spark", "spark", "apache kafka", "kafka", "airflow", "dbt", "snowflake", "bigquery",
        "databricks", "delta lake", "hadoop", "c++", "go", "golang", "java", "scala", "r",
        "agile", "scrum", "jira", "product management", "user research", "wireframing", "mixpanel",
        "selenium", "pytest", "penetration testing", "iam", "siem", "react native", "ios", "android"
    }
    
    DEGREE_PATTERNS = [
        (r"\bPh\.?D\.?\b|\bDoctor of Philosophy\b", "Ph.D."),
        (r"\bM\.?S\.?\b|\bMaster of Science\b|\bM\.?Tech\.?\b|\bMBA\b|\bMaster of Business\b", "Master's"),
        (r"\bB\.?S\.?\b|\bBachelor of Science\b|\bB\.?Tech\.?\b|\bB\.?A\.?\b|\bBachelor of Arts\b", "Bachelor's"),
    ]

    def extract_metadata(self, raw_text: str, file_path: str) -> Dict[str, Any]:
        """
        Extracts structured metadata payload from complete resume raw text.
        """
        filename = os.path.basename(file_path)
        resume_id = os.path.splitext(filename)[0]

        candidate_name = self._extract_name(raw_text, filename)
        experience_years = self._extract_experience_years(raw_text)
        skills = self._extract_skills(raw_text)
        education_degree = self._extract_education(raw_text)

        return {
            "resume_id": resume_id,
            "resume_path": file_path,
            "candidate_name": candidate_name,
            "total_experience_years": float(experience_years),
            "skills": skills,
            "skills_str": ", ".join(skills),
            "highest_degree": education_degree
        }

    def _extract_name(self, text: str, filename: str) -> str:
        """Extract candidate name from text headers or filename fallback."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        
        # Check explicit Name: header
        for line in lines[:5]:
            name_match = re.search(r"^(?:Name|Candidate Name):\s*([A-Za-z\s.'-]+)", line, re.IGNORECASE)
            if name_match:
                return name_match.group(1).strip()
        
        # First non-header line usually contains the candidate name
        if lines:
            first_line = lines[0]
            if len(first_line) < 40 and not first_line.isupper() and "RESUME" not in first_line.upper():
                return first_line.strip()

        # Filename fallback (e.g. jane_doe_sr_ml -> Jane Doe)
        clean_filename = re.sub(r"_(sr|jr|lead|ml|dev|pm|fe|sre|qa|sec|ds|cpp|dba|fullstack|mobile|ui|sys).*", "", os.path.splitext(filename)[0])
        words = clean_filename.replace("_", " ").replace("-", " ").split()
        return " ".join(w.capitalize() for w in words) if words else "Unknown Candidate"

    def _extract_experience_years(self, text: str) -> float:
        """Extract total experience years from text tags or employment date spans."""
        # 1. Look for explicit total experience tag (e.g. "Total Experience: 6 years")
        exp_match = re.search(r"(?:Total Experience|Experience Required|Years of Experience):\s*(\d+(?:\.\d+)?)\s*(?:\+)?\s*years?", text, re.IGNORECASE)
        if exp_match:
            try:
                return float(exp_match.group(1))
            except ValueError:
                pass

        # 2. Look for phrases like "6 years of experience" or "5+ years experience"
        phrase_match = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*years?(?:\s+of)?\s+experience", text, re.IGNORECASE)
        if phrase_match:
            try:
                return float(phrase_match.group(1))
            except ValueError:
                pass

        # 3. Calculate year span from date ranges like (2018 - 2024) or (2020 - Present)
        years = [int(y) for y in re.findall(r"\b(20[0-2][0-9]|19[89][0-9])\b", text)]
        if years:
            min_year = min(years)
            current_year = 2026
            max_year = current_year if "present" in text.lower() else max(years)
            calculated_span = max_year - min_year
            if 0 < calculated_span <= 30:
                return float(calculated_span)

        return 3.0  # Default baseline fallback if undetermined

    def _extract_skills(self, text: str) -> List[str]:
        """Extract list of recognized technical skills from resume text."""
        found_skills = set()
        text_lower = text.lower()

        # Exact skill search from taxonomy
        for skill in self.COMMON_SKILLS_TAXONOMY:
            pattern = r"\b" + re.escape(skill) + r"\b"
            if re.search(pattern, text_lower):
                # Standardize display formatting
                found_skills.add(self._standardize_skill_display(skill))

        # Additional skills extracted from explicit Skills section
        skills_sec_match = re.search(r"=== SKILLS ===\s*([\s\S]*?)(?:===|$)", text, re.IGNORECASE)
        if skills_sec_match:
            sec_text = skills_sec_match.group(1)
            raw_tokens = re.split(r"[,:\n;/•|]+", sec_text)
            for token in raw_tokens:
                cleaned = token.strip()
                if 2 <= len(cleaned) <= 30 and not re.search(r"\b(languages|frameworks|tools|databases|core|competencies)\b", cleaned, re.IGNORECASE):
                    found_skills.add(cleaned.capitalize())

        return sorted(list(found_skills))

    def _standardize_skill_display(self, skill: str) -> str:
        """Map skills to canonical display names."""
        mapping = {
            "python": "Python", "pytorch": "PyTorch", "tensorflow": "TensorFlow",
            "rag": "Retrieval-Augmented Generation", "llms": "LLMs",
            "langchain": "LangChain", "llamaindex": "LlamaIndex",
            "chromadb": "ChromaDB", "pinecone": "Pinecone", "weaviate": "Weaviate",
            "react": "React", "typescript": "TypeScript", "javascript": "JavaScript",
            "next.js": "Next.js", "docker": "Docker", "kubernetes": "Kubernetes",
            "k8s": "Kubernetes", "aws": "AWS", "gcp": "GCP", "sql": "SQL",
            "spark": "Apache Spark", "apache spark": "Apache Spark",
            "kafka": "Apache Kafka", "apache kafka": "Apache Kafka",
            "fastapi": "FastAPI", "graphql": "GraphQL", "terraform": "Terraform"
        }
        return mapping.get(skill.lower(), skill.capitalize())

    def _extract_education(self, text: str) -> str:
        """Extract highest degree level achieved."""
        for pattern, degree_label in self.DEGREE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return degree_label
        return "Bachelor's"
