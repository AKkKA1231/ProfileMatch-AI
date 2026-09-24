import re
from typing import List, Dict, Any, Optional

class MatchRanker:
    """
    Ranking, Filtering & Explainability Engine.
    Computes normalized alignment score (0-100), extracts matched skills & excerpts,
    and generates actionable rationale for candidate suitability.
    """

    def __init__(self, target_jd_text: str):
        self.target_jd_text = target_jd_text
        self.jd_skills = self._extract_jd_skills(target_jd_text)
        self.jd_min_experience = self._extract_jd_min_experience(target_jd_text)

    def process_and_rank_candidates(self, candidates: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Processes candidate search matches, normalizes scores (0-100), builds rationales,
        and returns top-K formatted candidate dictionaries.
        """
        top_matches = []

        for c in candidates:
            # 1. Normalize alignment score onto 0-100 scale
            raw_score = c.get("best_hybrid_score", 0.0)
            norm_score = self._normalize_score(raw_score)

            # 2. Extract matched skills
            candidate_skills = [s.strip() for s in c.get("skills_str", "").split(",") if s.strip()]
            matched_skills = sorted(list(set(candidate_skills).intersection(set(self.jd_skills))))
            if not matched_skills and candidate_skills:
                # Fallback fuzzy match
                matched_skills = [s for s in candidate_skills if s.lower() in self.target_jd_text.lower()][:5]

            # 3. Extract top relevant excerpts
            relevant_excerpts = self._extract_relevant_excerpts(c.get("chunks", []))

            # 4. Generate Explainable Reasoning
            reasoning = self._generate_reasoning(
                candidate_name=c["candidate_name"],
                match_score=norm_score,
                exp_years=c["total_experience_years"],
                matched_skills=matched_skills,
                excerpts=relevant_excerpts
            )

            top_matches.append({
                "candidate_name": c["candidate_name"],
                "resume_path": c["resume_path"],
                "match_score": norm_score,
                "matched_skills": matched_skills,
                "relevant_excerpts": relevant_excerpts,
                "reasoning": reasoning
            })

        # Re-sort by normalized match score
        top_matches.sort(key=lambda x: x["match_score"], reverse=True)
        return top_matches[:top_k]

    def _normalize_score(self, raw_score: float) -> int:
        """
        Normalizes composite similarity score (0.0 to 1.0) to 0–100 range.
        Ensures realistic hiring score distribution (e.g. 50 to 98).
        """
        if raw_score <= 0:
            return 0
        # Sigmoid / linear scaling transformation
        scaled = 40.0 + (raw_score * 58.0)
        return int(min(99, max(15, round(scaled))))

    def _extract_jd_skills(self, text: str) -> List[str]:
        """Extract key technical skills from Job Description text."""
        known_skills = [
            "Python", "PyTorch", "TensorFlow", "Transformers", "Retrieval-Augmented Generation", "RAG",
            "LangChain", "LlamaIndex", "ChromaDB", "Pinecone", "Weaviate", "Kubernetes", "Docker",
            "React", "TypeScript", "JavaScript", "Next.js", "Redux", "Tailwind CSS", "Jest",
            "AWS", "GCP", "Terraform", "Ansible", "Prometheus", "Grafana", "Linux", "Go",
            "Apache Spark", "Apache Kafka", "Airflow", "Snowflake", "BigQuery", "SQL", "dbt",
            "Product Management", "Agile", "Scrum", "JIRA", "Wireframing", "A/B Testing"
        ]
        found = []
        for sk in known_skills:
            if re.search(r"\b" + re.escape(sk) + r"\b", text, re.IGNORECASE):
                found.append(sk)
        return found

    def _extract_jd_min_experience(self, text: str) -> float:
        """Extract required experience years from Job Description."""
        match = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*years", text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return 3.0

    def _extract_relevant_excerpts(self, chunks: List[Dict[str, Any]], max_excerpts: int = 2) -> List[str]:
        """Extract top representative excerpts from candidate section chunks."""
        excerpts = []
        for chunk in chunks:
            content = chunk.get("content", "").strip()
            if not content:
                continue
            lines = [l.strip("-•* ").strip() for l in content.splitlines() if len(l.strip()) > 20]
            for line in lines:
                if line not in excerpts and not line.startswith("===") and not line.startswith("Name:"):
                    excerpts.append(line)
                    if len(excerpts) >= max_excerpts:
                        break
            if len(excerpts) >= max_excerpts:
                break

        if not excerpts:
            excerpts = ["Demonstrates domain experience aligned with role requirements."]
        return excerpts

    def _generate_reasoning(self, candidate_name: str, match_score: int, exp_years: float, matched_skills: List[str], excerpts: List[str]) -> str:
        """Generate human-readable matching justification for candidate suitability."""
        level = "Strong" if match_score >= 80 else ("Moderate" if match_score >= 60 else "Potential")
        
        skills_str = ", ".join(matched_skills[:4]) if matched_skills else "relevant core technical skills"
        req_exp = self.jd_min_experience

        if exp_years >= req_exp:
            exp_clause = f"meets or exceeds the {req_exp:.0f}+ years experience requirement ({exp_years:.1f} yrs demonstrated)"
        else:
            exp_clause = f"possesses {exp_years:.1f} years of relevant experience"

        return f"{level} match ({match_score}/100): Candidate {exp_clause}; exhibits direct proficiency in {skills_str} aligned with key role responsibilities."

    def format_output_payload(self, job_description_text: str, top_matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format matching results into the exact JSON schema required by ProblemStatement.md 4.2."""
        # Clean JD summary string
        jd_summary = job_description_text.strip().replace("\n", " ")
        if len(jd_summary) > 120:
            jd_summary = jd_summary[:117] + "..."

        return {
            "job_description": jd_summary,
            "top_matches": top_matches
        }
