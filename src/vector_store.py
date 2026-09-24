import os
import json
import shutil
import math
from typing import List, Dict, Any, Optional

class VectorStoreManager:
    """
    Vector Store Manager using ChromaDB with a lightweight persistent JSON/Cosine fallback.
    Handles embedding generation, chunk indexing, metadata persistence, and vector similarity search.
    """
    
    COLLECTION_NAME = "resume_chunks"
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, db_path: str = "data/chroma_db", embedding_model_name: Optional[str] = None):
        self.db_path = db_path
        self.model_name = embedding_model_name or self.MODEL_NAME
        self._embedder = None
        self._chroma_client = None
        self._collection = None
        self._use_fallback_store = False
        self._fallback_records = []

    def _get_embedder(self):
        """Lazy load SentenceTransformer embedding model or fallback embedder."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.model_name)
            except Exception:
                self._embedder = FallbackEmbedder()
        return self._embedder

    def _get_collection(self):
        """Lazy load persistent ChromaDB collection or initialize fallback store."""
        if self._collection is None and not self._use_fallback_store:
            try:
                import chromadb
                os.makedirs(self.db_path, exist_ok=True)
                self._chroma_client = chromadb.PersistentClient(path=self.db_path)
                self._collection = self._chroma_client.get_or_create_collection(
                    name=self.COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"}
                )
            except ImportError:
                self._use_fallback_store = True
                self._load_fallback_store()
        return self._collection

    def _load_fallback_store(self):
        """Load persistent JSON fallback vector store."""
        os.makedirs(self.db_path, exist_ok=True)
        file_path = os.path.join(self.db_path, "fallback_index.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    self._fallback_records = json.load(f)
            except Exception:
                self._fallback_records = []
        else:
            self._fallback_records = []

    def _save_fallback_store(self):
        """Save persistent JSON fallback vector store."""
        os.makedirs(self.db_path, exist_ok=True)
        file_path = os.path.join(self.db_path, "fallback_index.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self._fallback_records, f, indent=2)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a list of text strings."""
        embedder = self._get_embedder()
        embeddings = embedder.encode(texts, show_progress_bar=False)
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()
        return embeddings

    def add_resume_chunks(self, chunks: List[Dict[str, Any]], metadata: Dict[str, Any]):
        """
        Indexes a list of section chunks from a resume.
        Attaches candidate metadata to each chunk entry.
        """
        if not chunks:
            return

        self._get_collection()

        resume_id = metadata["resume_id"]
        ids = []
        documents = []
        metadatas = []
        texts_to_embed = []

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{resume_id}_chunk_{idx}"
            section_name = chunk.get("section_name", "GENERAL")
            content = chunk.get("content", "")

            embed_text = f"Candidate: {metadata['candidate_name']} | Section: {section_name}\nSkills: {metadata.get('skills_str', '')}\nContent: {content}"
            
            ids.append(chunk_id)
            documents.append(content)
            texts_to_embed.append(embed_text)

            chunk_meta = {
                "resume_id": resume_id,
                "resume_path": metadata["resume_path"],
                "candidate_name": metadata["candidate_name"],
                "section_name": section_name,
                "total_experience_years": float(metadata["total_experience_years"]),
                "highest_degree": metadata["highest_degree"],
                "skills_str": metadata.get("skills_str", ""),
                "chunk_index": idx
            }
            metadatas.append(chunk_meta)

        embeddings = self.embed_texts(texts_to_embed)

        if not self._use_fallback_store and self._collection is not None:
            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
        else:
            for c_id, doc, meta, emb in zip(ids, documents, metadatas, embeddings):
                # Remove existing record if re-indexing
                self._fallback_records = [r for r in self._fallback_records if r["chunk_id"] != c_id]
                self._fallback_records.append({
                    "chunk_id": c_id,
                    "content": doc,
                    "metadata": meta,
                    "embedding": emb
                })
            self._save_fallback_store()

    def query_similarity(self, query_text: str, top_k: int = 10, where_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Queries collection for top-K vector similarity matches.
        """
        self._get_collection()
        query_embedding = self.embed_texts([query_text])[0]

        if not self._use_fallback_store and self._collection is not None:
            query_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": min(top_k, max(1, self._collection.count()))
            }
            if where_filter:
                query_kwargs["where"] = where_filter

            results = self._collection.query(**query_kwargs)

            formatted_results = []
            if results and results.get("ids") and results["ids"][0]:
                ids = results["ids"][0]
                distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)
                documents = results["documents"][0] if results.get("documents") else [""] * len(ids)
                metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)

                for i in range(len(ids)):
                    sim_score = max(0.0, min(1.0, 1.0 - (distances[i] / 2.0)))
                    formatted_results.append({
                        "chunk_id": ids[i],
                        "similarity_score": sim_score,
                        "content": documents[i],
                        "metadata": metadatas[i]
                    })
            return formatted_results
        else:
            # Fallback cosine similarity match
            scores = []
            for record in self._fallback_records:
                meta = record["metadata"]
                
                # Apply where filter if specified (e.g. total_experience_years >= min_exp)
                if where_filter:
                    match = True
                    for k, v in where_filter.items():
                        if isinstance(v, dict) and "$gte" in v:
                            if meta.get(k, 0) < v["$gte"]:
                                match = False
                        elif meta.get(k) != v:
                            match = False
                    if not match:
                        continue

                sim = self._cosine_similarity(query_embedding, record["embedding"])
                scores.append({
                    "chunk_id": record["chunk_id"],
                    "similarity_score": sim,
                    "content": record["content"],
                    "metadata": meta
                })

            scores.sort(key=lambda x: x["similarity_score"], reverse=True)
            return scores[:top_k]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 > 0 and norm2 > 0:
            return max(0.0, min(1.0, dot / (norm1 * norm2)))
        return 0.0

    def reset_collection(self):
        """Resets/clears the persistent collection."""
        if self._chroma_client is not None:
            try:
                self._chroma_client.delete_collection(name=self.COLLECTION_NAME)
            except Exception:
                pass
            self._collection = None
        self._fallback_records = []
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path, ignore_errors=True)

    def count(self) -> int:
        """Returns total indexed chunks in collection."""
        self._get_collection()
        if not self._use_fallback_store and self._collection is not None:
            return self._collection.count()
        return len(self._fallback_records)


class FallbackEmbedder:
    """Fallback embedder when sentence-transformers is unavailable."""
    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts: List[str], show_progress_bar: bool = False):
        import hashlib

        embeddings = []
        for text in texts:
            vec = [0.0] * self.dim
            words = text.lower().split()
            for w in words:
                h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
                idx = h % self.dim
                vec[idx] += 1.0
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            embeddings.append(vec)
        return embeddings
