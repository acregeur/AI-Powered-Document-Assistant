"""Core document ingestion, retrieval, and RAG orchestration."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
import requests
from pypdf import PdfReader
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.models import (
    DocumentChunk,
    EvaluationQuestion,
    EvaluationResponse,
    EvaluationResult,
    IndexedSourceState,
    IngestResponse,
    QueryResponse,
    SearchResult,
    Settings,
    SourceReference,
)

logger = logging.getLogger(__name__)


class LocalLLM:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("response", "").strip()


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        embeddings = self.model.encode(list(texts), normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        embedding = self.model.encode(query, normalize_embeddings=True)
        return embedding.tolist()


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[dict[str, str | float]], top_k: int) -> list[dict[str, str | float]]:
        if not results:
            return []

        pairs = [(query, str(item["text"])) for item in results]
        scores = self.model.predict(pairs)

        rescored = []
        for item, score in zip(results, scores):
            updated = dict(item)
            updated["rerank_score"] = float(score)
            rescored.append(updated)

        rescored.sort(key=lambda item: float(item["rerank_score"]), reverse=True)
        return rescored[:top_k]


class DocumentIngestor:
    SUPPORTED_EXTENSIONS = {".txt", ".pdf"}

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest_folder(self, folder_path: Path) -> tuple[list[DocumentChunk], int]:
        folder = folder_path.expanduser().resolve()
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")

        chunks: list[DocumentChunk] = []
        indexed_files = 0

        for file_path in sorted(folder.rglob("*")):
            if not file_path.is_file() or file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            logger.info("Processing file: %s", file_path)
            file_chunks = self._process_file(file_path)
            if file_chunks:
                chunks.extend(file_chunks)
                indexed_files += 1

        return chunks, indexed_files

    def build_source_state(self, folder_path: Path) -> IndexedSourceState:
        folder = folder_path.expanduser().resolve()
        files: dict[str, float] = {}
        for file_path in sorted(folder.rglob("*")):
            if not file_path.is_file() or file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue
            files[str(file_path.resolve())] = file_path.stat().st_mtime
        return IndexedSourceState(folder_path=str(folder), files=files)

    def _process_file(self, file_path: Path) -> list[DocumentChunk]:
        if file_path.suffix.lower() == ".txt":
            return self._process_txt(file_path)
        if file_path.suffix.lower() == ".pdf":
            return self._process_pdf(file_path)
        return []

    def _process_txt(self, file_path: Path) -> list[DocumentChunk]:
        text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return []

        return [
            DocumentChunk(
                text=chunk_text,
                metadata={
                    "filename": file_path.name,
                    "filepath": str(file_path),
                    "chunk_index": index,
                    "page_number": None,
                },
            )
            for index, chunk_text in enumerate(self._chunk_text(text))
        ]

    def _process_pdf(self, file_path: Path) -> list[DocumentChunk]:
        reader = PdfReader(str(file_path))
        collected_chunks: list[DocumentChunk] = []
        chunk_index = 0

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue

            for chunk_text in self._chunk_text(text):
                collected_chunks.append(
                    DocumentChunk(
                        text=chunk_text,
                        metadata={
                            "filename": file_path.name,
                            "filepath": str(file_path),
                            "chunk_index": chunk_index,
                            "page_number": page_number,
                        },
                    )
                )
                chunk_index += 1

        return collected_chunks

    def _chunk_text(self, text: str) -> list[str]:
        normalized = " ".join(text.split())
        if len(normalized) <= self.chunk_size:
            return [normalized]

        chunks: list[str] = []
        start = 0
        step = max(1, self.chunk_size - self.chunk_overlap)

        while start < len(normalized):
            end = start + self.chunk_size
            chunks.append(normalized[start:end])
            start += step

        return chunks


class FaissVectorStore:
    INDEX_FILENAME = "index.faiss"
    CHUNKS_FILENAME = "chunks.json"
    STATE_FILENAME = "source_state.json"

    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.storage_dir / self.INDEX_FILENAME
        self.chunks_path = self.storage_dir / self.CHUNKS_FILENAME
        self.state_path = self.storage_dir / self.STATE_FILENAME

    def build(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
        source_state: IndexedSourceState,
    ) -> None:
        if not chunks:
            raise ValueError("No chunks available to index.")

        matrix = np.array(embeddings, dtype="float32")
        dimension = matrix.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(matrix)
        faiss.write_index(index, str(self.index_path))

        payload = [chunk.model_dump() for chunk in chunks]
        self.chunks_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.state_path.write_text(source_state.model_dump_json(indent=2), encoding="utf-8")

    def exists(self) -> bool:
        return self.index_path.exists() and self.chunks_path.exists() and self.state_path.exists()

    def load_source_state(self) -> IndexedSourceState:
        if not self.state_path.exists():
            raise FileNotFoundError("Indexed source state does not exist. Run ingestion first.")
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        return IndexedSourceState.model_validate(payload)

    def load_chunks(self) -> list[DocumentChunk]:
        if not self.chunks_path.exists():
            raise FileNotFoundError("Chunk store does not exist. Run ingestion first.")
        chunks_payload = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        return [DocumentChunk.model_validate(item) for item in chunks_payload]

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        if not self.exists():
            raise FileNotFoundError("Vector store does not exist. Run ingestion first.")

        index = faiss.read_index(str(self.index_path))
        chunks = self.load_chunks()
        query = np.array([query_embedding], dtype="float32")
        scores, indices = index.search(query, top_k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(chunks):
                continue
            chunk = chunks[idx]
            results.append(SearchResult(text=chunk.text, metadata=chunk.metadata, score=float(score)))

        return results


class Retriever:
    def __init__(
        self,
        embedder: SentenceTransformerEmbedder,
        vectorstore: FaissVectorStore,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.reranker = reranker

    def retrieve(self, question: str, top_k: int, candidate_k: int) -> list[SearchResult]:
        query_embedding = self.embedder.embed_query(question)
        limit = max(top_k, candidate_k)
        dense_candidates = self.vectorstore.search(query_embedding=query_embedding, top_k=limit)
        lexical_candidates = self._lexical_search(question=question, limit=limit)
        candidates = self._merge_candidates(dense_candidates, lexical_candidates)

        if not self.reranker:
            return candidates[:top_k]

        reranked_payload = self.reranker.rerank(
            query=question,
            results=[item.model_dump() for item in candidates],
            top_k=top_k,
        )
        return [SearchResult.model_validate(item) for item in reranked_payload]

    def _lexical_search(self, question: str, limit: int) -> list[SearchResult]:
        chunks = self.vectorstore.load_chunks()
        query = question.strip().lower()
        query_tokens = self._tokenize(query)
        scored: list[SearchResult] = []

        for chunk in chunks:
            chunk_text = chunk.text.lower()
            chunk_tokens = self._tokenize(chunk_text)
            score = self._lexical_score(query, query_tokens, chunk_text, chunk_tokens)
            if score <= 0:
                continue
            scored.append(SearchResult(text=chunk.text, metadata=chunk.metadata, score=score))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]

    def _merge_candidates(
        self,
        dense_candidates: list[SearchResult],
        lexical_candidates: list[SearchResult],
    ) -> list[SearchResult]:
        merged: dict[tuple[str, int, int | None], SearchResult] = {}

        for result in dense_candidates:
            merged[self._result_key(result)] = result

        for result in lexical_candidates:
            key = self._result_key(result)
            if key in merged and result.score <= merged[key].score:
                continue
            merged[key] = result

        return list(merged.values())

    def _lexical_score(
        self,
        query: str,
        query_tokens: list[str],
        chunk_text: str,
        chunk_tokens: list[str],
    ) -> float:
        if not query_tokens:
            return 0.0

        score = 0.0
        if query in chunk_text:
            score += 10.0

        overlap = sum(1 for token in query_tokens if token in chunk_tokens)
        score += overlap * 2.0

        for index in range(len(query_tokens) - 1):
            phrase = f"{query_tokens[index]} {query_tokens[index + 1]}"
            if phrase in chunk_text:
                score += 4.0

        return score

    def _result_key(self, result: SearchResult) -> tuple[str, int, int | None]:
        return (
            str(result.metadata.get("filename", "")),
            int(result.metadata.get("chunk_index", -1)),
            result.metadata.get("page_number"),
        )

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())


def build_rag_prompt(question: str, results: list[SearchResult]) -> str:
    context = "\n\n".join(
        (
            f"Source: {item.metadata.get('filename')} | "
            f"Chunk: {item.metadata.get('chunk_index')} | "
            f"Page: {item.metadata.get('page_number')}\n"
            f"{item.text}"
        )
        for item in results
    )
    return (
        "You are answering questions about local documents.\n"
        "Use ONLY the context below.\n"
        "If the context contains the answer, respond with a direct concise answer.\n"
        "If the context does not contain the answer, say exactly: I don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Helpful answer:"
    )


class RAGPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder = SentenceTransformerEmbedder(settings.embedding_model)
        self.reranker = CrossEncoderReranker(settings.reranker_model)
        self.ingestor = DocumentIngestor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self.llm = LocalLLM(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )

    def ingest(self, folder_path: Path) -> IngestResponse:
        chunks, indexed_files = self.ingestor.ingest_folder(folder_path)
        source_state = self.ingestor.build_source_state(folder_path)
        embeddings = self.embedder.embed_texts(chunk.text for chunk in chunks)
        vectorstore = FaissVectorStore(self._vectorstore_dir())
        vectorstore.build(chunks=chunks, embeddings=embeddings, source_state=source_state)
        return IngestResponse(
            folder_path=folder_path.resolve(),
            indexed_chunks=len(chunks),
            indexed_files=indexed_files,
            vectorstore_path=vectorstore.storage_dir.resolve(),
        )

    def answer_question(self, question: str, top_k: int | None = None) -> QueryResponse:
        vectorstore = FaissVectorStore(self._vectorstore_dir())
        self._refresh_if_source_changed(vectorstore)
        retriever = Retriever(
            embedder=self.embedder,
            vectorstore=vectorstore,
            reranker=self.reranker,
        )
        results = retriever.retrieve(
            question=question,
            top_k=top_k or self.settings.top_k,
            candidate_k=self.settings.retrieval_candidates,
        )

        if not results:
            return QueryResponse(answer="I don't know based on the indexed documents.", sources=[])

        answer = self.llm.generate(build_rag_prompt(question=question, results=results))
        sources = [
            SourceReference(
                filename=item.metadata.get("filename", "unknown"),
                chunk_index=int(item.metadata.get("chunk_index", -1)),
                page_number=item.metadata.get("page_number"),
                similarity_score=item.score,
            )
            for item in results
        ]
        return QueryResponse(answer=answer, sources=sources)

    def evaluate_questions(
        self,
        questions_file: Path,
        output_file: Path,
        top_k: int | None = None,
    ) -> EvaluationResponse:
        input_path = questions_file.expanduser().resolve()
        output_path = output_file.expanduser().resolve()

        if not input_path.exists():
            raise FileNotFoundError(f"Questions file not found: {input_path}")

        payload = json.loads(input_path.read_text(encoding="utf-8"))
        questions = self._parse_evaluation_questions(payload)
        results: list[EvaluationResult] = []

        for item in questions:
            response = self.answer_question(question=item.question, top_k=top_k)
            results.append(
                EvaluationResult(
                    id=item.id,
                    question=item.question,
                    answer=response.answer,
                    sources=response.sources,
                )
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_payload = {
            "questions_file": str(input_path),
            "total_questions": len(results),
            "results": [item.model_dump() for item in results],
        }
        output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

        return EvaluationResponse(
            questions_file=input_path,
            output_file=output_path,
            total_questions=len(results),
        )

    def _vectorstore_dir(self) -> Path:
        return self.settings.normalized_data_dir

    def _refresh_if_source_changed(self, vectorstore: FaissVectorStore) -> None:
        source_state = vectorstore.load_source_state()
        current_state = self.ingestor.build_source_state(Path(source_state.folder_path))
        if current_state.files == source_state.files:
            return

        chunks, _ = self.ingestor.ingest_folder(Path(source_state.folder_path))
        embeddings = self.embedder.embed_texts(chunk.text for chunk in chunks)
        vectorstore.build(chunks=chunks, embeddings=embeddings, source_state=current_state)

    def _parse_evaluation_questions(self, payload: object) -> list[EvaluationQuestion]:
        if not isinstance(payload, list):
            raise ValueError("Questions file must be a JSON array.")

        parsed: list[EvaluationQuestion] = []
        for index, item in enumerate(payload, start=1):
            if isinstance(item, str):
                parsed.append(EvaluationQuestion(id=str(index), question=item))
            elif isinstance(item, dict):
                parsed.append(EvaluationQuestion.model_validate(item))
            else:
                raise ValueError("Each question must be either a string or an object with a question field.")

        if not parsed:
            raise ValueError("Questions file is empty.")

        return parsed
