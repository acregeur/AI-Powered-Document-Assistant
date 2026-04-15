from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    app_name: str = "Local AI Document Assistant"
    app_version: str = "0.1.0"
    data_dir: Path = Field(default=Path("data"))
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    top_k: int = 5
    min_source_similarity_score: float = 0.35
    retrieval_candidates: int = 15
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @property
    def normalized_data_dir(self) -> Path:
        return self.data_dir.resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


class IngestRequest(BaseModel):
    folder_path: Path


class IngestResponse(BaseModel):
    folder_path: Path
    indexed_chunks: int
    indexed_files: int
    vectorstore_path: Path


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


class EvaluationRequest(BaseModel):
    questions_file: Path
    output_file: Path
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceReference(BaseModel):
    filename: str
    chunk_index: int
    page_number: int | None = None
    similarity_score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceReference]


class EvaluationQuestion(BaseModel):
    id: str | None = None
    question: str = Field(min_length=1)


class EvaluationResult(BaseModel):
    id: str | None = None
    question: str
    answer: str
    sources: list[SourceReference]


class EvaluationResponse(BaseModel):
    questions_file: Path
    output_file: Path
    total_questions: int


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    min_source_similarity_score: float


class DocumentChunk(BaseModel):
    text: str
    metadata: dict[str, Any]


class SearchResult(BaseModel):
    text: str
    metadata: dict[str, Any]
    score: float


class IndexedSourceState(BaseModel):
    folder_path: str
    files: dict[str, float]
