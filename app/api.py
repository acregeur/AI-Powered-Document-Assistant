from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException

from app.core import RAGPipeline
from app.models import (
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    Settings,
    get_settings,
)

router = APIRouter()


@lru_cache(maxsize=1)
def _get_cached_pipeline() -> RAGPipeline:
    return RAGPipeline(get_settings())


def get_pipeline(_: Settings = Depends(get_settings)) -> RAGPipeline:
    return _get_cached_pipeline()


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
        min_source_similarity_score=settings.min_source_similarity_score,
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest_documents(
    request: IngestRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> IngestResponse:
    try:
        return pipeline.ingest(folder_path=request.folder_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/query", response_model=QueryResponse)
def query_documents(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> QueryResponse:
    try:
        return pipeline.answer_question(question=request.question, top_k=request.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/evaluate", response_model=EvaluationResponse)
def evaluate_documents(
    request: EvaluationRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> EvaluationResponse:
    try:
        return pipeline.evaluate_questions(
            questions_file=request.questions_file,
            output_file=request.output_file,
            top_k=request.top_k,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
