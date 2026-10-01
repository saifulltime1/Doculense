"""DocuLens API: AI-assisted document summarization."""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .llm import abstractive_summarize, llm_configured
from .summarizer import extract_keywords, extractive_summarize

APP_VERSION = "1.0.0"
ALLOWED_EXTENSIONS = {".txt", ".md"}
VALID_LENGTHS = {"short", "medium", "detailed"}

app = FastAPI(title="DocuLens API", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_enabled: bool


class SummarizeResponse(BaseModel):
    summary: str
    mode: str = Field(description="'abstractive' (LLM) or 'extractive' (offline)")
    length: str
    keywords: list[str]
    sentences_used: Optional[int] = None
    char_count: int
    word_count: int


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=APP_VERSION, llm_enabled=llm_configured())


@app.post("/api/summarize", response_model=SummarizeResponse)
async def summarize(
    text: Optional[str] = Form(default=None),
    length: str = Form(default="medium"),
    file: Optional[UploadFile] = File(default=None),
) -> SummarizeResponse:
    if length not in VALID_LENGTHS:
        raise HTTPException(
            status_code=400,
            detail=f"length must be one of {sorted(VALID_LENGTHS)}",
        )

    content = (text or "").strip()
    if file is not None and file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Only {sorted(ALLOWED_EXTENSIONS)} files are supported",
            )
        raw = await file.read()
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            content = raw.decode("utf-8", errors="replace")

    if not content.strip():
        raise HTTPException(
            status_code=400,
            detail="Provide text or upload a .txt/.md file",
        )

    mode = "extractive"
    sentences_used: Optional[int] = None
    if llm_configured():
        try:
            summary = abstractive_summarize(content, length)
            mode = "abstractive"
        except Exception:
            summary, sentences_used = extractive_summarize(content, length)
    else:
        summary, sentences_used = extractive_summarize(content, length)

    words = content.split()
    return SummarizeResponse(
        summary=summary,
        mode=mode,
        length=length,
        keywords=extract_keywords(content),
        sentences_used=sentences_used,
        char_count=len(content),
        word_count=len(words),
    )
