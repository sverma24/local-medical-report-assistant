from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    project_root: Path = Path(__file__).resolve().parent.parent
    chroma_dir: Path = project_root / "data" / "chroma"
    knowledge_dir: Path = project_root / "data" / "knowledge"
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model: str = os.getenv("LLM_MODEL", "gemma4:e4b")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    ollama_timeout_seconds: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
    max_measurements_for_llm: int = int(os.getenv("MAX_MEASUREMENTS_FOR_LLM", "40"))
    report_retrieval_k: int = int(os.getenv("REPORT_RETRIEVAL_K", "2"))
    knowledge_retrieval_k: int = int(os.getenv("KNOWLEDGE_RETRIEVAL_K", "2"))
    retrieved_context_char_limit: int = int(
        os.getenv("RETRIEVED_CONTEXT_CHAR_LIMIT", "2800")
    )
    report_collection: str = "medical_reports"
    knowledge_collection: str = "medical_knowledge"
