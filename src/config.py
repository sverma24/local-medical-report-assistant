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
    llm_model: str = os.getenv("LLM_MODEL", "gemma4:26b")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    report_collection: str = "medical_reports"
    knowledge_collection: str = "medical_knowledge"

