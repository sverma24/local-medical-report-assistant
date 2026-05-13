from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig
from src.retrieval.vector_store import LocalVectorStore
from scripts.prepare_medlineplus_knowledge import prepare_medlineplus_knowledge


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare local knowledge files and rebuild Chroma knowledge collection."
    )
    parser.add_argument(
        "--clean-knowledge-dir",
        action="store_true",
        help="Delete existing MedlinePlus markdown files before redownloading.",
    )
    parser.add_argument(
        "--knowledge-output-dir",
        default="data/knowledge/medlineplus",
        help="Path to store MedlinePlus markdown files.",
    )
    args = parser.parse_args()

    output_dir = Path(args.knowledge_output_dir).resolve()
    pages_written, pages_failed = prepare_medlineplus_knowledge(
        output_dir=output_dir, clean=args.clean_knowledge_dir
    )

    config = AppConfig()
    vector_store = LocalVectorStore(config)
    chunk_count = vector_store.bootstrap_knowledge_base(force_rebuild=True)

    print(f"medlineplus_pages_written={pages_written}")
    print(f"medlineplus_pages_failed={pages_failed}")
    print(f"knowledge_chunks_indexed={chunk_count}")
    print(f"knowledge_output_dir={output_dir}")
    print(f"chroma_dir={config.chroma_dir.resolve()}")


if __name__ == "__main__":
    main()
