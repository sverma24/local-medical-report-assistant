from __future__ import annotations

import warnings
from pathlib import Path

from chromadb.config import Settings
from langchain_community.document_loaders import DirectoryLoader, TextLoader

warnings.filterwarnings(
    "ignore",
    message="The class `Chroma` was deprecated.*",
)

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import AppConfig


class LocalVectorStore:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.config.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = OllamaEmbeddings(
            model=config.embedding_model, base_url=config.ollama_base_url
        )
        chroma_settings = Settings(anonymized_telemetry=False)
        self.knowledge_store = Chroma(
            collection_name=config.knowledge_collection,
            embedding_function=self.embeddings,
            persist_directory=str(config.chroma_dir),
            client_settings=chroma_settings,
        )
        self.report_store = Chroma(
            collection_name=config.report_collection,
            embedding_function=self.embeddings,
            persist_directory=str(config.chroma_dir),
            client_settings=chroma_settings,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=900, chunk_overlap=120, separators=["\n\n", "\n", ". "]
        )

    def bootstrap_knowledge_base(self) -> int:
        if self.knowledge_store._collection.count() > 0:  # noqa: SLF001
            return self.knowledge_store._collection.count()  # noqa: SLF001

        docs = self._load_knowledge_docs(self.config.knowledge_dir)
        chunks = self.splitter.split_documents(docs)
        self.knowledge_store.add_documents(chunks)
        return len(chunks)

    def index_report(self, report_id: str, report_text: str) -> int:
        base_doc = Document(
            page_content=report_text,
            metadata={"source_type": "report", "report_id": report_id},
        )
        chunks = self.splitter.split_documents([base_doc])
        self.report_store.add_documents(chunks)
        return len(chunks)

    def retrieve_context(self, query: str, report_id: str, k: int = 4) -> list[Document]:
        report_docs = self.report_store.similarity_search(
            query, k=k, filter={"report_id": report_id}
        )
        knowledge_docs = self.knowledge_store.similarity_search(query, k=k)
        return report_docs + knowledge_docs

    @staticmethod
    def _load_knowledge_docs(knowledge_dir: Path) -> list[Document]:
        loader = DirectoryLoader(
            str(knowledge_dir),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        return loader.load()
