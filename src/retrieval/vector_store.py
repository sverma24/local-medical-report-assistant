from __future__ import annotations

import hashlib
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
    MAX_INSERT_BATCH_SIZE = 500

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

    def bootstrap_knowledge_base(self, force_rebuild: bool = False) -> int:
        if force_rebuild:
            self._recreate_knowledge_collection()

        if self.knowledge_store._collection.count() > 0:  # noqa: SLF001
            return self.knowledge_store._collection.count()  # noqa: SLF001

        docs = self._load_knowledge_docs(self.config.knowledge_dir)
        chunks = self.splitter.split_documents(docs)
        self._add_documents_in_batches(self.knowledge_store, chunks)
        return len(chunks)

    def index_report(self, report_id: str, report_text: str) -> int:
        base_doc = Document(
            page_content=report_text,
            metadata={"source_type": "report", "report_id": report_id},
        )
        chunks = self.splitter.split_documents([base_doc])
        self._add_documents_in_batches(self.report_store, chunks)
        return len(chunks)

    def retrieve_context(self, query: str, report_id: str) -> list[Document]:
        report_docs = self.report_store.similarity_search(
            query, k=self.config.report_retrieval_k, filter={"report_id": report_id}
        )
        knowledge_docs = self.knowledge_store.similarity_search(
            query, k=self.config.knowledge_retrieval_k
        )
        return self._truncate_docs(report_docs + knowledge_docs)

    def stable_report_id(self, report_text: str) -> str:
        digest = hashlib.sha256(report_text.encode("utf-8")).hexdigest()
        return f"report-{digest[:16]}"

    def report_is_indexed(self, report_id: str) -> bool:
        records = self.report_store._collection.get(  # noqa: SLF001
            where={"report_id": report_id},
            limit=1,
            include=[],
        )
        return bool(records.get("ids"))

    @staticmethod
    def _load_knowledge_docs(knowledge_dir: Path) -> list[Document]:
        loader = DirectoryLoader(
            str(knowledge_dir),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        return loader.load()

    def _recreate_knowledge_collection(self) -> None:
        try:
            self.knowledge_store._client.delete_collection(self.config.knowledge_collection)  # noqa: SLF001
        except Exception:
            pass
        self.knowledge_store = Chroma(
            collection_name=self.config.knowledge_collection,
            embedding_function=self.embeddings,
            persist_directory=str(self.config.chroma_dir),
            client_settings=Settings(anonymized_telemetry=False),
        )

    @classmethod
    def _add_documents_in_batches(cls, store: Chroma, docs: list[Document]) -> None:
        for start in range(0, len(docs), cls.MAX_INSERT_BATCH_SIZE):
            stop = start + cls.MAX_INSERT_BATCH_SIZE
            batch = docs[start:stop]
            store.add_documents(batch)

    def _truncate_docs(self, docs: list[Document]) -> list[Document]:
        limit = self.config.retrieved_context_char_limit
        if limit <= 0:
            return docs

        total = 0
        clipped: list[Document] = []
        for doc in docs:
            remaining = limit - total
            if remaining <= 0:
                break
            content = doc.page_content[:remaining]
            clipped.append(Document(page_content=content, metadata=doc.metadata))
            total += len(content)
        return clipped
