from __future__ import annotations

import logging
import os
import tempfile
from typing import List, Sequence, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader

from apps.documents.models import Document, DocumentChunk
from apps.shared.services.ai_service import AIService
from apps.shared.services.pinecone_service import PineconeService

logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Handle PDF ingestion for RAG:
      1. Persist uploaded file.
      2. Extract page content.
      3. Chunk content with LangChain.
      4. Embed chunks.
      5. Sync vectors to Pinecone and persist chunk metadata.

    All steps mirror the original implementation; behaviour is unchanged.
    """

    def __init__(self) -> None:
        self.ai_service = AIService()
        self.pinecone_service = PineconeService()

    def process(self, document: Document, uploaded_file, chat_id: str) -> bool:
        temp_path = None
        try:
            logger.info("Starting PDF processing for document %s", document.id)
            temp_path = self._write_temp_file(uploaded_file)
            pages = self._load_pdf(temp_path)
            if not pages:
                logger.warning("No pages loaded from PDF")
                return False

            chunks = self._split_documents(pages)
            texts = [chunk.page_content for chunk in chunks]

            embeddings = self._embed_chunks(texts)
            self._persist_vectors(document, chat_id, chunks, texts, embeddings)
            self._store_chunks(document, chunks, embeddings)

            logger.info("Successfully processed PDF with %s chunks", len(chunks))
            return True
        except Exception as exc:
            logger.exception("PDF processing error: %s", exc)
            return False
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    logger.debug("Cleaned up temp file %s", temp_path)
                except Exception as cleanup_exc:
                    logger.warning("Failed to clean up temp file %s: %s", temp_path, cleanup_exc)

    def _write_temp_file(self, uploaded_file) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            for chunk in uploaded_file.chunks():
                tmp.write(chunk)
            temp_path = tmp.name
        logger.debug("Saved PDF to temp file %s", temp_path)
        return temp_path

    def _load_pdf(self, temp_path: str):
        pages = []
        try:
            logger.debug("Trying PyMuPDFLoader")
            loader = PyMuPDFLoader(temp_path)
            pages = loader.load()
            logger.debug("PyMuPDFLoader loaded %s pages", len(pages))
        except Exception as pymupdf_exc:
            logger.warning("PyMuPDFLoader failed: %s", pymupdf_exc)
            try:
                logger.debug("Trying PyPDFLoader")
                loader = PyPDFLoader(temp_path)
                pages = loader.load()
                logger.debug("PyPDFLoader loaded %s pages", len(pages))
            except Exception as pypdf_exc:
                logger.error("PyPDFLoader also failed: %s", pypdf_exc)
                raise Exception(f"Both PDF loaders failed: {pymupdf_exc}, {pypdf_exc}") from pypdf_exc
        return pages

    def _split_documents(self, pages):
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=400)
            chunks = splitter.split_documents(pages)
            logger.info("Split document into %s chunks", len(chunks))
            return chunks
        except Exception as exc:
            logger.error("Text splitting failed: %s", exc)
            raise

    def _embed_chunks(self, texts: Sequence[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        logger.info("Generating embeddings for %s chunks", len(texts))
        for i, text in enumerate(texts):
            try:
                embedding = self.ai_service.generate_embedding(text)
                embeddings.append(embedding)
                if i % 5 == 0:
                    logger.debug("Generated %s/%s embeddings", i + 1, len(texts))
            except Exception as exc:
                logger.warning("Embedding generation failed for chunk %s: %s", i, exc)
                embeddings.append([0.0] * 768)
        logger.info("Generated %s embeddings", len(embeddings))
        return embeddings

    def _persist_vectors(self, document: Document, chat_id: str, chunks, texts, embeddings):
        try:
            if not self.pinecone_service.index:
                logger.info("Pinecone not available, skipping vector storage")
                return

            logger.debug("Clearing old vectors for chat %s", chat_id)
            self.pinecone_service.clear_old_vectors(chat_id)

            logger.info("Storing vectors in Pinecone for document %s", document.id)
            vectors_to_upsert: List[Tuple[str, List[float], dict]] = []
            for i, (embedding, text, chunk) in enumerate(zip(embeddings, texts, chunks)):
                vector_id = f"{document.id}_{i}"
                metadata = {
                    "pdf_id": str(document.id),
                    "chunk_id": i,
                    "page": chunk.metadata.get("page", 0),
                    "text": text[:1000],
                    "chat_id": chat_id,
                }
                vectors_to_upsert.append((vector_id, embedding, metadata))

            self.pinecone_service.upsert_vectors(vectors_to_upsert)
            logger.info("Stored %s vectors in Pinecone", len(vectors_to_upsert))
        except Exception as exc:
            logger.exception("Pinecone storage failed: %s", exc)

    def _store_chunks(self, document: Document, chunks, embeddings):
        logger.debug("Storing chunks in database for document %s", document.id)
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            try:
                DocumentChunk.objects.create(
                    document=document,
                    chunk_id=i,
                    page_number=chunk.metadata.get("page", 0),
                    content=chunk.page_content,
                    embedding=embedding if i < len(embeddings) else [],
                )
            except Exception as exc:
                logger.warning("Failed to store chunk %s: %s", i, exc)

