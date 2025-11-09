import logging
from typing import List, Optional, Sequence

import requests
from django.conf import settings

logger = logging.getLogger(__name__)


class AIService:
    """Service for AI interactions using the Hugging Face Router API."""

    def __init__(self):
        self.hf_token: Optional[str] = settings.HF_TOKEN
        self.model: str = settings.MODEL
        self.timeout: int = getattr(settings, "HF_TIMEOUT", 90)
        self.base_url: str = getattr(
            settings,
            "HF_API_URL",
            "https://router.huggingface.co/hf-inference",
        ).rstrip("/")
        self.chat_url: str = f"{self.base_url}/v1/chat/completions"
        self.embedding_url: str = f"{self.base_url}/v1/embeddings"

        self.session = requests.Session()
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"
        else:
            logger.warning("HF_TOKEN is not configured; responses will use fallback messaging.")
        self.session.headers.update(headers)

    # --------------------------------------------------------------------- #
    # Chat completion
    # --------------------------------------------------------------------- #
    def generate_response(self, message: str, conversation_history: Optional[Sequence[dict]] = None) -> str:
        """Generate an AI response, falling back to canned text on failure."""
        if not self.hf_token:
            return self._generate_fallback_response(message, conversation_history)

        payload = {
            "model": self.model,
            "messages": self._build_messages(message, conversation_history),
            "max_tokens": 900,
            "temperature": 0.4,
        }

        try:
            logger.debug("Sending chat completion request to Hugging Face router.")
            response = self.session.post(self.chat_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                logger.warning("Chat completion response missing choices: %s", data)
                return self._generate_fallback_response(message, conversation_history)

            content = (
                choices[0]
                .get("message", {})
                .get("content", "")
            )
            if not content:
                logger.warning("Chat completion choice missing content: %s", choices[0])
                return self._generate_fallback_response(message, conversation_history)

            logger.debug("AI generated response length=%s", len(content))
            return content.strip()

        except requests.HTTPError as http_error:
            logger.error(
                "Hugging Face router returned HTTP error %s: %s",
                http_error.response.status_code if http_error.response else "unknown",
                http_error,
            )
        except Exception as exc:
            logger.exception("Unexpected error during chat completion: %s", exc)

        return self._generate_fallback_response(message, conversation_history)

    def _build_messages(self, current_message: str, conversation_history: Optional[Sequence[dict]]) -> List[dict]:
        messages: List[dict] = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Respond concisely and clearly.",
            }
        ]

        if conversation_history:
            for msg in conversation_history[-6:]:
                role = "user" if msg.get("role") == "user" else "assistant"
                content = msg.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": current_message})
        return messages

    # --------------------------------------------------------------------- #
    # Fallback responses
    # --------------------------------------------------------------------- #
    def _generate_fallback_response(self, message: str, conversation_history: Optional[Sequence[dict]]) -> str:
        """Generate an informative fallback reply when we cannot reach the LLM."""
        message_lower = message.lower()

        if any(word in message_lower for word in ["pdf", "document", "file", "upload"]):
            return (
                "I can access the documents you've uploaded, but the language model is temporarily unavailable. "
                "Please try again in a few moments so I can analyse the content and share the details with you."
            )

        if any(word in message_lower for word in ["hello", "hi", "hey", "greetings"]):
            return (
                "Hello! I'm standing by to help, but my main language model is currently unavailable. "
                "Feel free to upload documents or share more context, and I'll respond once the service is restored."
            )

        if any(word in message_lower for word in ["thank", "thanks"]):
            return "You're very welcome! Let me know if there's anything else you need once the AI service is back."

        return (
            "I'm ready to help with that as soon as the language model is back online. "
            "Please retry in a moment, or provide any additional context you'd like me to consider."
        )

    # --------------------------------------------------------------------- #
    # Embeddings
    # --------------------------------------------------------------------- #
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings using the Hugging Face router API."""
        if not text.strip():
            return [0.0] * 768

        if not self.hf_token:
            logger.warning("Embedding requested but HF token is missing. Returning zero vector.")
            return [0.0] * 768

        payload = {
            "model": "sentence-transformers/all-mpnet-base-v2",
            "input": text,
        }

        try:
            response = self.session.post(self.embedding_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            embedding = (
                data.get("data", [{}])[0]
                .get("embedding")
            )
            if embedding:
                return embedding
        except requests.HTTPError as http_error:
            logger.error(
                "Embedding request failed with HTTP %s: %s",
                http_error.response.status_code if http_error.response else "unknown",
                http_error,
            )
        except Exception as exc:
            logger.exception("Unexpected error during embedding generation: %s", exc)

        logger.warning("Falling back to zero vector for embedding.")
        return [0.0] * 768

    # --------------------------------------------------------------------- #
    # Retrieval helpers (unchanged logic)
    # --------------------------------------------------------------------- #
    def retrieve_documents(self, query: str, top_k: int = 3, chat_id: Optional[str] = None) -> List[dict]:
        """Retrieve relevant document chunks using Pinecone with database fallback."""
        try:
            from .pinecone_service import PineconeService

            pinecone_service = PineconeService()

            logger.debug("RAG query: %s", query)
            if chat_id:
                logger.debug("RAG filtering for chat: %s", chat_id)

            sources: List[dict] = []
            if pinecone_service.index:
                try:
                    query_embedding = self.generate_embedding(query)
                    logger.debug("RAG query embedding length: %s", len(query_embedding))

                    results = pinecone_service.query_vectors(query_embedding, top_k)
                    logger.debug("RAG Pinecone returned %s results", len(results))

                    for i, match in enumerate(results):
                        logger.debug(
                            "RAG processing match %s score %s metadata keys %s",
                            i + 1,
                            getattr(match, "score", None),
                            list(getattr(match, "metadata", {}).keys()),
                        )

                        if chat_id and match.metadata.get("chat_id") != chat_id:
                            logger.debug(
                                "RAG skipping match %s due to chat mismatch (%s)",
                                i + 1,
                                match.metadata.get("chat_id"),
                            )
                            continue

                        text_content = match.metadata.get("text", "")
                        if not text_content:
                            try:
                                from apps.documents.models import DocumentChunk

                                chunk_id = match.metadata.get("chunk_id", 0)
                                pdf_id = match.metadata.get("pdf_id")
                                logger.debug("RAG looking up chunk %s in document %s", chunk_id, pdf_id)
                                if pdf_id:
                                    chunk = DocumentChunk.objects.filter(
                                        document_id=pdf_id,
                                        chunk_id=chunk_id,
                                    ).first()
                                    if chunk:
                                        text_content = chunk.content
                                        logger.debug("RAG retrieved chunk content length %s", len(text_content))
                            except Exception as exc:
                                logger.warning("Error retrieving chunk content: %s", exc)

                        if text_content and text_content.strip():
                            sources.append(
                                {
                                    "text": text_content,
                                    "page": match.metadata.get("page", 0),
                                    "score": match.score,
                                }
                            )
                            logger.debug(
                                "RAG added source %s with length %s",
                                len(sources),
                                len(text_content),
                            )
                except Exception as exc:
                    logger.exception("Pinecone query failed: %s", exc)
            else:
                logger.info("RAG Pinecone index not available")

            # Database fallback
            if not sources and chat_id:
                logger.info("RAG no Pinecone sources; trying database fallback")
                try:
                    from apps.chat.models import Chat
                    from apps.documents.models import Document, DocumentChunk

                    chat = Chat.objects.get(external_id=chat_id)
                    documents = Document.objects.filter(chat=chat)
                    logger.debug(
                        "RAG database fallback found %s documents for chat %s",
                        documents.count(),
                        chat_id,
                    )

                    chunks = DocumentChunk.objects.filter(document__in=documents)
                    logger.debug("RAG database fallback found %s chunks", chunks.count())

                    query_words = query.lower().split()
                    for chunk in chunks[:top_k]:
                        content_lower = chunk.content.lower()
                        if any(word in content_lower for word in query_words):
                            sources.append(
                                {
                                    "text": chunk.content,
                                    "page": chunk.page_number,
                                    "score": 0.8,
                                }
                            )
                            logger.debug(
                                "RAG database fallback added source length %s", len(chunk.content)
                            )

                    if not sources:
                        logger.info("RAG database fallback using first available chunks")
                        for chunk in chunks[:top_k]:
                            sources.append(
                                {
                                    "text": chunk.content,
                                    "page": chunk.page_number,
                                    "score": 0.5,
                                }
                            )
                except Exception as exc:
                    logger.exception("RAG database fallback failed: %s", exc)

            logger.debug("RAG returning %s sources", len(sources))
            return sources

        except Exception as exc:
            logger.exception("Document retrieval failed: %s", exc)
            return []

