import logging
from typing import List, Optional, Sequence

from django.conf import settings
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)


class AIService:
    """Service for AI interactions using Hugging Face's InferenceClient."""

    def __init__(self):
        self.hf_token: Optional[str] = settings.HF_TOKEN
        self.model: str = settings.MODEL
        self.embedding_model: str = getattr(
            settings, "EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"
        )
        self.timeout: int = getattr(settings, "HF_TIMEOUT", 90)
        self.api_url: str = getattr(
            settings,
            "HF_API_URL",
            "https://router.huggingface.co/hf-inference",
        )

        if not self.hf_token:
            logger.warning(
                "HF_TOKEN is not configured; the service will respond with fallback messaging."
            )
            self.llm_client = None
            self.embedding_client = None
            return

        try:
            self.llm_client = InferenceClient(
                model=self.model,
                token=self.hf_token,
                api_url=self.api_url,
                timeout=self.timeout,
            )
        except Exception as exc:
            logger.exception("Failed to initialise Hugging Face client for model %s: %s", self.model, exc)
            self.llm_client = None

        try:
            self.embedding_client = InferenceClient(
                model=self.embedding_model,
                token=self.hf_token,
                api_url=self.api_url,
                timeout=self.timeout,
            )
        except Exception as exc:
            logger.exception(
                "Failed to initialise Hugging Face client for embedding model %s: %s",
                self.embedding_model,
                exc,
            )
            self.embedding_client = None

    # ------------------------------------------------------------------ #
    # Chat completion
    # ------------------------------------------------------------------ #
    def generate_response(
        self,
        message: str,
        conversation_history: Optional[Sequence[dict]] = None,
    ) -> str:
        if not self.llm_client:
            return self._generate_fallback_response(message, conversation_history)

        messages = self._build_messages(message, conversation_history)

        try:
            if hasattr(self.llm_client, "chat_completion"):
                response = self.llm_client.chat_completion(
                    model=self.model,
                    messages=messages,
                    max_tokens=900,
                    temperature=0.4,
                )
                choices = getattr(response, "choices", None)
                if choices:
                    content = choices[0].message.get("content", "")
                    if content:
                        return content.strip()
                    logger.warning("Chat completion returned empty content: %s", choices[0])
                else:
                    logger.warning("Chat completion returned no choices: %s", response)
            else:
                # Fallback for older client versions: build a single prompt and call text_generation.
                prompt = self._messages_to_prompt(messages)
                logger.debug("chat_completion unavailable; falling back to text_generation")
                text = self.llm_client.text_generation(
                    prompt,
                    max_new_tokens=900,
                    temperature=0.4,
                )
                if text:
                    return text.strip()
        except Exception as exc:
            logger.exception("Error calling Hugging Face chat completion: %s", exc)

        return self._generate_fallback_response(message, conversation_history)

    def _build_messages(
        self,
        current_message: str,
        conversation_history: Optional[Sequence[dict]],
    ) -> List[dict]:
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

    @staticmethod
    def _messages_to_prompt(messages: Sequence[dict]) -> str:
        """Serialize chat history into a text prompt for text_generation fallback."""
        prompt_parts: List[str] = []
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)

    # ------------------------------------------------------------------ #
    # Fallback response
    # ------------------------------------------------------------------ #
    def _generate_fallback_response(
        self,
        message: str,
        conversation_history: Optional[Sequence[dict]],
    ) -> str:
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

    # ------------------------------------------------------------------ #
    # Embeddings
    # ------------------------------------------------------------------ #
    def generate_embedding(self, text: str) -> List[float]:
        if not text.strip():
            return [0.0] * 768

        if not self.embedding_client:
            logger.warning("Embedding client is unavailable. Returning a zero vector.")
            return [0.0] * 768

        try:
            embedding = self.embedding_client.feature_extraction(text)
            if isinstance(embedding, list):
                # feature_extraction may return [vector] or [[vector]]
                if embedding and isinstance(embedding[0], list):
                    return embedding[0]
                return embedding
        except Exception as exc:
            logger.exception("Error generating embedding via Hugging Face: %s", exc)

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

