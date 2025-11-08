import logging

from django.conf import settings
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

class AIService:
    """Service for AI interactions using Hugging Face"""
    
    def __init__(self):
        self.hf_token = settings.HF_TOKEN
        self.model = settings.MODEL
        try:
            self.llm_client = InferenceClient(model=self.model, token=self.hf_token)
        except Exception as e:
            logger.exception("Failed to initialize LLM client: %s", e)
            self.llm_client = None
    
    def generate_response(self, message, conversation_history=None):
        """Generate AI response using Hugging Face InferenceClient"""
        try:
            if not self.llm_client:
                logger.warning("LLM client not available; using fallback response")
                return self._generate_fallback_response(message, conversation_history)
            
            # Build conversation context
            if conversation_history:
                context = self._build_context(conversation_history)
                prompt = f"{context}\n\nHuman: {message}\n\nAssistant:"
            else:
                prompt = f"Human: {message}\n\nAssistant:"
            
            logger.debug("Generating AI response for prompt: %s", message[:100])
            
            # Try chat_completion first
            try:
                resp = self.llm_client.chat_completion(
                    model=self.model, 
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000, 
                    temperature=0.4, 
                    stream=False
                )
                # Extract response from chat completion
                if hasattr(resp, 'choices') and len(resp.choices) > 0:
                    response = resp.choices[0].message.content
                    logger.debug("AI generated response: %s", response[:100])
                    return response
                else:
                    logger.warning("Unexpected response format from chat_completion: %s", resp)
                    return str(resp)
            except Exception as e:
                logger.warning("Chat completion failed: %s. Falling back to text_generation.", e)
                try:
                    # Fallback to text_generation
                    gen = self.llm_client.text_generation(
                        prompt, 
                        max_new_tokens=2000, 
                        temperature=0.4
                    )
                    response = gen if isinstance(gen, str) else str(gen)
                    logger.debug("AI generated response via text_generation: %s", response[:100])
                    return response
                except Exception as e2:
                    logger.error("Text generation failed: %s", e2)
                    # Final fallback - return a helpful message with some intelligence
                    return self._generate_fallback_response(message, conversation_history)
                
        except Exception as e:
            logger.exception("Error generating AI response: %s", e)
            return f"Sorry, I encountered an error: {str(e)}"
    
    def _build_context(self, conversation_history):
        """Build conversation context from history"""
        context = "You are a helpful AI assistant. Here's our conversation so far:\n\n"
        for msg in conversation_history[-6:]:  # Last 6 messages for context
            role = "Human" if msg['role'] == 'user' else "Assistant"
            context += f"{role}: {msg['content']}\n"
        return context
    
    def _generate_fallback_response(self, message, conversation_history):
        """Generate a simple fallback response when AI service is down"""
        message_lower = message.lower()
        
        # Check if this is about uploaded documents
        if any(word in message_lower for word in ['name', 'candidate', 'pdf', 'document', 'file']):
            return (
                "I can access the documents you've uploaded, but my primary language model is temporarily unavailable. "
                "Please try again in a moment so I can analyze your files and share detailed findings."
            )
        
        # Simple pattern matching for common questions
        elif any(word in message_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return "Hello! I'm a RAG Chat Assistant. I can see you have uploaded documents, but I'm currently experiencing some technical difficulties with my AI service. I'm here to help once the service is restored!"
        
        elif any(word in message_lower for word in ['how are you', 'how do you do']):
            return "I'm doing well, thank you for asking! I'm a helpful AI assistant, though I'm currently running on backup systems due to some technical issues with my main AI service."
        
        elif any(word in message_lower for word in ['what can you do', 'help', 'assist']):
            return "I'm a RAG (Retrieval-Augmented Generation) Chat Assistant! I can help you analyze uploaded PDF documents and answer questions about their content. I'm currently running on backup systems, but I'm still here to help!"
        
        elif any(word in message_lower for word in ['thank', 'thanks']):
            return "You're very welcome! I'm happy to help. Is there anything else you'd like to know or discuss?"
        
        else:
            return (
                "I understand you're asking about this topic, and I'll be ready to help as soon as my language model is back online. "
                "Feel free to upload documents or add more context in the meantime."
            )
    
    def generate_embedding(self, text):
        """Generate embedding for text using Hugging Face"""
        try:
            # Use the embedding client
            from huggingface_hub import InferenceClient
            emb_client = InferenceClient(
                model="sentence-transformers/all-mpnet-base-v2", 
                token=self.hf_token
            )
            embedding = emb_client.feature_extraction(text)
            return embedding.tolist()
        except Exception as e:
            logger.warning("Embedding generation failed: %s", e)
            # Return zero vector as fallback
            return [0.0] * 768
    
    def retrieve_documents(self, query, top_k=3, chat_id=None):
        """Retrieve relevant documents using Pinecone for RAG"""
        try:
            from .pinecone_service import PineconeService
            pinecone_service = PineconeService()
            
            logger.debug("RAG query: %s", query)
            if chat_id:
                logger.debug("RAG filtering for chat: %s", chat_id)
            
            # First try Pinecone
            sources = []
            if pinecone_service.index:
                try:
                    # Generate query embedding
                    query_embedding = self.generate_embedding(query)
                    logger.debug("RAG query embedding length: %s", len(query_embedding))
                    
                    # Search in Pinecone
                    results = pinecone_service.query_vectors(query_embedding, top_k)
                    logger.debug("RAG Pinecone returned %s results", len(results))
                    
                    # Format results
                    for i, match in enumerate(results):
                        logger.debug(
                            "RAG processing match %s score %s metadata keys %s",
                            i + 1,
                            getattr(match, "score", None),
                            list(getattr(match, "metadata", {}).keys()),
                        )
                        
                        # Filter by chat_id if provided
                        if chat_id and match.metadata.get('chat_id') != chat_id:
                            logger.debug(
                                "RAG skipping match %s due to chat mismatch (%s)",
                                i + 1,
                                match.metadata.get('chat_id'),
                            )
                            continue
                        
                        # Get the full text from metadata or from database
                        text_content = match.metadata.get('text', '')
                        if not text_content:
                            # Try to get from database if not in metadata
                            try:
                                from apps.documents.models import DocumentChunk
                                chunk_id = match.metadata.get('chunk_id', 0)
                                pdf_id = match.metadata.get('pdf_id')
                                logger.debug(
                                    "RAG looking up chunk %s in document %s", chunk_id, pdf_id
                                )
                                if pdf_id:
                                    chunk = DocumentChunk.objects.filter(
                                        document_id=pdf_id,
                                        chunk_id=chunk_id
                                    ).first()
                                    if chunk:
                                        text_content = chunk.content
                                        logger.debug(
                                            "RAG retrieved chunk content length %s", len(text_content)
                                        )
                                    else:
                                        logger.debug("RAG chunk not found in database")
                            except Exception as e:
                                logger.warning("Error retrieving chunk content: %s", e)
                        
                        # Only add if we have actual content
                        if text_content and len(text_content.strip()) > 0:
                            sources.append({
                                'text': text_content,
                                'page': match.metadata.get('page', 0),
                                'score': match.score
                            })
                            logger.debug(
                                "RAG added source %s with length %s",
                                len(sources),
                                len(text_content),
                            )
                        else:
                            logger.debug("RAG skipping empty source %s", i + 1)
                except Exception as e:
                    logger.exception("Pinecone query failed: %s", e)
            else:
                logger.info("RAG Pinecone index not available")
            
            # If no sources from Pinecone, try database fallback
            if not sources and chat_id:
                logger.info("RAG no Pinecone sources; trying database fallback")
                try:
                    from apps.documents.models import Document, DocumentChunk
                    from apps.chat.models import Chat
                    
                    # Get chat and its documents
                    chat = Chat.objects.get(external_id=chat_id)
                    documents = Document.objects.filter(chat=chat)
                    logger.debug(
                        "RAG database fallback found %s documents for chat %s",
                        documents.count(),
                        chat_id,
                    )
                    
                    # Get all chunks from these documents
                    chunks = DocumentChunk.objects.filter(document__in=documents)
                    logger.debug("RAG database fallback found %s chunks", chunks.count())
                    
                    # Simple text matching fallback
                    query_words = query.lower().split()
                    for chunk in chunks[:top_k]:  # Limit to top_k chunks
                        content_lower = chunk.content.lower()
                        # Check if any query words are in the content
                        if any(word in content_lower for word in query_words):
                            sources.append({
                                'text': chunk.content,
                                'page': chunk.page_number,
                                'score': 0.8  # Default score for database fallback
                            })
                            logger.debug(
                                "RAG database fallback added source length %s", len(chunk.content)
                            )
                    
                    # If still no sources, just take the first few chunks
                    if not sources:
                        logger.info("RAG database fallback using first available chunks")
                        for chunk in chunks[:top_k]:
                            sources.append({
                                'text': chunk.content,
                                'page': chunk.page_number,
                                'score': 0.5  # Lower score for non-matching chunks
                            })
                            logger.debug(
                                "RAG fallback added chunk length %s", len(chunk.content)
                            )
                            
                except Exception as e:
                    logger.exception("RAG database fallback failed: %s", e)
            
            logger.debug("RAG returning %s sources", len(sources))
            return sources
        except Exception as e:
            logger.exception("Document retrieval failed: %s", e)
            return []