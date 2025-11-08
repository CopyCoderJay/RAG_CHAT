import logging

from django.conf import settings
from .models import Chat, Message
from apps.shared.services.ai_service import AIService
from apps.shared.services.pinecone_service import PineconeService

logger = logging.getLogger(__name__)

class ConversationService:
    """Service for managing conversation context and AI responses"""
    
    def __init__(self):
        self.ai_service = AIService()
        self.pinecone_service = PineconeService()
    
    def get_conversation_context(self, chat_id, limit=10):
        """Get recent conversation history for context"""
        try:
            chat = Chat.objects.get(external_id=chat_id)
            messages = Message.objects.filter(chat=chat).order_by('-created_at')[:limit]
            return [{
                'role': msg.role,
                'content': msg.content,
                'created_at': msg.created_at
            } for msg in reversed(messages)]
        except Chat.DoesNotExist:
            return []
    
    def generate_response_with_context(self, message, chat_id, use_rag=True):
        """Generate AI response with conversation context"""
        try:
            # Get conversation history
            conversation_history = self.get_conversation_context(chat_id)
            
            # If RAG is enabled, retrieve relevant documents
            sources = []
            rag_context = ""
            if use_rag:
                try:
                    logger.info("RAG search for chat %s message: %s", chat_id, message[:100])
                    sources = self.ai_service.retrieve_documents(message, chat_id=chat_id)
                    logger.info("RAG found %s sources", len(sources))
                    
                    # Debug: Print what we're actually retrieving
                    for i, source in enumerate(sources):
                        logger.debug(
                            "RAG source %s page %s score %s preview: %s",
                            i + 1,
                            source.get('page', 'unknown'),
                            source.get('score', 'unknown'),
                            source.get('text', '')[:200],
                        )
                    
                    # Build context from retrieved documents
                    if sources:
                        rag_context = "\n\nRelevant information from uploaded documents:\n"
                        for i, source in enumerate(sources):
                            text_content = source.get('text', '')
                            logger.debug("RAG source %s text length %s", i + 1, len(text_content))
                            if text_content:  # Only add non-empty content
                                rag_context += f"[Source {i+1}]: {text_content}\n"
                        logger.debug("RAG context length %s", len(rag_context))
                    else:
                        logger.info("RAG found no sources")
                except Exception as e:
                    logger.exception("RAG retrieval failed: %s", e)
                    sources = []
            
            # Generate AI response with RAG context
            if rag_context:
                # Add RAG context to the conversation
                enhanced_message = f"{message}\n\n{rag_context}"
                logger.debug("RAG enhanced message preview: %s", enhanced_message[:500])
                response = self.ai_service.generate_response(enhanced_message, conversation_history)
            else:
                logger.debug("RAG context empty; using plain generation")
                response = self.ai_service.generate_response(message, conversation_history)
            
            return response, sources
            
        except Exception as e:
            logger.exception("Error generating AI response: %s", e)
            return f"Sorry, I encountered an error while processing your message: {str(e)}", []
    
    def build_context_prompt(self, conversation_history, current_message):
        """Build a context-aware prompt for the AI"""
        if not conversation_history:
            return f"Human: {current_message}\n\nAssistant:"
        
        context = "You are a helpful AI assistant. Here's our conversation so far:\n\n"
        for msg in conversation_history[-6:]:  # Last 6 messages for context
            role = "Human" if msg['role'] == 'user' else "Assistant"
            context += f"{role}: {msg['content']}\n"
        
        context += f"\nHuman: {current_message}\n\nAssistant:"
        return context