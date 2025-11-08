# Services package initialization
from .pinecone_service import PineconeService
from .ai_service import AIService

# Initialize services
pinecone_service = PineconeService()
ai_service = AIService()
