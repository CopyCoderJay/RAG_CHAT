import logging

from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from .models import Document, DocumentChunk
from apps.chat.models import Chat
from .services.pdf_processor import PDFProcessor
from apps.shared.services.pinecone_service import PineconeService

logger = logging.getLogger(__name__)

@login_required
def upload_document(request, chat_id):
    """Upload and process PDF document"""
    try:
        chat = get_object_or_404(Chat, external_id=chat_id, user=request.user)
        logger.info("Upload request for chat %s by user %s", chat_id, request.user)
        
        if request.method == 'POST':
            uploaded_file = request.FILES.get('file')
            logger.debug("Uploaded file %s", uploaded_file)
            
            if not uploaded_file:
                return JsonResponse({'success': False, 'error': 'No file provided'})
            
            if not uploaded_file.name.lower().endswith('.pdf'):
                return JsonResponse({'success': False, 'error': 'Please upload a PDF file'})
            
            try:
                logger.debug("Creating document record for %s", uploaded_file.name)
                # Create document record
                document = Document.objects.create(
                    chat=chat,
                    filename=uploaded_file.name,
                    file_path=uploaded_file
                )
                logger.info("Created document %s for chat %s", document.id, chat_id)
                
                # Process PDF
                logger.info("Starting PDF processing for chat %s", chat_id)
                processor = PDFProcessor()
                success = processor.process(document, uploaded_file, chat_id)
                logger.info("PDF processing result for document %s: %s", document.id, success)
                
                if success:
                    return JsonResponse({
                        'success': True, 
                        'message': f'PDF "{uploaded_file.name}" uploaded and processed successfully!'
                    })
                else:
                    return JsonResponse({
                        'success': False, 
                        'error': 'PDF uploaded but processing failed. Check server logs for details.'
                    })
                    
            except Exception as e:
                logger.exception("Upload failed: %s", e)
                return JsonResponse({'success': False, 'error': f'Upload failed: {str(e)}'})
        
        return render(request, 'documents/upload.html', {'chat': chat})
    
    except Exception as e:
        logger.exception("Upload view error: %s", e)
        return JsonResponse({'success': False, 'error': f'Server error: {str(e)}'})

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_chat_documents(request, chat_id):
    """Get documents for a chat"""
    try:
        logger.debug("Fetching documents for chat %s by user %s", chat_id, request.user)
        chat = get_object_or_404(Chat, external_id=chat_id, user=request.user)
        documents = Document.objects.filter(chat=chat)
        logger.debug("Found %s documents for chat %s", documents.count(), chat_id)
        
        result = [{
            'id': str(doc.id),
            'filename': doc.filename,
            'created_at': doc.uploaded_at
        } for doc in documents]
        
        logger.debug("Returning documents metadata for chat %s", chat_id)
        return Response(result)
    except Exception as e:
        logger.exception("Error fetching documents: %s", e)
        return Response({'error': str(e)}, status=500)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def view_document(request, document_id):
    """Get document chunks for viewing"""
    try:
        document = get_object_or_404(Document, id=document_id, chat__user=request.user)
        chunks = DocumentChunk.objects.filter(document=document).order_by('page_number', 'chunk_id')
        
        return Response({
            'success': True,
            'document': {
                'id': str(document.id),
                'filename': document.filename,
                'created_at': document.created_at
            },
            'chunks': [{
                'chunk_id': chunk.chunk_id,
                'page_number': chunk.page_number,
                'content': chunk.content
            } for chunk in chunks]
        })
    except Exception as e:
        return Response({'success': False, 'error': str(e)}, status=500)

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_document(request, document_id):
    """Delete a document and all its associated data"""
    try:
        logger.info("Delete document request for %s by user %s", document_id, request.user)
        document = get_object_or_404(Document, id=document_id, chat__user=request.user)
        logger.debug("Found document %s (%s)", document.id, document.filename)
        
        # Delete document chunks
        chunks = DocumentChunk.objects.filter(document=document)
        logger.debug("Deleting %s chunks for document %s", chunks.count(), document.id)
        chunks.delete()
        
        # Delete from Pinecone if available
        try:
            from apps.shared.services.pinecone_service import PineconeService
            pinecone_service = PineconeService()
            if pinecone_service.index:
                # Delete vectors with document ID
                pinecone_service.delete_vectors([f"{document.id}_{i}" for i in range(100)])  # Delete up to 100 chunks
                logger.debug("Deleted Pinecone vectors for document %s", document.id)
        except Exception as e:
            logger.warning("Error deleting from Pinecone for document %s: %s", document.id, e)
        
        # Delete document file if it exists
        if document.file_path and document.file_path.name:
            try:
                document.file_path.delete(save=False)
                logger.debug("Deleted file %s", document.file_path.name)
            except Exception as e:
                logger.warning("Error deleting file %s: %s", document.file_path.name, e)
        
        # Delete document record
        document_filename = document.filename
        document.delete()
        logger.info("Deleted document %s", document_filename)
        
        return Response({
            'success': True, 
            'message': f'Document "{document_filename}" deleted successfully'
        })
        
    except Exception as e:
        logger.exception("Error deleting document %s: %s", document_id, e)
        return Response({'error': str(e)}, status=500)

