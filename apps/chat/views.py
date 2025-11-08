import logging

from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from django.contrib.auth.models import User
from .models import Chat, Message
from .services import ConversationService
from apps.accounts.models import UserProfile
from apps.documents.models import Document, DocumentChunk
import json

logger = logging.getLogger(__name__)

def ensure_user_profile(user):
    """Ensure user has a UserProfile, create if not exists"""
    try:
        return user.userprofile
    except UserProfile.DoesNotExist:
        return UserProfile.objects.create(user=user)

def get_guest_user():
    """Return a shared guest user account (created on first use)."""
    from django.contrib.auth.models import User
    guest, _ = User.objects.get_or_create(username='guest_user', defaults={'password': '!', 'email': 'guest@example.com'})
    return guest

def get_or_create_guest_chat(request):
    """Create or fetch the session-bound guest chat."""
    chat_id = request.session.get('guest_chat_id')
    if chat_id:
        try:
            return Chat.objects.get(external_id=chat_id)
        except Chat.DoesNotExist:
            pass
    # create new
    import uuid
    chat_id = str(uuid.uuid4())
    chat = Chat.objects.create(external_id=chat_id, user=get_guest_user(), title='Guest Chat')
    request.session['guest_chat_id'] = chat_id
    return chat

def dashboard_view(request):
    """Main chat dashboard"""
    if request.user.is_authenticated:
        ensure_user_profile(request.user)
        chats = Chat.objects.filter(user=request.user)
    else:
        # guest view: expose single guest chat ID
        chat = get_or_create_guest_chat(request)
        chats = [chat]
    return render(request, 'chat/dashboard.html', {'chats': chats})

def landing_view(request):
    """Public landing page. Redirect authenticated users to dashboard."""
    if request.user.is_authenticated:
        return redirect('chat:dashboard')
    return render(request, 'chat/landing_public.html')

@login_required
def chat_detail_view(request, chat_id):
    """Individual chat view"""
    chat = get_object_or_404(Chat, external_id=chat_id, user=request.user)
    messages = Message.objects.filter(chat=chat).order_by('created_at')
    return render(request, 'chat/chat_detail.html', {
        'chat': chat,
        'messages': messages
    })

# API Views
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_chats(request):
    """Get user's chats"""
    chats = Chat.objects.filter(user=request.user)
    return Response([{
        'id': chat.external_id,
        'title': chat.title,
        'created_at': chat.created_at
    } for chat in chats])

@api_view(['POST'])
@permission_classes([AllowAny])
def create_chat(request):
    """Create new chat"""
    logger.info("Create chat request from user %s", request.user)
    title = request.data.get('title', 'New Chat')
    logger.debug("Creating chat with title %s", title)
    
    try:
        if request.user.is_authenticated:
            logger.debug("Ensuring user profile for %s", request.user)
            user_profile = ensure_user_profile(request.user)
            logger.debug("User profile resolved: %s", user_profile)
            owner = request.user
        else:
            owner = get_guest_user()
        # Create a simple chat ID
        import uuid
        chat_id = str(uuid.uuid4())
        logger.debug("Generated chat ID %s", chat_id)
        
        # Create in Django
        logger.info("Creating chat in database for user %s", owner)
        chat = Chat.objects.create(
            external_id=chat_id,
            user=owner,
            title=title
        )
        logger.info("Chat %s created successfully", chat.id)
        
        return Response({
            'id': chat.external_id,
            'title': chat.title,
            'created_at': chat.created_at
        })
    except Exception as e:
        logger.exception("Error creating chat: %s", e)
        return Response({'error': str(e)}, status=500)

@api_view(['POST'])
@permission_classes([AllowAny])
def send_message(request):
    """Send message and get response"""
    chat_id = request.data.get('chat_id')
    message = request.data.get('message')
    use_rag = request.data.get('use_rag', True)
    
    if not chat_id or not message:
        return Response({'error': 'chat_id and message are required'}, status=400)
    
    try:
        if request.user.is_authenticated:
            chat = get_object_or_404(Chat, external_id=chat_id, user=request.user)
        else:
            # Guest: prefer session-bound chat, but if a valid guest chat_id was
            # created previously (e.g., in another tab) bind the session to it.
            guest_chat = get_or_create_guest_chat(request)
            if guest_chat.external_id != chat_id:
                try:
                    candidate = Chat.objects.get(external_id=chat_id, user=get_guest_user())
                    # Rebind session to this guest chat
                    request.session['guest_chat_id'] = candidate.external_id
                    chat = candidate
                except Chat.DoesNotExist:
                    return Response({'error': 'Invalid chat for guest'}, status=403)
            else:
                chat = guest_chat
        
        # Save user message to database
        Message.objects.create(
            chat=chat,
            role='user',
            content=message,
            sources=[]
        )
        
        # Generate AI response with conversation context
        conversation_service = ConversationService()
        response, sources = conversation_service.generate_response_with_context(
            message, chat_id, use_rag
        )
        
        # Save assistant response to database
        Message.objects.create(
            chat=chat,
            role='assistant',
            content=response,
            sources=sources
        )
        
        return Response({
            'response': response,
            'sources': sources
        })
    except Exception as e:
        return Response({'error': str(e)}, status=500)

@api_view(['GET'])
@permission_classes([AllowAny])
def get_messages(request, chat_id):
    """Get messages for a chat"""
    try:
        if request.user.is_authenticated:
            chat = get_object_or_404(Chat, external_id=chat_id, user=request.user)
        else:
            chat = get_or_create_guest_chat(request)
            if chat.external_id != chat_id:
                try:
                    candidate = Chat.objects.get(external_id=chat_id, user=get_guest_user())
                    request.session['guest_chat_id'] = candidate.external_id
                    chat = candidate
                except Chat.DoesNotExist:
                    return Response({'error': 'Invalid chat for guest'}, status=403)
        messages = Message.objects.filter(chat=chat).order_by('created_at')
        return Response([{
            'role': message.role,
            'content': message.content,
            'sources': message.sources or [],
            'created_at': message.created_at
        } for message in messages])
    except Exception as e:
        return Response({'error': str(e)}, status=500)

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_chat(request, chat_id):
    """Delete a chat and all its associated data"""
    try:
        logger.info("Delete request for chat %s by user %s", chat_id, request.user)
        chat = get_object_or_404(Chat, external_id=chat_id, user=request.user)
        logger.debug("Located chat %s", chat.title)
        
        # Delete associated documents and chunks
        documents = Document.objects.filter(chat=chat)
        logger.debug("Found %s documents to delete for chat %s", documents.count(), chat_id)
        
        for document in documents:
            # Delete document chunks
            chunks = DocumentChunk.objects.filter(document=document)
            logger.debug("Deleting %s chunks for document %s", chunks.count(), document.id)
            chunks.delete()
            
            # Delete document file if it exists
            if document.file_path and document.file_path.name:
                try:
                    document.file_path.delete(save=False)
                    logger.debug("Deleted file %s", document.file_path.name)
                except Exception as e:
                    logger.warning("Error deleting file %s: %s", document.file_path.name, e)
        
        # Delete documents
        documents.delete()
        logger.debug("Deleted documents for chat %s", chat_id)
        
        # Delete messages
        messages = Message.objects.filter(chat=chat)
        logger.debug("Deleting %s messages for chat %s", messages.count(), chat_id)
        messages.delete()
        
        # Delete chat
        chat_title = chat.title
        chat.delete()
        logger.info("Deleted chat %s", chat_title)
        
        return Response({'success': True, 'message': 'Chat deleted successfully'})
        
    except Exception as e:
        logger.exception("Error deleting chat %s: %s", chat_id, e)
        return Response({'error': str(e)}, status=500)

@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def rename_chat(request, chat_id):
    """Rename a chat"""
    try:
        chat = get_object_or_404(Chat, external_id=chat_id, user=request.user)
        new_title = request.data.get('title', '').strip()
        
        if not new_title:
            return Response({'error': 'Title cannot be empty'}, status=400)
        
        chat.title = new_title
        chat.save()
        
        return Response({
            'success': True, 
            'message': 'Chat renamed successfully',
            'new_title': new_title
        })
        
    except Exception as e:
        logger.exception("Error renaming chat %s: %s", chat_id, e)
        return Response({'error': str(e)}, status=500)

