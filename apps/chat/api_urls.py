from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.get_chats, name='get_chats'),
    path('chat/create/', views.create_chat, name='create_chat'),
    path('chat/send-message/', views.send_message, name='send_message'),
    path('chat/<str:chat_id>/messages/', views.get_messages, name='get_messages'),
    path('chat/<str:chat_id>/delete/', views.delete_chat, name='delete_chat'),
    path('chat/<str:chat_id>/rename/', views.rename_chat, name='rename_chat'),
]

