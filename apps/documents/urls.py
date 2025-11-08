from django.urls import path
from . import views

app_name = 'documents'
urlpatterns = [
    path('upload/<str:chat_id>/', views.upload_document, name='upload'),
    path('api/chat/<str:chat_id>/documents/', views.get_chat_documents, name='get_documents'),
    path('api/view/<str:document_id>/', views.view_document, name='view_document'),
    path('api/delete/<str:document_id>/', views.delete_document, name='delete_document'),
]

