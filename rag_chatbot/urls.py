"""
URL configuration for rag_chatbot project.
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from apps.chat import views as chat_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('apps.accounts.urls')),
    path('chat/', include('apps.chat.urls')),
    path('documents/', include('apps.documents.urls')),
    path('api/', include('apps.chat.api_urls')),
    # Public landing page for unauthenticated users
    path('', chat_views.landing_view, name='landing'),
]

# Serve static and media files in development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

