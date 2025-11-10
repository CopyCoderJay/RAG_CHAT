"""
URL configuration for rag_chatbot project.
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView
from apps.chat import views as chat_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('apps.accounts.urls')),
    path('chat/', include('apps.chat.urls')),
    path('documents/', include('apps.documents.urls')),
    path('api/', include('apps.chat.api_urls')),
    path('manifest.webmanifest', TemplateView.as_view(
        template_name='manifest.webmanifest',
        content_type='application/manifest+json'
    ), name='pwa-manifest'),
    path('service-worker.js', TemplateView.as_view(
        template_name='service-worker.js',
        content_type='application/javascript'
    ), name='service-worker'),
    # Public landing page for unauthenticated users
    path('', chat_views.landing_view, name='landing'),
]

# Serve static and media files in development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

