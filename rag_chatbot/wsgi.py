"""
WSGI config for rag_chatbot project.
"""
import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_chatbot.settings')

application = get_wsgi_application()

# Vercel requires this
app = application

