import os
import sys
from django.core.wsgi import get_wsgi_application

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_chatbot.settings')

application = get_wsgi_application()

# Vercel handler
def handler(request):
    return application(request)

