web: bash -c "python manage.py migrate && python manage.py collectstatic --noinput && gunicorn rag_chatbot.wsgi:application --bind 0.0.0.0:${PORT:-8000}"
