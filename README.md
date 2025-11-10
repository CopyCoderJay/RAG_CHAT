# AtlasMind

AtlasMind is a production-ready Retrieval-Augmented Generation (RAG) chatbot built on Django and Django REST Framework. The platform lets teams upload PDF knowledge bases, converts them into dense vector stores with Hugging Face embeddings, and serves contextual answers via the Pinecone vector database. Responses are grounded, cited, and formatted for readability across devices. A document-first chat interface keeps uploads visible, highlights which PDF is active, and supports inline question/answer follow-ups. AtlasMind ships with a polished landing page, progressive web app support, and a mobile-optimized layout, ensuring end users can install the assistant or run it straight from the browser.

For developers, the codebase exposes service layers that encapsulate ingestion (`PDFProcessor`), AI orchestration (`AIService`), and retrieval (`PineconeService`). LangChain handles chunking, Pinecone powers low-latency semantic search, and the Hugging Face Inference Router provides both chat completions and embeddings. Configuration relies on environment variables with fail-fast checks for production, enabling fast deployment to hosts like Railway or Render. In short, AtlasMind delivers a full-stack RAG assistant that is easy to customize, integrates modern AI tooling, and surfaces reliable, explainable responses from enterprise documents.

## ✨ Features

- **Document-first Chat** – Upload PDFs, keep them visible alongside the conversation, and route follow-up questions to the same document until you swap it.
- **RAG Workflow** – Uses LangChain for chunking, Pinecone for retrieval, and Hugging Face models for both chat completions and embeddings.
- **Professional UI** – Desktop & mobile responsive UI (now PWA-capable) with a polished landing page, upload progress toast, inline document chips, and structured assistant responses.
- **Progressive Web App** – Installable on desktop/mobile; works offline for static assets via service worker & manifest.
- **Structured Responses** – Primitive formatting keeps tables, bullet lists, and paragraphs readable across screen sizes.
- **Production-ready stack** – Django 4.2, DRF 3.14, Postgres (Railway-ready), Pinecone 5.x, and huggingface-hub ≥ 0.25.

## 🧱 Architecture

```
.
├── apps/
│   ├── accounts/          # Auth & user management
│   ├── ai/                # AI orchestration utilities
│   ├── chat/              # Chat views, APIs, routing
│   └── documents/         # Document upload, chunking, persistence
├── templates/             # Django templates (landing, dashboard, PWA assets)
├── static/
│   ├── css/chatgpt.css    # Core styling
│   ├── js/chat.js         # Frontend interactions
│   └── img/pwa-icon.svg   # PWA icon
├── requirements.txt
├── Dockerfile
└── rag_chatbot/settings.py
```

## 🧰 Tech Stack

### Backend
- **Django 4.2** – Core web framework, authentication, ORM, templating.
- **Django REST Framework 3.14** – RESTful API endpoints for chat, documents, and authentication flows.
- **PostgreSQL** – Production database (SQLite used for local development).
- **Psycopg2** – Postgres adapter with connection pooling.
- **Gunicorn + WhiteNoise** – Production WSGI server & static asset serving.
- **Structured Logging** – Configured via Django logging, tuned for Railway/Render logs.

### RAG & AI Layer
- **LangChain** – Chunking (`RecursiveCharacterTextSplitter`) and orchestration utilities.
- **Pinecone 5.x** – Vector database for embeddings with helper service wrappers.
- **Hugging Face hub 0.25** – Chat completions & embeddings via the inference router.
- **Custom service abstraction** – `AIService`, `PDFProcessor`, and Pinecone service encapsulate external calls with retry & fallback logic.
- **PyMuPDF / PyPDF** – Robust PDF parsing with fallback to ensure uploads always ingest.

### Frontend & UX
- **Django templates + vanilla JS** – Lightweight UI without a SPA framework.
- **CSS (chatgpt.css)** – Tailored responsive styles, message formatting, upload UI, toasts, mobile input bar.
- **PWA** – `manifest.webmanifest`, `service-worker.js`, and `pwa-icon.svg` allow installable experience and offline asset caching.
- **Responsive layout** – Landing page and chat dashboard designed for both desktop and mobile (clamp-based spacing, flex/grid layouts).

### Deployment & Tooling
- **Dockerfile** – Slim Python 3.11 image with build steps (collectstatic, gunicorn entrypoint).
- **Environment management** – `.env` (dotenv), fail-fast config for production if proper DB & API keys not set.
- **Railway / Render** – Defaults target containerized hosting with Postgres.

## ⚙️ Prerequisites

- Python 3.11
- PostgreSQL (Railway/production) or SQLite locally
- API keys:
  - `PINECONE_API_KEY`
  - `HF_TOKEN` (for chat + embeddings)

## 📦 Installation

```bash
git clone <repo>
cd RAG_Chatbot-main
python -m venv .venv
source .venv/bin/activate      # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Copy the environment template and fill in your secrets:

```bash
cp env.example .env
```

Key environment variables:

```
DEBUG=True
HF_TOKEN=...
MODEL=meta-llama/Meta-Llama-3-8B-Instruct
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=rag-chat-index
DATABASE_URL=postgres://...
```

## 🏃‍♂️ Running Locally

```bash
python manage.py migrate
python manage.py runserver
```

Visit `http://127.0.0.1:8000/` for the landing page (unauthenticated) or `/chat/` after logging in.

### Progressive Web App

The PWA manifest & service worker are served at:

- `/manifest.webmanifest`
- `/service-worker.js`

Icons are bundled under `static/img/pwa-icon.svg`. Installation prompts show automatically on supported browsers.

## 🐳 Docker (Optional)

```bash
docker build -t rag-chat .
docker run -p 8000:8000 --env-file .env rag-chat
```

The image collects static files, runs migrations, and launches Gunicorn.

## 🚀 Deployment Notes

- Designed for Railway, Render, or any container host.
- Ensure `DEBUG=False` and proper `ALLOWED_HOSTS`.
- Set `DATABASE_URL` for Postgres; SQLite is only allowed during local dev.
- Static files served by WhiteNoise (`STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"`).
- Service worker registration is environment-agnostic; ensure HTTPS in production.

## 🧠 RAG Pipeline Summary

1. Upload PDF → stored in Django, chunks persisted in DB.
2. LangChain chunking (RecursiveCharacterTextSplitter) → embeddings via Hugging Face.
3. Vectors synced to Pinecone (`pinecone-client==5.0.1`).
4. Chat requests call Hugging Face Inference Router (`https://router.huggingface.co`) with streaming fallback & graceful error handling.
5. Responses rendered with structured paragraphs, tables, and inline document chips.

## 🧪 Testing

```bash
python manage.py test
```

## 📚 Useful Commands

```bash
python manage.py createsuperuser
python manage.py collectstatic
python manage.py migrate
```

## 🙌 Acknowledgements

- [Django](https://www.djangoproject.com/)
- [DRF](https://www.django-rest-framework.org/)
- [LangChain](https://python.langchain.com/)
- [Pinecone](https://www.pinecone.io/)
- [Hugging Face Inference](https://huggingface.co/inference)
