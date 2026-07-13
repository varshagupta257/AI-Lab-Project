import os


# Project Paths


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DOCUMENT_FOLDER = os.path.join(BASE_DIR, "documents")

EMBEDDING_FOLDER = os.path.join(BASE_DIR, "embeddings")

DATABASE_FOLDER = os.path.join(BASE_DIR, "database")

UPLOAD_FOLDER = DOCUMENT_FOLDER


# Ollama

OLLAMA_MODEL = "llama3.2:latest"


# Upload Settings


ALLOWED_EXTENSIONS = {
    "pdf",
    "txt",
    "docx",
    "csv",
    "xlsx"
}

MAX_CONTENT_LENGTH = 50 * 1024 * 1024   # 50 MB


# Enterprise AI

APP_NAME = "Enterprise AI"

INDUSTRY = "Manufacturing"

ENABLE_CHAT_HISTORY = True

ENABLE_RAG = True