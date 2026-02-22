"""
OppForge AI Engine Configuration
Loads environment variables and provides config to all agents.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env from backend (shared config)
backend_env = Path(__file__).parent.parent / "backend" / ".env"
if backend_env.exists():
    load_dotenv(backend_env)

# AI & LLM Settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Ollama (Fallback)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")

# Vector Database
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chromadb")
CHROMA_COLLECTION = "opportunities"

# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 400MB, fast
# Advanced RAG: Cross-Encoder for highly accurate re-ranking
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Lightweight and very accurate

# Database (shared with backend)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///../backend/oppforge.db")

# API Settings
AI_ENGINE_HOST = "0.0.0.0"
AI_ENGINE_PORT = 8001

# Agent Settings
CLASSIFIER_TEMPERATURE = 0.3  # More deterministic
SCORER_TEMPERATURE = 0.7      # Balanced
CHAT_TEMPERATURE = 0.8        # More creative
RISK_TEMPERATURE = 0.2        # Very deterministic

MAX_TOKENS = 500
TIMEOUT = 30  # seconds

# Feature Flags
ENABLE_CHROMADB = True
ENABLE_LANGCHAIN = True
ENABLE_OLLAMA_FALLBACK = False  # Set True if Groq fails

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

print(f"✅ AI Engine Config Loaded")
print(f"   - Groq Model: {GROQ_MODEL}")
print(f"   - Embedding: {EMBEDDING_MODEL}")
print(f"   - ChromaDB: {CHROMA_PATH}")
print(f"   - Database: {DATABASE_URL[:30]}...")
