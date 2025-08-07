#!/bin/bash

# HarpoonAI Enhanced Installer with GPU Support, Vector Embeddings & Semantic Search

set -e  # Exit on error

echo "🚀 HarpoonAI Enhanced Installer Starting (Vector Embeddings & Semantic Search)..."
echo "📦 System: Fedora Plasma"

# Update system and install base dependencies
echo "📦 Installing system dependencies..."
sudo hostnamectl set-hostname harpoonai
sudo dnf update -y
sudo dnf install -y wget git python3-pip python3-devel net-tools gcc gcc-c++ make

# Install cmake and build tools
echo "🔧 Installing build tools..."
sudo dnf install -y cmake pkg-config
sudo dnf install -y libcurl-devel file-devel libmagic-devel openblas-devel lapack-devel --skip-unavailable

# Install NVIDIA drivers and CUDA (optional)
echo "🎮 Checking for NVIDIA GPU support..."
sudo dnf install -y https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm || true
sudo dnf install -y https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm || true

# Try to install NVIDIA drivers (will fail gracefully if no GPU)
sudo dnf install -y akmod-nvidia xorg-x11-drv-nvidia-cuda 2>/dev/null || echo "ℹ️ NVIDIA drivers not installed (GPU may not be present)"

# Check for NVIDIA GPU support
GPU_SUPPORT=""
CUDA_AVAILABLE=false

echo "🔍 Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "")
    if [ ! -z "$GPU_INFO" ]; then
        echo "✅ NVIDIA GPU detected: $GPU_INFO"
        
        if command -v nvcc &> /dev/null; then
            CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
            echo "✅ CUDA toolkit found: $CUDA_VERSION"
            CUDA_AVAILABLE=true
            GPU_SUPPORT="-DGGML_CUDA=ON"
        else
            echo "⚠️ NVIDIA GPU found but CUDA toolkit not installed"
            echo "📦 Attempting to install CUDA development tools..."
            sudo dnf install -y cuda-toolkit 2>/dev/null || sudo dnf install -y cuda-devel 2>/dev/null || echo "⚠️ Could not install CUDA automatically"
            
            if command -v nvcc &> /dev/null; then
                CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
                echo "✅ CUDA toolkit installed: $CUDA_VERSION"
                CUDA_AVAILABLE=true
                GPU_SUPPORT="-DGGML_CUDA=ON"
            fi
        fi
    fi
else
    echo "ℹ️ No NVIDIA GPU detected, using CPU-only build"
fi

# Set up project directories
PROJECT_DIR="$HOME/offline_ai_chat"
MODEL_DIR="$PROJECT_DIR/models"
BACKEND_DIR="$PROJECT_DIR/backend"
FRONTEND_DIR="$PROJECT_DIR/frontend"
VENV_DIR="$PROJECT_DIR/venv"
DOCS_DIR="$PROJECT_DIR/documents"
UPLOAD_DIR="$PROJECT_DIR/uploads"
SELF_LEARN_DIR="$PROJECT_DIR/self_learn"
VECTOR_DB_DIR="$PROJECT_DIR/vector_db"
EMBEDDINGS_DIR="$PROJECT_DIR/embeddings"

echo "📁 Setting up project directory: $PROJECT_DIR"
if [ "$CUDA_AVAILABLE" = true ]; then
    echo "🚀 GPU acceleration: ENABLED"
else
    echo "💻 GPU acceleration: DISABLED (CPU only)"
fi

mkdir -p "$PROJECT_DIR" "$MODEL_DIR" "$BACKEND_DIR" "$FRONTEND_DIR" "$DOCS_DIR" "$UPLOAD_DIR" "$SELF_LEARN_DIR" "$VECTOR_DB_DIR" "$EMBEDDINGS_DIR"
cd "$PROJECT_DIR"

# Check for Python
PYTHON_BIN=$(which python3)
if [ -z "$PYTHON_BIN" ]; then
    echo "❌ Python3 not found. Please install Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_BIN --version | awk '{print $2}')
echo "✅ Using Python: $PYTHON_BIN (version $PYTHON_VERSION)"

# Create virtual environment
echo "📦 Creating Python virtual environment..."
$PYTHON_BIN -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip and install wheel first
pip install --upgrade pip wheel setuptools

# Fix numpy version conflict first
echo "🔧 Fixing numpy version conflicts..."
pip uninstall -y numpy numba 2>/dev/null || true
pip install "numpy<2.0,>=1.24"

# Install numba with compatible numpy
pip install "numba>=0.59.0"

# Install core dependencies
echo "📦 Installing core Python dependencies..."
pip install fastapi uvicorn requests beautifulsoup4 tqdm pandas
pip install python-multipart aiofiles docx2txt PyPDF2 pdfplumber openpyxl lxml
pip install python-magic

# Install readability-lxml separately
echo "📦 Installing readability-lxml..."
pip install readability-lxml

# Install PyTorch CPU version (we'll install it separately from other packages)
echo "🧠 Installing PyTorch (CPU version)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install ML and Vector Database dependencies (without the PyTorch index URL)
echo "🧠 Installing Machine Learning and Vector Database dependencies..."
pip install sentence-transformers
pip install transformers
pip install chromadb
pip install scikit-learn
pip install nltk
pip install spacy
pip install umap-learn
pip install tiktoken
pip install langchain-text-splitters

# Install langchain-core with force reinstall to avoid conflicts
pip install --force-reinstall --no-deps langchain-core

# Download spacy model for better text processing
echo "🌐 Downloading spaCy language model..."
python -m spacy download en_core_web_sm || echo "⚠️ Could not download spaCy model, will use fallback"

# Download NLTK data
echo "📚 Downloading NLTK data..."
python -c "
import nltk
import ssl
import os

# Handle SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Set NLTK data path
nltk_data_dir = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    print('✅ NLTK data downloaded')
except Exception as e:
    print(f'⚠️ Could not download NLTK data: {e}')
"

# Test critical dependencies
echo "🧪 Testing critical dependencies..."
python3 -c "
import sys
test_failed = False

try:
    from readability import Document
    print('✅ readability-lxml: OK')
except ImportError as e:
    print(f'❌ readability-lxml: MISSING - {e}')
    test_failed = True

try:
    from bs4 import BeautifulSoup
    print('✅ BeautifulSoup: OK')
except ImportError as e:
    print(f'❌ BeautifulSoup: MISSING - {e}')
    test_failed = True

try:
    from sentence_transformers import SentenceTransformer
    print('✅ sentence-transformers: OK')
except ImportError as e:
    print(f'❌ sentence-transformers: MISSING - {e}')
    test_failed = True

try:
    import chromadb
    print('✅ ChromaDB: OK')
except ImportError as e:
    print(f'❌ ChromaDB: MISSING - {e}')
    test_failed = True

try:
    import torch
    print(f'✅ PyTorch: OK (version {torch.__version__})')
except ImportError as e:
    print(f'❌ PyTorch: MISSING - {e}')
    test_failed = True

try:
    import numpy
    print(f'✅ NumPy: OK (version {numpy.__version__})')
except ImportError as e:
    print(f'❌ NumPy: MISSING - {e}')
    test_failed = True

if test_failed:
    print('\\n⚠️ Some dependencies are missing. Attempting to fix...')
    sys.exit(1)
else:
    print('\\n✅ All critical dependencies are installed correctly!')
" || {
    echo "⚠️ Some dependencies missing. Attempting to reinstall..."
    pip install --upgrade --force-reinstall readability-lxml beautifulsoup4 lxml sentence-transformers chromadb
}

# Clone and build llama.cpp
echo "📥 Cloning llama.cpp..."
if [ -d "llama.cpp" ]; then
    echo "📁 llama.cpp directory already exists, pulling latest changes..."
    cd llama.cpp
    git pull
    cd ..
else
    git clone https://github.com/ggerganov/llama.cpp.git
fi

cd llama.cpp

# Clean previous build
echo "🧹 Cleaning previous build..."
rm -rf build

echo "🔨 Building llama.cpp..."
if [ "$CUDA_AVAILABLE" = true ]; then
    echo "🚀 Building with CUDA GPU support..."
else
    echo "💻 Building with CPU-only support..."
fi

mkdir -p build && cd build

# Configure with cmake
if [ "$CUDA_AVAILABLE" = true ]; then
    cmake .. -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DLLAMA_NATIVE=ON -DLLAMA_SERVER=ON $GPU_SUPPORT || {
        echo "⚠️ CMake with CUDA failed. Falling back to CPU-only build..."
        cmake .. -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DLLAMA_NATIVE=ON -DLLAMA_SERVER=ON
    }
else
    cmake .. -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DLLAMA_NATIVE=ON -DLLAMA_SERVER=ON
fi

# Build with make
make -j$(nproc) || { 
    echo "❌ Build failed. Attempting clean rebuild..."
    cd ..
    rm -rf build
    mkdir build && cd build
    cmake .. -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DLLAMA_NATIVE=ON -DLLAMA_SERVER=ON
    make -j$(nproc) || { echo "❌ Build failed completely."; exit 1; }
}

echo "✅ Build completed"

# Find server binary
LLAMA_SERVER_BIN=""
for bin in \
    "$PROJECT_DIR/llama.cpp/build/bin/llama-server" \
    "$PROJECT_DIR/llama.cpp/build/llama-server" \
    "$PROJECT_DIR/llama.cpp/build/bin/server" \
    "$PROJECT_DIR/llama.cpp/build/server" \
    "$PROJECT_DIR/llama.cpp/server"; do
    if [ -x "$bin" ]; then
        LLAMA_SERVER_BIN="$bin"
        echo "✅ Found LLaMA server binary: $LLAMA_SERVER_BIN"
        break
    fi
done

if [ -z "$LLAMA_SERVER_BIN" ]; then
    echo "❌ LLaMA server binary not found."
    echo "Available files in build directory:"
    ls -la "$PROJECT_DIR/llama.cpp/build/"
    ls -la "$PROJECT_DIR/llama.cpp/build/bin/" 2>/dev/null || true
    exit 1
fi

cd "$PROJECT_DIR"

# Download models
echo "⬇️ Downloading AI models..."
LLAMA_MODEL_DIR="$MODEL_DIR/llama3"
FALCON_MODEL_DIR="$MODEL_DIR/Falcon"
mkdir -p "$LLAMA_MODEL_DIR" "$FALCON_MODEL_DIR"

LLAMA_MODEL_FILE="$LLAMA_MODEL_DIR/luna-ai-llama2-uncensored.Q4_K_M.gguf"
FALCON_MODEL_FILE="$FALCON_MODEL_DIR/ehartford-WizardLM-Uncensored-Falcon-7b-Q2_K.gguf"

if [ ! -f "$LLAMA_MODEL_FILE" ]; then
    echo "📥 Downloading LLaMA model (this may take a while)..."
    wget --progress=bar:force -O "$LLAMA_MODEL_FILE" \
        "https://huggingface.co/TheBloke/Luna-AI-Llama2-Uncensored-GGUF/resolve/main/luna-ai-llama2-uncensored.Q4_K_M.gguf" || {
        echo "❌ Failed to download LLaMA model"
        exit 1
    }
else
    echo "✅ LLaMA model already exists"
fi

if [ ! -f "$FALCON_MODEL_FILE" ]; then
    echo "📥 Downloading Falcon model (this may take a while)..."
    wget --progress=bar:force -O "$FALCON_MODEL_FILE" \
        "https://huggingface.co/maddes8cht/ehartford-WizardLM-Uncensored-Falcon-7b-gguf/resolve/main/ehartford-WizardLM-Uncensored-Falcon-7b-Q2_K.gguf" || {
        echo "❌ Failed to download Falcon model"
        exit 1
    }
else
    echo "✅ Falcon model already exists"
fi

# Create ENHANCED Python server with Vector Embeddings and Semantic Search
echo "🔧 Creating ENHANCED backend server with Vector Embeddings..."
cat > "$BACKEND_DIR/server.py" << 'PYEOF'
import os, json, requests, time, uuid, shutil, re, subprocess, asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from urllib.parse import urlparse, urljoin
import mimetypes
import hashlib
import numpy as np
from collections import deque

try:
    import PyPDF2, pdfplumber, docx2txt
    import pandas as pd
    from bs4 import BeautifulSoup
    from readability import Document
    READABILITY_AVAILABLE = True
except ImportError as e:
    print(f"Missing dependency: {e}")
    BeautifulSoup = None
    READABILITY_AVAILABLE = False

try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False

# Enhanced ML imports for vector embeddings and semantic search
try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings
    from sklearn.metrics.pairwise import cosine_similarity
    import tiktoken
    VECTOR_SEARCH_AVAILABLE = True
    print("✅ Vector search capabilities loaded")
except ImportError as e:
    print(f"⚠️ Vector search not available: {e}")
    VECTOR_SEARCH_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None

def detect_gpu():
    gpu_info = {"has_gpu": False, "gpu_name": None, "vram_gb": 0, "recommended_layers": 0}
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines and lines[0]:
                gpu_name, vram_mb = lines[0].split(', ')
                vram_gb = int(vram_mb) // 1024
                gpu_info.update({
                    "has_gpu": True,
                    "gpu_name": gpu_name.strip(),
                    "vram_gb": vram_gb,
                    "recommended_layers": min(40, max(15, vram_gb * 3))
                })
    except:
        pass
    return gpu_info

GPU_INFO = detect_gpu()
PROJECT_DIR = os.environ.get('PROJECT_DIR', os.path.expanduser("~/offline_ai_chat"))
FRONTEND_DIR = f"{PROJECT_DIR}/frontend"
DOCS_DIR = f"{PROJECT_DIR}/documents"
UPLOAD_DIR = f"{PROJECT_DIR}/uploads"
SELF_LEARN_DIR = f"{PROJECT_DIR}/self_learn"
VECTOR_DB_DIR = f"{PROJECT_DIR}/vector_db"
EMBEDDINGS_DIR = f"{PROJECT_DIR}/embeddings"

LLAMA_API = "http://0.0.0.0:8080/completion"
FALCON_API = "http://0.0.0.0:8081/completion"
DOCS_INDEX_FILE = f"{DOCS_DIR}/index.json"
CONVERSATIONS_FILE = f"{SELF_LEARN_DIR}/conversation_memory.json"

app = FastAPI(title="Harpoon AI Enhanced", version="3.0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class DocumentChunker:
    """Enhanced document chunking with semantic awareness"""

    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoding = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Fallback to approximate word count * 1.3
            return int(len(text.split()) * 1.3)

    def semantic_chunk_text(self, text: str, title: str = "") -> List[Dict[str, Any]]:
        """Enhanced chunking with semantic awareness"""
        chunks = []

        # First try sentence-based chunking with spaCy
        if SPACY_AVAILABLE and nlp:
            doc = nlp(text[:1000000])  # Limit text size for spaCy
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        elif NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(text)
            except:
                sentences = self._fallback_sentence_split(text)
        else:
            sentences = self._fallback_sentence_split(text)

        current_chunk = ""
        current_size = 0
        chunk_num = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # If adding this sentence would exceed chunk size
            if current_size + sentence_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    "text": current_chunk.strip(),
                    "chunk_id": f"{title}_{chunk_num}" if title else f"chunk_{chunk_num}",
                    "token_count": int(current_size),
                    "chunk_number": chunk_num,
                    "source_title": title
                })
                chunk_num += 1

                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    # Take last few sentences for overlap
                    overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                    current_chunk = overlap_text + " " + sentence
                    current_size = self.count_tokens(current_chunk)
                else:
                    current_chunk = sentence
                    current_size = sentence_tokens
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_size += sentence_tokens

        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "chunk_id": f"{title}_{chunk_num}" if title else f"chunk_{chunk_num}",
                "token_count": int(current_size),
                "chunk_number": chunk_num,
                "source_title": title
            })

        return chunks

    def _fallback_sentence_split(self, text: str) -> List[str]:
        """Fallback sentence splitting when NLTK/spaCy not available"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Get last N tokens for overlap"""
        words = text.split()
        if len(words) <= overlap_tokens:
            return text
        return " ".join(words[-overlap_tokens:])

class VectorSearchEngine:
    """Enhanced vector search engine with ChromaDB"""

    def __init__(self):
        self.embeddings_model = None
        self.chroma_client = None
        self.collection = None
        self.chunker = DocumentChunker()
        self.initialized = False
        self._initialize()

    def _initialize(self):
        """Initialize the vector search engine"""
        if not VECTOR_SEARCH_AVAILABLE:
            print("⚠️ Vector search not available - missing dependencies")
            return

        try:
            # Initialize sentence transformer
            print("🧠 Loading sentence transformer model...")
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')

            # Initialize ChromaDB
            print("🗄️ Initializing ChromaDB...")
            self.chroma_client = chromadb.PersistentClient(
                path=VECTOR_DB_DIR,
                settings=Settings(anonymized_telemetry=False)
            )

            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection("documents")
                print("✅ Connected to existing ChromaDB collection")
            except:
                self.collection = self.chroma_client.create_collection(
                    name="documents",
                    metadata={"hnsw:space": "cosine"}
                )
                print("✅ Created new ChromaDB collection")

            self.initialized = True
            print("✅ Vector search engine initialized")

        except Exception as e:
            print(f"❌ Failed to initialize vector search: {e}")
            self.initialized = False

    def embed_document(self, doc_info: Dict[str, Any]) -> bool:
        """Embed a document into the vector database"""
        if not self.initialized:
            return False

        try:
            # Chunk the document
            chunks = self.chunker.semantic_chunk_text(
                doc_info["content"],
                doc_info["title"]
            )

            if not chunks:
                return False

            # Generate embeddings for chunks
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embeddings_model.encode(texts).tolist()

            # Prepare metadata
            metadatas = []
            ids = []

            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_info['id']}_chunk_{i}"
                ids.append(chunk_id)

                metadata = {
                    "document_id": doc_info["id"],
                    "document_title": doc_info["title"],
                    "chunk_number": chunk["chunk_number"],
                    "token_count": chunk["token_count"],
                    "upload_date": doc_info["upload_date"],
                    "file_type": doc_info.get("file_type", "unknown"),
                    "source_url": doc_info.get("source_url", "")
                }
                metadatas.append(metadata)

            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )

            print(f"✅ Embedded document '{doc_info['title']}' with {len(chunks)} chunks")
            return True

        except Exception as e:
            print(f"❌ Failed to embed document: {e}")
            return False

    def semantic_search(self, query: str, limit: int = 5, similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Perform semantic search using vector embeddings"""
        if not self.initialized:
            return []

        try:
            # Generate query embedding
            query_embedding = self.embeddings_model.encode([query]).tolist()[0]

            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(limit * 2, 20),  # Get more results to filter
                include=["documents", "metadatas", "distances"]
            )

            # Process results
            search_results = []
            seen_docs = set()

            for i, (doc, metadata, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                # Convert distance to similarity (ChromaDB returns cosine distance)
                similarity = 1 - distance

                if similarity < similarity_threshold:
                    continue

                doc_id = metadata["document_id"]

                # Avoid duplicate documents in top results
                if doc_id in seen_docs and len(search_results) >= limit:
                    continue

                seen_docs.add(doc_id)

                search_results.append({
                    "document_id": doc_id,
                    "title": metadata["document_title"],
                    "content": doc,
                    "similarity_score": similarity,
                    "chunk_number": metadata["chunk_number"],
                    "file_type": metadata.get("file_type", "unknown"),
                    "upload_date": metadata["upload_date"],
                    "source_url": metadata.get("source_url", "")
                })

            # Sort by similarity and limit
            search_results.sort(key=lambda x: x["similarity_score"], reverse=True)
            return search_results[:limit]

        except Exception as e:
            print(f"❌ Semantic search failed: {e}")
            return []

    def remove_document(self, document_id: str) -> bool:
        """Remove a document from the vector database"""
        if not self.initialized:
            return False

        try:
            # Query for chunks belonging to this document
            results = self.collection.get(
                where={"document_id": document_id},
                include=["ids"]
            )

            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                print(f"✅ Removed document {document_id} from vector database")
                return True

        except Exception as e:
            print(f"❌ Failed to remove document from vector database: {e}")

        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get vector database statistics"""
        if not self.initialized:
            return {"initialized": False}

        try:
            count = self.collection.count()
            return {
                "initialized": True,
                "total_chunks": count,
                "model_name": "all-MiniLM-L6-v2",
                "embedding_dimension": 384
            }
        except:
            return {"initialized": False}

class EnhancedConversationMemory:
    """Enhanced conversation memory with better context management"""

    def __init__(self, max_history=100, context_window=6):
        self.conversations = {}
        self.max_history = max_history
        self.context_window = context_window
        self.load_conversations()

    def load_conversations(self):
        if os.path.exists(CONVERSATIONS_FILE):
            try:
                with open(CONVERSATIONS_FILE, 'r') as f:
                    self.conversations = json.load(f)
            except:
                self.conversations = {}

    def save_conversations(self):
        os.makedirs(os.path.dirname(CONVERSATIONS_FILE), exist_ok=True)
        with open(CONVERSATIONS_FILE, 'w') as f:
            json.dump(self.conversations, f, indent=2)

    def get_conversation_context(self, session_id: str, max_exchanges: int = None) -> str:
        """Get conversation context with better formatting"""
        if max_exchanges is None:
            max_exchanges = self.context_window

        history = self.conversations.get(session_id, [])
        if not history:
            return ""

        # Get recent messages
        recent_messages = history[-(max_exchanges * 2):]
        context_parts = []

        for msg in recent_messages:
            role = "Human" if msg["role"] == "user" else "Assistant"
            # Truncate very long messages
            content = msg['content']
            if len(content) > 500:
                content = content[:500] + "..."
            context_parts.append(f"{role}: {content}")

        return "\n".join(context_parts) + "\n" if context_parts else ""

    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self.conversations:
            self.conversations[session_id] = []

        self.conversations[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

        # Trim history if too long
        if len(self.conversations[session_id]) > self.max_history:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history:]

        self.save_conversations()

    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary statistics for a conversation"""
        history = self.conversations.get(session_id, [])
        if not history:
            return {"message_count": 0, "start_time": None, "last_activity": None}

        user_messages = [msg for msg in history if msg["role"] == "user"]
        assistant_messages = [msg for msg in history if msg["role"] == "assistant"]

        return {
            "message_count": len(history),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "start_time": history[0]["timestamp"] if history else None,
            "last_activity": history[-1]["timestamp"] if history else None
        }

# Initialize enhanced components
vector_search = VectorSearchEngine()
conv_memory = EnhancedConversationMemory()

def load_document_index():
    if os.path.exists(DOCS_INDEX_FILE):
        try:
            with open(DOCS_INDEX_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"documents": [], "version": "3.0.1", "vector_enabled": VECTOR_SEARCH_AVAILABLE}

def save_document_index(index):
    os.makedirs(os.path.dirname(DOCS_INDEX_FILE), exist_ok=True)
    index["last_updated"] = datetime.now().isoformat()
    index["vector_enabled"] = VECTOR_SEARCH_AVAILABLE
    with open(DOCS_INDEX_FILE, 'w') as f:
        json.dump(index, f, indent=2)

def hybrid_search_documents(query, limit=5, use_vector=True):
    """Hybrid search combining vector and keyword search"""
    if not query.strip():
        return []

    results = []

    # Vector search (semantic)
    if use_vector and vector_search.initialized:
        vector_results = vector_search.semantic_search(query, limit=limit)
        for result in vector_results:
            result["search_type"] = "semantic"
            result["relevance_score"] = result["similarity_score"] * 100
        results.extend(vector_results)

    # Fallback to keyword search if vector search unavailable or as supplement
    if not vector_search.initialized or len(results) < limit:
        keyword_results = improved_keyword_search(query, limit=limit)
        for result in keyword_results:
            result["search_type"] = "keyword"
            # Don't duplicate documents already found by vector search
            if not any(r.get("document_id") == result.get("id") for r in results):
                results.append(result)

    # Sort by relevance and limit
    results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    return results[:limit]

def improved_keyword_search(query, limit=3):
    """Improved keyword search as fallback"""
    index = load_document_index()
    results = []
    query_lower = query.lower().strip()

    query_words = [word for word in re.findall(r'\b\w+\b', query_lower) if len(word) > 2]

    for doc in index["documents"]:
        content_lower = doc.get("content", "").lower()
        title_lower = doc.get("title", "").lower()

        score = 0

        # Exact phrase matching
        if query_lower in content_lower:
            score += content_lower.count(query_lower) * 15
        if query_lower in title_lower:
            score += title_lower.count(query_lower) * 25

        # Individual word matching
        for word in query_words:
            content_matches = content_lower.count(word)
            title_matches = title_lower.count(word)
            score += content_matches * 3
            score += title_matches * 8

        # Proximity bonus
        if len(query_words) > 1:
            for i in range(len(query_words) - 1):
                word1, word2 = query_words[i], query_words[i + 1]
                pattern = f"{word1}.{{0,100}}{word2}|{word2}.{{0,100}}{word1}"
                if re.search(pattern, content_lower):
                    score += 10

        if score > 0:
            results.append({**doc, "relevance_score": score})

    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results[:limit]

def get_enhanced_document_context(query, use_vector=True):
    """Enhanced document context with hybrid search"""
    relevant_docs = hybrid_search_documents(query, limit=3, use_vector=use_vector)
    if not relevant_docs:
        return ""

    context = "\n=== RELEVANT DOCUMENTS ===\n"
    total_length = 0
    max_context_length = 3000

    for doc in relevant_docs:
        search_type = doc.get("search_type", "unknown")
        score = doc.get("relevance_score", doc.get("similarity_score", 0))

        doc_context = f"\nDocument: {doc.get('title', 'Unknown')}\n"
        doc_context += f"Search Type: {search_type} | Score: {score:.1f}\n"

        # Get content (might be chunk for vector search or full content for keyword)
        content = doc.get("content", "")

        # For keyword search, try to find relevant excerpt
        if search_type == "keyword":
            content = get_relevant_excerpt(content, query, max_length=800)
        else:
            # For vector search, content is already a relevant chunk
            if len(content) > 800:
                content = content[:800] + "..."

        doc_context += content + "\n"

        if total_length + len(doc_context) > max_context_length:
            break

        context += doc_context
        total_length += len(doc_context)

    context += "=== END DOCUMENTS ===\n"
    return context

def get_relevant_excerpt(content, query, max_length=800):
    """Extract relevant excerpt from content"""
    query_lower = query.lower()
    content_lower = content.lower()

    # Find best match position
    best_pos = content_lower.find(query_lower)

    if best_pos == -1:
        # Try individual words
        query_words = re.findall(r'\b\w+\b', query_lower)
        for word in query_words:
            pos = content_lower.find(word)
            if pos != -1:
                best_pos = pos
                break

    if best_pos == -1:
        # Return beginning if no match
        excerpt = content[:max_length]
        if len(content) > max_length:
            excerpt += "..."
        return excerpt

    # Extract around the match
    start = max(0, best_pos - max_length // 3)
    end = min(len(content), best_pos + max_length * 2 // 3)
    excerpt = content[start:end]

    if start > 0:
        excerpt = "..." + excerpt
    if end < len(content):
        excerpt = excerpt + "..."

    return excerpt

def check_model_server(api_url):
    try:
        response = requests.post(api_url, json={"prompt": "test", "n_predict": 1}, timeout=5)
        return response.status_code == 200
    except:
        return False

def extract_text_from_file(file_path: str, filename: str) -> str:
    """Enhanced text extraction with better error handling"""
    file_extension = Path(filename).suffix.lower()
    content = ""

    try:
        if file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

        elif file_extension == '.pdf':
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    pages = []
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            pages.append(text)
                    content = "\n".join(pages)
            except:
                try:
                    with open(file_path, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        pages = []
                        for page in pdf_reader.pages:
                            text = page.extract_text()
                            if text:
                                pages.append(text)
                        content = "\n".join(pages)
                except Exception as e:
                    content = f"PDF processing failed: {str(e)}"

        elif file_extension == '.docx':
            try:
                content = docx2txt.process(file_path)
            except Exception as e:
                content = f"DOCX processing failed: {str(e)}"

        elif file_extension in ['.csv', '.xlsx', '.xls']:
            try:
                if file_extension == '.csv':
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                content = f"Data Summary:\nColumns: {', '.join(df.columns.tolist())}\nRows: {len(df)}\n\nFirst 10 rows:\n{df.head(10).to_string()}"
            except Exception as e:
                content = f"Spreadsheet processing failed: {str(e)}"

        elif file_extension in ['.html', '.htm']:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    html_content = f.read()
                if BeautifulSoup:
                    soup = BeautifulSoup(html_content, 'html.parser')
                    content = soup.get_text(separator='\n', strip=True)
                else:
                    content = html_content
            except Exception as e:
                content = f"HTML processing failed: {str(e)}"

        else:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except:
                content = f"File type {file_extension} not supported"

    except Exception as e:
        content = f"Error reading file: {str(e)}"

    return content.strip()

@app.get("/health")
async def health_check():
    llama_ok = check_model_server(LLAMA_API)
    falcon_ok = check_model_server(FALCON_API)
    doc_index = load_document_index()
    doc_count = len(doc_index["documents"])
    vector_stats = vector_search.get_stats()

    return {
        "status": "ok",
        "models": {"llama": "online" if llama_ok else "offline", "falcon": "online" if falcon_ok else "offline"},
        "documents": {"count": doc_count},
        "vector_search": vector_stats,
        "gpu_info": GPU_INFO,
        "features": {
            "readability": READABILITY_AVAILABLE,
            "magic": MAGIC_AVAILABLE,
            "vector_search": VECTOR_SEARCH_AVAILABLE,
            "nltk": NLTK_AVAILABLE,
            "spacy": SPACY_AVAILABLE
        },
        "version": "3.0.1-FIXED"
    }

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        prompt = data.get("message", "")
        model = data.get("model", "llama")
        response_length = data.get("response_length", "unlimited")
        use_documents = data.get("use_documents", True)
        use_vector_search = data.get("use_vector_search", True)
        session_id = data.get("session_id", "default")

        if not prompt.strip():
            return {"reply": "Please enter a message", "context_used": False}

        api = LLAMA_API if model == "llama" else FALCON_API
        model_name = "LLaMA" if model == "llama" else "Falcon"

        if not check_model_server(api):
            return {"reply": f"❌ {model_name} server not responding", "context_used": False}

        conversation_context = conv_memory.get_conversation_context(session_id, max_exchanges=3)
        document_context = ""
        context_used = False
        search_type = "none"

        if use_documents:
            document_context = get_enhanced_document_context(prompt, use_vector=use_vector_search)
            context_used = bool(document_context.strip())
            if context_used:
                search_type = "hybrid" if vector_search.initialized else "keyword"

        # Enhanced prompt structure
        enhanced_prompt = f"System: You are a helpful AI assistant."
        if document_context:
            enhanced_prompt += f" Use the following documents to help answer the user's question. Pay attention to the search scores and focus on the most relevant information:\n{document_context}"
        enhanced_prompt += "\n\n"

        if conversation_context:
            enhanced_prompt += conversation_context + "\n"
        enhanced_prompt += f"Human: {prompt}\nAssistant:"

        # Set response length
        length_mapping = {"short": 150, "medium": 400, "long": 800, "unlimited": -1}
        n_predict = length_mapping.get(response_length, -1)

        payload = {
            "prompt": enhanced_prompt,
            "n_predict": n_predict,
            "temperature": 0.3,
            "stop": ["Human:", "User:", "System:", "</s>"],
            "stream": False,
            "repeat_penalty": 1.1,
            "top_k": 30,
            "top_p": 0.85
        }

        response = requests.post(api, json=payload, timeout=120)
        if response.status_code != 200:
            return {"reply": f"❌ Model server error: HTTP {response.status_code}", "context_used": context_used}

        response_data = response.json()
        reply = response_data.get("content", "").strip()

        # Clean response
        patterns = [r'^(Human|User|Assistant):\s*', r'\n(Human|User|Assistant):\s*.*$']
        for pattern in patterns:
            reply = re.sub(pattern, '', reply, flags=re.MULTILINE | re.IGNORECASE)
        reply = reply.strip()

        if not reply:
            return {"reply": "❌ Empty response generated", "context_used": context_used}

        conv_memory.add_message(session_id, "user", prompt)
        conv_memory.add_message(session_id, "assistant", reply)

        return {
            "reply": reply,
            "context_used": context_used,
            "search_type": search_type,
            "session_id": session_id
        }

    except requests.exceptions.Timeout:
        return {"reply": "❌ Request timed out", "context_used": False}
    except Exception as e:
        return {"reply": f"❌ Error: {str(e)}", "context_used": False}

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        saved_filename = f"{file_id}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, saved_filename)

        os.makedirs(UPLOAD_DIR, exist_ok=True)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        content = extract_text_from_file(file_path, file.filename)

        if not content.strip():
            os.remove(file_path)
            return {"message": "Failed to extract text from document"}

        content_hash = hashlib.md5(content.encode()).hexdigest()

        doc_info = {
            "id": file_id,
            "title": file.filename,
            "filename": file.filename,
            "saved_filename": saved_filename,
            "file_type": file_extension.lower().lstrip('.'),
            "content": content,
            "content_hash": content_hash,
            "upload_date": datetime.now().isoformat(),
            "size": len(content)
        }

        index = load_document_index()

        # Check for duplicates
        for existing_doc in index["documents"]:
            if existing_doc.get("content_hash") == content_hash:
                os.remove(file_path)
                return {"message": f"Document already exists: {existing_doc['title']}", "duplicate": True}

        index["documents"].append(doc_info)
        save_document_index(index)

        # Add to vector database
        vector_embedded = False
        if vector_search.initialized:
            vector_embedded = vector_search.embed_document(doc_info)

        return {
            "message": "Document uploaded successfully",
            "title": doc_info["title"],
            "vector_embedded": vector_embedded
        }

    except Exception as e:
        return {"message": f"Upload failed: {str(e)}"}

@app.get("/documents")
async def list_documents():
    index = load_document_index()
    documents = []
    for doc in index["documents"]:
        documents.append({
            "id": doc["id"],
            "title": doc["title"],
            "filename": doc["filename"],
            "file_type": doc["file_type"],
            "upload_date": doc["upload_date"],
            "size": doc["size"]
        })
    return {"documents": documents}

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    index = load_document_index()
    doc_to_remove = None
    for i, doc in enumerate(index["documents"]):
        if doc["id"] == document_id:
            doc_to_remove = index["documents"].pop(i)
            break

    if not doc_to_remove:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = os.path.join(UPLOAD_DIR, doc_to_remove["saved_filename"])
    if os.path.exists(file_path):
        os.remove(file_path)

    # Remove from vector database
    if vector_search.initialized:
        vector_search.remove_document(document_id)

    save_document_index(index)
    return {"message": "Document deleted successfully"}

@app.post("/ingest")
async def ingest_url(request: Request):
    """Enhanced URL ingestion with vector embedding"""
    try:
        data = await request.json()
        url = data.get("url", "").strip()

        if not url:
            return {"success": False, "message": "URL is required"}

        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return {"success": False, "message": "Invalid URL format"}

        if not READABILITY_AVAILABLE:
            return {"success": False, "message": "URL ingestion not available - missing readability-lxml dependency"}

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=15, headers=headers, allow_redirects=True)
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    return {"success": False, "message": f"Failed to fetch URL after {max_retries} attempts: {str(e)}"}
                await asyncio.sleep(1)

        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type and 'application/xhtml' not in content_type:
            return {"success": False, "message": f"Unsupported content type: {content_type}"}

        try:
            doc = Document(response.text)
            title = doc.title() or f"Web Content from {parsed.netloc}"

            soup = BeautifulSoup(doc.summary(), "html.parser")

            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.decompose()

            text = soup.get_text(separator='\n', strip=True)

            lines = []
            for line in text.splitlines():
                line = line.strip()
                if line and len(line) > 3:
                    lines.append(line)

            clean_text = '\n'.join(lines)

        except Exception as e:
            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.title.string if soup.title else f"Web Content from {parsed.netloc}"

            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()

            clean_text = soup.get_text(separator='\n', strip=True)

        max_length = 20000
        if len(clean_text) > max_length:
            clean_text = clean_text[:max_length] + "\n... (content truncated)"

        if not clean_text.strip() or len(clean_text) < 100:
            return {"success": False, "message": "No meaningful content found at URL"}

        content_hash = hashlib.md5(clean_text.encode()).hexdigest()

        doc_info = {
            "id": str(uuid.uuid4()),
            "title": title.strip()[:200],
            "filename": f"web_{int(time.time())}.txt",
            "saved_filename": f"web_{int(time.time())}.txt",
            "file_type": "web",
            "content": clean_text,
            "content_hash": content_hash,
            "upload_date": datetime.now().isoformat(),
            "size": len(clean_text),
            "source_url": url
        }

        index = load_document_index()

        for existing_doc in index["documents"]:
            if existing_doc.get("content_hash") == content_hash:
                return {"success": False, "message": f"Content already exists: {existing_doc['title']}", "duplicate": True}

        index["documents"].append(doc_info)
        save_document_index(index)

        # Add to vector database
        vector_embedded = False
        if vector_search.initialized:
            vector_embedded = vector_search.embed_document(doc_info)

        return {
            "success": True,
            "message": f"Successfully ingested: {title}",
            "title": title,
            "vector_embedded": vector_embedded
        }

    except Exception as e:
        return {"success": False, "message": f"Error processing URL: {str(e)}"}

@app.get("/search")
async def search_documents_endpoint(q: str, limit: int = 5, use_vector: bool = True):
    """Search documents endpoint with hybrid search"""
    results = hybrid_search_documents(q, limit, use_vector=use_vector)
    return {"query": q, "results": results, "count": len(results), "vector_used": use_vector and vector_search.initialized}

@app.get("/vector-stats")
async def get_vector_stats():
    """Get vector database statistics"""
    return vector_search.get_stats()

@app.post("/reindex-vectors")
async def reindex_vectors():
    """Reindex all documents in vector database"""
    if not vector_search.initialized:
        return {"success": False, "message": "Vector search not available"}

    try:
        index = load_document_index()
        processed = 0
        errors = 0

        for doc in index["documents"]:
            try:
                if vector_search.embed_document(doc):
                    processed += 1
                else:
                    errors += 1
            except Exception as e:
                print(f"Error reindexing {doc['title']}: {e}")
                errors += 1

        return {
            "success": True,
            "message": f"Reindexing complete: {processed} documents processed, {errors} errors"
        }

    except Exception as e:
        return {"success": False, "message": f"Reindexing failed: {str(e)}"}

@app.get("/")
async def serve_index():
    return FileResponse(f"{FRONTEND_DIR}/index.html")

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

print(f"🚀 HarpoonAI Enhanced Server Starting...")
print(f"🖥️ GPU: {'ENABLED' if GPU_INFO.get('has_gpu') else 'DISABLED'}")
if GPU_INFO.get("has_gpu"):
    print(f"🎮 {GPU_INFO['gpu_name']} ({GPU_INFO['vram_gb']}GB)")
print(f"📚 Readability: {'AVAILABLE' if READABILITY_AVAILABLE else 'MISSING'}")
print(f"🔍 Magic: {'AVAILABLE' if MAGIC_AVAILABLE else 'MISSING'}")
print(f"🧠 Vector Search: {'AVAILABLE' if VECTOR_SEARCH_AVAILABLE else 'MISSING'}")
print(f"📝 NLTK: {'AVAILABLE' if NLTK_AVAILABLE else 'MISSING'}")
print(f"🔤 spaCy: {'AVAILABLE' if SPACY_AVAILABLE else 'MISSING'}")
print(f"✅ Enhanced features: Vector embeddings, semantic search, improved chunking, better context management")
PYEOF

# Copy the HTML file (same as before, no changes needed)
echo "🌐 Creating ENHANCED web interface..."
cat > "$FRONTEND_DIR/index.html" << 'HTMLEOF'
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HarpoonAI Enhanced - Vector Search & Semantic Analysis</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #000; color: #fff; font-family: system-ui, sans-serif; padding: 20px; min-height: 100vh; }
        .container { max-width: 1200px; margin: 0 auto; height: calc(100vh - 40px); display: flex; flex-direction: column; }
        .header { text-align: center; padding: 20px 0; border-bottom: 1px solid #333; margin-bottom: 20px; }
        .header h1 { font-size: 2rem; background: linear-gradient(135deg, #0ea5e9, #22c55e, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .status { font-size: 13px; color: #888; margin-top: 8px; }
        .enhanced-badge { background: linear-gradient(135deg, #22c55e, #0ea5e9); color: #000; padding: 4px 8px; border-radius: 4px; font-size: 11px; margin-top: 4px; display: inline-block; font-weight: bold; }
        .main-content { display: flex; gap: 20px; flex: 1; min-height: 0; }
        .chat-section { flex: 1; display: flex; flex-direction: column; }
        .controls-panel { width: 320px; background: #111; border-radius: 12px; padding: 20px; border: 1px solid #222; height: fit-content; max-height: calc(100vh - 200px); overflow-y: auto; }
        .control-group { margin-bottom: 16px; }
        .control-label { display: block; font-size: 13px; color: #ccc; margin-bottom: 8px; font-weight: 500; }
        .checkbox-wrapper { display: flex; align-items: center; gap: 8px; padding: 12px; background: #1a1a1a; border-radius: 8px; border: 1px solid #333; }
        .checkbox-wrapper.enhanced { border-color: #22c55e; background: #0a1a0a; }
        .file-input, .url-input, .response-select { width: 100%; padding: 10px; background: #1a1a1a; border: 1px solid #333; border-radius: 8px; color: #fff; font-size: 13px; }
        .file-input:focus, .url-input:focus, .response-select:focus { outline: none; border-color: #0ea5e9; }
        .btn { padding: 10px 16px; border: none; border-radius: 8px; font-size: 13px; cursor: pointer; transition: all 0.2s; font-weight: 500; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .btn-primary { background: #0ea5e9; color: #fff; }
        .btn-primary:hover:not(:disabled) { background: #0284c7; }
        .btn-secondary { background: #333; color: #fff; }
        .btn-secondary:hover:not(:disabled) { background: #404040; }
        .btn-vector { background: linear-gradient(135deg, #22c55e, #0ea5e9); color: #000; font-weight: bold; }
        .btn-vector:hover:not(:disabled) { background: linear-gradient(135deg, #16a34a, #0284c7); }
        .btn-group { display: flex; gap: 8px; }
        .progress-bar { width: 100%; height: 4px; background: #333; border-radius: 2px; overflow: hidden; margin-top: 8px; }
        .progress-fill { height: 100%; background: #0ea5e9; width: 0%; transition: width 0.3s; }
        .chat-container { flex: 1; background: #111; border-radius: 12px; border: 1px solid #222; display: flex; flex-direction: column; overflow: hidden; }
        .chat-area { flex: 1; padding: 20px; overflow-y: auto; }
        .message { margin: 16px 0; display: flex; align-items: flex-start; gap: 12px; }
        .message-content { max-width: 75%; padding: 12px 16px; border-radius: 12px; font-size: 14px; line-height: 1.5; }
        .message.user { justify-content: flex-end; }
        .message.user .message-content { background: #22c55e; color: #000; }
        .message.bot .message-content { background: #0ea5e9; color: #000; }
        .message.bot.context-used .message-content::before { content: "📄 "; }
        .message.bot.vector-used .message-content::before { content: "🧠 "; }
        .message.bot.hybrid-used .message-content::before { content: "🔬 "; }
        .message.error .message-content { background: #dc2626; color: #fff; }
        .message.system .message-content { background: #333; color: #ccc; font-style: italic; }
        .search-indicator { font-size: 10px; opacity: 0.7; margin-top: 4px; }
        .vector-indicator { color: #22c55e; }
        .keyword-indicator { color: #f59e0b; }
        .hybrid-indicator { color: #a855f7; }
        .code-block { background: #1a1a1a; border: 1px solid #333; border-radius: 8px; margin: 12px 0; overflow: hidden; }
        .code-header { background: #2a2a2a; padding: 8px 12px; border-bottom: 1px solid #333; display: flex; justify-content: space-between; font-size: 12px; color: #ccc; }
        .copy-btn { background: #333; border: none; color: #ccc; padding: 4px 8px; border-radius: 4px; font-size: 11px; cursor: pointer; }
        .copy-btn:hover { background: #404040; }
        .code-content { padding: 12px; overflow-x: auto; }
        .code-content pre { margin: 0; white-space: pre-wrap; color: #e6e6e6; font-family: monospace; font-size: 13px; line-height: 1.4; word-wrap: break-word; }
        .thinking-indicator { display: none; margin: 16px 0; }
        .thinking-content { background: #333; color: #ccc; padding: 12px 16px; border-radius: 12px; display: flex; align-items: center; gap: 8px; }
        .thinking-dots { display: flex; gap: 4px; }
        .thinking-dot { width: 6px; height: 6px; background: #666; border-radius: 50%; animation: thinking 1.4s infinite; }
        .thinking-dot:nth-child(2) { animation-delay: 0.2s; }
        .thinking-dot:nth-child(3) { animation-delay: 0.4s; }
        @keyframes thinking { 0%, 80%, 100% { opacity: 0.3; } 40% { opacity: 1; } }
        .input-section { padding: 20px; border-top: 1px solid #222; }
        .input-wrapper { display: flex; gap: 12px; align-items: flex-end; }
        .message-input { flex: 1; min-height: 44px; max-height: 120px; padding: 12px 16px; background: #1a1a1a; border: 1px solid #333; border-radius: 12px; color: #fff; font-size: 14px; resize: vertical; font-family: inherit; }
        .message-input:focus { outline: none; border-color: #0ea5e9; }
        .send-btn { padding: 12px 20px; min-width: 80px; }
        .documents-list { max-height: 200px; overflow-y: auto; border: 1px solid #333; border-radius: 8px; background: #1a1a1a; }
        .document-item { display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; border-bottom: 1px solid #333; font-size: 12px; }
        .document-item:last-child { border-bottom: none; }
        .document-info { flex: 1; min-width: 0; }
        .document-title { font-weight: 500; color: #fff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .document-meta { font-size: 10px; color: #888; margin-top: 2px; }
        .delete-btn { background: #dc2626; color: #fff; border: none; padding: 4px 8px; border-radius: 4px; font-size: 10px; cursor: pointer; }
        .delete-btn:hover { background: #b91c1c; }
        .no-documents { padding: 16px; text-align: center; color: #666; font-size: 12px; font-style: italic; }
        .url-section { display: flex; flex-direction: column; gap: 8px; }
        .url-progress { display: none; margin-top: 8px; }
        .status-indicator { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
        .status-online { background: #22c55e; }
        .status-offline { background: #dc2626; }
        .status-warning { background: #f59e0b; }
        .feature-status { display: flex; flex-wrap: wrap; gap: 8px; font-size: 10px; color: #888; margin-top: 4px; }
        .feature-item { display: flex; align-items: center; }
        .vector-stats { background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 12px; margin-top: 8px; font-size: 11px; }
        .vector-stats-title { font-weight: bold; color: #22c55e; margin-bottom: 6px; }
        .vector-stat { display: flex; justify-content: space-between; margin: 2px 0; }
        .enhanced-controls { border: 2px solid #22c55e; border-radius: 8px; padding: 12px; margin-bottom: 16px; background: rgba(34, 197, 94, 0.05); }
        .enhanced-controls h4 { color: #22c55e; margin-bottom: 8px; font-size: 13px; }
        @media (max-width: 768px) {
            .main-content { flex-direction: column; }
            .controls-panel { width: 100%; order: 2; max-height: 400px; }
            .chat-section { order: 1; }
            .message-content { max-width: 85%; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>HarpoonAI Enhanced</h1>
            <div class="enhanced-badge">🧠 Vector Embeddings & Semantic Search</div>
            <div class="status" id="status">Initializing Enhanced System...</div>
            <div class="feature-status" id="featureStatus"></div>
        </header>
        <div class="main-content">
            <aside class="controls-panel">
                <h3>Enhanced Configuration</h3>

                <div class="enhanced-controls">
                    <h4>🧠 AI Search Settings</h4>
                    <div class="checkbox-wrapper enhanced">
                        <input type="checkbox" id="useDocsCheckbox" checked>
                        <label for="useDocsCheckbox">Use document context</label>
                    </div>
                    <div class="checkbox-wrapper enhanced">
                        <input type="checkbox" id="useVectorSearchCheckbox" checked>
                        <label for="useVectorSearchCheckbox">Vector semantic search</label>
                    </div>
                </div>

                <div class="control-group">
                    <label class="control-label">AI Model</label>
                    <select id="modelSelect" class="response-select">
                        <option value="llama" selected>LLaMA</option>
                        <option value="falcon">Falcon</option>
                    </select>
                </div>
                <div class="control-group">
                    <label class="control-label">Response Length</label>
                    <select id="responseLengthSelect" class="response-select">
                        <option value="short">Short</option>
                        <option value="medium">Medium</option>
                        <option value="long">Long</option>
                        <option value="unlimited" selected>Unlimited</option>
                    </select>
                </div>
                <div class="control-group">
                    <label class="control-label">Upload Document</label>
                    <input type="file" id="fileInput" class="file-input" accept=".pdf,.docx,.txt,.csv,.xlsx,.html,.md,.htm">
                    <button class="btn btn-primary" onclick="uploadFile()" style="margin-top: 8px; width: 100%;" id="uploadBtn">Upload & Embed</button>
                    <div class="progress-bar" id="uploadProgress" style="display: none;">
                        <div class="progress-fill" id="uploadProgressFill"></div>
                    </div>
                </div>
                <div class="control-group">
                    <label class="control-label">Ingest URL</label>
                    <div class="url-section">
                        <input type="url" id="urlInput" class="url-input" placeholder="https://example.com/article">
                        <button class="btn btn-primary" onclick="ingestURL()" style="width: 100%;" id="ingestBtn">Ingest & Embed</button>
                        <div class="url-progress" id="urlProgress">
                            <div class="progress-bar">
                                <div class="progress-fill" id="urlProgressFill"></div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="control-group">
                    <label class="control-label">Documents</label>
                    <div class="documents-list" id="documentsList">
                        <div class="no-documents">No documents uploaded</div>
                    </div>
                    <button class="btn btn-secondary" onclick="refreshDocuments()" style="width: 100%; margin-top: 8px;">Refresh Documents</button>
                </div>

                <div class="vector-stats" id="vectorStats" style="display: none;">
                    <div class="vector-stats-title">🧠 Vector Database</div>
                    <div class="vector-stat"><span>Status:</span><span id="vectorStatus">-</span></div>
                    <div class="vector-stat"><span>Chunks:</span><span id="vectorChunks">-</span></div>
                    <div class="vector-stat"><span>Model:</span><span id="vectorModel">-</span></div>
                    <button class="btn btn-vector" onclick="reindexVectors()" style="width: 100%; margin-top: 8px; font-size: 11px;">Reindex All</button>
                </div>

                <div class="control-group">
                    <div class="btn-group">
                        <button class="btn btn-secondary" onclick="clearChat()">Clear Chat</button>
                        <button class="btn btn-secondary" onclick="checkStatus()">Refresh</button>
                    </div>
                    <button class="btn btn-primary" onclick="testConnection()" style="width: 100%; margin-top: 8px;">Test Models</button>
                </div>
            </aside>
            <main class="chat-section">
                <div class="chat-container">
                    <div class="chat-area" id="chatArea">
                        <div class="message bot">
                            <div class="message-content">
                                🧠 Welcome to HarpoonAI Enhanced with Vector Embeddings & Semantic Search!
                                <br><br><strong>🚀 New Features:</strong>
                                <br>✅ Vector embeddings for semantic understanding
                                <br>✅ ChromaDB vector database for intelligent search
                                <br>✅ Improved document chunking with sentence-transformers
                                <br>✅ Hybrid search combining semantic + keyword matching
                                <br>✅ Enhanced conversation context management
                                <br><br>Upload documents or ingest web content, then experience the power of semantic search!
                            </div>
                        </div>
                    </div>
                    <div class="thinking-indicator" id="thinkingIndicator">
                        <div class="thinking-content">
                            <span>HarpoonAI is thinking with enhanced semantic understanding</span>
                            <div class="thinking-dots">
                                <div class="thinking-dot"></div>
                                <div class="thinking-dot"></div>
                                <div class="thinking-dot"></div>
                            </div>
                        </div>
                    </div>
                    <div class="input-section">
                        <div class="input-wrapper">
                            <textarea id="messageInput" class="message-input" placeholder="Ask me anything with enhanced semantic understanding..." rows="1"></textarea>
                            <button id="sendButton" class="btn btn-primary send-btn" onclick="sendMessage()">Send</button>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-core.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/autoloader/prism-autoloader.min.js"></script>
    <script>
        let conversationHistory = [];
        let isLoading = false;

        function showThinking() {
            document.getElementById('thinkingIndicator').style.display = 'flex';
            scrollToBottom();
        }

        function hideThinking() {
            document.getElementById('thinkingIndicator').style.display = 'none';
        }

        function scrollToBottom() {
            const chatArea = document.getElementById('chatArea');
            setTimeout(() => chatArea.scrollTop = chatArea.scrollHeight, 100);
        }

        function showProgress(elementId, show = true) {
            const element = document.getElementById(elementId);
            if (element) {
                element.style.display = show ? 'block' : 'none';
            }
        }

        function updateProgress(elementId, percent) {
            const element = document.getElementById(elementId);
            if (element) {
                element.style.width = percent + '%';
            }
        }

        function formatMessage(content) {
            let formatted = content;

            formatted = formatted.replace(/```(\w+)?\n?([\s\S]*?)```/g, (match, language, code) => {
                const lang = language || 'text';
                const codeId = 'code-' + Math.random().toString(36).substr(2, 9);
                const cleanCode = code.trim();
                return `<div class="code-block">
                    <div class="code-header">
                        <span>${lang}</span>
                        <button class="copy-btn" onclick="copyCode('${codeId}')">Copy</button>
                    </div>
                    <div class="code-content">
                        <pre><code id="${codeId}">${escapeHtml(cleanCode)}</code></pre>
                    </div>
                </div>`;
            });

            formatted = formatted.replace(/`([^`]+)`/g, '<code style="background: rgba(0,0,0,0.3); padding: 2px 4px; border-radius: 3px; white-space: pre-wrap;">$1</code>');
            formatted = formatted.replace(/^### (.*$)/gm, '<h3 style="margin: 12px 0 8px 0; font-size: 15px;">$1</h3>');
            formatted = formatted.replace(/^## (.*$)/gm, '<h2 style="margin: 12px 0 8px 0; font-size: 16px;">$1</h2>');
            formatted = formatted.replace(/^# (.*$)/gm, '<h1 style="margin: 12px 0 8px 0; font-size: 18px;">$1</h1>');
            formatted = formatted.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
            formatted = formatted.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
            formatted = formatted.replace(/^[*-] (.+)$/gm, '<li style="margin: 4px 0;">$1</li>');
            formatted = formatted.replace(/(<li.*<\/li>)/gs, '<ul style="margin: 8px 0; padding-left: 20px;">$1</ul>');
            formatted = formatted.replace(/\n(?![^<]*<\/code>)/g, '<br>');

            return formatted;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function copyCode(codeId) {
            const codeElement = document.getElementById(codeId);
            navigator.clipboard.writeText(codeElement.textContent).then(() => {
                const copyBtn = codeElement.closest('.code-block').querySelector('.copy-btn');
                const originalText = copyBtn.textContent;
                copyBtn.textContent = 'Copied!';
                setTimeout(() => copyBtn.textContent = originalText, 2000);
            });
        }

        function addMessage(content, isUser = false, isError = false, searchInfo = null, isSystem = false) {
            const chatArea = document.getElementById('chatArea');
            const messageDiv = document.createElement('div');

            let className = `message ${isUser ? 'user' : 'bot'}${isError ? ' error' : ''}${isSystem ? ' system' : ''}`;

            if (searchInfo) {
                if (searchInfo.search_type === 'semantic') {
                    className += ' vector-used';
                } else if (searchInfo.search_type === 'hybrid') {
                    className += ' hybrid-used';
                } else if (searchInfo.context_used) {
                    className += ' context-used';
                }
            }

            messageDiv.className = className;

            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';

            if (!isUser && !isError) {
                contentDiv.innerHTML = formatMessage(content);

                // Add search indicator
                if (searchInfo && (searchInfo.context_used || searchInfo.search_type)) {
                    const indicator = document.createElement('div');
                    indicator.className = 'search-indicator';

                    if (searchInfo.search_type === 'semantic') {
                        indicator.innerHTML = '<span class="vector-indicator">🧠 Vector semantic search used</span>';
                    } else if (searchInfo.search_type === 'hybrid') {
                        indicator.innerHTML = '<span class="hybrid-indicator">🔬 Hybrid search (semantic + keyword)</span>';
                    } else if (searchInfo.search_type === 'keyword') {
                        indicator.innerHTML = '<span class="keyword-indicator">🔍 Keyword search used</span>';
                    }

                    if (indicator.innerHTML) {
                        contentDiv.appendChild(indicator);
                    }
                }

                setTimeout(() => {
                    const codeBlocks = contentDiv.querySelectorAll('pre code');
                    codeBlocks.forEach(block => {
                        if (window.Prism) Prism.highlightElement(block);
                    });
                }, 10);
            } else {
                contentDiv.textContent = content;
            }

            messageDiv.appendChild(contentDiv);
            chatArea.appendChild(messageDiv);
            scrollToBottom();
        }

        async function sendMessage() {
            if (isLoading) return;

            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            if (!message) return;

            const sendButton = document.getElementById('sendButton');
            const useDocuments = document.getElementById('useDocsCheckbox').checked;
            const useVectorSearch = document.getElementById('useVectorSearchCheckbox').checked;
            const selectedModel = document.getElementById('modelSelect').value;
            const responseLength = document.getElementById('responseLengthSelect').value;

            isLoading = true;
            sendButton.disabled = true;
            sendButton.textContent = 'Sending...';

            addMessage(message, true);
            input.value = '';
            showThinking();

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: message,
                        model: selectedModel,
                        response_length: responseLength,
                        use_documents: useDocuments,
                        use_vector_search: useVectorSearch
                    })
                });

                const data = await response.json();
                hideThinking();

                const searchInfo = {
                    context_used: data.context_used,
                    search_type: data.search_type
                };

                addMessage(data.reply || 'No response received', false, false, searchInfo);

            } catch (error) {
                console.error('Error:', error);
                hideThinking();
                addMessage(`Error: ${error.message}`, false, true);
            } finally {
                isLoading = false;
                sendButton.disabled = false;
                sendButton.textContent = 'Send';
                input.focus();
            }
        }

        async function uploadFile() {
            const fileInput = document.getElementById('fileInput');
            const uploadBtn = document.getElementById('uploadBtn');

            if (!fileInput.files[0]) {
                alert('Please select a file');
                return;
            }

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            uploadBtn.disabled = true;
            uploadBtn.textContent = 'Processing...';
            showProgress('uploadProgress');

            let progress = 0;
            const progressInterval = setInterval(() => {
                progress += Math.random() * 15;
                if (progress > 90) progress = 90;
                updateProgress('uploadProgressFill', progress);
            }, 200);

            try {
                const response = await fetch('/upload-document', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();

                clearInterval(progressInterval);
                updateProgress('uploadProgressFill', 100);

                let message = `📄 ${result.message}: ${result.title || 'Unknown'}`;
                if (result.vector_embedded) {
                    message += ' (🧠 Vector embedded)';
                }

                if (result.duplicate) {
                    addMessage(`📄 ${result.message}`, false, false, null, true);
                } else {
                    addMessage(message, false, false, null, true);
                }

                refreshDocuments();
                updateVectorStats();
                checkStatus();
            } catch (error) {
                clearInterval(progressInterval);
                addMessage(`❌ Upload error: ${error.message}`, false, true);
            } finally {
                uploadBtn.disabled = false;
                uploadBtn.textContent = 'Upload & Embed';
                showProgress('uploadProgress', false);
                fileInput.value = '';
            }
        }

        async function ingestURL() {
            const urlInput = document.getElementById('urlInput');
            const ingestBtn = document.getElementById('ingestBtn');
            const url = urlInput.value.trim();

            if (!url) {
                alert('Please enter a URL');
                return;
            }

            ingestBtn.disabled = true;
            ingestBtn.textContent = 'Processing...';
            showProgress('urlProgress');

            let progress = 0;
            const progressInterval = setInterval(() => {
                progress += Math.random() * 10;
                if (progress > 85) progress = 85;
                updateProgress('urlProgressFill', progress);
            }, 300);

            try {
                const response = await fetch('/ingest', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url: url })
                });
                const result = await response.json();

                clearInterval(progressInterval);
                updateProgress('urlProgressFill', 100);

                if (result.success) {
                    let message = `🌐 ${result.message}`;
                    if (result.vector_embedded) {
                        message += ' (🧠 Vector embedded)';
                    }
                    addMessage(message, false, false, null, true);
                    urlInput.value = '';
                    refreshDocuments();
                    updateVectorStats();
                    checkStatus();
                } else {
                    addMessage(`❌ URL ingestion failed: ${result.message}`, false, true);
                }
            } catch (error) {
                clearInterval(progressInterval);
                addMessage(`❌ URL ingestion error: ${error.message}`, false, true);
            } finally {
                ingestBtn.disabled = false;
                ingestBtn.textContent = 'Ingest & Embed';
                showProgress('urlProgress', false);
            }
        }

        async function refreshDocuments() {
            try {
                const response = await fetch('/documents');
                const data = await response.json();
                const documentsList = document.getElementById('documentsList');

                if (data.documents && data.documents.length > 0) {
                    documentsList.innerHTML = data.documents.map(doc => `
                        <div class="document-item">
                            <div class="document-info">
                                <div class="document-title">${doc.title}</div>
                                <div class="document-meta">${doc.file_type} • ${formatFileSize(doc.size)} • ${formatDate(doc.upload_date)}</div>
                            </div>
                            <button class="delete-btn" onclick="deleteDocument('${doc.id}', '${doc.title.replace(/'/g, "\\'")}')">×</button>
                        </div>
                    `).join('');
                } else {
                    documentsList.innerHTML = '<div class="no-documents">No documents uploaded</div>';
                }
            } catch (error) {
                console.error('Failed to load documents:', error);
            }
        }

        async function deleteDocument(documentId, documentTitle) {
            if (!confirm(`Delete "${documentTitle}"?`)) return;

            try {
                const response = await fetch(`/documents/${documentId}`, { method: 'DELETE' });
                if (response.ok) {
                    addMessage(`🗑️ Deleted: ${documentTitle} (including vector embeddings)`, false, false, null, true);
                    refreshDocuments();
                    updateVectorStats();
                    checkStatus();
                }
            } catch (error) {
                addMessage(`❌ Delete error: ${error.message}`, false, true);
            }
        }

        async function updateVectorStats() {
            try {
                const response = await fetch('/vector-stats');
                const stats = await response.json();
                const vectorStatsDiv = document.getElementById('vectorStats');

                if (stats.initialized) {
                    vectorStatsDiv.style.display = 'block';
                    document.getElementById('vectorStatus').textContent = 'Online';
                    document.getElementById('vectorChunks').textContent = stats.total_chunks || 0;
                    document.getElementById('vectorModel').textContent = stats.model_name || 'Unknown';
                } else {
                    vectorStatsDiv.style.display = 'none';
                }
            } catch (error) {
                document.getElementById('vectorStats').style.display = 'none';
            }
        }

        async function reindexVectors() {
            if (!confirm('Reindex all documents in vector database? This may take a while.')) return;

            try {
                addMessage('🧠 Starting vector reindexing...', false, false, null, true);
                const response = await fetch('/reindex-vectors', { method: 'POST' });
                const result = await response.json();

                if (result.success) {
                    addMessage(`✅ ${result.message}`, false, false, null, true);
                } else {
                    addMessage(`❌ Reindexing failed: ${result.message}`, false, true);
                }
                updateVectorStats();
            } catch (error) {
                addMessage(`❌ Reindexing error: ${error.message}`, false, true);
            }
        }

        function formatFileSize(bytes) {
            if (bytes === 0) return '0 B';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
        }

        function formatDate(dateString) {
            try {
                const date = new Date(dateString);
                return date.toLocaleDateString();
            } catch {
                return 'Unknown';
            }
        }

        async function testConnection() {
            addMessage('🔧 Testing enhanced system connections...', false, false, null, true);

            try {
                const models = ['llama', 'falcon'];
                for (const model of models) {
                    try {
                        const response = await fetch('/chat', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                message: 'Say exactly: SUCCESS TEST',
                                model: model,
                                response_length: 'short',
                                use_documents: false,
                                use_vector_search: false,
                                session_id: 'test-session'
                            })
                        });

                        const data = await response.json();
                        const result = data.reply || 'No response';
                        const status = result.includes('SUCCESS') ? '✅' : '⚠️';
                        addMessage(`${status} ${model.toUpperCase()}: ${result}`, false, false, null, true);

                    } catch (error) {
                        addMessage(`❌ ${model.toUpperCase()}: ${error.message}`, false, true);
                    }
                }
            } catch (error) {
                addMessage(`❌ Test failed: ${error.message}`, false, true);
            }
        }

        async function checkStatus() {
            try {
                const response = await fetch('/health');
                const data = await response.json();

                let statusText = '';
                const llamaStatus = data.models.llama === 'online' ? '✅' : '❌';
                const falconStatus = data.models.falcon === 'online' ? '✅' : '❌';

                statusText = `${llamaStatus} LLaMA | ${falconStatus} Falcon | 📄 ${data.documents.count} docs`;

                if (data.vector_search && data.vector_search.initialized) {
                    statusText += ` | 🧠 ${data.vector_search.total_chunks || 0} chunks`;
                }

                if (data.gpu_info && data.gpu_info.has_gpu) {
                    statusText += ` | 🎮 ${data.gpu_info.gpu_name}`;
                }

                if (data.version) {
                    statusText += ` | v${data.version}`;
                }

                document.getElementById('status').textContent = statusText;

                // Update feature status
                const features = [];
                if (data.features) {
                    if (data.features.readability) {
                        features.push('<div class="feature-item"><span class="status-indicator status-online"></span>URL Ingestion</div>');
                    } else {
                        features.push('<div class="feature-item"><span class="status-indicator status-offline"></span>URL Ingestion</div>');
                    }

                    if (data.features.vector_search) {
                        features.push('<div class="feature-item"><span class="status-indicator status-online"></span>Vector Search</div>');
                    } else {
                        features.push('<div class="feature-item"><span class="status-indicator status-offline"></span>Vector Search</div>');
                    }

                    if (data.features.nltk) {
                        features.push('<div class="feature-item"><span class="status-indicator status-online"></span>NLTK</div>');
                    }

                    if (data.features.spacy) {
                        features.push('<div class="feature-item"><span class="status-indicator status-online"></span>spaCy</div>');
                    }
                }

                document.getElementById('featureStatus').innerHTML = features.join('');

                // Update vector stats
                updateVectorStats();

            } catch (error) {
                document.getElementById('status').textContent = 'Status check failed';
                console.error('Status check error:', error);
            }
        }

        function clearChat() {
            if (confirm('Clear chat history?')) {
                document.getElementById('chatArea').innerHTML = `
                    <div class="message bot">
                        <div class="message-content">🧠 Chat cleared! Ready to help with enhanced semantic understanding.</div>
                    </div>
                `;
            }
        }

        // Auto-resize textarea
        const messageInput = document.getElementById('messageInput');
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });

        // Enter key support
        messageInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // Initialize
        window.addEventListener('load', function() {
            checkStatus();
            refreshDocuments();
            updateVectorStats();
            messageInput.focus();
            console.log('✅ HarpoonAI Enhanced Frontend loaded with Vector Search');
        });

        // Auto-refresh status every 30 seconds
        setInterval(checkStatus, 30000);
    </script>
</body>
</html>
HTMLEOF

# Create enhanced startup script
echo "🚀 Creating enhanced startup script..."
cat > "$PROJECT_DIR/start_harpoonai.sh" << 'STARTEOF'
#!/bin/bash

PROJECT_DIR="$HOME/offline_ai_chat"
cd "$PROJECT_DIR"

echo "🚀 Starting HarpoonAI Enhanced System with Vector Embeddings..."

# Check if ports are in use
check_port() {
    netstat -tuln | grep ":$1 " > /dev/null 2>&1
    return $?
}

# Start service in background
start_service() {
    local name="$1"
    local command="$2"
    local port="$3"
    local log_file="$4"

    if check_port "$port"; then
        echo "⚠️ Port $port already in use. $name may be running."
        return 1
    fi

    echo "🚀 Starting $name on port $port..."
    nohup $command > "$log_file" 2>&1 &
    local pid=$!
    echo "$pid" > "$PROJECT_DIR/${name,,}_pid.txt"

    sleep 3
    if kill -0 "$pid" 2>/dev/null; then
        echo "✅ $name started (PID: $pid)"
        return 0
    else
        echo "❌ Failed to start $name"
        return 1
    fi
}

# Wait for service
wait_for_service() {
    local name="$1"
    local url="$2"
    local max_attempts=30
    local attempt=1

    echo "⏳ Waiting for $name..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo "✅ $name ready!"
            return 0
        fi
        sleep 2
        attempt=$((attempt + 1))
    done
    echo "❌ $name failed to start"
    return 1
}

# Check GPU and display enhanced info
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ ! -z "$GPU_INFO" ]; then
        echo "🎮 GPU: $GPU_INFO"
        echo "🧠 Vector embeddings will use GPU acceleration when available"
    fi
else
    echo "💻 CPU-only mode - Vector embeddings will use CPU"
fi

# Find server binary
LLAMA_SERVER_BIN=""
for bin in "$PROJECT_DIR/llama.cpp/build/bin/llama-server" "$PROJECT_DIR/llama.cpp/build/llama-server" "$PROJECT_DIR/llama.cpp/build/bin/server" "$PROJECT_DIR/llama.cpp/build/server"; do
  if [ -x "$bin" ]; then
    LLAMA_SERVER_BIN="$bin"
    break
  fi
done

if [ -z "$LLAMA_SERVER_BIN" ]; then
    echo "❌ LLaMA server binary not found"
    exit 1
fi

# Model paths
LLAMA_MODEL="$PROJECT_DIR/models/llama3/luna-ai-llama2-uncensored.Q4_K_M.gguf"
FALCON_MODEL="$PROJECT_DIR/models/Falcon/ehartford-WizardLM-Uncensored-Falcon-7b-Q2_K.gguf"

# Check models
if [ ! -f "$LLAMA_MODEL" ]; then
    echo "❌ LLaMA model not found: $LLAMA_MODEL"
    exit 1
fi

if [ ! -f "$FALCON_MODEL" ]; then
    echo "❌ Falcon model not found: $FALCON_MODEL"
    exit 1
fi

# GPU parameters with enhanced settings for vector workloads
GPU_PARAMS_LLAMA=""
GPU_PARAMS_FALCON=""
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "🚀 GPU detected, enabling acceleration for LLM and vector operations..."
    GPU_PARAMS_LLAMA="-ngl 33"
    GPU_PARAMS_FALCON="-ngl 28"
fi

# Start LLaMA server with enhanced parameters
LLAMA_CMD="$LLAMA_SERVER_BIN -m $LLAMA_MODEL -c 4096 -b 1024 --host 0.0.0.0 --port 8080 -t $(nproc) $GPU_PARAMS_LLAMA --mlock"
start_service "LLaMA" "$LLAMA_CMD" "8080" "$PROJECT_DIR/llama_server.log"

# Start Falcon server
FALCON_CMD="$LLAMA_SERVER_BIN -m $FALCON_MODEL -c 2048 -b 512 --host 0.0.0.0 --port 8081 -t $(nproc) $GPU_PARAMS_FALCON --mlock"
start_service "Falcon" "$FALCON_CMD" "8081" "$PROJECT_DIR/falcon_server.log"

# Wait for models
wait_for_service "LLaMA" "http://localhost:8080/health"
wait_for_service "Falcon" "http://localhost:8081/health"

# Activate Python environment
if [ -f "$PROJECT_DIR/venv/bin/activate" ]; then
    source "$PROJECT_DIR/venv/bin/activate"
    echo "✅ Python environment activated"
fi

# Set environment variables for enhanced features
export PROJECT_DIR="$PROJECT_DIR"
export TOKENIZERS_PARALLELISM=false  # Avoid warnings with sentence-transformers
export CUDA_VISIBLE_DEVICES=0  # Use first GPU if available

# Start enhanced backend with vector capabilities
echo "🧠 Starting Enhanced Backend with Vector Embeddings..."
cd "$PROJECT_DIR/backend"
start_service "Backend" "uvicorn server:app --host 0.0.0.0 --port 8000" "8000" "$PROJECT_DIR/backend_server.log"

wait_for_service "Backend" "http://localhost:8000/health"

echo ""
echo "🎉 HarpoonAI Enhanced System Started Successfully!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 Web Interface: http://localhost:8000"
echo "🤖 LLaMA Server: http://localhost:8080"
echo "🦅 Falcon Server: http://localhost:8081"
echo ""
echo "🧠 ENHANCED FEATURES:"
echo "   ✅ Vector embeddings with sentence-transformers (all-MiniLM-L6-v2)"
echo "   ✅ ChromaDB vector database for semantic search"
echo "   ✅ Intelligent document chunking with semantic awareness"
echo "   ✅ Hybrid search (semantic + keyword matching)"
echo "   ✅ Enhanced conversation context management"
echo "   ✅ Improved document processing with spaCy/NLTK"
echo "   ✅ GPU acceleration for both LLM and ML workloads"
echo ""
echo "📁 Log Files:"
echo "   LLaMA:   $PROJECT_DIR/llama_server.log"
echo "   Falcon:  $PROJECT_DIR/falcon_server.log"
echo "   Backend: $PROJECT_DIR/backend_server.log"
echo ""
echo "📊 Data Directories:"
echo "   Documents: $PROJECT_DIR/documents"
echo "   Vector DB: $PROJECT_DIR/vector_db"
echo "   Embeddings: $PROJECT_DIR/embeddings"
echo ""
echo "🛑 To stop: $PROJECT_DIR/stop_harpoonai.sh"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STARTEOF

# Create enhanced stop script
echo "🛑 Creating enhanced stop script..."
cat > "$PROJECT_DIR/stop_harpoonai.sh" << 'STOPEOF'
#!/bin/bash

PROJECT_DIR="$HOME/offline_ai_chat"
cd "$PROJECT_DIR"

echo "🛑 Stopping HarpoonAI Enhanced System..."

stop_service() {
    local name="$1"
    local pid_file="$PROJECT_DIR/${name,,}_pid.txt"

    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "🛑 Stopping $name (PID: $pid)..."
            kill "$pid"
            sleep 2
            if kill -0 "$pid" 2>/dev/null; then
                echo "   Force killing $name..."
                kill -9 "$pid"
            fi
            echo "✅ $name stopped"
        fi
        rm -f "$pid_file"
    fi
}

stop_service "Backend"
stop_service "Falcon"
stop_service "LLaMA"

# Kill any remaining processes on ports
for port in 8000 8080 8081; do
    fuser -k ${port}/tcp 2>/dev/null || true
done

echo "✅ HarpoonAI Enhanced System stopped"
echo "📊 Vector database and documents preserved in $PROJECT_DIR"
STOPEOF

chmod +x "$PROJECT_DIR/start_harpoonai.sh"
chmod +x "$PROJECT_DIR/stop_harpoonai.sh"

# Create a system info script for troubleshooting
echo "📊 Creating system info script..."
cat > "$PROJECT_DIR/system_info.sh" << 'INFOEOF'
#!/bin/bash

PROJECT_DIR="$HOME/offline_ai_chat"

echo "🔍 HarpoonAI Enhanced System Information"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "📁 Project Directory: $PROJECT_DIR"
echo "🐍 Python Version: $(python3 --version 2>/dev/null || echo 'Not found')"

# Check GPU
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "🎮 GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits | head -1
    echo "🔧 CUDA Version: $(nvcc --version 2>/dev/null | grep release | sed 's/.*release \([0-9.]*\).*/\1/' || echo 'Not found')"
else
    echo "💻 GPU: Not available (CPU only)"
fi

# Check Python packages
echo ""
echo "📦 Python Packages:"
source "$PROJECT_DIR/venv/bin/activate" 2>/dev/null || echo "⚠️ Virtual environment not found"
pip list | grep -E "(sentence-transformers|chromadb|torch|transformers|nltk|spacy)" || echo "⚠️ Some packages missing"

# Check models
echo ""
echo "🤖 Models:"
[ -f "$PROJECT_DIR/models/llama3/luna-ai-llama2-uncensored.Q4_K_M.gguf" ] && echo "✅ LLaMA model" || echo "❌ LLaMA model missing"
[ -f "$PROJECT_DIR/models/Falcon/ehartford-WizardLM-Uncensored-Falcon-7b-Q2_K.gguf" ] && echo "✅ Falcon model" || echo "❌ Falcon model missing"

# Check vector database
echo ""
echo "🧠 Vector Database:"
if [ -d "$PROJECT_DIR/vector_db" ]; then
    echo "✅ ChromaDB directory exists"
    echo "📊 Size: $(du -sh $PROJECT_DIR/vector_db 2>/dev/null | cut -f1 || echo 'Unknown')"
else
    echo "❌ ChromaDB directory not found"
fi

# Check documents
echo ""
echo "📄 Documents:"
if [ -f "$PROJECT_DIR/documents/index.json" ]; then
    DOC_COUNT=$(python3 -c "import json; print(len(json.load(open('$PROJECT_DIR/documents/index.json'))['documents']))" 2>/dev/null || echo "Unknown")
    echo "✅ Document index exists ($DOC_COUNT documents)"
else
    echo "❌ Document index not found"
fi

# Check services
echo ""
echo "🔌 Service Status:"
curl -s http://localhost:8000/health > /dev/null 2>&1 && echo "✅ Backend API" || echo "❌ Backend API"
curl -s http://localhost:8080/health > /dev/null 2>&1 && echo "✅ LLaMA Server" || echo "❌ LLaMA Server"
curl -s http://localhost:8081/health > /dev/null 2>&1 && echo "✅ Falcon Server" || echo "❌ Falcon Server"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
INFOEOF

chmod +x "$PROJECT_DIR/system_info.sh"

echo ""
echo "✅ ENHANCED HarpoonAI Installation completed!"
echo ""
echo "🧠 ENHANCED FEATURES INSTALLED:"
echo "   ✅ Vector Embeddings: sentence-transformers (all-MiniLM-L6-v2)"
echo "   ✅ Vector Database: ChromaDB with persistent storage"
echo "   ✅ Semantic Search: Hybrid semantic + keyword matching"
echo "   ✅ Smart Chunking: Context-aware document segmentation"
echo "   ✅ Enhanced Context: Better conversation memory management"
echo "   ✅ ML Libraries: PyTorch, transformers, scikit-learn, NLTK, spaCy"
echo "   ✅ GPU Support: Automatic detection and acceleration"
echo ""
echo "🚀 GPU Status: $([ "$CUDA_AVAILABLE" = true ] && echo "ENABLED" || echo "DISABLED")"
echo "📁 Project directory: $PROJECT_DIR"
echo ""
echo "📊 DIRECTORY STRUCTURE:"
echo "   📄 Documents: $PROJECT_DIR/documents"
echo "   🧠 Vector DB: $PROJECT_DIR/vector_db"
echo "   🔧 Backend: $PROJECT_DIR/backend"
echo "   🌐 Frontend: $PROJECT_DIR/frontend"
echo "   📚 Models: $PROJECT_DIR/models"
echo "   🗂️ Uploads: $PROJECT_DIR/uploads"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 To start the ENHANCED system:"
echo "   $PROJECT_DIR/start_harpoonai.sh"
echo ""
echo "🛑 To stop all services:"
echo "   $PROJECT_DIR/stop_harpoonai.sh"
echo ""
echo "📊 To check system status:"
echo "   $PROJECT_DIR/system_info.sh"
echo ""
echo "🌐 Once started, access: http://localhost:8000"
echo ""
echo "🧠 WHAT'S NEW:"
echo "   • Upload documents and they're automatically embedded as vectors"
echo "   • Ask questions and get semantic search results, not just keyword matching"
echo "   • Improved document chunking for better context understanding"
echo "   • Hybrid search combines the best of semantic and keyword approaches"
echo "   • Enhanced conversation memory with better context management"
echo "   • All search results show relevance scores and search type used"
echo ""
echo "🚀 The system will automatically download sentence-transformer models on first use."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
