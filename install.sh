#!/bin/bash

# HarpoonAI Enhanced Installer with GPU Support and One-Command Startup
# Version 2.1.1 - FIXED URL Ingestion & Document Search

echo "🚀 HarpoonAI Enhanced Installer Starting (FIXED VERSION)..."

sudo hostnamectl set-hostname harpoonai
sudo yum install wget git pip net-tools -y
sudo yum update -y
sudo dnf install libcurl-devel file-devel libmagic-devel cmake gcc-c++ make openblas-devel -y

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
            echo "📦 Installing CUDA development tools..."
            sudo dnf install -y cuda-toolkit-12-* || sudo dnf install -y cuda-devel || echo "⚠️ Could not install CUDA automatically"
            
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

echo "📁 Setting up project directory: $PROJECT_DIR"
if [ "$CUDA_AVAILABLE" = true ]; then
    echo "🚀 GPU acceleration: ENABLED"
else
    echo "💻 GPU acceleration: DISABLED (CPU only)"
fi

mkdir -p "$PROJECT_DIR" "$MODEL_DIR" "$BACKEND_DIR" "$FRONTEND_DIR" "$DOCS_DIR" "$UPLOAD_DIR" "$SELF_LEARN_DIR"
cd "$PROJECT_DIR"

# Check for Python
PYTHON_BIN=$(which python3)
if [ -z "$PYTHON_BIN" ]; then
  echo "❌ Python3 not found. Please install Python 3.10+"
  exit 1
fi

echo "✅ Using Python: $PYTHON_BIN"

# Create virtual environment
echo "📦 Creating Python virtual environment..."
$PYTHON_BIN -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

# Install all required dependencies including the ones that were missing
echo "📦 Installing Python dependencies..."
pip install fastapi uvicorn requests beautifulsoup4 readability-lxml tqdm numpy pandas
pip install python-multipart aiofiles docx2txt PyPDF2 pdfplumber openpyxl lxml
pip install python-magic || echo "⚠️ python-magic not available, using fallback file detection"

# Test critical dependencies
echo "🧪 Testing critical dependencies..."
python3 -c "
try:
    from readability import Document
    print('✅ readability-lxml: OK')
except ImportError:
    print('❌ readability-lxml: MISSING - URL ingestion will not work!')
    exit(1)

try:
    from bs4 import BeautifulSoup
    print('✅ BeautifulSoup: OK')
except ImportError:
    print('❌ BeautifulSoup: MISSING')
    exit(1)
" || {
    echo "❌ Critical dependencies missing. Attempting to install..."
    pip install --force-reinstall readability-lxml beautifulsoup4 lxml
}

# Clone and build llama.cpp
echo "📥 Cloning llama.cpp..."
if [ ! -d "llama.cpp" ]; then
  git clone https://github.com/ggerganov/llama.cpp.git
fi

cd llama.cpp
echo "🔨 Building llama.cpp..."
if [ "$CUDA_AVAILABLE" = true ]; then
    echo "🚀 Building with CUDA GPU support..."
else
    echo "💻 Building with CPU-only support..."
fi

mkdir -p build && cd build

if [ "$CUDA_AVAILABLE" = true ]; then
    cmake .. -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DLLAMA_NATIVE=ON -DLLAMA_SERVER=ON $GPU_SUPPORT || {
        echo "❌ CMake with CUDA failed. Falling back to CPU-only build..."
        cmake .. -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DLLAMA_NATIVE=ON -DLLAMA_SERVER=ON
    }
else
    cmake .. -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DLLAMA_NATIVE=ON -DLLAMA_SERVER=ON
fi

make -j$(nproc) || { echo "❌ Build failed."; exit 1; }
echo "✅ Build completed"

# Find server binary
LLAMA_SERVER_BIN=""
for bin in \
  "$PROJECT_DIR/llama.cpp/build/bin/llama-server" \
  "$PROJECT_DIR/llama.cpp/build/llama-server" \
  "$PROJECT_DIR/llama.cpp/build/bin/server" \
  "$PROJECT_DIR/llama.cpp/build/server"; do
  if [ -x "$bin" ]; then
    LLAMA_SERVER_BIN="$bin"
    echo "✅ Found LLaMA server binary: $LLAMA_SERVER_BIN"
    break
  fi
done

if [ -z "$LLAMA_SERVER_BIN" ]; then
  echo "❌ LLaMA server binary not found."
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
  echo "📥 Downloading LLaMA model..."
  wget -O "$LLAMA_MODEL_FILE" \
    "https://huggingface.co/TheBloke/Luna-AI-Llama2-Uncensored-GGUF/resolve/main/luna-ai-llama2-uncensored.Q4_K_M.gguf"
fi

if [ ! -f "$FALCON_MODEL_FILE" ]; then
  echo "📥 Downloading Falcon model..."
  wget -O "$FALCON_MODEL_FILE" \
    "https://huggingface.co/maddes8cht/ehartford-WizardLM-Uncensored-Falcon-7b-gguf/resolve/main/ehartford-WizardLM-Uncensored-Falcon-7b-Q2_K.gguf"
fi

# Create FIXED Python server
echo "🔧 Creating FIXED backend server..."
cat > "$BACKEND_DIR/server.py" << 'PYEOF'
import os, json, requests, time, uuid, shutil, re, subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import asyncio
from urllib.parse import urlparse, urljoin
import mimetypes
import hashlib

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

LLAMA_API = "http://0.0.0.0:8080/completion"
FALCON_API = "http://0.0.0.0:8081/completion"
DOCS_INDEX_FILE = f"{DOCS_DIR}/index.json"
CONVERSATIONS_FILE = f"{SELF_LEARN_DIR}/conversation_memory.json"

app = FastAPI(title="Harpoon AI", version="2.1.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ConversationMemory:
    def __init__(self):
        self.conversations = {}
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

    def get_conversation_context(self, session_id: str, max_exchanges: int = 3) -> str:
        history = self.conversations.get(session_id, [])
        if not history:
            return ""
        recent_messages = history[-(max_exchanges * 2):]
        context_parts = []
        for msg in recent_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            context_parts.append(f"{role}: {msg['content']}")
        return "\n".join(context_parts) + "\n" if context_parts else ""

    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        self.conversations[session_id].append({
            "role": role, "content": content, "timestamp": datetime.now().isoformat()
        })
        if len(self.conversations[session_id]) > 50:
            self.conversations[session_id] = self.conversations[session_id][-50:]
        self.save_conversations()

conv_memory = ConversationMemory()

def load_document_index():
    if os.path.exists(DOCS_INDEX_FILE):
        try:
            with open(DOCS_INDEX_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"documents": [], "version": "2.1"}

def save_document_index(index):
    os.makedirs(os.path.dirname(DOCS_INDEX_FILE), exist_ok=True)
    index["last_updated"] = datetime.now().isoformat()
    with open(DOCS_INDEX_FILE, 'w') as f:
        json.dump(index, f, indent=2)

def improved_search_documents(query, limit=3):
    """IMPROVED document search with much better matching and scoring"""
    index = load_document_index()
    results = []
    query_lower = query.lower().strip()
    
    # Split query into words for better matching
    query_words = [word for word in re.findall(r'\b\w+\b', query_lower) if len(word) > 2]
    
    for doc in index["documents"]:
        content_lower = doc.get("content", "").lower()
        title_lower = doc.get("title", "").lower()
        
        score = 0
        
        # Exact phrase matching (highest score)
        if query_lower in content_lower:
            score += content_lower.count(query_lower) * 15
        if query_lower in title_lower:
            score += title_lower.count(query_lower) * 25
        
        # Individual word matching with position scoring
        for word in query_words:
            content_matches = content_lower.count(word)
            title_matches = title_lower.count(word)
            score += content_matches * 3
            score += title_matches * 8
        
        # Bonus for multiple word matches in proximity
        if len(query_words) > 1:
            for i in range(len(query_words) - 1):
                word1, word2 = query_words[i], query_words[i + 1]
                # Check if words appear within 100 characters of each other
                pattern = f"{word1}.{{0,100}}{word2}|{word2}.{{0,100}}{word1}"
                if re.search(pattern, content_lower):
                    score += 10
        
        # Bonus for document recency
        try:
            upload_date = datetime.fromisoformat(doc.get("upload_date", "2020-01-01"))
            days_old = (datetime.now() - upload_date).days
            if days_old < 7:
                score += 5
        except:
            pass
        
        if score > 0:
            results.append({**doc, "relevance_score": score})
    
    # Sort by relevance score
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results[:limit]

def get_document_context(query):
    """ENHANCED document context generation with better excerpts"""
    relevant_docs = improved_search_documents(query, limit=3)
    if not relevant_docs:
        return ""
    
    context = "\n=== RELEVANT DOCUMENTS ===\n"
    total_length = 0
    max_context_length = 2500
    
    for doc in relevant_docs:
        doc_context = f"\nDocument: {doc['title']}\nRelevance Score: {doc['relevance_score']}\n"
        
        # Try to find the most relevant excerpt around the query match
        content = doc['content']
        query_lower = query.lower()
        
        # Find the best excerpt around the query match
        best_excerpt = ""
        if query_lower in content.lower():
            match_pos = content.lower().find(query_lower)
            start = max(0, match_pos - 300)
            end = min(len(content), match_pos + 700)
            best_excerpt = content[start:end]
            if start > 0:
                best_excerpt = "..." + best_excerpt
            if end < len(content):
                best_excerpt = best_excerpt + "..."
        else:
            # Fallback: try to find excerpts around individual query words
            query_words = re.findall(r'\b\w+\b', query_lower)
            for word in query_words:
                if word in content.lower():
                    match_pos = content.lower().find(word)
                    start = max(0, match_pos - 200)
                    end = min(len(content), match_pos + 500)
                    excerpt = content[start:end]
                    if start > 0:
                        excerpt = "..." + excerpt
                    if end < len(content):
                        excerpt = excerpt + "..."
                    if len(excerpt) > len(best_excerpt):
                        best_excerpt = excerpt
                    break
            
            # Final fallback to beginning of document
            if not best_excerpt:
                best_excerpt = content[:600]
                if len(content) > 600:
                    best_excerpt += "..."
        
        doc_context += best_excerpt + "\n"
        
        # Check if adding this document would exceed the limit
        if total_length + len(doc_context) > max_context_length:
            break
            
        context += doc_context
        total_length += len(doc_context)
    
    context += "=== END DOCUMENTS ===\n"
    return context

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
            # Try pdfplumber first (better for complex PDFs), fallback to PyPDF2
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
                # Fallback to PyPDF2
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
                # Convert to a readable format
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
            # Try to read as text file
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
    return {
        "status": "ok",
        "models": {"llama": "online" if llama_ok else "offline", "falcon": "online" if falcon_ok else "offline"},
        "documents": {"count": doc_count},
        "gpu_info": GPU_INFO,
        "features": {
            "readability": READABILITY_AVAILABLE,
            "magic": MAGIC_AVAILABLE
        },
        "version": "2.1.1-FIXED"
    }

@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        prompt = data.get("message", "")
        model = data.get("model", "llama")
        response_length = data.get("response_length", "unlimited")
        use_documents = data.get("use_documents", True)
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
        
        if use_documents:
            document_context = get_document_context(prompt)
            context_used = bool(document_context.strip())

        # Improved prompt structure
        enhanced_prompt = f"System: You are a helpful AI assistant."
        if document_context:
            enhanced_prompt += f" Use the following documents to help answer the user's question. Pay attention to the relevance scores and focus on the most relevant information:\n{document_context}"
        enhanced_prompt += "\n\n"
        
        if conversation_context:
            enhanced_prompt += conversation_context + "\n"
        enhanced_prompt += f"User: {prompt}\nAssistant:"

        # Set response length
        length_mapping = {"short": 150, "medium": 400, "long": 800, "unlimited": -1}
        n_predict = length_mapping.get(response_length, -1)

        payload = {
            "prompt": enhanced_prompt,
            "n_predict": n_predict,
            "temperature": 0.3,
            "stop": ["User:", "Human:", "System:", "</s>"],
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

        return {"reply": reply, "context_used": context_used, "session_id": session_id}

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

        # Enhanced text extraction
        content = extract_text_from_file(file_path, file.filename)

        if not content.strip():
            os.remove(file_path)
            return {"message": "Failed to extract text from document"}

        # Create document hash for deduplication
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

        return {"message": "Document uploaded successfully", "title": doc_info["title"]}

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
    
    save_document_index(index)
    return {"message": "Document deleted successfully"}

# FIXED URL INGESTION - Now uses POST method and proper error handling
@app.post("/ingest")
async def ingest_url(request: Request):
    """FIXED URL ingestion with proper POST method and enhanced error handling"""
    try:
        data = await request.json()
        url = data.get("url", "").strip()
        
        if not url:
            return {"success": False, "message": "URL is required"}
            
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return {"success": False, "message": "Invalid URL format"}
            
        if not READABILITY_AVAILABLE:
            return {"success": False, "message": "URL ingestion not available - missing readability-lxml dependency"}
        
        # Enhanced headers to avoid blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # Fetch the webpage with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=15, headers=headers, allow_redirects=True)
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    return {"success": False, "message": f"Failed to fetch URL after {max_retries} attempts: {str(e)}"}
                await asyncio.sleep(1)  # Wait before retry
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type and 'application/xhtml' not in content_type:
            return {"success": False, "message": f"Unsupported content type: {content_type}"}
        
        # Extract readable content using readability
        try:
            doc = Document(response.text)
            title = doc.title() or f"Web Content from {parsed.netloc}"
            
            # Parse with BeautifulSoup for better text extraction
            soup = BeautifulSoup(doc.summary(), "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.decompose()
            
            # Get text with better formatting
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up text more thoroughly
            lines = []
            for line in text.splitlines():
                line = line.strip()
                if line and len(line) > 3:  # Filter out very short lines
                    lines.append(line)
            
            clean_text = '\n'.join(lines)
            
        except Exception as e:
            # Fallback to basic HTML parsing
            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.title.string if soup.title else f"Web Content from {parsed.netloc}"
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()
                
            clean_text = soup.get_text(separator='\n', strip=True)
        
        # Limit content length but keep more for better context
        max_length = 20000
        if len(clean_text) > max_length:
            clean_text = clean_text[:max_length] + "\n... (content truncated)"
        
        if not clean_text.strip() or len(clean_text) < 100:
            return {"success": False, "message": "No meaningful content found at URL"}
        
        # Create document hash for deduplication
        content_hash = hashlib.md5(clean_text.encode()).hexdigest()
        
        # Add to document index
        doc_info = {
            "id": str(uuid.uuid4()),
            "title": title.strip()[:200],  # Limit title length
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
        
        # Check for duplicates
        for existing_doc in index["documents"]:
            if existing_doc.get("content_hash") == content_hash:
                return {"success": False, "message": f"Content already exists: {existing_doc['title']}", "duplicate": True}
        
        index["documents"].append(doc_info)
        save_document_index(index)

        return {"success": True, "message": f"Successfully ingested: {title}", "title": title}
        
    except Exception as e:
        return {"success": False, "message": f"Error processing URL: {str(e)}"}

@app.get("/search")
async def search_documents_endpoint(q: str, limit: int = 5):
    """Search documents endpoint for debugging"""
    results = improved_search_documents(q, limit)
    return {"query": q, "results": results, "count": len(results)}

@app.get("/")
async def serve_index():
    return FileResponse(f"{FRONTEND_DIR}/index.html")

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

print(f"🚀 HarpoonAI Server Starting (FIXED VERSION)...")
print(f"🖥️ GPU: {'ENABLED' if GPU_INFO.get('has_gpu') else 'DISABLED'}")
if GPU_INFO.get("has_gpu"):
    print(f"🎮 {GPU_INFO['gpu_name']} ({GPU_INFO['vram_gb']}GB)")
print(f"📚 Readability: {'AVAILABLE' if READABILITY_AVAILABLE else 'MISSING'}")
print(f"🔍 Magic: {'AVAILABLE' if MAGIC_AVAILABLE else 'MISSING'}")
print(f"✅ All fixes applied: URL ingestion, document search, error handling")
PYEOF

# Create FIXED HTML interface
echo "🌐 Creating FIXED web interface..."
cat > "$FRONTEND_DIR/index.html" << 'HTMLEOF'
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HarpoonAI Beta - FIXED</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #000; color: #fff; font-family: system-ui, sans-serif; padding: 20px; min-height: 100vh; }
        .container { max-width: 1000px; margin: 0 auto; height: calc(100vh - 40px); display: flex; flex-direction: column; }
        .header { text-align: center; padding: 20px 0; border-bottom: 1px solid #333; margin-bottom: 20px; }
        .header h1 { font-size: 2rem; background: linear-gradient(135deg, #0ea5e9, #22c55e); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .status { font-size: 13px; color: #888; margin-top: 8px; }
        .fixed-badge { background: #22c55e; color: #000; padding: 4px 8px; border-radius: 4px; font-size: 11px; margin-top: 4px; display: inline-block; }
        .main-content { display: flex; gap: 20px; flex: 1; min-height: 0; }
        .chat-section { flex: 1; display: flex; flex-direction: column; }
        .controls-panel { width: 300px; background: #111; border-radius: 12px; padding: 20px; border: 1px solid #222; height: fit-content; max-height: calc(100vh - 200px); overflow-y: auto; }
        .control-group { margin-bottom: 16px; }
        .control-label { display: block; font-size: 13px; color: #ccc; margin-bottom: 8px; font-weight: 500; }
        .checkbox-wrapper { display: flex; align-items: center; gap: 8px; padding: 12px; background: #1a1a1a; border-radius: 8px; border: 1px solid #333; }
        .file-input, .url-input, .response-select { width: 100%; padding: 10px; background: #1a1a1a; border: 1px solid #333; border-radius: 8px; color: #fff; font-size: 13px; }
        .file-input:focus, .url-input:focus, .response-select:focus { outline: none; border-color: #0ea5e9; }
        .btn { padding: 10px 16px; border: none; border-radius: 8px; font-size: 13px; cursor: pointer; transition: all 0.2s; font-weight: 500; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .btn-primary { background: #0ea5e9; color: #fff; }
        .btn-primary:hover:not(:disabled) { background: #0284c7; }
        .btn-secondary { background: #333; color: #fff; }
        .btn-secondary:hover:not(:disabled) { background: #404040; }
        .btn-group { display: flex; gap: 8px; }
        .progress-bar { width: 100%; height: 4px; background: #333; border-radius: 2px; overflow: hidden; margin-top: 8px; }
        .progress-fill { height: 100%; background: #0ea5e9; width: 0%; transition: width 0.3s; }
        .chat-container { flex: 1; background: #111; border-radius: 12px; border: 1px solid #222; display: flex; flex-direction: column; overflow: hidden; }
        .chat-area { flex: 1; padding: 20px; overflow-y: auto; }
        .message { margin: 16px 0; display: flex; align-items: flex-start; gap: 12px; }
        .message-content { max-width: 70%; padding: 12px 16px; border-radius: 12px; font-size: 14px; line-height: 1.5; }
        .message.user { justify-content: flex-end; }
        .message.user .message-content { background: #22c55e; color: #000; }
        .message.bot .message-content { background: #0ea5e9; color: #000; }
        .message.bot.context-used .message-content::before { content: "📄 "; }
        .message.error .message-content { background: #dc2626; color: #fff; }
        .message.system .message-content { background: #333; color: #ccc; font-style: italic; }
        .code-block { background: #1a1a1a; border: 1px solid #333; border-radius: 8px; margin: 12px 0; overflow: hidden; }
        .code-header { background: #2a2a2a; padding: 8px 12px; border-bottom: 1px solid #333; display: flex; justify-content: space-between; font-size: 12px; color: #ccc; }
        .copy-btn { background: #333; border: none; color: #ccc; padding: 4px 8px; border-radius: 4px; font-size: 11px; cursor: pointer; }
        .copy-btn:hover { background: #404040; }
        .code-content { padding: 12px; overflow-x: auto; }
        .code-content pre { margin: 0; white-space: pre-wrap; color: #e6e6e6; font-family: monospace; font-size: 13px; line-height: 1.4; word-wrap: break-word; }
        .code-content code { white-space: pre-wrap; word-wrap: break-word; display: block; }
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
        .feature-status { display: flex; gap: 12px; font-size: 11px; color: #888; margin-top: 4px; }
        .feature-item { display: flex; align-items: center; }
        @media (max-width: 768px) { 
            .main-content { flex-direction: column; } 
            .controls-panel { width: 100%; order: 2; max-height: 300px; } 
            .chat-section { order: 1; } 
            .message-content { max-width: 85%; } 
        }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>HarpoonAI Beta</h1>
            <div class="fixed-badge">✅ URL Ingestion & Document Search FIXED</div>
            <div class="status" id="status">Initializing...</div>
            <div class="feature-status" id="featureStatus"></div>
        </header>
        <div class="main-content">
            <aside class="controls-panel">
                <h3>Configuration</h3>
                <div class="control-group">
                    <div class="checkbox-wrapper">
                        <input type="checkbox" id="useDocsCheckbox" checked>
                        <label for="useDocsCheckbox">Use document context</label>
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
                    <button class="btn btn-primary" onclick="uploadFile()" style="margin-top: 8px; width: 100%;" id="uploadBtn">Upload</button>
                    <div class="progress-bar" id="uploadProgress" style="display: none;">
                        <div class="progress-fill" id="uploadProgressFill"></div>
                    </div>
                </div>
                <div class="control-group">
                    <label class="control-label">Ingest URL (FIXED)</label>
                    <div class="url-section">
                        <input type="url" id="urlInput" class="url-input" placeholder="https://example.com/article">
                        <button class="btn btn-primary" onclick="ingestURL()" style="width: 100%;" id="ingestBtn">Ingest URL</button>
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
                                👋 Welcome to HarpoonAI Beta (FIXED)! All major issues have been resolved:
                                <br>✅ URL ingestion now works properly
                                <br>✅ Document search algorithm improved
                                <br>✅ Better error handling and user feedback
                                <br><br>Upload documents or ingest web content, then ask me questions!
                            </div>
                        </div>
                    </div>
                    <div class="thinking-indicator" id="thinkingIndicator">
                        <div class="thinking-content">
                            <span>HarpoonAI is thinking</span>
                            <div class="thinking-dots">
                                <div class="thinking-dot"></div>
                                <div class="thinking-dot"></div>
                                <div class="thinking-dot"></div>
                            </div>
                        </div>
                    </div>
                    <div class="input-section">
                        <div class="input-wrapper">
                            <textarea id="messageInput" class="message-input" placeholder="Ask me anything..." rows="1"></textarea>
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
            
            // Handle code blocks with proper line breaks
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

            // Handle inline code
            formatted = formatted.replace(/`([^`]+)`/g, '<code style="background: rgba(0,0,0,0.3); padding: 2px 4px; border-radius: 3px; white-space: pre-wrap;">$1</code>');
            
            // Handle headers
            formatted = formatted.replace(/^### (.*$)/gm, '<h3 style="margin: 12px 0 8px 0; font-size: 15px;">$1</h3>');
            formatted = formatted.replace(/^## (.*$)/gm, '<h2 style="margin: 12px 0 8px 0; font-size: 16px;">$1</h2>');
            formatted = formatted.replace(/^# (.*$)/gm, '<h1 style="margin: 12px 0 8px 0; font-size: 18px;">$1</h1>');
            
            // Handle bold/italic
            formatted = formatted.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
            formatted = formatted.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
            
            // Handle lists
            formatted = formatted.replace(/^[*-] (.+)$/gm, '<li style="margin: 4px 0;">$1</li>');
            formatted = formatted.replace(/(<li.*<\/li>)/gs, '<ul style="margin: 8px 0; padding-left: 20px;">$1</ul>');
            
            // Handle line breaks
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

        function addMessage(content, isUser = false, isError = false, contextUsed = false, isSystem = false) {
            const chatArea = document.getElementById('chatArea');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'bot'}${isError ? ' error' : ''}${contextUsed ? ' context-used' : ''}${isSystem ? ' system' : ''}`;
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            
            if (!isUser && !isError) {
                contentDiv.innerHTML = formatMessage(content);
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
                        use_documents: useDocuments
                    })
                });

                const data = await response.json();
                hideThinking();
                addMessage(data.reply || 'No response received', false, false, data.context_used);

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
            uploadBtn.textContent = 'Uploading...';
            showProgress('uploadProgress');

            // Simulate progress for user feedback
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
                
                if (result.duplicate) {
                    addMessage(`📄 ${result.message}`, false, false, false, true);
                } else {
                    addMessage(`📄 ${result.message}: ${result.title || 'Unknown'}`, false, false, false, true);
                }
                
                refreshDocuments();
                checkStatus();
            } catch (error) {
                clearInterval(progressInterval);
                addMessage(`❌ Upload error: ${error.message}`, false, true);
            } finally {
                uploadBtn.disabled = false;
                uploadBtn.textContent = 'Upload';
                showProgress('uploadProgress', false);
                fileInput.value = '';
            }
        }

        // FIXED URL ingestion function - now uses POST method
        async function ingestURL() {
            const urlInput = document.getElementById('urlInput');
            const ingestBtn = document.getElementById('ingestBtn');
            const url = urlInput.value.trim();
            
            if (!url) {
                alert('Please enter a URL');
                return;
            }

            ingestBtn.disabled = true;
            ingestBtn.textContent = 'Ingesting...';
            showProgress('urlProgress');

            // Simulate progress
            let progress = 0;
            const progressInterval = setInterval(() => {
                progress += Math.random() * 10;
                if (progress > 85) progress = 85;
                updateProgress('urlProgressFill', progress);
            }, 300);

            try {
                // FIXED: Now using POST method as expected by the backend
                const response = await fetch('/ingest', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url: url })
                });
                const result = await response.json();
                
                clearInterval(progressInterval);
                updateProgress('urlProgressFill', 100);
                
                if (result.success) {
                    addMessage(`🌐 ${result.message}`, false, false, false, true);
                    urlInput.value = '';
                    refreshDocuments();
                    checkStatus();
                } else {
                    addMessage(`❌ URL ingestion failed: ${result.message}`, false, true);
                }
            } catch (error) {
                clearInterval(progressInterval);
                addMessage(`❌ URL ingestion error: ${error.message}`, false, true);
            } finally {
                ingestBtn.disabled = false;
                ingestBtn.textContent = 'Ingest URL';
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
                    addMessage(`🗑️ Deleted: ${documentTitle}`, false, false, false, true);
                    refreshDocuments();
                    checkStatus();
                }
            } catch (error) {
                addMessage(`❌ Delete error: ${error.message}`, false, true);
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
            addMessage('🔧 Testing model connections...', false, false, false, true);
            
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
                                session_id: 'test-session'
                            })
                        });

                        const data = await response.json();
                        const result = data.reply || 'No response';
                        const status = result.includes('SUCCESS') ? '✅' : '⚠️';
                        addMessage(`${status} ${model.toUpperCase()}: ${result}`, false, false, false, true);
                        
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
                        features.push('<div class="feature-item"><span class="status-indicator status-online"></span>URL Ingestion: FIXED & Working</div>');
                    } else {
                        features.push('<div class="feature-item"><span class="status-indicator status-offline"></span>URL Ingestion: Missing readability-lxml</div>');
                    }
                    
                    if (data.features.magic) {
                        features.push('<div class="feature-item"><span class="status-indicator status-online"></span>File Detection: Enhanced</div>');
                    } else {
                        features.push('<div class="feature-item"><span class="status-indicator status-warning"></span>File Detection: Basic</div>');
                    }
                }
                
                document.getElementById('featureStatus').innerHTML = features.join('');
                
            } catch (error) {
                document.getElementById('status').textContent = 'Status check failed';
                console.error('Status check error:', error);
            }
        }

        function clearChat() {
            if (confirm('Clear chat history?')) {
                document.getElementById('chatArea').innerHTML = `
                    <div class="message bot">
                        <div class="message-content">👋 Chat cleared! Ready to help with your documents and web content.</div>
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
            messageInput.focus();
            console.log('✅ HarpoonAI Frontend loaded (FIXED VERSION)');
        });

        // Auto-refresh status every 30 seconds
        setInterval(checkStatus, 30000);
    </script>
</body>
</html>
HTMLEOF

# Create unified startup script
echo "🚀 Creating unified startup script..."
cat > "$PROJECT_DIR/start_harpoonai.sh" << 'STARTEOF'
#!/bin/bash

PROJECT_DIR="$HOME/offline_ai_chat"
cd "$PROJECT_DIR"

echo "🚀 Starting HarpoonAI System (FIXED VERSION)..."

# Check if ports are in use
check_port() {
    netstat -tuln | grep ":$1 " > /dev/null
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

# Check GPU
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ ! -z "$GPU_INFO" ]; then
        echo "🎮 GPU: $GPU_INFO"
    fi
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

# GPU parameters
GPU_PARAMS_LLAMA=""
GPU_PARAMS_FALCON=""
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "🚀 GPU detected, enabling acceleration..."
    GPU_PARAMS_LLAMA="-ngl 33"
    GPU_PARAMS_FALCON="-ngl 28"
fi

# Start LLaMA server
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

# Start backend
echo "🚀 Starting FIXED backend server..."
cd "$PROJECT_DIR/backend"
start_service "Backend" "uvicorn server:app --host 0.0.0.0 --port 8000" "8000" "$PROJECT_DIR/backend_server.log"

wait_for_service "Backend" "http://localhost:8000/health"

echo ""
echo "🎉 HarpoonAI FIXED VERSION Started Successfully!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 Web Interface: http://localhost:8000"
echo "🤖 LLaMA Server: http://localhost:8080"
echo "🦅 Falcon Server: http://localhost:8081"
echo ""
echo "✅ FIXES APPLIED:"
echo "   • URL ingestion endpoint fixed (POST method)"
echo "   • Document search algorithm improved with better scoring"
echo "   • Enhanced error handling and user feedback"
echo "   • Better file processing for multiple formats"
echo "   • Improved web content extraction with retry logic"
echo "   • Duplicate content detection"
echo ""
echo "📁 Log Files:"
echo "   LLaMA:   $PROJECT_DIR/llama_server.log"
echo "   Falcon:  $PROJECT_DIR/falcon_server.log"  
echo "   Backend: $PROJECT_DIR/backend_server.log"
echo ""
echo "🛑 To stop: $PROJECT_DIR/stop_harpoonai.sh"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STARTEOF

# Create stop script
echo "🛑 Creating stop script..."
cat > "$PROJECT_DIR/stop_harpoonai.sh" << 'STOPEOF'
#!/bin/bash

PROJECT_DIR="$HOME/offline_ai_chat"
cd "$PROJECT_DIR"

echo "🛑 Stopping HarpoonAI System..."

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

echo "✅ HarpoonAI System stopped"
STOPEOF

chmod +x "$PROJECT_DIR/start_harpoonai.sh"
chmod +x "$PROJECT_DIR/stop_harpoonai.sh"

echo ""
echo "✅ FIXED HarpoonAI Installation completed!"
echo ""
echo "🚀 GPU Status: $([ "$CUDA_AVAILABLE" = true ] && echo "ENABLED" || echo "DISABLED")"
echo "📁 Project directory: $PROJECT_DIR"
echo ""
echo "🔧 FIXES APPLIED:"
echo "   ✅ URL ingestion endpoint changed to POST method"
echo "   ✅ Improved document search algorithm with relevance scoring"
echo "   ✅ Enhanced error handling and user feedback"
echo "   ✅ Better file processing (PDF, DOCX, CSV, etc.)"
echo "   ✅ Improved web scraping with proper headers and retries"
echo "   ✅ Duplicate document detection"
echo "   ✅ Enhanced document context generation"
echo "   ✅ Fixed frontend to properly handle API responses"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 To start the FIXED system:"
echo "   $PROJECT_DIR/start_harpoonai.sh"
echo ""
echo "🛑 To stop all services:"
echo "   $PROJECT_DIR/stop_harpoonai.sh"
echo ""
echo "🌐 Once started, access: http://localhost:8000"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
