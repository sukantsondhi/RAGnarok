# ⚡ RAGnarok

A **Retrieval-Augmented Generation (RAG)** application built with Streamlit that enables users to ask questions about their documents and receive AI-generated answers based on the document content.

## ✨ Features

- **Dual Document Loading**: Upload via UI or load from disk
- **Local Models**: Uses locally stored AI models (no internet required for inference)
- **Configurable Models**: Change embedding model and LLM via `.env` file
- **Vector Search**: FAISS-powered semantic similarity search
- **Persistent Storage**: Save and load indexes using pickle
- **Batch Processing**: Efficient handling of large document sets
- **Interactive UI**: User-friendly Streamlit interface

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Architecture](#architecture)
7. [Troubleshooting](#troubleshooting)

---

## 🔧 Prerequisites

### System Requirements

| Requirement | Minimum             | Recommended |
| ----------- | ------------------- | ----------- |
| Python      | 3.8                 | 3.10+       |
| RAM         | 8GB                 | 16GB        |
| Storage     | 2GB                 | 5GB         |
| OS          | Windows/macOS/Linux | Any         |

### Required Python Packages

```
streamlit>=1.28.0
sentence-transformers>=2.2.0
transformers>=4.30.0
faiss-cpu>=1.7.4
torch>=2.0.0
python-dotenv>=1.0.0
numpy
```

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/sukantsondhi/Rag-Application.git
cd RAGnarok
```

### Step 2: Create Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit sentence-transformers transformers faiss-cpu torch python-dotenv numpy
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Or create it manually with the following content:

```env
# Embedding Model Configuration
# Options: sentence-transformers/all-MiniLM-L6-v2, sentence-transformers/all-mpnet-base-v2, etc.
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# LLM Configuration
# Options: google/flan-t5-small, google/flan-t5-base, google/flan-t5-large, etc.
LLM_MODEL_NAME=google/flan-t5-small

# Local model directory (models will be downloaded here)
MODELS_DIR=.models
```

### Step 5: Download Models

Models are downloaded automatically on first run, or you can pre-download them:

```bash
python -c "
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

os.makedirs('.models', exist_ok=True)

# Download embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.save('.models/all-MiniLM-L6-v2')

# Download LLM
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
llm = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')
tokenizer.save_pretrained('.models/flan-t5-small')
llm.save_pretrained('.models/flan-t5-small')
"
```

### Step 6: Prepare Documents

Place your `.txt` files in:

- `to_upload_from_UI/` - for UI upload testing
- `to_upload_from_disk/` - for bulk loading from disk

### Step 7: Run the Application

```bash
streamlit run RAGnarok.py
```

The app will open in your default browser at `http://localhost:8501`

---

## ⚙️ Configuration

### Environment Variables

Configure the application by editing the `.env` file:

| Variable               | Description                             | Default                                  |
| ---------------------- | --------------------------------------- | ---------------------------------------- |
| `EMBEDDING_MODEL_NAME` | HuggingFace embedding model name        | `sentence-transformers/all-MiniLM-L6-v2` |
| `LLM_MODEL_NAME`       | HuggingFace LLM model name              | `google/flan-t5-small`                   |
| `MODELS_DIR`           | Directory for storing downloaded models | `.models`                                |

### Supported Models

#### Embedding Models

| Model                                             | Size  | Dimensions | Best For              |
| ------------------------------------------------- | ----- | ---------- | --------------------- |
| `sentence-transformers/all-MiniLM-L6-v2`          | 80MB  | 384        | Fast, general purpose |
| `sentence-transformers/all-mpnet-base-v2`         | 420MB | 768        | Higher accuracy       |
| `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` | 80MB  | 384        | Q&A tasks             |

#### LLM Models

| Model                  | Size  | Parameters | Best For                    |
| ---------------------- | ----- | ---------- | --------------------------- |
| `google/flan-t5-small` | 300MB | 80M        | Fast inference, basic tasks |
| `google/flan-t5-base`  | 990MB | 250M       | Balanced performance        |
| `google/flan-t5-large` | 3GB   | 780M       | Higher quality answers      |

### Sidebar Settings

| Setting     | Range    | Default | Description                                 |
| ----------- | -------- | ------- | ------------------------------------------- |
| Chunk Size  | 200-1000 | 500     | Characters per text chunk                   |
| Overlap     | 0-200    | 50      | Overlapping characters between chunks       |
| Top K       | 1-10     | 3       | Number of relevant chunks to retrieve       |
| Temperature | 0.1-1.5  | 0.9     | LLM creativity (lower = more deterministic) |
| Top P       | 0.1-1.0  | 0.95    | Nucleus sampling threshold                  |

---

## 📖 Usage

### Method 1: Fresh Index (Upload Files)

1. Select **"Fresh Index"** option
2. Upload your `.txt` files using the file uploader
3. Click **"🚀 Build Index"**
4. Wait for processing to complete
5. Ask questions in the chat interface

### Method 2: Load from Disk

1. Place `.txt` files in the `to_upload_from_disk/` folder
2. Select **"Load from Disk"** option
3. Click **"🚀 Build Index"**
4. Ask questions in the chat interface

### Method 3: Add to Previous Index

1. Build an initial index using Method 1 or 2
2. Select **"Add to Previous Index"** option
3. Upload additional `.txt` files
4. Click **"🚀 Add to Index"**
5. The new documents will be merged with the existing index

### Asking Questions

1. Type your question in the text input
2. Press Enter or wait for automatic processing
3. View the generated answer
4. Expand **"📚 Sources"** to see the relevant document chunks

---

## 📁 Project Structure

```
Rag-Application/
├── RAGnarok.py                      # Main application file
├── README.md                       # Documentation
├── requirements.txt                # Python dependencies
├── .env                            # Environment configuration
├── .env.example                    # Example environment file
├── .gitignore                      # Git ignore patterns
│
├── .models/                        # Local AI models (auto-downloaded)
│   ├── all-MiniLM-L6-v2/          # Embedding model
│   └── flan-t5-small/             # Language model
│
├── to_upload_from_UI/             # Sample files for UI upload
│   └── *.txt
│
├── to_upload_from_disk/           # Bulk document folder
│   └── *.txt
│
├── faiss_index.pkl                # Saved FAISS index (auto-generated)
└── chunks.pkl                     # Saved text chunks (auto-generated)
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                          │
│                   (Streamlit App)                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   DOCUMENT INGESTION                        │
│  ┌──────────────┐              ┌──────────────┐             │
│  │ UI Upload    │              │ Disk Loader  │             │
│  └──────────────┘              └──────────────┘             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    TEXT PROCESSING                          │
│  Chunking (configurable size & overlap)                     │
│  Smart boundary detection (sentences, paragraphs)           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 EMBEDDING GENERATION                        │
│  SentenceTransformer (configurable via .env)                │
│  Batch processing for large datasets                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   VECTOR INDEXING                           │
│  FAISS IndexFlatL2 (L2/Euclidean distance)                  │
│  Pickle serialization for persistence                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    QUERY PROCESSING                         │
│  Question → Embedding → Vector similarity search            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   ANSWER GENERATION                         │
│  FLAN-T5 (configurable via .env)                            │
│  Context-aware text generation                              │
└─────────────────────────────────────────────────────────────┘
```

### How RAG Works

1. **Document Loading**: Text files are loaded from UI upload or disk
2. **Chunking**: Documents are split into overlapping chunks (default: 500 chars, 50 overlap)
3. **Embedding**: Each chunk is converted to a 384-dimensional vector
4. **Indexing**: Vectors are stored in a FAISS index for fast similarity search
5. **Query Processing**: User questions are embedded using the same model
6. **Retrieval**: Top-K most similar chunks are retrieved from the index
7. **Generation**: The LLM generates an answer using the retrieved context

---

## ❓ Troubleshooting

### Common Issues

#### Models not found

```
❌ Models not found
```

**Solution**: Click the "📥 Download Models" button or manually download models using the script in Step 5.

#### Out of memory error

**Solution**:

- Reduce chunk size in sidebar settings
- Use a smaller model in `.env`
- Close other applications

#### Slow embedding generation

**Solution**:

- Use batch processing (automatic for >100 chunks)
- Use a smaller embedding model
- Ensure you have sufficient RAM

#### FAISS import error

```
ImportError: No module named 'faiss'
```

**Solution**:

```bash
pip install faiss-cpu
```

#### .env file not loading

**Solution**:

- Ensure `.env` file is in the project root
- Install python-dotenv: `pip install python-dotenv`
- Restart the Streamlit app

### Performance Tips

1. **Use smaller models** for faster inference on limited hardware
2. **Adjust chunk size** based on your document structure
3. **Pre-build indexes** for frequently used document sets
4. **Use GPU** if available (install `faiss-gpu` instead of `faiss-cpu`)

---

## 📝 License

This project is open source and available under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Support

If you encounter any issues, please open an issue on GitHub.
