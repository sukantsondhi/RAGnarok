import os
from pathlib import Path
import pickle
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get model configuration from environment variables
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2')
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME', 'google/flan-t5-small')
MODELS_DIR = os.getenv('MODELS_DIR', '.models')


# Helper funtions for loading and processing text files

def load_text_file(file_path):
    """
    Load text from a file
    
    :param file_path: Path to teh text file
    :return: Contents of the file as a string
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_documents_from_folder(folder_path):
    """
    Load all text files form a folder
    
    :param folder_path: Path to teh folder containig text files
    :return: List of dictionaries with filename and contnet
    """
    documents = []
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                text = load_text_file(file_path)
                documents.append({'filename': filename, 'content': text})
    return documents

# Lets chunk teh text into smaller peices

# I'm using chunk 500 and overlap of 50 as it's 10% and maintians the context.
# You can use 300 and 30 for more precises retreival.
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    """
    Split text into overlaping chunks
    
    :param text: The text content to be chunkked
    :param chunk_size: Size of each chunk (defualt: 500)
    :param chunk_overlap: Overlap betwen chunks to maintain context (defualt: 50)
    :return: List of text chunks
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence or word boundaries
        if end < text_len:
            # Look for sentence end
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            last_space = chunk.rfind(' ')
            
            break_point = max(last_period, last_newline, last_space)
            if break_point > chunk_size * 0.5:  # Only break if we're past halfway
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1
        
        chunks.append(chunk.strip())
        start = end - chunk_overlap
    
    return chunks

# Createing Embeddings
def create_embeddings(texts, model):
    """
    Docstring for create_embeddings
    
    :param texts: Texts from teh text file
    :param model: Embedding Model
    """
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

def create_embeddings_batch(texts, model, batch_size=32):
    """
    Docstring for create_embeddings_batch
    
    :param texts: Texts from the text file
    :param model: Embedding Model
    :param batch_size: Size of the batches for memeory managment
    """
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.append(embeddings)
    
    return np.vstack(all_embeddings)

# Building FAISS Index
def build_faiss_index(embeddings):
    """
    Build FAISS index for similarity serach
    
    :param embeddings: The Embeddings numpy array
    :return: FAISS index object
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index

def save_index_and_chunks(index, chunks, embeddings, index_path='faiss_index.pkl', chunks_path='chunks.pkl'):
    """
    Save FAISS index and Chunks using pickle
    
    :param index: FAISS index object to seralize
    :param chunks: List of text chunks to save
    :param embeddings: Numpy array of embeddigns
    :param index_path: Path to save the FAISS index (defualt: 'faiss_index.pkl')
    :param chunks_path: Path to save the chunks (defualt: 'chunks.pkl')
    """
    with open(index_path, 'wb') as f:
        pickle.dump({'index': faiss.serialize_index(index), 'embeddings': embeddings}, f)
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)

# Let's load the chunks and index
def load_index_and_chunks(index_path='faiss_index.pkl', chunks_path='chunks.pkl'):
    """
    Load FAISS index and chunks from pickle files
    
    :param index_path: Path to the saved FAISS index file
    :param chunks_path: Path to the saved chunks file
    :return: Tuple of (index, chunks, embeddigns) or (None, None, None)
    """
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        with open(index_path, 'rb') as f:
            data = pickle.load(f)
            index = faiss.deserialize_index(data['index'])
            embeddings = data['embeddings']
        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)
        return index, chunks, embeddings
    return None, None, None

# Search similar chunk
def search_similar_chunks(query, query_embedding, index, chunks, k=3):
    """
    Searching Similar chinks with k nearest Neighbour

    :param query: User query as text
    :param query_embeddings: User query as embeddigns
    :param index: FAISS Vector Database
    :param chunks: Chunks of data on disk
    :param k: Nearest Neighbour is set to 3 by defualt
    """
    distances, indices = index.search(query_embedding.reshape(1, -1).astype('float32'), k)
    return [(chunks[i], distances[0][idx]) for idx, i in enumerate(indices[0])]

# Generating Answer
def generate_answer(question, context_chunks, generator):
    """
    Generating the answer usign the LLM

    :param question: User questoin
    :param context_chunks: Relevent chunks from the search
    :param generator: The LLM pipeline for genration
    """
    # Combine context togather
    context = "\n\n".join([chunk for chunk, _ in context_chunks])
    
    # Create prompt
    prompt = f"""Based on the following context, answer the question in a detailed and helpful way.

Context:
{context}

Question: {question}

Answer:"""
    
    # Generate answer
    result = generator(prompt, max_length=512, min_length=30, num_return_sequences=1, do_sample=True)
    
    # Extract just the answer portion
    generated_text = result[0]['generated_text']
    if "Answer:" in generated_text:
        answer = generated_text.split("Answer:")[-1].strip()
    else:
        answer = generated_text.strip()
    
    return answer

# Clear session state for frehs start
def clear_session_state():
    """
    Clear all sesison state for fresh start
    
    :return: None - just clears the state
    """
    keys_to_clear = ['faiss_index', 'chunks', 'embeddings', 'embedding_model', 
                     'generator', 'doc_count', 'chunk_count', 'chat_history']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

# UI Time!

# Title
st.set_page_config(page_title="RAGnarok", page_icon="⚡", layout="wide", initial_sidebar_state='auto')
st.title("⚡ RAGnarok")

# Model paths - derive folder names from model names in .env
embedding_model_folder = EMBEDDING_MODEL_NAME.split('/')[-1]
llm_model_folder = LLM_MODEL_NAME.split('/')[-1]
embedding_model_path = os.path.join(os.path.dirname(__file__), MODELS_DIR, embedding_model_folder)
llm_model_path = os.path.join(os.path.dirname(__file__), MODELS_DIR, llm_model_folder)

# Check and display model status
models_available = os.path.exists(embedding_model_path) and os.path.exists(llm_model_path)

if models_available:
    st.success("✅ Models ready")
else:
    st.error("❌ Models not found")
    st.info(f"📋 Models configured: `{EMBEDDING_MODEL_NAME}` and `{LLM_MODEL_NAME}`")
    if st.button("📥 Download Models"):
        with st.spinner(f"Downloading embedding model ({EMBEDDING_MODEL_NAME})..."):
            os.makedirs(MODELS_DIR, exist_ok=True)
            model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            model.save(embedding_model_path)
        
        with st.spinner(f"Downloading LLM ({LLM_MODEL_NAME})..."):
            tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
            llm = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
            tokenizer.save_pretrained(llm_model_path)
            llm.save_pretrained(llm_model_path)
        
        st.success("✅ Models downloaded!")
        st.rerun()
    st.stop()

# Initialize session state
if 'previous_source_option' not in st.session_state:
    st.session_state.previous_source_option = None

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    chunk_size = st.slider("Chunk Size", 200, 1000, 500, 50)
    chunk_overlap = st.slider("Overlap", 0, 200, 50, 10)
    top_k = st.slider("Top K", 1, 10, 3)
    
    st.markdown("---")
    temperature = st.slider("Temperature", 0.1, 1.5, 0.9, 0.1)
    top_p = st.slider("Top P", 0.1, 1.0, 0.95, 0.05)
    
    if "chunk_count" in st.session_state:
        st.markdown("---")
        st.metric("Chunks", st.session_state.chunk_count)

# Document source selection
st.markdown("### 📂 Source")
source_option = st.radio(
    "Select:",
    ["Fresh Index", "Load from Disk", "Add to Previous Index"],
    horizontal=True,
    label_visibility="collapsed"
)

# Reset state on option change
if st.session_state.previous_source_option != source_option:
    clear_session_state()
    st.session_state.previous_source_option = source_option

uploaded_files = None
folder_documents = None

# Fresh Index option
if source_option == "Fresh Index":
    st.caption("💡 Create a new index from uploaded files")
    uploaded_files = st.file_uploader(
        "Upload .txt files", 
        type=["txt"], 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    if uploaded_files:
        st.info(f"📄 {len(uploaded_files)} file(s) selected")

# Load from Disk option
elif source_option == "Load from Disk":
    disk_folder = os.path.join(os.path.dirname(__file__), 'to_upload_from_disk')
    if os.path.exists(disk_folder):
        folder_documents = load_documents_from_folder(disk_folder)
        st.caption(f"📁 {len(folder_documents)} files in Disk")
        
        with st.expander("View files"):
            for doc in folder_documents:
                st.text(f"• {doc['filename']}")
    else:
        st.error("❌ Folder not found")

# Add to Previous Index option
elif source_option == "Add to Previous Index":
    st.caption("💡 Add files to your existing index")
    
    # Auto-load previous index
    if 'faiss_index' not in st.session_state:
        index, chunks, embeddings = load_index_and_chunks()
        if index is not None:
            st.session_state.faiss_index = index
            st.session_state.chunks = chunks
            st.session_state.embeddings = embeddings
            st.session_state.chunk_count = len(chunks)
            st.success(f"✅ Loaded {len(chunks)} chunks")
        else:
            st.warning("⚠️ No previous index found")
    
    uploaded_files = st.file_uploader(
        "Upload .txt files", 
        type=["txt"], 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    if uploaded_files:
        st.info(f"📄 {len(uploaded_files)} file(s) to add")

# Check for documents
has_documents = (source_option in ["Fresh Index", "Add to Previous Index"] and uploaded_files) or \
                (source_option == "Load from Disk" and folder_documents)

# Button text
button_text = {
    "Fresh Index": "🚀 Build Index",
    "Load from Disk": "🚀 Build Index", 
    "Add to Previous Index": "🚀 Add to Index"
}.get(source_option, "🚀 Build")

if st.button(button_text, type="primary") and has_documents:
    with st.spinner("Processing..."):
        all_text = ""
        doc_count = 0
        
        if source_option in ["Fresh Index", "Add to Previous Index"]:
            # Process uploaded files
            for file in uploaded_files:
                text = file.read().decode('utf-8')
                all_text += f"\n\n--- {file.name} ---\n\n{text}"
                doc_count += 1
        else:
            # Process disk files
            for doc in folder_documents:
                all_text += f"\n\n--- {doc['filename']} ---\n\n{doc['content']}"
                doc_count += 1

        # Chunk the text
        new_chunks = chunk_text(all_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Load embedding model
        embedding_model = SentenceTransformer(embedding_model_path)
        st.session_state.embedding_model = embedding_model
        
        # Create embeddings for new chunks
        with st.spinner("🔍 Creating embeddings..."):
            if len(new_chunks) > 100:
                new_embeddings = create_embeddings_batch(new_chunks, embedding_model, batch_size=32)
            else:
                new_embeddings = create_embeddings(new_chunks, embedding_model)
        
        # Handle index based on source option
        if source_option == "Add to Previous Index" and 'faiss_index' in st.session_state:
            # Merge with existing index
            st.info("🔗 Merging with existing index...")
            existing_chunks = st.session_state.chunks
            existing_embeddings = st.session_state.embeddings
            
            # Combine chunks and embeddings
            chunks = existing_chunks + new_chunks
            embeddings = np.vstack([existing_embeddings, new_embeddings])
            
            st.session_state.doc_count = st.session_state.get('doc_count', 0) + doc_count
        else:
            # Fresh index
            chunks = new_chunks
            embeddings = new_embeddings
            st.session_state.doc_count = doc_count
        
        st.session_state.chunk_count = len(chunks)
        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings
        
        # Build FAISS index
        with st.spinner("🔍 Building search index..."):
            index = build_faiss_index(embeddings)
            st.session_state.faiss_index = index
        
        # Save index and chunks using pickle
        with st.spinner("💾 Saving index..."):
            save_index_and_chunks(index, chunks, embeddings)
            st.info("✅ Index saved for future use")
        
        # Load FLAN-T5 model from local folder
        with st.spinner("🤖 Loading language model..."):
            tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_path)
            
            generator = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.2,
                num_beams=2
            )
            
            st.session_state.generator = generator
        
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

    st.success(f"✅ Ready! {len(chunks)} chunks indexed")
    st.balloons()

# Chat interface
if "generator" in st.session_state and "faiss_index" in st.session_state:
    st.markdown("---")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader("💬 Ask")
    with col2:
        if st.button("🔄 Clear"):
            st.session_state.chat_history = []
            st.rerun()
    
    question = st.text_input(
        "Question:",
        placeholder="Ask about your documents...",
        label_visibility="collapsed"
    )
    
    if question:
        with st.spinner("Thinking..."):
            # Create query embedding
            query_embedding = st.session_state.embedding_model.encode([question])[0]
            
            # Search for relevant chunks
            similar_chunks = search_similar_chunks(
                question,
                query_embedding,
                st.session_state.faiss_index,
                st.session_state.chunks,
                k=top_k
            )
            
            # Generate answer
            answer = generate_answer(question, similar_chunks, st.session_state.generator)
            
            # Save to chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            st.session_state.chat_history.append((question, answer))
            
            # Display answer
            st.markdown("### 💡 Answer")
            st.markdown(answer)
            
            with st.expander("📚 Sources"):
                for i, (chunk, distance) in enumerate(similar_chunks):
                    st.caption(f"**Source {i+1}** (dist: {distance:.3f})")
                    st.text(chunk[:200] + "..." if len(chunk) > 200 else chunk)
else:
    st.info("📤 Select source and build index to start")

# st.markdown("---")
# rate = st.slider("Rate the RAG App", 0, 10, 0)
# st.write("### I would rate this App a Solid", rate)