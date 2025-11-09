# ingest.py — LawGPT Embedding Creation Script (Ollama Version)
# -------------------------------------------------------------
# This script loads PDFs from the `data/` folder,
# splits them into chunks, creates embeddings using
# a local Ollama model, and saves them in a FAISS vector DB.

import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# 1️⃣ Verify the data folder exists
if not os.path.exists("data"):
    raise FileNotFoundError("❌ 'data/' folder not found. Please create one and add your IPC PDFs inside it.")

print("🔹 Loading PDFs from the 'data' folder...")
loader = DirectoryLoader('data', glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
print(f"✅ Loaded {len(documents)} PDF documents.")

if len(documents) == 0:
    raise ValueError("❌ No PDFs found in the 'data' folder. Please add some legal documents (e.g. IPC sections).")

# 2️⃣ Split documents into smaller chunks
print("🔹 Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
print(f"✅ Split into {len(texts)} text chunks.")

# 3️⃣ Create text embeddings using Ollama
print("🔹 Creating text embeddings using Ollama model (llama3)...")
embeddings = OllamaEmbeddings(model="llama3")
print("✅ Ollama embedding model loaded successfully.")

# 4️⃣ Generate FAISS vector database
print("🔹 Generating FAISS vector database...")
faiss_db = FAISS.from_documents(texts, embeddings)
print("✅ Vector database created successfully.")

# 5️⃣ Save the FAISS vector database locally
output_dir = "ipc_vector_db"
print(f"🔹 Saving FAISS database to ./{output_dir} ...")
faiss_db.save_local(output_dir)

if os.path.exists(output_dir):
    print(f"✅ Successfully saved vector database in: {output_dir}/")
    print("🎉 Ingestion complete! You can now run: streamlit run app.py")
else:
    print("⚠️ Something went wrong — vector DB not found after saving.")
