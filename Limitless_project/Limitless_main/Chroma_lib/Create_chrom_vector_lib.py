# AI_Assistant_main/Chroma_pdf_to_vector.py

import os
import re
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

# Get current file directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Folder up
BASE_DIR_UP = os.path.dirname(os.path.dirname(BASE_DIR))

# === CONFIG ===
PDF_FOLDER = os.path.join(BASE_DIR_UP, "Offline_lib_pdf")         # Folder with your PDFs
CHROMA_DB_DIR = os.path.join(BASE_DIR_UP, "Offline_lib_vector")   # Persistent Chroma DB folder
CHUNK_SIZE, CHUNK_OVERLAP = 500, 50

# === Init Chroma and embeddings ===
os.makedirs(CHROMA_DB_DIR, exist_ok=True)
client = PersistentClient(path=CHROMA_DB_DIR)

# Local embedding model (runs offline)
embedder = SentenceTransformer("all-mpnet-base-v2")
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")

# Get or create collection
collection = client.get_or_create_collection(
    name="pdf_collection",
    embedding_function=embedding_func
)

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + size)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += size - overlap
    return chunks

def detect_section_type(text, page_num, total_pages):
    """Tag chunks as content, toc, or index."""
    text_lower = text.lower()
    toc_keywords = ["table of contents", "contents"]
    index_keywords = ["index", "glossary", "bibliography", "references", "appendix"]

    early = page_num <= 5
    late = page_num >= total_pages - 10

    if any(k in text_lower for k in toc_keywords):
        return "toc"
    if any(k in text_lower for k in index_keywords):
        return "index"

    # Check if many lines end with numbers (like page refs)
    lines = text.split("\n")
    num_lines = len(lines)
    num_lines_page_nums = sum(bool(re.search(r"\d+$", line.strip())) for line in lines)
    if (early or late) and (num_lines_page_nums / max(1, num_lines) > 0.3):
        return "toc" if early else "index"

    return "content"

def pdf_already_ingested(pdf_name):
    try:
        results = collection.get(where={"source": pdf_name}, limit=1)
        return len(results.get('ids', [])) > 0
    except Exception as e:
        print(f"Warning: Could not get collection entries for {pdf_name}: {e}")
        return False

# === Process PDFs incrementally ===
for pdf_name in os.listdir(PDF_FOLDER):
    if not pdf_name.lower().endswith(".pdf"):
        continue

    if pdf_already_ingested(pdf_name):
        print(f"Skipping already ingested PDF: {pdf_name}")
        continue

    pdf_path = os.path.join(PDF_FOLDER, pdf_name)
    print(f"Processing new PDF: {pdf_name}")

    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if not text.strip():
            continue

        section_type = detect_section_type(text, page_num, total_pages)
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            uid = f"{pdf_name}_p{page_num}_c{i}"  # unique ID
            metadata = {
                "source": pdf_name,
                "page": page_num,
                "section_type": section_type
            }
            # Add to Chroma
            collection.add(
                documents=[chunk],
                metadatas=[metadata],
                ids=[uid]
            )

print("Ingestion complete. Data stored in:", CHROMA_DB_DIR)
