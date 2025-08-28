from chromadb import PersistentClient
import fitz  # PyMuPDF for reading PDFs
import os

PDF_FOLDER = "pdfs_new"
CHROMA_DB_DIR = "chroma_store"

client = PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_collection("pdf_collection")

def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if not text.strip():
            continue
        # Your chunking and tagging here
        chunks = [text]  # or chunk_text(text) from before
        for i, chunk in enumerate(chunks):
            uid = f"{os.path.basename(pdf_path)}_p{page_num}_c{i}"
            metadata = {
                "source": os.path.basename(pdf_path),
                "page": page_num
            }
            collection.add(documents=[chunk], metadatas=[metadata], ids=[uid])

for pdf_file in os.listdir(PDF_FOLDER):
    if pdf_file.endswith(".pdf"):
        process_pdf(os.path.join(PDF_FOLDER, pdf_file))

client.persist()
print("Added new PDFs incrementally.")
