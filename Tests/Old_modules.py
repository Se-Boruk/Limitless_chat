from sentence_transformers import SentenceTransformer, CrossEncoder
from chromadb.utils import embedding_functions
from chromadb import PersistentClient
import os, gc, time, shutil, fitz, re
from docx import Document
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from ebooklib import epub


from config import RAG_EMBEDDER_PATH


class UniversalVectorStore:
    def __init__(self, data_folder, chroma_db_folder, chunk_size=256, chunk_overlap=64):
        self.data_folder = data_folder
        self.chroma_db_dir = chroma_db_folder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        os.makedirs(self.chroma_db_dir, exist_ok=True)

        # Embedder
        self.embedder = SentenceTransformer(RAG_EMBEDDER_PATH, trust_remote_code=True)
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=RAG_EMBEDDER_PATH, trust_remote_code=True)

        # Persistent Chroma client
        self.client = PersistentClient(path=self.chroma_db_dir)
        self.collection = self.client.get_or_create_collection(
            name="universal_collection",
            embedding_function=self.embedding_func
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if hasattr(self, 'client'):
                del self.client
            gc.collect()
            time.sleep(2)

            if os.path.exists(self.chroma_db_dir):
                for filename in os.listdir(self.chroma_db_dir):
                    file_path = os.path.join(self.chroma_db_dir, filename)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error during cleanup: {e}")
            
    def get_ingestion_status(self):
        status = {}
        for filename in os.listdir(self.data_folder):
            if not self.is_supported_file(filename):
                continue
            status[filename] = self.file_already_ingested(filename)
        return status

    # --- Text chunking ---
    def chunk_text(self, text):
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(len(words), start + self.chunk_size)
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    # --- Section detection for PDFs ---
    def detect_section_type(self, text, page_num=None, total_pages=None):
        text_lower = text.lower()
        toc_keywords = ["table of contents", "contents"]
        index_keywords = ["index", "glossary", "bibliography", "references", "appendix"]

        early = page_num is not None and page_num <= 5
        late = page_num is not None and total_pages is not None and page_num >= total_pages - 10

        if any(k in text_lower for k in toc_keywords):
            return "toc"
        if any(k in text_lower for k in index_keywords):
            return "index"

        lines = text.split("\n")
        num_lines = len(lines)
        num_lines_page_nums = sum(bool(re.search(r"\d+$", line.strip())) for line in lines)
        if (early or late) and (num_lines_page_nums / max(1, num_lines) > 0.3):
            return "toc" if early else "index"

        return "content"

    # --- Check if file already ingested ---
    def file_already_ingested(self, filename):
        try:
            results = self.collection.get(where={"source": filename}, limit=1)
            return len(results.get('ids', [])) > 0
        except Exception as e:
            print(f"Warning: Could not get collection entries for {filename}: {e}")
            return False

    # --- File ingestion methods ---
    def add_pdf(self, pdf_path):
        pdf_name = os.path.basename(pdf_path)
        if self.file_already_ingested(pdf_name):
            print(f"Skipping already ingested PDF: {pdf_name}")
            return

        print(f"Processing PDF: {pdf_name}")
        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if not text.strip():
                continue

            section_type = self.detect_section_type(text, page_num, total_pages)
            chunks = self.chunk_text(text)
            for i, chunk in enumerate(chunks):
                uid = f"{pdf_name}_p{page_num}_c{i}"
                metadata = {"source": pdf_name, "page": page_num, "section_type": section_type}
                self.collection.add(documents=[chunk], metadatas=[metadata], ids=[uid])

    def add_docx(self, docx_path):
        docx_name = os.path.basename(docx_path)
        if self.file_already_ingested(docx_name):
            print(f"Skipping already ingested DOCX: {docx_name}")
            return

        print(f"Processing DOCX: {docx_name}")
        doc = Document(docx_path)
        full_text = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
        text = "\n".join(full_text)
        chunks = self.chunk_text(text)
        for i, chunk in enumerate(chunks):
            uid = f"{docx_name}_chunk{i}"
            metadata = {"source": docx_name, "section_type": "content"}
            self.collection.add(documents=[chunk], metadatas=[metadata], ids=[uid])

    def add_txt(self, txt_path):
        txt_name = os.path.basename(txt_path)
        if self.file_already_ingested(txt_name):
            print(f"Skipping already ingested TXT: {txt_name}")
            return

        print(f"Processing TXT: {txt_name}")
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        chunks = self.chunk_text(text)
        for i, chunk in enumerate(chunks):
            uid = f"{txt_name}_chunk{i}"
            metadata = {"source": txt_name, "section_type": "content"}
            self.collection.add(documents=[chunk], metadatas=[metadata], ids=[uid])

    def add_xml(self, xml_path):
        xml_name = os.path.basename(xml_path)
        if self.file_already_ingested(xml_name):
            print(f"Skipping already ingested XML: {xml_name}")
            return

        print(f"Processing XML: {xml_name}")
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            texts = []

            def recursive_text_extract(elem):
                if elem.text and elem.text.strip():
                    texts.append(elem.text.strip())
                for child in elem:
                    recursive_text_extract(child)
                if elem.tail and elem.tail.strip():
                    texts.append(elem.tail.strip())

            recursive_text_extract(root)
            full_text = "\n".join(texts)
            chunks = self.chunk_text(full_text)
            for i, chunk in enumerate(chunks):
                uid = f"{xml_name}_chunk{i}"
                metadata = {"source": xml_name, "section_type": "content"}
                self.collection.add(documents=[chunk], metadatas=[metadata], ids=[uid])
        except Exception as e:
            print(f"Failed to parse XML {xml_name}: {e}")

    def add_epub(self, epub_path):
        epub_name = os.path.basename(epub_path)
        if self.file_already_ingested(epub_name):
            print(f"Skipping already ingested EPUB: {epub_name}")
            return

        print(f"Processing EPUB: {epub_name}")
        try:
            book = epub.read_epub(epub_path)
            texts = []
            for item in book.get_items():
                if item.get_type() == epub.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text = soup.get_text(separator='\n')
                    if text.strip():
                        texts.append(text.strip())
            full_text = "\n".join(texts)
            chunks = self.chunk_text(full_text)
            for i, chunk in enumerate(chunks):
                uid = f"{epub_name}_chunk{i}"
                metadata = {"source": epub_name, "section_type": "content"}
                self.collection.add(documents=[chunk], metadatas=[metadata], ids=[uid])
        except Exception as e:
            print(f"Failed to parse EPUB {epub_name}: {e}")

    def add_html(self, html_path):
        html_name = os.path.basename(html_path)
        if self.file_already_ingested(html_name):
            print(f"Skipping already ingested HTML: {html_name}")
            return

        print(f"Processing HTML: {html_name}")
        try:
            with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text(separator='\n')
            if not text.strip():
                print(f"No text extracted from HTML {html_name}")
                return
            chunks = self.chunk_text(text)
            for i, chunk in enumerate(chunks):
                uid = f"{html_name}_chunk{i}"
                metadata = {"source": html_name, "section_type": "content"}
                self.collection.add(documents=[chunk], metadatas=[metadata], ids=[uid])
        except Exception as e:
            print(f"Failed to parse HTML {html_name}: {e}")

    def add_file(self, file_path):
        ext = file_path.lower().split('.')[-1]
        if ext == "pdf":
            self.add_pdf(file_path)
        elif ext == "docx":
            self.add_docx(file_path)
        elif ext == "txt":
            self.add_txt(file_path)
        elif ext == "xml":
            self.add_xml(file_path)
        elif ext == "epub":
            self.add_epub(file_path)
        elif ext in ("html", "htm"):
            self.add_html(file_path)
        else:
            print(f"Unsupported file format: {file_path}")

    def ingest_all(self):
        for filename in os.listdir(self.data_folder):
            if not self.is_supported_file(filename):
                continue
            file_path = os.path.join(self.data_folder, filename)
            self.add_file(file_path)
        print("Incremental ingestion complete.")

    def is_supported_file(self, filename):
        lower = filename.lower()
        return lower.endswith((".pdf", ".docx", ".txt", ".xml", ".epub", ".html", ".htm"))

    # --- RAG-aware search with thresholds and re-ranking ---
    def search(self, query, top_n=3, similarity_threshold=0.5, absolute_min=0.15,
               reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    
        query_embedding = self.embedder.encode(query, convert_to_tensor=True).cpu().numpy()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_n,
            include=["documents", "distances"]
        )
    
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        similarities = [1 - d for d in distances]  # cosine similarity
    
        # Absolute minimum cutoff
        filtered = [(doc, sim) for doc, sim in zip(documents, similarities) if sim >= absolute_min]
    
        if not filtered:
            # rerank all top_n with cross-encoder
            cross_encoder = CrossEncoder(reranker_model)
            rerank_pairs = [[query, doc] for doc in documents]
            rerank_scores = cross_encoder.predict(rerank_pairs)
            reranked_docs = sorted(zip(documents, rerank_scores), key=lambda x: x[1], reverse=True)
            return reranked_docs[:top_n]
    
        # similarity threshold logic
        above_threshold = [(doc, sim) for doc, sim in filtered if sim >= similarity_threshold]
    
        if len(above_threshold) == top_n:
            return above_threshold
        elif len(above_threshold) > 0:
            return above_threshold
        else:
            # rerank filtered ones
            cross_encoder = CrossEncoder(reranker_model)
            rerank_pairs = [[query, doc] for doc, _ in filtered]
            rerank_scores = cross_encoder.predict(rerank_pairs)
            reranked_docs = sorted(zip([doc for doc, _ in filtered], rerank_scores), key=lambda x: x[1], reverse=True)
            return reranked_docs[:top_n]
    
    
    
    
    
    
    
    
