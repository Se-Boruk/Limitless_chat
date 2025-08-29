import os
import gc
import time
import shutil
import fitz
import xml.etree.ElementTree as ET
from docx import Document
from bs4 import BeautifulSoup
from ebooklib import epub
import re
from sentence_transformers import SentenceTransformer, CrossEncoder
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import numpy as np
from torch.amp import autocast
import torch

from transformers import AutoModel, AutoTokenizer
from sentence_transformers import models

from config import RAG_EMBEDDER_PATH, RAG_CROSS_ENC_PATH, RAG_BATCH_SIZE


class LocalEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, st_model):
        self.st_model = st_model

    # IMPORTANT: argument must be `input` (Chroma >=0.4.16)
    def __call__(self, input):
        # input can be a single string or list of strings
        if isinstance(input, str):
            return self.st_model.encode([input], convert_to_numpy=True, show_progress_bar=False)[0]
        else:
            return self.st_model.encode(input, convert_to_numpy=True, show_progress_bar=False)

    def name(self):
        # Required by Chroma
        return "local_embedder"
    
class UniversalVectorStore:
    def __init__(self, data_folder, chroma_db_folder, chunk_size=256, chunk_overlap=64, embedder_path=None):
        self.data_folder = data_folder
        self.chroma_db_dir = chroma_db_folder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        os.makedirs(self.chroma_db_dir, exist_ok=True)
        
    def unload(self):
        # --- Safe delete helper ---
        def safe_del(attr_name):
            if hasattr(self, attr_name):
                attr = getattr(self, attr_name)
                if attr is not None:
                    try:
                        # Move model off GPU if it has `.cpu()`
                        if hasattr(attr, 'cpu'):
                            attr.cpu()
                        delattr(self, attr_name)
                    except:
                        pass
                setattr(self, attr_name, None)
    
        # --- Delete all relevant attributes safely ---
        safe_del('embedder')
        safe_del('tokenizer')
        safe_del('embedding_func')
        safe_del('client')
        safe_del('collection')
        safe_del('cross_encoder')
    
        # --- Force garbage collection and free CUDA memory ---
        gc.collect()
        torch.cuda.empty_cache()  
 

    def load_model_fp32(self):
        self.unload()  # clean before loading new  
        
        print("Loading fp32 embedder for RAG...")
        # --- Load embedder on GPU ---
        self.embedder = SentenceTransformer(RAG_EMBEDDER_PATH, device='cuda', trust_remote_code=True)
    
        # --- Wrap embedding function for Chroma (float16) ---
        self.embedding_func = LocalEmbeddingFunction(self.embedder)
    
        # --- Persistent Chroma client ---
        self.client = PersistentClient(path=self.chroma_db_dir)
        self.collection = self.client.get_or_create_collection(
            name="universal_collection",
            embedding_function=self.embedding_func
        )
        print("Loaded!")
        # --- Force garbage collection and free CUDA memory ---
        gc.collect()
        torch.cuda.empty_cache()  

    def load_model_fp16(self):
        self.unload()  # clean before loading new  
        
        print("Loading fp16 embedder for RAG...")
        # --- Load tokenizer and model in FP16 ---
        tokenizer = AutoTokenizer.from_pretrained(RAG_EMBEDDER_PATH, trust_remote_code=True)
        base_model = AutoModel.from_pretrained(
            RAG_EMBEDDER_PATH,
            torch_dtype=torch.float16,   # FP16
            device_map="cuda",
            trust_remote_code=True
        )

        # --- Wrap model + tokenizer into SentenceTransformer ---
        # We keep the original ST style: pooling + transformer
        word_embedding_model = models.Transformer(RAG_EMBEDDER_PATH)
        word_embedding_model.auto_model = base_model
        word_embedding_model.tokenizer = tokenizer

        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        self.embedder = SentenceTransformer(modules=[word_embedding_model, pooling_model], device="cuda")

        # --- Wrap embedding function for Chroma (float16) ---
        self.embedding_func = LocalEmbeddingFunction(self.embedder)

        # --- Persistent Chroma client ---
        self.client = PersistentClient(path=self.chroma_db_dir)
        self.collection = self.client.get_or_create_collection(
            name="universal_collection",
            embedding_function=self.embedding_func
        )
        print("Loaded!")
        

        print("Loading cross encoder in FP16...")
        self.cross_encoder = CrossEncoder(RAG_CROSS_ENC_PATH, device='cuda')
        self.cross_encoder.model = self.cross_encoder.model.half()  # convert weights to FP16
        print("Loaded!")
        # --- Force garbage collection and free CUDA memory ---
        gc.collect()
        torch.cuda.empty_cache()  

    # --- Context manager for cleanup ---
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

    # --- Utility functions ---
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

    def file_already_ingested(self, filename):
        try:
            results = self.collection.get(where={"source": filename}, limit=1)
            return len(results.get('ids', [])) > 0
        except Exception as e:
            print(f"Warning: Could not get collection entries for {filename}: {e}")
            return False

    # --- File ingestion ---
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
            self._add_chunks(pdf_name, chunks, section_type, page_num)

    def add_docx(self, docx_path):
        docx_name = os.path.basename(docx_path)
        if self.file_already_ingested(docx_name):
            return
        print(f"Processing DOCX: {docx_name}")
        
        doc = Document(docx_path)
        text = "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
        chunks = self.chunk_text(text)
        self._add_chunks(docx_name, chunks, "content")

    def add_txt(self, txt_path):
        txt_name = os.path.basename(txt_path)
        if self.file_already_ingested(txt_name):
            return

        print(f"Processing TXT: {txt_name}")

        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        chunks = self.chunk_text(text)
        self._add_chunks(txt_name, chunks, "content")

    def add_xml(self, xml_path):
        xml_name = os.path.basename(xml_path)
        if self.file_already_ingested(xml_name):
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
            self._add_chunks(xml_name, chunks, "content")
        except Exception as e:
            print(f"Failed to parse XML {xml_name}: {e}")

    def add_epub(self, epub_path):
        epub_name = os.path.basename(epub_path)
        if self.file_already_ingested(epub_name):
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
            self._add_chunks(epub_name, chunks, "content")
        except Exception as e:
            print(f"Failed to parse EPUB {epub_name}: {e}")

    def add_html(self, html_path):
        html_name = os.path.basename(html_path)
        if self.file_already_ingested(html_name):
            return
        print(f"Processing HTML: {html_name}")
        
        try:
            with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text(separator='\n')
            if text.strip():
                chunks = self.chunk_text(text)
                self._add_chunks(html_name, chunks, "content")
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
            if self.is_supported_file(filename):
                self.add_file(os.path.join(self.data_folder, filename))
        print("Incremental ingestion complete.")

    def is_supported_file(self, filename):
        lower = filename.lower()
        return lower.endswith((".pdf", ".docx", ".txt", ".xml", ".epub", ".html", ".htm"))

    # --- Internal chunk addition with GPU embeddings ---
    def _add_chunks(self, source_name, chunks, section_type, page_num=None):
        # GPU batch processing
        batch_size = RAG_BATCH_SIZE
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            embeddings = self.embedder.encode(batch, convert_to_tensor=True, normalize_embeddings=True, device='cuda')
            embeddings = embeddings.cpu().numpy()
            for j, emb in enumerate(embeddings):
                uid = f"{source_name}_chunk{i+j}" + (f"_p{page_num}" if page_num else "")
                metadata = {"source": source_name, "section_type": section_type}
                if page_num:
                    metadata["page"] = page_num
                self.collection.add(documents=[batch[j]], metadatas=[metadata], ids=[uid])



    # --- GPU-accelerated RAG search ---
    def safe_cosine(self, a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


    def convert_distance_to_cosine(self, distances, doc_embeddings=None, query_embedding=None):
        if doc_embeddings is not None and query_embedding is not None:
            q = np.asarray(query_embedding, dtype=np.float32).ravel()
            return [self.safe_cosine(q, d) for d in doc_embeddings]
        d = np.array(distances, dtype=np.float32)
        if d.max() <= 1.05:
            return (1.0 - d).tolist()
        return (1.0 - (d ** 2) / 2.0).tolist()


    def truncate_doc_for_reranker(self, doc_text, cross_enc_model, max_len=480):
        try:
            tokenizer = cross_enc_model.tokenizer
            tokens = tokenizer(doc_text, truncation=True, max_length=max_len, return_tensors="pt")
            truncated = tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
            return truncated
        except Exception:
            return doc_text[: max_len * 4]


    def search(self, query, top_n=3, absolute_cosine_min=0.1, min_relevance=0.7):
        # --- Encode query ---
        query_embedding = self.embedder.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device='cuda'  # or 'cpu'
        )

        # --- Query vector DB ---
        results = self.collection.query(
            query_embeddings=[query_embedding.cpu().numpy()],
            n_results=top_n * 15,
            include=["documents", "distances", "embeddings"]
        )
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        doc_embeddings = results.get("embeddings", [[]])[0]

        # --- Convert distances to cosine ---
        use_doc_embeddings = doc_embeddings if (doc_embeddings is not None and len(doc_embeddings) > 0) else None
        cosines = self.convert_distance_to_cosine(distances, use_doc_embeddings, query_embedding.cpu().numpy())

        # --- Filter by absolute cosine min ---
        filtered = [(doc, sim, idx) for idx, (doc, sim) in enumerate(zip(documents, cosines)) if sim >= absolute_cosine_min]
        if not filtered:
            print("No documents above absolute cosine minimum.")
            return []

        filtered = sorted(filtered, key=lambda x: x[1], reverse=True)[:top_n * 15]

        print("Best embedder score: ",filtered[0][1])
        print("Best embedder paragraph:\n",filtered[0][0])

        rerank_pairs = []
        original_docs = []
        for doc, sim, idx in filtered:
            truncated = self.truncate_doc_for_reranker(doc, self.cross_encoder)
            rerank_pairs.append([query, truncated])
            original_docs.append((doc, sim))

        # --- Predict reranker scores ---
        with autocast(device_type='cuda', dtype=torch.float16):
            rerank_scores = self.cross_encoder.predict(rerank_pairs, show_progress_bar=False)
            
        rerank_scores = np.array(rerank_scores, dtype=np.float32)

        # --- Normalize to 0-1 using sigmoid ---
        rerank_scores_norm = 1 / (1 + np.exp(-rerank_scores))

        # --- Zip rerank scores with original docs ---
        reranked_docs = [(doc, float(score)) for (doc, _), score in zip(original_docs, rerank_scores_norm)]

        # --- Filter by minimum relevance ---
        relevant_docs = [d for d in reranked_docs if d[1] >= min_relevance]

        if not relevant_docs:
            best_parag, max_sim = max(reranked_docs, key=lambda x: x[1])
            print("No documents exceed minimum relevance threshold. Max: ",max_sim)
            print("Best paragraph (not, included):\n",best_parag)
            return []
        
        # --- Sort descending by reranker score ---
        relevant_docs = sorted(relevant_docs, key=lambda x: x[1], reverse=True)

        top_docs = relevant_docs[:top_n]
        _, max_rerank_sim = max(top_docs, key=lambda x: x[1])
        print("Best rerank score: ", max_rerank_sim)
        
        # --- Return top_n of relevant docs ---
        return top_docs
