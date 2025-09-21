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
import nltk
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import models
from tqdm import tqdm
from config import RAG_EMBEDDER_PATH, RAG_CROSS_ENC_PATH, RAG_BATCH_SIZE
import threading
import queue
import Functions


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
    def __init__(self, data_folder, chroma_db_folder, chunk_size=256, overlap_ratio=0.25, embedder_path=None):
        self.data_folder = data_folder
        self.chroma_db_dir = chroma_db_folder
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio

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
 

    def load_model_fp32(self, load_cross_encoder = True):
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
        
        if load_cross_encoder:
            print("Loading cross encoder in FP16...")
            self.cross_encoder = CrossEncoder(RAG_CROSS_ENC_PATH, device='cuda', trust_remote_code=True)
            self.cross_encoder.model = self.cross_encoder.model.half()  # convert weights to FP16
            print("Loaded!")
            # --- Force garbage collection and free CUDA memory ---
            gc.collect()
            torch.cuda.empty_cache()  

    def load_model_fp16(self, load_cross_encoder = True):
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
        word_embedding_model = models.Transformer(
            model_name_or_path=RAG_EMBEDDER_PATH,
            config_args={"trust_remote_code": True}  
        )
        
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
        
        if load_cross_encoder:
            print("Loading cross encoder in FP16...")
            self.cross_encoder = CrossEncoder(RAG_CROSS_ENC_PATH, device='cuda', trust_remote_code=True)
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

    def filter_contained_chunks(self, chunks):
        """
        Remove chunks that are entirely contained inside another chunk.
        """
        final_chunks = []
        for chunk in chunks:
            # Skip if chunk is contained in any of the already finalized chunks
            if any(chunk in prev_chunk for prev_chunk in final_chunks):
                continue
            final_chunks.append(chunk)
        return final_chunks
    
    
    def get_dynamic_chunk_size(self, total_lenght, base_chunk = 256, base_overlap = 0.25):

        print("Length: ",total_lenght)
        if total_lenght < 35_000:
            chunk_size = base_chunk
            chunk_overlap = base_overlap
            
            #print("chunk:", chunk_size)
            #print("chunk overlap:", chunk_overlap)
            return chunk_size, chunk_overlap
        
        elif total_lenght < 400_000:
            chunk_size = int(base_chunk*2)
            chunk_overlap = base_overlap*0.8
            
            #print("chunk:", chunk_size)
            #print("chunk overlap:", chunk_overlap)
            return chunk_size, chunk_overlap
        
        elif total_lenght < 1_000_000:
            chunk_size = int(base_chunk*3)
            chunk_overlap = base_overlap*0.7
            
            #print("chunk:", chunk_size)
            #print("chunk overlap:", chunk_overlap)
            return chunk_size, chunk_overlap
        
        else:
            chunk_size = int(base_chunk*4)
            chunk_overlap = base_overlap*0.6
            
            #print("chunk:", chunk_size)
            #print("chunk overlap:", chunk_overlap)
            return chunk_size, chunk_overlap
    
    
    
    def clean_epub_text(self, html_content: str) -> str:
        """Collapse EPUB HTML into coherent paragraphs."""
        soup = BeautifulSoup(html_content, 'html.parser')
    
        # Remove scripts, styles, footnotes
        for tag in soup(['script', 'style', 'aside', 'footer']):
            tag.decompose()
    
        # Get paragraphs/divs as block text
        blocks = []
        for block in soup.find_all(['p', 'div']):
            txt = block.get_text(" ", strip=True)
            if txt and len(txt) > 20:  # skip footnotes or tiny fragments
                blocks.append(txt)
    
        # Join with double newlines for paragraph separation
        return "\n".join(blocks)    
    
    
    # --- Utility functions ---

    
    

    def detect_section_type(self, chunks):
        
        return "content"*len(chunks)
    

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
    
        # Step 1: build full_text + record page spans
        all_text = []
        page_spans = []  # [(start_char, end_char, page_num)]
        cursor = 0
    
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if not text.strip():
                continue
    
            text = text.strip() + "    "  # add separator so pages don’t merge weirdly
            start = cursor
            end = cursor + len(text)
            page_spans.append((start, end, page_num))
            all_text.append(text)
            cursor = end
    
        full_text = "".join(all_text)
        chunk_size, overlap_ratio = self.get_dynamic_chunk_size(len(full_text))
        bs = int(RAG_BATCH_SIZE * (768/ chunk_size))
        #print("Batch size: ",bs)
    
        # Step 2: chunk full_text (keeps multi-page paragraphs intact)
        chunks = Functions.chunk_text(full_text, chunk_size=chunk_size, overlap_ratio = overlap_ratio)
        chunks = [f"[Source: {pdf_name}] {c}" for c in chunks]
        
        # Step 3: assign page numbers to chunks
        cursor = 0
        chunks_pages = []
        for chunk in chunks:
            start = full_text.find(chunk, cursor)  # find chunk in text
            if start == -1:
                start = cursor  # fallback if repeated text
            end = start + len(chunk)
            cursor = end
    
            # find pages overlapping this chunk
            chunk_pages = [p for (s, e, p) in page_spans if not (e < start or s > end)]
            try:
                chunk_page = chunk_pages[0]
            except:
                chunk_page = None
            chunks_pages.append(chunk_page)
    
    
        section_types = self.detect_section_type(chunks)
        self._add_chunks(source_name = pdf_name,
                         chunk_list = chunks,
                         section_type_list = section_types,
                         page_num_list = chunks_pages,
                         use_fp16 = True, 
                         batch_size = bs
                         )


    

    def add_epub(self, epub_path):
        epub_name = os.path.basename(epub_path)
        if self.file_already_ingested(epub_name):
            return
        print(f"Processing EPUB: {epub_name}")
    
        try:
            book = epub.read_epub(epub_path)
            book_items = [item for item in book.get_items() if isinstance(item, epub.EpubHtml)]
    
            # Collect all chunks and metadata before adding
            all_chunks, all_section_nums, all_section_types = [], [], []
            
            # Compute total length by clean_epub_text func
            total_length = 0
            for item in book_items:
                text = self.clean_epub_text(item.get_content())
                total_length += len(text)
            
            chunk_size, overlap_ratio = self.get_dynamic_chunk_size(total_length)
            
            bs = int(RAG_BATCH_SIZE * (1024/ chunk_size))
            #print("Batch size: ",bs)
            
            for section_num, item in enumerate(tqdm(book_items, desc="EPUB sections"), start=1):

                text = self.clean_epub_text(item.get_content())
                if not text:
                    continue
                    

                chunks = Functions.chunk_text(text, chunk_size = chunk_size, overlap_ratio = overlap_ratio)
                section_type = "content"
                chunks = [f"[Source: {epub_name}] {c}" for c in chunks]
                
                # Append to batch lists
                all_chunks.extend(chunks)
                all_section_nums.extend([section_num] * len(chunks))  # each chunk knows its section
                all_section_types.extend([section_type] * len(chunks))

            # Ship all chunks at once to _add_chunks (batch processing)
            if all_chunks:
                self._add_chunks(source_name = epub_name,
                                 chunk_list = all_chunks,
                                 section_type_list = all_section_types,
                                 page_num_list = all_section_nums,
                                 use_fp16 = True, 
                                 batch_size = bs
                                 )

    
        except Exception as e:
            print(f"Failed to parse EPUB {epub_name}: {e}")
            


    def add_file(self, file_path):
        ext = file_path.lower().split('.')[-1]
        if ext == "pdf":
            self.add_pdf(file_path)
        elif ext == "docx":
            #self.add_docx(file_path)
            print("Docx not supported yet")
        elif ext == "txt":
            #self.add_txt(file_path)
            print("Txt not supported yet")
        elif ext == "xml":
            #self.add_xml(file_path)
            print("xml not supported yet")
        elif ext == "epub":
            self.add_epub(file_path)
        elif ext in ("html", "htm"):
            #self.add_html(file_path)
            print("Xml not supported yet")
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
    def _add_chunks(
        self,
        source_name,
        chunk_list,
        section_type_list=None,
        page_num_list=None,
        use_fp16=True,
        batch_size = 16,
        chroma_max_batch=5000,
    ):
        """
        True streaming ingestion:
        - Encodes batches on GPU
        - Immediately flushes to Chroma in sub-batches
        - No vstack or large CPU buffers -> prevents end-of-process memory spike
        """
    
        section_type_list = section_type_list or [None] * len(chunk_list)
        page_num_list = page_num_list or [None] * len(chunk_list)
    
        pbar = tqdm(total=len(chunk_list), desc="Embedding chunks", unit="chunks")
    
        for start_idx in range(0, len(chunk_list), batch_size):
            batch = chunk_list[start_idx:start_idx + batch_size]
            batch_section = section_type_list[start_idx:start_idx + batch_size]
            batch_pages = page_num_list[start_idx:start_idx + batch_size]
    
            # --- Encode batch ---
            if use_fp16:
                with torch.amp.autocast("cuda"):
                    emb_gpu = self.embedder.encode(
                        batch, convert_to_tensor=True, normalize_embeddings=True, device="cuda"
                    )
                
            else:
                emb_gpu = self.embedder.encode(
                    batch, convert_to_tensor=True, normalize_embeddings=True, device="cuda"
                )
            emb_gpu = emb_gpu.to(torch.float32)
            emb_cpu = emb_gpu.cpu().numpy()
            del emb_gpu
            torch.cuda.empty_cache()
    
            # --- Build IDs and metadata ---
            batch_ids, batch_meta = [], []
            for j, _ in enumerate(batch):
                uid = f"{source_name}_chunk{start_idx + j}" + (f"_p{batch_pages[j]}" if batch_pages[j] else "")
                meta = {"source": source_name}
                if batch_section[j]:
                    meta["section_type"] = batch_section[j]
                if batch_pages[j]:
                    meta["page"] = batch_pages[j]
                batch_ids.append(uid)
                batch_meta.append(meta)
    
            # --- Flush in sub-batches straight to Chroma ---
            for sub_start in range(0, len(batch), chroma_max_batch):
                sub_end = sub_start + chroma_max_batch
                self.collection.add(
                    documents=batch[sub_start:sub_end],
                    metadatas=batch_meta[sub_start:sub_end],
                    ids=batch_ids[sub_start:sub_end],
                    embeddings=emb_cpu[sub_start:sub_end],
                )
    
            # Free batch memory before next one
            b_update = len(batch)
            del emb_cpu, batch, batch_ids, batch_meta
            gc.collect()
    
            pbar.update(b_update)
    
        pbar.close()
            
            
            

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


    def truncate_doc_for_reranker(self, doc_text, cross_enc_model, max_len=2048):
        try:
            tokenizer = cross_enc_model.tokenizer
            tokens = tokenizer(doc_text, truncation=True, max_length=max_len, return_tensors="pt")
            truncated = tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
            return truncated
        except Exception:
            return doc_text[: max_len * 4]


    def search(self, query, top_n=3, absolute_cosine_min=0.1, min_relevance=0.7, verbose = False):
        # --- Encode query ---
        query_embedding = self.embedder.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device='cuda'  # or 'cpu'
        )
        query_embedding = query_embedding.to(torch.float32)
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
        if verbose:
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

        # --- Normalize to 0-1 using sigmoid --- (Use with Mini LM12 rearnker)
        rerank_scores_norm = rerank_scores
        #rerank_scores_norm = 1 / (1 + np.exp(-rerank_scores))

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
        if verbose:
            print("Best rerank score: ", max_rerank_sim)
        
        # --- Return top_n of relevant docs ---
        return top_docs
