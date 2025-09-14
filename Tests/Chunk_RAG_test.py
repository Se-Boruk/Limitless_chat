import os
import fitz
import re
import nltk
from bs4 import BeautifulSoup
from tqdm import tqdm
from ebooklib import epub


#nltk.download("punkt", quiet=True)
#nltk.download('punkt_tab')
chunk_size = 256
chunk_overlap = 128
pdf_path = r"C:\Users\Stacja_Robocza\Desktop\Local_AI_assistant\AI_Assistant_project\Offline_lib_doc\Urlich_von_jungingen_summary.pdf"
pdf_path = r"C:\Users\Stacja_Robocza\Desktop\Local_AI_assistant\AI_Assistant_project\Offline_lib_doc\minecraft-construction-for-dummies-portable-edition.pdf"

#pdf_path = r"C:\Users\Stacja_Robocza\Desktop\Local_AI_assistant\AI_Assistant_project\Offline_lib_doc\Attention_GAN Unpaired Image to ImageTranslation using Attention-Guided GenerativeAdversarial Networks.pdf"

epub_path = r"C:\Users\Stacja_Robocza\Desktop\Local_AI_assistant\AI_Assistant_project\Offline_lib_doc\krzyzacy.epub"




def clean_epub_text(html_content: str) -> str:
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



def chunk_text(text, chunk_size=256, overlap_ratio=0.25):
    """
    Paragraph/sentence-aware chunking with dynamic overlap.
    - Uses NLTK for sentence splitting
    - Keeps paragraphs intact where possible
    - Handles long sentences safely
    """
    chunks = []
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n\s*\n', '\n\n', text)
    paragraphs = re.split(r'\n{2,}', text)

    prev_sentences = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        sentences = nltk.sent_tokenize(para)

        # paragraph overlap
        overlap_count = max(1, int(len(prev_sentences) * overlap_ratio)) if prev_sentences else 0
        sentences = prev_sentences[-overlap_count:] + sentences if overlap_count else sentences

        cur_words, cur_sents = [], []

        for sent in sentences:
            words = sent.split()

            # hard-split overlong sentences
            if len(words) > chunk_size:
                if cur_words:
                    chunks.append(" ".join(cur_words))
                    cur_words, cur_sents = [], []
                for i in range(0, len(words), chunk_size):
                    chunks.append(" ".join(words[i:i+chunk_size]))
                continue

            # flush if over limit
            if len(cur_words) + len(words) > chunk_size:
                chunks.append(" ".join(cur_words))
                overlap_count_chunk = max(1, int(len(cur_sents) * overlap_ratio))
                cur_words = []
                for s in cur_sents[-overlap_count_chunk:]:
                    cur_words.extend(s.split())
                cur_sents = cur_sents[-overlap_count_chunk:]

            cur_words.extend(words)
            cur_sents.append(sent)

        if cur_words:
            chunks.append(" ".join(cur_words))

        prev_sentences = sentences

    return chunks


def get_dynamic_chunk_size(total_lenght, base_chunk = 256, base_overlap = 0.25):
    """
    Choose chunk size dynamically based on document length.
    - short docs → smaller chunks for precision
    - long docs → bigger chunks for context
    """

    print("Length: ",total_lenght)
    if total_lenght < 35_000:
        chunk_size = base_chunk
        chunk_overlap = base_overlap
        
        print("chunk:", chunk_size)
        print("chunk overlap:", chunk_overlap)
        return chunk_size, chunk_overlap
    
    elif total_lenght < 400_000:
        chunk_size = int(base_chunk*2)
        chunk_overlap = base_overlap*0.8
        
        print("chunk:", chunk_size)
        print("chunk overlap:", chunk_overlap)
        return chunk_size, chunk_overlap
    
    elif total_lenght < 1_000_000:
        chunk_size = int(base_chunk*3)
        chunk_overlap = base_overlap*0.7
        
        print("chunk:", chunk_size)
        print("chunk overlap:", chunk_overlap)
        return chunk_size, chunk_overlap
    
    else:
        chunk_size = int(base_chunk*4)
        chunk_overlap = base_overlap*0.6
        
        print("chunk:", chunk_size)
        print("chunk overlap:", chunk_overlap)
        return chunk_size, chunk_overlap
        
    
 

def add_epub(epub_path):
    epub_name = os.path.basename(epub_path)


    try:
        book = epub.read_epub(epub_path)
        book_items = [item for item in book.get_items() if isinstance(item, epub.EpubHtml)]

        # Collect all chunks and metadata before adding
        all_chunks, all_section_nums, all_section_types = [], [], []
        
        # Compute total length by clean_epub_text func
        total_length = 0
        for item in book_items:
            text = clean_epub_text(item.get_content())
            total_length += len(text)
        
        chunk_size, overlap_ratio = get_dynamic_chunk_size(total_length)
        
        for section_num, item in enumerate(tqdm(book_items, desc="EPUB sections"), start=1):

            text = clean_epub_text(item.get_content())
            if not text:
                continue
        
            chunks = chunk_text(text, chunk_size=chunk_size, overlap_ratio=overlap_ratio)

            chunks = [f"[Source: {epub_name} ; Chapter {section_num}] {c}" for c in chunks]
            
            # Append to batch lists
            all_chunks.extend(chunks)


    except Exception as e:
        print(f"Failed to parse EPUB {epub_name}: {e}") 
        
    return all_chunks
    
 
    
 
    
#############################
# --- File ingestion ---
def add_pdf(pdf_path):
    pdf_name = os.path.basename(pdf_path)

    
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
    chunk_size, overlap_ratio = get_dynamic_chunk_size(len(full_text))

    # Step 2: chunk full_text (keeps multi-page paragraphs intact)
    chunks = chunk_text(full_text, chunk_size=chunk_size, overlap_ratio = overlap_ratio)
    chunks = [f"[Source: {pdf_name}] {c}" for c in chunks]
    
    return chunks




chunk_list_pdf = add_pdf(pdf_path)

chunk_list_epub = add_epub(epub_path)
        




        
        
        
        
        
        
        