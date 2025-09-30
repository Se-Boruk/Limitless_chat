import requests
from bs4 import BeautifulSoup
import random
import time
import urllib.parse
import re
import nltk
from urllib.parse import urlparse
from torch.amp import autocast
import torch
import numpy as np
from stem.process import launch_tor_with_config

try:
    import trafilatura
except Exception:
    trafilatura = None

try:
    from readability import Document
except Exception:
    Document = None

#####################################

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

def emergency_chat_prompt(messages, add_generation_prompt=True) -> str:
    """
    Build a ChatML-formatted prompt from a list of messages.

    Args:
        messages (list): List of dicts like [{'role': 'system'|'user'|'assistant', 'content': str}, ...]
        add_generation_prompt (bool): Whether to append the assistant preamble at the end for generation

    Returns:
        str: Formatted ChatML string for prompting the model
    """
    prompt = ""
    for message in messages:
        role = message['role']
        content = message['content'].strip()
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    
    if add_generation_prompt:
        prompt += "<|im_start|>assistant\n"
    
    RAG_response = (False, "None")
    return prompt, RAG_response



def local_rag_chat_prompt(tokenizer, messages, vector_lib, top_n=3, min_relevance = 0.75, absolute_cosine_min = 0.1, add_generation_prompt=True):
    
    
    RAG_response = (False, "None")
    user_messages = [m['content'] for m in messages if m['role'] == 'user']
    
    query = user_messages[-1]
    rag_results = vector_lib.search(query,
                                    top_n = top_n,
                                    absolute_cosine_min = absolute_cosine_min,
                                    min_relevance = min_relevance,
                                    verbose = True
                                    )
    

    if rag_results:
        # filter relevant
        relevant_chunks = [chunk for chunk, score in rag_results]
        if relevant_chunks:
            _, max_sim = max(rag_results, key=lambda x: x[1])
            RAG_response = (True, max_sim)
            
            context_text = "\n\n".join(relevant_chunks)

            # minimal rag-enriched history
            rag_history = [
                {"role": "system", "content": f"The following context may be useful:\n{context_text}"},
                {"role": "user", "content": query}
            ]

            prompt = tokenizer.apply_chat_template(
                rag_history,
                tokenize=False,
                add_generation_prompt=True
            )
            return prompt, RAG_response
        
        else:
            RAG_response = (False, "None")
            

    # if RAG enabled but no chunks found, just fall back to normal response
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt, RAG_response


def online_rag_chat_prompt(tokenizer, queries, messages, vector_lib, top_n=3, min_relevance = 0.75, absolute_cosine_min = 0.1, add_generation_prompt=True, TOR_search = False, TOR_server_on = False):
    
    #Inputs for both online and offline search
    RAG_response = (False, "None")
    user_messages = [m['content'] for m in messages if m['role'] == 'user']
    
    query = user_messages[-1]

    #Online RAG
    #####################################################
    #1 take and filter queries to search
    queries_separated = []
    for q in queries:
        try:
            _, q_sep = q.split(". ", 1)
        except:
            q_sep = q
            
        queries_separated.append(q_sep)
        

    #2 Send duckduckgo request
    if TOR_search:
        if not TOR_server_on:
            print("TOR server is not online. Turn it on to enable TOR search")
            return
        
    rag_results_online = scrape_queries_to_dict(queries_separated, max_sites_per_query=2, verbose=True, tor = TOR_search)
    
    #3 Chunk it
    chunk_list = []

    for url, text in rag_results_online.items():
        chunks = chunk_text(text)
        
        domain = urlparse(url).netloc  # extracts website domain
        domain = domain[:200]
        chunks = [f"[Source: {domain}] {c}" for c in chunks]
        
        chunk_list.extend(chunks)
        
    #5 Rerank the chunks found online
    rerank_pairs = []
    original_chunks = []
    for c in chunk_list:
        truncated = vector_lib.truncate_doc_for_reranker(c, vector_lib.cross_encoder)
        rerank_pairs.append([query, truncated])
        #In case chunk was cutted
        original_chunks.append(c)
        
    with autocast(device_type='cuda', dtype=torch.float16):
        rerank_scores = vector_lib.cross_encoder.predict(rerank_pairs, show_progress_bar=True)
        
    rerank_scores = np.array(rerank_scores, dtype=np.float32)
    rerank_scores_norm = rerank_scores
    ##########
    # --- Zip rerank scores with original docs ---
    reranked_docs = [(doc, float(score)) for doc, score in zip(original_chunks, rerank_scores_norm)]

    # --- Filter by minimum relevance ---
    relevant_online_chunks = [d for d in reranked_docs if d[1] >= min_relevance]
    

    #Offline RAG
    ##################
    relevant_offline_chunks = vector_lib.search(query,
                                    top_n = top_n,
                                    absolute_cosine_min = absolute_cosine_min,
                                    min_relevance = min_relevance
                                    )

    rag_results = relevant_online_chunks + relevant_offline_chunks
    
    if rag_results:
        # --- Sort descending by reranker score ---
        rag_results = sorted(rag_results, key=lambda x: x[1], reverse=True)

        rag_results = rag_results[:top_n*2]
        _, max_rerank_sim = max(rag_results, key=lambda x: x[1])
        print("Best rerank score (offline + online): ", max_rerank_sim)
        

        # filter relevant
        relevant_chunks = [chunk for chunk, score in rag_results]
        if relevant_chunks:
            _, max_sim = max(rag_results, key=lambda x: x[1])
            RAG_response = (True, max_sim)
            
            context_text = "\n\n".join(relevant_chunks)

            # minimal rag-enriched history
            rag_history = [
                {"role": "system", "content": f"The following context may be useful:\n{context_text}"},
                {"role": "user", "content": query}
            ]

            prompt = tokenizer.apply_chat_template(
                rag_history,
                tokenize=False,
                add_generation_prompt=True
            )
            return prompt, RAG_response
        
        else:
            RAG_response = (False, "None")
            

    # if RAG enabled but no chunks found, just fall back to normal response
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt, RAG_response

#Http scrapper functions
##################################################################################


def is_ad_url(url: str) -> bool:
    """
    Heuristic ad/sponsored/affiliate/tracking detector.
    Immediately blocks DuckDuckGo y.js ad_domain URLs.
    Returns True if the URL likely points to an ad/affiliate/tracking redirect or landing page.
    """
    if not url:
        return False

    url_s = url.strip()
    url_lower = url_s.lower()

    # ---------- HARD BLOCK ----------
    if url_lower.startswith("https://duckduckgo.com/y.js?ad_domain"):
        return True

    # quick sanity: only HTTP(S) URLs considered
    if not (url_lower.startswith("http://") or url_lower.startswith("https://")):
        return False

    try:
        from urllib.parse import urlparse, parse_qs, unquote_plus
        import re
    except Exception:
        return True  # if environment broken, better safe than sorry

    parsed = urlparse(url_s)
    netloc = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    query = parsed.query or ""
    qs = parse_qs(query)

    score = 0

    # ---------- Strong, explicit redirector/ad click handlers ----------
    strong_redirectors = (
        "duckduckgo.com/l/",
        "bing.com/aclick",
        "google.com/aclk",
        "googleadservices.com",
        "doubleclick.net",
        "pagead2.googlesyndication",
    )
    for sig in strong_redirectors:
        if sig in url_lower:
            score += 10  # high confidence

    # ---------- Ad / tracking query keys ----------
    ad_keys = {
        "ad_domain", "ad_provider", "ad_type", "click_metadata",
        "rut", "u3", "msclkid", "gclid", "fbclid",
        "utm_source", "utm_medium", "utm_campaign", "utm_term",
        "campaignid", "adid", "placementid", "creativeid",
        "affid", "aff_id", "affiliate", "aff", "tag", "ref", "refid", "rid",
    }
    present_ad_keys = set(qs.keys()) & ad_keys
    if present_ad_keys:
        score += 3 + min(len(present_ad_keys), 5)
        if "utm_medium" in qs:
            vals = [v.lower() for v in qs.get("utm_medium", []) if isinstance(v, str)]
            if any(x in "|".join(vals) for x in ("cpc", "ppc", "paid", "affiliate", "display")):
                score += 3

    # ---------- Embedded/double-encoded redirect targets ----------
    embedded_params = ("u", "u3", "url", "redirect", "target", "uddg", "rurl", "uinfo")
    for p in embedded_params:
        if p in qs:
            for raw in qs.get(p, []):
                decoded = unquote_plus(raw)
                if "http://" in decoded or "https://" in decoded or ("%3a%2f%2f" in raw.lower()):
                    score += 4
                    if any(x in decoded.lower() for x in ("amazon.", "tripadvisor.", "tiqets.", "booking.", "aff", "partner", "affiliate")):
                        score += 2

    # ---------- Netloc / hostname heuristics ----------
    netloc_ad_indicators = ("adservice", "adserver", "ads.", ".ads.", "tracking", "track.", "click", "clicks.", "clickserve", "affiliate", "affiliates", "partner", "partners", "sponsored", "promo", "promotions")
    if any(tok in netloc for tok in netloc_ad_indicators):
        score += 4

    ecommerce_hosts = ("amazon.", "ebay.", "aliexpress.", "etsy.", "booking.", "tripadvisor.", "expedia.", "airbnb.")
    if any(e in netloc for e in ecommerce_hosts):
        score += 2

    # ---------- Path heuristics ----------
    path_flags = ("/ad", "/ads", "/adclick", "/aclick", "/clk", "/sponsored", "/sponsor", "/affiliate", "/promo", "/promotions", "/aff")
    if any(pf in path for pf in path_flags):
        score += 3

    # ---------- Many params or very long query → tracking chain likely ----------
    if len(qs) >= 8 or len(query) > 180:
        score += 2

    # ---------- Generic keyword scan ----------
    generic_ad_keywords = ("advert", "ads", "sponsored", "promotion", "affiliate", "ref=", "affid", "msclkid")
    if any(kw in url_lower for kw in generic_ad_keywords):
        score += 1

    # ---------- Weak multi-signal heuristics ----------
    weak_signals = 0
    if any(k in netloc for k in ("amazon.", "booking.", "tripadvisor.", "tickets-", "tiqets.")):
        weak_signals += 1
    if any(k in path for k in ("/search", "/s/", "/products", "/dp/")):
        weak_signals += 1
    if re.search(r"[?&](utm_|gclid=|fbclid=|msclkid=)", url_lower):
        weak_signals += 1
    if weak_signals >= 2:
        score += 2

    # ---------- Final decision ----------
    return score >= 6



def safe_get(url, params=None, retries=5, timeout=15):
    """GET with retries, random UA; returns (html or None, final_url)"""
    
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Mozilla/5.0 (X11; Linux x86_64)",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)",
    ]
    sess = requests.Session()
    sess.headers.update({
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
    })
    
    for attempt in range(retries):
        try:
            sess.headers["User-Agent"] = random.choice(USER_AGENTS)
            r = sess.get(url, params=params, timeout=timeout, allow_redirects=True)
            
            if is_ad_url(r.url):
                time.sleep((2 ** attempt) + random.random())
                continue
            
            if r.status_code == 200:
                return r.text, r.url
            if r.status_code in (429, 403):
                time.sleep((2 ** attempt) + random.random())
                continue
            r.raise_for_status()
        except Exception:
            time.sleep((1.5 ** attempt) + random.random())
    return None, url


#get Tor users
def get_tor_user_agents():
    """Fetch user agents from online source or fallback to local list"""
    try:
        resp = requests.get("https://useragents.io/random?limit=100", timeout=10)
        resp.raise_for_status()
        user_agents = resp.json()
        return [ua['user_agent'] for ua in user_agents if 'user_agent' in ua]
    except Exception:
        # fallback local list
        return [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:54.0) Gecko/20100101 Firefox/54.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Edge/18.18363",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
            # ... add more entries to reach 30+
        ]

# ---------- Tor-safe GET ----------
def safe_get_tor(url, params=None, retries=5, timeout=15):
    """GET through Tor with retries, random UA; returns (html or None, final_url)"""
    USER_AGENTS = get_tor_user_agents()
    sess = requests.Session()
    sess.headers.update({
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
    })
    proxies = {
        "https": "socks5h://127.0.0.1:9050"
    }

    for attempt in range(retries):
        if not USER_AGENTS:
            USER_AGENTS = get_tor_user_agents()  # refresh if exhausted
        ua = random.choice(USER_AGENTS)
        USER_AGENTS.remove(ua)  # ensure each UA used only once per session
        sess.headers["User-Agent"] = ua

        try:
            r = sess.get(url, params=params, timeout=timeout,
                         allow_redirects=True, proxies=proxies)
            
            if is_ad_url(r.url):
                time.sleep((2 ** attempt) + random.random())
                continue
            
            if r.status_code == 200:
                return r.text, r.url
            if r.status_code in (403, 429):
                time.sleep((2 ** attempt) + random.random())
                continue
            r.raise_for_status()
        except Exception:
            time.sleep((1.5 ** attempt) + random.random())
    return None, url


def _word_count(t: str):
    return len(re.findall(r'\w+', t or ""))


def _decode_ddg_href(href: str):
    """Decode DuckDuckGo /l/?...&uddg= encoded links to real URLs"""
    DUCK_URL = "https://duckduckgo.com/html/"
    
    if not href:
        return href
    if href.startswith("http://") or href.startswith("https://"):
        return href
    parsed = urllib.parse.urlparse(href)
    qs = urllib.parse.parse_qs(parsed.query)
    if "uddg" in qs:
        return urllib.parse.unquote(qs["uddg"][0])
    return urllib.parse.urljoin(DUCK_URL, href)


def extract_main_text(html: str, min_block_words=40, top_blocks=2):
    """
    Heuristic extractor (prefers trafilatura/readability if available).
    Returns cleaned main text or empty string.
    """
    if not html:
        return ""

    # 1) trafilatura (preferred)
    if trafilatura is not None:
        try:
            t = trafilatura.extract(html, include_comments=False, include_tables=False)
            if t and len(t.strip()) >= 50:
                return t.strip()
        except Exception:
            pass

    # 2) readability (next)
    if Document is not None:
        try:
            doc = Document(html)
            summary_html = doc.summary() or ""
            if summary_html:
                s = BeautifulSoup(summary_html, "html.parser")
                plain = s.get_text("\n", strip=True)
                if plain and len(plain.strip()) >= 50:
                    return plain.strip()
        except Exception:
            pass

    # 3) aggressive heuristic cleaning
    soup = BeautifulSoup(html, "html.parser")

    # remove common UI elements
    selectors_to_remove = [
        "script", "style", "noscript", "svg", "iframe", "input", "button",
        "form", "header", "footer", "nav", "aside", "link", "meta"
    ]
    for sel in selectors_to_remove:
        for node in soup.select(sel):
            try:
                node.decompose()
            except Exception:
                pass

    for node in soup.select("[role=navigation], [role=complementary], [role=banner], [role=search], [role=contentinfo]"):
        try:
            node.decompose()
        except Exception:
            pass

    for node in soup.select(".share, .social, .newsletter, .promo, .cookie, .advert, .ads, .breadcrumb, .sidebar, .related, .comments"):
        try:
            node.decompose()
        except Exception:
            pass

    def link_density(el):
        text = el.get_text(" ", strip=True)
        if not text:
            return 1.0
        link_text = " ".join(a.get_text(" ", strip=True) for a in el.find_all("a"))
        return float(_word_count(link_text)) / max(1, _word_count(text))

    BOILERPLATE_PHRASES = [
        "subscribe", "follow us", "share this", "newsletter", "sign up",
        "cookie", "cookies", "privacy policy", "terms of service", "related posts",
        "read more", "advertisement", "ads", "©", "all rights reserved", "published in",
        "click here", "sponsored"
    ]

    def looks_like_boilerplate(text):
        if not text:
            return True
        t = text.lower()
        if _word_count(t) < 8:
            return True
        for p in BOILERPLATE_PHRASES:
            if p in t:
                return True
        return False

    candidates = []

    # semantic candidates first
    for el in soup.find_all(["article", "main"]):
        txt = el.get_text("\n", strip=True)
        if txt and _word_count(txt) >= min_block_words and link_density(el) < 0.6:
            candidates.append((el, txt))

    # then large div/section nodes
    for el in soup.find_all(["section", "div"], recursive=True):
        try:
            txt = el.get_text(" ", strip=True)
        except Exception:
            continue
        if not txt or len(txt) < 80:
            continue
        ld = link_density(el)
        wc = _word_count(txt)
        if ld > 0.45 or wc < min_block_words:
            continue
        candidates.append((el, txt))

    if not candidates:
        body = soup.body or soup
        txt = body.get_text("\n", strip=True)
        if not txt:
            return ""
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        filtered = [ln for ln in lines if not looks_like_boilerplate(ln)]
        return "\n\n".join(filtered).strip()

    # pick top blocks by word count
    candidates_sorted = sorted(candidates, key=lambda c: _word_count(c[1]), reverse=True)
    top = candidates_sorted[:top_blocks]

    final_blocks = []
    for el, txt in top:
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        lines = [ln for ln in lines if not looks_like_boilerplate(ln)]
        block_text = "\n".join(lines).strip()
        if _word_count(block_text) >= min_block_words:
            final_blocks.append(block_text)

    if not final_blocks:
        longest = max((c[1] for c in candidates), key=lambda t: _word_count(t), default="")
        return longest.strip()

    return "\n\n".join(final_blocks).strip()



def scrape_queries_to_dict(queries, max_sites_per_query=5, sleep_between_requests=0.5,
                           drop_empty=True, verbose=False, tor=False):

    per_url = {}
    seen = set()
    DUCK_URL = "https://duckduckgo.com/html/"

    # #############  SEARCH PHASE #############
    if verbose:
        print("SEARCH PHASE: running DuckDuckGo queries")

    for q in queries:
        if verbose:
            print("[search]", q)
        # fetch SERP HTML
        if tor:
            html, _ = safe_get_tor(DUCK_URL, params={"q": q})
        else:
            html, _ = safe_get(DUCK_URL, params={"q": q})

        if not html:
            if verbose:
                print("  search failed for:", q)
            continue

        soup = BeautifulSoup(html, "html.parser")
        links = []
        for a in soup.select(".result__body .result__a, .result__a")[:max_sites_per_query]:
            href = a.get("href") or a.get("data-href") or ""
            final = _decode_ddg_href(href)
            if final:
                links.append(final)
        if not links:
            for a in soup.select("a")[:max_sites_per_query]:
                final = _decode_ddg_href(a.get("href") or "")
                if final and final.startswith("http"):
                    links.append(final)
                    if len(links) >= max_sites_per_query:
                        break

        # #############  FETCH & EXTRACT PHASE #############
        for link in links:
            url_norm = (link or "").strip()
            if not url_norm or url_norm in seen:
                continue
            seen.add(url_norm)

            if verbose:
                print("  fetching:", url_norm)
            if tor:
                html_page, final_url = safe_get_tor(url_norm)
            else:
                html_page, final_url = safe_get(url_norm)

            text = ""
            if html_page:
                text = extract_main_text(html_page)
                # second-pass if extraction tiny and redirect occurred
                if (not text or len(text) < 50) and final_url != url_norm:
                    if tor:
                        html2, final_url2 = safe_get_tor(final_url)
                    else:
                        html2, final_url2 = safe_get(final_url)

                    if html2:
                        t2 = extract_main_text(html2)
                        if t2 and len(t2) > len(text):
                            text = t2
                            url_norm = final_url2

            per_url[url_norm] = text or ""
            time.sleep(sleep_between_requests + random.random() * 0.4)

    # #############  FINALIZE #############
    if drop_empty:
        per_url = {u: t for u, t in per_url.items() if t and len(t) >= 20}

    return per_url
##################################################################################










