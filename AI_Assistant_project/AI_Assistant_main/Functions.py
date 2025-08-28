

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


def emergency_chat_prompt_rag(messages, vector_lib, top_n=3, min_similarity=0.75, add_generation_prompt=True) -> str:
    """
    Build ChatML-formatted prompt using conversation history and RAG search on the latest user query.

    Args:
        messages (list): Chat history [{'role': 'system'|'user'|'assistant', 'content': str}, ...]
        vector_lib: Your PDFVectorStore-like object with .search(query, top_k) -> [(chunk, score), ...]
        top_n (int): Max number of chunks to inject.
        min_similarity (float): Similarity threshold for including chunks.
        add_generation_prompt (bool): Append assistant preamble at the end.

    Returns:
        str: ChatML prompt with relevant RAG context prepended.
    """
    prompt = ""

    # Take the latest user message as the retrieval query
    user_messages = [m['content'] for m in messages if m['role'] == 'user']
    if user_messages:
        query = user_messages[-1]
        rag_results = vector_lib.search(query, top_k=top_n)

        _, max_sim = max(rag_results, key=lambda x: x[1])
        
        # Filter and select chunks
        relevant_chunks = [chunk for chunk, score in rag_results if score >= min_similarity]
        if relevant_chunks:
            RAG_response = (True, max_sim)
            context_text = "\n\n".join(relevant_chunks[:top_n])
            prompt += f"<|im_start|>system\nThe following context may be useful:\n{context_text}\n<|im_end|>\n"
        else:
            RAG_response = (False, max_sim)
    # Add the conversation messages
    for message in messages:
        role = message['role']
        content = message['content'].strip()
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    if add_generation_prompt:
        prompt += "<|im_start|>assistant\n"

    return prompt, RAG_response




def native_chat_prompt_rag(tokenizer, messages, vector_lib, top_n=3, min_similarity=0.75, add_generation_prompt=True):
    
    
    RAG_response = (False, "None")
    user_messages = [m['content'] for m in messages if m['role'] == 'user']
    
    query = user_messages[-1]
    rag_results = vector_lib.search(query, top_k=top_n)
    
    _, max_sim = max(rag_results, key=lambda x: x[1])
    

    if rag_results:
        # filter relevant
        relevant_chunks = [chunk for chunk, score in rag_results if score >= min_similarity]
        if relevant_chunks:
            _, max_sim = max(rag_results, key=lambda x: x[1])
            RAG_response = (True, max_sim)
            context_text = "\n\n".join(relevant_chunks[:top_n])

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
            RAG_response = (False, max_sim)
            


    # if RAG enabled but no chunks found, just fall back to normal response
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt, RAG_response




