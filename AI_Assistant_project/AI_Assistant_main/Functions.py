

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



def native_chat_prompt_rag(tokenizer, messages, vector_lib, top_n=3, min_relevance = 0.75, absolute_cosine_min = 0.1, add_generation_prompt=True):
    
    
    RAG_response = (False, "None")
    user_messages = [m['content'] for m in messages if m['role'] == 'user']
    
    query = user_messages[-1]
    rag_results = vector_lib.search(query,
                                    top_n = top_n,
                                    absolute_cosine_min = absolute_cosine_min,
                                    min_relevance = min_relevance
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




