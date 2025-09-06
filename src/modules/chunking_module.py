from typing import List, Optional

class ChunkingError(Exception):
    """Custom exception for chunking errors."""
    pass

def _encode_len(text: str, tokenizer, add_special_tokens: bool = False) -> int:
    """
    Returns token count for text using the given tokenizer.
    Falls back to word count when tokenizer is None or unsupported.
    """
    if tokenizer is None:
        return len((text or "").split())
    try:
        # Prefer disabling special tokens for length accounting
        return len(tokenizer.encode(text, add_special_tokens=add_special_tokens))
    except TypeError:
        # Older tokenizers may not support add_special_tokens kwarg
        return len(tokenizer.encode(text))

def _max_tokens_for(tokenizer, default: int = 77, reserve_special: int = 2) -> int:
    """
    Determine effective chunk size. Reserves a couple tokens for BOS/EOS.
    """
    try:
        model_max = int(getattr(tokenizer, "model_max_length", default) or default)
    except Exception:
        model_max = default
    # Allow env override when module used standalone
    import os
    override = int(os.getenv("CHUNK_MAX_TOKENS", "0") or "0")
    if override > 0:
        model_max = override
    eff = max(8, model_max - int(reserve_special))
    return eff

def chunk_text(
    text: str,
    max_tokens: Optional[int] = None,
    tokenizer=None
) -> List[str]:
    """
    Splits input text into chunks that do not exceed max_tokens.
    If tokenizer provided, uses token lengths; otherwise uses words.

    Returns:
        List[str]: List of chunk strings (order preserved).
    """
    text = text or ""
    if not text.strip():
        return [""]

    eff_max = max_tokens if (isinstance(max_tokens, int) and max_tokens > 0) else _max_tokens_for(tokenizer)
    if tokenizer is not None:
        # Token-based chunking
        try:
            toks = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            toks = tokenizer.encode(text)
        chunks = []
        for i in range(0, len(toks), eff_max):
            chunk_toks = toks[i:i+eff_max]
            try:
                chunk = tokenizer.decode(chunk_toks, skip_special_tokens=True).strip()
            except TypeError:
                # Older tokenizers may not support skip_special_tokens
                chunk = tokenizer.decode(chunk_toks).strip()
            if chunk:
                chunks.append(chunk)
        return chunks if chunks else [text.strip()]
    else:
        # Fallback: naive word-based chunking
        words = (text or "").split()
        if not words:
            return [text]
        chunks = []
        for i in range(0, len(words), eff_max):
            chunk = " ".join(words[i:i+eff_max]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks if chunks else [text.strip()]

def is_within_token_limit(
    text: str,
    max_tokens: Optional[int] = None,
    tokenizer=None
) -> bool:
    """
    Checks if the text is within the token limit (excludes special tokens).
    """
    eff_max = max_tokens if (isinstance(max_tokens, int) and max_tokens > 0) else _max_tokens_for(tokenizer)
    return _encode_len(text or "", tokenizer, add_special_tokens=False) <= eff_max

if __name__ == "__main__":
    sample_text = "This is a very long prompt that might exceed the maximum token limit for the model."
    chunks = chunk_text(sample_text, max_tokens=10, tokenizer=None)
    print("Chunks:", chunks)
