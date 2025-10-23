# chunker.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging
from tqdm.auto import tqdm

from langchain_core.documents import Document

# shared helpers
from kg_utils import slugify, get_attr as _get

@dataclass(frozen=True)
class DocChunkerConfig:
    # ~1KB target by default (≈256 tokens @ ~4 chars/token)
    target_bytes: int = 1024
    # if None, we’ll try to infer from your OpenAI model; else use this tiktoken encoding
    tiktoken_encoding: Optional[str] = None       # e.g. "o200k_base" or "cl100k_base"
    # when model name is available (from your main), we try encoding_for_model first
    openai_model_hint: Optional[str] = None
    # default overlap as a fraction of chunk_size_tokens
    overlap_frac: float = 0.10
    # logger
    logger: Optional[logging.Logger] = None

class DocChunker:
    """
    Splits state['docs'] into semantically-meaningful chunks with semchunk.
    
    Returns {'docs': chunked_docs} replacing the input doc list.
    """

    def __init__(self, cfg: DocChunkerConfig):
        self.cfg = cfg
        self.logger = cfg.logger or logging.getLogger("doc_chunker")

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # resolve tokenizer/token_counter
        token_counter = self._get_token_counter()

        # size & overlap (allow per-run overrides from state)
        target_bytes = self.cfg.target_bytes
        chunk_size_tokens = max(1, target_bytes // 4)
        overlap_tokens = max(1, int(self.cfg.overlap_frac * chunk_size_tokens))

        # build semchunk chunker
        try:
            import semchunk
        except Exception as e:
            raise RuntimeError(
                "semchunk is required. Install with: pip install semchunk"
            ) from e
        chunker = semchunk.chunkerify(token_counter, chunk_size=chunk_size_tokens)

        docs = state.get("docs") or []
        if not docs:
            return {"docs": []}

        chunked_docs: List[Document] = []
        for doc_idx, doc in enumerate(tqdm(docs, desc="chunking docs", unit="doc")):
            text = getattr(doc, "page_content", "") or ""
            if not text.strip():
                continue

            chunks, offsets = chunker(text, offsets=True, overlap=overlap_tokens)

            parent_id = slugify(f"{_get(doc, 'metadata', {}).get('source', 'doc')}-{doc_idx}")
            base_meta = dict(getattr(doc, "metadata", {}) or {})
            base_meta.setdefault("source", base_meta.get("source", "workflow"))
            base_meta["parent_id"] = parent_id
            base_meta["parent_len"] = len(text)
            base_meta["chunk_size_tokens"] = chunk_size_tokens
            base_meta["chunk_overlap_tokens"] = overlap_tokens

            for i, (chunk, (start, end)) in enumerate(zip(chunks, offsets)):
                md = dict(base_meta)
                md.update({
                    "chunk_index": i,
                    "chunk_start": int(start),
                    "chunk_end": int(end),
                })
                chunked_docs.append(Document(page_content=chunk, metadata=md))

        self.logger.debug(
            "Chunked %d docs -> %d chunks (size=%d tokens, overlap=%d)",
            len(docs), len(chunked_docs), chunk_size_tokens, overlap_tokens
        )
        return {"docs": chunked_docs}

    # internals

    def _get_token_counter(self):
        """
        Prefer a tokenizer matching your LLM (via tiktoken); otherwise fall back
        to a simple ~4 chars/token heuristic.
        """
        # try tiktoken
        try:
            import tiktoken
            enc = None
            # 1) explicit encoding name
            if self.cfg.tiktoken_encoding:
                enc = tiktoken.get_encoding(self.cfg.tiktoken_encoding)
            # 2) encoding_for_model when we know the model
            elif self.cfg.openai_model_hint:
                try:
                    enc = tiktoken.encoding_for_model(self.cfg.openai_model_hint)
                except Exception:
                    # try common modern encodings as fallback
                    for name in ("o200k_base", "cl100k_base"):
                        try:
                            enc = tiktoken.get_encoding(name)
                            break
                        except Exception:
                            continue
            else:
                # no hint: pick a common default
                enc = tiktoken.get_encoding("cl100k_base")

            if enc is not None:
                def token_counter(s: str) -> int:
                    return len(enc.encode(s))
                return token_counter
        except Exception:
            # tiktoken unavailable or failed — fall back
            pass

        # heuristic fallback: ~4 chars/token
        return lambda s: max(1, len(s) // 4)