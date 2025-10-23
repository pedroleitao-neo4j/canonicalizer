# doc_builder.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
import logging

from langchain_core.documents import Document

@dataclass(frozen=True)
class DocBuilderConfig:
    logger: Optional[logging.Logger] = None
    
class DocBuilder:
    """
    Builds documents into the Knowledge Graph from input texts for further processing.
    """
    def __init__(self, cfg: DocBuilderConfig) -> None:
        self.cfg = cfg
        self.logger = cfg.logger or logging.getLogger("doc_builder")

    #
    # LangGraph node
    #
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Building documents into the Knowledge Graph.")
        
        # Check that input_texts exist
        if not state.get("input_texts"):
            self.logger.warning("No input_texts found in state. Returning empty docs.")
            return {"docs": []}
        
        texts = state.get("input_texts", [])

        # Dedup texts
        texts = list(dict.fromkeys(texts))

        start_idx = len(state.get("docs", [])) # Preserve existing docs if any
        self.logger.info("Adding %d new documents starting from index %d.", len(texts), start_idx)

        docs = [
            Document(
                page_content=s,
                metadata={"source": "workflow", "idx": start_idx + i}
            )
            for i, s in enumerate(texts)
        ]

        return {"docs": docs}