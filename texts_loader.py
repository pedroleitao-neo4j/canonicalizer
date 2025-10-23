# texts_loader.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
import logging
import re
import random

from datasets import load_dataset

@dataclass(frozen=True)
class TextsLoaderConfig:
    dataset: str = "Aletheia-ng/bloomberg-news-articles-pretraining-dataset"
    split_name: str = "train"
    text_column: str = "text"
    clean_regex: Optional[str] = None
    keywords: Optional[List[str]] = None
    max_texts: Optional[int] = None
    seed: int = 42
    logger: Optional[logging.Logger] = None
    
class TextsLoader:
    """
    Loads texts from a specified dataset for further processing.
    """
    def __init__(self, cfg: TextsLoaderConfig) -> None:
        self.cfg = cfg
        self.logger = cfg.logger or logging.getLogger("texts_loader")
        ds = load_dataset(self.cfg.dataset, split=self.cfg.split_name)
        self.texts = ds[self.cfg.text_column]

    #
    # LangGraph node
    #
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Loading texts from dataset: %s", self.cfg.dataset)

        dataset = load_dataset(self.cfg.dataset, split=self.cfg.split_name)

        raw_texts = [t for t in dataset[self.cfg.text_column] if isinstance(t, str) and t.strip()]
        
        cleaned = raw_texts
        if self.cfg.clean_regex:
            cleaned = [re.sub(self.cfg.clean_regex, "", t).strip() for t in raw_texts]

        texts = cleaned
        if self.cfg.keywords:
            pat = re.compile("|".join(map(re.escape, self.cfg.keywords)), re.IGNORECASE)
            texts = [t for t in texts if pat.search(t)]
            if not texts:
                self.logger.warning("No texts matched the provided keywords: %s", self.cfg.keywords)
                
        # Randomly select up to max_texts
        if self.cfg.max_texts and len(texts) > self.cfg.max_texts:
            random.seed(self.cfg.seed)
            texts = random.sample(texts, self.cfg.max_texts)
        
        self.logger.info("Loaded %d texts from dataset.", len(texts))

        return {"input_texts": texts}