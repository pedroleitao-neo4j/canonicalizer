# extractor.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import asyncio
import logging
import time

from langchain_core.documents import Document
from tqdm.auto import tqdm

@dataclass(frozen=True)
class ExtractGraphConfig:
    # how many docs to send per LLM call
    shard_size: int = 8
    # set >1 to enable conservative parallelism (each shard is one LLM call)
    max_concurrency: int = 1
    # optional best-effort RPM guard (per model/account). If None, no guard.
    rate_limit_rpm: Optional[int] = None
    # show tqdm progress in both modes
    show_progress: bool = True
    # logger
    logger: Optional[logging.Logger] = None

class ExtractGraph:
    """
    Parallelizable wrapper around LangChain's LLMGraphTransformer.convert_to_graph_documents.
    LangGraph-compatible: __call__(state) -> {'graph_docs': List[Any], 'summary': ...}

    - By default runs single-threaded (exactly like your current function).
    - Set max_concurrency > 1 to run shards concurrently via threads (I/O-bound).
    - Optional RPM gate to reduce 429s/overload.
    """

    def __init__(self, transformer, cfg: ExtractGraphConfig):
        self.tfm = transformer
        self.cfg = cfg
        self.logger = cfg.logger or logging.getLogger("extract_graph")

    # LangGraph node entrypoint
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        docs: List[Document] = state.get("docs") or []
        if not docs:
            return {"graph_docs": [], "summary": {"node_count": 0, "relationship_count": 0}}

        # allow per-run override, same as your old env/state 'KG_EXTRACT_BATCH_SIZE'
        shard_size = int(state.get("extract_batch_size") or self.cfg.shard_size)
        shards = [docs[i:i + shard_size] for i in range(0, len(docs), shard_size)]

        if self.cfg.max_concurrency <= 1:
            graph_docs, errors = self._run_sequential(shards)
        else:
            graph_docs, errors = asyncio.run(self._run_parallel(shards))

        summary = {
            "node_count": sum(len(getattr(d, "nodes", [])) for d in graph_docs),
            "relationship_count": sum(len(getattr(d, "relationships", [])) for d in graph_docs),
            "batch_size": shard_size,
        }
        if errors:
            summary["errors"] = errors

        return {"graph_docs": graph_docs, "summary": summary}

    # sequential path (default)
    def _run_sequential(self, shards: List[List[Document]]) -> tuple[List[Any], int]:
        out: List[Any] = []
        errors = 0
        iterator = tqdm(shards, desc="Extract graph", unit="shard") if self.cfg.show_progress else shards
        for shard in iterator:
            try:
                out.extend(self.tfm.convert_to_graph_documents(shard))
            except Exception as e:
                self.logger.warning("Failed to extract shard of %d docs: %s", len(shard), e)
                errors += len(shard)
        return out, errors

    # parallel path (conservative)
    async def _run_parallel(self, shards: List[List[Document]]) -> tuple[List[Any], int]:
        sem = asyncio.Semaphore(self.cfg.max_concurrency)
        gate = _RateGate(self.cfg.rate_limit_rpm) if self.cfg.rate_limit_rpm else None

        async def run_one(shard: List[Document]) -> tuple[List[Any], int]:
            async with sem:
                if gate:
                    await gate.wait()
                try:
                    # offload sync call to a worker thread
                    res = await asyncio.to_thread(self.tfm.convert_to_graph_documents, shard)
                    return res, 0
                except Exception as e:
                    self.logger.warning("Failed to extract shard of %d docs: %s", len(shard), e)
                    return [], len(shard)

        tasks = [run_one(s) for s in shards]
        results: Sequence[tuple[List[Any], int]] = []
        if self.cfg.show_progress:
            # progress over completions
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Extract graph (parallel)", unit="shard"):
                results.append(await coro)
        else:
            results = await asyncio.gather(*tasks)

        all_docs: List[Any] = []
        total_errors = 0
        for docs, err in results:
            all_docs.extend(docs)
            total_errors += err
        return all_docs, total_errors

class _RateGate:
    """Very light RPM gate (best-effort)."""
    def __init__(self, rpm: int):
        self.dt = 60.0 / max(1, int(rpm))
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self):
        async with self._lock:
            now = time.time()
            wait = max(0.0, self._next - now)
            if wait:
                await asyncio.sleep(wait)
            self._next = max(now, self._next) + self.dt