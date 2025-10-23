# extractor.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable
import asyncio
import logging
import time

from langchain_core.documents import Document
from tqdm.auto import tqdm
import inspect

@dataclass(frozen=True)
class ExtractGraphConfig:
    shard_size: int = 8
    max_concurrency: int = 1
    rate_limit_rpm: Optional[int] = None
    show_progress: bool = True
    logger: Optional[logging.Logger] = None

class ExtractGraph:
    """
    Parallelizable wrapper around LangChain's LLMGraphTransformer.*convert*_to_graph_documents.
    Uses the ASYNC aconvert_to_graph_documents when available to avoid thread-safety issues.
    """

    def __init__(self, cfg: ExtractGraphConfig, transformer: Any):
        self.tfm = transformer
        self.cfg = cfg
        self.logger = cfg.logger or logging.getLogger("extract_graph")

        # Resolve best available conversion function(s)
        self._async_convert: Optional[Callable[[List[Document]], Any]] = None
        self._sync_convert: Optional[Callable[[List[Document]], Any]] = None

        # Prefer async if present
        if hasattr(self.tfm, "aconvert_to_graph_documents"):
            fn = getattr(self.tfm, "aconvert_to_graph_documents")
            if inspect.iscoroutinefunction(fn):
                self._async_convert = fn  # coroutine
        # Keep sync fallback for older transformers or explicit single-thread usage
        if hasattr(self.tfm, "convert_to_graph_documents"):
            self._sync_convert = getattr(self.tfm, "convert_to_graph_documents")

        if not self._async_convert and not self._sync_convert:
            raise AttributeError(
                "Transformer must expose aconvert_to_graph_documents or convert_to_graph_documents."
            )

    # LangGraph node entrypoint
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        docs: List[Document] = state.get("docs") or []
        if not docs:
            return {"graph_docs": [], "summary": {"node_count": 0, "relationship_count": 0}}

        shard_size = int(state.get("extract_batch_size") or self.cfg.shard_size)
        shards = [docs[i:i + shard_size] for i in range(0, len(docs), shard_size)]

        # If we have async available, route through async runners even for sequential (to avoid thread races)
        if self._async_convert:
            if self.cfg.max_concurrency <= 1:
                graph_docs, errors = asyncio.run(self._run_sequential_async(shards))
            else:
                graph_docs, errors = asyncio.run(self._run_parallel_async(shards))
        else:
            # Fallback: purely synchronous, single-threaded only
            graph_docs, errors = self._run_sequential_sync(shards)

        summary = {
            "node_count": sum(len(getattr(d, "nodes", [])) for d in graph_docs),
            "relationship_count": sum(len(getattr(d, "relationships", [])) for d in graph_docs),
            "batch_size": shard_size,
        }
        if errors:
            summary["errors"] = errors

        return {"graph_docs": graph_docs, "summary": summary}

    # async paths

    async def _run_sequential_async(self, shards: List[List[Document]]) -> Tuple[List[Any], int]:
        assert self._async_convert is not None
        out: List[Any] = []
        errors = 0
        iterator = tqdm(shards, desc="Extract graph", unit="shard") if self.cfg.show_progress else shards
        for shard in iterator:
            try:
                res = await self._async_convert(shard)  # single coroutine call
                out.extend(res)
            except Exception as e:
                self.logger.warning("Failed to extract shard of %d docs: %s", len(shard), e)
                errors += len(shard)
        return out, errors

    async def _run_parallel_async(self, shards: List[List[Document]]) -> Tuple[List[Any], int]:
        assert self._async_convert is not None
        sem = asyncio.Semaphore(self.cfg.max_concurrency)
        gate = _RateGate(self.cfg.rate_limit_rpm) if self.cfg.rate_limit_rpm else None

        async def run_one(shard: List[Document]) -> Tuple[List[Any], int]:
            async with sem:
                if gate:
                    await gate.wait()
                try:
                    res = await self._async_convert(shard)  # stay on the event loop; no threads
                    return res, 0
                except Exception as e:
                    self.logger.warning("Failed to extract shard of %d docs: %s", len(shard), e)
                    return [], len(shard)

        tasks = [asyncio.create_task(run_one(s)) for s in shards]
        results: Sequence[Tuple[List[Any], int]] = []
        if self.cfg.show_progress:
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

    # sync fallback

    def _run_sequential_sync(self, shards: List[List[Document]]) -> Tuple[List[Any], int]:
        assert self._sync_convert is not None
        out: List[Any] = []
        errors = 0
        iterator = tqdm(shards, desc="Extract graph", unit="shard") if self.cfg.show_progress else shards
        for shard in iterator:
            try:
                out.extend(self._sync_convert(shard))
            except Exception as e:
                self.logger.warning("Failed to extract shard of %d docs: %s", len(shard), e)
                errors += len(shard)
        return out, errors

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
