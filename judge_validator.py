# judge_validator.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import logging
import gc
import torch

from tqdm.auto import tqdm

# Helpers shared with your pipeline
from kg_utils import (
    get_attr as _get, set_attr as _set,
    ensure_node_props, ensure_rel_props,
    get_node_label, get_node_id, get_node_name, set_node_name,
)

# T5 judge
from t5_judge.judge import T5Judge

# Deberta judge
from deberta_judge.judge import DebertaJudge


# Config

@dataclass(frozen=True)
class JudgeValidatorConfig:
    model_name: str = "google/flan-t5-large"
    device: Optional[str] = None
    action: str = "filter"           # 'filter' | 'annotate'
    batch_size: int = 16
    node_threshold: float = 0.4
    rel_threshold: float = 0.4
    auto_release_cuda: bool = True
    quantize: bool = False
    qbits: int = 8
    logger: Optional[logging.Logger] = None


# Default prompt templates (override-able)

DEFAULT_ENTITY_TEMPLATES: Dict[str, str] = {
    "Person": "is a person",
    "Organization": "is an organization",
    "Product": "is a product",
    "Location": "is a location",
    "City": "is a city",
    "Country": "is a country",
    "Government Body": "is a government body",
    "Year": "is a year",
}

DEFAULT_REL_TEMPLATES: Dict[str, str] = {
    "WORKS_FOR": "{src} works for {tgt}.",
    "FOUNDED": "{src} founded {tgt}.",
    "ACQUIRED": "{src} acquired {tgt}.",
    "LOCATED_IN": "{src} is located in {tgt}.",
    "AFFILIATED_WITH": "{src} is affiliated with {tgt}.",
    "COMPETES_WITH": "{src} competes with {tgt}.",
    "MADE_BY": "{src} is made by {tgt}.",
    "PARTNERED_WITH": "{src} partnered with {tgt}.",
}


# Main class (LangGraph-compatible callable)

class JudgeValidator:
    """
    Batched validation of nodes/relationships using a T5Judge.
    - 'filter' mode: drops low-confidence nodes/edges
    - 'annotate' mode: keeps everything, adds judge_score

    Usage as a LangGraph node: instance(state) -> partial KGState
    """

    def __init__(self,
                 cfg: JudgeValidatorConfig,
                 entity_templates: Optional[Dict[str, str]] = None,
                 relation_templates: Optional[Dict[str, str]] = None):
        self.cfg = cfg
        self.entity_templates = entity_templates or DEFAULT_ENTITY_TEMPLATES
        self.relation_templates = relation_templates or DEFAULT_REL_TEMPLATES
        self.logger = cfg.logger or logging.getLogger("judge_validator")

        # Lazily constructed when first used (so tests without CUDA don’t blow up)
        self._judge: Optional[T5Judge | DebertaJudge] = None
        
    def unload(self):
        if getattr(self, "_judge", None) is not None:
            try:
                # move off GPU & drop refs
                if hasattr(self._judge, "model"):
                    self._judge.model.to("cpu")
                del self._judge
            except Exception:
                pass
            self._judge = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # LangGraph entrypoint
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Read overrides from state (fallback to cfg)
            node_thr = float(state.get("judge_node_threshold") or self.cfg.node_threshold)
            rel_thr = float(state.get("judge_rel_threshold") or self.cfg.rel_threshold)
            model_name = state.get("judge_model_name") or self.cfg.model_name
            device = state.get("judge_device") or self.cfg.device
            action = (state.get("judge_action") or self.cfg.action).lower()
            keep_all = action == "annotate"
            batch_size = int(state.get("judge_batch_size") or self.cfg.batch_size)
            
            self.logger.debug("JudgeValidator called with model='%s', device='%s', action='%s', node_thr=%.2f, rel_thr=%.2f, batch_size=%d",
                              model_name, device or "default", action, node_thr, rel_thr, batch_size)
            self.logger.debug("JudgeValidator has entity templates: %s", self.entity_templates)
            self.logger.debug("JudgeValidator has relation templates: %s", self.relation_templates)

            # Init judge once
            judge = self._get_or_create_judge(model_name, device, quantize=self.cfg.quantize, qbits=self.cfg.qbits)
            if judge is None:
                self.logger.warning("Judge initialization failed. Skipping validation.")
                return {}

            docs = state.get("docs") or []
            gdocs = state.get("graph_docs") or []
            if not docs or not gdocs:
                return {}

            judge_results: List[Dict[str, Any]] = []
            updated_docs: List[Any] = []
            stats = {"nodes_removed": 0, "rels_removed": 0, "nodes_kept": 0, "rels_kept": 0, "mode": action}

            for doc, gd in tqdm(zip(docs, gdocs), total=min(len(docs), len(gdocs)), desc="Judge validation", unit="doc"):
                evidence = getattr(doc, "page_content", None) or ""

                # Nodes (collect pairs)
                node_items: List[Tuple[Any, str, str, str]] = []   # (node, label, name, fact)
                node_pairs: List[Tuple[str, str]] = []             # (evidence, fact)

                for node in getattr(gd, "nodes", []):
                    lbl = get_node_label(node)
                    name = get_node_name(node) or ""
                    if not lbl or not name:
                        if keep_all:
                            ensure_node_props(node).setdefault("judge_score", 0.0)
                            stats["nodes_kept"] += 1
                        else:
                            stats["nodes_removed"] += 1
                        continue
                    tpl = self.entity_templates.get(lbl)
                    if not tpl:
                        stats["nodes_kept"] += 1
                        continue
                    fact = f"{name} {tpl}"
                    node_items.append((node, lbl, name, fact))
                    node_pairs.append((evidence, fact))

                node_scores = self._batch_scores_safe(judge, node_pairs, batch_size=batch_size) if node_pairs else []

                # Nodes (apply)
                new_nodes: List[Any] = []
                valid_ids: set = set()
                for (node, lbl, name, fact_str), score in zip(node_items, node_scores):
                    ok = (score >= node_thr) or keep_all
                    judge_results.append({
                        "type": "node",
                        "name": lbl,
                        "fact": fact_str,
                        "evidence": evidence,
                        "score": float(score),
                        "accepted": bool(score >= node_thr),
                    })
                    ensure_node_props(node)["judge_score"] = float(score)
                    if ok:
                        new_nodes.append(node)
                        nid = get_node_id(node)
                        if nid:
                            valid_ids.add(nid)
                        stats["nodes_kept"] += 1
                    else:
                        stats["nodes_removed"] += 1

                if keep_all:
                    # keep original unscored/missing nodes too
                    kept_ids = {get_node_id(n) for n in new_nodes if get_node_id(n)}
                    for n in getattr(gd, "nodes", []):
                        nid = get_node_id(n)
                        if nid and nid in kept_ids:
                            continue
                        ensure_node_props(n).setdefault("judge_score", 0.0)
                        new_nodes.append(n)
                        if nid:
                            valid_ids.add(nid)

                _set(gd, "nodes", new_nodes)

                # Relationships (collect)
                rel_items: List[Tuple[Any, Optional[str], Optional[str]]] = []  # (rel, rtype|None, fact|None)
                rel_pairs: List[Tuple[str, str]] = []

                def _name_of(id_):
                    for n in new_nodes:
                        if get_node_id(n) == id_:
                            return get_node_name(n) or id_
                    return id_

                for rel in getattr(gd, "relationships", []):
                    rtype = _get(rel, "type") or _get(rel, "label")
                    src = _get(rel, "source"); tgt = _get(rel, "target")
                    src_id = _get(src, "id") if not isinstance(src, str) else src
                    tgt_id = _get(tgt, "id") if not isinstance(tgt, str) else tgt

                    if not rtype or not src_id or not tgt_id:
                        if keep_all:
                            ensure_rel_props(rel).setdefault("judge_score", 0.0)
                            rel_items.append((rel, None, None))
                        else:
                            stats["rels_removed"] += 1
                        continue

                    if not keep_all and (src_id not in valid_ids or tgt_id not in valid_ids):
                        stats["rels_removed"] += 1
                        continue

                    tpl = self.relation_templates.get(rtype)
                    if not tpl:
                        rel_items.append((rel, None, None))
                        continue

                    fact = tpl.format(src=_name_of(src_id), tgt=_name_of(tgt_id))
                    rel_items.append((rel, rtype, fact))
                    rel_pairs.append((evidence, fact))

                rel_scores = self._batch_scores_safe(judge, rel_pairs, batch_size=batch_size) if rel_pairs else []

                # Relationships (apply)
                new_rels: List[Any] = []
                score_iter = iter(rel_scores)
                for rel, rtype, fact_str in rel_items:
                    if rtype is None:
                        new_rels.append(rel)
                        stats["rels_kept"] += 1
                        continue

                    score = float(next(score_iter))
                    ok = (score >= rel_thr) or keep_all

                    judge_results.append({
                        "type": "relation",
                        "name": rtype,
                        "fact": fact_str,
                        "evidence": evidence,
                        "score": score,
                        "accepted": bool(score >= rel_thr),
                    })
                    ensure_rel_props(rel)["judge_score"] = score

                    if ok:
                        new_rels.append(rel)
                        stats["rels_kept"] += 1
                    else:
                        stats["rels_removed"] += 1

                _set(gd, "relationships", new_rels)
                updated_docs.append(gd)

            self.logger.info(
                "Judge done (mode=%s). nodes_kept=%d nodes_removed=%d rels_kept=%d rels_removed=%d",
                action, stats["nodes_kept"], stats["nodes_removed"], stats["rels_kept"], stats["rels_removed"]
            )
            return {"graph_docs": updated_docs, "judge_stats": stats, "judge_results": judge_results}
        finally:
            if self.cfg.auto_release_cuda:
                self.unload()

    # internals

    def _get_or_create_judge(self, model_name: str, device: Optional[str], quantize: bool = False, qbits: int = 8) -> Optional[T5Judge | DebertaJudge]:
        if self._judge is not None:
            return self._judge
        self.logger.debug("Initializing judge model '%s' on device '%s'.", model_name, device or "default")
        if "deberta" in model_name.lower():
            try:
                self._judge = DebertaJudge(model_name=model_name, device=device)
                return self._judge
            except Exception as e:
                self.logger.warning("Deberta Judge initialization failed (%s).", e)
                return None
        else:
            try:
                self._judge = T5Judge(model_name=model_name, device=device, logger=self.logger, quantize=quantize, qbits=qbits)
                return self._judge
            except Exception as e:
                self.logger.warning("FLAN-T5 Judge initialization failed (%s).", e)
                return None

    @staticmethod
    def _batch_scores_safe(judge: T5Judge | DebertaJudge,
                           pairs: Sequence[Tuple[str, str]],
                           batch_size: int = 16) -> List[float]:
        try:
            results = judge.batch_judge_claims(pairs, batch_size=batch_size)
            return [float(r.truth_score) for r in results]
        except (RuntimeError, ValueError):
            # keep-on-error semantics
            return [0.0] * len(pairs)