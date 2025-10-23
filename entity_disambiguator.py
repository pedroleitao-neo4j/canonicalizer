# entity_disambiguator.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging
from pathlib import Path
import gc

# sentence-transformers
from sentence_transformers import SentenceTransformer, util

# Shared helpers
from kg_utils import (
    slugify, normalized_id,
    get_attr as _get, set_attr as _set,
    get_node_label, get_node_name, get_node_id, set_node_name,
    remap_relationship_ids,
)

@dataclass(frozen=True)
class EntityDisambiguatorConfig:
    model_name: str = "Qwen/Qwen3-Embedding-4B"
    device: Optional[str] = ""
    threshold: float = 0.85
    include_snippet: bool = False
    default_labels: List[str] = field(default_factory=lambda: ["Product"])
    disambiguate_labels: List[str] = field(default_factory=lambda: [])
    write_yaml: bool = True            # write canonical_map.yaml
    auto_release_cuda: bool = True
    logger: Optional[logging.Logger] = None

class EntityDisambiguator:
    """
    Clusters contextual mentions per label and normalizes surface forms:
      - builds per-label mentions with short context windows
      - embeds with SentenceTransformer
      - clusters with community_detection
      - majority vote to pick canonical surface forms
      - rewrites node names & IDs, remaps relationship endpoints

    LangGraph-compatible: __call__(state) -> partial KGState
    """

    def __init__(self, cfg: EntityDisambiguatorConfig):
        self.cfg = cfg
        self.logger = cfg.logger or logging.getLogger("entity_disambiguator")
        self._model: Optional[SentenceTransformer] = None

    # LangGraph node
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        labels = self.cfg.disambiguate_labels or self.cfg.default_labels
        threshold = float(state.get("disambiguation_threshold") or self.cfg.threshold)
        self.logger.info("Disambiguation start: labels=%s threshold=%.2f", labels, threshold)

        docs = state.get("docs") or []
        gdocs = state.get("graph_docs") or []

        # Collect contextual mentions
        per_label_mentions: Dict[str, List[Dict[str, Any]]] = {lbl: [] for lbl in labels}
        for i, gd in enumerate(gdocs):
            evidence = _get(docs[i], "page_content", "") if i < len(docs) else ""
            for node in getattr(gd, "nodes", []):
                label = get_node_label(node)
                if label in labels:
                    name = get_node_name(node)
                    if not name:
                        continue
                    if self.cfg.include_snippet:
                        ctx = self._context_snippet(evidence, name, max_chars=512)
                        embed_text = f"[{label}] {name}\nContext: {ctx}"
                    else:
                        embed_text = f"[{label}] {name}"
                    per_label_mentions[label].append({"name": name, "embed_text": embed_text})

        if all(len(v) == 0 for v in per_label_mentions.values()):
            self.logger.info("No names found for configured labels; skipping disambiguation.")
            return {"canonical_map": {}}

        model = self._get_or_load_model()

        canonical_map: Dict[str, Dict[str, str]] = {lbl: {} for lbl in labels}
        for label, mentions in per_label_mentions.items():
            if not mentions:
                continue
            texts = [m["embed_text"] for m in mentions]
            self.logger.debug("Encoding %d mentions for label %s", len(texts), label)
            emb = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
            clusters = util.community_detection(emb, threshold=threshold, min_community_size=1)
            self.logger.info("Label %s clustered into %d groups", label, len(clusters))

            # votes[name][canonical] = count
            votes: Dict[str, Dict[str, int]] = {}
            for cl in clusters:
                cl_names = [mentions[idx]["name"] for idx in cl]
                canonical = max(cl_names, key=len)  # longest tokenized string as tie-breaker
                for n in cl_names:
                    votes.setdefault(n, {})
                    votes[n][canonical] = votes[n].get(canonical, 0) + 1

            for n, cand_counts in votes.items():
                best = sorted(cand_counts.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)[0][0]
                canonical_map[label][n] = best

        # Optionally write YAML for inspection
        if self.cfg.write_yaml:
            try:
                import yaml
                Path("out").mkdir(parents=True, exist_ok=True)
                with open("out/canonical_map.yaml", "w") as f:
                    yaml.dump(canonical_map, f)
            except Exception as e:
                self.logger.debug("Failed to write canonical_map.yaml: %s", e)

        # Apply canonical forms & remap relationships
        updated_docs = []
        total_id_changes = total_name_changes = total_rel_src_changes = total_rel_tgt_changes = 0
        for doc in gdocs:
            id_map: Dict[str, str] = {}
            for node in getattr(doc, "nodes", []):
                label = get_node_label(node)
                if label in labels:
                    old_name = get_node_name(node)
                    new_name = canonical_map.get(label, {}).get(old_name, old_name)
                    if new_name and new_name != old_name:
                        self.logger.info("Name remap [%s]: '%s' -> '%s'", label, old_name, new_name)
                        set_node_name(node, new_name)
                        total_name_changes += 1
                    old_id = get_node_id(node) or slugify(str(old_name))
                    new_id = normalized_id(new_name or old_name, label)
                    if new_id != old_id:
                        self.logger.info("ID remap   [%s]: '%s' -> '%s'", label, old_id, new_id)
                        _set(node, "id", new_id)
                        id_map[old_id] = new_id
                        total_id_changes += 1
            rel_stats = remap_relationship_ids(doc, id_map)
            total_rel_src_changes += rel_stats["rel_src"]
            total_rel_tgt_changes += rel_stats["rel_tgt"]
            updated_docs.append(doc)

        self.logger.info(
            "Disambiguation done. name_changes=%d id_changes=%d rel_src_updates=%d rel_tgt_updates=%d",
            total_name_changes, total_id_changes, total_rel_src_changes, total_rel_tgt_changes,
        )

        # Always free CUDA if configured
        self._auto_release()
        return {"graph_docs": updated_docs, "canonical_map": canonical_map}

    # internals
    def _get_or_load_model(self) -> SentenceTransformer:
        if self._model is not None:
            return self._model
        self._model = SentenceTransformer(self.cfg.model_name, device=self.cfg.device or "cpu")
        return self._model

    @staticmethod
    def _context_snippet(text: str, name: str, max_chars: int = 512) -> str:
        if not text:
            return ""
        t = text
        needle = (name or "").lower()
        idx = t.lower().find(needle) if needle else -1
        if idx == -1:
            return t[:max_chars]
        half = max_chars // 2
        start = max(0, idx - half)
        end = min(len(t), idx + len(name or "") + half)
        return t[start:end]

    def _auto_release(self):
        """Optionally release CUDA memory used by the embedder."""
        if not self.cfg.auto_release_cuda:
            return
        try:
            import torch
            # If model is on CUDA, move to CPU & clear cache
            if self._model is not None and any(p.is_cuda for p in self._model.parameters()):
                try:
                    self._model.to("cpu")
                except Exception:
                    pass
            self._model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            # best-effort cleanup
            pass