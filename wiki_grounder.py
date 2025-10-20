# wiki_grounder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import logging
import re
import difflib
import html

from functools import lru_cache
from joblib import Memory
from tqdm.auto import tqdm

import pywikibot
from pywikibot.data.api import Request as ApiRequest

from kg_utils import (
    slugify,
    normalized_id,
    get_attr as _get,
    set_attr as _set,
    ensure_node_props,
    get_node_label,
    get_node_id,
    get_node_name,
    set_node_name,
    remap_relationship_ids,
    DEFAULT_WIKI_CATEGORY,
)


try:
    from rerankers import Reranker
    from rerankers import Document as RRDocument
except Exception:
    Reranker = None
    RRDocument = None

DEFAULT_WIKI_CATEGORY: Dict[str, str] = {
    "Organization": "Companies",
    "Product": "Products",
    "Person": "People",
    "Country": "Countries",
    "Government Body": "Government bodies",
    "Location": "Places",
}

# Public config dataclass

@dataclass(frozen=True)
class WikipediaGrounderConfig:
    cache_dir: str = ".kg_cache"
    user_agent: str = "PedroSearchBot/1.0"
    reranker_model: Optional[str] = "mixedbread-ai/mxbai-rerank-base-v1"
    log: Optional[logging.Logger] = None

# Main class

class WikipediaGrounder:
    """
    Ground graph nodes (by label) to Wikipedia titles/Wikidata IDs, with:
    - CirrusSearch (intitle: / deepcat:)
    - Optional cross-encoder reranking (rerankers)
    - Fallback to difflib string similarity
    Callable as a LangGraph node: __call__(state) -> partial KGState.
    """

    def __init__(self, cfg: WikipediaGrounderConfig):
        self.cfg = cfg
        self.logger = cfg.log or logging.getLogger("wiki_grounder")
        self.memory = Memory(location=cfg.cache_dir, verbose=0)

        # Lazily wrap impl functions with joblib cache so we keep clean method signatures
        self.wiki_search_titles = self.memory.cache(self._wiki_search_titles_impl)
        self.wiki_page_info = self.memory.cache(self._wiki_page_info_impl)

        # Reranker (optional)
        self._reranker = None
        if Reranker is not None and cfg.reranker_model:
            try:
                self._reranker = Reranker(cfg.reranker_model, model_type="cross-encoder", verbose=0)
            except Exception as e:
                self.logger.warning("Reranker init failed: %s", e)
                self._reranker = None

        # Make pywikibot happy for read-only ops
        pywikibot.config.user_agent = cfg.user_agent
        # Avoid requiring a user-config file
        import os
        os.environ.setdefault("PYWIKIBOT_NO_USER_CONFIG", "1")

    # Public entrypoint (LangGraph node)

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Grounds nodes in state['graph_docs'] using the Wikipedia API."""
        labels = state.get("ground_labels") or ["Organization", "Product"]
        lang = state.get("wiki_language") or "en"
        k = int(state.get("wiki_search_k") or 5)
        use_deepcat = bool(state.get("wiki_use_deepcat") or False)
        threshold = float(state.get("wiki_match_threshold") or 0.9)
        hints_map: Dict[str, str] = state.get("wiki_query_hints") or {}

        self.logger.info(
            "Wikipedia grounding start: labels=%s lang=%s k=%d threshold=%.2f",
            labels, lang, k, threshold,
        )

        updated_docs = []
        total_name_changes = total_id_changes = total_rel_src_changes = total_rel_tgt_changes = grounded_nodes_count = 0
        docs = state.get("docs") or []

        for idx, gd in enumerate(tqdm(state.get("graph_docs") or [], desc="Wikipedia grounding", unit="doc")):
            id_map: Dict[str, str] = {}
            for node in getattr(gd, "nodes", []):
                lbl = get_node_label(node)
                if lbl not in labels:
                    continue

                query = get_node_name(node)
                if not query:
                    continue

                hint = hints_map.get(lbl) or DEFAULT_WIKI_CATEGORY.get(lbl)
                results = self.wiki_search_titles(query, lang=lang, limit=k, hint=hint, use_deepcat=use_deepcat)
                best = self._best_wiki_match(query, results, threshold)
                if not best:
                    continue

                title = best.get("title") or query
                info = self.wiki_page_info(title, lang=lang)
                fullurl = info.get("fullurl") or f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
                pageid = info.get("pageid") or best.get("pageid")

                old_name = get_node_name(node)
                props = ensure_node_props(node)
                props.update({
                    "wiki_title": title,
                    "wiki_pageid": pageid,
                    "wiki_url": fullurl,
                    "wiki_lang": lang,
                })
                if info.get("description") is not None:
                    props["wiki_description"] = info.get("description")
                if info.get("wikidata_id") is not None:
                    props["wikidata_id"] = info.get("wikidata_id")

                self.logger.info(
                    "Enriched wiki [%s]: name='%s', title='%s', pageid=%s, qid=%s, url=%s (hint=%s)",
                    lbl, old_name, title, str(pageid), str(info.get("wikidata_id")), fullurl, hint,
                )
                grounded_nodes_count += 1

                if title and title != old_name:
                    self.logger.info("Ground name [%s]: '%s' -> '%s'", lbl, old_name, title)
                    set_node_name(node, title)
                    total_name_changes += 1

                old_id = get_node_id(node) or slugify(str(old_name))
                new_id = normalized_id(title or old_name, lbl)
                if new_id != old_id:
                    self.logger.info("Ground id   [%s]: '%s' -> '%s'", lbl, old_id, new_id)
                    _set(node, "id", new_id)
                    id_map[old_id] = new_id
                    total_id_changes += 1

            rel_stats = remap_relationship_ids(gd, id_map)
            total_rel_src_changes += rel_stats["rel_src"]
            total_rel_tgt_changes += rel_stats["rel_tgt"]
            updated_docs.append(gd)

        self.logger.info(
            "Wikipedia grounding done. grounded_nodes=%d name_changes=%d id_changes=%d rel_src_updates=%d rel_tgt_updates=%d",
            grounded_nodes_count, total_name_changes, total_id_changes, total_rel_src_changes, total_rel_tgt_changes,
        )
        return {"graph_docs": updated_docs}

    # Cached IO implementations

    def _wiki_search_titles_impl(self, query: str, lang: str, limit: int,
                                 hint: Optional[str], use_deepcat: bool) -> List[Dict[str, Any]]:
        if not query:
            return []
        try:
            # Build CirrusSearch query
            safe = re.sub(r"\s+", " ", query.strip().replace('"', ""))
            eff = f'intitle:"{safe}"'
            if use_deepcat and hint:
                eff += f' deepcat:"{hint}"'
            elif hint:
                eff += f" {hint}"

            site = pywikibot.Site(lang, "wikipedia")
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": eff,
                "srnamespace": 0,
                "srlimit": int(limit or 10),
                "srprop": "snippet|size|wordcount|timestamp|titlesnippet",
            }
            data = ApiRequest(site=site, **params).submit()
            hits = data.get("query", {}).get("search", []) or []
            out: List[Dict[str, Any]] = []
            for h in hits:
                snippet_html = h.get("snippet") or ""
                out.append({
                    "title": h.get("title"),
                    "pageid": h.get("pageid"),
                    "snippet": html.unescape(snippet_html),
                    "wordcount": h.get("wordcount"),
                    "size": h.get("size"),
                    "timestamp": h.get("timestamp"),
                })
            return out
        except Exception as e:
            self.logger.warning("Wikipedia title search failed for %r: %s", query, e)
            return []

    def _wiki_page_info_impl(self, title: str, lang: str) -> Dict[str, Any]:
        try:
            site = pywikibot.Site(lang, "wikipedia")
            page = pywikibot.Page(site, title)
            props = page.properties()
            desc = page.description() if hasattr(page, "description") else None
            return {
                "pageid": page.pageid,
                "title": page.title(),
                "fullurl": page.full_url(),
                "description": desc,
                "wikidata_id": props.get("wikibase_item"),
            }
        except Exception as e:
            self.logger.warning("Wikipedia page info failed for title=%r: %s", title, e)
            return {}

    # Ranking / matching

    def _best_wiki_match(self, query: str, results: List[Dict[str, Any]], thr: float) -> Optional[Dict[str, Any]]:
        if not results:
            return None

        q_slug = slugify(query)
        for r in results:
            if slugify(r.get("title", "")) == q_slug:
                self.logger.debug("Exact slug match: %r", r.get("title"))
                return r

        # Reranker path
        if self._reranker is not None and RRDocument is not None:
            docs = [RRDocument(text=r["title"] + ": " + r.get("snippet", ""), doc_id=str(i))
                    for i, r in enumerate(results) if r.get("title")]
            if docs:
                try:
                    ranked = self._reranker.rank(query=query, docs=docs) or []
                    if ranked:
                        top = ranked[0]
                        score = float(getattr(top, "score", 0.0))
                        self.logger.debug("Reranker top: %r (%.4f)", getattr(top, "text", None), score)
                        if score >= thr:
                            by_title = {r["title"]: r for r in results if r.get("title")}
                            return by_title.get(getattr(top, "text", ""))
                except Exception as e:
                    self.logger.debug("Reranker error, fallback to difflib: %s", e)

        # Fallback: difflib
        scored = []
        ql = (query or "").lower()
        for r in results:
            t = r.get("title", "")
            if not t:
                continue
            ratio = difflib.SequenceMatcher(a=ql, b=t.lower()).ratio()
            scored.append((ratio, r))
        if not scored:
            return None
        scored.sort(key=lambda x: x[0], reverse=True)
        best_ratio, best = scored[0]
        self.logger.debug("Fallback best: %r (%.4f)", best.get("title"), best_ratio)
        return best if best_ratio >= thr else None