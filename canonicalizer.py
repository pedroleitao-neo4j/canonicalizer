# Cleaned & compact KG workflow
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, TypedDict, Optional, Tuple, Sequence

import os
import re
import logging
from dotenv import load_dotenv, find_dotenv

from joblib import Memory
from tqdm.auto import tqdm

import pandas as pd

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph
from langgraph.graph import StateGraph, START, END

from kg_utils import (
    ensure_node_props as ensure_node_props,
    ensure_rel_props as ensure_rel_props,
    get_node_label as get_node_label,
    get_node_id as get_node_id,
    get_node_name as get_node_name,
    set_node_name as set_node_name,
    remap_relationship_ids as remap_relationship_ids,
    DEFAULT_WIKI_CATEGORY,
)

# Chunker
from chunker import DocChunker, DocChunkerConfig

# LLM Graph Extractor
from extractor import ExtractGraph, ExtractGraphConfig

# Judge
from judge_validator import JudgeValidatorConfig, JudgeValidator

# Wikipedia Grounder
from wiki_grounder import WikipediaGrounder, WikipediaGrounderConfig

# Disambiguator
from entity_disambiguator import EntityDisambiguatorConfig, EntityDisambiguator

# Load environment variables from a .env file early so KGConfig sees them    
_DOTENV_PATH = find_dotenv(usecwd=True) or str(Path(__file__).resolve().parent / ".env")
if _DOTENV_PATH:
    load_dotenv(_DOTENV_PATH, override=False)


# Config + Logging

@dataclass(frozen=True)
class KGConfig:
    log_file: str = os.getenv("KG_LOG_FILE", "out/kg_workflow.log")
    log_format: str = os.getenv("KG_LOG_FORMAT", "%(asctime)s %(levelname)s - %(message)s")
    log_level: str = os.getenv("KG_LOG_LEVEL", "DEBUG")
    cache_dir: str = os.getenv("KG_CACHE_DIR", ".kg_cache")

    llm_provider: str = os.getenv("LLM_PROVIDER", "openai").lower()
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma3:27b-it-qat")

    judge_model: str = os.getenv("KG_JUDGE_MODEL", "google/flan-t5-large")
    judge_action: str = os.getenv("KG_JUDGE_ACTION", "filter")  # or "annotate"

    wiki_lang_default: str = os.getenv("KG_WIKI_LANG", "en")


def setup_logging(cfg: KGConfig) -> logging.Logger:
    Path(cfg.log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=cfg.log_file,
        format=cfg.log_format,
        level=getattr(logging, cfg.log_level.upper(), logging.DEBUG),
        encoding="utf-8",
    )
    if os.getenv("KG_LOG_CONSOLE", "0") == "1":
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(cfg.log_format))
        logging.getLogger().addHandler(console)
    return logging.getLogger("kg_disambiguation")


CFG = KGConfig()
logger = setup_logging(CFG)

grounder = WikipediaGrounder(
    WikipediaGrounderConfig(
        cache_dir=CFG.cache_dir,
        user_agent="PedroSearchBot/1.0",
        reranker_model=os.getenv("RERANKERS_MODEL", "mixedbread-ai/mxbai-rerank-base-v1"),
        log=logger,
    )
)

DEFINITIONS = {
    "entities": {
        "Person": "An individual named human being, e.g., John Doe, Alice Smith.",
        "Organization": "A group of people working together, e.g., Apple Inc, NASA.",
        "Product": "A specific item produced for sale, e.g., iPhone, Microsoft Windows.",
        "Location": "A specific place, e.g., factory, or office.",
        "City": "A large town, e.g., New York, San Francisco.",
        "Country": "A nation with its own government, e.g., United States, Canada.",
        "Government Body": "An organization that governs a specific area, e.g., United Nations, Federal Reserve.",
        "Year": "A specific year in time, e.g., 2020, 1999.",
    },
    "relationships": {
        "WORKS_FOR": "Indicates that a person is employed by an organization.",
        "FOUNDED": "Indicates that a person established or started an organization.",
        "ACQUIRED": "Indicates that one organization has purchased another organization.",
        "LOCATED_IN": "Indicates the geographical location of an organization.",
        "AFFILIATED_WITH": "Indicates a formal association between a person and an organization.",
        "COMPETES_WITH": "Indicates that two organizations or products are in competition with each other.",
        "MADE_BY": "Indicates the organization that manufactures or produces a product.",
        "PARTNERED_WITH": "Indicates a collaborative relationship between two organizations.",
    }
}

# Judge templates
ENTITY_TYPE_TEMPLATES: Dict[str, str] = {
    "Person": f"is a person.",
    "Organization": f"is an organization.",
    "Product": f"is a product.",
    "Location": f"is a location.",
    "City": f"is a city.",
    "Country": f"is a country.",
    "Government Body": f"is a government body.",
    "Year": f"is a year.",
}

RELATION_TEMPLATES: Dict[str, str] = {
    "WORKS_FOR": "{src} works for {tgt}.",
    "FOUNDED": "{src} founded {tgt}.",
    "ACQUIRED": "{src} acquired {tgt}.",
    "LOCATED_IN": "{src} is located in {tgt}.",
    "AFFILIATED_WITH": "{src} is affiliated with {tgt}.",
    "COMPETES_WITH": "{src} competes with {tgt}.",
    "MADE_BY": "{src} is made by {tgt}.",
    "PARTNERED_WITH": "{src} partnered with {tgt}.",
}

chunker = DocChunker(
    DocChunkerConfig(
        target_bytes_default=1024,
        tiktoken_encoding=None,                 # let it infer
        openai_model_hint=CFG.openai_model if CFG.llm_provider == "openai" else None,
        default_overlap_frac=0.10,
        logger=logger,
    )
)

validator = JudgeValidator(
    JudgeValidatorConfig(
        model_name=CFG.judge_model,
        device=None,                      # or pass through env
        action=CFG.judge_action,          # 'filter' or 'annotate'
        batch_size=16,
        node_threshold=0.4,
        rel_threshold=0.4,
        auto_release_cuda=True,
        quantize=True,
        qbits=8,
        logger=logger,
    ),
    entity_templates=ENTITY_TYPE_TEMPLATES,
    relation_templates=RELATION_TEMPLATES,
)

disambiguator = EntityDisambiguator(
    EntityDisambiguatorConfig(
        model_name="Qwen/Qwen3-Embedding-4B",
        include_snippet=False,
        device="cuda",
        threshold=0.85,
        default_labels=[],
        write_yaml=True,
        auto_release_cuda=True,
        logger=logger,
    )
)


# Caching for expensive ops
memory = Memory(location=CFG.cache_dir, verbose=0)

# Make pywikibot happy for read-only ops
os.environ.setdefault("PYWIKIBOT_NO_USER_CONFIG", "1")

# State definition

class KGState(TypedDict, total=False):
    sentences: List[str]
    docs: List[Document]
    graph_docs: List[Any]
    node_samples: List[Dict[str, Any]]
    rel_samples: List[Dict[str, Any]]
    summary: Dict[str, Any]
    # Disambiguation config + results
    disambiguate_labels: List[str]
    disambiguation_threshold: float
    canonical_map: Dict[str, Dict[str, str]]  # label -> {original_name: canonical_name}
    # Wikipedia grounding config
    ground_labels: List[str]
    wiki_language: str
    wiki_search_k: int
    wiki_match_threshold: float
    wiki_use_deepcat: bool
    wiki_query_hints: Dict[str, str]
    # Chunking
    chunk_target_bytes: int
    # Judge config + results
    judge_node_threshold: float
    judge_rel_threshold: float
    judge_model_name: str
    judge_device: str
    judge_batch_size: int
    judge_action: str  # 'filter' | 'annotate'
    judge_stats: Dict[str, Any]
    judge_results: List[Dict[str, Any]]

# Extraction config

allowed_nodes = [
    "Person",
    "Organization",
    "Product",
    "Location",
    "City",
    "Country",
    "Government Body",
    "Year",
]

allowed_relationships = [
    ("Person", "WORKS_FOR", "Organization"),
    ("Person", "FOUNDED", "Organization"),
    ("Organization", "ACQUIRED", "Organization"),
    ("Organization", "LOCATED_IN", "Location"),
    ("Person", "AFFILIATED_WITH", "Organization"),
    ("Organization", "COMPETES_WITH", "Organization"),
    ("Product", "MADE_BY", "Organization"),
    ("Organization", "PARTNERED_WITH", "Organization"),
    ("Product", "COMPETES_WITH", "Product"),
]

node_properties = [
    "name",
    "wiki_title",
    "wiki_pageid",
    "wiki_url",
    "wiki_lang",
    "wiki_description",
    "wikidata_id",
    "judge_score",
]

additional_instructions = (
    "When extracting nodes, always include a 'name' property which should contain the text extracted.\n\n"
    "Here are the definitions of the allowed node types:\n\n"
    + "\n".join(f"- {k}: {v}" for k, v in DEFINITIONS["entities"].items())
    + "\n\nHere are the definitions of the allowed relationship types:\n\n"
    + "\n".join(f"- {k}: {v}" for k, v in DEFINITIONS["relationships"].items())
)

# LLM factory + transformer

def make_llm(cfg: KGConfig):
    if cfg.llm_provider == "openai":
        return ChatOpenAI(model=cfg.openai_model, temperature=0, openai_api_key=cfg.openai_api_key)
    return ChatOllama(base_url=cfg.ollama_base_url, model=cfg.ollama_model, temperature=0)


llm = make_llm(CFG)

transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=allowed_nodes,
    allowed_relationships=allowed_relationships,
    node_properties=node_properties,
    additional_instructions=additional_instructions,
    strict_mode=True,
)

extractor = ExtractGraph(
    transformer,
    ExtractGraphConfig(
        shard_size=int(os.getenv("KG_EXTRACT_BATCH_SIZE", "8")),
        # keep conservative defaults; bump when you’re ready
        max_concurrency=int(os.getenv("KG_EXTRACT_MAX_CONCURRENCY", "2")),
        rate_limit_rpm=os.getenv("KG_EXTRACT_RPM") and int(os.getenv("KG_EXTRACT_RPM")) or None,
        show_progress=True,
        logger=logger,
    ),
)

# Nodes

def build_docs(state: KGState) -> KGState:
    sentences = state.get("sentences") or []
    if not sentences:
        raise ValueError("No sentences provided in state['sentences']")
    return {"docs": [Document(page_content=s, metadata={"source": "workflow"}) for s in sentences]}


def extract_graph(state: KGState) -> KGState:
    docs = state.get("docs") or []
    if not docs:
        return {"graph_docs": [], "summary": {"node_count": 0, "relationship_count": 0}}

    batch_size = int(state.get("extract_batch_size") or os.getenv("KG_EXTRACT_BATCH_SIZE", "8"))

    graph_docs, errors = [], 0
    with tqdm(total=len(docs), desc="Extract graph", unit="doc") as pbar:
        for start in range(0, len(docs), batch_size):
            batch = docs[start : start + batch_size]
            try:
                gds = transformer.convert_to_graph_documents(batch)
                graph_docs.extend(gds)
            except Exception as e:
                logger.warning("Failed to extract batch %d:%d: %s", start, start + len(batch), e)
                errors += len(batch)
            finally:
                pbar.update(len(batch))

    summary = {
        "node_count": sum(len(d.nodes) for d in graph_docs),
        "relationship_count": sum(len(d.relationships) for d in graph_docs),
        "batch_size": batch_size,
    }
    if errors:
        summary["errors"] = errors
    return {"graph_docs": graph_docs, "summary": summary}


# Neo4j ingestions

_NEO4J_GRAPH_SINGLETON: Optional[Neo4jGraph] = None

def _get_neo4j_graph() -> Neo4jGraph:
    global _NEO4J_GRAPH_SINGLETON
    if _NEO4J_GRAPH_SINGLETON is None:
        for k in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"):
            if not os.environ.get(k):
                raise EnvironmentError(f"Missing environment variable: {k}")
        _NEO4J_GRAPH_SINGLETON = Neo4jGraph(
            url=os.environ["NEO4J_URI"],
            username=os.environ["NEO4J_USER"],
            password=os.environ["NEO4J_PASSWORD"],
        )
        _NEO4J_GRAPH_SINGLETON.query(
            """
            CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
            FOR (n:Entity) REQUIRE n.id IS UNIQUE
            """
        )
    return _NEO4J_GRAPH_SINGLETON


def write_to_neo4j(state: KGState) -> KGState:
    graph = _get_neo4j_graph()
    graph.add_graph_documents(state["graph_docs"], include_source=True)
    return {}


def sanity_checks(state: KGState) -> KGState:
    graph = _get_neo4j_graph()
    node_rows = graph.query(
        """
        MATCH (n:Entity)
        RETURN
          n.name AS name,
          n.wiki_title AS wiki_title,
          n.wikidata_id AS wikidata_id,
          n.wiki_url AS wiki_url,
          n.wiki_pageid AS wiki_pageid,
          n.wiki_lang AS wiki_lang,
          n.wiki_description AS wiki_description,
          labels(n) AS labels
        LIMIT 10
        """
    )
    rel_rows = graph.query(
        """
        MATCH (a)-[r]->(b)
        RETURN a.name AS from, TYPE(r) AS rel, b.name AS to, r.source AS source
        LIMIT 10
        """
    )
    wiki_counts = graph.query(
        """
        MATCH (n:Entity)
        RETURN
          count(n) AS total_nodes,
          sum(CASE WHEN n.wiki_title IS NOT NULL THEN 1 ELSE 0 END) AS grounded_nodes,
          sum(CASE WHEN n.wikidata_id IS NOT NULL THEN 1 ELSE 0 END) AS with_wikidata
        """
    )
    return {"node_samples": node_rows, "rel_samples": rel_rows, "wiki_counts": wiki_counts}


# Graph wiring

builder = StateGraph(KGState)
builder.add_node("build_docs", build_docs)
builder.add_node("chunk_docs", chunker)
builder.add_node("extract_graph", extractor)
# validation
builder.add_node("validate_with_judge", validator)
builder.add_node("disambiguate_entities", disambiguator)
builder.add_node("ground_with_wikipedia", grounder)
# io
builder.add_node("write_to_neo4j", write_to_neo4j)
builder.add_node("sanity_checks", sanity_checks)

builder.add_edge(START, "build_docs")
builder.add_edge("build_docs", "chunk_docs")
builder.add_edge("chunk_docs", "extract_graph")
# judge first, then disambiguation, then grounding
builder.add_edge("extract_graph", "validate_with_judge")
builder.add_edge("validate_with_judge", "disambiguate_entities")
# builder.add_edge("disambiguate_entities", "ground_with_wikipedia")
# builder.add_edge("ground_with_wikipedia", "write_to_neo4j")
builder.add_edge("disambiguate_entities", "write_to_neo4j")
builder.add_edge("write_to_neo4j", "sanity_checks")
builder.add_edge("sanity_checks", END)

workflow = builder.compile()


# Demo: load HF dataset and run

def load_bloomberg_texts(max_items: int = 200, seed: int = 42, keywords: Optional[List[str]] = None) -> List[str]:
    """Load Bloomberg pretraining dataset, clean, keyword-filter, and sample texts."""
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            "HuggingFace 'datasets' is required. Install with: pip install datasets"
        ) from e

    ds = load_dataset("Aletheia-ng/bloomberg-news-articles-pretraining-dataset")
    split_name = "train" if "train" in ds else next(iter(ds.keys()))
    split = ds[split_name]

    raw_texts = [t for t in split["text"] if isinstance(t, str) and t.strip()]

    # Truncate boilerplate like: "To contact the reporters on this story ..."
    cleaned = [re.sub(r"(?is)\bTo contact the reporters?\b.*", "", t).strip() for t in raw_texts]

    kws = keywords or ["Apple", "Google", "Microsoft", "Amazon"]
    pat = re.compile("|".join(map(re.escape, kws)), re.IGNORECASE)
    filtered = [t for t in cleaned if pat.search(t)]

    if not filtered:
        raise ValueError("No texts matched the keyword filter; adjust 'keywords'.")

    import random
    random.seed(seed)
    sampled = random.sample(filtered, min(max_items, len(filtered)))
    logger.info("Selected %d texts for processing out of %d after keyword filter (%s)", len(sampled), len(filtered), ", ".join(kws))
    return sampled


# Entry point

def run_workflow(
    sentences: List[str],
    *,
    chunk_target_bytes: Optional[int] = None,
    disambiguate_labels: Optional[List[str]] = None,
    disambiguation_threshold: Optional[float] = None,
    ground_labels: Optional[List[str]] = None,
    wiki_language: Optional[str] = None,
    wiki_search_k: Optional[int] = None,
    wiki_match_threshold: Optional[float] = None,
    wiki_use_deepcat: Optional[bool] = None,
    wiki_query_hints: Optional[Dict[str, str]] = None,
    judge_batch_size: Optional[int] = None,
    judge_node_threshold: Optional[float] = None,
    judge_rel_threshold: Optional[float] = None,
    judge_model_name: Optional[str] = None,
    judge_action: Optional[str] = None,
) -> KGState:
    pipeline_setup: KGState = {"sentences": sentences}

    if chunk_target_bytes is not None: pipeline_setup["chunk_target_bytes"] = chunk_target_bytes
    if disambiguate_labels is not None: pipeline_setup["disambiguate_labels"] = disambiguate_labels
    if disambiguation_threshold is not None: pipeline_setup["disambiguation_threshold"] = disambiguation_threshold
    if ground_labels is not None: pipeline_setup["ground_labels"] = ground_labels
    if wiki_language is not None: pipeline_setup["wiki_language"] = wiki_language
    if wiki_search_k is not None: pipeline_setup["wiki_search_k"] = wiki_search_k
    if wiki_use_deepcat is not None: pipeline_setup["wiki_use_deepcat"] = wiki_use_deepcat
    if wiki_match_threshold is not None: pipeline_setup["wiki_match_threshold"] = wiki_match_threshold
    if wiki_query_hints is not None: pipeline_setup["wiki_query_hints"] = wiki_query_hints
    if judge_batch_size is not None: pipeline_setup["judge_batch_size"] = judge_batch_size
    if judge_node_threshold is not None: pipeline_setup["judge_node_threshold"] = judge_node_threshold
    if judge_rel_threshold is not None: pipeline_setup["judge_rel_threshold"] = judge_rel_threshold
    if judge_model_name is not None: pipeline_setup["judge_model_name"] = judge_model_name
    if judge_action is not None: pipeline_setup["judge_action"] = judge_action

    result_state = workflow.invoke(pipeline_setup)
    summary = result_state.get("summary", {})
    print(f"Extracted {summary.get('node_count', 0)} nodes and {summary.get('relationship_count', 0)} relationships.")
    if result_state.get("canonical_map"):
        print("\nDisambiguation applied for labels:")
        for lbl, mp in result_state["canonical_map"].items():
            if mp:
                print(f"- {lbl}: {len(mp)} names mapped")

    return result_state


def run_demo_from_hf_dataset() -> pd.DataFrame:
    """Convenience runner that pulls a sample from HF and executes the workflow."""
    sampled_texts = load_bloomberg_texts(max_items=100, seed=42,
                                         keywords=["Apple", "Google", "Microsoft", "Amazon"]) 
    print(f"Selected {len(sampled_texts)} texts for processing.")

    state = run_workflow(
        sampled_texts,
        chunk_target_bytes=1024,
        disambiguate_labels=["Product", "Organization", "Country", "Government Body", "Person", "Location", "City"],
        disambiguation_threshold=1.0,
        ground_labels=["Product", "Organization", "Country", "Government Body", "Person", "Location", "City"],
        wiki_language="en",
        wiki_search_k=50,
        wiki_use_deepcat=False,
        wiki_match_threshold=0.9,  # works well for title similarity; reranker path also uses this threshold
        judge_batch_size=4,
        judge_node_threshold=0.3,
        judge_rel_threshold=0.3,
        judge_model_name="google/flan-t5-xxl",
        judge_action="annotate",  # set to 'filter' to drop low-confidence items
    )
    
    return state.get('judge_results')


if __name__ == "__main__":
    print("Starting KG workflow demo from HF dataset...")
    # Print available RAM, GPU info, etc.
    try:
        import torch
        print(f"Torch version: {torch.__version__}")
        if torch.cuda.is_available():
            print("CUDA is available. Device count:", torch.cuda.device_count())
            for i in range(torch.cuda.device_count()):
                print(f"Device {i}: {torch.cuda.get_device_name(i)}, Memory Allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB, Memory Cached: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB, Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        else:
            print("CUDA is not available.")
    except ImportError:
        print("Torch not installed; skipping GPU info.")
        
    df_results = run_demo_from_hf_dataset()
    
    # Save results to CSV for analysis
    if df_results:
        df = pd.DataFrame(df_results)
        Path("out").mkdir(parents=True, exist_ok=True)
        output_csv = "out/judge_results.csv"
        df.to_csv(output_csv, index=False)
        print(f"Saved judge results to {output_csv} for analysis.")
    
    print("Done!")