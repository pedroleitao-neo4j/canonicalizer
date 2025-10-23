# Cleaned & compact KG workflow
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, TypedDict, Optional

import os
import logging
import json
from dotenv import load_dotenv, find_dotenv

from joblib import Memory

import pandas as pd

import torch

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
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


# Texts Loader
from texts_loader import TextsLoader, TextsLoaderConfig

# Doc Builder
from doc_builder import DocBuilder, DocBuilderConfig

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

# Neo4j Writer
from neo4j_writer import Neo4jWriter, Neo4jWriterConfig

#
# Definitions for entities and relationships
#

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

#
# Judge model templates
#

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

ALLOWED_NODES = [
    "Person",
    "Organization",
    "Product",
    "Location",
    "City",
    "Country",
    "Government Body",
    "Year",
]

ALLOWED_RELATIONSHIPS = [
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

NODE_PROPERTIES = [
    "name",
    "wiki_title",
    "wiki_pageid",
    "wiki_url",
    "wiki_lang",
    "wiki_description",
    "wikidata_id",
    "judge_score",
]

ADDITIONAL_INSTRUCTIONS = (
    "When extracting nodes, always include a 'name' property which should contain the text extracted.\n\n"
    "Here are the definitions of the allowed node types:\n\n"
    + "\n".join(f"- {k}: {v}" for k, v in DEFINITIONS["entities"].items())
    + "\n\nHere are the definitions of the allowed relationship types:\n\n"
    + "\n".join(f"- {k}: {v}" for k, v in DEFINITIONS["relationships"].items())
)

#
# Load environment variables from a .env file early so KGConfig sees them
#

_DOTENV_PATH = find_dotenv(usecwd=True) or str(Path(__file__).resolve().parent / ".env")
if _DOTENV_PATH:
    load_dotenv(_DOTENV_PATH, override=False)


class KGState(TypedDict, total=False):
    """
    State dictionary for the KG workflow.
    """
    input_texts: List[str]
    docs: List[Any]
    graph_docs: List[Any]
    node_samples: List[Dict[str, Any]]
    rel_samples: List[Dict[str, Any]]
    summary: Dict[str, Any]
    canonical_map: Dict[str, Dict[str, str]]  # label -> {original_name: canonical_name}
    # Wikipedia grounding config
    ground_labels: List[str]
    wiki_language: str
    wiki_search_k: int
    wiki_match_threshold: float
    wiki_use_deepcat: bool
    wiki_query_hints: Dict[str, str]
    judge_stats: Dict[str, Any]
    judge_results: List[Dict[str, Any]]


def setup_logging() -> logging.Logger:
    """
    Setup logging based on configuration.
    """
    log_file = os.getenv("KG_LOG_FILE", "out/kg_workflow.log")
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        format=os.getenv("KG_LOG_FORMAT", "%(asctime)s %(levelname)s - %(message)s"),
        level=getattr(logging, os.getenv("KG_LOG_LEVEL", "DEBUG").upper(), logging.DEBUG),
        encoding="utf-8",
    )
    if os.getenv("KG_LOG_CONSOLE", "0") == "1":
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(os.getenv("KG_LOG_FORMAT", "%(asctime)s %(levelname)s - %(message)s")))
        logging.getLogger().addHandler(console)
    return logging.getLogger("canonicalizer")


def build_llm(provider: str, model: str, key: Optional[str], url: Optional[str]) -> Any:
    """
    Build the LLM based on configuration.
    """
    if provider == "openai":
        return ChatOpenAI(model=model, temperature=0, openai_api_key=key)
    return ChatOllama(base_url=url, model=model, temperature=0)


#
# Instantiate components
#

logger = setup_logging()

llm = build_llm(
    os.getenv("LLM_PROVIDER"),
    os.getenv("OPENAI_MODEL") if os.getenv("LLM_PROVIDER") == "openai" else os.getenv("OLLAMA_MODEL"),
    os.getenv("OPENAI_API_KEY") if os.getenv("LLM_PROVIDER") == "openai" else None,
    os.getenv("OLLAMA_BASE_URL") if os.getenv("LLM_PROVIDER") == "ollama" else None
)

transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=ALLOWED_NODES,
    allowed_relationships=ALLOWED_RELATIONSHIPS,
    node_properties=NODE_PROPERTIES,
    additional_instructions=ADDITIONAL_INSTRUCTIONS,
    strict_mode=True,
)

doc_builder = DocBuilder(
    DocBuilderConfig(
        logger=logger,
    )
)

input_loader = TextsLoader(
    TextsLoaderConfig(
        dataset="Aletheia-ng/bloomberg-news-articles-pretraining-dataset",
        split_name="train",
        text_column="text",
        clean_regex=r"(?is)\bTo contact the reporters?\b.*",  # remove boilerplate
        keywords=["Apple", "Google", "Microsoft", "Amazon", "Facebook", "Tesla"],
        max_texts=10,
        seed=42,
        logger=logger,
    )
)

grounder = WikipediaGrounder(
    WikipediaGrounderConfig(
        cache_dir=os.getenv("WIKI_CACHE_DIR", "wiki_cache"),
        user_agent="PedroSearchBot/1.0",
        reranker_model=os.getenv("RERANKERS_MODEL", "mixedbread-ai/mxbai-rerank-base-v1"),
        wiki_lang=os.getenv("WIKI_LANG", "en"),
        wiki_search_k=int(os.getenv("WIKI_SEARCH_K", 50)),
        wiki_match_threshold=float(os.getenv("WIKI_MATCH_THRESHOLD", 0.9)),
        wiki_use_deepcat=bool(os.getenv("WIKI_USE_DEEPCAT", False)),
        wiki_query_hints=json.loads(os.getenv("WIKI_QUERY_HINTS", "{}")),
        ground_labels=json.loads(os.getenv("WIKI_GROUND_LABELS", '[]')), # leave empty to ground nothing on wikipedia
        logger=logger,
    )
)

chunker = DocChunker(
    DocChunkerConfig(
        target_bytes=1024,
        tiktoken_encoding=None,  # infer automatically from model
        openai_model_hint=os.getenv("OPENAI_MODEL") if os.getenv("LLM_PROVIDER") == "openai" else None,  # for tokenization
        overlap_frac=0.10,
        logger=logger,
    )
)

validator = JudgeValidator(
    JudgeValidatorConfig(
        model_name=os.getenv("KG_JUDGE_MODEL", "google/flan-t5-xxl"),
        device=None,                      # or pass through env
        action=os.getenv("KG_JUDGE_ACTION", "annotate"),  # 'filter' or 'annotate'
        batch_size=16,
        node_threshold=os.getenv("KG_JUDGE_NODE_THRESHOLD") and float(os.getenv("KG_JUDGE_NODE_THRESHOLD")) or 0.5,
        rel_threshold=os.getenv("KG_JUDGE_REL_THRESHOLD") and float(os.getenv("KG_JUDGE_REL_THRESHOLD")) or 0.5,
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
        disambiguate_labels=["Product", "Organization", "Country", "Government Body", "Person", "Location", "City"],
        include_snippet=False,
        device="cuda",
        threshold=0.85,
        default_labels=[],
        write_yaml=True,
        auto_release_cuda=True,
        logger=logger,
    )
)

neo4j_writer = Neo4jWriter(
    Neo4jWriterConfig(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "test"),
        logger=logger,
    )
)

extractor = ExtractGraph(
    ExtractGraphConfig(
        shard_size=int(os.getenv("KG_EXTRACT_BATCH_SIZE", "8")),
        # keep conservative defaults; bump when you’re ready
        max_concurrency=int(os.getenv("KG_EXTRACT_MAX_CONCURRENCY", "2")),
        rate_limit_rpm=os.getenv("KG_EXTRACT_RPM") and int(os.getenv("KG_EXTRACT_RPM")) or None,
        show_progress=True,
        logger=logger,
    ),
    transformer=transformer,
)

#
# Graph workflow assembly
#

builder = StateGraph(KGState)
builder.add_node("input_loader", input_loader)
builder.add_node("build_docs", doc_builder)
builder.add_node("chunk_docs", chunker)
builder.add_node("extract_graph", extractor)
# validation
builder.add_node("validate_with_judge", validator)
builder.add_node("disambiguate_entities", disambiguator)
builder.add_node("ground_with_wikipedia", grounder)
# io
builder.add_node("write_to_neo4j", neo4j_writer)

builder.add_edge(START, "input_loader")
builder.add_edge("input_loader", "build_docs")
builder.add_edge("build_docs", "chunk_docs")
builder.add_edge("chunk_docs", "extract_graph")
# judge first, then disambiguation, then grounding
builder.add_edge("extract_graph", "validate_with_judge")
builder.add_edge("validate_with_judge", "disambiguate_entities")
builder.add_edge("disambiguate_entities", "ground_with_wikipedia")
builder.add_edge("ground_with_wikipedia", "write_to_neo4j")
#builder.add_edge("disambiguate_entities", "write_to_neo4j")
builder.add_edge("write_to_neo4j", END)

workflow = builder.compile()

#
# Entry point method
#

def run_workflow() -> KGState:
    pipeline_setup: KGState = {}

    
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
    state = run_workflow()
     
    return state.get('judge_results')


if __name__ == "__main__":
    print("Starting KG workflow demo from HF dataset...")
    # Print available RAM, GPU info, etc.
    
    print(f"Torch version: {torch.__version__}")
    if torch.cuda.is_available():
        print("CUDA is available. Device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}, Memory Allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB, Memory Cached: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB, Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA is not available.")
        
    df_results = run_demo_from_hf_dataset()
    
    # Save results to CSV for analysis
    if df_results:
        df = pd.DataFrame(df_results)
        Path("out").mkdir(parents=True, exist_ok=True)
        output_csv = "out/judge_results.csv"
        df.to_csv(output_csv, index=False)
        print(f"Saved judge results to {output_csv} for analysis.")
    
    print("Done!")