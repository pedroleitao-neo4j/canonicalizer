# extractor.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable
import logging

from langchain_community.graphs import Neo4jGraph

@dataclass(frozen=True)
class Neo4jWriterConfig:
    uri: str
    user: str
    password: str
    logger: Optional[logging.Logger] = None
    
class Neo4jWriter:
    """
    Writes graph documents into a Neo4j database.
    """
    def __init__(self, cfg: Neo4jWriterConfig):
        self.cfg = cfg
        self.logger = cfg.logger or logging.getLogger("neo4j_writer")
        try:
            self.neo4j_driver = Neo4jGraph(
                url=self.cfg.uri,
                username=self.cfg.user,
                password=self.cfg.password
            )
            self.neo4j_driver.query(
                """
                CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
                FOR (n:Entity) REQUIRE n.id IS UNIQUE
                """
            )
        except Exception as e:
            self.logger.error("Failed to initialize Neo4j driver: %s", e)
            raise
        
    #
    # LangGraph node
    #
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Writing graph documents to Neo4j at %s", self.cfg.uri)
        graph_docs: List[Any] = state.get("graph_docs") or []
        
        if not graph_docs:
            self.logger.warning("No graph documents provided to write.")
            return state
        
        self.neo4j_driver.add_graph_documents(graph_docs, include_source=True)
        
        self.logger.info("Successfully processed %d graph documents.", len(graph_docs))
        return state