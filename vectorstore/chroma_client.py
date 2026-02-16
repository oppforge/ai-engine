"""
ChromaDB Vector Store Client for AI Engine
Enhanced version with better semantic search and scoring.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import numpy as np

from config import CHROMA_PATH, CHROMA_COLLECTION, EMBEDDING_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaDBClient:
    """
    Manages ChromaDB for semantic search and opportunity embeddings.
    """
    
    def __init__(self):
        # Ensure data directory exists
        Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Load embedding model
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"✅ Embedding model loaded. Dim: {self.embedder.get_sentence_embedding_dimension()}")
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=CHROMA_COLLECTION)
            logger.info(f"✅ Loaded existing collection: {CHROMA_COLLECTION} ({self.collection.count()} items)")
        except:
            self.collection = self.client.create_collection(
                name=CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"✅ Created new collection: {CHROMA_COLLECTION}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for text."""
        return self.embedder.encode(text).tolist()
    
    def add_opportunity(self, opportunity_id: str, opportunity_data: Dict[str, Any]):
        """Add or update an opportunity in the vector store."""
        # Create searchable text by combining key fields
        searchable_text = " ".join(filter(None, [
            opportunity_data.get("title", ""),
            opportunity_data.get("description", "")[:500],  # Limit description
            opportunity_data.get("category", ""),
            opportunity_data.get("chain", ""),
            " ".join(opportunity_data.get("tags", [])),
            " ".join(opportunity_data.get("required_skills", []))
        ]))
        
        # Generate embedding
        embedding = self.embed_text(searchable_text)
        
        # Store in ChromaDB
        self.collection.upsert(
            ids=[opportunity_id],
            embeddings=[embedding],
            metadatas=[{
                "title": opportunity_data.get("title", "")[:100],
                "category": opportunity_data.get("category", ""),
                "chain": opportunity_data.get("chain", ""),
                "ai_score": float(opportunity_data.get("ai_score", 0)),
            }],
            documents=[searchable_text]
        )
        
        logger.debug(f"Added opportunity to ChromaDB: {opportunity_id}")
    
    def semantic_search(
        self, 
        query: str, 
        n_results: int = 20,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Semantic search for opportunities."""
        query_embedding = self.embed_text(query)
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filters
        )
        
        # Format results
        opportunities = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                opportunities.append({
                    "id": results['ids'][0][i],
                    "similarity_score": float(1 - results['distances'][0][i]),
                    "metadata": results['metadatas'][0][i],
                    "text": results['documents'][0][i]
                })
        
        return opportunities
    
    def calculate_match_score(
        self, 
        user_profile: Dict[str, Any], 
        opportunity_data: Dict[str, Any]
    ) -> float:
        """Calculate semantic match score between user and opportunity."""
        # Create user profile text
        user_text = " ".join(filter(None, [
            " ".join(user_profile.get("skills", [])),
            user_profile.get("bio", ""),
            " ".join(user_profile.get("preferred_chains", [])),
            user_profile.get("experience_level", "")
        ]))
        
        # Create opportunity text
        opp_text = " ".join(filter(None, [
            opportunity_data.get("title", ""),
            opportunity_data.get("description", "")[:300],
            " ".join(opportunity_data.get("required_skills", [])),
            opportunity_data.get("chain", "")
        ]))
        
        if not user_text or not opp_text:
            return 50.0  # Default score
        
        # Generate embeddings
        user_emb = self.embedder.encode(user_text)
        opp_emb = self.embedder.encode(opp_text)
        
        # Calculate cosine similarity
        similarity = np.dot(user_emb, opp_emb) / (np.linalg.norm(user_emb) * np.linalg.norm(opp_emb))
        
        # Convert to 0-100 score
        match_score = float(max(0, min(100, (similarity + 1) * 50)))  # Map [-1,1] to [0,100]
        
        return match_score
    
    def bulk_add_opportunities(self, opportunities: List[Dict[str, Any]]):
        """Bulk add multiple opportunities."""
        if not opportunities:
            return
        
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        for opp in opportunities:
            opp_id = str(opp.get("id"))
            
            # Create searchable text
            searchable_text = " ".join(filter(None, [
                opp.get("title", ""),
                opp.get("description", "")[:500],
                opp.get("category", ""),
                opp.get("chain", ""),
                " ".join(opp.get("tags", [])),
            ]))
            
            ids.append(opp_id)
            embeddings.append(self.embed_text(searchable_text))
            metadatas.append({
                "title": opp.get("title", "")[:100],
                "category": opp.get("category", ""),
                "chain": opp.get("chain", ""),
            })
            documents.append(searchable_text)
        
        # Bulk upsert
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        
        logger.info(f"✅ Bulk added {len(opportunities)} opportunities to ChromaDB")
    
    def count(self) -> int:
        """Get total number of opportunities in vector store."""
        return self.collection.count()


# Singleton instance
_chroma_client = None

def get_chroma_client() -> ChromaDBClient:
    """Get or create ChromaDB client instance."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = ChromaDBClient()
    return _chroma_client
