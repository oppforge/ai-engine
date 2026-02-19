"""
OppForge AI Engine - FastAPI Server
Provides AI-powered opportunity analysis, scoring, and chat.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import logging

from config import AI_ENGINE_HOST, AI_ENGINE_PORT, ENABLE_CHROMADB
from agents.scoring_agent import get_scoring_agent
from agents.chat_agent import get_chat_agent
from agents.risk_agent import get_risk_agent
from agents.classifier_agent import get_classifier_agent

if ENABLE_CHROMADB:
    from vectorstore.chroma_client import get_chroma_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OppForge AI Engine",
    description="AI-powered opportunity analysis and recommendation system",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Request/Response Models
# ==========================================

class OpportunityData(BaseModel):
    id: Optional[str] = None
    title: str
    description: str
    category: Optional[str] = None
    chain: Optional[str] = None
    reward_pool: Optional[str] = None
    deadline: Optional[datetime] = None
    source: Optional[str] = None
    tags: Optional[List[str]] = []
    required_skills: Optional[List[str]] = []
    mission_requirements: Optional[List[str]] = []
    trust_score: int = 70
    url: Optional[str] = None
    is_verified: bool = False


class UserProfile(BaseModel):
    id: Optional[uuid.UUID] = None
    full_name: Optional[str] = None
    username: Optional[str] = None
    skills: List[str] = []
    experience_level: str = "Intermediate"
    preferred_chains: List[str] = []
    preferred_categories: List[str] = []
    xp: int = 0
    level: int = 1
    bio: Optional[str] = None


class ScoreRequest(BaseModel):
    opportunity: OpportunityData
    user_profile: Optional[UserProfile] = None


class ChatRequest(BaseModel):
    message: str
    user_profile: Optional[UserProfile] = None
    opportunity: Optional[OpportunityData] = None
    conversation_history: Optional[List[Dict[str, str]]] = None


class SemanticSearchRequest(BaseModel):
    query: str
    n_results: int = 20
    filters: Optional[Dict[str, Any]] = None


class ClassifyRequest(BaseModel):
    raw_text: str
    source: str = "Unknown"
    model: Optional[str] = None


class RiskRequest(BaseModel):
    opportunity: OpportunityData
    ecosystem: Optional[Dict[str, Any]] = None


# ==========================================
# Health & Status
# ==========================================

@app.get("/")
def root():
    return {
        "service": "OppForge AI Engine",
        "status": "operational",
        "version": "1.0.1",
        "last_updated": "2026-02-19T10:30:00"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "chromadb_enabled": ENABLE_CHROMADB,
        "agents": {
            "scoring": "operational",
            "chat": "operational",
            "risk": "operational",
            "classifier": "operational"
        }
    }


# ==========================================
# AI Endpoints
# ==========================================

@app.post("/ai/score")
async def score_opportunity(request: ScoreRequest):
    """
    Score an opportunity (0-100) with detailed breakdown.
    """
    try:
        agent = get_scoring_agent()
        
        opp_dict = request.opportunity.dict()
        user_dict = request.user_profile.dict() if request.user_profile else None
        
        result = await agent.score_opportunity(opp_dict, user_dict)
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/chat")
async def chat(request: ChatRequest):
    """
    Chat with Forge AI assistant.
    """
    try:
        agent = get_chat_agent()
        
        opp_dict = request.opportunity.dict() if request.opportunity else None
        user_dict = request.user_profile.dict() if request.user_profile else None
        
        response = await agent.chat(
            message=request.message,
            user_profile=user_dict,
            opportunity=opp_dict,
            conversation_history=request.conversation_history
        )
        
        return {
            "success": True,
            "response": response
        }
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/classify")
async def classify(request: ClassifyRequest):
    """
    Classify raw text as opportunity and extract metadata.
    """
    try:
        agent = get_classifier_agent()
        result = await agent.classify(request.raw_text, request.source, request.model)
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/risk-assess")
async def assess_risk(request: RiskRequest):
    """
    Assess opportunity risk and legitimacy.
    """
    try:
        agent = get_risk_agent()
        
        opp_dict = request.opportunity.dict()
        ecosystem_dict = request.ecosystem
        
        result = await agent.assess_risk(opp_dict, ecosystem_dict)
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Risk assessment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# Vector Search Endpoints (ChromaDB)
# ==========================================

if ENABLE_CHROMADB:
    
    @app.post("/ai/semantic-search")
    def semantic_search(request: SemanticSearchRequest):
        """
        Semantic search across opportunities.
        """
        try:
            client = get_chroma_client()
            
            # Sanitize filters: Remove empty dictionaries or nulls that crash ChromaDB
            clean_filters = None
            if request.filters:
                clean_filters = {k: v for k, v in request.filters.items() if v and v != {}}
                if not clean_filters:
                    clean_filters = None
            
            results = client.semantic_search(
                query=request.query,
                n_results=request.n_results,
                filters=clean_filters
            )
            
            return {
                "success": True,
                "results": results,
                "count": len(results)
            }
            
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @app.post("/ai/match-score")
    def calculate_match_score(request: ScoreRequest):
        """
        Calculate semantic match score between user and opportunity.
        """
        try:
            client = get_chroma_client()
            
            opp_dict = request.opportunity.dict()
            user_dict = request.user_profile.dict() if request.user_profile else {}
            
            score = client.calculate_match_score(user_dict, opp_dict)
            
            return {
                "success": True,
                "match_score": score
            }
            
        except Exception as e:
            logger.error(f"Match score error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @app.get("/ai/vector-stats")
    def vector_stats():
        """
        Get ChromaDB statistics.
        """
        try:
            client = get_chroma_client()
            count = client.count()
            
            return {
                "success": True,
                "total_opportunities": count,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384
            }
            
        except Exception as e:
            logger.error(f"Vector stats error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# Utility Endpoints
# ==========================================

@app.get("/ai/agents")
def list_agents():
    """List all available AI agents."""
    return {
        "agents": [
            {
                "name": "Scoring Agent",
                "description": "Scores opportunities 0-100 with detailed breakdown",
                "endpoint": "/ai/score"
            },
            {
                "name": "Chat Agent",
                "description": "Forge AI conversational assistant",
                "endpoint": "/ai/chat"
            },
            {
                "name": "Classifier Agent",
                "description": "Classifies and extracts metadata from raw text",
                "endpoint": "/ai/classify"
            },
            {
                "name": "Risk Agent",
                "description": "Assesses opportunity safety and legitimacy",
                "endpoint": "/ai/risk-assess"
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    import os
    
    # Priority: Environment Var (Railway) > Config File > Default
    port = int(os.getenv("PORT", AI_ENGINE_PORT))
    
    logger.info(f"🚀 Starting OppForge AI Engine on {AI_ENGINE_HOST}:{port}")
    
    uvicorn.run(
        "main:app",
        host=AI_ENGINE_HOST,
        port=port,
        reload=False, # Disable reload in production
        log_level="info"
    )
