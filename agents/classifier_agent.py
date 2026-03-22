"""
Classifier Agent - Categorizes and extracts metadata from raw opportunities.
"""

import httpx
import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ValidationError
import logging

from config import GROQ_API_KEY, GROQ_MODEL, GROQ_API_URL, CLASSIFIER_TEMPERATURE, MAX_TOKENS, TIMEOUT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassificationOutput(BaseModel):
    is_opportunity: bool = Field(description="Whether the text represents a valid actionable Web3 opportunity")
    category: str = Field(description="Must be exactly: Grant, Hackathon, Bounty, Airdrop, or Testnet")
    title: str = Field(description="Action-oriented title of the opportunity")
    chain: str = Field(description="Blockchain network, e.g. Ethereum, Solana, Multi-chain")
    required_skills: List[str] = Field(description="List of skills required")
    estimated_reward: str = Field(description="Value of the reward, e.g. '$50,000' or 'Unknown'")
    deadline: Optional[str] = Field(description="Deadline in YYYY-MM-DD format, or null")
    difficulty: str = Field(description="Beginner, Intermediate, or Expert")
    confidence: int = Field(description="Confidence score from 0-100")

class ClassifierAgent:
    """
    Classifies opportunities and extracts structured metadata.
    """
    
    def __init__(self):
        self.api_key = GROQ_API_KEY
        self.model = GROQ_MODEL
    
    async def classify(self, raw_text: str, source: str = "Unknown", model: str = None) -> Dict[str, Any]:
        """
        Classify raw text into structured opportunity data.
        
        Returns:
            {
                "is_opportunity": bool,
                "category": "Grant" | "Hackathon" | "Bounty" | "Airdrop" | "Testnet",
                "title": str,
                "chain": str,
                "required_skills": List[str],
                "estimated_reward": str,
                "deadline": str (ISO format or null),
                "difficulty": "Beginner" | "Intermediate" | "Expert"
            }
        """
        if not self.api_key:
            return self._fallback_classification(raw_text)
        
        prompt = f"""Analyze this text and determine if it's an actionable Web3 opportunity.

Text: {raw_text[:1000]}
Source: {source}

Web3 opportunities include: grants, hackathons, bounties, airdrops, testnets, ambassador programs.
A BOUNTY is any task where someone can earn a reward by completing work — building, writing, designing, researching, auditing, or creating content. Bounties are ALWAYS valid opportunities even if they don't say "apply" or "register".

Extract the following:
1. Category (must be one of: Grant, Hackathon, Bounty, Airdrop, Testnet, Ambassador)
2. Clear action-oriented title
3. Blockchain/chain involved
4. Required skills: Focus on technical Web3 skills (e.g., Solidity, Rust, Foundry, ZK-Rollups, Cairo, Ethers.js, React, Node.js, Content Writing, Data Analysis). Be exhaustive for matchmaking.
5. Estimated reward amount
6. Deadline (YYYY-MM-DD format or null)
7. Difficulty level

Rules:
- INCLUDE as Bounty: Any task with a reward for building, writing, designing, creating, researching, or auditing (e.g. "Build a dashboard", "Write a thread", "Create a tutorial")
- INCLUDE as Grant: Funding opportunities with application/proposal process
- INCLUDE as Hackathon: Time-boxed building competitions
- INCLUDE as Airdrop/Testnet: Network participation incentives
- EXCLUDE: Pure news articles, partnership announcements with no reward, generic blog posts, job listings without payment
- When source is Superteam, DoraHacks, ETHGlobal, HackQuest, Devfolio, Questbook — always mark is_opportunity=true

Respond with JSON only:
{{
    "is_opportunity": true/false,
    "category": "Bounty",
    "title": "Clear Actionable Title",
    "chain": "Solana",
    "required_skills": ["React", "Solana", "Data Visualization"],
    "estimated_reward": "$5,000 USDC",
    "deadline": "2026-04-10",
    "difficulty": "Intermediate",
    "confidence": 0-100
}}
"""
        
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                response = await client.post(
                    GROQ_API_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model or self.model,
                        "messages": [
                            {"role": "system", "content": "You are a Web3 opportunity classifier. Always respond with valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": CLASSIFIER_TEMPERATURE,
                        "max_tokens": MAX_TOKENS,
                        "response_format": {"type": "json_object"}
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    
                    # Force strict Pydantic Structured Output Validation
                    try:
                        validated_data = ClassificationOutput.model_validate_json(content)
                        return validated_data.model_dump()
                    except ValidationError as ve:
                        logger.error(f"Pydantic Validation failed, invalid AI schema: {ve}")
                        return self._fallback_classification(raw_text)
                else:
                    logger.error(f"Classification API error: {response.status_code}")
                    return self._fallback_classification(raw_text)
                    
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return self._fallback_classification(raw_text)
    
    def _fallback_classification(self, raw_text: str) -> Dict[str, Any]:
        """Rule-based fallback classification."""
        text_lower = raw_text.lower()
        
        # Check if it's an opportunity
        action_keywords = ['apply', 'register', 'submit', 'join', 'grant', 'hackathon', 'bounty', 'airdrop']
        is_opportunity = any(kw in text_lower for kw in action_keywords)
        
        if not is_opportunity:
            return {"is_opportunity": False}
        
        # Determine category
        category = "Grant"  # Default
        if 'hackathon' in text_lower:
            category = "Hackathon"
        elif 'bounty' in text_lower or 'bug' in text_lower:
            category = "Bounty"
        elif 'airdrop' in text_lower:
            category = "Airdrop"
        elif 'testnet' in text_lower:
            category = "Testnet"
        
        # Extract chain
        chains = ['ethereum', 'solana', 'arbitrum', 'optimism', 'polygon', 'base', 'sui', 'aptos']
        chain = "Multi-chain"
        for c in chains:
            if c in text_lower:
                chain = c.capitalize()
                break
        
        # Basic title (first 60 chars)
        title = raw_text[:60].strip()
        if not title:
            title = f"{category} Opportunity"
        
        return {
            "is_opportunity": True,
            "category": category,
            "title": title,
            "chain": chain,
            "required_skills": [],
            "estimated_reward": "Unknown",
            "deadline": None,
            "difficulty": "Intermediate",
            "confidence": 50
        }


# Singleton instance
_classifier_agent = None

def get_classifier_agent() -> ClassifierAgent:
    """Get or create classifier agent instance."""
    global _classifier_agent
    if _classifier_agent is None:
        _classifier_agent = ClassifierAgent()
    return _classifier_agent
