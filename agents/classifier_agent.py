"""
Classifier Agent - Categorizes and extracts metadata from raw opportunities.
"""

import httpx
import json
from typing import Dict, Any, List
import logging

from config import GROQ_API_KEY, GROQ_MODEL, GROQ_API_URL, CLASSIFIER_TEMPERATURE, MAX_TOKENS, TIMEOUT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassifierAgent:
    """
    Classifies opportunities and extracts structured metadata.
    """
    
    def __init__(self):
        self.api_key = GROQ_API_KEY
        self.model = GROQ_MODEL
    
    async def classify(self, raw_text: str, source: str = "Unknown") -> Dict[str, Any]:
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

If it's a legitimate opportunity (grant, hackathon, bounty, airdrop, testnet), extract:
1. Category (must be one of: Grant, Hackathon, Bounty, Airdrop, Testnet)
2. Clear action-oriented title
3. Blockchain/chain involved
4. Required skills (programming languages, domains)
5. Estimated reward amount
6. Deadline (YYYY-MM-DD format or null)
7. Difficulty level

Rules:
- EXCLUDE: Generic news, partnerships without CTA, jobs, marketing hype
- INCLUDE: Only opportunities with clear call-to-action (apply, submit, register, join)

Respond with JSON only:
{{
    "is_opportunity": true/false,
    "category": "Grant",
    "title": "Clear Actionable Title",
    "chain": "Ethereum",
    "required_skills": ["Solidity", "React"],
    "estimated_reward": "$50,000",
    "deadline": "2026-03-15",
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
                        "model": self.model,
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
                    result = json.loads(data["choices"][0]["message"]["content"])
                    return result
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
