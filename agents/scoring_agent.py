"""
Scoring Agent - Calculates opportunity scores using hybrid AI + ML approach.
"""

import httpx
import json
from typing import Dict, Any, Optional
import logging

from config import GROQ_API_KEY, GROQ_MODEL, GROQ_API_URL, SCORER_TEMPERATURE, MAX_TOKENS, TIMEOUT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScoringAgent:
    """
    Scores opportunities 0-100 using hybrid approach:
    - LLM analysis for quality assessment
    - Rule-based scoring for objective factors
    - User profile matching
    """
    
    def __init__(self):
        self.api_key = GROQ_API_KEY
        self.model = GROQ_MODEL
        
    async def score_opportunity(
        self, 
        opportunity: Dict[str, Any],
        user_profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Score an opportunity with detailed breakdown.
        
        Returns:
            {
                "overall_score": 0-100,
                "breakdown": {
                    "quality_score": 0-100,
                    "skill_match": 0-100,
                    "reward_score": 0-100,
                    "urgency_score": 0-100,
                    "difficulty_match": 0-100
                },
                "reasoning": "Explanation"
            }
        """
        try:
            # 1. Get AI quality assessment
            quality_score, reasoning = await self._assess_quality(opportunity)
            
            # 2. Calculate objective scores
            reward_score = self._calculate_reward_score(opportunity)
            urgency_score = self._calculate_urgency_score(opportunity)
            
            # 3. Calculate user-specific scores
            skill_match = 0
            difficulty_match = 0
            
            if user_profile:
                skill_match = self._calculate_skill_match(opportunity, user_profile)
                difficulty_match = self._calculate_difficulty_match(opportunity, user_profile)
            else:
                # Default scores if no user profile
                skill_match = 50
                difficulty_match = 50
            
            # 4. Weighted overall score
            overall_score = (
                quality_score * 0.30 +
                skill_match * 0.25 +
                reward_score * 0.20 +
                urgency_score * 0.15 +
                difficulty_match * 0.10
            )
            
            return {
                "overall_score": round(overall_score, 1),
                "breakdown": {
                    "quality_score": round(quality_score, 1),
                    "skill_match": round(skill_match, 1),
                    "reward_score": round(reward_score, 1),
                    "urgency_score": round(urgency_score, 1),
                    "difficulty_match": round(difficulty_match, 1)
                },
                "reasoning": reasoning,
                "recommendation": self._get_recommendation(overall_score)
            }
            
        except Exception as e:
            logger.error(f"Error scoring opportunity: {e}")
            return {
                "overall_score": 50,
                "breakdown": {},
                "reasoning": f"Error calculating score: {str(e)}",
                "recommendation": "Unable to assess"
            }
    
    async def _assess_quality(self, opportunity: Dict[str, Any]) -> tuple[float, str]:
        """Use LLM to assess opportunity quality."""
        if not self.api_key:
            return 50.0, "AI assessment unavailable"
        
        prompt = f"""
Assess this Web3 opportunity's quality and legitimacy.

Opportunity:
- Title: {opportunity.get('title', 'N/A')}
- Description: {opportunity.get('description', 'N/A')[:500]}
- Category: {opportunity.get('category', 'N/A')}
- Chain: {opportunity.get('chain', 'N/A')}
- Reward: {opportunity.get('reward_pool', 'N/A')}
- Source: {opportunity.get('source', 'N/A')}

Rate quality (0-100) considering:
- Legitimacy (is this real and safe?)
- Clarity (is the opportunity well-defined?)
- Value proposition (is it worth pursuing?)
- Competition level (how competitive?)

Respond with JSON only:
{{
    "quality_score": 0-100,
    "reasoning": "Brief explanation (2 sentences max)"
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
                            {"role": "system", "content": "You are an expert Web3 opportunity analyst. Respond only with valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": SCORER_TEMPERATURE,
                        "max_tokens": MAX_TOKENS,
                        "response_format": {"type": "json_object"}
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = json.loads(data["choices"][0]["message"]["content"])
                    return float(content.get("quality_score", 50)), content.get("reasoning", "")
                else:
                    logger.warning(f"Groq API error: {response.status_code}")
                    return 50.0, "API error"
                    
        except Exception as e:
            logger.error(f"Quality assessment error: {e}")
            return 50.0, str(e)
    
    def _calculate_reward_score(self, opportunity: Dict[str, Any]) -> float:
        """Calculate score based on reward value."""
        reward_str = str(opportunity.get('reward_pool', '')).lower()
        
        # Extract numeric value from reward string
        import re
        numbers = re.findall(r'[\d,]+', reward_str.replace('$', '').replace('k', '000'))
        
        if not numbers:
            return 40.0  # Unknown reward
        
        try:
            value = int(numbers[0].replace(',', ''))
            
            # Score tiers
            if value >= 100000:
                return 100.0
            elif value >= 50000:
                return 90.0
            elif value >= 20000:
                return 80.0
            elif value >= 10000:
                return 70.0
            elif value >= 5000:
                return 60.0
            elif value >= 1000:
                return 50.0
            else:
                return 40.0
                
        except:
            return 40.0
    
    def _calculate_urgency_score(self, opportunity: Dict[str, Any]) -> float:
        """Calculate score based on deadline proximity."""
        deadline = opportunity.get('deadline')
        
        if not deadline:
            return 50.0  # No deadline = medium urgency
        
        from datetime import datetime, timezone
        
        try:
            if isinstance(deadline, str):
                deadline = datetime.fromisoformat(deadline.replace('Z', '+00:00'))
            
            now = datetime.now(timezone.utc)
            days_left = (deadline - now).days
            
            if days_left < 0:
                return 0.0  # Expired
            elif days_left <= 2:
                return 100.0  # Very urgent
            elif days_left <= 7:
                return 85.0
            elif days_left <= 14:
                return 70.0
            elif days_left <= 30:
                return 55.0
            else:
                return 40.0
                
        except:
            return 50.0
    
    def _calculate_skill_match(
        self, 
        opportunity: Dict[str, Any], 
        user_profile: Dict[str, Any]
    ) -> float:
        """Calculate how well user skills match opportunity requirements."""
        user_skills = set([s.lower() for s in user_profile.get('skills', [])])
        
        required_skills = set([
            s.lower() for s in opportunity.get('required_skills', [])
        ])
        opp_tags = set([t.lower() for t in opportunity.get('tags', [])])
        
        all_required = required_skills.union(opp_tags)
        
        if not all_required:
            return 50.0  # No requirements = medium match
        
        # Calculate overlap
        matches = user_skills.intersection(all_required)
        match_ratio = len(matches) / len(all_required)
        
        # Bonus for extra skills
        bonus = min(len(matches) * 5, 20)
        
        base_score = match_ratio * 80
        final_score = min(100, base_score + bonus)
        
        return float(final_score)
    
    def _calculate_difficulty_match(
        self,
        opportunity: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> float:
        """Match opportunity difficulty to user experience."""
        opp_difficulty = opportunity.get('difficulty', 'Intermediate').lower()
        user_level = user_profile.get('experience_level', 'Intermediate').lower()
        
        # Difficulty matrix
        difficulty_map = {
            'beginner': 1,
            'intermediate': 2,
            'expert': 3,
            'advanced': 3
        }
        
        opp_level = difficulty_map.get(opp_difficulty, 2)
        user_level_num = difficulty_map.get(user_level, 2)
        
        # Perfect match = 100, one level diff = 70, two levels = 40
        diff = abs(opp_level - user_level_num)
        
        if diff == 0:
            return 100.0
        elif diff == 1:
            return 70.0
        else:
            return 40.0
    
    def _get_recommendation(self, score: float) -> str:
        """Get action recommendation based on score."""
        if score >= 85:
            return "Highly Recommended - Apply ASAP"
        elif score >= 70:
            return "Recommended - Strong Match"
        elif score >= 55:
            return "Consider - Decent Fit"
        elif score >= 40:
            return "Review Carefully"
        else:
            return "Not Recommended"


# Singleton instance
_scoring_agent = None

def get_scoring_agent() -> ScoringAgent:
    """Get or create scoring agent instance."""
    global _scoring_agent
    if _scoring_agent is None:
        _scoring_agent = ScoringAgent()
    return _scoring_agent
