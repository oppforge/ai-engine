"""
Risk Assessment Agent - Evaluates opportunity safety and legitimacy.
"""

import httpx
import json
from typing import Dict, Any
import logging
from datetime import datetime
import re

from config import GROQ_API_KEY, GROQ_MODEL, GROQ_API_URL, RISK_TEMPERATURE, MAX_TOKENS, TIMEOUT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskAgent:
    """
    Assesses opportunity risk and legitimacy.
    """
    
    def __init__(self):
        self.api_key = GROQ_API_KEY
        self.model = GROQ_MODEL
    
    async def assess_risk(
        self,
        opportunity: Dict[str, Any],
        ecosystem: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive risk assessment.
        
        Returns:
            {
                "risk_score": 0-100 (higher = safer),
                "risk_level": "LOW" | "MEDIUM" | "HIGH",
                "flags": ["flag1", "flag2"],
                "details": {...}
            }
        """
        try:
            # Calculate individual risk scores
            scam_score = self._detect_scam_patterns(opportunity)
            source_score = self._assess_source_reliability(opportunity)
            url_score = self._check_url_safety(opportunity)
            legitimacy_score = await self._verify_legitimacy(opportunity, ecosystem)
            reward_score = self._assess_reward_realism(opportunity)
            
            # Weighted overall risk score
            risk_score = (
                scam_score * 0.30 +
                source_score * 0.25 +
                legitimacy_score * 0.20 +
                url_score * 0.15 +
                reward_score * 0.10
            )
            
            # Determine risk level
            if risk_score >= 75:
                risk_level = "LOW"
            elif risk_score >= 50:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"
            
            # Collect flags
            flags = []
            if scam_score < 50:
                flags.append("Potential scam patterns detected")
            if source_score < 50:
                flags.append("Unverified source")
            if url_score < 50:
                flags.append("Suspicious URL")
            if reward_score < 50:
                flags.append("Unrealistic reward amount")
            if legitimacy_score < 50:
                flags.append("Legitimacy concerns")
            
            return {
                "risk_score": round(risk_score, 1),
                "risk_level": risk_level,
                "flags": flags,
                "verified_source": source_score >= 75,
                "details": {
                    "scam_detection": round(scam_score, 1),
                    "source_reliability": round(source_score, 1),
                    "url_safety": round(url_score, 1),
                    "legitimacy": round(legitimacy_score, 1),
                    "reward_realism": round(reward_score, 1)
                }
            }
            
        except Exception as e:
            logger.error(f"Risk assessment error: {e}")
            return {
                "risk_score": 50.0,
                "risk_level": "MEDIUM",
                "flags": ["Unable to complete risk assessment"],
                "verified_source": False,
                "details": {}
            }
    
    def _detect_scam_patterns(self, opportunity: Dict[str, Any]) -> float:
        """Detect common scam red flags."""
        score = 100.0
        text = (opportunity.get('title', '') + ' ' + opportunity.get('description', '')).lower()
        
        # Red flag keywords
        scam_keywords = [
            'guaranteed', 'risk-free', '100% profit', 'double your',
            'limited time', 'act now', 'secret', 'exclusive insider',
            'instant riches', 'no experience needed', 'get rich quick',
            'make money fast', 'financial freedom', 'x1000', 'moon'
        ]
        
        for keyword in scam_keywords:
            if keyword in text:
                score -= 15
        
        # Check for excessive punctuation (!!!!, ????)
        if text.count('!') > 5 or text.count('?') > 5:
            score -= 10
        
        # Check for all caps (SHOUTING)
        if any(word.isupper() and len(word) > 5 for word in text.split()):
            score -= 10
        
        # Check for suspicious emojis overuse
        emoji_count = len(re.findall(r'[🚀💰💸🤑💎🔥⚡️]', text))
        if emoji_count > 10:
            score -= 15
        
        return max(0, score)
    
    def _assess_source_reliability(self, opportunity: Dict[str, Any]) -> float:
        """Assess source trustworthiness."""
        source = opportunity.get('source', '').lower()
        url = opportunity.get('url', '').lower()
        
        # Trusted sources
        trusted_sources = [
            'gitcoin', 'devpost', 'dorahacks', 'immunefi',
            'ethereum', 'solana', 'arbitrum', 'optimism', 'polygon'
        ]
        
        # Check source
        for trusted in trusted_sources:
            if trusted in source or trusted in url:
                return 95.0
        
        # Platform domains
        trusted_domains = [
            'github.com', 'ethereum.org', 'solana.com', 'arbitrum.io',
            'optimism.io', 'polygon.technology', 'gitcoin.co'
        ]
        
        for domain in trusted_domains:
            if domain in url:
                return 90.0
        
        # Has official domain
        if any(tld in url for tld in ['.org', '.io', '.com', '.network']):
            return 60.0
        
        # Unknown source
        return 40.0
    
    def _check_url_safety(self, opportunity: Dict[str, Any]) -> float:
        """Basic URL safety checks."""
        url = opportunity.get('url', '').lower()
        
        if not url:
            return 50.0  # No URL = neutral
        
        score = 100.0
        
        # Check for HTTPS
        if not url.startswith('https://'):
            score -= 30
        
        # Check for suspicious TLDs
        suspicious_tlds = ['.xyz', '.tk', '.ml', '.ga', '.cf', '.gq']
        if any(tld in url for tld in suspicious_tlds):
            score -= 40
        
        # Check for IP address instead of domain
        if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url):
            score -= 50
        
        # Check for excessive subdomains
        domain_parts = url.replace('https://', '').replace('http://', '').split('/')[0].split('.')
        if len(domain_parts) > 4:
            score -= 20
        
        return max(0, score)
    
    async def _verify_legitimacy(
        self,
        opportunity: Dict[str, Any],
        ecosystem: Optional[Dict[str, Any]]
    ) -> float:
        """Use AI to verify legitimacy."""
        if not self.api_key:
            return 60.0  # Neutral if no AI
        
        # Quick legitimacy check
        if ecosystem:
            # Check if URL matches ecosystem domain
            opp_url = opportunity.get('url', '').lower()
            eco_website = ecosystem.get('official_website', '').lower()
            
            if eco_website and eco_website.replace('https://', '').replace('www.', '') in opp_url:
                return 95.0  # Strong match
        
        # AI verification
        prompt = f"""Is this a legitimate Web3 opportunity or potentially fraudulent?

Title: {opportunity.get('title', '')}
Description: {opportunity.get('description', '')[:300]}
Source: {opportunity.get('source', '')}
URL: {opportunity.get('url', '')}

Respond with JSON only:
{{
    "is_legitimate": true/false,
    "confidence": 0-100,
    "reasoning": "brief explanation"
}}
"""
        
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(
                    GROQ_API_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "You are a Web3 security expert. Respond only with valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": RISK_TEMPERATURE,
                        "max_tokens": 300,
                        "response_format": {"type": "json_object"}
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = json.loads(data["choices"][0]["message"]["content"])
                    
                    if content.get("is_legitimate"):
                        return float(content.get("confidence", 70))
                    else:
                        return max(0, 100 - float(content.get("confidence", 70)))
            
            return 60.0
            
        except Exception as e:
            logger.error(f"Legitimacy verification error: {e}")
            return 60.0
    
    def _assess_reward_realism(self, opportunity: Dict[str, Any]) -> float:
        """Check if reward amount is realistic."""
        reward_str = str(opportunity.get('reward_pool', '')).lower()
        
        # Extract numbers
        numbers = re.findall(r'[\d,]+', reward_str.replace('$', '').replace('k', '000').replace('m', '000000'))
        
        if not numbers:
            return 60.0  # Unknown = neutral
        
        try:
            value = int(numbers[0].replace(',', ''))
            
            # Realistic ranges for different categories
            category = opportunity.get('category', '').lower()
            
            if category == 'grant':
                if value > 10_000_000:  # $10M+ grant is suspicious
                    return 30.0
                return 90.0
            
            elif category == 'hackathon':
                if value > 5_000_000:  # $5M+ hackathon rare but possible
                    return 40.0
                return 90.0
            
            elif category == 'bounty':
                if value > 2_000_000:  # $2M+ bounty very rare
                    return 50.0
                return 90.0
            
            elif category == 'airdrop':
                if value > 100_000_000:  # $100M+ airdrop suspicious
                    return 20.0
                return 70.0
            
            else:
                # General check
                if value > 50_000_000:
                    return 30.0
                return 70.0
                
        except:
            return 60.0


# Singleton instance
_risk_agent = None

def get_risk_agent() -> RiskAgent:
    """Get or create risk agent instance."""
    global _risk_agent
    if _risk_agent is None:
        _risk_agent = RiskAgent()
    return _risk_agent
