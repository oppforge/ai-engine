"""
Chat Agent - Conversational AI assistant for OppForge (Forge AI).
"""

import httpx
import json
from typing import Dict, Any, List, Optional
import logging

from config import GROQ_API_KEY, GROQ_MODEL, GROQ_API_URL, CHAT_TEMPERATURE, MAX_TOKENS, TIMEOUT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatAgent:
    """
    Forge AI - Conversational assistant for opportunity guidance.
    """
    
    def __init__(self):
        self.api_key = GROQ_API_KEY
        self.model = GROQ_MODEL
        
        self.system_prompt = """You are Forge AI, the intelligent assistant for OppForge - a Web3 opportunity discovery platform.

Your role:
- Help users evaluate grants, hackathons, bounties, airdrops, and testnets
- Provide tactical, data-driven advice on winning opportunities
- Be concise, direct, and actionable (2-3 sentences max unless asked for detail)
- Focus on ROI, deadline urgency, and skill match
- Never make up information - if you don't know, say so

Tone: Professional yet friendly, like a knowledgeable Web3 mentor."""

    async def chat(
        self,
        message: str,
        user_profile: Optional[Dict[str, Any]] = None,
        opportunity: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Process chat message and return response.
        
        Args:
            message: User's message
            user_profile: User's profile data for context
            opportunity: Specific opportunity being discussed
            conversation_history: Previous messages in format [{"role": "user/assistant", "content": "..."}]
        
        Returns:
            AI response string
        """
        if not self.api_key:
            return "I'm unable to respond right now. Please configure Groq API key."
        
        try:
            # Build context
            context_parts = []
            
            if user_profile:
                context_parts.append(f"User Profile: {user_profile.get('full_name', 'User')}")
                if user_profile.get('skills'):
                    context_parts.append(f"Skills: {', '.join(user_profile['skills'][:5])}")
                if user_profile.get('preferred_chains'):
                    context_parts.append(f"Preferred Chains: {', '.join(user_profile['preferred_chains'][:3])}")
                context_parts.append(f"Experience: {user_profile.get('experience_level', 'Intermediate')}")
            
            if opportunity:
                context_parts.append(f"\nOpportunity Context:")
                context_parts.append(f"- Title: {opportunity.get('title', 'N/A')}")
                context_parts.append(f"- Category: {opportunity.get('category', 'N/A')}")
                context_parts.append(f"- Chain: {opportunity.get('chain', 'N/A')}")
                context_parts.append(f"- Reward: {opportunity.get('reward_pool', 'N/A')}")
                if opportunity.get('deadline'):
                    context_parts.append(f"- Deadline: {opportunity['deadline']}")
                if opportunity.get('ai_score'):
                    context_parts.append(f"- AI Score: {opportunity['ai_score']}/100")
            
            context = "\n".join(context_parts) if context_parts else ""
            
            # Build messages
            messages = [{"role": "system", "content": self.system_prompt}]
            
            if context:
                messages.append({"role": "system", "content": f"Context:\n{context}"})
            
            # Add conversation history
            if conversation_history:
                messages.extend(conversation_history[-10:])  # Last 10 messages
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            # Call Groq API
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                response = await client.post(
                    GROQ_API_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": CHAT_TEMPERATURE,
                        "max_tokens": MAX_TOKENS
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    logger.error(f"Groq API error: {response.status_code} - {response.text}")
                    return "I encountered an error. Please try again."
                    
        except Exception as e:
            logger.error(f"Chat agent error: {e}")
            return f"I encountered an issue: {str(e)[:100]}"
    
    async def quick_evaluate(self, opportunity: Dict[str, Any]) -> str:
        """Quick evaluation of an opportunity."""
        prompt = f"""Evaluate this opportunity in 2-3 sentences:

Title: {opportunity.get('title', 'N/A')}
Category: {opportunity.get('category', 'N/A')}
Reward: {opportunity.get('reward_pool', 'N/A')}
Deadline: {opportunity.get('deadline', 'N/A')}

Is it worth applying to? Be direct."""
        
        return await self.chat(prompt)
    
    async def suggest_strategy(self, opportunity: Dict[str, Any], user_profile: Dict[str, Any]) -> str:
        """Suggest winning strategy for an opportunity."""
        prompt = "Given my skills and this opportunity, what's the best strategy to win? Give me 3 specific tactics."
        
        return await self.chat(prompt, user_profile=user_profile, opportunity=opportunity)


# Singleton instance
_chat_agent = None

def get_chat_agent() -> ChatAgent:
    """Get or create chat agent instance."""
    global _chat_agent
    if _chat_agent is None:
        _chat_agent = ChatAgent()
    return _chat_agent
