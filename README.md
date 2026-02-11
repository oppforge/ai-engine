# OppForge AI Engine

AI/ML processing engine for OppForge — LangChain agents, opportunity scoring, chat assistant, and vector search.

## Tech Stack

- **Agent Framework**: LangChain + LangGraph
- **Local LLM**: Ollama (Llama 3 / Mistral) — **100% free**
- **Fallback LLM**: Groq API (free tier)
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Vector DB**: ChromaDB
- **ML Scoring**: scikit-learn

## Getting Started

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Pull LLM model via Ollama
ollama pull llama3:8b

# Start engine
uvicorn main:app --reload --port 8001
```

## Project Structure

```
ai-engine/
├── config.py                  # AI engine configuration
├── agents/
│   ├── classifier_agent.py    # Classify opportunity type/chain
│   ├── scoring_agent.py       # Score opportunities 0-100
│   ├── chat_agent.py          # Forge AI conversational assistant
│   ├── proposal_agent.py      # Generate proposal drafts
│   ├── strategy_agent.py      # Generate farming strategies
│   └── enrichment_agent.py    # Enrich raw data with context
├── models/
│   ├── scoring_model.py       # scikit-learn scoring model
│   ├── embeddings.py          # Sentence-Transformers
│   └── feature_extractor.py   # Feature engineering
├── prompts/                   # Prompt templates per agent
├── pipelines/
│   ├── ingestion_pipeline.py  # Raw data → structured
│   ├── scoring_pipeline.py    # Structured → scored
│   ├── dedup_pipeline.py      # Deduplication logic
│   └── enrichment_pipeline.py # Add metadata
└── vectorstore/
    ├── chroma_client.py       # ChromaDB connection
    └── embedder.py            # Document embedding
```

## Agents

| Agent | Purpose |
|-------|---------|
| **Classifier** | Categorizes raw opportunities (grant/airdrop/hackathon/bounty) and identifies chain |
| **Scorer** | Generates 0-100 scores using hybrid LLM + ML approach |
| **Chat** | Powers the Forge AI conversational assistant |
| **Proposal** | Generates full proposal/application drafts |
| **Strategy** | Creates step-by-step airdrop farming strategies |
| **Enrichment** | Adds protocol metadata, TVL data, and context |

## Scoring Algorithm

```
Score = weighted sum of:
  - Reward value (20%)
  - Skill match (25%)
  - Deadline proximity (10%)
  - Difficulty match (15%)
  - Competition level (10%)
  - Source reliability (10%)
  - Chain preference (10%)
```

## Environment Variables

```bash
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3:8b
GROQ_API_KEY=your-groq-key     # Free tier fallback
CHROMA_PATH=./data/chromadb
```
