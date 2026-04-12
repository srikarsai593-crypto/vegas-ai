# Vegas AI

A Streamlit app that analyzes casino player behavior using feature engineering, an ML classifier, a strategy engine, ChromaDB-backed semantic search, and LLM explanations. It supports a custom multi-agent flow plus optional LangGraph, CrewAI, LangChain, and LlamaIndex orchestration modes.

## Why this project fits the theme

Vegas AI is designed for the HackNite 2026 Sin City theme. It models casino-style volatility, streaks, bankroll pressure, and game selection decisions to identify whether performance looks more skill-driven or luck-driven.

## Rubric-aligned highlights

- Problem statement and idea: casino intelligence dashboard for classifying player behavior and recommending safer strategy.
- AI/ML implementation: feature-engineered Random Forest classifier, confidence calibration, risk heuristics, and LLM-generated reasoning.
- System design: chunked document ingestion, ChromaDB retrieval, embeddings, semantic search, and optional multi-agent orchestration.
- Code quality: modular separation across `utils/agent.py`, `utils/semantic_search.py`, `utils/framework_agents.py`, and model training code.
- Documentation and explainability: benchmark metrics, confusion matrix, feature-driver explanations, and clear setup instructions.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Optional environment variables:

- `GROQ_API_KEY`: enables LLM explanations
- `GROQ_MODEL`: optional override for the Groq model ID. Defaults to `llama-3.3-70b-versatile`

If `GROQ_API_KEY` is not set, the app falls back to deterministic summaries.
If `CrewAI` dependencies are unavailable or incompatible in the current environment, the app now marks that framework as unavailable and falls back safely.

## Deployment

This repo is prepared for Streamlit-style deployment.

- Entry point: `app.py`
- Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- Python runtime: `3.11`

### Streamlit Community Cloud

1. Push this repo to GitHub.
2. Create a new Streamlit app pointing to `app.py`.
3. Add `GROQ_API_KEY` in app secrets if you want LLM output.

### Render or similar platforms

Use the start command from the `Procfile` and Python `3.11`.

## Notes

- `plotly` is required by the dashboard UI.
- Semantic search now uses a local persistent ChromaDB vector store with `sentence-transformers` plus chunked knowledge documents and historical cases.
- The app supports five orchestration modes: `Custom`, `LangGraph`, `CrewAI`, `LangChain`, and `LlamaIndex`.
- If an optional framework is unavailable, the app falls back to the custom multi-agent system.
- The Streamlit UI includes a sidebar control center, live simulation, dashboard metrics, strategy and risk tabs, semantic matches, and a leaderboard.
- The app now includes benchmark evaluation metrics, confusion-matrix style reporting, and feature-level explanation signals for the predicted class.
- Benchmarking now includes 5-fold cross-validation, hold-out metrics, a calibration curve, and a short limitations section to make evaluation more realistic and credible.
- The app does not require LLM access to boot successfully.
