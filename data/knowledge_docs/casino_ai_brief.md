Vegas AI is a Sin City themed decision-support system for analyzing whether a casino player's outcomes look more skill-driven or luck-driven.

The project combines classical machine learning, retrieval-augmented generation, and optional agent orchestration frameworks to produce decisions that are both predictive and explainable.

Core goals:
- classify player profiles as likely skilled or likely lucky
- estimate volatility and projected downside
- recommend a safer allocation across blackjack, poker, roulette, and slots
- retrieve relevant historical cases and strategic guidance
- explain the reasoning behind the system output

Why the problem fits the theme:
Las Vegas is built on high-variance outcomes, streak psychology, table choice, and bankroll decisions. This project turns those patterns into an intelligent analytics workflow that fits the Sin City theme while staying grounded in measurable signals.

AI and ML components used in the project:
- feature engineered random forest classifier for player-type prediction
- sentence-transformer embeddings for semantic understanding
- ChromaDB local vector store for retrieval
- framework-based LLM orchestration using LangGraph, CrewAI, LangChain, and LlamaIndex

Responsible gambling layer:
The system does not just optimize returns. It also detects elevated behavioral risk, high volatility, and loss exposure so that the recommendation can favor safer play when warning signals are present.

Explainability layer:
Each analysis should be backed by feature-level drivers, benchmark metrics, retrieved evidence, and a summary that translates the technical result into plain language for judges and users.
