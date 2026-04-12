from functools import lru_cache
from pathlib import Path

import pandas as pd

try:
    import chromadb
except ImportError:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


CHROMA_DIR = Path("data/chroma_db")
COLLECTION_NAME = "casino_knowledge"
KNOWLEDGE_DOCS_DIR = Path("data/knowledge_docs")

STRATEGY_KNOWLEDGE = [
    {
        "id": "strategy_poker_edge",
        "title": "Poker rewards decision quality",
        "text": (
            "Poker usually rewards decision quality, patience, bet sizing, and table reading "
            "more than pure-chance games."
        ),
        "source": "knowledge_base",
    },
    {
        "id": "strategy_blackjack_control",
        "title": "Blackjack supports disciplined play",
        "text": (
            "Blackjack can provide a more controlled risk profile when a player avoids chasing "
            "losses and keeps bet sizing disciplined."
        ),
        "source": "knowledge_base",
    },
    {
        "id": "strategy_roulette_randomness",
        "title": "Roulette amplifies randomness",
        "text": (
            "Roulette results are strongly driven by randomness, so short streaks are a weak "
            "signal of repeatable skill."
        ),
        "source": "knowledge_base",
    },
    {
        "id": "strategy_slots_volatility",
        "title": "Slots are highly volatile",
        "text": (
            "Slots usually create the highest volatility and the least room for player skill, "
            "which makes them poor for stable long-term strategy."
        ),
        "source": "knowledge_base",
    },
    {
        "id": "strategy_bankroll",
        "title": "Bankroll management matters",
        "text": (
            "A strong casino strategy balances expected return with bankroll protection. "
            "High average bets become more dangerous when variance is elevated."
        ),
        "source": "knowledge_base",
    },
]


def player_row_to_document(row):
    label = "Skilled" if int(row["is_skilled"]) == 1 else "Lucky"
    profile = "consistent" if float(row["variance"]) < 0.15 else "volatile"

    return {
        "id": f"player_{int(row['player_id'])}",
        "title": f"Historical player {int(row['player_id'])}",
        "text": (
            f"Historical player case labeled {label}. "
            f"Games: {int(row['num_games'])}. "
            f"Win rate: {float(row['win_rate']):.3f}. "
            f"Variance: {float(row['variance']):.3f}. "
            f"Profit: {int(row['profit'])}. "
            f"Average bet: {int(row['avg_bet'])}. "
            f"Streak length: {int(row['streak_length'])}. "
            f"This is a {profile} profile."
        ),
        "source": "dataset",
    }


def split_text(text, chunk_size=260, overlap=50):
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start = max(end - overlap, start + 1)

    return chunks


def chunk_document(document, chunk_size=260, overlap=50):
    chunks = split_text(document["text"], chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        chunks = [document["text"]]

    chunked = []
    for index, chunk in enumerate(chunks):
        chunked.append(
            {
                "id": f"{document['id']}_chunk_{index}",
                "title": document["title"],
                "text": chunk,
                "source": document["source"],
            }
        )
    return chunked


def load_external_knowledge_docs():
    documents = []
    if not KNOWLEDGE_DOCS_DIR.exists():
        return documents

    for path in sorted(KNOWLEDGE_DOCS_DIR.glob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        documents.append(
            {
                "id": f"knowledge_{path.stem}",
                "title": path.stem.replace("_", " ").title(),
                "text": text,
                "source": "knowledge_doc",
            }
        )
    return documents


@lru_cache(maxsize=1)
def load_corpus(dataset_path="data/dataset.csv"):
    df = pd.read_csv(dataset_path)
    documents = []

    for _, row in df.iterrows():
        documents.extend(chunk_document(player_row_to_document(row), chunk_size=70, overlap=12))

    for document in STRATEGY_KNOWLEDGE:
        documents.extend(chunk_document(document, chunk_size=55, overlap=10))

    for document in load_external_knowledge_docs():
        documents.extend(chunk_document(document, chunk_size=110, overlap=18))

    return documents


@lru_cache(maxsize=1)
def get_embedding_model():
    if SentenceTransformer is None:
        return None

    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


def build_player_query(goal, player_data, prediction_label, risk_level):
    return (
        f"{goal}. "
        f"Prediction: {prediction_label}. "
        f"Risk level: {risk_level}. "
        f"Win rate: {player_data['win_rate']}. "
        f"Variance: {player_data['variance']:.2f}. "
        f"Profit: {player_data['profit']}. "
        f"Average bet: {player_data['avg_bet']}. "
        f"Streak: {player_data['streak']}."
    )


def _build_fallback_results(query, documents, top_k):
    query_terms = set(query.lower().split())
    scored = []

    for document in documents:
        doc_terms = set(document["text"].lower().split())
        overlap = len(query_terms & doc_terms)
        scored.append({
            **document,
            "score": round(overlap / max(len(query_terms), 1), 4),
        })

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


@lru_cache(maxsize=1)
def get_chroma_collection(dataset_path="data/dataset.csv"):
    if chromadb is None:
        return None

    model = get_embedding_model()
    if model is None:
        return None

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    documents = load_corpus(dataset_path)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    existing_count = collection.count()

    if existing_count != len(documents):
        if existing_count:
            client.delete_collection(COLLECTION_NAME)
            collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

        ids = [doc["id"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadatas = [
            {
                "title": doc["title"],
                "source": doc["source"],
            }
            for doc in documents
        ]
        embeddings = model.encode(texts, convert_to_numpy=True).tolist()

        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    return collection


def semantic_search(query, top_k=4, dataset_path="data/dataset.csv"):
    collection = get_chroma_collection(dataset_path)

    if collection is None:
        documents = load_corpus(dataset_path)
        return {
            "mode": "fallback_in_memory",
            "results": _build_fallback_results(query, documents, top_k),
        }

    model = get_embedding_model()
    query_embedding = model.encode([query], convert_to_numpy=True).tolist()
    response = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    ids = response.get("ids", [[]])[0]
    documents = response.get("documents", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]

    results = []
    for doc_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        similarity = max(0.0, 1 - float(distance))
        results.append({
            "id": doc_id,
            "title": metadata.get("title", doc_id),
            "text": document,
            "source": metadata.get("source", "vector_store"),
            "score": round(similarity, 4),
        })

    return {
        "mode": "chromadb_sentence_transformers",
        "results": results,
    }
