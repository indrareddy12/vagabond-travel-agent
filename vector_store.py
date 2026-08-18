import os
import numpy as np
from typing import List, Dict, Any, Tuple
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from config import logger

DB_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(DB_DIR, "faiss_index")

class DeterministicEmbeddings(Embeddings):
    """
    Deterministic local embedding generator for testing environment consistency.
    Generates 1536-dimensional normalized vectors seeded from input text strings.
    """
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            seed = sum(ord(c) * (i + 1) for i, c in enumerate(text)) % 999983
            rng = np.random.default_rng(seed)
            vector = rng.standard_normal(1536)
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            results.append(vector.tolist())
        return results

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# Pre-populated facts for target destinations
CITY_FACTS = {
    "tokyo": (
        "Tokyo, the capital city of Japan, is a vibrant metropolis that blends ultra-modern "
        "skyscrapers with historic Shinto shrines and temples. Renowned for its exceptional "
        "public transport, world-class culinary scene (holding the most Michelin stars of any city), "
        "and distinctive neighborhoods like Shibuya, Shinjuku, and Akihabara. Key landmarks include "
        "the historic Senso-ji Temple in Asakusa, the Tokyo Skytree, and the majestic Imperial Palace. "
        "Tokyo is a global hub for technology, fashion, pop culture, and business."
    ),
    "paris": (
        "Paris, the capital of France, is a global center for art, fashion, gastronomy, and culture. "
        "Its 19th-century cityscape is crisscrossed by wide boulevards and the River Seine. Famous landmarks "
        "include the iconic Eiffel Tower, the Gothic Notre-Dame Cathedral, the Louvre Museum (housing the Mona Lisa), "
        "and the Arc de Triomphe at the end of the Champs-Élysées. Known as the 'City of Light' and the capital of "
        "romance, Paris is celebrated for its cafe culture, haute couture, and classical architecture."
    ),
    "new york": (
        "New York City (NYC) comprises 5 boroughs sitting where the Hudson River meets the Atlantic Ocean. "
        "At its core is Manhattan, a densely populated borough that's among the world's major commercial, "
        "financial, and cultural centers. Iconic sites include skyscrapers like the Empire State Building and "
        "sprawling Central Park. Broadway theater is staged in neon-lit Times Square. NYC is famous for its "
        "diverse neighborhoods, high-energy lifestyle, yellow cabs, and landmarks like the Statue of Liberty."
    )
}

def initialize_vector_store() -> FAISS:
    """
    Creates and populates the local FAISS index with city facts if not already present on disk.
    """
    embeddings = DeterministicEmbeddings()
    
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            logger.info("Loading existing FAISS index from disk...")
            db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            return db
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}. Re-creating index...")

    documents = []
    for city, fact in CITY_FACTS.items():
        doc = Document(
            page_content=fact,
            metadata={"city": city, "title": f"Facts about {city.title()}"}
        )
        documents.append(doc)
        
    logger.info("Initializing new local FAISS vector store...")
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(FAISS_INDEX_PATH)
    logger.info(f"FAISS index successfully saved to {FAISS_INDEX_PATH}")
    return db

def check_city_in_store(city_name: str, db: FAISS) -> Tuple[bool, str]:
    """
    Queries the vector store for matching city records.
    Returns (is_present, content_summary).
    """
    city_name_lower = city_name.strip().lower()
    
    try:
        # String match verification
        for key in CITY_FACTS.keys():
            if key in city_name_lower or city_name_lower in key:
                return True, CITY_FACTS[key]
                
        # Cosine distance semantic lookup
        results_with_scores = db.similarity_search_with_score(city_name, k=1)
        if results_with_scores:
            doc, score = results_with_scores[0]
            # Since vector embeddings are unit length, L2 distance score < 0.8 is threshold
            if score < 0.8:
                matched_city = doc.metadata.get("city", "")
                logger.info(f"Matched city '{matched_city}' in FAISS with score {score:.4f}")
                return True, doc.page_content
                
    except Exception as e:
        logger.error(f"Error querying local store: {e}")
        
    return False, ""
