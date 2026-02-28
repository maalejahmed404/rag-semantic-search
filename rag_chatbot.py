"""
rag_chatbot.py
RAG chatbot with:
- LLM Metadata Extraction: extracts ingredient name from query for filtering
- HyDE: generates hypothetical answer for better embedding match
- Dual-Language Query Translation: EN + FR for cross-language retrieval
- ChromaDB METADATA filtering on single 'ingredient' field
- Cross-Encoder Reranking: reranks top candidates for content precision
"""
import os
import json
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI

load_dotenv()

# ── Config ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "chroma_db")
COLLECTION_NAME = "ingredients_rag"
TOP_K = 3
PRE_FETCH_K = 15

# Lightning AI LLM
LLM_CLIENT = OpenAI(
    base_url="https://lightning.ai/api/v1/",
    api_key=os.environ.get("LIGHTNING_API_KEY", "YOUR_API_KEY_HERE"),
)
LLM_MODEL = "openai/gpt-5-nano"

# ── Load Models (once) ──
print("Loading embedding model ...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Loading cross-encoder reranker ...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── Connect to ChromaDB ──
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)
print(f"Connected to ChromaDB collection '{COLLECTION_NAME}' ({collection.count()} entries).")

# ── Build ingredient list from DB (single 'ingredient' field now) ──
all_metas = collection.get(limit=collection.count(), include=["metadatas"])
KNOWN_INGREDIENTS = sorted(set(
    m.get("ingredient", "").strip()
    for m in all_metas["metadatas"]
    if m.get("ingredient", "").strip()
))
print(f"Known ingredients: {len(KNOWN_INGREDIENTS)}\n")


# ═══════════════════════════════════════════════════════════
# STEP 0: Extract metadata (ingredient) from query
# ═══════════════════════════════════════════════════════════

def extract_ingredient(query: str):
    """Use fast matching + LLM fallback to extract ingredient from query.
    Returns a list of matching ingredient names, or None."""
    
    # First: try fast exact/substring match (no LLM needed)
    query_lower = query.lower()
    for ing in KNOWN_INGREDIENTS:
        ing_lower = ing.lower()
        # Clean versions: remove "TDS", "pdf", parentheses
        clean = ing_lower.replace("tds", "").replace("pdf", "").replace("(1)", "").strip()
        clean_words = [w for w in clean.split() if len(w) > 2]
        
        if ing_lower in query_lower:
            return [ing]
        if clean_words and all(w in query_lower for w in clean_words):
            return [ing]
    
    # If no fast match: ask the LLM
    ing_list = "\n".join(f"- {ing}" for ing in KNOWN_INGREDIENTS)
    prompt = (
        "From the following user query, extract the ingredient name if one is mentioned.\n"
        "Here is the list of known ingredients:\n"
        f"{ing_list}\n\n"
        "User query: " + query + "\n\n"
        "RULES:\n"
        "- If the query mentions a specific ingredient from the list above, return EXACTLY that ingredient name.\n"
        "- If the query mentions a general ingredient type (like 'alpha-amylase', 'xylanase', 'lipase', 'transglutaminase', 'glucose oxidase', 'ascorbic acid'), return ALL matching ingredients separated by '|'.\n"
        "- If no specific ingredient is mentioned, return exactly: NONE\n"
        "- Return ONLY the ingredient name(s), nothing else."
    )
    
    completion = LLM_CLIENT.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    )
    response = completion.choices[0].message.content.strip()
    
    if response.upper() == "NONE" or len(response) > 500:
        return None
    
    # Parse response - could be single ingredient or multiple separated by |
    extracted = [r.strip() for r in response.split("|")]
    valid = [e for e in extracted if e in KNOWN_INGREDIENTS]
    
    if valid:
        return valid
    return None


# ═══════════════════════════════════════════════════════════
# STEP 1: Query Translation
# ═══════════════════════════════════════════════════════════

def translate_query(query: str):
    """Use the LLM to translate the query into both English and French."""
    prompt = (
        "Translate the following query into BOTH English and French. "
        "Return ONLY two lines, nothing else:\n"
        "EN: <english translation>\n"
        "FR: <french translation>\n\n"
        f"Query: {query}"
    )
    completion = LLM_CLIENT.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    )
    response = completion.choices[0].message.content.strip()
    
    en_query = query
    fr_query = query
    
    for line in response.split("\n"):
        line = line.strip()
        if line.upper().startswith("EN:"):
            en_query = line[3:].strip()
        elif line.upper().startswith("FR:"):
            fr_query = line[3:].strip()
    
    return en_query, fr_query


# ═══════════════════════════════════════════════════════════
# STEP 2: HyDE
# ═══════════════════════════════════════════════════════════

def generate_hypothetical_doc(query: str):
    """HyDE: Generate a hypothetical technical sheet excerpt."""
    prompt = (
        "You are a technical writer for bakery ingredient data sheets. "
        "Write a SHORT (2-3 sentences max) technical data sheet excerpt that would answer this question. "
        "Use the same style as a real product specification sheet (dosage in ppm, ingredient names, regulatory limits, etc). "
        "Write in BOTH English and French in the same response. "
        "Do NOT say 'I don't know'. Just write a plausible technical excerpt.\n\n"
        f"Question: {query}"
    )
    completion = LLM_CLIENT.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    )
    return completion.choices[0].message.content.strip()


# ═══════════════════════════════════════════════════════════
# STEP 3: Retrieval with METADATA filtering
# ═══════════════════════════════════════════════════════════

def retrieve_multi(query: str):
    """
    Full retrieval pipeline:
    0. Extract ingredient for metadata filtering
    1. Generate hypothetical document (HyDE)
    2. Translate query to EN + FR
    3. Retrieve Top-15 with each vector (with optional ingredient filter)
    4. If filtered results are too few, fallback to unfiltered
    5. Merge, deduplicate, sort by best score, return Top-3
    """
    # Step 0: Extract ingredient
    print("Extracting metadata from query...")
    ingredient_filter = extract_ingredient(query)
    
    if ingredient_filter:
        print(f"  Filter: {len(ingredient_filter)} ingredients: {', '.join(ingredient_filter[:5])}")
    else:
        print("  Filter: None (general query)")
    print()

    # Step 1: HyDE
    print("Generating hypothetical document (HyDE)...")
    hyde_doc = generate_hypothetical_doc(query)
    print(f"  HyDE: {hyde_doc[:150]}...")
    print()

    # Step 2: Translate
    print("Translating query...")
    en_query, fr_query = translate_query(query)
    print(f"  EN: {en_query}")
    print(f"  FR: {fr_query}")
    print()

    # Step 3: Embed all vectors
    hyde_emb = embed_model.encode(hyde_doc, normalize_embeddings=True).tolist()
    en_emb = embed_model.encode(en_query, normalize_embeddings=True).tolist()
    fr_emb = embed_model.encode(fr_query, normalize_embeddings=True).tolist()

    all_queries = [
        (hyde_emb, "HyDE"),
        (en_emb, "EN"),
        (fr_emb, "FR"),
    ]

    # Step 4: Build ChromaDB METADATA filter (on 'ingredient' field)
    where_filter = None
    if ingredient_filter:
        if len(ingredient_filter) == 1:
            where_filter = {"ingredient": ingredient_filter[0]}
        else:
            where_filter = {"ingredient": {"$in": ingredient_filter}}

    # Retrieve with filter
    seen = {}
    for emb, lang in all_queries:
        try:
            results = collection.query(
                query_embeddings=[emb],
                n_results=PRE_FETCH_K,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            print(f"  (ChromaDB filter error: {e})")
            results = collection.query(
                query_embeddings=[emb],
                n_results=PRE_FETCH_K,
                include=["documents", "metadatas", "distances"]
            )
        
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            sim = round(1 - dist, 4)
            key = doc[:100]
            if key not in seen or sim > seen[key]["score"]:
                seen[key] = {"text": doc, "metadata": meta, "score": sim, "matched_lang": lang}

    # Step 5: If filter gave too few results, also do unfiltered retrieval
    if where_filter and len(seen) < TOP_K:
        print("  (Filter too restrictive, adding unfiltered results)")
        for emb, lang in all_queries:
            results = collection.query(
                query_embeddings=[emb],
                n_results=PRE_FETCH_K,
                include=["documents", "metadatas", "distances"]
            )
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                sim = round(1 - dist, 4)
                key = doc[:100]
                if key not in seen or sim > seen[key]["score"]:
                    seen[key] = {"text": doc, "metadata": meta, "score": sim, "matched_lang": lang}

    # Step 6: Cross-encoder reranking using English query
    candidates = list(seen.values())
    if len(candidates) > 1:
        pairs = [(en_query, c["text"]) for c in candidates]
        rerank_scores = reranker.predict(pairs)
        for c, score in zip(candidates, rerank_scores):
            c["rerank_score"] = float(score)
        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    else:
        ranked = candidates

    return ranked[:TOP_K]


# ═══════════════════════════════════════════════════════════
# STEP 4: LLM Answer Generation
# ═══════════════════════════════════════════════════════════

def generate_answer(query: str, passages: list):
    """Send the retrieved passages + question to Lightning AI LLM."""
    context = ""
    for i, p in enumerate(passages, 1):
        context += f"Fragment {i} (Score: {p['score']}):\n\"{p['text']}\"\nIngrédient: {p['metadata'].get('ingredient', 'N/A')}\n\n"

    system_prompt = (
        "Tu es un assistant expert en boulangerie et pâtisserie. "
        "Réponds à la question de l'utilisateur EN UTILISANT UNIQUEMENT le contexte fourni ci-dessous. "
        "Si le contexte ne contient pas la réponse, dis-le clairement. Ne fais pas d'hallucinations."
    )
    user_prompt = f"Question: {query}\n\nContexte:\n{context}\nRéponds de manière claire et structurée."

    completion = LLM_CLIENT.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ],
    )
    return completion.choices[0].message.content


def chat(query: str):
    """Full RAG pipeline: extract → HyDE → translate → filter → retrieve → generate."""
    print(f"\n{'='*60}")
    print(f"Question: {query}")
    print(f"{'='*60}\n")

    passages = retrieve_multi(query)

    print("Retrieved Passages:")
    print("-" * 40)
    for i, p in enumerate(passages, 1):
        print(f"Résultat {i}")
        print(f"Texte : \"{p['text']}\"")
        print(f"Score : {p['score']}")
        print(f"Ingrédient : {p['metadata'].get('ingredient', 'N/A')}")
        print(f"(matched via {p['matched_lang']} query)")
        print()

    print("-" * 40)
    print("LLM Answer:")
    answer = generate_answer(query, passages)
    print(answer)
    print(f"{'='*60}\n")
    return answer


if __name__ == "__main__":
    print("=" * 60)
    print("  RAG Chatbot - Assistance Boulangerie")
    print("  Type 'quit' to exit")
    print("=" * 60)

    while True:
        query = input("\nVotre question: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("Au revoir!")
            break
        if not query:
            continue
        chat(query)
