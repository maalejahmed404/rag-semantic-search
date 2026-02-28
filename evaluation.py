"""
evaluation.py
Comprehensive RAG Evaluation Benchmark
- 100 queries generated from actual chunk data with STRICT ground truth
- Retrieval metrics: Hit Rate@K, MRR@K, Precision@K, Recall@K, NDCG@K
- STRICT relevance: exact ingredient match + content match (not just keyword)
- Context Precision: are top results for the CORRECT ingredient?
- LLM Faithfulness evaluation on a subset
- Results saved to evaluation_results.txt
"""
import json
import time
import math
import os
import random
import numpy as np
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from openai import OpenAI

# ── Config ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_FILE = os.path.join(SCRIPT_DIR, "all_extracted_chunks_merged.json")
DB_PATH = os.path.join(SCRIPT_DIR, "chroma_db")
COLLECTION_NAME = "ingredients_rag"
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "evaluation_results.txt")
TOP_K_VALUES = [1, 3, 5, 10]

# Lightning AI LLM (free tier: 700M+ tokens)
# Set your API key: set LIGHTNING_API_KEY=your-key-here
LLM_CLIENT = OpenAI(
    base_url="https://lightning.ai/api/v1/",
    api_key=os.environ.get("LIGHTNING_API_KEY", "YOUR_API_KEY_HERE"),
)
LLM_MODEL = "lightning-ai/gpt-oss-120b"

# ── Load resources ──
print("Loading resources...")
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)
print(f"Loaded {len(chunks)} raw chunks, {collection.count()} DB entries.\n")

# Build set of known ingredients
all_metas = collection.get(limit=collection.count(), include=["metadatas"])
KNOWN_INGREDIENTS = sorted(set(
    m.get("ingredient", "").strip()
    for m in all_metas["metadatas"]
    if m.get("ingredient", "").strip()
))
print(f"Known ingredients: {len(KNOWN_INGREDIENTS)}\n")


# ═══════════════════════════════════════════════════════════
# STEP 1: Generate 100 queries with STRICT ground truth
# ═══════════════════════════════════════════════════════════

def generate_benchmark_queries(chunks):
    """Generate 100 diverse queries at MULTIPLE DIFFICULTY LEVELS.
    
    DIFFICULTY TIERS:
    - EASY (40%): Direct questions with exact ingredient name
    - MEDIUM (30%): Partial names, abbreviations, indirect product references
    - HARD (20%): Natural language, problem-solving, no exact ingredient name
    - ADVERSARIAL (10%): Cross-ingredient comparison, confusion queries
    
    Each query has:
    - query_en / query_fr: the question in EN and FR
    - expected_ingredient: the EXACT ingredient name that MUST match
    - ground_truth_text: the exact text that should be in the retrieved doc
    - category: query category for per-category analysis
    - difficulty: easy/medium/hard/adversarial
    """
    queries = []
    seen_keys = set()
    
    # Build per-ingredient index for harder query generation
    ingredient_data = {}
    for idx, c in enumerate(chunks):
        text = c.get("chunk text", c.get("row text", "")).replace("\n", " ").strip()
        while "  " in text:
            text = text.replace("  ", " ")
        if len(text) < 10:
            continue
        section = c.get("section title", "").lower()
        ingredients = c.get("ingredients", [])
        for ing in ingredients:
            if ing not in ingredient_data:
                ingredient_data[ing] = []
            ingredient_data[ing].append({
                "text": text, "section": section, "chunk_index": idx, "chunk": c
            })
    
    # ═══════════════════════════════════════
    # TIER 1: EASY — exact ingredient name, direct question
    # ═══════════════════════════════════════
    for idx, c in enumerate(chunks):
        text = c.get("chunk text", c.get("row text", "")).replace("\n", " ").strip()
        while "  " in text:
            text = text.replace("  ", " ")
        if len(text) < 10:
            continue
        section = c.get("section title", "").lower()
        ingredients = c.get("ingredients", [])
        if not ingredients:
            continue
        
        for ing in ingredients:
            key = q_en = q_fr = cat = None
            
            if section.startswith("dosage") and "ppm" in text.lower():
                key = f"dosage_{ing}"
                q_en = f"What is the dosage of {ing}?"
                q_fr = f"Quel est le dosage de {ing} ?"
                cat = "dosage"
            elif section.startswith("effective material") and len(text) > 20:
                key = f"effective_material_{ing}"
                q_en = f"What is {ing} made from? What organism produces it?"
                q_fr = f"De quoi est fait {ing} ? Quel organisme le produit ?"
                cat = "effective_material"
            elif section.startswith("application") and len(text) > 30:
                key = f"application_{ing}"
                q_en = f"What is {ing} used for in bakery?"
                q_fr = f"A quoi sert {ing} en boulangerie ?"
                cat = "application"
            elif section.startswith("function") and len(text) > 30:
                key = f"function_{ing}"
                q_en = f"What is the function of {ing}?"
                q_fr = f"Quelle est la fonction de {ing} ?"
                cat = "function"
            elif section.startswith("activity") and any(ch.isdigit() for ch in text):
                key = f"activity_{ing}"
                q_en = f"What is the enzyme activity of {ing}?"
                q_fr = f"Quelle est l'activité enzymatique de {ing} ?"
                cat = "activity"
            elif section.startswith("storage") and len(text) > 20:
                key = f"storage_{ing}"
                q_en = f"How should {ing} be stored?"
                q_fr = f"Comment conserver {ing} ?"
                cat = "storage"
            elif "package" in section and len(text) > 10:
                key = f"packaging_{ing}"
                q_en = f"What is the packaging of {ing}?"
                q_fr = f"Quel est l'emballage de {ing} ?"
                cat = "packaging"
            elif "product description" in section and len(text) > 20:
                key = f"product_desc_{ing}"
                q_en = f"What type of enzyme is {ing}?"
                q_fr = f"Quel type d'enzyme est {ing} ?"
                cat = "product_description"
            elif "table type" in c and "allergen" in c.get("row title", "").lower():
                key = f"allergen_{ing}"
                q_en = f"What allergens does {ing} contain?"
                q_fr = f"Quels allergènes contient {ing} ?"
                cat = "allergens"
            elif "dosages recommandés" in section and "table type" in c:
                row = c.get("row title", "")
                if row and "dosage" not in row.lower():
                    key = f"aa_dosage_{row}_{ing}"
                    q_en = f"What is the {ing} dosage for {row}?"
                    q_fr = f"Quel est le dosage de {ing} pour {row} ?"
                    cat = "aa_dosage_table"
            elif "conversion" in section and "table type" in c:
                row = c.get("row title", "")
                if row:
                    key = f"aa_conversion_{row}_{ing}"
                    q_en = f"How much {ing} for {row} of flour?"
                    q_fr = f"Combien de {ing} pour {row} de farine ?"
                    cat = "aa_conversion"
            elif section.startswith("propriétés") and len(text) > 50:
                key = f"properties_{ing}"
                q_en = f"What are the main properties of {ing}?"
                q_fr = f"Quelles sont les propriétés principales de {ing} ?"
                cat = "properties"
            elif section.startswith("points importants") and len(text) > 30:
                key = f"points_{ing}"
                q_en = f"What are the important points about {ing}?"
                q_fr = f"Quels sont les points importants de {ing} ?"
                cat = "important_points"
            elif section.startswith("limitations") and len(text) > 30:
                key = f"limitations_{ing}"
                q_en = f"What are the limitations of {ing}?"
                q_fr = f"Quelles sont les limitations de {ing} ?"
                cat = "limitations"
            elif "alternatives" in section and "table type" in c:
                alt = c.get("row title", "")
                if alt:
                    key = f"alternatives_{alt}_{ing}"
                    q_en = f"What are the advantages of {alt} as alternative to {ing}?"
                    q_fr = f"Quels sont les avantages de {alt} comme alternative au {ing} ?"
                    cat = "alternatives"
            elif "statut légal" in section and len(text) > 20:
                key = f"legal_{ing}"
                q_en = f"What is the legal status of {ing}?"
                q_fr = f"Quel est le statut légal de {ing} ?"
                cat = "legal"
            elif "mode d'emploi" in section and len(text) > 30:
                key = f"mode_emploi_{ing}"
                q_en = f"How to use {ing} in production?"
                q_fr = f"Comment utiliser {ing} en production ?"
                cat = "mode_emploi"
            elif "recommandations" in section and len(text) > 50:
                key = f"recommendations_{ing}"
                q_en = f"What are the recommendations for {ing}?"
                q_fr = f"Quelles sont les recommandations pour {ing} ?"
                cat = "recommendations"
            
            if key and key not in seen_keys:
                seen_keys.add(key)
                queries.append({
                    "query_en": q_en, "query_fr": q_fr,
                    "expected_ingredient": ing,
                    "ground_truth_text": text,
                    "expected_section": cat, "category": cat,
                    "difficulty": "easy", "chunk_index": idx,
                })
    
    # ═══════════════════════════════════════
    # TIER 2: MEDIUM — partial names, abbreviations, no "TDS"/"BVZyme" prefix
    # ═══════════════════════════════════════
    for ing, data_list in ingredient_data.items():
        short = ing.replace("TDS ", "").replace("BVZyme ", "").replace("BVzyme ", "").strip()
        if short == ing:
            continue
        for d in data_list:
            s = d["section"]
            text = d["text"]
            key = cat = q_en = q_fr = None
            if s.startswith("dosage") and "ppm" in text.lower():
                key = f"med_dosage_{ing}"
                q_en = f"How much {short} should I add to dough?"
                q_fr = f"Combien de {short} dois-je ajouter à la pâte ?"
                cat = "dosage"
            elif s.startswith("function") and len(text) > 30:
                key = f"med_function_{ing}"
                q_en = f"What does {short} do to the bread?"
                q_fr = f"Que fait {short} au pain ?"
                cat = "function"
            elif s.startswith("application") and len(text) > 30:
                key = f"med_application_{ing}"
                q_en = f"Which products can I use {short} in?"
                q_fr = f"Dans quels produits puis-je utiliser {short} ?"
                cat = "application"
            elif s.startswith("storage") and len(text) > 20:
                key = f"med_storage_{ing}"
                q_en = f"At what temperature should {short} be kept?"
                q_fr = f"À quelle température garder {short} ?"
                cat = "storage"
            elif s.startswith("activity") and any(ch.isdigit() for ch in text):
                key = f"med_activity_{ing}"
                q_en = f"How many units per gram for {short}?"
                q_fr = f"Combien d'unités par gramme pour {short} ?"
                cat = "activity"
            elif "product description" in s and len(text) > 20:
                key = f"med_desc_{ing}"
                q_en = f"Tell me about {short}, what kind of enzyme?"
                q_fr = f"Parlez-moi de {short}, quel type d'enzyme ?"
                cat = "product_description"
            if key and key not in seen_keys:
                seen_keys.add(key)
                queries.append({
                    "query_en": q_en, "query_fr": q_fr,
                    "expected_ingredient": ing,
                    "ground_truth_text": text,
                    "expected_section": cat, "category": cat,
                    "difficulty": "medium", "chunk_index": d["chunk_index"],
                })
    
    # ═══════════════════════════════════════
    # TIER 3: HARD — natural language, problem-solving, NO ingredient name
    # ═══════════════════════════════════════
    for ing, data_list in ingredient_data.items():
        for d in data_list:
            s = d["section"]
            text = d["text"]
            key = cat = q_en = q_fr = None
            if s.startswith("function") and "gluten" in text.lower():
                key = f"hard_gluten_{ing}"
                q_en = "Which enzyme improves gluten strength and gas retention?"
                q_fr = "Quel enzyme améliore la force du gluten et la rétention de gaz ?"
                cat = "function"
            elif s.startswith("function") and "volume" in text.lower():
                key = f"hard_volume_{ing}"
                q_en = "My bread has low volume, which enzyme increases oven spring?"
                q_fr = "Mon pain a un faible volume, quel enzyme augmente le développement au four ?"
                cat = "function"
            elif s.startswith("application") and "biscuit" in text.lower():
                key = f"hard_biscuit_{ing}"
                q_en = "I need an enzyme for biscuit production, what do you recommend?"
                q_fr = "J'ai besoin d'un enzyme pour la production de biscuits, que recommandez-vous ?"
                cat = "application"
            elif s.startswith("application") and "croissant" in text.lower():
                key = f"hard_croissant_{ing}"
                q_en = "Which enzyme is suitable for croissant and puff pastry?"
                q_fr = "Quel enzyme convient pour les croissants et la pâte feuilletée ?"
                cat = "application"
            elif s.startswith("function") and "fresh" in text.lower():
                key = f"hard_fresh_{ing}"
                q_en = "How can I extend the shelf life and freshness of my bread?"
                q_fr = "Comment prolonger la durée de conservation et la fraîcheur de mon pain ?"
                cat = "function"
            elif s.startswith("function") and "soft" in text.lower():
                key = f"hard_soft_{ing}"
                q_en = "I want softer crumb texture, what enzyme should I use?"
                q_fr = "Je veux une mie plus moelleuse, quel enzyme utiliser ?"
                cat = "function"
            elif s.startswith("effective material") and "xylanase" in text.lower():
                key = f"hard_xylanase_{ing}"
                q_en = "Do you have any xylanase-based product?"
                q_fr = "Avez-vous un produit à base de xylanase ?"
                cat = "effective_material"
            elif s.startswith("effective material") and "lipase" in text.lower():
                key = f"hard_lipase_{ing}"
                q_en = "I'm looking for a lipase enzyme for bakery"
                q_fr = "Je cherche une enzyme lipase pour la boulangerie"
                cat = "effective_material"
            elif s.startswith("effective material") and "amylase" in text.lower():
                key = f"hard_amylase_{ing}"
                q_en = "Which products contain amylase?"
                q_fr = "Quels produits contiennent de l'amylase ?"
                cat = "effective_material"
            if key and key not in seen_keys:
                seen_keys.add(key)
                queries.append({
                    "query_en": q_en, "query_fr": q_fr,
                    "expected_ingredient": ing,
                    "ground_truth_text": text,
                    "expected_section": cat, "category": cat,
                    "difficulty": "hard", "chunk_index": d["chunk_index"],
                })
    
    # ═══════════════════════════════════════
    # TIER 4: ADVERSARIAL — cross-ingredient comparison, confusion
    # ═══════════════════════════════════════
    all_ings = list(ingredient_data.keys())
    random.seed(42)
    for i in range(min(15, len(all_ings) - 1)):
        ing1 = all_ings[i]
        ing2 = all_ings[(i + 7) % len(all_ings)]
        if ing1 == ing2:
            continue
        s1 = {d["section"].split()[0] for d in ingredient_data[ing1] if d["section"]}
        s2 = {d["section"].split()[0] for d in ingredient_data[ing2] if d["section"]}
        common = s1 & s2
        if "dosage" in common:
            dd = [d for d in ingredient_data[ing1] if d["section"].startswith("dosage")]
            if dd:
                key = f"adv_dosage_{ing1}_{ing2}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    queries.append({
                        "query_en": f"Compare dosage of {ing1} vs {ing2}, which needs more?",
                        "query_fr": f"Comparez le dosage de {ing1} et {ing2}, lequel nécessite plus ?",
                        "expected_ingredient": ing1,
                        "ground_truth_text": dd[0]["text"],
                        "expected_section": "dosage", "category": "dosage",
                        "difficulty": "adversarial", "chunk_index": dd[0]["chunk_index"],
                    })
        if "function" in common:
            fd = [d for d in ingredient_data[ing2] if d["section"].startswith("function")]
            if fd:
                key = f"adv_func_{ing1}_{ing2}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    queries.append({
                        "query_en": f"Is {ing2} better than {ing1} for improving texture?",
                        "query_fr": f"Est-ce que {ing2} est meilleur que {ing1} pour la texture ?",
                        "expected_ingredient": ing2,
                        "ground_truth_text": fd[0]["text"],
                        "expected_section": "function", "category": "function",
                        "difficulty": "adversarial", "chunk_index": fd[0]["chunk_index"],
                    })
    
    # ── Balance to ~100 with difficulty distribution ──
    target = {"easy": 40, "medium": 25, "hard": 20, "adversarial": 15}
    selected = []
    random.seed(42)
    for diff, n in target.items():
        pool = [q for q in queries if q["difficulty"] == diff]
        random.shuffle(pool)
        selected.extend(pool[:n])
    if len(selected) < 100:
        remaining = [q for q in queries if q not in selected]
        random.shuffle(remaining)
        selected.extend(remaining[:100 - len(selected)])
    return selected[:100]


# ═══════════════════════════════════════════════════════════
# STEP 2: Retrieval Evaluation (STRICT)
# ═══════════════════════════════════════════════════════════

def strict_match(retrieved_doc, retrieved_meta, query_data):
    """STRICT relevance check: 
    1. The retrieved doc's ingredient MUST match the expected ingredient
    2. The content must overlap with ground truth
    
    Returns: (ingredient_match, content_match, full_match)
    """
    expected_ing = query_data["expected_ingredient"]
    retrieved_ing = retrieved_meta.get("ingredient", "")
    
    # Ingredient match: exact
    ing_match = (retrieved_ing == expected_ing)
    
    # Content match: check if ground truth text overlaps with retrieved doc
    gt = query_data["ground_truth_text"].lower().replace("\n", " ").strip()
    # Remove common prefixes that aren't in the embedded doc
    for prefix in ["dosage", "activity", "function", "application", "effective material",
                    "product description", "storage", "package:", "packaging"]:
        if gt.startswith(prefix):
            gt_content = gt[len(prefix):].strip()
            if gt_content:
                gt = gt_content
            break
    
    doc = retrieved_doc.lower()
    
    content_match = False
    
    # 1. Direct substring: is the ground truth text (or core of it) in the doc?
    if len(gt) <= 30:
        # Short text: check if entire GT appears in doc
        if gt in doc:
            content_match = True
        else:
            # Try with cleaned/key parts (e.g., "5-40 ppm" from "Dosage 5-40 ppm")
            # Extract numeric parts as key tokens
            key_tokens = [w for w in gt.split() if any(ch.isdigit() for ch in w)]
            if key_tokens and all(t in doc for t in key_tokens):
                content_match = True
    
    # 2. Sliding window for longer texts
    if not content_match and len(gt) > 15:
        window = min(25, len(gt))
        min_len = min(10, len(gt) - 1)
        for i in range(0, min(len(gt), 150), 5):
            snippet = gt[i:i+window]
            if len(snippet) >= min_len and snippet in doc:
                content_match = True
                break
    
    # 3. Fallback: check if the original GT (with prefix) is in the doc
    if not content_match:
        full_gt = query_data["ground_truth_text"].lower().replace("\n", " ").strip()
        if len(full_gt) <= 50 and full_gt in doc:
            content_match = True
        elif len(full_gt) > 50:
            # Check first 40 chars
            if full_gt[:40] in doc:
                content_match = True
    
    return ing_match, content_match, (ing_match and content_match)


def evaluate_retrieval(queries, top_k=10):
    """Evaluate retrieval with STRICT metrics.
    
    Evaluates:
    A) Unfiltered retrieval (pure semantic search, EN+FR)
    B) Filtered retrieval (metadata filter on ingredient)
    C) Filtered + Reranked (cross-encoder reranking after filtering)
    """
    results_unfiltered = []
    results_filtered = []
    results_reranked = []
    
    for i, q in enumerate(queries):
        # Embed both languages
        en_emb = model.encode(q["query_en"], normalize_embeddings=True).tolist()
        fr_emb = model.encode(q["query_fr"], normalize_embeddings=True).tolist()
        
        # ── A) UNFILTERED retrieval ──
        en_res = collection.query(query_embeddings=[en_emb], n_results=top_k, 
                                   include=["documents", "metadatas", "distances"])
        fr_res = collection.query(query_embeddings=[fr_emb], n_results=top_k, 
                                   include=["documents", "metadatas", "distances"])
        
        seen = {}
        for docs, metas, dists in [
            (en_res["documents"][0], en_res["metadatas"][0], en_res["distances"][0]),
            (fr_res["documents"][0], fr_res["metadatas"][0], fr_res["distances"][0]),
        ]:
            for doc, meta, dist in zip(docs, metas, dists):
                sim = 1 - dist
                key = doc[:80]
                if key not in seen or sim > seen[key][1]:
                    seen[key] = (doc, sim, meta)
        
        ranked = sorted(seen.values(), key=lambda x: x[1], reverse=True)
        
        relevance = []
        for doc, sim, meta in ranked[:top_k]:
            ing_match, content_match, full_match = strict_match(doc, meta, q)
            relevance.append({
                "doc": doc[:150], "sim": round(sim, 4),
                "ingredient": meta.get("ingredient", ""),
                "ing_match": ing_match, "content_match": content_match,
                "full_match": full_match,
            })
        
        results_unfiltered.append({
            "query_id": i + 1,
            "query": q["query_fr"],
            "category": q["category"],
            "expected_ingredient": q["expected_ingredient"],
            "ground_truth": q["ground_truth_text"][:100],
            "results": relevance,
        })
        
        # ── B) FILTERED retrieval (metadata filter) ──
        where_filter = {"ingredient": q["expected_ingredient"]}
        try:
            en_res_f = collection.query(query_embeddings=[en_emb], n_results=top_k,
                                         where=where_filter,
                                         include=["documents", "metadatas", "distances"])
            fr_res_f = collection.query(query_embeddings=[fr_emb], n_results=top_k,
                                         where=where_filter,
                                         include=["documents", "metadatas", "distances"])
        except Exception:
            en_res_f = en_res
            fr_res_f = fr_res
        
        seen_f = {}
        for docs, metas, dists in [
            (en_res_f["documents"][0], en_res_f["metadatas"][0], en_res_f["distances"][0]),
            (fr_res_f["documents"][0], fr_res_f["metadatas"][0], fr_res_f["distances"][0]),
        ]:
            for doc, meta, dist in zip(docs, metas, dists):
                sim = 1 - dist
                key = doc[:80]
                if key not in seen_f or sim > seen_f[key][1]:
                    seen_f[key] = (doc, sim, meta)
        
        ranked_f = sorted(seen_f.values(), key=lambda x: x[1], reverse=True)
        
        relevance_f = []
        for doc, sim, meta in ranked_f[:top_k]:
            ing_match, content_match, full_match = strict_match(doc, meta, q)
            relevance_f.append({
                "doc": doc[:150], "sim": round(sim, 4),
                "ingredient": meta.get("ingredient", ""),
                "ing_match": ing_match, "content_match": content_match,
                "full_match": full_match,
            })
        
        results_filtered.append({
            "query_id": i + 1,
            "query": q["query_fr"],
            "category": q["category"],
            "expected_ingredient": q["expected_ingredient"],
            "ground_truth": q["ground_truth_text"][:100],
            "results": relevance_f,
        })
        
        # ── C) FILTERED + RERANKED retrieval ──
        # Take ALL candidates from filtered retrieval, rerank with cross-encoder
        candidates_f = list(seen_f.values())  # (doc, sim, meta)
        if len(candidates_f) > 1:
            pairs = [(q["query_en"], doc) for doc, sim, meta in candidates_f]
            rerank_scores = reranker.predict(pairs)
            scored = [(doc, float(rs), meta) for (doc, sim, meta), rs in zip(candidates_f, rerank_scores)]
            ranked_r = sorted(scored, key=lambda x: x[1], reverse=True)
        else:
            ranked_r = [(doc, sim, meta) for doc, sim, meta in candidates_f]
        
        relevance_r = []
        for doc, score, meta in ranked_r[:top_k]:
            ing_match, content_match, full_match = strict_match(doc, meta, q)
            relevance_r.append({
                "doc": doc[:150], "sim": round(score, 4),
                "ingredient": meta.get("ingredient", ""),
                "ing_match": ing_match, "content_match": content_match,
                "full_match": full_match,
            })
        
        results_reranked.append({
            "query_id": i + 1,
            "query": q["query_fr"],
            "category": q["category"],
            "expected_ingredient": q["expected_ingredient"],
            "ground_truth": q["ground_truth_text"][:100],
            "results": relevance_r,
        })
        
        if (i + 1) % 20 == 0:
            print(f"  Evaluated {i+1}/{len(queries)} queries...")
    
    return results_unfiltered, results_filtered, results_reranked


def compute_metrics(eval_results, k, match_type="full_match"):
    """Compute IR metrics at a given K.
    
    match_type: 'full_match' (ingredient+content), 'ing_match', 'content_match'
    """
    hit_count = 0
    rr_sum = 0
    precision_sum = 0
    ndcg_sum = 0
    recall_sum = 0
    
    for r in eval_results:
        top_k_results = r["results"][:k]
        
        # Hit Rate@K
        hit = any(res[match_type] for res in top_k_results)
        hit_count += int(hit)
        
        # MRR
        rr = 0
        for rank, res in enumerate(top_k_results, 1):
            if res[match_type]:
                rr = 1.0 / rank
                break
        rr_sum += rr
        
        # Precision@K
        relevant_count = sum(1 for res in top_k_results if res[match_type])
        precision_sum += relevant_count / k
        
        # Recall@K (assume 1 ground truth doc per query for simplicity)
        recall_sum += min(1, relevant_count)
        
        # NDCG@K
        dcg = 0
        for rank, res in enumerate(top_k_results, 1):
            if res[match_type]:
                dcg += 1.0 / math.log2(rank + 1)
        idcg = 1.0 / math.log2(2)
        ndcg_sum += (dcg / idcg) if idcg > 0 else 0
    
    n = len(eval_results)
    return {
        "Hit Rate": round(hit_count / n, 4),
        "MRR": round(rr_sum / n, 4),
        "Precision": round(precision_sum / n, 4),
        "Recall": round(recall_sum / n, 4),
        "NDCG": round(ndcg_sum / n, 4),
    }


def compute_ingredient_precision(eval_results, k):
    """Context Precision: what fraction of top-K results are for the CORRECT ingredient?"""
    scores = []
    for r in eval_results:
        top_k = r["results"][:k]
        if top_k:
            correct = sum(1 for res in top_k if res["ing_match"])
            scores.append(correct / len(top_k))
    return round(np.mean(scores), 4) if scores else 0


# ═══════════════════════════════════════════════════════════
# STEP 3: LLM Faithfulness (subset)
# ═══════════════════════════════════════════════════════════

def evaluate_llm_faithfulness(queries, eval_results_filtered, n_samples=15):
    """Evaluate LLM faithfulness using filtered retrieval results."""
    # Pick diverse samples across categories
    by_cat = defaultdict(list)
    for i, q in enumerate(queries):
        by_cat[q["category"]].append(i)
    
    sample_indices = []
    cats_list = sorted(by_cat.keys())
    per_cat = max(1, n_samples // len(cats_list))
    for cat in cats_list:
        indices = by_cat[cat]
        sample_indices.extend(indices[:per_cat])
    sample_indices = sample_indices[:n_samples]
    
    faithfulness_results = []
    for idx in sample_indices:
        q = queries[idx]
        r = eval_results_filtered[idx]
        
        # Build context from top 3 retrieved docs
        context = ""
        for j, res in enumerate(r["results"][:3], 1):
            context += f"Fragment {j} [{res['ingredient']}]: \"{res['doc']}\"\n"
        
        # Generate answer
        system_prompt = (
            "Tu es un assistant expert en boulangerie. "
            "Réponds EN UTILISANT UNIQUEMENT le contexte fourni. "
            "Si le contexte ne contient pas la réponse, dis 'Information non disponible'."
        )
        user_prompt = f"Question: {q['query_fr']}\n\nContexte:\n{context}\nRéponds brièvement."
        
        try:
            completion = LLM_CLIENT.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
                ],
            )
            answer = completion.choices[0].message.content.strip()
        except Exception as e:
            answer = f"ERROR: {str(e)}"
        
        # Check faithfulness: does answer contain key info from ground truth?
        gt = q["ground_truth_text"].lower()
        answer_lower = answer.lower()
        
        # Extract key tokens from ground truth (numbers, technical terms)
        gt_tokens = set()
        for word in gt.split():
            word_clean = word.strip(".,;:()[]{}\"'")
            if any(ch.isdigit() for ch in word_clean) and len(word_clean) >= 2:
                gt_tokens.add(word_clean)
            if word_clean in ["ppm", "u/g", "nmau/g", "skb/g", "fau/g", "xylh/g", "agi/g"]:
                gt_tokens.add(word_clean)
        
        if gt_tokens:
            token_hits = sum(1 for t in gt_tokens if t in answer_lower)
            faithfulness_score = token_hits / len(gt_tokens)
        else:
            # Fallback: sliding window overlap
            window = 30
            found = False
            for i in range(0, min(len(gt), 100), 10):
                snippet = gt[i:i+window]
                if len(snippet) >= 20 and snippet in answer_lower:
                    found = True
                    break
            faithfulness_score = 1.0 if found else 0.0
        
        refused = any(phrase in answer_lower for phrase in 
                      ["non disponible", "pas disponible", "ne contient pas", 
                       "ne mentionne", "pas de réponse", "pas d'information"])
        
        retrieval_had_answer = any(res["full_match"] for res in r["results"][:3])
        
        faithfulness_results.append({
            "query_id": idx + 1,
            "query": q["query_fr"][:80],
            "expected_ingredient": q["expected_ingredient"],
            "ground_truth": q["ground_truth_text"][:80],
            "answer": answer[:250],
            "faithfulness": round(faithfulness_score, 2),
            "refused": refused,
            "retrieval_had_answer": retrieval_had_answer,
            "top_3_ingredients": [res["ingredient"] for res in r["results"][:3]],
        })
        
        print(f"  LLM eval {len(faithfulness_results)}/{n_samples}...")
    
    return faithfulness_results


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    start_time = time.time()
    
    # ── Generate queries ──
    print("=" * 60)
    print("STEP 1: Generating benchmark queries...")
    queries = generate_benchmark_queries(chunks)
    print(f"Generated {len(queries)} queries.\n")
    
    cats = Counter(q["category"] for q in queries)
    diffs = Counter(q.get("difficulty", "easy") for q in queries)
    print("Category distribution:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    
    print("\nDifficulty distribution:")
    for diff in ["easy", "medium", "hard", "adversarial"]:
        print(f"  {diff}: {diffs.get(diff, 0)}")
    
    # Check ingredient coverage
    ing_counts = Counter(q["expected_ingredient"] for q in queries)
    print(f"\nIngredient coverage: {len(ing_counts)} unique ingredients")
    print()
    
    # ── Retrieval evaluation ──
    print("=" * 60)
    print("STEP 2: Evaluating retrieval (unfiltered vs filtered vs reranked)...")
    results_unfiltered, results_filtered, results_reranked = evaluate_retrieval(queries, top_k=max(TOP_K_VALUES))
    
    # ── Compute metrics ──
    print("\n" + "=" * 60)
    print("RETRIEVAL METRICS (STRICT: ingredient + content must match)")
    print("=" * 60)
    
    print("\n--- A) UNFILTERED (pure semantic search, EN+FR) ---")
    metrics_unfiltered = {}
    for k in TOP_K_VALUES:
        m = compute_metrics(results_unfiltered, k, "full_match")
        metrics_unfiltered[k] = m
        print(f"  @K={k}: Hit={m['Hit Rate']:.2%}, MRR={m['MRR']:.4f}, P@K={m['Precision']:.4f}, R@K={m['Recall']:.2%}, NDCG={m['NDCG']:.4f}")
    
    print(f"\n  Context Precision (ingredient accuracy):")
    for k in [1, 3, 5]:
        cp = compute_ingredient_precision(results_unfiltered, k)
        print(f"    @K={k}: {cp:.2%}")
    
    print("\n--- B) FILTERED (metadata filter on ingredient) ---")
    metrics_filtered = {}
    for k in TOP_K_VALUES:
        m = compute_metrics(results_filtered, k, "full_match")
        metrics_filtered[k] = m
        print(f"  @K={k}: Hit={m['Hit Rate']:.2%}, MRR={m['MRR']:.4f}, P@K={m['Precision']:.4f}, R@K={m['Recall']:.2%}, NDCG={m['NDCG']:.4f}")
    
    print(f"\n  Context Precision (ingredient accuracy):")
    for k in [1, 3, 5]:
        cp = compute_ingredient_precision(results_filtered, k)
        print(f"    @K={k}: {cp:.2%}")
    
    print("\n--- C) FILTERED + RERANKED (cross-encoder reranking) ---")
    metrics_reranked = {}
    for k in TOP_K_VALUES:
        m = compute_metrics(results_reranked, k, "full_match")
        metrics_reranked[k] = m
        print(f"  @K={k}: Hit={m['Hit Rate']:.2%}, MRR={m['MRR']:.4f}, P@K={m['Precision']:.4f}, R@K={m['Recall']:.2%}, NDCG={m['NDCG']:.4f}")
    
    print(f"\n  Context Precision (ingredient accuracy):")
    for k in [1, 3, 5]:
        cp = compute_ingredient_precision(results_reranked, k)
        print(f"    @K={k}: {cp:.2%}")
    
    # ── Per-category breakdown ──
    print("\nPer-Category Strict Hit Rate @K=3:")
    print(f"  {'Category':<25s} | {'Unfiltered':>12s} | {'Filtered':>12s} | {'Reranked':>12s}")
    print("  " + "-" * 70)
    for cat in sorted(cats.keys()):
        uf_results = [r for r, q in zip(results_unfiltered, queries) if q["category"] == cat]
        f_results = [r for r, q in zip(results_filtered, queries) if q["category"] == cat]
        r_results = [r for r, q in zip(results_reranked, queries) if q["category"] == cat]
        uf_hits = sum(1 for r in uf_results if any(res["full_match"] for res in r["results"][:3]))
        f_hits = sum(1 for r in f_results if any(res["full_match"] for res in r["results"][:3]))
        r_hits = sum(1 for r in r_results if any(res["full_match"] for res in r["results"][:3]))
        uf_rate = f"{uf_hits}/{len(uf_results)}" if uf_results else "-"
        f_rate = f"{f_hits}/{len(f_results)}" if f_results else "-"
        r_rate = f"{r_hits}/{len(r_results)}" if r_results else "-"
        print(f"  {cat:<25s} | {uf_rate:>12s} | {f_rate:>12s} | {r_rate:>12s}")
    
    # ── Ingredient precision gap ──
    print("\nIngredient Filtering Impact:")
    for k in [1, 3]:
        cp_uf = compute_ingredient_precision(results_unfiltered, k)
        cp_f = compute_ingredient_precision(results_filtered, k)
        print(f"  @K={k}: Unfiltered={cp_uf:.2%}, Filtered={cp_f:.2%}, Improvement={cp_f-cp_uf:+.2%}")
    
    # ── Similarity scores ──
    uf_sims = [r["results"][0]["sim"] for r in results_unfiltered if r["results"]]
    f_sims = [r["results"][0]["sim"] for r in results_filtered if r["results"]]
    print(f"\nSimilarity Score Distribution (Top-1):")
    print(f"  Unfiltered: Mean={np.mean(uf_sims):.4f}, Median={np.median(uf_sims):.4f}, Min={np.min(uf_sims):.4f}, Max={np.max(uf_sims):.4f}")
    print(f"  Filtered:   Mean={np.mean(f_sims):.4f}, Median={np.median(f_sims):.4f}, Min={np.min(f_sims):.4f}, Max={np.max(f_sims):.4f}")
    
    # ── LLM Faithfulness ──
    print("\n" + "=" * 60)
    print("STEP 3: Evaluating LLM faithfulness (15 samples, using filtered retrieval)...")
    faith_results = evaluate_llm_faithfulness(queries, results_filtered, n_samples=15)
    
    avg_faith = np.mean([f["faithfulness"] for f in faith_results])
    refusal_rate = np.mean([f["refused"] for f in faith_results])
    correct_refusals = sum(1 for f in faith_results if f["refused"] and not f["retrieval_had_answer"])
    incorrect_refusals = sum(1 for f in faith_results if f["refused"] and f["retrieval_had_answer"])
    
    print(f"\n  Avg Faithfulness: {avg_faith:.2%}")
    print(f"  Refusal Rate: {refusal_rate:.2%}")
    print(f"  Correct Refusals: {correct_refusals}, Incorrect Refusals: {incorrect_refusals}")
    
    # ═══════════════════════════════════════════════════════════
    # STEP 4: Write comprehensive report
    # ═══════════════════════════════════════════════════════════
    
    elapsed = time.time() - start_time
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("RAG SYSTEM EVALUATION REPORT\n")
        f.write(f"Date: 2026-02-28 | Queries: {len(queries)} | Time: {elapsed:.1f}s\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Embedding Model: all-MiniLM-L6-v2 (384d)\n")
        f.write(f"Vector DB: ChromaDB ({collection.count()} entries, exploded)\n")
        f.write(f"Similarity: Cosine\n")
        f.write(f"Retrieval: Dual-language (EN+FR)\n")
        f.write(f"Filtering: Metadata 'ingredient' field (exact match)\n")
        f.write(f"LLM: Lightning AI gpt-oss-120b\n")
        f.write(f"Relevance: STRICT (ingredient + content must match)\n\n")
        
        f.write("QUERY DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
            f.write(f"  {cat:25s}: {count:3d} queries\n")
        f.write(f"  {'TOTAL':25s}: {len(queries):3d} queries\n")
        f.write(f"  Ingredient coverage: {len(ing_counts)} unique ingredients\n\n")
        
        # ── Section A: Unfiltered metrics ──
        f.write("=" * 70 + "\n")
        f.write("A) UNFILTERED RETRIEVAL METRICS (pure semantic search)\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"{'K':>4s} | {'Hit Rate':>10s} | {'MRR':>8s} | {'P@K':>8s} | {'R@K':>8s} | {'NDCG@K':>8s}\n")
        f.write("-" * 58 + "\n")
        for k in TOP_K_VALUES:
            m = metrics_unfiltered[k]
            f.write(f"{k:4d} | {m['Hit Rate']:10.2%} | {m['MRR']:8.4f} | {m['Precision']:8.4f} | {m['Recall']:8.2%} | {m['NDCG']:8.4f}\n")
        
        f.write(f"\nContext Precision (ingredient accuracy):\n")
        for k in [1, 3, 5]:
            cp = compute_ingredient_precision(results_unfiltered, k)
            f.write(f"  @K={k}: {cp:.2%}\n")
        
        # ── Section B: Filtered metrics ──
        f.write(f"\n{'=' * 70}\n")
        f.write("B) FILTERED RETRIEVAL METRICS (metadata filter on ingredient)\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"{'K':>4s} | {'Hit Rate':>10s} | {'MRR':>8s} | {'P@K':>8s} | {'R@K':>8s} | {'NDCG@K':>8s}\n")
        f.write("-" * 58 + "\n")
        for k in TOP_K_VALUES:
            m = metrics_filtered[k]
            f.write(f"{k:4d} | {m['Hit Rate']:10.2%} | {m['MRR']:8.4f} | {m['Precision']:8.4f} | {m['Recall']:8.2%} | {m['NDCG']:8.4f}\n")
        
        f.write(f"\nContext Precision (ingredient accuracy):\n")
        for k in [1, 3, 5]:
            cp = compute_ingredient_precision(results_filtered, k)
            f.write(f"  @K={k}: {cp:.2%}\n")
        
        # ── Section C: Reranked metrics ──
        f.write(f"\n{'=' * 70}\n")
        f.write("C) FILTERED + RERANKED METRICS (cross-encoder reranking)\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"{'K':>4s} | {'Hit Rate':>10s} | {'MRR':>8s} | {'P@K':>8s} | {'R@K':>8s} | {'NDCG@K':>8s}\n")
        f.write("-" * 58 + "\n")
        for k in TOP_K_VALUES:
            m = metrics_reranked[k]
            f.write(f"{k:4d} | {m['Hit Rate']:10.2%} | {m['MRR']:8.4f} | {m['Precision']:8.4f} | {m['Recall']:8.2%} | {m['NDCG']:8.4f}\n")
        
        f.write(f"\nContext Precision (ingredient accuracy):\n")
        for k in [1, 3, 5]:
            cp = compute_ingredient_precision(results_reranked, k)
            f.write(f"  @K={k}: {cp:.2%}\n")
        
        # ── Filtering Impact ──
        f.write(f"\n{'=' * 70}\n")
        f.write("FILTERING IMPACT ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Metric @K=3':<30s} | {'Unfiltered':>12s} | {'Filtered':>12s} | {'Reranked':>12s}\n")
        f.write("-" * 80 + "\n")
        for metric_name in ["Hit Rate", "MRR", "Precision", "Recall", "NDCG"]:
            uf_val = metrics_unfiltered[3][metric_name]
            f_val = metrics_filtered[3][metric_name]
            r_val = metrics_reranked[3][metric_name]
            f.write(f"  {metric_name:<28s} | {uf_val:12.4f} | {f_val:12.4f} | {r_val:12.4f}\n")
        
        cp_uf_3 = compute_ingredient_precision(results_unfiltered, 3)
        cp_f_3 = compute_ingredient_precision(results_filtered, 3)
        cp_r_3 = compute_ingredient_precision(results_reranked, 3)
        f.write(f"  {'Context Precision':<28s} | {cp_uf_3:12.4f} | {cp_f_3:12.4f} | {cp_r_3:12.4f}\n")
        
        # ── Per-category ──
        f.write(f"\nPer-Category Strict Hit Rate @K=3:\n")
        f.write(f"  {'Category':<25s} | {'Unfiltered':>12s} | {'Filtered':>12s} | {'Reranked':>12s}\n")
        f.write("  " + "-" * 70 + "\n")
        for cat in sorted(cats.keys()):
            uf_results = [r for r, q in zip(results_unfiltered, queries) if q["category"] == cat]
            f_results = [r for r, q in zip(results_filtered, queries) if q["category"] == cat]
            r_results = [r for r, q in zip(results_reranked, queries) if q["category"] == cat]
            uf_hits = sum(1 for r in uf_results if any(res["full_match"] for res in r["results"][:3]))
            f_hits = sum(1 for r in f_results if any(res["full_match"] for res in r["results"][:3]))
            r_hits = sum(1 for r in r_results if any(res["full_match"] for res in r["results"][:3]))
            f.write(f"  {cat:<25s} | {uf_hits:5d}/{len(uf_results):<5d} | {f_hits:5d}/{len(f_results):<5d} | {r_hits:5d}/{len(r_results):<5d}\n")
        
        # ── Similarity scores ──
        f.write(f"\nSimilarity Score Distribution (Top-1):\n")
        f.write(f"  Unfiltered: Mean={np.mean(uf_sims):.4f}, Median={np.median(uf_sims):.4f}, Min={np.min(uf_sims):.4f}, Max={np.max(uf_sims):.4f}\n")
        f.write(f"  Filtered:   Mean={np.mean(f_sims):.4f}, Median={np.median(f_sims):.4f}, Min={np.min(f_sims):.4f}, Max={np.max(f_sims):.4f}\n")
        
        # ── LLM Faithfulness ──
        f.write(f"\n\n{'=' * 70}\n")
        f.write(f"LLM FAITHFULNESS ({len(faith_results)} samples, filtered retrieval)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Average Faithfulness Score: {avg_faith:.2%}\n")
        f.write(f"Refusal Rate: {refusal_rate:.2%}\n")
        f.write(f"Correct Refusals (no relevant retrieval): {correct_refusals}\n")
        f.write(f"Incorrect Refusals (had relevant retrieval): {incorrect_refusals}\n\n")
        
        for fr in faith_results[:5]:  # show 5 samples
            f.write(f"Q{fr['query_id']}: {fr['query']}\n")
            f.write(f"  Expected Ingredient: {fr['expected_ingredient']}\n")
            f.write(f"  Ground Truth: {fr['ground_truth']}\n")
            f.write(f"  Faithfulness: {fr['faithfulness']:.0%}\n")
            f.write(f"  Refused: {fr['refused']}, Retrieval had answer: {fr['retrieval_had_answer']}\n")
            f.write(f"  LLM Answer: {fr['answer'][:200]}...\n\n")
        if len(faith_results) > 5:
            f.write(f"  ... ({len(faith_results) - 5} more samples omitted)\n")
        
        # ── Detailed results (failures focus) ──
        f.write(f"\n{'=' * 70}\n")
        f.write("DETAILED RESULTS: FILTERED + RERANKED RETRIEVAL\n")
        f.write("=" * 70 + "\n\n")
        
        # Show all misses first
        misses = [(r, q) for r, q in zip(results_reranked, queries) 
                  if not any(res["full_match"] for res in r["results"][:3])]
        
        if misses:
            f.write(f"--- MISSES ({len(misses)} queries where ground truth NOT in Top-3, showing first 5) ---\n\n")
            for r, q in misses[:5]:  # show only first 5 misses
                f.write(f"Q{r['query_id']} [{q['category']}]: {r['query'][:70]}\n")
                f.write(f"  Expected: [{q['expected_ingredient']}] {r['ground_truth'][:80]}\n")
                for j, res in enumerate(r["results"][:3], 1):
                    ing_marker = "✓" if res["ing_match"] else "✗"
                    cnt_marker = "✓" if res["content_match"] else "✗"
                    f.write(f"  R{j} [ing:{ing_marker} cnt:{cnt_marker}] (sim={res['sim']:.4f}) [{res['ingredient']}]: {res['doc'][:100]}\n")
                f.write("\n")
            if len(misses) > 5:
                f.write(f"  ... ({len(misses) - 5} more misses omitted)\n\n")
        
        # Then show all hits
        hits_list = [(r, q) for r, q in zip(results_reranked, queries) 
                     if any(res["full_match"] for res in r["results"][:3])]
        
        f.write(f"\n--- HITS ({len(hits_list)} queries with ground truth in Top-3, showing first 10) ---\n\n")
        for r, q in hits_list[:10]:  # show first 10
            rank = next((j+1 for j, res in enumerate(r["results"][:3]) if res["full_match"]), "?")
            f.write(f"Q{r['query_id']} [{q['category']}] Rank={rank}: {r['query'][:70]}\n")
            f.write(f"  [{q['expected_ingredient']}] {r['ground_truth'][:80]}\n")
        if len(hits_list) > 10:
            f.write(f"\n  ... ({len(hits_list) - 10} more hits omitted)\n")
        
        # ── Unfiltered vs Filtered comparison for failures ──
        f.write(f"\n\n{'=' * 70}\n")
        f.write("UNFILTERED vs FILTERED: Cases where filtering HELPED\n")
        f.write("=" * 70 + "\n\n")
        
        helped = 0
        hurt = 0
        for r_uf, r_f, q in zip(results_unfiltered, results_filtered, queries):
            uf_hit = any(res["full_match"] for res in r_uf["results"][:3])
            f_hit = any(res["full_match"] for res in r_f["results"][:3])
            if f_hit and not uf_hit:
                helped += 1
                f.write(f"  HELPED: Q{r_f['query_id']} [{q['category']}]: {q['query_fr'][:60]}\n")
                f.write(f"    Ingredient: {q['expected_ingredient']}\n")
            elif uf_hit and not f_hit:
                hurt += 1
                f.write(f"  HURT: Q{r_f['query_id']} [{q['category']}]: {q['query_fr'][:60]}\n")
                f.write(f"    Ingredient: {q['expected_ingredient']}\n")
        
        f.write(f"\n  Filtering helped: {helped} queries")
        f.write(f"\n  Filtering hurt: {hurt} queries\n")
    
    print(f"\n{'=' * 60}")
    print(f"EVALUATION COMPLETE in {elapsed:.1f}s")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"{'=' * 60}")
