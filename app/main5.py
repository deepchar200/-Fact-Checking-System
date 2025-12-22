# imports
from fastapi import FastAPI, HTTPException, Depends
from sqlmodel import Session, select
from typing import List
import requests
from pydantic import BaseModel
from pathlib import Path
import wikipedia
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from difflib import SequenceMatcher
from sentence_transformers import CrossEncoder
import string
import re # <--- Ensure this is imported at the top
# --- for spliting the sentences of the premise into parts ---
from sentence_transformers import util
from rake_nltk import Rake
import nltk
nltk.download('punkt_tab')
import string
# This variable contains all standard ASCII punctuation:
# ('!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~')
PUNCT_TRANSLATION_TABLE = str.maketrans('', '', string.punctuation)
# --- Local Imports ---
from .models import create_db_and_tables, PredictionHistory, NewsRequest
from .database import get_session, engine

app = FastAPI()

# --- CONFIGURATION ---
NEWS_API_KEY = "f20e28ccefdba6327e432510be360a93"  # <--- PASTE KEY HERE

TRUSTED_DOMAINS = [
    "timesofindia.indiatimes.com", "thehindu.com", "ndtv.com",
    "indianexpress.com", "bbc.com", "bbc.co.uk", "hindustantimes.com",
    "livemint.com", "business-standard.com", "economictimes.indiatimes.com", "theprint.in"
]
# --- LOAD MODELS ---
BASE_DIR = Path(__file__).resolve().parent.parent
DL_MODEL_PATH = BASE_DIR / "indic-bert-model" 

dl_tokenizer = None
dl_model = None
nli_model = None 

try:
    print("\n[INIT] ⏳ Loading Models...")
    dl_tokenizer = AutoTokenizer.from_pretrained(DL_MODEL_PATH)
    dl_model = AutoModelForSequenceClassification.from_pretrained(DL_MODEL_PATH)
    dl_model.eval()
    print("[INIT] ✅ Indic-BERT Loaded!")
    
    # Old model (Prone to bias):
    # nli_model = CrossEncoder('cross-encoder/nli-distilroberta-base')
    
    # 🔴 OLD MODEL (Simple, Fast, but gets confused by dates)
    # nli_model = CrossEncoder('cross-encoder/nli-roberta-base')
    
    # 🟢 NEW MODEL (Smarter, DeBERTa-v3-base)
    # This model is specifically designed to solve the "Overlap Bias" you found!
    nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
    
    print("[INIT] ✅ NLI Model (DeBERTa) Loaded!")
    
except Exception as e:
    print(f"[INIT] ❌ Error loading Models: {e}")
    dl_tokenizer = None
    dl_model = None
    nli_model = None

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

# --- HELPER FUNCTIONS ---

def check_tier_1_trusted_sources(claim: str) -> dict:

    """

    Tier 1: Live News Check with STRICT Subject Matching.

    """
    
    url = "https://gnews.io/api/v4/search"
    
    # 1. RAKE Extraction Debug
    r = Rake()
    r.extract_keywords_from_text(claim)
    phrases = r.get_ranked_phrases()
    entities = [w for p in phrases for w in p.split(' ')]
    print(f"[DEBUG] Phrases Extracted: {phrases}")
    print(f"[DEBUG] Entities identified: {entities}")

    # 2. Required Names Debug
    common_places = ["India", "US", "UK", "USA", "China", "World"]
    required_names = [e for e in entities if e not in common_places]
    if not required_names: required_names = entities
    print(f"[DEBUG] Required Names (for strict matching): {required_names}")

    # 3. Query Strategy
    query_strict = " AND ".join(entities[:4])
    query_broad = " OR ".join(entities[:3])
    
    def run_api_search(query_str, search_type):
        if not query_str: return []
        print(f"\n[Tier 1] ({search_type}) GNews Search: '{query_str}'")
        
        params = {
            "q": query_str, "apikey": NEWS_API_KEY, "lang": "en",
            "country": "in", "max": 10, "in": "title"
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            print(f"[DEBUG] API Status: {response.status_code} | Total Articles: {data.get('totalArticles', 0)}")
            
            if data.get("totalArticles", 0) > 0:
                verified = []
                for article in data.get("articles", []):
                    headline = article["title"]
                    source_url = article["source"]["url"]
                    source_name = article["source"]["name"]
                    print(f"\n  🔍 Processing: {headline[:60]}... | Source: {source_url}")

                    # Domain Filter
                    is_trusted = any(d in source_url for d in TRUSTED_DOMAINS)
                    if not is_trusted and source_name.lower() in [d.split('.')[0] for d in TRUSTED_DOMAINS]:

                        is_trusted = True
                    
                    # Subject Validation Debug
                    hits = sum(1 for name in required_names if name.lower() in headline.lower())
                    hit_ratio = hits / len(required_names) if required_names else 1.0
                    subject_match = hit_ratio >= 0.5
                    print(f"  [DEBUG] Subject Match: {subject_match} (Hits: {hits}/{len(required_names)})")

                    # Similarity Debug
                    similarity = SequenceMatcher(None, claim.lower(), headline.lower()).ratio()
                    entity_hits = sum(1 for e in entities if e.lower() in headline.lower())
                    print(f"  [DEBUG] Similarity Score: {round(similarity, 2)} | Entity Hits: {entity_hits}")
                    
                    if is_trusted and subject_match and (similarity > 0.3 or entity_hits >= 2):
                        print(f"    ✅ FINAL MATCH: {headline}")
                        verified.append({"title": headline, "url": article["url"], "source": {"name": article["source"]["name"]}})
                    else:
                        print(f"    ❌ REJECTED: Logic condition not met (Trusted: {is_trusted})")
                return verified
        except Exception as e:
            print(f"Tier 1 Exception: {e}")
        return []

    # Execution Flow
    res = run_api_search(query_strict, "Strict")
    if not res: res = run_api_search(query_broad, "Broad")
    
    return {"status": "found" if res else "not_found", "articles": res}

# --- NLI LOGIC ---
def check_tier_1_8_nli(claim: str, evidence: str) -> dict:
    """Tier 1.8 with Tensor Fix."""
    print(f"\n--- 🧠 [DEBUG] TIER 1.8: NLI CHECK ---")
    print(f"Hypothesis: {claim}")
    print(f"Premise: {evidence}")
    
    if not nli_model:
        return {"label": "neutral", "score": 0.0}

    # Download sentence tokenizer if not present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt_tab')

    # 1. Split Evidence into Sentences
    sentences = nltk.sent_tokenize(evidence)
    
    # 2. Find the "Best Matching" Sentence (Semantic Search)
    # We use a simple keyword overlap score for speed (or you can use embedding if available)
    best_sentence = ""
    best_score = -1
    best_index = -1
    
    stopwords = {
    'i', 'me', 'is', 'upon', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
    'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
    'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 
    'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', 
    "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', 
    "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 
    'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}
    # 1. Clean Punctuation and lowercase the entire string
    cleaned_claim = claim.lower().translate(PUNCT_TRANSLATION_TABLE)
    
    # 2. Split and filter stopwords in one step
    claim_words = {w for w in cleaned_claim.split() if w not in stopwords}
    
    for index, sent in enumerate(sentences):
        # Calculate overlap (How many claim words are in this sentence?)
        cleaned_sent = sent.lower().translate(PUNCT_TRANSLATION_TABLE)
        
        sent_words = {w for w in cleaned_sent.split() if w not in stopwords}
        
        # calcualte the no of overlap words in the claim and evidence for the each sentence
        overlap =len(claim_words.intersection(sent_words))
        score = overlap
        
        # Priority Bonus: If sentence has the same YEAR as claim, boost score
        if any(year in sent for year in [w for w in claim_words if w.isdigit()]):
            overlap += 5  # Huge bonus for matching dates
            
        if score > best_score:
            best_score = score
            best_sentence = sent
            best_index = index
        
    print(f"   Original Evidence Length: {len(evidence)} chars")
    print(f"   Selected 'Golden' Sentence: '{best_sentence}'")

    # --- Context Expansion Logic ---
    final_evidence = best_sentence
    
    # Use the stored index to check the previous sentence
    # FIX: Context Expansion
        # If the best sentence has a pronoun like "he", "she", "it", or "they",
        # OR if it's just short, we prepend the previous sentence to get the Subject Name.
    if best_index > 0:
        prev_sentence = sentences[best_index - 1]
        final_evidence = f"{prev_sentence} {best_sentence}"
        print(f"   Using Expanded Context (Index {best_index} + Prev): '{final_evidence}'")

    # 3. Run NLI ONLY on the Expanded Evidence
    # Now the model won't get distracted by "Luna 2 (1959)"
    
    try:
        # 1. Get raw scores
        scores = nli_model.predict([(claim, final_evidence)])[0]
        
        # 2. FIX: Convert to Tensor and SQUEEZE to remove batch dimension
        # Shape becomes [3] instead of [1, 3]
        tensor_scores = torch.tensor(scores).squeeze()
        
        # 3. Apply Softmax
        probabilities = torch.nn.functional.softmax(tensor_scores, dim=0)
        
        # 4. Extract scalar values safely
        c_score = probabilities[0].item()
        e_score = probabilities[1].item()
        n_score = probabilities[2].item()
        
        print(f"Scores -> Contradiction: {c_score:.2f} | Entailment: {e_score:.2f} | Neutral: {n_score:.2f}")
        
        label_map = ['contradiction', 'entailment', 'neutral']
        confidence, winner_index = torch.max(probabilities, dim=0)
        winner_label = label_map[winner_index.item()]
        
        print(f"NLI Verdict: {winner_label.upper()} ({confidence.item():.2f})")
        return {"label": winner_label, "score": round(confidence.item() * 100, 2)}
        
    except Exception as e:
        print(f"NLI Error: {e}")
        return {"label": "neutral", "score": 0.0}

# TIER 1.5 Checking with the wikipedia page for archiving purpose
def check_tier_1_5_knowledge_base(claim: str) -> dict:
    """
    Tier 1.5: Multi-Page Wiki Check with Soft Fallback.
    """
    print(f"\n--- 📚 [DEBUG] TIER 1.5: MULTI-PAGE WIKI CHECK ---")
    
    # 1. Setup Search Query
    clean_claim = claim.lower().translate(PUNCT_TRANSLATION_TABLE)
    print(f"Claim : '{clean_claim}'")
    keywords = [word for word in clean_claim.split() if word.istitle() and len(word) > 3]
    print(f"Keywords : '{keywords}'")
    keyword_query = " ".join(keywords) if keywords else claim
    print(f"Keyword_Query : '{keyword_query}'")

    def get_wiki_candidates(query):
        try:
            return wikipedia.search(query, results=3)
        except: return []

    # Helper function to process a single page
    def process_page(title):
        try:
            print(f"   🔎 Checking Candidate: '{title}'...")
            page = wikipedia.page(title, auto_suggest=False)
            
            # Context Extraction
            full_summary = page.summary
            sentences = [s.strip() + "." for s in full_summary.split('. ') if s]
            
            evidence_sentences = []
            evidence_sentences.extend(sentences[:2]) 
            
            date_sentences = [s for s in sentences if re.search(r'\b\d{4}\b', s)]
            for s in date_sentences:
                if s not in evidence_sentences:
                    evidence_sentences.append(s)
            
            clean_evidence = []
            for item in evidence_sentences[:6]:
                clean_evidence.append(str(item) if not isinstance(item, list) else " ".join(item))
            
            optimized_premise = " ".join(clean_evidence)

            # --- FIX: Pronoun Replacement ---
            # If the evidence starts with "He", "She", or "It", replace it with the Page Title.
            # This forces the model to see "Jawaharlal Nehru served..." instead of "He served..."
            
            final_evidence_list = []
            page_title_clean = page.title.split('(')[0].strip() # Remove (1889-1964) etc
            
            for sent in clean_evidence:
                # Simple check for starting pronouns
                if sent.lower().startswith(("he ", "she ", "it ", "they ")):
                    # Replace the first word with the Page Title
                    # e.g. "He served..." -> "Jawaharlal Nehru served..."
                    parts = sent.split(" ", 1)
                    if len(parts) > 1:
                        sent = f"{page_title_clean} {parts[1]}"
                
                final_evidence_list.append(sent)
                
            optimized_premise = " ".join(final_evidence_list)
            
            # Run NLI
            nli_result = check_tier_1_8_nli(claim, optimized_premise)
            label = nli_result['label']
            score = nli_result['score']
            
            # --- OVERRIDE LOGIC ---
            claim_year_match = re.search(r'\b\d{4}\b', claim)
            claim_year = claim_year_match.group(0) if claim_year_match else None
            evidence_years = set(re.findall(r'\b\d{4}\b', optimized_premise))
            
            is_strong_disproof = False
            
            if claim_year and label == "entailment":
                if claim_year not in evidence_years:
                    if any(y != claim_year for y in evidence_years):
                        print(f"         🚨 OVERRIDE: Date mismatch detected ({claim_year} missing)")
                        label = "contradiction"
                        score = 99.99 
                        is_strong_disproof = True
            
            return {
                "page": page,
                "summary": optimized_premise,
                "label": label,
                "score": score,
                "is_override": is_strong_disproof
            }
        except Exception as e:
            print(f"      ❌ Processing Error: {e}")
            return None

    # 2. Get Candidates
    candidates = get_wiki_candidates(claim)
    if not candidates and keyword_query:
        candidates = get_wiki_candidates(keyword_query)
    
    if not candidates:
        return {"status": "not_found"}

    print(f"   [DEBUG] Found Candidates: {candidates}")

    best_result = None
    highest_score = 0.0

    # 3. LOOP THROUGH CANDIDATES
    for title in candidates:
        result = process_page(title)
        
        if result:
            label = result['label']
            score = result['score']
            is_override = result['is_override']
            
            print(f"      👉 Verdict: {label.upper()} ({score:.2f}%)")

            # A. STRICT MATCH (>85% or Override) -> Return Immediately
            if is_override or (score > 85 and label in ['entailment', 'contradiction']):
                return {
                    "status": "verified" if label == "entailment" else "contradiction",
                    "title": result['page'].title,
                    "summary": result['summary'],
                    "url": result['page'].url,
                    "nli_verdict": label,
                    "nli_confidence": score
                }
            
            # B. TRACK BEST CANDIDATE (For Soft Fallback)
            # If this score is better than previous ones, save it.
            if score > highest_score and label in ['entailment', 'contradiction']:
                highest_score = score
                best_result = result

    # --- 👇👇👇 PASTE THE NEW CODE HERE 👇👇👇 ---
    # 4. SOFT FALLBACK (Run only if loop finishes without strict match)
    
    if best_result and highest_score > 70:
        print(f"   ⚠️ No 85% match found. Accepting Best Candidate ({highest_score:.2f}%)")
        
        label = best_result['label']
        return {
            "status": "verified" if label == "entailment" else "contradiction",
            "title": best_result['page'].title,
            "summary": best_result['summary'],
            "url": best_result['page'].url,
            "nli_verdict": label,
            "nli_confidence": highest_score
        }
    # --- 👆👆👆 END OF NEW CODE 👆👆👆 ---

    print("   [DEBUG] No candidate met the 70% Soft Threshold.")
    return {"status": "not_found"}


# TIER 2
def predict_tier_2_dl(text: str) -> dict:
    
    print(f"\n--- 🤖 [DEBUG] TIER 2: AI STYLE CHECK ---")
    
    if not dl_model or not dl_tokenizer:
        # Return a specific error status when the model is missing
        return {"label": "Model Disabled", "score": 0.0, "error": "Tier 2 Model not found locally."}
    
    try:
        inputs = dl_tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
        with torch.no_grad():
            logits = dl_model(**inputs).logits
        
        probs = F.softmax(logits, dim=1)
        print(f"Raw Logits: {logits}")
        print(f"Probabilities: {probs}")
        
        confidence, predicted_class = torch.max(probs, dim=1)
        label_map = {0: "Likely FAKE", 1: "Likely REAL"}
        result = label_map.get(predicted_class.item(), "Unknown")
        
        print(f"Final Verdict: {result} ({confidence.item():.2f})")
        return {
            "label": result,
            "score": round(confidence.item() * 100, 2)
        }
    except Exception as e:
        return {"label": f"Error: {e}", "score": 0.0}

# --- ENDPOINTS ---

@app.get("/verify-news")
def verify_news(claim: str, session: Session = Depends(get_session)):
    results = []
    final_verdict = ""
    verification_method = ""
    wiki_data = None

    # === TIER 1: Live News Check ===

    tier_1 = check_tier_1_trusted_sources(claim)

    # The rest of the logic remains exactly the same!
    if tier_1["status"] == "found":
        final_verdict = "REAL (Verified by Live News)"
        verification_method = "Tier 1: Trusted Source Coverage"
        for article in tier_1["articles"]:
            results.append({
                "headline": article["title"],
                "source": article["source"]["name"],
                "url": article["url"],
                "tier": "Tier 1 (News)"
            })
            
    else:
        # TIER 1.5 + 1.8
        tier_1_5 = check_tier_1_5_knowledge_base(claim)
        
        if tier_1_5["status"] == "verified":
            final_verdict = "REAL (Verified by Knowledge Base)"
            verification_method = f"Tier 1.5: Fact Context ({tier_1_5['nli_confidence']}%)"
            wiki_data = tier_1_5
            
        elif tier_1_5["status"] == "contradiction":
            final_verdict = "FAKE (Refuted by Knowledge Base)"
            verification_method = f"Tier 1.8: Fact Contradiction ({tier_1_5['nli_confidence']}%)"
            wiki_data = tier_1_5
            
        else:
            # TIER 2
            dl_result = predict_tier_2_dl(claim)
            model_label = dl_result["label"]
            confidence = dl_result["score"]
            final_verdict = f"Unverified ({model_label})"
            verification_method = f"Tier 2: AI Analysis (Indic-BERT)"
            
            note = f"Confidence: {confidence}%."
            if tier_1_5["status"] != "not_found":
                note += f" (Wiki checked, NLI was {tier_1_5.get('nli_verdict', 'neutral').upper()}.)"
                
            results.append({
                "headline": claim,
                "source": "User Input",
                "url": "N/A",
                "tier": f"{model_label}",
                "score": confidence,
                "note": note
            })

    # Save History
    history = PredictionHistory(
        input_text=claim,
        input_topic=verification_method,
        prediction_label=final_verdict
    )
    session.add(history)
    session.commit()

    return {
        "claim": claim,
        "verdict": final_verdict,
        "method": verification_method,
        "evidence": results,
        "wiki_context": wiki_data
    }

@app.get("/history", response_model=List[PredictionHistory])
def get_history(session: Session = Depends(get_session)):
    return session.exec(select(PredictionHistory)).all()
