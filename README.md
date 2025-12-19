Automated Hybrid Fact-Checking & Semantic Verification System

🛡️ Project Overview

This project implements a comprehensive, multi-tiered pipeline for automatically verifying the truthfulness of natural language claims. It moves beyond simple keyword matching by integrating real-time news APIs with advanced Natural Language Inference (NLI) to provide evidence-based, high-confidence verdicts (Entailment, Contradiction, or Neutral).

The system is deployed as a user-friendly web application using Streamlit.

🧠 Core Architecture (3-Tier Pipeline)

The verification process follows a cascading logic to maximize efficiency and accuracy:

Tier 1: Live News Coverage: Checks trusted sources (via GNews API) for recent articles matching key entities. Uses custom stopword/punctuation filters and Jaccard similarity scoring to find high-relevance matches.

Tier 1.8: Knowledge Base + Semantic NLI: If no live news match, the system queries a knowledge base (Wikipedia) for context. It uses a DeBERTa-based NLI Cross-Encoder to determine if the claim is logically Entailed or Contradicted by the extracted text.

🤖 Tier 2: AI Stylistic Analysis (Indic-BERT)
When real-time news (Tier 1) and factual context (Tier 1.5/1.8) are unavailable—often the case with rumors, opinions, or newly emerging misinformation—the system activates Tier 2.

1. Logic: Detecting the "DNA" of Fake News
Unlike the previous tiers which seek external evidence, Tier 2 analyzes the internal structure of the claim. It looks for linguistic "red flags" common in misinformation:

Sensationalism: Overuse of inflammatory or emotionally charged words.

Urgency Patterns: Clickbait-style phrasing (e.g., "Must read!", "Viral message").

Grammatical Markers: Specific patterns of syntax that differentiate professional journalism from fabricated content.

2. The Model: Indic-BERT
We utilize Indic-BERT, a multilingual model specifically pre-trained on 12 major Indian languages.

Contextual Advantage: Standard models often fail on Indian-English nuances; Indic-BERT is fine-tuned to understand the specific cultural and linguistic patterns of the Indian media landscape.

Task: Binary classification (Likely REAL vs. Likely FAKE) with a probability confidence score.

3. Technical Implementation & Robustness
Local Loading: The system is configured to load model weights from the /indic-bert-model/ directory.

Graceful Degradation: To ensure the app remains functional for recruiters without the 180MB model files, I implemented a Safety Guard in main.py. If the model is not found locally, the system catches the error at startup and disables Tier 2 while keeping Tier 1 and 1.5 fully operational.

💡 Innovation Spotlight: Coreference Resolution Fix

The system addresses a critical failure point in NLI models: Pronoun Ambiguity.

Problem: When Wikipedia evidence contained pronouns (e.g., "He served as the first PM...") referencing a subject in a previous sentence (e.g., "Jawaharlal Nehru was..."), the NLI model would fail Coreference Resolution, resulting in an inaccurate "Neutral" verdict.

Solution: A Pronoun Replacement Heuristic was implemented to dynamically inject the primary entity name (e.g., "Jawaharlal Nehru") directly into the start of the pronoun-dependent sentence. This forced a direct entity comparison ("Mahatma Gandhi" vs. "Jawaharlal Nehru") and correctly yielded a Contradiction verdict, demonstrating system robustness.

⚙️ Setup and Installation

1. Repository Structure
```text
.
├── .gitignore          (Ensures large models/keys are ignored)
├── app/                (Core logic: database, NLI calls, main pipeline)
│   ├── database.py
│   ├── main5.py
│   └── models.py
├── Dataset/            (Contains small, sample CSV data)
├── frontend/           (Streamlit application UI)
│   └── dashboard1.py
└── indic-bert-model/   (**Manually placed model files**)
```

2. Dependencies

Create a virtual environment and install the required packages:

pip install -r requirements.txt
Note: You may also need to install specific NLTK components


3. API Key Setup

For security, API keys are loaded via environment variables.

Create a file named .env in the root directory.

Add your keys to the file:

NEWS_API_KEY="YOUR_GNEWS_API_KEY"
GEMINI_API_KEY="YOUR_GEMINI_API_KEY" (If used outside the web app)


4. Tier 2 Model Setup (Manual Download Required)

The large Indic-BERT model weights are not stored in this repository to comply with GitHub file size limits.

Action Required: You must manually recreate the indic-bert-model/ directory structure and place the necessary weight and configuration files inside it for Tier 2 to run.

Required Files (Must be placed inside the indic-bert-model/ folder):

pytorch_model.bin (or equivalent .safetensors file)

config.json

tokenizer.json

vocab.txt

(Please place your existing local model files into this folder now.)

🚀 How to Run the System

The primary application runs via Streamlit:

Ensure all setup steps above are complete.

Navigate to the frontend/ directory.

Run the Streamlit dashboard:

streamlit run dashboard1.py
