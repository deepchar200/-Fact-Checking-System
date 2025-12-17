import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import time

# --- CONFIGURATION ---
st.set_page_config(
    page_title="News Verifier Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API URL (Ensure FastAPI is running on this port)
API_BASE_URL = "http://127.0.0.1:8000"

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .big-font { font-size:20px !important; }
    .stAlert { padding: 0.8rem; border-radius: 8px; }
    .metric-card { background-color: #0e1117; padding: 15px; border-radius: 10px; border: 1px solid #303030; }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.markdown("<h1>🛡️</h1>", unsafe_allow_html=True)
with col_title:
    st.title("Fake News Detection")
    st.caption("Hybrid Architecture: Live News (GNews) + Knowledge Base (Wiki) + Deep Learning (Indic-BERT)")

st.divider()

# --- TABS ---
tab_verify, tab_analytics = st.tabs(["🕵️ Verification Engine", "📊 Analytics Dashboard"])

# ==============================================================================
# TAB 1: VERIFICATION INTERFACE
# ==============================================================================
with tab_verify:
    # Layout: Input on left, Logic explanation on right
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Verify a Claim")
        user_claim = st.text_area(
            "Enter a news headline or statement:", 
            placeholder="e.g., 'India wins 1983 World Cup' or 'Dharmendra is died'",
            height=100
        )
        
        if st.button("Verify Claim", type="primary", use_container_width=True):
            if not user_claim.strip():
                st.warning("Please enter some text to verify.")
            else:
                try:
                    # Status Indicator
                    status_container = st.status("Initializing Cascade Protocol...", expanded=True)
                    
                    # 1. Call API
                    status_container.write("🔍 Tier 1: Scanning Trusted News Sources...")
                    response = requests.get(f"{API_BASE_URL}/verify-news", params={"claim": user_claim})
                    
                    if response.status_code == 200:
                        try:
                            data = response.json()
                            if isinstance(data, dict):
                                 verdict = data.get("verdict", "Unknown")
                                 method = data.get("method", "Unknown")
                                 evidence = data.get("evidence", [])
                                 wiki_data = data.get("wiki_context")
                            else:
                                 print("Expected a dictionary, got:", type(data))
                        except ValueError:
                                 print("Response is not valid JSON")

                        # 2. Update Status based on Method
                        if "Tier 1:" in method: 
                            status_container.update(label="✅ Verified by Live News Coverage!", state="complete", expanded=False)
                        elif "Tier 1.5" in method: 
                            status_container.update(label="📚 Verified by Historical Knowledge Base!", state="complete", expanded=False)
                        elif "Tier 1.8" in method:
                            status_container.update(label="🛑 Refuted by Knowledge Base Contradiction!", state="complete", expanded=False)
                        else: 
                            status_container.update(label="⚠️ Unverified. AI Analysis Complete.", state="complete", expanded=False)
                        
                        st.divider()
                        
                        # --- VERDICT BANNER ---
                        if "REAL" in verdict:
                            st.success(f"### {verdict}")
                        elif "FAKE" in verdict:
                            st.error(f"### {verdict}")
                        else:
                            st.warning(f"### {verdict}")
                        
                        st.caption(f"**Method Used:** {method}")
                        
                        # --- EVIDENCE DISPLAY SECTION ---
                        
                        # SCENARIO A: Wikipedia Context (Tier 1.5 / 1.8)
                        if wiki_data:
                            nli_label = wiki_data.get('nli_verdict', 'neutral')
                            nli_conf = wiki_data.get('nli_confidence', 0.0)
                            
                            # Display based on NLI Result
                            if nli_label == 'contradiction':
                                st.error("### 🛑 Fact Check: Contradiction Found")
                                st.caption(f"Semantic Confidence: {nli_conf}%")
                                st.markdown(f"**Knowledge Base:** {wiki_data['title']}")
                                st.markdown(f"*{wiki_data['summary']}*")
                                st.markdown(f"[Read full article on Wikipedia]({wiki_data['url']})")
                            elif nli_label == 'entailment':
                                st.info("### 📖 Historical Context Found")
                                st.caption(f"Semantic Confidence: {nli_conf}%")
                                st.markdown(f"**{wiki_data['title']}**")
                                st.markdown(f"*{wiki_data['summary']}*")
                                st.markdown(f"[Read full article on Wikipedia]({wiki_data['url']})")
                        
                        # SCENARIO B: Live News Found (Tier 1)
                        elif evidence and "Tier 1" in method:
                            st.subheader(f"📰 Live Coverage Found ({len(evidence)} sources)")
                            for item in evidence:
                                # --- CRASH PROOFING ---
                                if isinstance(item, dict):
                                    # It is a valid dictionary
                                    title = item.get('title', 'No Title')
                                    source = item.get('source', {}).get('name', 'Unknown')
                                    url = item.get('url', '#')
                                    
                                    with st.expander(f"{source}: {title}"):
                                        if url != "N/A":
                                            st.markdown(f"[Read Full Article]({url})")
                                        else:
                                            st.write("No direct link available.")
                                else:
                                    # It is a string (Error message)
                                    st.warning(f"⚠️ Metadata Error: {str(item)}")

                        # SCENARIO C: AI Analysis (Tier 2)
                        elif "Tier 2" in method:
                            st.subheader("🤖 AI Style Analysis")
                            st.markdown("Trusted sources and Knowledge Base provided no confirmation. The AI model analyzed the writing style:")
                            
                            for item in evidence:
                                verdict_text = item.get('tier', 'Unknown')
                                
                                # Get raw score from backend
                                score_val = item.get('score', 99.0)
                                
                                # AMBIGUITY CHECK (40-60%)
                                if 40 < score_val < 60:
                                    st.warning(f"⚠️ **Ambiguous Result** (Confidence: {score_val}%)")
                                    st.markdown("""
                                    The AI is **uncertain**. The writing style contains a mix of legitimate journalistic phrasing and suspicious assertion. 
                                    *Recommendation: Verify with a secondary source manually.*
                                    """)
                                else:
                                    # Standard Confident Result
                                    color = "red" if "FAKE" in verdict_text else "green"
                                    st.markdown(f":{color}[**{verdict_text}**]")
                                    st.caption(f"Model Confidence: {score_val}%")
                                    st.caption(item.get('note', ''))
                        
                    else:
                        status_container.update(label="❌ Connection Error", state="error")
                        st.error(f"Server Error: {response.status_code}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to Backend. Is Uvicorn running?")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

    # Right Sidebar Logic Explanation
    with col2:
        with st.container(border=True):
            st.info("ℹ️ **How Logic Flows**")
            st.markdown("""
            **1. GNews API (Live)**
            *Checks:* Recent articles (last 30 days) from trusted domains (TOI, BBC, etc.).
            *Success:* Returns **REAL**.
            
            ⬇️ *(If not found)*
            
            **2. Wikipedia + NLI (Fact Check)**
            *Checks:* Established facts & history.
            *Validation:* Uses **NLI Model** to check if Wiki summary confirms or contradicts your claim.
            *Success:* Returns **REAL** or **FAKE (Refuted)**.
            
            ⬇️ *(If no match)*
            
            **3. Indic-BERT (Fallback)**
            *Checks:* Grammar, sensationalism, "Fake News" style patterns.
            *Result:* **Unverified (Likely Real/Fake)**.
            """)

# ==============================================================================
# TAB 2: ANALYTICS INTERFACE
# ==============================================================================
with tab_analytics:
    c1, c2 = st.columns([4, 1])
    with c1:
        st.header("System Performance & Logs")
    with c2:
        if st.button("Refresh Data"):
            st.rerun()
        
    try:
        # Fetch History
        hist_response = requests.get(f"{API_BASE_URL}/history")
        if hist_response.status_code == 200:
            hist_data = hist_response.json()
            df = pd.DataFrame(hist_data)
            
            if not df.empty:
                # Stats Metrics
                total = len(df)
                tier1_count = len(df[df['input_topic'].astype(str).str.contains("Tier 1:", na=False)])
                tier15_count = len(df[df['input_topic'].astype(str).str.contains("Tier 1.5|Tier 1.8", na=False)])
                tier2_count = len(df[df['input_topic'].astype(str).str.contains("Tier 2", na=False)])
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Verifications", total)
                m2.metric("Live News Hits", tier1_count)
                m3.metric("Wiki Fact Checks", tier15_count)
                m4.metric("AI Predictions", tier2_count)
                
                st.divider()
                
                # Charts
                chart1, chart2 = st.columns(2)
                
                with chart1:
                    st.subheader("Verification Methods Used")
                    method_counts = df['input_topic'].value_counts().reset_index()
                    method_counts.columns = ['Method', 'Count']
                    fig_pie = px.pie(method_counts, names='Method', values='Count', hole=0.4, title="Distribution of Logic Tiers")
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                with chart2:
                    st.subheader("Verdict Distribution")
                    # Clean verdict string to just "REAL", "FAKE", "Unverified"
                    df['clean_label'] = df['prediction_label'].apply(lambda x: x.split('(')[0].strip())
                    verdict_counts = df['clean_label'].value_counts().reset_index()
                    verdict_counts.columns = ['Verdict', 'Count']
                    
                    color_map = {"REAL": "green", "FAKE": "red", "Unverified": "orange", "Error": "gray"}
                    fig_bar = px.bar(verdict_counts, x='Verdict', y='Count', color='Verdict', title="Outcome Analysis", color_discrete_map=color_map)
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Data Table
                st.subheader("Recent Request Logs")
                st.dataframe(
                    df[['timestamp', 'input_text', 'input_topic', 'prediction_label']].sort_values(by='timestamp', ascending=False),
                    use_container_width=True,
                    column_config={
                        "timestamp": "Time",
                        "input_text": "Claim",
                        "input_topic": "Method Used",
                        "prediction_label": "Verdict"
                    }
                )
            else:
                st.info("No history data found yet. Try verifying some news!")
    except Exception as e:
        st.error(f"Error loading analytics: {e}")