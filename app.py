import streamlit as st
import pandas as pd
import requests
import os

# --- Configuration ---
CSV_PATH = r'C:\Users\Admin\Desktop\LLM_Stocks\stock_data_summary_20250726_085144.csv'
API_KEY = "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"

# --- API Setup ---
url = "https://api.perplexity.ai/chat/completions"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# --- Page Config ---
st.set_page_config(
    page_title="Stock Scanner AI",
    layout="centered",
    page_icon="📈"
)

# --- Minimal CSS ---
st.markdown("""
<style>
    .main { padding-top: 2rem; }
    .stTextInput > div > div > input { 
        border-radius: 20px; 
        border: 2px solid #e0e0e0;
        padding: 10px 15px;
    }
    .user-msg { 
        background: #f0f8ff; 
        padding: 10px 15px; 
        border-radius: 15px; 
        margin: 5px 0; 
        text-align: right;
    }
    .bot-msg { 
        background: #f5f5f5; 
        padding: 10px 15px; 
        border-radius: 15px; 
        margin: 5px 0; 
    }
    .header { 
        text-align: center; 
        padding: 1rem 0; 
        border-bottom: 1px solid #eee; 
        margin-bottom: 2rem; 
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="header">
    <h1>📈 Stock Scanner AI</h1>
    <p>Fast • Simple • Smart</p>
</div>
""", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_data():
    if not os.path.exists(CSV_PATH):
        return None
    return pd.read_csv(CSV_PATH)

df = load_data()

if df is None:
    st.error("❌ CSV file not found")
    st.stop()

# --- Data Info ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📊 Stocks", len(df))
with col2:
    st.metric("📋 Columns", len(df.columns))
with col3:
    if 'Close' in df.columns:
        st.metric("💰 Avg Price", f"${df['Close'].mean():.2f}")
    else:
        st.metric("🔍 Records", "Ready")

# --- Initialize Chat ---
if 'messages' not in st.session_state:
    st.session_state.messages = []

# --- AI Function ---
def get_response(query):
    try:
        # Simple data context
        context = f"Dataset: {len(df)} stocks, Columns: {list(df.columns)[:5]}"
        sample = df.head(3).to_csv(index=False)
        
        prompt = f"""You are a stock analysis expert. 

Data Context: {context}
Sample Data:
{sample}

Question: {query}

Give a brief, precise answer in 1-2 sentences. Be specific with numbers when possible."""

        payload = {
            "model": "sonar-pro",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 80
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            return "❌ Error getting response"
            
    except Exception as e:
        return f"❌ Error: {str(e)}"

# --- Chat Interface ---
st.markdown("### 💬 Ask about your stocks")

# Chat input
if prompt := st.chat_input("Type your question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get AI response
    with st.spinner("🤔"):
        response = get_response(prompt)
    
    # Add AI response
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display chat
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-msg"><b>You:</b> {message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg"><b>🤖:</b> {message["content"]}</div>', unsafe_allow_html=True)

# --- Quick Actions (Minimal) ---
if len(st.session_state.messages) == 0:
    st.markdown("#### 🚀 Try asking:")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📈 Top performers", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What are the top performing stocks?"})
            with st.spinner("🤔"):
                response = get_response("What are the top performing stocks?")
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    
    with col2:
        if st.button("📊 Market summary", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Give me a market summary"})
            with st.spinner("🤔"):
                response = get_response("Give me a market summary")
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

# --- Data Preview (Collapsible) ---
with st.expander("📋 View Data Sample"):
    st.dataframe(df.head(10), use_container_width=True)

# --- Clear Chat ---
if len(st.session_state.messages) > 0:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()