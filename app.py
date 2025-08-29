import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import plotly.express as px
from openai import OpenAI
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Page Config ---
st.set_page_config(
    page_title="AI Stock Analyzer", 
    layout="wide", 
    page_icon="📈"
)

# --- Optimized Styling ---
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    padding: 1.5rem;
    border-radius: 10px;
    text-align: center;
    color: white;
    margin-bottom: 1rem;
}

.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    border-left: 3px solid #6366f1;
}

.chat-message {
    padding: 0.8rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}

.user-msg { background: #f0f9ff; border-left: 3px solid #0ea5e9; }
.ai-msg { background: #f8fafc; border-left: 3px solid #8b5cf6; }

.stock-item {
    background: white;
    padding: 0.8rem;
    border-radius: 6px;
    margin: 0.3rem 0;
    border-left: 3px solid #10b981;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h2>📈 AI Stock Analyzer</h2>
    <p>Quick stock insights and screening</p>
</div>
""", unsafe_allow_html=True)

# --- Optimized Data Loading ---
CSV_PATH = "stock_data_summary_20250802_122812.csv"

@st.cache_data(ttl=600)
def load_stock_data():
    """Load and process stock data efficiently"""
    if not os.path.exists(CSV_PATH):
        return None, None
    
    df = pd.read_csv(CSV_PATH)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Essential calculations only
    df['momentum_score'] = calculate_simple_momentum(df)
    df['signal'] = get_simple_signal(df)
    df['breakout'] = df['close'] > df.get('Prev Day High', df['high'])
    
    # Quick market summary
    summary = {
        'total': len(df),
        'bullish': len(df[df['momentum_score'] > 60]),
        'breakouts': len(df[df['breakout'] == True]),
        'top_stocks': df.nlargest(5, 'momentum_score')[['symbol', 'close', 'momentum_score']].to_dict('records')
    }
    
    return df, summary

def calculate_simple_momentum(df):
    """Simple momentum calculation"""
    score = 0
    
    # RSI component (0-50)
    if 'RSI_14' in df.columns:
        score += np.clip((df['RSI_14'] - 30) / 40 * 50, 0, 50)
    
    # MACD component (0-30)
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        score += np.where(df['MACD'] > df['MACD_Signal'], 30, 0)
    
    # Price vs EMA (0-20)
    if 'EMA_50' in df.columns:
        score += np.where(df['close'] > df['EMA_50'], 20, 0)
    
    return np.clip(score, 0, 100)

def get_simple_signal(df):
    """Simple signal classification"""
    return np.where(df['momentum_score'] >= 70, 'Strong',
           np.where(df['momentum_score'] >= 50, 'Moderate', 'Weak'))

# --- Load Data ---
@st.cache_resource
def initialize_data():
    return load_stock_data()

with st.spinner("Loading data..."):
    df, market_summary = initialize_data()

if df is None:
    st.error("Data file not found. Please upload your CSV file.")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.markdown("## Market Overview")
    
    st.metric("Total Stocks", market_summary['total'])
    st.metric("Bullish", market_summary['bullish'])
    st.metric("Breakouts", market_summary['breakouts'])
    
    st.markdown("### Top Momentum")
    for stock in market_summary['top_stocks'][:3]:
        st.markdown(f"**{stock['symbol']}** - {stock['momentum_score']:.0f}")

# --- Simplified AI System ---
def get_system_prompt():
    return """You are a stock screening assistant. Provide concise, factual responses.

RULES:
- List specific stock symbols with key metrics
- Keep responses under 200 words
- No buy/sell recommendations
- Focus on technical screening only
- Use bullet points for clarity

Available data: symbol, close price, RSI_14, MACD, EMA_50, momentum_score (0-100)
"""

def query_stocks(user_query: str, df: pd.DataFrame) -> str:
    """Process user query and return stock list"""
    try:
        # Simple filtering
        filtered_df = filter_stocks(user_query, df)
        
        if filtered_df.empty:
            return "No stocks match your criteria."
        
        # Get top 10 results
        top_stocks = filtered_df.nlargest(10, 'momentum_score')
        
        # Prepare data for AI
        stock_data = top_stocks[['symbol', 'close', 'momentum_score', 'signal', 'RSI_14']].round(2).to_dict('records')
        
        prompt = f"Query: {user_query}\n\nStock Data: {json.dumps(stock_data[:8])}\n\nProvide a concise stock list with key metrics."
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=300
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error: {str(e)}"

def filter_stocks(query: str, df: pd.DataFrame) -> pd.DataFrame:
    """Filter stocks based on query"""
    query_lower = query.lower()
    filtered_df = df.copy()
    
    if 'breakout' in query_lower:
        filtered_df = filtered_df[filtered_df['breakout'] == True]
    
    if 'high momentum' in query_lower:
        filtered_df = filtered_df[filtered_df['momentum_score'] >= 70]
    elif 'momentum' in query_lower:
        filtered_df = filtered_df[filtered_df['momentum_score'] >= 50]
    
    if 'oversold' in query_lower and 'RSI_14' in df.columns:
        filtered_df = filtered_df[filtered_df['RSI_14'] < 35]
    
    if 'strong' in query_lower:
        filtered_df = filtered_df[filtered_df['signal'] == 'Strong']
    
    return filtered_df

# --- Chat Interface ---
st.markdown("## Chat")

if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": f"📊 Analyzed {market_summary['total']} stocks. Found {market_summary['bullish']} bullish opportunities and {market_summary['breakouts']} breakouts. Ask me about specific stock screens!"}
    ]

# Display messages
for msg in st.session_state.messages:
    css_class = "user-msg" if msg["role"] == "user" else "ai-msg"
    st.markdown(f'<div class="chat-message {css_class}">{msg["content"]}</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask for stock screens (e.g., 'high momentum stocks')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Analyzing..."):
        response = query_stocks(prompt, df)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# --- Quick Buttons ---
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🚀 High Momentum"):
        query = "Show high momentum stocks above 75"
        st.session_state.messages.append({"role": "user", "content": query})
        with st.spinner("Analyzing..."):
            response = query_stocks(query, df)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

with col2:
    if st.button("💥 Breakouts"):
        query = "Find breakout stocks above previous day high"
        st.session_state.messages.append({"role": "user", "content": query})
        with st.spinner("Analyzing..."):
            response = query_stocks(query, df)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

with col3:
    if st.button("📈 Strong Signals"):
        query = "Show stocks with strong signals"
        st.session_state.messages.append({"role": "user", "content": query})
        with st.spinner("Analyzing..."):
            response = query_stocks(query, df)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# --- Data Table ---
if st.checkbox("Show Top Stocks Table"):
    top_20 = df.nlargest(20, 'momentum_score')[['symbol', 'close', 'momentum_score', 'signal', 'RSI_14']].round(2)
    st.dataframe(top_20, use_container_width=True)

# --- Performance Info ---
st.markdown("---")
st.caption("Optimized for speed • Simplified analysis • No financial advice")
