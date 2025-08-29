import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# --- Config ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
st.set_page_config(page_title="Stock Screener", layout="wide", page_icon="📈")

# --- Minimal CSS ---
st.markdown("""
<style>
.main { padding: 0.5rem; }
.stButton button { width: 100%; height: 2.5rem; font-size: 0.9rem; }
.metric { text-align: center; }
.chat-msg { padding: 0.5rem; margin: 0.2rem 0; border-radius: 5px; }
.user { background: #e3f2fd; }
.ai { background: #f3e5f5; }
</style>
""", unsafe_allow_html=True)

st.title("📈 AI Stock Screener")

# --- Fast Data Loading ---
CSV_PATH = "stock_data_summary_20250802_122812.csv"

@st.cache_data(ttl=1800, show_spinner=False)
def load_data():
    if not os.path.exists(CSV_PATH):
        return None, None
    
    df = pd.read_csv(CSV_PATH)
    
    # Minimal processing
    df['rsi'] = df.get('RSI_14', 50).fillna(50)
    df['macd_bull'] = (df.get('MACD', 0) > df.get('MACD_Signal', 0)).fillna(False)
    df['above_ema'] = (df['close'] > df.get('EMA_50', df['close'])).fillna(False)
    
    # Simple momentum (0-100)
    df['momentum'] = (
        np.clip((df['rsi'] - 30) / 40 * 50, 0, 50) +
        np.where(df['macd_bull'], 30, 0) +
        np.where(df['above_ema'], 20, 0)
    ).clip(0, 100)
    
    df['breakout'] = df['close'] > df.get('Prev Day High', df['close'])
    
    # Summary
    summary = {
        'total': len(df),
        'bullish': len(df[df['momentum'] > 60]),
        'breakouts': len(df[df['breakout'] == True])
    }
    
    return df, summary

# Load data
df, summary = load_data()
if df is None:
    st.error("CSV file not found!")
    st.stop()

# --- Quick Stats ---
col1, col2, col3 = st.columns(3)
col1.metric("Total", summary['total'])
col2.metric("Bullish", summary['bullish'])
col3.metric("Breakouts", summary['breakouts'])

# --- AI System ---
def get_system_prompt():
    return """You are a stock screener. Return ONLY a simple list of stocks with basic info.

FORMAT EXACTLY LIKE THIS:
• AAPL - $150.25 (Score: 85)
• TSLA - $245.60 (Score: 82)
• NVDA - $420.15 (Score: 79)

RULES:
- Max 10 stocks
- No explanations or advice
- Just symbol, price, momentum score
- Use bullet points only"""

def query_ai(user_query: str, df: pd.DataFrame) -> str:
    try:
        # Filter data
        filtered_df = filter_data(user_query, df)
        if filtered_df.empty:
            return "No stocks found matching criteria."
        
        # Get top 10
        top = filtered_df.nlargest(10, 'momentum')
        stock_data = top[['symbol', 'close', 'momentum']].round(2).to_dict('records')
        
        prompt = f"Query: {user_query}\nData: {json.dumps(stock_data[:10])}\nList stocks in bullet format."
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error: {str(e)}"

def filter_data(query: str, df: pd.DataFrame) -> pd.DataFrame:
    q = query.lower()
    filtered = df.copy()
    
    if 'breakout' in q:
        filtered = filtered[filtered['breakout'] == True]
    if 'high momentum' in q or 'momentum > 70' in q:
        filtered = filtered[filtered['momentum'] > 70]
    elif 'momentum' in q:
        filtered = filtered[filtered['momentum'] > 50]
    if 'oversold' in q:
        filtered = filtered[filtered['rsi'] < 35]
    if 'strong' in q:
        filtered = filtered[filtered['momentum'] > 70]
    
    return filtered

# --- Chat Interface ---
if 'msgs' not in st.session_state:
    st.session_state.msgs = [
        {"role": "ai", "content": f"📊 Ready! Found {summary['bullish']} bullish stocks and {summary['breakouts']} breakouts. Ask me for stock lists!"}
    ]

# Display chat
for msg in st.session_state.msgs:
    css = "user" if msg["role"] == "user" else "ai"
    st.markdown(f'<div class="chat-msg {css}">{msg["content"]}</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask for stocks (e.g., 'high momentum stocks')"):
    st.session_state.msgs.append({"role": "user", "content": prompt})
    
    with st.spinner("🔍"):
        response = query_ai(prompt, df)
    
    st.session_state.msgs.append({"role": "ai", "content": response})
    st.rerun()

# --- Quick Buttons ---
st.markdown("**Quick Screens:**")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🚀 High Momentum"):
        query = "high momentum stocks above 70"
        st.session_state.msgs.append({"role": "user", "content": query})
        with st.spinner("🔍"):
            response = query_ai(query, df)
        st.session_state.msgs.append({"role": "ai", "content": response})
        st.rerun()

with col2:
    if st.button("💥 Breakouts"):
        query = "breakout stocks"
        st.session_state.msgs.append({"role": "user", "content": query})
        with st.spinner("🔍"):
            response = query_ai(query, df)
        st.session_state.msgs.append({"role": "ai", "content": response})
        st.rerun()

with col3:
    if st.button("📈 Oversold"):
        query = "oversold stocks with momentum"
        st.session_state.msgs.append({"role": "user", "content": query})
        with st.spinner("🔍"):
            response = query_ai(query, df)
        st.session_state.msgs.append({"role": "ai", "content": response})
        st.rerun()

with col4:
    if st.button("🎯 Top 10"):
        query = "top momentum stocks"
        st.session_state.msgs.append({"role": "user", "content": query})
        with st.spinner("🔍"):
            response = query_ai(query, df)
        st.session_state.msgs.append({"role": "ai", "content": response})
        st.rerun()

# --- Manual Filter ---
with st.expander("🔧 Manual Filter"):
    col1, col2 = st.columns(2)
    min_mom = col1.slider("Min Momentum", 0, 100, 50)
    max_rsi = col2.slider("Max RSI", 30, 100, 70)
    
    if st.button("Apply Filter"):
        filtered = df[(df['momentum'] >= min_mom) & (df['rsi'] <= max_rsi)].nlargest(15, 'momentum')
        result = ""
        for _, row in filtered.iterrows():
            result += f"• **{row['symbol']}** - ${row['close']:.2f} (Score: {row['momentum']:.0f})\n"
        
        st.session_state.msgs.append({"role": "user", "content": f"Filter: momentum>={min_mom}, RSI<={max_rsi}"})
        st.session_state.msgs.append({"role": "ai", "content": result})
        st.rerun()

# --- Raw Data ---
if st.checkbox("📊 Show Data"):
    display_cols = ['symbol', 'close', 'momentum', 'rsi', 'breakout']
    st.dataframe(df[display_cols].head(30).round(2), use_container_width=True)

st.caption("⚡ Fast AI screening with GPT-4o-mini")
