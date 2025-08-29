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
    return """You are an advanced NLP stock screener for Indian stock market. Understand natural language queries and return simple stock lists.

UNDERSTAND QUERIES LIKE:
- "show me breakout stocks" → filter breakout=True
- "which stocks have RSI below 30?" → filter rsi<30
- "find momentum above 75" → filter momentum>75
- "oversold stocks that might bounce" → rsi<35 AND momentum>40
- "strong stocks breaking out" → breakout=True AND momentum>70
- "stocks above ₹1000" → filter price>1000
- "cheap stocks under ₹100" → filter price<100
- "high volume stocks" → sort by volume
- "stocks near 52 week high" → close price near year high

INDIAN STOCK CONTEXT:
- Prices are in Indian Rupees (₹)
- Stocks like MIDCPNIFTY, ZFCVINDIA, ZENTEC, ZENSARTECH, ZEEL
- Price ranges from ₹100 to ₹13000+
- Volume in lakhs/crores

FORMAT EXACTLY LIKE THIS:
• MIDCPNIFTY - ₹12930.45 (Score: 85)
• ZFCVINDIA - ₹1350.00 (Score: 82)
• ZENTEC - ₹1773.20 (Score: 79)

RULES:
- Max 10 stocks
- No explanations or advice
- Use ₹ symbol for Indian Rupees
- Just symbol, price, momentum score
- Use bullet points only
- Understand natural language queries"""

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
    
    # Enhanced NLP understanding for Indian stocks
    if any(word in q for word in ['breakout', 'broke above', 'breaking out', 'breakthrough']):
        filtered = filtered[filtered['breakout'] == True]
    
    if any(word in q for word in ['high momentum', 'momentum above', 'momentum > 70', 'strong momentum']):
        filtered = filtered[filtered['momentum'] > 70]
    elif any(word in q for word in ['momentum', 'mom']):
        filtered = filtered[filtered['momentum'] > 50]
    
    if any(word in q for word in ['oversold', 'rsi below', 'rsi < 30', 'rsi under']):
        filtered = filtered[filtered['rsi'] < 35]
    
    if any(word in q for word in ['overbought', 'rsi above 70', 'rsi > 70']):
        filtered = filtered[filtered['rsi'] > 70]
    
    if any(word in q for word in ['strong', 'powerful', 'top', 'best']):
        filtered = filtered[filtered['momentum'] > 70]
    
    if any(word in q for word in ['bounce', 'reversal', 'recovery', 'turnaround']):
        filtered = filtered[(filtered['rsi'] < 35) & (filtered['momentum'] > 40)]
    
    # Price filters in Indian Rupees
    if any(word in q for word in ['expensive', 'above 1000', '> 1000', 'high price']):
        filtered = filtered[filtered['close'] > 1000]
    
    if any(word in q for word in ['cheap', 'under 100', '< 100', 'low price', 'below 100']):
        filtered = filtered[filtered['close'] < 100]
    
    if any(word in q for word in ['mid cap', 'medium price', '200 to 1000']):
        filtered = filtered[(filtered['close'] >= 200) & (filtered['close'] <= 1000)]
    
    if any(word in q for word in ['penny stocks', 'very cheap', 'under 50']):
        filtered = filtered[filtered['close'] < 50]
    
    # Volume filters
    if any(word in q for word in ['high volume', 'active', 'heavy trading']):
        if 'volume' in filtered.columns:
            median_vol = filtered['volume'].median()
            filtered = filtered[filtered['volume'] > median_vol * 2]
    
    # 52 week filters
    if any(word in q for word in ['52 week high', 'yearly high', 'near high']):
        if 'year' in filtered.columns:
            filtered = filtered[filtered['close'] >= filtered['year'] * 0.95]
    
    # Number filters
    if 'above 75' in q or '> 75' in q:
        filtered = filtered[filtered['momentum'] > 75]
    if 'above 80' in q or '> 80' in q:
        filtered = filtered[filtered['momentum'] > 80]
    if 'below 30' in q or '< 30' in q:
        filtered = filtered[filtered['rsi'] < 30]
    
    return filtered

# --- NLP Chat Interface ---
st.markdown("**💬 AI Chat - Ask in Natural Language:**")

if 'msgs' not in st.session_state:
    st.session_state.msgs = [
        {"role": "ai", "content": f"📊 Ready! Found {summary['bullish']} bullish stocks and {summary['breakouts']} breakouts. Ask me anything about stocks!"}
    ]

# Display chat
for msg in st.session_state.msgs:
    css = "user" if msg["role"] == "user" else "ai"
    st.markdown(f'<div class="chat-msg {css}">{msg["content"]}</div>', unsafe_allow_html=True)

# Chat input with NLP examples for Indian stocks
if prompt := st.chat_input("💭 Ask in Hindi/English: 'saste stocks dikhao', 'expensive stocks above ₹1000', 'breakout stocks', 'RSI 30 ke neeche'"):
    st.session_state.msgs.append({"role": "user", "content": prompt})
    
    with st.spinner("🧠 AI analyzing..."):
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
