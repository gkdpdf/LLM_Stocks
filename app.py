import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# --- Load OpenAI API Key ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Config ---
CSV_PATH = "stock_data_summary_20250802_122812.csv"
st.set_page_config(page_title="📱 AI Stock Scanner", layout="centered", page_icon="📱")

# --- WhatsApp-style CSS ---
st.markdown("""
<style>
body { background-color: #e5ddd5; }
.user-message {
    background-color: #dcf8c6; text-align: right;
    padding: 10px; border-radius: 10px 10px 0px 10px;
    margin: 10px 0; word-wrap: break-word;
}
.bot-message {
    background-color: #ffffff; text-align: left;
    padding: 10px; border-radius: 10px 10px 10px 0px;
    margin: 10px 0; word-wrap: break-word; border: 1px solid #ccc;
}
.chat-box {
    background-color: #f0f0f0; border-radius: 10px;
    padding: 20px; max-width: 700px; margin: auto;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div style='text-align:center;'><h2>📱 AI Stock Assistant</h2><p>Chat with your stock data like WhatsApp!</p></div>", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_data():
    if not os.path.exists(CSV_PATH):
        return None
    df = pd.read_csv(CSV_PATH)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

df = load_data()
if df is None:
    st.error("❌ CSV file not found.")
    st.stop()

# --- Enhance DataFrame with Signals ---
def enhance_with_signals(df):
    df = df.copy()
    df['price_above_prev_day_high'] = df['close'] > df['Prev Day High']
    df['price_change'] = df['close'] - df['EMA_50']
    df['price_above_200EMA'] = df['close'] > df['EMA_200']
    df['rsi_zone'] = pd.cut(df['RSI_14'], bins=[0, 30, 70, 100], labels=['Oversold', 'Neutral', 'Overbought'])
    df['macd_trend'] = np.where(df['MACD'] > df['MACD_Signal'], 'Bullish', 'Bearish')
    df['bb_position'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    return df

# --- System Prompt with Column Descriptions and Logic Hint ---
def system_prompt():
    return (
        "You are a WhatsApp-style financial assistant that helps analyze a stock dataset. "
        "Columns include:\n\n"
        "- `symbol`: Stock ticker\n"
        "- `close`, `high`, `low`: Price fields\n"
        "- `EMA_20`, `EMA_50`, `EMA_100`, `EMA_200`: Moving averages\n"
        "- `RSI_14`: Relative Strength Index\n"
        "- `MACD`, `MACD_Signal`, `MACD_Hist`: Trend signals\n"
        "- `BB_Upper`, `BB_Lower`: Bollinger Bands\n"
        "- `Prev Day High`: Yesterday’s high\n"
        "- `Previous Week High`, etc.: Weekly levels\n"
        "- `1D_PctRange`, `4D_PctRange`: % change past days\n\n"
        "🚨 RULE: If user asks for 'stocks above previous day high', return only those where `close` > `Prev Day High`. Do not use `high` > `Prev Day High`.\n\n"
        "💬 Reply concisely with emojis and relevant stocks only, like a WhatsApp chat."
    )

# --- GPT Query Logic ---
def query_ai_stock_insights(user_query, df):
    try:
        df = enhance_with_signals(df)

        # Auto-filter if user wants only stocks above previous day high
        if "above previous day high" in user_query.lower():
            df = df[df['price_above_prev_day_high'] == True]

        sample = df.sample(n=min(50, len(df)), random_state=42).copy()
        sample = sample.round(2)
        sample = sample.astype({col: str for col in sample.select_dtypes(include='datetime64').columns})
        sample_dict = sample.to_dict('records')

        prompt = f"""
USER QUERY:
{user_query}

--- STOCK DATA SAMPLE START ---
{json.dumps(sample_dict, indent=2)}
--- STOCK DATA SAMPLE END ---

🎯 Respond only with relevant stocks. Format like a WhatsApp chat. Use emojis.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ AI Analysis Error: {e}"

# --- WhatsApp Chat UI ---
st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
if 'messages' not in st.session_state:
    st.session_state.messages = []

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("💭 Thinking..."):
        ai_reply = query_ai_stock_insights(prompt, df)
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})

for msg in st.session_state.messages:
    css_class = 'user-message' if msg['role'] == 'user' else 'bot-message'
    st.markdown(f"<div class='{css_class}'>{msg['content']}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
---
<div style="text-align:center; color: #555; font-size: 12px;">
<br>
Ask about RSI, MACD, breakouts, and trend levels 📈
</div>
""", unsafe_allow_html=True)
