import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Optional
import asyncio
import warnings
warnings.filterwarnings('ignore')

# --- Enhanced Configuration ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Page Config ---
st.set_page_config(
    page_title="🚀 AI Stock Command Center", 
    layout="wide", 
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# --- Advanced Styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --primary-color: #6366f1;
    --secondary-color: #8b5cf6;
    --success-color: #10b981;
    --danger-color: #ef4444;
    --warning-color: #f59e0b;
    --dark-bg: #0f172a;
    --light-bg: #f8fafc;
}

.main-header {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 10px 25px rgba(99, 102, 241, 0.3);
}

.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    border-left: 4px solid var(--primary-color);
    margin: 1rem 0;
    transition: transform 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

.chat-container {
    background: linear-gradient(145deg, #f8fafc 0%, #e2e8f0 100%);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
}

.user-message {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    color: white;
    padding: 1rem 1.5rem;
    border-radius: 18px 18px 5px 18px;
    margin: 1rem 0 1rem auto;
    max-width: 80%;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    font-weight: 500;
}

.assistant-message {
    background: white;
    color: #1e293b;
    padding: 1rem 1.5rem;
    border-radius: 18px 18px 18px 5px;
    margin: 1rem auto 1rem 0;
    max-width: 85%;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    border: 1px solid #e2e8f0;
}

.stock-alert {
    background: linear-gradient(135deg, var(--success-color) 0%, #059669 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    font-weight: 500;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
}

.sidebar-content {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.performance-badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 0.2rem;
}

.badge-bullish { background: #dcfce7; color: #166534; }
.badge-bearish { background: #fef2f2; color: #dc2626; }
.badge-neutral { background: #f3f4f6; color: #374151; }

* {
    font-family: 'Inter', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>🚀 AI Stock Command Center</h1>
    <p style="font-size: 1.2em; opacity: 0.9;">Advanced AI-Powered Stock Analysis & Insights</p>
    <p style="font-size: 0.9em; opacity: 0.8;">Real-time data • Professional signals • Smart recommendations</p>
</div>
""", unsafe_allow_html=True)

# --- Enhanced Data Loading ---
CSV_PATH = "stock_data_summary_20250802_122812.csv"

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_and_enhance_data():
    """Load and enhance stock data with advanced technical indicators"""
    if not os.path.exists(CSV_PATH):
        return None, None
    
    df = pd.read_csv(CSV_PATH)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Enhanced technical analysis
    df = enhance_technical_indicators(df)
    
    # Generate market summary
    summary = generate_market_summary(df)
    
    return df, summary

def enhance_technical_indicators(df):
    """Add advanced technical indicators and signals"""
    df = df.copy()
    
    # Price action signals
    df['price_above_prev_day_high'] = df['close'] > df['Prev Day High']
    df['price_above_week_high'] = df['close'] > df.get('Previous Week High', df['high'])
    df['price_above_200EMA'] = df['close'] > df['EMA_200']
    df['price_above_50EMA'] = df['close'] > df['EMA_50']
    
    # Advanced RSI analysis
    df['rsi_oversold'] = df['RSI_14'] < 30
    df['rsi_overbought'] = df['RSI_14'] > 70
    df['rsi_zone'] = pd.cut(df['RSI_14'], bins=[0, 30, 40, 60, 70, 100], 
                           labels=['Oversold', 'Weak', 'Neutral', 'Strong', 'Overbought'])
    
    # MACD signals
    df['macd_bullish'] = (df['MACD'] > df['MACD_Signal']) & (df['MACD_Hist'] > 0)
    df['macd_bearish'] = (df['MACD'] < df['MACD_Signal']) & (df['MACD_Hist'] < 0)
    df['macd_trend'] = np.where(df['macd_bullish'], 'Bullish', 
                               np.where(df['macd_bearish'], 'Bearish', 'Neutral'))
    
    # Bollinger Bands analysis
    df['bb_position'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    df['bb_squeeze'] = (df['BB_Upper'] - df['BB_Lower']) / df['close'] < 0.1
    df['bb_breakout_up'] = df['close'] > df['BB_Upper']
    df['bb_breakout_down'] = df['close'] < df['BB_Lower']
    
    # Volume analysis (if available)
    if 'volume' in df.columns:
        df['volume_ma'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).mean())
        df['volume_spike'] = df['volume'] > df['volume_ma'] * 1.5
    
    # Momentum scoring
    df['momentum_score'] = calculate_momentum_score(df)
    df['signal_strength'] = calculate_signal_strength(df)
    
    return df

def calculate_momentum_score(df):
    """Calculate composite momentum score (0-100)"""
    score = 0
    
    # RSI component (30 points)
    rsi_score = np.clip((df['RSI_14'] - 30) / 40 * 30, 0, 30)
    score += rsi_score
    
    # MACD component (25 points)
    macd_score = np.where(df['macd_bullish'], 25, 
                         np.where(df['macd_bearish'], 0, 12.5))
    score += macd_score
    
    # EMA alignment (25 points)
    ema_score = 0
    if 'EMA_20' in df.columns and 'EMA_50' in df.columns:
        ema_score += np.where(df['close'] > df['EMA_20'], 8, 0)
        ema_score += np.where(df['close'] > df['EMA_50'], 8, 0)
        ema_score += np.where(df['EMA_20'] > df['EMA_50'], 9, 0)
    score += ema_score
    
    # Price breakouts (20 points)
    breakout_score = 0
    breakout_score += np.where(df['price_above_prev_day_high'], 10, 0)
    breakout_score += np.where(df['bb_breakout_up'], 10, 0)
    score += breakout_score
    
    return np.clip(score, 0, 100)

def calculate_signal_strength(df):
    """Calculate signal strength: Strong, Moderate, Weak"""
    conditions = [
        df['momentum_score'] >= 75,
        df['momentum_score'] >= 50,
        df['momentum_score'] >= 25
    ]
    choices = ['Strong', 'Moderate', 'Weak']
    return np.select(conditions, choices, default='Very Weak')

def generate_market_summary(df):
    """Generate comprehensive market summary"""
    summary = {}
    
    # Basic stats
    summary['total_stocks'] = len(df)
    summary['bullish_stocks'] = len(df[df['momentum_score'] >= 60])
    summary['bearish_stocks'] = len(df[df['momentum_score'] <= 40])
    summary['breakout_stocks'] = len(df[df['price_above_prev_day_high']])
    
    # Top performers
    summary['top_momentum'] = df.nlargest(5, 'momentum_score')[['symbol', 'momentum_score', 'signal_strength']].to_dict('records')
    summary['recent_breakouts'] = df[df['price_above_prev_day_high']].nlargest(5, 'momentum_score')[['symbol', 'close', 'Prev Day High']].to_dict('records')
    
    # Market sentiment
    bullish_pct = (summary['bullish_stocks'] / summary['total_stocks']) * 100
    summary['market_sentiment'] = 'Bullish' if bullish_pct > 60 else 'Bearish' if bullish_pct < 40 else 'Neutral'
    summary['sentiment_score'] = bullish_pct
    
    return summary

# --- Load Data ---
with st.spinner("🔄 Loading and analyzing market data..."):
    df, market_summary = load_and_enhance_data()

if df is None:
    st.error("❌ Data file not found. Please upload your stock data CSV file.")
    st.stop()

# --- Sidebar: Market Overview ---
with st.sidebar:
    st.markdown("## 📊 Market Overview")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📈 Total Stocks", market_summary['total_stocks'])
        st.metric("🚀 Bullish", market_summary['bullish_stocks'])
    with col2:
        st.metric("📉 Bearish", market_summary['bearish_stocks'])
        st.metric("💥 Breakouts", market_summary['breakout_stocks'])
    
    # Market sentiment gauge
    sentiment_color = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "🟡"}
    st.markdown(f"### {sentiment_color[market_summary['market_sentiment']]} Market Sentiment")
    st.progress(market_summary['sentiment_score']/100)
    st.write(f"**{market_summary['market_sentiment']}** ({market_summary['sentiment_score']:.1f}%)")
    
    # Top momentum stocks
    st.markdown("### 🔥 Top Momentum")
    for stock in market_summary['top_momentum'][:3]:
        badge_class = "badge-bullish" if stock['momentum_score'] > 70 else "badge-neutral"
        st.markdown(f"""
        <div class="performance-badge {badge_class}">
            {stock['symbol']} - {stock['momentum_score']:.0f}
        </div>
        """, unsafe_allow_html=True)

# --- Enhanced AI Query System ---
def get_enhanced_system_prompt():
    return f"""
You are a world-class financial AI assistant with deep expertise in technical analysis and market insights.

CURRENT MARKET STATUS:
- Total Stocks: {market_summary['total_stocks']}
- Bullish Sentiment: {market_summary['sentiment_score']:.1f}%
- Market Mood: {market_summary['market_sentiment']}
- Active Breakouts: {market_summary['breakout_stocks']}

DATASET COLUMNS & SIGNALS:
- symbol: Stock ticker
- close, high, low: Price data
- EMA_20, EMA_50, EMA_100, EMA_200: Moving averages
- RSI_14: Relative Strength Index (30=oversold, 70=overbought)
- MACD, MACD_Signal, MACD_Hist: Trend momentum
- BB_Upper, BB_Lower: Bollinger Bands
- Prev Day High, Previous Week High: Key resistance levels
- momentum_score: Composite score (0-100, higher=stronger bullish momentum)
- signal_strength: Strong/Moderate/Weak/Very Weak

ADVANCED ANALYSIS RULES:
🎯 For breakout queries: Use `price_above_prev_day_high` = True
🎯 For momentum queries: Sort by `momentum_score` descending
🎯 For oversold: RSI_14 < 30 AND momentum_score > 40
🎯 For strong signals: signal_strength = 'Strong'

RESPONSE STYLE:
- Professional yet conversational
- Use relevant emojis strategically
- Provide specific stock symbols with key metrics
- Include actionable insights
- Format with clear structure
- Always explain the reasoning behind recommendations
"""

def query_enhanced_ai_insights(user_query: str, df: pd.DataFrame) -> str:
    """Enhanced AI query processing with better accuracy"""
    try:
        # Smart filtering based on query intent
        filtered_df = smart_filter_stocks(user_query, df)
        
        if filtered_df.empty:
            return "🔍 No stocks match your criteria. Try adjusting your search parameters."
        
        # Prepare enhanced sample data
        sample_size = min(30, len(filtered_df))
        sample = filtered_df.nlargest(sample_size, 'momentum_score').round(2)
        
        # Convert to clean format for AI
        relevant_columns = ['symbol', 'close', 'momentum_score', 'signal_strength', 
                          'RSI_14', 'macd_trend', 'price_above_prev_day_high']
        
        if 'Prev Day High' in sample.columns:
            relevant_columns.append('Prev Day High')
            
        sample_data = sample[relevant_columns].to_dict('records')
        
        prompt = f"""
MARKET ANALYSIS REQUEST:
{user_query}

RELEVANT STOCK DATA:
{json.dumps(sample_data, indent=2)}

ANALYSIS CONTEXT:
- Found {len(filtered_df)} stocks matching criteria
- Showing top {sample_size} by momentum score
- Current market sentiment: {market_summary['market_sentiment']}

Provide professional analysis with specific recommendations.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": get_enhanced_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,  # Lower temperature for more consistent results
            max_tokens=1500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"⚠️ Analysis Error: {str(e)}. Please try rephrasing your query."

def smart_filter_stocks(query: str, df: pd.DataFrame) -> pd.DataFrame:
    """Smart filtering based on query intent"""
    query_lower = query.lower()
    filtered_df = df.copy()
    
    # Breakout filters
    if any(term in query_lower for term in ['breakout', 'above previous day', 'broke above']):
        filtered_df = filtered_df[filtered_df['price_above_prev_day_high'] == True]
    
    # Momentum filters
    if 'high momentum' in query_lower or 'strong momentum' in query_lower:
        filtered_df = filtered_df[filtered_df['momentum_score'] >= 70]
    elif 'momentum' in query_lower:
        filtered_df = filtered_df[filtered_df['momentum_score'] >= 50]
    
    # RSI filters
    if 'oversold' in query_lower:
        filtered_df = filtered_df[filtered_df['RSI_14'] < 35]
    elif 'overbought' in query_lower:
        filtered_df = filtered_df[filtered_df['RSI_14'] > 70]
    
    # MACD filters
    if 'bullish macd' in query_lower:
        filtered_df = filtered_df[filtered_df['macd_trend'] == 'Bullish']
    
    # Signal strength filters
    if 'strong signal' in query_lower:
        filtered_df = filtered_df[filtered_df['signal_strength'] == 'Strong']
    
    # Bollinger band filters
    if 'bollinger' in query_lower and 'breakout' in query_lower:
        filtered_df = filtered_df[filtered_df['bb_breakout_up'] == True]
    
    return filtered_df

# --- Main Chat Interface ---
st.markdown("## 💬 AI Stock Analysis Chat")

# Initialize chat history
if 'enhanced_messages' not in st.session_state:
    st.session_state.enhanced_messages = []
    # Add welcome message
    welcome_msg = f"""
🎯 **Welcome to AI Stock Command Center!**

I've analyzed **{market_summary['total_stocks']} stocks** and found:
- 🚀 **{market_summary['bullish_stocks']} bullish opportunities**
- 💥 **{market_summary['breakout_stocks']} fresh breakouts**  
- 📊 Market sentiment: **{market_summary['market_sentiment']}**

**Try asking:**
- "Show me stocks with strong momentum above 70"
- "Find breakouts above previous day high with good RSI"
- "Which stocks are oversold but showing bullish MACD?"
- "Top 5 stocks for swing trading"
"""
    st.session_state.enhanced_messages.append({"role": "assistant", "content": welcome_msg})

# Chat container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.enhanced_messages:
    css_class = "user-message" if message["role"] == "user" else "assistant-message"
    st.markdown(f'<div class="{css_class}">{message["content"]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("💭 Ask me anything about stocks... (e.g., 'Find high momentum breakouts')"):
    # Add user message
    st.session_state.enhanced_messages.append({"role": "user", "content": prompt})
    
    # Generate AI response
    with st.spinner("🧠 Analyzing market data..."):
        ai_response = query_enhanced_ai_insights(prompt, df)
    
    # Add AI response
    st.session_state.enhanced_messages.append({"role": "assistant", "content": ai_response})
    
    # Rerun to display new messages
    st.rerun()

# --- Quick Action Buttons ---
st.markdown("### ⚡ Quick Analysis")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🚀 High Momentum", use_container_width=True):
        query = "Show me stocks with momentum score above 75 and strong signals"
        st.session_state.enhanced_messages.append({"role": "user", "content": query})
        with st.spinner("Analyzing..."):
            response = query_enhanced_ai_insights(query, df)
        st.session_state.enhanced_messages.append({"role": "assistant", "content": response})
        st.rerun()

with col2:
    if st.button("💥 Fresh Breakouts", use_container_width=True):
        query = "Find stocks breaking above previous day high with good momentum"
        st.session_state.enhanced_messages.append({"role": "user", "content": query})
        with st.spinner("Analyzing..."):
            response = query_enhanced_ai_insights(query, df)
        st.session_state.enhanced_messages.append({"role": "assistant", "content": response})
        st.rerun()

with col3:
    if st.button("📈 Oversold Bounce", use_container_width=True):
        query = "Show oversold stocks with RSI below 35 but bullish MACD signals"
        st.session_state.enhanced_messages.append({"role": "user", "content": query})
        with st.spinner("Analyzing..."):
            response = query_enhanced_ai_insights(query, df)
        st.session_state.enhanced_messages.append({"role": "assistant", "content": response})
        st.rerun()

with col4:
    if st.button("🎯 Swing Trades", use_container_width=True):
        query = "Best swing trading opportunities with strong signals and good risk-reward"
        st.session_state.enhanced_messages.append({"role": "user", "content": query})
        with st.spinner("Analyzing..."):
            response = query_enhanced_ai_insights(query, df)
        st.session_state.enhanced_messages.append({"role": "assistant", "content": response})
        st.rerun()

# --- Data Dashboard ---
with st.expander("📊 Advanced Market Dashboard", expanded=False):
    
    # Market metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_momentum = df['momentum_score'].mean()
        st.metric("📈 Avg Momentum", f"{avg_momentum:.1f}", f"{avg_momentum-50:.1f}")
    
    with col2:
        rsi_oversold = len(df[df['RSI_14'] < 30])
        st.metric("🔄 Oversold Count", rsi_oversold)
    
    with col3:
        macd_bullish = len(df[df['macd_trend'] == 'Bullish'])
        st.metric("🚀 Bullish MACD", macd_bullish)
    
    with col4:
        strong_signals = len(df[df['signal_strength'] == 'Strong'])
        st.metric("💪 Strong Signals", strong_signals)
    
    # Top performers table
    st.markdown("### 🏆 Top 10 Momentum Stocks")
    top_stocks = df.nlargest(10, 'momentum_score')[
        ['symbol', 'close', 'momentum_score', 'signal_strength', 'RSI_14', 'macd_trend']
    ].round(2)
    
    # Style the dataframe
    st.dataframe(
        top_stocks,
        use_container_width=True,
        column_config={
            "symbol": "Symbol",
            "close": st.column_config.NumberColumn("Price ($)", format="%.2f"),
            "momentum_score": st.column_config.ProgressColumn("Momentum", min_value=0, max_value=100),
            "signal_strength": "Signal",
            "RSI_14": st.column_config.NumberColumn("RSI", format="%.1f"),
            "macd_trend": "MACD Trend"
        }
    )

# --- Footer ---
st.markdown("""
---
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 15px; margin-top: 2rem;">
    <h3 style="color: #475569;">🚀 AI Stock Command Center</h3>
    <p style="color: #64748b;">Professional-grade stock analysis powered by advanced AI</p>
    <p style="font-size: 0.9em; color: #94a3b8;">
        Features: Advanced Technical Analysis • Smart Filtering • Real-time Insights • Professional Signals
    </p>
</div>
""", unsafe_allow_html=True)
