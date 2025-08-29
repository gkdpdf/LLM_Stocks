import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# --- Minimal Config ---
st.set_page_config(page_title="Stock Screener", layout="wide", page_icon="📈")

# --- Ultra-Light Styling ---
st.markdown("""
<style>
.main { padding: 1rem; }
.stButton button { width: 100%; height: 3rem; }
.metric { background: #f8f9fa; padding: 0.5rem; border-radius: 4px; margin: 0.2rem; }
.stock-list { font-family: monospace; background: #f8f9fa; padding: 1rem; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Stock Screener")

# --- Ultra-Fast Data Loading ---
CSV_PATH = "stock_data_summary_20250802_122812.csv"

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    if not os.path.exists(CSV_PATH):
        return None
    
    df = pd.read_csv(CSV_PATH)
    
    # Ultra-minimal processing
    if 'RSI_14' in df.columns:
        df['rsi'] = df['RSI_14'].fillna(50)
    else:
        df['rsi'] = 50
        
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        df['macd_bull'] = df['MACD'] > df['MACD_Signal']
    else:
        df['macd_bull'] = False
        
    # Simple momentum (0-100)
    df['momentum'] = ((df['rsi'] - 30) / 40 * 60).clip(0, 60) + np.where(df['macd_bull'], 40, 0)
    df['momentum'] = df['momentum'].clip(0, 100)
    
    # Breakout check
    if 'Prev Day High' in df.columns:
        df['breakout'] = df['close'] > df['Prev Day High']
    else:
        df['breakout'] = False
    
    return df

# Load data instantly
df = load_data()
if df is None:
    st.error("CSV file not found!")
    st.stop()

# --- Quick Stats (No AI) ---
total = len(df)
high_mom = len(df[df['momentum'] > 70])
breakouts = len(df[df['breakout'] == True])
oversold = len(df[df['rsi'] < 30]) if 'rsi' in df.columns else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total", total)
col2.metric("High Momentum", high_mom)
col3.metric("Breakouts", breakouts)
col4.metric("Oversold", oversold)

# --- Instant Stock Lists (No AI) ---
def get_stock_list(filter_type):
    if filter_type == "high_momentum":
        stocks = df[df['momentum'] > 70].nlargest(15, 'momentum')
        return stocks[['symbol', 'close', 'momentum']].round(2)
    
    elif filter_type == "breakouts":
        stocks = df[df['breakout'] == True].nlargest(15, 'momentum')
        return stocks[['symbol', 'close', 'momentum']].round(2)
    
    elif filter_type == "oversold":
        stocks = df[(df['rsi'] < 35) & (df['momentum'] > 30)].nlargest(15, 'momentum')
        return stocks[['symbol', 'close', 'rsi']].round(2)
    
    elif filter_type == "strong_macd":
        stocks = df[(df['macd_bull'] == True) & (df['momentum'] > 50)].nlargest(15, 'momentum')
        return stocks[['symbol', 'close', 'momentum']].round(2)
    
    else:  # top_momentum
        stocks = df.nlargest(15, 'momentum')
        return stocks[['symbol', 'close', 'momentum']].round(2)

# --- Instant Buttons ---
st.markdown("### Quick Screens")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🚀 HIGH MOMENTUM (70+)"):
        result = get_stock_list("high_momentum")
        st.markdown("**HIGH MOMENTUM STOCKS:**")
        for _, row in result.iterrows():
            st.markdown(f"• **{row['symbol']}** - ${row['close']} (Score: {row['momentum']:.0f})")

with col2:
    if st.button("💥 BREAKOUTS"):
        result = get_stock_list("breakouts")
        st.markdown("**BREAKOUT STOCKS:**")
        for _, row in result.iterrows():
            st.markdown(f"• **{row['symbol']}** - ${row['close']} (Score: {row['momentum']:.0f})")

with col3:
    if st.button("📈 OVERSOLD BOUNCE"):
        result = get_stock_list("oversold")
        st.markdown("**OVERSOLD STOCKS:**")
        for _, row in result.iterrows():
            st.markdown(f"• **{row['symbol']}** - ${row['close']} (RSI: {row['rsi']:.0f})")

col4, col5 = st.columns(2)

with col4:
    if st.button("💪 STRONG MACD"):
        result = get_stock_list("strong_macd")
        st.markdown("**STRONG MACD STOCKS:**")
        for _, row in result.iterrows():
            st.markdown(f"• **{row['symbol']}** - ${row['close']} (Score: {row['momentum']:.0f})")

with col5:
    if st.button("🏆 TOP MOMENTUM"):
        result = get_stock_list("top_momentum")
        st.markdown("**TOP MOMENTUM STOCKS:**")
        for _, row in result.iterrows():
            st.markdown(f"• **{row['symbol']}** - ${row['close']} (Score: {row['momentum']:.0f})")

# --- Manual Filters ---
st.markdown("### Manual Filters")

col1, col2 = st.columns(2)
with col1:
    min_momentum = st.slider("Min Momentum", 0, 100, 50)
with col2:
    max_rsi = st.slider("Max RSI", 30, 100, 70)

if st.button("🔍 APPLY FILTERS"):
    filtered = df[(df['momentum'] >= min_momentum) & (df['rsi'] <= max_rsi)].nlargest(20, 'momentum')
    st.markdown("**FILTERED STOCKS:**")
    for _, row in filtered.iterrows():
        st.markdown(f"• **{row['symbol']}** - ${row['close']:.2f} (Mom: {row['momentum']:.0f}, RSI: {row['rsi']:.0f})")

# --- Raw Data Toggle ---
if st.checkbox("Show Raw Data"):
    display_df = df[['symbol', 'close', 'momentum', 'rsi', 'breakout']].round(2)
    st.dataframe(display_df.head(50), use_container_width=True)

st.caption("⚡ Ultra-fast screening • No AI delays • Pure data")
