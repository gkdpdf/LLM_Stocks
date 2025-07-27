import streamlit as st
import pandas as pd
import requests
import os
import json
import numpy as np
from datetime import datetime, timedelta
from euriai import EuriaiClient

# --- Configuration ---
CSV_PATH = r'stock_data_summary_20250726_085144.csv'

# EuriAI Configuration
euron_key = 'euri-579485c6e40c4fa4b2405dc944f3e5c1482479c6264698d7de4a510d6400a210'

client = EuriaiClient(
    api_key=euron_key,
    model="gpt-4.1-nano"
)

# --- Page Config ---
st.set_page_config(
    page_title="Stock Scanner AI",
    layout="wide",
    page_icon="📈"
)

# --- Enhanced CSS ---
st.markdown("""
<style>
    .main { padding-top: 2rem; }
    .stTextInput > div > div > input { 
        border-radius: 20px; 
        border: 2px solid #e0e0e0;
        padding: 10px 15px;
    }
    .user-msg { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 18px; 
        border-radius: 18px; 
        margin: 8px 0; 
        text-align: right;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .bot-msg { 
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 12px 18px; 
        border-radius: 18px; 
        margin: 8px 0; 
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .header { 
        text-align: center; 
        padding: 1rem 0; 
        border-bottom: 2px solid #667eea; 
        margin-bottom: 2rem; 
    }
    .stock-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="header">
    <h1>📈 Advanced Stock Scanner AI</h1>
    <p>Technical Analysis • Smart Filtering • AI-Powered Insights</p>
</div>
""", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_data():
    if not os.path.exists(CSV_PATH):
        return None
    df = pd.read_csv(CSV_PATH)
    # Convert date column to datetime if it exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

df = load_data()

if df is None:
    st.error("❌ CSV file not found. Please ensure 'stock_data_summary_20250726_085144.csv' exists in the current directory.")
    st.stop()

# --- Data Info Dashboard ---
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("📊 Total Stocks", len(df))
with col2:
    st.metric("💰 Avg Close", f"₹{df['close'].mean():.2f}")
with col3:
    st.metric("📈 Avg Volume", f"{df['volume'].mean():,.0f}")
with col4:
    st.metric("📉 Avg RSI", f"{df['RSI_14'].mean():.1f}")
with col5:
    st.metric("⚡ Signals", len(df[df['RSI_14'] < 30]) + len(df[df['RSI_14'] > 70]))

# --- Initialize Chat ---
if 'messages' not in st.session_state:
    st.session_state.messages = []

# --- Enhanced Data Processing Functions ---
def calculate_technical_signals(df):
    """Calculate additional technical signals"""
    df_copy = df.copy()
    
    # Price change calculations
    df_copy['price_change'] = df_copy['close'] - df_copy['open']
    df_copy['price_change_pct'] = (df_copy['price_change'] / df_copy['open']) * 100
    
    # Bollinger Bands signals
    df_copy['bb_position'] = (df_copy['close'] - df_copy['BB_Lower']) / (df_copy['BB_Upper'] - df_copy['BB_Lower'])
    
    # MACD signals
    df_copy['macd_signal'] = np.where(df_copy['MACD'] > df_copy['MACD_Signal'], 'Bullish', 'Bearish')
    
    # RSI categories
    df_copy['rsi_category'] = pd.cut(df_copy['RSI_14'], 
                                   bins=[0, 30, 70, 100], 
                                   labels=['Oversold', 'Neutral', 'Overbought'])
    
    # Volume analysis
    df_copy['volume_ratio'] = df_copy['volume'] / df_copy['volume'].mean()
    
    return df_copy

def get_system_prompt():
    """Enhanced system prompt for better stock analysis"""
    return """You are an expert stock market analyst with deep knowledge of technical analysis, fundamental analysis, and market trends. 

Your task is to analyze stock data and provide accurate, data-driven insights based on the following dataset columns:
- Basic Data: date, open, high, low, close, volume, symbol
- Technical Indicators: EMA_20, EMA_44, EMA_50, EMA_100, EMA_200 (Exponential Moving Averages)
- Bollinger Bands: BB_Middle, BB_Upper, BB_Lower
- Momentum: RSI_14 (Relative Strength Index), MACD, MACD_Signal, MACD_Hist
- Trend: ADX_14 (Average Directional Index)
- Volume: VWAP (Volume Weighted Average Price)
- Historical Data: Prev Day High/Low, Previous Week High/Low, Previous Quarter Profit

Key Analysis Guidelines:
1. RSI < 30 = Oversold (potential buy signal)
2. RSI > 70 = Overbought (potential sell signal)  
3. MACD > MACD_Signal = Bullish momentum
4. Price above EMA_20 = Short-term uptrend
5. High volume (>2x average) = Strong conviction
6. ADX > 25 = Strong trend
7. Price near BB_Upper = Potential resistance
8. Price near BB_Lower = Potential support

Always provide:
- Specific stock symbols and current prices
- Technical reasoning for recommendations
- Risk assessment
- Actionable insights
- Data-backed conclusions only

Format responses with clear structure and use emojis for better readability."""

def analyze_stocks_with_ai(query, df):
    """Enhanced AI-powered stock analysis"""
    try:
        # Calculate additional technical signals
        df_enhanced = calculate_technical_signals(df)
        
        # Prepare relevant data summary for AI
        market_summary = {
            "total_stocks": len(df_enhanced),
            "avg_rsi": df_enhanced['RSI_14'].mean(),
            "oversold_stocks": len(df_enhanced[df_enhanced['RSI_14'] < 30]),
            "overbought_stocks": len(df_enhanced[df_enhanced['RSI_14'] > 70]),
            "bullish_macd": len(df_enhanced[df_enhanced['MACD'] > df_enhanced['MACD_Signal']]),
            "high_volume_stocks": len(df_enhanced[df_enhanced['volume'] > df_enhanced['volume'].mean() * 2]),
            "strong_trend_stocks": len(df_enhanced[df_enhanced['ADX_14'] > 25])
        }
        
        # Get top/bottom performers for context
        top_performers = df_enhanced.nlargest(5, 'price_change_pct')[['symbol', 'close', 'price_change_pct', 'RSI_14', 'volume']].to_dict('records')
        bottom_performers = df_enhanced.nsmallest(5, 'price_change_pct')[['symbol', 'close', 'price_change_pct', 'RSI_14', 'volume']].to_dict('records')
        
        # Create comprehensive prompt
        system_prompt = get_system_prompt()
        
        user_prompt = f"""
Query: {query}

Market Summary:
- Total Stocks: {market_summary['total_stocks']}
- Average RSI: {market_summary['avg_rsi']:.1f}
- Oversold Stocks (RSI < 30): {market_summary['oversold_stocks']}
- Overbought Stocks (RSI > 70): {market_summary['overbought_stocks']}
- Bullish MACD Signals: {market_summary['bullish_macd']}
- High Volume Stocks: {market_summary['high_volume_stocks']}
- Strong Trend Stocks (ADX > 25): {market_summary['strong_trend_stocks']}

Top 5 Performers:
{json.dumps(top_performers, indent=2)}

Bottom 5 Performers:
{json.dumps(bottom_performers, indent=2)}

Available columns for analysis: {list(df_enhanced.columns)}

Please provide a detailed analysis addressing the user's query with specific stock recommendations, technical reasoning, and actionable insights.
"""

        # Generate AI response
        response = client.generate_completion(
            prompt=f"{system_prompt}\n\n{user_prompt}",
            temperature=0.3,
            max_tokens=800
        )
        
        return response
        
    except Exception as e:
        return f"❌ AI Analysis Error: {str(e)}"

def get_rule_based_analysis(query, df):
    """Enhanced rule-based analysis with technical indicators"""
    try:
        df_enhanced = calculate_technical_signals(df)
        query_lower = query.lower()
        results = []
        
        if any(word in query_lower for word in ['oversold', 'buy', 'undervalued']):
            oversold = df_enhanced[df_enhanced['RSI_14'] < 30].nlargest(10, 'volume')
            results.append("🔥 **OVERSOLD STOCKS (RSI < 30) - Potential Buy Opportunities:**\n")
            for i, (_, row) in enumerate(oversold.iterrows(), 1):
                results.append(f"{i}. **{row['symbol']}** - ₹{row['close']:.2f} | RSI: {row['RSI_14']:.1f} | Vol: {row['volume']:,}")
                
        elif any(word in query_lower for word in ['overbought', 'sell', 'overvalued']):
            overbought = df_enhanced[df_enhanced['RSI_14'] > 70].nlargest(10, 'volume')
            results.append("⚠️ **OVERBOUGHT STOCKS (RSI > 70) - Potential Sell Signals:**\n")
            for i, (_, row) in enumerate(overbought.iterrows(), 1):
                results.append(f"{i}. **{row['symbol']}** - ₹{row['close']:.2f} | RSI: {row['RSI_14']:.1f} | Vol: {row['volume']:,}")
                
        elif any(word in query_lower for word in ['bullish', 'momentum', 'macd']):
            bullish = df_enhanced[df_enhanced['MACD'] > df_enhanced['MACD_Signal']].nlargest(10, 'MACD_Hist')
            results.append("🚀 **BULLISH MOMENTUM (MACD > Signal):**\n")
            for i, (_, row) in enumerate(bullish.iterrows(), 1):
                results.append(f"{i}. **{row['symbol']}** - ₹{row['close']:.2f} | MACD: {row['MACD']:.2f} | Vol: {row['volume']:,}")
                
        elif any(word in query_lower for word in ['high volume', 'active', 'volume']):
            high_vol = df_enhanced.nlargest(10, 'volume')
            results.append("📊 **HIGH VOLUME STOCKS:**\n")
            for i, (_, row) in enumerate(high_vol.iterrows(), 1):
                results.append(f"{i}. **{row['symbol']}** - ₹{row['close']:.2f} | Vol: {row['volume']:,} | RSI: {row['RSI_14']:.1f}")
                
        elif any(word in query_lower for word in ['trending', 'adx', 'strong trend']):
            trending = df_enhanced[df_enhanced['ADX_14'] > 25].nlargest(10, 'ADX_14')
            results.append("📈 **STRONG TRENDING STOCKS (ADX > 25):**\n")
            for i, (_, row) in enumerate(trending.iterrows(), 1):
                results.append(f"{i}. **{row['symbol']}** - ₹{row['close']:.2f} | ADX: {row['ADX_14']:.1f} | RSI: {row['RSI_14']:.1f}")
                
        elif any(word in query_lower for word in ['gainers', 'winners', 'top perform']):
            gainers = df_enhanced.nlargest(10, 'price_change_pct')
            results.append("🏆 **TOP GAINERS:**\n")
            for i, (_, row) in enumerate(gainers.iterrows(), 1):
                change_pct = row['price_change_pct']
                results.append(f"{i}. **{row['symbol']}** - ₹{row['close']:.2f} | Change: {change_pct:+.2f}% | Vol: {row['volume']:,}")
                
        elif any(word in query_lower for word in ['losers', 'decliners', 'worst']):
            losers = df_enhanced.nsmallest(10, 'price_change_pct')
            results.append("📉 **TOP LOSERS:**\n")
            for i, (_, row) in enumerate(losers.iterrows(), 1):
                change_pct = row['price_change_pct']
                results.append(f"{i}. **{row['symbol']}** - ₹{row['close']:.2f} | Change: {change_pct:+.2f}% | Vol: {row['volume']:,}")
                
        elif any(word in query_lower for word in ['breakout', 'bollinger', 'resistance']):
            # Stocks near upper Bollinger Band
            breakout = df_enhanced[df_enhanced['bb_position'] > 0.8].nlargest(10, 'volume')
            results.append("🔥 **POTENTIAL BREAKOUT STOCKS (Near Upper BB):**\n")
            for i, (_, row) in enumerate(breakout.iterrows(), 1):
                results.append(f"{i}. **{row['symbol']}** - ₹{row['close']:.2f} | BB Pos: {row['bb_position']:.2f} | RSI: {row['RSI_14']:.1f}")
                
        elif any(word in query_lower for word in ['support', 'bounce', 'oversold bounce']):
            # Stocks near lower Bollinger Band with low RSI
            support = df_enhanced[(df_enhanced['bb_position'] < 0.2) & (df_enhanced['RSI_14'] < 40)].nlargest(10, 'volume')
            results.append("🛡️ **STOCKS NEAR SUPPORT (Lower BB + Low RSI):**\n")
            for i, (_, row) in enumerate(support.iterrows(), 1):
                results.append(f"{i}. **{row['symbol']}** - ₹{row['close']:.2f} | BB Pos: {row['bb_position']:.2f} | RSI: {row['RSI_14']:.1f}")
                
        elif any(word in query_lower for word in ['summary', 'overview', 'market']):
            # Market overview
            avg_rsi = df_enhanced['RSI_14'].mean()
            bullish_count = len(df_enhanced[df_enhanced['MACD'] > df_enhanced['MACD_Signal']])
            oversold_count = len(df_enhanced[df_enhanced['RSI_14'] < 30])
            overbought_count = len(df_enhanced[df_enhanced['RSI_14'] > 70])
            
            results.append("📊 **MARKET OVERVIEW:**\n")
            results.append(f"• **Total Stocks Analyzed:** {len(df_enhanced)}")
            results.append(f"• **Average RSI:** {avg_rsi:.1f}")
            results.append(f"• **Bullish MACD Signals:** {bullish_count} ({bullish_count/len(df_enhanced)*100:.1f}%)")
            results.append(f"• **Oversold Stocks:** {oversold_count}")
            results.append(f"• **Overbought Stocks:** {overbought_count}")
            results.append(f"• **Strong Trends (ADX>25):** {len(df_enhanced[df_enhanced['ADX_14'] > 25])}")
            
        else:
            # Use AI for complex queries
            return analyze_stocks_with_ai(query, df_enhanced)
            
        if results:
            return "\n".join(results)
        else:
            return analyze_stocks_with_ai(query, df_enhanced)
            
    except Exception as e:
        return f"❌ Analysis Error: {str(e)}"

# --- Main Analysis Function ---
def get_stock_analysis(query, df):
    """Main analysis function combining rule-based and AI approaches"""
    try:
        # First try rule-based analysis for common queries
        result = get_rule_based_analysis(query, df)
        return result
        
    except Exception as e:
        return f"❌ Error analyzing stocks: {str(e)}"

# --- Chat Interface ---
st.markdown("### 💬 Ask Your Stock Analysis Questions")

# Chat input
if prompt := st.chat_input("Ask about stocks, technical analysis, trends... (e.g., 'Show me oversold stocks with high volume')"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get AI response
    with st.spinner("🧠 Analyzing market data..."):
        response = get_stock_analysis(prompt, df)
    
    # Add AI response
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display chat
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-msg"><b>You:</b> {message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg"><b>🤖 Stock AI:</b> {message["content"]}</div>', unsafe_allow_html=True)

# --- Enhanced Quick Actions ---
if len(st.session_state.messages) == 0:
    st.markdown("#### 🚀 Quick Technical Analysis:")
    
    col1, col2, col3, col4 = st.columns(4)
    
    quick_queries = [
        ("🔥 Oversold Stocks", "oversold stocks"),
        ("⚠️ Overbought Stocks", "overbought stocks"),
        ("🚀 Bullish MACD", "bullish momentum stocks"),
        ("📊 High Volume", "high volume stocks"),
        ("📈 Strong Trends", "trending stocks with strong ADX"),
        ("🏆 Top Gainers", "top gainers today"),
        ("📉 Top Losers", "worst performing stocks"),
        ("🔥 Breakout Potential", "breakout stocks near resistance"),
        ("🛡️ Support Levels", "stocks near support levels"),
        ("📊 Market Overview", "market summary and overview"),
        ("💎 Value Picks", "undervalued stocks with good technicals"),
        ("⚡ Momentum Plays", "high momentum stocks")
    ]
    
    for i, (button_text, query) in enumerate(quick_queries):
        col = [col1, col2, col3, col4][i % 4]
        with col:
            if st.button(button_text, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": query})
                with st.spinner("🧠 Analyzing..."):
                    response = get_stock_analysis(query, df)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

# --- Technical Analysis Dashboard ---
if st.checkbox("📊 Show Technical Analysis Dashboard"):
    df_enhanced = calculate_technical_signals(df)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🔥 Oversold Opportunities")
        oversold = df_enhanced[df_enhanced['RSI_14'] < 30].nsmallest(5, 'RSI_14')
        for _, row in oversold.iterrows():
            st.markdown(f"""
            <div class="stock-card">
                <strong>{row['symbol']}</strong><br>
                Price: ₹{row['close']:.2f}<br>
                RSI: {row['RSI_14']:.1f}<br>
                Volume: {row['volume']:,}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### ⚠️ Overbought Alerts")
        overbought = df_enhanced[df_enhanced['RSI_14'] > 70].nlargest(5, 'RSI_14')
        for _, row in overbought.iterrows():
            st.markdown(f"""
            <div class="stock-card">
                <strong>{row['symbol']}</strong><br>
                Price: ₹{row['close']:.2f}<br>
                RSI: {row['RSI_14']:.1f}<br>
                Volume: {row['volume']:,}
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("#### 🚀 Strong Momentum")
        momentum = df_enhanced[df_enhanced['MACD'] > df_enhanced['MACD_Signal']].nlargest(5, 'MACD_Hist')
        for _, row in momentum.iterrows():
            st.markdown(f"""
            <div class="stock-card">
                <strong>{row['symbol']}</strong><br>
                Price: ₹{row['close']:.2f}<br>
                MACD: {row['MACD']:.2f}<br>
                Volume: {row['volume']:,}
            </div>
            """, unsafe_allow_html=True)

# --- Data Explorer ---
with st.expander("📋 Advanced Data Explorer"):
    st.markdown("**Technical Indicators Overview:**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Moving Averages:**")
        st.write("- EMA_20, EMA_44, EMA_50, EMA_100, EMA_200")
        st.write("**Bollinger Bands:**")
        st.write("- BB_Upper, BB_Middle, BB_Lower")
    
    with col2:
        st.write("**Momentum Indicators:**")
        st.write("- RSI_14, MACD, MACD_Signal, MACD_Hist")
        st.write("**Trend & Volume:**")
        st.write("- ADX_14, VWAP, Volume")
    
    if st.checkbox("Show full dataset with technical indicators"):
        st.dataframe(df, use_container_width=True)
    else:
        # Show key columns only
        key_cols = ['symbol', 'close', 'volume', 'RSI_14', 'MACD', 'ADX_14']
        available_cols = [col for col in key_cols if col in df.columns]
        st.dataframe(df[available_cols].head(20), use_container_width=True)

# --- Advanced Screener ---
with st.expander("🔧 Advanced Technical Screener"):
    st.markdown("**Create Custom Technical Screens:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rsi_min = st.slider("RSI Min", 0, 100, 0)
        rsi_max = st.slider("RSI Max", 0, 100, 100)
    
    with col2:
        volume_min = st.number_input("Min Volume", value=0)
        price_min = st.number_input("Min Price", value=0.0)
    
    with col3:
        adx_min = st.slider("ADX Min (Trend Strength)", 0, 100, 0)
        macd_positive = st.checkbox("MACD > Signal (Bullish)")
    
    if st.button("🔍 Apply Technical Screen"):
        filtered_df = df.copy()
        
        # Apply filters
        filtered_df = filtered_df[
            (filtered_df['RSI_14'] >= rsi_min) & 
            (filtered_df['RSI_14'] <= rsi_max) &
            (filtered_df['volume'] >= volume_min) &
            (filtered_df['close'] >= price_min) &
            (filtered_df['ADX_14'] >= adx_min)
        ]
        
        if macd_positive:
            filtered_df = filtered_df[filtered_df['MACD'] > filtered_df['MACD_Signal']]
        
        st.write(f"**Screened Results ({len(filtered_df)} stocks):**")
        if len(filtered_df) > 0:
            display_cols = ['symbol', 'close', 'volume', 'RSI_14', 'MACD', 'ADX_14']
            available_display_cols = [col for col in display_cols if col in filtered_df.columns]
            st.dataframe(filtered_df[available_display_cols])
        else:
            st.warning("No stocks match your criteria. Try adjusting the filters.")

# --- Sidebar Settings ---
with st.sidebar:
    st.markdown("### ⚙️ Settings & Info")
    
    st.markdown("**🤖 AI Configuration:**")
    st.success("✅ EuriAI Connected (GPT-4.1-nano)")
    
    st.markdown("**📊 Dataset Info:**")
    st.write(f"📁 File: {CSV_PATH}")
    st.write(f"📈 Stocks: {len(df)}")
    st.write(f"📋 Columns: {len(df.columns)}")
    st.write(f"📅 Data Date: {df['date'].iloc[0] if 'date' in df.columns else 'N/A'}")
    
    st.markdown("**🎯 Available Indicators:**")
    st.write("• RSI, MACD, ADX")
    st.write("• EMAs (20,44,50,100,200)")
    st.write("• Bollinger Bands")
    st.write("• VWAP, Volume Analysis")
    
    if st.button("🔄 Reload Data"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown("**💡 Example Queries:**")
    st.write("• 'Show oversold stocks'")
    st.write("• 'Find bullish MACD signals'")
    st.write("• 'High volume breakouts'")
    st.write("• 'Stocks near support'")

# --- Clear Chat ---
if len(st.session_state.messages) > 0:
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px;">
    Advanced Stock Scanner AI • Technical Analysis • Powered by EuriAI (GPT-4.1-nano) • Built with Streamlit
    <br>
    💡 <em>Use natural language to query technical indicators, trends, and market opportunities</em>
</div>
""", unsafe_allow_html=True)