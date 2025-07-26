# import streamlit as st
# import pandas as pd
# import requests
# import os
# from euriai import EuriaiClient

# euron_key = 'euri-9abeec33995ca4541991ffc24a66a27928da68feeb7ecc83b0c26a7a905d9643'


# client = EuriaiClient(
#     api_key=euron_key,
#     model="gpt-4.1-nano"
# )

# response = client.generate_completion(
#     prompt="Write a short poem about artificial intelligence.",
#     temperature=0.7,
#     max_tokens=300
# )

# print(response)

# # --- Configuration ---
# CSV_PATH = r'stock_data_summary_20250726_085144.csv'

# API_KEY = "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"

# # --- API Setup ---
# url = "https://api.perplexity.ai/chat/completions"
# headers = {
#     "Authorization": f"Bearer {API_KEY}",
#     "Content-Type": "application/json"
# }
import streamlit as st
import pandas as pd
import requests
import os
import json
from euriai import EuriaiClient

# --- Configuration ---
CSV_PATH = r'stock_data_summary_20250726_085144.csv'

# EuriAI Configuration
euron_key = 'euri-9abeec33995ca4541991ffc24a66a27928da68feeb7ecc83b0c26a7a905d9643'

client = EuriaiClient(
    api_key=euron_key,
    model="gpt-4.1-nano"
)

# --- Page Config ---
st.set_page_config(
    page_title="Stock Scanner AI",
    layout="centered",
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
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="header">
    <h1>📈 Stock Scanner AI</h1>
    <p>Smart Stock Analysis • Real-time Filtering • AI-Powered</p>
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
    st.error("❌ CSV file not found. Please ensure 'stock_data_summary_20250726_085144.csv' exists in the current directory.")
    st.stop()

# --- Data Info Dashboard ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📊 Total Stocks", len(df))
with col2:
    st.metric("📋 Data Columns", len(df.columns))
with col3:
    if 'Close' in df.columns:
        st.metric("💰 Avg Close Price", f"${df['Close'].mean():.2f}")
    elif 'Price' in df.columns:
        st.metric("💰 Avg Price", f"${df['Price'].mean():.2f}")
    else:
        st.metric("📊 Data", "Loaded")
with col4:
    if 'Volume' in df.columns:
        st.metric("📈 Avg Volume", f"{df['Volume'].mean():,.0f}")
    else:
        st.metric("🔍 Status", "Ready")

# --- Initialize Chat ---
if 'messages' not in st.session_state:
    st.session_state.messages = []

# --- Direct Stock Filter Function ---
def get_direct_stock_list(query, df):
    """Direct data processing for clean stock lists - handles duplicates by keeping most recent entry"""
    try:
        # Find stock identifier column
        stock_col = None
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['symbol', 'stock', 'name', 'ticker', 'company']):
                stock_col = col
                break
        if not stock_col:
            stock_col = df.columns[0]
        
        # Find date column for filtering duplicates
        date_col = None
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp', 'day']):
                date_col = col
                break
        
        # Remove duplicates by keeping most recent entry for each stock
        if date_col:
            # Convert date column to datetime if it's not already
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
            # Sort by date and keep last (most recent) entry for each stock
            df_unique = df_copy.sort_values(date_col).drop_duplicates(subset=[stock_col], keep='last')
        else:
            # If no date column, just remove duplicates by stock symbol (keep last occurrence)
            df_unique = df.drop_duplicates(subset=[stock_col], keep='last')
        
        # Find numeric columns for filtering
        numeric_cols = df_unique.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        query_lower = query.lower()
        result_stocks = []
        
        # Handle common queries directly
        if 'top' in query_lower and ('perform' in query_lower or 'gain' in query_lower):
            # Find price or close column
            price_col = None
            for col in numeric_cols:
                if any(word in col.lower() for word in ['close', 'price', 'last']):
                    price_col = col
                    break
            
            if price_col:
                top_stocks = df_unique.nlargest(10, price_col)
                for i, (_, row) in enumerate(top_stocks.iterrows(), 1):
                    result_stocks.append(f"{i}. {row[stock_col]} - ${row[price_col]:.2f}")
        
        elif 'volume' in query_lower and 'high' in query_lower:
            # Find volume column
            volume_col = None
            for col in numeric_cols:
                if 'volume' in col.lower():
                    volume_col = col
                    break
            
            if volume_col:
                high_vol_stocks = df_unique.nlargest(10, volume_col)
                for i, (_, row) in enumerate(high_vol_stocks.iterrows(), 1):
                    result_stocks.append(f"{i}. {row[stock_col]} - Volume: {row[volume_col]:,.0f}")
        
        elif 'low' in query_lower and ('price' in query_lower or 'cheap' in query_lower):
            # Find lowest priced stocks
            price_col = None
            for col in numeric_cols:
                if any(word in col.lower() for word in ['close', 'price', 'last']):
                    price_col = col
                    break
            
            if price_col:
                low_price_stocks = df_unique.nsmallest(10, price_col)
                for i, (_, row) in enumerate(low_price_stocks.iterrows(), 1):
                    result_stocks.append(f"{i}. {row[stock_col]} - ${row[price_col]:.2f}")
        
        else:
            # Return random sample of unique stocks
            sample_stocks = df_unique.sample(min(10, len(df_unique)))
            for i, (_, row) in enumerate(sample_stocks.iterrows(), 1):
                result_stocks.append(f"{i}. {row[stock_col]}")
        
        if result_stocks:
            unique_count = len(df_unique)
            total_count = len(df)
            return f"📊 **Stock List** (Showing unique stocks: {unique_count} out of {total_count} total records):\n\n" + "\n".join(result_stocks)
        else:
            return "❌ No matching stocks found for your query."
            
    except Exception as e:
        return f"❌ Error processing stocks: {str(e)}"

# --- Enhanced AI Function ---
def get_stock_analysis(query, df):
    try:
        # First remove duplicates to get unique stocks
        stock_identifier = None
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['symbol', 'stock', 'name', 'ticker', 'company']):
                stock_identifier = col
                break
        if not stock_identifier:
            stock_identifier = df.columns[0]
        
        # Find date column for filtering duplicates
        date_col = None
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp', 'day']):
                date_col = col
                break
        
        # Remove duplicates by keeping most recent entry for each stock
        if date_col:
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
            df_unique = df_copy.sort_values(date_col).drop_duplicates(subset=[stock_identifier], keep='last')
        else:
            df_unique = df.drop_duplicates(subset=[stock_identifier], keep='last')
        
        # First try direct processing for common queries
        direct_result = get_direct_stock_list(query, df)
        if "Error" not in direct_result and len(direct_result.split('\n')) > 2:
            return direct_result
        
        # Get unique stock symbols/names for AI context
        all_stocks = df_unique[stock_identifier].astype(str).tolist()
        
        # Create simple prompt for stock list
        prompt = f"""Based on the stock data, provide ONLY a clean numbered list of stocks for this query.

Available Unique Stocks: {', '.join(all_stocks[:30])}
Dataset Columns: {', '.join(df_unique.columns)}
Total Unique Stocks: {len(df_unique)}

Query: "{query}"

Return format:
1. STOCK1
2. STOCK2
3. STOCK3
(etc, max 10 unique stocks)

Stock List:"""

        # Use EuriAI client
        response = client.generate_completion(
            prompt=prompt,
            temperature=0.1,
            max_tokens=200
        )
        
        # Extract content from response if it's a dict/JSON
        if isinstance(response, dict):
            if 'choices' in response and len(response['choices']) > 0:
                content = response['choices'][0].get('message', {}).get('content', '')
                return content.strip()
            elif 'content' in response:
                return response['content'].strip()
            else:
                return str(response)
        else:
            return str(response).strip()
        
    except Exception as e:
        return f"❌ Error analyzing stocks: {str(e)}\n\nPlease check your EuriAI connection."

# Alternative function using Groq API (uncomment to use)
# def get_stock_analysis_groq(query, df):
#     """Alternative using Groq API - faster and cheaper"""
#     try:
#         # Same data preparation as above
#         stock_identifier = None
#         for col in df.columns:
#             if any(keyword in col.lower() for keyword in ['symbol', 'stock', 'name', 'ticker']):
#                 stock_identifier = col
#                 break
#         if not stock_identifier:
#             stock_identifier = df.columns[0]
        
#         prompt = f"""Analyze stock data and answer: "{query}"
        
# Available stocks: {', '.join(df[stock_identifier].astype(str).tolist()[:20])}
# Dataset columns: {list(df.columns)}
# Total stocks: {len(df)}

# Provide a clear, numbered list of relevant stocks based on the query."""

#         headers = {
#             "Authorization": f"Bearer {GROQ_API_KEY}",
#             "Content-Type": "application/json"
#         }
        
#         payload = {
#             "model": "mixtral-8x7b-32768",  # or "llama2-70b-4096"
#             "messages": [{"role": "user", "content": prompt}],
#             "temperature": 0.3,
#             "max_tokens": 500
#         }
        
#         response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
        
#         if response.status_code == 200:
#             return response.json()["choices"][0]["message"]["content"]
#         else:
#             return f"❌ API Error: {response.status_code}"
            
#     except Exception as e:
#         return f"❌ Error: {str(e)}"

# --- Chat Interface ---
st.markdown("### 💬 Ask about your stocks")

# Chat input
if prompt := st.chat_input("Ask me about stocks... (e.g., 'Show me top 10 stocks by volume')"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get AI response
    with st.spinner("🧠 Analyzing stocks..."):
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
    st.markdown("#### 🚀 Quick Analysis Options:")
    
    col1, col2, col3 = st.columns(3)
    
    quick_queries = [
        ("📈 Top Performers", "Show me the top 10 best performing stocks"),
        ("📊 High Volume", "List stocks with highest trading volume"),
        ("💰 Price Analysis", "Show me stocks with prices above average"),
        ("🔍 Market Overview", "Give me an overview of the stock market data"),
        ("📉 Low Price Stocks", "Show me stocks under $50"),
        ("⚡ Most Active", "List the most actively traded stocks")
    ]
    
    for i, (button_text, query) in enumerate(quick_queries):
        col = [col1, col2, col3][i % 3]
        with col:
            if st.button(button_text, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": query})
                with st.spinner("🧠 Analyzing..."):
                    response = get_stock_analysis(query, df)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

# --- Data Explorer ---
with st.expander("📋 Explore Your Stock Data"):
    st.markdown("**Dataset Overview:**")
    st.write(f"- **Total Records:** {len(df)}")
    st.write(f"- **Columns:** {', '.join(df.columns)}")
    
    if st.checkbox("Show full dataset"):
        st.dataframe(df, use_container_width=True)
    else:
        st.dataframe(df.head(10), use_container_width=True)
    
    # Basic statistics
    if st.checkbox("Show statistics"):
        st.write("**Numerical Columns Statistics:**")
        st.dataframe(df.describe())

# --- Advanced Filters ---
with st.expander("🔧 Advanced Filtering"):
    st.markdown("**Create Custom Filters:**")
    
    # Select column for filtering
    filter_col = st.selectbox("Select column to filter by:", df.columns)
    
    if df[filter_col].dtype in ['int64', 'float64']:
        min_val = st.number_input(f"Minimum {filter_col}:", value=float(df[filter_col].min()))
        max_val = st.number_input(f"Maximum {filter_col}:", value=float(df[filter_col].max()))
        
        if st.button("Apply Filter"):
            filtered_df = df[(df[filter_col] >= min_val) & (df[filter_col] <= max_val)]
            st.write(f"**Filtered Results ({len(filtered_df)} stocks):**")
            st.dataframe(filtered_df)
    else:
        unique_values = df[filter_col].unique()
        selected_values = st.multiselect(f"Select {filter_col} values:", unique_values)
        
        if st.button("Apply Filter") and selected_values:
            filtered_df = df[df[filter_col].isin(selected_values)]
            st.write(f"**Filtered Results ({len(filtered_df)} stocks):**")
            st.dataframe(filtered_df)

# --- Settings ---
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    st.markdown("**API Configuration:**")
    st.success("✅ EuriAI Connected (GPT-4.1-nano)")
    
    st.markdown("**Data Info:**")
    st.write(f"📁 File: {CSV_PATH}")
    st.write(f"📊 Stocks: {len(df)}")
    st.write(f"📋 Columns: {len(df.columns)}")
    
    if st.button("🔄 Reload Data"):
        st.cache_data.clear()
        st.rerun()

# --- Clear Chat ---
if len(st.session_state.messages) > 0:
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px;">
    Stock Scanner AI • Powered by EuriAI (GPT-4.1-nano) • Built with Streamlit
</div>
""", unsafe_allow_html=True)