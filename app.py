import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# --- App Configuration ---
st.set_page_config(
    page_title="The Algorithmic Analyst (India)",
    page_icon="🇮🇳",
    layout="wide"
)

# --- 1. SECTOR TO TICKER MAPPING (Indian Equities) ---
# This is our "database" of companies.
# All tickers must end with .NS for Yahoo Finance (National Stock Exchange)
SECTOR_TICKER_MAP = {
    "IT & Software": [
        "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", 
        "LTIM.NS", "MPHASIS.NS", "PERSISTENT.NS", "COFORGE.NS"
    ],
    "FMCG": [
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", 
        "DABUR.NS", "MARICO.NS", "GODREJCP.NS", "COLPAL.NS"
    ],
    "Automotive": [
        "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", 
        "EICHERMOT.NS", "HEROMOTOCO.NS", "TVSMOTOR.NS", "ASHOKLEY.NS"
    ],
    "Pharma": [
        "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", 
        "LUPIN.NS", "AUROPHARMA.NS", "TORNTPHARM.NS"
    ],
    "Banking (Private)": [
        "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", 
        "INDUSINDBK.NS", "IDFCFIRSTB.NS"
    ],
    "Oil & Gas": [
        "RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS", "GAIL.NS", "HINDPETRO.NS"
    ]
}

# --- 2. Helper Functions (Cached) ---
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_stock_data(tickers):
    """
    Pulls key financial data for a list of tickers from yfinance.
    """
    data = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Check for necessary data points
            if all(k in info for k in ['enterpriseValue', 'ebitda', 'revenueGrowth', 'ebitdaMargins']):
                data.append({
                    "Ticker": ticker,
                    "Company Name": info.get('shortName', ticker),
                    "EV/EBITDA": info.get('enterpriseValue', np.nan) / info.get('ebitda', np.nan),
                    "Revenue Growth (TTM)": info.get('revenueGrowth', np.nan),
                    "EBITDA Margin": info.get('ebitdaMargins', np.nan),
                    "Enterprise Value": info.get('enterpriseValue', np.nan),
                    "EBITDA": info.get('ebitda', np.nan)
                })
        except Exception as e:
            st.warning(f"Could not retrieve valid data for {ticker}.")
            
    df = pd.DataFrame(data).set_index("Ticker")
    # Clean data: remove NaNs and outliers for a stable model
    df = df.dropna()
    # Filter out extreme/non-sensical outliers
    df = df[(df['EV/EBITDA'] < 100) & (df['EV/EBITDA'] > 0)] 
    df = df[(df['Revenue Growth (TTM)'] < 2) & (df['EBITDA Margin'] < 1)] # Growth < 200%, Margin < 100%
    return df

# --- 3. Sidebar Inputs ---
st.sidebar.header("Valuation Inputs")

# 1. Select Sector
sector_list = list(SECTOR_TICKER_MAP.keys())
selected_sector = st.sidebar.selectbox("1. Select a Sector", sector_list)

# 2. Get Peer List from Sector
peer_tickers = SECTOR_TICKER_MAP[selected_sector]

# 3. Select Target Company (dynamically populated from the sector list)
target_ticker = st.sidebar.selectbox("2. Select Company to Value", peer_tickers)

# --- 4. Main App ---
st.title(f"The Algorithmic Analyst 🇮🇳")
st.subheader(f"Sector-Based 'Smart Multiple' Valuation for: {selected_sector}")

# --- Special Warning for Banking Sector ---
if "Banking" in selected_sector:
    st.warning("⚠️ **Valuation Warning:** You've selected the Banking sector. "
             "EV/EBITDA is **not** a standard metric for valuing banks due to their "
             "unique capital structure and business model. "
             "Metrics like P/B (Price-to-Book) or P/E are preferred. "
             "The model below may produce unreliable results.")

try:
    # 1. Get Live Data
    with st.spinner(f"Pulling live data for {len(peer_tickers)} companies in the {selected_sector} sector..."):
        df = get_stock_data(peer_tickers)

    if df.empty or len(df) < 3: # Need at least 3 data points to build a model
        st.error(f"Could not retrieve enough valid data to build a model for the {selected_sector} sector. Try another sector.")
    else:
        st.subheader("Live Peer Group Financials")
        st.dataframe(df.style.format({
            "EV/EBITDA": "{:,.2f}x",
            "Revenue Growth (TTM)": "{:,.2%}",
            "EBITDA Margin": "{:,.2%}",
            "Enterprise Value": "₹{:,.0f}",
            "EBITDA": "₹{:,.0f}",
        }))

        # 2. Build the "Smart Multiple" Model
        st.subheader("Building the Valuation Model")
        
        # Define X and y variables for the regression
        X = df[['Revenue Growth (TTM)', 'EBITDA Margin']]
        y = df['EV/EBITDA']

        # Create and fit the model
        model = LinearRegression()
        model.fit(X, y)

        # Show model coefficients
        coef = model.coef_
        intercept = model.intercept_
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            The model learns the 'fair' price for 1 unit of growth or margin within this sector:
            - **'Fair' price for +1% Revenue Growth:** `{coef[0]:.2f}x`
            - **'Fair' price for +1% EBITDA Margin:** `{coef[1]:.2f}x`
            - **Base Multiple (Intercept):** `{intercept:.2f}x`
            """)
        with col2:
            # Display the regression formula in LaTeX
            st.latex(f"EV/EBITDA = {intercept:.2f} + ({coef[0]:.2f} \\times Growth) + ({coef[1]:.2f} \\times Margin)")
            
        # 3. Visualize the Model
        df['Predicted EV/EBITDA'] = model.predict(X)
        
        fig = px.scatter(
            df, 
            x="Revenue Growth (TTM)", 
            y="EV/EBITDA",
            text=df.index,
            title=f"{selected_sector}: Revenue Growth vs. EV/EBITDA",
            labels={"Revenue Growth (TTM)": "Revenue Growth (TTM)", "EV/EBITDA": "Live EV/EBITDA Multiple"},
            hover_data=['Company Name', 'EBITDA Margin']
        )
        # Add the regression "plane" as a line (simplified)
        fig.add_scatter(
            x=df["Revenue Growth (TTM)"],
            y=df['Predicted EV/EBITDA'],
            mode='lines',
            name='"Fair Value" Line'
        )
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("Companies *above* the line are 'expensive' (trading higher than their fundamentals suggest). Companies *below* are 'cheap'.")

        # 4. Value the Target Ticker
        st.subheader(f"Valuation for {target_ticker}")
        
        if target_ticker not in df.index:
            st.error(f"Could not get valid data for {target_ticker}. It may have been filtered out during data cleaning (e.g., missing data or extreme outlier).")
        else:
            # Get target stock's data
            target_data = df.loc[target_ticker]
            target_fundamentals = target_data[['Revenue Growth (TTM)', 'EBITDA Margin']]
            
            # Predict "Fair" Multiple
            predicted_multiple = model.predict([target_fundamentals])[0]
            live_multiple = target_data['EV/EBITDA']
            
            # Calculate Fair Value
            fair_enterprise_value = predicted_multiple * target_data['EBITDA']
            live_enterprise_value = target_data['Enterprise Value']
            
            # Calculate Upside/Downside
            upside = (fair_enterprise_value / live_enterprise_value) - 1

            # Display results
            col1, col2, col3 = st.columns(3)
            col1.metric("Live EV/EBITDA", f"{live_multiple:.2f}x")
            col2.metric("'Fair' EV/EBITDA (Model)", f"{predicted_multiple:.2f}x")
            col3.metric("Model Upside/Downside", f"{upside:,.1%}")

            st.markdown("---")
            col1, col2 = st.columns(2)
            col1.metric("Live Enterprise Value", f"₹{live_enterprise_value:,.0f}")
            col2.metric("'Fair' Enterprise Value (Model)", f"₹{fair_enterprise_value:,.0f}")
            
            # Final Recommendation
            if upside > 0.15:
                st.success(f"**Recommendation: Potentially UNDERVALUED**")
                st.markdown(f"The model suggests {target_ticker}'s fundamentals (growth of {target_fundamentals[0]:.2%} and margin of {target_fundamentals[1]:.2%}) justify a multiple of **{predicted_multiple:.2f}x**, which is higher than its current trading multiple of **{live_multiple:.2f}x**.")
            elif upside < -0.15:
                st.error(f"**Recommendation: Potentially OVERVALUED**")
                st.markdown(f"The model suggests {target_ticker}'s fundamentals only justify a multiple of **{predicted_multiple:.2f}x**, which is lower than its 'live' multiple of **{live_multiple:.2f}x**.")
            else:
                st.info(f"**Recommendation: FAIRLY VALUED**")
                st.markdown(f"The model's 'fair' multiple of **{predicted_multiple:.2f}x** is in line with its current 'live' multiple of **{live_multiple:.2f}x**.")

except Exception as e:
    st.error(f"An error occurred. This could be due to a temporary issue with the Yahoo Finance API or the selected data.")
    st.exception(e)
