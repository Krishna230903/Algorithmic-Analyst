import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# --- App Configuration ---
st.set_page_config(
    page_title="Algorithmic Analyst (Pro)",
    page_icon="🇮🇳",
    layout="wide"
)

# --- 1. SECTOR TO TICKER MAPPING (Indian Equities) ---
SECTOR_TICKER_MAP = {
    "IT & Software": [
        "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", 
        "LTIM.NS", "MPHASIS.NS", "PERSISTENT.NS", "COFORGE.NS", "TATATECH.NS"
    ],
    "FMCG": [
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", 
        "DABUR.NS", "MARICO.NS", "GODREJCP.NS", "COLPAL.NS", "PGHH.NS"
    ],
    "Automotive": [
        "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", 
        "EICHERMOT.NS", "HEROMOTOCO.NS", "TVSMOTOR.NS", "ASHOKLEY.NS", "BHARATFORG.NS"
    ],
    "Pharma": [
        "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", 
        "LUPIN.NS", "AUROPHARMA.NS", "TORNTPHARM.NS", "ALKEM.NS", "GLENMARK.NS"
    ],
    "Banking (Private)": [
        "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", 
        "INDUSINDBK.NS", "IDFCFIRSTB.NS", "YESBANK.NS", "BANDHANBNK.NS"
    ],
    "Oil & Gas": [
        "RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS", "GAIL.NS", 
        "HINDPETRO.NS", "IGL.NS", "MGL.NS"
    ],
    "Consumer Durables": [
        "TITAN.NS", "HAVells.NS", "VOLTAS.NS", "WHIRLPOOL.NS", 
        "DIXON.NS", "TTKPRESTIG.NS", "CROMPTON.NS"
    ]
}

# --- 2. Helper Functions ---
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_stock_data(tickers):
    """
    Pulls key financial data for a list of tickers from yfinance.
    Now includes Return on Assets (ROA).
    """
    data = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Check for necessary data points for our 3-factor model
            if all(k in info and info[k] is not None for k in 
                   ['enterpriseValue', 'ebitda', 'revenueGrowth', 'ebitdaMargins', 'returnOnAssets']):
                data.append({
                    "Ticker": ticker,
                    "Company Name": info.get('shortName', ticker),
                    "EV/EBITDA": info.get('enterpriseValue') / info.get('ebitda'),
                    "Revenue Growth (TTM)": info.get('revenueGrowth'),
                    "EBITDA Margin": info.get('ebitdaMargins'),
                    "ROA": info.get('returnOnAssets'), # Return on Assets
                    "P/E Ratio": info.get('trailingPE', np.nan),
                    "Enterprise Value": info.get('enterpriseValue'),
                    "EBITDA": info.get('ebitda')
                })
        except Exception as e:
            st.warning(f"Could not retrieve valid data for {ticker}.")
            
    df = pd.DataFrame(data).set_index("Ticker")
    return df

def clean_data(df, columns):
    """
    Removes NaNs and statistical outliers using the IQR method.
    """
    df_cleaned = df.dropna(subset=columns)
    
    for col in columns:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
        
    return df_cleaned

# --- 3. Sidebar Inputs ---
st.sidebar.header("Valuation Inputs")

# 1. Select Sector
sector_list = list(SECTOR_TICKER_MAP.keys())
selected_sector = st.sidebar.selectbox("1. Select a Sector", sector_list)

# 2. Get Peer List from Sector
peer_tickers = SECTOR_TICKER_MAP[selected_sector]

# 3. Select Target Company
target_ticker = st.sidebar.selectbox("2. Select Company to Value", peer_tickers)

# --- 4. Main App UI (with Tabs) ---
st.title(f"The Algorithmic Analyst 🇮🇳")
st.subheader(f"Sector-Based 'Smart Multiple' Valuation for: {selected_sector}")

tab1, tab2 = st.tabs(["Valuation Dashboard", "Methodology Explained"])

# --- TAB 1: VALUATION DASHBOARD ---
with tab1:
    if "Banking" in selected_sector:
        st.warning("⚠️ **Valuation Warning:** You've selected the Banking sector. "
                 "EV/EBITDA is **not** a standard metric for valuing banks. "
                 "This model is likely unreliable for this sector. Prefer P/B or P/E.")

    try:
        # 1. Get Live Data
        with st.spinner(f"Pulling live data for {len(peer_tickers)} companies..."):
            df_raw = get_stock_data(peer_tickers)
        
        st.subheader("Peer Group - Raw Data")
        st.dataframe(df_raw.style.format({
            "EV/EBITDA": "{:,.2f}x", "P/E Ratio": "{:,.2f}x",
            "Revenue Growth (TTM)": "{:,.2%}", "EBITDA Margin": "{:,.2%}", "ROA": "{:,.2%}",
            "Enterprise Value": "₹{:,.0f}", "EBITDA": "₹{:,.0f}",
        }))

        # 2. Clean Data & Build Model
        st.subheader("Building the Valuation Model")
        
        model_columns = ['EV/EBITDA', 'Revenue Growth (TTM)', 'EBITDA Margin', 'ROA']
        df_cleaned = clean_data(df_raw, model_columns)
        
        st.info(f"Original peers: {len(df_raw)} | Model peers: {len(df_cleaned)} "
                f"(After removing outliers & missing data)")

        if len(df_cleaned) < 5: # Need enough data for a 3-variable regression
            st.error(f"Not enough clean data ({len(df_cleaned)} companies) to build a reliable model for this sector.")
        else:
            # Define X (features) and y (target)
            X = df_cleaned[['Revenue Growth (TTM)', 'EBITDA Margin', 'ROA']]
            y = df_cleaned['EV/EBITDA']

            model = LinearRegression()
            model.fit(X, y)
            coef = model.coef_
            intercept = model.intercept_

            # Display Model Formula in a container
            with st.container(border=True):
                st.markdown("##### Regression Model Formula")
                st.markdown(f"""
                The model learns the 'fair' price for each fundamental driver:
                - **'Fair' price for +1% Revenue Growth:** `{coef[0]:.2f}x`
                - **'Fair' price for +1% EBITDA Margin:** `{coef[1]:.2f}x`
                - **'Fair' price for +1% ROA:** `{coef[2]:.2f}x`
                - **Base Multiple (Intercept):** `{intercept:.2f}x`
                """)
                st.latex(f"EV/EBITDA = {intercept:.2f} + ({coef[0]:.2f} \\times Growth) + ({coef[1]:.2f} \\times Margin) + ({coef[2]:.2f} \\times ROA)")

            # 3. Visualize the Model (3D Plot)
            st.subheader("Interactive 3D Model Visualization")
            df_cleaned['Predicted EV/EBITDA'] = model.predict(X)
            
            fig = px.scatter_3d(
                df_cleaned,
                x='Revenue Growth (TTM)',
                y='EBITDA Margin',
                z='ROA',
                color='EV/EBITDA', # Color dots by the 'true' multiple
                text='Company Name',
                title=f"{selected_sector}: Fundamental Drivers vs. EV/EBITDA",
                hover_data=['Company Name', 'EV/EBITDA', 'Predicted EV/EBITDA']
            )
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("This 3D chart shows the relationship between the 3 drivers and the EV/EBITDA multiple (color).")

            # 4. Value the Target Ticker
            st.subheader(f"Valuation for {target_ticker}")
            
            if target_ticker not in df_raw.index:
                st.error(f"Could not get valid data for {target_ticker}. It may be a new listing or data is missing from the provider.")
            elif target_ticker not in df_cleaned.index:
                st.warning(f"**Valuation Halted:** {target_ticker} was filtered out as a statistical outlier compared to its peers. The model cannot produce a reliable valuation.")
            else:
                with st.container(border=True):
                    # Get target stock's data
                    target_data = df_cleaned.loc[target_ticker]
                    target_fundamentals = target_data[['Revenue Growth (TTM)', 'EBITDA Margin', 'ROA']]
                    
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
                    col1.metric("Live EV/EBITDA", f"{live_multiple:.2f}x", 
                                help="The multiple the company is currently trading at in the market.")
                    col2.metric("'Fair' EV/EBITDA (Model)", f"{predicted_multiple:.2f}x",
                                help="The multiple the model thinks the company *should* trade at based on its fundamentals.")
                    col3.metric("Model Upside/Downside", f"{upside:,.1%}", 
                                help="The percentage difference between the 'Fair' value and the 'Live' value.")

                    st.markdown("---")
                    
                    # Final Recommendation
                    if upside > 0.15:
                        st.success(f"**Recommendation: Potentially UNDERVALUED**")
                        st.markdown(f"The model suggests {target_ticker}'s fundamentals (Growth: {target_fundamentals[0]:.2%}, Margin: {target_fundamentals[1]:.2%}, ROA: {target_fundamentals[2]:.2%}) justify a multiple of **{predicted_multiple:.2f}x**, which is higher than its current multiple of **{live_multiple:.2f}x**.")
                    elif upside < -0.15:
                        st.error(f"**Recommendation: Potentially OVERVALUED**")
                        st.markdown(f"The model suggests {target_ticker}'s fundamentals only justify a multiple of **{predicted_multiple:.2f}x**, which is lower than its 'live' multiple of **{live_multiple:.2f}x**.")
                    else:
                        st.info(f"**Recommendation: FAIRLY VALUED**")
                        st.markdown(f"The model's 'fair' multiple of **{predicted_multiple:.2f}x** is in line with its current 'live' multiple of **{live_multiple:.2f}x**.")

    except Exception as e:
        st.error(f"An error occurred. This could be due to a temporary issue with the Yahoo Finance API or the selected data.")
        st.exception(e)

# --- TAB 2: METHODOLOGY EXPLAINED ---
with tab2:
    st.header("How This Works: The 'Algorithmic Analyst'")
    st.markdown("""
    This tool performs a **quantitative relative valuation** to determine if a stock is 
    overvalued or undervalued compared to its peers. It's designed to be 'smarter' 
    than a simple "industry average" valuation.
    """)

    with st.expander("Step 1: The Problem with 'Simple Average' Valuation"):
        st.markdown("""
        A common valuation method is to take the average EV/EBITDA multiple of an industry (e.g., "the IT sector trades at 25x") and apply it to your company.
        
        **This is a flawed, 'dumb' approach.**
        
        Why? Because not all companies are equal. A company with **50% revenue growth** and **30% margins** *deserves* to trade at a higher multiple than a company with **5% growth** and **10% margins**. A simple average ignores this.
        """)

    with st.expander("Step 2: The 'Smart Multiple' (Multiple Linear Regression)"):
        st.markdown("""
        This tool fixes that problem. It uses a **Multiple Linear Regression** model to find the *statistical relationship* between a company's fundamentals and its valuation multiple.
        
        We build a model based on this formula:
        """)
        st.latex("Fair\\ Multiple = \\beta_0 + (\\beta_1 \\times Growth) + (\\beta_2 \\times Margin) + (\\beta_3 \\times ROA)")
        st.markdown("""
        - **Fair Multiple (EV/EBITDA):** The target we are trying to predict.
        - **Growth (Revenue Growth TTM):** How fast the company is growing.
        - **Margin (EBITDA Margin):** How profitable the company is.
        - **ROA (Return on Assets):** How efficiently the company uses its assets to generate profit.
        - **$\beta_0, \beta_1, \beta_2, \beta_3$:** These are the coefficients (prices) the model *learns* from the peer data. For example, it might learn that "for every 1% of growth, the multiple increases by 0.5x."
        """)

    with st.expander("Step 3: The Calculation (Putting it all together)"):
        st.markdown("""
        Here is the step-by-step process:
        
        1.  **Fetch Data:** You select a sector. The app pulls live financial data (EV/EBITDA, Growth, Margin, ROA) for all companies in that sector.
        2.  **Clean Data:** It removes any statistical outliers (e.g., a company with a 500x multiple) that would skew the model.
        3.  **Train Model:** It trains the regression model on this cleaned peer data to find the "fair" pricing for Growth, Margin, and ROA.
        4.  **Get Target Fundamentals:** It takes the company you want to value (e.g., `TCS.NS`) and gets its *specific* fundamentals (e.g., 10% Growth, 25% Margin, 18% ROA).
        5.  **Predict 'Fair' Multiple:** It plugs these fundamentals into the model's formula to get a 'Fair' multiple.
        6.  **Compare:** It compares this **'Fair' Multiple (Model)** to the **'Live' Multiple (Market)** to generate the "Upside/Downside" and the final recommendation.
        """)
        
    with st.expander("Disclaimer"):
        st.warning("""
        **This is not financial advice.** This tool is for educational and illustrative purposes only. 
        Valuations are complex and depend on many factors. The model is highly sensitive to the 
        peer group, the data provider (Yahoo Finance), and market conditions. 
        Always do your own research.
        """)
