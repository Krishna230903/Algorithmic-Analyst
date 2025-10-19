import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

# --- App Configuration ---
st.set_page_config(
    page_title="FMVA Valuation Dashboard",
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
    ]
}

# --- 2. Session State Initialization ---
if 'valuation_results' not in st.session_state:
    st.session_state.valuation_results = {
        "Comps (EV/EBITDA)": None,
        "Comps (P/E)": None,
        "DCF": None,
        "Current Price": None
    }

# --- 3. Helper Functions ---
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_stock_data(tickers):
    """
    Pulls a comprehensive set of financial data for a list of tickers.
    """
    data = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # --- Data for Comps ---
            ev_ebitda = info.get('enterpriseValue', np.nan) / info.get('ebitda', np.nan)
            pe_ratio = info.get('trailingPE', np.nan)
            rev_growth = info.get('revenueGrowth', np.nan)
            ebitda_margin = info.get('ebitdaMargins', np.nan)
            roe = info.get('returnOnEquity', np.nan)
            earnings_growth = info.get('earningsGrowth', np.nan)
            
            # --- Data for DCF ---
            op_cashflow = info.get('operatingCashflow', np.nan)
            cap_ex = info.get('capitalExpenditure', np.nan)
            fcf = (op_cashflow or 0) + (cap_ex or 0) # CapEx is negative, so we add
            
            data.append({
                "Ticker": ticker,
                "Company Name": info.get('shortName', ticker),
                "Current Price": info.get('currentPrice', np.nan),
                "EV/EBITDA": ev_ebitda,
                "P/E Ratio": pe_ratio,
                "Revenue Growth": rev_growth,
                "EBITDA Margin": ebitda_margin,
                "ROE": roe,
                "Earnings Growth": earnings_growth,
                "FCF (TTM)": fcf,
                "Enterprise Value": info.get('enterpriseValue', np.nan),
                "Market Cap": info.get('marketCap', np.nan),
                "Shares Outstanding": info.get('sharesOutstanding', np.nan)
            })
        except Exception as e:
            pass # Silently skip tickers with errors
            
    df = pd.DataFrame(data).set_index("Ticker")
    return df

def clean_data(df, columns):
    """Removes NaNs and statistical outliers using the IQR method."""
    df_cleaned = df.dropna(subset=columns)
    
    for col in columns:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
        
    return df_cleaned

def run_regression_model(df, target_y, features_x):
    """Runs a linear regression and returns the model and cleaned df."""
    model_cols = [target_y] + features_x
    df_cleaned = clean_data(df, model_cols)
    
    if len(df_cleaned) < len(features_x) + 2:
        return None, None # Not enough data
        
    X = df_cleaned[features_x]
    y = df_cleaned[target_y]
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model, df_cleaned

def create_football_field(results, target_ticker):
    """Creates a Plotly Football Field chart."""
    data = []
    
    # Add valuation methods
    for method, value in results.items():
        if value is not None and method != "Current Price":
            data.append(dict(Task=method, Start=value, Finish=value, Label=f"₹{value:.2f}"))
    
    if not data:
        st.info("Run valuations in other tabs to populate this chart.")
        return
        
    # Sort for chart
    df = pd.DataFrame(data).sort_values(by="Start")
    
    # Create the horizontal bar chart
    fig = px.bar(df, x_start="Start", x_end="Finish", y="Task", text="Label",
                 title=f"Valuation Summary for {target_ticker}")
    
    fig.update_traces(textposition='outside')
    
    # Add current price as a vertical line
    if results["Current Price"] is not None:
        fig.add_vline(x=results["Current Price"], 
                      line_width=3, line_dash="dash", line_color="red",
                      annotation_text=f"Current Price: ₹{results['Current Price']:.2f}",
                      annotation_position="bottom right")

    fig.update_layout(
        yaxis_title=None,
        xaxis_title="Implied Share Price (₹)",
        plot_bgcolor='white',
        xaxis_showgrid=True,
        yaxis_showgrid=False
    )
    st.plotly_chart(fig, use_container_width=True)


# --- 4. Sidebar Inputs ---
st.sidebar.header("Valuation Inputs")
sector_list = list(SECTOR_TICKER_MAP.keys())
selected_sector = st.sidebar.selectbox("1. Select a Sector", sector_list)
peer_tickers = SECTOR_TICKER_MAP[selected_sector]
target_ticker = st.sidebar.selectbox("2. Select Company to Value", peer_tickers)

# --- 5. Main App UI (with Tabs) ---
st.title(f"Multi-Method Valuation Dashboard 🇮🇳")
st.subheader(f"Target: {target_ticker} | Sector: {selected_sector}")

# --- Fetch Data Once ---
try:
    with st.spinner(f"Pulling live data for {len(peer_tickers)} companies..."):
        df_raw = get_stock_data(peer_tickers)
        
    if target_ticker not in df_raw.index:
        st.error(f"Could not fetch data for {target_ticker}. Please try another company.")
        st.stop()
        
    target_info = df_raw.loc[target_ticker]
    st.session_state.valuation_results["Current Price"] = target_info["Current Price"]
    
except Exception as e:
    st.error(f"An error occurred fetching data: {e}")
    st.stop()


# --- Define Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Valuation Summary", 
    "📈 Comparable Analysis", 
    "💸 Intrinsic (DCF) Analysis", 
    "Methodology"
])

# --- TAB 1: VALUATION SUMMARY ---
with tab1:
    st.header("Valuation 'Football Field' Summary")
    st.markdown("""
    This chart summarizes the valuation ranges from all methods. 
    Run the models in the other tabs to populate it.
    """)
    
    create_football_field(st.session_state.valuation_results, target_ticker)
    
    # Display results in text
    with st.container(border=True):
        st.subheader("Summary of Results")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"₹{st.session_state.valuation_results['Current Price']:.2f}")
        col2.metric("Comps (EV/EBITDA)", f"₹{st.session_state.valuation_results['Comps (EV/EBITDA)']:.2f}" if st.session_state.valuation_results['Comps (EV/EBITDA)'] else "N/A")
        col3.metric("Comps (P/E)", f"₹{st.session_state.valuation_results['Comps (P/E)']:.2f}" if st.session_state.valuation_results['Comps (P/E)'] else "N/A")
        col4.metric("DCF", f"₹{st.session_state.valuation_results['DCF']:.2f}" if st.session_state.valuation_results['DCF'] else "N/A")

# --- TAB 2: COMPARABLE ANALYSIS ---
with tab2:
    st.header("Comparable Company Analysis (Comps)")
    st.markdown("This uses quantitative models to find a 'fair' multiple based on fundamentals.")

    if "Banking" in selected_sector:
        st.warning("⚠️ **Warning:** Standard Comps (EV/EBITDA, P/E) are not reliable for banks. P/B is preferred.")
    
    # --- Comps Model 1: EV/EBITDA ---
    with st.container(border=True):
        st.subheader("Model 1: 'Smart' EV/EBITDA")
        features_ev = ['Revenue Growth', 'EBITDA Margin']
        model_ev, df_cleaned_ev = run_regression_model(df_raw, 'EV/EBITDA', features_ev)
        
        if model_ev is None:
            st.error("Not enough clean data to build the EV/EBITDA model.")
        else:
            # Get target fundamentals
            target_fundamentals_ev = target_info[features_ev]
            if target_fundamentals_ev.isnull().any():
                st.warning(f"{target_ticker} is missing data for this model.")
            else:
                # Predict 'Fair' Multiple
                fair_multiple_ev = model_ev.predict([target_fundamentals_ev])[0]
                
                # Calculate 'Fair' Enterprise Value
                fair_ev = fair_multiple_ev * target_info['EBITDA']
                
                # Calculate Fair Equity Value & Price
                net_debt = target_info['Enterprise Value'] - target_info['Market Cap']
                fair_equity_value_ev = fair_ev - net_debt
                fair_price_ev = fair_equity_value_ev / target_info['Shares Outstanding']
                
                # Store result
                st.session_state.valuation_results["Comps (EV/EBITDA)"] = fair_price_ev
                
                # Display
                col1, col2, col3 = st.columns(3)
                col1.metric("Live EV/EBITDA", f"{target_info['EV/EBITDA']:.2f}x")
                col2.metric("'Fair' EV/EBITDA", f"{fair_multiple_ev:.2f}x")
                col3.metric("Implied Fair Price", f"₹{fair_price_ev:.2f}")

    # --- Comps Model 2: P/E Ratio ---
    with st.container(border=True):
        st.subheader("Model 2: 'Smart' P/E Ratio")
        features_pe = ['Earnings Growth', 'ROE']
        model_pe, df_cleaned_pe = run_regression_model(df_raw, 'P/E Ratio', features_pe)
        
        if model_pe is None:
            st.error("Not enough clean data to build the P/E model.")
        else:
            # Get target fundamentals
            target_fundamentals_pe = target_info[features_pe]
            if target_fundamentals_pe.isnull().any():
                st.warning(f"{target_ticker} is missing data for this model.")
            else:
                # Predict 'Fair' Multiple
                fair_multiple_pe = model_pe.predict([target_fundamentals_pe])[0]
                
                # Calculate 'Fair' Price
                # P/E = Price / EPS. So, Price = P/E * EPS
                # We can use (Fair P/E) / (Live P/E) * (Live Price)
                fair_price_pe = (fair_multiple_pe / target_info['P/E Ratio']) * target_info['Current Price']
                
                # Store result
                st.session_state.valuation_results["Comps (P/E)"] = fair_price_pe
                
                # Display
                col1, col2, col3 = st.columns(3)
                col1.metric("Live P/E", f"{target_info['P/E Ratio']:.2f}x")
                col2.metric("'Fair' P/E", f"{fair_multiple_pe:.2f}x")
                col3.metric("Implied Fair Price", f"₹{fair_price_pe:.2f}")

# --- TAB 3: INTRINSIC (DCF) ANALYSIS ---
with tab3:
    st.header("Intrinsic Valuation (Discounted Cash Flow)")
    
    with st.container(border=True):
        st.subheader("DCF Assumptions")
        
        # Get TTM FCF
        ttm_fcf = target_info['FCF (TTM)']
        if pd.isna(ttm_fcf) or ttm_fcf == 0:
            st.error("Cannot run DCF: TTM Free Cash Flow data is missing or zero.")
        else:
            st.metric("Last 12 Months Free Cash Flow (TTM)", f"₹{ttm_fcf:,.0f}")
            
            # DCF Assumption Sliders
            col1, col2, col3 = st.columns(3)
            with col1:
                g_short = st.slider("Short-Term Growth (Yrs 1-5)", 0.0, 0.25, 0.10, 0.01)
            with col2:
                wacc = st.slider("Discount Rate (WACC)", 0.05, 0.15, 0.10, 0.005)
            with col3:
                g_long = st.slider("Terminal Growth Rate", 0.0, 0.05, 0.03, 0.005)

            # Run DCF Calculation
            if wacc <= g_long:
                st.error("WACC must be greater than Terminal Growth Rate.")
            else:
                # 1. Project FCF for 5 years
                fcf_forecasts = []
                for year in range(1, 6):
                    fcf = ttm_fcf * (1 + g_short)**year
                    fcf_forecasts.append(fcf)
                
                # 2. Calculate Terminal Value
                fcf_yr5 = fcf_forecasts[-1]
                terminal_value = (fcf_yr5 * (1 + g_long)) / (wacc - g_long)
                
                # 3. Discount all cash flows
                discounted_fcf = []
                for year, fcf in enumerate(fcf_forecasts, 1):
                    dfcf = fcf / (1 + wacc)**year
                    discounted_fcf.append(dfcf)
                
                d_terminal_value = terminal_value / (1 + wacc)**5
                
                # 4. Calculate Enterprise Value
                dcf_enterprise_value = sum(discounted_fcf) + d_terminal_value
                
                # 5. Calculate Equity Value & Price
                net_debt = target_info['Enterprise Value'] - target_info['Market Cap']
                dcf_equity_value = dcf_enterprise_value - net_debt
                dcf_price_per_share = dcf_equity_value / target_info['Shares Outstanding']
                
                # Store result
                st.session_state.valuation_results["DCF"] = dcf_price_per_share
                
                # Display DCF results
                st.subheader("DCF Valuation Results")
                col1, col2, col3 = st.columns(3)
                col1.metric("Implied Enterprise Value", f"₹{dcf_enterprise_value:,.0f}")
                col2.metric("Implied Equity Value", f"₹{dcf_equity_value:,.0f}")
                col3.metric("Implied Fair Price per Share", f"₹{dcf_price_per_share:,.2f}")
                
                # Show forecast table
                with st.expander("View 5-Year FCF Forecast"):
                    forecast_df = pd.DataFrame({
                        'Year': [1, 2, 3, 4, 5],
                        'Projected FCF': fcf_forecasts,
                        'Discounted FCF': discounted_fcf
                    })
                    forecast_df.loc['Terminal'] = ['Terminal', terminal_value, d_terminal_value]
                    st.dataframe(forecast_df.style.format("₹{:,.0f}"))


# --- TAB 4: METHODOLOGY ---
with tab4:
    st.header("Methodology Explained")

    with st.expander("Valuation Summary ('Football Field')"):
        st.markdown("""
        This chart is the standard output in investment banking. It plots the implied valuation
        from each method on a single graph.
        
        - **Why?** It quickly shows you the *range* of possible values.
        - **How to read:** If the **Current Price** (red line) is far to the left of most
          valuation bars, the stock may be **undervalued**. If it's far to the right, it
          may be **overvalued**.
        """)
        
    with st.expander("Comparable Analysis ('Smart' Comps)"):
        st.markdown("""
        This is a quantitative, 'algorithmic' approach to relative valuation.
        
        1.  **Problem:** Simple industry-average multiples are dumb. A fast-growing,
            high-margin company *deserves* a higher multiple.
        2.  **Solution:** We use **Multiple Linear Regression** to find the *statistical*
            relationship between fundamentals and valuation multiples.
        3.  **Models Used:**
            - `EV/EBITDA = f(Revenue Growth, EBITDA Margin)`
            - `P/E Ratio = f(Earnings Growth, Return on Equity)`
        4.  **Result:** The model gives a 'Fair' multiple for our target company. We
            use this to calculate an implied share price.
        """)
        
    with st.expander("Intrinsic Valuation (DCF)"):
        st.markdown("""
        This method ignores the market and values a company based on its *intrinsic*
        ability to generate cash.
        
        1.  **Get FCF:** We start with the Free Cash Flow (FCF) from the last 12 months.
        2.  **Forecast FCF:** We project this FCF for 5 years using the
            **'Short-Term Growth'** rate you provide.
        3.  **Calculate Terminal Value:** We estimate the value of all cash flows
            *after* year 5 using a perpetual growth model (with your
            **'Terminal Growth Rate'**).
        4.  **Discount:** We pull all future cash flows (Years 1-5 + Terminal Value)
            back to today's value using the **'WACC'** (Weighted Average Cost of Capital)
            you provide.
        5.  **Calculate Price:** This gives us the 'Fair' Enterprise Value. We subtract
            net debt and divide by shares outstanding to get the final implied price.
        """)

    with st.expander("Disclaimer"):
        st.warning("""
        **This is not financial advice.** This tool is for educational purposes.
        Financial data from APIs can be imperfect or 'dirty'. DCF models are
        extremely sensitive to assumptions.
        """)
