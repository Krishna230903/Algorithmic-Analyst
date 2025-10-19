import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

# --- App Configuration ---
st.set_page_config(
    page_title="FMVA Valuation Dashboard (Pro)",
    page_icon="🇮🇳",
    layout="wide"
)

# --- 1. SECTOR TO TICKER MAPPING (Expanded) ---
SECTOR_TICKER_MAP = {
    "IT & Software": [
        "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", 
        "LTIM.NS", "MPHASIS.NS", "PERSISTENT.NS", "COFORGE.NS", "TATATECH.NS",
        "TATAELXSI.NS", "KPITTECH.NS", "LTT.NS", "OFSS.NS", "ZENSARTECH.NS"
    ],
    "FMCG": [
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", 
        "DABUR.NS", "MARICO.NS", "GODREJCP.NS", "COLPAL.NS", "PGHH.NS",
        "VBL.NS", "RADICO.NS", "UBL.NS", "EMAMILTD.NS", "GSKCONS.NS"
    ],
    "Automotive": [
        "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", 
        "EICHERMOT.NS", "HEROMOTOCO.NS", "TVSMOTOR.NS", "ASHOKLEY.NS", 
        "BHARATFORG.NS", "BOSCHLTD.NS", "MRF.NS", "APOLLOTYRE.NS", "BALKRISIND.NS"
    ],
    "Pharma": [
        "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", 
        "LUPIN.NS", "AUROPHARMA.NS", "TORNTPHARM.NS", "ALKEM.NS", "GLENMARK.NS",
        "ZYDUSLIFE.NS", "BIOCON.NS", "GLAND.NS", "LAURUSLABS.NS"
    ],
    "Banking (Private & PSU)": [
        "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", 
        "INDUSINDBK.NS", "IDFCFIRSTB.NS", "YESBANK.NS", "BANDHANBNK.NS",
        "SBIN.NS", "BANKBARODA.NS", "PNB.NS", "CANBK.NS", "FEDERALBNK.NS", "RBLBANK.NS"
    ],
    "Oil & Gas": [
        "RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS", "GAIL.NS", 
        "HINDPETRO.NS", "IGL.NS", "MGL.NS", "PETRONET.NS", "ATGL.NS"
    ],
    "Retail & Consumer": [
        "DMART.NS", "TITAN.NS", "RELAXO.NS", "BATAINDIA.NS", "TRENT.NS",
        "VEDANTFASH.NS", "ADITYAFRL.NS", "METROBRAND.NS", "PAGEIND.NS"
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
    """Pulls a comprehensive set of financial data for a list of tickers."""
    data = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            ebitda = info.get('ebitda', np.nan)
            enterprise_value = info.get('enterpriseValue', np.nan)
            op_cashflow = info.get('operatingCashflow', np.nan)
            cap_ex = info.get('capitalExpenditure', np.nan)

            ev_ebitda = (enterprise_value / ebitda) if (pd.notna(ebitda) and ebitda != 0) else np.nan
            fcf = (op_cashflow or 0) + (cap_ex or 0) 

            data.append({
                "Ticker": ticker,
                "Company Name": info.get('shortName', ticker),
                "Current Price": info.get('currentPrice', np.nan),
                "EV/EBITDA": ev_ebitda,
                "P/E Ratio": info.get('trailingPE', np.nan),
                "Revenue Growth": info.get('revenueGrowth', np.nan),
                "EBITDA Margin": info.get('ebitdaMargins', np.nan),
                "ROE": info.get('returnOnEquity', np.nan),
                "Earnings Growth": info.get('earningsGrowth', np.nan),
                "FCF (TTM)": fcf,
                "Enterprise Value": enterprise_value,
                "EBITDA": ebitda,
                "Market Cap": info.get('marketCap', np.nan),
                "Shares Outstanding": info.get('sharesOutstanding', np.nan)
            })
        except Exception as e:
            pass 
            
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
        return None, None 
        
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
        return None
        
    df = pd.DataFrame(data).sort_values(by="Start")
    
    # --- *** BUG FIX IS HERE *** ---
    # Used px.timeline instead of px.bar
    fig = px.timeline(df, 
                      x_start="Start", 
                      x_end="Finish", 
                      y="Task", 
                      text="Label",
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
        yaxis_showgrid=False,
        # Make bars thicker
        bargap=0.8 
    )
    fig.update_yaxes(categoryorder='array', categoryarray=df.sort_values(by="Start", ascending=False)['Task'])
    return fig


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
        
    if target_ticker not in df_raw.index or pd.isna(df_raw.loc[target_ticker, "Current Price"]):
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
    "🧠 Methodology & Live Calcs"
])

# --- TAB 1: VALUATION SUMMARY ---
with tab1:
    st.header("Valuation 'Football Field' Summary")
    st.markdown("This chart summarizes the valuation ranges from all methods. Run the models in the other tabs to populate it.")
    
    fig = create_football_field(st.session_state.valuation_results, target_ticker)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # --- NEW: Final Verdict ---
    with st.container(border=True):
        st.subheader("Final Verdict")
        
        valid_results = [
            v for k, v in st.session_state.valuation_results.items() 
            if v is not None and k != "Current Price"
        ]
        
        if not valid_results:
            st.info("Run valuations in the 'Comps' and 'DCF' tabs to generate a verdict.")
        else:
            avg_valuation = np.mean(valid_results)
            current_price = st.session_state.valuation_results["Current Price"]
            upside = (avg_valuation / current_price) - 1
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"₹{current_price:.2f}")
            col2.metric("Average 'Fair' Value", f"₹{avg_valuation:.2f}")
            col3.metric("Implied Upside", f"{upside:.2%}")
            
            if upside > 0.20:
                st.success(f"**Verdict: Potentially Undervalued.** The average 'fair' value of ₹{avg_valuation:.2f} is significantly higher than the current price.")
            elif upside < -0.20:
                st.error(f"**Verdict: Potentially Overvalued.** The average 'fair' value of ₹{avg_valuation:.2f} is significantly lower than the current price.")
            else:
                st.info(f"**Verdict: Fairly Valued.** The average 'fair' value of ₹{avg_valuation:.2f} is in line with the current price.")

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
            target_fundamentals_ev = target_info[features_ev]
            if target_fundamentals_ev.isnull().any():
                st.warning(f"{target_ticker} is missing data for this model.")
            else:
                fair_multiple_ev = model_ev.predict([target_fundamentals_ev])[0]
                fair_ev = fair_multiple_ev * target_info['EBITDA']
                net_debt = target_info['Enterprise Value'] - target_info['Market Cap']
                fair_equity_value_ev = fair_ev - net_debt
                fair_price_ev = fair_equity_value_ev / target_info['Shares Outstanding']
                
                st.session_state.valuation_results["Comps (EV/EBITDA)"] = fair_price_ev
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Live EV/EBITDA", f"{target_info['EV/EBITDA']:.2f}x")
                col2.metric("'Fair' EV/EBITDA", f"{fair_multiple_ev:.2f}x")
                col3.metric("Implied Fair Price", f"₹{fair_price_ev:.2f}")

            with st.expander("View Model Data & Formula"):
                st.latex(f"EV/EBITDA = {model_ev.intercept_:.2f} + ({model_ev.coef_[0]:.2f} \\times Growth) + ({model_ev.coef_[1]:.2f} \\times Margin)")
                st.dataframe(df_cleaned_ev[features_ev + ['EV/EBITDA']].style.format("{:,.2f}"))

    # --- Comps Model 2: P/E Ratio ---
    with st.container(border=True):
        st.subheader("Model 2: 'Smart' P/E Ratio")
        features_pe = ['Earnings Growth', 'ROE']
        model_pe, df_cleaned_pe = run_regression_model(df_raw, 'P/E Ratio', features_pe)
        
        if model_pe is None:
            st.error("Not enough clean data to build the P/E model.")
        else:
            target_fundamentals_pe = target_info[features_pe]
            if target_fundamentals_pe.isnull().any() or target_info['P/E Ratio'] is None or target_info['P/E Ratio'] == 0:
                st.warning(f"{target_ticker} is missing data for this model.")
            else:
                fair_multiple_pe = model_pe.predict([target_fundamentals_pe])[0]
                fair_price_pe = (fair_multiple_pe / target_info['P/E Ratio']) * target_info['Current Price']
                
                st.session_state.valuation_results["Comps (P/E)"] = fair_price_pe
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Live P/E", f"{target_info['P/E Ratio']:.2f}x")
                col2.metric("'Fair' P/E", f"{fair_multiple_pe:.2f}x")
                col3.metric("Implied Fair Price", f"₹{fair_price_pe:.2f}")

            with st.expander("View Model Data & Formula"):
                st.latex(f"P/E = {model_pe.intercept_:.2f} + ({model_pe.coef_[0]:.2f} \\times E.Growth) + ({model_pe.coef_[1]:.2f} \\times ROE)")
                st.dataframe(df_cleaned_pe[features_pe + ['P/E Ratio']].style.format("{:,.2f}"))

# --- TAB 3: INTRINSIC (DCF) ANALYSIS ---
with tab3:
    st.header("Intrinsic Valuation (Discounted Cash Flow)")
    
    with st.container(border=True):
        st.subheader("DCF Assumptions")
        
        ttm_fcf = target_info['FCF (TTM)']
        if pd.isna(ttm_fcf) or ttm_fcf == 0:
            st.error("Cannot run DCF: TTM Free Cash Flow data is missing or zero.")
        else:
            st.metric("Last 12 Months Free Cash Flow (TTM)", f"₹{ttm_fcf:,.0f}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                g_short = st.slider("Short-Term Growth (Yrs 1-5)", 0.0, 0.25, 0.10, 0.01, key="dcf_g_short")
            with col2:
                wacc = st.slider("Discount Rate (WACC)", 0.05, 0.15, 0.10, 0.005, key="dcf_wacc")
            with col3:
                g_long = st.slider("Terminal Growth Rate", 0.0, 0.05, 0.03, 0.005, key="dcf_g_long")

            if wacc <= g_long:
                st.error("WACC must be greater than Terminal Growth Rate.")
            else:
                fcf_forecasts = [ttm_fcf * (1 + g_short)**year for year in range(1, 6)]
                
                fcf_yr5 = fcf_forecasts[-1]
                terminal_value = (fcf_yr5 * (1 + g_long)) / (wacc - g_long)
                
                discounted_fcf = [fcf / (1 + wacc)**year for year, fcf in enumerate(fcf_forecasts, 1)]
                d_terminal_value = terminal_value / (1 + wacc)**5
                
                dcf_enterprise_value = sum(discounted_fcf) + d_terminal_value
                
                net_debt = target_info['Enterprise Value'] - target_info['Market Cap']
                dcf_equity_value = dcf_enterprise_value - net_debt
                dcf_price_per_share = dcf_equity_value / target_info['Shares Outstanding']
                
                st.session_state.valuation_results["DCF"] = dcf_price_per_share
                
                st.subheader("DCF Valuation Results")
                col1, col2, col3 = st.columns(3)
                col1.metric("Implied Enterprise Value", f"₹{dcf_enterprise_value:,.0f}")
                col2.metric("Implied Equity Value", f"₹{dcf_equity_value:,.0f}")
                col3.metric("Implied Fair Price per Share", f"₹{dcf_price_per_share:,.2f}")
                
                with st.expander("View 5-Year FCF Forecast"):
                    forecast_df = pd.DataFrame({
                        'Year': [1, 2, 3, 4, 5],
                        'Projected FCF': fcf_forecasts,
                        'Discounted FCF': discounted_fcf
                    })
                    forecast_df.loc['Terminal'] = ['Terminal', terminal_value, d_terminal_value]
                    st.dataframe(forecast_df.style.format("₹{:,.0f}"))


# --- TAB 4: METHODOLOGY & LIVE CALCS ---
with tab4:
    st.header("Methodology & Live Calculators")

    with st.expander("Comparable Analysis ('Smart' Comps)", expanded=True):
        st.markdown("""
        This is a quantitative, 'algorithmic' approach to relative valuation.
        
        1.  **Problem:** Simple industry-average multiples are dumb. A fast-growing,
            high-margin company *deserves* a higher multiple.
        2.  **Solution:** We use **Multiple Linear Regression** to find the *statistical*
            relationship between fundamentals and valuation multiples.
        3.  **Models Used:**
            - `EV/EBITDA = f(Revenue Growth, EBITDA Margin)`
            - `P/E Ratio = f(Earnings Growth, Return on Equity)`
        
        #### Live P/E Calculator
        See how a P/E multiple translates to a price.
        """)
        
        col1, col2 = st.columns(2)
        pe_calc = col1.number_input("Example 'Fair' P/E Multiple", value=25.0, step=0.5)
        eps_calc = col2.number_input("Example Earnings Per Share (EPS)", value=50.0, step=1.0)
        
        st.latex(f"Implied\\ Price = P/E \\times EPS")
        st.latex(f"Implied\\ Price = {pe_calc:,.2f} \\times ₹{eps_calc:,.2f} = ₹{pe_calc * eps_calc:,.2f}")
        
    with st.expander("Intrinsic Valuation (DCF)", expanded=True):
        st.markdown("""
        This method ignores the market and values a company based on its *intrinsic*
        ability to generate cash.
        
        1.  **Forecast FCF:** We project Free Cash Flow (FCF) for 5 years.
        2.  **Calculate Terminal Value:** We estimate the value of all cash flows *after* year 5.
        3.  **Discount:** We pull all future cash flows back to today's value using the WACC.
        
        #### Live DCF (Terminal Value) Calculator
        See how small changes in WACC and Growth dramatically change the value.
        """)
        
        col1, col2, col3 = st.columns(3)
        fcf_calc = col1.number_input("Example: Final Year FCF", value=1000.0, step=10.0)
        wacc_calc = col2.number_input("Example: Discount Rate (WACC)", value=0.10, step=0.005, format="%.3f")
        g_calc = col3.number_input("Example: Terminal Growth Rate", value=0.03, step=0.005, format="%.3f")

        st.latex(f"Terminal\\ Value = \\frac{{FCF_n \\times (1 + g)}}{{WACC - g}}")
        
        if wacc_calc <= g_calc:
            st.error("WACC must be greater than Growth Rate.")
        else:
            tv_calc = (fcf_calc * (1 + g_calc)) / (wacc_calc - g_calc)
            st.latex(f"Terminal\\ Value = \\frac{{₹{fcf_calc:,.0f} \\times (1 + {g_calc})}}{{{wacc_calc} - {g_calc}}} = ₹{tv_calc:,.0f}")
            st.markdown(f"**Present Value of TV (in 5 yrs):** `₹{tv_calc:,.0f} / (1 + {wacc_calc})^5` = **`₹{tv_calc / (1+wacc_calc)**5:,.0f}`**")

    with st.expander("Disclaimer"):
        st.warning("""
        **This is not financial advice.** This tool is for educational purposes.
        Financial data from APIs can be imperfect or 'dirty'. DCF models are
        extremely sensitive to assumptions.
        """)
