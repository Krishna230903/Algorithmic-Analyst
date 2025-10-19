import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression

# --- App Configuration ---
st.set_page_config(
    page_title="Professional FMVA Dashboard",
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
    ]
}

# --- 2. Session State Initialization ---
if 'valuation_results' not in st.session_state:
    st.session_state.valuation_results = {
        "Comps (EV/EBITDA)": None,
        "Comps (P/E)": None,
        "DCF": None,
        "DDM": None,
        "Current Price": None
    }

# --- 3. HARDCODED ASSUMPTIONS (for WACC/DDM) ---
RISK_FREE_RATE = 0.07  # Indian 10-Year Bond Yield (approx)
MARKET_RETURN = 0.12   # Nifty 50 Long-Term Average Return

# --- 4. NEW DATA SOURCING FUNCTIONS ---

@st.cache_data(ttl=3600)
def get_target_data(ticker):
    """
    Performs a deep-dive data pull for ONLY the target company.
    Pulls raw statements to calculate robust metrics.
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # Get 4 years of data to calculate 3-year metrics
    financials = stock.financials.iloc[:, :4]
    bs = stock.balance_sheet.iloc[:, :4]
    cf = stock.cash_flow.iloc[:, :4]

    data = {}
    
    try:
        # --- Basic Info ---
        data['Current Price'] = info.get('currentPrice', 0)
        data['Market Cap'] = info.get('marketCap', 0)
        data['Shares Outstanding'] = info.get('sharesOutstanding', 0)
        data['Beta'] = info.get('beta', 1.0) # Default to 1.0 if missing
        data['Dividend Rate'] = info.get('dividendRate', 0) # For DDM
        data['Company Name'] = info.get('shortName', ticker)

        # --- Calculate Averages (more stable) ---
        rev = financials.loc['Total Revenue']
        data['Revenue (TTM)'] = rev.iloc[0]
        data['Revenue CAGR (3Y)'] = ((rev.iloc[0] / rev.iloc[3])**(1/3)) - 1
        
        ebit = financials.loc['Operating Income']
        dep = cf.loc['Depreciation']
        ebitda = ebit + dep
        data['EBITDA (TTM)'] = ebitda.iloc[0]
        data['EBITDA Margin (3Y Avg)'] = (ebitda / rev).mean()

        op_cf = cf.loc['Total Cash From Operating Activities']
        capex = cf.loc['Capital Expenditures']
        fcf = op_cf + capex # Capex is negative
        data['FCF (3Y Avg)'] = fcf.iloc[:3].mean() # Use 3-year avg for DCF base
        
        # --- P/E Model Metrics ---
        net_income = financials.loc['Net Income']
        data['Earnings Growth (3Y CAGR)'] = ((net_income.iloc[0] / net_income.iloc[3])**(1/3)) - 1
        equity = bs.loc['Total Stockholder Equity']
        data['ROE (3Y Avg)'] = (net_income / equity).mean()
        data['P/E Ratio (TTM)'] = info.get('trailingPE', 0)
        
        # --- EV/EBITDA Model Metrics ---
        ev = info.get('enterpriseValue', 0)
        data['EV/EBITDA (TTM)'] = ev / ebitda.iloc[0]
        
        # --- WACC Metrics ---
        tax_expense = financials.loc['Income Tax Expense']
        ebt = financials.loc['Income Before Tax']
        data['Tax Rate (3Y Avg)'] = (tax_expense / ebt).mean()
        
        data['Interest Expense'] = financials.loc['Interest Expense'].iloc[0]
        data['Total Debt'] = bs.loc['Short Long Term Debt'].iloc[0] + bs.loc['Long Term Debt'].iloc[0]
        
        return data

    except KeyError as e:
        st.error(f"Data missing for {ticker}: {e}. Try another company.")
        return None
    except Exception as e:
        st.error(f"An error occurred fetching data for {ticker}: {e}")
        return None

@st.cache_data(ttl=3600)
def get_peer_data(tickers):
    """
    Pulls data for the peer group, focusing only on metrics for regression.
    This is faster than a deep-dive for every peer.
    """
    data = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get 4 years of financials
            rev = stock.financials.loc['Total Revenue'].iloc[:4]
            ebit = stock.financials.loc['Operating Income'].iloc[:4]
            dep = stock.cash_flow.loc['Depreciation'].iloc[:4]
            ebitda = ebit + dep
            net_income = stock.financials.loc['Net Income'].iloc[:4]
            equity = stock.balance_sheet.loc['Total Stockholder Equity'].iloc[:4]

            # Calculate robust metrics
            rev_cagr = ((rev.iloc[0] / rev.iloc[-1])**(1/len(rev-1))) - 1
            avg_ebitda_margin = (ebitda / rev).mean()
            avg_roe = (net_income / equity).mean()
            
            data.append({
                "Ticker": ticker,
                "EV/EBITDA": info.get('enterpriseValue', np.nan) / ebitda.iloc[0],
                "P/E Ratio": info.get('trailingPE', np.nan),
                "Revenue CAGR (3Y)": rev_cagr,
                "EBITDA Margin (3Y Avg)": avg_ebitda_margin,
                "ROE (3Y Avg)": avg_roe,
                "Earnings Growth (TTM)": info.get('earningsGrowth', np.nan) # Use TTM for peers as CAGR is slow
            })
        except Exception as e:
            pass # Skip ticker if data is bad
            
    df = pd.DataFrame(data).set_index("Ticker")
    return df

# --- 5. HELPER FUNCTIONS (Valuation & UI) ---

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
    if len(df_cleaned) < len(features_x) + 2: return None, None 
    X = df_cleaned[features_x]
    y = df_cleaned[target_y]
    model = LinearRegression().fit(X, y)
    return model, df_cleaned

def create_football_field(results, target_ticker):
    """Creates a Plotly Football Field chart."""
    data = []
    for method, value in results.items():
        if value is not None and method != "Current Price" and not np.isnan(value):
            data.append(dict(Task=method, Start=value, Finish=value, Label=f"₹{value:.2f}"))
    if not data:
        st.info("Run valuations in other tabs to populate this chart.")
        return None
    df = pd.DataFrame(data).sort_values(by="Start")
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", text="Label",
                      title=f"Valuation Summary for {target_ticker}")
    fig.update_traces(textposition='outside')
    if results["Current Price"] is not None:
        fig.add_vline(x=results["Current Price"], line_width=3, line_dash="dash", line_color="red",
                      annotation_text=f"Current Price: ₹{results['Current Price']:.2f}",
                      annotation_position="bottom right")
    fig.update_layout(yaxis_title=None, xaxis_title="Implied Share Price (₹)", plot_bgcolor='white',
                      xaxis_showgrid=True, yaxis_showgrid=False, bargap=0.8)
    fig.update_yaxes(categoryorder='array', categoryarray=df.sort_values(by="Start", ascending=False)['Task'])
    return fig

def calculate_wacc(target_info):
    """Calculates WACC from raw target company data."""
    try:
        # 1. Cost of Equity (Ke) using CAPM
        Ke = RISK_FREE_RATE + target_info['Beta'] * (MARKET_RETURN - RISK_FREE_RATE)
        
        # 2. Cost of Debt (Kd)
        Kd = target_info['Interest Expense'] / target_info['Total Debt']
        
        # 3. Market Values
        E = target_info['Market Cap']
        D = target_info['Total Debt'] # Using book value of debt as proxy
        V = E + D
        
        # 4. Tax Rate
        tax_rate = target_info['Tax Rate (3Y Avg)']
        
        # 5. WACC Formula
        wacc = (E/V * Ke) + (D/V * Kd * (1 - tax_rate))
        return wacc, Ke, Kd
    except Exception as e:
        return None, None, None

def create_sensitivity_table(base_fcf, g_short, wacc, g_long, net_debt, shares):
    """Creates a 2D sensitivity table for DCF."""
    wacc_range = np.linspace(wacc - 0.01, wacc + 0.01, 5)
    g_long_range = np.linspace(g_long - 0.01, g_long + 0.01, 5)
    
    table = pd.DataFrame(index=wacc_range, columns=g_long_range)
    
    for w in wacc_range:
        for g in g_long_range:
            if w <= g:
                table.loc[w, g] = np.nan
                continue
            
            # Run DCF calc
            fcf_forecasts = [base_fcf * (1 + g_short)**year for year in range(1, 6)]
            fcf_yr5 = fcf_forecasts[-1]
            terminal_value = (fcf_yr5 * (1 + g)) / (w - g)
            discounted_fcf = [fcf / (1 + w)**year for year, fcf in enumerate(fcf_forecasts, 1)]
            d_terminal_value = terminal_value / (1 + w)**5
            
            dcf_enterprise_value = sum(discounted_fcf) + d_terminal_value
            dcf_equity_value = dcf_enterprise_value - net_debt
            dcf_price_per_share = dcf_equity_value / shares
            table.loc[w, g] = dcf_price_per_share
            
    table.index.name = "WACC"
    table.columns.name = "Terminal Growth"
    return table.style.format("₹{:.2f}").background_gradient(cmap='RdYlGn', axis=None)


# --- 6. SIDEBAR INPUTS ---
st.sidebar.header("Valuation Inputs")
sector_list = list(SECTOR_TICKER_MAP.keys())
selected_sector = st.sidebar.selectbox("1. Select a Sector", sector_list)
peer_tickers = SECTOR_TICKER_MAP[selected_sector]
target_ticker = st.sidebar.selectbox("2. Select Company to Value", peer_tickers)

# --- 7. MAIN APP ---
st.title(f"Professional FMVA Dashboard 🇮🇳")
st.subheader(f"Target: {target_ticker} | Sector: {selected_sector}")

# --- Data Fetching ---
try:
    with st.spinner(f"Running deep-dive analysis on {target_ticker}..."):
        target_info = get_target_data(target_ticker)
    
    if target_info is None:
        st.error("Failed to get target data. App cannot proceed.")
        st.stop()
        
    with st.spinner(f"Fetching data for {len(peer_tickers)} peers..."):
        df_peers_raw = get_peer_data(peer_tickers)
        
    st.session_state.valuation_results["Current Price"] = target_info["Current Price"]
    
except Exception as e:
    st.error(f"An error occurred during data fetching: {e}")
    st.stop()

# --- Define Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Valuation Summary", 
    "📈 Comparable Analysis", 
    "💸 Intrinsic (DCF) Analysis", 
    "🧠 Methodology & Calcs"
])

# --- TAB 1: VALUATION SUMMARY ---
with tab1:
    st.header("Valuation 'Football Field' Summary")
    st.markdown("Run models in other tabs to populate this chart. Values are auto-saved.")
    
    fig = create_football_field(st.session_state.valuation_results, target_ticker)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    with st.container(border=True):
        st.subheader("Final Verdict")
        valid_results = [v for k, v in st.session_state.valuation_results.items() 
                         if v is not None and k != "Current Price" and not np.isnan(v)]
        
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
            
            if upside > 0.20: st.success(f"**Verdict: Potentially Undervalued.**")
            elif upside < -0.20: st.error(f"**Verdict: Potentially Overvalued.**")
            else: st.info(f"**Verdict: Fairly Valued.**")

# --- TAB 2: COMPARABLE ANALYSIS ---
with tab2:
    st.header("Comparable Company Analysis (Comps)")
    if "Banking" in selected_sector:
        st.warning("⚠️ **Warning:** P/E and EV/EBITDA are not reliable for banks. P/B is preferred.")

    # --- Comps Model 1: EV/EBITDA ---
    with st.container(border=True):
        st.subheader("Model 1: 'Smart' EV/EBITDA")
        features_ev = ['Revenue CAGR (3Y)', 'EBITDA Margin (3Y Avg)']
        model_ev, df_cleaned_ev = run_regression_model(df_peers_raw, 'EV/EBITDA', features_ev)
        
        if model_ev is None:
            st.error("Not enough clean peer data to build the EV/EBITDA model.")
        else:
            target_fundamentals_ev = pd.Series(target_info)[features_ev]
            if target_fundamentals_ev.isnull().any():
                st.warning(f"{target_ticker} is missing data for this model.")
            else:
                fair_multiple_ev = model_ev.predict([target_fundamentals_ev])[0]
                fair_ev = fair_multiple_ev * target_info['EBITDA (TTM)']
                net_debt = target_info['Market Cap'] - target_info['Market Cap'] # EV - Market Cap
                fair_equity_value_ev = fair_ev - net_debt
                fair_price_ev = fair_equity_value_ev / target_info['Shares Outstanding']
                
                st.session_state.valuation_results["Comps (EV/EBITDA)"] = fair_price_ev
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Live EV/EBITDA", f"{target_info['EV/EBITDA (TTM)']:.2f}x")
                col2.metric("'Fair' EV/EBITDA", f"{fair_multiple_ev:.2f}x")
                col3.metric("Implied Fair Price", f"₹{fair_price_ev:.2f}")

    # --- Comps Model 2: P/E Ratio ---
    with st.container(border=True):
        st.subheader("Model 2: 'Smart' P/E Ratio")
        features_pe = ['Earnings Growth (TTM)', 'ROE (3Y Avg)']
        model_pe, df_cleaned_pe = run_regression_model(df_peers_raw, 'P/E Ratio', features_pe)
        
        if model_pe is None:
            st.error("Not enough clean peer data to build the P/E model.")
        else:
            target_fundamentals_pe = pd.Series(target_info)[['Earnings Growth (3Y CAGR)', 'ROE (3Y Avg)']]
            target_fundamentals_pe.rename(index={'Earnings Growth (3Y CAGR)': 'Earnings Growth (TTM)'}, inplace=True) # Align col names
            
            if target_fundamentals_pe.isnull().any() or target_info['P/E Ratio (TTM)'] == 0:
                st.warning(f"{target_ticker} is missing data for this model.")
            else:
                fair_multiple_pe = model_pe.predict([target_fundamentals_pe])[0]
                fair_price_pe = (fair_multiple_pe / target_info['P/E Ratio (TTM)']) * target_info['Current Price']
                st.session_state.valuation_results["Comps (P/E)"] = fair_price_pe
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Live P/E", f"{target_info['P/E Ratio (TTM)']:.2f}x")
                col2.metric("'Fair' P/E", f"{fair_multiple_pe:.2f}x")
                col3.metric("Implied Fair Price", f"₹{fair_price_pe:.2f}")

    # --- Model 3: Dividend Discount Model (DDM) ---
    with st.container(border=True):
        st.subheader("Model 3: Dividend Discount Model (DDM)")
        if target_info['Dividend Rate'] == 0:
            st.info(f"{target_ticker} does not pay a dividend. DDM is not applicable.")
        else:
            wacc, Ke, Kd = calculate_wacc(target_info)
            if Ke is None:
                st.warning("Could not calculate Cost of Equity (Ke) for DDM.")
            else:
                st.markdown(f"Using a calculated **Cost of Equity (Ke) of {Ke:.2%}**.")
                g_long_ddm = st.slider("Long-Term Dividend Growth Rate (g)", 0.0, 0.05, 0.03, 0.005, key="ddm_g")
                
                if Ke <= g_long_ddm:
                    st.error("Cost of Equity (Ke) must be greater than growth rate (g).")
                else:
                    # Price = D1 / (Ke - g)  where D1 = D0 * (1+g)
                    d0 = target_info['Dividend Rate']
                    d1 = d0 * (1 + g_long_ddm)
                    ddm_price = d1 / (Ke - g_long_ddm)
                    st.session_state.valuation_results["DDM"] = ddm_price
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Last Dividend (D0)", f"₹{d0:.2f}")
                    col2.metric("Implied Fair Price", f"₹{ddm_price:.2f}")
                    st.latex(f"Price = \\frac{{D_0 \\times (1 + g)}}{{K_e - g}} = \\frac{{₹{d0:.2f} \\times (1 + {g_long_ddm})}}{{{Ke:.2%} - {g_long_ddm}}} = ₹{ddm_price:.2f}")

# --- TAB 3: INTRINSIC (DCF) ANALYSIS ---
with tab3:
    st.header("Intrinsic Valuation (Discounted Cash Flow)")
    
    with st.container(border=True):
        st.subheader("DCF Assumptions")
        
        # 1. Get Base FCF
        base_fcf = target_info['FCF (3Y Avg)']
        if pd.isna(base_fcf) or base_fcf == 0:
            st.error("Cannot run DCF: 3-Year Avg. Free Cash Flow is missing or zero.")
        else:
            st.metric("3-Year Average FCF (Base)", f"₹{base_fcf:,.0f}")
            
            # 2. Calculate WACC
            wacc, Ke, Kd = calculate_wacc(target_info)
            if wacc is None:
                st.error("Could not calculate WACC. Check company data.")
                st.stop()
            
            st.subheader("Calculated WACC: {wacc:.2%}")
            with st.expander("View WACC Calculation"):
                st.markdown(f"- **Cost of Equity (Ke):** {Ke:.2%} (Risk-Free: {RISK_FREE_RATE:.1%}, Beta: {target_info['Beta']:.2f}, Market Return: {MARKET_RETURN:.1%})")
                st.markdown(f"- **Cost of Debt (Kd):** {Kd:.2%}")
                st.markdown(f"- **After-Tax Cost of Debt:** {Kd * (1 - target_info['Tax Rate (3Y Avg)']):.2%}")
                st.markdown(f"- **Weights:** Equity: {target_info['Market Cap'] / (target_info['Market Cap'] + target_info['Total Debt']):.1%}, Debt: {target_info['Total Debt'] / (target_info['Market Cap'] + target_info['Total Debt']):.1%}")

            # 3. Get User Assumptions
            col1, col2, col3 = st.columns(3)
            wacc_adj = col1.slider("Adjust WACC", -0.02, 0.02, 0.0, 0.001, format="%.3f")
            g_short = col2.slider("FCF Growth (Yrs 1-5)", 0.0, 0.25, 0.10, 0.01, key="dcf_g_short")
            g_long = col3.slider("Terminal Growth (Yr 5+)", 0.0, 0.05, 0.03, 0.005, key="dcf_g_long")
            
            final_wacc = wacc + wacc_adj

            # 4. Run DCF
            if final_wacc <= g_long:
                st.error("Adjusted WACC must be greater than Terminal Growth Rate.")
            else:
                fcf_forecasts = [base_fcf * (1 + g_short)**year for year in range(1, 6)]
                fcf_yr5 = fcf_forecasts[-1]
                terminal_value = (fcf_yr5 * (1 + g_long)) / (final_wacc - g_long)
                discounted_fcf = [fcf / (1 + final_wacc)**year for year, fcf in enumerate(fcf_forecasts, 1)]
                d_terminal_value = terminal_value / (1 + final_wacc)**5
                
                dcf_enterprise_value = sum(discounted_fcf) + d_terminal_value
                
                net_debt = target_info['Total Debt'] # Simple proxy
                dcf_equity_value = dcf_enterprise_value - net_debt
                dcf_price_per_share = dcf_equity_value / target_info['Shares Outstanding']
                
                st.session_state.valuation_results["DCF"] = dcf_price_per_share
                
                # 5. Display Results
                st.subheader("DCF Valuation Results")
                col1, col2, col3 = st.columns(3)
                col1.metric("Implied Enterprise Value", f"₹{dcf_enterprise_value:,.0f}")
                col2.metric("Implied Equity Value", f"₹{dcf_equity_value:,.0f}")
                col3.metric("Implied Fair Price per Share", f"₹{dcf_price_per_share:,.2f}")
                
                # 6. Sensitivity Analysis
                st.subheader("Sensitivity Analysis: Price per Share")
                sensitivity_df = create_sensitivity_table(base_fcf, g_short, final_wacc, g_long, net_debt, target_info['Shares Outstanding'])
                st.dataframe(sensitivity_df)

# --- TAB 4: METHODOLOGY & CALCS ---
with tab4:
    st.header("Methodology & Live Calculators")

    with st.expander("Core Valuation Concepts"):
        st.markdown("""
        - **Intrinsic Value (DCF, DDM):** What a company is *fundamentally* worth based on its ability to generate cash, regardless of what the market thinks.
        - **Relative Value (Comps):** What a company is worth *compared to its peers*. This is a beauty contest; it just tells you if a stock is cheaper or more expensive than others in its sector.
        - **Football Field:** A chart that plots all valuation methods to show a *range* of possible "fair" values.
        """)

    with st.expander("Comparable Analysis ('Smart' Comps)"):
        st.markdown("""
        We use **Multiple Linear Regression** to find the statistical relationship between fundamentals and valuation multiples.
        - **Models:**
            - `EV/EBITDA = f(3Y Revenue CAGR, 3Y Avg. EBITDA Margin)`
            - `P/E Ratio = f(TTM Earnings Growth, 3Y Avg. ROE)`
        - **Why?** This is 'smarter' than a simple average because it correctly gives a higher 'fair' multiple to high-growth, high-profitability companies.
        """)

    with st.expander("Dividend Discount Model (DDM)"):
        st.markdown("Used for mature, dividend-paying companies. It assumes the value is the sum of all future dividends.")
        st.latex(f"Price = \\frac{{D_0 \\times (1 + g)}}{{K_e - g}}")
        st.markdown("- **$D_0$**: Last annual dividend.\n- **$g$**: Long-term (terminal) dividend growth rate.\n- **$K_e$**: Cost of Equity (calculated via CAPM).")

    with st.expander("Intrinsic Valuation (DCF) & WACC"):
        st.markdown("The DCF values a company by projecting its future Free Cash Flow (FCF) and discounting it back to today.")
        st.subheader("Live WACC Calculator")
        st.markdown("WACC is the 'discount rate' used. A higher WACC means future cash is worth less today, lowering the valuation.")
        
        col1, col2, col3 = st.columns(3)
        ke_calc = col1.number_input("Cost of Equity (Ke)", value=0.12, step=0.005, format="%.3f")
        kd_calc = col2.number_input("Cost of Debt (Kd)", value=0.08, step=0.005, format="%.3f")
        tax_calc = col3.number_input("Tax Rate", value=0.25, step=0.01, format="%.2f")
        col1, col2 = st.columns(2)
        e_calc = col1.number_input("Market Cap (E)", value=100000)
        d_calc = col2.number_input("Total Debt (D)", value=20000)
        
        v_calc = e_calc + d_calc
        wacc_calc = (e_calc / v_calc * ke_calc) + (d_calc / v_calc * kd_calc * (1 - tax_calc))
        st.latex(f"WACC = (E/V \\times Ke) + (D/V \\times Kd \\times (1 - t))")
        st.latex(f"WACC = ({e_calc / v_calc:.1%} \\times {ke_calc:.1%}) + ({d_calc / v_calc:.1%} \\times {kd_calc:.1%} \\times (1 - {tax_calc:.0%})) = {wacc_calc:.2%}")

    with st.expander("Disclaimer"):
        st.warning("""
        **This is not financial advice.** This tool is for educational purposes.
        Financial data from APIs can be imperfect or 'dirty'. DCF models are
        extremely sensitive to assumptions.
        """)
