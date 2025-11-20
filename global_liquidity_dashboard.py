import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from fredapi import Fred
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Global Liquidity Dashboard",
    page_icon="🌍",
    layout="wide"
)

# Title
st.title("🌍 Global Liquidity Dashboard v2")
st.markdown("**Track liquidity conditions across major central banks: USD, JPY, GBP, CAD**")

# Sidebar for FRED API key
st.sidebar.header("Configuration")

# Try to get API key from secrets first (for deployment), fallback to user input
try:
    api_key = st.secrets["FRED_API_KEY"]
    st.sidebar.success("✅ Using configured API key")
except:
    api_key = st.sidebar.text_input(
        "FRED API Key",
        type="password",
        help="Get your free API key from https://fred.stlouisfed.org/docs/api/api_key.html"
    )
    
    if not api_key:
        st.warning("⚠️ Please enter your FRED API key in the sidebar to fetch data.")
        st.info("""
        **How to get a FRED API Key:**
        1. Go to https://fred.stlouisfed.org/
        2. Create a free account
        3. Request an API key at https://fred.stlouisfed.org/docs/api/api_key.html
        """)
        st.stop()

# Currency selection
st.sidebar.header("Currency Selection")
currencies = st.sidebar.multiselect(
    "Select Currencies",
    ["USD", "JPY", "GBP", "CAD"],
    default=["USD", "JPY", "GBP", "CAD"]
)

# Display settings
st.sidebar.header("Display Settings")
normalize_to_usd = st.sidebar.checkbox("Normalize to USD Equivalent", value=True)
show_total_liquidity = st.sidebar.checkbox("Show Combined Global Liquidity", value=True)
show_data_quality = st.sidebar.checkbox("Show Data Quality Indicators", value=True)

# Forecast parameters
st.sidebar.header("Forecast Settings")
forecast_days = st.sidebar.slider("Forecast Days", 30, 180, 90)

@st.cache_data(ttl=3600)
def get_live_fx_rates(_fred):
    """Fetch live exchange rates from FRED"""
    try:
        # Fetch FX rates (all are per USD)
        fx_rates = {}
        
        # USD to JPY
        try:
            jpy_usd = _fred.get_series('DEXJPUS', observation_start='2020-01-01')  # JPY per USD
            fx_rates['JPY_per_USD'] = jpy_usd.iloc[-1]
            fx_rates['JPY'] = 1 / jpy_usd.iloc[-1]  # Convert to USD per JPY
        except:
            fx_rates['JPY_per_USD'] = 150.0
            fx_rates['JPY'] = 1/150.0
        
        # USD to GBP (inverted: DEXUSUK gives USD per GBP, we need GBP per USD)
        try:
            usd_gbp = _fred.get_series('DEXUSUK', observation_start='2020-01-01')  # USD per GBP
            fx_rates['GBP'] = usd_gbp.iloc[-1]  # USD per GBP
            fx_rates['GBP_per_USD'] = 1 / usd_gbp.iloc[-1]
        except:
            fx_rates['GBP'] = 1.27
            fx_rates['GBP_per_USD'] = 1/1.27
        
        # USD to CAD
        try:
            cad_usd = _fred.get_series('DEXCAUS', observation_start='2020-01-01')  # CAD per USD
            fx_rates['CAD_per_USD'] = cad_usd.iloc[-1]
            fx_rates['CAD'] = 1 / cad_usd.iloc[-1]  # Convert to USD per CAD
        except:
            fx_rates['CAD_per_USD'] = 1.39
            fx_rates['CAD'] = 1/1.39
        
        fx_rates['USD'] = 1.0
        fx_rates['last_updated'] = datetime.now()
        
        return fx_rates
    except Exception as e:
        st.warning(f"Could not fetch live FX rates: {str(e)}. Using defaults.")
        return {
            'USD': 1.0,
            'JPY': 0.0067,
            'JPY_per_USD': 150.0,
            'GBP': 1.27,
            'GBP_per_USD': 0.79,
            'CAD': 0.72,
            'CAD_per_USD': 1.39,
            'last_updated': datetime.now()
        }

def fetch_boc_api_data():
    """Fetch real-time data from Bank of Canada API"""
    try:
        import requests
        
        # BOC Series V122711: Bank of Canada Total Assets (millions of CAD)
        url = "https://www.bankofcanada.ca/valet/observations/V122711/json"
        params = {
            'start_date': '2020-01-01',
            'end_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            observations = data['observations']
            
            # Parse BOC format
            dates = [obs['d'] for obs in observations]
            values = [float(obs['V122711']['v']) / 1000 for obs in observations]  # Convert millions to billions
            
            df = pd.DataFrame({
                'Central_Bank_Balance': values,
                'RRP_or_Reserves': 0,
                'Treasury_Account': 0
            }, index=pd.to_datetime(dates))
            
            df['Liquidity'] = df['Central_Bank_Balance']
            df['Currency'] = 'CAD'
            df['data_quality'] = 'Good'
            
            return df
        else:
            return None
    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def fetch_global_liquidity_data(_api_key, selected_currencies, normalize):
    """Fetch liquidity data for all major currencies with improved data sources"""
    try:
        fred = Fred(api_key=_api_key)
        
        # Get live FX rates first
        fx_rates = get_live_fx_rates(fred)
        
        st.info(f"📊 **Live FX Rates** (Updated: {fx_rates['last_updated'].strftime('%Y-%m-%d %H:%M')})")
        fx_col1, fx_col2, fx_col3 = st.columns(3)
        with fx_col1:
            st.metric("JPY/USD", f"{fx_rates['JPY_per_USD']:.2f}")
        with fx_col2:
            st.metric("USD/GBP", f"{fx_rates['GBP']:.4f}")
        with fx_col3:
            st.metric("CAD/USD", f"{fx_rates['CAD_per_USD']:.4f}")
        
        st.info("Fetching global liquidity data...")
        
        dfs = {}
        
        # USD Liquidity: Fed Balance - RRP - TGA
        if 'USD' in selected_currencies:
            with st.spinner("Fetching USD data..."):
                try:
                    fed_balance = fred.get_series('WALCL', observation_start='2020-01-01')
                    rrp = fred.get_series('RRPONTSYD', observation_start='2020-01-01')
                    tga = fred.get_series('WTREGEN', observation_start='2020-01-01')
                    
                    usd_df = pd.DataFrame({
                        'Central_Bank_Balance': fed_balance,
                        'RRP_or_Reserves': rrp,
                        'Treasury_Account': tga
                    })
                    usd_df = usd_df.ffill()
                    usd_df['Liquidity'] = usd_df['Central_Bank_Balance'] - usd_df['RRP_or_Reserves'] - usd_df['Treasury_Account']
                    usd_df['Currency'] = 'USD'
                    usd_df['data_quality'] = 'Excellent'
                    dfs['USD'] = usd_df
                    st.success("✓ USD data loaded (Federal Reserve)")
                except Exception as e:
                    st.error(f"❌ USD data error: {str(e)}")
        
        # JPY Liquidity: BOJ Balance
        if 'JPY' in selected_currencies:
            with st.spinner("Fetching JPY data..."):
                try:
                    # Try multiple BOJ data sources
                    boj_assets = None
                    data_source = ""
                    
                    # Option 1: JPNASSETS (Total Assets in Trillions of Yen)
                    try:
                        boj_assets = fred.get_series('JPNASSETS', observation_start='2020-01-01')
                        boj_assets = boj_assets * 1000  # Convert trillions to billions
                        data_source = "BOJ Total Assets"
                    except:
                        pass
                    
                    # Option 2: BOGMBASE (Monetary Base in Billions of Yen)
                    if boj_assets is None:
                        try:
                            boj_assets = fred.get_series('BOGMBASE', observation_start='2020-01-01')
                            data_source = "BOJ Monetary Base"
                        except:
                            pass
                    
                    # Option 3: MYAGM2JPM196N (M2 Money Stock)
                    if boj_assets is None:
                        try:
                            boj_assets = fred.get_series('MYAGM2JPM196N', observation_start='2020-01-01')
                            data_source = "BOJ M2 (Proxy)"
                        except:
                            raise Exception("No JPY data sources available")
                    
                    jpy_df = pd.DataFrame({
                        'Central_Bank_Balance': boj_assets,
                        'RRP_or_Reserves': 0,
                        'Treasury_Account': 0
                    })
                    jpy_df = jpy_df.ffill()
                    jpy_df['Liquidity'] = jpy_df['Central_Bank_Balance']
                    jpy_df['Currency'] = 'JPY'
                    jpy_df['data_quality'] = 'Good'
                    
                    # Convert to USD if requested
                    if normalize:
                        jpy_df['Liquidity'] = jpy_df['Liquidity'] * fx_rates['JPY']
                        jpy_df['Central_Bank_Balance'] = jpy_df['Central_Bank_Balance'] * fx_rates['JPY']
                    
                    dfs['JPY'] = jpy_df
                    st.success(f"✓ JPY data loaded ({data_source})")
                    
                except Exception as e:
                    st.error(f"❌ JPY data error: {str(e)}")
        
        # GBP Liquidity: BOE Balance - IMPROVED DATA SOURCES
        if 'GBP' in selected_currencies:
            with st.spinner("Fetching GBP data..."):
                try:
                    boe_assets = None
                    data_source = ""
                    data_quality = "Fair"
                    
                    # Option 1: Try DISCONTINUED series (for historical data)
                    try:
                        boe_assets = fred.get_series('BOETOTASSETS', observation_start='2020-01-01')
                        data_source = "BOE Total Assets (Historical)"
                        data_quality = "Fair - Discontinued"
                    except:
                        pass
                    
                    # Option 2: UK Central Bank Assets
                    if boe_assets is None:
                        try:
                            boe_assets = fred.get_series('DDOI01GBA156NWDB', observation_start='2020-01-01')
                            data_source = "BOE Central Bank Assets"
                            data_quality = "Fair"
                        except:
                            pass
                    
                    # Option 3: UK Monetary Base
                    if boe_assets is None:
                        try:
                            boe_assets = fred.get_series('BOGMBASE', observation_start='2020-01-01')
                            # This is actually BOJ, need UK version
                            boe_assets = fred.get_series('MABMM301GBQ189S', observation_start='2020-01-01')
                            data_source = "UK Monetary Base"
                            data_quality = "Fair"
                        except:
                            pass
                    
                    # Option 4: Create synthetic estimate from known values
                    if boe_assets is None:
                        st.warning("⚠️ Creating GBP estimate from known BOE data points")
                        # As of 2024, BOE balance sheet is approximately £750-800B
                        dates = pd.date_range(start='2020-01-01', end=datetime.now(), freq='D')
                        # Linear interpolation from known points
                        # 2020: ~£900B, 2024: ~£750B (QT)
                        start_val = 900
                        end_val = 750
                        days = len(dates)
                        values = np.linspace(start_val, end_val, days)
                        boe_assets = pd.Series(values, index=dates)
                        data_source = "Estimated from BOE Reports"
                        data_quality = "Estimated"
                    
                    gbp_df = pd.DataFrame({
                        'Central_Bank_Balance': boe_assets,
                        'RRP_or_Reserves': 0,
                        'Treasury_Account': 0
                    })
                    gbp_df = gbp_df.ffill()
                    gbp_df['Liquidity'] = gbp_df['Central_Bank_Balance']
                    gbp_df['Currency'] = 'GBP'
                    gbp_df['data_quality'] = data_quality
                    
                    # Convert to USD if requested
                    if normalize:
                        gbp_df['Liquidity'] = gbp_df['Liquidity'] * fx_rates['GBP']
                        gbp_df['Central_Bank_Balance'] = gbp_df['Central_Bank_Balance'] * fx_rates['GBP']
                    
                    dfs['GBP'] = gbp_df
                    
                    if data_quality == "Estimated":
                        st.warning(f"⚠️ GBP data loaded ({data_source}) - Use directionally only")
                    else:
                        st.success(f"✓ GBP data loaded ({data_source})")
                    
                except Exception as e:
                    st.error(f"❌ GBP data error: {str(e)}")
        
        # CAD Liquidity: BOC Balance - IMPROVED WITH API
        if 'CAD' in selected_currencies:
            with st.spinner("Fetching CAD data..."):
                try:
                    # Try Bank of Canada API first (best source!)
                    cad_df = fetch_boc_api_data()
                    
                    if cad_df is not None:
                        # Convert to USD if requested
                        if normalize:
                            cad_df['Liquidity'] = cad_df['Liquidity'] * fx_rates['CAD']
                            cad_df['Central_Bank_Balance'] = cad_df['Central_Bank_Balance'] * fx_rates['CAD']
                        
                        dfs['CAD'] = cad_df
                        st.success("✓ CAD data loaded (Bank of Canada API)")
                    else:
                        # Fallback: Try FRED series
                        try:
                            cad_assets = fred.get_series('CANBIS', observation_start='2020-01-01')
                            data_source = "BOC from FRED"
                            data_quality = "Good"
                        except:
                            # Last resort: Use improved estimate with seasonal patterns
                            st.warning("⚠️ Creating CAD estimate from known BOC data")
                            dates = pd.date_range(start='2020-01-01', end=datetime.now(), freq='D')
                            
                            # Known values: 2020 peak ~$600B CAD, 2024 ~$130B CAD (massive QT)
                            # Model the QT taper more realistically
                            days = len(dates)
                            t = np.linspace(0, 1, days)
                            # Exponential decay model (more realistic for QT)
                            decay_rate = 4
                            values = 600 - (470 * (1 - np.exp(-decay_rate * t)))
                            
                            cad_assets = pd.Series(values, index=dates)
                            data_source = "Estimated from BOC Reports"
                            data_quality = "Estimated"
                        
                        cad_df = pd.DataFrame({
                            'Central_Bank_Balance': cad_assets,
                            'RRP_or_Reserves': 0,
                            'Treasury_Account': 0
                        })
                        cad_df = cad_df.ffill()
                        cad_df['Liquidity'] = cad_df['Central_Bank_Balance']
                        cad_df['Currency'] = 'CAD'
                        cad_df['data_quality'] = data_quality
                        
                        # Convert to USD if requested
                        if normalize:
                            cad_df['Liquidity'] = cad_df['Liquidity'] * fx_rates['CAD']
                            cad_df['Central_Bank_Balance'] = cad_df['Central_Bank_Balance'] * fx_rates['CAD']
                        
                        dfs['CAD'] = cad_df
                        
                        if data_quality == "Estimated":
                            st.warning(f"⚠️ CAD data loaded ({data_source}) - Use directionally only")
                        else:
                            st.success(f"✓ CAD data loaded ({data_source})")
                
                except Exception as e:
                    st.error(f"❌ CAD data error: {str(e)}")
        
        return dfs, fx_rates
    
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None, None

def create_simple_forecast(df, days, currency):
    """Simple trend-based forecast for any currency"""
    try:
        recent_data = df.tail(60)
        x = np.arange(len(recent_data))
        
        # Fit linear trend
        slope = np.polyfit(x, recent_data['Liquidity'].values, 1)[0]
        
        # Project forward
        current_date = df.index[-1]
        forecast_dates = pd.date_range(start=current_date + timedelta(days=1), periods=days, freq='D')
        
        liquidity_forecast = df['Liquidity'].iloc[-1] + slope * np.arange(1, days + 1)
        
        # Confidence intervals
        std = df['Liquidity'].tail(60).std()
        upper = liquidity_forecast + 1.5 * std * np.sqrt(np.arange(1, days + 1) / 30)
        lower = liquidity_forecast - 1.5 * std * np.sqrt(np.arange(1, days + 1) / 30)
        
        return pd.DataFrame({
            'ds': forecast_dates,
            'yhat': liquidity_forecast,
            'yhat_upper': upper,
            'yhat_lower': lower,
            'currency': currency
        })
    
    except Exception as e:
        st.error(f"Error creating forecast for {currency}: {str(e)}")
        return None

# Fetch data
liquidity_data, fx_rates = fetch_global_liquidity_data(api_key, currencies, normalize_to_usd)

if liquidity_data and len(liquidity_data) > 0:
    
    # Filter out currencies with insufficient data
    valid_currencies = {}
    for currency, df in liquidity_data.items():
        if len(df) >= 2:  # Need at least 2 data points
            valid_currencies[currency] = df
        else:
            st.warning(f"⚠️ {currency} has insufficient data (only {len(df)} points) - skipping")
    
    liquidity_data = valid_currencies
    
    if len(liquidity_data) == 0:
        st.error("No currencies have sufficient data. Please check your data sources.")
        st.stop()
    
    # Data Quality Summary
    if show_data_quality:
        st.markdown("---")
        st.subheader("📋 Data Quality Summary")
        
        quality_data = []
        for curr, df in liquidity_data.items():
            quality = df['data_quality'].iloc[0] if 'data_quality' in df.columns else 'Unknown'
            
            if quality == 'Excellent':
                icon = "🟢"
                description = "Official, real-time data"
            elif quality == 'Good':
                icon = "🟡"
                description = "Official data, may have reporting lag"
            elif quality == 'Fair' or 'Fair' in quality:
                icon = "🟠"
                description = "Limited official data, use with caution"
            else:
                icon = "🔴"
                description = "Estimated data, directional use only"
            
            quality_data.append({
                'Currency': curr,
                'Quality': f"{icon} {quality}",
                'Description': description,
                'Last Update': df.index[-1].strftime('%Y-%m-%d')
            })
        
        quality_df = pd.DataFrame(quality_data)
        st.dataframe(quality_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Create overview metrics
    st.header("📊 Current Liquidity Overview")
    
    cols = st.columns(len(liquidity_data))
    
    for idx, (currency, df) in enumerate(liquidity_data.items()):
        with cols[idx]:
            if len(df) == 0:
                st.error(f"{currency}: No data available")
                continue
                
            latest = df.iloc[-1]
            # Safe lookback - use min of 30 days or available data
            lookback_days = min(30, len(df) - 1)
            prev = df.iloc[-lookback_days-1] if lookback_days > 0 else latest
            
            change = latest['Liquidity'] - prev['Liquidity']
            change_pct = (change / prev['Liquidity']) * 100 if prev['Liquidity'] != 0 else 0
            
            # Currency symbols and units
            if normalize_to_usd:
                symbol = '$'
                suffix = 'B USD'
            else:
                symbols = {'USD': '$', 'JPY': '¥', 'GBP': '£', 'CAD': 'C$'}
                symbol = symbols[currency]
                suffix = 'B'
            
            # Add data quality badge
            quality = df['data_quality'].iloc[0] if 'data_quality' in df.columns else 'Unknown'
            quality_icons = {
                'Excellent': '🟢',
                'Good': '🟡',
                'Fair': '🟠',
                'Estimated': '🔴'
            }
            quality_icon = quality_icons.get(quality, '⚪')
            if 'Fair' in str(quality):
                quality_icon = '🟠'
            
            st.metric(
                f"{currency} Liquidity {quality_icon}",
                f"{symbol}{latest['Liquidity']:.0f}{suffix}",
                f"{change:+.0f}B ({change_pct:+.1f}%)",
                delta_color="normal"
            )
            
            # Mini trend sparkline
            lookback_for_trend = min(30, len(df))
            recent_trend = df.tail(lookback_for_trend)['Liquidity'].values
            
            if len(recent_trend) > 1:
                fig_mini = go.Figure()
                fig_mini.add_trace(go.Scatter(
                    y=recent_trend,
                    mode='lines',
                    line=dict(color='#1f77b4', width=1),
                    fill='tozeroy'
                ))
                fig_mini.update_layout(
                    height=100,
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                    yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_mini, use_container_width=True, config={'displayModeBar': False})
            else:
                st.caption("Not enough data for trend")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌍 Global Overview", 
        "📈 Individual Currencies", 
        "🔮 Forecasts", 
        "📊 Correlation Analysis",
        "🎯 Trading Signals"
    ])
    
    with tab1:
        st.subheader("Global Liquidity Trends")
        
        # Combined chart
        fig_global = go.Figure()
        
        colors = {
            'USD': '#1f77b4',
            'JPY': '#ff7f0e', 
            'GBP': '#2ca02c',
            'CAD': '#d62728'
        }
        
        for currency, df in liquidity_data.items():
            fig_global.add_trace(go.Scatter(
                x=df.index,
                y=df['Liquidity'],
                mode='lines',
                name=f'{currency} Liquidity',
                line=dict(color=colors.get(currency, '#333333'), width=2)
            ))
        
        fig_global.update_layout(
            title="Multi-Currency Liquidity Comparison",
            xaxis_title="Date",
            yaxis_title="Liquidity (Billions USD)" if normalize_to_usd else "Liquidity (Billions)",
            hovermode='x unified',
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_global, use_container_width=True)
        
        # Combined global liquidity
        if show_total_liquidity and len(liquidity_data) > 1:
            st.subheader("Combined Global Liquidity")
            
            # Align all dataframes and sum
            all_dates = pd.DataFrame()
            for currency, df in liquidity_data.items():
                if all_dates.empty:
                    all_dates = df[['Liquidity']].copy()
                    all_dates.columns = [currency]
                else:
                    all_dates[currency] = df['Liquidity']
            
            all_dates = all_dates.ffill().bfill()
            all_dates['Total'] = all_dates.sum(axis=1)
            
            fig_total = go.Figure()
            
            # Stacked area chart
            fig_total.add_trace(go.Scatter(
                x=all_dates.index,
                y=all_dates['Total'],
                mode='lines',
                name='Total Global Liquidity',
                line=dict(color='purple', width=3),
                fill='tozeroy'
            ))
            
            fig_total.update_layout(
                title="Total Global Liquidity (All Currencies Combined)",
                xaxis_title="Date",
                yaxis_title="Total Liquidity (Billions USD)" if normalize_to_usd else "Total Liquidity",
                height=400
            )
            
            st.plotly_chart(fig_total, use_container_width=True)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                current_total = all_dates['Total'].iloc[-1]
                st.metric("Current Global Liquidity", f"${current_total:.0f}B" if normalize_to_usd else f"{current_total:.0f}B")
            
            with col2:
                lookback_30 = min(30, len(all_dates) - 1)
                change_30d = all_dates['Total'].iloc[-1] - all_dates['Total'].iloc[-lookback_30-1] if lookback_30 > 0 else 0
                st.metric("30-Day Change", f"{change_30d:+.0f}B")
            
            with col3:
                if len(all_dates) > 0:
                    dominant = all_dates[currencies].iloc[-1].idxmax()
                    dominant_pct = (all_dates[dominant].iloc[-1] / current_total) * 100
                    st.metric("Dominant Currency", f"{dominant} ({dominant_pct:.0f}%)")
                else:
                    st.metric("Dominant Currency", "N/A")
    
    with tab2:
        st.subheader("Individual Currency Analysis")
        
        selected_currency = st.selectbox("Select Currency for Detailed View", list(liquidity_data.keys()))
        
        if selected_currency:
            df = liquidity_data[selected_currency]
            
            # Main chart
            fig_individual = go.Figure()
            
            fig_individual.add_trace(go.Scatter(
                x=df.index,
                y=df['Liquidity'],
                mode='lines',
                name='Liquidity',
                line=dict(color=colors.get(selected_currency, '#1f77b4'), width=2),
                fill='tozeroy'
            ))
            
            fig_individual.update_layout(
                title=f"{selected_currency} Liquidity Over Time",
                xaxis_title="Date",
                yaxis_title="Liquidity (Billions)",
                height=500
            )
            
            st.plotly_chart(fig_individual, use_container_width=True)
            
            # Components breakdown
            st.subheader(f"{selected_currency} Components")
            
            fig_components = make_subplots(
                rows=3, cols=1,
                subplot_titles=(
                    f"{selected_currency} Central Bank Balance",
                    f"{selected_currency} Reserves/RRP",
                    f"{selected_currency} Treasury Account" if selected_currency == 'USD' else "Other"
                ),
                vertical_spacing=0.1
            )
            
            fig_components.add_trace(
                go.Scatter(x=df.index, y=df['Central_Bank_Balance'], 
                          name='Balance Sheet', line=dict(color='green')),
                row=1, col=1
            )
            
            fig_components.add_trace(
                go.Scatter(x=df.index, y=df['RRP_or_Reserves'], 
                          name='Reserves/RRP', line=dict(color='red')),
                row=2, col=1
            )
            
            if selected_currency == 'USD':
                fig_components.add_trace(
                    go.Scatter(x=df.index, y=df['Treasury_Account'], 
                              name='TGA', line=dict(color='orange')),
                    row=3, col=1
                )
            
            fig_components.update_layout(height=800, showlegend=False)
            st.plotly_chart(fig_components, use_container_width=True)
            
            # Statistics table
            st.subheader("Statistics")
            
            # Safe lookbacks
            lookback_30 = min(30, len(df) - 1)
            lookback_90 = min(90, len(df) - 1)
            
            recent_30 = df.tail(lookback_30 + 1) if lookback_30 > 0 else df
            recent_90 = df.tail(lookback_90 + 1) if lookback_90 > 0 else df
            
            # Calculate changes safely
            change_30d = df['Liquidity'].iloc[-1] - df['Liquidity'].iloc[-lookback_30-1] if lookback_30 > 0 else 0
            change_90d = df['Liquidity'].iloc[-1] - df['Liquidity'].iloc[-lookback_90-1] if lookback_90 > 0 else 0
            
            # YTD change
            ytd_data = df.loc[df.index.year == df.index[-1].year]
            ytd_change = df['Liquidity'].iloc[-1] - ytd_data['Liquidity'].iloc[0] if len(ytd_data) > 0 else 0
            
            stats_df = pd.DataFrame({
                'Metric': [
                    'Current Value',
                    '30-Day Change',
                    '90-Day Change',
                    '30-Day Volatility',
                    'YTD Change',
                    'Peak (2020+)',
                    'Trough (2020+)'
                ],
                'Value': [
                    f"{df['Liquidity'].iloc[-1]:.1f}B",
                    f"{change_30d:+.1f}B" if lookback_30 > 0 else "N/A",
                    f"{change_90d:+.1f}B" if lookback_90 > 0 else "N/A",
                    f"{recent_30['Liquidity'].std():.1f}B" if len(recent_30) > 1 else "N/A",
                    f"{ytd_change:+.1f}B" if len(ytd_data) > 0 else "N/A",
                    f"{df['Liquidity'].max():.1f}B",
                    f"{df['Liquidity'].min():.1f}B"
                ]
            })
            
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("Liquidity Forecasts")
        
        forecast_currency = st.selectbox("Select Currency to Forecast", list(liquidity_data.keys()), key='forecast_select')
        
        if forecast_currency:
            df = liquidity_data[forecast_currency]
            forecast = create_simple_forecast(df, forecast_days, forecast_currency)
            
            if forecast is not None:
                fig_forecast = go.Figure()
                
                # Historical
                fig_forecast.add_trace(go.Scatter(
                    x=df.tail(90).index,
                    y=df.tail(90)['Liquidity'],
                    mode='lines',
                    name='Historical',
                    line=dict(color=colors.get(forecast_currency, '#1f77b4'), width=2)
                ))
                
                # Forecast
                fig_forecast.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='orange', width=2, dash='dash')
                ))
                
                # Confidence intervals
                fig_forecast.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat_upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig_forecast.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat_lower'],
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(255, 127, 14, 0.2)',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig_forecast.update_layout(
                    title=f"{forecast_currency} Liquidity Forecast ({forecast_days} days)",
                    xaxis_title="Date",
                    yaxis_title="Liquidity (Billions)",
                    height=500
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Forecast stats
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    forecast_end = forecast.iloc[-1]['yhat']
                    current = df['Liquidity'].iloc[-1]
                    change = forecast_end - current
                    st.metric(
                        f"Forecast ({forecast_days}d)",
                        f"{forecast_end:.0f}B",
                        f"{change:+.0f}B"
                    )
                
                with col2:
                    st.metric("Upper Bound", f"{forecast.iloc[-1]['yhat_upper']:.0f}B")
                
                with col3:
                    st.metric("Lower Bound", f"{forecast.iloc[-1]['yhat_lower']:.0f}B")
        
        # Multi-currency forecast comparison
        st.subheader("Multi-Currency Forecast Comparison")
        
        fig_multi_forecast = go.Figure()
        
        for curr, df in liquidity_data.items():
            forecast = create_simple_forecast(df, forecast_days, curr)
            if forecast is not None:
                fig_multi_forecast.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    mode='lines',
                    name=f'{curr} Forecast',
                    line=dict(color=colors.get(curr, '#333333'), width=2, dash='dash')
                ))
        
        fig_multi_forecast.update_layout(
            title="Multi-Currency Forecast Comparison",
            xaxis_title="Date",
            yaxis_title="Forecasted Liquidity (Billions)",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_multi_forecast, use_container_width=True)
    
    with tab4:
        st.subheader("Cross-Currency Correlation Analysis")
        
        if len(liquidity_data) > 1:
            # Create correlation matrix
            corr_df = pd.DataFrame()
            for currency, df in liquidity_data.items():
                corr_df[currency] = df['Liquidity']
            
            corr_df = corr_df.ffill().bfill()
            correlation_matrix = corr_df.corr()
            
            # Heatmap
            fig_corr = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdBu',
                zmid=0,
                text=correlation_matrix.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 14},
                colorbar=dict(title="Correlation")
            ))
            
            fig_corr.update_layout(
                title="Liquidity Correlation Matrix",
                height=500
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Rolling correlation
            st.subheader("Rolling 30-Day Correlation")
            
            if len(currencies) >= 2:
                curr1 = st.selectbox("Currency 1", currencies, index=0)
                curr2 = st.selectbox("Currency 2", currencies, index=1 if len(currencies) > 1 else 0)
                
                if curr1 != curr2 and curr1 in liquidity_data and curr2 in liquidity_data:
                    df1 = liquidity_data[curr1]['Liquidity']
                    df2 = liquidity_data[curr2]['Liquidity']
                    
                    # Align and calculate rolling correlation
                    combined = pd.DataFrame({curr1: df1, curr2: df2})
                    combined = combined.ffill()
                    rolling_corr = combined[curr1].rolling(30).corr(combined[curr2])
                    
                    fig_roll = go.Figure()
                    fig_roll.add_trace(go.Scatter(
                        x=rolling_corr.index,
                        y=rolling_corr.values,
                        mode='lines',
                        name=f'{curr1} vs {curr2}',
                        line=dict(color='purple', width=2)
                    ))
                    
                    fig_roll.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_roll.add_hline(y=0.5, line_dash="dot", line_color="green", annotation_text="Strong positive")
                    fig_roll.add_hline(y=-0.5, line_dash="dot", line_color="red", annotation_text="Strong negative")
                    
                    fig_roll.update_layout(
                        title=f"30-Day Rolling Correlation: {curr1} vs {curr2}",
                        xaxis_title="Date",
                        yaxis_title="Correlation",
                        height=400
                    )
                    
                    st.plotly_chart(fig_roll, use_container_width=True)
        else:
            st.info("Select at least 2 currencies to view correlation analysis")
    
    with tab5:
        st.subheader("🎯 Multi-Currency Trading Signals")
        
        signals_df = []
        
        for currency, df in liquidity_data.items():
            if len(df) < 2:
                # Skip if not enough data
                continue
                
            latest = df.iloc[-1]
            
            # Safe lookbacks
            lookback_30 = min(30, len(df) - 1)
            lookback_90 = min(90, len(df) - 1)
            
            # Calculate signals with safe indexing
            ma_30 = df['Liquidity'].tail(lookback_30 + 1).mean() if lookback_30 > 0 else latest['Liquidity']
            ma_90 = df['Liquidity'].tail(lookback_90 + 1).mean() if lookback_90 > 0 else latest['Liquidity']
            
            momentum = df['Liquidity'].iloc[-1] - df['Liquidity'].iloc[-lookback_30-1] if lookback_30 > 0 else 0
            momentum_pct = (momentum / df['Liquidity'].iloc[-lookback_30-1]) * 100 if lookback_30 > 0 and df['Liquidity'].iloc[-lookback_30-1] != 0 else 0
            
            # Determine regime
            if latest['Liquidity'] > ma_90 and momentum > 0:
                regime = "🟢 Bullish"
                score = 2
            elif latest['Liquidity'] > ma_90:
                regime = "🟡 Neutral+"
                score = 1
            elif momentum > 0:
                regime = "🟡 Neutral"
                score = 0
            else:
                regime = "🔴 Bearish"
                score = -1
            
            # Data quality indicator
            quality = df['data_quality'].iloc[0] if 'data_quality' in df.columns else 'Unknown'
            if quality == 'Excellent':
                quality_badge = "🟢"
            elif quality == 'Good':
                quality_badge = "🟡"
            elif 'Fair' in str(quality):
                quality_badge = "🟠"
            else:
                quality_badge = "🔴"
            
            signals_df.append({
                'Currency': f"{currency} {quality_badge}",
                'Current': f"{latest['Liquidity']:.0f}B",
                '30-Day Change': f"{momentum:+.0f}B ({momentum_pct:+.1f}%)",
                'vs 90-Day MA': f"{latest['Liquidity'] - ma_90:+.0f}B",
                'Regime': regime,
                'Score': score
            })
        
        signals_table = pd.DataFrame(signals_df)
        st.dataframe(signals_table, use_container_width=True, hide_index=True)
        
        # Global signal
        st.subheader("🌍 Global Liquidity Signal")
        
        avg_score = signals_table['Score'].mean()
        
        if avg_score >= 1:
            st.success("🚀 **GLOBALLY BULLISH** - Most major currencies showing expansionary liquidity")
        elif avg_score >= 0:
            st.info("📊 **MIXED/NEUTRAL** - Divergent liquidity conditions across currencies")
        else:
            st.error("🔻 **GLOBALLY BEARISH** - Contracting liquidity across major currencies")
        
        # Key insights
        st.markdown("### Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Strongest Liquidity:**")
            strongest = signals_table.loc[signals_table['Score'].idxmax()]
            st.write(f"- {strongest['Currency']}: {strongest['Regime']}")
            
            st.markdown("**Best Momentum:**")
            momentum_values = [float(x.split('(')[0].replace('B','').replace('+','').replace(' ','')) for x in signals_table['30-Day Change']]
            best_momentum_idx = momentum_values.index(max(momentum_values))
            st.write(f"- {signals_table.iloc[best_momentum_idx]['Currency']}: {signals_table.iloc[best_momentum_idx]['30-Day Change']}")
        
        with col2:
            st.markdown("**Weakest Liquidity:**")
            weakest = signals_table.loc[signals_table['Score'].idxmin()]
            st.write(f"- {weakest['Currency']}: {weakest['Regime']}")
            
            st.markdown("**Worst Momentum:**")
            worst_momentum_idx = momentum_values.index(min(momentum_values))
            st.write(f"- {signals_table.iloc[worst_momentum_idx]['Currency']}: {signals_table.iloc[worst_momentum_idx]['30-Day Change']}")
    
    # Download data
    st.sidebar.markdown("---")
    st.sidebar.header("Export Data")
    
    # Combine all data for export
    export_df = pd.DataFrame()
    for currency, df in liquidity_data.items():
        df_copy = df.copy()
        df_copy['Currency'] = currency
        export_df = pd.concat([export_df, df_copy])
    
    csv = export_df.to_csv()
    st.sidebar.download_button(
        label="Download All Data (CSV)",
        data=csv,
        file_name=f"global_liquidity_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

else:
    st.error("Unable to fetch liquidity data. Please check your API key and try again.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Sources:**")
st.sidebar.markdown("""
- 🇺🇸 **USD**: Federal Reserve (WALCL, RRPONTSYD, WTREGEN) - Real-time
- 🇯🇵 **JPY**: Bank of Japan (JPNASSETS, BOGMBASE) - Monthly
- 🇬🇧 **GBP**: Bank of England (Multiple sources) - Use caution
- 🇨🇦 **CAD**: Bank of Canada API (V122711) - Weekly

**FX Rates**: Live from FRED (DEXJPUS, DEXUSUK, DEXCAUS)
""")

st.sidebar.markdown("---")
st.sidebar.info("""
**💡 Recommendation:**

- ✅ **Trust USD most** - Best data quality
- ✅ **JPY good for trends** - Reliable but monthly lag
- ⚠️ **GBP/CAD use directionally** - Limited data availability

For professional trading, consider Bloomberg Terminal for GBP/CAD data.
""")