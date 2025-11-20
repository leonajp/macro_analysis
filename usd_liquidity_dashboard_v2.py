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
    page_title="USD Liquidity Dashboard",
    page_icon="💵",
    layout="wide"
)

# Title
st.title("💵 USD Liquidity Analysis & Forecast")
st.markdown("**Formula:** USD Liquidity = Fed Balance Sheet - Reverse Repo (RRP) - Treasury General Account (TGA)")

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

# Forecast parameters
st.sidebar.header("Forecast Settings")
forecast_days = st.sidebar.slider("Forecast Days", 30, 365, 90)
forecast_method = st.sidebar.selectbox(
    "Forecast Method",
    ["Enhanced (Treasury Operations)", "Simple Trend", "Regression"]
)

# Treasury Operations (Manual Inputs)
st.sidebar.header("Treasury Operations")
st.sidebar.markdown("_Adjust expected changes:_")

with st.sidebar.expander("Fed Balance Sheet"):
    fed_qt_monthly = st.number_input(
        "QT (Monthly Reduction, $B)", 
        value=-60.0, 
        step=10.0,
        help="Quantitative Tightening pace"
    )
    fed_adjust = st.number_input(
        "One-time Adjustment ($B)", 
        value=0.0, 
        step=50.0
    )

with st.sidebar.expander("TGA (Treasury Account)"):
    tga_target = st.number_input(
        "Target TGA Level ($B)", 
        value=750.0, 
        step=50.0,
        help="Treasury typically maintains ~$750B"
    )
    debt_ceiling_event = st.checkbox("Debt Ceiling Event Expected")
    if debt_ceiling_event:
        debt_ceiling_date = st.date_input("Event Date", value=datetime.now() + timedelta(days=60))

with st.sidebar.expander("RRP Dynamics"):
    rrp_drain_rate = st.slider(
        "RRP Drain Rate (%/month)", 
        -20.0, 20.0, -5.0, 0.5,
        help="Expected monthly change in RRP"
    )

with st.sidebar.expander("Treasury Issuance"):
    quarterly_issuance = st.number_input(
        "Quarterly Net Issuance ($B)", 
        value=500.0, 
        step=100.0,
        help="Net new debt issuance per quarter"
    )
    quarterly_interest = st.number_input(
        "Quarterly Interest Payment ($B)", 
        value=250.0, 
        step=50.0,
        help="Interest payments drain TGA"
    )

@st.cache_data(ttl=3600)
def fetch_fred_data(api_key):
    """Fetch data from FRED API"""
    try:
        fred = Fred(api_key=api_key)
        
        st.info("Fetching data from Federal Reserve...")
        
        fed_balance = fred.get_series('WALCL', observation_start='2020-01-01')
        rrp = fred.get_series('RRPONTSYD', observation_start='2020-01-01')
        tga = fred.get_series('WTREGEN', observation_start='2020-01-01')
        
        # Create DataFrame
        df = pd.DataFrame({
            'Fed_Balance': fed_balance,
            'RRP': rrp,
            'TGA': tga
        })
        
        # Forward fill missing values
        df = df.fillna(method='ffill')
        
        # Calculate USD Liquidity
        df['USD_Liquidity'] = df['Fed_Balance'] - df['RRP'] - df['TGA']
        
        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        return df
    
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def create_enhanced_forecast(df, days, fed_qt_monthly, fed_adjust, tga_target, 
                            rrp_drain_rate, quarterly_issuance, quarterly_interest,
                            debt_ceiling_event=False, debt_ceiling_date=None):
    """
    Enhanced forecast incorporating Treasury operations
    """
    try:
        # Get current values
        current_fed = df['Fed_Balance'].iloc[-1]
        current_rrp = df['RRP'].iloc[-1]
        current_tga = df['TGA'].iloc[-1]
        current_date = df.index[-1]
        
        # Calculate historical trends
        recent_30d = df.tail(30)
        fed_trend = (recent_30d['Fed_Balance'].iloc[-1] - recent_30d['Fed_Balance'].iloc[0]) / 30
        rrp_trend = (recent_30d['RRP'].iloc[-1] - recent_30d['RRP'].iloc[0]) / 30
        tga_volatility = recent_30d['TGA'].std()
        
        # Create forecast dates
        forecast_dates = pd.date_range(start=current_date + timedelta(days=1), periods=days, freq='D')
        
        # Initialize forecast arrays
        fed_forecast = np.zeros(days)
        rrp_forecast = np.zeros(days)
        tga_forecast = np.zeros(days)
        
        for i in range(days):
            day = i + 1
            
            # Fed Balance Sheet: QT + one-time adjustments
            monthly_reduction = (fed_qt_monthly / 30) * day
            fed_forecast[i] = current_fed + monthly_reduction + fed_adjust
            
            # RRP: Exponential drain based on rate
            monthly_factor = (1 + rrp_drain_rate / 100)
            rrp_forecast[i] = current_rrp * (monthly_factor ** (day / 30))
            rrp_forecast[i] = max(rrp_forecast[i], 0)  # Can't go negative
            
            # TGA: Mean reversion to target with quarterly ops
            current_quarter = (day // 90) + 1
            
            # Interest payments (drain TGA) - typically around mid-Feb, May, Aug, Nov
            interest_impact = 0
            day_of_quarter = day % 90
            if 40 <= day_of_quarter <= 50:  # Mid-quarter interest payment
                interest_impact = -quarterly_interest * (day_of_quarter - 40) / 10
            
            # Debt issuance (increases TGA) - spread throughout quarter
            issuance_impact = (quarterly_issuance / 90) * (day % 90)
            
            # Mean reversion to target
            tga_gap = tga_target - current_tga
            mean_reversion = tga_gap * (1 - np.exp(-day / 60))  # 60-day half-life
            
            # Add seasonal volatility
            seasonal = tga_volatility * np.sin(2 * np.pi * day / 365) * 0.3
            
            tga_forecast[i] = current_tga + mean_reversion + interest_impact + issuance_impact + seasonal
            
            # Debt ceiling impact
            if debt_ceiling_event and debt_ceiling_date:
                days_to_ceiling = (debt_ceiling_date - current_date).days
                if day >= days_to_ceiling:
                    # TGA drains rapidly approaching zero
                    drain_factor = min(1.0, (day - days_to_ceiling) / 30)
                    tga_forecast[i] = tga_forecast[i] * (1 - drain_factor * 0.8)
        
        # Calculate liquidity forecast
        liquidity_forecast = fed_forecast - rrp_forecast - tga_forecast
        
        # Calculate confidence intervals based on historical volatility
        liquidity_std = df['USD_Liquidity'].tail(90).std()
        upper_bound = liquidity_forecast + 1.96 * liquidity_std * np.sqrt(np.arange(1, days + 1) / 30)
        lower_bound = liquidity_forecast - 1.96 * liquidity_std * np.sqrt(np.arange(1, days + 1) / 30)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'ds': forecast_dates,
            'Fed_Balance': fed_forecast,
            'RRP': rrp_forecast,
            'TGA': tga_forecast,
            'yhat': liquidity_forecast,
            'yhat_upper': upper_bound,
            'yhat_lower': lower_bound
        })
        
        return forecast_df
    
    except Exception as e:
        st.error(f"Error creating forecast: {str(e)}")
        return None

def create_simple_forecast(df, days):
    """Simple trend-based forecast"""
    try:
        # Linear trend for each component
        recent_data = df.tail(90)
        x = np.arange(len(recent_data))
        
        # Fit trends
        fed_slope = np.polyfit(x, recent_data['Fed_Balance'].values, 1)[0]
        rrp_slope = np.polyfit(x, recent_data['RRP'].values, 1)[0]
        tga_slope = np.polyfit(x, recent_data['TGA'].values, 1)[0]
        
        # Project forward
        current_date = df.index[-1]
        forecast_dates = pd.date_range(start=current_date + timedelta(days=1), periods=days, freq='D')
        
        fed_forecast = df['Fed_Balance'].iloc[-1] + fed_slope * np.arange(1, days + 1)
        rrp_forecast = df['RRP'].iloc[-1] + rrp_slope * np.arange(1, days + 1)
        tga_forecast = df['TGA'].iloc[-1] + tga_slope * np.arange(1, days + 1)
        
        liquidity_forecast = fed_forecast - rrp_forecast - tga_forecast
        
        # Simple confidence intervals
        std = df['USD_Liquidity'].tail(90).std()
        upper = liquidity_forecast + 1.5 * std
        lower = liquidity_forecast - 1.5 * std
        
        return pd.DataFrame({
            'ds': forecast_dates,
            'yhat': liquidity_forecast,
            'yhat_upper': upper,
            'yhat_lower': lower,
            'Fed_Balance': fed_forecast,
            'RRP': rrp_forecast,
            'TGA': tga_forecast
        })
    
    except Exception as e:
        st.error(f"Error creating forecast: {str(e)}")
        return None

# Fetch data
df = fetch_fred_data(api_key)

if df is not None:
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    latest = df.iloc[-1]
    prev = df.iloc[-30] if len(df) > 30 else df.iloc[0]
    
    with col1:
        st.metric(
            "Current USD Liquidity",
            f"${latest['USD_Liquidity']:.1f}B",
            f"{latest['USD_Liquidity'] - prev['USD_Liquidity']:.1f}B (30d)"
        )
    
    with col2:
        st.metric(
            "Fed Balance Sheet",
            f"${latest['Fed_Balance']:.1f}B",
            f"{latest['Fed_Balance'] - prev['Fed_Balance']:.1f}B (30d)"
        )
    
    with col3:
        st.metric(
            "Reverse Repo (RRP)",
            f"${latest['RRP']:.1f}B",
            f"{latest['RRP'] - prev['RRP']:.1f}B (30d)"
        )
    
    with col4:
        st.metric(
            "Treasury General Account",
            f"${latest['TGA']:.1f}B",
            f"{latest['TGA'] - prev['TGA']:.1f}B (30d)"
        )
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Historical Data", "🔮 Forecast", "📈 Components", "🎯 Trading Signals"])
    
    with tab1:
        st.subheader("Historical USD Liquidity")
        
        # Create interactive plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['USD_Liquidity'],
            mode='lines',
            name='USD Liquidity',
            line=dict(color='#1f77b4', width=2),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.2)'
        ))
        
        fig.update_layout(
            title="USD Liquidity Over Time",
            xaxis_title="Date",
            yaxis_title="Liquidity (Billions USD)",
            hovermode='x unified',
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Components breakdown
        st.subheader("Components Breakdown")
        
        fig2 = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Fed Balance Sheet", "Reverse Repo (RRP)", "Treasury General Account (TGA)"),
            vertical_spacing=0.1
        )
        
        fig2.add_trace(
            go.Scatter(x=df.index, y=df['Fed_Balance'], name='Fed Balance', line=dict(color='green')),
            row=1, col=1
        )
        
        fig2.add_trace(
            go.Scatter(x=df.index, y=df['RRP'], name='RRP', line=dict(color='red')),
            row=2, col=1
        )
        
        fig2.add_trace(
            go.Scatter(x=df.index, y=df['TGA'], name='TGA', line=dict(color='orange')),
            row=3, col=1
        )
        
        fig2.update_layout(height=800, showlegend=False)
        fig2.update_yaxes(title_text="Billions USD", row=1, col=1)
        fig2.update_yaxes(title_text="Billions USD", row=2, col=1)
        fig2.update_yaxes(title_text="Billions USD", row=3, col=1)
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.subheader(f"USD Liquidity Forecast ({forecast_days} days)")
        
        # Create forecast based on selected method
        if forecast_method == "Enhanced (Treasury Operations)":
            dc_date = None
            if debt_ceiling_event:
                dc_date = datetime.combine(debt_ceiling_date, datetime.min.time())
            
            forecast = create_enhanced_forecast(
                df, forecast_days, fed_qt_monthly, fed_adjust, tga_target,
                rrp_drain_rate, quarterly_issuance, quarterly_interest,
                debt_ceiling_event, dc_date
            )
            
            st.info("📌 **Enhanced Forecast** incorporates: QT schedule, RRP dynamics, TGA mean reversion, debt issuance, and interest payments")
        else:
            forecast = create_simple_forecast(df, forecast_days)
            st.info("📌 **Simple Forecast** based on recent linear trends")
        
        if forecast is not None:
            # Create forecast plot
            fig3 = go.Figure()
            
            # Historical data
            fig3.add_trace(go.Scatter(
                x=df.index,
                y=df['USD_Liquidity'],
                mode='lines',
                name='Historical',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Forecast
            fig3.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color='#ff7f0e', width=2, dash='dash')
            ))
            
            # Confidence intervals
            fig3.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_upper'],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig3.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_lower'],
                mode='lines',
                name='Lower Bound',
                fill='tonexty',
                fillcolor='rgba(255, 127, 14, 0.2)',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig3.update_layout(
                title="USD Liquidity Forecast with Confidence Intervals",
                xaxis_title="Date",
                yaxis_title="Liquidity (Billions USD)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig3, use_container_width=True)
            
            # Component forecasts
            if 'Fed_Balance' in forecast.columns:
                st.subheader("Component Forecasts")
                
                fig_comp = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=("Fed Balance Sheet Projection", "RRP Projection", "TGA Projection"),
                    vertical_spacing=0.1
                )
                
                # Fed Balance
                fig_comp.add_trace(
                    go.Scatter(x=df.tail(90).index, y=df.tail(90)['Fed_Balance'], 
                              name='Historical', line=dict(color='green')),
                    row=1, col=1
                )
                fig_comp.add_trace(
                    go.Scatter(x=forecast['ds'], y=forecast['Fed_Balance'], 
                              name='Forecast', line=dict(color='green', dash='dash')),
                    row=1, col=1
                )
                
                # RRP
                fig_comp.add_trace(
                    go.Scatter(x=df.tail(90).index, y=df.tail(90)['RRP'], 
                              name='Historical', line=dict(color='red')),
                    row=2, col=1
                )
                fig_comp.add_trace(
                    go.Scatter(x=forecast['ds'], y=forecast['RRP'], 
                              name='Forecast', line=dict(color='red', dash='dash')),
                    row=2, col=1
                )
                
                # TGA
                fig_comp.add_trace(
                    go.Scatter(x=df.tail(90).index, y=df.tail(90)['TGA'], 
                              name='Historical', line=dict(color='orange')),
                    row=3, col=1
                )
                fig_comp.add_trace(
                    go.Scatter(x=forecast['ds'], y=forecast['TGA'], 
                              name='Forecast', line=dict(color='orange', dash='dash')),
                    row=3, col=1
                )
                
                fig_comp.update_layout(height=800, showlegend=False)
                fig_comp.update_yaxes(title_text="Billions USD", row=1, col=1)
                fig_comp.update_yaxes(title_text="Billions USD", row=2, col=1)
                fig_comp.update_yaxes(title_text="Billions USD", row=3, col=1)
                
                st.plotly_chart(fig_comp, use_container_width=True)
            
            # Forecast statistics
            st.subheader("Forecast Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            forecast_end = forecast.iloc[-1]
            change = forecast_end['yhat'] - latest['USD_Liquidity']
            change_pct = (change / latest['USD_Liquidity']) * 100
            
            with col1:
                st.metric(
                    f"Forecast ({forecast_days}d)",
                    f"${forecast_end['yhat']:.1f}B",
                    f"{change:.1f}B ({change_pct:+.1f}%)"
                )
            
            with col2:
                st.metric(
                    "Upper Bound",
                    f"${forecast_end['yhat_upper']:.1f}B"
                )
            
            with col3:
                st.metric(
                    "Lower Bound",
                    f"${forecast_end['yhat_lower']:.1f}B"
                )
    
    with tab3:
        st.subheader("Component Analysis")
        
        # Calculate correlations
        corr = df[['Fed_Balance', 'RRP', 'TGA', 'USD_Liquidity']].corr()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Recent Trends (30 days)**")
            recent = df.tail(30)
            
            trends = pd.DataFrame({
                'Component': ['Fed Balance', 'RRP', 'TGA', 'USD Liquidity'],
                'Change': [
                    recent['Fed_Balance'].iloc[-1] - recent['Fed_Balance'].iloc[0],
                    recent['RRP'].iloc[-1] - recent['RRP'].iloc[0],
                    recent['TGA'].iloc[-1] - recent['TGA'].iloc[0],
                    recent['USD_Liquidity'].iloc[-1] - recent['USD_Liquidity'].iloc[0]
                ],
                '% Change': [
                    ((recent['Fed_Balance'].iloc[-1] - recent['Fed_Balance'].iloc[0]) / recent['Fed_Balance'].iloc[0]) * 100,
                    ((recent['RRP'].iloc[-1] - recent['RRP'].iloc[0]) / recent['RRP'].iloc[0]) * 100 if recent['RRP'].iloc[0] > 0 else 0,
                    ((recent['TGA'].iloc[-1] - recent['TGA'].iloc[0]) / recent['TGA'].iloc[0]) * 100 if recent['TGA'].iloc[0] > 0 else 0,
                    ((recent['USD_Liquidity'].iloc[-1] - recent['USD_Liquidity'].iloc[0]) / recent['USD_Liquidity'].iloc[0]) * 100
                ]
            })
            
            st.dataframe(trends.style.format({'Change': '${:.1f}B', '% Change': '{:.2f}%'}))
        
        with col2:
            st.write("**Correlation Matrix**")
            st.dataframe(corr.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1).format('{:.2f}'))
        
        # Velocity analysis
        st.subheader("Velocity Analysis")
        df['Liquidity_Change'] = df['USD_Liquidity'].diff()
        df['Liquidity_MA7'] = df['USD_Liquidity'].rolling(7).mean()
        df['Liquidity_MA30'] = df['USD_Liquidity'].rolling(30).mean()
        
        fig_velocity = go.Figure()
        fig_velocity.add_trace(go.Scatter(
            x=df.tail(180).index, 
            y=df.tail(180)['Liquidity_Change'],
            name='Daily Change',
            line=dict(color='lightgray')
        ))
        fig_velocity.add_trace(go.Scatter(
            x=df.tail(180).index, 
            y=df.tail(180)['Liquidity_Change'].rolling(7).mean(),
            name='7-Day Average',
            line=dict(color='blue', width=2)
        ))
        fig_velocity.update_layout(
            title="Liquidity Change Velocity (Last 180 Days)",
            yaxis_title="Daily Change (Billions USD)",
            height=400
        )
        st.plotly_chart(fig_velocity, use_container_width=True)
        
    with tab4:
        st.subheader("🎯 Trading Signals")
        
        # Calculate signals
        current_liq = latest['USD_Liquidity']
        ma_30 = df['USD_Liquidity'].rolling(30).mean().iloc[-1]
        ma_90 = df['USD_Liquidity'].rolling(90).mean().iloc[-1]
        
        momentum = df['USD_Liquidity'].iloc[-1] - df['USD_Liquidity'].iloc[-30]
        momentum_pct = (momentum / df['USD_Liquidity'].iloc[-30]) * 100
        
        # Signal interpretation
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Current Regime")
            
            if current_liq > ma_90:
                regime = "🟢 **EXPANSIONARY**"
                regime_desc = "Liquidity above 90-day average. Supportive for risk assets."
            else:
                regime = "🔴 **CONTRACTIONARY**"
                regime_desc = "Liquidity below 90-day average. Risk-off environment."
            
            st.markdown(regime)
            st.write(regime_desc)
            
            # Momentum
            st.markdown("### Momentum (30-day)")
            if momentum > 0:
                st.success(f"📈 Positive: +${momentum:.1f}B ({momentum_pct:+.2f}%)")
            else:
                st.error(f"📉 Negative: ${momentum:.1f}B ({momentum_pct:+.2f}%)")
        
        with col2:
            st.markdown("### Component Signals")
            
            # RRP drain = liquidity positive
            rrp_change = latest['RRP'] - df['RRP'].iloc[-30]
            if rrp_change < -50:
                st.success("✅ RRP draining significantly (+bullish)")
            elif rrp_change < 0:
                st.info("→ RRP declining (neutral/+bullish)")
            else:
                st.warning("⚠️ RRP rising (-bearish)")
            
            # TGA
            tga_change = latest['TGA'] - df['TGA'].iloc[-30]
            if tga_change < -50:
                st.success("✅ TGA declining (+bullish)")
            elif tga_change > 50:
                st.warning("⚠️ TGA rising (-bearish)")
            else:
                st.info("→ TGA stable (neutral)")
            
            # QT
            fed_change = latest['Fed_Balance'] - df['Fed_Balance'].iloc[-30]
            if fed_change < -50:
                st.warning("⚠️ Active QT (-bearish)")
            elif fed_change > 50:
                st.success("✅ Balance sheet expanding (+bullish)")
            else:
                st.info("→ Balance sheet stable (neutral)")
        
        # Combined signal
        st.markdown("---")
        st.markdown("### 🎲 Combined Signal")
        
        signal_score = 0
        if current_liq > ma_90:
            signal_score += 2
        if momentum > 0:
            signal_score += 2
        if rrp_change < 0:
            signal_score += 1
        if tga_change < 0:
            signal_score += 1
        if fed_change > 0:
            signal_score += 1
        
        if signal_score >= 5:
            st.success("🚀 **STRONG BULLISH** - High liquidity conditions favor risk assets")
        elif signal_score >= 3:
            st.info("📊 **NEUTRAL/BULLISH** - Mixed signals, lean bullish")
        elif signal_score >= 1:
            st.warning("⚠️ **NEUTRAL/BEARISH** - Mixed signals, lean bearish")
        else:
            st.error("🔻 **BEARISH** - Contracting liquidity, risk-off environment")
        
        # Key levels
        st.markdown("### 📊 Key Levels")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current", f"${current_liq:.0f}B")
        with col2:
            st.metric("30-Day MA", f"${ma_30:.0f}B", f"{current_liq - ma_30:+.0f}B")
        with col3:
            st.metric("90-Day MA", f"${ma_90:.0f}B", f"{current_liq - ma_90:+.0f}B")
    
    # Download data
    st.sidebar.markdown("---")
    st.sidebar.header("Export Data")
    
    csv = df.to_csv()
    st.sidebar.download_button(
        label="Download Historical CSV",
        data=csv,
        file_name=f"usd_liquidity_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Sources:**")
st.sidebar.markdown("- Federal Reserve Economic Data (FRED)")
st.sidebar.markdown("- Updated weekly")