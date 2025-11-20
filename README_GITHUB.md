# 💵 USD Liquidity Dashboard

Real-time analysis and forecasting of USD liquidity conditions for macro traders and investors.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## 📊 What is USD Liquidity?

**USD Liquidity = Fed Balance Sheet - Reverse Repo (RRP) - Treasury General Account (TGA)**

This represents the amount of money available in the financial system for investing. Rising liquidity typically supports risk assets, while falling liquidity creates headwinds.

## ✨ Features

- 📈 **Real-time Data** from Federal Reserve Economic Data (FRED)
- 🔮 **Smart Forecasting** with Treasury operations modeling
- 🎯 **Trading Signals** - Automated bullish/bearish regime detection
- 📊 **Component Analysis** - Track Fed Balance, RRP, and TGA
- 💾 **Data Export** - Download historical data as CSV

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements_v2.txt

# Run the dashboard
python -m streamlit run usd_liquidity_dashboard_v2.py
```

### Get FRED API Key

1. Go to [FRED website](https://fred.stlouisfed.org/)
2. Create a free account
3. Request an API key at [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)

## 🌐 Deploy to Web

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions on deploying to:
- Streamlit Cloud (recommended, free)
- Render
- Railway
- Self-hosted options

## 📚 Documentation

- **[README_v2.md](README_v2.md)** - Complete feature documentation
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Step-by-step deployment
- **[TREASURY_OPERATIONS_GUIDE.md](TREASURY_OPERATIONS_GUIDE.md)** - Understanding Treasury operations

## 🎯 For Traders

The dashboard incorporates:
- ✅ Quantitative Tightening (QT) schedules
- ✅ RRP drain dynamics
- ✅ TGA mean reversion patterns
- ✅ Quarterly debt issuance
- ✅ Interest payment dates
- ✅ Debt ceiling scenarios

## 📸 Screenshots

### Historical Analysis
View complete liquidity history with component breakdowns.

### Forecast with Treasury Operations
Model future liquidity with customizable Fed and Treasury parameters.

### Trading Signals
Automated regime detection and actionable signals.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional data sources (ECB, BoJ balance sheets)
- Correlation with asset prices (S&P 500, BTC, Gold)
- Alert system for regime changes
- API endpoints for algorithmic trading

## 📄 License

MIT License - feel free to use and modify for your needs.

## ⚠️ Disclaimer

This tool is for informational and educational purposes only. It is not financial advice. Always conduct your own research and implement proper risk management.

## 🙏 Acknowledgments

- Data from [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/)
- Built with [Streamlit](https://streamlit.io/)
- Visualization with [Plotly](https://plotly.com/)

---

**Star ⭐ this repo if you find it useful!**
