# 🚀 OnChain Macro Trends Analysis

A comprehensive framework for analyzing blockchain data to predict macroeconomic trends, integrating DeFi metrics with traditional financial indicators through network analysis and machine learning.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

### 🕸️ **Network Analysis Engine**
- **Protocol Relationship Mapping** - Analyze connections between 300+ DeFi protocols
- **Centrality Analysis** - Identify the most important protocols in the ecosystem  
- **Systemic Risk Assessment** - Evaluate ecosystem-wide vulnerabilities
- **Contagion Simulation** - Model how shocks spread through the network

### 📊 **Real-Time Data Integration**
- **DeFiLlama API** - Live TVL and protocol data
- **CoinGecko API** - Global crypto market metrics
- **Multi-Chain Support** - Ethereum, BSC, Polygon, Arbitrum, and more
- **Macro Indicators** - Traditional financial data integration

### 🎯 **Interactive Dashboards**
- **Network Visualization** - Interactive protocol relationship graphs
- **Risk Monitoring** - Real-time systemic risk scoring
- **Market Analytics** - Comprehensive DeFi ecosystem analysis
- **Data Export** - Download analysis results in CSV format

## 🖥️ Screenshots

### Network Analysis Dashboard
![Network Dashboard](docs/screenshots/network_dashboard.png)

### Enhanced Analytics Dashboard  
![Enhanced Dashboard](docs/screenshots/enhanced_dashboard.png)

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/onchain-macro-analysis.git
cd onchain-macro-analysis
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys (optional - works without keys)
```

5. **Initialize database**
```bash
python scripts/setup_database.py
```

### 🎯 Launch Dashboards

#### Basic Network Analysis
```bash
python -m streamlit run src/visualization/network_dashboard.py
```

#### Enhanced Real-Time Dashboard
```bash
python -m streamlit run src/visualization/enhanced_dashboard.py
```

Open your browser to `http://localhost:8501`

## 🧪 Testing

### Run the complete test suite
```bash
python test_pipeline.py
```

### Test network analysis specifically
```bash
python test_network_analysis.py
```

### Test real data fetching
```bash
python src/data/real_data_fetcher.py
```

## 📁 Project Structure

```
onchain-macro-analysis/
├── 📊 src/
│   ├── 🔍 analysis/
│   │   └── network_analysis/     # Network analysis algorithms
│   ├── 📡 data/
│   │   ├── collectors/           # Data collection modules
│   │   ├── processors/          # Data cleaning & feature engineering
│   │   └── storage/             # Database & caching
│   ├── 📈 visualization/
│   │   ├── network_dashboard.py     # Basic network analysis UI
│   │   └── enhanced_dashboard.py   # Advanced real-time dashboard
│   └── 🛠️ utils/                # Utilities and helpers
├── 📓 notebooks/                # Jupyter analysis notebooks
├── 🧪 tests/                   # Test suites
├── 📜 scripts/                 # Automation scripts
└── 📚 docs/                    # Documentation
```

## 🔧 Core Components

### Data Pipeline (`src/data/`)
- **Multi-source collectors** for DeFi protocols, gas prices, macro indicators
- **Intelligent data cleaning** with validation and error handling
- **Feature engineering** with 150+ technical indicators
- **Caching system** for optimal performance

### Network Analysis (`src/analysis/network_analysis/`)
- **Protocol network construction** based on TVL similarity and chain relationships
- **Centrality metrics** (degree, betweenness, closeness, eigenvector)
- **Risk assessment** using network topology analysis
- **Contagion modeling** to simulate shock propagation

### Visualization (`src/visualization/`)
- **Interactive network graphs** with Plotly and NetworkX
- **Real-time dashboards** with Streamlit
- **Professional charts** for centrality, risk, and market analysis
- **Export capabilities** for further analysis

## 📊 Supported Data Sources

### Blockchain Data
- **DeFiLlama** - Protocol TVL and metadata (no API key required)
- **Etherscan** - Gas prices and network metrics (API key recommended)
- **The Graph** - On-chain protocol interactions (API key optional)

### Traditional Finance  
- **FRED** - Federal Reserve economic data (API key required)
- **Yahoo Finance** - Stock market indices and commodities
- **CoinGecko** - Cryptocurrency market data (no API key required)

## 🎯 Use Cases

### 🏦 **Financial Analysis**
- Portfolio risk assessment across DeFi protocols
- Correlation analysis between crypto and traditional markets
- Liquidity flow analysis and capital allocation optimization

### 🔬 **Academic Research**
- DeFi ecosystem structure and evolution studies
- Systemic risk measurement in decentralized finance
- Network effects and contagion modeling

### 💼 **Investment Strategy**
- Protocol importance ranking for investment decisions
- Risk-adjusted yield farming strategies
- Market timing based on network activity

### 🏛️ **Regulatory Insight**
- Systemic risk monitoring for policy makers
- Market concentration analysis
- Cross-protocol dependency mapping

## 🔑 API Keys (Optional)

While the system works without API keys, adding them enables additional features:

```bash
# Blockchain Data
ETHERSCAN_API_KEY=your_etherscan_key
INFURA_PROJECT_ID=your_infura_key

# Economic Data  
FRED_API_KEY=your_fred_key

# Market Data
COINGECKO_API_KEY=your_coingecko_key
```

## 📈 Analysis Capabilities

### Network Metrics
- **Degree Centrality** - Number of direct connections
- **Betweenness Centrality** - Bridge importance between protocols  
- **Closeness Centrality** - Average distance to all other protocols
- **Eigenvector Centrality** - Influence based on connected protocols' importance

### Risk Indicators
- **Network Density** - Overall interconnectedness level
- **Clustering Coefficient** - Local network cohesion
- **Connected Components** - Separate network clusters
- **Risk Score** - Composite systemic risk measure (0-1 scale)

### Contagion Modeling
- **Shock Propagation** - How failures spread through the network
- **Impact Assessment** - Total value at risk from contagion
- **Critical Path Analysis** - Most vulnerable transmission routes

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [DeFiLlama](https://defillama.com/) for comprehensive DeFi data
- [CoinGecko](https://coingecko.com/) for cryptocurrency market data
- [NetworkX](https://networkx.org/) for network analysis algorithms
- [Streamlit](https://streamlit.io/) for rapid dashboard development
- [Plotly](https://plotly.com/) for interactive visualizations

## ⚠️ Disclaimer

This software is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Cryptocurrency and DeFi investments carry high risk. Always conduct your own research and consult with financial advisors before making investment decisions.
