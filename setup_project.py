#!/usr/bin/env python3
"""
项目结构自动生成脚本
运行此脚本会创建完整的 onchain-macro-analysis 项目结构
"""

import os
from pathlib import Path

def create_project_structure():
    """创建完整的项目目录结构"""
    
    # 项目根目录
    project_root = "onchain-macro-analysis"
    
    # 定义完整的目录结构
    directories = [
        # 根目录文件
        "",
        
        # 配置目录
        "config",
        
        # 源码目录
        "src",
        "src/data",
        "src/data/collectors",
        "src/data/processors", 
        "src/data/storage",
        
        # 分析模块
        "src/analysis",
        "src/analysis/network_analysis",
        "src/analysis/regression",
        "src/analysis/risk",
        
        # 可视化模块
        "src/visualization",
        
        # 工具模块
        "src/utils",
        
        # 数据目录
        "data",
        "data/raw",
        "data/processed",
        "data/external",
        
        # Jupyter notebooks
        "notebooks",
        
        # 测试目录
        "tests",
        
        # 脚本目录
        "scripts",
        
        # 文档目录
        "docs",
        
        # 日志目录
        "logs"
    ]
    
    # 定义需要创建的文件及其内容
    files = {
        # 根目录文件
        "README.md": get_readme_content(),
        "requirements.txt": get_requirements_content(),
        ".env.example": get_env_example_content(),
        ".gitignore": get_gitignore_content(),
        "LICENSE": get_license_content(),
        "docker-compose.yml": get_docker_compose_content(),
        "Dockerfile": get_dockerfile_content(),
        
        # 配置文件
        "config/__init__.py": "",
        "config/settings.py": get_settings_content(),
        "config/data_sources.yaml": get_data_sources_content(),
        
        # 源码 __init__.py 文件
        "src/__init__.py": "",
        "src/data/__init__.py": "",
        "src/data/collectors/__init__.py": "",
        "src/data/processors/__init__.py": "",
        "src/data/storage/__init__.py": "",
        "src/analysis/__init__.py": "",
        "src/analysis/network_analysis/__init__.py": "",
        "src/analysis/regression/__init__.py": "",
        "src/analysis/risk/__init__.py": "",
        "src/visualization/__init__.py": "",
        "src/utils/__init__.py": "",
        
        # 核心模块文件
        "src/utils/logger.py": get_logger_content(),
        "src/utils/constants.py": get_constants_content(),
        "src/utils/helpers.py": get_helpers_content(),
        "src/data/storage/database.py": get_database_content(),
        "src/data/collectors/base_collector.py": get_base_collector_content(),
        
        # 数据目录的 .gitkeep 文件
        "data/raw/.gitkeep": "",
        "data/processed/.gitkeep": "",
        "data/external/.gitkeep": "",
        "logs/.gitkeep": "",
        
        # 测试文件
        "tests/__init__.py": "",
        "tests/test_collectors.py": get_test_collectors_content(),
        "tests/test_processors.py": get_test_processors_content(),
        "tests/test_analysis.py": get_test_analysis_content(),
        
        # 脚本文件
        "scripts/setup_database.py": get_setup_database_content(),
        "scripts/daily_update.py": get_daily_update_content(),
        "scripts/generate_report.py": get_generate_report_content(),
        
        # Notebook 文件
        "notebooks/01_data_exploration.ipynb": get_notebook_content("Data Exploration"),
        "notebooks/02_feature_engineering.ipynb": get_notebook_content("Feature Engineering"),
        "notebooks/03_network_analysis.ipynb": get_notebook_content("Network Analysis"),
        "notebooks/04_regression_modeling.ipynb": get_notebook_content("Regression Modeling"),
        "notebooks/05_macro_integration.ipynb": get_notebook_content("Macro Integration"),
        
        # 文档文件
        "docs/api_reference.md": get_api_docs_content(),
        "docs/data_sources.md": get_data_sources_docs_content(),
        "docs/methodology.md": get_methodology_docs_content(),
    }
    
    print(f"🚀 开始创建项目结构: {project_root}")
    
    # 创建项目根目录
    Path(project_root).mkdir(exist_ok=True)
    os.chdir(project_root)
    
    # 创建所有目录
    print("📁 创建目录结构...")
    for directory in directories:
        if directory:  # 跳过空字符串（根目录）
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"   ✓ 创建目录: {directory}")
    
    # 创建所有文件
    print("📄 创建文件...")
    for file_path, content in files.items():
        file_obj = Path(file_path)
        file_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_obj, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"   ✓ 创建文件: {file_path}")
    
    # 返回上级目录
    os.chdir("..")
    
    print(f"\n🎉 项目结构创建完成！")
    print(f"📂 项目位置: ./{project_root}")
    print(f"\n🔧 下一步操作:")
    print(f"   1. cd {project_root}")
    print(f"   2. python -m venv venv")
    print(f"   3. source venv/bin/activate  (Linux/Mac) 或 venv\\Scripts\\activate (Windows)")
    print(f"   4. pip install -r requirements.txt")
    print(f"   5. cp .env.example .env")
    print(f"   6. 编辑 .env 文件，添加你的 API keys")
    print(f"   7. python scripts/setup_database.py")

def get_readme_content():
    return """# On-Chain Macro Trends Analysis

A comprehensive framework for analyzing blockchain data to predict macroeconomic trends, integrating DeFi metrics with traditional financial indicators through regression analysis and network modeling.

## 🚀 Features

- **Multi-Chain Data Collection**: Ethereum, Layer 2s, and cross-chain metrics
- **DeFi Protocol Analysis**: TVL, yield rates, and protocol interactions  
- **Network Analysis**: Protocol dependency mapping and systemic risk assessment
- **Macro Integration**: Traditional financial indicators correlation analysis
- **Predictive Modeling**: Machine learning enhanced trend forecasting
- **Real-time Dashboard**: Interactive visualization and monitoring

## 📊 Key Metrics Tracked

### DeFi Metrics
- Total Value Locked (TVL) across protocols and chains
- Gas fees and network congestion indicators
- Active addresses and transaction volumes
- Yield farming rates and liquidity flows

### Macro Indicators
- Federal Reserve economic data (interest rates, inflation)
- Stock market indices (S&P 500, NASDAQ, VIX)
- Currency exchange rates and commodity prices
- Institutional crypto adoption metrics

## 🛠 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/onchain-macro-analysis.git
cd onchain-macro-analysis
```

2. **Set up Python environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

3. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Set up database**
```bash
python scripts/setup_database.py
```

5. **Run initial data collection**
```bash
python scripts/daily_update.py
```

## 🚦 Quick Start

### Start the dashboard
```bash
streamlit run src/visualization/dashboard.py
```

### Run analysis notebooks
```bash
jupyter notebook notebooks/
```

### API server
```bash
uvicorn src.api:app --reload
```

## 📈 Analysis Modules

### 1. Data Collection (`src/data/`)
- Multi-source API integrations
- Real-time WebSocket connections
- Data validation and cleaning

### 2. Network Analysis (`src/analysis/network_analysis/`)
- Protocol dependency graphs
- Centrality and risk metrics
- Contagion modeling

### 3. Regression Models (`src/analysis/regression/`)
- Time series forecasting
- Multivariate analysis
- Machine learning integration

### 4. Risk Assessment (`src/analysis/risk/`)
- Systemic risk indicators
- Scenario analysis
- Stress testing

## 📚 Documentation

- [API Reference](docs/api_reference.md)
- [Data Sources](docs/data_sources.md)
- [Methodology](docs/methodology.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Cryptocurrency and DeFi investments carry high risk.
"""

def get_requirements_content():
    return """# Web3 and Blockchain
web3==6.15.1
etherscan-python==2.2.0
python-decouple==3.8

# Data Collection
requests==2.31.0
aiohttp==3.9.3
websockets==12.0
yfinance==0.2.18
fredapi==0.5.1
alpha-vantage==2.3.1

# Data Processing
pandas==2.2.0
numpy==1.26.3
polars==0.20.6
pyarrow==15.0.0

# Database
psycopg2-binary==2.9.9
sqlalchemy==2.0.25
redis==5.0.1
alembic==1.13.1

# Machine Learning
scikit-learn==1.4.0
statsmodels==0.14.1
xgboost==2.0.3
lightgbm==4.3.0
prophet==1.1.5
pmdarima==2.0.4

# Network Analysis
networkx==3.2.1
python-igraph==0.11.4
community==1.0.0b1

# Visualization
streamlit==1.30.0
plotly==5.18.0
dash==2.16.1
seaborn==0.13.2
matplotlib==3.8.2

# API Framework
fastapi==0.109.0
uvicorn==0.27.0
pydantic==2.5.3

# Task Queue
celery==5.3.6

# Development
pytest==8.0.0
black==24.1.1
flake8==7.0.0
mypy==1.8.0
pre-commit==3.6.0
jupyter==1.0.0

# Monitoring
prometheus-client==0.19.0
sentry-sdk==1.40.0
"""

def get_env_example_content():
    return """# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/onchain_analysis
REDIS_URL=redis://localhost:6379/0

# Blockchain APIs
INFURA_PROJECT_ID=your_infura_project_id
ALCHEMY_API_KEY=your_alchemy_api_key
ETHERSCAN_API_KEY=your_etherscan_api_key
THE_GRAPH_API_KEY=your_graph_api_key

# Financial Data APIs
FRED_API_KEY=your_fred_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
COINMARKETCAP_API_KEY=your_cmc_api_key
COINGECKO_API_KEY=your_coingecko_api_key

# Configuration
LOG_LEVEL=INFO
ENVIRONMENT=development
DEBUG=True

# External Services
SENTRY_DSN=your_sentry_dsn
PROMETHEUS_PORT=8000
"""

def get_gitignore_content():
    return """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Logs
logs/*.log
!logs/.gitkeep

# Jupyter
.ipynb_checkpoints

# Cache
.cache/
.pytest_cache/

# Documentation
docs/_build/

# Database
*.db
*.sqlite3

# Docker
.dockerignore
"""

def get_license_content():
    return """MIT License

Copyright (c) 2025 On-Chain Macro Analysis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

def get_docker_compose_content():
    return """version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: onchain_analysis
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  app:
    build: .
    depends_on:
      - postgres
      - redis
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/onchain_analysis
      - REDIS_URL=redis://redis:6379/0
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs

volumes:
  postgres_data:
"""

def get_dockerfile_content():
    return """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed logs

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
"""

def get_settings_content():
    return """import os
from typing import Dict, Any
from decouple import config
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    # Environment
    ENVIRONMENT: str = config('ENVIRONMENT', default='development')
    DEBUG: bool = config('DEBUG', default=True, cast=bool)
    LOG_LEVEL: str = config('LOG_LEVEL', default='INFO')
    
    # Database
    DATABASE_URL: str = config('DATABASE_URL', default='sqlite:///./data/onchain_analysis.db')
    REDIS_URL: str = config('REDIS_URL', default='redis://localhost:6379/0')
    
    # API Keys
    INFURA_PROJECT_ID: str = config('INFURA_PROJECT_ID', default='')
    ALCHEMY_API_KEY: str = config('ALCHEMY_API_KEY', default='')
    ETHERSCAN_API_KEY: str = config('ETHERSCAN_API_KEY', default='')
    THE_GRAPH_API_KEY: str = config('THE_GRAPH_API_KEY', default='')
    
    FRED_API_KEY: str = config('FRED_API_KEY', default='')
    ALPHA_VANTAGE_API_KEY: str = config('ALPHA_VANTAGE_API_KEY', default='')
    COINMARKETCAP_API_KEY: str = config('COINMARKETCAP_API_KEY', default='')
    COINGECKO_API_KEY: str = config('COINGECKO_API_KEY', default='')
    
    # Paths
    DATA_DIR: Path = BASE_DIR / 'data'
    LOGS_DIR: Path = BASE_DIR / 'logs'
    
    def __init__(self):
        # Create directories if they don't exist
        self.DATA_DIR.mkdir(exist_ok=True)
        self.LOGS_DIR.mkdir(exist_ok=True)
        (self.DATA_DIR / 'raw').mkdir(exist_ok=True)
        (self.DATA_DIR / 'processed').mkdir(exist_ok=True)

settings = Settings()
"""

def get_data_sources_content():
    return """# 数据源配置
ethereum:
  mainnet:
    rpc_url: "https://mainnet.infura.io/v3/${INFURA_PROJECT_ID}"
    ws_url: "wss://mainnet.infura.io/ws/v3/${INFURA_PROJECT_ID}"
    etherscan_url: "https://api.etherscan.io/api"

defi_protocols:
  defillama:
    base_url: "https://api.llama.fi"
    endpoints:
      tvl: "/tvl"
      protocols: "/protocols"
      chains: "/chains"

macro_data:
  fred:
    base_url: "https://api.stlouisfed.org/fred"
    series:
      - "FEDFUNDS"    # Federal Funds Rate
      - "DGS10"       # 10-Year Treasury Rate
      - "DEXUSEU"     # USD/EUR Exchange Rate

crypto_market:
  coingecko:
    base_url: "https://api.coingecko.com/api/v3"
"""

def get_logger_content():
    return """import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger
"""

def get_constants_content():
    return """from enum import Enum

class Chain(Enum):
    ETHEREUM = "ethereum"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    POLYGON = "polygon"

class Protocol(Enum):
    UNISWAP = "uniswap"
    AAVE = "aave"
    COMPOUND = "compound"
    MAKERDAO = "makerdao"

class MetricType(Enum):
    TVL = "tvl"
    VOLUME = "volume"
    ACTIVE_USERS = "active_users"
"""

def get_helpers_content():
    return """from typing import Any, Dict, List
import pandas as pd
from datetime import datetime

def validate_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    \"\"\"验证数据格式\"\"\"
    valid_data = []
    for item in data:
        if isinstance(item, dict) and 'timestamp' in item:
            valid_data.append(item)
    return valid_data

def format_timestamp(timestamp: Any) -> datetime:
    \"\"\"格式化时间戳\"\"\"
    if isinstance(timestamp, str):
        return pd.to_datetime(timestamp)
    elif isinstance(timestamp, (int, float)):
        return datetime.fromtimestamp(timestamp)
    return timestamp
"""

def get_database_content():
    return """from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from config.settings import settings

Base = declarative_base()
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class BaseModel(Base):
    __abstract__ = True
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ChainMetrics(BaseModel):
    __tablename__ = "chain_metrics"
    
    chain = Column(String(50), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    tvl_usd = Column(Float)
    gas_price_gwei = Column(Float)
    active_addresses = Column(Integer)

def create_tables():
    Base.metadata.create_all(bind=engine)
"""

def get_base_collector_content():
    return """from abc import ABC, abstractmethod
from typing import Dict, List, Any
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class BaseCollector(ABC):
    def __init__(self, source_name: str):
        self.source_name = source_name
    
    @abstractmethod
    async def collect_data(self, **kwargs) -> List[Dict[str, Any]]:
        pass
    
    def validate_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [item for item in data if self._is_valid_record(item)]
    
    def _is_valid_record(self, record: Dict[str, Any]) -> bool:
        return 'timestamp' in record and record['timestamp'] is not None
"""

def get_test_collectors_content():
    return """import pytest
from src.data.collectors.base_collector import BaseCollector

class TestBaseCollector:
    def test_validate_data(self):
        # Test data validation logic
        pass
"""

def get_test_processors_content():
    return """import pytest

class TestDataProcessors:
    def test_data_cleaning(self):
        # Test data cleaning logic
        pass
"""

def get_test_analysis_content():
    return """import pytest

class TestAnalysisModules:
    def test_network_analysis(self):
        # Test network analysis logic
        pass
"""

def get_setup_database_content():
    return """#!/usr/bin/env python3
from src.data.storage.database import create_tables
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    logger.info("Setting up database...")
    create_tables()
    logger.info("Database setup complete!")

if __name__ == "__main__":
    main()
"""

def get_daily_update_content():
    return """#!/usr/bin/env python3
import asyncio
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

async def main():
    logger.info("Starting daily data update...")
    # Add data collection logic here
    logger.info("Daily update complete!")

if __name__ == "__main__":
    asyncio.run(main())
"""

def get_generate_report_content():
    return """#!/usr/bin/env python3
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    logger.info("Generating analysis report...")
    # Add report generation logic here
    logger.info("Report generation complete!")

if __name__ == "__main__":
    main()
"""

def get_notebook_content(title: str):
    return f'''{{
 "cells": [
  {{
   "cell_type": "markdown",
   "metadata": {{}},
   "source": [
    "# {title}\\n",
    "\\n",
    "This notebook contains analysis for {title.lower()}."
   ]
  }},
  {{
   "cell_type": "code",
   "execution_count": null,
   "metadata": {{}},
   "outputs": [],
   "source": [
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "\\n",
    "# Set up plotting\\n",
    "plt.style.use('seaborn-v0_8')\\n",
    "sns.set_palette('husl')"
   ]
  }}
 ],
 "metadata": {{
  "kernelspec": {{
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }},
  "language_info": {{
   "name": "python",
   "version": "3.11.0"
  }}
 }},
 "nbformat": 4,
 "nbformat_minor": 4
}}'''

def get_api_docs_content():
    return """# API Reference

## Data Collection APIs

### Collectors

#### BaseCollector
Base class for all data collectors.

#### DefiLlamaCollector
Collects DeFi protocol data from DeFiLlama API.

#### EthereumGasCollector  
Collects Ethereum gas prices and network metrics.

## Analysis APIs

### Network Analysis
Functions for analyzing protocol relationships and systemic risk.

### Regression Models
Time series and multivariate regression analysis tools.
"""

def get_data_sources_docs_content():
    return """# Data Sources

## Blockchain Data
- **Ethereum**: Infura, Alchemy, Etherscan
- **Layer 2**: Arbitrum, Optimism, Polygon
- **DeFi Protocols**: DeFiLlama, The Graph

## Traditional Finance
- **FRED**: Federal Reserve Economic Data
- **Yahoo Finance**: Stock market data
- **Alpha Vantage**: Financial market data

## Crypto Market Data
- **CoinGecko**: Cryptocurrency market data
- **CoinMarketCap**: Market capitalization data
"""

def get_methodology_docs_content():
    return """# Methodology

## Data Collection
1. Multi-source data aggregation
2. Real-time and historical data collection
3. Data validation and cleaning

## Network Analysis
1. Protocol dependency mapping
2. Centrality metrics calculation
3. Systemic risk assessment

## Regression Analysis
1. Time series modeling
2. Multivariate correlation analysis
3. Machine learning integration

## Risk Assessment
1. Scenario analysis
2. Stress testing
3. Contagion modeling
"""

if __name__ == "__main__":
    create_project_structure()