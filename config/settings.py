import os
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
