# src/data/collectors/defi_collector.py
import aiohttp
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from .base_collector import BaseCollector
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DefiLlamaCollector(BaseCollector):
    """DeFiLlama数据收集器 - 获取TVL和协议数据"""
    
    def __init__(self):
        super().__init__("defillama")
        self.base_url = "https://api.llama.fi"
    
    async def collect_data(self, **kwargs) -> List[Dict[str, Any]]:
        """收集DeFi协议数据"""
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._collect_protocols(session),
                self._collect_chains(session),
                self._collect_yields(session)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_data = []
            for result in results:
                if isinstance(result, list):
                    all_data.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Error in data collection: {result}")
            
            return self.validate_data(all_data)
    
    async def _collect_protocols(self, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        """收集协议数据"""
        url = f"{self.base_url}/protocols"
        data = await self._make_request(session, url)
        
        if not data:
            return []
        
        protocols = []
        for protocol in data[:50]:  # 限制前50个协议
            try:
                protocols.append({
                    'data_type': 'protocol_metrics',
                    'protocol': protocol['name'].lower().replace(' ', '_'),
                    'chain': protocol.get('chain', 'ethereum').lower(),
                    'timestamp': datetime.utcnow(),
                    'tvl_usd': float(protocol.get('tvl', 0)),
                    'volume_24h': float(protocol.get('volume', 0)) if protocol.get('volume') else None,
                    'users_24h': protocol.get('users', None),
                    'transactions_24h': protocol.get('transactions', None),
                    'metadata': {
                        'category': protocol.get('category', 'unknown'),
                        'url': protocol.get('url', ''),
                        'logo': protocol.get('logo', ''),
                        'description': protocol.get('description', '')[:200]  # 限制长度
                    }
                })
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Error processing protocol {protocol.get('name', 'unknown')}: {e}")
                continue
        
        logger.info(f"Collected {len(protocols)} protocol metrics")
        return protocols
    
    async def _collect_chains(self, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        """收集链数据"""
        url = f"{self.base_url}/chains"
        data = await self._make_request(session, url)
        
        if not data:
            return []
        
        chains = []
        for chain_info in data:
            try:
                chains.append({
                    'data_type': 'chain_metrics',
                    'chain': chain_info['name'].lower(),
                    'timestamp': datetime.utcnow(),
                    'tvl_usd': float(chain_info.get('tvl', 0)),
                    'metadata': {
                        'gecko_id': chain_info.get('gecko_id', ''),
                        'symbol': chain_info.get('tokenSymbol', ''),
                        'cmc_id': chain_info.get('cmcId', '')
                    }
                })
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Error processing chain {chain_info.get('name', 'unknown')}: {e}")
                continue
        
        logger.info(f"Collected {len(chains)} chain metrics")
        return chains
    
    async def _collect_yields(self, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        """收集收益率数据"""
        url = f"{self.base_url}/yields"
        data = await self._make_request(session, url)
        
        if not data or 'data' not in data:
            return []
        
        yields = []
        for pool in data['data'][:100]:  # 限制前100个池子
            try:
                yields.append({
                    'data_type': 'yield_metrics',
                    'protocol': pool.get('project', '').lower(),
                    'chain': pool.get('chain', '').lower(),
                    'timestamp': datetime.utcnow(),
                    'apy': float(pool.get('apy', 0)),
                    'tvl_usd': float(pool.get('tvlUsd', 0)),
                    'metadata': {
                        'pool_id': pool.get('pool', ''),
                        'symbol': pool.get('symbol', ''),
                        'apy_base': pool.get('apyBase', 0),
                        'apy_reward': pool.get('apyReward', 0),
                        'il_risk': pool.get('ilRisk', 'unknown'),
                        'outlook': pool.get('outlook', 'unknown')
                    }
                })
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Error processing yield pool: {e}")
                continue
        
        logger.info(f"Collected {len(yields)} yield metrics")
        return yields
    
    def transform_data(self, raw_data: Any) -> List[Dict[str, Any]]:
        """数据转换 - 由具体的收集方法处理"""
        return []

# ---

# src/data/collectors/gas_collector.py
import aiohttp
import asyncio
from web3 import Web3
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from .base_collector import BaseCollector
from src.utils.logger import setup_logger
from config.settings import settings

logger = setup_logger(__name__)

class EthereumGasCollector(BaseCollector):
    """以太坊Gas费用和网络数据收集器"""
    
    def __init__(self):
        super().__init__("etherscan", settings.ETHERSCAN_API_KEY)
        self.etherscan_url = "https://api.etherscan.io/api"
        
        # 初始化Web3连接
        if settings.INFURA_PROJECT_ID:
            infura_url = f"https://mainnet.infura.io/v3/{settings.INFURA_PROJECT_ID}"
            self.web3 = Web3(Web3.HTTPProvider(infura_url))
            logger.info("Web3 connected via Infura")
        else:
            self.web3 = None
            logger.warning("No Infura project ID provided")
    
    async def collect_data(self, **kwargs) -> List[Dict[str, Any]]:
        """收集Gas和网络数据"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            # Gas数据收集
            if self.api_key:
                tasks.extend([
                    self._collect_gas_prices(session),
                    self._collect_gas_oracle(session),
                    self._collect_eth_supply(session)
                ])
            
            # Web3数据收集
            if self.web3 and self.web3.is_connected():
                tasks.append(self._collect_block_data())
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_data = []
            for result in results:
                if isinstance(result, list):
                    all_data.extend(result)
                elif isinstance(result, dict):
                    all_data.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Error in gas data collection: {result}")
            
            return self.validate_data(all_data)
    
    async def _collect_gas_prices(self, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        """收集Gas价格数据"""
        params = {
            'module': 'gastracker',
            'action': 'gasoracle',
            'apikey': self.api_key
        }
        
        data = await self._make_request(session, self.etherscan_url, params=params)
        
        if not data or data.get('status') != '1':
            logger.warning("Failed to fetch gas prices from Etherscan")
            return []
        
        result = data.get('result', {})
        
        return [{
            'data_type': 'gas_metrics',
            'chain': 'ethereum',
            'timestamp': datetime.utcnow(),
            'gas_price_gwei': float(result.get('ProposeGasPrice', 0)),
            'metadata': {
                'safe_gas_price': float(result.get('SafeGasPrice', 0)),
                'fast_gas_price': float(result.get('FastGasPrice', 0)),
                'suggested_base_fee': float(result.get('suggestBaseFee', 0)),
                'gas_used_ratio': result.get('gasUsedRatio', '')
            }
        }]
    
    async def _collect_gas_oracle(self, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        """收集Gas预言机数据"""
        params = {
            'module': 'gastracker',
            'action': 'gasestimate',
            'gasprice': '2000000000',  # 2 Gwei
            'apikey': self.api_key
        }
        
        data = await self._make_request(session, self.etherscan_url, params=params)
        
        if not data or data.get('status') != '1':
            return []
        
        return [{
            'data_type': 'gas_estimate',
            'chain': 'ethereum',
            'timestamp': datetime.utcnow(),
            'estimated_confirmation_time': float(data.get('result', 0)),
            'metadata': {
                'gas_price_used': 2.0,  # Gwei
                'source': 'etherscan_oracle'
            }
        }]
    
    async def _collect_eth_supply(self, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        """收集ETH供应量数据"""
        params = {
            'module': 'stats',
            'action': 'ethsupply',
            'apikey': self.api_key
        }
        
        data = await self._make_request(session, self.etherscan_url, params=params)
        
        if not data or data.get('status') != '1':
            return []
        
        return [{
            'data_type': 'supply_metrics',
            'chain': 'ethereum',
            'timestamp': datetime.utcnow(),
            'total_supply': float(data.get('result', 0)) / 1e18,  # Wei to ETH
            'metadata': {
                'source': 'etherscan',
                'unit': 'ETH'
            }
        }]
    
    async def _collect_block_data(self) -> Dict[str, Any]:
        """收集当前区块数据"""
        try:
            latest_block = self.web3.eth.get_block('latest')
            
            return {
                'data_type': 'block_metrics',
                'chain': 'ethereum',
                'timestamp': datetime.fromtimestamp(latest_block['timestamp']),
                'block_number': latest_block['number'],
                'transaction_count': len(latest_block['transactions']),
                'gas_used': latest_block['gasUsed'],
                'gas_limit': latest_block['gasLimit'],
                'metadata': {
                    'gas_utilization': (latest_block['gasUsed'] / latest_block['gasLimit']) * 100,
                    'base_fee_per_gas': latest_block.get('baseFeePerGas', 0),
                    'difficulty': latest_block.get('difficulty', 0),
                    'size': latest_block.get('size', 0)
                }
            }
        except Exception as e:
            logger.error(f"Error collecting block data: {e}")
            return {}
    
    def transform_data(self, raw_data: Any) -> List[Dict[str, Any]]:
        """数据转换"""
        return []

# ---

# src/data/collectors/macro_collector.py
import aiohttp
import asyncio
import yfinance as yf
import pandas as pd
from fredapi import Fred
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from .base_collector import BaseCollector
from src.utils.logger import setup_logger
from config.settings import settings

logger = setup_logger(__name__)

class MacroDataCollector(BaseCollector):
    """宏观经济数据收集器"""
    
    def __init__(self):
        super().__init__("macro_data")
        
        # FRED API
        self.fred = None
        if settings.FRED_API_KEY:
            try:
                self.fred = Fred(api_key=settings.FRED_API_KEY)
                logger.info("FRED API initialized")
            except Exception as e:
                logger.error(f"Failed to initialize FRED API: {e}")
        
        # 经济指标
        self.fred_indicators = {
            'FEDFUNDS': 'federal_funds_rate',
            'DGS10': 'treasury_10y',
            'DEXUSEU': 'usd_eur_rate',
            'DTWEXBGS': 'dollar_index',
            'CPIAUCSL': 'cpi_inflation',
            'UNRATE': 'unemployment_rate'
        }
        
        # 市场指标
        self.market_symbols = {
            '^GSPC': 'sp500',
            '^IXIC': 'nasdaq',
            '^VIX': 'vix',
            'GLD': 'gold_etf',
            'TLT': 'treasury_etf'
        }
    
    async def collect_data(self, days_back: int = 30, **kwargs) -> List[Dict[str, Any]]:
        """收集宏观数据"""
        tasks = []
        
        # FRED数据
        if self.fred:
            tasks.append(self._collect_fred_data(days_back))
        
        # 市场数据
        tasks.append(self._collect_market_data(days_back))
        
        # 加密市场情绪
        tasks.append(self._collect_crypto_sentiment())
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_data = []
        for result in results:
            if isinstance(result, list):
                all_data.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Error in macro data collection: {result}")
        
        return self.validate_data(all_data)
    
    async def _collect_fred_data(self, days_back: int) -> List[Dict[str, Any]]:
        """收集FRED经济数据"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        fred_data = []
        
        for fred_code, indicator_name in self.fred_indicators.items():
            try:
                await self._rate_limit_check(fred_code)
                
                series = self.fred.get_series(
                    fred_code, 
                    start=start_date, 
                    end=end_date
                )
                
                if not series.empty:
                    # 取最新值
                    latest_date = series.index[-1]
                    latest_value = series.iloc[-1]
                    
                    if not pd.isna(latest_value):
                        fred_data.append({
                            'data_type': 'macro_indicator',
                            'indicator': indicator_name,
                            'timestamp': pd.to_datetime(latest_date),
                            'value': float(latest_value),
                            'source': 'fred',
                            'frequency': 'daily',
                            'metadata': {
                                'fred_code': fred_code,
                                'data_points': len(series)
                            }
                        })
                
            except Exception as e:
                logger.error(f"Error collecting FRED data for {fred_code}: {e}")
                continue
        
        logger.info(f"Collected {len(fred_data)} FRED indicators")
        return fred_data
    
    async def _collect_market_data(self, days_back: int) -> List[Dict[str, Any]]:
        """收集市场数据"""
        period = f"{days_back}d"
        market_data = []
        
        for symbol, name in self.market_symbols.items():
            try:
                await self._rate_limit_check(symbol)
                
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                
                if not hist.empty:
                    latest = hist.iloc[-1]
                    latest_date = hist.index[-1]
                    
                    market_data.append({
                        'data_type': 'market_indicator',
                        'indicator': name,
                        'timestamp': pd.to_datetime(latest_date),
                        'value': float(latest['Close']),
                        'source': 'yahoo_finance',
                        'frequency': 'daily',
                        'metadata': {
                            'symbol': symbol,
                            'open': float(latest['Open']),
                            'high': float(latest['High']),
                            'low': float(latest['Low']),
                            'volume': int(latest['Volume']) if not pd.isna(latest['Volume']) else 0,
                            'change_pct': ((latest['Close'] - latest['Open']) / latest['Open']) * 100
                        }
                    })
                
            except Exception as e:
                logger.error(f"Error collecting market data for {symbol}: {e}")
                continue
        
        logger.info(f"Collected {len(market_data)} market indicators")
        return market_data
    
    async def _collect_crypto_sentiment(self) -> List[Dict[str, Any]]:
        """收集加密货币情绪数据"""
        async with aiohttp.ClientSession() as session:
            # Fear & Greed Index
            fear_greed_url = "https://api.alternative.me/fng/"
            
            try:
                data = await self._make_request(session, fear_greed_url)
                
                if data and 'data' in data and len(data['data']) > 0:
                    latest = data['data'][0]
                    
                    return [{
                        'data_type': 'sentiment_indicator',
                        'indicator': 'fear_greed_index',
                        'timestamp': datetime.fromtimestamp(int(latest['timestamp'])),
                        'value': float(latest['value']),
                        'source': 'alternative_me',
                        'frequency': 'daily',
                        'metadata': {
                            'classification': latest['value_classification'],
                            'description': 'Crypto Fear & Greed Index (0=Extreme Fear, 100=Extreme Greed)'
                        }
                    }]
                
            except Exception as e:
                logger.error(f"Error collecting Fear & Greed Index: {e}")
        
        return []
    
    def transform_data(self, raw_data: Any) -> List[Dict[str, Any]]:
        """数据转换"""
        return []

# ---

# src/data/collectors/address_collector.py
import aiohttp
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from .base_collector import BaseCollector
from src.utils.logger import setup_logger
from config.settings import settings

logger = setup_logger(__name__)

class AddressActivityCollector(BaseCollector):
    """地址活跃度数据收集器"""
    
    def __init__(self):
        super().__init__("address_activity")
        self.etherscan_url = "https://api.etherscan.io/api"
        self.api_key = settings.ETHERSCAN_API_KEY
    
    async def collect_data(self, **kwargs) -> List[Dict[str, Any]]:
        """收集地址活跃度数据"""
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._collect_unique_addresses(session),
                self._collect_transaction_stats(session),
            ]
            
            # 如果有The Graph API key，添加更详细的地址分析
            if settings.THE_GRAPH_API_KEY:
                tasks.append(self._collect_address_distribution(session))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_data = []
            for result in results:
                if isinstance(result, list):
                    all_data.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Error in address data collection: {result}")
            
            return self.validate_data(all_data)
    
    async def _collect_unique_addresses(self, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        """收集唯一地址数据"""
        if not self.api_key:
            return []
        
        # 获取最近的交易统计
        params = {
            'module': 'stats',
            'action': 'ethsupply',
            'apikey': self.api_key
        }
        
        # 由于Etherscan没有直接的唯一地址API，我们通过其他方式估算
        # 这里我们收集一些基础的网络统计数据
        data = await self._make_request(session, self.etherscan_url, params=params)
        
        if not data or data.get('status') != '1':
            return []
        
        # 这是一个简化的实现，实际项目中需要更复杂的地址分析
        return [{
            'data_type': 'address_metrics',
            'chain': 'ethereum',
            'timestamp': datetime.utcnow(),
            'metric': 'network_stats',
            'metadata': {
                'source': 'etherscan',
                'note': 'Basic network statistics - detailed address analysis requires additional data sources'
            }
        }]
    
    async def _collect_transaction_stats(self, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        """收集交易统计数据"""
        if not self.api_key:
            return []
        
        # 获取最新区块信息来估算交易活跃度
        params = {
            'module': 'proxy',
            'action': 'eth_blockNumber',
            'apikey': self.api_key
        }
        
        data = await self._make_request(session, self.etherscan_url, params=params)
        
        if not data or not data.get('result'):
            return []
        
        try:
            latest_block = int(data['result'], 16)
            
            return [{
                'data_type': 'transaction_metrics',
                'chain': 'ethereum',
                'timestamp': datetime.utcnow(),
                'latest_block': latest_block,
                'metadata': {
                    'source': 'etherscan',
                    'data_type': 'block_height'
                }
            }]
            
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing block number: {e}")
            return []
    
    async def _collect_address_distribution(self, session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        """通过The Graph收集地址分布数据"""
        # 这是一个示例GraphQL查询，实际需要根据具体的subgraph调整
        query = """
        {
          ethereum(network: ethereum) {
            transactions(date: {since: "2025-01-01"}) {
              count(uniq: senders)
              count(uniq: receivers)
            }
          }
        }
        """
        
        # 由于没有确切的The Graph endpoint，这里返回空数据
        # 实际实现需要配置正确的subgraph endpoint
        logger.info("Address distribution collection requires specific subgraph configuration")
        return []
    
    def transform_data(self, raw_data: Any) -> List[Dict[str, Any]]:
        """数据转换"""
        return []

# ---

# src/data/collectors/__init__.py
"""
数据收集器模块

提供各种数据源的收集器：
- DefiLlamaCollector: DeFi协议数据
- EthereumGasCollector: Gas费用和网络数据  
- MacroDataCollector: 宏观经济数据
- AddressActivityCollector: 地址活跃度数据
"""

from .base_collector import BaseCollector
from .defi_collector import DefiLlamaCollector
from .gas_collector import EthereumGasCollector
from .macro_collector import MacroDataCollector
from .address_collector import AddressActivityCollector

__all__ = [
    'BaseCollector',
    'DefiLlamaCollector', 
    'EthereumGasCollector',
    'MacroDataCollector',
    'AddressActivityCollector'
]

# ---

# src/data/storage/cache.py
import redis
import json
import pickle
from typing import Any, Optional, Union
from datetime import timedelta
from src.utils.logger import setup_logger
from config.settings import settings

logger = setup_logger(__name__)

class CacheManager:
    """Redis缓存管理器"""
    
    def __init__(self):
        try:
            self.redis_client = redis.from_url(
                settings.REDIS_URL,
                decode_responses=False  # 允许存储二进制数据
            )
            # 测试连接
            self.redis_client.ping()
            logger.info("Redis cache connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def set(self, key: str, value: Any, expire: Optional[Union[int, timedelta]] = None) -> bool:
        """设置缓存值"""
        if not self.redis_client:
            return False
        
        try:
            # 序列化数据
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value).encode('utf-8')
            else:
                serialized_value = pickle.dumps(value)
            
            # 设置过期时间
            if isinstance(expire, timedelta):
                expire = int(expire.total_seconds())
            
            result = self.redis_client.set(key, serialized_value, ex=expire)
            return result is True
            
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if not self.redis_client:
            return None
        
        try:
            value = self.redis_client.get(key)
            if value is None:
                return None
            
            # 尝试JSON反序列化
            try:
                return json.loads(value.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # 如果JSON失败，尝试pickle
                return pickle.loads(value)
                
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """删除缓存键"""
        if not self.redis_client:
            return False
        
        try:
            result = self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        if not self.redis_client:
            return False
        
        try:
            return self.redis_client.exists(key) > 0
        except Exception as e:
            logger.error(f"Error checking cache key {key}: {e}")
            return False
    
    def set_json(self, key: str, value: dict, expire: Optional[Union[int, timedelta]] = None) -> bool:
        """专门用于JSON数据的缓存方法"""
        return self.set(key, value, expire)
    
    def get_json(self, key: str) -> Optional[dict]:
        """专门用于获取JSON数据的方法"""
        result = self.get(key)
        return result if isinstance(result, dict) else None
    
    def cache_function_result(self, cache_key: str, expire: Union[int, timedelta] = 3600):
        """装饰器：缓存函数结果"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # 生成完整的缓存键
                full_key = f"{cache_key}:{hash(str(args) + str(kwargs))}"
                
                # 尝试从缓存获取
                cached_result = self.get(full_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for {full_key}")
                    return cached_result
                
                # 执行函数并缓存结果
                result = func(*args, **kwargs)
                self.set(full_key, result, expire)
                logger.debug(f"Cache set for {full_key}")
                
                return result
            return wrapper
        return decorator

# 全局缓存实例
cache_manager = CacheManager()