# src/data/real_data_fetcher.py
"""
真实数据获取器 - 集成多个数据源
无需API key也能获取部分数据
"""

import aiohttp
import asyncio
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import time

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class RealDataFetcher:
    """真实数据获取器"""
    
    def __init__(self):
        self.session = None
        self.rate_limits = {
            'coingecko': {'calls': 0, 'reset_time': 0, 'limit': 50},  # 每分钟50次
            'defillama': {'calls': 0, 'reset_time': 0, 'limit': 300}   # 每分钟300次
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'OnChain-Macro-Analysis/1.0'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _rate_limit_check(self, source: str):
        """检查和执行速率限制"""
        current_time = time.time()
        rate_info = self.rate_limits.get(source, {'calls': 0, 'reset_time': 0, 'limit': 60})
        
        # 如果超过1分钟，重置计数器
        if current_time - rate_info['reset_time'] > 60:
            rate_info['calls'] = 0
            rate_info['reset_time'] = current_time
        
        # 如果达到限制，等待
        if rate_info['calls'] >= rate_info['limit']:
            wait_time = 60 - (current_time - rate_info['reset_time'])
            if wait_time > 0:
                logger.info(f"Rate limit reached for {source}, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                rate_info['calls'] = 0
                rate_info['reset_time'] = time.time()
        
        rate_info['calls'] += 1
    
    async def fetch_defillama_protocols(self) -> List[Dict[str, Any]]:
        """获取DeFiLlama协议数据（无需API key）"""
        await self._rate_limit_check('defillama')
        
        try:
            url = "https://api.llama.fi/protocols"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Fetched {len(data)} protocols from DeFiLlama")
                    return data
                else:
                    logger.warning(f"DeFiLlama API returned status {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching DeFiLlama data: {e}")
            return []
    
    async def fetch_defillama_chains(self) -> List[Dict[str, Any]]:
        """获取链数据"""
        await self._rate_limit_check('defillama')
        
        try:
            url = "https://api.llama.fi/chains"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Fetched {len(data)} chains from DeFiLlama")
                    return data
                else:
                    logger.warning(f"DeFiLlama chains API returned status {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching chain data: {e}")
            return []
    
    async def fetch_coingecko_global(self) -> Dict[str, Any]:
        """获取CoinGecko全局数据（无需API key）"""
        await self._rate_limit_check('coingecko')
        
        try:
            url = "https://api.coingecko.com/api/v3/global"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("Fetched global crypto data from CoinGecko")
                    return data
                else:
                    logger.warning(f"CoinGecko API returned status {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Error fetching CoinGecko data: {e}")
            return {}
    
    async def fetch_coingecko_defi(self) -> Dict[str, Any]:
        """获取DeFi市场数据"""
        await self._rate_limit_check('coingecko')
        
        try:
            url = "https://api.coingecko.com/api/v3/global/decentralized_finance_defi"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("Fetched DeFi market data from CoinGecko")
                    return data
                else:
                    logger.warning(f"CoinGecko DeFi API returned status {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Error fetching DeFi market data: {e}")
            return {}
    
    def transform_defillama_protocols(self, raw_data: List[Dict]) -> List[Dict[str, Any]]:
        """转换DeFiLlama协议数据"""
        transformed = []
        
        for protocol in raw_data:
            try:
                # 基础数据验证
                if not protocol.get('name') or not isinstance(protocol.get('tvl'), (int, float)):
                    continue
                
                # 跳过TVL太小的协议
                if protocol.get('tvl', 0) < 1000000:  # 100万美元阈值
                    continue
                
                transformed_protocol = {
                    'protocol': protocol['name'].lower().replace(' ', '_').replace('-', '_'),
                    'chain': protocol.get('chain', protocol.get('chains', ['ethereum'])[0] if protocol.get('chains') else 'ethereum').lower(),
                    'tvl_usd': float(protocol['tvl']),
                    'volume_24h': None,  # DeFiLlama协议端点不包含交易量
                    'users_24h': None,
                    'transactions_24h': None,
                    'yield_rate': None,
                    'metadata': {
                        'category': protocol.get('category', 'unknown'),
                        'url': protocol.get('url', ''),
                        'logo': protocol.get('logo', ''),
                        'description': protocol.get('description', '')[:200] if protocol.get('description') else '',
                        'chains': protocol.get('chains', []),
                        'module': protocol.get('module', ''),
                        'twitter': protocol.get('twitter', ''),
                        'forked_from': protocol.get('forkedFrom', []),
                        'gecko_id': protocol.get('gecko_id', ''),
                        'cmcId': protocol.get('cmcId', ''),
                        'listedAt': protocol.get('listedAt', 0),
                        'methodology': protocol.get('methodology', {})
                    }
                }
                
                transformed.append(transformed_protocol)
                
            except (KeyError, ValueError, TypeError) as e:
                logger.debug(f"Skipping protocol due to data issue: {e}")
                continue
        
        logger.info(f"Transformed {len(transformed)} protocols from DeFiLlama")
        return transformed
    
    def transform_chain_data(self, raw_data: List[Dict]) -> List[Dict[str, Any]]:
        """转换链数据"""
        transformed = []
        
        for chain in raw_data:
            try:
                if not chain.get('name') or not isinstance(chain.get('tvl'), (int, float)):
                    continue
                
                transformed_chain = {
                    'data_type': 'chain_metrics',
                    'chain': chain['name'].lower(),
                    'timestamp': datetime.utcnow(),
                    'tvl_usd': float(chain['tvl']),
                    'metadata': {
                        'gecko_id': chain.get('gecko_id', ''),
                        'symbol': chain.get('tokenSymbol', ''),
                        'cmc_id': chain.get('cmcId', ''),
                        'categories': chain.get('categories', [])
                    }
                }
                
                transformed.append(transformed_chain)
                
            except (KeyError, ValueError, TypeError) as e:
                logger.debug(f"Skipping chain due to data issue: {e}")
                continue
        
        return transformed
    
    def transform_global_data(self, coingecko_global: Dict, coingecko_defi: Dict) -> List[Dict[str, Any]]:
        """转换全局市场数据"""
        indicators = []
        timestamp = datetime.utcnow()
        
        # CoinGecko全局数据
        if 'data' in coingecko_global:
            global_data = coingecko_global['data']
            
            # 总市值
            if 'total_market_cap' in global_data and 'usd' in global_data['total_market_cap']:
                indicators.append({
                    'data_type': 'macro_indicator',
                    'indicator': 'crypto_total_market_cap',
                    'timestamp': timestamp,
                    'value': float(global_data['total_market_cap']['usd']),
                    'source': 'coingecko',
                    'frequency': 'daily',
                    'metadata': {'unit': 'usd', 'type': 'market_cap'}
                })
            
            # 总交易量
            if 'total_volume' in global_data and 'usd' in global_data['total_volume']:
                indicators.append({
                    'data_type': 'macro_indicator',
                    'indicator': 'crypto_total_volume_24h',
                    'timestamp': timestamp,
                    'value': float(global_data['total_volume']['usd']),
                    'source': 'coingecko',
                    'frequency': 'daily',
                    'metadata': {'unit': 'usd', 'type': 'volume'}
                })
            
            # Bitcoin主导地位
            if 'market_cap_percentage' in global_data and 'btc' in global_data['market_cap_percentage']:
                indicators.append({
                    'data_type': 'macro_indicator',
                    'indicator': 'bitcoin_dominance',
                    'timestamp': timestamp,
                    'value': float(global_data['market_cap_percentage']['btc']),
                    'source': 'coingecko',
                    'frequency': 'daily',
                    'metadata': {'unit': 'percentage', 'type': 'dominance'}
                })
        
        # DeFi数据
        if 'data' in coingecko_defi:
            defi_data = coingecko_defi['data']
            
            # DeFi市值
            if 'defi_market_cap' in defi_data:
                indicators.append({
                    'data_type': 'macro_indicator',
                    'indicator': 'defi_market_cap',
                    'timestamp': timestamp,
                    'value': float(defi_data['defi_market_cap']),
                    'source': 'coingecko',
                    'frequency': 'daily',
                    'metadata': {
                        'unit': 'usd',
                        'type': 'defi_market_cap',
                        'defi_to_eth_ratio': defi_data.get('defi_to_eth_ratio', 0),
                        'trading_volume_24h': defi_data.get('trading_volume_24h', 0)
                    }
                })
        
        return indicators

async def fetch_real_data() -> Dict[str, List[Dict[str, Any]]]:
    """获取所有真实数据"""
    logger.info("Starting real data collection...")
    
    async with RealDataFetcher() as fetcher:
        # 并行获取所有数据
        tasks = [
            fetcher.fetch_defillama_protocols(),
            fetcher.fetch_defillama_chains(),
            fetcher.fetch_coingecko_global(),
            fetcher.fetch_coingecko_defi()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        protocols_raw, chains_raw, global_raw, defi_raw = results
        
        # 转换数据
        all_data = {
            'protocols': [],
            'chains': [],
            'global_indicators': []
        }
        
        # 转换协议数据
        if isinstance(protocols_raw, list):
            all_data['protocols'] = fetcher.transform_defillama_protocols(protocols_raw)
        
        # 转换链数据
        if isinstance(chains_raw, list):
            all_data['chains'] = fetcher.transform_chain_data(chains_raw)
        
        # 转换全局指标
        if isinstance(global_raw, dict) and isinstance(defi_raw, dict):
            all_data['global_indicators'] = fetcher.transform_global_data(global_raw, defi_raw)
        
        logger.info(f"Real data collection completed:")
        logger.info(f"  - Protocols: {len(all_data['protocols'])}")
        logger.info(f"  - Chains: {len(all_data['chains'])}")
        logger.info(f"  - Global indicators: {len(all_data['global_indicators'])}")
        
        return all_data

# 测试函数
async def test_real_data_fetch():
    """测试真实数据获取"""
    print("🌐 Testing Real Data Fetch")
    print("="*50)
    
    try:
        data = await fetch_real_data()
        
        print(f"✅ Data fetch completed:")
        print(f"   - Protocols: {len(data['protocols'])}")
        print(f"   - Chains: {len(data['chains'])}")
        print(f"   - Global indicators: {len(data['global_indicators'])}")
        
        # 显示前5个协议
        if data['protocols']:
            print(f"\n📊 Top 5 Protocols by TVL:")
            top_protocols = sorted(data['protocols'], key=lambda x: x['tvl_usd'], reverse=True)[:5]
            for i, protocol in enumerate(top_protocols):
                print(f"   {i+1}. {protocol['protocol'].upper()}: ${protocol['tvl_usd']:,.0f}")
        
        # 显示链数据
        if data['chains']:
            print(f"\n⛓️ Top 5 Chains by TVL:")
            top_chains = sorted(data['chains'], key=lambda x: x['tvl_usd'], reverse=True)[:5]
            for i, chain in enumerate(top_chains):
                print(f"   {i+1}. {chain['chain'].upper()}: ${chain['tvl_usd']:,.0f}")
        
        # 显示全局指标
        if data['global_indicators']:
            print(f"\n🌍 Global Indicators:")
            for indicator in data['global_indicators']:
                name = indicator['indicator'].replace('_', ' ').title()
                value = indicator['value']
                unit = indicator['metadata'].get('unit', '')
                
                if unit == 'usd':
                    print(f"   - {name}: ${value:,.0f}")
                elif unit == 'percentage':
                    print(f"   - {name}: {value:.1f}%")
                else:
                    print(f"   - {name}: {value:,.0f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Real data fetch failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_real_data_fetch())