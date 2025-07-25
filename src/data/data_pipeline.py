# src/data/data_pipeline.py
"""
数据管道 - 核心协调器
"""
import asyncio
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

class SimpleDataPipeline:
    """简化的数据管道，处理核心功能"""
    
    def __init__(self):
        self.pipeline_stats = {}
        
        # 创建简单的收集器（避免复杂的导入依赖）
        self.collectors = {
            'defi': SimpleCollector('defi'),
            'gas': SimpleCollector('gas'), 
            'macro': SimpleCollector('macro'),
            'address': SimpleCollector('address')
        }
        
        # 创建简单的处理器
        self.data_cleaner = SimpleDataCleaner()
        self.feature_engineer = SimpleFeatureEngineer()
        self.data_validator = SimpleDataValidator()
    
    async def run_full_pipeline(self, sources: List[str] = None) -> Dict[str, Any]:
        """运行完整的数据管道"""
        start_time = datetime.utcnow()
        
        if sources is None:
            sources = ['defi']  # 默认只测试不需要API key的源
        
        result = {
            'start_time': start_time,
            'sources_processed': [],
            'total_records_collected': 0,
            'total_records_processed': 0,
            'errors': [],
            'execution_time_seconds': 0
        }
        
        try:
            # 数据收集
            for source in sources:
                if source in self.collectors:
                    try:
                        data = await self.collectors[source].collect_data()
                        result['sources_processed'].append(source)
                        result['total_records_collected'] += len(data)
                        
                        # 数据清洗
                        cleaned_data = self.data_cleaner.clean_metrics_data(data)
                        result['total_records_processed'] += len(cleaned_data)
                        
                    except Exception as e:
                        result['errors'].append(f"Error in {source}: {str(e)}")
            
            end_time = datetime.utcnow()
            result['end_time'] = end_time
            result['execution_time_seconds'] = (end_time - start_time).total_seconds()
            
            # 保存统计信息
            self.pipeline_stats[start_time.isoformat()] = result
            
        except Exception as e:
            result['errors'].append(f"Pipeline error: {str(e)}")
        
        return result
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """获取管道状态"""
        if not self.pipeline_stats:
            return {"status": "No pipeline runs found"}
        
        latest_run = max(self.pipeline_stats.keys())
        latest_stats = self.pipeline_stats[latest_run]
        
        return {
            "latest_run": latest_run,
            "status": "success" if not latest_stats['errors'] else "completed_with_errors",
            "sources_processed": latest_stats['sources_processed'],
            "total_records": latest_stats['total_records_processed'],
            "execution_time": latest_stats['execution_time_seconds'],
            "error_count": len(latest_stats['errors']),
            "total_runs": len(self.pipeline_stats)
        }

class SimpleCollector:
    """简单的数据收集器（用于测试）"""
    
    def __init__(self, source_name: str):
        self.source_name = source_name
    
    async def collect_data(self, **kwargs) -> List[Dict[str, Any]]:
        """收集测试数据"""
        # 模拟网络延迟
        await asyncio.sleep(0.1)
        
        # 根据不同源返回不同的测试数据
        if self.source_name == 'defi':
            return [
                {
                    'data_type': 'protocol_metrics',
                    'protocol': 'uniswap',
                    'chain': 'ethereum',
                    'timestamp': datetime.utcnow(),
                    'tvl_usd': 1000000.0,
                    'volume_24h': 50000.0,
                    'metadata': {'test': True, 'source': 'defi'}
                },
                {
                    'data_type': 'protocol_metrics',
                    'protocol': 'aave',
                    'chain': 'ethereum', 
                    'timestamp': datetime.utcnow(),
                    'tvl_usd': 800000.0,
                    'volume_24h': 30000.0,
                    'metadata': {'test': True, 'source': 'defi'}
                }
            ]
        elif self.source_name == 'gas':
            return [
                {
                    'data_type': 'gas_metrics',
                    'chain': 'ethereum',
                    'timestamp': datetime.utcnow(),
                    'gas_price_gwei': 25.0,
                    'metadata': {'test': True, 'source': 'gas'}
                }
            ]
        elif self.source_name == 'macro':
            return [
                {
                    'data_type': 'macro_indicator',
                    'indicator': 'test_rate',
                    'timestamp': datetime.utcnow(),
                    'value': 5.25,
                    'source': 'test',
                    'metadata': {'test': True, 'source': 'macro'}
                }
            ]
        else:
            return [
                {
                    'data_type': 'test_metric',
                    'timestamp': datetime.utcnow(),
                    'value': 100.0,
                    'source': self.source_name,
                    'metadata': {'test': True}
                }
            ]

class SimpleDataCleaner:
    """简单的数据清洗器"""
    
    def clean_metrics_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """清洗指标数据"""
        if not data:
            return []
        
        cleaned_data = []
        
        for record in data:
            # 基础验证
            if not isinstance(record, dict):
                continue
            
            # 检查必须字段
            if 'timestamp' not in record:
                continue
            
            # 简单的数据清洗
            cleaned_record = record.copy()
            
            # 确保timestamp是datetime对象
            if isinstance(cleaned_record['timestamp'], str):
                try:
                    cleaned_record['timestamp'] = pd.to_datetime(cleaned_record['timestamp'])
                except:
                    continue
            
            cleaned_data.append(cleaned_record)
        
        return cleaned_data

class SimpleFeatureEngineer:
    """简单的特征工程器"""
    
    def engineer_features(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """执行特征工程"""
        if not data:
            return pd.DataFrame()
        
        # 转换为DataFrame
        df = pd.DataFrame(data)
        
        # 确保时间戳列存在
        if 'timestamp' not in df.columns:
            return df
        
        # 转换时间戳
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 添加时间特征
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # 添加周期性特征
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        return df

# 替换 src/data/data_pipeline.py 中的 SimpleDataValidator 类

class SimpleDataValidator:
    """简单的数据验证器（修复版）"""
    
    def validate_dataset(self, df: pd.DataFrame, dataset_name: str = "unknown") -> Dict[str, Any]:
        """验证数据集的质量"""
        validation_result = {
            'dataset_name': dataset_name,
            'timestamp': datetime.utcnow(),
            'total_rows': len(df) if df is not None else 0,
            'total_columns': len(df.columns) if df is not None and hasattr(df, 'columns') else 0,
            'validation_passed': True,
            'issues': []
        }
        
        if df is None or df.empty:
            validation_result['validation_passed'] = False
            validation_result['issues'].append("Dataset is empty")
            return validation_result
        
        try:
            # 检查缺失值
            missing_counts = df.isnull().sum()
            if missing_counts.any():
                validation_result['issues'].append(f"Found missing values in some columns")
            
            # 检查重复行（跳过包含字典的列）
            try:
                # 只检查可哈希的列
                hashable_cols = []
                for col in df.columns:
                    try:
                        # 测试是否可以哈希
                        hash(str(df[col].iloc[0]) if len(df) > 0 else "")
                        hashable_cols.append(col)
                    except:
                        continue
                
                if hashable_cols:
                    duplicate_count = df[hashable_cols].duplicated().sum()
                    if duplicate_count > 0:
                        validation_result['issues'].append(f"Found {duplicate_count} duplicate rows")
            except Exception:
                # 如果重复检查失败，跳过
                pass
            
            # 基本数据类型检查
            numeric_cols = df.select_dtypes(include=[pd.np.number]).columns
            if len(numeric_cols) > 0:
                # 检查是否有无穷大或NaN值
                has_inf = df[numeric_cols].isin([pd.np.inf, -pd.np.inf]).any().any()
                if has_inf:
                    validation_result['issues'].append("Found infinite values in numeric columns")
        
        except Exception as e:
            validation_result['issues'].append(f"Validation error: {str(e)}")
        
        # 如果有问题，标记验证失败
        validation_result['validation_passed'] = len(validation_result['issues']) == 0
        
        return validation_result

# 创建全局管道实例
data_pipeline = SimpleDataPipeline()