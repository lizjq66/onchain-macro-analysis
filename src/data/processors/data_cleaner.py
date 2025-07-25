# src/data/processors/data_cleaner.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DataCleaner:
    """数据清洗和预处理器"""
    
    def __init__(self):
        self.cleaning_stats = {}
    
    def clean_metrics_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """清洗指标数据"""
        if not data:
            return []
        
        cleaned_data = []
        removed_count = 0
        
        for record in data:
            cleaned_record = self._clean_single_record(record)
            if cleaned_record:
                cleaned_data.append(cleaned_record)
            else:
                removed_count += 1
        
        logger.info(f"Data cleaning completed: {len(cleaned_data)} valid records, {removed_count} removed")
        return cleaned_data
    
    def _clean_single_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """清洗单条记录"""
        try:
            # 基础验证
            if not isinstance(record, dict):
                return None
            
            # 时间戳验证和标准化
            timestamp = self._validate_timestamp(record.get('timestamp'))
            if not timestamp:
                return None
            
            # 数值验证和清洗
            cleaned_record = record.copy()
            cleaned_record['timestamp'] = timestamp
            
            # 清洗数值字段
            numeric_fields = ['value', 'tvl_usd', 'volume_24h', 'gas_price_gwei', 'apy']
            for field in numeric_fields:
                if field in cleaned_record:
                    cleaned_value = self._clean_numeric_value(cleaned_record[field])
                    if cleaned_value is not None:
                        cleaned_record[field] = cleaned_value
                    else:
                        # 如果是关键字段且无效，删除整条记录
                        if field in ['value', 'tvl_usd']:
                            return None
            
            # 清洗字符串字段
            string_fields = ['chain', 'protocol', 'indicator']
            for field in string_fields:
                if field in cleaned_record:
                    cleaned_record[field] = self._clean_string_value(cleaned_record[field])
            
            # 验证数据合理性
            if not self._validate_data_ranges(cleaned_record):
                return None
            
            return cleaned_record
            
        except Exception as e:
            logger.warning(f"Error cleaning record: {e}")
            return None
    
    def _validate_timestamp(self, timestamp: Any) -> Optional[datetime]:
        """验证和标准化时间戳"""
        if not timestamp:
            return None
        
        try:
            if isinstance(timestamp, str):
                # 尝试解析字符串时间戳
                parsed_time = pd.to_datetime(timestamp)
                if parsed_time.tz is not None:
                    parsed_time = parsed_time.tz_convert('UTC').tz_localize(None)
                return parsed_time
            elif isinstance(timestamp, (int, float)):
                # Unix时间戳
                return datetime.fromtimestamp(timestamp)
            elif isinstance(timestamp, datetime):
                return timestamp
            else:
                return None
        except Exception as e:
            logger.debug(f"Invalid timestamp {timestamp}: {e}")
            return None
    
    def _clean_numeric_value(self, value: Any) -> Optional[float]:
        """清洗数值"""
        if value is None or value == '':
            return None
        
        try:
            # 转换为浮点数
            if isinstance(value, str):
                # 移除逗号和货币符号
                cleaned_str = value.replace(',', '').replace('$', '').replace('%', '').strip()
                if cleaned_str == '' or cleaned_str.lower() in ['null', 'nan', 'none']:
                    return None
                numeric_value = float(cleaned_str)
            else:
                numeric_value = float(value)
            
            # 检查无效值
            if np.isnan(numeric_value) or np.isinf(numeric_value):
                return None
            
            return numeric_value
            
        except (ValueError, TypeError):
            return None
    
    def _clean_string_value(self, value: Any) -> str:
        """清洗字符串值"""
        if not value:
            return ''
        
        try:
            cleaned = str(value).strip().lower()
            # 移除特殊字符，保留字母数字和下划线
            cleaned = ''.join(c if c.isalnum() or c in ['_', '-'] else '_' for c in cleaned)
            return cleaned
        except Exception:
            return ''
    
    def _validate_data_ranges(self, record: Dict[str, Any]) -> bool:
        """验证数据范围的合理性"""
        try:
            # TVL应该为正数且在合理范围内
            if 'tvl_usd' in record:
                tvl = record['tvl_usd']
                if tvl < 0 or tvl > 1e12:  # 1万亿美元上限
                    return False
            
            # Gas价格应该在合理范围内
            if 'gas_price_gwei' in record:
                gas_price = record['gas_price_gwei']
                if gas_price < 0 or gas_price > 10000:  # 10000 Gwei上限
                    return False
            
            # APY应该在合理范围内
            if 'apy' in record:
                apy = record['apy']
                if apy < -100 or apy > 10000:  # -100%到10000%
                    return False
            
            # 时间戳不应该太久远或太未来
            if 'timestamp' in record:
                timestamp = record['timestamp']
                now = datetime.utcnow()
                if timestamp < now - timedelta(days=3650) or timestamp > now + timedelta(days=1):
                    return False
            
            return True
            
        except Exception:
            return False

# ---

# src/data/processors/feature_engineer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class FeatureEngineer:
    """特征工程处理器"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_stats = {}
    
    def engineer_features(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """执行特征工程"""
        if not data:
            return pd.DataFrame()
        
        # 转换为DataFrame
        df = pd.DataFrame(data)
        
        if df.empty:
            return df
        
        # 确保时间戳列存在且格式正确
        if 'timestamp' not in df.columns:
            logger.error("No timestamp column found")
            return df
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 时间特征
        df = self._create_time_features(df)
        
        # 技术指标特征
        df = self._create_technical_features(df)
        
        # 相对变化特征
        df = self._create_change_features(df)
        
        # 滚动统计特征
        df = self._create_rolling_features(df)
        
        # 交叉特征
        df = self._create_cross_features(df)
        
        # 异常值检测特征
        df = self._create_anomaly_features(df)
        
        logger.info(f"Feature engineering completed: {len(df)} rows, {len(df.columns)} features")
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建时间相关特征"""
        df = df.copy()
        
        # 基础时间特征
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['year'] = df['timestamp'].dt.year
        
        # 周期性特征（正弦余弦编码）
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 交易时间特征
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        df['is_asian_trading'] = ((df['hour'] >= 0) & (df['hour'] <= 8)).astype(int)
        df['is_european_trading'] = ((df['hour'] >= 8) & (df['hour'] <= 16)).astype(int)
        df['is_us_trading'] = ((df['hour'] >= 14) & (df['hour'] <= 22)).astype(int)
        
        return df
    
    def _create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建技术指标特征"""
        df = df.copy()
        
        # 识别数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in ['hour', 'day_of_week', 'month', 'year']:  # 跳过时间特征
                continue
            
            series = df[col].fillna(method='ffill')
            
            # 移动平均
            for window in [7, 14, 30]:
                df[f'{col}_ma_{window}'] = series.rolling(window=window, min_periods=1).mean()
                df[f'{col}_std_{window}'] = series.rolling(window=window, min_periods=1).std()
            
            # 指数移动平均
            for span in [7, 14, 30]:
                df[f'{col}_ema_{span}'] = series.ewm(span=span).mean()
            
            # 布林带
            ma_20 = series.rolling(window=20, min_periods=1).mean()
            std_20 = series.rolling(window=20, min_periods=1).std()
            df[f'{col}_bb_upper'] = ma_20 + (2 * std_20)
            df[f'{col}_bb_lower'] = ma_20 - (2 * std_20)
            df[f'{col}_bb_position'] = (series - df[f'{col}_bb_lower']) / (df[f'{col}_bb_upper'] - df[f'{col}_bb_lower'])
            
            # RSI (简化版本)
            delta = series.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss.replace(0, np.inf)
            df[f'{col}_rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _create_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建变化率特征"""
        df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in ['hour', 'day_of_week', 'month', 'year'] or '_change' in col:
                continue
            
            series = df[col]
            
            # 绝对变化
            for lag in [1, 7, 30]:
                df[f'{col}_change_{lag}d'] = series.diff(lag)
                df[f'{col}_pct_change_{lag}d'] = series.pct_change(lag)
            
            # 对数变化（对于价格类数据）
            if 'price' in col.lower() or 'tvl' in col.lower():
                log_series = np.log1p(series.clip(lower=0))
                df[f'{col}_log_change_1d'] = log_series.diff(1)
                df[f'{col}_log_change_7d'] = log_series.diff(7)
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建滚动统计特征"""
        df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        base_cols = [col for col in numeric_cols if not any(suffix in col for suffix in ['_ma_', '_std_', '_ema_', '_bb_', '_rsi', '_change'])]
        
        for col in base_cols:
            if col in ['hour', 'day_of_week', 'month', 'year']:
                continue
            
            series = df[col].fillna(method='ffill')
            
            # 滚动统计
            for window in [7, 14, 30]:
                df[f'{col}_min_{window}d'] = series.rolling(window=window, min_periods=1).min()
                df[f'{col}_max_{window}d'] = series.rolling(window=window, min_periods=1).max()
                df[f'{col}_quantile_25_{window}d'] = series.rolling(window=window, min_periods=1).quantile(0.25)
                df[f'{col}_quantile_75_{window}d'] = series.rolling(window=window, min_periods=1).quantile(0.75)
                df[f'{col}_skew_{window}d'] = series.rolling(window=window, min_periods=3).skew()
                df[f'{col}_kurt_{window}d'] = series.rolling(window=window, min_periods=4).kurt()
        
        return df
    
    def _create_cross_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建交叉特征"""
        df = df.copy()
        
        # TVL与Volume的比率
        if 'tvl_usd' in df.columns and 'volume_24h' in df.columns:
            df['tvl_volume_ratio'] = df['tvl_usd'] / (df['volume_24h'] + 1e-8)
        
        # Gas价格与网络活跃度的关系
        if 'gas_price_gwei' in df.columns and 'transaction_count' in df.columns:
            df['gas_tx_ratio'] = df['gas_price_gwei'] / (df['transaction_count'] + 1e-8)
        
        # APY与TVL的关系
        if 'apy' in df.columns and 'tvl_usd' in df.columns:
            df['risk_adjusted_yield'] = df['apy'] / np.log1p(df['tvl_usd'])
        
        return df
    
    def _create_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建异常值检测特征"""
        df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        base_cols = [col for col in numeric_cols if not any(suffix in col for suffix in ['_ma_', '_std_', '_ema_', '_bb_', '_rsi', '_change', '_min_', '_max_', '_quantile_'])]
        
        for col in base_cols:
            if col in ['hour', 'day_of_week', 'month', 'year']:
                continue
            
            series = df[col].fillna(method='ffill')
            
            # Z-score异常检测
            rolling_mean = series.rolling(window=30, min_periods=1).mean()
            rolling_std = series.rolling(window=30, min_periods=1).std()
            df[f'{col}_zscore'] = (series - rolling_mean) / (rolling_std + 1e-8)
            df[f'{col}_is_outlier'] = (np.abs(df[f'{col}_zscore']) > 3).astype(int)
            
            # IQR异常检测
            rolling_q1 = series.rolling(window=30, min_periods=1).quantile(0.25)
            rolling_q3 = series.rolling(window=30, min_periods=1).quantile(0.75)
            iqr = rolling_q3 - rolling_q1
            df[f'{col}_iqr_outlier'] = ((series < rolling_q1 - 1.5 * iqr) | (series > rolling_q3 + 1.5 * iqr)).astype(int)
        
        return df
    
    def scale_features(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """标准化特征"""
        df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_scale = [col for col in numeric_cols if col not in ['hour', 'day_of_week', 'month', 'year', 'is_weekend']]
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            logger.warning(f"Unknown scaling method: {method}, using standard")
            scaler = StandardScaler()
        
        if cols_to_scale:
            df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale].fillna(0))
            self.scalers[method] = scaler
        
        return df

# ---

# src/data/processors/data_validator.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DataValidator:
    """数据质量验证器"""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_dataset(self, df: pd.DataFrame, dataset_name: str = "unnamed") -> Dict[str, Any]:
        """验证整个数据集的质量"""
        validation_result = {
            'dataset_name': dataset_name,
            'timestamp': datetime.utcnow(),
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'validation_passed': True,
            'issues': []
        }
        
        if df.empty:
            validation_result['validation_passed'] = False
            validation_result['issues'].append("Dataset is empty")
            return validation_result
        
        # 基础数据质量检查
        validation_result.update(self._check_basic_quality(df))
        
        # 时间序列完整性检查
        if 'timestamp' in df.columns:
            validation_result.update(self._check_time_series_integrity(df))
        
        # 数值合理性检查
        validation_result.update(self._check_numeric_reasonableness(df))
        
        # 数据一致性检查
        validation_result.update(self._check_data_consistency(df))
        
        # 异常值检查
        validation_result.update(self._check_outliers(df))
        
        # 判断总体验证结果
        validation_result['validation_passed'] = len(validation_result['issues']) == 0
        
        self.validation_results[dataset_name] = validation_result
        
        if validation_result['validation_passed']:
            logger.info(f"Data validation passed for {dataset_name}")
        else:
            logger.warning(f"Data validation failed for {dataset_name}: {len(validation_result['issues'])} issues found")
        
        return validation_result
    
    def _check_basic_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """基础数据质量检查"""
        result = {
            'missing_data': {},
            'duplicate_rows': 0,
            'data_types': {}
        }
        
        # 缺失值检查
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        for col in df.columns:
            if missing_counts[col] > 0:
                result['missing_data'][col] = {
                    'count': int(missing_counts[col]),
                    'percentage': float(missing_percentages[col])
                }
                
                # 如果关键列缺失超过20%，标记为问题
                if col in ['timestamp', 'value', 'tvl_usd'] and missing_percentages[col] > 20:
                    result.setdefault('issues', []).append(
                        f"Critical column '{col}' has {missing_percentages[col]:.1f}% missing values"
                    )
        
        # 重复行检查
        result['duplicate_rows'] = df.duplicated().sum()
        if result['duplicate_rows'] > 0:
            result.setdefault('issues', []).append(
                f"Found {result['duplicate_rows']} duplicate rows"
            )
        
        # 数据类型检查
        for col in df.columns:
            result['data_types'][col] = str(df[col].dtype)
        
        return result
    
    def _check_time_series_integrity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """时间序列完整性检查"""
        result = {
            'time_range': {},
            'time_gaps': [],
            'frequency_analysis': {}
        }
        
        if 'timestamp' not in df.columns:
            return result
        
        try:
            timestamps = pd.to_datetime(df['timestamp']).sort_values()
            
            # 时间范围
            result['time_range'] = {
                'start': timestamps.min().isoformat(),
                'end': timestamps.max().isoformat(),
                'duration_days': (timestamps.max() - timestamps.min()).days
            }
            
            # 时间间隔分析
            intervals = timestamps.diff().dropna()
            if not intervals.empty:
                result['frequency_analysis'] = {
                    'median_interval_minutes': float(intervals.median().total_seconds() / 60),
                    'mean_interval_minutes': float(intervals.mean().total_seconds() / 60),
                    'std_interval_minutes': float(intervals.std().total_seconds() / 60)
                }
                
                # 检测异常大的时间间隔（可能的数据缺失）
                threshold = intervals.median() * 3
                large_gaps = intervals[intervals > threshold]
                
                if not large_gaps.empty:
                    result['time_gaps'] = [
                        {
                            'gap_start': (timestamps.iloc[i] - large_gaps.iloc[j]).isoformat(),
                            'gap_duration_hours': float(large_gaps.iloc[j].total_seconds() / 3600)
                        }
                        for j, i in enumerate(large_gaps.index)
                    ]
                    
                    if len(result['time_gaps']) > 5:
                        result.setdefault('issues', []).append(
                            f"Found {len(result['time_gaps'])} significant time gaps in data"
                        )
        
        except Exception as e:
            result.setdefault('issues', []).append(f"Error analyzing time series: {str(e)}")
        
        return result
    
    def _check_numeric_reasonableness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """数值合理性检查"""
        result = {
            'numeric_ranges': {},
            'zero_values': {},
            'negative_values': {}
        }
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            series = df[col].dropna()
            if series.empty:
                continue
            
            # 数值范围
            result['numeric_ranges'][col] = {
                'min': float(series.min()),
                'max': float(series.max()),
                'mean': float(series.mean()),
                'std': float(series.std())
            }
            
            # 零值检查
            zero_count = (series == 0).sum()
            if zero_count > 0:
                result['zero_values'][col] = {
                    'count': int(zero_count),
                    'percentage': float((zero_count / len(series)) * 100)
                }
            
            # 负值检查（对于应该为正的指标）
            if col in ['tvl_usd', 'volume_24h', 'gas_price_gwei']:
                negative_count = (series < 0).sum()
                if negative_count > 0:
                    result['negative_values'][col] = {
                        'count': int(negative_count),
                        'percentage': float((negative_count / len(series)) * 100)
                    }
                    result.setdefault('issues', []).append(
                        f"Column '{col}' has {negative_count} negative values (should be positive)"
                    )
            
            # 极值检查
            if col == 'tvl_usd' and series.max() > 1e12:  # 1万亿美元
                result.setdefault('issues', []).append(
                    f"TVL values seem unreasonably high (max: ${series.max():,.0f})"
                )
            
            if col == 'gas_price_gwei' and series.max() > 10000:  # 10000 Gwei
                result.setdefault('issues', []).append(
                    f"Gas prices seem unreasonably high (max: {series.max():.0f} Gwei)"
                )
        
        return result
    
    def _check_data_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """数据一致性检查"""
        result = {
            'consistency_checks': {}
        }
        
        # 检查TVL与Volume的关系
        if 'tvl_usd' in df.columns and 'volume_24h' in df.columns:
            valid_data = df[['tvl_usd', 'volume_24h']].dropna()
            if not valid_data.empty:
                # Volume通常不应该远大于TVL
                ratio = valid_data['volume_24h'] / (valid_data['tvl_usd'] + 1e-8)
                extreme_ratios = (ratio > 100).sum()  # Volume超过TVL 100倍
                
                result['consistency_checks']['tvl_volume_ratio'] = {
                    'extreme_ratio_count': int(extreme_ratios),
                    'median_ratio': float(ratio.median())
                }
                
                if extreme_ratios > len(valid_data) * 0.1:  # 超过10%的数据异常
                    result.setdefault('issues', []).append(
                        f"Volume to TVL ratios seem inconsistent in {extreme_ratios} records"
                    )
        
        # 检查协议名称一致性
        if 'protocol' in df.columns:
            protocol_variations = df['protocol'].value_counts()
            # 查找可能的重复协议（名称相似但不完全相同）
            protocol_names = set(protocol_variations.index)
            similar_protocols = []
            
            for proto1 in protocol_names:
                for proto2 in protocol_names:
                    if proto1 != proto2 and proto1 in proto2:
                        similar_protocols.append((proto1, proto2))
            
            if similar_protocols:
                result['consistency_checks']['similar_protocol_names'] = similar_protocols[:10]  # 限制数量
        
        return result
    
    def _check_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """异常值检查"""
        result = {
            'outliers': {}
        }
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            series = df[col].dropna()
            if len(series) < 10:  # 数据太少无法检测异常值
                continue
            
            # 使用IQR方法检测异常值
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            if not outliers.empty:
                result['outliers'][col] = {
                    'count': len(outliers),
                    'percentage': float((len(outliers) / len(series)) * 100),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'extreme_values': outliers.nlargest(5).tolist() + outliers.nsmallest(5).tolist()
                }
                
                # 如果异常值超过5%，标记为问题
                if len(outliers) / len(series) > 0.05:
                    result.setdefault('issues', []).append(
                        f"Column '{col}' has {len(outliers)} outliers ({len(outliers)/len(series)*100:.1f}%)"
                    )
        
        return result
    
    def generate_validation_report(self, dataset_name: str = None) -> str:
        """生成验证报告"""
        if dataset_name and dataset_name in self.validation_results:
            results = [self.validation_results[dataset_name]]
        else:
            results = list(self.validation_results.values())
        
        if not results:
            return "No validation results available"
        
        report_lines = ["=== DATA VALIDATION REPORT ===\n"]
        
        for result in results:
            report_lines.append(f"Dataset: {result['dataset_name']}")
            report_lines.append(f"Validation Status: {'PASSED' if result['validation_passed'] else 'FAILED'}")
            report_lines.append(f"Total Rows: {result['total_rows']:,}")
            report_lines.append(f"Total Columns: {result['total_columns']}")
            
            if result['issues']:
                report_lines.append("\nISSUES FOUND:")
                for issue in result['issues']:
                    report_lines.append(f"  • {issue}")
            
            if 'missing_data' in result and result['missing_data']:
                report_lines.append("\nMISSING DATA:")
                for col, info in result['missing_data'].items():
                    report_lines.append(f"  • {col}: {info['count']} ({info['percentage']:.1f}%)")
            
            if 'outliers' in result and result['outliers']:
                report_lines.append("\nOUTLIERS DETECTED:")
                for col, info in result['outliers'].items():
                    report_lines.append(f"  • {col}: {info['count']} ({info['percentage']:.1f}%)")
            
            report_lines.append("\n" + "="*50 + "\n")
        
        return "\n".join(report_lines)

# ---

# src/data/data_pipeline.py
import asyncio
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from src.data.collectors import (
    DefiLlamaCollector,
    EthereumGasCollector, 
    MacroDataCollector,
    AddressActivityCollector
)
from src.data.processors.data_cleaner import DataCleaner
from src.data.processors.feature_engineer import FeatureEngineer
from src.data.processors.data_validator import DataValidator
from src.data.storage.database import db_manager
from src.data.storage.cache import cache_manager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DataPipeline:
    """数据管道协调器"""
    
    def __init__(self):
        # 初始化收集器
        self.collectors = {
            'defi': DefiLlamaCollector(),
            'gas': EthereumGasCollector(),
            'macro': MacroDataCollector(),
            'address': AddressActivityCollector()
        }
        
        # 初始化处理器
        self.data_cleaner = DataCleaner()
        self.feature_engineer = FeatureEngineer()
        self.data_validator = DataValidator()
        
        # 管道统计
        self.pipeline_stats = {}
    
    async def run_full_pipeline(self, sources: List[str] = None) -> Dict[str, Any]:
        """运行完整的数据管道"""
        start_time = datetime.utcnow()
        logger.info("Starting full data pipeline execution")
        
        # 默认运行所有数据源
        if sources is None:
            sources = list(self.collectors.keys())
        
        pipeline_result = {
            'start_time': start_time,
            'sources_processed': [],
            'total_records_collected': 0,
            'total_records_processed': 0,
            'errors': [],
            'execution_time_seconds': 0
        }
        
        try:
            # 1. 数据收集阶段
            logger.info("Phase 1: Data Collection")
            collected_data = await self._collect_all_data(sources)
            
            for source, data in collected_data.items():
                if isinstance(data, list):
                    pipeline_result['total_records_collected'] += len(data)
                    pipeline_result['sources_processed'].append(source)
                else:
                    pipeline_result['errors'].append(f"Collection failed for {source}")
            
            # 2. 数据清洗阶段
            logger.info("Phase 2: Data Cleaning")
            cleaned_data = {}
            for source, raw_data in collected_data.items():
                if isinstance(raw_data, list) and raw_data:
                    cleaned = self.data_cleaner.clean_metrics_data(raw_data)
                    cleaned_data[source] = cleaned
                    logger.info(f"Cleaned {len(cleaned)} records from {source}")
            
            # 3. 数据存储阶段
            logger.info("Phase 3: Data Storage")
            storage_results = await self._store_data(cleaned_data)
            pipeline_result.update(storage_results)
            
            # 4. 特征工程阶段（对需要的数据）
            logger.info("Phase 4: Feature Engineering")
            engineered_data = {}
            for source, data in cleaned_data.items():
                if data:  # 只处理有数据的源
                    try:
                        df = self.feature_engineer.engineer_features(data)
                        if not df.empty:
                            engineered_data[source] = df
                            pipeline_result['total_records_processed'] += len(df)
                            logger.info(f"Engineered {len(df)} features for {source}")
                    except Exception as e:
                        logger.error(f"Feature engineering failed for {source}: {e}")
                        pipeline_result['errors'].append(f"Feature engineering failed for {source}: {str(e)}")
            
            # 5. 数据验证阶段
            logger.info("Phase 5: Data Validation")
            validation_results = {}
            for source, df in engineered_data.items():
                try:
                    validation_result = self.data_validator.validate_dataset(df, source)
                    validation_results[source] = validation_result
                    
                    if not validation_result['validation_passed']:
                        pipeline_result['errors'].extend([
                            f"Validation failed for {source}: {issue}" 
                            for issue in validation_result['issues']
                        ])
                except Exception as e:
                    logger.error(f"Validation failed for {source}: {e}")
                    pipeline_result['errors'].append(f"Validation failed for {source}: {str(e)}")
            
            # 6. 缓存结果
            logger.info("Phase 6: Caching Results")
            await self._cache_results(engineered_data, validation_results)
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            pipeline_result['errors'].append(f"Pipeline execution failed: {str(e)}")
        
        finally:
            end_time = datetime.utcnow()
            pipeline_result['end_time'] = end_time
            pipeline_result['execution_time_seconds'] = (end_time - start_time).total_seconds()
            
            # 更新统计信息
            self.pipeline_stats[start_time.isoformat()] = pipeline_result
            
            logger.info(f"Pipeline completed in {pipeline_result['execution_time_seconds']:.2f} seconds")
            logger.info(f"Processed {pipeline_result['total_records_processed']} records from {len(pipeline_result['sources_processed'])} sources")
            
            if pipeline_result['errors']:
                logger.warning(f"Pipeline completed with {len(pipeline_result['errors'])} errors")
        
        return pipeline_result
    
    async def _collect_all_data(self, sources: List[str]) -> Dict[str, Any]:
        """并行收集所有数据源的数据"""
        collection_tasks = []
        
        for source in sources:
            if source in self.collectors:
                collector = self.collectors[source]
                task = asyncio.create_task(
                    self._collect_with_retry(collector, source)
                )
                collection_tasks.append((source, task))
        
        collected_data = {}
        results = await asyncio.gather(*[task for _, task in collection_tasks], return_exceptions=True)
        
        for (source, _), result in zip(collection_tasks, results):
            if isinstance(result, Exception):
                logger.error(f"Collection failed for {source}: {result}")
                collected_data[source] = result
            else:
                collected_data[source] = result
                logger.info(f"Successfully collected data from {source}: {len(result) if isinstance(result, list) else 'N/A'} records")
        
        return collected_data
    
    async def _collect_with_retry(self, collector, source: str, max_retries: int = 3) -> List[Dict[str, Any]]:
        """带重试机制的数据收集"""
        for attempt in range(max_retries):
            try:
                # 检查缓存
                cache_key = f"collected_data:{source}:{datetime.utcnow().strftime('%Y-%m-%d-%H')}"
                cached_data = cache_manager.get(cache_key)
                
                if cached_data:
                    logger.info(f"Using cached data for {source}")
                    return cached_data
                
                # 收集新数据
                data = await collector.collect_data()
                
                # 缓存结果（1小时）
                if data:
                    cache_manager.set(cache_key, data, expire=3600)
                
                return data
                
            except Exception as e:
                logger.warning(f"Collection attempt {attempt + 1} failed for {source}: {e}")
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(2 ** attempt)  # 指数退避
        
        return []
    
    async def _store_data(self, cleaned_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """存储清洗后的数据"""
        storage_result = {
            'stored_records': 0,
            'storage_errors': []
        }
        
        for source, data in cleaned_data.items():
            if not data:
                continue
            
            try:
                # 根据数据类型选择存储表
                categorized_data = self._categorize_data_for_storage(data)
                
                for table_name, records in categorized_data.items():
                    if not records:
                        continue
                    
                    # 存储到数据库
                    success = False
                    if table_name == 'chain_metrics':
                        success = db_manager.insert_chain_metrics(records)
                    elif table_name == 'protocol_metrics':
                        success = db_manager.insert_protocol_metrics(records)
                    elif table_name == 'macro_indicators':
                        success = db_manager.insert_macro_indicators(records)
                    
                    if success:
                        storage_result['stored_records'] += len(records)
                        logger.info(f"Stored {len(records)} records in {table_name} from {source}")
                    else:
                        error_msg = f"Failed to store {len(records)} records in {table_name} from {source}"
                        storage_result['storage_errors'].append(error_msg)
                        logger.error(error_msg)
                        
            except Exception as e:
                error_msg = f"Storage failed for {source}: {str(e)}"
                storage_result['storage_errors'].append(error_msg)
                logger.error(error_msg)
        
        return storage_result
    
    def _categorize_data_for_storage(self, data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """根据数据类型分类数据以便存储"""
        categorized = {
            'chain_metrics': [],
            'protocol_metrics': [],
            'macro_indicators': []
        }
        
        for record in data:
            data_type = record.get('data_type', '')
            
            if data_type in ['chain_metrics', 'gas_metrics', 'block_metrics']:
                # 转换为chain_metrics表格式
                chain_record = {
                    'chain': record.get('chain', 'unknown'),
                    'timestamp': record.get('timestamp'),
                    'block_number': record.get('block_number'),
                    'transaction_count': record.get('transaction_count'),
                    'active_addresses': record.get('active_addresses'),
                    'gas_price_gwei': record.get('gas_price_gwei'),
                    'gas_used': record.get('gas_used'),
                    'tvl_usd': record.get('tvl_usd'),
                    'metadata': record.get('metadata', {})
                }
                categorized['chain_metrics'].append(chain_record)
                
            elif data_type in ['protocol_metrics', 'yield_metrics']:
                # 转换为protocol_metrics表格式
                protocol_record = {
                    'protocol': record.get('protocol', 'unknown'),
                    'chain': record.get('chain', 'unknown'),
                    'timestamp': record.get('timestamp'),
                    'tvl_usd': record.get('tvl_usd'),
                    'volume_24h': record.get('volume_24h'),
                    'users_24h': record.get('users_24h'),
                    'transactions_24h': record.get('transactions_24h'),
                    'yield_rate': record.get('apy', record.get('yield_rate')),
                    'metadata': record.get('metadata', {})
                }
                categorized['protocol_metrics'].append(protocol_record)
                
            elif data_type in ['macro_indicator', 'market_indicator', 'sentiment_indicator']:
                # 转换为macro_indicators表格式
                macro_record = {
                    'indicator': record.get('indicator', 'unknown'),
                    'timestamp': record.get('timestamp'),
                    'value': record.get('value'),
                    'source': record.get('source', 'unknown'),
                    'frequency': record.get('frequency', 'daily'),
                    'metadata': record.get('metadata', {})
                }
                categorized['macro_indicators'].append(macro_record)
        
        return categorized
    
    async def _cache_results(self, engineered_data: Dict[str, pd.DataFrame], validation_results: Dict[str, Any]):
        """缓存处理结果"""
        try:
            # 缓存工程化数据的摘要
            for source, df in engineered_data.items():
                if not df.empty:
                    summary = {
                        'shape': df.shape,
                        'columns': list(df.columns),
                        'last_timestamp': df['timestamp'].max().isoformat() if 'timestamp' in df.columns else None,
                        'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[pd.np.number]).columns) > 0 else {}
                    }
                    
                    cache_key = f"data_summary:{source}:{datetime.utcnow().strftime('%Y-%m-%d')}"
                    cache_manager.set(cache_key, summary, expire=86400)  # 24小时
            
            # 缓存验证结果
            cache_key = f"validation_results:{datetime.utcnow().strftime('%Y-%m-%d-%H')}"
            cache_manager.set(cache_key, validation_results, expire=3600)  # 1小时
            
            logger.info("Results cached successfully")
            
        except Exception as e:
            logger.error(f"Failed to cache results: {e}")
    
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
    
    async def run_incremental_update(self, sources: List[str] = None, hours_back: int = 24) -> Dict[str, Any]:
        """运行增量更新（只获取最近的数据）"""
        logger.info(f"Starting incremental update for last {hours_back} hours")
        
        # 这里可以实现更智能的增量更新逻辑
        # 比如检查数据库中的最新时间戳，只获取之后的数据
        
        # 目前简化为运行完整管道，但可以传递时间范围参数给收集器
        return await self.run_full_pipeline(sources)

# 全局管道实例
data_pipeline = DataPipeline()

# ---

# scripts/daily_update.py (更新版本)
#!/usr/bin/env python3
"""
每日数据更新脚本
可以通过cron或其他调度器定期运行
"""

import asyncio
import sys
import os
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_pipeline import data_pipeline
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

async def main():
    """主函数"""
    try:
        logger.info("=== Starting Daily Data Update ===")
        
        # 运行完整数据管道
        result = await data_pipeline.run_full_pipeline()
        
        # 输出结果摘要
        logger.info("=== Update Summary ===")
        logger.info(f"Sources processed: {', '.join(result['sources_processed'])}")
        logger.info(f"Total records collected: {result['total_records_collected']:,}")
        logger.info(f"Total records processed: {result['total_records_processed']:,}")
        logger.info(f"Execution time: {result['execution_time_seconds']:.2f} seconds")
        
        if result['errors']:
            logger.warning("=== Errors Encountered ===")
            for error in result['errors']:
                logger.warning(f"  • {error}")
        
        logger.info("=== Daily Update Completed ===")
        
        # 返回适当的退出码
        return 0 if not result['errors'] else 1
        
    except Exception as e:
        logger.error(f"Daily update failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)