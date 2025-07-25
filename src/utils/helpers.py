from typing import Any, Dict, List
import pandas as pd
from datetime import datetime

def validate_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """验证数据格式"""
    valid_data = []
    for item in data:
        if isinstance(item, dict) and 'timestamp' in item:
            valid_data.append(item)
    return valid_data

def format_timestamp(timestamp: Any) -> datetime:
    """格式化时间戳"""
    if isinstance(timestamp, str):
        return pd.to_datetime(timestamp)
    elif isinstance(timestamp, (int, float)):
        return datetime.fromtimestamp(timestamp)
    return timestamp
