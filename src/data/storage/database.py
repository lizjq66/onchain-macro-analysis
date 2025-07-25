# src/data/storage/database.py
"""
数据库管理器 - 简化版本
"""
from datetime import datetime
from typing import List, Dict, Any, Optional

class SimpleDatabaseManager:
    """简化的数据库管理器（内存存储）"""
    
    def __init__(self):
        # 使用内存存储模拟数据库
        self.data_store = {
            'chain_metrics': [],
            'protocol_metrics': [],
            'macro_indicators': [],
            'network_analysis': [],
            'prediction_results': []
        }
        self.created_tables = False
    
    def create_tables(self):
        """创建表（模拟）"""
        self.created_tables = True
        print("✅ Database tables created (in-memory simulation)")
        return True
    
    def get_session(self):
        """获取数据库会话（模拟）"""
        return self
    
    def insert_chain_metrics(self, metrics: List[Dict[str, Any]]) -> bool:
        """插入链指标数据"""
        try:
            if not self.created_tables:
                self.create_tables()
            
            for metric in metrics:
                # 添加插入时间戳
                metric_copy = metric.copy()
                metric_copy['inserted_at'] = datetime.utcnow()
                self.data_store['chain_metrics'].append(metric_copy)
            
            print(f"✅ Inserted {len(metrics)} chain metrics records")
            return True
        except Exception as e:
            print(f"❌ Error inserting chain metrics: {e}")
            return False
    
    def insert_protocol_metrics(self, metrics: List[Dict[str, Any]]) -> bool:
        """插入协议指标数据"""
        try:
            if not self.created_tables:
                self.create_tables()
            
            for metric in metrics:
                metric_copy = metric.copy()
                metric_copy['inserted_at'] = datetime.utcnow()
                self.data_store['protocol_metrics'].append(metric_copy)
            
            print(f"✅ Inserted {len(metrics)} protocol metrics records")
            return True
        except Exception as e:
            print(f"❌ Error inserting protocol metrics: {e}")
            return False
    
    def insert_macro_indicators(self, indicators: List[Dict[str, Any]]) -> bool:
        """插入宏观指标数据"""
        try:
            if not self.created_tables:
                self.create_tables()
            
            for indicator in indicators:
                indicator_copy = indicator.copy()
                indicator_copy['inserted_at'] = datetime.utcnow()
                self.data_store['macro_indicators'].append(indicator_copy)
            
            print(f"✅ Inserted {len(indicators)} macro indicators records")
            return True
        except Exception as e:
            print(f"❌ Error inserting macro indicators: {e}")
            return False
    
    def get_latest_metrics(self, table_name: str, limit: int = 100) -> List[Dict]:
        """获取最新指标"""
        try:
            if table_name not in self.data_store:
                return []
            
            # 返回最新的记录
            data = self.data_store[table_name]
            if not data:
                return []
            
            # 按时间戳排序（如果有的话）
            try:
                sorted_data = sorted(data, key=lambda x: x.get('timestamp', datetime.min), reverse=True)
                return sorted_data[:limit]
            except:
                return data[-limit:]  # 如果排序失败，返回最后的记录
            
        except Exception as e:
            print(f"❌ Error getting latest metrics: {e}")
            return []
    
    def get_data_summary(self) -> Dict[str, Any]:
        """获取数据摘要"""
        summary = {
            'tables': {},
            'total_records': 0,
            'created_tables': self.created_tables
        }
        
        for table_name, data in self.data_store.items():
            record_count = len(data)
            summary['tables'][table_name] = record_count
            summary['total_records'] += record_count
        
        return summary
    
    def clear_all_data(self):
        """清空所有数据（用于测试）"""
        for table_name in self.data_store:
            self.data_store[table_name] = []
        print("🗑️ All data cleared")
    
    def _row_to_dict(self, row) -> Dict:
        """将行对象转换为字典（兼容方法）"""
        if isinstance(row, dict):
            return row
        else:
            # 如果是其他类型，尝试转换
            return {'data': str(row)}

# 创建全局数据库管理器实例
db_manager = SimpleDatabaseManager()