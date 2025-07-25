from abc import ABC, abstractmethod
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
