# src/utils/logger.py
"""
日志模块 - 简化版本
"""
import logging
import sys
from pathlib import Path

def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """设置日志器"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 如果logs目录存在，也添加文件处理器
    logs_dir = Path('logs')
    if logs_dir.exists():
        try:
            file_handler = logging.FileHandler(logs_dir / f"{name}.log")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception:
            # 如果文件处理器创建失败，只使用控制台处理器
            pass
    
    return logger