import logging
import os
from typing import Optional


def get_logger(
    name: str = 'TIMR',
    log_file: Optional[str] = None,
    log_level: int = logging.INFO
) -> logging.Logger:
    logger = logging.getLogger(name)
    
    # 避免重复配置
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(log_level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger 