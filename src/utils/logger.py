# -*- coding: utf-8 -*-
"""
日志工具模块
提供日志记录功能
"""

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(name: str = "ECOSight", 
                 log_dir: Optional[str] = None,
                 level: int = logging.INFO,
                 console: bool = True) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_dir: 日志目录路径
        level: 日志级别
        console: 是否输出到控制台
        
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台输出
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件输出
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"log_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class Timer:
    """
    计时器类
    
    用于测量代码执行时间
    
    Example:
        with Timer("数据处理"):
            # 处理代码
            pass
    """
    
    def __init__(self, name: str = "", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger("ECOSight")
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if self.name:
            self.logger.info(f"{self.name} 完成，耗时: {elapsed:.2f} 秒")
        else:
            self.logger.info(f"操作完成，耗时: {elapsed:.2f} 秒")


class ProgressTracker:
    """
    进度追踪器
    
    用于追踪长时间任务的进度
    
    Example:
        tracker = ProgressTracker(total=100, name="训练")
        for i in range(100):
            tracker.update(1)
    """
    
    def __init__(self, total: int, name: str = "", 
                 logger: Optional[logging.Logger] = None,
                 report_interval: int = 10):
        self.total = total
        self.name = name
        self.logger = logger or logging.getLogger("ECOSight")
        self.report_interval = report_interval
        self.current = 0
        self.start_time = datetime.now()
    
    def update(self, amount: int = 1):
        """更新进度"""
        self.current += amount
        if self.current % max(1, self.total // self.report_interval) == 0:
            self._report()
    
    def _report(self):
        """报告进度"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        progress = self.current / self.total * 100
        if self.current > 0:
            eta = elapsed / self.current * (self.total - self.current)
            self.logger.info(f"{self.name} 进度: {progress:.1f}% ({self.current}/{self.total}), ETA: {eta:.0f}s")
        else:
            self.logger.info(f"{self.name} 进度: {progress:.1f}%")
    
    def finish(self):
        """完成任务"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(f"{self.name} 完成，总耗时: {elapsed:.2f} 秒")
