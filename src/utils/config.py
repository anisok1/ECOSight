# -*- coding: utf-8 -*-
"""
配置管理模块
提供全局配置和路径管理功能
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """
    ECOSight 全局配置类
    
    管理项目的所有路径和参数配置
    
    Attributes:
        project_root: 项目根目录
        data_dir: 数据目录
        intermediate_dir: 中间数据目录
        model_dir: 模型保存目录
        output_dir: 输出目录
        log_dir: 日志目录
        design: 当前设计名称
    """
    
    # 项目根目录（自动检测）
    project_root: str = None
    
    # 数据目录
    data_dir: str = "data"
    
    # 中间数据目录
    intermediate_dir: str = "intermediate"
    
    # 模型保存目录
    model_dir: str = "models"
    
    # 输出目录
    output_dir: str = "output"
    
    # 日志目录
    log_dir: str = "logs"
    
    # 当前设计名称
    design: str = "b05"
    
    def __post_init__(self):
        """初始化后自动设置项目根目录"""
        if self.project_root is None:
            # 自动检测项目根目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.project_root = os.path.dirname(os.path.dirname(current_dir))
    
    @property
    def data_path(self) -> str:
        """获取数据目录绝对路径"""
        return os.path.join(self.project_root, self.data_dir)
    
    @property
    def intermediate_path(self) -> str:
        """获取中间数据目录绝对路径"""
        return os.path.join(self.project_root, self.intermediate_dir)
    
    @property
    def model_path(self) -> str:
        """获取模型目录绝对路径"""
        return os.path.join(self.project_root, self.model_dir)
    
    @property
    def output_path(self) -> str:
        """获取输出目录绝对路径"""
        return os.path.join(self.project_root, self.output_dir)
    
    @property
    def log_path(self) -> str:
        """获取日志目录绝对路径"""
        return os.path.join(self.project_root, self.log_dir)
    
    def get_design_data_path(self, design: Optional[str] = None) -> str:
        """
        获取设计数据路径
        
        Args:
            design: 设计名称，默认使用当前设计
            
        Returns:
            设计数据目录路径
        """
        design = design or self.design
        return os.path.join(self.data_path, "design", design)
    
    def get_design_intermediate_path(self, design: Optional[str] = None) -> str:
        """
        获取设计中间数据路径
        
        Args:
            design: 设计名称，默认使用当前设计
            
        Returns:
            设计中间数据目录路径
        """
        design = design or self.design
        return os.path.join(self.intermediate_path, design)
    
    def get_def_path(self, design: Optional[str] = None) -> str:
        """获取DEF文件路径"""
        return os.path.join(self.get_design_data_path(design), "def", f"{design or self.design}_incr.def")
    
    def get_netlist_path(self, design: Optional[str] = None) -> str:
        """获取网表文件路径"""
        return os.path.join(self.get_design_data_path(design), "netlist", f"{design or self.design}.v")
    
    def get_timing_report_path(self, design: Optional[str] = None) -> str:
        """获取时序报告路径"""
        return os.path.join(self.get_design_data_path(design), "timing_report", "setup.wo_io_all.rpt")
    
    def get_model_file(self, design: Optional[str] = None) -> str:
        """获取模型文件路径"""
        design = design or self.design
        return os.path.join(self.model_path, f"{design}_model.pth")
    
    def get_lef_paths(self) -> list:
        """获取LEF文件路径列表"""
        lef_dir = os.path.join(self.data_path, "lef")
        return [
            os.path.join(lef_dir, "demo_hvt.lef"),
            os.path.join(lef_dir, "demo_lvt.lef"),
            os.path.join(lef_dir, "demo_rvt.lef")
        ]
    
    def get_tech_lef_path(self) -> str:
        """获取Tech LEF文件路径"""
        return os.path.join(self.data_path, "lef", "demo_tech.lef")
    
    def get_lib_paths(self) -> list:
        """获取LIB文件路径列表"""
        lib_dir = os.path.join(self.data_path, "lib")
        return [
            os.path.join(lib_dir, "demo_hvt.lib"),
            os.path.join(lib_dir, "demo_lvt.lib"),
            os.path.join(lib_dir, "demo_rvt.lib")
        ]
    
    def ensure_directories(self):
        """确保所有必要目录存在"""
        for path in [self.intermediate_path, self.model_path, self.output_path, self.log_path]:
            os.makedirs(path, exist_ok=True)
        # 确保设计的中间数据目录存在
        os.makedirs(self.get_design_intermediate_path(), exist_ok=True)


# 默认配置实例
default_config = Config()
