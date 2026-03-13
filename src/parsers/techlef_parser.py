#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tech LEF文件解析器模块

该模块用于解析Technology LEF文件，提取工艺信息。
主要功能：
- 解析UNITS信息
- 提取MANUFACTURINGGRID信息
- 提取金属层信息（PITCH, WIDTH）
- 提取VIA信息

作者: ECOsight Team
"""

import re
import os
import pandas as pd
from typing import Dict, Tuple, Optional


class TechLefParser:
    """
    Tech LEF文件解析器类
    
    用于解析Technology LEF格式的文件，提取工艺相关参数。
    
    属性:
        tech_lef_path (str): Tech LEF文件路径
        unit (str): 设计单位
        grid (str): 制造网格
        metal_dict (Dict[str, float]): 金属层宽度字典
        via_df (pd.DataFrame): VIA信息表
    """
    
    def __init__(self, tech_lef_path: str):
        """
        初始化Tech LEF解析器
        
        参数:
            tech_lef_path: Tech LEF文件路径
        """
        self.tech_lef_path = tech_lef_path
        self.unit: str = ""
        self.grid: str = ""
        self.metal_dict: Dict[str, float] = {}
        self.via_df: pd.DataFrame = pd.DataFrame(
            columns=["via_name", "ME1", "ME2", "ME3", "ME4", "ME5", "ME6", "ME7", "ME8"]
        )
    
    def parse(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        解析Tech LEF文件
        
        参数:
            output_path: 输出CSV文件路径（可选）
        
        返回:
            VIA信息DataFrame
        """
        cnt = 0
        
        with open(self.tech_lef_path, "r") as f:
            content = f.readlines()
            i = 0
            
            while i < len(content):
                line = content[i]
                
                # 获取单位信息
                if re.findall("^UNITS", line):
                    unit_pattern = r"^\s*DATABASE\s+MICRONS\s+(\d+)\s*;"
                    self.unit = re.findall(unit_pattern, content[i + 1])[0]
                
                # 获取制造网格
                elif re.findall("^MANUFACTURINGGRID\s*(\d*\.*\d+)\s*;", line):
                    self.grid = re.findall("^MANUFACTURINGGRID\s*(\d*\.*\d+)\s*;", line)[0]
                
                # 获取金属层信息
                elif re.findall("^LAYER\s+ME\d+", line):
                    metal = re.findall("^LAYER\s+(ME\d+)", line)[0]
                    pitch = ""
                    width = ""
                    
                    while True:
                        i += 1
                        if re.findall("^END\s+{}".format(metal), content[i]):
                            break
                        else:
                            # 获取间距
                            if re.findall(r"^\s*PITCH\s+(\d*\.*\d+)\s*;", content[i]):
                                pitch = re.findall(r"^\s*PITCH\s+(\d*\.*\d+)\s*;", content[i])[0]
                            
                            # 获取宽度
                            elif re.findall(r"^\s*WIDTH\s+(\d*\.*\d+)\s*;", content[i]):
                                width = re.findall(r"^\s*WIDTH\s+(\d*\.*\d+)\s*;", content[i])[0]
                                self.metal_dict[metal] = float(width)
                
                # 获取VIA信息
                elif re.findall("^VIA\s+(.*)", content[i]):
                    via_name = re.findall("^VIA\s+(.*)", content[i].replace("\n", ""))[0]
                    
                    # 处理DEFAULT关键字
                    if "DEFAULT" in via_name:
                        via_name = via_name.replace("DEFAULT", "")
                    via_name = via_name.strip()
                    
                    self.via_df.loc[cnt, "via_name"] = via_name
                    self.via_df.iloc[cnt, 1:] = 0
                    
                    while True:
                        i += 1
                        if re.findall("^END\s+{}".format(via_name), content[i]):
                            break
                        else:
                            # 获取VIA连接的金属层
                            if re.findall(r"LAYER\s+(ME\d+)\s*;", content[i]):
                                metal = re.findall(r"LAYER\s+(ME\d+)\s*;", content[i])[0]
                            
                            # 获取VIA的矩形坐标
                            elif re.findall(
                                r"^\s+RECT\s+-*(\d*\.*\d+)\s+-*(\d*\.*\d+)\s+-*(\d*\.*\d+)\s+-*(\d*\.*\d+)\s*;",
                                content[i]
                            ):
                                rect_pattern = r"^\s+RECT\s+(-*\d*\.*\d+)\s+(-*\d*\.*\d+)\s+(-*\d*\.*\d+)\s+(-*\d*\.*\d+)\s*;"
                                self.via_df.loc[cnt, metal] = re.findall(rect_pattern, content[i])[0]
                    
                    cnt += 1
                
                i += 1
        
        # 保存到CSV文件
        if output_path:
            self.via_df.to_csv(output_path, index=False)
        
        return self.via_df
    
    def get_unit(self) -> str:
        """获取设计单位"""
        return self.unit
    
    def get_grid(self) -> str:
        """获取制造网格"""
        return self.grid
    
    def get_metal_width(self, metal_name: str) -> Optional[float]:
        """
        获取指定金属层的宽度
        
        参数:
            metal_name: 金属层名称
        
        返回:
            金属层宽度，如果不存在则返回None
        """
        return self.metal_dict.get(metal_name)
    
    def get_all_metal_info(self) -> Dict[str, float]:
        """获取所有金属层的宽度信息"""
        return self.metal_dict


def TechLefExtraction(tech_lef: str, design: str, output_dir: str = None) -> pd.DataFrame:
    """
    Tech LEF信息提取函数（兼容旧接口）
    
    从Tech LEF文件中提取VIA信息并保存到CSV文件。
    
    参数:
        tech_lef: Tech LEF文件路径
        design: 设计名称
        output_dir: 输出目录（可选）
    
    返回:
        VIA信息DataFrame
    """
    parser = TechLefParser(tech_lef)
    
    if output_dir is None:
        output_dir = "../Intermediate_data"
    
    output_path = os.path.join(output_dir, "via_info.csv")
    return parser.parse(output_path)


if __name__ == "__main__":
    import time
    
    tech_lef_path = "../data/lef/demo_tech.lef"
    design = "b05"
    
    start_time = time.time()
    
    # 使用类方式解析
    parser = TechLefParser(tech_lef_path)
    via_df = parser.parse("../Intermediate_data/via_info.csv")
    
    # 或使用兼容函数
    # via_df = TechLefExtraction(tech_lef_path, design)
    
    elapsed_time = time.time() - start_time
    
    print(f"解析完成，耗时: {elapsed_time:.2f} 秒")
    print(f"单位: {parser.get_unit()}")
    print(f"制造网格: {parser.get_grid()}")
    print(f"金属层信息: {parser.get_all_metal_info()}")
    print(f"VIA数量: {len(via_df)}")
