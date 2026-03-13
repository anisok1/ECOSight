#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LEF文件解析器模块

该模块用于解析LEF (Library Exchange Format) 文件，提取标准单元的引脚位置信息。
主要功能：
- 解析MACRO定义，获取单元名称和尺寸
- 提取输入/输出引脚的位置、层信息和尺寸
- 生成输入引脚和输出引脚的DataFrame

作者: ECOsight Team
"""

import re
import os
import pandas as pd
import numpy as np


class LEFParser:
    """
    LEF文件解析器类
    
    用于解析LEF格式的库文件，提取单元的引脚位置信息。
    
    属性:
        lef_path (str): LEF文件路径
        inpin_csv (DataFrame): 输入引脚信息表
        outpin_csv (DataFrame): 输出引脚信息表
        inpin_count (int): 输入引脚计数器
        outpin_count (int): 输出引脚计数器
    """
    
    def __init__(self, lef_path):
        """
        初始化LEF解析器
        
        参数:
            lef_path (str): LEF文件的完整路径
        """
        self.lef_path = lef_path
        self.inpin_csv = pd.DataFrame(
            columns=["cell_name", "inpin", "layer", "center_x", "center_y", "size", "by"]
        )
        self.outpin_csv = pd.DataFrame(
            columns=["cell_name", "outpin", "layer", "center_x", "center_y", "size", "by"]
        )
        self.inpin_count = 0
        self.outpin_count = 0
    
    def parse(self):
        """
        解析LEF文件
        
        读取LEF文件并提取所有单元的引脚位置信息。
        
        返回:
            tuple: (inpin_csv, outpin_csv) 输入引脚和输出引脚的DataFrame
        """
        with open(self.lef_path, "r") as f:
            content = f.readlines()
            i = 0
            while i < len(content) - 1:
                i += 1
                line = content[i].replace("\n", "").strip()
                
                # 检测MACRO定义开始
                if re.findall("^MACRO", line):
                    cell_name = line.split(" ")[-1]
                    
                    # 遍历MACRO内的内容
                    while True:
                        i += 1
                        line_1 = content[i].replace("\n", "").strip()
                        
                        # MACRO定义结束
                        if line_1 == f"END {cell_name}":
                            break
                        
                        # 获取原点坐标
                        if re.findall("^ORIGIN", line_1):
                            origin_matches = re.findall(
                                r"^ORIGIN\s+(-*\d*\.*\d+)\s+(-*\d*\.*\d+)\s*;", line_1
                            )[0]
                            origin_coordinate_x, origin_coordinate_y = origin_matches
                            if int(origin_coordinate_x) != 0 or int(origin_coordinate_y) != 0:
                                print(f"警告: 原点坐标非零: {origin_coordinate_x} {origin_coordinate_y}")
                        
                        # 获取单元尺寸
                        if re.findall("^SIZE", line_1):
                            size_matches = re.findall(
                                r"^SIZE\s+(\d*\.*\d+)\s+BY\s+(\d*\.*\d+)\s+;", line_1
                            )[0]
                            size, by = size_matches
                        
                        # 处理PIN定义
                        if re.findall("^PIN", line_1):
                            pin_name = re.findall("^PIN\s+(\w+)", line_1)[0]
                            center_pointx = 0
                            center_pointy = 0
                            
                            while True:
                                i += 1
                                line_2 = content[i].replace("\n", "").strip()
                                
                                # PIN定义结束
                                if re.findall(f"^END {pin_name}", line_2):
                                    break
                                
                                # 获取引脚方向
                                if re.findall("^DIRECTION", line_2):
                                    direction = re.findall("^DIRECTION\s+(\w+)\s+;", line_2)[0]
                                
                                # 跳过电源和地引脚
                                if re.findall("^USE", line_2):
                                    use = re.findall("^USE\s+(\w+)\s+;", line_2)[0]
                                    if use in ["POWER", "GROUND"]:
                                        break
                                
                                # 处理PORT定义（引脚几何信息）
                                if re.findall("^PORT", line_2):
                                    while True:
                                        i += 1
                                        line_3 = content[i].replace("\n", "").strip()
                                        
                                        if re.findall("^END", line_3):
                                            i -= 1
                                            break
                                        
                                        # 获取金属层信息
                                        if re.findall("^LAYER", line_3):
                                            metal = re.findall("^LAYER\s+(\w+)\s+;", line_3)[0]
                                            pointx_list = []
                                            pointy_list = []
                                            
                                            while True:
                                                i += 1
                                                line_4 = content[i].replace("\n", "").strip()
                                                
                                                # 当前层结束或新层开始
                                                if re.findall("^LAYER", line_4) or re.findall("^END", line_4):
                                                    i -= 1
                                                    break
                                                
                                                # 获取矩形坐标
                                                if re.findall("^RECT", line_4):
                                                    rect_matches = re.findall(
                                                        r"^RECT\s+(-*\d*\.*\d+)\s+(-*\d*\.*\d+)\s+"
                                                        r"(-*\d*\.*\d+)\s+(-*\d*\.*\d+)\s*;", line_4
                                                    )[0]
                                                    llx, lly, urx, ury = rect_matches
                                                    pointx = (float(llx) + float(urx)) / 2
                                                    pointy = (float(lly) + float(ury)) / 2
                                                    pointx_list.append(pointx)
                                                    pointy_list.append(pointy)
                                                else:
                                                    print(f"错误 RECT: {line_4}")
                                            
                                            # 计算中心点
                                            center_pointx = np.average(pointx_list)
                                            center_pointy = np.average(pointy_list)
                            
                            # 根据方向保存引脚信息
                            if direction == "INPUT":
                                self._add_inpin(
                                    cell_name, pin_name, metal, 
                                    center_pointx, center_pointy, 
                                    float(origin_coordinate_x), float(origin_coordinate_y),
                                    size, by
                                )
                            elif direction == "OUTPUT":
                                self._add_outpin(
                                    cell_name, pin_name, metal,
                                    center_pointx, center_pointy,
                                    float(origin_coordinate_x), float(origin_coordinate_y),
                                    size, by
                                )
                            elif direction == "INOUT":
                                continue
                            else:
                                print(f"错误 未知方向: {direction}")
        
        return self.inpin_csv, self.outpin_csv
    
    def _add_inpin(self, cell_name, pin_name, metal, cx, cy, ox, oy, size, by):
        """
        添加输入引脚信息到DataFrame
        
        参数:
            cell_name: 单元名称
            pin_name: 引脚名称
            metal: 金属层
            cx, cy: 中心坐标
            ox, oy: 原点坐标
            size, by: 单元尺寸
        """
        self.inpin_csv.loc[self.inpin_count, "cell_name"] = cell_name
        self.inpin_csv.loc[self.inpin_count, "inpin"] = pin_name
        self.inpin_csv.loc[self.inpin_count, "layer"] = metal
        self.inpin_csv.loc[self.inpin_count, "center_x"] = round(cx - ox, 3)
        self.inpin_csv.loc[self.inpin_count, "center_y"] = round(cy - oy, 3)
        self.inpin_csv.loc[self.inpin_count, "size"] = size
        self.inpin_csv.loc[self.inpin_count, "by"] = by
        self.inpin_count += 1
    
    def _add_outpin(self, cell_name, pin_name, metal, cx, cy, ox, oy, size, by):
        """
        添加输出引脚信息到DataFrame
        
        参数:
            cell_name: 单元名称
            pin_name: 引脚名称
            metal: 金属层
            cx, cy: 中心坐标
            ox, oy: 原点坐标
            size, by: 单元尺寸
        """
        self.outpin_csv.loc[self.outpin_count, "cell_name"] = cell_name
        self.outpin_csv.loc[self.outpin_count, "outpin"] = pin_name
        self.outpin_csv.loc[self.outpin_count, "layer"] = metal
        self.outpin_csv.loc[self.outpin_count, "center_x"] = round(cx - ox, 3)
        self.outpin_csv.loc[self.outpin_count, "center_y"] = round(cy - oy, 3)
        self.outpin_csv.loc[self.outpin_count, "size"] = size
        self.outpin_csv.loc[self.outpin_count, "by"] = by
        self.outpin_count += 1


def write_csv(df, df_path):
    """
    将DataFrame写入CSV文件
    
    如果文件不存在则创建，如果存在则追加数据。
    
    参数:
        df: 要写入的DataFrame
        df_path: CSV文件路径
    """
    if not os.path.isfile(df_path):
        df.to_csv(df_path, index=False)
    else:
        df_old = pd.read_csv(df_path)
        df_merge = pd.concat([df_old, df], ignore_index=True)
        df_merge.to_csv(df_path, index=False)


def delete_csv(df_path):
    """
    删除CSV文件
    
    参数:
        df_path: 要删除的CSV文件路径
    """
    if os.path.isfile(df_path):
        os.remove(df_path)


# 保持向后兼容的别名
lef_parser = LEFParser


if __name__ == "__main__":
    import time
    
    lef_list = [
        "../data/lef/demo_hvt.lef",
        "../data/lef/demo_lvt.lef",
        "../data/lef/demo_rvt.lef"
    ]
    inpin_path = "../Intermediate_data/lef/inpin_loc.csv"
    outpin_path = "../Intermediate_data/lef/outpin_loc.csv"

    start_time = time.time()

    delete_csv(inpin_path)
    delete_csv(outpin_path)

    for lef in lef_list:
        parser = LEFParser(lef)
        inpin_csv, outpin_csv = parser.parse()
        write_csv(inpin_csv, inpin_path)
        write_csv(outpin_csv, outpin_path)

    elapsed_time = time.time() - start_time
    print(f"处理完成，耗时: {elapsed_time:.2f} 秒")
