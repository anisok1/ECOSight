#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LIB文件解析器模块

该模块用于解析LIB (Library) 文件，提取标准单元的时序信息。
主要功能：
- 解析单元定义和属性
- 提取输入引脚电容信息
- 提取时序查找表（延迟和转换时间）

作者: ECOsight Team
"""

import re
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


def get_brace_end(content: List[str], start: int) -> int:
    """
    获取大括号块的结束位置
    
    参数:
        content: 文件内容列表
        start: 起始位置
    
    返回:
        结束位置的索引
    """
    flag = 0
    i = start
    while True:
        if re.findall(r"\{", content[i]):
            flag += 1
        if re.findall(r"\}", content[i]):
            flag -= 1
        if flag == 0:
            return i
        i += 1


def get_complete_line(content: List[str], start: int) -> str:
    """
    获取完整的语句行
    
    LIB文件中有些语句可能跨越多行，此函数用于将其合并。
    
    参数:
        content: 文件内容列表
        start: 起始位置
    
    返回:
        完整的语句行
    """
    line = ""
    i = start
    while True:
        if re.findall(";", content[i]):
            line += content[i]
            break
        else:
            line += content[i]
        i += 1
    line = line.replace("\n", "").replace("\\", "")
    return line


class InputPin:
    """
    输入引脚类
    
    存储输入引脚的电容信息。
    """
    
    def __init__(self, name: str):
        """
        初始化输入引脚
        
        参数:
            name: 引脚名称
        """
        self.name = name
        self.capacitance = 0.0
    
    def get_capacitance(self, content: List[str], start: int, end: int) -> bool:
        """
        从文件内容中提取电容值
        
        参数:
            content: 文件内容列表
            start: 起始位置
            end: 结束位置
        
        返回:
            是否成功提取电容值
        """
        for j in range(start, end + 1):
            pattern = r"^capacitance:(\d*\.\*\d+);"
            if re.findall(pattern, content[j].replace(" ", "")):
                self.capacitance = re.findall(
                    pattern, content[j].replace(" ", "")
                )[0]
                return True
        return False


class OutputPin:
    """
    输出引脚类
    
    存储输出引脚的功能和时序信息。
    """
    
    def __init__(self, name: str):
        """
        初始化输出引脚
        
        参数:
            name: 引脚名称
        """
        self.name = name
        self.function = ""
        self.timing_table: List['TimingInfo'] = []
    
    def get_function(self, content: List[str], start: int, end: int) -> None:
        """
        提取引脚功能表达式
        
        参数:
            content: 文件内容列表
            start: 起始位置
            end: 结束位置
        """
        for j in range(start, end + 1):
            pattern = r'^function:"(.*?)";'
            if re.findall(pattern, content[j].replace(" ", "")):
                self.function = re.findall(
                    pattern, content[j].replace(" ", "")
                )[0]
                break
    
    def get_timing_info(self, content: List[str], start: int, end: int) -> None:
        """
        提取时序信息
        
        参数:
            content: 文件内容列表
            start: 起始位置
            end: 结束位置
        """
        for j in range(start, end + 1):
            pattern = r"^timing\(\)\{"
            if re.findall(pattern, content[j].replace(" ", "")):
                end_timing = get_brace_end(content, j)
                timing_info = TimingInfo()
                
                for k in range(j, end_timing + 1):
                    line = content[k].replace(" ", "")
                    
                    # 提取相关引脚
                    if re.findall(r'^related_pin:(.*?);', line):
                        timing_info.get_related_pin(content[k])
                    
                    # 提取时序感知
                    if re.findall(r'^timing_sense:(.*?);', line):
                        timing_info.get_timing_sense(content[k])
                    
                    # 提取条件
                    if re.findall(r'^when:(.*?);', line):
                        timing_info.get_when(content[k])
                    
                    # 提取时序表
                    if re.findall(r"^cell_rise\(.*?\)\{", line):
                        timing_info.get_timing_table(content, k, "cell_rise")
                    if re.findall(r"^cell_fall\(.*?\)\{", line):
                        timing_info.get_timing_table(content, k, "cell_fall")
                    if re.findall(r"^rise_transition\(.*?\)\{", line):
                        timing_info.get_timing_table(content, k, "rise_transition")
                    if re.findall(r"^fall_transition\(.*?\)\{", line):
                        timing_info.get_timing_table(content, k, "fall_transition")
                
                self.timing_table.append(timing_info)


class TimingInfo:
    """
    时序信息类
    
    存储单元的时序查找表数据。
    """
    
    def __init__(self, table_size: int = 8):
        """
        初始化时序信息
        
        参数:
            table_size: 查找表大小（默认8x8）
        """
        self.num = table_size
        self.related_pin = ""
        self.timing_sense = ""
        self.when = ""
        
        # 上升延迟表
        self.cell_rise_trans = ["0" for _ in range(table_size)]
        self.cell_rise_loads = ["0" for _ in range(table_size)]
        self.cell_rise_values = ["0" for _ in range(table_size ** 2)]
        
        # 下降延迟表
        self.cell_fall_trans = ["0" for _ in range(table_size)]
        self.cell_fall_loads = ["0" for _ in range(table_size)]
        self.cell_fall_values = ["0" for _ in range(table_size ** 2)]
        
        # 上升转换时间表
        self.rise_transition_trans = ["0" for _ in range(table_size)]
        self.rise_transition_loads = ["0" for _ in range(table_size)]
        self.rise_transition_values = ["0" for _ in range(table_size ** 2)]
        
        # 下降转换时间表
        self.fall_transition_trans = ["0" for _ in range(table_size)]
        self.fall_transition_loads = ["0" for _ in range(table_size)]
        self.fall_transition_values = ["0" for _ in range(table_size ** 2)]
    
    def get_related_pin(self, line: str) -> None:
        """提取相关引脚名称"""
        self.related_pin = re.findall(
            r'^related_pin:(.*?);', line.replace(" ", "")
        )[0].replace("\"", "").replace("\'", "")
    
    def get_timing_sense(self, line: str) -> None:
        """提取时序感知类型"""
        self.timing_sense = re.findall(
            r'^timing_sense:(.*?);', line.replace(" ", "")
        )[0]
    
    def get_when(self, line: str) -> None:
        """提取条件表达式"""
        self.when = re.findall(
            r'^when:(.*?);', line.replace(" ", "")
        )[0].replace("\"", "").replace("\'", "")
    
    def get_timing_table(self, content: List[str], start: int, table_type: str) -> int:
        """
        提取时序查找表
        
        参数:
            content: 文件内容列表
            start: 起始位置
            table_type: 表类型 (cell_rise/cell_fall/rise_transition/fall_transition)
        
        返回:
            是否成功提取
        """
        end = get_brace_end(content, start)
        
        for j in range(start, end + 1):
            # 提取index_1 (转换时间)
            if re.findall("^index_1", content[j].replace(" ", "")):
                line = get_complete_line(content, j)
                line = line.replace(" ", "").replace("'", "").replace("\"", "")
                index1 = re.findall(r'^index_1\((.*?)\);', line)[0].split(",")
                
                # 填充或截断索引
                if len(index1) < self.num:
                    index1 = index1 + ["0"] * (self.num - len(index1))
                if len(index1) > self.num:
                    index1 = index1[:self.num]
                
                if table_type == "cell_rise":
                    self.cell_rise_trans = index1
                elif table_type == "cell_fall":
                    self.cell_fall_trans = index1
                elif table_type == "rise_transition":
                    self.rise_transition_trans = index1
                elif table_type == "fall_transition":
                    self.fall_transition_trans = index1
            
            # 提取index_2 (负载)
            if re.findall("^index_2", content[j].replace(" ", "")):
                line = get_complete_line(content, j)
                line = line.replace(" ", "").replace("'", "").replace("\"", "")
                index2 = re.findall(r'^index_2\((.*?)\);', line)[0].split(",")
                
                if len(index2) < self.num:
                    index2 = index2 + ["0"] * (self.num - len(index2))
                if len(index2) > self.num:
                    index2 = index2[:self.num]
                
                if table_type == "cell_rise":
                    self.cell_rise_loads = index2
                elif table_type == "cell_fall":
                    self.cell_fall_loads = index2
                elif table_type == "rise_transition":
                    self.rise_transition_loads = index2
                elif table_type == "fall_transition":
                    self.fall_transition_loads = index2
            
            # 提取values (延迟值)
            if re.findall("^values", content[j].replace(" ", "")):
                line = get_complete_line(content, j)
                line = line.replace(" ", "").replace("'", "").replace("\"", "")
                values = re.findall(r'^values\((.*?)\);', line)[0].split(",")
                
                # 调整values大小
                if len(values) < self.num ** 2:
                    shape = int(len(values) ** 0.5)
                    reshape_matrix = np.array(values).reshape((shape, shape))
                    expanded_matrix = np.zeros((self.num, self.num))
                    expanded_matrix[:shape, :shape] = reshape_matrix
                    values = expanded_matrix.flatten().tolist()
                if len(values) > self.num ** 2:
                    shape = int(len(values) ** 0.5)
                    original_matrix = np.array(values).reshape((shape, shape))
                    cropped_matrix = original_matrix[:-1, :-1]
                    values = cropped_matrix.flatten().tolist()
                
                if table_type == "cell_rise":
                    self.cell_rise_values = values
                elif table_type == "cell_fall":
                    self.cell_fall_values = values
                elif table_type == "rise_transition":
                    self.rise_transition_values = values
                elif table_type == "fall_transition":
                    self.fall_transition_values = values
                
                return 1
        
        return 0


class Cell:
    """
    单元类
    
    存储标准单元的属性和引脚信息。
    """
    
    def __init__(self, cell_name: str):
        """
        初始化单元
        
        参数:
            cell_name: 单元名称
        """
        self.name = cell_name
        self.footprint = ""
        self.area = ""
        self.inpins: List[InputPin] = []
        self.outpins: List[OutputPin] = []
    
    def get_cell_footprint(self, line: str) -> None:
        """提取单元封装信息"""
        self.footprint = re.findall(r'^cell_footprint:"(.*?)";', line)[0]
    
    def get_area(self, line: str) -> None:
        """提取单元面积"""
        self.area = re.findall(r'^area:(\d*\.\*\d+);', line)[0]
    
    def get_pin(self, pin_name: str, content: List[str], start: int) -> None:
        """
        提取引脚信息
        
        参数:
            pin_name: 引脚名称
            content: 文件内容列表
            start: 起始位置
        """
        end = get_brace_end(content, start)
        
        for j in range(start, end + 1):
            # 输入引脚
            if re.findall(r'^direction.*?input.*?', content[j].replace(" ", "")):
                inpin = InputPin(pin_name)
                flag = inpin.get_capacitance(content, start, end)
                if flag:
                    self.inpins.append(inpin)
                break
            
            # 输出引脚
            if re.findall(r'^direction.*?output.*?', content[j].replace(" ", "")):
                outpin = OutputPin(pin_name)
                outpin.get_function(content, start, end)
                outpin.get_timing_info(content, start, end)
                self.outpins.append(outpin)
                break


class LibParser:
    """
    LIB文件解析器类
    
    用于解析LIB格式的库文件，提取单元的时序信息。
    
    属性:
        file_path (str): LIB文件路径
        footprint_dict_driving (Dict): 单元封装到驱动强度的映射
    """
    
    def __init__(self, file_path: str, footprint_dict_driving: Dict = None):
        """
        初始化LIB解析器
        
        参数:
            file_path: LIB文件路径
            footprint_dict_driving: 单元封装字典（可选）
        """
        self.file_path = file_path
        self.footprint_dict_driving = footprint_dict_driving or {}
    
    def parse(self, table_size: int = 8) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        解析LIB文件
        
        参数:
            table_size: 查找表大小
        
        返回:
            (输入引脚电容表, 延迟表, 转换时间表)
        """
        # 创建DataFrame
        df_inpin = pd.DataFrame(columns=["std_cell", "inpin", "cap"])
        
        trans_cols = [f"tran{i}" for i in range(1, table_size + 1)]
        load_cols = [f"load{i}" for i in range(1, table_size + 1)]
        value_cols = [f"value{i}" for i in range(1, table_size ** 2 + 1)]
        
        df_delay = pd.DataFrame(
            columns=["std_cell", "inpin", "outpin", "rf", "when"] + trans_cols + load_cols + value_cols
        )
        df_trans = pd.DataFrame(
            columns=["std_cell", "inpin", "outpin", "rf", "when"] + trans_cols + load_cols + value_cols
        )
        
        cnt_inpin = 0
        cnt_delay = 0
        cnt_trans = 0
        
        with open(self.file_path, "r") as f:
            content = f.readlines()
            file_end = len(content)
            i = 0
            
            while i < file_end:
                # 检测cell定义
                pattern = r"^cell\(.*?\)\{"
                if re.findall(pattern, content[i].replace(" ", "")):
                    cell_name = re.findall(
                        r"^cell\((.*?)\)\{", content[i].replace(" ", "")
                    )[0]
                    
                    # 创建单元实例
                    cell = Cell(cell_name)
                    
                    # 保存驱动强度
                    cell_type = cell_name.split("_")[0]
                    if cell_type in self.footprint_dict_driving:
                        self.footprint_dict_driving[cell_type].append(cell.name)
                    else:
                        self.footprint_dict_driving[cell_type] = [cell.name]
                    
                    end = get_brace_end(content, i)
                    
                    for j in range(i, end + 1):
                        # 提取单元属性
                        if re.findall(r'^cell_footprint:"(.*?)";', content[j].replace(" ", "")):
                            cell.get_cell_footprint(content[j].replace(" ", ""))
                        
                        if re.findall(r'^area:(\d*\.\*\d+);', content[j].replace(" ", "")):
                            cell.get_area(content[j].replace(" ", ""))
                        
                        # 提取引脚信息
                        pin_pattern = r"^[pin|bus]*\((.*?)\)\{"
                        if re.findall(pin_pattern, content[j].replace(" ", "")):
                            pin_name = re.findall(
                                pin_pattern, content[j].replace(" ", "")
                            )[0]
                            cell.get_pin(pin_name, content, j)
                    
                    i = end
                    
                    # 保存输入引脚信息
                    if cell.inpins:
                        for inpin in cell.inpins:
                            df_inpin.loc[cnt_inpin, "std_cell"] = cell.name
                            df_inpin.loc[cnt_inpin, "inpin"] = inpin.name
                            df_inpin.loc[cnt_inpin, "cap"] = inpin.capacitance
                            cnt_inpin += 1
                    else:
                        df_inpin.loc[cnt_inpin, "std_cell"] = cell.name
                        df_inpin.loc[cnt_inpin, "inpin"] = None
                        df_inpin.loc[cnt_inpin, "cap"] = None
                        cnt_inpin += 1
                    
                    # 保存时序信息
                    for outpin in cell.outpins:
                        for timing_info in outpin.timing_table:
                            # 写入上升延迟
                            self._write_timing_row(
                                df_delay, cnt_delay, cell.name, outpin.name,
                                timing_info, "r", table_size
                            )
                            cnt_delay += 1
                            
                            # 写入下降延迟
                            self._write_timing_row(
                                df_delay, cnt_delay, cell.name, outpin.name,
                                timing_info, "f", table_size
                            )
                            cnt_delay += 1
                            
                            # 写入上升转换时间
                            self._write_trans_row(
                                df_trans, cnt_trans, cell.name, outpin.name,
                                timing_info, "r", table_size
                            )
                            cnt_trans += 1
                            
                            # 写入下降转换时间
                            self._write_trans_row(
                                df_trans, cnt_trans, cell.name, outpin.name,
                                timing_info, "f", table_size
                            )
                            cnt_trans += 1
                
                i += 1
        
        return df_inpin, df_delay, df_trans
    
    def _write_timing_row(self, df: pd.DataFrame, cnt: int, cell_name: str,
                          outpin_name: str, timing_info: TimingInfo, 
                          rf: str, table_size: int) -> None:
        """写入延迟表行"""
        df.loc[cnt, "std_cell"] = cell_name
        df.loc[cnt, "outpin"] = outpin_name
        df.loc[cnt, "inpin"] = timing_info.related_pin
        df.loc[cnt, "rf"] = rf
        df.loc[cnt, "when"] = timing_info.when
        
        if rf == "r":
            df.iloc[cnt, 5:5 + table_size] = timing_info.cell_rise_trans
            df.iloc[cnt, 5 + table_size:5 + table_size * 2] = timing_info.cell_rise_loads
            df.iloc[cnt, 5 + table_size * 2:] = timing_info.cell_rise_values
        else:
            df.iloc[cnt, 5:5 + table_size] = timing_info.cell_fall_trans
            df.iloc[cnt, 5 + table_size:5 + table_size * 2] = timing_info.cell_fall_loads
            df.iloc[cnt, 5 + table_size * 2:] = timing_info.cell_fall_values
    
    def _write_trans_row(self, df: pd.DataFrame, cnt: int, cell_name: str,
                         outpin_name: str, timing_info: TimingInfo,
                         rf: str, table_size: int) -> None:
        """写入转换时间表行"""
        df.loc[cnt, "std_cell"] = cell_name
        df.loc[cnt, "outpin"] = outpin_name
        df.loc[cnt, "inpin"] = timing_info.related_pin
        df.loc[cnt, "rf"] = rf
        df.loc[cnt, "when"] = timing_info.when
        
        if rf == "r":
            df.iloc[cnt, 5:5 + table_size] = timing_info.rise_transition_trans
            df.iloc[cnt, 5 + table_size:5 + table_size * 2] = timing_info.rise_transition_loads
            df.iloc[cnt, 5 + table_size * 2:] = timing_info.rise_transition_values
        else:
            df.iloc[cnt, 5:5 + table_size] = timing_info.fall_transition_trans
            df.iloc[cnt, 5 + table_size:5 + table_size * 2] = timing_info.fall_transition_loads
            df.iloc[cnt, 5 + table_size * 2:] = timing_info.fall_transition_values


def write_csv(df: pd.DataFrame, df_path: str) -> None:
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


def delete_csv(df_path: str) -> None:
    """
    删除CSV文件
    
    参数:
        df_path: 要删除的CSV文件路径
    """
    if os.path.isfile(df_path):
        os.remove(df_path)


# 保持向后兼容的别名
lib_parser = LibParser
inpin = InputPin
outpin = OutputPin
timing_info = TimingInfo
cell = Cell


if __name__ == "__main__":
    import time
    
    lib_list = [
        "../data/lib/demo_rvt.lib",
    ]
    
    time1 = time.time()
    path1 = "./Intermediate_data/lib/inpin_cap.csv"
    path2 = "./Intermediate_data/lib/delay.csv"
    path3 = "./Intermediate_data/lib/trans.csv"
    
    delete_csv(path1)
    delete_csv(path2)
    delete_csv(path3)
    
    footprint_dict_driving = {}
    for lib in lib_list:
        print(f"正在处理库文件: {lib}")
        parser = LibParser(lib, footprint_dict_driving)
        df1, df2, df3 = parser.parse()
        write_csv(df1, path1)
        write_csv(df2, path2)
        write_csv(df3, path3)
        print(f"完成处理库文件: {lib}")
    
    time2 = time.time()
    print(f"处理完成，耗时: {time2 - time1:.2f} 秒")
