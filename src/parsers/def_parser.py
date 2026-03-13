#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DEF文件解析器模块

该模块用于解析DEF (Design Exchange Format) 文件，提取设计中的实例、端口、网表等信息。
主要功能：
- 解析UNITS和TRACKS信息
- 提取COMPONENTS（实例）信息
- 解析PINS（端口）信息
- 提取NETS（网表）连接信息

作者: ECOsight Team
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from .data_structures import (
    Cell, Inst, Wire, Port, Track, InPin_Inst, OutPin_Inst,
    InPin_Cell, OutPin_Cell
)


def complete_line(index: int, content: List[str]) -> Tuple[int, str]:
    """
    合并多行语句为完整的一行
    
    DEF文件中有些语句可能跨越多行，此函数用于将其合并。
    
    参数:
        index: 当前行索引
        content: 文件内容列表
    
    返回:
        (新索引, 完整语句)
    """
    words = ""
    while index <= len(content):
        line = content[index].replace("\n", "")
        if line[-1] != ";":
            words += line
            index += 1
        else:
            words += line
            break
    return index, words


class DEFParser:
    """
    DEF文件解析器类
    
    用于解析DEF格式的设计文件，提取实例、端口和网表信息。
    
    属性:
        filepath (str): DEF文件路径
        unit (float): 设计单位
        metal_list (List[str]): 金属层列表
        track (Dict[str, Track]): 轨道信息字典
        cell_list (Dict[str, Cell]): 标准单元字典
        inst_list (Dict[str, Inst]): 实例字典
        wire_list (Dict[str, Wire]): 连线字典
        ports (Dict[str, Port]): 端口字典
        matrix (List[np.ndarray]): 布线矩阵列表
    """
    
    def __init__(self, filepath: str, inpin_loc_csvpath: str, outpin_loc_csvpath: str):
        """
        初始化DEF解析器
        
        参数:
            filepath: DEF文件路径
            inpin_loc_csvpath: 输入引脚位置CSV文件路径
            outpin_loc_csvpath: 输出引脚位置CSV文件路径
        """
        self.filepath = filepath
        self.cell_list: Dict[str, Cell] = {}
        self.inst_list: Dict[str, Inst] = {}
        self.wire_list: Dict[str, Wire] = {}
        self.ports: Dict[str, Port] = {}
        self.metal_list: List[str] = []
        self.track: Dict[str, Track] = {}
        self.unit: float = 0
        
        # 解析轨道信息和引脚信息
        self._parse_track_info()
        self._parse_pin_info(inpin_loc_csvpath, outpin_loc_csvpath)
        self.matrix: List[np.ndarray] = self._define_matrix()
    
    def _define_matrix(self) -> List[np.ndarray]:
        """
        定义布线矩阵
        
        为每个金属层创建一个布线矩阵。
        
        返回:
            布线矩阵列表
        """
        wire_matrix = []
        for metal in self.metal_list:
            matrix = np.zeros(
                (self.track[metal].numy, self.track[metal].numx), 
                dtype=np.int8
            )
            wire_matrix.append(matrix)
        return wire_matrix
    
    def _parse_track_info(self) -> None:
        """
        解析轨道信息
        
        从DEF文件中提取UNITS和TRACKS信息。
        """
        with open(self.filepath, "r") as f:
            content = f.readlines()
            i = 0
            while i < len(content):
                line = content[i]
                
                # 获取单位信息
                if re.findall("^UNITS DISTANCE MICRONS (\d+)", line):
                    self.unit = float(re.findall("^UNITS DISTANCE MICRONS (\d+)", line)[0])
                
                # 获取轨道信息
                if re.findall("^TRACKS", line):
                    tmp = line.replace(";", "").replace("\n", "").strip().split(" ")
                    direction = tmp[1]
                    bias = float(tmp[2]) / self.unit
                    num = int(tmp[4])
                    step = float(tmp[6]) / self.unit
                    metal = tmp[-1]
                    
                    # 创建或更新轨道对象
                    if metal not in self.track:
                        track = Track()
                    else:
                        track = self.track[metal]
                    
                    if direction == "X":
                        track.biasx = bias
                        track.stepx = step
                        track.numx = num
                        track.metal = metal
                    elif direction == "Y":
                        track.biasy = bias
                        track.stepy = step
                        track.numy = num
                        track.metal = metal
                    else:
                        print(f"错误: 未知的轨道方向 {direction}")
                    
                    # 设置轨道方向（根据层号交替）
                    if len(self.metal_list) % 2 == 0:
                        track.direction = "Y"
                    else:
                        track.direction = "X"
                    
                    self.track[metal] = track
                    if metal not in self.metal_list:
                        self.metal_list.insert(0, metal)
                
                i += 1
    
    def _parse_pin_info(self, inpin_info_filepath: str, outpin_info_filepath: str) -> None:
        """
        解析引脚信息
        
        从CSV文件中读取标准单元的输入/输出引脚位置信息。
        
        参数:
            inpin_info_filepath: 输入引脚CSV文件路径
            outpin_info_filepath: 输出引脚CSV文件路径
        """
        inpin_csv = pd.read_csv(inpin_info_filepath)
        outpin_csv = pd.read_csv(outpin_info_filepath)
        
        # 处理输入引脚
        for index, row in inpin_csv.iterrows():
            cell_name = row[0]
            if cell_name not in self.cell_list:
                cell = Cell(cell_name)
                cell.size = float(row[5])
                cell.by = float(row[6])
            else:
                cell = self.cell_list[cell_name]
            
            inpin = InPin_Cell(row[1], cell)
            metal_index = self.metal_list.index(row[2])
            inpin.loc = (metal_index, float(row[3]), float(row[4]))
            cell.inpin_dict[row[1]] = inpin
            self.cell_list[cell_name] = cell
        
        # 处理输出引脚
        for index, row in outpin_csv.iterrows():
            cell_name = row[0]
            if cell_name not in self.cell_list:
                cell = Cell(cell_name)
                cell.size = float(row[5])
                cell.by = float(row[6])
            else:
                cell = self.cell_list[cell_name]
            
            outpin = OutPin_Cell(row[1], cell)
            metal_index = self.metal_list.index(row[2])
            outpin.loc = (metal_index, float(row[3]), float(row[4]))
            cell.outpin_dict[row[1]] = outpin
            self.cell_list[cell_name] = cell
    
    def _get_pin_loc_for_inst(self, inst: Inst) -> None:
        """
        计算实例的引脚位置
        
        根据实例的方向和位置，计算其实际引脚坐标。
        
        参数:
            inst: 实例对象
        """
        std_cell_name = inst.std_cell
        if std_cell_name not in self.cell_list:
            return
        
        std_cell = self.cell_list[std_cell_name]
        
        # 计算输入引脚位置
        for pin_name in std_cell.inpin_dict:
            inpin_cell = std_cell.inpin_dict[pin_name]
            inpin_inst = InPin_Inst(pin_name, inst)
            
            # 根据方向计算坐标
            if inst.orient == "N":
                inpin_inst.loc = (
                    int(inpin_cell.loc[0]),
                    round(inst.loc[0] + inpin_cell.loc[1], 3),
                    round(inst.loc[1] + inpin_cell.loc[2], 3)
                )
            elif inst.orient == "S":
                inpin_inst.loc = (
                    int(inpin_cell.loc[0]),
                    round(inst.loc[0] - inpin_cell.loc[1] + std_cell.size, 3),
                    round(inst.loc[1] - inpin_cell.loc[2] + std_cell.by, 3)
                )
            elif inst.orient == "FN":
                inpin_inst.loc = (
                    int(inpin_cell.loc[0]),
                    round(inst.loc[0] - inpin_cell.loc[1] + std_cell.size, 3),
                    round(inst.loc[1] + inpin_cell.loc[2], 3)
                )
            elif inst.orient == "FS":
                inpin_inst.loc = (
                    int(inpin_cell.loc[0]),
                    round(inst.loc[0] + inpin_cell.loc[1], 3),
                    round(inst.loc[1] - inpin_cell.loc[2] + std_cell.by, 3)
                )
            else:
                print(f"错误: 未知方向 {inst.orient}")
            
            inst.inpin_dict[pin_name] = inpin_inst
        
        # 计算输出引脚位置
        for pin_name in std_cell.outpin_dict:
            outpin_cell = std_cell.outpin_dict[pin_name]
            outpin_inst = OutPin_Inst(pin_name, inst)
            
            if inst.orient == "N":
                outpin_inst.loc = (
                    int(outpin_cell.loc[0]),
                    round(inst.loc[0] + outpin_cell.loc[1], 3),
                    round(inst.loc[1] + outpin_cell.loc[2], 3)
                )
            elif inst.orient == "S":
                outpin_inst.loc = (
                    int(outpin_cell.loc[0]),
                    round(inst.loc[0] - outpin_cell.loc[1] + std_cell.size, 3),
                    round(inst.loc[1] - outpin_cell.loc[2] + std_cell.by, 3)
                )
            elif inst.orient == "FN":
                outpin_inst.loc = (
                    int(outpin_cell.loc[0]),
                    round(inst.loc[0] - outpin_cell.loc[1] + std_cell.size, 3),
                    round(inst.loc[1] + outpin_cell.loc[2], 3)
                )
            elif inst.orient == "FS":
                outpin_inst.loc = (
                    int(outpin_cell.loc[0]),
                    round(inst.loc[0] + outpin_cell.loc[1], 3),
                    round(inst.loc[1] - outpin_cell.loc[2] + std_cell.by, 3)
                )
            else:
                print(f"错误: 未知方向 {inst.orient}")
            
            inst.outpin_dict[pin_name] = outpin_inst
    
    def parse(self) -> None:
        """
        解析DEF文件
        
        提取COMPONENTS、PINS和NETS信息。
        """
        with open(self.filepath, "r") as f:
            content = f.readlines()
            i = 0
            while i < len(content):
                # 解析COMPONENTS
                if re.findall("^\s*COMPONENTS", content[i]):
                    while True:
                        if re.findall("^\s*END COMPONENTS", content[i]):
                            break
                        if re.findall("^\s*-", content[i]):
                            i, words = complete_line(i, content)
                            words = words.replace(";", "")
                            inst = Inst(self.unit)
                            inst.get_info(words)
                            self._get_pin_loc_for_inst(inst)
                            self.inst_list[inst.name] = inst
                        i += 1
                
                # 解析PINS
                elif re.findall("^PINS", content[i]):
                    while True:
                        if re.findall("^END PINS", content[i]):
                            break
                        else:
                            if re.findall("^-", content[i]):
                                pattern = r"^-\s+(.*?)\s+\+\s+NET\s+(.*?)\s+\+\s+DIRECTION\s+(.*?)\s+\+\s+USE\s+(.*?)$"
                                port_name, net_name, direction, purpose = re.findall(pattern, content[i])[0]
                                
                                if purpose == "SIGNAL":
                                    port = Port(port_name)
                                    self.ports[port_name] = port
                                    
                                    if direction == "INPUT":
                                        port.in_out_port = 0
                                    else:
                                        port.in_out_port = 1
                                    
                                    while True:
                                        i += 1
                                        if re.findall("^\s*\+\s+LAYER", content[i]):
                                            metal = re.findall("^\s*\+\s+LAYER\s+(\w+)\s+.*?", content[i])[0]
                                            layer = self.metal_list.index(metal)
                                        elif re.findall("^\s*\+\s+PLACED", content[i]):
                                            placed_pattern = r"^\s*\+\s+PLACED\s+\(\s+(\d+)\s+(\d+)\s+\)\s+(.*?)\s+;"
                                            x, y, orient = re.findall(placed_pattern, content[i])[0]
                                            x = round(float(x) / self.unit, 3)
                                            y = round(float(y) / self.unit, 3)
                                            port.loc = (layer, x, y)
                                        else:
                                            print(f"错误: {content[i]}")
                                        
                                        if re.findall("^-", content[i + 1]):
                                            break
                        i += 1
                
                # 解析NETS
                elif re.findall("^NETS", content[i]):
                    while True:
                        i += 1
                        if re.findall("^END\s+NETS", content[i]):
                            break
                        else:
                            if re.findall("^-\s+(.*?)$", content[i]):
                                net_name = re.findall("^-\s+(.*?)$", content[i])[0]
                                wire = Wire(net_name, self)
                                self.wire_list[net_name] = wire
                                
                                while True:
                                    i += 1
                                    if re.findall("^\s*\+", content[i]):
                                        break
                                    else:
                                        result = re.findall(r"\((.*?)\)+", content[i])
                                        for tmp in result:
                                            inst, pin_name = tmp.strip().split(" ")
                                            if inst != "PIN":
                                                _inst = self.inst_list[inst]
                                                if pin_name in _inst.inpin_dict:
                                                    pin = _inst.inpin_dict[pin_name]
                                                    wire.sink_pin.append(pin)
                                                elif pin_name in _inst.outpin_dict:
                                                    pin = _inst.outpin_dict[pin_name]
                                                    wire.source_pin.append(pin)
                                                else:
                                                    print(f"错误: 找不到引脚 {inst} / {pin_name}")
                                            else:
                                                if self.ports[pin_name].in_out_port == 0:
                                                    wire.source_pin.append(self.ports[pin_name])
                                                else:
                                                    wire.sink_pin.append(self.ports[pin_name])
                                
                                while True:
                                    if re.findall("^\s*;", content[i]):
                                        break
                                    else:
                                        if re.findall("^\s*\+\s+ROUTED", content[i]):
                                            line = content[i].replace("ROUTED", "").replace("+", "").strip()
                                        elif re.findall("^\s*NEW", content[i]):
                                            line = content[i].replace("NEW", "").strip()
                                        else:
                                            i += 1
                                            continue
                                        wire.parser_route(line, self)
                                    i += 1
                                
                                if len(wire.source_pin) != 1:
                                    print("错误: 源引脚数量不为1!")
                                else:
                                    if isinstance(wire.source_pin[0], OutPin_Inst):
                                        new_ports = wire.get_new_ports()
                                        source_loc = wire.source_pin[0].loc
                                        port_loc, port = wire.get_close_loc(source_loc, new_ports)
                                        wire.source_pin[0].loc = (
                                            wire.source_pin[0].loc[0], 
                                            port_loc[1], port_loc[2]
                                        )
                                        new_ports.remove(port)
                                        
                                        for sink_pin in wire.sink_pin:
                                            if isinstance(sink_pin, InPin_Inst):
                                                sink_loc = sink_pin.loc
                                                port_loc, port = wire.get_close_loc(sink_loc, new_ports)
                                                sink_pin.loc = (sink_pin.loc[0], port_loc[1], port_loc[2])
                                                new_ports.remove(port)
                
                i += 1


# 保持向后兼容的别名
Parser = DEFParser


if __name__ == "__main__":
    import time
    
    filepath = "../data/design/b05/def/b05_incr.def"
    design = "b05"
    inpin_loc_csvpath = f"../Intermediate_data/{design}/inpin_loc.csv"
    outpin_loc_csvpath = f"../Intermediate_data/{design}/outpin_loc.csv"
    
    start_time = time.time()
    def_parser = DEFParser(filepath, inpin_loc_csvpath, outpin_loc_csvpath)
    def_parser.parse()
    elapsed_time = time.time() - start_time
    
    print(f"解析完成，耗时: {elapsed_time:.2f} 秒")
    print(f"实例数量: {len(def_parser.inst_list)}")
    print(f"连线数量: {len(def_parser.wire_list)}")
    print(f"端口数量: {len(def_parser.ports)}")
