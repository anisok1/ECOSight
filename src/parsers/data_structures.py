# -*- coding: utf-8 -*-
"""
数据结构定义模块
定义 ECOSight 中使用的核心数据结构，包括：
- Node: A*寻路算法节点
- Track: 布线轨道信息
- Pin: 引脚（输入/输出）
- Cell: 标准单元
- Inst: 实例
- Wire: 连线
"""

import re
from typing import Union, List, Dict, Tuple
import numpy as np
import pandas as pd
from collections import Counter


class Node:
    """
    A*寻路算法中使用的节点类
    
    用于在布线矩阵中进行路径搜索，支持优先队列比较
    """
    
    def __init__(self, parent=None, position=None):
        """
        初始化节点
        
        Args:
            parent: 父节点，用于回溯路径
            position: 节点位置 (layer, x, y)
        """
        self.parent = parent
        self.position = position
        self.g = 0  # 实际代价（从起点到当前节点）
    
    def __eq__(self, other):
        """比较两个节点是否相等（基于位置）"""
        return self.position == other.position
    
    def __lt__(self, other):
        """用于优先队列比较，代价小的优先"""
        return self.g < other.g


class Track:
    """
    布线轨道信息类
    
    存储金属层的轨道参数，用于计算布线位置
    """
    
    def __init__(self, metal=None, biasx=None, biasy=None, 
                 stepx=None, stepy=None, numx=None, numy=None, direction=None):
        """
        初始化轨道信息
        
        Args:
            metal: 金属层名称
            biasx: X方向偏移
            biasy: Y方向偏移
            stepx: X方向步长
            stepy: Y方向步长
            numx: X方向轨道数
            numy: Y方向轨道数
            direction: 轨道方向（'X' 或 'Y'）
        """
        self.direction = direction
        self.metal = metal
        self.biasx = biasx
        self.biasy = biasy
        self.stepx = stepx
        self.stepy = stepy
        self.numx = numx
        self.numy = numy
    
    def print_attr(self):
        """打印轨道属性"""
        print("metal: {}".format(self.metal))
        print("biasx: {}, biasy: {}".format(self.biasx, self.biasy))
        print("stepx: {}, stepy: {}".format(self.stepx, self.stepy))
        print("numx: {}, numy: {}".format(self.numx, self.numy))
        print("direction: {}".format(self.direction))


class InPin_Inst:
    """实例的输入引脚"""
    
    def __init__(self, name, inst):
        """
        Args:
            name: 引脚名称
            inst: 所属实例
        """
        self.name = name
        self.loc = (0, 0, 0)  # (层, x, y)
        self.inst = inst
    
    def print_attr(self):
        print("pin name: {}, loc: {}".format(self.name, self.loc))


class OutPin_Inst:
    """实例的输出引脚"""
    
    def __init__(self, name, inst):
        self.name = name
        self.loc = (0, 0, 0)
        self.inst = inst
    
    def print_attr(self):
        print("pin name: {}, loc: {}".format(self.name, self.loc))


class Port:
    """设计端口"""
    
    def __init__(self, name):
        self.name = name
        self.loc = (0, 0, 0)
        self.in_out_port = 0  # 0=输入端口, 1=输出端口
    
    def print_attr(self):
        direction = "INPUT" if self.in_out_port == 0 else "OUTPUT"
        print("port: {}, loc: {}, direction: {}".format(self.name, self.loc, direction))


class InPin_Cell:
    """标准单元的输入引脚"""
    
    def __init__(self, name, cell):
        self.name = name
        self.loc = (0, 0, 0)
        self.cell = cell
    
    def print_attr(self):
        print("cell inpin: {}, loc: {}".format(self.name, self.loc))


class OutPin_Cell:
    """标准单元的输出引脚"""
    
    def __init__(self, name, cell):
        self.name = name
        self.loc = (0, 0, 0)
        self.cell = cell
    
    def print_attr(self):
        print("cell outpin: {}, loc: {}".format(self.name, self.loc))


class Cell:
    """
    标准单元类
    
    存储标准单元的引脚信息和尺寸
    """
    
    def __init__(self, name):
        self.name = name
        self.inpin_dict = {}
        self.outpin_dict = {}
        self.size = 0  # 单元宽度
        self.by = 0    # 单元高度
    
    def print_attr(self):
        print("cell: {}, size: {}, by: {}".format(self.name, self.size, self.by))
        print("inpins:", list(self.inpin_dict.keys()))
        print("outpins:", list(self.outpin_dict.keys()))


class Inst:
    """
    实例类
    
    表示设计中的一个实例化单元
    """
    
    def __init__(self, unit):
        self.name = ""
        self.std_cell = ""  # 标准单元类型
        self.orient = ""    # 方向 (N/S/FN/FS)
        self.loc = (0, 0)  # 位置
        self.unit = unit
        self.inpin_dict = {}
        self.outpin_dict = {}
    
    def get_info(self, words):
        """
        从DEF语句解析实例信息
        
        Args:
            words: DEF文件中的实例定义语句
        """
        inst, std_cell = re.findall(r"-\s*(\S+)\s+(\S+)\s+.*?", words)[0]
        locx, locy = re.findall(r"\(\s+(\d+)\s+(\d+)\s+\)", words)[0]
        orient = re.findall(r"\s+(FS|S|N|FN)\s+", words)[0]
        
        self.name = inst
        self.std_cell = std_cell
        self.orient = orient
        self.loc = (round(float(locx) / self.unit, 3), 
                   round(float(locy) / self.unit, 3))
    
    def print_attr(self):
        print("inst: {}, cell: {}".format(self.name, self.std_cell))
        print("orient: {}, loc: {}".format(self.orient, self.loc))


class Wire:
    """
    连线类
    
    表示设计中的一条连线，包含路径矩阵和长度计算
    """
    
    def __init__(self, name, parser=None):
        self.name = name
        self.parser = parser
        if parser is not None:
            self.metal_list = parser.metal_list
            self.track = parser.track
            self.unit = parser.unit
        else:
            self.metal_list = []
            self.track = {}
            self.unit = 1.0
        self.wire_path = []
        self.length = {metal: 0 for metal in self.metal_list}
        self.via = []
        self.ports = []
        self.source_pin = []
        self.sink_pin = []
        self.boundury = []
    
    def define_matrix(self):
        """定义连线路径矩阵"""
        wire_matrix = []
        for metal in self.metal_list:
            matrix = np.zeros((self.track[metal].numy, self.track[metal].numx), dtype=np.int8)
            wire_matrix.append(matrix)
        return wire_matrix
    
    def get_inpin(self, inst_name, inpin_name):
        """获取输入引脚"""
        for inpin in self.sink_pin:
            if isinstance(inpin, InPin_Inst) and inpin.name == inpin_name and inpin.inst.name == inst_name:
                return inpin
        return None
    
    def get_outpin(self, inst_name, outpin_name):
        """获取输出引脚"""
        for outpin in self.source_pin:
            if isinstance(outpin, OutPin_Inst) and outpin.name == outpin_name and outpin.inst.name == inst_name:
                return outpin
        return None
    
    def get_boundury(self):
        """获取边界坐标"""
        if not self.ports:
            return 0, 0, 0, 0
        min_x = min(self.ports, key=lambda coord: coord[1])[1] / self.unit
        max_x = max(self.ports, key=lambda coord: coord[1])[1] / self.unit
        min_y = min(self.ports, key=lambda coord: coord[2])[2] / self.unit
        max_y = max(self.ports, key=lambda coord: coord[2])[2] / self.unit
        return min_x, min_y, max_x, max_y
    
    def cut_matrix(self):
        """截取矩阵"""
        min_x, min_y, max_x, max_y = self.get_boundury()
        for i in range(len(self.metal_list)):
            metal = self.metal_list[i]
            min_x_num = max(round((min_x - self.track[metal].biasx) / self.track[metal].stepx) - 1, 0)
            max_x_num = min(round((max_x - self.track[metal].biasx) / self.track[metal].stepx) + 1, self.track[metal].numx)
            min_y_num = max(round((min_y - self.track[metal].biasy) / self.track[metal].stepy) - 1, 0)
            max_y_num = min(round((max_y - self.track[metal].biasy) / self.track[metal].stepy) + 1, self.track[metal].numy)
            self.boundury.append((min_x_num, min_y_num, max_x_num, max_y_num))
            self.wire_path[i] = self.wire_path[i][min_y_num:max_y_num, min_x_num:max_x_num]
    
    def GetPath4TwoPoint(self, source_pin, sink_pin):
        """获取两点间的路径长度"""
        return {metal: 0 for metal in self.metal_list}