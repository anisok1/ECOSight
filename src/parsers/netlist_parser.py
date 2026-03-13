#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
网表解析器模块

该模块用于解析Verilog网表文件，提取设计中的实例、引脚和连线信息。
主要功能：
- 解析Verilog模块定义
- 提取实例和引脚信息
- 建立单元间连接关系
- 计算负载和驱动强度

作者: ECOsight Team
"""

import re
import os
import copy
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set


class NetlistParser:
    """
    网表解析器类
    
    用于解析Verilog格式的网表文件。
    
    属性:
        design (str): 设计名称
        top_module (str): 顶层模块名称
        netlist_path (str): 网表文件路径
        in_pin_list (List[str]): 输入引脚名称列表
        out_pin_list (List[str]): 输出引脚名称列表
    """
    
    # 标准单元引脚名称定义
    DEFAULT_IN_PINS = ["A", "B", "CI", "C", "D", "A0", "A0N", "A1", "A1N", "B0", 
                       "B0N", "B1", "B1N", "C0", "A2", "CIN", "CKN", "R", "SN", 
                       "D", "GN", "RN", "G", "S0", "BN", "AN", "CN", "DN", "E", 
                       "SEN", "SE", "CK", "SI"]
    DEFAULT_OUT_PINS = ["S", "CO", "Y", "CON", "ECK", "Q"]
    
    def __init__(self, design: str, netlist_path: str, top_module: str = None):
        """
        初始化网表解析器
        
        参数:
            design: 设计名称
            netlist_path: 网表文件路径
            top_module: 顶层模块名称（可选，默认与design相同）
        """
        self.design = design
        self.netlist_path = netlist_path
        self.top_module = top_module or design
        
        self.in_pin_list = self.DEFAULT_IN_PINS.copy()
        self.out_pin_list = self.DEFAULT_OUT_PINS.copy()
        
        # 内部数据结构
        self._module_position: Dict[str, Tuple[int, int]] = {}
        self._inpin_dict: Dict[str, float] = {}
        self._cell_list: Set[str] = set()
        self._lut_set: Set[str] = set()
        self._wire_connection: Dict[str, List] = {}
        
        # 输出DataFrame
        self.inpin_csv = pd.DataFrame(
            columns=["std_cell", "inst_pin", "PI", "cap", "position_x", "position_y", "layer"]
        )
        self.outpin_csv = pd.DataFrame(
            columns=["std_cell", "inst_pin", "PO", "load", "position_x", "position_y", "layer"]
        )
        self.cell_csv = pd.DataFrame(
            columns=["std_cell", "inst", "in_degree", "out_degree", "driving", "B", "M", "A"]
        )
    
    def load_library_info(self, inpin_cap_path: str, lut_delay_path: str) -> None:
        """
        加载库信息
        
        参数:
            inpin_cap_path: 输入引脚电容CSV文件路径
            lut_delay_path: 延迟查找表CSV文件路径
        """
        # 加载输入引脚电容
        inpin_csv = pd.read_csv(inpin_cap_path)
        for index, row in inpin_csv.iterrows():
            if not row.isnull().values.any():
                key = row['std_cell'] + "_" + row['inpin']
                self._inpin_dict[key] = row['cap']
        
        # 获取所有标准单元
        self._cell_list = set(inpin_csv.iloc[:, 0])
        
        # 加载延迟查找表
        lut_csv = pd.read_csv(lut_delay_path)
        for index, row in lut_csv.iterrows():
            key = row['std_cell'] + "_" + row['inpin'] + "_" + row['outpin'] + "_" + row['rf']
            self._lut_set.add(key)
    
    def parse(self, output_dir: str = None) -> Dict:
        """
        解析网表文件
        
        参数:
            output_dir: 输出目录路径
        
        返回:
            解析结果字典
        """
        if output_dir is None:
            output_dir = f"../Intermediate_data/{self.design}"
        
        # 确保输出目录存在
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        
        # 读取网表文件
        with open(self.netlist_path, 'r') as f:
            lines = f.readlines()
        
        # 预处理：获取模块位置
        print("预处理中...")
        self._get_module_positions(lines)
        print("预处理完成")
        
        # 初始化计数器和数据结构
        cnt = 0  # 实例计数
        cnt1 = 0  # 输入引脚计数
        cnt2 = 0  # 输出引脚计数
        
        _in = []  # 输入信号列表
        _out = []  # 输出信号列表
        high = []  # 高电平信号
        low = []  # 低电平信号
        self._wire_connection = {}
        
        # 打开输出文件
        net_file = open(os.path.join(output_dir, "NetEdgeLink.txt"), 'w')
        cell_file = open(os.path.join(output_dir, "CellEdgeLink.txt"), 'w')
        inst_file = open(os.path.join(output_dir, "inst_all.txt"), 'w')
        wire_file = open(os.path.join(output_dir, "wire_all.txt"), 'w')
        
        # 获取主模块行号范围
        main_begin = self._module_position[self.top_module][0]
        main_end = self._module_position[self.top_module][1]
        main_lines = lines[main_begin:main_end]
        
        print(f"分析顶层模块 {self.top_module}...")
        
        # 解析主模块
        i = 0
        part_number = max(1, round(len(main_lines) / 10))
        
        while i in range(len(main_lines)):
            # 跳过注释和空行
            if re.findall(r"^\s*\/\/", main_lines[i]):
                pass
            elif not re.findall(r"\S+", main_lines[i]):
                pass
            elif re.findall(r"\*\/", main_lines[i]):
                pass
            elif re.findall(r"\/\*", main_lines[i]):
                pass
            # 遇到endmodule则跳出
            elif re.findall("^endmodule", main_lines[i]):
                break
            else:
                # 获取完整语句
                i, line = self._complete_line(i, main_lines)
                
                # 跳过module声明
                if re.findall("^module", line):
                    pass
                
                # 处理input声明
                elif re.findall(r"^\s*input", line):
                    self._parse_port_declaration(line, _in, "input")
                
                # 处理output声明
                elif re.findall(r"^\s*output", line):
                    self._parse_port_declaration(line, _out, "output")
                
                # 处理assign语句
                elif re.findall(r"^\s*assign", line):
                    self._parse_assign(line, _in, _out, high, low)
                
                # 处理wire声明
                elif re.findall(r"^\s*wire", line):
                    self._parse_wire_declaration(line, _in, _out)
                
                # 处理实例
                else:
                    if re.findall(r"\.\S+\s*\(\s*\S+\s*\)", line):
                        result = self._parse_instance(
                            lines, line, _in, _out, high, low,
                            cnt, cnt1, cnt2, [], inst_file, net_file, cell_file
                        )
                        if result:
                            cnt, cnt1, cnt2 = result
            
            i += 1
            
            if i % part_number == 0:
                print(f"处理进度: {i / len(main_lines) * 100:.1f}%")
        
        # 处理连线连接关系
        for wire, connect in self._wire_connection.items():
            wire_file.write(wire + "\n")
            cnt2_out = connect[0]
            load = 0
            cnt1_list = connect[1]
            
            if cnt2_out == -1:
                print(f"连线错误/输出引脚: {wire}")
            elif len(cnt1_list) == 0:
                print(f"连线错误/输入引脚: {wire}")
            else:
                for tmp in cnt1_list:
                    load += float(self.inpin_csv.loc[tmp, "cap"])
                    line = str(cnt2_out) + " " + str(tmp) + " " + wire + "\n"
                    net_file.write(line)
                self.outpin_csv.loc[cnt2_out, "load"] = load
        
        # 保存CSV文件
        self.cell_csv.to_csv(os.path.join(output_dir, "CellEdgeFeature.csv"), index=False)
        self.inpin_csv.to_csv(os.path.join(output_dir, "InPinFeature.csv"), index=False)
        self.outpin_csv.to_csv(os.path.join(output_dir, "OutPinFeature.csv"), index=False)
        
        # 关闭文件
        net_file.close()
        cell_file.close()
        inst_file.close()
        wire_file.close()
        
        return {
            "cell_count": len(self.cell_csv),
            "inpin_count": len(self.inpin_csv),
            "outpin_count": len(self.outpin_csv),
            "wire_count": len(self._wire_connection)
        }
    
    def _complete_line(self, index: int, content: List[str]) -> Tuple[int, str]:
        """
        合并多行语句为完整的一行
        
        参数:
            index: 当前行索引
            content: 文件内容列表
        
        返回:
            (新索引, 完整语句)
        """
        word = ""
        while index <= len(content):
            line = content[index].replace("\n", "").replace("\\", "")
            if line[-1] != ";":
                word += line
                index += 1
            else:
                word += line
                break
        return index, word
    
    def _get_module_positions(self, lines: List[str]) -> None:
        """
        获取所有模块的行号范围
        
        参数:
            lines: 文件内容列表
        """
        module_name = ""
        module_begin = 0
        
        for i in range(len(lines)):
            if re.findall("^module", lines[i]):
                module_name = re.findall("^module\s+(\S+)\s*\(.*", lines[i])[0]
                module_begin = i
            elif re.findall("^endmodule", lines[i]):
                self._module_position[module_name] = (module_begin, i)
    
    def _parse_port_declaration(self, line: str, port_list: List[str], port_type: str) -> None:
        """
        解析端口声明
        
        参数:
            line: 语句行
            port_list: 端口列表
            port_type: 端口类型 ("input" 或 "output")
        """
        line = line.replace(port_type, "").replace(" ", "").replace("\n", '').replace(";", "")
        
        # 处理位宽声明
        if re.findall(r"(\[\d+:\d+\])", line):
            line = line.replace(re.findall(r"(\[\d+:\d+\])", line)[0], "")
        
        if re.findall(",", line):
            port_list += line.split(",")
        else:
            port_list.append(line)
    
    def _parse_assign(self, line: str, _in: List, _out: List, high: List, low: List) -> None:
        """
        解析assign语句
        
        参数:
            line: 语句行
            _in: 输入信号列表
            _out: 输出信号列表
            high: 高电平信号列表
            low: 低电平信号列表
        """
        line1, line2 = re.findall(r"assign(.*?)=(.*)", line.replace(" ", ""))[0]
        
        if (line1 in _in) and (line2 not in _in):
            _in.append(line2)
        elif (line2 in _in) and (line1 not in _in):
            _in.append(line1)
        elif (line1 in _out) and (line2 not in _out):
            _out.append(line2)
        elif (line2 in _out) and (line1 not in _out):
            _out.append(line1)
        elif line1 == "1'b1":
            high.append(line2)
        elif line1 == "1'b0":
            low.append(line2)
        elif line2 == "1'b1":
            high.append(line1)
        elif line2 == "1'b0":
            low.append(line1)
    
    def _parse_wire_declaration(self, line: str, _in: List, _out: List) -> None:
        """
        解析wire声明
        
        参数:
            line: 语句行
            _in: 输入信号列表
            _out: 输出信号列表
        """
        line = line.replace(" ", "").replace(";", "").replace("wire", "")
        
        # 处理位宽声明
        if re.findall(r"\[\d+:\d+\]", line):
            d1, d2 = re.findall(r"\[(\d+):(\d+)\]", line)[0]
            line = line.replace(f"[{d1}:{d2}]", "")
            line_list = line.split(",")
            
            for tmp in line_list:
                if (tmp not in _in) and (tmp not in _out):
                    for idx in range(int(d2), int(d1) + 1):
                        wire_name = tmp + "[" + str(idx) + "]"
                        self._wire_connection.setdefault(wire_name, [-1, []])
        else:
            line_list = line.split(",")
            for tmp in line_list:
                if (tmp not in _in) and (tmp not in _out):
                    self._wire_connection.setdefault(tmp, [-1, []])
    
    def _parse_instance(self, lines: List[str], line: str, _in: List, _out: List,
                        high: List, low: List, cnt: int, cnt1: int, cnt2: int,
                        connect_replace_list: List, inst_file, net_file, 
                        cell_file) -> Optional[Tuple[int, int, int]]:
        """
        解析实例
        
        参数:
            lines: 文件内容列表
            line: 语句行
            _in: 输入信号列表
            _out: 输出信号列表
            high: 高电平信号列表
            low: 低电平信号列表
            cnt: 实例计数
            cnt1: 输入引脚计数
            cnt2: 输出引脚计数
            connect_replace_list: 连接替换列表
            inst_file: 实例文件句柄
            net_file: 网络文件句柄
            cell_file: 单元文件句柄
        
        返回:
            更新后的计数器元组 (cnt, cnt1, cnt2)
        """
        # 提取实例信息
        pattern = r"^\s*(\S+)\s+(\S*?)\s*\((.*)\);"
        std_cell, inst, connect = re.findall(pattern, line)[0]
        
        drive = std_cell.split("_")[1]  # 驱动强度
        
        # 标准单元
        if std_cell in self._cell_list:
            inst_file.write(inst + "\n")
        else:
            # 子模块处理（递归）
            print(f"开始处理子模块: {std_cell}")
            return None
        
        # 计算驱动强度数值
        drive_number = drive[1:-1]
        if re.findall(r"\d+P\d+", drive_number):
            num_list = re.findall(r"(\d+)P(\d+)", drive_number)[0]
            number = float(num_list[0]) + 0.1 * float(num_list[1])
        else:
            number = int(drive_number)
        
        # 解析连接关系
        connect_list = connect.replace(" ", "").split("),")
        for ii in range(len(connect_list) - 1):
            connect_list[ii] = connect_list[ii] + ")"
        
        inpin = []
        outpin = []
        in_degree = 0
        
        for e in connect_list:
            pin, wire = re.findall(r"\.(\S+)\((\S+)\)", e)[0]
            
            # 处理输入引脚
            if pin in self.in_pin_list:
                in_degree += 1
                inpin.append(str(cnt1) + "_" + pin)
                
                pin_name = std_cell + "_" + pin
                self.inpin_csv.loc[cnt1, "cap"] = float(self._inpin_dict.get(pin_name, 0))
                self.inpin_csv.loc[cnt1, "std_cell"] = std_cell
                self.inpin_csv.loc[cnt1, "inst_pin"] = inst + "_" + pin
                
                wire_without_number = wire
                if re.findall(r"\[\d+\]", wire):
                    num = re.findall(r"(\[\d+\])", wire)[0]
                    wire_without_number = wire.replace(num, "")
                
                if wire in _in or wire_without_number in _in:
                    self.inpin_csv.loc[cnt1, "PI"] = 1
                elif wire in _out or wire_without_number in _out:
                    self.inpin_csv.loc[cnt1, "PO"] = 1
                else:
                    self.inpin_csv.loc[cnt1, "PI"] = 0
                    if ("1'b1" not in wire) and ("1'b0" not in wire) and \
                       (wire_without_number not in _in) and (wire_without_number not in _out) and \
                       (wire not in high) and (wire not in low):
                        self._wire_connection.setdefault(wire, [-1, []])[1].append(cnt1)
                
                cnt1 += 1
            
            # 处理输出引脚
            elif pin in self.out_pin_list:
                outpin.append(str(cnt2) + "_" + pin)
                
                self.outpin_csv.loc[cnt2, "std_cell"] = std_cell
                self.outpin_csv.loc[cnt2, "inst_pin"] = inst + "_" + pin
                
                wire_without_number = wire
                if re.findall(r"\[\d+\]", wire):
                    num = re.findall(r"(\[\d+\])", wire)[0]
                    wire_without_number = wire.replace(num, "")
                
                if wire in _out or wire_without_number in _out:
                    self.outpin_csv.loc[cnt2, "PO"] = 1
                    self.outpin_csv.loc[cnt2, "load"] = 0
                else:
                    self.outpin_csv.loc[cnt2, "PO"] = 0
                    if ("1'b1" not in wire) and ("1'b0" not in wire) and \
                       (wire_without_number not in _in) and (wire_without_number not in _out) and \
                       (wire not in high) and (wire not in low):
                        self._wire_connection.setdefault(wire, [-1, []])[0] = cnt2
                
                cnt2 += 1
        
        # 记录单元边连接关系
        for in_pin in inpin:
            for out_pin in outpin:
                inpin_name = in_pin.rsplit("_", 1)[-1]
                outpin_name = out_pin.rsplit("_", 1)[-1]
                key = std_cell + "_" + inpin_name + "_" + outpin_name + "_r"
                
                if key in self._lut_set:
                    self.cell_csv.loc[cnt, 'std_cell'] = std_cell
                    self.cell_csv.loc[cnt, 'inst'] = inst
                    self.cell_csv.loc[cnt, 'in_degree'] = in_degree
                    self.cell_csv.loc[cnt, "driving"] = number
                    self.cell_csv.loc[cnt, "out_degree"] = len(outpin)
                    
                    # 设置Beta比例标记
                    if drive[-1] == "B":
                        self.cell_csv.loc[cnt, "B"] = 1
                        self.cell_csv.loc[cnt, "M"] = 0
                        self.cell_csv.loc[cnt, "A"] = 0
                    elif drive[-1] == "M":
                        self.cell_csv.loc[cnt, "B"] = 0
                        self.cell_csv.loc[cnt, "M"] = 1
                        self.cell_csv.loc[cnt, "A"] = 0
                    elif drive[-1] == "A":
                        self.cell_csv.loc[cnt, "B"] = 0
                        self.cell_csv.loc[cnt, "M"] = 0
                        self.cell_csv.loc[cnt, "A"] = 1
                    else:
                        print(f"未知的Beta比例字母: {drive[-1]}")
                    
                    line_out = in_pin.rsplit("_", 1)[0] + " " + out_pin.rsplit("_", 1)[0] + "\n"
                    cell_file.write(line_out)
                    cnt += 1
        
        return cnt, cnt1, cnt2


def netlist_extraction(design: str, netlist_path: str, top: str = None,
                       lib_dir: str = None) -> Dict:
    """
    网表提取函数（兼容旧接口）
    
    从Verilog网表中提取实例和连接信息。
    
    参数:
        design: 设计名称
        netlist_path: 网表文件路径
        top: 顶层模块名称
        lib_dir: 库文件目录
    
    返回:
        解析结果字典
    """
    if lib_dir is None:
        lib_dir = "../Intermediate_data/lib"
    
    parser = NetlistParser(design, netlist_path, top)
    parser.load_library_info(
        os.path.join(lib_dir, "ULVT_inpin_cap.csv"),
        os.path.join(lib_dir, "ULVT_delay.csv")
    )
    
    output_dir = f"../Intermediate_data/{design}"
    return parser.parse(output_dir)


if __name__ == "__main__":
    import time
    
    design = "b05"
    top = "b05"
    netlist = f"../data/design/{design}/netlist/{design}.v"
    
    print(f"处理网表提取: {netlist}")
    start_time = time.time()
    result = netlist_extraction(design, netlist, top)
    elapsed_time = time.time() - start_time
    
    print(f"处理完成，耗时: {elapsed_time:.2f} 秒")
    print(f"单元数量: {result['cell_count']}")
    print(f"输入引脚数量: {result['inpin_count']}")
    print(f"输出引脚数量: {result['outpin_count']}")