#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
时序报告解析器模块

该模块用于解析时序分析工具生成的报告文件。
主要功能：
- 解析时序路径信息
- 提取起点和终点信息
- 计算数据路径延迟
- 提取转换时间和单元信息

作者: ECOsight Team
"""

import re
import os
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional


class TimingReportParser:
    """
    时序报告解析器类
    
    用于解析时序分析工具生成的报告文件（如PrimeTime报告）。
    
    属性:
        design (str): 设计名称
        report_path (str): 报告文件路径
        paths (List[Dict]): 解析出的路径列表
        all_inst_dict (Dict[str, str]): 实例到单元类型的映射
        all_term_dict (Dict[str, str]): 实例到终结点的映射
    """
    
    def __init__(self, design: str, report_path: str):
        """
        初始化时序报告解析器
        
        参数:
            design: 设计名称
            report_path: 时序报告文件路径
        """
        self.design = design
        self.report_path = report_path
        self.paths: List[Dict] = []
        self.all_inst_dict: Dict[str, str] = {}
        self.all_term_dict: Dict[str, str] = {}
    
    def parse(self, cell_feature_path: str, inpin_feature_path: str, 
              outpin_feature_path: str, output_dir: str = None) -> Dict:
        """
        解析时序报告
        
        参数:
            cell_feature_path: CellEdgeFeature.csv路径
            inpin_feature_path: InPinFeature.csv路径
            outpin_feature_path: OutPinFeature.csv路径
            output_dir: 输出目录路径
        
        返回:
            解析结果字典
        """
        if output_dir is None:
            output_dir = f"../Intermediate_data/{self.design}"
        
        # 确保输出目录存在
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        
        # 加载特征文件
        lib = pd.read_csv(cell_feature_path)
        in_pin_csv = pd.read_csv(inpin_feature_path)
        out_pin_csv = pd.read_csv(outpin_feature_path)
        
        inst_lib = list(lib.loc[:, "inst"])
        in_pin_lib = list(in_pin_csv.loc[:, "inst_pin"])
        out_pin_lib = list(out_pin_csv.loc[:, "inst_pin"])
        
        # 打开输出文件
        paths_file = open(os.path.join(output_dir, "paths.txt"), "w")
        label_file = open(os.path.join(output_dir, "label.txt"), "w")
        label_outtran = open(os.path.join(output_dir, "label_outtran.txt"), "w")
        label_intran = open(os.path.join(output_dir, "label_intran.txt"), "w")
        
        path_count = 0
        
        with open(self.report_path, "r") as f:
            rpt_lines = f.readlines()
        
        i = 0
        part_number = max(1, round(len(rpt_lines) / 10))
        
        while i in range(len(rpt_lines)):
            # 检测路径起点
            if re.findall(r"^\s*Startpoint:", rpt_lines[i]):
                path_count += 1
                flag = 1
                out_trans_list = []
                cell_num_list = []
                
                # 提取起点信息
                startpoint = re.findall(r"^\s*Startpoint:(.*)", rpt_lines[i])[0].replace(" ", "")
                if re.findall(r"\(.*?\)", startpoint):
                    startpoint = startpoint.replace(re.findall(r"(\(.*?\))", startpoint)[0], "")
                
                while True:
                    i += 1
                    
                    # 提取终点信息
                    if re.findall(r"^\s*Endpoint:", rpt_lines[i]):
                        endpoint = re.findall(r"^\s*Endpoint:(.*)", rpt_lines[i])[0].replace(" ", "")
                        if re.findall(r"\(.*?\)", endpoint.replace(" ", "")):
                            endpoint = endpoint.replace(re.findall(r"(\(.*?\))", endpoint)[0], "")
                    
                    # 提取路径类型
                    elif re.findall("Path Type", rpt_lines[i]):
                        path_type = re.findall("Path Type:(.*)", rpt_lines[i])[0].replace(" ", "")
                    
                    # 处理路径点
                    elif re.findall(r"^\s*Point", rpt_lines[i]):
                        ps = "^" + startpoint + r".*?\((.*?)\)"
                        pe = "^" + endpoint + r".*?\((.*?)\)"
                        
                        # 检查路径有效性
                        if (startpoint == endpoint) or startpoint not in inst_lib or endpoint not in inst_lib:
                            flag = 0
                            break
                        
                        # 获取传播延迟
                        propagated_pattern = r"^clock network delay \(propagated\)\s+(\d+\.\d+)\s+.*?"
                        if re.findall(propagated_pattern, rpt_lines[i + 3].strip()):
                            propagated_time = re.findall(propagated_pattern, rpt_lines[i + 3].strip())[0]
                        else:
                            print(f"错误: 无法获取传播延迟, 行内容: {rpt_lines[i + 3].strip()}")
                        
                        # 解析路径点
                        while True:
                            i += 1
                            if re.findall(startpoint, rpt_lines[i]):
                                if re.findall(ps, rpt_lines[i].replace(" ", ""))[0] == "in":
                                    i += 2
                                    in_trans_pattern = r".*/.*?\(.*?\)\s+(\d+\.\d+)\s+\d+\.\d+\s+\d+\.\d+"
                                    in_trans = re.findall(in_trans_pattern, rpt_lines[i])[0]
                                else:
                                    inst, pin_in, cell = re.findall(
                                        r"(.*)/(.*?)\((.*?)\).*?", rpt_lines[i].replace(" ", "")
                                    )[0]
                                    self.all_inst_dict[inst] = cell
                                break
                        
                        # 遍历路径上的单元
                        while True:
                            inst, pin_in, cell = re.findall(
                                r"(.*)/(.*?)\((.*?)\).*?", rpt_lines[i].replace(" ", "")
                            )[0]
                            self.all_inst_dict[inst] = cell
                            
                            tmp_in = inst + "_" + pin_in
                            rf_in = rpt_lines[i].replace("\n", "").replace(" ", "")[-1]
                            cell_num = str(in_pin_lib.index(tmp_in)) + rf_in
                            cell_num_list.append(cell_num)
                            
                            if inst != endpoint and inst != startpoint:
                                self.all_term_dict[inst] = pin_in
                            
                            if inst == endpoint:
                                break
                            
                            if inst == startpoint:
                                in_trans_pattern = r".*/.*?\(.*?\)\s+(\d+\.\d+)\s+\d+\.\d+\s+&?\s+\d+\.\d+"
                                in_trans = re.findall(in_trans_pattern, rpt_lines[i])[0]
                            
                            if inst != endpoint:
                                # 获取输出转换时间
                                pin_out = re.findall(
                                    r".*/(.*?)\(.*?\).*?", rpt_lines[i + 1].replace(" ", "")
                                )[0].replace(" ", "")
                                out_trans_pattern = r".*/.*?\(.*?\)\s+(\d+\.\d+)\s+\d+\.\d+\s+&?\s+\d+\.\d+"
                                out_trans = re.findall(out_trans_pattern, rpt_lines[i + 1])[0]
                                out_trans_list.append(out_trans)
                                
                                rf_out = rpt_lines[i + 1].replace("\n", "").replace(" ", "")[-1]
                                tmp_out = inst + "_" + pin_out
                                
                                if inst not in inst_lib:
                                    flag = 0
                                else:
                                    cell_num = str(out_pin_lib.index(tmp_out)) + rf_out
                                    cell_num_list.append(cell_num)
                            
                            i += 3
                    
                    # 处理数据到达时间
                    elif re.findall(r"^\s*data arrival time", rpt_lines[i]):
                        Time = re.findall(r"^\s*data arrival time\s*(\d+\.\d+)", rpt_lines[i])[0]
                        data_path_time = round(float(Time) - float(propagated_time), 3)
                        
                        if flag:
                            label_file.write(str(data_path_time) + "\n")
                            label_intran.write(in_trans + "\n")
                            
                            for trans in out_trans_list:
                                label_outtran.write(trans + " ")
                            label_outtran.write("\n")
                            
                            for cell_num in cell_num_list:
                                paths_file.write(cell_num + " ")
                            paths_file.write("\n")
                        break
            
            i += 1
            
            # 显示进度
            if i % part_number == 0:
                print(f"处理进度: {i / len(rpt_lines) * 100:.1f}%")
        
        # 保存实例和终结点信息
        all_inst_json = os.path.join(output_dir, "all_inst.json")
        all_term_json = os.path.join(output_dir, "all_term.json")
        
        with open(all_inst_json, 'w') as f:
            json.dump(self.all_inst_dict, f)
        with open(all_term_json, 'w') as f:
            json.dump(self.all_term_dict, f)
        
        # 关闭文件
        paths_file.close()
        label_file.close()
        label_intran.close()
        label_outtran.close()
        
        print(f"路径数量: {path_count}")
        
        return {
            "path_count": path_count,
            "all_inst_dict": self.all_inst_dict,
            "all_term_dict": self.all_term_dict
        }


def timing_report_extraction(design: str, rpt: str) -> Dict:
    """
    时序报告提取函数（兼容旧接口）
    
    从时序报告中提取路径信息。
    
    参数:
        design: 设计名称
        rpt: 时序报告文件路径
    
    返回:
        解析结果字典
    """
    output_dir = f"../Intermediate_data/{design}"
    cell_feature_path = os.path.join(output_dir, "CellEdgeFeature.csv")
    inpin_feature_path = os.path.join(output_dir, "InPinFeature.csv")
    outpin_feature_path = os.path.join(output_dir, "OutPinFeature.csv")
    
    parser = TimingReportParser(design, rpt)
    return parser.parse(cell_feature_path, inpin_feature_path, outpin_feature_path, output_dir)


if __name__ == "__main__":
    import time
    
    design = "b05"
    rpt = f"../data/design/{design}/timing_report/setup.wo_io.rpt"
    
    start_time = time.time()
    result = timing_report_extraction(design, rpt)
    elapsed_time = time.time() - start_time
    
    print(f"解析完成，耗时: {elapsed_time:.2f} 秒")
    print(f"路径数量: {result['path_count']}")
    print(f"实例数量: {len(result['all_inst_dict'])}")
