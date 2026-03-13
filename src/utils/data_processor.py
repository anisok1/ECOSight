# -*- coding: utf-8 -*-
"""
数据处理器模块
整合数据提取和处理流程
"""

import os
import time
import pickle
import json
import logging
from typing import List, Optional

from ..parsers import (
    LEFParser, LibParser, TechLefParser, 
    TimingReportParser, NetlistParser, DEFParser
)
from .config import Config
from .file_utils import ensure_dir, write_csv, delete_csv, write_json, read_json


class DataProcessor:
    """
    数据处理器
    
    负责协调各解析器完成数据提取和处理流程
    
    Args:
        config: 配置对象
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.config.ensure_directories()
        self.logger = logging.getLogger("ECOSight.DataProcessor")
    
    def process_lib(self, lib_paths: List[str]) -> dict:
        """
        处理LIB文件
        
        Args:
            lib_paths: LIB文件路径列表
            
        Returns:
            footprint_driving字典
        """
        output_dir = os.path.join(self.config.intermediate_path, "lib")
        ensure_dir(output_dir)
        
        path1 = os.path.join(output_dir, "ULVT_inpin_cap.csv")
        path2 = os.path.join(output_dir, "ULVT_delay.csv")
        path3 = os.path.join(output_dir, "ULVT_trans.csv")
        
        if all(os.path.exists(p) for p in [path1, path2, path3]):
            self.logger.info("LIB数据已存在，跳过处理")
            with open(os.path.join(output_dir, "footlogging.info_driving.json"), 'r') as f:
                return json.load(f)
        
        # 删除旧文件
        for p in [path1, path2, path3]:
            delete_csv(p)
        
        footprint_dict_driving = {}
        for lib_path in lib_paths:
            self.logger.info(f"处理LIB文件: {lib_path}")
            start_time = time.time()
            
            parser = LibParser(lib_path, footprint_dict_driving)
            df1, df2, df3 = parser.parse()
            
            write_csv(df1, path1)
            write_csv(df2, path2)
            write_csv(df3, path3)
            
            elapsed = time.time() - start_time
            self.logger.info(f"LIB处理完成，耗时: {elapsed:.2f}s")
        
        # 保存footprint信息
        write_json(footprint_dict_driving, 
                   os.path.join(output_dir, "footlogging.info_driving.json"))
        
        return footprint_dict_driving
    
    def process_tech_lef(self, tech_lef_path: str) -> bool:
        """
        处理Tech LEF文件
        
        Args:
            tech_lef_path: Tech LEF文件路径
            
        Returns:
            是否成功
        """
        output_path = os.path.join(self.config.intermediate_path, "via_info.csv")
        
        if os.path.exists(output_path):
            self.logger.info("Tech LEF数据已存在，跳过处理")
            return True
        
        self.logger.info(f"处理Tech LEF文件: {tech_lef_path}")
        start_time = time.time()
        
        parser = TechLefParser(tech_lef_path)
        parser.extract_to_csv(output_path)
        
        elapsed = time.time() - start_time
        self.logger.info(f"Tech LEF处理完成，耗时: {elapsed:.2f}s")
        return True
    
    def process_lef(self, lef_paths: List[str]) -> bool:
        """
        处理LEF文件
        
        Args:
            lef_paths: LEF文件路径列表
            
        Returns:
            是否成功
        """
        output_dir = os.path.join(self.config.intermediate_path, "lef")
        ensure_dir(output_dir)
        
        inpin_path = os.path.join(output_dir, "inpin_loc.csv")
        outpin_path = os.path.join(output_dir, "outpin_loc.csv")
        
        if os.path.exists(inpin_path) and os.path.exists(outpin_path):
            self.logger.info("LEF数据已存在，跳过处理")
            return True
        
        delete_csv(inpin_path)
        delete_csv(outpin_path)
        
        for lef_path in lef_paths:
            self.logger.info(f"处理LEF文件: {lef_path}")
            start_time = time.time()
            
            parser = LEFParser(lef_path)
            inpin_csv, outpin_csv = parser.parse()
            
            write_csv(inpin_csv, inpin_path)
            write_csv(outpin_csv, outpin_path)
            
            elapsed = time.time() - start_time
            self.logger.info(f"LEF处理完成，耗时: {elapsed:.2f}s")
        
        return True
    
    def process_netlist(self, netlist_path: str, design: Optional[str] = None) -> bool:
        """
        处理网表文件
        
        Args:
            netlist_path: 网表文件路径
            design: 设计名称
            
        Returns:
            是否成功
        """
        design = design or self.config.design
        output_dir = self.config.get_design_intermediate_path(design)
        ensure_dir(output_dir)
        
        self.logger.info(f"处理网表文件: {netlist_path}")
        start_time = time.time()
        
        parser = NetlistParser(netlist_path, output_dir, design)
        parser.extract()
        
        elapsed = time.time() - start_time
        self.logger.info(f"网表处理完成，耗时: {elapsed:.2f}s")
        return True
    
    def process_def(self, def_path: str, design: Optional[str] = None) -> 'DEFParser':
        """
        处理DEF文件
        
        Args:
            def_path: DEF文件路径
            design: 设计名称
            
        Returns:
            DEF解析器对象
        """
        design = design or self.config.design
        output_dir = self.config.get_design_intermediate_path(design)
        ensure_dir(output_dir)
        
        lef_dir = os.path.join(self.config.intermediate_path, "lef")
        inpin_path = os.path.join(lef_dir, "inpin_loc.csv")
        outpin_path = os.path.join(lef_dir, "outpin_loc.csv")
        
        self.logger.info(f"处理DEF文件: {def_path}")
        start_time = time.time()
        
        parser = DEFParser(def_path, inpin_path, outpin_path)
        parser.parse()
        
        # 提取网络边特征和负载
        parser.extract_net_features(output_dir)
        
        # 保存解析器对象
        with open(os.path.join(output_dir, "def_parser.pkl"), "wb") as f:
            pickle.dump(parser, f)
        
        elapsed = time.time() - start_time
        self.logger.info(f"DEF处理完成，耗时: {elapsed:.2f}s")
        return parser
    
    def process_timing_report(self, timing_path: str, design: Optional[str] = None) -> bool:
        """
        处理时序报告
        
        Args:
            timing_path: 时序报告文件路径
            design: 设计名称
            
        Returns:
            是否成功
        """
        design = design or self.config.design
        output_dir = self.config.get_design_intermediate_path(design)
        ensure_dir(output_dir)
        
        self.logger.info(f"处理时序报告: {timing_path}")
        start_time = time.time()
        
        parser = TimingReportParser(timing_path, output_dir)
        parser.extract()
        
        elapsed = time.time() - start_time
        self.logger.info(f"时序报告处理完成，耗时: {elapsed:.2f}s")
        return True
    
    def run_full_pipeline(self, design: Optional[str] = None) -> bool:
        """
        运行完整的数据处理流程
        
        Args:
            design: 设计名称
            
        Returns:
            是否成功
        """
        design = design or self.config.design
        self.logger.info(f"开始完整数据处理流程: {design}")
        total_start = time.time()
        
        try:
            # 1. 处理LIB
            self.process_lib(self.config.get_lib_paths())
            
            # 2. 处理Tech LEF
            self.process_tech_lef(self.config.get_tech_lef_path())
            
            # 3. 处理LEF
            self.process_lef(self.config.get_lef_paths())
            
            # 4. 处理网表
            self.process_netlist(self.config.get_netlist_path(design), design)
            
            # 5. 处理DEF
            self.process_def(self.config.get_def_path(design), design)
            
            # 6. 处理时序报告
            self.process_timing_report(self.config.get_timing_report_path(design), design)
            
            total_elapsed = time.time() - total_start
            self.logger.info(f"完整数据处理流程完成，总耗时: {total_elapsed:.2f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"数据处理失败: {e}")
            raise


def data_extraction(design: str, libs: List[str], techlef: str, lefs: List[str],
                   netlist: str, def_file: str, rpt: str, top: str = "") -> None:
    """
    数据提取主函数（兼容原接口）
    
    Args:
        design: 设计名称
        libs: LIB文件路径列表
        techlef: Tech LEF路径
        lefs: LEF文件路径列表
        netlist: 网表路径
        def_file: DEF文件路径
        rpt: 时序报告路径
        top: 顶层模块名
    """
    config = Config(design=design)
    processor = DataProcessor(config)
    
    # 处理LIB
    processor.process_lib(libs)
    
    # 处理Tech LEF
    processor.process_tech_lef(techlef)
    
    # 处理LEF
    processor.process_lef(lefs)
    
    # 处理网表
    processor.process_netlist(netlist, design)
    
    # 处理DEF
    processor.process_def(def_file, design)
    
    # 处理时序报告
    processor.process_timing_report(rpt, design)
