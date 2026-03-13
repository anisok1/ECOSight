# -*- coding: utf-8 -*-
"""
ECOsight 解析器模块
包含所有文件解析器和数据结构定义

模块列表:
- data_structures: 数据结构定义 (Cell, Inst, Pin, Wire, Track等)
- def_parser: DEF文件解析器
- lef_parser: LEF文件解析器
- lib_parser: LIB文件解析器
- techlef_parser: Tech LEF解析器
- netlist_parser: 网表解析器
- timing_parser: 时序报告解析器

使用示例:
    from parsers import DEFParser, LEFParser, LibParser
    from parsers import Cell, Inst, Wire
"""

# 数据结构
from .data_structures import (
    Node,
    Track,
    InPin_Inst,
    OutPin_Inst,
    Port,
    InPin_Cell,
    OutPin_Cell,
    Cell,
    Inst,
    Wire
)

# DEF解析器
from .def_parser import (
    DEFParser,
    Parser,  # 向后兼容别名
    complete_line
)

# LEF解析器
from .lef_parser import (
    LEFParser,
    lef_parser,  # 向后兼容别名
    write_csv,
    delete_csv
)

# LIB解析器
from .lib_parser import (
    LibParser,
    lib_parser,  # 向后兼容别名
    Cell as LibCell,  # LIB中的Cell类
    InputPin,
    OutputPin,
    TimingInfo,
    write_csv as write_lib_csv,
    delete_csv as delete_lib_csv
)

# Tech LEF解析器
from .techlef_parser import (
    TechLefParser,
    TechLefExtraction
)

# 网表解析器
from .netlist_parser import (
    NetlistParser,
    netlist_extraction
)

# 时序报告解析器
from .timing_parser import (
    TimingReportParser,
    timing_report_extraction
)


__all__ = [
    # 数据结构
    'Node',
    'Track',
    'InPin_Inst',
    'OutPin_Inst',
    'Port',
    'InPin_Cell',
    'OutPin_Cell',
    'Cell',
    'Inst',
    'Wire',
    
    # DEF解析器
    'DEFParser',
    'Parser',
    'complete_line',
    
    # LEF解析器
    'LEFParser',
    'lef_parser',
    'write_csv',
    'delete_csv',
    
    # LIB解析器
    'LibParser',
    'lib_parser',
    'LibCell',
    'InputPin',
    'OutputPin',
    'TimingInfo',
    'write_lib_csv',
    'delete_lib_csv',
    
    # Tech LEF解析器
    'TechLefParser',
    'TechLefExtraction',
    
    # 网表解析器
    'NetlistParser',
    'netlist_extraction',
    
    # 时序报告解析器
    'TimingReportParser',
    'timing_report_extraction'
]


# 版本信息
__version__ = '1.0.0'
__author__ = 'ECOsight Team'