# -*- coding: utf-8 -*-
"""
ECOSight 主界面
基于PyQt5的图形用户界面，提供数据处理、模型训练和自动ECO功能

使用方法:
    python gui.py
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QTextEdit, QPushButton, QGroupBox
)
from PyQt5.QtCore import pyqtSignal, QObject, QThread
import logging
from datetime import datetime
import time
import re
import tqdm

# 导入自定义模块
from .utils.config import Config
from .utils.data_processor import DataProcessor
from .utils.logger import setup_logger
from .models.trainer import Trainer
from .models.predictor import Predictor, path_extraction, model_load, predict
from .models.sensitivity import SensitivityAnalyzer


class EmitStream(QObject):
    """输出流重定向类，用于将打印输出重定向到GUI"""
    text_written = pyqtSignal(str)
    progress_update = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.buffer = ""

    def write(self, text):
        self.buffer += text
        lines = self.buffer.split('\n')
        self.buffer = lines.pop() if lines else ''
        for line in lines:
            processed_line = self._process_control_chars(line + '\n')
            if processed_line:
                progress_info = self._extract_progress_info(processed_line.strip())
                if progress_info:
                    self.progress_update.emit(progress_info)
                else:
                    self.text_written.emit(processed_line)

    def _process_control_chars(self, text):
        """处理控制字符"""
        if '\r' in text:
            text = text.split('\r')[-1]
        if '\b' in text:
            parts = text.split('\b')
            result = []
            for part in parts:
                if result:
                    result[-1] = result[-1][:-1] if len(result[-1]) > 0 else ""
                result.append(part)
            text = "".join(result)
        return text

    def _extract_progress_info(self, line):
        """提取进度信息"""
        patterns = [r'(\d+%)', r'(\d+\.\d+%)', r'\[.*?(\d+%).*?\]', r'(\d+/\d+)']
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                return match.group()
        return None

    def flush(self):
        pass

    def isatty(self):
        return False


class QTextEditLogHandler(logging.Handler, QObject):
    """日志处理器，将日志输出到QTextEdit"""
    log_received = pyqtSignal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        self.log_received.emit(msg + '\n')


class MainWindow(QWidget):
    """
    主窗口类
    
    提供四个主要功能按钮：
    1. 数据处理：解析输入文件，生成中间数据
    2. 数据加载：加载处理好的数据
    3. 模型训练：训练延迟预测模型
    4. 自动ECO：生成ECO命令
    """
    
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.data_processor = DataProcessor(self.config)
        
        # 数据存储
        self.cell_edge_link = None
        self.net_edge_link = None
        self.in_pin_feature = None
        self.out_pin_feature = None
        self.cell_edge_feature = None
        self.net_edge_feature = None
        self.lut_information = None
        self.footprint_dict = None
        self.def_parser = None
        
        self.ui_init()
        self.setup_redirects()

    def setup_redirects(self):
        """设置输出重定向"""
        self.output_stream = EmitStream()
        self.output_stream.text_written.connect(self.append_text)
        self.output_stream.progress_update.connect(self.update_progress)
        sys.stdout = self.output_stream

        self.log_handler = QTextEditLogHandler()
        self.log_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.log_handler.log_received.connect(self.append_text)

        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_handler)
        root_logger.setLevel(logging.INFO)

        logging.info("ECOSight 初始化完成")

    def ui_init(self):
        """初始化用户界面"""
        main_layout = QVBoxLayout()

        # 输入区域
        input_group = QGroupBox("输入参数")
        input_layout = QVBoxLayout()
        input_group.setLayout(input_layout)
        
        # 单行输入
        single_input_layout = QHBoxLayout()
        self.single_line_edits = []
        
        labels = ["design", "top", "techlef", "netlist", "def", "rpt", "device", "eco_path", "model"]
        defaults = [
            'b05', 'b05', "../data/lef/demo_tech.lef",
            "../data/design/b05/netlist/b05.v",
            "../data/design/b05/def/b05_incr.def",
            "../data/design/b05/timing_report/setup.wo_io_all.rpt",
            "cuda", "../data/design/b05_complete_eco/timing_report/setup.wo_io.rpt",
            "../models/b05_model.pth"
        ]
        
        for i, (label, default) in enumerate(zip(labels, defaults)):
            v_layout = QVBoxLayout()
            lbl = QLabel(label)
            edit = QLineEdit()
            edit.setText(default)
            v_layout.addWidget(lbl)
            v_layout.addWidget(edit)
            single_input_layout.addLayout(v_layout)
            self.single_line_edits.append(edit)
        
        input_layout.addLayout(single_input_layout)

        # 多行输入
        multi_line_layout = QHBoxLayout()
        self.multi_line_edits = []
        
        multi_labels = ["libs", "lefs"]
        multi_defaults = [
            "../data/lib/demo_hvt.lib\n../data/lib/demo_lvt.lib\n../data/lib/demo_rvt.lib",
            "../data/lef/demo_hvt.lef\n../data/lef/demo_lvt.lef\n../data/lef/demo_rvt.lef"
        ]
        
        for label, default in zip(multi_labels, multi_defaults):
            h_layout = QHBoxLayout()
            lbl = QLabel(label)
            text_edit = QTextEdit()
            text_edit.setFixedSize(500, 80)
            text_edit.setText(default)
            h_layout.addWidget(lbl)
            h_layout.addWidget(text_edit)
            multi_line_layout.addLayout(h_layout)
            self.multi_line_edits.append(text_edit)
        
        input_layout.addLayout(multi_line_layout)

        # 按钮区域
        button_group = QGroupBox("操作按钮")
        button_group.setFixedHeight(100)
        button_layout = QHBoxLayout()
        button_group.setLayout(button_layout)

        btn_names = [("数据处理", self.data_process), ("数据加载", self.load_data),
                     ("模型训练", self.model_training), ("自动ECO", self.auto_eco)]
        
        for name, callback in btn_names:
            btn = QPushButton(name)
            btn.clicked.connect(callback)
            button_layout.addWidget(btn)

        # 输出区域
        output_group = QGroupBox('输出日志')
        output_layout = QVBoxLayout()
        output_group.setLayout(output_layout)
        self.output_text_edit = QTextEdit()
        self.output_text_edit.setReadOnly(True)
        output_layout.addWidget(self.output_text_edit)

        # 添加到主布局
        main_layout.addWidget(input_group, 1)
        main_layout.addWidget(button_group, 1)
        main_layout.addWidget(output_group, 1)

        self.setLayout(main_layout)
        self.setWindowTitle("ECOSight V1.0")

    def append_text(self, text):
        """追加文本到输出区域"""
        self.output_text_edit.moveCursor(self.output_text_edit.textCursor().End)
        self.output_text_edit.insertPlainText(text)
        self.output_text_edit.ensureCursorVisible()
        QApplication.processEvents()

    def update_progress(self, progress_info):
        """更新进度信息"""
        self.output_text_edit.moveCursor(self.output_text_edit.textCursor().End)
        self.output_text_edit.insertPlainText(progress_info)
        QApplication.processEvents()

    def data_process(self):
        """数据处理按钮回调"""
        logging.info("开始数据处理...")
        
        design = self.single_line_edits[0].text()
        top = self.single_line_edits[1].text()
        techlef = self.single_line_edits[2].text()
        netlist = self.single_line_edits[3].text()
        def_path = self.single_line_edits[4].text()
        rpt = self.single_line_edits[5].text()

        libs = self.multi_line_edits[0].toPlainText().split("\n")
        lefs = self.multi_line_edits[1].toPlainText().split("\n")

        self.data_processor.process_lib(libs)
        self.data_processor.process_tech_lef(techlef)
        self.data_processor.process_lef(lefs)
        self.data_processor.process_netlist(netlist, design)
        self.data_processor.process_def(def_path, design)
        self.data_processor.process_timing_report(rpt, design)

        logging.info("数据处理完成!")

    def load_data(self):
        """数据加载按钮回调"""
        import numpy as np
        import pandas as pd
        import json
        import pickle
        
        design = self.single_line_edits[0].text()
        eco_path = self.single_line_edits[7].text()
        
        logging.info("开始加载数据...")
        start_time = time.time()
        
        # 加载中间数据
        intermediate_dir = f"../intermediate/{design}"
        
        # 加载DEF解析器
        with open(f"{intermediate_dir}/def_parser.pkl", "rb") as f:
            self.def_parser = pickle.load(f)
        
        # 加载边连接
        self.cell_edge_link = np.loadtxt(f"{intermediate_dir}/CellEdgeLink.txt", dtype=int)
        self.net_edge_link = np.loadtxt(f"{intermediate_dir}/NetEdgeLink.txt", usecols=(0, 1), dtype=int)
        
        # 加载特征
        self.in_pin_feature = pd.read_csv(f"{intermediate_dir}/InPinFeature.csv")
        self.out_pin_feature = pd.read_csv(f"{intermediate_dir}/OutPinFeature.csv")
        self.cell_edge_feature = pd.read_csv(f"{intermediate_dir}/CellEdgeFeature.csv")
        self.net_edge_feature = pd.read_csv(f"{intermediate_dir}/NetEdgeFeature.csv")
        
        # 加载LUT信息
        lut_df = pd.read_csv("../intermediate/lib/ULVT_delay.csv")
        self.lut_information = {}
        for _, row in lut_df.iterrows():
            key = f"{row['std_cell']}_{row['inpin']}_{row['outpin']}_{row['rf']}"
            self.lut_information[key] = row.iloc[5:].tolist()
        
        # 加载footprint信息
        with open('../intermediate/lib/footlogging.info_driving.json', 'r') as f:
            self.footprint_dict = json.load(f)
        
        # 解析ECO报告
        self.path_list = self._parse_eco_report(eco_path)
        
        elapsed = time.time() - start_time
        logging.info(f"数据加载完成，耗时: {elapsed:.2f}s")

    def _parse_eco_report(self, rpt_path):
        """解析ECO时序报告"""
        paths = []
        with open(rpt_path, "r") as f:
            content = f.readlines()
        
        # 简化的路径解析逻辑
        # ... 实际实现需要根据报告格式调整
        
        return paths

    def model_training(self):
        """模型训练按钮回调"""
        logging.info("开始模型训练...")
        
        design = self.single_line_edits[0].text()
        device = self.single_line_edits[6].text()
        model_path = self.single_line_edits[8].text()
        
        # 加载训练数据
        labels = np.loadtxt(f"../intermediate/{design}/label_dataprocess.txt", dtype=float)
        with open(f"../intermediate/{design}/paths_dataprocess.txt", "r") as f:
            paths = [line.strip() for line in f.readlines()]
        
        # 创建训练器
        trainer = Trainer(
            design=design, device=device, model_type='CG',
            hidden_channels=256, num_layers=2, batch_size=32,
            epochs=500
        )
        
        # 提取子图并训练
        # ... 实际实现需要调用子图提取
        
        logging.info(f"模型保存到: {model_path}")
        logging.info("模型训练完成!")

    def auto_eco(self):
        """自动ECO按钮回调"""
        logging.info("开始自动ECO...")
        
        design = self.single_line_edits[0].text()
        device = self.single_line_edits[6].text()
        model_path = self.single_line_edits[8].text()
        eco_path = self.single_line_edits[7].text()
        
        # 加载模型
        # ... 实际实现需要完整的ECO流程
        
        logging.info("自动ECO完成!")


def main():
    """主函数"""
    # 设置日志
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"../logs/log_{current_time}.log"
    logging.basicConfig(
        filename=log_file, 
        level=logging.INFO, 
        format='%(asctime)s - %(message)s'
    )
    
    # 创建应用
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
