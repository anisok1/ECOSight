# ECOSight

基于图神经网络的可解释性 Timing ECO 自动化决策工具

## 项目简介

ECOSight 是一个用于集成电路设计中 **Timing ECO（Engineering Change Order）** 自动化决策的 AI 工具。该项目通过分析时序报告，使用异构图神经网络预测路径延迟，并基于敏感性分析自动生成 ECO 命令来修复时序违例。

### 核心功能

- **数据处理**：解析 DEF、LEF、LIB、网表和时序报告文件，提取特征
- **模型训练**：使用异构图神经网络（GNN）预测路径延迟
- **自动 ECO**：基于敏感性分析自动生成单元尺寸调整命令

## 项目结构

```
ECOsight/
├── src/                      # 源代码
│   ├── parsers/             # 文件解析器
│   │   ├── __init__.py
│   │   ├── data_structures.py
│   │   ├── lef_parser.py
│   │   ├── lib_parser.py
│   │   ├── techlef_parser.py
│   │   ├── timing_parser.py
│   │   ├── netlist_parser.py
│   │   └── def_parser.py
│   ├── models/              # 模型定义
│   │   ├── __init__.py
│   │   ├── gnn_models.py    # GNN模型
│   │   ├── trainer.py       # 训练器
│   │   ├── predictor.py     # 预测器
│   │   ├── sensitivity.py   # 敏感性分析
│   │   └── subgraph.py      # 子图提取
│   ├── utils/               # 工具函数
│   │   ├── __init__.py
│   │   ├── config.py        # 配置管理
│   │   ├── data_processor.py
│   │   ├── file_utils.py
│   │   └── logger.py
│   └── gui.py               # 主入口GUI
├── data/                     # 输入数据
│   ├── design/              # 设计数据
│   ├── lef/                 # LEF文件
│   └── lib/                 # LIB文件
├── models/                   # 训练好的模型
├── intermediate/            # 中间处理数据
├── output/                  # 输出结果
├── logs/                    # 日志文件
├── requirements.txt         # 依赖列表
├── checkpoint.md            # 优化进度记录
└── README.md                # 项目说明
```

## 安装

### 环境要求

- Python 3.7+
- CUDA（推荐）

### 安装依赖

```bash
conda activate ecosight_opt
pip install -r requirements.txt
```

## 使用方法

### 方式一：通过 GUI 运行

```bash
cd ECOsight
python -m src.gui
```

GUI 提供四个主要操作按钮：
1. **数据处理**：解析输入文件，生成中间数据
2. **数据加载**：加载处理好的数据
3. **模型训练**：训练延迟预测模型
4. **自动 ECO**：生成 ECO 命令

### 方式二：命令行运行

```python
from src.utils import Config, DataProcessor
from src.models import Trainer

# 数据处理
config = Config(design="b05")
processor = DataProcessor(config)
processor.run_full_pipeline()

# 模型训练
trainer = Trainer(design="b05", device="cuda")
trainer.fit(graph_list, save_path="models/b05_model.pth")
```

## 输入文件格式

| 文件类型     | 说明        | 示例路径                                                |
| -------- | --------- | --------------------------------------------------- |
| DEF      | 设计交换格式    | `data/design/b05/def/b05_incr.def`                  |
| 网表       | Verilog网表 | `data/design/b05/netlist/b05.v`                     |
| 时序报告     | PT时序报告    | `data/design/b05/timing_report/setup.wo_io_all.rpt` |
| Tech LEF | 工艺LEF     | `data/lef/demo_tech.lef`                            |
| LEF      | 单元LEF     | `data/lef/demo_hvt.lef`                             |
| LIB      | 时序库       | `data/lib/demo_hvt.lib`                             |

## 输出文件

| 文件类型 | 说明 |
|---------|------|
| 模型文件 | `models/{design}_model.pth` |
| ECO命令 | `output/{design}/innovus_eco_command.tcl` |
| 日志文件 | `logs/log_{timestamp}.log` |

## 技术架构

### 图神经网络模型

项目实现了多种异构图神经网络：
- **HeteroGNN_GAT**：基于 GAT 的异构 GNN
- **HeteroGNN_GATv2**：基于 GATv2 的异构 GNN
- **HeteroGNN_Transformer**：基于 Transformer 的异构 GNN
- **HeteroGNN_CG**：基于 CGConv 的异构 GNN（主要使用）

### 图结构

- **节点类型**：`inpin`（输入引脚）、`outpin`（输出引脚）
- **边类型**：`inpin -> cell -> outpin`（单元边）、`outpin -> net -> inpin`（网络边）

## 参考论文

**ECOSight: an explainable graph AI tool for automated decision making in timing ECO**

**Paper describing this work has been received in _Frontiers of Computer Science_****（****FCS****）** **special column “**[**Code & Data**](https://journal.hep.com.cn/fcs/EN/subject/showCollection.do?subjectId=1710741206314)**”.**

**Cited as**: Huiqing YOU, Xiaowei HE, Wencheng JIANG, Bo HU, Peiyun BIAN, Zexiang CHENG, Chaochao FENG,

Daheng LE, Pengcheng HUANG, Chiyuan MA, Zhenyu ZHAO. ECOSight: an explainable graph AI tool for automated decisionmaking in timing ECO. Front. Comput. Sci., 2026, DOI: 10.1007/s11704-026-51891-6


## 联系方式

youhuiqing@nudt.edu.cn
