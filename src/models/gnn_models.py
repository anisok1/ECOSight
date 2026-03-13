# -*- coding: utf-8 -*-
"""
图神经网络模型定义模块
定义了多种异构图神经网络模型用于时序路径延迟预测

模型列表:
- HeteroGNN_GAT: 基于图注意力网络的异构GNN
- HeteroGNN_GATv2: 基于GATv2的异构GNN
- HeteroGNN_Transformer: 基于Transformer的异构GNN
- HeteroGNN_CG: 基于CGConv的异构GNN (主要使用)
"""

import torch
import torch.nn.functional as F
from torch.nn import Sequential, ReLU, Linear, LayerNorm
from torch_geometric.nn import (
    HeteroConv, GATConv, GATv2Conv, 
    TransformerConv, CGConv, BatchNorm,
    global_add_pool, global_mean_pool
)


class HeteroGNN_GAT(torch.nn.Module):
    """
    基于图注意力网络(GAT)的异构图神经网络
    
    使用GAT层处理异构图数据，适用于时序路径延迟预测
    
    Args:
        g: 异构图数据对象，用于获取图结构信息
        hidden_channels: 隐藏层通道数
        out_channels: 输出通道数
        num_layers: GAT层数
        edge_dim: 边特征维度字典
    """
    
    def __init__(self, g, hidden_channels, out_channels, num_layers, edge_dim):
        super().__init__()
        
        # 构建多层GAT卷积
        self.convs = torch.nn.ModuleList()
        self.batch_norms_ins = torch.nn.ModuleList()
        self.batch_norms_outs = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: GATConv(
                    (-1, -1), hidden_channels, heads=1,
                    add_self_loops=False, edge_dim=edge_dim[edge_type], dropout=0
                )
                for edge_type in g.metadata()[1]
            }, aggr='mean')
            self.convs.append(conv)
            self.batch_norms_ins.append(BatchNorm(hidden_channels))
            self.batch_norms_outs.append(BatchNorm(hidden_channels))
        
        # 输出MLP层
        self.mlp = Sequential(
            Linear(hidden_channels, 50), ReLU(),
            Linear(50, 25), ReLU(),
            Linear(25, out_channels)
        )
    
    def forward(self, x_dict, edge_index_dict, edge_attr_dict, batch_in, batch_out):
        """
        前向传播
        
        Args:
            x_dict: 节点特征字典 {节点类型: 特征张量}
            edge_index_dict: 边索引字典 {边类型: 边索引张量}
            edge_attr_dict: 边特征字典 {边类型: 边特征张量}
            batch_in: 输入引脚的batch索引
            batch_out: 输出引脚的batch索引
            
        Returns:
            预测的延迟值
        """
        for conv, batch_norm_in, batch_norm_out in zip(
            self.convs, self.batch_norms_ins, self.batch_norms_outs
        ):
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict["inpin"] = batch_norm_in(x_dict["inpin"])
            x_dict["outpin"] = batch_norm_out(x_dict["outpin"])
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        
        # 全局池化
        x_dict["inpin"] = global_add_pool(x_dict["inpin"], batch_in)
        x_dict["outpin"] = global_add_pool(x_dict["outpin"], batch_out)
        x = x_dict["inpin"] + x_dict["outpin"]
        
        return self.mlp(x)


class HeteroGNN_GATv2(torch.nn.Module):
    """
    基于GATv2的异构图神经网络
    
    GATv2改进了GAT的注意力机制，动态计算注意力分数
    """
    
    def __init__(self, g, hidden_channels, out_channels, num_layers, edge_dim):
        super().__init__()
        
        self.convs = torch.nn.ModuleList()
        self.batch_norms_ins = torch.nn.ModuleList()
        self.batch_norms_outs = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: GATv2Conv(
                    (-1, -1), hidden_channels, heads=1,
                    add_self_loops=False, edge_dim=edge_dim[edge_type], dropout=0
                )
                for edge_type in g.metadata()[1]
            }, aggr='mean')
            self.convs.append(conv)
            self.batch_norms_ins.append(BatchNorm(hidden_channels))
            self.batch_norms_outs.append(BatchNorm(hidden_channels))
        
        self.mlp = Sequential(
            Linear(hidden_channels, 50), ReLU(),
            Linear(50, 25), ReLU(),
            Linear(25, out_channels)
        )
    
    def forward(self, x_dict, edge_index_dict, edge_attr_dict, batch_in, batch_out):
        for conv, batch_norm_in, batch_norm_out in zip(
            self.convs, self.batch_norms_ins, self.batch_norms_outs
        ):
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict["inpin"] = batch_norm_in(x_dict["inpin"])
            x_dict["outpin"] = batch_norm_out(x_dict["outpin"])
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        
        x_dict["inpin"] = global_add_pool(x_dict["inpin"], batch_in)
        x_dict["outpin"] = global_add_pool(x_dict["outpin"], batch_out)
        x = x_dict["inpin"] + x_dict["outpin"]
        
        return self.mlp(x)


class HeteroGNN_Transformer(torch.nn.Module):
    """
    基于Transformer的异构图神经网络
    
    使用Transformer卷积层处理图数据
    """
    
    def __init__(self, g, hidden_channels, out_channels, num_layers, edge_dim):
        super().__init__()
        
        self.convs = torch.nn.ModuleList()
        self.batch_norms_ins = torch.nn.ModuleList()
        self.batch_norms_outs = torch.nn.ModuleList()
        
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: TransformerConv(
                    (-1, -1), hidden_channels, heads=1,
                    dropout=0, edge_dim=edge_dim[edge_type]
                )
                for edge_type in g.metadata()[1]
            }, aggr='mean')
            self.convs.append(conv)
            self.batch_norms_ins.append(BatchNorm(hidden_channels))
            self.batch_norms_outs.append(BatchNorm(hidden_channels))
        
        self.mlp = Sequential(
            Linear(hidden_channels, 50), ReLU(),
            Linear(50, 25), ReLU(),
            Linear(25, out_channels)
        )
    
    def forward(self, x_dict, edge_index_dict, edge_attr_dict, batch_in, batch_out):
        for conv, batch_norm_in, batch_norm_out in zip(
            self.convs, self.batch_norms_ins, self.batch_norms_outs
        ):
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict["inpin"] = batch_norm_in(x_dict["inpin"])
            x_dict["outpin"] = batch_norm_out(x_dict["outpin"])
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        
        x_dict["inpin"] = global_add_pool(x_dict["inpin"], batch_in)
        x_dict["outpin"] = global_add_pool(x_dict["outpin"], batch_out)
        x = x_dict["inpin"] + x_dict["outpin"]
        
        return self.mlp(x)


class HeteroGNN_CG(torch.nn.Module):
    """
    基于CGConv(Compact Graph Convolution)的异构图神经网络
    
    这是项目主要使用的模型，CGConv通过边特征和节点特征
    的组合来学习图表示，效果最好
    
    Args:
        g: 异构图数据对象
        hidden_channels: 隐藏层通道数
        out_channels: 输出通道数
        num_layers: CGConv层数
        edge_dim: 边特征维度字典
        node_dim: 节点特征维度字典
    """
    
    def __init__(self, g, hidden_channels, out_channels, num_layers, edge_dim, node_dim):
        super().__init__()
        
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        # 层归一化
        self.layer_norms = torch.nn.ModuleDict({
            "inpin": LayerNorm(node_dim[("inpin", "cell", "outpin")][0]),
            "outpin": LayerNorm(node_dim[("inpin", "cell", "outpin")][1])
        })
        
        # 边特征MLP
        self.edge_mlps = torch.nn.ModuleDict({
            edge_type[1]: Sequential(
                Linear(edge_dim[edge_type], 32), ReLU(),
                Linear(32, hidden_channels)
            )
            for edge_type in g.metadata()[1]
        })
        
        # 节点特征MLP
        self.mlp_in = Sequential(
            Linear(node_dim[("inpin", "cell", "outpin")][0], 32), 
            ReLU(), 
            Linear(32, hidden_channels)
        )
        self.mlp_out = Sequential(
            Linear(node_dim[("inpin", "cell", "outpin")][1], 32), 
            ReLU(), 
            Linear(32, hidden_channels)
        )
        
        # CGConv层
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: CGConv(channels=hidden_channels, dim=hidden_channels)
                for edge_type in g.metadata()[1]
            }, aggr='mean')
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_channels))
        
        # 输出MLP
        self.mlp = Sequential(
            Linear(hidden_channels * 4, 128), ReLU(),
            Linear(128, 64), ReLU(),
            Linear(64, out_channels)
        )
    
    def forward(self, x_dict, edge_index_dict, edge_attr_dict, batch_in, batch_out):
        """
        前向传播
        
        处理流程:
        1. 对边特征进行MLP变换
        2. 对节点特征进行MLP变换
        3. 通过多层CGConv进行消息传递
        4. 全局池化并输出预测值
        """
        # 边特征变换
        edge_attr_dict = {
            edge_type: self.edge_mlps[edge_type[1]](edge_attr)
            for edge_type, edge_attr in edge_attr_dict.items()
        }
        
        # 节点特征变换
        x_dict["inpin"] = self.mlp_in(x_dict["inpin"])
        x_dict["outpin"] = self.mlp_out(x_dict["outpin"])
        
        # CGConv层处理
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        
        # 全局池化（同时使用add和mean pooling）
        x_dict["inpin"] = torch.cat([
            global_add_pool(x_dict["inpin"], batch_in),
            global_mean_pool(x_dict["inpin"], batch_in)
        ], dim=-1)
        x_dict["outpin"] = torch.cat([
            global_add_pool(x_dict["outpin"], batch_out),
            global_mean_pool(x_dict["outpin"], batch_out)
        ], dim=-1)
        
        # 拼接并输出
        x = torch.cat([x_dict["inpin"], x_dict["outpin"]], dim=-1)
        return self.mlp(x)
