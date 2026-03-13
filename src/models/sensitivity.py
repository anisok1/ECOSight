# -*- coding: utf-8 -*-
"""
敏感性分析模块
提供基于梯度和特征遮掩的敏感性分析方法
"""

import torch
from torch_geometric.loader import DataLoader
import numpy as np


class SensitivityAnalyzer:
    """
    敏感性分析器
    
    分析图神经网络中各特征对预测结果的影响程度
    
    Args:
        model: 训练好的模型
        device: 计算设备
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    def gradients_based(self, graph_data, except_list=None):
        """
        基于梯度的敏感性分析
        
        通过计算输出对各特征的梯度来确定敏感性
        
        Args:
            graph_data: 图数据
            except_list: 排除的索引列表 [(index, type), ...]
            
        Returns:
            max_index: 最敏感特征的索引
            type: 特征类型 ('cell', 'net', 'cell_load', 'net_load')
        """
        if except_list is None:
            except_list = []
        
        graph = graph_data.to(self.device)
        self.model.eval()
        
        # 启用梯度追踪
        for node_type, node_feat in graph.x_dict.items():
            node_feat.requires_grad = True
        for edge_type, edge_feat in graph.edge_attr_dict.items():
            edge_feat.requires_grad = True
        
        # 前向传播
        train_loader = DataLoader([graph], batch_size=1)
        for g in train_loader:
            g = g.to(self.device)
            out = self.model(
                g.x_dict, g.edge_index_dict, g.edge_attr_dict,
                g["inpin"].batch, g["outpin"].batch
            )
        
        out.squeeze().backward()
        
        # 提取梯度
        node_gradients = {nt: g.x_dict[nt].grad.clone() for nt in g.x_dict.keys()}
        edge_gradients = {et: g.edge_attr_dict[et].grad.clone() for et in g.edge_attr_dict.keys()}
        
        # 分析输出引脚特征梯度
        feature_range = slice(7, 13)
        grad = node_gradients["outpin"].abs()
        feature_grad_cap = grad[:, 0]
        feature_grad_net = grad[:, feature_range].mean(dim=1)
        
        # 应用排除列表
        for index, t in except_list:
            if t == "cell_load":
                feature_grad_cap[index] = 0
            elif t == "net_load":
                feature_grad_net[index] = 0
        
        # 找最大梯度
        max_grad = float("-inf")
        max_index = 0
        type_str = ""
        
        max_values, max_cap_index = feature_grad_cap.max(dim=0)
        if max_grad < max_values:
            max_grad = max_values
            max_index = int(max_cap_index.item())
            type_str = "cell_load"
        
        max_values, max_net_index = feature_grad_net.max(dim=0)
        if max_grad < max_values:
            max_grad = max_values
            max_index = int(max_net_index.item())
            type_str = "net_load"
        
        # 分析边特征梯度
        for edge_type, grad in edge_gradients.items():
            grad_mean = grad.abs().mean(dim=1)
            edge_type_label = "cell" if edge_type == ('inpin', 'cell', 'outpin') else "net"
            
            if edge_type_label == "cell":
                for index, t in except_list:
                    if t == "cell":
                        grad_mean[index] = 0
                
                max_values, max_cell_index = grad_mean.max(dim=0)
                if max_grad < max_values:
                    max_grad = max_values
                    max_index = int(max_cell_index.item())
                    type_str = "cell"
        
        return max_index, type_str
    
    def gradients_based_list(self, graph_data):
        """
        基于梯度的敏感性分析（返回排序列表）
        
        Args:
            graph_data: 图数据
            
        Returns:
            排序后的敏感性列表 [(敏感度, 索引, 类型), ...]
        """
        graph = graph_data.to(self.device)
        self.model.eval()
        
        # 启用梯度追踪
        for node_type, node_feat in graph.x_dict.items():
            node_feat.requires_grad = True
        for edge_type, edge_feat in graph.edge_attr_dict.items():
            edge_feat.requires_grad = True
        
        train_loader = DataLoader([graph], batch_size=1)
        for g in train_loader:
            g = g.to(self.device)
            out = self.model(
                g.x_dict, g.edge_index_dict, g.edge_attr_dict,
                g["inpin"].batch, g["outpin"].batch
            )
        
        out.squeeze().backward()
        
        node_gradients = {nt: g.x_dict[nt].grad for nt in g.x_dict.keys()}
        edge_gradients = {et: g.edge_attr_dict[et].grad for et in g.edge_attr_dict.keys()}
        
        # 收集所有敏感性
        indices_and_type = []
        grad = node_gradients["outpin"].abs()
        feature_grad_cap = grad[:, 0]
        
        indices_and_type.extend([
            (value.item(), idx, "cell_load")
            for idx, value in enumerate(feature_grad_cap)
        ])
        
        for edge_type, grad in edge_gradients.items():
            grad_mean = grad.abs().mean(dim=1)
            edge_type_label = "cell" if edge_type == ('inpin', 'cell', 'outpin') else "net"
            if edge_type_label == "cell":
                indices_and_type.extend([
                    (value.item(), idx, edge_type_label)
                    for idx, value in enumerate(grad_mean)
                ])
        
        # 按敏感度降序排序
        indices_and_type.sort(reverse=True, key=lambda x: x[0])
        return indices_and_type
    
    def feature_masking_based(self, graph_data, except_list=None):
        """
        基于特征遮掩的敏感性分析
        
        通过逐个遮掩特征并观察输出变化来确定敏感性
        
        Args:
            graph_data: 图数据
            except_list: 排除的索引列表
            
        Returns:
            max_index: 最敏感特征的索引
            type: 特征类型
        """
        if except_list is None:
            except_list = []
        
        graph = graph_data.to(self.device)
        self.model.eval()
        
        train_loader = DataLoader([graph], batch_size=1)
        max_score = float("-inf")
        max_index = 0
        type_str = ""
        
        for g in train_loader:
            # 原始输出
            original_output = self.model(
                g.x_dict, g.edge_index_dict, g.edge_attr_dict,
                g["inpin"].batch, g["outpin"].batch
            )
            original_value = original_output.squeeze().item()
            
            # 分析输出引脚特征
            num_nodes = g.x_dict["outpin"].shape[0]
            for node_index in range(num_nodes):
                # cap特征
                cap = g.x_dict["outpin"][node_index, 0].clone()
                g.x_dict["outpin"][node_index, 0] = 0
                masked_output = self.model(
                    g.x_dict, g.edge_index_dict, g.edge_attr_dict,
                    g["inpin"].batch, g["outpin"].batch
                )
                score = abs(original_value - masked_output.squeeze().item())
                g.x_dict["outpin"][node_index, 0] = cap
                
                if score > max_score and (node_index, "cell_load") not in except_list:
                    max_score = score
                    max_index = node_index
                    type_str = "cell_load"
                
                # net特征
                net = g.x_dict["outpin"][node_index, 7:13].clone()
                g.x_dict["outpin"][node_index, 7:13] = 0
                masked_output = self.model(
                    g.x_dict, g.edge_index_dict, g.edge_attr_dict,
                    g["inpin"].batch, g["outpin"].batch
                )
                score = abs(original_value - masked_output.squeeze().item())
                g.x_dict["outpin"][node_index, 7:13] = net
                
                if score > max_score and (node_index, "net_load") not in except_list:
                    max_score = score
                    max_index = node_index
                    type_str = "net_load"
            
            # 分析边特征
            num_cell = g.edge_attr_dict[('inpin', 'cell', 'outpin')].shape[0]
            for cell_index in range(num_cell):
                cell = g.edge_attr_dict[('inpin', 'cell', 'outpin')][cell_index, :].clone()
                g.edge_attr_dict[('inpin', 'cell', 'outpin')][cell_index, :] = 0
                masked_output = self.model(
                    g.x_dict, g.edge_index_dict, g.edge_attr_dict,
                    g["inpin"].batch, g["outpin"].batch
                )
                score = abs(original_value - masked_output.squeeze().item())
                g.edge_attr_dict[('inpin', 'cell', 'outpin')][cell_index, :] = cell
                
                if score > max_score and (cell_index, "cell") not in except_list:
                    max_score = score
                    max_index = cell_index
                    type_str = "cell"
        
        return max_index, type_str


# 兼容原接口的函数
def gradients_based(g_data, model, except_list=None):
    """基于梯度的敏感性分析（兼容原接口）"""
    analyzer = SensitivityAnalyzer(model)
    return analyzer.gradients_based(g_data, except_list or [])


def gradients_based_list(g_data, model):
    """基于梯度的敏感性分析列表（兼容原接口）"""
    analyzer = SensitivityAnalyzer(model)
    return analyzer.gradients_based_list(g_data)


def feature_masking_based_sensitivity(g_data, model, except_list=None):
    """基于特征遮掩的敏感性分析（兼容原接口）"""
    analyzer = SensitivityAnalyzer(model)
    return analyzer.feature_masking_based(g_data, except_list or [])
