# -*- coding: utf-8 -*-
"""
延迟预测器模块
提供模型加载和延迟预测功能
"""

import torch
import scipy.sparse as ssp
from torch_geometric.utils import convert
from torch_geometric.data import HeteroData
import numpy as np

from .gnn_models import HeteroGNN_CG


class Predictor:
    """
    延迟预测器
    
    负责加载训练好的模型并进行延迟预测
    
    Args:
        model_path: 模型文件路径
        device: 计算设备
    """
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_path = model_path
    
    def load_model(self, sample_graph):
        """
        加载模型
        
        Args:
            sample_graph: 示例图，用于获取特征维度
        """
        # 固定特征维度（与训练时保持一致）
        feature_edge = {
            ("inpin", "cell", "outpin"): 80,
            ("outpin", "net", "inpin"): 4
        }
        node_dim = {
            ("inpin", "cell", "outpin"): (4, 11),
            ("outpin", "net", "inpin"): (11, 4)
        }
        
        # 创建模型
        self.model = HeteroGNN_CG(
            sample_graph, hidden_channels=256, out_channels=1,
            num_layers=2, edge_dim=feature_edge, node_dim=node_dim
        ).to(self.device)
        
        # 加载权重
        state_dict = torch.load(self.model_path, weights_only=True)
        self.model.load_state_dict(state_dict['model'])
        self.model.eval()
    
    def predict(self, graph):
        """
        预测延迟
        
        Args:
            graph: 异构图数据
            
        Returns:
            预测的延迟值
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        self.model.eval()
        with torch.no_grad():
            graph = graph.to(self.device)
            out = self.model(
                graph.x_dict, graph.edge_index_dict, graph.edge_attr_dict,
                graph["inpin"].batch, graph["outpin"].batch
            )
        return out.squeeze()


def model_load(graph, model_name, device):
    """
    加载模型（兼容原接口）
    
    Args:
        graph: 示例图
        model_name: 模型路径
        device: 设备
        
    Returns:
        加载好的模型
    """
    predictor = Predictor(model_name, device)
    predictor.load_model(graph)
    return predictor.model


def predict(graph, model, device):
    """
    预测延迟（兼容原接口）
    
    Args:
        graph: 异构图数据
        model: 模型
        device: 设备
        
    Returns:
        预测值
    """
    model.eval()
    with torch.no_grad():
        graph = graph.to(device)
        out = model(
            graph.x_dict, graph.edge_index_dict, graph.edge_attr_dict,
            graph["inpin"].batch, graph["outpin"].batch
        )
    return torch.squeeze(out)


def path_extraction(cell_link, net_link, cell_feature, in_pin_feature, out_pin_feature,
                   net_feature, LUT_information, path, hops=0):
    """
    从时序路径提取异构图
    
    Args:
        cell_link: 单元边连接
        net_link: 网络边连接
        cell_feature: 单元特征
        in_pin_feature: 输入引脚特征
        out_pin_feature: 输出引脚特征
        net_feature: 网络特征
        LUT_information: 查找表信息
        path: 路径（格式: inst/pin_r inst/pin_f ...）
        hops: 子图跳数
        
    Returns:
        g: 异构图数据
        in_pin_feature_new: 处理后的输入引脚特征
        out_pin_feature_new: 处理后的输出引脚特征
        ...
    """
    # 创建邻接矩阵
    max_idx_in = np.max(cell_link[:, 0])
    max_idx_out = np.max(cell_link[:, 1])
    cell_matrix = _create_matrix(cell_link, max_idx_in, max_idx_out)
    net_matrix = _create_matrix(net_link, max_idx_out, max_idx_in)
    
    # 创建上升/下降标志列表
    out_rf_list = _create_rf_list(path, max_idx_in, max_idx_out, net_matrix)
    
    # 提取子图
    subgraph_cell, subgraph_net, in_pin_feature_new, out_pin_feature_new, out_rf_list_new, \
        in_pin_feature_origin, out_pin_feature_origin = _get_subgraphs(
            hops, path, cell_matrix, net_matrix, in_pin_feature, out_pin_feature, out_rf_list
        )
    
    # 转换为PyG格式
    edge_index_cell, edge_weight_cell = convert.from_scipy_sparse_matrix(subgraph_cell)
    edge_index_net, edge_weight_net = convert.from_scipy_sparse_matrix(subgraph_net)
    
    # 准备节点特征
    InPinFeature = in_pin_feature_new[["cap", "position_x", "position_y", "layer"]].to_numpy() * [1000, 1, 1, 1]
    OutPinFeature = out_pin_feature_new[["load", "position_x", "position_y", "layer"]].to_numpy() * [1000, 1, 1, 1]
    out_rf_list_new = np.array(out_rf_list_new).reshape(-1, 1)
    OutPinFeature = np.hstack((OutPinFeature, out_pin_feature_new.iloc[:, 7:13].to_numpy(), out_rf_list_new))
    
    # 准备边特征
    CellFeature = []
    edge_weight_cell_int = edge_weight_cell.to(torch.int) - 1
    for i, cell_idx in enumerate(edge_weight_cell_int):
        cell_idx = cell_idx.item()
        std_cell = cell_feature.loc[cell_idx, "std_cell"]
        rf = out_rf_list_new[edge_index_cell[1, i]]
        pin_in = in_pin_feature_new.loc[int(edge_index_cell[0, i]), "inst_pin"].rsplit("_", 1)[-1]
        pin_out = out_pin_feature_new.loc[int(edge_index_cell[1, i]), "inst_pin"].rsplit("_", 1)[-1]
        name = f"{std_cell}_{pin_in}_{pin_out}_{'r' if rf == 1 else 'f'}"
        feature_cell = LUT_information[name]
        CellFeature.append(feature_cell)
    
    edge_weight_net_int = edge_weight_net.to(torch.int) - 1
    if isinstance(edge_weight_net_int, torch.Tensor):
        edge_weight_net_int = edge_weight_net_int.cpu().numpy().astype(int)
    Netfeature = net_feature.iloc[edge_weight_net_int, 1:5].to_numpy()
    
    # 构建异构图
    g = HeteroData()
    g['inpin'].x = torch.tensor(InPinFeature).float()
    g['outpin'].x = torch.tensor(OutPinFeature).float()
    g["inpin", "cell", "outpin"].edge_attr = torch.tensor(CellFeature).float()
    g["outpin", "net", "inpin"].edge_attr = torch.tensor(Netfeature).float()
    g["inpin", "cell", "outpin"].edge_index = torch.tensor(edge_index_cell)
    g["outpin", "net", "inpin"].edge_index = torch.tensor(edge_index_net)
    
    return g, in_pin_feature_new, out_pin_feature_new, in_pin_feature_origin, \
           out_pin_feature_origin, edge_weight_cell, edge_weight_net


def _create_matrix(links_idx, max_idx_s, max_idx_d):
    """创建稀疏邻接矩阵"""
    return ssp.csc_matrix(
        (np.arange(1, len(links_idx) + 1), (links_idx[:, 0], links_idx[:, 1])),
        shape=(max_idx_s + 1, max_idx_d + 1)
    )


def _create_rf_list(path, max_idx_in, max_idx_out, net_matrix):
    """创建上升/下降标志列表"""
    in_list = np.full(max_idx_in + 1, -1, dtype=np.int8)
    out_list = np.full(max_idx_out + 1, -1, dtype=np.int8)
    
    path_idx = np.array([int(p[:-1]) for p in path], dtype=np.int32)
    path_rf = np.array([1 if p[-1] == "r" else 0 for p in path], dtype=np.int8)
    
    for i, (idx, rf) in enumerate(zip(path_idx, path_rf)):
        if i % 2 == 0:
            in_list[idx] = rf
            nei = ssp.find(net_matrix[:, idx])[0]
            out_list[nei] = rf
        else:
            out_list[idx] = rf
    
    return out_list


def _get_subgraphs(hops, path, cell_matrix, net_matrix, in_pin_feature, out_pin_feature, out_rf_list):
    """提取子图"""
    in_nodes = [int(path[i][:-1]) for i in range(len(path)) if i % 2 == 0]
    out_nodes = [int(path[i][:-1]) for i in range(len(path)) if i % 2 != 0]
    
    subgraph_cell = cell_matrix[in_nodes, :][:, out_nodes]
    subgraph_net = net_matrix[out_nodes, :][:, in_nodes]
    
    in_pin_feature_new = in_pin_feature.iloc[in_nodes, :].reset_index(drop=True)
    out_pin_feature_new = out_pin_feature.iloc[out_nodes, :].reset_index(drop=True)
    out_rf_list_new = [out_rf_list[i] for i in out_nodes]
    
    return subgraph_cell, subgraph_net, in_pin_feature_new, out_pin_feature_new, \
           out_rf_list_new, in_pin_feature.iloc[in_nodes, :], out_pin_feature.iloc[out_nodes, :]
