# -*- coding: utf-8 -*-
"""
子图提取模块
提供从时序路径中提取异构子图的功能
"""

import torch
from tqdm import tqdm
import numpy as np
import scipy.sparse as ssp
from torch_geometric.data import HeteroData
from torch_geometric.utils import convert


class SubgraphExtractor:
    """
    子图提取器
    
    从时序路径数据中提取异构图数据用于模型训练
    
    Args:
        cell_link: 单元边连接矩阵
        net_link: 网络边连接矩阵
        cell_feature: 单元特征
        in_pin_feature: 输入引脚特征
        out_pin_feature: 输出引脚特征
        net_feature: 网络特征
        LUT_information: 延迟查找表
    """
    
    def __init__(self, cell_link, net_link, cell_feature, in_pin_feature,
                 out_pin_feature, net_feature, LUT_information):
        self.cell_link = cell_link
        self.net_link = net_link
        self.cell_feature = cell_feature
        self.in_pin_feature = in_pin_feature
        self.out_pin_feature = out_pin_feature
        self.net_feature = net_feature
        self.LUT_information = LUT_information
        
        # 预计算邻接矩阵
        self.max_idx_in = np.max(cell_link[:, 0])
        self.max_idx_out = np.max(cell_link[:, 1])
        self.cell_matrix = self._create_matrix(cell_link, self.max_idx_in, self.max_idx_out)
        self.net_matrix = self._create_matrix(net_link, self.max_idx_out, self.max_idx_in)
    
    def _create_matrix(self, links_idx, max_idx_s, max_idx_d):
        """创建稀疏邻接矩阵"""
        return ssp.csc_matrix(
            (np.arange(1, len(links_idx) + 1), (links_idx[:, 0], links_idx[:, 1])),
            shape=(max_idx_s + 1, max_idx_d + 1)
        )
    
    def extract_batch(self, labels, paths, hops=0):
        """
        批量提取子图
        
        Args:
            labels: 延迟标签列表
            paths: 路径列表
            hops: 子图扩展跳数
            
        Returns:
            图数据列表
        """
        graph_list = []
        
        for iii in tqdm(range(len(paths)), desc='提取子图'):
            path = paths[iii].strip().split(" ")
            label = labels[iii]
            
            graph = self._extract_single(path, label, hops)
            graph_list.append(graph)
        
        return graph_list
    
    def _extract_single(self, path, label, hops):
        """提取单个子图"""
        # 创建RF列表
        out_rf_list = self._create_rf_list(path)
        
        # 获取子图节点集合
        subgraph_cell, subgraph_net, in_pin_feature_new, out_pin_feature_new, out_rf_list_new = \
            self._get_subgraphs(hops, path, out_rf_list)
        
        # 转换边索引
        edge_index_cell, edge_weight_cell = convert.from_scipy_sparse_matrix(subgraph_cell)
        edge_index_net, edge_weight_net = convert.from_scipy_sparse_matrix(subgraph_net)
        
        # 准备节点特征
        InPinFeature = self._prepare_inpin_features(in_pin_feature_new)
        OutPinFeature = self._prepare_outpin_features(out_pin_feature_new, out_rf_list_new)
        
        # 准备边特征
        CellFeature = self._prepare_cell_features(
            edge_index_cell, edge_weight_cell, in_pin_feature_new, 
            out_pin_feature_new, out_rf_list_new
        )
        Netfeature = self._prepare_net_features(edge_weight_net)
        
        # 构建异构图
        g = HeteroData()
        g['inpin'].x = torch.tensor(InPinFeature).float()
        g['outpin'].x = torch.tensor(OutPinFeature).float()
        g["label"].x = torch.tensor(label).unsqueeze(0).float()
        g["inpin", "cell", "outpin"].edge_attr = torch.tensor(CellFeature).float()
        g["outpin", "net", "inpin"].edge_attr = torch.tensor(Netfeature).float()
        g["inpin", "cell", "outpin"].edge_index = torch.tensor(edge_index_cell)
        g["outpin", "net", "inpin"].edge_index = torch.tensor(edge_index_net)
        
        return g
    
    def _create_rf_list(self, path):
        """创建上升/下降标志列表"""
        in_list = [-1] * (self.max_idx_in + 1)
        out_list = [-1] * (self.max_idx_out + 1)
        
        for i in range(len(path)):
            if i % 2 == 0:
                idx = int(path[i][:-1])
                rf = path[i][-1]
                in_list[idx] = 1 if rf == "r" else 0
                nei, _, _ = ssp.find(self.net_matrix[:, idx])
                for ii in nei:
                    out_list[ii] = 1 if rf == "r" else 0
            else:
                idx = int(path[i][:-1])
                rf = path[i][-1]
                out_list[idx] = 1 if rf == "r" else 0
        
        return out_list
    
    def _get_subgraphs(self, hops, path, out_rf_list):
        """获取子图节点和特征"""
        in_nodes = set()
        out_nodes = set()
        out_rf_list_new = []
        
        for i in range(len(path)):
            if i % 2 == 0:
                in_nodes.add(int(path[i][:-1]))
            else:
                out_nodes.add(int(path[i][:-1]))
        
        # 扩展邻居
        for _ in range(hops):
            in_nodes, out_nodes = self._get_neighbors(in_nodes, out_nodes)
        
        in_nodes = list(in_nodes)
        out_nodes = list(out_nodes)
        
        subgraph_cell = self.cell_matrix[in_nodes, :][:, out_nodes]
        subgraph_net = self.net_matrix[out_nodes, :][:, in_nodes]
        
        in_pin_feature_new = self.in_pin_feature.iloc[in_nodes, :].reset_index(drop=True)
        out_pin_feature_new = self.out_pin_feature.iloc[out_nodes, :].reset_index(drop=True)
        
        for i in out_nodes:
            out_rf_list_new.append(out_rf_list[i])
        
        return subgraph_cell, subgraph_net, in_pin_feature_new, out_pin_feature_new, out_rf_list_new
    
    def _get_neighbors(self, in_nodes, out_nodes):
        """获取邻居节点"""
        out_res = set()
        in_res = set()
        
        for node in in_nodes:
            nei, _, _ = ssp.find(self.net_matrix[:, node])
            out_res.update(nei)
        
        for node in out_nodes:
            nei, _, _ = ssp.find(self.cell_matrix[:, node])
            in_res.update(nei)
        
        return in_res, out_res
    
    def _prepare_inpin_features(self, in_pin_feature_new):
        """准备输入引脚特征"""
        features = []
        for i in range(in_pin_feature_new.shape[0]):
            feature = [
                in_pin_feature_new.loc[i, "cap"] * 1000,
                float(in_pin_feature_new.loc[i, "position_x"]),
                float(in_pin_feature_new.loc[i, "position_y"]),
                float(in_pin_feature_new.loc[i, "layer"])
            ]
            features.append(feature)
        return features
    
    def _prepare_outpin_features(self, out_pin_feature_new, out_rf_list_new):
        """准备输出引脚特征"""
        features = []
        for i in range(out_pin_feature_new.shape[0]):
            feature = [
                round(float(out_pin_feature_new.loc[i, "load"]) * 1000, 3),
                float(out_pin_feature_new.loc[i, "position_x"]),
                float(out_pin_feature_new.loc[i, "position_y"]),
                float(out_pin_feature_new.loc[i, "layer"])
            ]
            feature += list(out_pin_feature_new.iloc[i, 7:13])
            feature.append(out_rf_list_new[i])
            features.append(feature)
        return features
    
    def _prepare_cell_features(self, edge_index_cell, edge_weight_cell, 
                               in_pin_feature_new, out_pin_feature_new, out_rf_list_new):
        """准备单元边特征"""
        features = []
        for i in range(len(edge_weight_cell)):
            cell_index = int(edge_weight_cell[i] - 1)
            std_cell = self.cell_feature.loc[cell_index, "std_cell"]
            rf = out_rf_list_new[edge_index_cell[1][i]]
            
            pin_in = in_pin_feature_new.loc[int(edge_index_cell[0][i]), "inst_pin"].rsplit("_", 1)[-1]
            pin_out = out_pin_feature_new.loc[int(edge_index_cell[1][i]), "inst_pin"].rsplit("_", 1)[-1]
            
            name = f"{std_cell}_{pin_in}_{pin_out}_{'r' if rf == 1 else 'f'}"
            c = self.LUT_information[name]
            features.append(list(map(float, c)))
        
        return features
    
    def _prepare_net_features(self, edge_weight_net):
        """准备网络边特征"""
        features = []
        for i in range(len(edge_weight_net)):
            net_index = int(edge_weight_net[i] - 1)
            feature = self.net_feature.iloc[net_index, 1:5].tolist()
            features.append(feature)
        return features


def get_batched_graph(cell_link, net_link, cell_feature, in_pin_feature, out_pin_feature,
                     net_feature, LUT_information, labels, paths, hops):
    """
    批量提取子图（兼容原接口）
    
    Args:
        cell_link: 单元边连接
        net_link: 网络边连接
        cell_feature: 单元特征
        in_pin_feature: 输入引脚特征
        out_pin_feature: 输出引脚特征
        net_feature: 网络特征
        LUT_information: 延迟查找表
        labels: 延迟标签
        paths: 路径列表
        hops: 子图跳数
        
    Returns:
        图数据列表
    """
    extractor = SubgraphExtractor(
        cell_link, net_link, cell_feature, in_pin_feature,
        out_pin_feature, net_feature, LUT_information
    )
    return extractor.extract_batch(labels, paths, hops)


def r2_loss(output, target):
    """计算R2损失"""
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    return 1 - ss_res / ss_tot
