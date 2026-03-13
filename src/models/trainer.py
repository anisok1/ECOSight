# -*- coding: utf-8 -*-
"""
模型训练器模块
提供模型训练、验证和保存功能
"""

import torch
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import SmoothL1Loss
import time
import os
import logging
from .gnn_models import HeteroGNN_GAT, HeteroGNN_GATv2, HeteroGNN_Transformer, HeteroGNN_CG


class Trainer:
    """
    模型训练器
    
    负责图神经网络模型的训练、验证和模型保存
    
    Args:
        design: 设计名称
        device: 计算设备 ('cuda' 或 'cpu')
        model_type: 模型类型 ('GAT', 'GATv2', 'Transformer', 'CG')
        hidden_channels: 隐藏层通道数
        num_layers: 网络层数
        batch_size: 批次大小
        train_ratio: 训练集比例
        epochs: 训练轮数
    """
    
    # 支持的模型类型映射
    MODEL_MAP = {
        'GAT': HeteroGNN_GAT,
        'GATv2': HeteroGNN_GATv2,
        'Transformer': HeteroGNN_Transformer,
        'CG': HeteroGNN_CG
    }
    
    def __init__(self, design='b05', device='cuda', model_type='CG',
                 hidden_channels=256, num_layers=2, batch_size=32,
                 train_ratio=0.8, epochs=500):
        self.design = design
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.epochs = epochs
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = SmoothL1Loss()
        
        logging.info(f"Trainer初始化完成: design={design}, device={self.device}, model={model_type}")
    
    def build_model(self, sample_graph):
        """
        构建模型
        
        Args:
            sample_graph: 示例图，用于获取特征维度
        """
        # 获取特征维度
        feature_edge = {}
        feature_edge[("inpin", "cell", "outpin")] = sample_graph["inpin", "cell", "outpin"].edge_attr.shape[1]
        feature_edge[("outpin", "net", "inpin")] = sample_graph["outpin", "net", "inpin"].edge_attr.shape[1]
        
        node_dim = {}
        node_dim[("inpin", "cell", "outpin")] = (
            sample_graph['inpin'].x.shape[1],
            sample_graph['outpin'].x.shape[1]
        )
        node_dim[("outpin", "net", "inpin")] = (
            sample_graph['outpin'].x.shape[1],
            sample_graph['inpin'].x.shape[1]
        )
        
        # 创建模型实例
        model_class = self.MODEL_MAP.get(self.model_type, HeteroGNN_CG)
        
        if self.model_type == 'CG':
            self.model = model_class(
                sample_graph, self.hidden_channels, 1, self.num_layers,
                edge_dim=feature_edge, node_dim=node_dim
            ).to(self.device)
        else:
            self.model = model_class(
                sample_graph, self.hidden_channels, 1, self.num_layers,
                edge_dim=feature_edge
            ).to(self.device)
        
        # 初始化优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.001, weight_decay=1e-5
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=20, min_lr=0.00001
        )
        
        logging.info(f"模型构建完成: {self.model_type}")
    
    def load_pretrained(self, model_path):
        """加载预训练模型"""
        state_dict = torch.load(model_path, weights_only=True)
        self.model.load_state_dict(state_dict['model'])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.00001, weight_decay=1e-5
        )
        logging.info(f"加载预训练模型: {model_path}")
    
    def train_epoch(self, train_loader):
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            loss: 平均损失
            r2: R2分数
        """
        self.model.train()
        total_loss = 0
        outputs = []
        labels = []
        
        for data in train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            out = self.model(
                data.x_dict, data.edge_index_dict, data.edge_attr_dict,
                data["inpin"].batch, data["outpin"].batch
            )
            
            loss = (out.squeeze() - data.x_dict["label"]).abs().mean()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * data.num_graphs
            outputs.append(out.squeeze().detach())
            labels.append(data.x_dict["label"])
        
        # 计算R2分数
        outputs = torch.cat(outputs)
        labels = torch.cat(labels)
        r2 = self._compute_r2(outputs, labels)
        
        return total_loss / len(train_loader.dataset), r2
    
    def validate(self, val_loader):
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            loss: 平均损失
            r2: R2分数
        """
        self.model.eval()
        total_loss = 0
        outputs = []
        labels = []
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)
                out = self.model(
                    data.x_dict, data.edge_index_dict, data.edge_attr_dict,
                    data["inpin"].batch, data["outpin"].batch
                )
                loss = (out.squeeze() - data.x_dict["label"]).abs().mean()
                total_loss += loss.item() * data.num_graphs
                outputs.append(out.squeeze())
                labels.append(data.x_dict["label"])
        
        outputs = torch.cat(outputs)
        labels = torch.cat(labels)
        r2 = self._compute_r2(outputs, labels)
        
        return total_loss / len(val_loader.dataset), r2
    
    def fit(self, graph_list, save_path=None, log_path=None):
        """
        训练模型
        
        Args:
            graph_list: 图数据列表
            save_path: 模型保存路径
            log_path: 日志文件路径
            
        Returns:
            best_loss: 最佳验证损失
        """
        # 构建模型
        if self.model is None:
            self.build_model(graph_list[0])
        
        # 划分训练集和验证集
        train_size = int(self.train_ratio * len(graph_list))
        train_loader = DataLoader(graph_list[:train_size], batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(graph_list[train_size:], batch_size=self.batch_size, shuffle=True)
        
        # 打开日志文件
        log_file = open(log_path, 'w') if log_path else None
        
        best_loss = None
        start_time = time.time()
        
        for epoch in range(1, self.epochs + 1):
            train_loss, train_r2 = self.train_epoch(train_loader)
            val_loss, val_r2 = self.validate(val_loader)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 记录日志
            msg = f'Epoch {epoch:03d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, r2={val_r2:.4f}'
            logging.info(msg)
            if log_file:
                log_file.write(msg + '\n')
            
            # 保存最佳模型
            if best_loss is None or val_loss < best_loss:
                best_loss = val_loss
                if save_path:
                    self.save(save_path)
        
        elapsed = time.time() - start_time
        logging.info(f'训练完成，耗时: {elapsed:.1f}s')
        
        if log_file:
            log_file.write(f'训练耗时: {elapsed:.1f}s\n')
            log_file.close()
        
        return best_loss
    
    def save(self, path):
        """保存模型"""
        torch.save({'model': self.model.state_dict()}, path)
        logging.info(f'模型已保存: {path}')
    
    def _compute_r2(self, output, target):
        """计算R2分数"""
        target_mean = torch.mean(target)
        ss_tot = torch.sum((target - target_mean) ** 2)
        ss_res = torch.sum((target - output) ** 2)
        return 1 - ss_res / ss_tot


def model_generation(design, hop, bs, nl, tn, hc, rf, epochs, device, pn, model,
                    CellEdgeLink, NetEdgeLink, CellEdgeFeature, NetEdgeFeature,
                    InpinFeature, OutpinFeature, LUT_information_dict, rpttxt, pathtxt,
                    module_name_new, **kwargs):
    """
    模型训练主函数（兼容原接口）
    
    Args:
        design: 设计名称
        hop: 子图跳数
        bs: 批次大小
        nl: 网络层数
        tn: 训练集比例
        hc: 隐藏通道数
        rf: 结果文件名
        epochs: 训练轮数
        device: 设备
        pn: 路径数量
        model: 模型类型
        ...其他特征参数
    """
    from .subgraph import get_batched_graph
    import random
    
    # 准备目录
    os.makedirs(f"../result/{design}", exist_ok=True)
    os.makedirs("../model", exist_ok=True)
    
    # 采样路径
    pn = min(pn, len(pathtxt))
    random_indices = random.sample(range(len(pathtxt)), pn)
    random_paths = [pathtxt[i] for i in random_indices]
    random_rpt = rpttxt[random_indices]
    
    # 提取子图
    graph_list = get_batched_graph(
        CellEdgeLink, NetEdgeLink, CellEdgeFeature, InpinFeature,
        OutpinFeature, NetEdgeFeature, LUT_information_dict,
        random_rpt, random_paths, hop
    )
    
    # 创建训练器并训练
    trainer = Trainer(
        design=design, device=device, model_type=model,
        hidden_channels=hc, num_layers=nl, batch_size=bs,
        train_ratio=tn, epochs=epochs
    )
    
    log_path = f"../result/{design}/{rf}.txt" if rf else None
    trainer.fit(graph_list, save_path=module_name_new, log_path=log_path)