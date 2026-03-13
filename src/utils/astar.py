# -*- coding: utf-8 -*-
"""
A*寻路算法模块
提供三维布线路径搜索功能，支持多层金属层间的路径规划

主要功能:
- A*算法寻路
- 路径长度计算
- 启发式函数计算
"""

import heapq
import numpy as np
from typing import List, Tuple, Dict, Optional

# 导入数据结构
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parsers.data_structures import Node


def heuristic(node: Node, end_node: Node, goal_list: List[Tuple] = None) -> float:
    """
    计算节点到目标节点的估计代价（启发式函数）
    
    使用曼哈顿距离作为启发式函数，考虑多个目标节点的情况
    
    Args:
        node: 当前节点
        end_node: 目标节点
        goal_list: 其他目标节点列表（用于多目标优化）
    
    Returns:
        float: 估计代价
    """
    # 基本曼哈顿距离
    result = abs(node.position[1] - end_node.position[1]) + \
             abs(node.position[2] - end_node.position[2])
    
    # 如果有多个目标点，考虑平均距离
    if goal_list:
        for goal in goal_list:
            result += (abs(node.position[1] - goal[1]) + 
                      abs(node.position[2] - goal[2])) * (1 / len(goal_list))
    
    return result


def get_neighbor(loc: Tuple[float, float], biasx: float, biasy: float, 
                 stepx: float, stepy: float) -> Tuple[int, int]:
    """
    计算最近邻的网格坐标
    
    根据给定的位置和轨道参数，计算最近的轨道网格坐标
    
    Args:
        loc: 当前位置 (x, y)
        biasx: X方向偏移
        biasy: Y方向偏移
        stepx: X方向步长
        stepy: Y方向步长
    
    Returns:
        Tuple[int, int]: 最近网格坐标 (x, y)
    """
    nearest_x = round((loc[0] - biasx) / stepx)
    nearest_y = round((loc[1] - biasy) / stepy)
    return nearest_x, nearest_y


def calculate_path_length(path: List[Tuple], track_info: Dict, 
                         metal_list: List[str], maze: List[np.ndarray]) -> Tuple[Dict, Dict, int]:
    """
    计算路径的长度信息
    
    分析路径在每一金属层的长度，包括已布线部分和总长度
    
    Args:
        path: 路径点列表 [(layer, x, y), ...]
        track_info: 轨道信息字典
        metal_list: 金属层列表
        maze: 布线矩阵列表
    
    Returns:
        Tuple[Dict, Dict, int]: 
            - layer_lengths: 每层有效布线长度
            - layer_lengths_all: 每层总长度
            - via_num: 通孔数量
    """
    layer_lengths = {i: 0 for i in range(len(metal_list))}
    layer_lengths_all = {i: 0 for i in range(len(metal_list))}
    via_num = 0
    
    for i in range(1, len(path)):
        z0, x0, y0 = path[i - 1]
        z1, x1, y1 = path[i]
        
        # 同一层内Y方向移动
        if z0 == z1 and abs(x1 - x0) < 0.00001:
            dy = abs(y1 - y0)
            num = 0
            distance = dy * track_info[metal_list[z0]].stepy
            layer_lengths_all[z0] += distance
            
            # 统计有效布线段
            for j in range(min(y0, y1), max(y0, y1) + 1):
                if maze[z1][j][x1] != 2:
                    num += 1
            layer_lengths[z0] += num * track_info[metal_list[z0]].stepy
            
        # 同一层内X方向移动
        elif z0 == z1 and abs(y1 - y0) < 0.00001:
            dx = abs(x1 - x0)
            num = 0
            distance = dx * track_info[metal_list[z0]].stepx
            layer_lengths_all[z0] += distance
            
            for j in range(min(x0, x1), max(x0, x1) + 1):
                if maze[z1][y1][j] != 2:
                    num += 1
            layer_lengths[z0] += num * track_info[metal_list[z0]].stepx
            
        # 层间移动（通孔）
        elif abs(y1 - y0) < 0.00001 or abs(x1 - x0) < 0.00001:
            if maze[z1][y1][x1] != 2:
                via_num += 1
        else:
            print(f"路径错误: {z0, x0, y0} -> {z1, x1, y1}")
    
    return layer_lengths, layer_lengths_all, via_num


class AStarPathfinder:
    """
    A*寻路算法类
    
    用于在三维布线矩阵中寻找两点间的最优路径
    支持多层金属层间的路径规划
    """
    
    def __init__(self, metal_list: List[str], track_info: Dict):
        """
        初始化寻路器
        
        Args:
            metal_list: 金属层名称列表
            track_info: 轨道信息字典 {metal_name: Track对象}
        """
        self.metal_list = metal_list
        self.track_info = track_info
        self.layer = len(metal_list)
    
    def find_path(self, maze: List[np.ndarray], wire_path: List[np.ndarray],
                  startpointz: int, startpointx: float, startpointy: float,
                  endpointz: int, endpointx: float, endpointy: float,
                  other_nodes: List = None, cost: List[float] = None) -> Optional[Tuple]:
        """
        使用A*算法寻找路径
        
        Args:
            maze: 布线矩阵列表（包含障碍物信息）
            wire_path: 连线路径矩阵列表
            startpointz: 起点层
            startpointx: 起点X坐标
            startpointy: 起点Y坐标
            endpointz: 终点层
            endpointx: 终点X坐标
            endpointy: 终点Y坐标
            other_nodes: 其他目标节点（用于多目标优化）
            cost: 每层成本系数
        
        Returns:
            Optional[Tuple]: (layer_lengths, layer_lengths_all, via_num, maze, wire_path)
                            如果找不到路径则返回None
        """
        if cost is None:
            cost = [1.0] * self.layer
        
        if other_nodes is None:
            other_nodes = []
        
        # 初始化起点和终点节点
        start = (startpointz, startpointx, startpointy)
        end = (endpointz, endpointx, endpointy)
        startnode = Node(parent=None, position=start)
        endnode = Node(parent=None, position=end)
        
        # 使用优先队列保存待探索的节点
        pq = [(0, startnode)]
        heapq.heapify(pq)
        
        # 已访问节点字典
        visit = {}
        
        # 开始搜索
        while pq:
            cost_val, current = heapq.heappop(pq)
            
            # 到达终点，回溯路径
            if current == endnode:
                path = []
                while current.parent:
                    path.append(current.position)
                    current = current.parent
                path.append(start)
                path.reverse()
                
                # 计算路径长度
                layer_lengths, layer_lengths_all, via_num = calculate_path_length(
                    path, self.track_info, self.metal_list, maze
                )
                
                # 标记路径
                for node in path:
                    maze[node[0]][node[2]][node[1]] = 2
                    wire_path[node[0]][node[2]][node[1]] = 1
                
                return layer_lengths, layer_lengths_all, via_num, maze, wire_path
            
            z, x, y = current.position
            locx = x * self.track_info[self.metal_list[z]].stepx + \
                   self.track_info[self.metal_list[z]].biasx
            locy = y * self.track_info[self.metal_list[z]].stepy + \
                   self.track_info[self.metal_list[z]].biasy
            
            # 获取相邻节点（同层四个方向）
            near_node = [
                (z, x, y + 1), (z, x, y - 1),
                (z, x - 1, y), (z, x + 1, y)
            ]
            
            # 添加层间邻居节点
            if z == 0:
                # 最底层，只能向上
                newx, newy = get_neighbor(
                    (locx, locy),
                    self.track_info[self.metal_list[1]].biasx,
                    self.track_info[self.metal_list[1]].biasy,
                    self.track_info[self.metal_list[1]].stepx,
                    self.track_info[self.metal_list[1]].stepy
                )
                near_node.append((1, newx, newy))
            elif z == self.layer - 1:
                # 最顶层，只能向下
                newx, newy = get_neighbor(
                    (locx, locy),
                    self.track_info[self.metal_list[self.layer - 2]].biasx,
                    self.track_info[self.metal_list[self.layer - 2]].biasy,
                    self.track_info[self.metal_list[self.layer - 2]].stepx,
                    self.track_info[self.metal_list[self.layer - 2]].stepy
                )
                near_node.append((self.layer - 2, newx, newy))
            else:
                # 中间层，可以上下
                for metal in (z - 1, z + 1):
                    newx, newy = get_neighbor(
                        (locx, locy),
                        self.track_info[self.metal_list[metal]].biasx,
                        self.track_info[self.metal_list[metal]].biasy,
                        self.track_info[self.metal_list[metal]].stepx,
                        self.track_info[self.metal_list[metal]].stepy
                    )
                    near_node.append((metal, newx, newy))
            
            # 遍历相邻节点
            for j in range(len(near_node)):
                nz, nx, ny = near_node[j]
                
                # 检查边界
                if not (0 <= nz < self.layer and 
                       0 <= nx < maze[nz].shape[1] and 
                       0 <= ny < maze[nz].shape[0]):
                    continue
                
                # 计算移动代价
                direction = self.track_info[self.metal_list[nz]].direction
                
                if int(maze[nz][ny][nx]) == 0:
                    cc = cost[nz] * 5  # 空位
                elif int(maze[nz][ny][nx]) == 1:
                    cc = cost[nz] * 10000  # 已占用（高代价）
                else:
                    cc = cost[nz]  # 已布线
                
                # 根据方向调整代价
                if nx != x and nz == z:
                    if direction != "X":
                        new_cost = current.g + cc * 2  # 非首选方向
                    else:
                        new_cost = current.g + cc
                elif ny != y and nz == z:
                    if direction != "Y":
                        new_cost = current.g + cc * 2
                    else:
                        new_cost = current.g + cc
                else:
                    new_cost = current.g + 20  # 层间移动
                
                # 更新代价并加入队列
                if (nz, nx, ny) not in visit or new_cost < visit[(nz, nx, ny)].g:
                    tmp_node = Node(current, (nz, nx, ny))
                    tmp_node.g = new_cost
                    visit[(nz, nx, ny)] = tmp_node
                    priority = new_cost + heuristic(tmp_node, endnode, other_nodes)
                    heapq.heappush(pq, (priority, tmp_node))
        
        # 无法找到路径
        return None


def find_path_astar(maze: List[np.ndarray], wire_path: List[np.ndarray],
                    startpointz: int, startpointx: float, startpointy: float,
                    endpointz: int, endpointx: float, endpointy: float,
                    parser, other_nodes: List = None, c: List[float] = None) -> Optional[Tuple]:
    """
    A*寻路算法的便捷函数
    
    Args:
        maze: 布线矩阵列表
        wire_path: 连线路径矩阵列表
        startpointz: 起点层
        startpointx: 起点X坐标
        startpointy: 起点Y坐标
        endpointz: 终点层
        endpointx: 终点X坐标
        endpointy: 终点Y坐标
        parser: 解析器对象（包含metal_list和track属性）
        other_nodes: 其他目标节点
        c: 成本系数列表
    
    Returns:
        Optional[Tuple]: 路径计算结果
    """
    pathfinder = AStarPathfinder(parser.metal_list, parser.track)
    return pathfinder.find_path(
        maze, wire_path,
        startpointz, startpointx, startpointy,
        endpointz, endpointx, endpointy,
        other_nodes, c
    )
