# -*- coding: utf-8 -*-
"""
文件操作工具模块
提供文件和目录操作的辅助函数
"""

import os
import pandas as pd
from typing import Optional


def ensure_dir(path: str) -> str:
    """
    确保目录存在，不存在则创建
    
    Args:
        path: 目录路径
        
    Returns:
        目录路径
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def write_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
    """
    写入CSV文件，支持追加模式
    
    Args:
        df: DataFrame数据
        path: 文件路径
        index: 是否写入索引
    """
    if not os.path.isfile(path):
        df.to_csv(path, index=index)
    else:
        df_old = pd.read_csv(path)
        df_merge = pd.concat([df_old, df], ignore_index=True)
        df_merge.to_csv(path, index=index)


def delete_csv(path: str) -> bool:
    """
    删除CSV文件
    
    Args:
        path: 文件路径
        
    Returns:
        是否删除成功
    """
    if os.path.isfile(path):
        os.remove(path)
        return True
    return False


def read_json(path: str) -> dict:
    """
    读取JSON文件
    
    Args:
        path: 文件路径
        
    Returns:
        JSON数据字典
    """
    import json
    with open(path, 'r') as f:
        return json.load(f)


def write_json(data: dict, path: str) -> None:
    """
    写入JSON文件
    
    Args:
        data: 数据字典
        path: 文件路径
    """
    import json
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_lines(path: str) -> list:
    """
    读取文件所有行
    
    Args:
        path: 文件路径
        
    Returns:
        行列表
    """
    with open(path, 'r') as f:
        return f.readlines()


def write_lines(lines: list, path: str) -> None:
    """
    写入多行到文件
    
    Args:
        lines: 行列表
        path: 文件路径
    """
    with open(path, 'w') as f:
        f.writelines(lines)


def file_exists(path: str) -> bool:
    """
    检查文件是否存在
    
    Args:
        path: 文件路径
        
    Returns:
        是否存在
    """
    return os.path.isfile(path)


def dir_exists(path: str) -> bool:
    """
    检查目录是否存在
    
    Args:
        path: 目录路径
        
    Returns:
        是否存在
    """
    return os.path.isdir(path)


def get_file_size(path: str) -> int:
    """
    获取文件大小（字节）
    
    Args:
        path: 文件路径
        
    Returns:
        文件大小
    """
    if file_exists(path):
        return os.path.getsize(path)
    return 0
