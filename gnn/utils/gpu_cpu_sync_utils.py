"""
GPU-CPU同步优化工具模块
提供向量化操作以减少GPU-CPU同步开销
"""

import torch
from typing import Dict, List, Tuple, Optional, Union
import numpy as np


def create_node_mask(node_list: List[int], 
                    max_node_id: int, 
                    device: torch.device) -> torch.Tensor:
    """
    创建节点掩码张量，用于向量化节点筛选
    
    Args:
        node_list: 节点ID列表
        max_node_id: 最大节点ID
        device: 设备
        
    Returns:
        node_mask: 布尔掩码张量 [max_node_id+1]
    """
    node_mask = torch.zeros(max_node_id + 1, dtype=torch.bool, device=device)
    if node_list:
        node_indices = torch.tensor(node_list, device=device, dtype=torch.long)
        node_mask[node_indices] = True
    return node_mask


def vectorized_edge_filter(edge_index: torch.Tensor,
                          candidate_nodes: List[int]) -> torch.Tensor:
    """
    向量化边筛选，只保留两端都在候选节点中的边
    
    Args:
        edge_index: 边索引 [2, num_edges]
        candidate_nodes: 候选节点列表
        
    Returns:
        mask: 边掩码 [num_edges]
    """
    device = edge_index.device
    max_node_id = max(max(candidate_nodes), edge_index.max().item())
    
    # 创建候选节点掩码
    cand_mask = create_node_mask(candidate_nodes, max_node_id, device)
    
    # 向量化检查边的两端
    src_in_cand = cand_mask[edge_index[0]]  # [num_edges]
    dst_in_cand = cand_mask[edge_index[1]]  # [num_edges]
    
    return src_in_cand & dst_in_cand


def dict_to_tensor(data_dict: Dict[int, Union[int, float]], 
                  max_key: int,
                  default_value: Union[int, float],
                  device: torch.device,
                  dtype: torch.dtype = torch.float) -> torch.Tensor:
    """
    将字典转换为张量，支持向量化索引
    
    Args:
        data_dict: 数据字典 {node_id: value}
        max_key: 最大键值
        default_value: 默认值
        device: 设备
        dtype: 数据类型
        
    Returns:
        tensor: 张量 [max_key+1]
    """
    tensor = torch.full((max_key + 1,), default_value, device=device, dtype=dtype)
    
    if data_dict:
        for key, value in data_dict.items():
            if key <= max_key:
                tensor[key] = value
    
    return tensor


def batch_cpu_transfer(tensors: List[torch.Tensor]) -> List[np.ndarray]:
    """
    批量将张量转移到CPU并转换为numpy数组
    
    Args:
        tensors: 张量列表
        
    Returns:
        arrays: numpy数组列表
    """
    return [tensor.cpu().numpy() for tensor in tensors]


def vectorized_node_mapping(edge_index: torch.Tensor,
                           node_mapping: Dict[int, int],
                           max_node_id: int) -> torch.Tensor:
    """
    向量化节点ID重映射
    
    Args:
        edge_index: 边索引 [2, num_edges]
        node_mapping: 节点映射字典 {old_id: new_id}
        max_node_id: 最大节点ID
        
    Returns:
        remapped_edge_index: 重映射后的边索引 [2, num_edges]
    """
    device = edge_index.device
    
    # 创建映射张量
    mapping_tensor = torch.full((max_node_id + 1,), -1, device=device, dtype=torch.long)
    for old_id, new_id in node_mapping.items():
        if old_id <= max_node_id:
            mapping_tensor[old_id] = new_id
    
    # 批量重映射
    remapped_edge_index = edge_index.clone()
    remapped_edge_index[0] = mapping_tensor[edge_index[0]]
    remapped_edge_index[1] = mapping_tensor[edge_index[1]]
    
    return remapped_edge_index


def compute_edge_costs_vectorized(edge_index: torch.Tensor,
                                 distances: Dict[int, Union[int, float]],
                                 relation_costs: Optional[torch.Tensor] = None,
                                 beta: float = 0.2,
                                 gamma: float = 0.1,
                                 default_distance: float = 10.0) -> torch.Tensor:
    """
    向量化计算边成本
    
    Args:
        edge_index: 边索引 [2, num_edges]
        distances: 距离字典 {node_id: distance}
        relation_costs: 关系成本 [num_edges] (可选)
        beta: 距离权重
        gamma: 关系权重
        default_distance: 默认距离
        
    Returns:
        edge_costs: 边成本 [num_edges]
    """
    device = edge_index.device
    num_edges = edge_index.size(1)
    
    # 将距离字典转换为张量
    max_node_id = max(max(distances.keys()) if distances else 0, 
                     edge_index.max().item())
    distance_tensor = dict_to_tensor(distances, max_node_id, default_distance, 
                                   device, torch.float)
    
    # 向量化计算距离成本
    src_distances = distance_tensor[edge_index[0]]  # [num_edges]
    dst_distances = distance_tensor[edge_index[1]]  # [num_edges]
    dist_costs = beta * (src_distances + dst_distances) / 2.0
    
    # 添加关系成本
    if relation_costs is not None:
        rel_costs = gamma * relation_costs
    else:
        rel_costs = torch.zeros(num_edges, device=device)
    
    return dist_costs + rel_costs


def create_target_mask_vectorized(selected_indices: List[int],
                                 total_size: int,
                                 device: torch.device) -> torch.Tensor:
    """
    向量化创建目标掩码
    
    Args:
        selected_indices: 选中的索引列表
        total_size: 总大小
        device: 设备
        
    Returns:
        mask: 目标掩码 [total_size]
    """
    mask = torch.zeros(total_size, device=device, dtype=torch.float)
    
    if selected_indices:
        valid_indices = [idx for idx in selected_indices if 0 <= idx < total_size]
        if valid_indices:
            indices_tensor = torch.tensor(valid_indices, device=device, dtype=torch.long)
            mask[indices_tensor] = 1.0
    
    return mask


class GPUCPUSyncProfiler:
    """GPU-CPU同步性能分析器"""
    
    def __init__(self):
        self.sync_counts = {}
        self.sync_times = {}
        self.operation_times = {}
    
    def record_sync(self, operation_name: str, sync_count: int = 1):
        """记录同步操作"""
        self.sync_counts[operation_name] = self.sync_counts.get(operation_name, 0) + sync_count
    
    def get_stats(self) -> Dict[str, int]:
        """获取同步统计信息"""
        return {
            'sync_counts': self.sync_counts.copy(),
            'operation_times': self.operation_times.copy()
        }
    
    def reset(self):
        """重置统计信息"""
        self.sync_counts.clear()
        self.sync_times.clear()
        self.operation_times.clear()
    
    def profile_context(self, operation_name: str):
        """性能分析上下文管理器"""
        import time
        from contextlib import contextmanager
        
        @contextmanager
        def context():
            start_time = time.time()
            try:
                yield
            finally:
                elapsed_time = time.time() - start_time
                self.operation_times[operation_name] = self.operation_times.get(operation_name, 0) + elapsed_time
        
        return context()


# 全局同步分析器实例
gpu_cpu_sync_profiler = GPUCPUSyncProfiler()


def get_sync_stats() -> Dict[str, int]:
    """获取GPU-CPU同步统计信息"""
    return gpu_cpu_sync_profiler.get_stats()


def reset_sync_stats():
    """重置GPU-CPU同步统计信息"""
    gpu_cpu_sync_profiler.reset()