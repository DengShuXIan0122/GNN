import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os

# 添加根目录到路径以导入bmssp_sssp模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from bmssp_sssp import DirectedGraph, bmssp_sssp_driver


class BMSSPSolver:
    """BMSSP最短路径算法的PyTorch包装器"""
    
    def __init__(self, k_param: Optional[int] = None, t_param: Optional[int] = None):
        """
        Args:
            k_param: BMSSP算法的k参数 (默认根据图大小自动计算)
            t_param: BMSSP算法的t参数 (默认根据图大小自动计算)
        """
        self.k_param = k_param
        self.t_param = t_param
    
    def compute_distances(self, edge_index: torch.Tensor, edge_weights: Optional[torch.Tensor],
                         num_nodes: int, source_nodes: List[int]) -> torch.Tensor:
        """
        使用BMSSP算法计算从源节点到所有节点的最短距离
        
        Args:
            edge_index: 边索引 [2, num_edges]
            edge_weights: 边权重 [num_edges] (可选，默认为1.0)
            num_nodes: 节点总数
            source_nodes: 源节点列表
            
        Returns:
            distances: 距离矩阵 [len(source_nodes), num_nodes]
        """
        device = edge_index.device
        
        # 转换为numpy进行计算
        edge_index_np = edge_index.cpu().numpy()
        if edge_weights is not None:
            edge_weights_np = edge_weights.cpu().numpy()
        else:
            edge_weights_np = np.ones(edge_index.size(1))
        
        # 构建DirectedGraph
        graph = DirectedGraph(num_nodes)
        for i in range(edge_index.size(1)):
            u, v = edge_index_np[0, i], edge_index_np[1, i]
            w = edge_weights_np[i]
            graph.add_edge(u, v, w)
        
        # 为每个源节点计算距离
        all_distances = []
        for source in source_nodes:
            if 0 <= source < num_nodes:
                distances_dict = bmssp_sssp_driver(graph, source, self.k_param, self.t_param)
                
                # 转换为张量
                distances = torch.full((num_nodes,), float('inf'), device=device)
                for node, dist in distances_dict.items():
                    if 0 <= node < num_nodes:
                        distances[node] = dist
                
                all_distances.append(distances)
            else:
                # 无效源节点，返回无穷大距离
                distances = torch.full((num_nodes,), float('inf'), device=device)
                all_distances.append(distances)
        
        return torch.stack(all_distances, dim=0)
    
    def compute_single_source_distances(self, edge_index: torch.Tensor, 
                                      edge_weights: Optional[torch.Tensor],
                                      num_nodes: int, source: int) -> torch.Tensor:
        """
        计算单源最短路径
        
        Args:
            edge_index: 边索引 [2, num_edges]
            edge_weights: 边权重 [num_edges] (可选)
            num_nodes: 节点总数
            source: 源节点
            
        Returns:
            distances: 距离向量 [num_nodes]
        """
        distances = self.compute_distances(edge_index, edge_weights, num_nodes, [source])
        return distances[0]
    
    def compute_candidate_distances(self, edge_index: torch.Tensor, 
                                  edge_weights: Optional[torch.Tensor],
                                  num_nodes: int, source_nodes: List[int],
                                  candidate_nodes: List[int]) -> torch.Tensor:
        """
        计算候选节点的最短距离
        
        Args:
            edge_index: 边索引 [2, num_edges]
            edge_weights: 边权重 [num_edges] (可选)
            num_nodes: 节点总数
            source_nodes: 源节点列表
            candidate_nodes: 候选节点列表
            
        Returns:
            distances: 候选节点距离 [len(candidate_nodes)]
        """
        # 计算所有源节点的距离
        all_distances = self.compute_distances(edge_index, edge_weights, num_nodes, source_nodes)
        
        # 对每个候选节点，取到所有源节点的最小距离
        candidate_distances = []
        for candidate in candidate_nodes:
            if 0 <= candidate < num_nodes:
                min_dist = torch.min(all_distances[:, candidate])
                candidate_distances.append(min_dist)
            else:
                candidate_distances.append(torch.tensor(float('inf'), device=all_distances.device))
        
        return torch.stack(candidate_distances)


def convert_torch_graph_to_directed_graph(edge_index: torch.Tensor, 
                                        edge_weights: Optional[torch.Tensor],
                                        num_nodes: int) -> DirectedGraph:
    """
    将PyTorch图转换为DirectedGraph格式
    
    Args:
        edge_index: 边索引 [2, num_edges]
        edge_weights: 边权重 [num_edges] (可选)
        num_nodes: 节点总数
        
    Returns:
        graph: DirectedGraph对象
    """
    graph = DirectedGraph(num_nodes)
    
    edge_index_np = edge_index.cpu().numpy()
    if edge_weights is not None:
        edge_weights_np = edge_weights.cpu().numpy()
    else:
        edge_weights_np = np.ones(edge_index.size(1))
    
    for i in range(edge_index.size(1)):
        u, v = edge_index_np[0, i], edge_index_np[1, i]
        w = edge_weights_np[i]
        graph.add_edge(u, v, w)
    
    return graph


def batch_bmssp_distances(edge_index: torch.Tensor, edge_weights: Optional[torch.Tensor],
                         num_nodes: int, source_nodes: List[int],
                         k_param: Optional[int] = None, t_param: Optional[int] = None) -> torch.Tensor:
    """
    批量计算BMSSP距离的便捷函数
    
    Args:
        edge_index: 边索引 [2, num_edges]
        edge_weights: 边权重 [num_edges] (可选)
        num_nodes: 节点总数
        source_nodes: 源节点列表
        k_param: BMSSP的k参数
        t_param: BMSSP的t参数
        
    Returns:
        distances: 距离矩阵 [len(source_nodes), num_nodes]
    """
    solver = BMSSPSolver(k_param, t_param)
    return solver.compute_distances(edge_index, edge_weights, num_nodes, source_nodes)