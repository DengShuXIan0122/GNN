import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np
import sys
import os

# 添加utils路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from distance import get_distance_and_direction_cached, convert_to_tensor_indices, get_dde_cache_stats


class DDE(nn.Module):
    """
    Directional Distance Encoding (DDE)
    方向-距离编码层，为GNN消息传递注入几何偏置
    实现GNN-RAG³的第3层增强：DDE-Enhanced GNN Propagation
    """
    
    def __init__(self, 
                 hop_dim: int = 16,
                 dir_dim: int = 8,
                 hidden_dim: int = 64,
                 max_distance: int = 10,
                 dropout: float = 0.1):
        """
        Args:
            hop_dim: 距离嵌入维度
            dir_dim: 方向嵌入维度  
            hidden_dim: 隐层维度
            max_distance: 最大距离（超过则截断）
            dropout: dropout率
        """
        super(DDE, self).__init__()
        
        self.hop_dim = hop_dim
        self.dir_dim = dir_dim
        self.hidden_dim = hidden_dim
        self.max_distance = max_distance
        
        # 距离嵌入：E_h(d)
        self.hop_embedding = nn.Embedding(max_distance + 1, hop_dim)
        
        # 方向嵌入：E_d(dir) - in/out/self
        # 0: self-loop, 1: incoming, 2: outgoing
        self.dir_embedding = nn.Embedding(3, dir_dim)
        
        # 几何门控网络
        self.geo_gate_net = nn.Sequential(
            nn.Linear(hop_dim + dir_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.hop_embedding.weight)
        nn.init.xavier_uniform_(self.dir_embedding.weight)
        
        for module in self.geo_gate_net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, 
                dist_u: torch.Tensor, 
                dist_v: torch.Tensor, 
                direction: torch.Tensor) -> torch.Tensor:
        """
        前向传播：计算几何门控
        
        Args:
            dist_u: 源节点到种子的距离 [batch_size] 或 [num_edges]
            dist_v: 目标节点到种子的距离 [batch_size] 或 [num_edges]  
            direction: 边方向 [batch_size] 或 [num_edges]
                      0: self-loop, 1: incoming, 2: outgoing
                      
        Returns:
            geo_gate: 几何门控值 [batch_size, 1] 或 [num_edges, 1]
        """
        # 计算最小距离：h = min(dist_u, dist_v)
        min_dist = torch.min(dist_u, dist_v)
        
        # 截断距离
        min_dist = torch.clamp(min_dist, 0, self.max_distance)
        
        # 距离嵌入
        hop_emb = self.hop_embedding(min_dist)  # [batch_size, hop_dim]
        
        # 方向嵌入
        dir_emb = self.dir_embedding(direction)  # [batch_size, dir_dim]
        
        # 拼接特征
        geo_feat = torch.cat([hop_emb, dir_emb], dim=-1)  # [batch_size, hop_dim + dir_dim]
        
        # 计算几何门控
        geo_gate = self.geo_gate_net(geo_feat)  # [batch_size, 1]
        
        return geo_gate
    
    def compute_edge_features(self, 
                             edge_index: torch.Tensor,
                             distances: Dict[int, int],
                             num_edges: int) -> torch.Tensor:
        """
        为所有边计算DDE特征
        
        Args:
            edge_index: 边索引 [2, num_edges]
            distances: 节点到种子的距离字典 {node_id: distance}
            num_edges: 边数量
            
        Returns:
            edge_features: 边的DDE特征 [num_edges, 1]
        """
        device = edge_index.device
        
        # 提取源节点和目标节点
        src_nodes = edge_index[0]  # [num_edges]
        dst_nodes = edge_index[1]  # [num_edges]
        
        # 向量化获取距离
        # 将距离字典转换为张量以支持向量化操作
        max_node_id = max(max(distances.keys()) if distances else 0, 
                         edge_index.max().item()) + 1
        distance_tensor = torch.full((max_node_id,), self.max_distance, 
                                   device=device, dtype=torch.long)
        
        # 批量设置已知距离
        if distances:
            for node_id, dist in distances.items():
                if node_id < max_node_id:
                    distance_tensor[node_id] = min(dist, self.max_distance)
        
        # 向量化获取源节点和目标节点的距离
        src_distances = distance_tensor[src_nodes]  # [num_edges]
        dst_distances = distance_tensor[dst_nodes]  # [num_edges]
        
        # 计算方向：简化版本，可以根据需要扩展
        # 这里假设所有边都是outgoing (2)
        directions = torch.full((num_edges,), 2, dtype=torch.long, device=device)
        
        # 计算DDE特征
        edge_features = self.forward(src_distances, dst_distances, directions)
        
        return edge_features
    
    def compute_edge_features_cached(self,
                                   graph,
                                   edge_index: torch.Tensor,
                                   seeds: List[int],
                                   cand_nodes: List[int],
                                   node_mapping: Dict[int, int]) -> torch.Tensor:
        """
        使用缓存优化的边特征计算
        
        Args:
            graph: 图结构
            edge_index: 边索引 [2, num_edges]
            seeds: 种子节点列表
            cand_nodes: 候选节点列表
            node_mapping: 原节点ID到子图节点ID的映射
            
        Returns:
            edge_features: 边的DDE特征 [num_edges, 1]
        """
        device = edge_index.device
        
        # 使用缓存获取距离和方向索引
        hop_distances, direction_indices = get_distance_and_direction_cached(
            graph, seeds, cand_nodes, max_hops=self.max_distance
        )
        
        # 转换为张量索引
        hop_idx_tensor, dir_idx_tensor = convert_to_tensor_indices(
            hop_distances, direction_indices, edge_index, node_mapping, device
        )
        
        # 提取源节点和目标节点的距离
        src_nodes = edge_index[0]  # [num_edges]
        dst_nodes = edge_index[1]  # [num_edges]
        
        src_distances = hop_idx_tensor[src_nodes]  # [num_edges]
        dst_distances = hop_idx_tensor[dst_nodes]  # [num_edges]
        
        # 计算DDE特征
        edge_features = self.forward(src_distances, dst_distances, dir_idx_tensor)
        
        return edge_features


class DDEMessagePassing(nn.Module):
    """
    集成DDE的消息传递层
    在原有的关系门控基础上增加几何门控
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 hop_dim: int = 16,
                 dir_dim: int = 8,
                 dropout: float = 0.1):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐层维度
            hop_dim: DDE距离嵌入维度
            dir_dim: DDE方向嵌入维度
            dropout: dropout率
        """
        super(DDEMessagePassing, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # DDE编码器
        self.dde = DDE(hop_dim, dir_dim, hidden_dim, dropout=dropout)
        
        # 消息变换
        self.message_net = nn.Linear(input_dim, hidden_dim)
        
        # 门控融合
        self.gate_fusion = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, 
                h_u: torch.Tensor,
                h_v: torch.Tensor, 
                rel_gate: torch.Tensor,
                dist_u: torch.Tensor,
                dist_v: torch.Tensor,
                direction: torch.Tensor) -> torch.Tensor:
        """
        DDE增强的消息传递
        
        Args:
            h_u: 源节点特征 [num_edges, input_dim]
            h_v: 目标节点特征 [num_edges, input_dim]
            rel_gate: 关系门控 [num_edges, 1]
            dist_u: 源节点距离 [num_edges]
            dist_v: 目标节点距离 [num_edges]
            direction: 边方向 [num_edges]
            
        Returns:
            messages: 消息 [num_edges, hidden_dim]
        """
        # 计算几何门控
        geo_gate = self.dde(dist_u, dist_v, direction)  # [num_edges, 1]
        
        # 消息变换
        messages = self.message_net(h_u)  # [num_edges, hidden_dim]
        
        # 双重门控：关系门控 × 几何门控
        combined_gate = rel_gate * geo_gate  # [num_edges, 1]
        
        # 应用门控
        gated_messages = messages * combined_gate  # [num_edges, hidden_dim]
        
        return gated_messages


def compute_distances_bfs(edge_index: torch.Tensor, 
                         num_nodes: int,
                         seeds: List[int]) -> Dict[int, int]:
    """
    使用BFS计算从种子节点到所有节点的最短距离
    
    Args:
        edge_index: 边索引 [2, num_edges]
        num_nodes: 节点总数
        seeds: 种子节点列表
        
    Returns:
        distances: 节点ID -> 最短距离的字典
    """
    from collections import deque
    
    # 向量化构建邻接表
    adj = {i: [] for i in range(num_nodes)}
    
    # 批量获取边信息
    edge_index_cpu = edge_index.cpu().numpy()
    for i in range(edge_index.size(1)):
        u, v = int(edge_index_cpu[0, i]), int(edge_index_cpu[1, i])
        adj[u].append(v)
        adj[v].append(u)  # 无向图
    
    # 多源BFS
    distances = {}
    queue = deque()
    visited = set()
    
    # 初始化种子节点
    for seed in seeds:
        if seed < num_nodes:
            distances[seed] = 0
            queue.append((seed, 0))
            visited.add(seed)
    
    # BFS遍历
    while queue:
        node, dist = queue.popleft()
        
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))
    
    return distances