import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from dataclasses import dataclass
from collections import OrderedDict
import xxhash
import time
from typing_extensions import TypedDict

from .ppr import PersonalizedPageRank, DirectedDistanceEncoding
from .bmssp import BMSSPSolver


class PCSTPrepared(TypedDict):
    """PCST预计算数据结构"""
    edge_index: torch.Tensor      # [2, E] on device
    rel_ids: torch.LongTensor     # [E] 关系ID
    edge_cost_base: torch.Tensor  # [E] 基础边成本 (β * dist(e))
    node_mapping: Dict[int, int]  # 原节点ID -> 局部节点ID
    rel_frequency: Dict[Tuple[int, ...], int]  # 关系频率统计


@dataclass
class PCSTEdge:
    """PCST边数据结构"""
    src: int
    dst: int
    cost: float
    original_idx: int


@dataclass
class PCSTNode:
    """PCST节点数据结构"""
    node_id: int
    prize: float


class PCSTSolver:
    """
    Prize-Collecting Steiner Tree求解器
    This module implements PCST for combining PPR, DDE, and BMSSP algorithms
    """
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.2, gamma: float = 0.1):
        """
        Args:
            alpha: Weight for PPR scores
            beta: Weight for distance costs
            gamma: Weight for relation costs
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def solve_pcst(self, nodes: List[PCSTNode], edges: List[PCSTEdge], 
                   root: Optional[int] = None) -> Tuple[List[int], List[int]]:
        """Solve PCST problem using approximation algorithm"""
        if not nodes or not edges:
            return [], []
        
        # 构建邻接表
        adj = {}
        edge_map = {}
        
        for node in nodes:
            adj[node.node_id] = []
        
        for i, edge in enumerate(edges):
            if edge.src not in adj:
                adj[edge.src] = []
            if edge.dst not in adj:
                adj[edge.dst] = []
            
            adj[edge.src].append(edge.dst)
            adj[edge.dst].append(edge.src)  # 无向图
            edge_map[(edge.src, edge.dst)] = edge
            edge_map[(edge.dst, edge.src)] = edge
        
        # Run approximation algorithm
        selected_nodes, selected_edges = self._gw_approximation(
            adj, edge_map, {n.node_id: n.prize for n in nodes}, root
        )
        
        return selected_nodes, selected_edges
    
    def _gw_approximation(self, adj: Dict[int, List[int]], edge_map: Dict[Tuple[int, int], PCSTEdge],
                         node_prizes: Dict[int, float], root: Optional[int]) -> Tuple[List[int], List[int]]:
        """Goemans-Williamson approximation algorithm for PCST"""
        # 简化的贪心近似算法
        selected_nodes = set()
        selected_edges = []
        
        if not node_prizes:
            return [], []
        
        # 选择起始节点（最高奖励或指定根节点）
        if root is not None and root in node_prizes:
            current = root
        else:
            current = max(node_prizes.keys(), key=lambda x: node_prizes[x])
        
        selected_nodes.add(current)
        remaining_nodes = set(node_prizes.keys()) - selected_nodes
        
        # 贪心扩展（限制最大节点数）
        max_nodes = min(10, len(node_prizes))  # 限制最大节点数
        
        while remaining_nodes and len(selected_nodes) < max_nodes:
            best_ratio = -float('inf')
            best_node = None
            best_edge = None
            
            # 寻找最佳的prize/cost比率
            for node in list(selected_nodes):
                for neighbor in adj.get(node, []):
                    if neighbor in remaining_nodes:
                        edge_key = (node, neighbor)
                        if edge_key in edge_map:
                            edge = edge_map[edge_key]
                            prize = node_prizes[neighbor]
                            cost = edge.cost
                            
                            if cost > 0:
                                ratio = prize / cost
                            else:
                                ratio = float('inf') if prize > 0 else 0
                            
                            if ratio > best_ratio:
                                best_ratio = ratio
                                best_node = neighbor
                                best_edge = edge.original_idx
            
            # 如果找到有利的扩展，添加节点和边（改进的动态阈值 + 同簇补位策略）
            # 使用统计式阈值策略
            avg_prize = sum(node_prizes.values()) / len(node_prizes)
            std_prize = (sum((v - avg_prize)**2 for v in node_prizes.values()) / len(node_prizes))**0.5
            threshold = max(0.0, avg_prize + 0.5 * std_prize)  # 更合理的统计式阈值
            
            if best_node is not None and best_ratio > threshold:
                selected_nodes.add(best_node)
                selected_edges.append(best_edge)
                remaining_nodes.remove(best_node)
                
                # 同簇补位策略：如果选中了Top-1节点，检查同簇相似节点
                self._add_cluster_nodes(best_node, selected_nodes, remaining_nodes, 
                                      node_prizes, adj, edge_map, selected_edges, threshold)
            else:
                break
        
        return list(selected_nodes), selected_edges
    
    def _add_cluster_nodes(self, anchor_node: int, selected_nodes: Set[int], 
                          remaining_nodes: Set[int], node_prizes: Dict[int, float],
                          adj: Dict[int, List[int]], edge_map: Dict[Tuple[int, int], PCSTEdge],
                          selected_edges: List[int], base_threshold: float):
        """
        同簇补位策略：为高质量节点添加相似的邻居节点
        
        Args:
            anchor_node: 已选中的锚点节点
            selected_nodes: 已选中的节点集合
            remaining_nodes: 剩余候选节点集合
            node_prizes: 节点奖励字典
            adj: 邻接表
            edge_map: 边映射
            selected_edges: 已选中的边列表
            base_threshold: 基础阈值
        """
        if anchor_node not in node_prizes:
            return
            
        anchor_prize = node_prizes[anchor_node]
        
        # 只对高质量节点进行同簇补位（Top-1或高于平均值的节点）
        avg_prize = sum(node_prizes.values()) / len(node_prizes)
        if anchor_prize < avg_prize * 1.2:  # 只对明显高于平均的节点进行补位
            return
        
        # 放宽阈值进行同簇搜索
        relaxed_threshold = base_threshold * 0.6  # 降低60%的阈值
        max_cluster_size = 3  # 限制每个簇的最大补位数量
        added_count = 0
        
        # 搜索锚点的邻居节点
        for neighbor in adj.get(anchor_node, []):
            if (neighbor in remaining_nodes and 
                added_count < max_cluster_size and
                neighbor in node_prizes):
                
                neighbor_prize = node_prizes[neighbor]
                edge_key = (anchor_node, neighbor)
                
                if edge_key in edge_map:
                    edge = edge_map[edge_key]
                    cost = edge.cost
                    
                    # 计算相似性比率（使用放宽的阈值）
                    if cost > 0:
                        ratio = neighbor_prize / cost
                    else:
                        ratio = float('inf') if neighbor_prize > 0 else 0
                    
                    # 同簇条件：1) 超过放宽阈值 2) 奖励相近（相似性检查）
                    similarity_ratio = min(neighbor_prize, anchor_prize) / max(neighbor_prize, anchor_prize)
                    
                    if (ratio > relaxed_threshold and 
                        similarity_ratio > 0.7):  # 要求70%的相似性
                        
                        selected_nodes.add(neighbor)
                        selected_edges.append(edge.original_idx)
                        remaining_nodes.remove(neighbor)
                        added_count += 1


class PCSTCache:
    """
    PCST图结构预计算缓存器
    缓存与候选子图结构相关的预计算数据
    """
    
    def __init__(self, max_items: int = 256):
        self.cache = OrderedDict()
        self.max_items = max_items
        self.hit_count = 0
        self.miss_count = 0
    
    def get_cache_key(self, cand_nodes: List[int]) -> str:
        """生成缓存键"""
        return xxhash.xxh64(str(sorted(cand_nodes))).hexdigest()
    
    def get(self, key: str) -> Optional[PCSTPrepared]:
        """获取缓存的预计算数据"""
        if key in self.cache:
            # 移动到末尾（LRU）
            value = self.cache[key]
            self.cache.move_to_end(key)
            self.hit_count += 1
            return value
        else:
            self.miss_count += 1
            return None
    
    def put(self, key: str, value: PCSTPrepared):
        """添加到缓存"""
        self.cache[key] = value
        self.cache.move_to_end(key)
        
        # LRU淘汰
        if len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
    
    def get_hit_rate(self) -> float:
        """获取缓存命中率"""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0


# 全局PCST缓存实例
pcst_cache = PCSTCache()


def prepare_pcst_tensors(edge_index: torch.Tensor,
                        cand_nodes: List[int],
                        distance_features: torch.Tensor,
                        graph_data=None,
                        device: torch.device = None) -> PCSTPrepared:
    """
    预计算PCST所需的张量数据
    
    Args:
        edge_index: 边索引 [2, num_edges]
        cand_nodes: 候选节点列表
        distance_features: 距离特征 [num_nodes]
        graph_data: 图数据（包含边属性）
        device: 设备
        
    Returns:
        预计算的PCST数据
    """
    if device is None:
        device = edge_index.device
    
    # 生成缓存键
    cache_key = pcst_cache.get_cache_key(cand_nodes)
    
    # 尝试从缓存获取
    cached_result = pcst_cache.get(cache_key)
    if cached_result is not None:
        return cached_result
    
    # 缓存未命中，重新计算
    start_time = time.time()
    
    # 创建候选节点集合和映射
    cand_set = set(cand_nodes)
    node_mapping = {node_id: idx for idx, node_id in enumerate(cand_nodes)}
    
    # 筛选候选子图的边
    valid_edges = []
    rel_ids = []
    edge_costs_base = []
    
    for edge_idx in range(edge_index.size(1)):
        src = int(edge_index[0, edge_idx].item())
        dst = int(edge_index[1, edge_idx].item())
        
        if src in cand_set and dst in cand_set:
            valid_edges.append([node_mapping[src], node_mapping[dst]])
            
            # 计算基础边成本 (β * distance)
            src_idx = node_mapping[src]
            dst_idx = node_mapping[dst]
            avg_distance = (distance_features[src_idx] + distance_features[dst_idx]) / 2
            edge_costs_base.append(avg_distance.item())
            
            # 提取关系ID（简化版本）
            if graph_data is not None and hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None:
                if edge_idx < len(graph_data.edge_attr):
                    edge_feat = graph_data.edge_attr[edge_idx]
                    if edge_feat.dim() > 0 and edge_feat.numel() > 1:
                        rel_id = int(edge_feat[0].item())
                    else:
                        rel_id = int(edge_feat.item())
                else:
                    rel_id = 0
            else:
                rel_id = (src + dst) % 100  # 简化的关系ID
            rel_ids.append(rel_id)
    
    # 转换为张量
    if valid_edges:
        edge_index_local = torch.tensor(valid_edges, dtype=torch.long, device=device).t()
        rel_ids_tensor = torch.tensor(rel_ids, dtype=torch.long, device=device)
        edge_cost_base_tensor = torch.tensor(edge_costs_base, dtype=torch.float, device=device)
    else:
        edge_index_local = torch.empty((2, 0), dtype=torch.long, device=device)
        rel_ids_tensor = torch.empty(0, dtype=torch.long, device=device)
        edge_cost_base_tensor = torch.empty(0, dtype=torch.float, device=device)
    
    # 计算关系频率统计
    rel_frequency = {}
    for rel_id in rel_ids:
        rel_key = (rel_id,)
        rel_frequency[rel_key] = rel_frequency.get(rel_key, 0) + 1
    
    # 创建预计算数据
    prepared_data = PCSTPrepared(
        edge_index=edge_index_local,
        rel_ids=rel_ids_tensor,
        edge_cost_base=edge_cost_base_tensor,
        node_mapping=node_mapping,
        rel_frequency=rel_frequency
    )
    
    # 添加到缓存
    pcst_cache.put(cache_key, prepared_data)
    
    compute_time = time.time() - start_time
    
    return prepared_data


def get_pcst_cache_stats() -> Dict[str, float]:
    """获取PCST缓存统计信息"""
    return {
        "hit_rate": pcst_cache.get_hit_rate(),
        "cache_size": len(pcst_cache.cache),
        "max_size": pcst_cache.max_items,
        "hit_count": pcst_cache.hit_count,
        "miss_count": pcst_cache.miss_count
    }


def clear_pcst_cache():
    """清空PCST缓存"""
    pcst_cache.clear()


class IntegratedRetriever(nn.Module):
    """Integrated retrieval system combining PPR, DDE, BMSSP with PCST"""
    
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256, 
                 ppr_alpha: float = 0.15, dde_layers: int = 2, alpha: float = 0.7,
                 beta: float = 0.2, gamma: float = 0.1, penalty_range: Tuple[float, float] = (0.1, 5.0),
                 use_log_penalty: bool = False):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha  # PPR weight
        self.beta = beta    # Distance weight
        self.gamma = gamma  # Relation weight
        
        # 量化参数常量，确保relation_type生成的一致性
        self._rel_quant_scale = 100
        
        # 关系惩罚参数
        self.penalty_min, self.penalty_max = penalty_range
        self.use_log_penalty = use_log_penalty
        
        # 算法组件
        self.ppr = PersonalizedPageRank(alpha=ppr_alpha)
        self.dde = DirectedDistanceEncoding(max_distance=10, embedding_dim=hidden_dim)
        self.bmssp = BMSSPSolver()
        self.pcst = PCSTSolver(alpha=alpha, beta=beta, gamma=gamma)
        
        # 特征投影层
        self.ppr_proj = nn.Linear(1, hidden_dim)
        self.distance_proj = nn.Linear(1, hidden_dim)
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 最终预测层
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, graph_data, query_embedding: torch.Tensor, 
                query_nodes: List[int], candidate_nodes: List[int],
                edge_weights: Optional[torch.Tensor] = None,
                distance_info: Optional[Dict] = None) -> Tuple[torch.Tensor, List[int], List[int]]:
        """
        完整的多层检索流程
        
        Args:
            graph_data: 图数据 (包含edge_index, x, edge_attr等)
            query_embedding: 查询嵌入 [embedding_dim]
            query_nodes: 查询节点列表
            candidate_nodes: 候选节点列表
            edge_weights: 边权重 (可选)
            
        Returns:
            node_scores: 节点分数 [len(candidate_nodes)]
            selected_nodes: PCST选中的节点
            selected_edges: PCST选中的边
        """
        edge_index = graph_data.edge_index
        num_nodes = graph_data.num_nodes
        device = edge_index.device
        
        # Step 1: Compute PPR scores
        ppr_scores = self._compute_ppr_scores(edge_index, query_nodes, candidate_nodes,
                                            edge_weights, num_nodes, device)
        
        # Step 2: Use provided distance info or compute BMSSP distances
        if distance_info is not None and 'distances' in distance_info:
            # Use pre-computed distances from BMSSP
            distances_data = distance_info['distances']
            
            # Handle both dict and tensor formats
            if isinstance(distances_data, dict):
                # Convert dict to tensor in candidate_nodes order
                candidate_distances = []
                for node_idx in candidate_nodes:
                    if node_idx in distances_data:
                        dist_val = distances_data[node_idx]
                        # Ensure scalar tensor
                        if isinstance(dist_val, torch.Tensor):
                            if dist_val.dim() == 0:
                                candidate_distances.append(dist_val.to(device))
                            else:
                                candidate_distances.append(dist_val.squeeze().to(device))
                        else:
                            candidate_distances.append(torch.tensor(float(dist_val), device=device))
                    else:
                        candidate_distances.append(torch.tensor(float('inf'), device=device))
                candidate_distances = torch.stack(candidate_distances)
            else:
                # Already a tensor, ensure proper device and shape
                candidate_distances = distances_data.to(device)
                if candidate_distances.dim() > 1:
                    candidate_distances = candidate_distances.squeeze()
                
            distance_features = candidate_distances
        else:
            # Fallback: compute distances ourselves
            distance_features = self._compute_distance_features(edge_index, edge_weights,
                                                              num_nodes, query_nodes, candidate_nodes, device)
            candidate_distances = self.bmssp.compute_candidate_distances(
                edge_index, edge_weights, num_nodes, query_nodes, candidate_nodes
            )
        
        # Step 3: Compute DDE features using real distances
        dde_features = self._compute_dde_features(edge_index, candidate_distances, num_nodes, 
                                                query_nodes, candidate_nodes)
        
        # Step 4: Fuse features
        fused_features = self._fuse_features(ppr_scores, dde_features, distance_features,
                                           candidate_nodes, device)
        
        # Step 5: Compute node scores
        node_scores = self.predictor(fused_features).squeeze(-1)
        
        # Step 6: Apply PCST for subgraph selection
        selected_nodes, selected_edges = self._apply_pcst(ppr_scores, distance_features,
                                                        candidate_nodes, edge_index, node_scores, graph_data)
        
        return node_scores, selected_nodes, selected_edges
    
    def _compute_ppr_scores(self, edge_index: torch.Tensor, query_nodes: List[int],
                          candidate_nodes: List[int], edge_weights: Optional[torch.Tensor],
                          num_nodes: int, device: torch.device) -> torch.Tensor:
        """Compute PPR scores for candidate nodes"""
        # Compute PPR scores for all query nodes at once
        scores = self.ppr.compute_ppr(edge_index, num_nodes, query_nodes)
        
        # Extract scores for candidate nodes
        ppr_scores = torch.zeros(len(candidate_nodes), device=device, dtype=scores.dtype)
        for j, candidate in enumerate(candidate_nodes):
            if 0 <= candidate < num_nodes:
                ppr_scores[j] = scores[candidate]
        
        return ppr_scores
    
    def _compute_distance_features(self, edge_index: torch.Tensor, edge_weights: Optional[torch.Tensor],
                                 num_nodes: int, query_nodes: List[int], candidate_nodes: List[int],
                                 device: torch.device) -> torch.Tensor:
        """Compute distance features using BMSSP"""
        # Compute distances from query nodes to all nodes
        distances = self.bmssp.compute_candidate_distances(
            edge_index, edge_weights, num_nodes, query_nodes, candidate_nodes
        )
        
        return distances
    
    def _compute_dde_features(self, edge_index: torch.Tensor, distances: torch.Tensor,
                            num_nodes: int, query_nodes: List[int], candidate_nodes: List[int]) -> torch.Tensor:
        """Compute DDE features using real BMSSP distances"""
        # Create full distance vector for all nodes, initialized with inf
        full_distances = torch.full((num_nodes,), float('inf'), device=distances.device)
        
        # Map real BMSSP distances to the full distance vector
        # distances shape: [len(candidate_nodes)] - distances from query_nodes to candidate_nodes
        for i, candidate_idx in enumerate(candidate_nodes):
            if i < len(distances) and candidate_idx < num_nodes:
                full_distances[candidate_idx] = distances[i]
        
        # Set query nodes distance to 0
        for query_idx in query_nodes:
            if query_idx < num_nodes:
                full_distances[query_idx] = 0.0
        
        # Now DDE receives real distances instead of all inf
        edge_encodings = self.dde(edge_index, full_distances, num_nodes)
        
        return edge_encodings
    
    def _fuse_features(self, ppr_scores: torch.Tensor, dde_features: torch.Tensor,
                     distance_features: torch.Tensor, candidate_nodes: List[int],
                     device: torch.device) -> torch.Tensor:
        """Fuse PPR, DDE, and BMSSP features"""
        # Project PPR scores
        ppr_proj = self.ppr_proj(ppr_scores.unsqueeze(-1))  # [num_candidates, hidden_dim]
        
        # Project distance features
        dist_proj = self.distance_proj(distance_features.unsqueeze(-1))  # [num_candidates, hidden_dim]
        
        # For DDE features, we need to aggregate edge features to node features
        # This is a simplified approach - you might want to use more sophisticated aggregation
        if dde_features.size(0) > 0:
            # Average pooling of edge features (simplified)
            dde_dim = dde_features.size(-1)
            if dde_dim == self.hidden_dim:
                dde_proj = torch.mean(dde_features, dim=0, keepdim=True).expand(len(candidate_nodes), -1)
            else:
                # If dimensions don't match, create a projection layer or use zero features
                dde_proj = torch.zeros(len(candidate_nodes), self.ppr_proj.out_features, device=device, dtype=ppr_scores.dtype)
        else:
            dde_proj = torch.zeros(len(candidate_nodes), self.hidden_dim, device=device, dtype=ppr_scores.dtype)
        
        # Concatenate all features
        combined = torch.cat([ppr_proj, dde_proj, dist_proj], dim=-1)  # [num_candidates, 3*hidden_dim]
        
        # Fuse features
        fused = self.fusion_layer(combined)  # [num_candidates, hidden_dim]
        
        return fused
    
    def _apply_pcst(self, ppr_scores: torch.Tensor, distance_features: torch.Tensor,
                   candidate_nodes: List[int], edge_index: torch.Tensor,
                   node_scores: torch.Tensor, graph_data=None) -> Tuple[List[int], List[int]]:
        """Apply PCST algorithm for subgraph selection with improved cost function and caching"""
        device = edge_index.device
        
        # 使用PCST缓存机制预计算图结构
        prepared_data = prepare_pcst_tensors(
            edge_index=edge_index,
            cand_nodes=candidate_nodes,
            distance_features=distance_features,
            graph_data=graph_data,
            device=device
        )
        
        # Create PCST nodes with combined prizes
        pcst_nodes = []
        for i, node_id in enumerate(candidate_nodes):
            # Combine PPR and node scores as prize
            prize = self.alpha * ppr_scores[i].item() + (1 - self.alpha) * torch.sigmoid(node_scores[i]).item()
            pcst_nodes.append(PCSTNode(node_id, prize))
        
        # Create PCST edges using cached data
        pcst_edges = []
        
        if prepared_data['edge_index'].size(1) > 0:  # 有边存在
            # 使用预计算的边成本基础值
            edge_cost_base = prepared_data['edge_cost_base']
            rel_ids = prepared_data['rel_ids']
            rel_frequency = prepared_data['rel_frequency']
            node_mapping = prepared_data['node_mapping']
            
            # 计算关系惩罚（向量化）
            rel_penalties = torch.zeros_like(rel_ids, dtype=torch.float, device=device)
            for i, rel_id in enumerate(rel_ids):
                rel_key = (int(rel_id.item()),)
                freq = rel_frequency.get(rel_key, 1)
                if self.use_log_penalty:
                    penalty = self.penalty_min + (self.penalty_max - self.penalty_min) * (1 / (1 + np.log(freq + 1)))
                else:
                    penalty = self.penalty_min + (self.penalty_max - self.penalty_min) / (freq + 1)
                rel_penalties[i] = penalty
            
            # 计算最终边成本（向量化）
            final_edge_costs = edge_cost_base + self.gamma * rel_penalties
            
            # 构建PCST边
            edge_index_local = prepared_data['edge_index']
            for i in range(edge_index_local.size(1)):
                src_local = int(edge_index_local[0, i].item())
                dst_local = int(edge_index_local[1, i].item())
                
                # 转换回原始节点ID
                src_global = candidate_nodes[src_local]
                dst_global = candidate_nodes[dst_local]
                
                cost = final_edge_costs[i].item()
                pcst_edges.append(PCSTEdge(src_global, dst_global, cost, i))
        
        # Solve PCST
        if pcst_nodes and pcst_edges:
            selected_nodes, selected_edges = self.pcst.solve_pcst(pcst_nodes, pcst_edges)
        else:
            selected_nodes, selected_edges = [], []
        
        return selected_nodes, selected_edges
    
    def _compute_relation_frequency(self, edge_index: torch.Tensor, graph_data=None) -> Dict[Tuple[int,...], int]:
        """计算关系类型的频率，用于稀有度惩罚"""
        relation_freq = {}
        
        if graph_data is not None and hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None:
            for i in range(edge_index.size(1)):
                if i < len(graph_data.edge_attr):
                    edge_feat = graph_data.edge_attr[i]
                    if edge_feat.dim() > 0 and edge_feat.numel() > 1:
                        feat_signature = edge_feat[:min(3, edge_feat.size(0))]
                        quantized = torch.round(feat_signature * self._rel_quant_scale).int()
                        relation_type = tuple(int(x.item()) for x in quantized)
                    else:
                        # 单维也返回 tuple 以保持类型一致
                        relation_type = (int(round(float(edge_feat.item()) * self._rel_quant_scale)),)
                    relation_freq[relation_type] = relation_freq.get(relation_type, 0) + 1
        else:
            for i in range(edge_index.size(1)):
                src = int(edge_index[0, i].item())
                dst = int(edge_index[1, i].item())
                relation_type = (min(src, dst) % 10, )
                relation_freq[relation_type] = relation_freq.get(relation_type, 0) + 1
        
        return relation_freq
    
    def _compute_relation_penalty(self, edge_idx: int, relation_freq: Dict[Tuple[int,...], int], graph_data=None) -> float:
        """计算关系稀有度惩罚：稀有关系更贵，常见关系更便宜
        
        Args:
            edge_idx: 边索引
            relation_freq: 关系频率字典
            graph_data: 图数据
            
        Returns:
            惩罚值，范围在 [penalty_min, penalty_max]，支持对数缩放
        """
        # 构造 relation_type 与 compute_frequency 保持一致
        if graph_data is not None and hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None and edge_idx < len(graph_data.edge_attr):
            edge_feat = graph_data.edge_attr[edge_idx]
            if edge_feat.dim() > 0 and edge_feat.numel() > 1:
                feat_signature = edge_feat[:min(3, edge_feat.size(0))]
                quantized = torch.round(feat_signature * self._rel_quant_scale).int()
                relation_type = tuple(int(x.item()) for x in quantized)
            else:
                relation_type = (int(round(float(edge_feat.item()) * self._rel_quant_scale)),)
        else:
            if hasattr(graph_data, 'edge_index'):
                src = int(graph_data.edge_index[0, edge_idx].item())
                dst = int(graph_data.edge_index[1, edge_idx].item())
                relation_type = (min(src, dst) % 10,)
            else:
                relation_type = (0,)

        freq = relation_freq.get(relation_type, 1)
        total_edges = sum(relation_freq.values()) if relation_freq else 1
        
        # 计算稀有度比例 (0, 1]，越稀有越接近1
        rarity_ratio = total_edges / (freq * total_edges)  # 简化为 1/freq
        
        if self.use_log_penalty:
            # 对数缩放：log(1 + rarity_ratio * scale_factor)
            # 使用自然对数，scale_factor调整敏感度
            scale_factor = 10.0
            raw_penalty = np.log(1 + rarity_ratio * scale_factor)
            # 归一化到 [0, 1] 然后映射到 [penalty_min, penalty_max]
            max_log_penalty = np.log(1 + scale_factor)  # 最大可能的对数惩罚
            normalized_penalty = raw_penalty / max_log_penalty
        else:
            # 线性缩放：直接使用稀有度比例
            normalized_penalty = min(1.0, rarity_ratio)
        
        # 映射到指定范围
        penalty = self.penalty_min + normalized_penalty * (self.penalty_max - self.penalty_min)
        
        return float(penalty)