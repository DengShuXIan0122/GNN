import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import networkx as nx
import sys
import os
from collections import defaultdict

# 添加根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class PPRCalculator:
    """
    个性化PageRank计算器
    为GNN-RAG³的混合检索提供结构先验
    """
    
    def __init__(self, 
                 alpha: float = 0.85,
                 tolerance: float = 1e-6,
                 max_iterations: int = 100,
                 use_push: bool = True):
        """
        Args:
            alpha: 阻尼系数 (通常0.85)
            tolerance: 收敛容差
            max_iterations: 最大迭代次数
            use_push: 是否使用push-based算法
        """
        self.alpha = alpha
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.use_push = use_push
        
        # 使用内置的PPR实现
        self.ppr_module = PersonalizedPageRank(alpha=1-alpha, tol=tolerance, max_iter=max_iterations)
    
    def compute_appr(self, 
                    graph: Union[nx.Graph, Dict, torch.Tensor],
                    seeds: List[int],
                    alpha: Optional[float] = None,
                    tolerance: Optional[float] = None) -> Dict[int, float]:
        """
        计算近似个性化PageRank (APPR)
        
        Args:
            graph: 图结构 (NetworkX图、邻接表或邻接矩阵)
            seeds: 种子节点列表
            alpha: 阻尼系数 (覆盖默认值)
            tolerance: 收敛容差 (覆盖默认值)
            
        Returns:
            appr_scores: APPR分数字典 {node_id: score}
        """
        if alpha is None:
            alpha = self.alpha
        if tolerance is None:
            tolerance = self.tolerance
        
        # 尝试使用现有PPR模块
        if self.ppr_module is not None:
            try:
                return self._compute_with_existing_module(graph, seeds, alpha, tolerance)
            except Exception as e:
                print(f"Existing PPR module failed: {e}, using fallback")
        
        # 使用fallback实现
        return self._compute_appr_fallback(graph, seeds, alpha, tolerance)
    
    def compute_appr_optimized(self, 
                              graph: Union[nx.Graph, Dict, torch.Tensor],
                              seeds: List[int],
                              alpha: Optional[float] = None,
                              tolerance: Optional[float] = None,
                              warm_start_p: Optional[Dict[int, float]] = None,
                              warm_start_r: Optional[Dict[int, float]] = None,
                              topk: Optional[int] = None,
                              return_residuals: bool = False) -> Union[Dict[int, float], Tuple[Dict[int, float], Dict[int, float]]]:
        """
        优化版本的APPR计算，支持warm-start、topk截断和稀疏返回
        
        Args:
            graph: 图结构
            seeds: 种子节点列表
            alpha: 重启概率
            tolerance: 收敛容差
            warm_start_p: 热启动的概率分布
            warm_start_r: 热启动的残差分布
            topk: 返回top-k节点，None表示返回所有
            return_residuals: 是否返回残差（用于后续warm start）
            
        Returns:
            如果return_residuals=False: Dict[int, float] APPR分数
            如果return_residuals=True: Tuple[Dict[int, float], Dict[int, float]] (APPR分数, 残差)
        """
        if alpha is None:
            alpha = self.alpha
        if tolerance is None:
            tolerance = self.tolerance
            
        # 使用优化的push-based算法
        appr_scores, residuals = self._push_based_appr(
            graph, seeds, alpha, tolerance, warm_start_p, warm_start_r, topk
        )
        
        if return_residuals:
            return appr_scores, residuals
        else:
            return appr_scores
    
    def _compute_with_existing_module(self, 
                                     graph: Union[nx.Graph, Dict, torch.Tensor],
                                     seeds: List[int],
                                     alpha: float,
                                     tolerance: float) -> Dict[int, float]:
        """使用内置的PPR模块计算"""
        try:
            # 转换图格式
            if isinstance(graph, nx.Graph):
                edge_index = torch.tensor(list(graph.edges())).T
                num_nodes = len(graph.nodes())
            elif isinstance(graph, dict):
                edges = []
                for src, targets in graph.items():
                    for tgt in targets:
                        edges.append([src, tgt])
                edge_index = torch.tensor(edges).T if edges else torch.empty((2, 0))
                # 安全地计算节点数量，避免空序列错误
                max_key = max(graph.keys()) if graph else 0
                max_target = 0
                if graph:
                    for targets in graph.values():
                        if targets:  # 只处理非空的targets列表
                            max_target = max(max_target, max(targets))
                num_nodes = max(max_key, max_target) + 1
            else:
                edge_index = graph
                # 安全地处理空tensor的情况
                if graph.numel() > 0:
                    num_nodes = int(graph.max()) + 1
                else:
                    num_nodes = 1  # 默认至少有一个节点
            
            # 计算PPR
            ppr_tensor = self.ppr_module.compute_ppr(edge_index, num_nodes, seeds)
            return {i: float(ppr_tensor[i]) for i in range(num_nodes) if ppr_tensor[i] > tolerance}
        except Exception as e:
            print(f"Warning: PPR module failed: {e}, using fallback")
            return self._compute_appr_fallback(graph, seeds, alpha, tolerance)
    
    def _compute_appr_fallback(self, 
                              graph: Union[nx.Graph, Dict, torch.Tensor],
                              seeds: List[int],
                              alpha: float,
                              tolerance: float,
                              warm_start_p: Optional[Dict[int, float]] = None,
                              warm_start_r: Optional[Dict[int, float]] = None,
                              topk: Optional[int] = None) -> Union[Dict[int, float], Tuple[Dict[int, float], Dict[int, float]]]:
        """
        备用APPR计算方法，支持warm-start和topk截断
        """
        if self.use_push:
            appr_scores, residuals = self._push_based_appr(
                graph, seeds, alpha, tolerance, warm_start_p, warm_start_r, topk
            )
            # 为了向后兼容，如果没有warm start参数，只返回appr_scores
            if warm_start_p is None and warm_start_r is None:
                return appr_scores
            else:
                return appr_scores, residuals
        else:
            return self._power_iteration_appr(graph, seeds, alpha, tolerance)
    
    def _push_based_appr(self, 
                        graph: Union[nx.Graph, Dict, torch.Tensor],
                        seeds: List[int],
                        alpha: float,
                        tolerance: float,
                        warm_start_p: Optional[Dict[int, float]] = None,
                        warm_start_r: Optional[Dict[int, float]] = None,
                        topk: Optional[int] = None) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Push-based APPR算法，支持warm-start和topk截断
        更适合稀疏图和局部查询
        
        Args:
            graph: 图结构
            seeds: 种子节点列表
            alpha: damping factor
            tolerance: 收敛容差
            warm_start_p: 热启动的概率分布
            warm_start_r: 热启动的残差分布
            topk: 返回top-k节点，None表示返回所有
            
        Returns:
            (appr_scores, residuals): APPR分数和残差
        """
        # 构建邻接表
        if isinstance(graph, nx.Graph):
            adj_list = {node: list(graph.neighbors(node)) for node in graph.nodes()}
            all_nodes = set(graph.nodes())
        elif isinstance(graph, dict):
            adj_list = graph
            all_nodes = set(adj_list.keys())
            for neighbors in adj_list.values():
                all_nodes.update(neighbors)
        else:
            adj_list, all_nodes = self._tensor_to_adj_list(graph)
        
        # 初始化 - 支持warm start
        if warm_start_p is not None:
            p = defaultdict(float, warm_start_p)
        else:
            p = defaultdict(float)
            
        if warm_start_r is not None:
            r = defaultdict(float, warm_start_r)
        else:
            r = defaultdict(float)
            # 设置种子节点
            for seed in seeds:
                r[seed] = 1.0 / len(seeds)
        
        # Push操作
        queue = []
        # 初始化队列 - 包含所有有残差的节点
        for node, residual in r.items():
            if residual > tolerance:
                queue.append(node)
        
        visited = set()
        
        while queue:
            node = queue.pop(0)
            if node in visited and r[node] <= tolerance:
                continue
            
            visited.add(node)
            
            if r[node] <= tolerance:
                continue
            
            # Push操作
            push_mass = r[node]
            p[node] += alpha * push_mass
            r[node] = 0
            
            # 分发到邻居
            neighbors = adj_list.get(node, [])
            if neighbors:
                mass_per_neighbor = (1 - alpha) * push_mass / len(neighbors)
                for neighbor in neighbors:
                    r[neighbor] += mass_per_neighbor
                    if neighbor not in visited and r[neighbor] > tolerance:
                        queue.append(neighbor)
        
        # 过滤小值并应用topk
        appr_scores = {node: score for node, score in p.items() if score > tolerance}
        
        if topk is not None and len(appr_scores) > topk:
            # 按分数排序并取top-k
            sorted_items = sorted(appr_scores.items(), key=lambda x: x[1], reverse=True)
            appr_scores = dict(sorted_items[:topk])
        
        # 返回残差用于后续的warm start
        residuals = {node: res for node, res in r.items() if res > tolerance}
        
        return appr_scores, residuals
    
    def _power_iteration_appr(self, 
                             graph: Union[nx.Graph, Dict, torch.Tensor],
                             seeds: List[int],
                             alpha: float,
                             tolerance: float) -> Dict[int, float]:
        """
        幂迭代APPR算法
        适合密集图
        """
        # 转换为邻接矩阵
        if isinstance(graph, torch.Tensor):
            adj_matrix = graph
            num_nodes = adj_matrix.size(0)
        else:
            adj_matrix, num_nodes = self._graph_to_matrix(graph)
        
        # 计算转移矩阵
        degree = adj_matrix.sum(dim=1, keepdim=True)
        degree[degree == 0] = 1  # 避免除零
        transition_matrix = adj_matrix / degree
        
        # 初始化分布
        p = torch.zeros(num_nodes)
        for seed in seeds:
            if seed < num_nodes:
                p[seed] = 1.0 / len(seeds)
        
        # 幂迭代
        for _ in range(self.max_iterations):
            p_new = alpha * p + (1 - alpha) * torch.mv(transition_matrix.t(), p)
            
            # 检查收敛
            if torch.norm(p_new - p, p=1) < tolerance:
                break
            
            p = p_new
        
        # 转换为字典格式
        appr_scores = {}
        for i, score in enumerate(p.cpu().numpy()):
            if score > tolerance:
                appr_scores[i] = float(score)
        
        return appr_scores
    
    def _graph_to_tensor(self, graph: Union[nx.Graph, Dict]) -> torch.Tensor:
        """将图转换为邻接矩阵张量"""
        if isinstance(graph, nx.Graph):
            nodes = list(graph.nodes())
            num_nodes = len(nodes)
            node_to_idx = {node: i for i, node in enumerate(nodes)}
            
            adj_matrix = torch.zeros(num_nodes, num_nodes)
            for edge in graph.edges():
                i, j = node_to_idx[edge[0]], node_to_idx[edge[1]]
                adj_matrix[i, j] = 1.0
                adj_matrix[j, i] = 1.0  # 无向图
            
        elif isinstance(graph, dict):
            all_nodes = set(graph.keys())
            for neighbors in graph.values():
                all_nodes.update(neighbors)
            
            nodes = sorted(all_nodes)
            num_nodes = len(nodes)
            node_to_idx = {node: i for i, node in enumerate(nodes)}
            
            adj_matrix = torch.zeros(num_nodes, num_nodes)
            for node, neighbors in graph.items():
                i = node_to_idx[node]
                for neighbor in neighbors:
                    j = node_to_idx[neighbor]
                    adj_matrix[i, j] = 1.0
        
        return adj_matrix
    
    def _graph_to_matrix(self, graph: Union[nx.Graph, Dict]) -> Tuple[torch.Tensor, int]:
        """将图转换为邻接矩阵和节点数"""
        adj_matrix = self._graph_to_tensor(graph)
        return adj_matrix, adj_matrix.size(0)
    
    def _tensor_to_adj_list(self, adj_matrix: torch.Tensor) -> Tuple[Dict[int, List[int]], set]:
        """将邻接矩阵转换为邻接表"""
        num_nodes = adj_matrix.size(0)
        adj_list = {}
        all_nodes = set(range(num_nodes))
        
        for i in range(num_nodes):
            neighbors = []
            for j in range(num_nodes):
                if adj_matrix[i, j] > 0:
                    neighbors.append(j)
            adj_list[i] = neighbors
        
        return adj_list, all_nodes
    
    def compute_multi_seed_appr(self, 
                               graph: Union[nx.Graph, Dict, torch.Tensor],
                               seed_groups: List[List[int]],
                               alpha: Optional[float] = None) -> List[Dict[int, float]]:
        """
        计算多组种子的APPR
        
        Args:
            graph: 图结构
            seed_groups: 种子组列表
            alpha: 阻尼系数
            
        Returns:
            appr_results: 每组种子的APPR结果列表
        """
        results = []
        for seeds in seed_groups:
            appr_scores = self.compute_appr(graph, seeds, alpha)
            results.append(appr_scores)
        
        return results
    
    def normalize_appr_scores(self, appr_scores: Dict[int, float]) -> Dict[int, float]:
        """归一化APPR分数"""
        if not appr_scores:
            return {}
        
        total_score = sum(appr_scores.values())
        if total_score == 0:
            return appr_scores
        
        normalized_scores = {node: score / total_score 
                           for node, score in appr_scores.items()}
        
        return normalized_scores
    
    def filter_top_k(self, 
                    appr_scores: Dict[int, float], 
                    k: int) -> Dict[int, float]:
        """保留top-k节点"""
        if len(appr_scores) <= k:
            return appr_scores
        
        sorted_items = sorted(appr_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_items = sorted_items[:k]
        
        return dict(top_k_items)


def compute_appr(graph: Union[nx.Graph, Dict, torch.Tensor],
                seeds: List[int],
                alpha: float = 0.85,
                tolerance: float = 1e-6) -> Dict[int, float]:
    """
    统一的APPR计算接口
    
    Args:
        graph: 图结构
        seeds: 种子节点列表
        alpha: 阻尼系数
        tolerance: 收敛容差
        
    Returns:
        appr_scores: APPR分数字典 {node_id: score}
    """
    calculator = PPRCalculator(alpha=alpha, tolerance=tolerance)
    return calculator.compute_appr(graph, seeds, alpha, tolerance)


def compute_appr_optimized(graph: Union[nx.Graph, Dict, torch.Tensor],
                          seeds: List[int],
                          alpha: float = 0.85,
                          tolerance: float = 1e-3,
                          warm_start_p: Optional[Dict[int, float]] = None,
                          warm_start_r: Optional[Dict[int, float]] = None,
                          topk: Optional[int] = 500,
                          return_residuals: bool = False) -> Union[Dict[int, float], Tuple[Dict[int, float], Dict[int, float]]]:
    """
    优化版本的APPR计算便利函数，支持warm-start、topk截断和稀疏返回
    
    Args:
        graph: 图结构
        seeds: 种子节点列表
        alpha: 重启概率
        tolerance: 收敛容差（默认放宽到1e-3以提升速度）
        warm_start_p: 热启动的概率分布
        warm_start_r: 热启动的残差分布
        topk: 返回top-k节点（默认500）
        return_residuals: 是否返回残差（用于后续warm start）
        
    Returns:
        如果return_residuals=False: Dict[int, float] APPR分数
        如果return_residuals=True: Tuple[Dict[int, float], Dict[int, float]] (APPR分数, 残差)
    """
    calculator = PPRCalculator(alpha=alpha, tolerance=tolerance, use_push=True)
    return calculator.compute_appr_optimized(
        graph, seeds, alpha, tolerance, warm_start_p, warm_start_r, topk, return_residuals
    )


class APPRCache:
    """
    APPR结果缓存器
    避免重复计算相同参数的APPR
    """
    
    def __init__(self, max_cache_size: int = 50):
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.access_count = defaultdict(int)
    
    def get_cache_key(self, 
                     graph_hash: str, 
                     seeds: List[int], 
                     alpha: float) -> str:
        """生成缓存键"""
        seeds_str = "_".join(map(str, sorted(seeds)))
        return f"{graph_hash}_{seeds_str}_{alpha:.3f}"
    
    def get_appr(self, 
                graph_hash: str,
                graph: Union[nx.Graph, Dict, torch.Tensor],
                seeds: List[int],
                alpha: float = 0.85,
                tolerance: float = 1e-6) -> Dict[int, float]:
        """
        获取APPR结果（带缓存）
        
        Args:
            graph_hash: 图的哈希值
            graph: 图结构
            seeds: 种子节点
            alpha: 阻尼系数
            tolerance: 收敛容差
            
        Returns:
            appr_scores: APPR分数字典
        """
        cache_key = self.get_cache_key(graph_hash, seeds, alpha)
        
        if cache_key in self.cache:
            self.access_count[cache_key] += 1
            return self.cache[cache_key]
        
        # 计算APPR
        appr_scores = compute_appr(graph, seeds, alpha, tolerance)
        
        # 缓存结果
        self._add_to_cache(cache_key, appr_scores)
        
        return appr_scores
    
    def _add_to_cache(self, key: str, appr_scores: Dict[int, float]):
        """添加到缓存"""
        if len(self.cache) >= self.max_cache_size:
            # 移除最少使用的缓存项
            least_used_key = min(self.cache.keys(), 
                                key=lambda k: self.access_count[k])
            del self.cache[least_used_key]
            del self.access_count[least_used_key]
        
        self.cache[key] = appr_scores
        self.access_count[key] = 1
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        self.access_count.clear()


# 全局APPR缓存实例
appr_cache = APPRCache()


def compute_appr_with_cache(graph_hash: str,
                           graph: Union[nx.Graph, Dict, torch.Tensor],
                           seeds: List[int],
                           alpha: float = 0.85,
                           tolerance: float = 1e-6) -> Dict[int, float]:
    """
    带缓存的APPR计算
    
    Args:
        graph_hash: 图的哈希值
        graph: 图结构
        seeds: 种子节点
        alpha: 阻尼系数
        tolerance: 收敛容差
        
    Returns:
        appr_scores: APPR分数字典
    """
    return appr_cache.get_appr(graph_hash, graph, seeds, alpha, tolerance)


# ==================== PPR和DDE实现 ====================

class PersonalizedPageRank:
    """个性化PageRank算法实现"""
    
    def __init__(self, alpha: float = 0.15, tol: float = 1e-6, max_iter: int = 100):
        """
        Args:
            alpha: 重启概率 (damping factor)
            tol: 收敛容忍度
            max_iter: 最大迭代次数
        """
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
    
    def compute_ppr(self, edge_index: torch.Tensor, num_nodes: int, 
                   topic_nodes: List[int], edge_weights: Optional[torch.Tensor] = None,
                   apply_degree_penalty: bool = True) -> torch.Tensor:
        """
        计算个性化PageRank分数
        
        Args:
            edge_index: 边索引 [2, num_edges]
            num_nodes: 节点总数
            topic_nodes: 主题节点列表 (personalization seeds)
            edge_weights: 边权重 [num_edges] (可选)
            
        Returns:
            ppr_scores: PPR分数 [num_nodes]
        """
        device = edge_index.device
        
        # 构建邻接矩阵
        # 确保数据类型一致性
        target_dtype = edge_index.dtype if edge_index.dtype.is_floating_point else torch.float32
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1), device=device, dtype=target_dtype)
        
        # 创建稀疏邻接矩阵
        adj = torch.sparse_coo_tensor(
            edge_index, edge_weights, (num_nodes, num_nodes), device=device
        ).coalesce()
        
        # 计算出度并归一化
        out_degree = torch.sparse.sum(adj, dim=1).to_dense()
        out_degree = torch.clamp(out_degree, min=1e-8)  # 避免除零
        
        # 归一化邻接矩阵 (列随机化)
        row_indices = adj.indices()[0]
        col_indices = adj.indices()[1]
        values = adj.values() / out_degree[row_indices]
        
        adj_norm = torch.sparse_coo_tensor(
            torch.stack([row_indices, col_indices]), values, 
            (num_nodes, num_nodes), device=device, dtype=target_dtype
        ).coalesce()
        
        # 初始化个性化向量
        # Get dtype from edge_index to ensure consistency
        target_dtype = edge_index.dtype if edge_index.dtype.is_floating_point else torch.float32
        personalization = torch.zeros(num_nodes, device=device, dtype=target_dtype)
        for node in topic_nodes:
            if 0 <= node < num_nodes:
                personalization[node] = 1.0 / len(topic_nodes)
        
        # 初始化PPR向量
        ppr = personalization.clone()
        
        # 幂迭代法求解PPR
        for _ in range(self.max_iter):
            ppr_new = (1 - self.alpha) * torch.sparse.mm(adj_norm.t(), ppr.unsqueeze(1)).squeeze(1) + \
                      self.alpha * personalization
            
            # 检查收敛
            if torch.norm(ppr_new - ppr, p=1) < self.tol:
                break
                
            ppr = ppr_new
        
        # 应用大度数节点penalty机制
        if apply_degree_penalty:
            ppr = self._apply_degree_penalty(ppr, edge_index, num_nodes)
        
        return ppr
    
    def _apply_degree_penalty(self, ppr_scores: torch.Tensor, edge_index: torch.Tensor, 
                            num_nodes: int) -> torch.Tensor:
        """
        对大度数节点应用penalty，避免hub节点反复出现
        
        Args:
            ppr_scores: 原始PPR分数 [num_nodes]
            edge_index: 边索引 [2, num_edges]
            num_nodes: 节点总数
            
        Returns:
            penalized_scores: 应用penalty后的PPR分数 [num_nodes]
        """
        device = ppr_scores.device
        
        # 计算每个节点的度数
        degrees = torch.zeros(num_nodes, device=device, dtype=ppr_scores.dtype)
        unique_nodes, counts = torch.unique(edge_index.flatten(), return_counts=True)
        # 确保unique_nodes为long类型，用于tensor索引
        unique_nodes = unique_nodes.long()
        # 确保counts的数据类型与degrees一致
        counts = counts.to(dtype=ppr_scores.dtype)
        degrees[unique_nodes] = counts
        
        # 应用penalty: score = ppr_score / log(1 + degree)
        # 使用log(1 + degree)避免除零，同时对高度数节点施加更强的penalty
        penalty_factor = torch.log(1.0 + degrees)
        penalty_factor = torch.clamp(penalty_factor, min=1.0)  # 确保penalty_factor >= 1
        
        penalized_scores = ppr_scores / penalty_factor
        
        return penalized_scores


class DirectedDistanceEncoding(nn.Module):
    """方向距离编码器 (DDE)"""
    
    def __init__(self, max_distance: int = 10, embedding_dim: int = 64):
        """
        Args:
            max_distance: 最大距离
            embedding_dim: 嵌入维度
        """
        super().__init__()
        self.max_distance = max_distance
        self.embedding_dim = embedding_dim
        
        # 距离嵌入
        self.distance_embedding = nn.Embedding(max_distance + 1, embedding_dim)
        
        # 方向嵌入 (in/out/self)
        self.direction_embedding = nn.Embedding(3, embedding_dim)
        
        # 组合层
        self.combine_layer = nn.Linear(2 * embedding_dim, embedding_dim)
        
    def forward(self, edge_index: torch.Tensor, distances: torch.Tensor, 
                num_nodes: int) -> torch.Tensor:
        """
        计算边的方向距离编码
        
        Args:
            edge_index: 边索引 [2, num_edges]
            distances: 节点到查询节点的距离 [num_nodes]
            num_nodes: 节点总数
            
        Returns:
            edge_encodings: 边编码 [num_edges, embedding_dim]
        """
        src_nodes, dst_nodes = edge_index[0], edge_index[1]
        
        # 确保节点索引为long类型，用于tensor索引
        src_nodes = src_nodes.long()
        dst_nodes = dst_nodes.long()
        
        # 获取源节点和目标节点的距离
        src_distances = distances[src_nodes]
        dst_distances = distances[dst_nodes]
        
        # 计算边的距离特征 (取最小距离)
        edge_distances = torch.min(src_distances, dst_distances)
        edge_distances = torch.clamp(edge_distances, 0, self.max_distance).long()
        
        # 计算方向特征
        # 0: in (dst距离更小), 1: out (src距离更小), 2: self (距离相等)
        direction = torch.where(
            src_distances < dst_distances, 
            torch.tensor(1, device=edge_index.device),  # out
            torch.where(
                src_distances > dst_distances,
                torch.tensor(0, device=edge_index.device),  # in
                torch.tensor(2, device=edge_index.device)   # self
            )
        )
        
        # 获取嵌入
        distance_emb = self.distance_embedding(edge_distances)
        direction_emb = self.direction_embedding(direction)
        
        # 组合距离和方向嵌入
        combined = torch.cat([distance_emb, direction_emb], dim=-1)
        edge_encodings = self.combine_layer(combined)
        
        return edge_encodings


class PEConv(nn.Module):
    """位置编码卷积层"""
    
    def __init__(self, input_dim: int, hidden_dim: int, pe_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pe_dim = pe_dim
        
        # 消息函数
        self.message_net = nn.Sequential(
            nn.Linear(input_dim + pe_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 更新函数
        self.update_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_pe: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            edge_pe: 边位置编码 [num_edges, pe_dim]
            
        Returns:
            updated_x: 更新后的节点特征 [num_nodes, input_dim]
        """
        src, dst = edge_index[0], edge_index[1]
        
        # 确保节点索引为long类型，用于tensor索引
        src = src.long()
        dst = dst.long()
        
        # 计算消息
        src_features = x[src]  # [num_edges, input_dim]
        messages_input = torch.cat([src_features, edge_pe], dim=-1)
        messages = self.message_net(messages_input)  # [num_edges, hidden_dim]
        
        # 聚合消息
        num_nodes = x.size(0)
        aggregated = torch.zeros(num_nodes, self.hidden_dim, device=x.device, dtype=x.dtype)
        aggregated.index_add_(0, dst, messages)
        
        # 更新节点特征
        update_input = torch.cat([x, aggregated], dim=-1)
        updated_x = self.update_net(update_input)
        
        return updated_x


class DDEModule(nn.Module):
    """DDE模块，包含正向和反向PEConv层"""
    
    def __init__(self, input_dim: int, hidden_dim: int, pe_dim: int, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        
        # 正向和反向PEConv层
        self.forward_convs = nn.ModuleList([
            PEConv(input_dim, hidden_dim, pe_dim) for _ in range(num_layers)
        ])
        
        self.backward_convs = nn.ModuleList([
            PEConv(input_dim, hidden_dim, pe_dim) for _ in range(num_layers)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(input_dim) for _ in range(num_layers * 2)
        ])
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_pe: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            edge_pe: 边位置编码 [num_edges, pe_dim]
            
        Returns:
            output: 处理后的节点特征 [num_nodes, input_dim]
        """
        # 创建反向边索引
        reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        
        current_x = x
        
        for i in range(self.num_layers):
            # 正向传播
            forward_x = self.forward_convs[i](current_x, edge_index, edge_pe)
            forward_x = self.layer_norms[i * 2](forward_x)
            
            # 反向传播
            backward_x = self.backward_convs[i](current_x, reverse_edge_index, edge_pe)
            backward_x = self.layer_norms[i * 2 + 1](backward_x)
            
            # 残差连接
            current_x = current_x + forward_x + backward_x
        
        return current_x