import torch
import numpy as np
from collections import deque, defaultdict, OrderedDict
from typing import Dict, List, Set, Union, Optional, Tuple
import networkx as nx
import sys
import os
import xxhash
import time

# 添加根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
try:
    from bmssp import BMSSPSolver
except ImportError:
    print("Warning: BMSSP not available, using BFS fallback")
    BMSSPSolver = None


class DistanceCalculator:
    """
    距离计算器
    支持BFS、Dijkstra和BMSSP算法
    为GNN-RAG³的DDE层提供距离信息
    """
    
    def __init__(self, method: str = "bfs", max_distance: int = 10):
        """
        Args:
            method: 距离计算方法 ("bfs", "dijkstra", "bmssp")
            max_distance: 最大距离限制
        """
        self.method = method
        self.max_distance = max_distance
        
        if method == "bmssp" and BMSSPSolver is not None:
            self.bmssp_solver = BMSSPSolver()
        else:
            self.bmssp_solver = None
    
    def compute_distances(self, 
                         graph: Union[nx.Graph, Dict],
                         seeds: List[int],
                         weighted: bool = False,
                         edge_weights: Optional[Dict[Tuple[int, int], float]] = None) -> Dict[int, int]:
        """
        计算从种子节点到所有节点的距离
        
        Args:
            graph: 图结构 (NetworkX图或邻接表)
            seeds: 种子节点列表
            weighted: 是否使用边权重
            edge_weights: 边权重字典 {(src, dst): weight}
            
        Returns:
            distances: 节点距离字典 {node_id: distance}
        """
        if self.method == "bfs" and not weighted:
            return self._bfs_distances(graph, seeds)
        elif self.method == "dijkstra" or weighted:
            return self._dijkstra_distances(graph, seeds, edge_weights)
        elif self.method == "bmssp" and self.bmssp_solver is not None:
            return self._bmssp_distances(graph, seeds, edge_weights)
        else:
            # 默认使用BFS
            return self._bfs_distances(graph, seeds)
    
    def _bfs_distances(self, 
                      graph: Union[nx.Graph, Dict],
                      seeds: List[int]) -> Dict[int, int]:
        """
        BFS计算距离
        
        Args:
            graph: 图结构
            seeds: 种子节点
            
        Returns:
            distances: 距离字典
        """
        distances = {}
        queue = deque()
        
        # 初始化种子节点
        for seed in seeds:
            distances[seed] = 0
            queue.append((seed, 0))
        
        # BFS遍历
        while queue:
            node, dist = queue.popleft()
            
            if dist >= self.max_distance:
                continue
            
            # 获取邻居节点
            neighbors = self._get_neighbors(graph, node)
            
            for neighbor in neighbors:
                if neighbor not in distances:
                    distances[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
        
        return distances
    
    def _dijkstra_distances(self, 
                           graph: Union[nx.Graph, Dict],
                           seeds: List[int],
                           edge_weights: Optional[Dict[Tuple[int, int], float]] = None) -> Dict[int, int]:
        """
        Dijkstra算法计算距离
        
        Args:
            graph: 图结构
            seeds: 种子节点
            edge_weights: 边权重
            
        Returns:
            distances: 距离字典
        """
        import heapq
        
        distances = {}
        heap = []
        
        # 初始化种子节点
        for seed in seeds:
            distances[seed] = 0
            heapq.heappush(heap, (0, seed))
        
        while heap:
            dist, node = heapq.heappop(heap)
            
            if dist > distances.get(node, float('inf')):
                continue
            
            if dist >= self.max_distance:
                continue
            
            # 获取邻居节点
            neighbors = self._get_neighbors(graph, node)
            
            for neighbor in neighbors:
                # 计算边权重
                if edge_weights:
                    weight = edge_weights.get((node, neighbor), 1.0)
                    weight = max(weight, edge_weights.get((neighbor, node), 1.0))
                else:
                    weight = 1.0
                
                new_dist = dist + weight
                
                if new_dist < distances.get(neighbor, float('inf')):
                    distances[neighbor] = new_dist
                    heapq.heappush(heap, (new_dist, neighbor))
        
        # 转换为整数距离
        int_distances = {}
        for node, dist in distances.items():
            int_distances[node] = int(round(dist))
        
        return int_distances
    
    def _bmssp_distances(self, 
                        graph: Union[nx.Graph, Dict],
                        seeds: List[int],
                        edge_weights: Optional[Dict[Tuple[int, int], float]] = None) -> Dict[int, int]:
        """
        BMSSP算法计算距离
        
        Args:
            graph: 图结构
            seeds: 种子节点
            edge_weights: 边权重
            
        Returns:
            distances: 距离字典
        """
        if self.bmssp_solver is None:
            print("BMSSP solver not available, falling back to BFS")
            return self._bfs_distances(graph, seeds)
        
        try:
            # 转换图格式为BMSSP所需格式
            if isinstance(graph, nx.Graph):
                adj_list = {}
                for node in graph.nodes():
                    adj_list[node] = list(graph.neighbors(node))
            else:
                adj_list = graph
            
            # 使用BMSSP计算距离
            distances = self.bmssp_solver.compute_distances_from_sources(
                adj_list, seeds, max_distance=self.max_distance
            )
            
            return distances
            
        except Exception as e:
            print(f"BMSSP computation failed: {e}, falling back to BFS")
            return self._bfs_distances(graph, seeds)
    
    def _get_neighbors(self, graph: Union[nx.Graph, Dict], node: int) -> List[int]:
        """获取节点的邻居"""
        if isinstance(graph, nx.Graph):
            return list(graph.neighbors(node))
        elif isinstance(graph, dict):
            return graph.get(node, [])
        else:
            raise ValueError(f"Unsupported graph type: {type(graph)}")
    
    def compute_pairwise_distances(self, 
                                  graph: Union[nx.Graph, Dict],
                                  node_pairs: List[Tuple[int, int]],
                                  weighted: bool = False,
                                  edge_weights: Optional[Dict[Tuple[int, int], float]] = None) -> Dict[Tuple[int, int], int]:
        """
        计算节点对之间的距离
        
        Args:
            graph: 图结构
            node_pairs: 节点对列表
            weighted: 是否使用权重
            edge_weights: 边权重
            
        Returns:
            pair_distances: 节点对距离字典
        """
        pair_distances = {}
        
        # 收集所有唯一的源节点
        sources = list(set([pair[0] for pair in node_pairs]))
        
        # 批量计算距离
        all_distances = {}
        for source in sources:
            distances = self.compute_distances(graph, [source], weighted, edge_weights)
            all_distances[source] = distances
        
        # 提取节点对距离
        for src, dst in node_pairs:
            if src in all_distances and dst in all_distances[src]:
                pair_distances[(src, dst)] = all_distances[src][dst]
            else:
                pair_distances[(src, dst)] = self.max_distance  # 不可达
        
        return pair_distances


def bfs_or_dijkstra(graph: Union[nx.Graph, Dict], 
                   seeds: List[int], 
                   weighted: bool = False,
                   edge_weights: Optional[Dict[Tuple[int, int], float]] = None,
                   max_distance: int = 10) -> Dict[int, int]:
    """
    统一的距离计算接口
    
    Args:
        graph: 图结构
        seeds: 种子节点列表
        weighted: 是否使用权重
        edge_weights: 边权重字典
        max_distance: 最大距离
        
    Returns:
        distances: 距离字典 {node_id: distance}
    """
    method = "dijkstra" if weighted else "bfs"
    calculator = DistanceCalculator(method=method, max_distance=max_distance)
    return calculator.compute_distances(graph, seeds, weighted, edge_weights)


def compute_distances_bfs(graph: Union[nx.Graph, Dict], 
                         seeds: List[int],
                         max_hops: int = 10) -> Dict[int, int]:
    """
    BFS距离计算（与DDE层兼容的接口）
    
    Args:
        graph: 图结构
        seeds: 种子节点
        max_hops: 最大跳数
        
    Returns:
        distances: 距离字典
    """
    return bfs_or_dijkstra(graph, seeds, weighted=False, max_distance=max_hops)


def compute_direction_encoding(edge_index: torch.Tensor, 
                              distances: Dict[int, int],
                              num_directions: int = 8) -> torch.Tensor:
    """
    计算边的方向编码
    
    Args:
        edge_index: 边索引 [2, num_edges]
        distances: 节点距离字典
        num_directions: 方向类别数
        
    Returns:
        directions: 方向编码 [num_edges]
    """
    num_edges = edge_index.size(1)
    directions = torch.zeros(num_edges, dtype=torch.long)
    
    for i in range(num_edges):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        src_dist = distances.get(src, 10)
        dst_dist = distances.get(dst, 10)
        
        # 方向编码逻辑
        if src_dist < dst_dist:
            direction = 0  # 远离种子
        elif src_dist > dst_dist:
            direction = 1  # 靠近种子
        else:
            direction = 2  # 同距离
        
        # 可以扩展为更复杂的方向编码
        directions[i] = direction % num_directions
    
    return directions


class GraphDistanceCache:
    """
    图距离缓存器
    避免重复计算相同图的距离
    """
    
    def __init__(self, max_cache_size: int = 100):
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.access_count = defaultdict(int)
    
    def get_cache_key(self, graph_hash: str, seeds: List[int], method: str) -> str:
        """生成缓存键"""
        seeds_str = "_".join(map(str, sorted(seeds)))
        return f"{graph_hash}_{seeds_str}_{method}"
    
    def get_distances(self, 
                     graph_hash: str,
                     graph: Union[nx.Graph, Dict],
                     seeds: List[int],
                     method: str = "bfs",
                     **kwargs) -> Dict[int, int]:
        """
        获取距离（带缓存）
        
        Args:
            graph_hash: 图的哈希值
            graph: 图结构
            seeds: 种子节点
            method: 计算方法
            **kwargs: 其他参数
            
        Returns:
            distances: 距离字典
        """
        cache_key = self.get_cache_key(graph_hash, seeds, method)
        
        if cache_key in self.cache:
            self.access_count[cache_key] += 1
            return self.cache[cache_key]
        
        # 计算距离
        calculator = DistanceCalculator(method=method)
        distances = calculator.compute_distances(graph, seeds, **kwargs)
        
        # 缓存结果
        self._add_to_cache(cache_key, distances)
        
        return distances
    
    def _add_to_cache(self, key: str, distances: Dict[int, int]):
        """添加到缓存"""
        if len(self.cache) >= self.max_cache_size:
            # 移除最少使用的缓存项
            least_used_key = min(self.cache.keys(), 
                                key=lambda k: self.access_count[k])
            del self.cache[least_used_key]
            del self.access_count[least_used_key]
        
        self.cache[key] = distances
        self.access_count[key] = 1
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        self.access_count.clear()


# 全局距离缓存实例
distance_cache = GraphDistanceCache()


def compute_graph_hash(graph: Union[nx.Graph, Dict]) -> str:
    """计算图的哈希值"""
    try:
        if isinstance(graph, nx.Graph):
            # 对于NetworkX图，使用边列表的哈希
            edges = sorted(graph.edges())
            return str(hash(tuple(edges)))
        elif isinstance(graph, dict):
            # 对于邻接表，使用键值对的哈希
            items = sorted([(k, tuple(sorted(v))) for k, v in graph.items()])
            return str(hash(tuple(items)))
        else:
            return "unknown"
    except:
        return "unknown"


class DDEDistanceCache:
    """
    DDE距离和方向索引缓存器
    缓存索引而非嵌入张量，因为embedding是可学习参数
    """
    
    def __init__(self, max_items: int = 256):
        self.cache = OrderedDict()
        self.max_items = max_items
        self.hit_count = 0
        self.miss_count = 0
        
    def get_cache_key(self, 
                     cand_nodes: List[int], 
                     query_nodes: List[int]) -> str:
        """生成缓存键"""
        cand_fingerprint = xxhash.xxh64(str(sorted(cand_nodes))).hexdigest()
        query_fingerprint = xxhash.xxh64(str(sorted(query_nodes))).hexdigest()
        return f"{cand_fingerprint}_{query_fingerprint}"
    
    def get(self, key: str) -> Optional[Tuple[Dict[int, int], Dict[Tuple[int, int], int]]]:
        """获取缓存的距离和方向索引"""
        if key in self.cache:
            # 移动到末尾（LRU）
            value = self.cache[key]
            self.cache.move_to_end(key)
            self.hit_count += 1
            return value
        else:
            self.miss_count += 1
            return None
    
    def put(self, key: str, value: Tuple[Dict[int, int], Dict[Tuple[int, int], int]]):
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


# 全局DDE距离缓存实例
dde_distance_cache = DDEDistanceCache()


def bfs_distance_and_direction(graph: Union[nx.Graph, Dict], 
                              seeds: List[int], 
                              cand_nodes: List[int],
                              max_hops: int = 6) -> Tuple[Dict[int, int], Dict[Tuple[int, int], int]]:
    """
    使用BFS计算距离和方向索引
    
    Args:
        graph: 图结构 (NetworkX图或邻接字典)
        seeds: 种子节点列表
        cand_nodes: 候选节点列表
        max_hops: 最大跳数
        
    Returns:
        hop_distances: {node_id -> hop_distance}
        direction_indices: {(src, dst) -> direction_index}
                          direction_index: 0=out, 1=in, 2=self
    """
    hop_distances = {}
    direction_indices = {}
    
    # 初始化种子节点距离为0
    queue = deque()
    for seed in seeds:
        if seed in cand_nodes:
            hop_distances[seed] = 0
            queue.append((seed, 0))
    
    # BFS遍历
    visited = set(seeds)
    
    while queue:
        current_node, current_dist = queue.popleft()
        
        if current_dist >= max_hops:
            continue
            
        # 获取邻居节点
        if isinstance(graph, nx.Graph):
            neighbors = list(graph.neighbors(current_node))
        else:
            neighbors = graph.get(current_node, [])
        
        for neighbor in neighbors:
            if neighbor in cand_nodes:
                # 记录方向索引
                direction_indices[(current_node, neighbor)] = 0  # out
                direction_indices[(neighbor, current_node)] = 1  # in
                
                # 更新距离
                if neighbor not in visited:
                    hop_distances[neighbor] = current_dist + 1
                    queue.append((neighbor, current_dist + 1))
                    visited.add(neighbor)
    
    # 添加自环方向
    for node in cand_nodes:
        direction_indices[(node, node)] = 2  # self
        if node not in hop_distances:
            hop_distances[node] = max_hops  # 不可达节点设为最大距离
    
    return hop_distances, direction_indices


def get_distance_and_direction_cached(graph: Union[nx.Graph, Dict],
                                    seeds: List[int],
                                    cand_nodes: List[int],
                                    max_hops: int = 6) -> Tuple[Dict[int, int], Dict[Tuple[int, int], int]]:
    """
    带缓存的距离和方向索引计算
    
    Args:
        graph: 图结构
        seeds: 种子节点列表  
        cand_nodes: 候选节点列表
        max_hops: 最大跳数
        
    Returns:
        hop_distances: {node_id -> hop_distance}
        direction_indices: {(src, dst) -> direction_index}
    """
    # 生成缓存键
    cache_key = dde_distance_cache.get_cache_key(cand_nodes, seeds)
    
    # 尝试从缓存获取
    cached_result = dde_distance_cache.get(cache_key)
    if cached_result is not None:
        return cached_result
    
    # 缓存未命中，重新计算
    start_time = time.time()
    hop_distances, direction_indices = bfs_distance_and_direction(
        graph, seeds, cand_nodes, max_hops
    )
    compute_time = time.time() - start_time
    
    # 添加到缓存
    dde_distance_cache.put(cache_key, (hop_distances, direction_indices))
    
    return hop_distances, direction_indices


def convert_to_tensor_indices(hop_distances: Dict[int, int],
                            direction_indices: Dict[Tuple[int, int], int],
                            edge_index: torch.Tensor,
                            node_mapping: Dict[int, int],
                            device: torch.device) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    将距离和方向索引转换为张量格式
    
    Args:
        hop_distances: 节点跳数距离字典
        direction_indices: 边方向索引字典
        edge_index: 边索引张量 [2, num_edges]
        node_mapping: 原节点ID到子图节点ID的映射
        device: 设备
        
    Returns:
        hop_idx_tensor: 节点跳数索引张量 [num_nodes]
        dir_idx_tensor: 边方向索引张量 [num_edges]
    """
    num_nodes = len(node_mapping)
    num_edges = edge_index.size(1)
    
    # 创建跳数索引张量
    hop_idx_tensor = torch.zeros(num_nodes, dtype=torch.long, device=device)
    for orig_node_id, local_node_id in node_mapping.items():
        hop_dist = hop_distances.get(orig_node_id, 6)  # 默认最大距离
        hop_idx_tensor[local_node_id] = min(hop_dist, 6)  # 限制最大跳数
    
    # 创建方向索引张量
    dir_idx_tensor = torch.zeros(num_edges, dtype=torch.long, device=device)
    
    # 反向映射：local_id -> orig_id
    local_to_orig = {v: k for k, v in node_mapping.items()}
    
    for edge_idx in range(num_edges):
        src_local = edge_index[0, edge_idx].item()
        dst_local = edge_index[1, edge_idx].item()
        
        src_orig = local_to_orig[src_local]
        dst_orig = local_to_orig[dst_local]
        
        # 获取方向索引
        dir_idx = direction_indices.get((src_orig, dst_orig), 0)  # 默认为out
        dir_idx_tensor[edge_idx] = dir_idx
    
    return hop_idx_tensor, dir_idx_tensor


def get_dde_cache_stats() -> Dict[str, float]:
    """获取DDE缓存统计信息"""
    return {
        "hit_rate": dde_distance_cache.get_hit_rate(),
        "cache_size": len(dde_distance_cache.cache),
        "max_size": dde_distance_cache.max_items,
        "hit_count": dde_distance_cache.hit_count,
        "miss_count": dde_distance_cache.miss_count
    }


def clear_dde_distance_cache():
    """清空DDE距离缓存"""
    dde_distance_cache.clear()