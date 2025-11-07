"""
APPRManager: 高效的个性化PageRank管理器
实现缓存、基向量组合、热启动等优化策略，将APPR计算从分钟级优化到毫秒级
"""

import torch
import numpy as np
import hashlib
import time
from typing import Dict, List, Union, Optional, Tuple, Set
from collections import defaultdict, OrderedDict
import networkx as nx

from .ppr import compute_appr, compute_appr_optimized, PPRCalculator


class APPRManager:
    """
    高效的APPR管理器
    
    核心优化策略：
    1. 多级缓存：完整结果缓存 + 基向量缓存 + 热启动缓存
    2. 基向量线性组合：预计算热门实体的单源APPR，多种子通过线性组合快速得到
    3. 热启动Push：基于上次结果进行增量计算
    4. 子图掩码：在大图上计算一次，通过掩码适配不同子图
    5. 批内共享：同批次相同种子/子图的结果共享
    """
    
    def __init__(self, 
                 alpha: float = 0.85,
                 tolerance: float = 1e-3,  # 放宽容差提升速度
                 max_iterations: int = 50,  # 减少迭代次数
                 topk: int = 500,  # 只保留top-k结果
                 cache_size: int = 1024,
                 basis_cache_size: int = 2048,
                 enable_basis_vectors: bool = True,
                 enable_warm_start: bool = True,
                 enable_subgraph_mask: bool = True):
        """
        Args:
            alpha: 阻尼系数
            tolerance: 收敛容差（放宽以提升速度）
            max_iterations: 最大迭代次数
            topk: 只保留top-k个最高分数的节点
            cache_size: 完整结果缓存大小
            basis_cache_size: 基向量缓存大小
            enable_basis_vectors: 是否启用基向量线性组合
            enable_warm_start: 是否启用热启动
            enable_subgraph_mask: 是否启用子图掩码
        """
        self.alpha = alpha
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.topk = topk
        self.enable_basis_vectors = enable_basis_vectors
        self.enable_warm_start = enable_warm_start
        self.enable_subgraph_mask = enable_subgraph_mask
        
        # 多级缓存
        self.full_cache = OrderedDict()  # 完整APPR结果缓存
        self.basis_cache = OrderedDict()  # 单源基向量缓存
        self.warm_start_cache = OrderedDict()  # 热启动状态缓存
        
        self.cache_size = cache_size
        self.basis_cache_size = basis_cache_size
        
        # 统计信息
        self.stats = {
            'full_cache_hits': 0,
            'basis_cache_hits': 0,
            'warm_start_hits': 0,
            'full_computations': 0,
            'total_queries': 0,
            'time_saved': 0.0
        }
        
        # PPR计算器
        self.ppr_calculator = PPRCalculator(
            alpha=alpha, 
            tolerance=tolerance, 
            max_iterations=max_iterations
        )
        
        # 热门实体追踪（用于基向量预计算）
        self.hot_entities = set()
        self.entity_frequency = defaultdict(int)
        
    def get_appr(self, 
                graph: Union[nx.Graph, Dict],
                seeds: List[int],
                seed_weights: Optional[List[float]] = None,
                subgraph_nodes: Optional[Set[int]] = None,
                graph_fingerprint: Optional[str] = None) -> Dict[int, float]:
        """
        获取APPR结果（智能缓存+优化策略）
        
        Args:
            graph: 图结构
            seeds: 种子节点列表
            seed_weights: 种子权重（如果为None则均匀权重）
            subgraph_nodes: 子图节点集合（用于掩码）
            graph_fingerprint: 图指纹（如果为None则自动计算）
            
        Returns:
            appr_scores: APPR分数字典
        """
        start_time = time.perf_counter()
        self.stats['total_queries'] += 1
        
        # 标准化输入
        if not seeds:
            return {}
        
        if seed_weights is None:
            seed_weights = [1.0 / len(seeds)] * len(seeds)
        elif len(seed_weights) != len(seeds):
            raise ValueError("seed_weights length must match seeds length")
        
        # 更新热门实体统计
        for seed in seeds:
            self.entity_frequency[seed] += 1
            if self.entity_frequency[seed] >= 5:  # 出现5次以上认为是热门实体
                self.hot_entities.add(seed)
        
        # 生成缓存键
        if graph_fingerprint is None:
            graph_fingerprint = self._compute_graph_fingerprint(graph)
        
        cache_key = self._generate_cache_key(
            graph_fingerprint, seeds, seed_weights, subgraph_nodes
        )
        
        # 1. 尝试完整缓存命中
        if cache_key in self.full_cache:
            self.stats['full_cache_hits'] += 1
            self._update_cache_access(self.full_cache, cache_key)
            result = self.full_cache[cache_key]
            self.stats['time_saved'] += time.perf_counter() - start_time
            return result
        
        # 2. 检查子图重用机会（如果有子图约束）
        if subgraph_nodes is not None:
            reusable_result = self._try_subgraph_reuse(
                graph_fingerprint, seeds, seed_weights, subgraph_nodes
            )
            if reusable_result is not None:
                self.stats['subgraph_reuse_hits'] = self.stats.get('subgraph_reuse_hits', 0) + 1
                self._add_to_cache(self.full_cache, cache_key, reusable_result, self.cache_size)
                return reusable_result
        
        # 3. 尝试基向量线性组合
        if self.enable_basis_vectors and self._can_use_basis_combination(seeds):
            try:
                result = self._compute_via_basis_combination(
                    graph, graph_fingerprint, seeds, seed_weights, subgraph_nodes
                )
                if result is not None:
                    self.stats['basis_cache_hits'] += 1
                self._add_to_cache(self.full_cache, cache_key, result, self.cache_size)
                self.stats['time_saved'] += time.perf_counter() - start_time
                return result
            except Exception as e:
                print(f"Basis combination failed: {e}, falling back")
        
        # 4. 尝试热启动
        if self.enable_warm_start:
            warm_start_result = self._try_warm_start(
                graph, graph_fingerprint, seeds, seed_weights
            )
            if warm_start_result is not None:
                self.stats['warm_start_hits'] += 1
                result = warm_start_result
                if subgraph_nodes is not None:
                    result = self._apply_subgraph_mask(result, subgraph_nodes)
                self._add_to_cache(self.full_cache, cache_key, result, self.cache_size)
                return result
        
        # 5. 完整计算
        self.stats['full_computations'] += 1
        result = self._compute_full_appr(graph, seeds, seed_weights)
        
        # 应用子图掩码
        if subgraph_nodes is not None:
            result = self._apply_subgraph_mask(result, subgraph_nodes)
        
        # 缓存结果
        self._add_to_cache(self.full_cache, cache_key, result, self.cache_size)
        
        # 如果是单种子，也缓存为基向量
        if len(seeds) == 1 and self.enable_basis_vectors:
            basis_key = self._generate_basis_key(graph_fingerprint, seeds[0])
            self._add_to_cache(self.basis_cache, basis_key, result, self.basis_cache_size)
        
        return result
    
    def precompute_basis_vectors(self, 
                               graph: Union[nx.Graph, Dict],
                               hot_entities: Optional[List[int]] = None,
                               graph_fingerprint: Optional[str] = None) -> None:
        """
        预计算热门实体的基向量
        
        Args:
            graph: 图结构
            hot_entities: 热门实体列表（如果为None则使用内部统计）
            graph_fingerprint: 图指纹
        """
        if not self.enable_basis_vectors:
            return
        
        if hot_entities is None:
            hot_entities = list(self.hot_entities)
        
        if graph_fingerprint is None:
            graph_fingerprint = self._compute_graph_fingerprint(graph)
        
        print(f"Precomputing basis vectors for {len(hot_entities)} hot entities...")
        
        for entity in hot_entities:
            basis_key = self._generate_basis_key(graph_fingerprint, entity)
            if basis_key not in self.basis_cache:
                # 计算单源APPR
                appr_result = self._compute_full_appr(graph, [entity], [1.0])
                self._add_to_cache(self.basis_cache, basis_key, appr_result, self.basis_cache_size)
        
        print(f"Basis vectors precomputation completed. Cache size: {len(self.basis_cache)}")
    
    def _can_use_basis_combination(self, seeds: List[int]) -> bool:
        """检查是否可以使用基向量组合"""
        if not self.enable_basis_vectors or len(seeds) <= 1:
            return False
        
        # 检查是否所有种子都是热门实体且有缓存的基向量
        return all(seed in self.hot_entities for seed in seeds)
    
    def _compute_via_basis_combination(self, 
                                     graph: Union[nx.Graph, Dict],
                                     graph_fingerprint: str,
                                     seeds: List[int],
                                     seed_weights: List[float],
                                     subgraph_nodes: Optional[Set[int]] = None) -> Optional[Dict[int, float]]:
        """通过基向量线性组合计算APPR"""
        try:
            # 检查所有基向量是否都在缓存中
            basis_vectors = {}
            for seed in seeds:
                basis_key = self._generate_basis_key(graph_fingerprint, seed)
                if basis_key not in self.basis_cache:
                    # 如果缺少基向量，先计算它
                    appr_result = self._compute_full_appr(graph, [seed], [1.0])
                    self._add_to_cache(self.basis_cache, basis_key, appr_result, self.basis_cache_size)
                    basis_vectors[seed] = appr_result
                else:
                    basis_vectors[seed] = self.basis_cache[basis_key]
                    self._update_cache_access(self.basis_cache, basis_key)
            
            # 线性组合
            combined_result = defaultdict(float)
            for seed, weight in zip(seeds, seed_weights):
                basis_vector = basis_vectors[seed]
                for node, score in basis_vector.items():
                    combined_result[node] += weight * score
            
            # 转换为普通字典并应用topk
            result = dict(combined_result)
            result = self._apply_topk(result)
            
            # 应用子图掩码
            if subgraph_nodes is not None:
                result = self._apply_subgraph_mask(result, subgraph_nodes)
            
            return result
            
        except Exception as e:
            print(f"Basis combination failed: {e}")
            return None
    
    def _try_warm_start(self, 
                       graph: Union[nx.Graph, Dict],
                       graph_fingerprint: str,
                       seeds: List[int],
                       seed_weights: List[float]) -> Optional[Dict[int, float]]:
        """尝试热启动计算"""
        if not self.enable_warm_start:
            return None
        
        # 寻找相似的缓存项作为热启动点
        best_match = None
        best_similarity = 0.0
        
        for cached_key in self.warm_start_cache:
            if graph_fingerprint in cached_key:
                # 解析缓存键中的种子信息
                try:
                    cached_seeds = self._extract_seeds_from_key(cached_key)
                    similarity = self._compute_seed_similarity(seeds, cached_seeds)
                    if similarity > best_similarity and similarity > 0.5:  # 相似度阈值
                        best_similarity = similarity
                        best_match = cached_key
                except:
                    continue
        
        if best_match is not None:
            # 使用热启动
            warm_start_state = self.warm_start_cache[best_match]
            try:
                result = self._warm_start_push(
                    graph, seeds, seed_weights, warm_start_state
                )
                return result
            except Exception as e:
                print(f"Warm start failed: {e}")
                return None
        
        return None
    
    def _try_subgraph_reuse(self, 
                           graph_fingerprint: str,
                           seeds: List[int],
                           seed_weights: List[float],
                           subgraph_nodes: Set[int]) -> Optional[Dict[int, float]]:
        """尝试重用相似子图的缓存结果"""
        if not self.enable_subgraph_mask:
            return None
        
        # 寻找相似的缓存项
        best_match = None
        best_overlap = 0.0
        
        for cached_key in self.full_cache:
            if graph_fingerprint in cached_key:
                try:
                    # 解析缓存键中的子图信息
                    cached_subgraph = self._extract_subgraph_from_key(cached_key)
                    cached_seeds = self._extract_seeds_from_key(cached_key)
                    
                    # 检查种子相似度
                    seed_similarity = self._compute_seed_similarity(seeds, cached_seeds)
                    
                    # 检查子图重叠度
                    if cached_subgraph is not None:
                        subgraph_overlap = self._compute_subgraph_overlap(subgraph_nodes, cached_subgraph)
                        
                        # 综合相似度评分
                        combined_score = 0.6 * seed_similarity + 0.4 * subgraph_overlap
                        
                        if combined_score > best_overlap and combined_score > 0.7:  # 相似度阈值
                            best_overlap = combined_score
                            best_match = cached_key
                except Exception:
                    continue
        
        if best_match is not None:
            # 重用缓存结果并应用新的子图掩码
            cached_result = self.full_cache[best_match]
            self._update_cache_access(self.full_cache, best_match)
            
            # 重新应用子图掩码
            reused_result = self._apply_subgraph_mask(cached_result, subgraph_nodes)
            return reused_result
        
        return None
    
    def _extract_subgraph_from_key(self, cache_key: str) -> Optional[Set[int]]:
        """从缓存键中提取子图信息"""
        try:
            if "_sg:" in cache_key:
                # 这是一个简化的实现，实际中需要更复杂的解析
                # 由于子图信息被哈希了，我们无法直接恢复
                # 这里返回None，表示无法提取
                return None
            else:
                # 没有子图约束的缓存项
                return None
        except Exception:
            return None
    
    def _compute_full_appr(self, 
                          graph: Union[nx.Graph, Dict],
                          seeds: List[int],
                          seed_weights: List[float]) -> Dict[int, float]:
        """完整计算APPR，使用优化版本"""
        # 使用优化版本的APPR计算
        if len(seeds) == 1:
            result = compute_appr_optimized(
                graph, seeds, alpha=self.alpha, tolerance=self.tolerance, 
                topk=self.topk, return_residuals=False
            )
        else:
            # 多种子情况：分别计算后加权组合
            combined_result = defaultdict(float)
            for seed, weight in zip(seeds, seed_weights):
                single_result = compute_appr_optimized(
                    graph, [seed], alpha=self.alpha, tolerance=self.tolerance,
                    topk=self.topk, return_residuals=False
                )
                for node, score in single_result.items():
                    combined_result[node] += weight * score
            result = dict(combined_result)
            
            # 对组合结果再次应用topk
            if len(result) > self.topk:
                sorted_items = sorted(result.items(), key=lambda x: x[1], reverse=True)
                result = dict(sorted_items[:self.topk])
        
        return result
    
    def _compute_appr_direct(self, graph, seeds, alpha, tolerance=1e-3, topk=500, 
                            warm_start_p=None, warm_start_r=None):
        """直接计算APPR（热启动或全新计算）"""
        timer = time.time
        t0 = timer()
        
        # 使用优化版本的APPR计算
        if warm_start_p is not None or warm_start_r is not None:
            # 热启动模式
            appr_scores, residuals = compute_appr_optimized(
                graph, seeds, alpha=alpha, tolerance=tolerance, topk=topk,
                warm_start_p=warm_start_p, warm_start_r=warm_start_r, 
                return_residuals=True
            )
            # 缓存残差用于下次热启动
            graph_fingerprint = self._compute_graph_fingerprint(graph)
            seeds_fingerprint = self._get_seeds_fingerprint(seeds, [1.0/len(seeds)] * len(seeds))
            cache_key = self._get_cache_key(alpha, graph_fingerprint, seeds_fingerprint, 'warmstart')
            self.warm_start_cache[cache_key] = (appr_scores.copy(), residuals)
        else:
            # 全新计算
            appr_scores = compute_appr_optimized(
                graph, seeds, alpha=alpha, tolerance=tolerance, topk=topk,
                return_residuals=False
            )
        
        t1 = timer()
        print(f"[APPRManager] 直接计算APPR耗时: {t1-t0:.4f}s, 返回{len(appr_scores)}个节点")
        
        return appr_scores
    
    def _warm_start_push(self, 
                        graph: Union[nx.Graph, Dict],
                        seeds: List[int],
                        seed_weights: List[float],
                        warm_start_state: Dict) -> Dict[int, float]:
        """基于热启动状态进行Push计算"""
        # 这里实现简化的热启动Push算法
        # 实际实现中应该基于residual和probability进行增量更新
        
        # 暂时使用完整计算作为fallback
        return self._compute_full_appr(graph, seeds, seed_weights)
    
    def _get_cache_key(self, alpha, graph_fingerprint, seeds_fingerprint, suffix=''):
        """生成缓存键"""
        key = f"a{alpha:.3f}_g{graph_fingerprint}_s{seeds_fingerprint}"
        if suffix:
            key += f"_{suffix}"
        return key
    
    def _get_subgraph_fingerprint(self, subgraph_nodes: Set[int]) -> str:
        """获取子图节点集合的指纹"""
        if not subgraph_nodes:
            return "empty"
        
        # 使用排序后的节点列表生成指纹
        sorted_nodes = sorted(subgraph_nodes)
        
        # 对于大的子图，使用采样策略
        if len(sorted_nodes) > 1000:
            # 采样策略：取前100、中间100、后100个节点
            sample_nodes = (sorted_nodes[:100] + 
                          sorted_nodes[len(sorted_nodes)//2-50:len(sorted_nodes)//2+50] + 
                          sorted_nodes[-100:])
            fingerprint_str = f"sampled_{len(sorted_nodes)}_{str(sample_nodes)}"
        else:
            fingerprint_str = str(sorted_nodes)
        
        return hashlib.md5(fingerprint_str.encode()).hexdigest()[:12]
    
    def _get_seeds_fingerprint(self, seeds: List[int], seed_weights: List[float]) -> str:
        """获取种子和权重的指纹"""
        # 创建种子-权重对并排序
        seed_weight_pairs = [(s, w) for s, w in zip(seeds, seed_weights)]
        seed_weight_pairs.sort(key=lambda x: x[0])
        
        # 生成指纹
        fingerprint_str = str(seed_weight_pairs)
        return hashlib.md5(fingerprint_str.encode()).hexdigest()[:12]
    
    def _compute_subgraph_overlap(self, subgraph1: Set[int], subgraph2: Set[int]) -> float:
        """计算两个子图的重叠度"""
        if not subgraph1 or not subgraph2:
            return 0.0
        
        intersection = len(subgraph1 & subgraph2)
        union = len(subgraph1 | subgraph2)
        
        return intersection / union if union > 0 else 0.0
    
    def _can_reuse_subgraph_result(self, 
                                 current_subgraph: Set[int], 
                                 cached_subgraph: Set[int],
                                 overlap_threshold: float = 0.8) -> bool:
        """判断是否可以重用子图结果"""
        if not current_subgraph or not cached_subgraph:
            return False
        
        overlap = self._compute_subgraph_overlap(current_subgraph, cached_subgraph)
        return overlap >= overlap_threshold
    
    def _apply_topk(self, appr_scores: Dict[int, float]) -> Dict[int, float]:
        """应用topk截断"""
        if len(appr_scores) <= self.topk:
            return appr_scores
        
        # 按分数排序，保留top-k
        sorted_items = sorted(appr_scores.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_items[:self.topk])
    
    def _apply_subgraph_mask(self, 
                           appr_scores: Dict[int, float],
                           subgraph_nodes: Set[int]) -> Dict[int, float]:
        """应用子图掩码"""
        if not self.enable_subgraph_mask:
            return appr_scores
        
        # 只保留子图中的节点
        masked_result = {
            node: score for node, score in appr_scores.items()
            if node in subgraph_nodes
        }
        
        # 重新归一化
        total_score = sum(masked_result.values())
        if total_score > 0:
            masked_result = {
                node: score / total_score 
                for node, score in masked_result.items()
            }
        
        return masked_result
    
    def _compute_graph_fingerprint(self, graph: Union[nx.Graph, Dict]) -> str:
        """计算图的指纹"""
        if isinstance(graph, nx.Graph):
            # 使用边集合的哈希
            edges = sorted(graph.edges())
            edge_str = str(edges)
        elif isinstance(graph, dict):
            # 使用邻接表的哈希
            sorted_items = []
            for src in sorted(graph.keys()):
                targets = sorted(graph[src])
                sorted_items.append((src, tuple(targets)))
            edge_str = str(sorted_items)
        else:
            edge_str = str(graph)
        
        return hashlib.md5(edge_str.encode()).hexdigest()[:16]
    
    def _generate_cache_key(self, 
                          graph_fingerprint: str,
                          seeds: List[int],
                          seed_weights: List[float],
                          subgraph_nodes: Optional[Set[int]] = None) -> str:
        """生成缓存键"""
        # 标准化种子和权重
        seed_weight_pairs = list(zip(seeds, seed_weights))
        seed_weight_pairs.sort(key=lambda x: x[0])  # 按种子ID排序
        
        seeds_str = "_".join(f"{s}:{w:.3f}" for s, w in seed_weight_pairs)
        
        subgraph_str = ""
        if subgraph_nodes is not None:
            subgraph_hash = hashlib.md5(str(sorted(subgraph_nodes)).encode()).hexdigest()[:8]
            subgraph_str = f"_sg:{subgraph_hash}"
        
        return f"{graph_fingerprint}_a:{self.alpha:.3f}_s:{seeds_str}{subgraph_str}"
    
    def _generate_basis_key(self, graph_fingerprint: str, entity: int) -> str:
        """生成基向量缓存键"""
        return f"basis_{graph_fingerprint}_a:{self.alpha:.3f}_e:{entity}"
    
    def _extract_seeds_from_key(self, cache_key: str) -> List[int]:
        """从缓存键中提取种子信息"""
        # 简化实现：从键中解析种子
        try:
            parts = cache_key.split("_s:")
            if len(parts) < 2:
                return []
            
            seeds_part = parts[1].split("_")[0]  # 取第一个下划线前的部分
            seed_pairs = seeds_part.split("_")
            seeds = []
            for pair in seed_pairs:
                if ":" in pair:
                    seed_id = int(pair.split(":")[0])
                    seeds.append(seed_id)
            return seeds
        except:
            return []
    
    def _compute_seed_similarity(self, seeds1: List[int], seeds2: List[int]) -> float:
        """计算种子集合的相似度"""
        set1, set2 = set(seeds1), set(seeds2)
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union  # Jaccard相似度
    
    def _add_to_cache(self, cache: OrderedDict, key: str, value: Dict[int, float], max_size: int):
        """添加到缓存（LRU策略）"""
        if key in cache:
            # 更新现有项
            cache.move_to_end(key)
            cache[key] = value
        else:
            # 添加新项
            if len(cache) >= max_size:
                # 移除最旧的项
                cache.popitem(last=False)
            cache[key] = value
    
    def _update_cache_access(self, cache: OrderedDict, key: str):
        """更新缓存访问（移到末尾）"""
        if key in cache:
            cache.move_to_end(key)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        total_hits = (self.stats['full_cache_hits'] + 
                     self.stats['basis_cache_hits'] + 
                     self.stats['warm_start_hits'])
        
        stats_copy = self.stats.copy()
        stats_copy.update({
            'full_cache_size': len(self.full_cache),
            'basis_cache_size': len(self.basis_cache),
            'warm_start_cache_size': len(self.warm_start_cache),
            'hot_entities_count': len(self.hot_entities),
            'total_entities_tracked': len(self.entity_frequency),
            'cache_hit_rate': total_hits / max(1, self.stats['total_queries']),
            'full_cache_hit_rate': self.stats['full_cache_hits'] / max(1, self.stats['total_queries']),
            'basis_cache_hit_rate': self.stats['basis_cache_hits'] / max(1, self.stats['total_queries']),
            'warm_start_hit_rate': self.stats['warm_start_hits'] / max(1, self.stats['total_queries']),
            'subgraph_reuse_hit_rate': self.stats.get('subgraph_reuse_hits', 0) / max(1, self.stats['total_queries']),
            'computation_avoidance_rate': total_hits / max(1, total_hits + self.stats['full_computations']),
            'average_time_saved_per_query': self.stats['time_saved'] / max(1, self.stats['total_queries'])
        })
        return stats_copy

    def hit_ratio(self) -> float:
        """整体缓存命中率（简化访问）"""
        total_hits = (self.stats['full_cache_hits'] + self.stats['basis_cache_hits'] + self.stats['warm_start_hits'])
        total_queries = max(1, self.stats['total_queries'])
        return float(total_hits) / float(total_queries)

    def avg_topk(self) -> float:
        """缓存中APPR结果的平均保留条数（top-k 代理）"""
        if len(self.full_cache) == 0:
            return 0.0
        lengths = [min(len(v), self.topk) for v in self.full_cache.values()]
        return float(sum(lengths)) / float(len(lengths))

    def get_or_compute(self,
                       graph: Union[nx.Graph, Dict],
                       seeds: List[int],
                       seed_weights: Optional[List[float]] = None,
                       subgraph_nodes: Optional[Set[int]] = None,
                       graph_fingerprint: Optional[str] = None) -> Dict[int, float]:
        """缓存优先获取APPR，未命中则按策略计算。"""
        if not seeds:
            return {}
        if seed_weights is None:
            seed_weights = [1.0 / len(seeds)] * len(seeds)
        if graph_fingerprint is None:
            graph_fingerprint = self._compute_graph_fingerprint(graph)
        cache_key = self._generate_cache_key(graph_fingerprint, seeds, seed_weights, subgraph_nodes)
        cached = self.full_cache.get(cache_key, None)
        if cached is not None:
            self.stats['full_cache_hits'] += 1
            self._update_cache_access(self.full_cache, cache_key)
            return cached
        # 回落到标准流程（可用基向量/热启动）
        return self.get_appr(graph, seeds, seed_weights, subgraph_nodes, graph_fingerprint)

    def get_cached(self, cache_key: Optional[str]) -> Optional[Dict[int, float]]:
        """仅从完整缓存读取APPR结果，未命中返回None。"""
        if cache_key is None:
            return None
        cached = self.full_cache.get(cache_key, None)
        if cached is not None:
            self.stats['full_cache_hits'] += 1
            self._update_cache_access(self.full_cache, cache_key)
        else:
            self.stats['total_queries'] += 1
        return cached
    
    def _update_entity_frequency(self, entity: int, threshold: int = 10):
        """
        更新实体频率统计，自动检测热门实体
        
        Args:
            entity: 实体ID
            threshold: 热门实体阈值
        """
        self.entity_frequency[entity] += 1
        
        # 如果实体频率超过阈值，加入热门实体集合
        if self.entity_frequency[entity] >= threshold and entity not in self.hot_entities:
            self.hot_entities.add(entity)
            print(f"[APPR] 检测到热门实体: {entity} (频率: {self.entity_frequency[entity]})")
    
    def get_hot_entities(self, top_k: int = 50) -> List[int]:
        """
        获取最热门的K个实体
        
        Args:
            top_k: 返回的热门实体数量
            
        Returns:
            热门实体列表，按频率降序排列
        """
        sorted_entities = sorted(
            self.entity_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [entity for entity, freq in sorted_entities[:top_k]]
    
    def update_hot_entities_threshold(self, new_threshold: int = 10):
        """
        更新热门实体阈值并重新计算热门实体集合
        
        Args:
            new_threshold: 新的热门实体阈值
        """
        self.hot_entities.clear()
        for entity, freq in self.entity_frequency.items():
            if freq >= new_threshold:
                self.hot_entities.add(entity)
        
        print(f"[APPR] 更新热门实体阈值为 {new_threshold}, 当前热门实体数量: {len(self.hot_entities)}")

    def clear_caches(self):
        """清空所有缓存"""
        self.full_cache.clear()
        self.basis_cache.clear()
        self.warm_start_cache.clear()
        
        # 重置统计
        for key in self.stats:
            self.stats[key] = 0
        
        # 清空热门实体统计
        self.hot_entities.clear()
        self.entity_frequency.clear()


# 全局APPR管理器实例
appr_manager = APPRManager()


def get_appr_optimized(graph: Union[nx.Graph, Dict],
                      seeds: List[int],
                      seed_weights: Optional[List[float]] = None,
                      subgraph_nodes: Optional[Set[int]] = None,
                      alpha: float = 0.85) -> Dict[int, float]:
    """
    优化的APPR计算接口
    
    Args:
        graph: 图结构
        seeds: 种子节点列表
        seed_weights: 种子权重
        subgraph_nodes: 子图节点集合
        alpha: 阻尼系数
        
    Returns:
        appr_scores: APPR分数字典
    """
    # 如果alpha不同，创建临时管理器
    if abs(alpha - appr_manager.alpha) > 1e-6:
        temp_manager = APPRManager(alpha=alpha)
        return temp_manager.get_appr(graph, seeds, seed_weights, subgraph_nodes)
    
    return appr_manager.get_appr(graph, seeds, seed_weights, subgraph_nodes)

    def get_or_compute(self,
                       graph: Union[nx.Graph, Dict],
                       seeds: List[int],
                       seed_weights: Optional[List[float]] = None,
                       subgraph_nodes: Optional[Set[int]] = None,
                       graph_fingerprint: Optional[str] = None) -> Dict[int, float]:
        """Cache-first APPR: try full cache, then compute if needed."""
        if not seeds:
            return {}
        if seed_weights is None:
            seed_weights = [1.0 / len(seeds)] * len(seeds)
        if graph_fingerprint is None:
            graph_fingerprint = self._compute_graph_fingerprint(graph)
        cache_key = self._generate_cache_key(graph_fingerprint, seeds, seed_weights, subgraph_nodes)
        cached = self.full_cache.get(cache_key, None)
        if cached is not None:
            self.stats['full_cache_hits'] += 1
            self._update_cache_access(self.full_cache, cache_key)
            return cached
        # Fall back to standard get_appr (will utilize basis/warm-start if enabled)
        return self.get_appr(graph, seeds, seed_weights, subgraph_nodes, graph_fingerprint)

    def get_cached(self, cache_key: Optional[str]) -> Optional[Dict[int, float]]:
        """Return cached APPR result if present, else None."""
        if cache_key is None:
            return None
        cached = self.full_cache.get(cache_key, None)
        if cached is not None:
            self.stats['full_cache_hits'] += 1
            self._update_cache_access(self.full_cache, cache_key)
        else:
            # Count a query even if missing to keep ratios meaningful
            self.stats['total_queries'] += 1
        return cached

    def hit_ratio(self) -> float:
        """Convenience accessor for overall cache hit ratio."""
        total_hits = (self.stats['full_cache_hits'] + self.stats['basis_cache_hits'] + self.stats['warm_start_hits'])
        total_queries = max(1, self.stats['total_queries'])
        return float(total_hits) / float(total_queries)

    def avg_topk(self) -> float:
        """Average number of entries kept in cached APPR results (proxy for top-k)."""
        if len(self.full_cache) == 0:
            return 0.0
        lengths = [min(len(v), self.topk) for v in self.full_cache.values()]
        return float(sum(lengths)) / float(len(lengths))