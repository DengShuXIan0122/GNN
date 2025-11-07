"""
自适应候选子图大小模块
基于问题复杂度动态调整候选节点数量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import re
from collections import Counter


class QuestionComplexityAnalyzer:
    """问题复杂度分析器"""
    
    def __init__(self):
        # 复杂度指示词
        self.multi_hop_keywords = [
            'and', 'also', 'both', 'either', 'neither', 'as well as',
            'in addition', 'furthermore', 'moreover', 'besides',
            'what else', 'who else', 'where else', 'when else'
        ]
        
        self.temporal_keywords = [
            'before', 'after', 'during', 'when', 'while', 'since',
            'until', 'first', 'last', 'previous', 'next', 'earlier',
            'later', 'recent', 'current', 'past', 'future'
        ]
        
        self.comparative_keywords = [
            'more', 'less', 'most', 'least', 'better', 'worse',
            'larger', 'smaller', 'higher', 'lower', 'compare',
            'versus', 'vs', 'than', 'between', 'among'
        ]
        
        self.aggregation_keywords = [
            'total', 'sum', 'count', 'number', 'how many', 'all',
            'every', 'each', 'average', 'mean', 'maximum', 'minimum'
        ]
        
        self.negation_keywords = [
            'not', 'no', 'never', 'none', 'nothing', 'nobody',
            'nowhere', 'neither', 'nor', 'without', 'except'
        ]
    
    def analyze_question_complexity(self, question: str) -> Dict[str, float]:
        """
        分析问题复杂度
        
        Args:
            question: 问题文本
            
        Returns:
            complexity_features: 复杂度特征字典
        """
        question_lower = question.lower()
        
        # 1. 词汇复杂度
        words = question_lower.split()
        word_count = len(words)
        unique_words = len(set(words))
        lexical_diversity = unique_words / word_count if word_count > 0 else 0
        
        # 2. 句法复杂度
        sentence_count = len(re.split(r'[.!?]+', question))
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else word_count
        
        # 3. 语义复杂度指标
        multi_hop_score = sum(1 for kw in self.multi_hop_keywords if kw in question_lower)
        temporal_score = sum(1 for kw in self.temporal_keywords if kw in question_lower)
        comparative_score = sum(1 for kw in self.comparative_keywords if kw in question_lower)
        aggregation_score = sum(1 for kw in self.aggregation_keywords if kw in question_lower)
        negation_score = sum(1 for kw in self.negation_keywords if kw in question_lower)
        
        # 4. 实体密度
        # 简单启发式：大写单词可能是实体
        potential_entities = len([w for w in words if w[0].isupper() and len(w) > 1])
        entity_density = potential_entities / word_count if word_count > 0 else 0
        
        # 5. 疑问词复杂度
        wh_words = ['what', 'who', 'where', 'when', 'why', 'how', 'which']
        wh_count = sum(1 for wh in wh_words if wh in question_lower)
        
        return {
            'word_count': word_count,
            'lexical_diversity': lexical_diversity,
            'avg_sentence_length': avg_sentence_length,
            'multi_hop_score': multi_hop_score,
            'temporal_score': temporal_score,
            'comparative_score': comparative_score,
            'aggregation_score': aggregation_score,
            'negation_score': negation_score,
            'entity_density': entity_density,
            'wh_count': wh_count
        }
    
    def analyze_complexity(self, question: str) -> Dict[str, float]:
        """analyze_question_complexity的别名，保持向后兼容"""
        return self.analyze_question_complexity(question)
    
    def compute_complexity_score(self, question: str) -> float:
        """
        计算综合复杂度分数
        
        Args:
            question: 问题文本
            
        Returns:
            complexity_score: 复杂度分数 [0, 1]
        """
        features = self.analyze_question_complexity(question)
        
        # 权重设计
        weights = {
            'word_count': 0.1,
            'lexical_diversity': 0.15,
            'avg_sentence_length': 0.1,
            'multi_hop_score': 0.25,
            'temporal_score': 0.1,
            'comparative_score': 0.1,
            'aggregation_score': 0.1,
            'negation_score': 0.05,
            'entity_density': 0.1,
            'wh_count': 0.05
        }
        
        # 归一化特征
        normalized_features = {
            'word_count': min(features['word_count'] / 50.0, 1.0),  # 50词为满分
            'lexical_diversity': features['lexical_diversity'],
            'avg_sentence_length': min(features['avg_sentence_length'] / 30.0, 1.0),  # 30词/句为满分
            'multi_hop_score': min(features['multi_hop_score'] / 3.0, 1.0),  # 3个关键词为满分
            'temporal_score': min(features['temporal_score'] / 2.0, 1.0),
            'comparative_score': min(features['comparative_score'] / 2.0, 1.0),
            'aggregation_score': min(features['aggregation_score'] / 2.0, 1.0),
            'negation_score': min(features['negation_score'] / 2.0, 1.0),
            'entity_density': min(features['entity_density'] * 2.0, 1.0),  # 50%实体密度为满分
            'wh_count': min(features['wh_count'] / 3.0, 1.0)
        }
        
        # 加权求和
        complexity_score = sum(weights[k] * normalized_features[k] for k in weights)
        
        return complexity_score


class AdaptiveSubgraphSizer(nn.Module):
    """自适应候选子图大小调节器"""
    
    def __init__(self, 
                 min_candidates: int = 300,
                 max_candidates: int = 2000,
                 base_candidates: int = 1200,
                 complexity_weight: float = 0.7,
                 seed_weight: float = 0.2,
                 graph_weight: float = 0.1):
        """
        Args:
            min_candidates: 最小候选节点数
            max_candidates: 最大候选节点数
            base_candidates: 基础候选节点数
            complexity_weight: 问题复杂度权重
            seed_weight: 种子节点数权重
            graph_weight: 图结构权重
        """
        super(AdaptiveSubgraphSizer, self).__init__()
        
        self.min_candidates = min_candidates
        self.max_candidates = max_candidates
        self.base_candidates = base_candidates
        self.complexity_weight = complexity_weight
        self.seed_weight = seed_weight
        self.graph_weight = graph_weight
        
        self.complexity_analyzer = QuestionComplexityAnalyzer()
        
        # 可学习的调节网络
        self.size_predictor = nn.Sequential(
            nn.Linear(13, 64),  # 10个复杂度特征 + 3个图特征
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出[0,1]范围的调节因子
        )
    
    def compute_graph_features(self, 
                              edge_index: torch.Tensor,
                              num_nodes: int,
                              seeds: List[int]) -> torch.Tensor:
        """
        计算图结构特征
        
        Args:
            edge_index: 边索引
            num_nodes: 节点总数
            seeds: 种子节点列表
            
        Returns:
            graph_features: 图特征 [3]
        """
        device = edge_index.device
        
        # 1. 图密度
        num_edges = edge_index.size(1)
        max_edges = num_nodes * (num_nodes - 1) // 2
        graph_density = num_edges / max_edges if max_edges > 0 else 0
        
        # 2. 种子节点比例
        seed_ratio = len(seeds) / num_nodes if num_nodes > 0 else 0
        
        # 3. 平均度数
        degrees = torch.zeros(num_nodes, device=device)
        degrees.scatter_add_(0, edge_index[0], torch.ones(num_edges, device=device))
        degrees.scatter_add_(0, edge_index[1], torch.ones(num_edges, device=device))
        avg_degree = degrees.mean().item() if num_nodes > 0 else 0
        avg_degree_normalized = min(avg_degree / 20.0, 1.0)  # 20为满分度数
        
        return torch.tensor([graph_density, seed_ratio, avg_degree_normalized], 
                          device=device, dtype=torch.float)
    
    def forward(self, 
                question: str,
                edge_index: torch.Tensor,
                num_nodes: int,
                seeds: List[int]) -> int:
        """
        前向传播：计算自适应候选节点数
        
        Args:
            question: 问题文本
            edge_index: 边索引
            num_nodes: 节点总数
            seeds: 种子节点列表
            
        Returns:
            adaptive_candidates: 自适应候选节点数
        """
        device = edge_index.device
        
        # 1. 问题复杂度分析
        complexity_features = self.complexity_analyzer.analyze_question_complexity(question)
        complexity_tensor = torch.tensor([
            complexity_features['word_count'] / 50.0,
            complexity_features['lexical_diversity'],
            complexity_features['avg_sentence_length'] / 30.0,
            complexity_features['multi_hop_score'] / 3.0,
            complexity_features['temporal_score'] / 2.0,
            complexity_features['comparative_score'] / 2.0,
            complexity_features['aggregation_score'] / 2.0,
            complexity_features['negation_score'] / 2.0,
            complexity_features['entity_density'] * 2.0,
            complexity_features['wh_count'] / 3.0
        ], device=device, dtype=torch.float)
        
        # 限制在[0,1]范围
        complexity_tensor = torch.clamp(complexity_tensor, 0, 1)
        
        # 2. 图结构特征
        graph_features = self.compute_graph_features(edge_index, num_nodes, seeds)
        
        # 3. 合并特征
        combined_features = torch.cat([complexity_tensor, graph_features], dim=0)
        
        # 4. 预测调节因子
        adjustment_factor = self.size_predictor(combined_features).item()
        
        # 5. 计算自适应候选节点数
        # 基于调节因子在min和max之间插值
        adaptive_candidates = int(
            self.min_candidates + 
            adjustment_factor * (self.max_candidates - self.min_candidates)
        )
        
        return adaptive_candidates
    
    def get_complexity_stats(self, question: str) -> Dict[str, float]:
        """获取问题复杂度统计信息"""
        return self.complexity_analyzer.analyze_question_complexity(question)


class SubgraphSizeCache:
    """候选子图大小缓存"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get_cache_key(self, question: str, num_seeds: int, num_nodes: int) -> str:
        """生成缓存键"""
        import hashlib
        key_str = f"{question}_{num_seeds}_{num_nodes}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, question: str, num_seeds: int, num_nodes: int) -> Optional[int]:
        """获取缓存的候选节点数"""
        key = self.get_cache_key(question, num_seeds, num_nodes)
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def put(self, question: str, num_seeds: int, num_nodes: int, candidates: int):
        """缓存候选节点数"""
        if len(self.cache) >= self.max_size:
            # LRU淘汰
            lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            del self.cache[lru_key]
            del self.access_count[lru_key]
        
        key = self.get_cache_key(question, num_seeds, num_nodes)
        self.cache[key] = candidates
        self.access_count[key] = 1
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_count.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        return {
            'cache_size': len(self.cache),
            'total_accesses': sum(self.access_count.values())
        }


# 全局缓存实例
subgraph_size_cache = SubgraphSizeCache()


def get_adaptive_candidates(question: str,
                          edge_index: torch.Tensor,
                          num_nodes: int,
                          seeds: List[int],
                          sizer: AdaptiveSubgraphSizer,
                          use_cache: bool = True) -> int:
    """
    获取自适应候选节点数（带缓存）
    
    Args:
        question: 问题文本
        edge_index: 边索引
        num_nodes: 节点总数
        seeds: 种子节点列表
        sizer: 自适应大小调节器
        use_cache: 是否使用缓存
        
    Returns:
        candidates: 候选节点数
    """
    if use_cache:
        cached_result = subgraph_size_cache.get(question, len(seeds), num_nodes)
        if cached_result is not None:
            return cached_result
    
    # 计算自适应候选节点数
    candidates = sizer(question, edge_index, num_nodes, seeds)
    
    if use_cache:
        subgraph_size_cache.put(question, len(seeds), num_nodes, candidates)
    
    return candidates


def get_subgraph_size_cache_stats() -> Dict[str, int]:
    """获取候选子图大小缓存统计信息"""
    return subgraph_size_cache.get_stats()


def clear_subgraph_size_cache():
    """清空候选子图大小缓存"""
    subgraph_size_cache.clear()