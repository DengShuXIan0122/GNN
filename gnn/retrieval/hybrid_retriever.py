import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import networkx as nx
from collections import defaultdict
import sys
import os

# 添加根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from ppr_dde import PersonalizedPageRank
from bmssp import BMSSPSolver

# 添加自适应候选子图大小模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from adaptive_subgraph import AdaptiveSubgraphSizer, get_adaptive_candidates


class HybridRetriever(nn.Module):
    """
    混合检索器：融合APPR结构先验与语义相似度
    实现GNN-RAG³的第2层：Hybrid Retrieval (Embedding + APPR)
    """
    
    def __init__(self, 
                 input_dim: int = 768,
                 hidden_dim: int = 256,
                 appr_alpha: float = 0.85,
                 appr_tol: float = 1e-6,
                 max_candidates: int = 1200,
                 use_adaptive_size: bool = True,
                 min_candidates: int = 300,
                 max_adaptive_candidates: int = 2000):
        """
        Args:
            input_dim: 问题嵌入维度
            hidden_dim: 隐层维度
            appr_alpha: APPR重启系数
            appr_tol: APPR收敛容忍度
            max_candidates: 最大候选节点数（非自适应模式）
            use_adaptive_size: 是否使用自适应候选子图大小
            min_candidates: 自适应模式最小候选节点数
            max_adaptive_candidates: 自适应模式最大候选节点数
        """
        super(HybridRetriever, self).__init__()
        
        self.appr_alpha = appr_alpha
        self.appr_tol = appr_tol
        self.max_candidates = max_candidates
        self.use_adaptive_size = use_adaptive_size
        
        # 自适应候选子图大小调节器
        if use_adaptive_size:
            self.adaptive_sizer = AdaptiveSubgraphSizer(
                min_candidates=min_candidates,
                max_candidates=max_adaptive_candidates,
                base_candidates=max_candidates
            )
        else:
            self.adaptive_sizer = None
        
        # 可学习融合权重
        self.fusion_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # PPR计算器
        self.ppr_computer = PersonalizedPageRank(
            alpha=appr_alpha, 
            tol=appr_tol, 
            max_iter=100
        )
        
    def compute_appr(self, 
                     edge_index: torch.Tensor, 
                     num_nodes: int,
                     seeds: List[int], 
                     edge_weights: Optional[torch.Tensor] = None) -> Dict[int, float]:
        """
        计算Approximate Personalized PageRank
        
        Args:
            edge_index: 边索引 [2, num_edges]
            num_nodes: 节点总数
            seeds: 种子节点列表 (T_q)
            edge_weights: 边权重 [num_edges]
            
        Returns:
            appr_scores: 节点ID -> APPR分数的字典
        """
        # 使用已有的PPR实现
        ppr_tensor = self.ppr_computer.compute_ppr(
            edge_index=edge_index,
            num_nodes=num_nodes,
            topic_nodes=seeds,
            edge_weights=edge_weights,
            apply_degree_penalty=True
        )
        
        # 转换为字典格式，只保留非零分数
        appr_scores = {}
        for node_id in range(num_nodes):
            score = ppr_tensor[node_id].item()
            if score > self.appr_tol:
                appr_scores[node_id] = score
                
        return appr_scores

    # --- 新增：z-score 门控与融合接口 ---
    def _zscore(self, scores: Dict[int, float]) -> Dict[int, float]:
        if not scores:
            return {}
        vals = np.array(list(scores.values()), dtype=np.float32)
        m = float(vals.mean())
        s = float(vals.std()) + 1e-8
        return {nid: (float(v) - m) / s for nid, v in scores.items()}

    def fuse_scores(self,
                    z_q: torch.Tensor,
                    sim_scores: Dict[int, float],
                    appr_scores: Dict[int, float],
                    gate_negative: bool = True,
                    normalize: bool = False) -> Dict[int, float]:
        """
        融合 APPR 与 语义分数，支持 z-score 门控。
        返回未归一化的融合分数（用于 logits 损失域）。
        """
        alpha_q = self.fusion_net(z_q).squeeze()  # 标量权重
        z_sim = self._zscore(sim_scores)
        z_appr = self._zscore(appr_scores)
        all_nodes = set(sim_scores.keys()) | set(appr_scores.keys())
        fused: Dict[int, float] = {}
        for nid in all_nodes:
            s = z_sim.get(nid, -1e9 if gate_negative else 0.0)
            a = z_appr.get(nid, -1e9 if gate_negative else 0.0)
            s_pos = max(0.0, float(s))
            a_pos = max(0.0, float(a))
            val = alpha_q * a_pos + (1.0 - alpha_q) * s_pos
            fused[nid] = val.item() if torch.is_tensor(val) else float(val)
        if normalize and len(fused) > 0:
            arr = np.array(list(fused.values()), dtype=np.float32)
            mi, ma = float(arr.min()), float(arr.max())
            rng = ma - mi if ma > mi else 1.0
            fused = {nid: (score - mi) / rng for nid, score in fused.items()}
        return fused
    
    def hybrid_rank(self, 
                    z_q: torch.Tensor,
                    sim_scores: Dict[int, float],
                    appr_scores: Dict[int, float],
                    adaptive_candidates: Optional[int] = None,
                    use_zscore_gate: bool = True) -> List[int]:
        """
        混合排序：可学习融合语义相似度与结构相关度
        
        Args:
            z_q: 问题嵌入 [hidden_dim]
            sim_scores: 节点ID -> 语义相似度分数
            appr_scores: 节点ID -> APPR分数
            adaptive_candidates: 自适应候选节点数（可选）
            use_zscore_gate: 是否在融合前进行z-score门控
            
        Returns:
            cand_nodes: 排序后的候选节点列表
        """
        # 计算可学习融合权重
        alpha_q = self.fusion_net(z_q).squeeze()  # 标量权重
        
        # 获取所有候选节点
        all_nodes = set(sim_scores.keys()) | set(appr_scores.keys())
        
        # 计算融合分数
        fused_scores = {}
        if use_zscore_gate:
            fused_scores = self.fuse_scores(z_q, sim_scores, appr_scores, gate_negative=True, normalize=False)
        else:
            for node_id in all_nodes:
                sim_score = max(0.0, sim_scores.get(node_id, 0.0))
                appr_score = max(0.0, appr_scores.get(node_id, 0.0))
                fused_score = alpha_q * appr_score + (1 - alpha_q) * sim_score
                fused_scores[node_id] = fused_score.item() if torch.is_tensor(fused_score) else fused_score
        
        # 确定候选节点数量
        num_candidates = adaptive_candidates if adaptive_candidates is not None else self.max_candidates
        
        # 按分数排序，取前N个
        sorted_nodes = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        cand_nodes = [node_id for node_id, _ in sorted_nodes[:num_candidates]]
        
        return cand_nodes
    
    def build_candidate_subgraph(self, 
                                 edge_index: torch.Tensor,
                                 edge_attr: Optional[torch.Tensor],
                                 cand_nodes: List[int],
                                 num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int]]:
        """
        构建候选子图
        
        Args:
            edge_index: 原图边索引 [2, num_edges]
            edge_attr: 边属性 [num_edges, attr_dim]
            cand_nodes: 候选节点列表
            num_nodes: 原图节点总数
            
        Returns:
            sub_edge_index: 子图边索引 [2, sub_num_edges]
            sub_edge_attr: 子图边属性 [sub_num_edges, attr_dim]
            node_mapping: 原节点ID -> 子图节点ID的映射
        """
        device = edge_index.device
        cand_set = set(cand_nodes)
        
        # 创建节点映射
        node_mapping = {old_id: new_id for new_id, old_id in enumerate(cand_nodes)}
        
        # 向量化边筛选：使用候选节点掩码
        max_node_id = max(max(cand_nodes), edge_index.max().item()) + 1
        cand_mask = torch.zeros(max_node_id, dtype=torch.bool, device=device)
        cand_indices = torch.tensor(cand_nodes, device=device, dtype=torch.long)
        cand_mask[cand_indices] = True
        
        # 向量化检查边的两端是否都在候选节点中
        src_in_cand = cand_mask[edge_index[0]]  # [num_edges]
        dst_in_cand = cand_mask[edge_index[1]]  # [num_edges]
        mask = src_in_cand & dst_in_cand  # [num_edges]
        
        # 提取子图边
        sub_edge_index = edge_index[:, mask]
        if edge_attr is not None:
            sub_edge_attr = edge_attr[mask]
        else:
            sub_edge_attr = None
        
        # 向量化重新映射节点ID
        if sub_edge_index.size(1) > 0:
            # 创建节点映射张量
            node_mapping_tensor = torch.full((max_node_id,), -1, device=device, dtype=torch.long)
            for new_id, old_id in enumerate(cand_nodes):
                node_mapping_tensor[old_id] = new_id
            
            # 批量重新映射
            sub_edge_index[0] = node_mapping_tensor[sub_edge_index[0]]
            sub_edge_index[1] = node_mapping_tensor[sub_edge_index[1]]
        
        return sub_edge_index, sub_edge_attr, node_mapping
    
    def forward(self, 
                edge_index: torch.Tensor,
                num_nodes: int,
                z_q: torch.Tensor,
                seeds: List[int],
                sim_scores: Dict[int, float],
                edge_weights: Optional[torch.Tensor] = None,
                edge_attr: Optional[torch.Tensor] = None,
                question: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor, List[int], Dict[int, int]]:
        """
        前向传播：执行混合检索
        
        Args:
            edge_index: 边索引 [2, num_edges]
            num_nodes: 节点总数
            z_q: 问题嵌入 [hidden_dim]
            seeds: 种子节点列表
            sim_scores: 语义相似度分数字典
            edge_weights: 边权重 [num_edges]
            edge_attr: 边属性 [num_edges, attr_dim]
            question: 问题文本（用于自适应大小调节）
            
        Returns:
            sub_edge_index: 候选子图边索引
            sub_edge_attr: 候选子图边属性
            cand_nodes: 候选节点列表
            node_mapping: 节点映射字典
        """
        # 1. 计算APPR分数
        appr_scores = self.compute_appr(edge_index, num_nodes, seeds, edge_weights)
        
        # 2. 自适应候选节点数量计算
        adaptive_candidates = None
        if self.use_adaptive_size and self.adaptive_sizer is not None and question is not None:
            adaptive_candidates = get_adaptive_candidates(
                question, edge_index, num_nodes, seeds, self.adaptive_sizer
            )
        
        # 3. 混合排序
        cand_nodes = self.hybrid_rank(z_q, sim_scores, appr_scores, adaptive_candidates)
        
        # 4. 构建候选子图
        sub_edge_index, sub_edge_attr, node_mapping = self.build_candidate_subgraph(
            edge_index, edge_attr, cand_nodes, num_nodes
        )
        
        return sub_edge_index, sub_edge_attr, cand_nodes, node_mapping
    
    def get_complexity_stats(self, question: str) -> Dict[str, float]:
        """
        获取问题复杂度统计信息
        
        Args:
            question: 问题文本
            
        Returns:
            complexity_stats: 复杂度统计字典
        """
        if self.adaptive_sizer is not None:
            return self.adaptive_sizer.get_complexity_stats(question)
        else:
            return {}
    
    def get_adaptive_info(self, 
                         question: str,
                         edge_index: torch.Tensor,
                         num_nodes: int,
                         seeds: List[int]) -> Dict[str, Union[int, float, Dict]]:
        """
        获取自适应调节的详细信息
        
        Args:
            question: 问题文本
            edge_index: 边索引
            num_nodes: 节点总数
            seeds: 种子节点列表
            
        Returns:
            adaptive_info: 自适应信息字典
        """
        if not self.use_adaptive_size or self.adaptive_sizer is None:
            return {
                'adaptive_enabled': False,
                'candidates': self.max_candidates
            }
        
        adaptive_candidates = get_adaptive_candidates(
            question, edge_index, num_nodes, seeds, self.adaptive_sizer
        )
        
        complexity_stats = self.get_complexity_stats(question)
        
        return {
            'adaptive_enabled': True,
            'candidates': adaptive_candidates,
            'base_candidates': self.max_candidates,
            'complexity_stats': complexity_stats,
            'num_seeds': len(seeds),
            'num_nodes': num_nodes
        }


def compute_semantic_similarity(z_q: torch.Tensor, 
                               node_embeddings: torch.Tensor,
                               node_ids: List[int]) -> Dict[int, float]:
    """
    计算语义相似度（辅助函数）
    
    Args:
        z_q: 问题嵌入 [hidden_dim]
        node_embeddings: 节点嵌入 [num_nodes, hidden_dim]
        node_ids: 需要计算相似度的节点ID列表
        
    Returns:
        sim_scores: 节点ID -> 相似度分数的字典
    """
    sim_scores = {}
    z_q_norm = F.normalize(z_q.unsqueeze(0), p=2, dim=1)
    
    for node_id in node_ids:
        node_emb = node_embeddings[node_id].unsqueeze(0)
        node_emb_norm = F.normalize(node_emb, p=2, dim=1)
        
        # 余弦相似度
        similarity = torch.mm(z_q_norm, node_emb_norm.t()).squeeze().item()
        sim_scores[node_id] = max(0.0, similarity)  # 确保非负
    
    return sim_scores