import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np
import sys
import os

# 添加根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from pcst import PCSTSolver, PCSTNode, PCSTEdge


class PCSTLoss(nn.Module):
    """
    PCST损失函数模块
    实现GNN-RAG³的第4层：PCST-regularized Evidence Graph
    包含蒸馏损失和软正则约束
    """
    
    def __init__(self, 
                 lambda_cost: float = 0.1,
                 lambda_conn: float = 0.1, 
                 lambda_sparse: float = 0.05,
                 temperature: float = 2.0,
                 min_temperature: float = 0.5):
        """
        Args:
            lambda_cost: 成本权重
            lambda_conn: 连通性权重
            lambda_sparse: 稀疏性权重
            temperature: Gumbel-Sigmoid初始温度
            min_temperature: 最小温度
        """
        super(PCSTLoss, self).__init__()
        
        self.lambda_cost = lambda_cost
        self.lambda_conn = lambda_conn
        self.lambda_sparse = lambda_sparse
        self.temperature = temperature
        self.min_temperature = min_temperature
        
        # PCST求解器
        self.pcst_solver = PCSTSolver()
        
        # 边选择头
        self.edge_selector = nn.Sequential(
            nn.Linear(128, 64),  # 假设边特征维度为128
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def pcst_soft_regularizer(self, 
                             pi_edge: torch.Tensor,
                             edge_cost: torch.Tensor,
                             laplacian: torch.Tensor,
                             temperature: Optional[float] = None) -> torch.Tensor:
        """
        PCST软正则化损失
        
        Args:
            pi_edge: 边选择概率 [num_edges]
            edge_cost: 边成本 [num_edges]
            laplacian: 图拉普拉斯矩阵 [num_nodes, num_nodes]
            temperature: 当前温度
            
        Returns:
            pcst_loss: PCST正则化损失
        """
        if temperature is None:
            temperature = self.temperature
        
        # 应用温度缩放
        pi_edge_temp = torch.sigmoid(torch.logit(pi_edge) / temperature)
        
        # 成本项：λ1 * (cost · π)
        cost_term = self.lambda_cost * torch.sum(edge_cost * pi_edge_temp)
        
        # 连通性项：λ2 * π^T * L * π (近似)
        # 这里简化为边选择的方差惩罚
        conn_term = self.lambda_conn * torch.var(pi_edge_temp)
        
        # 稀疏性项：λ3 * ||π||_1
        sparse_term = self.lambda_sparse * torch.sum(pi_edge_temp)
        
        total_loss = cost_term + conn_term + sparse_term
        
        return total_loss
    
    def pcst_distill_bce(self, 
                        pred_edge_mask: torch.Tensor,
                        target_edge_mask: torch.Tensor) -> torch.Tensor:
        """
        PCST蒸馏二元交叉熵损失
        
        Args:
            pred_edge_mask: 预测的边选择概率 [num_edges]
            target_edge_mask: 目标边掩码（来自PCST近似解） [num_edges]
            
        Returns:
            distill_loss: 蒸馏损失
        """
        return F.binary_cross_entropy(pred_edge_mask, target_edge_mask.float())
    
    def compute_pcst_target(self, 
                           node_scores: torch.Tensor,
                           edge_index: torch.Tensor,
                           edge_costs: torch.Tensor,
                           appr_scores: Optional[torch.Tensor] = None,
                           alpha: float = 0.5,
                           use_fast_heuristic: bool = True) -> torch.Tensor:
        """
        计算PCST目标掩码（用于蒸馏）
        
        Args:
            node_scores: 节点重要性分数 [num_nodes]
            edge_index: 边索引 [2, num_edges]
            edge_costs: 边成本 [num_edges]
            appr_scores: APPR分数 [num_nodes]
            alpha: APPR与节点分数的融合权重
            use_fast_heuristic: 是否使用快速启发式方法
            
        Returns:
            target_mask: 目标边掩码 [num_edges]
        """
        device = node_scores.device
        num_nodes = node_scores.size(0)
        num_edges = edge_index.size(1)
        
        # 计算节点奖励
        if appr_scores is not None:
            node_prizes = alpha * appr_scores + (1 - alpha) * node_scores
        else:
            node_prizes = node_scores
        
        # 优先使用快速启发式方法，避免CPU回退
        if use_fast_heuristic or num_edges > 1000:  # 大图时强制使用启发式
            return self._fast_heuristic_selection(node_prizes, edge_index, edge_costs)
        
        # 原始PCST求解（仅用于小图）
        try:
            # 向量化转换为PCST格式
            # 转换节点奖励到CPU（批量操作）
            node_prizes_cpu = node_prizes.cpu().numpy()
            nodes = [PCSTNode(node_id=i, prize=float(node_prizes_cpu[i])) for i in range(num_nodes)]
            
            # 向量化转换边到CPU（批量操作）
            edge_index_cpu = edge_index.cpu().numpy()
            edge_costs_cpu = edge_costs.cpu().numpy()
            edges = []
            for i in range(num_edges):
                src, dst = int(edge_index_cpu[0, i]), int(edge_index_cpu[1, i])
                cost = float(edge_costs_cpu[i])
                edges.append(PCSTEdge(src=src, dst=dst, cost=cost, original_idx=i))
            
            selected_nodes, selected_edges = self.pcst_solver.solve_pcst(nodes, edges)
            
            # 创建目标掩码（向量化）
            target_mask = torch.zeros(num_edges, device=device)
            if selected_edges:
                # 批量设置选中的边
                selected_edge_indices = torch.tensor([idx for idx in selected_edges if idx < num_edges], 
                                                   device=device, dtype=torch.long)
                if len(selected_edge_indices) > 0:
                    target_mask[selected_edge_indices] = 1.0
                    
        except Exception as e:
            # 如果PCST求解失败，使用启发式方法
            print(f"PCST solving failed: {e}, using heuristic fallback")
            target_mask = self._fast_heuristic_selection(node_prizes, edge_index, edge_costs)
            
        
        return target_mask
    
    def _heuristic_edge_selection(self, 
                                 node_scores: torch.Tensor,
                                 edge_index: torch.Tensor,
                                 edge_costs: torch.Tensor,
                                 top_k: int) -> torch.Tensor:
        """
        启发式边选择（PCST求解失败时的备选方案）
        
        Args:
            node_scores: 节点分数 [num_nodes]
            edge_index: 边索引 [2, num_edges]
            edge_costs: 边成本 [num_edges]
            top_k: 选择的边数
            
        Returns:
            edge_mask: 边掩码 [num_edges]
        """
        device = node_scores.device
        num_edges = edge_index.size(1)
        
        # 计算边的重要性：(node_score[src] + node_score[dst]) / edge_cost
        # 确保索引为long类型，用于tensor索引
        src_indices = edge_index[0].long()
        dst_indices = edge_index[1].long()
        src_scores = node_scores[src_indices]
        dst_scores = node_scores[dst_indices]
        edge_importance = (src_scores + dst_scores) / (edge_costs + 1e-8)
        
        # 选择top-k边
        _, top_indices = torch.topk(edge_importance, min(top_k, num_edges))
        
        edge_mask = torch.zeros(num_edges, device=device)
        edge_mask[top_indices] = 1.0
        
        return edge_mask
    
    def compute_edge_costs(self, 
                          edge_index: torch.Tensor,
                          distances: Dict[int, int],
                          relation_costs: Optional[torch.Tensor] = None,
                          beta: float = 0.2,
                          gamma: float = 0.1) -> torch.Tensor:
        """
        计算边成本
        
        Args:
            edge_index: 边索引 [2, num_edges]
            distances: 节点到种子的距离字典
            relation_costs: 关系成本 [num_edges]
            beta: 距离权重
            gamma: 关系权重
            
        Returns:
            edge_costs: 边成本 [num_edges]
        """
        device = edge_index.device
        num_edges = edge_index.size(1)
        
        # 向量化距离成本计算
        # 将距离字典转换为张量以支持向量化操作
        max_node_id = max(max(distances.keys()) if distances else 0, 
                         edge_index.max().item()) + 1
        distance_tensor = torch.full((max_node_id,), 10.0, device=device, dtype=torch.float)
        
        # 批量设置已知距离
        if distances:
            for node_id, dist in distances.items():
                if node_id < max_node_id:
                    distance_tensor[node_id] = float(dist)
        
        # 向量化计算边的距离成本
        src_distances = distance_tensor[edge_index[0]]  # [num_edges]
        dst_distances = distance_tensor[edge_index[1]]  # [num_edges]
        dist_costs = beta * (src_distances + dst_distances) / 2.0
        
        # 关系成本
        if relation_costs is not None:
            rel_costs = gamma * relation_costs
        else:
            rel_costs = torch.zeros(num_edges, device=device)
        
        total_costs = dist_costs + rel_costs + 1e-6  # 避免零成本
        
        return total_costs
    
    def forward(self, 
                node_scores: torch.Tensor,
                edge_features: torch.Tensor,
                edge_index: torch.Tensor,
                distances: Dict[int, int],
                appr_scores: Optional[torch.Tensor] = None,
                relation_costs: Optional[torch.Tensor] = None,
                stage: str = "distill",
                temperature: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播：计算PCST损失
        
        Args:
            node_scores: 节点重要性分数 [num_nodes]
            edge_features: 边特征 [num_edges, feature_dim]
            edge_index: 边索引 [2, num_edges]
            distances: 节点距离字典
            appr_scores: APPR分数 [num_nodes]
            relation_costs: 关系成本 [num_edges]
            stage: 训练阶段 ("distill" 或 "joint")
            temperature: 当前温度
            
        Returns:
            losses: 损失字典
        """
        losses = {}
        
        # 计算边成本
        edge_costs = self.compute_edge_costs(edge_index, distances, relation_costs)
        
        # 边选择概率
        pi_edge = self.edge_selector(edge_features).squeeze()  # [num_edges]
        
        if stage == "distill":
            # 蒸馏阶段：只计算蒸馏损失
            target_mask = self.compute_pcst_target(
                node_scores, edge_index, edge_costs, appr_scores
            )
            losses["pcst_distill"] = self.pcst_distill_bce(pi_edge, target_mask)
            
        elif stage == "joint":
            # 联合阶段：计算软正则损失
            laplacian = self._compute_laplacian(edge_index, node_scores.size(0))
            losses["pcst_soft"] = self.pcst_soft_regularizer(
                pi_edge, edge_costs, laplacian, temperature
            )
            
        else:
            # 完整阶段：计算所有损失
            target_mask = self.compute_pcst_target(
                node_scores, edge_index, edge_costs, appr_scores
            )
            losses["pcst_distill"] = self.pcst_distill_bce(pi_edge, target_mask)
            
            laplacian = self._compute_laplacian(edge_index, node_scores.size(0))
            losses["pcst_soft"] = self.pcst_soft_regularizer(
                pi_edge, edge_costs, laplacian, temperature
            )
        
        # 返回边选择概率用于后续处理
        losses["edge_probs"] = pi_edge
        
        return losses
    
    def _compute_laplacian(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """计算图拉普拉斯矩阵"""
        device = edge_index.device
        
        # 度矩阵
        degree = torch.zeros(num_nodes, device=device)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i], edge_index[1, i]
            degree[src] += 1
            degree[dst] += 1
        
        # 拉普拉斯矩阵（简化版）
        laplacian = torch.diag(degree)
        
        return laplacian
    
    def update_temperature(self, current_step: int, total_steps: int):
        """更新温度（退火）"""
        progress = current_step / total_steps
        self.temperature = max(
            self.min_temperature,
            self.temperature * (1 - progress)
        )