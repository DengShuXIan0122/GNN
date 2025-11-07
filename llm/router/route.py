"""
GNN-RAG+Route: 智能路由机制
根据问题复杂度和类型选择最优的推理路径
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class RouteType(Enum):
    """路由类型枚举"""
    GNN = "gnn"                      # GNN推理路径
    LONG_CONTEXT = "long_context"    # 长上下文路径

@dataclass
class RouteDecision:
    """路由决策结果"""
    route_type: RouteType
    confidence: float
    use_appr: bool
    use_dde: bool
    use_pcst: bool

def route_question(question: str, answers_gnn: list, triples_long: list, k: int = 100) -> RouteDecision:
    """
    简化的路由决策函数
    
    Args:
        question: 输入问题
        answers_gnn: GNN推理得到的答案列表
        triples_long: 长上下文中的三元组列表
        k: 检查的三元组数量限制
        
    Returns:
        RouteDecision: 路由决策结果
    """
    # 若 gnn 答案存在不在长上下文 triples_long 的实体 → 走 GNN
    # 提取长上下文中的所有实体
    long_context_entities = set()
    for triple in triples_long[:k]:  # 限制检查的三元组数量
        if isinstance(triple, (list, tuple)) and len(triple) >= 3:
            long_context_entities.add(triple[0])  # 主语
            long_context_entities.add(triple[2])  # 宾语
        elif isinstance(triple, str):
            # 如果是字符串格式，尝试解析
            parts = triple.split()
            if len(parts) >= 3:
                long_context_entities.add(parts[0])
                long_context_entities.add(parts[2])
    
    # 检查GNN答案是否都在长上下文中
    covered = all(str(answer) in long_context_entities for answer in answers_gnn)
    
    if not covered:
        return RouteDecision(RouteType.GNN, 0.8, True, True, True)
    return RouteDecision(RouteType.LONG_CONTEXT, 0.7, False, False, False)


# ==================== 原有的复杂路由系统 ====================

class ComplexRouteType(Enum):
    """复杂路由类型枚举（保留原有功能）"""
    DIRECT_KG = "direct_kg"           # 直接KG查询
    SINGLE_HOP = "single_hop"         # 单跳推理
    MULTI_HOP = "multi_hop"          # 多跳推理
    COMPLEX_REASONING = "complex"     # 复杂推理
    HYBRID = "hybrid"                # 混合推理

@dataclass
class ComplexRouteDecision:
    """复杂路由决策结果（保留原有功能）"""
    route_type: ComplexRouteType
    confidence: float
    reasoning_steps: int
    use_appr: bool
    use_dde: bool
    use_pcst: bool
    max_hops: int
    beam_size: int
    explanation: str

class QuestionComplexityAnalyzer:
    """问题复杂度分析器"""
    
    def __init__(self):
        # 复杂度指标关键词
        self.multi_hop_keywords = [
            "how many", "what is the", "which", "who is", "when did",
            "where is", "why", "what are", "list", "name", "find"
        ]
        
        self.complex_keywords = [
            "compare", "difference", "similar", "relationship", "between",
            "both", "either", "neither", "all", "most", "least", "best", "worst"
        ]
        
        self.temporal_keywords = [
            "before", "after", "during", "since", "until", "when", "while",
            "first", "last", "latest", "earliest", "recent", "old", "new"
        ]
        
        self.aggregation_keywords = [
            "total", "sum", "count", "average", "maximum", "minimum",
            "how many", "number of", "amount of"
        ]
    
    def analyze_question(self, question: str) -> Dict[str, float]:
        """
        分析问题复杂度
        
        Args:
            question: 输入问题
            
        Returns:
            复杂度分析结果
        """
        question_lower = question.lower()
        
        # 基础特征
        word_count = len(question.split())
        entity_count = self._count_entities(question)
        
        # 关键词匹配
        multi_hop_score = self._keyword_match_score(question_lower, self.multi_hop_keywords)
        complex_score = self._keyword_match_score(question_lower, self.complex_keywords)
        temporal_score = self._keyword_match_score(question_lower, self.temporal_keywords)
        aggregation_score = self._keyword_match_score(question_lower, self.aggregation_keywords)
        
        # 语法复杂度
        syntax_complexity = self._analyze_syntax_complexity(question)
        
        return {
            "word_count": word_count,
            "entity_count": entity_count,
            "multi_hop_score": multi_hop_score,
            "complex_score": complex_score,
            "temporal_score": temporal_score,
            "aggregation_score": aggregation_score,
            "syntax_complexity": syntax_complexity,
            "overall_complexity": self._compute_overall_complexity(
                word_count, entity_count, multi_hop_score, complex_score,
                temporal_score, aggregation_score, syntax_complexity
            )
        }
    
    def _count_entities(self, question: str) -> int:
        """估算问题中的实体数量"""
        # 简化实现：大写开头的词作为潜在实体
        words = question.split()
        entity_count = sum(1 for word in words if word[0].isupper() and len(word) > 1)
        return max(entity_count, 1)
    
    def _keyword_match_score(self, question: str, keywords: List[str]) -> float:
        """计算关键词匹配分数"""
        matches = sum(1 for keyword in keywords if keyword in question)
        return min(matches / len(keywords), 1.0)
    
    def _analyze_syntax_complexity(self, question: str) -> float:
        """分析语法复杂度"""
        # 简化实现：基于标点符号和连接词
        complexity_indicators = [",", ";", "and", "or", "but", "however", "therefore"]
        score = sum(1 for indicator in complexity_indicators if indicator in question.lower())
        return min(score / 5.0, 1.0)
    
    def _compute_overall_complexity(self, word_count: int, entity_count: int,
                                  multi_hop: float, complex: float,
                                  temporal: float, aggregation: float,
                                  syntax: float) -> float:
        """计算总体复杂度分数"""
        # 加权组合各个因子
        word_factor = min(word_count / 20.0, 1.0)  # 标准化到[0,1]
        entity_factor = min(entity_count / 5.0, 1.0)
        
        complexity_score = (
            0.2 * word_factor +
            0.15 * entity_factor +
            0.25 * multi_hop +
            0.2 * complex +
            0.1 * temporal +
            0.05 * aggregation +
            0.05 * syntax
        )
        
        return min(complexity_score, 1.0)

class GNNRAGRouter(nn.Module):
    """GNN-RAG智能路由器"""
    
    def __init__(self, 
                 hidden_dim: int = 256,
                 num_route_types: int = 5,
                 complexity_threshold_low: float = 0.3,
                 complexity_threshold_high: float = 0.7):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_route_types = num_route_types
        self.complexity_threshold_low = complexity_threshold_low
        self.complexity_threshold_high = complexity_threshold_high
        
        # 复杂度分析器
        self.complexity_analyzer = QuestionComplexityAnalyzer()
        
        # 特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Linear(7, hidden_dim),  # 7个复杂度特征
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # 路由分类器
        self.route_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, num_route_types)
        )
        
        # 置信度预测器
        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, complexity_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            complexity_features: 复杂度特征 [batch_size, 7]
            
        Returns:
            route_logits: 路由类型logits [batch_size, num_route_types]
            confidence: 置信度分数 [batch_size, 1]
        """
        # 特征编码
        encoded_features = self.feature_encoder(complexity_features)
        
        # 路由分类
        route_logits = self.route_classifier(encoded_features)
        
        # 置信度预测
        confidence = self.confidence_predictor(encoded_features)
        
        return route_logits, confidence
    
    def route_question(self, question: str, context: Optional[Dict] = None) -> ComplexRouteDecision:
        """
        为问题选择最优路由
        
        Args:
            question: 输入问题
            context: 可选的上下文信息
            
        Returns:
            路由决策结果
        """
        # 分析问题复杂度
        complexity_analysis = self.complexity_analyzer.analyze_question(question)
        
        # 构建特征向量
        features = torch.tensor([
            complexity_analysis["word_count"] / 20.0,  # 标准化
            complexity_analysis["entity_count"] / 5.0,
            complexity_analysis["multi_hop_score"],
            complexity_analysis["complex_score"],
            complexity_analysis["temporal_score"],
            complexity_analysis["aggregation_score"],
            complexity_analysis["syntax_complexity"]
        ], dtype=torch.float32).unsqueeze(0)
        
        # 模型预测
        with torch.no_grad():
            route_logits, confidence = self.forward(features)
            route_probs = F.softmax(route_logits, dim=-1)
            predicted_route_idx = torch.argmax(route_probs, dim=-1).item()
            confidence_score = confidence.item()
        
        # 基于规则的后处理
        route_decision = self._post_process_route_decision(
            predicted_route_idx, confidence_score, complexity_analysis, question
        )
        
        return route_decision
    
    def _post_process_route_decision(self, 
                                   predicted_route_idx: int,
                                   confidence_score: float,
                                   complexity_analysis: Dict[str, float],
                                   question: str) -> ComplexRouteDecision:
        """后处理路由决策"""
        
        route_types = list(ComplexRouteType)
        predicted_route = route_types[predicted_route_idx]
        overall_complexity = complexity_analysis["overall_complexity"]
        
        # 基于复杂度的规则调整
        if overall_complexity < self.complexity_threshold_low:
            # 低复杂度：倾向于简单路由
            if predicted_route in [ComplexRouteType.COMPLEX_REASONING, ComplexRouteType.HYBRID]:
                predicted_route = ComplexRouteType.SINGLE_HOP
                confidence_score *= 0.9  # 降低置信度
        
        elif overall_complexity > self.complexity_threshold_high:
            # 高复杂度：倾向于复杂路由
            if predicted_route == ComplexRouteType.DIRECT_KG:
                predicted_route = ComplexRouteType.MULTI_HOP
                confidence_score *= 0.8
        
        # 根据路由类型设置参数
        route_config = self._get_route_config(predicted_route, complexity_analysis)
        
        return ComplexRouteDecision(
            route_type=predicted_route,
            confidence=confidence_score,
            reasoning_steps=route_config["reasoning_steps"],
            use_appr=route_config["use_appr"],
            use_dde=route_config["use_dde"],
            use_pcst=route_config["use_pcst"],
            max_hops=route_config["max_hops"],
            beam_size=route_config["beam_size"],
            explanation=route_config["explanation"]
        )
    
    def _get_route_config(self, route_type: ComplexRouteType, complexity_analysis: Dict[str, float]) -> Dict:
        """获取路由配置"""
        
        base_configs = {
            ComplexRouteType.DIRECT_KG: {
                "reasoning_steps": 1,
                "use_appr": False,
                "use_dde": False,
                "use_pcst": False,
                "max_hops": 1,
                "beam_size": 1,
                "explanation": "直接KG查询，适用于简单事实性问题"
            },
            ComplexRouteType.SINGLE_HOP: {
                "reasoning_steps": 2,
                "use_appr": True,
                "use_dde": False,
                "use_pcst": False,
                "max_hops": 2,
                "beam_size": 3,
                "explanation": "单跳推理，使用APPR增强检索"
            },
            ComplexRouteType.MULTI_HOP: {
                "reasoning_steps": 4,
                "use_appr": True,
                "use_dde": True,
                "use_pcst": False,
                "max_hops": 4,
                "beam_size": 5,
                "explanation": "多跳推理，使用APPR+DDE增强"
            },
            ComplexRouteType.COMPLEX_REASONING: {
                "reasoning_steps": 6,
                "use_appr": True,
                "use_dde": True,
                "use_pcst": True,
                "max_hops": 6,
                "beam_size": 8,
                "explanation": "复杂推理，启用完整GNN-RAG³流程"
            },
            ComplexRouteType.HYBRID: {
                "reasoning_steps": 5,
                "use_appr": True,
                "use_dde": True,
                "use_pcst": True,
                "max_hops": 5,
                "beam_size": 6,
                "explanation": "混合推理，平衡效率和准确性"
            }
        }
        
        config = base_configs[route_type].copy()
        
        # 根据复杂度动态调整参数
        if complexity_analysis["aggregation_score"] > 0.5:
            config["beam_size"] = min(config["beam_size"] + 2, 10)
        
        if complexity_analysis["temporal_score"] > 0.5:
            config["max_hops"] = min(config["max_hops"] + 1, 8)
        
        return config

class RouteOptimizer:
    """路由优化器，基于历史性能优化路由决策"""
    
    def __init__(self, history_file: str = "route_history.json"):
        self.history_file = history_file
        self.performance_history = self._load_history()
    
    def _load_history(self) -> Dict:
        """加载历史性能数据"""
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"routes": {}, "questions": []}
    
    def _save_history(self):
        """保存历史性能数据"""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.performance_history, f, ensure_ascii=False, indent=2)
    
    def record_performance(self, question: str, route_decision: ComplexRouteDecision, 
                         performance_metrics: Dict[str, float]):
        """记录路由性能"""
        route_key = route_decision.route_type.value
        
        if route_key not in self.performance_history["routes"]:
            self.performance_history["routes"][route_key] = {
                "count": 0,
                "total_accuracy": 0.0,
                "total_latency": 0.0,
                "total_tokens": 0.0
            }
        
        route_stats = self.performance_history["routes"][route_key]
        route_stats["count"] += 1
        route_stats["total_accuracy"] += performance_metrics.get("accuracy", 0.0)
        route_stats["total_latency"] += performance_metrics.get("latency", 0.0)
        route_stats["total_tokens"] += performance_metrics.get("tokens", 0.0)
        
        # 记录问题级别的信息
        self.performance_history["questions"].append({
            "question": question,
            "route": route_key,
            "confidence": route_decision.confidence,
            "metrics": performance_metrics
        })
        
        # 保持最近1000条记录
        if len(self.performance_history["questions"]) > 1000:
            self.performance_history["questions"] = self.performance_history["questions"][-1000:]
        
        self._save_history()
    
    def get_route_performance(self, route_type: ComplexRouteType) -> Dict[str, float]:
        """获取路由性能统计"""
        route_key = route_type.value
        
        if route_key not in self.performance_history["routes"]:
            return {"accuracy": 0.0, "latency": 0.0, "tokens": 0.0, "count": 0}
        
        stats = self.performance_history["routes"][route_key]
        count = stats["count"]
        
        if count == 0:
            return {"accuracy": 0.0, "latency": 0.0, "tokens": 0.0, "count": 0}
        
        return {
            "accuracy": stats["total_accuracy"] / count,
            "latency": stats["total_latency"] / count,
            "tokens": stats["total_tokens"] / count,
            "count": count
        }
    
    def optimize_route_decision(self, original_decision: ComplexRouteDecision, 
                              question: str) -> ComplexRouteDecision:
        """基于历史性能优化路由决策"""
        
        # 获取所有路由的性能
        route_performances = {}
        for route_type in ComplexRouteType:
            route_performances[route_type] = self.get_route_performance(route_type)
        
        # 如果原始路由性能较差，考虑切换
        original_perf = route_performances[original_decision.route_type]
        
        if original_perf["count"] > 10 and original_perf["accuracy"] < 0.6:
            # 寻找性能更好的替代路由
            best_route = original_decision.route_type
            best_score = original_perf["accuracy"]
            
            for route_type, perf in route_performances.items():
                if perf["count"] > 5:  # 有足够的历史数据
                    # 综合考虑准确性和效率
                    score = perf["accuracy"] - 0.1 * (perf["latency"] / 1000.0)  # 延迟惩罚
                    if score > best_score:
                        best_route = route_type
                        best_score = score
            
            if best_route != original_decision.route_type:
                # 创建优化后的决策
                optimized_decision = ComplexRouteDecision(
                    route_type=best_route,
                    confidence=original_decision.confidence * 0.9,  # 降低置信度
                    reasoning_steps=original_decision.reasoning_steps,
                    use_appr=original_decision.use_appr,
                    use_dde=original_decision.use_dde,
                    use_pcst=original_decision.use_pcst,
                    max_hops=original_decision.max_hops,
                    beam_size=original_decision.beam_size,
                    explanation=f"基于历史性能优化：{original_decision.explanation}"
                )
                return optimized_decision
        
        return original_decision

# 全局路由器实例
_global_router = None
_global_optimizer = None

def get_router() -> GNNRAGRouter:
    """获取全局路由器实例"""
    global _global_router
    if _global_router is None:
        _global_router = GNNRAGRouter()
    return _global_router

def get_optimizer() -> RouteOptimizer:
    """获取全局优化器实例"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = RouteOptimizer()
    return _global_optimizer

def route_question_complex(question: str, context: Optional[Dict] = None, 
                          optimize: bool = True) -> ComplexRouteDecision:
    """
    为问题选择最优路由（便捷函数）
    
    Args:
        question: 输入问题
        context: 可选的上下文信息
        optimize: 是否使用历史性能优化
        
    Returns:
        路由决策结果
    """
    router = get_router()
    decision = router.route_question(question, context)
    
    if optimize:
        optimizer = get_optimizer()
        decision = optimizer.optimize_route_decision(decision, question)
    
    return decision

if __name__ == "__main__":
    # 测试路由器
    test_questions = [
        "What is the capital of France?",  # 简单事实
        "Who directed the movie that won the Oscar for Best Picture in 2020?",  # 单跳
        "What movies did the director of Inception also direct?",  # 多跳
        "Compare the box office performance of Marvel and DC movies in the last decade",  # 复杂
        "List all actors who have worked with both Christopher Nolan and Martin Scorsese"  # 混合
    ]
    
    router = get_router()
    
    print("=== GNN-RAG+Route 测试 ===")
    for i, question in enumerate(test_questions, 1):
        print(f"\n问题 {i}: {question}")
        decision = route_question_complex(question)
        print(f"路由类型: {decision.route_type.value}")
        print(f"置信度: {decision.confidence:.3f}")
        print(f"推理步数: {decision.reasoning_steps}")
        print(f"配置: APPR={decision.use_appr}, DDE={decision.use_dde}, PCST={decision.use_pcst}")
        print(f"说明: {decision.explanation}")