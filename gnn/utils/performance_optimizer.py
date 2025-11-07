"""
性能优化集成模块
整合GPU-CPU同步优化、自适应子图大小和内存监控功能
"""

import torch
import time
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager
import warnings

from .gpu_cpu_sync_utils import GPUCPUSyncProfiler, gpu_cpu_sync_profiler
from .adaptive_subgraph import AdaptiveSubgraphSizer, QuestionComplexityAnalyzer
from .memory_monitor import (
    system_monitor, MemoryContext, monitor_performance,
    get_memory_stats, get_performance_stats, optimize_system
)


class GNNRAGOptimizer:
    """GNN-RAG性能优化器"""
    
    def __init__(self, 
                 enable_sync_optimization: bool = True,
                 enable_adaptive_subgraph: bool = True,
                 enable_memory_monitoring: bool = True,
                 monitoring_interval: float = 1.0):
        """
        初始化优化器
        
        Args:
            enable_sync_optimization: 是否启用GPU-CPU同步优化
            enable_adaptive_subgraph: 是否启用自适应子图大小
            enable_memory_monitoring: 是否启用内存监控
            monitoring_interval: 监控间隔（秒）
        """
        self.enable_sync_optimization = enable_sync_optimization
        self.enable_adaptive_subgraph = enable_adaptive_subgraph
        self.enable_memory_monitoring = enable_memory_monitoring
        self.monitoring_interval = monitoring_interval
        
        # 初始化组件
        self.sync_profiler = gpu_cpu_sync_profiler if enable_sync_optimization else None
        self.adaptive_sizer = AdaptiveSubgraphSizer() if enable_adaptive_subgraph else None
        self.complexity_analyzer = QuestionComplexityAnalyzer() if enable_adaptive_subgraph else None
        
        # 性能统计
        self.optimization_stats = {
            'sync_optimizations': 0,
            'adaptive_adjustments': 0,
            'memory_cleanups': 0,
            'total_time_saved': 0.0
        }
        
        # 启动监控
        if enable_memory_monitoring:
            system_monitor.start_monitoring(monitoring_interval)
    
    def optimize_retrieval(self, 
                          retriever,
                          question: str,
                          seed_nodes: torch.Tensor,
                          edge_index: torch.Tensor,
                          **kwargs) -> Dict[str, Any]:
        """
        优化检索过程
        
        Args:
            retriever: 检索器实例
            question: 问题文本
            seed_nodes: 种子节点
            edge_index: 边索引
            **kwargs: 其他参数
            
        Returns:
            优化结果和统计信息
        """
        start_time = time.time()
        optimization_info = {
            'sync_optimization': False,
            'adaptive_subgraph': False,
            'memory_optimization': False,
            'original_candidates': getattr(retriever, 'max_candidates', 1200),
            'optimized_candidates': None,
            'time_saved': 0.0
        }
        
        # 1. GPU-CPU同步优化
        if self.enable_sync_optimization and self.sync_profiler:
            with self.sync_profiler.profile_context("retrieval_optimization"):
                # 记录同步操作
                optimization_info['sync_optimization'] = True
                self.optimization_stats['sync_optimizations'] += 1
        
        # 2. 自适应子图大小优化
        if self.enable_adaptive_subgraph and self.adaptive_sizer:
            try:
                # 分析问题复杂度
                complexity_stats = self.complexity_analyzer.analyze_complexity(question)
                
                # 计算自适应候选数量
                adaptive_candidates = self.adaptive_sizer.forward(
                    question=question,
                    edge_index=edge_index,
                    num_nodes=edge_index.max().item() + 1,
                    seeds=seed_nodes.tolist() if hasattr(seed_nodes, 'tolist') else list(seed_nodes)
                )
                
                optimization_info['adaptive_subgraph'] = True
                optimization_info['optimized_candidates'] = adaptive_candidates
                optimization_info['complexity_stats'] = complexity_stats
                self.optimization_stats['adaptive_adjustments'] += 1
                
            except Exception as e:
                warnings.warn(f"自适应子图优化失败: {e}")
                optimization_info['optimized_candidates'] = optimization_info['original_candidates']
        
        # 3. 内存优化
        if self.enable_memory_monitoring:
            memory_stats = get_memory_stats()
            if memory_stats.get('cpu', {}).get('current_mb', 0) > 8000:  # 8GB阈值
                optimize_system()
                optimization_info['memory_optimization'] = True
                self.optimization_stats['memory_cleanups'] += 1
        
        # 计算优化时间
        optimization_time = time.time() - start_time
        optimization_info['optimization_time'] = optimization_time
        
        return optimization_info
    
    def get_optimization_recommendations(self) -> List[str]:
        """获取优化建议"""
        recommendations = []
        
        # 获取各组件的统计信息
        memory_stats = get_memory_stats() if self.enable_memory_monitoring else {}
        performance_stats = get_performance_stats() if self.enable_memory_monitoring else {}
        sync_stats = self.sync_profiler.get_stats() if self.sync_profiler else {}
        
        # GPU-CPU同步建议
        if sync_stats:
            total_sync_time = sum(sync_stats.get('operation_times', {}).values())
            if total_sync_time > 1.0:  # 超过1秒
                recommendations.append(
                    f"检测到GPU-CPU同步耗时{total_sync_time:.2f}秒，建议进一步向量化操作"
                )
        
        # 内存使用建议
        if memory_stats:
            cpu_memory = memory_stats.get('cpu', {}).get('current_mb', 0)
            gpu_memory = memory_stats.get('gpu', {}).get('current_mb', 0)
            
            if cpu_memory > 16000:  # 16GB
                recommendations.append("CPU内存使用过高，建议减少批处理大小或使用数据流处理")
            
            if gpu_memory > 12000:  # 12GB
                recommendations.append("GPU内存使用过高，建议使用梯度检查点或模型并行")
        
        # 自适应子图建议
        if self.enable_adaptive_subgraph:
            adj_ratio = self.optimization_stats['adaptive_adjustments']
            if adj_ratio > 0:
                recommendations.append(
                    f"已进行{adj_ratio}次自适应子图调整，有助于提高检索效率"
                )
        
        return recommendations
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """获取综合性能报告"""
        report = {
            'timestamp': time.time(),
            'optimization_enabled': {
                'sync_optimization': self.enable_sync_optimization,
                'adaptive_subgraph': self.enable_adaptive_subgraph,
                'memory_monitoring': self.enable_memory_monitoring
            },
            'optimization_stats': self.optimization_stats.copy(),
            'recommendations': self.get_optimization_recommendations()
        }
        
        # 添加各组件的详细统计
        if self.enable_memory_monitoring:
            report['memory_stats'] = get_memory_stats()
            report['performance_stats'] = get_performance_stats()
        
        if self.sync_profiler:
            report['sync_stats'] = self.sync_profiler.get_stats()
        
        return report
    
    def save_report(self, filepath: str):
        """保存性能报告"""
        import json
        report = self.get_comprehensive_report()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
    def cleanup(self):
        """清理资源"""
        if self.enable_memory_monitoring:
            system_monitor.stop_monitoring()
        
        if self.sync_profiler:
            self.sync_profiler.reset()


@contextmanager
def optimized_retrieval_context(question: str = None,
                               enable_sync_opt: bool = True,
                               enable_adaptive: bool = True,
                               enable_memory: bool = True):
    """
    优化检索上下文管理器
    
    Args:
        question: 问题文本
        enable_sync_opt: 启用同步优化
        enable_adaptive: 启用自适应子图
        enable_memory: 启用内存监控
    """
    optimizer = GNNRAGOptimizer(
        enable_sync_optimization=enable_sync_opt,
        enable_adaptive_subgraph=enable_adaptive,
        enable_memory_monitoring=enable_memory
    )
    
    try:
        yield optimizer
    finally:
        optimizer.cleanup()


class PerformanceBenchmark:
    """性能基准测试"""
    
    def __init__(self):
        self.baseline_times = {}
        self.optimized_times = {}
        self.improvement_ratios = {}
    
    @monitor_performance("benchmark_retrieval")
    def benchmark_retrieval(self, 
                           retriever,
                           questions: List[str],
                           seed_nodes_list: List[torch.Tensor],
                           edge_index: torch.Tensor,
                           use_optimization: bool = True) -> Dict[str, Any]:
        """
        基准测试检索性能
        
        Args:
            retriever: 检索器
            questions: 问题列表
            seed_nodes_list: 种子节点列表
            edge_index: 边索引
            use_optimization: 是否使用优化
            
        Returns:
            基准测试结果
        """
        total_time = 0.0
        results = []
        
        with optimized_retrieval_context(
            enable_sync_opt=use_optimization,
            enable_adaptive=use_optimization,
            enable_memory=use_optimization
        ) as optimizer:
            
            for i, (question, seed_nodes) in enumerate(zip(questions, seed_nodes_list)):
                start_time = time.time()
                
                # 执行检索
                with MemoryContext(f"retrieval_{i}"):
                    if hasattr(retriever, 'forward'):
                        # 如果检索器支持question参数
                        try:
                            result = retriever.forward(
                                seed_nodes=seed_nodes,
                                edge_index=edge_index,
                                question=question
                            )
                        except TypeError:
                            # 回退到不带question的调用
                            result = retriever.forward(
                                seed_nodes=seed_nodes,
                                edge_index=edge_index
                            )
                    
                    # 获取优化信息
                    if use_optimization:
                        opt_info = optimizer.optimize_retrieval(
                            retriever=retriever,
                            question=question,
                            seed_nodes=seed_nodes,
                            edge_index=edge_index
                        )
                    else:
                        opt_info = {}
                
                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                
                results.append({
                    'question_idx': i,
                    'time': elapsed_time,
                    'optimization_info': opt_info
                })
        
        return {
            'total_time': total_time,
            'avg_time': total_time / len(questions),
            'results': results,
            'optimization_enabled': use_optimization
        }
    
    def compare_performance(self,
                           retriever,
                           questions: List[str],
                           seed_nodes_list: List[torch.Tensor],
                           edge_index: torch.Tensor) -> Dict[str, Any]:
        """
        比较优化前后的性能
        
        Returns:
            性能比较结果
        """
        # 基线测试（无优化）
        print("运行基线测试...")
        baseline_results = self.benchmark_retrieval(
            retriever, questions, seed_nodes_list, edge_index, use_optimization=False
        )
        
        # 优化测试
        print("运行优化测试...")
        optimized_results = self.benchmark_retrieval(
            retriever, questions, seed_nodes_list, edge_index, use_optimization=True
        )
        
        # 计算改进
        baseline_time = baseline_results['total_time']
        optimized_time = optimized_results['total_time']
        improvement_ratio = (baseline_time - optimized_time) / baseline_time * 100
        
        return {
            'baseline': baseline_results,
            'optimized': optimized_results,
            'improvement': {
                'time_saved': baseline_time - optimized_time,
                'improvement_ratio': improvement_ratio,
                'speedup': baseline_time / optimized_time if optimized_time > 0 else float('inf')
            }
        }


# 全局优化器实例
global_optimizer = None


def get_global_optimizer() -> GNNRAGOptimizer:
    """获取全局优化器实例"""
    global global_optimizer
    if global_optimizer is None:
        global_optimizer = GNNRAGOptimizer()
    return global_optimizer


def initialize_optimization(enable_sync: bool = True,
                          enable_adaptive: bool = True,
                          enable_memory: bool = True,
                          monitoring_interval: float = 1.0):
    """初始化全局优化"""
    global global_optimizer
    global_optimizer = GNNRAGOptimizer(
        enable_sync_optimization=enable_sync,
        enable_adaptive_subgraph=enable_adaptive,
        enable_memory_monitoring=enable_memory,
        monitoring_interval=monitoring_interval
    )
    return global_optimizer


def cleanup_optimization():
    """清理全局优化"""
    global global_optimizer
    if global_optimizer:
        global_optimizer.cleanup()
        global_optimizer = None