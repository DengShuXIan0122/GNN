"""
GNN-RAG性能优化示例
展示如何使用GPU-CPU同步优化、自适应子图大小和内存监控功能
"""

import torch
import numpy as np
import time
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn.utils.performance_optimizer import (
    GNNRAGOptimizer, optimized_retrieval_context, PerformanceBenchmark,
    initialize_optimization, cleanup_optimization
)
from gnn.utils.memory_monitor import MemoryContext, monitor_performance
from gnn.utils.gpu_cpu_sync_utils import gpu_cpu_sync_profiler
from gnn.retrieval.hybrid_retriever import HybridRetriever


def create_sample_data(num_nodes: int = 10000, num_edges: int = 50000):
    """创建示例数据"""
    print(f"创建示例数据: {num_nodes}个节点, {num_edges}条边")
    
    # 创建随机图结构
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # 创建节点特征
    node_features = torch.randn(num_nodes, 768)  # 假设768维特征
    
    # 创建种子节点
    seed_nodes = torch.randint(0, num_nodes, (50,))  # 50个种子节点
    
    # 示例问题
    questions = [
        "What is the relationship between entity A and entity B?",
        "How does concept X relate to concept Y in the knowledge graph?",
        "Find all entities connected to the given seed nodes.",
        "What are the most important connections in this subgraph?",
        "Identify the shortest path between these entities."
    ]
    
    return {
        'edge_index': edge_index,
        'node_features': node_features,
        'seed_nodes': seed_nodes,
        'questions': questions
    }


@monitor_performance("demo_retrieval")
def demo_basic_optimization():
    """演示基本优化功能"""
    print("\n=== 基本优化功能演示 ===")
    
    # 创建示例数据
    data = create_sample_data()
    
    # 初始化优化器
    optimizer = initialize_optimization(
        enable_sync=True,
        enable_adaptive=True,
        enable_memory=True,
        monitoring_interval=0.5
    )
    
    print("优化器初始化完成")
    
    # 创建检索器（简化版本）
    class SimpleRetriever:
        def __init__(self, max_candidates=1200):
            self.max_candidates = max_candidates
        
        def forward(self, seed_nodes, edge_index, question=None):
            # 模拟检索过程
            time.sleep(0.1)  # 模拟计算时间
            
            # 模拟一些GPU-CPU同步操作
            with gpu_cpu_sync_profiler.profile_context("retrieval_sync"):
                # 模拟.item()调用
                for i in range(10):
                    _ = seed_nodes[i % len(seed_nodes)].item()
            
            return torch.randint(0, 1000, (self.max_candidates,))
    
    retriever = SimpleRetriever()
    
    # 使用优化上下文
    with optimized_retrieval_context(
        question=data['questions'][0],
        enable_sync_opt=True,
        enable_adaptive=True,
        enable_memory=True
    ) as opt:
        
        print("开始优化检索...")
        
        # 执行优化检索
        optimization_info = opt.optimize_retrieval(
            retriever=retriever,
            question=data['questions'][0],
            seed_nodes=data['seed_nodes'],
            edge_index=data['edge_index']
        )
        
        print("优化信息:")
        for key, value in optimization_info.items():
            print(f"  {key}: {value}")
        
        # 获取优化建议
        recommendations = opt.get_optimization_recommendations()
        print("\n优化建议:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    cleanup_optimization()


def demo_memory_monitoring():
    """演示内存监控功能"""
    print("\n=== 内存监控功能演示 ===")
    
    # 使用内存监控上下文
    with MemoryContext("memory_demo", cleanup_after=True):
        print("开始内存密集型操作...")
        
        # 创建大量张量模拟内存使用
        tensors = []
        for i in range(100):
            tensor = torch.randn(1000, 1000)
            if torch.cuda.is_available():
                tensor = tensor.cuda()
            tensors.append(tensor)
            
            if i % 20 == 0:
                print(f"  创建了 {i+1} 个张量")
        
        print("内存密集型操作完成")
    
    print("内存监控演示完成")


def demo_performance_benchmark():
    """演示性能基准测试"""
    print("\n=== 性能基准测试演示 ===")
    
    # 创建测试数据
    data = create_sample_data(num_nodes=5000, num_edges=25000)
    
    # 创建检索器
    class BenchmarkRetriever:
        def __init__(self, max_candidates=1200):
            self.max_candidates = max_candidates
        
        def forward(self, seed_nodes, edge_index, question=None):
            # 模拟更复杂的检索过程
            time.sleep(0.05)  # 基础计算时间
            
            # 模拟GPU-CPU同步
            sync_operations = 20 if question and len(question) > 50 else 10
            for i in range(sync_operations):
                _ = seed_nodes[i % len(seed_nodes)].item()
            
            return torch.randint(0, 1000, (self.max_candidates,))
    
    retriever = BenchmarkRetriever()
    
    # 创建基准测试器
    benchmark = PerformanceBenchmark()
    
    # 准备测试数据
    questions = data['questions']
    seed_nodes_list = [data['seed_nodes'] for _ in questions]
    
    print("开始性能基准测试...")
    
    # 运行比较测试
    comparison_results = benchmark.compare_performance(
        retriever=retriever,
        questions=questions,
        seed_nodes_list=seed_nodes_list,
        edge_index=data['edge_index']
    )
    
    # 显示结果
    baseline = comparison_results['baseline']
    optimized = comparison_results['optimized']
    improvement = comparison_results['improvement']
    
    print(f"\n基准测试结果:")
    print(f"  基线时间: {baseline['total_time']:.3f}秒")
    print(f"  优化时间: {optimized['total_time']:.3f}秒")
    print(f"  时间节省: {improvement['time_saved']:.3f}秒")
    print(f"  性能提升: {improvement['improvement_ratio']:.1f}%")
    print(f"  加速比: {improvement['speedup']:.2f}x")


def demo_adaptive_subgraph():
    """演示自适应子图大小功能"""
    print("\n=== 自适应子图大小演示 ===")
    
    from gnn.utils.adaptive_subgraph import QuestionComplexityAnalyzer, AdaptiveSubgraphSizer
    
    # 创建分析器
    complexity_analyzer = QuestionComplexityAnalyzer()
    adaptive_sizer = AdaptiveSubgraphSizer()
    
    # 测试不同复杂度的问题
    test_questions = [
        "What is A?",  # 简单问题
        "How does entity A relate to entity B in the context of concept C?",  # 中等复杂度
        "Find all entities that are connected to A through B, considering the relationships with C, D, and E, and analyze their impact on the overall network structure."  # 复杂问题
    ]
    
    data = create_sample_data()
    
    print("分析不同复杂度问题的子图大小调整:")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n问题 {i}: {question}")
        
        # 分析复杂度
        complexity_stats = complexity_analyzer.analyze_complexity(question)
        complexity_score = complexity_analyzer.compute_complexity_score(question)
        print(f"  复杂度分数: {complexity_score:.2f}")
        print(f"  词汇多样性: {complexity_stats['lexical_diversity']:.2f}")
        print(f"  多跳指标: {complexity_stats['multi_hop_score']}")
        
        # 计算自适应候选数量
        adaptive_candidates = adaptive_sizer.forward(
            question=question,
            edge_index=data['edge_index'],
            num_nodes=len(data['node_features']),
            seeds=data['seed_nodes']
        )
        
        print(f"  推荐候选数量: {adaptive_candidates}")
        print()


def main():
    """主函数"""
    print("GNN-RAG性能优化演示")
    print("=" * 50)
    
    try:
        # 1. 基本优化功能演示
        demo_basic_optimization()
        
        # 2. 内存监控演示
        demo_memory_monitoring()
        
        # 3. 自适应子图演示
        demo_adaptive_subgraph()
        
        # 4. 性能基准测试演示
        demo_performance_benchmark()
        
        print("\n" + "=" * 50)
        print("所有演示完成！")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        cleanup_optimization()


if __name__ == "__main__":
    main()