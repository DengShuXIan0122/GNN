# -*- coding: utf-8 -*-
"""
CPU回退演示 - 模拟GNN-RAG中的实际场景
"""
import torch
import time
import numpy as np

def demo_cpu_fallback():
    print("=== CPU回退性能影响演示 ===")
    
    if not torch.cuda.is_available():
        print("CUDA不可用，使用CPU演示")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    
    # 模拟GNN-RAG中的数据规模
    num_nodes = 10000
    num_edges = 50000
    
    # 创建模拟数据
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    node_scores = torch.randn(num_nodes, device=device)
    edge_costs = torch.rand(num_edges, device=device) + 0.1
    
    print(f"\n数据规模:")
    print(f"节点数: {num_nodes:,}")
    print(f"边数: {num_edges:,}")
    
    # 方法1: 有CPU回退的原始方法（模拟PCST损失函数）
    print(f"\n=== 方法1: 有CPU回退的原始方法 ===")
    start_time = time.time()
    
    # 模拟原始PCST损失函数中的CPU回退操作
    selected_edges = []
    for i in range(min(1000, num_edges)):  # 只测试1000条边避免太慢
        src_id = edge_index[0, i].item()  # CPU回退
        dst_id = edge_index[1, i].item()  # CPU回退
        src_score = node_scores[src_id].item()  # CPU回退
        dst_score = node_scores[dst_id].item()  # CPU回退
        edge_cost = edge_costs[i].item()  # CPU回退
        
        # 简单的选择逻辑
        if (src_score + dst_score) / edge_cost > 0.5:
            selected_edges.append(i)
    
    cpu_fallback_time = time.time() - start_time
    print(f"处理1000条边用时: {cpu_fallback_time:.4f}秒")
    print(f"选中边数: {len(selected_edges)}")
    
    # 方法2: 优化后的向量化方法
    print(f"\n=== 方法2: 优化后的向量化方法 ===")
    start_time = time.time()
    
    # 向量化操作，无CPU回退
    src_scores = node_scores[edge_index[0]]  # 向量化索引
    dst_scores = node_scores[edge_index[1]]  # 向量化索引
    edge_importance = (src_scores + dst_scores) / edge_costs
    
    # 选择重要的边
    mask = edge_importance > 0.5
    selected_edges_vec = torch.nonzero(mask).squeeze()
    
    vectorized_time = time.time() - start_time
    print(f"处理{num_edges:,}条边用时: {vectorized_time:.4f}秒")
    print(f"选中边数: {len(selected_edges_vec)}")
    
    # 性能对比
    print(f"\n=== 性能对比 ===")
    # 估算处理相同数量边的时间
    estimated_cpu_time = cpu_fallback_time * (num_edges / 1000)
    speedup = estimated_cpu_time / vectorized_time
    
    print(f"CPU回退方法估算时间: {estimated_cpu_time:.4f}秒")
    print(f"向量化方法实际时间: {vectorized_time:.4f}秒")
    print(f"性能提升: {speedup:.1f}x")
    
    # 内存使用情况
    if torch.cuda.is_available():
        print(f"\nGPU内存使用:")
        print(f"已分配: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        print(f"缓存: {torch.cuda.memory_reserved()/1024**2:.1f} MB")

if __name__ == '__main__':
    demo_cpu_fallback()