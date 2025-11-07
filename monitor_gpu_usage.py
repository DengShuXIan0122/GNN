# -*- coding: utf-8 -*-
import torch
import time
import psutil

def monitor_gpu_cpu_usage():
    print('=== GPU/CPU使用率监控 ===')
    if torch.cuda.is_available():
        print(f'GPU数量: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    
    # 创建测试tensor
    if torch.cuda.is_available():
        device = torch.device('cuda')
        x = torch.randn(1000, 1000, device=device)
        
        # 测试CPU回退
        print('\n测试CPU回退影响:')
        
        # 无CPU回退的操作
        start_time = time.time()
        for _ in range(100):
            y = torch.matmul(x, x.T)  # 纯GPU操作
        gpu_time = time.time() - start_time
        print(f'纯GPU操作时间: {gpu_time:.4f}秒')
        
        # 有CPU回退的操作
        start_time = time.time()
        for _ in range(100):
            y = torch.matmul(x, x.T)
            _ = y[0, 0].item()  # 触发CPU回退
        cpu_fallback_time = time.time() - start_time
        print(f'包含CPU回退时间: {cpu_fallback_time:.4f}秒')
        print(f'性能损失: {(cpu_fallback_time/gpu_time - 1)*100:.1f}%')
    else:
        print('CUDA不可用，无法演示GPU操作')
    
    print(f'\nCPU使用率: {psutil.cpu_percent()}%')
    print(f'内存使用率: {psutil.virtual_memory().percent}%')

if __name__ == '__main__':
    monitor_gpu_cpu_usage()