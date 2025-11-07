"""
内存优化和性能监控模块
提供内存使用监控、优化建议和性能分析功能
"""

import torch
import psutil
import gc
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import defaultdict, deque
import numpy as np
import json
from datetime import datetime


class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.memory_history = deque(maxlen=max_history)
        self.gpu_memory_history = deque(maxlen=max_history)
        self.peak_memory = {'cpu': 0, 'gpu': 0}
        self.monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 1.0  # 秒
        
    def get_cpu_memory_info(self) -> Dict[str, float]:
        """获取CPU内存信息"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # 物理内存
            'vms_mb': memory_info.vms / 1024 / 1024,  # 虚拟内存
            'percent': process.memory_percent(),       # 内存占用百分比
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """获取GPU内存信息"""
        if not torch.cuda.is_available():
            return {'allocated_mb': 0, 'cached_mb': 0, 'max_allocated_mb': 0}
        
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        cached = torch.cuda.memory_reserved() / 1024 / 1024
        max_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        return {
            'allocated_mb': allocated,
            'cached_mb': cached,
            'max_allocated_mb': max_allocated
        }
    
    def record_memory_snapshot(self) -> Dict[str, Any]:
        """记录内存快照"""
        timestamp = time.time()
        cpu_info = self.get_cpu_memory_info()
        gpu_info = self.get_gpu_memory_info()
        
        snapshot = {
            'timestamp': timestamp,
            'cpu': cpu_info,
            'gpu': gpu_info
        }
        
        # 更新峰值
        self.peak_memory['cpu'] = max(self.peak_memory['cpu'], cpu_info['rss_mb'])
        self.peak_memory['gpu'] = max(self.peak_memory['gpu'], gpu_info['allocated_mb'])
        
        return snapshot
    
    def start_monitoring(self, interval: float = 1.0):
        """开始监控"""
        self.monitor_interval = interval
        self.monitoring = True
        
        def monitor_loop():
            while self.monitoring:
                snapshot = self.record_memory_snapshot()
                self.memory_history.append(snapshot)
                time.sleep(self.monitor_interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        if not self.memory_history:
            return {}
        
        cpu_rss = [h['cpu']['rss_mb'] for h in self.memory_history]
        gpu_allocated = [h['gpu']['allocated_mb'] for h in self.memory_history]
        
        return {
            'cpu': {
                'current_mb': cpu_rss[-1] if cpu_rss else 0,
                'peak_mb': self.peak_memory['cpu'],
                'avg_mb': np.mean(cpu_rss) if cpu_rss else 0,
                'std_mb': np.std(cpu_rss) if cpu_rss else 0
            },
            'gpu': {
                'current_mb': gpu_allocated[-1] if gpu_allocated else 0,
                'peak_mb': self.peak_memory['gpu'],
                'avg_mb': np.mean(gpu_allocated) if gpu_allocated else 0,
                'std_mb': np.std(gpu_allocated) if gpu_allocated else 0
            },
            'history_length': len(self.memory_history)
        }
    
    def clear_history(self):
        """清空历史记录"""
        self.memory_history.clear()
        self.gpu_memory_history.clear()
        self.peak_memory = {'cpu': 0, 'gpu': 0}


class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.counters = defaultdict(int)
        self.active_timers = {}
        
    def start_timer(self, name: str):
        """开始计时"""
        self.active_timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """结束计时并返回耗时"""
        if name in self.active_timers:
            elapsed = time.time() - self.active_timers[name]
            self.timings[name].append(elapsed)
            del self.active_timers[name]
            return elapsed
        return 0.0
    
    def increment_counter(self, name: str, value: int = 1):
        """增加计数器"""
        self.counters[name] += value
    
    def get_timing_stats(self, name: str) -> Dict[str, float]:
        """获取计时统计"""
        if name not in self.timings or not self.timings[name]:
            return {}
        
        times = self.timings[name]
        return {
            'count': len(times),
            'total_s': sum(times),
            'avg_s': np.mean(times),
            'min_s': min(times),
            'max_s': max(times),
            'std_s': np.std(times)
        }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """获取所有统计信息"""
        timing_stats = {}
        for name in self.timings:
            timing_stats[name] = self.get_timing_stats(name)
        
        return {
            'timings': timing_stats,
            'counters': dict(self.counters)
        }
    
    def reset(self):
        """重置所有统计"""
        self.timings.clear()
        self.counters.clear()
        self.active_timers.clear()


class MemoryOptimizer:
    """内存优化器"""
    
    @staticmethod
    def cleanup_gpu_memory():
        """清理GPU内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def cleanup_cpu_memory():
        """清理CPU内存"""
        gc.collect()
    
    @staticmethod
    def optimize_tensor_memory(tensor: torch.Tensor, 
                             target_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """优化张量内存使用"""
        if target_dtype is not None and tensor.dtype != target_dtype:
            # 类型转换以节省内存
            if target_dtype in [torch.float16, torch.bfloat16] and tensor.dtype == torch.float32:
                return tensor.to(target_dtype)
        
        # 确保张量是连续的
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        return tensor
    
    @staticmethod
    def get_memory_recommendations(memory_stats: Dict[str, Any]) -> List[str]:
        """获取内存优化建议"""
        recommendations = []
        
        cpu_stats = memory_stats.get('cpu', {})
        gpu_stats = memory_stats.get('gpu', {})
        
        # CPU内存建议
        if cpu_stats.get('current_mb', 0) > 8000:  # 8GB
            recommendations.append("CPU内存使用较高，考虑减少批处理大小或使用数据流处理")
        
        if cpu_stats.get('peak_mb', 0) > cpu_stats.get('current_mb', 0) * 2:
            recommendations.append("检测到内存峰值较高，可能存在内存泄漏")
        
        # GPU内存建议
        if gpu_stats.get('current_mb', 0) > 10000:  # 10GB
            recommendations.append("GPU内存使用较高，考虑使用梯度检查点或模型并行")
        
        if gpu_stats.get('peak_mb', 0) > gpu_stats.get('current_mb', 0) * 1.5:
            recommendations.append("GPU内存峰值较高，考虑调用torch.cuda.empty_cache()")
        
        return recommendations


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, max_memory_mb: float = 1000):
        self.max_memory_mb = max_memory_mb
        self.caches = {}
        self.cache_stats = defaultdict(lambda: {'hits': 0, 'misses': 0, 'size': 0})
    
    def register_cache(self, name: str, cache_obj: Any):
        """注册缓存对象"""
        self.caches[name] = cache_obj
    
    def get_cache_memory_usage(self) -> Dict[str, float]:
        """获取缓存内存使用情况"""
        usage = {}
        for name, cache in self.caches.items():
            if hasattr(cache, 'get_memory_usage'):
                usage[name] = cache.get_memory_usage()
            elif hasattr(cache, '__sizeof__'):
                usage[name] = cache.__sizeof__() / 1024 / 1024  # MB
            else:
                usage[name] = 0
        return usage
    
    def cleanup_caches(self, threshold_mb: Optional[float] = None):
        """清理缓存"""
        if threshold_mb is None:
            threshold_mb = self.max_memory_mb * 0.8
        
        total_usage = sum(self.get_cache_memory_usage().values())
        
        if total_usage > threshold_mb:
            # 清理所有注册的缓存
            for name, cache in self.caches.items():
                if hasattr(cache, 'clear'):
                    cache.clear()
                elif hasattr(cache, 'cache_clear'):
                    cache.cache_clear()
    
    def get_cache_stats(self) -> Dict[str, Dict[str, int]]:
        """获取缓存统计信息"""
        stats = {}
        for name, cache in self.caches.items():
            if hasattr(cache, 'get_stats'):
                stats[name] = cache.get_stats()
            else:
                stats[name] = dict(self.cache_stats[name])
        return stats


class SystemMonitor:
    """系统监控器"""
    
    def __init__(self):
        self.memory_monitor = MemoryMonitor()
        self.performance_profiler = PerformanceProfiler()
        self.memory_optimizer = MemoryOptimizer()
        self.cache_manager = CacheManager()
        
    def start_monitoring(self, interval: float = 1.0):
        """开始系统监控"""
        self.memory_monitor.start_monitoring(interval)
    
    def stop_monitoring(self):
        """停止系统监控"""
        self.memory_monitor.stop_monitoring()
    
    def get_system_report(self) -> Dict[str, Any]:
        """获取系统报告"""
        memory_stats = self.memory_monitor.get_memory_stats()
        performance_stats = self.performance_profiler.get_all_stats()
        cache_stats = self.cache_manager.get_cache_stats()
        cache_memory = self.cache_manager.get_cache_memory_usage()
        
        recommendations = self.memory_optimizer.get_memory_recommendations(memory_stats)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'memory': memory_stats,
            'performance': performance_stats,
            'cache': {
                'stats': cache_stats,
                'memory_usage_mb': cache_memory
            },
            'recommendations': recommendations
        }
    
    def optimize_system(self):
        """优化系统性能"""
        # 清理内存
        self.memory_optimizer.cleanup_cpu_memory()
        self.memory_optimizer.cleanup_gpu_memory()
        
        # 清理缓存
        self.cache_manager.cleanup_caches()
    
    def save_report(self, filepath: str):
        """保存系统报告到文件"""
        report = self.get_system_report()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


# 全局系统监控器实例
system_monitor = SystemMonitor()


def get_memory_stats() -> Dict[str, Any]:
    """获取内存统计信息"""
    return system_monitor.memory_monitor.get_memory_stats()


def get_performance_stats() -> Dict[str, Any]:
    """获取性能统计信息"""
    return system_monitor.performance_profiler.get_all_stats()


def start_system_monitoring(interval: float = 1.0):
    """开始系统监控"""
    system_monitor.start_monitoring(interval)


def stop_system_monitoring():
    """停止系统监控"""
    system_monitor.stop_monitoring()


def optimize_system():
    """优化系统性能"""
    system_monitor.optimize_system()


def get_system_report() -> Dict[str, Any]:
    """获取系统报告"""
    return system_monitor.get_system_report()


def save_system_report(filepath: str):
    """保存系统报告"""
    system_monitor.save_report(filepath)


# 装饰器：性能监控
def monitor_performance(name: str):
    """性能监控装饰器"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            system_monitor.performance_profiler.start_timer(name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                system_monitor.performance_profiler.end_timer(name)
        return wrapper
    return decorator


# 上下文管理器：内存监控
class MemoryContext:
    """内存监控上下文管理器"""
    
    def __init__(self, name: str, cleanup_after: bool = True):
        self.name = name
        self.cleanup_after = cleanup_after
        self.start_memory = None
    
    def __enter__(self):
        self.start_memory = system_monitor.memory_monitor.record_memory_snapshot()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_memory = system_monitor.memory_monitor.record_memory_snapshot()
        
        # 计算内存变化
        cpu_diff = end_memory['cpu']['rss_mb'] - self.start_memory['cpu']['rss_mb']
        gpu_diff = end_memory['gpu']['allocated_mb'] - self.start_memory['gpu']['allocated_mb']
        
        print(f"[{self.name}] 内存变化: CPU {cpu_diff:+.2f}MB, GPU {gpu_diff:+.2f}MB")
        
        if self.cleanup_after:
            system_monitor.optimize_system()