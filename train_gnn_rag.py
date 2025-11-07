#!/usr/bin/env python3
"""
GNN-RAG3 四阶段训练脚本
支持完整的参数配置和阶段控制

使用方法:
    python train_gnn_rag.py --dataset webqsp --gpu 0
    python train_gnn_rag.py --dataset cwq --gpu 0 --stages warmup,joint1
    python train_gnn_rag.py --config configs/webqsp_config.json
"""

import argparse
import json
import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入性能优化模块
try:
    from gnn.utils.performance_optimizer import (
        initialize_optimization, cleanup_optimization, get_global_optimizer
    )
    from gnn.utils.memory_monitor import start_system_monitoring, stop_system_monitoring
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    logging.warning("性能优化模块不可用，将使用标准训练模式")

def setup_logging(log_file: str = None):
    """设置日志"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)

class GNNRAGTrainer:
    """GNN-RAG3 训练器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.dataset = config['dataset']
        self.gpu_id = config['gpu_id']
        self.base_dir = Path(config.get('base_dir', PROJECT_ROOT))
        self.data_dir = self.base_dir / 'gnn' / 'data' / self.dataset
        self.output_dir = self.base_dir / 'gnn' / 'outputs' / f'gnn_rag3_{self.dataset}'
        
        # 初始化性能优化
        self.use_optimization = config.get('use_optimization', True) and OPTIMIZATION_AVAILABLE
        self.optimizer = None
        
        if self.use_optimization:
            logging.info("初始化性能优化功能...")
            self.optimizer = initialize_optimization(
                enable_sync=config.get('enable_sync_optimization', True),
                enable_adaptive=config.get('enable_adaptive_subgraph', True),
                enable_memory=config.get('enable_memory_monitoring', True),
                monitoring_interval=config.get('monitoring_interval', 2.0)
            )
            logging.info("性能优化功能已启用")
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        log_file = self.output_dir / 'training.log'
        setup_logging(str(log_file))
        
        logging.info(f"=== GNN-RAG3 Four-Stage Training Initialization ===")
        logging.info(f"Dataset: {self.dataset}")
        logging.info(f"GPU: {self.gpu_id}")
        logging.info(f"Output Directory: {self.output_dir}")
    
    def get_common_args(self) -> List[str]:
        """获取通用参数"""
        args = [
            'ReaRev',  # 模型名称，必须是第一个位置参数
            '--name', self.dataset,
            '--data_folder', str(self.data_dir),
            '--checkpoint_dir', str(self.output_dir)
        ]
        
        # 语言模型配置（从 config 注入到每个阶段）
        lm_cfg = self.config.get('language_model', {})
        if lm_cfg:
            lm = lm_cfg.get('lm')
            if lm:
                args.extend(['--lm', str(lm)])
            if 'lm_frozen' in lm_cfg:
                args.extend(['--lm_frozen', str(lm_cfg.get('lm_frozen', 1))])
            if 'lm_dropout' in lm_cfg:
                args.extend(['--lm_dropout', str(lm_cfg.get('lm_dropout', 0.3))])
        
        # 模型维度/结构（可选：按配置覆盖默认值，确保一致性）
        model_cfg = self.config.get('model_params', {})
        if model_cfg:
            if 'entity_dim' in model_cfg:
                args.extend(['--entity_dim', str(model_cfg['entity_dim'])])
            if 'num_iter' in model_cfg:
                args.extend(['--num_iter', str(model_cfg['num_iter'])])
            if 'num_ins' in model_cfg:
                args.extend(['--num_ins', str(model_cfg['num_ins'])])
            if 'num_gnn' in model_cfg:
                args.extend(['--num_gnn', str(model_cfg['num_gnn'])])
        
        # 多GPU配置
        multi_gpu_config = self.config.get('multi_gpu', {})
        if multi_gpu_config.get('enabled', False):
            args.extend(['--use_multi_gpu'])
            gpu_ids = multi_gpu_config.get('gpu_ids', [0, 1])
            args.extend(['--gpu_ids', ','.join(map(str, gpu_ids))])
        
        return args
    
    def get_rag3_args(self) -> List[str]:
        """获取GNN-RAG3特定参数"""
        config = self.config.get('rag3_params', {})
        args = []
        
        # APPR参数
        if config.get('use_appr', True):
            args.extend(['--use_appr'])
            args.extend(['--appr_alpha', str(config.get('appr_alpha', 0.85))])
            args.extend(['--cand_n', str(config.get('cand_n', 1200))])
        
        # DDE参数
        if config.get('use_dde', True):
            args.extend(['--use_dde'])
            args.extend(['--hop_dim', str(config.get('hop_dim', 16))])
            args.extend(['--dir_dim', str(config.get('dir_dim', 8))])
        
        # PCST参数
        if config.get('use_pcst', True):
            args.extend(['--use_pcst'])
            pcst_lambda = config.get('pcst_lambda', [0.1, 0.1, 0.05])
            args.extend(['--pcst_lambda', ','.join(map(str, pcst_lambda))])
        
        # Mid-layer restart
        if config.get('mid_restart', True):
            args.extend(['--mid_restart'])
        
        return args

    def get_opt_args(self) -> List[str]:
        """获取性能优化相关参数（所有阶段都应传递）"""
        if not self.use_optimization:
            return []
        return [
            '--use_optimization', 'True',
            '--enable_sync_optimization', str(self.config.get('enable_sync_optimization', True)),
            '--enable_adaptive_subgraph', str(self.config.get('enable_adaptive_subgraph', True)),
            '--enable_memory_monitoring', str(self.config.get('enable_memory_monitoring', True)),
            '--min_candidates', str(self.config.get('min_candidates', 300)),
            '--max_candidates', str(self.config.get('max_candidates', 2000)),
        ]

    def _resolve_checkpoint(self, ckpt_name: str) -> Optional[str]:
        """解析上一阶段检查点，支持未带前缀的简写并回退到 f1/h1/final。
        优先匹配以实验名前缀开头的文件，例如 gnn_rag3_<dataset>-<stage>_best.pt。
        """
        prefix = f"gnn_rag3_{self.dataset}-"
        suffixes = ['_best.pt', '-f1.ckpt', '-h1.ckpt', '-final.ckpt']

        def strip_suffix(name: str) -> str:
            for s in suffixes:
                if name.endswith(s):
                    return name[:-len(s)]
            return name

        # 如果传入的是简写（如 warmup_best.pt），尝试带前缀与不带前缀两套候选
        bases = []
        if ckpt_name.startswith(prefix):
            bases.append(strip_suffix(ckpt_name))
        else:
            bases.append(strip_suffix(ckpt_name))  # 例如 'warmup'
            bases.append(prefix + strip_suffix(ckpt_name))  # 例如 'gnn_rag3_webqsp-warmup'

        candidates = []
        for base in bases:
            candidates.extend([
                f"{base}_best.pt",
                f"{base}-f1.ckpt",
                f"{base}-h1.ckpt",
                f"{base}-final.ckpt",
            ])

        for fname in candidates:
            p = self.output_dir / fname
            if p.exists():
                return p.name  # 返回相对 checkpoint_dir 的文件名
        return None

    def run_stage(self, stage_name: str, stage_config: Dict) -> bool:
        """运行训练阶段"""
        logging.info(f"\n=== Stage: {stage_name} ===")
        
        # 构建命令 - 使用gnn.main而不是train_model
        cmd = ['python', '-m', 'gnn.main']
        cmd.extend(self.get_common_args())
        
        # 为该阶段设置唯一的 experiment_name，确保产出 *_best.pt
        stage_exp_name = f"gnn_rag3_{self.dataset}-{stage_name}"
        cmd.extend(['--experiment_name', stage_exp_name])
        
        # 阶段特定算法（warmup不加RAG3算法开关）
        if stage_name != 'warmup':
            cmd.extend(self.get_rag3_args())
        else:
            # Warmup: optionally enable APPR if training hybrid only
            if stage_config.get('train_hybrid_only', False):
                rag3_cfg = self.config.get('rag3_params', {})
                cmd += ['--use_appr']
                cmd += ['--appr_alpha', str(rag3_cfg.get('appr_alpha', 0.85))]
                cmd += ['--cand_n', str(rag3_cfg.get('cand_n', 1200))]
        # Optimization args
        cmd += self.get_opt_args()
        # Stage-specific toggles
        if stage_config.get('train_hybrid_only', False):
            cmd += ['--train_hybrid_only']
        if stage_config.get('freeze_gnn', False):
            cmd += ['--freeze_gnn']
        
        # 基础训练参数 - 使用 gnn.main 支持的参数
        eval_every_raw = stage_config.get('eval_every', 1000)
        eval_every = max(1, eval_every_raw // 100) if eval_every_raw >= 100 else eval_every_raw
        cmd.extend([
            '--num_epoch', str(stage_config.get('num_epoch', stage_config.get('max_steps', 10000) // 100)),
            '--lr', str(stage_config.get('learning_rate', 1e-4)),
            '--batch_size', str(stage_config.get('batch_size', 8)),
            '--eval_every', str(eval_every),
        ])
        
        # 加载上一阶段的检查点（支持简写和前缀）
        if 'load_checkpoint' in stage_config:
            resolved = self._resolve_checkpoint(stage_config['load_checkpoint'])
            if resolved:
                cmd.extend(['--load_experiment', resolved])
            else:
                logging.warning(f"未找到上一阶段检查点或回退文件: {stage_config['load_checkpoint']}")
        
        logging.info(f"Executing command: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
            logging.info(f"Stage {stage_name} completed")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Stage {stage_name} failed: {e}")
            return False
    
    def train(self, stages: Optional[List[str]] = None):
        """执行训练"""
        # 默认训练阶段配置（针对双GPU优化）
        default_stages = {
            'warmup': {
                'max_steps': 5000,
                'learning_rate': 1e-3,
                'freeze_gnn': True,
                'train_hybrid_only': True,
                'batch_size': 32,  # 双GPU：16 -> 32
                'eval_every': 500,
                'save_every': 1000
            },
            'joint1': {
                'max_steps': 20000,
                'learning_rate': 5e-4,
                'num_gnn_layers': 6,
                'hidden_dim': 256,
                'dropout': 0.1,
                'num_heads': 3,
                'batch_size': 16,  # 双GPU：8 -> 16，充分利用48GB内存
                'eval_every': 1000,  # 减少评估频率以加速训练
                'save_every': 5000,
                'load_checkpoint': 'warmup_best.pt'
            },
            'pcst_distill': {
                'max_steps': 10000,
                'learning_rate': 1e-4,
                'pcst_distill_only': True,
                'gumbel_temp': 2.0,
                'batch_size': 16,  # 双GPU：8 -> 16
                'eval_every': 1000,
                'save_every': 2500,
                'load_checkpoint': 'joint1_best.pt'
            },
            'joint2': {
                'max_steps': 20000,
                'learning_rate': 2e-4,
                'gumbel_temp_start': 2.0,
                'gumbel_temp_end': 0.5,
                'temp_anneal_steps': 15000,
                'lambda_ret': 0.1,
                'lambda_pcst': 0.2,
                'batch_size': 16,  # 双GPU：8 -> 16
                'eval_every': 1000,  # 减少评估频率以加速训练
                'save_every': 5000,
                'load_checkpoint': 'pcst_distill_best.pt'
            }
        }
        
        # 合并用户配置
        stage_configs = self.config.get('stages', {})
        for stage_name, default_config in default_stages.items():
            if stage_name in stage_configs:
                default_config.update(stage_configs[stage_name])
            stage_configs[stage_name] = default_config
        
        # 确定要运行的阶段
        if stages is None:
            stages = list(default_stages.keys())
        
        # 执行训练阶段
        for stage_name in stages:
            if stage_name not in stage_configs:
                logging.error(f"Unknown stage: {stage_name}")
                return False
            
            success = self.run_stage(stage_name, stage_configs[stage_name])
            if not success:
                logging.error(f"Training failed at stage {stage_name}")
                return False
        
        # 最终评估
        self.final_evaluation()
        return True
    
    def _aggregate_info_metrics(self, info_file: Path) -> Dict:
        """读取 *_test.info 并聚合为 final_results.json 指标"""
        metrics = {
            'count': 0,
            'f1': 0.0,
            'hit_10': 0.0,  # 这里用 hit 字段近似
            'em': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
        if not info_file.exists():
            return metrics
        try:
            with open(info_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    metrics['count'] += 1
                    metrics['f1'] += float(obj.get('f1', 0.0))
                    # evaluate.py 使用键名 'precison'，兼容拼写错误
                    prec = obj.get('precison', obj.get('precision', 0.0))
                    metrics['precision'] += float(prec)
                    metrics['recall'] += float(obj.get('recall', 0.0))
                    metrics['hit_10'] += float(obj.get('hit', 0.0))
                    metrics['em'] += float(obj.get('em', 0.0))
            if metrics['count'] > 0:
                for k in ['f1', 'precision', 'recall', 'hit_10', 'em']:
                    metrics[k] = metrics[k] / metrics['count']
        except Exception as e:
            logging.error(f"聚合评估指标失败: {e}")
        return metrics

    def final_evaluation(self):
        """最终评估：使用 gnn.main 的 --is_eval 生成 *_test.info，并写 final_results.json"""
        logging.info("\n=== Final Evaluation ===")
        # 解析 joint2 的检查点（允许回退）
        resolved = self._resolve_checkpoint('joint2_best.pt')
        if not resolved:
            logging.error("未找到 joint2 的评估检查点，跳过最终评估")
            return
        
        # 设置评估 experiment_name 与阶段保持一致，便于生成 info 文件
        eval_experiment = f"gnn_rag3_{self.dataset}-joint2"
        
        cmd = [
            'python', '-m', 'gnn.main',
            *self.get_common_args(),
            '--is_eval',
            '--load_experiment', resolved,
            '--experiment_name', eval_experiment
        ]
        
        try:
            subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
            logging.info("Evaluation via gnn.main completed")
            # 读取 *_test.info 并写 final_results.json
            info_file = self.output_dir / f"{eval_experiment}_test.info"
            metrics = self._aggregate_info_metrics(info_file)
            out_file = self.output_dir / 'final_results.json'
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            logging.info(f"Final metrics written: {out_file}")
            self.generate_report()
        except subprocess.CalledProcessError as e:
            logging.error(f"Final evaluation failed: {e}")
    
    def generate_report(self):
        """生成训练报告"""
        logging.info("\n=== Training Report ===")
        
        stages = ['warmup', 'joint1', 'pcst_distill', 'joint2']
        for stage in stages:
            log_file = self.output_dir / f'{stage}.log'
            if log_file.exists():
                logging.info(f'{stage}: Log file exists')
            else:
                logging.info(f'{stage}: Log file missing')
        
        results_file = self.output_dir / 'final_results.json'
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                logging.info(f"\nFinal Metrics:")
                logging.info(f'Hit@10: {results.get("hit_10", "N/A")}')
                logging.info(f'F1: {results.get("f1", "N/A")}')
                logging.info(f'Connectivity: {results.get("connectivity", "N/A")}')
                logging.info(f'KG Tokens: {results.get("kg_tokens", "N/A")}')
            except Exception as e:
                logging.error(f"Failed to read results file: {e}")
        
        # 生成性能优化报告
        if self.use_optimization and self.optimizer:
            try:
                performance_report = self.optimizer.get_comprehensive_report()
                report_file = self.output_dir / 'performance_optimization_report.json'
                
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(performance_report, f, indent=2, ensure_ascii=False)
                
                logging.info(f"\n=== Performance Optimization Report ===")
                logging.info(f"报告已保存到: {report_file}")
                
                # 显示关键优化统计
                opt_stats = performance_report.get('optimization_stats', {})
                logging.info(f"GPU-CPU同步优化次数: {opt_stats.get('sync_optimizations', 0)}")
                logging.info(f"自适应子图调整次数: {opt_stats.get('adaptive_adjustments', 0)}")
                logging.info(f"内存清理次数: {opt_stats.get('memory_cleanups', 0)}")
                
                # 显示优化建议
                recommendations = performance_report.get('recommendations', [])
                if recommendations:
                    logging.info("\n优化建议:")
                    for i, rec in enumerate(recommendations, 1):
                        logging.info(f"  {i}. {rec}")
                
            except Exception as e:
                logging.error(f"生成性能优化报告失败: {e}")
    
    def cleanup(self):
        """清理资源"""
        if self.use_optimization:
            logging.info("清理性能优化资源...")
            cleanup_optimization()
            logging.info("性能优化资源清理完成")

def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_default_config(dataset: str, gpu_id: int) -> Dict:
    """创建默认配置"""
    return {
        'dataset': dataset,
        'gpu_id': gpu_id,
        'base_dir': str(PROJECT_ROOT),
        'multi_gpu': {
            'enabled': True,  # 默认启用多GPU
            'gpu_ids': [0, 1]  # 使用GPU 0和1
        },
        'rag3_params': {
            'use_appr': True,
            'appr_alpha': 0.85,
            'cand_n': 1200,
            'use_dde': True,
            'hop_dim': 16,
            'dir_dim': 8,
            'use_pcst': True,
            'pcst_lambda': [0.1, 0.1, 0.05],
            'mid_restart': True
        }
    }

def main():
    parser = argparse.ArgumentParser(description='GNN-RAG3 四阶段训练脚本')
    parser.add_argument('--dataset', type=str, default='webqsp', 
                       choices=['webqsp', 'cwq'], help='数据集名称')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--stages', type=str, default='warmup,joint1,pcst_distill,joint2', help='要运行的阶段，逗号分隔')
    parser.add_argument('--checkpoint', type=str, help='checkpoint路径')
    parser.add_argument('--output_dir', type=str, help='输出目录')
    
    # GNN-RAG3 特定参数
    parser.add_argument('--use_appr', action='store_true', help='启用APPR (Approximate Personalized PageRank)')
    parser.add_argument('--appr_alpha', type=float, default=0.85, help='APPR alpha参数')
    parser.add_argument('--cand_n', type=int, default=1200, help='候选实体数量')
    parser.add_argument('--use_dde', action='store_true', help='启用DDE (Direction-aware Distance Encoding)')
    parser.add_argument('--hop_dim', type=int, default=16, help='跳数维度')
    parser.add_argument('--dir_dim', type=int, default=8, help='方向维度')
    parser.add_argument('--use_pcst', action='store_true', help='启用PCST (Prize-Collecting Steiner Tree)')
    parser.add_argument('--pcst_lambda', type=str, default='0.1,0.1,0.05', help='PCST lambda参数 (cost,conn,sparse)')
    parser.add_argument('--mid_layer_restart', action='store_true', help='启用中层重启')
    
    # 调试和日志参数
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--log_level', default='INFO', help='日志级别')
    
    # 性能优化参数
    parser.add_argument('--use_optimization', action='store_true', help='启用性能优化功能')
    parser.add_argument('--enable_sync_optimization', action='store_true', help='启用GPU-CPU同步优化')
    parser.add_argument('--enable_adaptive_subgraph', action='store_true', help='启用自适应子图大小调整')
    parser.add_argument('--enable_memory_monitoring', action='store_true', help='启用内存监控')
    parser.add_argument('--min_candidates', type=int, default=300, help='最小候选数量')
    parser.add_argument('--max_candidates', type=int, default=2000, help='最大候选数量')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # 加载配置
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config(args.dataset, args.gpu)
    
    # 覆盖命令行参数
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # 处理GNN-RAG3参数
    rag3_params = config.setdefault('rag3_params', {})
    if args.use_appr:
        rag3_params['use_appr'] = True
        rag3_params['appr_alpha'] = args.appr_alpha
        rag3_params['cand_n'] = args.cand_n
    if args.use_dde:
        rag3_params['use_dde'] = True
        rag3_params['hop_dim'] = args.hop_dim
        rag3_params['dir_dim'] = args.dir_dim
    if args.use_pcst:
        rag3_params['use_pcst'] = True
        # 解析PCST lambda参数
        lam_cost, lam_conn, lam_sparse = map(float, args.pcst_lambda.split(','))
        rag3_params['pcst_lambda'] = [lam_cost, lam_conn, lam_sparse]
    if args.mid_layer_restart:
        rag3_params['mid_restart'] = True
    
    # 处理性能优化参数
    optimization_params = config.setdefault('optimization', {})
    if args.use_optimization:
        optimization_params['enabled'] = True
        optimization_params['sync_optimization'] = args.enable_sync_optimization
        optimization_params['adaptive_subgraph'] = args.enable_adaptive_subgraph
        optimization_params['memory_monitoring'] = args.enable_memory_monitoring
        optimization_params['min_candidates'] = args.min_candidates
        optimization_params['max_candidates'] = args.max_candidates
    
    # 解析阶段
    stages = None
    if args.stages:
        stages = [s.strip() for s in args.stages.split(',')]
    
    # 创建训练器并开始训练
    trainer = GNNRAGTrainer(config)
    
    try:
        success = trainer.train(stages)
        
        if success:
            logging.info("\n=== GNN-RAG3 Training Completed ===")
            logging.info(f"Final model: {trainer.output_dir}/joint2_best.pt")
            logging.info(f"Results file: {trainer.output_dir}/final_results.json")
            
            if trainer.use_optimization:
                logging.info(f"Performance report: {trainer.output_dir}/performance_optimization_report.json")
        else:
            logging.error("Training failed")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logging.info("训练被用户中断")
        sys.exit(1)
    except Exception as e:
        logging.error(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # 确保清理资源
        trainer.cleanup()

if __name__ == '__main__':
    main()