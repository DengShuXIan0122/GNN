#!/usr/bin/env python3
"""
GNN-RAG³ 推理脚本
支持完整的推理流程和结果分析

使用方法:
    python infer_gnn_rag.py --dataset webqsp --checkpoint path/to/model.pt --gpu 0
    python infer_gnn_rag.py --config configs/inference_config.json
    python infer_gnn_rag.py --dataset cwq --checkpoint model.pt --stages hybrid,gnn --batch_size 16
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

class GNNRAGInferencer:
    """GNN-RAG³ 推理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.dataset = config['dataset']
        self.checkpoint_path = Path(config['checkpoint_path'])
        self.gpu_id = config['gpu_id']
        self.base_dir = Path(config.get('base_dir', PROJECT_ROOT))
        self.data_dir = self.base_dir / 'gnn' / 'data' / self.dataset
        self.output_dir = self.base_dir / 'gnn' / 'outputs' / f'inference_{self.dataset}'
        
        # 检查checkpoint
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint文件不存在: {self.checkpoint_path}")
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        log_file = self.output_dir / 'inference.log'
        setup_logging(str(log_file))
        
        logging.info(f"=== GNN-RAG³ 推理初始化 ===")
        logging.info(f"数据集: {self.dataset}")
        logging.info(f"模型: {self.checkpoint_path}")
        logging.info(f"GPU: {self.gpu_id}")
        logging.info(f"输出目录: {self.output_dir}")
    
    def get_common_args(self) -> List[str]:
        """获取通用参数"""
        return [
            '--dataset', self.dataset,
            '--gpu', str(self.gpu_id),
            '--data_folder', str(self.data_dir),
            '--output_dir', str(self.output_dir),
            '--load_checkpoint', str(self.checkpoint_path)
        ]
    
    def get_rag3_args(self) -> List[str]:
        """获取GNN-RAG³特定参数"""
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
    
    def run_hybrid_retrieval(self, stage_config: Dict) -> bool:
        """运行混合检索阶段"""
        logging.info("\n=== 阶段1: 混合检索 + 候选子图构建 ===")
        
        cmd = [
            'python', '-m', 'gnn.inference',
            *self.get_common_args(),
            *self.get_rag3_args(),
            '--stage', 'hybrid_retrieval',
            '--eval_split', stage_config.get('eval_split', 'test'),
            '--batch_size', str(stage_config.get('batch_size', 16)),
            '--output_file', str(self.output_dir / 'candidate_subgraphs.json'),
            '--log_file', str(self.output_dir / 'hybrid_retrieval.log')
        ]
        
        # 添加额外参数
        for key, value in stage_config.items():
            if key not in ['eval_split', 'batch_size'] and not key.startswith('_'):
                if isinstance(value, bool) and value:
                    cmd.append(f'--{key}')
                elif not isinstance(value, bool):
                    cmd.extend([f'--{key}', str(value)])
        
        logging.info(f"执行命令: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
            logging.info("混合检索完成，候选子图已保存")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"混合检索阶段失败: {e}")
            return False
    
    def run_gnn_reasoning(self, stage_config: Dict) -> bool:
        """运行GNN推理阶段"""
        logging.info("\n=== 阶段2: GNN推理 + DDE增强 ===")
        
        cmd = [
            'python', '-m', 'gnn.inference',
            *self.get_common_args(),
            *self.get_rag3_args(),
            '--stage', 'gnn_reasoning',
            '--eval_split', stage_config.get('eval_split', 'test'),
            '--batch_size', str(stage_config.get('batch_size', 8)),
            '--input_file', str(self.output_dir / 'candidate_subgraphs.json'),
            '--output_file', str(self.output_dir / 'gnn_predictions.json'),
            '--log_file', str(self.output_dir / 'gnn_reasoning.log'),
            '--compute_distances',
            '--apply_dde'
        ]
        
        # 添加额外参数
        for key, value in stage_config.items():
            if key not in ['eval_split', 'batch_size'] and not key.startswith('_'):
                if isinstance(value, bool) and value:
                    cmd.append(f'--{key}')
                elif not isinstance(value, bool):
                    cmd.extend([f'--{key}', str(value)])
        
        logging.info(f"执行命令: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
            logging.info("GNN推理完成，预测结果已保存")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"GNN推理阶段失败: {e}")
            return False
    
    def run_pcst_selection(self, stage_config: Dict) -> bool:
        """运行PCST证据选择阶段"""
        logging.info("\n=== 阶段3: PCST证据选择 ===")
        
        cmd = [
            'python', '-m', 'gnn.inference',
            *self.get_common_args(),
            *self.get_rag3_args(),
            '--stage', 'pcst_selection',
            '--eval_split', stage_config.get('eval_split', 'test'),
            '--batch_size', str(stage_config.get('batch_size', 8)),
            '--input_file', str(self.output_dir / 'gnn_predictions.json'),
            '--output_file', str(self.output_dir / 'pcst_evidence.json'),
            '--log_file', str(self.output_dir / 'pcst_selection.log'),
            '--compute_connectivity',
            '--compute_kg_tokens'
        ]
        
        # 添加额外参数
        for key, value in stage_config.items():
            if key not in ['eval_split', 'batch_size'] and not key.startswith('_'):
                if isinstance(value, bool) and value:
                    cmd.append(f'--{key}')
                elif not isinstance(value, bool):
                    cmd.extend([f'--{key}', str(value)])
        
        logging.info(f"执行命令: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
            logging.info("PCST证据选择完成")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"PCST证据选择阶段失败: {e}")
            return False
    
    def run_llm_reasoning(self, stage_config: Dict) -> bool:
        """运行LLM推理阶段"""
        logging.info("\n=== 阶段4: LLM推理 ===")
        
        cmd = [
            'python', '-m', 'llm.src.qa_prediction.predict',
            '--dataset', self.dataset,
            '--input_file', str(self.output_dir / 'pcst_evidence.json'),
            '--output_file', str(self.output_dir / 'final_predictions.json'),
            '--log_file', str(self.output_dir / 'llm_reasoning.log')
        ]
        
        # LLM特定参数
        llm_config = stage_config.get('llm_params', {})
        for key, value in llm_config.items():
            if isinstance(value, bool) and value:
                cmd.append(f'--{key}')
            elif not isinstance(value, bool):
                cmd.extend([f'--{key}', str(value)])
        
        logging.info(f"执行命令: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
            logging.info("LLM推理完成")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"LLM推理阶段失败: {e}")
            return False
    
    def inference(self, stages: Optional[List[str]] = None):
        """执行推理"""
        # 默认推理阶段配置
        default_stages = {
            'hybrid': {
                'batch_size': 16,
                'eval_split': 'test'
            },
            'gnn': {
                'batch_size': 8,
                'eval_split': 'test'
            },
            'pcst': {
                'batch_size': 8,
                'eval_split': 'test'
            },
            'llm': {
                'llm_params': {
                    'model_name': 'llama2-7b',
                    'max_length': 2048,
                    'temperature': 0.1,
                    'top_p': 0.9
                }
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
            stages = ['hybrid', 'gnn', 'pcst', 'llm']
        
        # 阶段映射
        stage_functions = {
            'hybrid': self.run_hybrid_retrieval,
            'gnn': self.run_gnn_reasoning,
            'pcst': self.run_pcst_selection,
            'llm': self.run_llm_reasoning
        }
        
        # 执行推理阶段
        for stage_name in stages:
            if stage_name not in stage_functions:
                logging.error(f"未知阶段: {stage_name}")
                return False
            
            success = stage_functions[stage_name](stage_configs[stage_name])
            if not success:
                logging.error(f"推理在阶段 {stage_name} 失败")
                return False
        
        # 最终评估
        self.final_evaluation()
        return True
    
    def final_evaluation(self):
        """最终评估"""
        logging.info("\n=== 最终评估 ===")
        
        # 检查最终预测文件
        final_predictions = self.output_dir / 'final_predictions.json'
        if not final_predictions.exists():
            logging.warning("最终预测文件不存在，跳过评估")
            return
        
        cmd = [
            'python', '-m', 'gnn.evaluate',
            '--dataset', self.dataset,
            '--predictions_file', str(final_predictions),
            '--output_file', str(self.output_dir / 'evaluation_results.json'),
            '--compute_metrics'
        ]
        
        try:
            subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
            logging.info("最终评估完成")
            self.generate_report()
        except subprocess.CalledProcessError as e:
            logging.error(f"最终评估失败: {e}")
    
    def generate_report(self):
        """生成推理报告"""
        logging.info("\n=== 推理报告 ===")
        
        # 检查各阶段输出文件
        output_files = {
            'candidate_subgraphs.json': '候选子图',
            'gnn_predictions.json': 'GNN预测',
            'pcst_evidence.json': 'PCST证据',
            'final_predictions.json': '最终预测',
            'evaluation_results.json': '评估结果'
        }
        
        for filename, description in output_files.items():
            filepath = self.output_dir / filename
            if filepath.exists():
                logging.info(f'{description}: ✓ {filepath}')
            else:
                logging.info(f'{description}: ✗ 文件不存在')
        
        # 显示评估结果
        results_file = self.output_dir / 'evaluation_results.json'
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                logging.info(f"\n最终指标:")
                logging.info(f'Hit@10: {results.get("hit_10", "N/A")}')
                logging.info(f'F1: {results.get("f1", "N/A")}')
                logging.info(f'EM: {results.get("exact_match", "N/A")}')
                logging.info(f'连通率: {results.get("connectivity", "N/A")}')
                logging.info(f'KG Tokens: {results.get("kg_tokens", "N/A")}')
            except Exception as e:
                logging.error(f"读取评估结果失败: {e}")

def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_default_config(dataset: str, checkpoint_path: str, gpu_id: int) -> Dict:
    """创建默认配置"""
    return {
        'dataset': dataset,
        'checkpoint_path': checkpoint_path,
        'gpu_id': gpu_id,
        'base_dir': str(PROJECT_ROOT),
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
    parser = argparse.ArgumentParser(description='GNN-RAG³ 推理脚本')
    parser.add_argument('--dataset', type=str, default='webqsp', 
                       choices=['webqsp', 'cwq'], help='数据集名称')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--stages', type=str, default='hybrid,gnn,pcst,llm', help='要运行的阶段，逗号分隔')
    parser.add_argument('--output_dir', type=str, help='输出目录')
    parser.add_argument('--batch_size', type=int, help='批处理大小')
    
    # GNN-RAG³ 特定参数
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
        config = create_default_config(args.dataset, args.checkpoint, args.gpu)
    
    # 覆盖命令行参数
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.batch_size:
        config.setdefault('stages', {})
        for stage in ['hybrid', 'gnn', 'pcst']:
            config['stages'].setdefault(stage, {})['batch_size'] = args.batch_size
    
    # 处理GNN-RAG³参数
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
    
    # 解析阶段
    stages = None
    if args.stages:
        stages = [s.strip() for s in args.stages.split(',')]
    
    # 创建推理器并开始推理
    try:
        inferencer = GNNRAGInferencer(config)
        success = inferencer.inference(stages)
        
        if success:
            logging.info("\n=== GNN-RAG3 推理完成 ===")
            logging.info(f"输出目录: {inferencer.output_dir}")
        else:
            logging.error("推理失败")
            sys.exit(1)
    except Exception as e:
        logging.error(f"推理初始化失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()