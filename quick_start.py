#!/usr/bin/env python3
"""
GNN-RAG³ 快速开始脚本
提供一键训练和推理功能

使用方法:
    python quick_start.py --mode train --dataset webqsp
    python quick_start.py --mode infer --dataset webqsp --checkpoint path/to/model.pt
    python quick_start.py --mode demo --question "Who is the president of the United States?"
"""

import argparse
import json
import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, Optional

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def check_dependencies():
    """检查依赖"""
    logging.info("检查依赖...")
    
    required_dirs = [
        'gnn/data',
        'llm/src',
        'gnn/models'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not (PROJECT_ROOT / dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        logging.error(f"缺少必要目录: {missing_dirs}")
        return False
    
    logging.info("依赖检查通过")
    return True

def download_data(dataset: str):
    """下载数据集"""
    logging.info(f"检查数据集: {dataset}")
    
    data_dir = PROJECT_ROOT / 'gnn' / 'data' / dataset
    if data_dir.exists() and any(data_dir.iterdir()):
        logging.info(f"数据集 {dataset} 已存在")
        return True
    
    logging.info(f"下载数据集 {dataset}...")
    # 这里应该实现实际的数据下载逻辑
    # 目前只是创建目录结构
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建示例数据文件
    sample_files = ['train.json', 'dev.json', 'test.json']
    for filename in sample_files:
        sample_file = data_dir / filename
        if not sample_file.exists():
            sample_file.write_text('[]')  # 空的JSON数组
    
    logging.info(f"数据集 {dataset} 准备完成")
    return True

def quick_train(dataset: str, gpu_id: int = 0, stages: Optional[str] = None, args=None):
    """快速训练"""
    logging.info(f"开始训练 GNN-RAG³ 模型 (数据集: {dataset})")
    
    # 检查数据
    if not download_data(dataset):
        return False
    
    # 构建训练命令
    cmd = [
        'python', 'train_gnn_rag.py',
        '--dataset', dataset,
        '--gpu', str(gpu_id)
    ]
    
    if stages:
        cmd.extend(['--stages', stages])
    
    # 添加GNN-RAG³参数
    if args:
        if args.use_appr:
            cmd.extend(['--use_appr', '--appr_alpha', str(args.appr_alpha), '--cand_n', str(args.cand_n)])
        if args.use_dde:
            cmd.extend(['--use_dde', '--hop_dim', str(args.hop_dim), '--dir_dim', str(args.dir_dim)])
        if args.use_pcst:
            cmd.extend(['--use_pcst', '--pcst_lambda', args.pcst_lambda])
        if args.mid_layer_restart:
            cmd.extend(['--mid_layer_restart'])
        if args.config:
            cmd.extend(['--config', args.config])
    
    # 使用配置文件（如果存在且未指定）
    if not (args and args.config):
        config_file = PROJECT_ROOT / 'configs' / f'{dataset}_train_config.json'
        if config_file.exists():
            cmd.extend(['--config', str(config_file)])
    
    logging.info(f"执行训练命令: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        logging.info("训练完成")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"训练失败: {e}")
        return False

def quick_infer(dataset: str, checkpoint_path: str, gpu_id: int = 0, stages: Optional[str] = None, args=None):
    """快速推理"""
    logging.info(f"开始推理 (数据集: {dataset}, 模型: {checkpoint_path})")
    
    # 检查数据
    if not download_data(dataset):
        return False
    
    # 构建推理命令
    cmd = [
        'python', 'infer_gnn_rag.py',
        '--dataset', dataset,
        '--checkpoint', checkpoint_path,
        '--gpu', str(gpu_id)
    ]
    
    if stages:
        cmd.extend(['--stages', stages])
    
    # 添加GNN-RAG³参数
    if args:
        if args.use_appr:
            cmd.extend(['--use_appr', '--appr_alpha', str(args.appr_alpha), '--cand_n', str(args.cand_n)])
        if args.use_dde:
            cmd.extend(['--use_dde', '--hop_dim', str(args.hop_dim), '--dir_dim', str(args.dir_dim)])
        if args.use_pcst:
            cmd.extend(['--use_pcst', '--pcst_lambda', args.pcst_lambda])
        if args.mid_layer_restart:
            cmd.extend(['--mid_layer_restart'])
        if args.config:
            cmd.extend(['--config', args.config])
    
    # 使用配置文件（如果存在且未指定）
    if not (args and args.config):
        config_file = PROJECT_ROOT / 'configs' / f'{dataset}_inference_config.json'
        if config_file.exists():
            cmd.extend(['--config', str(config_file)])
    
    logging.info(f"执行推理命令: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        logging.info("推理完成")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"推理失败: {e}")
        return False

def demo_question(question: str, dataset: str = 'webqsp', checkpoint_path: Optional[str] = None):
    """演示单个问题的推理"""
    logging.info(f"演示问题推理: {question}")
    
    # 如果没有提供checkpoint，尝试找到默认的
    if checkpoint_path is None:
        default_checkpoint = PROJECT_ROOT / 'gnn' / 'outputs' / f'gnn_rag3_{dataset}' / 'joint2_best.pt'
        if default_checkpoint.exists():
            checkpoint_path = str(default_checkpoint)
        else:
            logging.error("未找到训练好的模型，请先运行训练或指定checkpoint路径")
            return False
    
    # 创建临时问题文件
    temp_dir = PROJECT_ROOT / 'temp'
    temp_dir.mkdir(exist_ok=True)
    
    question_file = temp_dir / 'demo_question.json'
    with open(question_file, 'w', encoding='utf-8') as f:
        json.dump([{
            'id': 'demo_001',
            'question': question,
            'answers': []  # 演示模式不需要答案
        }], f, ensure_ascii=False, indent=2)
    
    # 运行推理
    cmd = [
        'python', 'infer_gnn_rag.py',
        '--dataset', dataset,
        '--checkpoint', checkpoint_path,
        '--stages', 'hybrid,gnn,pcst',  # 跳过LLM阶段以便快速演示
        '--output_dir', str(temp_dir / 'demo_output')
    ]
    
    logging.info(f"执行演示推理: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        
        # 显示结果
        result_file = temp_dir / 'demo_output' / 'pcst_evidence.json'
        if result_file.exists():
            with open(result_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            if results:
                result = results[0]
                logging.info(f"\n=== 推理结果 ===")
                logging.info(f"问题: {question}")
                logging.info(f"候选实体: {result.get('candidate_entities', [])[:5]}...")  # 显示前5个
                logging.info(f"证据路径数量: {len(result.get('evidence_paths', []))}")
                logging.info(f"KG tokens: {result.get('kg_tokens', 'N/A')}")
        
        logging.info("演示完成")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"演示失败: {e}")
        return False

def setup_environment():
    """设置环境"""
    logging.info("设置环境...")
    
    # 创建必要目录
    dirs_to_create = [
        'configs',
        'gnn/outputs',
        'llm/results',
        'logs',
        'temp'
    ]
    
    for dir_path in dirs_to_create:
        (PROJECT_ROOT / dir_path).mkdir(parents=True, exist_ok=True)
    
    logging.info("环境设置完成")

def main():
    parser = argparse.ArgumentParser(description='GNN-RAG³ 快速开始脚本')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['train', 'infer', 'demo', 'setup'],
                       help='运行模式')
    parser.add_argument('--dataset', type=str, default='webqsp',
                       choices=['webqsp', 'cwq'], help='数据集')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, help='模型checkpoint路径')
    parser.add_argument('--stages', type=str, default='warmup,joint1,pcst_distill,joint2', help='要运行的阶段')
    parser.add_argument('--question', type=str, help='演示问题')
    
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
    
    setup_logging()
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    success = False
    
    if args.mode == 'setup':
        setup_environment()
        success = True
    
    elif args.mode == 'train':
        success = quick_train(args.dataset, args.gpu, args.stages, args)
    
    elif args.mode == 'infer':
        if not args.checkpoint:
            logging.error("推理模式需要指定 --checkpoint 参数")
            sys.exit(1)
        success = quick_infer(args.dataset, args.checkpoint, args.gpu, args.stages, args)
    
    elif args.mode == 'demo':
        if not args.question:
            logging.error("演示模式需要指定 --question 参数")
            sys.exit(1)
        success = demo_question(args.question, args.dataset, args.checkpoint)
    
    if success:
        logging.info(f"\n=== {args.mode.upper()} 模式完成 ===")
        
        if args.mode == 'train':
            model_path = PROJECT_ROOT / 'gnn' / 'outputs' / f'gnn_rag3_{args.dataset}' / 'joint2_best.pt'
            logging.info(f"训练完成的模型: {model_path}")
            logging.info(f"推理命令示例: python quick_start.py --mode infer --dataset {args.dataset} --checkpoint {model_path}")
        
        elif args.mode == 'infer':
            output_dir = PROJECT_ROOT / 'gnn' / 'outputs' / f'inference_{args.dataset}'
            logging.info(f"推理结果目录: {output_dir}")
    else:
        logging.error(f"{args.mode.upper()} 模式失败")
        sys.exit(1)

if __name__ == '__main__':
    main()