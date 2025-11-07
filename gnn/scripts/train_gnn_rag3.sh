#!/bin/bash

# GNN-RAG³ 四阶段训练脚本
# 使用方法: bash train_gnn_rag3.sh [dataset] [gpu_id]

DATASET=${1:-"webqsp"}
GPU_ID=${2:-0}
BASE_DIR="."
DATA_DIR="${BASE_DIR}/gnn/data/${DATASET}"
OUTPUT_DIR="${BASE_DIR}/gnn/outputs/gnn_rag3_${DATASET}"

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

echo "=== GNN-RAG³ 四阶段训练开始 ==="
echo "数据集: ${DATASET}"
echo "GPU: ${GPU_ID}"
echo "输出目录: ${OUTPUT_DIR}"

# 基础参数
COMMON_ARGS="--dataset ${DATASET} --gpu ${GPU_ID} --data_folder ${DATA_DIR} --checkpoint_dir ${OUTPUT_DIR}"

# GNN-RAG³ 特定参数
RAG3_ARGS="--use_appr --appr_alpha 0.85 --cand_n 1200 --use_dde --hop_dim 16 --dir_dim 8 --use_pcst --pcst_lambda 0.1,0.1,0.05 --mid_restart"

# ============================================================================
# 阶段1: Warm-up (5k steps) - 训练混合检索
# ============================================================================
echo ""
echo "=== 阶段1: Warm-up (5k steps) - 训练混合检索 ==="
python -m gnn.train_model \
    ${COMMON_ARGS} \
    ${RAG3_ARGS} \
    --stage warmup \
    --max_steps 5000 \
    --learning_rate 1e-3 \
    --freeze_gnn \
    --train_hybrid_only \
    --batch_size 16 \
    --eval_every 500 \
    --save_every 1000 \
    --log_file ${OUTPUT_DIR}/warmup.log \
    --checkpoint_prefix warmup

if [ $? -ne 0 ]; then
    echo "阶段1训练失败，退出"
    exit 1
fi

echo "阶段1完成，检查Hit@10指标..."

# ============================================================================
# 阶段2: Joint-1 (20k steps) - GNN + DDE，不开PCST
# ============================================================================
echo ""
echo "=== 阶段2: Joint-1 (20k steps) - GNN + DDE，不开PCST ==="
python -m gnn.train_model \
    ${COMMON_ARGS} \
    --use_appr --appr_alpha 0.85 --cand_n 1200 \
    --use_dde --hop_dim 16 --dir_dim 8 \
    --mid_restart \
    --stage joint1 \
    --max_steps 20000 \
    --learning_rate 5e-4 \
    --num_gnn_layers 6 \
    --hidden_dim 256 \
    --dropout 0.1 \
    --num_heads 3 \
    --batch_size 8 \
    --eval_every 2000 \
    --save_every 5000 \
    --log_file ${OUTPUT_DIR}/joint1.log \
    --checkpoint_prefix joint1 \
    --load_checkpoint ${OUTPUT_DIR}/warmup_best.pt

if [ $? -ne 0 ]; then
    echo "阶段2训练失败，退出"
    exit 1
fi

echo "阶段2完成，检查多跳题F1指标..."

# ============================================================================
# 阶段3: PCST Distill (10k steps) - PCST蒸馏训练
# ============================================================================
echo ""
echo "=== 阶段3: PCST Distill (10k steps) - PCST蒸馏训练 ==="
python -m gnn.train_model \
    ${COMMON_ARGS} \
    --use_appr --appr_alpha 0.85 --cand_n 1200 \
    --use_dde --hop_dim 16 --dir_dim 8 \
    --use_pcst --pcst_lambda 0.0,0.0,0.0 \
    --mid_restart \
    --stage pcst_distill \
    --max_steps 10000 \
    --learning_rate 1e-4 \
    --pcst_distill_only \
    --gumbel_temp 2.0 \
    --batch_size 8 \
    --eval_every 1000 \
    --save_every 2500 \
    --log_file ${OUTPUT_DIR}/pcst_distill.log \
    --checkpoint_prefix pcst_distill \
    --load_checkpoint ${OUTPUT_DIR}/joint1_best.pt

if [ $? -ne 0 ]; then
    echo "阶段3训练失败，退出"
    exit 1
fi

echo "阶段3完成，检查边选择器准确率..."

# ============================================================================
# 阶段4: Joint-2 (20k steps) - 启用软约束，温度退火
# ============================================================================
echo ""
echo "=== 阶段4: Joint-2 (20k steps) - 启用软约束，温度退火 ==="
python -m gnn.train_model \
    ${COMMON_ARGS} \
    ${RAG3_ARGS} \
    --stage joint2 \
    --max_steps 20000 \
    --learning_rate 2e-4 \
    --gumbel_temp_start 2.0 \
    --gumbel_temp_end 0.5 \
    --temp_anneal_steps 15000 \
    --lambda_ret 0.1 \
    --lambda_pcst 0.2 \
    --batch_size 8 \
    --eval_every 2000 \
    --save_every 5000 \
    --log_file ${OUTPUT_DIR}/joint2.log \
    --checkpoint_prefix joint2 \
    --load_checkpoint ${OUTPUT_DIR}/pcst_distill_best.pt

if [ $? -ne 0 ]; then
    echo "阶段4训练失败，退出"
    exit 1
fi

echo "阶段4完成，检查证据连通率和KG tokens数量..."

# ============================================================================
# 最终评估
# ============================================================================
echo ""
echo "=== 最终评估 ==="
python -m gnn.evaluate \
    ${COMMON_ARGS} \
    ${RAG3_ARGS} \
    --load_checkpoint ${OUTPUT_DIR}/joint2_best.pt \
    --eval_split test \
    --output_file ${OUTPUT_DIR}/final_results.json \
    --compute_connectivity \
    --compute_kg_tokens

echo ""
echo "=== GNN-RAG³ 四阶段训练完成 ==="
echo "最终模型: ${OUTPUT_DIR}/joint2_best.pt"
echo "结果文件: ${OUTPUT_DIR}/final_results.json"
echo "日志文件: ${OUTPUT_DIR}/*.log"

# 生成训练报告
python -c "
import json
import os

output_dir = '${OUTPUT_DIR}'
stages = ['warmup', 'joint1', 'pcst_distill', 'joint2']

print('\\n=== 训练报告 ===')
for stage in stages:
    log_file = os.path.join(output_dir, f'{stage}.log')
    if os.path.exists(log_file):
        print(f'{stage}: 日志文件存在')
    else:
        print(f'{stage}: 日志文件缺失')

if os.path.exists(os.path.join(output_dir, 'final_results.json')):
    with open(os.path.join(output_dir, 'final_results.json'), 'r') as f:
        results = json.load(f)
    print(f'\\n最终指标:')
    print(f'Hit@10: {results.get(\"hit_10\", \"N/A\")}')
    print(f'F1: {results.get(\"f1\", \"N/A\")}')
    print(f'连通率: {results.get(\"connectivity\", \"N/A\")}')
    print(f'KG Tokens: {results.get(\"kg_tokens\", \"N/A\")}')
"