#!/bin/bash

# GNN-RAG³ 推理脚本
# 使用方法: bash infer_gnn_rag3.sh [dataset] [checkpoint_path] [gpu_id]

DATASET=${1:-"webqsp"}
CHECKPOINT_PATH=${2:-""}
GPU_ID=${3:-0}
BASE_DIR="."
DATA_DIR="${BASE_DIR}/gnn/data/${DATASET}"
OUTPUT_DIR="${BASE_DIR}/gnn/outputs/inference_${DATASET}"

# 检查checkpoint路径
if [ -z "${CHECKPOINT_PATH}" ]; then
    echo "错误: 请提供checkpoint路径"
    echo "使用方法: bash infer_gnn_rag3.sh [dataset] [checkpoint_path] [gpu_id]"
    exit 1
fi

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "错误: checkpoint文件不存在: ${CHECKPOINT_PATH}"
    exit 1
fi

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

echo "=== GNN-RAG³ 推理开始 ==="
echo "数据集: ${DATASET}"
echo "模型: ${CHECKPOINT_PATH}"
echo "GPU: ${GPU_ID}"
echo "输出目录: ${OUTPUT_DIR}"

# 基础参数
COMMON_ARGS="--dataset ${DATASET} --gpu ${GPU_ID} --data_folder ${DATA_DIR} --output_dir ${OUTPUT_DIR}"

# GNN-RAG³ 特定参数
RAG3_ARGS="--use_appr --appr_alpha 0.85 --cand_n 1200 --use_dde --hop_dim 16 --dir_dim 8 --use_pcst --pcst_lambda 0.1,0.1,0.05 --mid_restart"

# ============================================================================
# 阶段1: 混合检索 + 候选子图构建
# ============================================================================
echo ""
echo "=== 阶段1: 混合检索 + 候选子图构建 ==="
python -m gnn.inference \
    ${COMMON_ARGS} \
    ${RAG3_ARGS} \
    --load_checkpoint ${CHECKPOINT_PATH} \
    --stage hybrid_retrieval \
    --eval_split test \
    --batch_size 16 \
    --output_file ${OUTPUT_DIR}/candidate_subgraphs.json \
    --log_file ${OUTPUT_DIR}/hybrid_retrieval.log

if [ $? -ne 0 ]; then
    echo "混合检索阶段失败，退出"
    exit 1
fi

echo "混合检索完成，候选子图已保存"

# ============================================================================
# 阶段2: GNN推理 + DDE增强
# ============================================================================
echo ""
echo "=== 阶段2: GNN推理 + DDE增强 ==="
python -m gnn.inference \
    ${COMMON_ARGS} \
    ${RAG3_ARGS} \
    --load_checkpoint ${CHECKPOINT_PATH} \
    --stage gnn_reasoning \
    --eval_split test \
    --batch_size 8 \
    --input_file ${OUTPUT_DIR}/candidate_subgraphs.json \
    --output_file ${OUTPUT_DIR}/gnn_predictions.json \
    --log_file ${OUTPUT_DIR}/gnn_reasoning.log \
    --compute_distances \
    --apply_dde

if [ $? -ne 0 ]; then
    echo "GNN推理阶段失败，退出"
    exit 1
fi

echo "GNN推理完成，预测结果已保存"

# ============================================================================
# 阶段3: PCST证据选择（可选）
# ============================================================================
echo ""
echo "=== 阶段3: PCST证据选择 ==="
python -m gnn.inference \
    ${COMMON_ARGS} \
    ${RAG3_ARGS} \
    --load_checkpoint ${CHECKPOINT_PATH} \
    --stage pcst_selection \
    --eval_split test \
    --batch_size 8 \
    --input_file ${OUTPUT_DIR}/gnn_predictions.json \
    --output_file ${OUTPUT_DIR}/pcst_evidence.json \
    --log_file ${OUTPUT_DIR}/pcst_selection.log \
    --apply_pcst \
    --cumulative_prob_threshold 0.95

if [ $? -ne 0 ]; then
    echo "PCST证据选择失败，使用原始预测"
    cp ${OUTPUT_DIR}/gnn_predictions.json ${OUTPUT_DIR}/pcst_evidence.json
fi

echo "PCST证据选择完成"

# ============================================================================
# 阶段4: 路径抽取 + 文本化
# ============================================================================
echo ""
echo "=== 阶段4: 路径抽取 + 文本化 ==="
python -m gnn.export_paths \
    ${COMMON_ARGS} \
    --input_file ${OUTPUT_DIR}/pcst_evidence.json \
    --output_file ${OUTPUT_DIR}/reasoning_paths.json \
    --extract_shortest_paths \
    --verbalize_paths \
    --max_path_length 5 \
    --log_file ${OUTPUT_DIR}/path_extraction.log

if [ $? -ne 0 ]; then
    echo "路径抽取失败，退出"
    exit 1
fi

echo "路径抽取完成"

# ============================================================================
# 最终评估和统计
# ============================================================================
echo ""
echo "=== 最终评估和统计 ==="
python -m gnn.evaluate \
    ${COMMON_ARGS} \
    ${RAG3_ARGS} \
    --load_checkpoint ${CHECKPOINT_PATH} \
    --eval_split test \
    --input_file ${OUTPUT_DIR}/reasoning_paths.json \
    --output_file ${OUTPUT_DIR}/final_metrics.json \
    --compute_connectivity \
    --compute_kg_tokens \
    --compute_hit_at_k 1,5,10 \
    --compute_f1

echo ""
echo "=== GNN-RAG³ 推理完成 ==="
echo "候选子图: ${OUTPUT_DIR}/candidate_subgraphs.json"
echo "GNN预测: ${OUTPUT_DIR}/gnn_predictions.json"
echo "PCST证据: ${OUTPUT_DIR}/pcst_evidence.json"
echo "推理路径: ${OUTPUT_DIR}/reasoning_paths.json"
echo "最终指标: ${OUTPUT_DIR}/final_metrics.json"

# 生成推理报告
python -c "
import json
import os

output_dir = '${OUTPUT_DIR}'
files = ['candidate_subgraphs.json', 'gnn_predictions.json', 'pcst_evidence.json', 'reasoning_paths.json', 'final_metrics.json']

print('\\n=== 推理报告 ===')
for file in files:
    file_path = os.path.join(output_dir, file)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f'{file}: {len(data)} 条记录')
    else:
        print(f'{file}: 文件缺失')

# 显示最终指标
metrics_file = os.path.join(output_dir, 'final_metrics.json')
if os.path.exists(metrics_file):
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    print(f'\\n=== 最终指标 ===')
    print(f'Hit@1: {metrics.get(\"hit_1\", \"N/A\")}')
    print(f'Hit@5: {metrics.get(\"hit_5\", \"N/A\")}')
    print(f'Hit@10: {metrics.get(\"hit_10\", \"N/A\")}')
    print(f'F1: {metrics.get(\"f1\", \"N/A\")}')
    print(f'连通率: {metrics.get(\"connectivity_rate\", \"N/A\")}')
    print(f'平均KG Tokens: {metrics.get(\"avg_kg_tokens\", \"N/A\")}')
    print(f'推理时间(秒): {metrics.get(\"inference_time\", \"N/A\")}')
"

echo ""
echo "推理完成！结果已保存到 ${OUTPUT_DIR}"