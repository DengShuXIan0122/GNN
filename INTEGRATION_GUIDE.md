# GNN-RAG³ Integration Guide

This guide provides detailed instructions for integrating and using the enhanced GNN-RAG³ system.

## 🏗️ Architecture Overview

GNN-RAG³ extends the original GNN-RAG with four key enhancements:

1. **Hybrid Retrieval**: Combines APPR and semantic similarity
2. **DDE (Directional Distance Encoding)**: Geometric bias injection
3. **PCST (Prize-Collecting Steiner Tree)**: Evidence graph regularization
4. **Intelligent Routing**: Adaptive strategy selection

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (for GPU acceleration)
- 16GB+ RAM recommended
- 50GB+ disk space for datasets

### Dependencies
```bash
# Core dependencies (updated for GNN-RAG³)
pip install torch>=1.12.0 torchvision torchaudio
pip install torch-geometric>=2.0.0
pip install transformers>=4.20.0
pip install networkx>=2.8
pip install scipy>=1.8.0
pip install scikit-learn>=1.1.0
pip install numpy>=1.21.0
pip install tqdm>=4.64.0
pip install wandb>=0.12.0  # for experiment tracking

# GNN-RAG³ specific dependencies
pip install openai>=0.27.0  # for LLM integration
pip install anthropic>=0.3.0  # for Claude integration
pip install faiss-cpu>=1.7.0  # for semantic similarity
pip install sentence-transformers>=2.2.0  # for embeddings
```

## 🚀 Step-by-Step Integration

### 1. Environment Setup

```bash
# Clone and setup
git clone <repository-url>
cd GNN-RAG-main

# Create virtual environment
python -m venv gnn_rag3_env
source gnn_rag3_env/bin/activate  # Linux/Mac
# or
gnn_rag3_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Download datasets
cd gnn/data
wget <dataset-urls>
unzip *.zip

# Preprocess knowledge graphs
python preprocess_kg.py --dataset webqsp
python preprocess_kg.py --dataset cwq
```

### 3. Model Configuration

Create a configuration file `config/gnn_rag3.yaml`:

```yaml
# GNN-RAG³ Configuration
model:
  name: "rearev_gnn_rag3"
  
  # Core GNN settings
  num_gnn: 4
  entity_dim: 768
  relation_dim: 768
  
  # GNN-RAG³ enhancements
  use_appr: true           # Enable Approximate Personalized PageRank
  appr_alpha: 0.85         # APPR damping factor (0.0-1.0)
  cand_n: 1200            # Candidate subgraph size
  
  use_dde: true           # Enable Directional Distance Encoding
  hop_dim: 16             # Hop distance embedding dimension
  dir_dim: 8              # Direction embedding dimension
  
  use_pcst: true          # Enable Prize-Collecting Steiner Tree
  pcst_lambda: [0.1, 0.1, 0.05]  # PCST loss weights [warmup, joint1, joint2]
  gumbel_temp: 1.0        # Gumbel softmax temperature
  
  mid_restart: true       # Enable mid-layer restart at L/2

training:
  # 4-stage training
  stages:
    warmup:
      steps: 5000
      lr: 1e-4
      features: ["appr"]
    joint1:
      steps: 20000
      lr: 5e-5
      features: ["appr", "dde", "mid_restart"]
    pcst_distill:
      steps: 10000
      lr: 1e-5
      features: ["pcst"]
    joint2:
      steps: 20000
      lr: 1e-5
      features: ["all"]
      
  batch_size: 16
  gradient_clip: 1.0
  warmup_steps: 1000

routing:
  complexity_threshold_low: 0.3    # Threshold for simple questions (use LONG_CONTEXT)
  complexity_threshold_high: 0.7   # Threshold for complex questions (use GNN)
  confidence_threshold: 0.8        # Minimum confidence for route selection
  
  # Route-specific configurations
  gnn_route:
    max_hops: 4                     # Maximum reasoning hops for GNN
    beam_size: 5                    # Beam search width
    use_appr: true                  # Enable APPR for GNN route
    use_dde: true                   # Enable DDE for GNN route
    use_pcst: true                  # Enable PCST for GNN route
    
  long_context_route:
    max_tokens: 4096                # Maximum context length
    use_retrieval: false            # Disable additional retrieval
```

### 4. Training Pipeline

#### Option 1: Automated Training (Recommended)
```bash
# One-click training with all stages
python train_gnn_rag.py --dataset webqsp --gpu 0

# Using configuration file
python train_gnn_rag.py --config configs/webqsp_train_config.json

# Training specific stages only
python train_gnn_rag.py --dataset webqsp --stages warmup,joint1,joint2
```

#### Option 2: Manual Stage-by-Stage Training
```bash
# Stage 1: Warm-up Training
python train_gnn_rag.py \
    --dataset webqsp \
    --gpu 0 \
    --stages warmup \
    --max_steps 5000 \
    --learning_rate 1e-4

# Stage 2: Joint-1 Training  
python train_gnn_rag.py \
    --dataset webqsp \
    --gpu 0 \
    --stages joint1 \
    --checkpoint outputs/gnn_rag3_webqsp/warmup_best.pt \
    --max_steps 20000 \
    --learning_rate 5e-5

# Stage 3: PCST Distillation
python train_gnn_rag.py \
    --dataset webqsp \
    --gpu 0 \
    --stages pcst_distill \
    --checkpoint outputs/gnn_rag3_webqsp/joint1_best.pt \
    --max_steps 10000 \
    --learning_rate 1e-5

# Stage 4: Joint-2 Training
python train_gnn_rag.py \
    --dataset webqsp \
    --gpu 0 \
    --stages joint2 \
    --checkpoint outputs/gnn_rag3_webqsp/pcst_distill_best.pt \
    --max_steps 20000 \
    --learning_rate 1e-5
```

### 5. Inference Setup

#### Option 1: Automated Inference (Recommended)
```bash
# Complete inference pipeline
python infer_gnn_rag.py --dataset webqsp --checkpoint outputs/gnn_rag3_webqsp/joint2_best.pt --gpu 0

# Using configuration file
python infer_gnn_rag.py --config configs/webqsp_inference_config.json

# Running specific stages only
python infer_gnn_rag.py --dataset webqsp --checkpoint model.pt --stages hybrid,gnn,pcst
```

#### Option 2: Programmatic Inference
```python
from gnn.models.ReaRev.rearev import ReaRev
from llm.router.route import route_question, route_question_complex
import torch

# Load trained model
model = ReaRev.load_from_checkpoint("outputs/gnn_rag3_webqsp/joint2_best.pt")
model.eval()

# Example inference
question = "Who directed the movie that won Oscar for Best Picture in 2020?"

# 1. Simple route selection (GNN vs Long Context)
route_decision = route_question(question)
print(f"Selected route: {route_decision.route_type.value}")

# 2. Complex route selection (detailed strategy)
complex_decision = route_question_complex(question)
print(f"Complex route: {complex_decision.route_type.value}")
print(f"Confidence: {complex_decision.confidence:.3f}")

# 3. Configure model based on route
model.configure_for_route(complex_decision)

# 4. Run inference
with torch.no_grad():
    results = model.inference(
        question=question,
        kg_data=kg_data,
        use_appr=complex_decision.use_appr,
        use_dde=complex_decision.use_dde,
        use_pcst=complex_decision.use_pcst
    )

print(f"Answer: {results['answer']}")
print(f"Reasoning path: {results['path']}")
print(f"Confidence: {results['confidence']}")
```

## 🔧 Component Integration

### Hybrid Retrieval Integration

```python
from gnn.retrieval.hybrid_retriever import HybridRetriever

# Initialize hybrid retriever
retriever = HybridRetriever(
    kg_embeddings=kg_embeddings,
    appr_alpha=0.85,
    semantic_weight=0.3
)

# Retrieve candidates
candidates = retriever.retrieve(
    query_entities=query_entities,
    question_embedding=question_emb,
    top_k=1200
)
```

### DDE Layer Integration

```python
from gnn.layers.dde import DDELayer

# Initialize DDE layer
dde_layer = DDELayer(
    hop_dim=16,
    dir_dim=8,
    entity_dim=768
)

# Compute geometric gates
distances = compute_distances_bfs(kg_adj, query_entities)
geo_gates = dde_layer(distances, directions)
```

### PCST Loss Integration

```python
from gnn.losses.pcst_loss import PCSTLoss

# Initialize PCST loss
pcst_loss = PCSTLoss(lambda_val=0.1)

# Compute PCST regularization
edge_probs = model.get_edge_probabilities()
edge_costs = model.get_edge_costs()
pcst_reg = pcst_loss.pcst_soft_regularizer(edge_probs, edge_costs)
```

### Router Integration

```python
from llm.router.route import GNNRAGRouter, RouteOptimizer

# Initialize router
router = GNNRAGRouter(
    complexity_analyzer=complexity_analyzer,
    route_predictor=route_predictor
)

# Initialize optimizer
optimizer = RouteOptimizer()

# Route and optimize
decision = router.route(question)
optimized_decision = optimizer.optimize_route_decision(decision, question)
```

## 📊 Monitoring and Evaluation

### Training Monitoring

```python
import wandb

# Initialize tracking
wandb.init(project="gnn-rag3", config=config)

# Log metrics during training
wandb.log({
    "stage": current_stage,
    "step": step,
    "loss": loss.item(),
    "hit_at_10": hit_at_10,
    "multi_hop_f1": multi_hop_f1,
    "pcst_accuracy": pcst_acc,
    "connectivity_rate": conn_rate
})
```

### Evaluation Scripts

```bash
# Evaluate on test set
python evaluate.py \
    --checkpoint checkpoints/final/best.pt \
    --dataset webqsp \
    --split test \
    --output_file results/webqsp_test.json

# Generate evaluation report
python generate_report.py \
    --results_file results/webqsp_test.json \
    --output_dir reports/
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. Memory Issues
```python
# Reduce batch size
config.training.batch_size = 8

# Enable gradient checkpointing
config.model.gradient_checkpointing = True

# Use mixed precision
config.training.mixed_precision = True
```

#### 2. PCST Solver Timeout
```python
# Increase timeout
config.pcst.timeout = 30.0

# Use heuristic fallback
config.pcst.use_heuristic_fallback = True
```

#### 3. DDE Dimension Mismatch
```python
# Ensure compatibility
assert config.model.hop_dim % config.model.num_heads == 0
assert config.model.dir_dim <= config.model.entity_dim
```

#### 4. Route Confidence Issues
```python
# Retrain router with more data
python train_router.py --data_size 50000

# Adjust thresholds
config.routing.confidence_threshold = 0.6
```

### Debug Mode

```bash
# Enable debug logging
export GNN_RAG_DEBUG=1
export CUDA_LAUNCH_BLOCKING=1

# Run with detailed logs
python train_model.py --debug --log_level DEBUG
```

## 🔄 Migration from Original GNN-RAG

### Code Migration

1. **Update imports**:
```python
# Old
from gnn.models.ReaRev.rearev import ReaRev

# New
from gnn.models.ReaRev.rearev import ReaRev  # Enhanced version
```

2. **Update configuration**:
```python
# Add GNN-RAG³ parameters
config.update({
    'use_appr': True,
    'use_dde': True,
    'use_pcst': True,
    'mid_restart': True
})
```

3. **Update training loop**:
```python
# Add stage-based training
for stage in ['warmup', 'joint1', 'pcst_distill', 'joint2']:
    train_stage(model, stage, config)
```

### Data Migration

```bash
# Convert old format to new format
python migrate_data.py \
    --input_dir old_data/ \
    --output_dir new_data/ \
    --format gnn_rag3
```

## 📈 Performance Optimization

### GPU Optimization
```python
# Enable optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Use DataParallel for multiple GPUs
model = torch.nn.DataParallel(model)
```

### Memory Optimization
```python
# Gradient accumulation
config.training.gradient_accumulation_steps = 4

# Dynamic batching
config.training.dynamic_batching = True
```

### Inference Optimization
```python
# Model quantization
model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear})

# TensorRT optimization (if available)
model = torch.jit.script(model)
```

## 🚀 Production Deployment

### Docker Setup

```dockerfile
FROM pytorch/pytorch:1.12.1-cuda11.3-devel

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "serve.py"]
```

### API Server

```python
from fastapi import FastAPI
from gnn_rag3 import GNNRAGModel

app = FastAPI()
model = GNNRAGModel.load("checkpoints/final/best.pt")

@app.post("/answer")
async def answer_question(question: str):
    result = model.answer(question)
    return {
        "answer": result.answer,
        "confidence": result.confidence,
        "reasoning_path": result.path
    }
```

This integration guide provides a comprehensive roadmap for implementing and deploying GNN-RAG³ in your environment. For additional support, please refer to the troubleshooting section or open an issue in the repository.