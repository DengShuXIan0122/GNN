import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import sys
import os

try:
    from ..base_model import BaseModel
except ImportError:
    from models.base_model import BaseModel
try:
    from ...modules.kg_reasoning.reasongnn import ReasonGNNLayer
    from ...modules.question_encoding.lstm_encoder import LSTMInstruction
    from ...modules.question_encoding.bert_encoder import BERTInstruction
    from ...modules.layer_init import TypeLayer
    from ...modules.query_update import AttnEncoder, Fusion, QueryReform
except ImportError:
    from modules.kg_reasoning.reasongnn import ReasonGNNLayer
    from modules.question_encoding.lstm_encoder import LSTMInstruction
    from modules.question_encoding.bert_encoder import BERTInstruction
    from modules.layer_init import TypeLayer
    from modules.query_update import AttnEncoder, Fusion, QueryReform

# 添加新的DDE和混合检索模块
gnn_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if gnn_dir not in sys.path:
    sys.path.insert(0, gnn_dir)

from layers.dde import DDE, DDEMessagePassing
from retrieval.hybrid_retriever import HybridRetriever
from losses.pcst_loss import PCSTLoss
from utils.distance import compute_distances_bfs, compute_direction_encoding
from utils.ppr import compute_appr
from gnn.utils.appr_manager import APPRManager, get_appr_optimized

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000



class ReaRev(BaseModel):
    def __init__(self, args, num_entity, num_relation, num_word):
        """
        Init ReaRev model.
        """
        super(ReaRev, self).__init__(args, num_entity, num_relation, num_word)
        #self.embedding_def()
        #self.share_module_def()
        self.norm_rel = args['norm_rel']
        self.layers(args)
        

        self.loss_type =  args['loss_type']
        self.num_iter = args['num_iter']
        self.num_ins = args['num_ins']
        self.num_gnn = args['num_gnn']
        self.alg = args['alg']
        assert self.alg == 'bfs'
        self.lm = args['lm']
        
        # GNN-RAG³ 相关参数
        self.use_appr = args.get('use_appr', False)
        self.use_dde = args.get('use_dde', False)
        self.use_pcst = args.get('use_pcst', False)
        self.appr_alpha = args.get('appr_alpha', 0.85)
        self.cand_n = args.get('cand_n', 1200)
        self.hop_dim = args.get('hop_dim', 16)
        self.dir_dim = args.get('dir_dim', 8)
        pcst_lambda_raw = args.get('pcst_lambda', [0.1, 0.1, 0.05])
        # Handle string format from command line arguments
        if isinstance(pcst_lambda_raw, str):
            self.pcst_lambda = [float(x) for x in pcst_lambda_raw.split(',')]
        else:
            self.pcst_lambda = pcst_lambda_raw
        self.gumbel_temp = args.get('gumbel_temp', 2.0)
        self.mid_restart = args.get('mid_restart', True)
        # 新增：读取 warmup-only 与冻结标志
        self.train_hybrid_only = args.get('train_hybrid_only', False)
        self.freeze_gnn = args.get('freeze_gnn', False)
        
        self.private_module_def(args, num_entity, num_relation)

        self.to(self.device)
        self.lin = nn.Linear(3*self.entity_dim, self.entity_dim)

        self.fusion = Fusion(self.entity_dim)
        self.reforms = []
        for i in range(self.num_ins):
            self.add_module('reform' + str(i), QueryReform(self.entity_dim))
        
        # GNN-RAG³ 模块初始化
        if self.use_appr:
            self.hybrid_retriever = HybridRetriever(
                input_dim=self.entity_dim,
                appr_alpha=self.appr_alpha,
                max_candidates=self.cand_n
            )
            # 初始化APPRManager用于优化APPR计算
            self.appr_manager = APPRManager(
                alpha=self.appr_alpha,
                tolerance=1e-3,  # 放宽容差提升速度
                topk=500,  # 限制返回节点数
                cache_size=1024,
                enable_basis_vectors=True,
                enable_warm_start=True,
                enable_subgraph_mask=True
            )
        
        if self.use_dde:
            self.dde_layer = DDE(
                hop_dim=self.hop_dim,
                dir_dim=self.dir_dim,
                hidden_dim=self.entity_dim
            )
        
        if self.use_pcst:
            self.pcst_loss = PCSTLoss(
                lambda_cost=self.pcst_lambda[0],
                lambda_conn=self.pcst_lambda[1],
                lambda_sparse=self.pcst_lambda[2],
                temperature=self.gumbel_temp
            )
        # self.reform_rel = QueryReform(self.entity_dim)
        # self.add_module('reform', QueryReform(self.entity_dim))

    def private_module_def(self, args, num_entity, num_relation):
        """
        Building modules: LM encoder, GNN, etc.
        """
        # initialize entity embedding
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        entity_dim = self.entity_dim
        self.reasoning = ReasonGNNLayer(args, num_entity, num_relation, entity_dim, self.alg)
        # 新增：根据标志冻结GNN推理层参数
        if getattr(self, 'freeze_gnn', False):
            for p in self.reasoning.parameters():
                p.requires_grad = False
            self.reasoning.eval()
        if args['lm'] == 'lstm':
            self.instruction = LSTMInstruction(args, self.word_embedding, self.num_word)
            self.relation_linear = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        else:
            self.instruction = BERTInstruction(args, self.word_embedding, self.num_word, args['lm'])
            #self.relation_linear = nn.Linear(in_features=self.instruction.word_dim, out_features=entity_dim)
        # self.relation_linear = nn.Linear(in_features=entity_dim, out_features=entity_dim)
        # self.relation_linear_inv = nn.Linear(in_features=entity_dim, out_features=entity_dim)

    def layers(self, args):
        """
        Initialize projection layers and attention used by ReaRev.
        Mirrors NSM/GraftNet patterns to ensure compatibility.
        """
        entity_dim = self.entity_dim
        self.linear_dropout = args['linear_dropout']
        # Project raw embeddings to model entity_dim
        self.entity_linear = nn.Linear(in_features=self.ent_dim, out_features=entity_dim)
        self.relation_linear1 = nn.Linear(in_features=self.rel_dim, out_features=entity_dim)
        # Common dropout
        self.linear_drop = nn.Dropout(p=self.linear_dropout)
        # Type-based entity initialization when no KGE provided
        if self.encode_type:
            self.type_layer = TypeLayer(
                in_features=entity_dim,
                out_features=entity_dim,
                linear_drop=self.linear_drop,
                device=self.device,
                norm_rel=self.norm_rel
            )
        # Relation text attention encoder
        self.self_att_r = AttnEncoder(self.entity_dim)

    def get_ent_init(self, local_entity, kb_adj_mat, rel_features):
        """Compute initial local entity embeddings (type-layer or KGE -> linear)."""
        if self.encode_type:
            local_entity_emb = self.type_layer(
                local_entity=local_entity,
                edge_list=kb_adj_mat,
                rel_features=rel_features
            )
        else:
            local_entity_emb = self.entity_embedding(local_entity)
            local_entity_emb = self.entity_linear(local_entity_emb)
        return local_entity_emb

    def get_rel_feature(self):
        """Forward relation features (projected to entity_dim)."""
        if self.rel_texts is None:
            rel_features = self.relation_embedding.weight
            rel_features = self.relation_linear1(rel_features)
        else:
            # Use encoded relation texts -> map to entity_dim -> attention pool
            rel_features = self.instruction.question_emb(self.rel_features)
            rel_features = self.self_att_r(
                rel_features,
                (self.rel_texts != self.instruction.pad_val).float()
            )
            if self.lm == 'lstm':
                rel_features = self.self_att_r(
                    rel_features,
                    (self.rel_texts != self.num_relation + 1).float()
                )
        return rel_features

    def get_rel_feature_inv(self):
        """Inverse relation features (projected to entity_dim)."""
        if getattr(self, 'rel_texts_inv', None) is None:
            # Fall back to inverse relation embeddings when no texts
            rel_features_inv = self.relation_embedding_inv.weight
            rel_features_inv = self.relation_linear1(rel_features_inv)
        else:
            rel_features_inv = self.instruction.question_emb(self.rel_features_inv)
            rel_features_inv = self.self_att_r(
                rel_features_inv,
                (self.rel_texts_inv != self.instruction.pad_val).float()
            )
            if self.lm == 'lstm':
                rel_features_inv = self.self_att_r(
                    rel_features_inv,
                    (self.rel_texts_inv != self.num_relation + 1).float()
                )
        return rel_features_inv

    def init_reason(self, curr_dist, local_entity, kb_adj_mat, q_input, query_entities):
        """
        Prepare state and wire inputs to ReasonGNNLayer.
        Note: `query_node_emb` is set by `instruction.init_reason(q_input)` in forward;
        here we pass None so ReasonGNNLayer can proceed without it.
        """
        self.local_entity = local_entity
        # Relation features (forward + inverse)
        rel_features = self.get_rel_feature()
        rel_features_inv = self.get_rel_feature_inv()
        # Initial entity embeddings
        self.local_entity_emb = self.get_ent_init(local_entity, kb_adj_mat, rel_features)
        # Distributions and tracking
        self.curr_dist = curr_dist
        self.dist_history = []
        self.action_probs = []
        self.seed_entities = curr_dist
        # Initialize reasoning layer
        self.reasoning.init_reason(
            local_entity=local_entity,
            kb_adj_mat=kb_adj_mat,
            local_entity_emb=self.local_entity_emb,
            rel_features=rel_features,
            rel_features_inv=rel_features_inv,
            query_entities=query_entities,
            query_node_emb=None
        )

    def forward(self, batch, training=False):
        """
        Forward function: creates instructions and performs GNN reasoning.
        """
        import time
        def timer(): return time.perf_counter()
        
        t0 = timer()  # 总计时开始

        # 1) 数据预处理和转换
        t_data0 = timer()
        # local_entity, query_entities, kb_adj_mat, query_text, seed_dist, answer_dist = batch
        local_entity, query_entities, kb_adj_mat, query_text, seed_dist, true_batch_id, answer_dist = batch
        local_entity = torch.from_numpy(local_entity).type('torch.LongTensor').to(self.device)
        # local_entity_mask = (local_entity != self.num_entity).float()
        query_entities = torch.from_numpy(query_entities).type('torch.FloatTensor').to(self.device)
        answer_dist = torch.from_numpy(answer_dist).type('torch.FloatTensor').to(self.device)
        seed_dist = torch.from_numpy(seed_dist).type('torch.FloatTensor').to(self.device)
        current_dist = Variable(seed_dist, requires_grad=True)

        q_input= torch.from_numpy(query_text).type('torch.LongTensor').to(self.device)
        #query_text2 = torch.from_numpy(query_text2).type('torch.LongTensor').to(self.device)
        if self.lm != 'lstm':
            pad_val = self.instruction.pad_val #tokenizer.convert_tokens_to_ids(self.instruction.tokenizer.pad_token)
            query_mask = (q_input != pad_val).float()
            
        else:
            query_mask = (q_input != self.num_word).float()
        t_data1 = timer()

        # 2) 初始化推理和GNN-RAG³组件
        t_init0 = timer()
        """
        Instruction generations
        """
        self.init_reason(curr_dist=current_dist, local_entity=local_entity,
                         kb_adj_mat=kb_adj_mat, q_input=q_input, query_entities=query_entities)
        self.instruction.init_reason(q_input)
        for i in range(self.num_ins):
            relational_ins, attn_weight = self.instruction.get_instruction(self.instruction.relational_ins, step=i) 
            self.instruction.instructions.append(relational_ins.unsqueeze(1))
            self.instruction.relational_ins = relational_ins
        #relation_ins = torch.cat(self.instruction.instructions, dim=1)
        #query_emb = None
        self.dist_history.append(self.curr_dist)
        t_init1 = timer()


        # 新增：warmup-only 混合检索路径（跳过GNN推理，仅训练融合/QA损失）
        if training and getattr(self, 'train_hybrid_only', False) and self.use_appr and hasattr(self, 'hybrid_retriever'):
            pred_dist = self._compute_hybrid_pred_distribution(local_entity, kb_adj_mat, seed_dist)
            answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
            case_valid = (answer_number > 0).float()
            qa_loss = self.calc_loss_label(curr_dist=pred_dist, teacher_dist=answer_dist, label_valid=case_valid)
            pred = torch.max(pred_dist, dim=1)[1]
            h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
            tp_list = [h1.tolist(), f1.tolist()]
            return qa_loss, pred, pred_dist, tp_list

        # 3) GNN推理循环
        t_gnn0 = timer()
        """
        BFS + GNN reasoning with GNN-RAG³ enhancements
        """

        for t in range(self.num_iter):
            relation_ins = torch.cat(self.instruction.instructions, dim=1)
            self.curr_dist = current_dist            
            for j in range(self.num_gnn):
                # 中途重启机制：在L/2层重置分布
                if self.mid_restart and j == self.num_gnn // 2:
                    self.curr_dist = self.seed_entities.clone()
                
                # 传递DDE信息到reasoning层
                if self.use_dde:
                    self.curr_dist, global_rep = self.reasoning(
                        self.curr_dist, relation_ins, step=j, 
                        distances=getattr(self, 'distances', None),
                        dde_layer=getattr(self, 'dde_layer', None)
                    )
                else:
                    self.curr_dist, global_rep = self.reasoning(self.curr_dist, relation_ins, step=j)
            
            self.dist_history.append(self.curr_dist)
            qs = []

            """
            Instruction Updates
            """
            for j in range(self.num_ins):
                reform = getattr(self, 'reform' + str(j))
                q = reform(self.instruction.instructions[j].squeeze(1), global_rep, query_entities, local_entity)
                qs.append(q.unsqueeze(1))
                self.instruction.instructions[j] = q.unsqueeze(1)
        t_gnn1 = timer()
        
        # 4) 损失计算和PCST处理
        t_loss0 = timer()
        """
        Answer Predictions with GNN-RAG³ Loss Integration
        """
        pred_dist = self.dist_history[-1]
        answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
        case_valid = (answer_number > 0).float()
        
        # 基础QA损失
        qa_loss = self.calc_loss_label(curr_dist=pred_dist, teacher_dist=answer_dist, label_valid=case_valid)
        
        # GNN-RAG³ 增强损失
        total_loss = qa_loss
        loss_dict = {'qa_loss': qa_loss.item()}
        
        if training and self.use_pcst and hasattr(self, 'pcst_loss'):
            # PCST损失计算
            pcst_loss_val = self._compute_pcst_loss(pred_dist, local_entity, kb_adj_mat)
            # 使用pcst_lambda的总和作为整体权重，并转换为tensor
            pcst_weight_val = sum(self.pcst_lambda) if isinstance(self.pcst_lambda, (list, tuple)) else self.pcst_lambda
            pcst_weight = torch.tensor(pcst_weight_val, device=pred_dist.device, dtype=pcst_loss_val.dtype)
            total_loss = total_loss + pcst_weight * pcst_loss_val
            loss_dict['pcst_loss'] = pcst_loss_val.item()
        
        # 混合检索损失（如果在训练阶段）
        if training and self.use_appr and hasattr(self, 'hybrid_retriever'):
            # 这里可以添加检索相关的损失
            pass
        
        pred_dist = self.dist_history[-1]
        pred = torch.max(pred_dist, dim=1)[1]
        if training:
            h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
            tp_list = [h1.tolist(), f1.tolist()]
            # 注释掉额外的loss_dict以保持与训练代码的兼容性
            # tp_list.append(loss_dict)
        else:
            tp_list = None
        
        t_loss1 = timer()
        tt = timer() - t0
        
        # 详细计时输出

        
        return total_loss, pred, pred_dist, tp_list

    def _compute_pcst_loss(self, pred_dist, local_entity, kb_adj_mat):
        """
        计算PCST损失
        """
        try:
            # 构建边概率和成本
            batch_size = pred_dist.size(0)
            # 初始化为tensor而不是float
            total_pcst_loss = torch.tensor(0.0, device=pred_dist.device, requires_grad=True)
            
            for b in range(batch_size):
                # 简化的PCST损失计算
                # 实际实现中应该基于图结构和边权重
                edge_probs = torch.sigmoid(pred_dist[b])  # 简化的边概率
                edge_costs = torch.ones_like(edge_probs)  # 简化的边成本
                
                # 软正则化损失
                soft_loss = self.pcst_loss.pcst_soft_regularizer(
                    edge_probs, edge_costs, 
                    laplacian=None,  # 应该从kb_adj_mat构建
                    temperature=self.gumbel_temp  # 修正参数名
                )
                total_pcst_loss = total_pcst_loss + soft_loss
            
            return total_pcst_loss / batch_size
        except Exception as e:
            # 如果PCST损失计算失败，返回零损失
            return torch.tensor(0.0, device=pred_dist.device, requires_grad=True)

    
    def _compute_hybrid_pred_distribution(self, local_entity, kb_adj_mat, seed_dist):
        """
        使用 HybridRetriever 的融合权重将 APPR 与语义相似度融合，直接生成每个样本的答案分布。
        local_entity: [B, max_local_entity]，其中无效实体以 num_entity 填充
        kb_adj_mat: (batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list, weight_rel_list)
        seed_dist: [B, max_local_entity]，种子分布（>0 为种子）
        返回: [B, max_local_entity] 的概率分布
        """
        import torch.nn.functional as F
        device = self.device
        batch_size = local_entity.size(0)
        max_local_entity = local_entity.size(1)
        
        batch_heads, batch_rels, batch_tails, batch_bids, fact_ids, weight_list, weight_rel_list = kb_adj_mat
        # 转为 numpy 以便掩码，再到 torch
        import numpy as np
        heads_np = np.array(batch_heads)
        tails_np = np.array(batch_tails)
        bids_np = np.array(batch_bids)
        weights_np = np.array(weight_list)
        
        pred_dists = []
        for b in range(batch_size):
            # 有效节点掩码
            valid_mask = (local_entity[b] != self.num_entity)
            num_nodes = int(valid_mask.sum().item()) if hasattr(valid_mask, 'sum') else max_local_entity
            # 索引偏移：数据构造时为每个 batch 增加偏移 i*max_local_entity
            index_bias = b * max_local_entity
            mask_b = (bids_np == b)
            heads_b = torch.from_numpy(heads_np[mask_b] - index_bias).long().to(device)
            tails_b = torch.from_numpy(tails_np[mask_b] - index_bias).long().to(device)
            edge_index_b = torch.stack([heads_b, tails_b], dim=0) if heads_b.numel() > 0 else torch.empty((2,0), dtype=torch.long, device=device)
            weights_b = torch.from_numpy(weights_np[mask_b]).float().to(device) if weights_np.size > 0 else None
            
            # 取查询嵌入与本地实体嵌入
            z_q_b = self.instruction.query_node_emb[b, 0, :].to(device)  # [D]
            ent_emb_b = self.local_entity_emb[b]  # [max_local_entity, D]
            
            # 计算 APPR 分布（字典）
            seed_nodes = torch.nonzero(seed_dist[b] > 0, as_tuple=True)[0].tolist()
            appr_dict = {}
            if heads_b.numel() > 0 and len(seed_nodes) > 0:
                appr_dict = self.hybrid_retriever.compute_appr(edge_index_b, max_local_entity, seed_nodes, edge_weights=weights_b)
            appr_vec = torch.zeros(max_local_entity, device=device)
            if len(appr_dict) > 0:
                idxs = torch.tensor(list(appr_dict.keys()), dtype=torch.long, device=device)
                vals = torch.tensor(list(appr_dict.values()), dtype=torch.float, device=device)
                appr_vec[idxs] = vals
            
            # 语义相似度分数（余弦归一化后点积）
            z_norm = F.normalize(z_q_b.unsqueeze(0), p=2, dim=1)  # [1,D]
            ent_norm = F.normalize(ent_emb_b, p=2, dim=1)        # [N,D]
            sim_vec = (ent_norm @ z_norm.t()).squeeze(1).clamp(min=0.0)  # [N]
            
            # 融合权重 alpha_q（标量）
            alpha_q = self.hybrid_retriever.fusion_net(z_q_b).squeeze()
            fused = alpha_q * appr_vec + (1.0 - alpha_q) * sim_vec
            
            # 屏蔽无效实体并softmax
            VERY_NEG = -1e9
            fused = fused.clone()
            fused[~valid_mask] = VERY_NEG
            pred_b = torch.softmax(fused, dim=0)
            pred_dists.append(pred_b)
        
        return torch.stack(pred_dists, dim=0)

    def calc_loss_label(self, curr_dist, teacher_dist, label_valid):
        tp_loss = self.get_loss(pred_dist=curr_dist, answer_dist=teacher_dist, reduction='none')
        tp_loss = tp_loss * label_valid
        cur_loss = torch.sum(tp_loss) / curr_dist.size(0)
        return cur_loss

    