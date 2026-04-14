from torch import nn as nn
import torch
from trainer import util, sampling
import os
import math
from models.Syn_GCN import GCN
from models.Sem_GCN import SemGCN
from models.Attention_Module import SelfAttention
from models.TIN_GCN import TIN, FeatureStacking
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from models.Channel_Fusion import Orthographic_projection_fusion, TextCentredSP
from transformers import PreTrainedModel
from transformers import AutoConfig, AutoModel


def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """Get specific token embedding (e.g. [CLS])"""
    emb_size = h.shape[-1]

    token_h = h.view(-1, emb_size)
    flat = x.contiguous().view(-1)

    # get contextualized embedding of given token
    token_h = token_h[flat == token, :]

    return token_h


class D2E2SModel(PreTrainedModel):
    VERSION = "1.1"
    _tied_weights_keys = []
    all_tied_weights_keys = {}

    def __init__(
        self,
        config: AutoConfig,
        cls_token: int,
        sentiment_types: int,
        entity_types: int,
        args,
    ):
        super(D2E2SModel, self).__init__(config)
        # 1、parameters init
        self.args = args
        self._size_embedding = self.args.size_embedding
        self._prop_drop = self.args.prop_drop
        self._freeze_transformer = self.args.freeze_transformer
        self.drop_rate = self.args.drop_out_rate
        self._is_bidirectional = self.args.is_bidirect
        self.layers = self.args.lstm_layers
        self._hidden_dim = self.args.hidden_dim
        self.mem_dim = self.args.mem_dim
        self._emb_dim = self.args.emb_dim
        self.output_size = self._emb_dim
        self.batch_size = self.args.batch_size
        self.max_pairs = self.args.max_pairs
        self.deberta_feature_dim = self.args.deberta_feature_dim
        self.gcn_dim = self.args.gcn_dim
        self.gcn_dropout = self.args.gcn_dropout

        # 2、DEBERT model
        # Build from config to avoid nested `from_pretrained` calls inside outer
        # model loading contexts (can trigger meta-device errors in newer Transformers).
        self.deberta = AutoModel.from_config(config)

        # self.BertAdapterModel = BertAdapterModel(config)
        self.Syn_gcn = GCN(self._emb_dim)
        self.Sem_gcn = SemGCN(self.args, self._emb_dim)
        self.senti_classifier = nn.Linear(
            config.hidden_size * 5 + self._size_embedding * 4, sentiment_types
        )
        self.sentence_classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size),
            nn.GELU(),
            nn.Dropout(self._prop_drop),
            nn.Linear(config.hidden_size, 1),
        )
        self.entity_classifier = nn.Linear(
            config.hidden_size * 2 + self._size_embedding, entity_types
        )
        self.size_embeddings = nn.Embedding(100, self._size_embedding)
        self.dropout = nn.Dropout(self._prop_drop)
        self._cls_token = cls_token
        self._sentiment_types = sentiment_types
        self._entity_types = entity_types
        self._max_pairs = self.max_pairs
        self._max_role_candidates = self.args.max_role_candidates
        self.neg_span_all = 0
        self.neg_span = 0
        self.number = 1

        # 3、LSTM Layers + Attention Layers
        self.lstm = nn.LSTM(
            self._emb_dim,
            int(self._hidden_dim),
            self.layers,
            batch_first=True,
            bidirectional=self._is_bidirectional,
            dropout=self.drop_rate,
        )
        self.attention_layer = SelfAttention(self.args)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0)
        self.lstm_dropout = nn.Dropout(self.drop_rate)

        # 4、linear and sigmoid layers
        if self._is_bidirectional:
            self.fc = nn.Linear(int(self._hidden_dim * 2), self.output_size)
        else:
            self.fc = nn.Linear(int(self._hidden_dim), self.output_size)

        if self._is_bidirectional:
            self.number = 2

        # 6、weight initialization
        self.init_weights()
        if self._freeze_transformer:
            print("Freeze transformer weights")

            # freeze all transformer weights
            for param in self.deberta.parameters():
                param.requires_grad = False

        # # 7、Mutual Biaffine Model
        # self.affine1 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        # self.affine2 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        # self.gcn_drop = nn.Dropout(self.args.gcn_dropout)
        # # 7、MLP with Biaffine Attention
        # self.Biaffine_ATT = BiaffineAttention(self.bert_feature_dim, self.bert_feature_dim)

        # 7、feature merge model
        self.TIN = TIN(self.deberta_feature_dim)
        # self.TextCentredSP = TextCentredSP(self.bert_feature_dim*2, self.shared_dim, self.private_dim)

    def _forward_train(
        self,
        encodings: torch.tensor,
        context_masks: torch.tensor,
        entity_masks: torch.tensor,
        entity_sizes: torch.tensor,
        sentiments: torch.tensor,
        senti_masks: torch.tensor,
        adj,
    ):

        # Parameters init
        context_masks = context_masks.float()
        self.context_masks = context_masks
        batch_size = encodings.shape[0]
        seq_lens = encodings.shape[1]

        # encoder layer
        # h = self.BertAdapterModel(input_ids=encodings, attention_mask=self.context_masks)[0]
        h = self.deberta(input_ids=encodings, attention_mask=self.context_masks)[0]
        self.output, _ = self.lstm(h)
        self.deberta_lstm_output = self.lstm_dropout(self.output)
        self.deberta_lstm_att_feature = self.deberta_lstm_output

        # attention layers
        # bert_lstm_feature_attention = self.attention_layer(self.bert_lstm_output, self.bert_lstm_output, self.context_masks[:,:seq_lens])
        # self.bert_lstm_att_feature = self.bert_lstm_output + bert_lstm_feature_attention

        # gcn layer
        h_syn_ori, pool_mask_origin = self.Syn_gcn(adj, h)
        h_syn_gcn, pool_mask = self.Syn_gcn(adj, self.deberta_lstm_att_feature)
        h_sem_ori, adj_sem_ori = self.Sem_gcn(h, encodings, seq_lens)
        h_sem_gcn, adj_sem_gcn = self.Sem_gcn(
            self.deberta_lstm_att_feature, encodings, seq_lens
        )

        # fusion layer
        h1 = self.TIN(
            h, h_syn_ori, h_syn_gcn, h_sem_ori, h_sem_gcn, adj_sem_ori, adj_sem_gcn
        )
        h = self.attention_layer(h1, h1, self.context_masks[:, :seq_lens]) + h1
        sentence_logits = self._classify_sentence(h, context_masks)
        # h_feature, h_syn_origin, h_syn_feature, h_sem_origin, h_sem_feature = self.TIN(h, h_syn_ori, h_syn_gcn, h_sem_ori, h_sem_gcn)
        # h = self.TextCentredSP(h_syn_feature, h_sem_feature)

        size_embeddings = self.size_embeddings(entity_sizes)
        entity_clf, entity_spans_pool = self._classify_entities(
            encodings, h, entity_masks, size_embeddings, self.args
        )

        # relation_classify
        h_large = h.unsqueeze(1).repeat(
            1, max(min(sentiments.shape[1], self._max_pairs), 1), 1, 1
        )
        senti_clf = torch.zeros(
            [batch_size, sentiments.shape[1], self._sentiment_types]
        ).to(self.senti_classifier.weight.device)

        # obtain sentiment logits
        # chunk processing to reduce memory usage
        for i in range(0, sentiments.shape[1], self._max_pairs):
            # classify sentiment candidates
            chunk_senti_logits = self._classify_sentiments(
                entity_spans_pool, size_embeddings, sentiments, senti_masks, h_large, i
            )
            senti_clf[:, i : i + self._max_pairs, :] = chunk_senti_logits

        batch_loss = compute_loss(adj_sem_ori, adj_sem_gcn, adj)

        return entity_clf, senti_clf, sentence_logits, batch_loss

    def _forward_eval(
        self,
        encodings: torch.tensor,
        context_masks: torch.tensor,
        entity_masks: torch.tensor,
        entity_sizes: torch.tensor,
        entity_spans: torch.tensor,
        entity_sample_masks: torch.tensor,
        adj,
    ):
        context_masks = context_masks.float()
        self.context_masks = context_masks
        batch_size = encodings.shape[0]
        seq_lens = encodings.shape[1]

        # encoder layer
        # h = self.BertAdapterModel(input_ids=encodings, attention_mask=self.context_masks)[0]
        h = self.deberta(input_ids=encodings, attention_mask=self.context_masks)[0]
        self.output, _ = self.lstm(h)
        self.deberta_lstm_output = self.lstm_dropout(self.output)
        self.deberta_lstm_att_feature = self.deberta_lstm_output

        # attention layers
        # bert_lstm_feature_attention = self.attention_layer(self.bert_lstm_output, self.bert_lstm_output, self.context_masks[:,:seq_lens])
        # self.bert_lstm_att_feature = self.bert_lstm_output + bert_lstm_feature_attention
        # self.bert_lstm_att_feature = bert_lstm_feature_attention

        # gcn layer
        h_syn_ori, pool_mask_origin = self.Syn_gcn(adj, h)
        h_syn_gcn, pool_mask = self.Syn_gcn(adj, self.deberta_lstm_att_feature)
        h_sem_ori, adj_sem_ori = self.Sem_gcn(h, encodings, seq_lens)
        h_sem_gcn, adj_sem_gcn = self.Sem_gcn(
            self.deberta_lstm_att_feature, encodings, seq_lens
        )

        # fusion layer
        h1 = self.TIN(
            h, h_syn_ori, h_syn_gcn, h_sem_ori, h_sem_gcn, adj_sem_ori, adj_sem_gcn
        )
        h = self.attention_layer(h1, h1, self.context_masks[:, :seq_lens]) + h1
        sentence_logits = self._classify_sentence(h, context_masks)
        # h_feature, h_syn_origin, h_syn_feature, h_sem_origin, h_sem_feature = self.TIN(h, h_syn_ori, h_syn_gcn, h_sem_ori, h_sem_gcn)
        # h = self.TextCentredSP(h_syn_feature, h_sem_feature)

        # entity_classify
        size_embeddings = self.size_embeddings(
            entity_sizes
        )  # embed entity candidate sizes
        entity_clf, entity_spans_pool = self._classify_entities(
            encodings, h, entity_masks, size_embeddings, self.args
        )

        # ignore entity candidates that do not constitute an actual entity for sentiments (based on classifier)
        ctx_size = context_masks.shape[-1]
        sentiments, senti_masks, senti_sample_masks = self._filter_spans(
            entity_clf, entity_spans, entity_sample_masks, ctx_size
        )
        senti_sample_masks = senti_sample_masks.float().unsqueeze(-1)
        h_large = h.unsqueeze(1).repeat(
            1, max(min(sentiments.shape[1], self._max_pairs), 1), 1, 1
        )
        senti_clf = torch.zeros(
            [batch_size, sentiments.shape[1], self._sentiment_types]
        ).to(self.senti_classifier.weight.device)

        # obtain sentiment logits
        # chunk processing to reduce memory usage
        for i in range(0, sentiments.shape[1], self._max_pairs):
            # classify sentiment candidates
            chunk_senti_logits = self._classify_sentiments(
                entity_spans_pool, size_embeddings, sentiments, senti_masks, h_large, i
            )
            # apply sigmoid
            chunk_senti_clf = torch.sigmoid(chunk_senti_logits)
            senti_clf[:, i : i + self._max_pairs, :] = chunk_senti_clf

        senti_clf = senti_clf * senti_sample_masks  # mask

        sentence_scores = torch.sigmoid(sentence_logits).view(-1, 1, 1)
        sentence_gate = (sentence_scores >= self.args.sentence_filter_threshold).float()
        senti_clf = senti_clf * sentence_gate

        # apply softmax
        entity_clf = torch.softmax(entity_clf, dim=2)

        return entity_clf, senti_clf, sentiments, sentence_logits

    def _classify_sentence(self, h, context_masks):
        token_mask = context_masks[:, : h.shape[1]].float()
        cls_repr = h[:, 0]
        mean_repr = (h * token_mask.unsqueeze(-1)).sum(dim=1) / token_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        max_repr = h.masked_fill(token_mask.unsqueeze(-1) == 0, -1e30).max(dim=1).values
        sentence_repr = torch.cat([cls_repr, mean_repr, max_repr], dim=-1)
        return self.sentence_classifier(sentence_repr).squeeze(-1)

    def _classify_entities(self, encodings, h, entity_masks, size_embeddings, args):
        # entity_masks: tensor(4,132,24) 4:batch_size, 132: entities count, 24: one sentence token count and one entity need 24 mask
        # size_embedding: tensor(4,132,25) 4：batch_size, 132:entities_size, 25:each entities Embedding Dimension
        # h: tensor(4,24,768) -> (4,1,24,768) -> (4,132,24,768)
        # m: tensor(4,132,24,1)
        # encoding: tensor(4,24)
        # entity_spans_pool: tensor(4，132，24，768) -> tensor(4,132,768)
        m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)
        entity_spans_pool = m + h.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)

        self.args = args
        if self.args.span_generator == "Average" or self.args.span_generator == "Max":
            if self.args.span_generator == "Max":
                entity_spans_pool = entity_spans_pool.max(dim=2)[0]
            else:
                entity_spans_pool = entity_spans_pool.mean(dim=2, keepdim=True).squeeze(
                    -2
                )

        # Use the first token representation as CLS context. Selecting by token-id
        # can accidentally include padded positions for models where cls_token_id
        # equals a common pad value (e.g. XLM-R with id=0).
        entity_ctx = h[:, 0, :]

        # create candidate representations including context, max pooled span and size embedding
        entity_repr = torch.cat(
            [
                entity_ctx.unsqueeze(1).repeat(1, entity_spans_pool.shape[1], 1),
                entity_spans_pool,
                size_embeddings,
            ],
            dim=2,
        )
        entity_repr = self.dropout(entity_repr)

        # classify entity candidates
        entity_clf = self.entity_classifier(entity_repr)

        return entity_clf, entity_spans_pool

    def _classify_sentiments(
        self, entity_spans, size_embeddings, sentiments, senti_masks, h, chunk_start
    ):
        batch_size = sentiments.shape[0]

        # create chunks if necessary
        if sentiments.shape[1] > self._max_pairs:
            sentiments = sentiments[:, chunk_start : chunk_start + self._max_pairs]
            senti_masks = senti_masks[:, chunk_start : chunk_start + self._max_pairs]
            h = h[:, : sentiments.shape[1], :]

        # get quadruples of entity candidate representations  [batch, n_quints, 4, hidden]
        entity_quads = util.batch_index(entity_spans, sentiments)
        entity_quads = entity_quads.view(batch_size, entity_quads.shape[1], -1)  # [batch, n_quints, 4*hidden]

        # get corresponding size embeddings  [batch, n_quints, 4, size_emb]
        size_quad_embeddings = util.batch_index(size_embeddings, sentiments)
        size_quad_embeddings = size_quad_embeddings.view(
            batch_size, size_quad_embeddings.shape[1], -1
        )  # [batch, n_quints, 4*size_emb]

        # sentiment context (context spanning all entity spans in the quintuple)
        m = ((senti_masks == 0).float() * (-1e30)).unsqueeze(-1)
        senti_ctx = m + h
        senti_ctx = senti_ctx.max(dim=2)[0]
        senti_ctx[senti_masks.to(torch.uint8).any(-1) == 0] = 0

        # create sentiment candidate representations:
        #   context  [hidden]  +  4 entities [4*hidden]  +  4 size embs [4*size_emb]
        senti_repr = torch.cat([senti_ctx, entity_quads, size_quad_embeddings], dim=2)
        senti_repr = self.dropout(senti_repr)

        chunk_senti_logits = self.senti_classifier(senti_repr)
        return chunk_senti_logits

    def log_sample_total(self, neg_entity_count_all):
        log_path = os.path.join("./log/Sample/", "countSample.txt")
        with open(log_path, mode="a", encoding="utf-8") as f:
            f.write("neg_entity_count_all: \n")
            self.neg_span_all += len(neg_entity_count_all)
            f.write(str(self.neg_span_all))
            f.write("\nneg_entity_count: \n")
            self.neg_span += len((neg_entity_count_all != 0).nonzero())
            f.write(str(self.neg_span))
            f.write("\n")
        f.close()

    def _select_role_indices(self, entity_type_ids, role_scores, non_zero_indices, role_id, allow_null):
        role_indices = [idx for idx in non_zero_indices if entity_type_ids[idx] == role_id]

        if self._max_role_candidates > 0 and len(role_indices) > self._max_role_candidates:
            role_indices = sorted(
                role_indices,
                key=lambda idx: role_scores[idx],
                reverse=True,
            )[: self._max_role_candidates]

        if allow_null:
            # Keep non-null candidates first so Cartesian product prioritizes
            # full quadruples before partial tuples when _max_pairs truncates.
            return role_indices + [0]
        return role_indices

    def _filter_spans(self, entity_clf, entity_spans, entity_sample_masks, ctx_size):
        """Generate candidate quintuples (s, o, a, p) from predicted entity spans.
        Index 0 in entity_spans is always the NULL entity (CLS span).
        """
        from itertools import product as iproduct
        batch_size = entity_clf.shape[0]
        entity_logits_max = (
            entity_clf.argmax(dim=-1) * entity_sample_masks.long()
        )  # get entity type (including none)
        batch_sentiments = []
        batch_senti_masks = []
        batch_senti_sample_masks = []

        for i in range(batch_size):
            rels = []
            senti_masks = []
            sample_masks = []

            self.log_sample_total(entity_logits_max[i])

            # Non-zero (non-None) entity indices (excluding index 0 = null)
            non_zero_indices = (entity_logits_max[i] != 0).nonzero().view(-1).tolist()
            entity_type_ids = entity_logits_max[i].tolist()

            if getattr(self.args, "dataset", "") == "cameraCOQE_quintuple" and self._entity_types >= 5:
                role_probabilities = torch.softmax(entity_clf[i], dim=-1)
                subject_indices = self._select_role_indices(
                    entity_type_ids,
                    role_probabilities[:, 1].tolist(),
                    non_zero_indices,
                    1,
                    allow_null=True,
                )
                object_indices = self._select_role_indices(
                    entity_type_ids,
                    role_probabilities[:, 2].tolist(),
                    non_zero_indices,
                    2,
                    allow_null=True,
                )
                aspect_indices = self._select_role_indices(
                    entity_type_ids,
                    role_probabilities[:, 3].tolist(),
                    non_zero_indices,
                    3,
                    allow_null=True,
                )
                predicate_indices = self._select_role_indices(
                    entity_type_ids,
                    role_probabilities[:, 4].tolist(),
                    non_zero_indices,
                    4,
                    allow_null=False,
                )
                role_products = iproduct(subject_indices, object_indices, aspect_indices, predicate_indices)
            else:
                # All candidate indices include null (0) and real entities.
                all_indices = [0] + non_zero_indices
                role_products = iproduct(all_indices, all_indices, all_indices, all_indices)

            # Enumerate all 4-tuples; limit to _max_pairs to control memory
            for i1, i2, i3, i4 in role_products:
                # At least one slot must be a real (non-null) entity
                if i1 == 0 and i2 == 0 and i3 == 0 and i4 == 0:
                    continue
                rels.append((i1, i2, i3, i4))

                # Context mask: span region covering all non-null entities
                real_span_list = [
                    entity_spans[i][idx].tolist()
                    for idx in (i1, i2, i3, i4) if idx != 0
                ]
                if len(real_span_list) >= 2:
                    sorted_s = sorted(real_span_list, key=lambda s: s[0])
                    cs, ce = sorted_s[0][1], sorted_s[-1][0]
                    mask = torch.zeros(ctx_size, dtype=torch.bool)
                    if cs < ce:
                        mask[cs:ce] = 1
                else:
                    mask = torch.zeros(ctx_size, dtype=torch.bool)
                senti_masks.append(mask)
                sample_masks.append(1)

                if len(rels) >= self._max_pairs:
                    break

            if not rels:
                batch_sentiments.append(torch.tensor([[0, 0, 0, 0]], dtype=torch.long))
                batch_senti_masks.append(
                    torch.tensor([[0] * ctx_size], dtype=torch.bool)
                )
                batch_senti_sample_masks.append(torch.tensor([0], dtype=torch.bool))
            else:
                batch_sentiments.append(torch.tensor(rels, dtype=torch.long))
                batch_senti_masks.append(torch.stack(senti_masks))
                batch_senti_sample_masks.append(
                    torch.tensor(sample_masks, dtype=torch.bool)
                )

        device = self.senti_classifier.weight.device
        batch_sentiments = util.padded_stack(batch_sentiments).to(device)
        batch_senti_masks = util.padded_stack(batch_senti_masks).to(device)
        batch_senti_sample_masks = util.padded_stack(batch_senti_sample_masks).to(
            device
        )

        return batch_sentiments, batch_senti_masks, batch_senti_sample_masks

    def forward(self, *args, evaluate=False, **kwargs):
        if not evaluate:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_eval(*args, **kwargs)


def compute_loss(p, q, k):

    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(k, dim=-1), reduction="none")
    k_loss = F.kl_div(F.log_softmax(k, dim=-1), F.softmax(p, dim=-1), reduction="none")

    p_loss = p_loss.sum()
    k_loss = k_loss.sum()
    total_loss = math.log(1 + 5 / (torch.abs((p_loss + k_loss) / 2)))

    return total_loss
