r"""
FDSA
################################################

Reference:
    Tingting Zhang et al. "Feature-level Deeper Self-Attention Network for Sequential Recommendation."
    In IJCAI 2019

"""

import copy
import torch
from torch import nn
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, MLPLayers
from recbole.model.loss import BPRLoss

class FDSA(SequentialRecommender):
    r"""
    FDSA is similar with the GRU4RecF implemented in RecBole, which uses two different Transformer encoders to
    encode items and features respectively and concatenates the two subparts' outputs as the final output.

    """

    def __init__(self, config, dataset):
        super(FDSA, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.selected_features = config['selected_features']
        self.pooling_mode = config['pooling_mode']
        self.device = config['device']
        self.num_feature_field = len(config['selected_features'])

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.plm_size = dataset.plm_size
        self.plm_embedding = copy.deepcopy(dataset.plm_embedding)
        self.adapter = MLPLayers(config['adapter_layers'])

        self.item_trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        # For simplicity, we use same architecture for item_trm and feature_trm
        self.feature_trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.concat_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)
        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0., -10000.)
        return extended_attention_mask

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        # get item_trm_input
        # item position add position embedding
        item_emb = item_emb + position_embedding
        item_emb = self.LayerNorm(item_emb)
        item_trm_input = self.dropout(item_emb)

        # feature position add position embedding
        feature_emb = self.adapter(self.plm_embedding(item_seq))
        feature_emb = feature_emb + position_embedding
        feature_emb = self.LayerNorm(feature_emb)
        feature_trm_input = self.dropout(feature_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        item_trm_output = self.item_trm_encoder(item_trm_input, extended_attention_mask, output_all_encoded_layers=True)
        item_output = item_trm_output[-1]

        feature_trm_output = self.feature_trm_encoder(
            feature_trm_input, extended_attention_mask, output_all_encoded_layers=True
        )  # [B Len H]
        feature_output = feature_trm_output[-1]

        item_output = self.gather_indexes(item_output, item_seq_len - 1)  # [B H]
        feature_output = self.gather_indexes(feature_output, item_seq_len - 1)  # [B H]

        output_concat = torch.cat((item_output, feature_output), -1)  # [B 2*H]
        output = self.concat_layer(output_concat)
        output = self.LayerNorm(output)
        seq_output = self.dropout(output)
        return seq_output  # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores
