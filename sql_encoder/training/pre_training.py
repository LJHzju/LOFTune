import numpy as np
import torch
from torch import nn

from sql_encoder.model.tbcnn_encoder import TBCNNEncoder


class SubtreePredictionModel(nn.Module):
    def __init__(self, config):
        super(SubtreePredictionModel, self).__init__()

        self.num_subtrees = config['num_subtrees']
        self.subtree_dim = config['subtree_dim']

        self.softmax_w = torch.nn.Parameter(torch.randn(self.num_subtrees, self.subtree_dim) / np.sqrt(self.subtree_dim))
        self.softmax_b = torch.nn.Parameter(torch.zeros(self.num_subtrees))

    def forward(self, sql_embedding):
        # sql_embedding: batch_size × subtree_dim
        # labels: batch_size × num_subtree
        # log_softmax: batch_size × num_subtree
        logits = torch.matmul(sql_embedding, self.softmax_w.t()) + self.softmax_b
        return logits


class PretrainSQLEncoder(nn.Module):
    def __init__(self, encoder_config, subtree_config):
        super(PretrainSQLEncoder, self).__init__()

        self.encoder = TBCNNEncoder(encoder_config)
        self.subtree_predictor = SubtreePredictionModel(subtree_config)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, node_type, node_tokens, children_index, subtree_labels):
        code_vector, _ = self.encoder(node_type, node_tokens, children_index)

        prediction_logit = self.subtree_predictor(code_vector)  # batch_size × num_subtrees
        prediction_loss = self.criterion(prediction_logit, subtree_labels)
        return prediction_loss, prediction_logit

    def predict_subtrees(self, node_type, node_tokens, children_index):
        code_vector, _ = self.encoder(node_type, node_tokens, children_index)

        prediction_logit = self.subtree_predictor(code_vector)  # batch_size × num_subtrees
        probability = torch.sigmoid(prediction_logit)
        subtree_ids = []
        for sql in probability:
            ids = torch.nonzero(sql > 0.5).tolist()
            ids = [id[0] for id in ids]
            subtree_ids.append(ids)
        return subtree_ids

    def get_encoder_param_size(self):
        return sum(p.numel() for p in self.encoder.parameters())

