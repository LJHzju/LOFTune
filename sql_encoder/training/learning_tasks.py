import torch
import torch.nn as nn
import numpy as np


class SubtreePredictionModel(nn.Module):
    def __init__(self, config):
        super(SubtreePredictionModel, self).__init__()

        self.num_objects = config['num_objects']
        self.object_dim = config['object_dim']

        self.softmax_w = torch.nn.Parameter(torch.randn(self.num_objects, self.object_dim) / np.sqrt(self.object_dim))
        self.softmax_b = torch.nn.Parameter(torch.zeros(self.num_objects))

    def forward(self, sql_embedding):
        # sql_embedding: batch_size × subtree_dim
        # labels: batch_size × num_subtree
        # log_softmax: batch_size × num_subtree
        logits = torch.matmul(sql_embedding, self.softmax_w.t()) + self.softmax_b
        return logits


class SequenceTaggingModel(nn.Module):
    def __init__(self, config):
        super(SequenceTaggingModel, self).__init__()

        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_labels = config['num_labels']

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        self.output_layer = nn.Linear(self.hidden_dim, self.num_labels)
        self.dropout = nn.Dropout(config['dropout_rate'])

    def forward(self, node_embeddings):
        # node_embeddings: (batch_size, seq_length, hidden_dim)
        lstm_out, _ = self.lstm(node_embeddings)

        # Pass the output of the BiLSTM through the output layer
        lstm_out = self.dropout(lstm_out)
        tag_space = self.output_layer(lstm_out)  # (batch_size, seq_length, num_labels)

        return tag_space