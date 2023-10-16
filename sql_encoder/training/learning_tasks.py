import numpy as np
import torch
from torch import nn


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


class SequenceTaggingModel(nn.Module):
    def __init__(self, config):
        super(SequenceTaggingModel, self).__init__()

        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_labels = config['num_labels']

        self.lstm = nn.LSTM(self.embedding_dim * 2, self.hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(config['dropout_rate'])
        self.output_layer = nn.Linear(self.hidden_dim * 2, self.num_labels)

    def forward(self, node_embeddings, context_vector, mask_nodes):
        # node_embeddings shape: (batch_size, seq_length, embedding_dim)
        # context_vector shape: (batch_size, embedding_dim)

        # Duplicate the context vector to match the sequence length and concatenate with node_embeddings
        context_vector_repeated = context_vector.unsqueeze(1).repeat(1, node_embeddings.size(1),
                                                                     1)  # (batch_size, seq_length, embedding_dim)
        concatenated_input = torch.cat((node_embeddings, context_vector_repeated),
                                       dim=2)  # (batch_size, seq_length, embedding_dim * 2)

        masked_input = concatenated_input * mask_nodes.unsqueeze(-1).type(torch.float)

        # Pass the augmented input through the BiLSTM
        lstm_out, _ = self.lstm(masked_input)  # (batch_size, seq_length, hidden_dim * 2)
        lstm_out = self.dropout(lstm_out)

        # Pass the output of the BiLSTM through the output layer
        tag_space = self.output_layer(lstm_out)  # (batch_size, seq_length, num_labels)

        return tag_space