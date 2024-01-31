import torch
from torch import nn
import torch.nn.functional as F


class SelfAttentionLayer(nn.Module):
    def __init__(self, input_size):
        super(SelfAttentionLayer, self).__init__()

        # Define linear transformations for query, key, and value
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)

    def forward(self, nodes, mask):
        # nodes: batch_size × seq_length × input_size
        # mask: batch_size × seq_length (1 for real, 0 for padded)
        # key_node_positions: batch_size × seq_length (1 for key nodes, 0 otherwise)

        # Linear transformations
        query = self.query(nodes)
        key = self.key(nodes)
        value = self.value(nodes)

        # Calculate attention scores
        # (batch_size, seq_length, seq_length)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / key.size(-1) ** 0.5

        # Apply mask to the scores
        new_mask = mask.unsqueeze(1).expand_as(attention_scores)
        adjusted_scores = attention_scores.masked_fill(new_mask == 0, float('-inf'))

        # Calculate attention weights using softmax
        # (batch_size, seq_length, seq_length)
        attention_weights = F.softmax(adjusted_scores, dim=-1)

        # Apply attention weights to the values
        # (batch_size, seq_length, seq_length) * (batch_size, seq_length, input_size)
        weighted_sum = torch.matmul(attention_weights, value)
        mean_weighted_sum = torch.mean(weighted_sum, dim=1)

        return mean_weighted_sum
