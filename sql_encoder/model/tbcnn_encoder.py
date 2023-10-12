import torch
from torch import nn
from sql_encoder.model.conv_node import ConvNode


def gen_children_embeddings(parent_node_embeddings, children_index, node_embedding_dim):
    # parent_node_type_embeddings: batch_size × num_nodes × embedding_dim
    batch_size, num_nodes, _ = parent_node_embeddings.size()
    _, _, max_children = children_index.size()
    device = parent_node_embeddings.device

    # replace the root node with the zero vector so lookups for the 0th
    # vector return 0 instead of the root vector
    # zero_vecs is (batch_size, 1, node_type_dim)
    zero_vecs = torch.zeros((batch_size, 1, node_embedding_dim), device=device)
    # vector_lookup is (batch_size, num_nodes, node_type_dim)
    vector_lookup = torch.cat([zero_vecs, parent_node_embeddings[:, 1:, :]], dim=1)
    # children_indices is (batch_size, num_nodes, num_children, 1)
    children_index = children_index.unsqueeze(3)
    # prepend the batch indices to the 4th dimension of children
    # batch_indices is (batch_size, 1, 1, 1)
    batch_indices = torch.arange(0, batch_size).view(batch_size, 1, 1, 1).to(device)
    # batch_indices is (batch_size, num_nodes, num_children, 1)
    batch_indices = batch_indices.repeat(1, num_nodes, max_children, 1)
    # children_indices is (batch_size, num_nodes, num_children, 2)
    children_index = torch.cat([batch_indices, children_index], dim=3)
    # output will have shape (batch_size, num_nodes, num_children, node_type_dim)
    return vector_lookup[children_index[..., 0], children_index[..., 1]]


class AttentionLayer(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.a = nn.Linear(self.input_size, 1, bias=False)

    def forward(self, nodes, mask):
        # (batch_size, max_tree_size, output_size)
        max_tree_size = nodes.shape[1]

        flat_nodes = torch.reshape(nodes, [-1, self.input_size])
        aggregated_vector = self.a(flat_nodes)
        # aggregated_vector = tf.matmul(flat_nodes, w_attention)
        attention_score = torch.reshape(aggregated_vector, [-1, max_tree_size, 1])

        mask = mask.unsqueeze(-1)
        attention_score = attention_score.masked_fill(mask == 0, -1e12)

        attention_weights = torch.sigmoid(attention_score)

        nodes = nodes * mask
        weighted_average_nodes = torch.mean(input=torch.multiply(nodes, attention_weights), dim=1)

        return weighted_average_nodes


class ConvolutionLayer(nn.Module):
    def __init__(self, config):
        super(ConvolutionLayer, self).__init__()
        self.config = config
        self.conv_num = self.config['num_conv_layers']
        config['output_size'] = self.config['conv_output']
        config['feature_size'] = config['embedding_dim']  # embedding_dim是最终输出的维数，conv_output是卷积输出的维数
        self.conv_nodes = nn.ModuleList([ConvNode(config=config) for _ in range(self.conv_num)])

    def forward(self, nodes, children, children_embedding):
        for conv_node in self.conv_nodes:
            nodes = conv_node(nodes, children, children_embedding)
            children_embedding = gen_children_embeddings(nodes, children, self.config['conv_output'])
        return nodes


class TBCNNEncoder(nn.Module):
    def __init__(self, config):
        super(TBCNNEncoder, self).__init__()
        self.config = config

        self.node_type_embedding_dim = config['type_embedding_dim']
        self.node_token_embedding_dim = config['token_embedding_dim']
        self.node_embedding_dim = config['node_embedding_dim']

        # Embedding
        self.node_type_embedding_layer = nn.Embedding(num_embeddings=config['num_types'],
                                                      embedding_dim=self.node_type_embedding_dim,
                                                      padding_idx=0)
        self.node_token_embedding_layer = nn.Embedding(num_embeddings=config['num_tokens'],
                                                       embedding_dim=self.node_token_embedding_dim,
                                                       padding_idx=0)

        self.parent_node_embedding_layer = nn.Linear(self.node_type_embedding_dim + self.node_token_embedding_dim,
                                                     self.node_embedding_dim)

        # output is (batch_size, max_tree_size, output_size)
        self.conv_layer = ConvolutionLayer(config)

        self.aggr_layer = AttentionLayer(config['output_size'])

    def forward(self, node_type, node_tokens, children_index):
        parent_node_type_embeddings = self.node_type_embedding_layer(node_type)
        parent_node_token_embeddings = torch.sum(self.node_token_embedding_layer(node_tokens), dim=2)
        parent_node_embeddings = torch.cat([parent_node_type_embeddings, parent_node_token_embeddings], -1)
        parent_node_embeddings = self.parent_node_embedding_layer(parent_node_embeddings)
        children_embeddings = gen_children_embeddings(parent_node_embeddings, children_index, self.node_embedding_dim)

        conv_output = self.conv_layer(parent_node_embeddings, children_index, children_embeddings)
        mask = (node_type != 0)
        code_vector = self.aggr_layer(conv_output, mask)

        return code_vector, conv_output
