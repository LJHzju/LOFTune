from sql_encoder.model.aggr_layer import *
from sql_encoder.model.conv_layer import ConvolutionLayer


class TBCNNEncoder(nn.Module):
    def __init__(self, config):
        super(TBCNNEncoder, self).__init__()
        self.config = config

        self.node_type_embedding_dim = config['type_embedding_dim']
        self.node_token_embedding_dim = config['token_embedding_dim']
        self.node_embedding_dim = config['node_embedding_dim']
        self.dropout_rate = config['dropout_rate']

        # Embedding
        self.node_type_embedding_layer = nn.Embedding(num_embeddings=config['num_types'],
                                                      embedding_dim=self.node_type_embedding_dim,
                                                      padding_idx=0)
        self.node_token_embedding_layer = nn.Embedding(num_embeddings=config['num_tokens'],
                                                       embedding_dim=self.node_token_embedding_dim,
                                                       padding_idx=0)
        self.node_token_embedding_dropout = nn.Dropout(self.dropout_rate)

        self.parent_node_embedding_layer = nn.Linear(self.node_type_embedding_dim + self.node_token_embedding_dim,
                                                     self.node_embedding_dim)

        self.conv_layers = nn.ModuleList([ConvolutionLayer(self.node_embedding_dim, self.node_embedding_dim, self.dropout_rate)
                                          for _ in range(config['num_conv_layers'])])

        self.aggr_layer = SelfAttentionLayer(self.node_embedding_dim)

    def forward(self, node_type, node_tokens, children_index):
        mask = node_type != 0
        parent_node_type_embeddings = self.node_type_embedding_layer(node_type)
        parent_node_token_embeddings = torch.sum(self.node_token_embedding_layer(node_tokens), dim=2)
        parent_node_token_embeddings = self.node_token_embedding_dropout(parent_node_token_embeddings)

        parent_node_embeddings = torch.cat([parent_node_type_embeddings, parent_node_token_embeddings], -1)
        parent_node_embeddings = self.parent_node_embedding_layer(parent_node_embeddings)
        parent_node_embeddings = parent_node_embeddings * mask.unsqueeze(-1).float()

        conv_output = parent_node_embeddings
        for conv in self.conv_layers:
            conv_output = conv(conv_output, children_index)
        conv_output = conv_output * mask.unsqueeze(-1).float()

        code_vector = self.aggr_layer(conv_output, mask)

        return code_vector, conv_output
