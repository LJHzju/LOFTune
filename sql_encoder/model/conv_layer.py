import math

import torch
from torch import nn
import torch.nn.functional as F


def gen_children_embeddings(parent_node_embedding, children_index):
    batch_size, num_nodes, embedding_dim = parent_node_embedding.size()
    _, _, max_children = children_index.size()

    mask = children_index == 0

    # Gather the embeddings
    # batch_size × max_tree_size × max_children × embedding_dim
    # for a node: children: [1, 2, 3, 0, 0] -> [[1, 1, 1, ..., 1(x embedding_dim)], [2, 2, 2, ..., 2], ...]
    children_index_expanded = children_index.unsqueeze(-1).expand(-1, -1, -1, embedding_dim)
    # batch_size × max_tree_size × max_children × embedding_dim
    # for a node: [0.1, 0.2, 0.3] -> [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], ..., [...]], (x max_children)

    child_embeddings = torch.gather(parent_node_embedding.unsqueeze(2).expand(-1, -1, max_children, -1), 1,
                                    children_index_expanded)

    # Apply the mask to set embeddings of padded positions to zero
    child_embeddings[mask.unsqueeze(-1).expand(-1, -1, -1, embedding_dim)] = 0

    # child_embeddings now has the shape (batch_size, num_nodes, max_children, embedding_dim)
    # and embeddings for padded indices are zero
    return child_embeddings


class ConvolutionLayer(nn.Module):
    def __init__(self, node_embedding_dim, conv_output_dim, dropout_rate):
        super(ConvolutionLayer, self).__init__()
        self.conv_input_dim = node_embedding_dim
        self.conv_output_dim = conv_output_dim
        std = 1.0 / math.sqrt(self.conv_input_dim)
        self.w_t = nn.Parameter(
            torch.normal(size=(self.conv_input_dim, self.conv_output_dim), std=std, mean=0))
        self.w_l = nn.Parameter(
            torch.normal(size=(self.conv_input_dim, self.conv_output_dim), std=std, mean=0))
        self.w_r = nn.Parameter(
            torch.normal(size=(self.conv_input_dim, self.conv_output_dim), std=std, mean=0))
        self.conv = nn.Parameter(
            torch.normal(size=(self.conv_output_dim,), std=math.sqrt(2.0 / self.conv_input_dim), mean=0))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, nodes, children):
        # nodes is shape (batch_size x max_tree_size x feature_size)
        # children is shape (batch_size x max_tree_size x max_children)

        # children_vectors will have shape
        # (batch_size x max_tree_size x max_children x feature_size)

        children_vectors = gen_children_embeddings(nodes, children)
        tree_tensor = torch.cat((nodes.unsqueeze(2), children_vectors), dim=2)
        del children_vectors

        # coefficient tensors are shape (batch_size x max_tree_size x max_children + 1)
        c_t = eta_t(children)
        c_r = eta_r(children, c_t)
        c_l = eta_l(children, c_t, c_r)

        # concatenate the position coefficients into a tensor
        # (batch_size x max_tree_size x max_children + 1 x 3)
        coef = torch.stack((c_t, c_r, c_l), dim=3)

        # stack weight matrices on top to make a weight tensor
        # (3, feature_size, output_size)
        weights = torch.stack((self.w_t, self.w_r, self.w_l), dim=0)

        # combine
        batch_size, max_tree_size, max_children = children.shape
        x = batch_size * max_tree_size
        y = 1 + max_children
        # result = torch.reshape(tree_tensor, (x, y, self.node_embedding_dim))
        # coef = coef.view(x, y, 3)
        # result = torch.transpose(result, 1, 2)
        # result = torch.matmul(result, coef).view(batch_size, max_tree_size, 3, self.node_embedding_dim)
        result = tree_tensor.view(x, y, self.conv_input_dim)
        # bmm((x, emb_dim, y), (x, y, 3)) -> (x, emb_dim, 3)
        result = torch.bmm(result.transpose(1, 2), coef.view(x, y, 3)).view(batch_size, max_tree_size, 3, self.conv_input_dim)

        # output is (batch_size, max_tree_size, output_size)
        # (bs, node, 3, emb_dim) * (3, emb_dim, out_dim) ->
        result = torch.tensordot(result, weights, [[2, 3], [0, 1]])

        # output is (batch_size, max_tree_size, output_size)
        output = F.leaky_relu(result + self.conv)
        output = self.dropout(output)
        return output


def eta_t(children):
    """Compute weight matrix for how much each vector belongs to the 'top'"""
    # children is shape (batch_size x max_tree_size x max_children)

    batch_size, max_tree_size, max_children = children.shape
    device = children.device

    # eta_t is shape (batch_size x max_tree_size x max_children + 1)
    result = torch.zeros(batch_size, max_tree_size, 1 + max_children, device=device)
    result[:, :, 0] = 1
    return result


def eta_r(children, t_coef):
    """Compute weight matrix for how much each vector belongs to the 'right'"""
    # children is shape (batch_size x max_tree_size x max_children)
    batch_size, max_tree_size, max_children = children.shape
    device = children.device

    # num_siblings is shape (batch_size x max_tree_size x 1)
    num_siblings = torch.count_nonzero(children, dim=2).float().unsqueeze(2)

    # num_siblings is shape (batch_size x max_tree_size x max_children + 1)
    num_siblings = torch.tile(
        num_siblings, (1, 1, max_children + 1)
    )
    # creates a mask of 1's and 0's where 1 means there is a child there
    # has shape (batch_size x max_tree_size x max_children + 1)
    mask = torch.cat(
        (torch.zeros((batch_size, max_tree_size, 1), device=device),
         children.clamp_max(1)), dim=2
    )

    # child indices for every tree (batch_size x max_tree_size x max_children + 1)
    p = torch.arange(-1.0, max_children, 1.0, dtype=torch.float32, device=device).expand_as(mask)
    child_indices = torch.multiply(p, mask)

    # weights for every tree node in the case that num_siblings = 0
    # shape is (batch_size x max_tree_size x max_children + 1)
    singles = torch.zeros(batch_size, max_tree_size, 1 + max_children, dtype=torch.float, device=device)
    singles[:, :, 1] = 0.5

    # eta_r is shape (batch_size x max_tree_size x max_children + 1)
    return torch.where(
        num_siblings == 1.0,
        # avoid division by 0 when num_siblings == 1
        singles,
        # the normal case where num_siblings != 1
        (1.0 - t_coef) * (child_indices / (num_siblings - 1.0))
    )


def eta_l(children, coef_t, coef_r):
    """Compute weight matrix for how much each vector belongs to the 'left'"""
    batch_size, max_tree_size, max_children = children.shape
    device = children.device
    # creates a mask of 1's and 0's where 1 means there is a child there
    # has shape (batch_size x max_tree_size x max_children + 1)
    mask = torch.cat(
        (torch.zeros((batch_size, max_tree_size, 1), device=device),
         children.clamp_max(1)),
        dim=2)

    # eta_l is shape (batch_size x max_tree_size x max_children + 1)
    return (1.0 - coef_t) * (1.0 - coef_r) * mask