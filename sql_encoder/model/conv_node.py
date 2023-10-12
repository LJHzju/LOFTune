import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvNode(nn.Module):
    def __init__(self, config):
        super(ConvNode, self).__init__()
        self.config = config
        std = 1.0 / math.sqrt(self.config['feature_size'])
        self.w_t = nn.Parameter(
            torch.normal(size=(self.config['feature_size'], self.config['output_size']), std=std, mean=0))
        self.w_l = nn.Parameter(
            torch.normal(size=(self.config['feature_size'], self.config['output_size']), std=std, mean=0))
        self.w_r = nn.Parameter(
            torch.normal(size=(self.config['feature_size'], self.config['output_size']), std=std, mean=0))
        self.conv = nn.Parameter(
            torch.normal(size=(self.config['output_size'],), std=math.sqrt(2.0 / self.config['feature_size']), mean=0))
        self.dropout = nn.Dropout(self.config['dropout_rate'])

    def forward(self, nodes, children, children_vectors):
        # nodes is shape (batch_size x max_tree_size x feature_size)
        # children is shape (batch_size x max_tree_size x max_children)

        # children_vectors will have shape
        # (batch_size x max_tree_size x max_children x feature_size)

        # add a 4th dimension to the nodes tensor
        # nodes is now shape (batch_size x max_tree_size x 1 x feature_size)
        nodes = torch.unsqueeze(nodes, dim=2)
        # tree_tensor is shape
        # (batch_size x max_tree_size x max_children + 1 x feature_size)
        tree_tensor = torch.cat((nodes, children_vectors), dim=2)

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
        batch_size = children.shape[0]
        max_tree_size = children.shape[1]
        max_children = children.shape[2]

        # reshape for matrix multiplication
        x = batch_size * max_tree_size
        y = max_children + 1
        result = torch.reshape(tree_tensor, (x, y, self.config['feature_size']))
        coef = torch.reshape(coef, (x, y, 3))
        result = torch.transpose(result, 1, 2)
        result = torch.matmul(result, coef)
        result = torch.reshape(result, (batch_size, max_tree_size, 3, self.config['feature_size']))

        # output is (batch_size, max_tree_size, output_size)
        result = torch.tensordot(result, weights, [[2, 3], [0, 1]])

        # output is (batch_size, max_tree_size, output_size)
        output = F.leaky_relu(result + self.conv)
        output_dropout = self.dropout(output)
        return output_dropout


def eta_t(children):
    """Compute weight matrix for how much each vector belongs to the 'top'"""
    # children is shape (batch_size x max_tree_size x max_children)
    batch_size = children.shape[0]
    max_tree_size = children.shape[1]
    max_children = children.shape[2]
    device = children.device
    # eta_t is shape (batch_size x max_tree_size x max_children + 1)
    return torch.tile(torch.unsqueeze(torch.concat(
        [torch.ones((max_tree_size, 1), device=device), torch.zeros((max_tree_size, max_children), device=device)],
        dim=1), dim=0,
    ), (batch_size, 1, 1))


def eta_r(children, t_coef):
    """Compute weight matrix for how much each vector belongs to the 'right'"""
    # children is shape (batch_size x max_tree_size x max_children)
    batch_size = children.shape[0]
    max_tree_size = children.shape[1]
    max_children = children.shape[2]
    device = children.device

    # num_siblings is shape (batch_size x max_tree_size x 1)
    num_siblings = torch.count_nonzero(children, dim=2).float().reshape(batch_size, max_tree_size, 1)

    # num_siblings is shape (batch_size x max_tree_size x max_children + 1)
    num_siblings = torch.tile(
        num_siblings, (1, 1, max_children + 1)
    )
    # creates a mask of 1's and 0's where 1 means there is a child there
    # has shape (batch_size x max_tree_size x max_children + 1)
    mask = torch.cat(
        [torch.zeros((batch_size, max_tree_size, 1), device=device),
         torch.minimum(children, torch.ones(children.shape, device=device))],
        dim=2
    )

    # child indices for every tree (batch_size x max_tree_size x max_children + 1)
    p = torch.tile(
        torch.unsqueeze(
            torch.unsqueeze(
                torch.arange(-1.0, max_children, 1.0, dtype=torch.float32, device=device),
                dim=0
            ),
            dim=0
        ),
        (batch_size, max_tree_size, 1)
    )
    child_indices = torch.multiply(p, mask)

    # weights for every tree node in the case that num_siblings = 0
    # shape is (batch_size x max_tree_size x max_children + 1)
    singles = torch.cat((torch.zeros((batch_size, max_tree_size, 1), device=device).float(),
                         torch.tensor((), dtype=torch.float, device=device).new_full((batch_size, max_tree_size, 1), 0.5),
                         torch.zeros((batch_size, max_tree_size, max_children - 1), device=device).float()), 2)

    # eta_r is shape (batch_size x max_tree_size x max_children + 1)
    return torch.where(
        num_siblings == 1.0,
        # avoid division by 0 when num_siblings == 1
        singles,
        # the normal case where num_siblings != 1
        torch.multiply((1.0 - t_coef), torch.divide(child_indices, num_siblings - 1.0))
    )


def eta_l(children, coef_t, coef_r):
    """Compute weight matrix for how much each vector belongs to the 'left'"""
    batch_size = children.shape[0]
    max_tree_size = children.shape[1]
    device = children.device
    # creates a mask of 1's and 0's where 1 means there is a child there
    # has shape (batch_size x max_tree_size x max_children + 1)
    mask = torch.cat(
        [torch.zeros((batch_size, max_tree_size, 1), device=device),
         torch.minimum(children, torch.ones(children.shape, device=device))],
        dim=2)

    # eta_l is shape (batch_size x max_tree_size x max_children + 1)
    return torch.multiply(
        torch.multiply((1.0 - coef_t), (1.0 - coef_r)), mask
    )