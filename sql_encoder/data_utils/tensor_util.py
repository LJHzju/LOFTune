import numpy as np


def transform_tree_to_index(tree):
    node_type = []
    node_type_id = []
    node_tokens_id = []
    node_index = []
    children_index = []

    queue = [(tree, -1)]
    while queue:
        node, parent_ind = queue.pop(0)
        node_ind = len(node_type)
        queue.extend([(child, node_ind) for child in node['children']])
        children_index.append([])
        if parent_ind > -1:
            children_index[parent_ind].append(node_ind)

        node_type.append(node["node_type"])
        node_type_id.append(node["node_type_id"])
        node_tokens_id.append(node["node_tokens_id"])
        node_index.append(node_ind)

    data = {"node_type_id": node_type_id, "node_tokens_id": node_tokens_id, "children_index": children_index}

    return data


def trees_to_batch_tensors(all_tree_indices):
    batch_node_type_id = []
    batch_node_tokens_id = []
    batch_children_index = []
    batch_subtree_id = []

    for tree_indices in all_tree_indices:
        batch_node_type_id.append(tree_indices["node_type_id"])
        batch_node_tokens_id.append(tree_indices["node_tokens_id"])
        batch_children_index.append(tree_indices["children_index"])

        if "subtree_id" in tree_indices:
            batch_subtree_id.append(tree_indices["subtree_id"])

    # [[]]
    # batch_size × max_tree_size
    batch_node_type_id = pad_batch_2D(batch_node_type_id)
    # [[[]]]
    # batch_size × max_tree_size × max_token_length
    batch_node_tokens_id = pad_batch_3D(batch_node_tokens_id)
    # [[[]]]
    # batch_size × max_tree_size × max_children
    batch_children_index = pad_batch_3D(batch_children_index)

    batch_obj = {
        "batch_node_type_id": batch_node_type_id,
        "batch_node_tokens_id": batch_node_tokens_id,
        "batch_children_index": batch_children_index,
    }

    if len(batch_subtree_id) != 0:
        batch_obj["batch_subtree_id"] = batch_subtree_id

    return batch_obj


def pad_batch_2D(batch):
    max_2nd_D = max([len(x) for x in batch])

    tensor = np.zeros((len(batch), max_2nd_D), dtype=np.int32)
    for dim_1, _1 in enumerate(batch):
        for dim_2, value in enumerate(_1):
            if value != 0:
                tensor[dim_1][dim_2] = value
    return tensor


def pad_batch_3D(batch):
    max_2nd_D = max([len(x) for x in batch])
    max_3rd_D = max([len(c) for n in batch for c in n])

    tensor = np.zeros((len(batch), max_2nd_D, max_3rd_D), dtype=np.int32)
    for dim_1, _1 in enumerate(batch):
        for dim_2, _2 in enumerate(_1):
            for dim_3, value in enumerate(_2):
                if value != 0:
                    tensor[dim_1][dim_2][dim_3] = value
    return tensor
