from collections import deque
import array
import numpy as np


def transform_tree_to_index(tree, tree_size=600):
    node_type_id = np.zeros(tree_size, dtype=np.int16)
    node_tokens_id = np.zeros(tree_size, dtype=object)
    node_tag = np.zeros(tree_size, dtype=np.int16)
    children_index = np.zeros(tree_size, dtype=object)

    queue = deque([(tree, -1)])
    node_ind = 0
    while queue:
        node, parent_ind = queue.popleft()
        queue.extend([(child, node_ind) for child in node['children']])
        children_index[node_ind] = array.array('h')
        if parent_ind > -1:
            children_index[parent_ind].append(node_ind)

        node_tag[node_ind] = node["node_tag"]
        node_type_id[node_ind] = node["node_type_id"]
        node_tokens_id[node_ind] = node["node_tokens_id"]
        node_ind += 1

    max_tokens_length = max(len(tokens) for tokens in node_tokens_id)
    max_children_length = max(len(children) for children in children_index)

    # Create new padded arrays
    padded_node_tokens_id = np.zeros((len(node_tokens_id), max_tokens_length),
                                     dtype=np.int16)
    padded_children_index = np.zeros((len(children_index), max_children_length),
                                     dtype=np.int16)

    # Copy and pad data
    for i, tokens in enumerate(node_tokens_id):
        padded_node_tokens_id[i, :len(tokens)] = tokens

    for i, children in enumerate(children_index):
        padded_children_index[i, :len(children)] = children

    data = {"node_type_id": node_type_id,
            "node_tokens_id": padded_node_tokens_id,
            "children_index": padded_children_index,
            "node_tag": node_tag}

    return data


def trees_to_batch_tensors(all_tree_indices):
    batch_node_type_id = []
    batch_node_tag = []
    batch_node_tokens_id = []
    batch_children_index = []
    batch_subtree_id = []

    for tree_indices in all_tree_indices:
        batch_node_type_id.append(tree_indices["node_type_id"])
        batch_node_tokens_id.append(tree_indices["node_tokens_id"])
        batch_children_index.append(tree_indices["children_index"])
        if "node_tag" in tree_indices:
            batch_node_tag.append(tree_indices["node_tag"])
        if "subtree_id" in tree_indices:
            batch_subtree_id.append(tree_indices["subtree_id"])

    # [[]]
    # batch_size × max_tree_size
    batch_node_type_id = _pad_batch_2D(batch_node_type_id)
    # [[]]
    # batch_size × max_tree_size
    if len(batch_node_tag) > 0:
        batch_node_tag = _pad_batch_2D(batch_node_tag, fill_value=-1, dtype=np.int64)
    # [[]]
    # batch_size × max_tree_size
    if len(batch_subtree_id) > 0:
        batch_subtree_id = _pad_batch_2D(batch_subtree_id, fill_value=-1, dtype=np.int64)
    # [[[]]]
    # batch_size × max_tree_size × max_token_length
    batch_node_tokens_id = _pad_batch_3D(batch_node_tokens_id, dtype=np.int32)
    # [[[]]]
    # batch_size × max_tree_size × max_children
    batch_children_index = _pad_batch_3D(batch_children_index, dtype=np.int64)

    return {
        "batch_node_type_id": batch_node_type_id,
        "batch_node_tokens_id": batch_node_tokens_id,
        "batch_children_index": batch_children_index,
        "batch_node_tag": batch_node_tag,
        "batch_subtree_id": batch_subtree_id
    }


def _pad_batch_2D(batch, fill_value=0, dtype=np.int32):
    max_2nd_D = max([len(x) for x in batch])

    if fill_value == 0:
        tensor = np.zeros((len(batch), max_2nd_D), dtype=dtype)
    else:
        tensor = np.full((len(batch), max_2nd_D), fill_value=fill_value, dtype=dtype)
    for dim_1, _1 in enumerate(batch):
        tensor[dim_1, :len(_1)] = _1
    return tensor


def _pad_batch_3D(batch, dtype):
    max_2nd_D = max(x.shape[0] for x in batch)
    max_3rd_D = max(x.shape[1] for x in batch)

    tensor = np.zeros((len(batch), max_2nd_D, max_3rd_D), dtype=dtype)

    for dim_1, _1 in enumerate(batch):
        rows, cols = _1.shape
        tensor[dim_1, :rows, :cols] = _1

    return tensor
