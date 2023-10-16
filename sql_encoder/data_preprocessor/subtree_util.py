from copy import deepcopy


ignore_types = ["\n", ",", ".", "'", "(", ")", ";", "alias"]
ignore_children_types = ["string", "NULL"]

unfold_expression_types = ['unary_expression', 'binary_expression', 'boolean_expression']


def gen_tree_str(node):
    node_str = f"{node['type']}"
    if len(node['children']) > 0:
        node_str += "("
    for index, child in enumerate(node['children']):
        node_str += f"{gen_tree_str(child)}"
        if index < len(node['children']) - 1:
            node_str += ", "
    if len(node['children']) > 0:
        node_str += ")"
    return node_str


def swap_node(node_1, node_2):
    node_1_type = gen_tree_str(node_1)
    node_2_type = gen_tree_str(node_2)
    if node_1_type < node_2_type:
        return deepcopy(node_1), deepcopy(node_2)
    else:
        return deepcopy(node_2), deepcopy(node_1)


def modify_subtree_by_commutation(current_node):
    if current_node['type'] == 'boolean_expression':
        if len(current_node['children']) == 3:
            operand_1, operand_2 = swap_node(current_node['children'][0], current_node['children'][2])
            current_node['children'][0] = operand_1
            current_node['children'][2] = operand_2

    # + / *
    elif current_node['type'] == 'binary_expression':
        if len(current_node['children']) == 3:
            operator = current_node['children'][1]['type']
            if operator in ['+', '*']:
                operand_1, operand_2 = swap_node(current_node['children'][0], current_node['children'][2])
                current_node['children'][0] = operand_1
                current_node['children'][2] = operand_2


def extract_subtree_of_node(node):
    subtree = {
        "type": node['type'],
        "parent": node['parent'],
        "children": []
    }
    children_nodes = []
    for child in node['children']:
        child_type = child['type']

        subtree_child = {
            "type": child_type,
            "parent": node['type'],
            "children": []
        }

        if child_type in unfold_expression_types:
            for child_child in child['children']:
                subtree_child['children'].append({"type": child_child['type'], "parent": child_type, "children": []})
            modify_subtree_by_commutation(subtree_child)
            children_nodes.append(child)

        elif child_type == 'function_call':
            function_name_node = child['children'][0] if child['children'][0]['type'] != 'LATERAL' else child['children'][1]
            subtree_child['children'].append({"type": function_name_node['type'], "parent": child_type, "children": []})
            children_nodes.append(child)

        elif child_type == 'asterisk_expression':
            subtree_child = child

        else:
            children_nodes.append(child)

        subtree['children'].append(subtree_child)

    modify_subtree_by_commutation(subtree)
    return subtree, children_nodes


def extract_separated_subtrees(root):
    subtrees = []
    for child in root['children']:
        separated_subtree = {
            "type": root['type'],
            "parent": root['parent'],
            "children": [child]
        }
        subtrees.append(separated_subtree)
    return subtrees


def extract_group_by_subtrees(node):
    subtrees = []
    children_nodes = []
    for child in node['children']:
        if child['type'] == 'having_clause':
            children_nodes.append(child)
            continue
        elif child['type'] == 'GROUP_BY':
            continue
        subtree_child, group_expression_children = extract_subtree_of_node(child)  # group_expression
        separated_subtree = {
            "type": node['type'],
            "parent": node['parent'],
            "children": [subtree_child]
        }
        children_nodes.extend(group_expression_children)
        subtrees.append(separated_subtree)
    return subtrees, children_nodes


def extract_order_by_subtrees(node):
    subtrees = []
    children_nodes = []
    for child in node['children']:
        if child['type'] == 'ORDER_BY':
            continue
        subtree_child, _ = extract_subtree_of_node(child)  # order_expression
        separated_subtree = {
            "type": node['type'],
            "parent": node['parent'],
            "children": [subtree_child]
        }
        children_nodes.append(child['children'][0])
        subtrees.append(separated_subtree)
    return subtrees, children_nodes


def extract_explicit_join_subtree(node):
    subtree = {
        "type": node['type'],
        "parent": node['parent'],
        "children": []
    }
    children_nodes = []
    for child in node['children']:
        child_type = child['type']

        if child_type == 'join_type':
            subtree_child = child
        elif child_type == 'join_condition':
            subtree_child, _ = extract_subtree_of_node(child)
            children_nodes.append(child['children'][1])
        elif child_type == 'AS':
            continue
        else:
            subtree_child = {
                "type": child_type,
                "parent": node['type'],
                "children": []
            }
            children_nodes.append(child)

        subtree['children'].append(subtree_child)

    return subtree, children_nodes


def extract_select_clause_body_subtrees(node):
    subtrees = []

    child_types = [child['type'] for child in node['children']]
    children = []
    for index, child in enumerate(child_types):
        if child == 'AS':
            continue
        else:
            children.append([index])

    for child in children:
        root_children = []
        for index in child:
            root_children.append({"type": node['children'][index]['type'], "parent": node['type'], "children": node['children'][index]['children']})
        separated_subtree = {
            "type": node['type'],
            "parent": node['parent'],
            "children": root_children
        }
        subtrees.append(separated_subtree)

    return subtrees


def extract_implicit_join_subtrees(node):
    subtrees = []

    child_types = [child['type'] for child in node['children'][1:]]
    children = []
    for index, child in enumerate(child_types):
        if child == 'AS':
            continue
        else:
            children.append([index + 1])

    for child in children:
        root_children = []
        for index in child:
            root_children.append({"type": node['children'][index]['type'], "parent": node['type'], "children": []})
        separated_subtree = {
            "type": node['type'],
            "parent": node['parent'],
            "children": root_children
        }
        subtrees.append(separated_subtree)

    children_nodes = []
    for child in node['children']:
        if child['type'] == 'select_subexpression':
            children_nodes.append(child)
    return subtrees, children_nodes


def extract_subquery_root_subtree(node):
    subtree = {
        "type": node['type'],
        "parent": node['parent'],
        "children": []
    }
    children_nodes = []
    for child in node['children']:
        child_type = child['type']

        subtree_child = {
            "type": child_type,
            "parent": node['type'],
            "children": []
        }

        if child_type == 'select_statement':
            for child_child in child['children']:
                subtree_child['children'].append({"type": child_child['type'], "parent": child_type, "children": []})
                children_nodes.append(child_child)

        subtree['children'].append(subtree_child)

    return subtree, children_nodes


def extract_with_subtrees(node):
    subtrees = []
    children_nodes = []
    for child in node['children'][1:]:
        # cte
        subtree, nodes = extract_subquery_root_subtree(child)
        separated_subtree = {
            "type": node['type'],
            "parent": node['parent'],
            "children": [subtree]
        }
        subtrees.append(separated_subtree)
        children_nodes.extend(nodes)
    return subtrees, children_nodes


def extract_subtree(root):
    subtrees = []
    queue = [root]
    while queue:
        current_node = queue.pop(0)
        current_node_type = current_node['type']

        if len(current_node['children']) == 0:
            continue

        if current_node_type == 'from_clause':
            if current_node['children'][1]['type'] != 'join_clause':
                implicit_join_subtrees, children_nodes = extract_implicit_join_subtrees(current_node)
                subtrees.extend(implicit_join_subtrees)
                queue.extend(children_nodes)
            else:
                queue.extend(current_node['children'])

        elif current_node_type == 'join_clause':
            explicit_join_subtree, children_nodes = extract_explicit_join_subtree(current_node)
            subtrees.append(explicit_join_subtree)
            queue.extend(children_nodes)

        elif current_node_type == 'group_by_clause':
            group_by_subtrees, children_nodes = extract_group_by_subtrees(current_node)
            subtrees.extend(group_by_subtrees)
            queue.extend(children_nodes)

        elif current_node_type == 'order_by_clause':
            order_by_subtrees, children_nodes = extract_order_by_subtrees(current_node)
            subtrees.extend(order_by_subtrees)
            queue.extend(children_nodes)

        elif current_node_type == 'with_clause':
            with_subtrees, children_nodes = extract_with_subtrees(current_node)
            subtrees.extend(with_subtrees)
            queue.extend(children_nodes)

        elif current_node_type == 'select_clause_body':
            subtree, children_nodes = extract_subtree_of_node(current_node)
            separated_subtrees = extract_select_clause_body_subtrees(subtree)
            subtrees.extend(separated_subtrees)
            queue.extend(children_nodes)

        elif current_node_type == 'source_file' or current_node_type == 'tuple':
            subtree, children_nodes = extract_subtree_of_node(current_node)
            separated_subtrees = extract_separated_subtrees(subtree)
            subtrees.extend(separated_subtrees)
            queue.extend(children_nodes)

        elif current_node_type == 'select_clause':
            subtree, children_nodes = extract_subtree_of_node(current_node)
            queue.extend(children_nodes)

        elif current_node_type == 'select_subexpression':
            subtree, children_nodes = extract_subquery_root_subtree(current_node)
            subtrees.append(subtree)
            queue.extend(children_nodes)

        else:
            subtree, children_nodes = extract_subtree_of_node(current_node)
            subtrees.append(subtree)
            queue.extend(children_nodes)

    return subtrees


def extract_sub_queries(root):
    exclude_parents = ['source_file', 'set_select_statement']
    sub_queries = []
    queue = [root]
    while queue:
        current_node = queue.pop(0)

        if current_node['type'] == 'select_statement' and current_node['parent'] not in exclude_parents:
            sub_queries.append(deepcopy(current_node))
            current_node['children'] = []
        else:
            queue.extend(current_node['children'])

    return root, sub_queries


def convert_tree(root):
    queue = [root]
    root_json = {
        "type": root.type,
        "parent": None,
        "children": []
    }
    queue_json = [root_json]
    while queue:
        current_node = queue.pop(0)
        current_node_json = queue_json.pop(0)

        if current_node.type == "dotted_name":
            current_node_json["type"] = "identifier"
            continue

        elif current_node.type == "<>":
            current_node_json["type"] = "!="
            continue

        elif current_node.type in ignore_children_types:
            continue

        for index, child in enumerate(current_node.children):
            if child.type not in ignore_types:
                queue.append(child)

                child_json = {
                    "type": child.type,
                    "parent": current_node.type,
                    "children": []
                }

                if current_node.type == "function_call":
                    if (index == 0 and child.type != 'LATERAL') or (index == 1 and current_node.children[0].type == 'LATERAL'):
                        child_json = {
                            "type": str.upper(str(child.text, 'utf-8')),
                            "parent": current_node.type,
                            "children": []
                        }

                current_node_json['children'].append(child_json)
                queue_json.append(child_json)

    return root_json


def convert_subtrees_to_str(subtrees, is_sub_query):
    subtree_strs = []
    for subtree in subtrees:
        subtree_str = gen_tree_str(subtree)
        if is_sub_query:
            subtree_str = f"SUBQUERY({subtree_str})"
        subtree_strs.append(subtree_str)
    return subtree_strs


def extract_subtrees(tree):
    root = convert_tree(tree.root_node)

    subtrees = extract_subtree(root)
    all_subtree_strs = convert_subtrees_to_str(subtrees, is_sub_query=False)

    subtree_map = {}
    for subtree_str in all_subtree_strs:
        if subtree_str not in subtree_map.keys():
            subtree_map[subtree_str] = 0
        subtree_map[subtree_str] += 1

    return subtree_map
