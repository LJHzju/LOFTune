import array

from sqlglot import parse_one, exp
from tree_sitter import Language, Parser
from config.encoder_config import tree_sitter_sql_lib_path

IGNORE_TYPES = ["\n", ",", ".", "'", "(", ")", ";", "AS", "comment"]
TABLE = 1
COLUMN = 2
ALIAS = 3
OTHER = 0


def get_table_column_names(text):
    parsed = parse_one(text, read='spark')

    cte_names = [_.alias for _ in parsed.find_all(exp.CTE)]
    tables = [_.name for _ in parsed.find_all(exp.Table)]
    columns = [_.name for _ in parsed.find_all(exp.Column)]
    table_aliases = [_.name for _ in parsed.find_all(exp.TableAlias)]
    column_aliases = [_.alias for _ in parsed.find_all(exp.Alias)]

    return set(tables) - set(cte_names), set(table_aliases), set(columns), set(column_aliases)


class ASTParser:
    def __init__(self, type_vocab=None, token_vocab=None):
        SQL_LANGUAGE = Language(tree_sitter_sql_lib_path, 'sql')

        self.sql_parser = Parser()
        self.sql_parser.set_language(SQL_LANGUAGE)

        self.type_vocab = type_vocab
        self.token_vocab = token_vocab

    def get_node_cnt(self, tree):
        root = tree.root_node

        ignore_types = IGNORE_TYPES
        num_nodes = 0
        queue = [root]

        while queue:
            current_node = queue.pop(0)
            num_nodes += 1

            for child in current_node.children:
                child_type = str(child.type)
                if child_type not in ignore_types:
                    queue.append(child)

        return num_nodes

    # Simplify the AST
    def simplify_ast(self, tree, text):
        root = tree.root_node
        table_names, table_aliases, column_names, column_aliases = get_table_column_names(text)

        ignore_types = IGNORE_TYPES
        num_nodes = 0
        root_type = str(root.type)
        root_type_id = self.type_vocab.encode(root_type).ids[0] if self.type_vocab is not None else 0
        queue = [root]

        root_json = {
            "node_type_id": root_type_id,
            "node_tokens_id": [],
            "children": []
        }

        queue_json = [root_json]
        while queue:
            current_node = queue.pop(0)
            current_node_json = queue_json.pop(0)
            num_nodes += 1

            for child in current_node.children:
                child_type = str(child.type)
                if child_type not in ignore_types:
                    queue.append(child)

                    child_type_id = self.type_vocab.encode(child_type).ids[0] if self.type_vocab is not None else 0

                    child_token = ""
                    child_sub_tokens_id = []

                    if len(child.children) == 0:
                        child_token = text[child.start_byte: child.end_byte]
                        if self.token_vocab is not None:
                            output = self.token_vocab.encode(child_token)
                            child_sub_tokens_id = output.ids

                        if child_type == 'identifier':
                            if current_node.type in ['cte', 'alias']:
                                node_tag = ALIAS
                            elif current_node.type == 'dotted_name':
                                if text[child.start_byte - 1] == '.':
                                    if child_token in column_names:
                                        node_tag = COLUMN
                                    elif child_token in column_aliases:
                                        node_tag = ALIAS
                                else:
                                    if child_token in table_names:
                                        node_tag = TABLE
                                    elif child_token in table_aliases:
                                        node_tag = ALIAS
                            elif current_node.type in ['from_clause', 'join_clause']:
                                if child_token in table_names:
                                    node_tag = TABLE
                                elif child_token in table_aliases:
                                    node_tag = ALIAS
                            else:
                                if child_token in column_names:
                                    node_tag = COLUMN
                                elif child_token in column_aliases:
                                    node_tag = ALIAS

                    if len(child_sub_tokens_id) == 0:
                        child_sub_tokens_id.append(0)
                    else:
                        child_sub_tokens_id = array.array('h', (x for x in child_sub_tokens_id if x != 0))

                    child_json = {
                        "node_type_id": child_type_id,
                        "node_tag": node_tag,
                        "node_tokens_id": child_sub_tokens_id,
                        "children": []
                    }

                    current_node_json['children'].append(child_json)
                    queue_json.append(child_json)

        return root_json, num_nodes

    def parse(self, sql):
        return self.sql_parser.parse(bytes(sql, "utf8"))
