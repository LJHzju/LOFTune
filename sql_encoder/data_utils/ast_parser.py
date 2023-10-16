from tree_sitter import Language, Parser
from config.encoder_config import tree_sitter_sql_lib_path


class ASTParser:
    def __init__(self, type_vocab=None, token_vocab=None):
        SQL_LANGUAGE = Language(tree_sitter_sql_lib_path, 'sql')

        self.sql_parser = Parser()
        self.sql_parser.set_language(SQL_LANGUAGE)

        self.type_vocab = type_vocab
        self.token_vocab = token_vocab
        self.ignore_types = ["\n", ",", ".", "(", ")", "'", ";", "comment"]

    # Simplify the AST
    def simplify_ast(self, tree, text):
        root = tree.root_node

        num_nodes = 0
        root_type = str(root.type)
        root_type_id = self.type_vocab.encode(root_type).ids[0]
        queue = [root]

        root_json = {
            "node_type": root_type,
            "node_type_id": root_type_id,
            "node_tokens": [],
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
                if child_type not in self.ignore_types:
                    queue.append(child)

                    child_type_id = self.type_vocab.encode(child_type).ids[0]

                    child_token = ""
                    child_sub_tokens_id = []
                    child_sub_tokens = []

                    has_child = len(child.children) > 0

                    if not has_child:
                        child_token = text[child.start_byte: child.end_byte]
                        output = self.token_vocab.encode(child_token)
                        child_sub_tokens_id = output.ids
                        child_sub_tokens = output.tokens

                    if len(child_sub_tokens_id) == 0:
                        child_sub_tokens_id.append(0)
                    else:
                        child_sub_tokens_id = [x for x in child_sub_tokens_id if x != 0]

                    child_json = {
                        "node_type": child_type,
                        "node_type_id": child_type_id,
                        "node_tokens": child_sub_tokens,
                        "node_tokens_id": child_sub_tokens_id,
                        "children": []
                    }

                    current_node_json['children'].append(child_json)
                    queue_json.append(child_json)

        return root_json, num_nodes

    def get_node_cnt(self, tree):
        root = tree.root_node
        num_nodes = 0
        queue = [root]

        while queue:
            current_node = queue.pop(0)
            num_nodes += 1

            for child in current_node.children:
                child_type = str(child.type)
                if child_type not in self.ignore_types:
                    queue.append(child)

        return num_nodes

    def parse(self, sql):
        return self.sql_parser.parse(bytes(sql, "utf8"))
