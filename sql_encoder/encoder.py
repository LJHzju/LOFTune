import os
from tokenizers import Tokenizer

from sql_encoder.data_utils import tensor_util
from sql_encoder.data_utils.ast_parser import ASTParser
from sql_encoder.model.tbcnn_encoder import TBCNNEncoder
from config.encoder_config import *


class SQLEncoder:
    def __init__(self):
        self.type_vocab = Tokenizer.from_file(data_path['node_type_vocab_model_path'])
        self.token_vocab = Tokenizer.from_file(data_path['node_token_vocab_model_path'])

        self.ast_parser = ASTParser(type_vocab=self.type_vocab, token_vocab=self.token_vocab)

        self.encoder_checkpoint_file_path = train_config['encoder_checkpoint_file_path']

        encoder_config['num_types'] = self.type_vocab.get_vocab_size()
        encoder_config['num_tokens'] = self.token_vocab.get_vocab_size()

        self.device = device

        self.encoder = TBCNNEncoder(encoder_config).to(self.device)

    def load_encoder(self):
        if not os.path.exists(self.encoder_checkpoint_file_path):
            return False
        checkpoint = torch.load(self.encoder_checkpoint_file_path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['model_state_dict'])
        return True

    def sqls_to_tensors(self, batch_sql_snippets):
        batch_tree_indexes = []
        for sql_snippet in batch_sql_snippets:
            ast = self.ast_parser.parse(sql_snippet)
            tree_representation, _ = self.ast_parser.simplify_ast(ast, sql_snippet)
            tree_indexes = tensor_util.transform_tree_to_index(tree_representation)
            batch_tree_indexes.append(tree_indexes)

        tensors = tensor_util.trees_to_batch_tensors(batch_tree_indexes)
        return tensors

    def encode(self, sqls):
        tensors = self.sqls_to_tensors(sqls)
        self.encoder.eval()
        with torch.no_grad():
            batch_node_type = torch.tensor(tensors["batch_node_type_id"]).to(self.device)
            batch_node_tokens = torch.tensor(tensors["batch_node_tokens_id"]).to(self.device)
            batch_children_index = torch.tensor(tensors["batch_children_index"]).to(self.device)
            code_vector, _ = self.encoder(batch_node_type, batch_node_tokens, batch_children_index)

        return code_vector
