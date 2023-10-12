import numpy as np
import os
from sql_encoder.data_preprocessor.subtree_util import extract_subtrees
from sql_encoder.data_utils.tensor_util import transform_tree_to_index
from sql_encoder.data_utils.ast_parser import ASTParser
from config.common import big_gap
from collections import defaultdict
from tokenizers import Tokenizer
import pickle
import concurrent.futures


def process_file(file_path, type_vocab, token_vocab, subtree_vocab):
    bucket_sizes = np.array(list(range(20, 7500, 20)))
    local_buckets = defaultdict(list)

    with open(file_path, "r", encoding='UTF-8') as f:
        sqls = f.readlines()
    print(f"load {file_path} completed...", flush=True)

    ast_parser = ASTParser(type_vocab=type_vocab, token_vocab=token_vocab)
    for code_snippet in sqls:
        ast = ast_parser.parse(code_snippet)
        tree_representation, tree_size = ast_parser.simplify_ast(ast, code_snippet)
        tree_indexes = transform_tree_to_index(tree_representation)
        tree_indexes["size"] = tree_size

        subtree_map = extract_subtrees(ast)
        subtree_id_map = {}
        for subtree, count in subtree_map.items():
            subtree_id = subtree_vocab.encode(subtree).ids[0]
            if subtree_id != 3:
                subtree_id_map[subtree_id] = count
        tree_indexes["subtree_id"] = list(subtree_id_map.keys())

        chosen_bucket_idx = np.argmax(bucket_sizes > tree_size)
        local_buckets[chosen_bucket_idx].append(tree_indexes)

    return local_buckets


class DatasetProcessor:
    def __init__(self, input_data_path, output_tensors_path, type_vocab, token_vocab, subtree_vocab):
        self.input_data_path = input_data_path
        self.output_tensors_path = output_tensors_path

        self.type_vocab = type_vocab
        self.token_vocab = token_vocab
        self.subtree_vocab = subtree_vocab

    def put_trees_into_buckets(self):
        buckets = defaultdict(list)
        file_paths = []
        index = 0

        for subdir, dirs, files in os.walk(self.input_data_path):
            for file in files:
                file_path = os.path.join(subdir, file)
                file_paths.append(file_path)

        with concurrent.futures.ProcessPoolExecutor(max_workers=int(os.cpu_count() / 2)) as executor:
            futures = {executor.submit(process_file, file_path, self.type_vocab, self.token_vocab, self.subtree_vocab): file_path
                       for file_path in file_paths}

            for future in concurrent.futures.as_completed(futures):
                local_buckets = future.result()

                for idx, trees in local_buckets.items():
                    buckets[idx].extend(trees)

                index += 1
                print(f"Processed {futures[future]}, {index} / {len(file_paths)}", flush=True)

        pickle.dump(buckets, open(self.output_tensors_path, "wb"))

        return buckets


def process_dataset(data_path):
    type_vocab = Tokenizer.from_file(data_path['node_type_vocab_model_path'])
    token_vocab = Tokenizer.from_file(data_path['node_token_vocab_model_path'])
    subtree_vocab = Tokenizer.from_file(data_path['subtree_vocab_model_path'])

    print("Training data process starts...", flush=True)
    if os.path.exists(data_path['training_data_path']):
        os.remove(data_path['training_data_path'])
    training_data_processor = DatasetProcessor(input_data_path=data_path['training_queries_path'],
                                               output_tensors_path=data_path['training_data_path'],
                                               type_vocab=type_vocab, token_vocab=token_vocab, subtree_vocab=subtree_vocab)
    training_data_processor.put_trees_into_buckets()
    print("Training data process finishes...", flush=True)
    print(big_gap, flush=True)

    print("Evaluation data process starts...", flush=True)
    if os.path.exists(data_path['evaluation_data_path']):
        os.remove(data_path['evaluation_data_path'])
    evaluation_data_processor = DatasetProcessor(input_data_path=data_path['evaluation_queries_path'],
                                                 output_tensors_path=data_path['evaluation_data_path'],
                                                 type_vocab=type_vocab, token_vocab=token_vocab, subtree_vocab=subtree_vocab)
    evaluation_data_processor.put_trees_into_buckets()
    print("Evaluation data process finishes...", flush=True)
    print(big_gap, flush=True)
