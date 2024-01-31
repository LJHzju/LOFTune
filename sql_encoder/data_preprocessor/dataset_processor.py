import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import os
import time
from subtree_util import extract_subtrees
from sql_encoder.data_utils.tensor_util import transform_tree_to_index
from sql_encoder.data_utils.ast_parser import ASTParser
from collections import defaultdict
from tokenizers import Tokenizer
from data_augmentor import augment_data, init_tokens
import concurrent.futures
import traceback
import array
import joblib


def process_file(file_path, local_buckets_path, type_vocab, token_vocab, subtree_vocab):
    bucket_sizes = np.array(list(range(20, 1000, 20)))
    local_buckets = defaultdict(list)

    with open(file_path, "r", encoding='UTF-8') as f:
        sqls = f.readlines()
    print(f"load {file_path} completed...", flush=True)

    start = time.time()
    ast_parser = ASTParser(type_vocab=type_vocab, token_vocab=token_vocab)
    for i, sql in enumerate(sqls):
        if i % 100 == 1:
            print(f"In {file_path}, process {i} items successfully, use time = {time.time() - start}...", flush=True)
        pos, negs = augment_data(sql)
        augmented_sqls = [sql, pos] + negs
        augmented_tree_indexes = []
        original_size = -1

        skip_flag = False
        for index, code_snippet in enumerate(augmented_sqls):
            ast = ast_parser.parse(code_snippet)
            if str(ast.root_node.sexp()).find("ERROR") != -1:
                print(f"Tree-sitter parse error in SQL = {code_snippet}")
                skip_flag = True
                break
            try:
                tree_representation, tree_size = ast_parser.simplify_ast(ast, code_snippet)
            except Exception as exc:
                print(f"Error in SQL = {code_snippet}")
                traceback.print_exception(type(exc), exc, exc.__traceback__)
                skip_flag = True
                break
            tree_indexes = transform_tree_to_index(tree_representation, tree_size)
            if tree_indexes['children_index'].shape[1] > 16:
                print(f"too many children = {code_snippet}, index = {tree_representation}")
            if index > 0 and tree_size != original_size:
                print(f"inconsistent nodes = {code_snippet}, \n original = {augmented_sqls[0]}")
            original_size = tree_size if index == 0 else original_size

            subtree_map = extract_subtrees(ast, code_snippet)
            subtree_id_map = {}
            for subtree, count in subtree_map.items():
                subtree_id = subtree_vocab.encode(subtree).ids[0]
                if subtree_id != 3:
                    subtree_id_map[subtree_id] = count
            tree_indexes["subtree_id"] = array.array('h', subtree_id_map.keys())
            augmented_tree_indexes.append(tree_indexes)

        if skip_flag:
            continue
        chosen_bucket_idx = np.argmax(bucket_sizes > original_size)
        local_buckets[chosen_bucket_idx].append(augmented_tree_indexes)

    file_id = int(file_path.split("/")[-1][:-4].split("_")[-1])
    for idx, trees in local_buckets.items():
        bucket_path = f"{local_buckets_path}/{file_id}_{idx}"
        joblib.dump(trees, bucket_path)


class DatasetProcessor:
    def __init__(self, input_data_path, local_buckets_path, output_tensors_path, type_vocab, token_vocab, subtree_vocab):
        self.input_data_path = input_data_path
        self.local_buckets_path = local_buckets_path
        self.output_tensors_path = output_tensors_path

        self.type_vocab = type_vocab
        self.token_vocab = token_vocab
        self.subtree_vocab = subtree_vocab

        init_tokens(list(self.token_vocab.get_vocab().keys()))

    def put_trees_into_buckets(self):
        file_paths = []
        index = 0

        for subdir, dirs, files in os.walk(self.input_data_path):
            for file in files:
                file_path = os.path.join(subdir, file)
                file_paths.append(file_path)

        with concurrent.futures.ProcessPoolExecutor(max_workers=int(4 * os.cpu_count() / 5)) as executor:
            futures = {executor.submit(process_file, file_path, self.local_buckets_path, self.type_vocab, self.token_vocab, self.subtree_vocab): file_path
                       for file_path in file_paths}

            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f'File {futures[future]} generated an exception:')
                    traceback.print_exception(type(exc), exc, exc.__traceback__)
                    continue

                index += 1
                print(f"Processed {futures[future]}, {index} / {len(file_paths)}", flush=True)


def reduce_to_buckets(local_buckets_path, final_buckets_path):
    bucket_files = defaultdict(list)
    for subdir, dirs, files in os.walk(local_buckets_path):
        for file in files:
            file_index, bucket_index = file.split("_")[0], file.split("_")[1]
            bucket_files[bucket_index].append(file_index)

        for bucket_index, file_indexes in bucket_files.items():
            all_trees = []
            for file_index in file_indexes:
                all_trees.extend(joblib.load(os.path.join(subdir, f"{file_index}_{bucket_index}")))

            joblib.dump(all_trees, f"{final_buckets_path}/{bucket_index}")


def load_buckets(final_buckets_path):
    start = time.time()
    buckets = {}
    for subdir, dirs, files in os.walk(final_buckets_path):
        for file in files:
            start_1 = time.time()
            buckets[int(file)] = joblib.load(os.path.join(subdir, file))
            print(f"Load bucket {file}, pickle load time = {time.time() - start_1} s")
    print(f"Load all buckets time = {time.time() - start} s")


def process_dataset(data_path):
    type_vocab = Tokenizer.from_file(data_path['node_type_vocab_model_path'])
    token_vocab = Tokenizer.from_file(data_path['node_token_vocab_model_path'])
    subtree_vocab = Tokenizer.from_file(data_path['subtree_vocab_model_path'])

    print("Training data process starts...", flush=True)
    if os.path.exists(data_path['training_data_path']):
        os.remove(data_path['training_data_path'])
    training_data_processor = DatasetProcessor(input_data_path=data_path['training_queries_path'],
                                               local_buckets_path='./numpy/training_buckets',
                                               output_tensors_path=data_path['training_data_path'],
                                               type_vocab=type_vocab, token_vocab=token_vocab, subtree_vocab=subtree_vocab)
    training_data_processor.put_trees_into_buckets()
    reduce_to_buckets('./numpy/training_buckets', './numpy/final_training_buckets')
    print("Training data process finishes...", flush=True)
    print("=======================================================", flush=True)

    print("Evaluation data process starts...", flush=True)
    if os.path.exists(data_path['evaluation_data_path']):
        os.remove(data_path['evaluation_data_path'])
    evaluation_data_processor = DatasetProcessor(input_data_path=data_path['evaluation_queries_path'],
                                                 local_buckets_path='./numpy/evaluation_buckets',
                                                 output_tensors_path=data_path['evaluation_data_path'],
                                                 type_vocab=type_vocab, token_vocab=token_vocab, subtree_vocab=subtree_vocab)
    evaluation_data_processor.put_trees_into_buckets()
    reduce_to_buckets('./numpy/evaluation_buckets', './numpy/final_evaluation_buckets')
    print("Evaluation data process finishes...", flush=True)
    print("=======================================================", flush=True)

