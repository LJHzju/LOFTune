import os
import concurrent.futures
import argparse
from sql_encoder.data_utils.ast_parser import ASTParser
from config.encoder_config import base_dir
import random

ast_parser = ASTParser()


def list_split(items, n):
    return [items[i:i + n] for i in range(0, len(items), n)]


def process_file(file_path, node_cnt_threshold):
    filter_sqls = []
    with open(file_path, "r", errors="replace") as sqls_file:
        sqls = sqls_file.readlines()
        for sql in sqls:
            tree = ast_parser.parse(sql)
            root_node = tree.root_node
            if str(root_node.sexp()).find("ERROR") == -1:
                num_nodes = ast_parser.get_node_cnt(tree)
                if num_nodes <= node_cnt_threshold:
                    filter_sqls.append(sql)

    return filter_sqls


def filter_queries(input_data_path, dataset_name, training_size, node_cnt_threshold, split_size):
    training_sqls = []
    evaluation_sqls = []
    index = 0
    file_paths = []

    for subdir, dirs, files in os.walk(input_data_path):
        for file in files:
            file_path = os.path.join(subdir, file)
            file_paths.append(file_path)

    random.shuffle(file_paths)
    training_file_cnt = int(0.8 * len(file_paths))

    with concurrent.futures.ProcessPoolExecutor(max_workers=int(os.cpu_count() / 2)) as executor:
        futures = {executor.submit(process_file, file_path, node_cnt_threshold): file_path for file_path in file_paths}

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if index < training_file_cnt:
                training_sqls.extend(result)
            else:
                evaluation_sqls.extend(result)
            index += 1
            print(f"Processed {futures[future]}, {index} / {len(file_paths)}", flush=True)

    print("extend data completed...", flush=True)

    training_queries_path = f"{base_dir}/data/{dataset_name}/raw_queries/"
    evaluation_queries_path = f"{base_dir}/data/{dataset_name}/evaluation_queries/"
    os.mkdir(f"{base_dir}/data/{dataset_name}")
    os.mkdir(training_queries_path)
    os.mkdir(evaluation_queries_path)

    training_sqls = list(set(training_sqls))
    training_sqls = random.sample(training_sqls, training_size)
    random.shuffle(training_sqls)

    evaluation_sqls = list(set(evaluation_sqls))
    evaluation_sqls = random.sample(evaluation_sqls, int(0.25 * training_size))
    random.shuffle(evaluation_sqls)

    training_queries_split = list_split(training_sqls, split_size)
    evaluation_queries_split = list_split(evaluation_sqls, split_size)
    for index, queries in enumerate(training_queries_split):
        with open(f"{training_queries_path}/queries_split_{index}.txt", "w") as training_queries_file:
            for query in queries:
                training_queries_file.write(query)
    for index, queries in enumerate(evaluation_queries_split):
        with open(f"{evaluation_queries_path}/queries_split_{index}.txt", "w") as evaluation_queries_file:
            for query in queries:
                evaluation_queries_file.write(query)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries_path', type=str, default='', help="The path of raw SQL queries.")
    parser.add_argument('--dataset_name', type=str, default='', help="The name of generated SQL dataset.")
    parser.add_argument('--training_set_size', type=int, default=100000, help="The count of training SQL queries.")
    parser.add_argument('--node_cnt_threshold', type=int, default=600, help="The threshold of AST node count. "
                                                                            "The SQL query with AST node count higher than this value will be removed.")
    parser.add_argument('--sql_cnt_per_file', type=int, default=10000, help="The count of SQL queries in each output file.")
    opt = parser.parse_args()
    path = opt.queries_path
    dataset = opt.dataset_name
    training_size = opt.training_set_size
    node_cnt_threshold = opt.node_cnt_threshold
    split_size = opt.sql_cnt_per_file
    filter_queries(path, dataset, training_size, node_cnt_threshold, split_size)
