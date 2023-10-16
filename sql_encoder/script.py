import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sql_encoder.training.multi_task_trainer import MultiTaskTrainer
from sql_encoder.encoder import SQLEncoder
from sql_encoder.data_preprocessor.gen_vocabulary import gen_all_vocab
from sql_encoder.data_preprocessor.dataset_processor import process_dataset
from config.encoder_config import base_dir
import os
import numpy as np
import argparse


def process_data(dataset):
    data_path = {
        'node_type_vocab_model_path': f'{base_dir}/data/{dataset}/{dataset}_type.json',
        'node_token_vocab_model_path': f'{base_dir}/data/{dataset}/{dataset}_token.json',
        'subtree_vocab_model_path': f'{base_dir}/data/{dataset}/{dataset}_subtree.json',
        'training_queries_path': f'{base_dir}/data/{dataset}/raw_queries/',
        'evaluation_queries_path': f'{base_dir}/data/{dataset}/evaluation_queries/',
        'training_data_path': f'{base_dir}/data/{dataset}/{dataset}_processed_training_data.pkl',
        'evaluation_data_path': f'{base_dir}/data/{dataset}/{dataset}_processed_evaluation_data.pkl',
    }

    if not os.path.exists(data_path['training_queries_path']):
        print(f"The training queries of dataset {dataset} not exists, "
              f"desired queries path: {data_path['training_queries_path']}")
        return
    if not os.path.exists(data_path['evaluation_queries_path']):
        print(f"The evaluation queries of dataset {dataset} not exists, "
              f"desired queries path: {data_path['evaluation_queries_path']}")
        return

    gen_all_vocab(data_path)
    process_dataset(data_path)


def train():
    trainer = MultiTaskTrainer()
    trainer.load_checkpoint()
    trainer.train()


def encode(sql):
    sql_encoder = SQLEncoder()
    if not sql_encoder.load_encoder():
        print("Encoder checkpoint file not found, can't encode the sql...")
        return
    sql_embedding = sql_encoder.encode([sql])[0].tolist()
    norm = np.linalg.norm(sql_embedding)
    sql_embedding = list(map(lambda x: x / norm, sql_embedding))
    return sql_embedding


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='encode', choices=['encode', 'train', 'process-data'], help="Decide what to do.")
    parser.add_argument('--sql', type=str, default='', help="The SQL statement to encode.")
    parser.add_argument('--dataset_name', type=str, default='', help="The name of dataset.")
    args = parser.parse_args()
    mode = args.mode
    if mode == 'train':
        train()
    elif mode == 'encode':
        embedding = encode(args.sql)
        print(f"The embedding of SQL is {embedding}")
    elif mode == 'process-data':
        if args.dataset_name == '':
            print("Please specify the name of dataset to be processed...")
            sys.exit()
        process_data(args.dataset_name)
