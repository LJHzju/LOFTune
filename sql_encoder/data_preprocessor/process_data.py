import os
from gen_vocabulary import gen_all_vocab
from dataset_processor import process_dataset

dataset_path = '../data/text2sql_300k'


def process_data(dataset):
    data_path = {
        'node_type_vocab_model_path': f'{dataset_path}/{dataset}_type.json',
        'node_token_vocab_model_path': f'{dataset_path}/{dataset}_token.json',
        'subtree_vocab_model_path': f'{dataset_path}/{dataset}_subtree.json',
        'training_queries_path': f'{dataset_path}/raw_queries/',
        'evaluation_queries_path': f'{dataset_path}/evaluation_queries/'
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


if __name__ == "__main__":
    process_data("text2sql_300k")
