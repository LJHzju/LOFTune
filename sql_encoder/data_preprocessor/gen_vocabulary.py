from sql_encoder.data_preprocessor.token_vocab_extractor import TokenVocabExtractor
from sql_encoder.data_preprocessor.subtree_vocab_extractor import SubtreeVocabExtractor
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from config.encoder_config import base_dir
from config.common import big_gap
import os
import json


def get_node_types():
    node_type_file_path = f'node-type-file-path'
    node_types = json.load(open(node_type_file_path))
    node_type_set = set()
    for item in node_types:
        node_type_set.add(item['type'])
    return list(node_type_set)


def gen_type_vocab(output_path):
    print("Node type vocabulary generation starts...", flush=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    tokenizer = Tokenizer(WordLevel(vocab=None, unk_token="<unk>"))
    trainer = WordLevelTrainer(special_tokens=["<pad>", "<sos>", "<eos>", "<unk>", "<tb>", "</tb>", "<col>"],
                               min_frequency=1, show_progress=True, vocab_size=100000)

    node_types = get_node_types()
    tokenizer.train_from_iterator(node_types, trainer)
    tokenizer.save(output_path)
    print("Node type vocabulary generation finishes...", flush=True)
    print(big_gap, flush=True)


def gen_token_vocab(queries_file, output_path):
    print("Node token vocabulary generation starts...", flush=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    token_vocab_extractor = TokenVocabExtractor(token_vocab_model_path=output_path)
    token_vocab_extractor.create_vocab_from_dir(queries_file)
    print("Node token vocabulary generation finishes...", flush=True)
    print(big_gap, flush=True)


def gen_subtree_vocab(queries_file, output_path):
    print("Subtree vocabulary generation starts...", flush=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    subtree_vocab_extractor = SubtreeVocabExtractor(subtree_vocab_model_path=output_path)
    subtree_vocab_extractor.create_vocab_from_dir(queries_file)
    print("Subtree vocabulary generation finishes...", flush=True)
    print(big_gap, flush=True)


def gen_all_vocab(data_path):
    training_queries_path = data_path['training_queries_path']

    gen_type_vocab(data_path['node_type_vocab_model_path'])
    gen_token_vocab(training_queries_path, data_path['node_token_vocab_model_path'])
    gen_subtree_vocab(training_queries_path, data_path['subtree_vocab_model_path'])
