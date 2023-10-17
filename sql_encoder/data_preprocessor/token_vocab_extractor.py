import os
import concurrent.futures
from tokenizers import Tokenizer, trainers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from sqlglot.tokens import Tokenizer as SQLTokenizer


def process_file(file_path):
    with open(file_path, "r", errors="ignore", encoding='UTF-8') as f:
        sqls = f.readlines()
        print(f"load {file_path} completed...", flush=True)
        tokenizer = SQLTokenizer()
        result = []
        for index, sql in enumerate(sqls):
            tokens = tokenizer.tokenize(sql)
            split_tokens = []
            for token in tokens:
                split_tokens.append(token.text)
            result.append(" ".join(split_tokens))
        return file_path, result


class TokenVocabExtractor:
    def __init__(self, token_vocab_model_path):
        self.token_vocab_model_path = token_vocab_model_path
        self.tokenizer = Tokenizer(BPE(vocab=None, unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.trainer = trainers.BpeTrainer(
            special_tokens=["<pad>", "<sos>", "<eos>", "<unk>", "<tb>", "</tb>", "<col>"],
            min_frequency=1, show_progress=True, vocab_size=10000
        )

    def create_vocab_from_dir(self, input_data_path: str):
        data = []
        index = 0
        file_paths = []

        for subdir, dirs, files in os.walk(input_data_path):
            for file in files:
                file_path = os.path.join(subdir, file)
                file_paths.append(file_path)

        with concurrent.futures.ProcessPoolExecutor(max_workers=int(os.cpu_count() / 2)) as executor:
            results = executor.map(process_file, file_paths)

            for file_path, result in results:
                data.extend(result)
                index += 1
                print(f"Processed {file_path}, {index} / {len(file_paths)}", flush=True)

        print("extend data completed...", flush=True)
        self.tokenizer.train_from_iterator(data, self.trainer)
        self.tokenizer.save(self.token_vocab_model_path)
