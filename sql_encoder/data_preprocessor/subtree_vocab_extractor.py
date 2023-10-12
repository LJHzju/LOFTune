import os
import concurrent.futures
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from sql_encoder.data_utils.ast_parser import ASTParser
from sql_encoder.data_preprocessor.subtree_util import extract_subtrees


# 处理file_path指向的文件，读取该文件内的所有SQL，并将每条SQL拆成若干个子树
def process_file(file_path):
    ast_parser = ASTParser()
    with open(file_path, "r", errors="ignore", encoding='UTF-8') as f:
        sqls = f.readlines()
        print(f"load {file_path} completed...", flush=True)
        result = []
        for sql in sqls:
            tree = ast_parser.parse(sql)
            subtrees = extract_subtrees(tree)
            result.extend(subtrees.keys())
        return file_path, result


class SubtreeVocabExtractor:
    def __init__(self, subtree_vocab_model_path):
        self.subtree_vocab_model_path = subtree_vocab_model_path
        self.tokenizer = Tokenizer(WordLevel(vocab=None, unk_token="<unk>"))
        self.trainer = WordLevelTrainer(
            special_tokens=["<pad>", "<sos>", "<eos>", "<unk>", "<tb>", "</tb>", "<col>"],
            min_frequency=1, show_progress=True, vocab_size=100000
        )

    def create_vocab_from_dir(self, input_data_path: str):
        data = []
        file_paths = []
        index = 0

        for subdir, dirs, files in os.walk(input_data_path):
            for file in files:
                file_path = os.path.join(subdir, file)
                file_paths.append(file_path)

        # 并行
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(os.cpu_count() / 2)) as executor:
            results = executor.map(process_file, file_paths)

            for file_path, result in results:
                data.extend(result)
                index += 1
                print(f"Processed {file_path}, {index} / {len(file_paths)}", flush=True)

        print("extend data completed...", flush=True)
        self.tokenizer.train_from_iterator(data, self.trainer)
        self.tokenizer.save(self.subtree_vocab_model_path)
