import logging.config
import os
import joblib

from sql_encoder.data_utils.data_loader import DataLoader
from config.encoder_config import *
from config.logging_config import *
from tokenizers import Tokenizer

logging.config.dictConfig(LOGGING_CONFIG)


def load_buckets(final_buckets_path):
    buckets = {}
    for subdir, dirs, files in os.walk(final_buckets_path):
        for file in files:
            buckets[int(file)] = joblib.load(os.path.join(subdir, file))
    return buckets


class BaseTrainer:
    def __init__(self):
        self.logger = logging.getLogger('trainer')

        self.type_vocab = Tokenizer.from_file(data_path['node_type_vocab_model_path'])
        self.token_vocab = Tokenizer.from_file(data_path['node_token_vocab_model_path'])

        self.training_buckets = load_buckets(data_path['training_data_path'])
        self.evaluation_buckets = load_buckets(data_path['evaluation_data_path'])

        self.batch_size = train_config['batch_size']
        self.train_data_loader = DataLoader(self.batch_size)
        self.eval_data_loader = DataLoader(self.batch_size)

        self.num_epoch = train_config['num_epoch']
        self.start_epoch = 1

        self.checkpoint_file_path = train_config['checkpoint_file_path']
        self.encoder_checkpoint_file_path = train_config['encoder_checkpoint_file_path']

        encoder_config['num_types'] = self.type_vocab.get_vocab_size()
        encoder_config['num_tokens'] = self.token_vocab.get_vocab_size()

        self.device = device
        self.model = None
        self.optimizer = None

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_file_path):
            return
        checkpoint = torch.load(self.checkpoint_file_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1

    def load_batch_tree_data(self, batch_data):
        batch_node_type = torch.from_numpy(batch_data["batch_node_type_id"]).to(self.device, non_blocking=True)
        batch_node_tokens = torch.from_numpy(batch_data["batch_node_tokens_id"]).to(self.device)
        batch_children_index = torch.from_numpy(batch_data["batch_children_index"]).to(self.device)

        return batch_node_type, batch_node_tokens, batch_children_index

    def eval(self):
        pass

    def save_model(self, epoch):
        torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, self.checkpoint_file_path)
        torch.save({'model_state_dict': self.model.encoder.state_dict()}, f"{self.encoder_checkpoint_file_path}_{epoch}")
