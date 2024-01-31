import torch
from config.common import cwd
from config.config import encoding_model

base_dir = f'{cwd}/sql_encoder'
tree_sitter_sql_lib_path = f"{cwd}/sql_encoder/tree-sitter/sql.so"

dataset = 'text2sql_300k'
device = torch.device('cpu')
sql_embedding_dim = 768 if encoding_model == 'bert' else 128

data_path = {
    'node_type_vocab_model_path': f'{base_dir}/data/{dataset}/{dataset}_type.json',
    'node_token_vocab_model_path': f'{base_dir}/data/{dataset}/{dataset}_token.json',
    'subtree_vocab_model_path': f'{base_dir}/data/{dataset}/{dataset}_subtree.json',
    'training_data_path': f'{base_dir}/data/{dataset}/final_training_buckets',
    'evaluation_data_path': f'{base_dir}/data/{dataset}/final_evaluation_buckets',
}

train_config = {
    'num_epoch': 20,
    'batch_size': 16,
    'lr': 0.0001,
    'weight_decay': 1e-8,
    'log_step_interval': 1000,
    'checkpoint_file_path': f'{base_dir}/checkpoints/checkpoint',
    'best_eval_path': f'{base_dir}/checkpoints/best_eval',
    'encoder_checkpoint_file_path': f'{base_dir}/checkpoints/encoder_checkpoint',
}


encoder_config = {
    'type_embedding_dim': sql_embedding_dim,
    'token_embedding_dim': sql_embedding_dim,
    'node_embedding_dim': sql_embedding_dim,
    'conv_output_dim': sql_embedding_dim,
    'num_conv_layers': 2,
    'dropout_rate': 0.2
}

subtree_prediction_config = {
    'object_dim': sql_embedding_dim
}

contrastive_config = {
    'num_pos': 1,
    'num_neg_random': 6,
    'num_neg_in_batch': train_config['batch_size'] - 1
}

sequence_tagging_config = {
    'num_labels': 4,  # other/table/column/alias
    'hidden_dim': sql_embedding_dim,
    'embedding_dim': sql_embedding_dim,
    'dropout_rate': 0.2
}