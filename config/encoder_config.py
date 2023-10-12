import torch
from config.common import cwd

base_dir = f'{cwd}/sql_encoder'
tree_sitter_sql_lib_path = f"{cwd}/sql_encoder/tree-sitter/sql.so"

dataset = 'text2sql_400k'
device = torch.device('cpu')
sql_embedding_dim = 768

data_path = {
    'node_type_vocab_model_path': f'{base_dir}/data/{dataset}/{dataset}_type.json',
    'node_token_vocab_model_path': f'{base_dir}/data/{dataset}/{dataset}_token.json',
    'subtree_vocab_model_path': f'{base_dir}/data/{dataset}/{dataset}_subtree.json',
    'training_data_path': f'{base_dir}/data/{dataset}/{dataset}_processed_training_data.pkl',
    'evaluation_data_path': f'{base_dir}/data/{dataset}/{dataset}_processed_evaluation_data.pkl',
}

train_config = {
    'num_epoch': 2,
    'batch_size': 8,
    'lr': 0.001,
    'weight_decay': 1e-8,
    'log_step_interval': 1000,
    'checkpoint_file_path': f'{base_dir}/checkpoints/checkpoint',
    'encoder_checkpoint_file_path': f'{base_dir}/checkpoints/encoder_checkpoint',
}

encoder_config = {
    'type_embedding_dim': sql_embedding_dim,
    'token_embedding_dim': sql_embedding_dim,
    'node_embedding_dim': sql_embedding_dim,
    'embedding_dim': sql_embedding_dim,
    'num_conv_layers': 3,
    'conv_output': sql_embedding_dim,
    'dropout_rate': 0.3
}

subtree_prediction_config = {
    'subtree_dim': sql_embedding_dim
}
