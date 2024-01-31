import os

from config.config import *

cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sql_base_path = f"{cwd}/data/test-sqls"
config_path = f"{os.getenv('SPARK_HOME')}/benchmarks/conf"

data_path = f'{cwd}/data/{workload}_{data_size}G_{mode}'
raw_history_data_file_path = f'{data_path}/raw_history_data.csv'
all_history_data_file_path = f'{data_path}/bert_history_data.csv' if encoding_model == 'bert' else f'{data_path}/all_history_data.csv'
tuneful_mapping_file_path = f'{data_path}/tuneful_mapping.json'
rover_mapping_file_path = f'{data_path}/rover_mapping.json'
new_task_file_path = f'{data_path}/new_tasks{task_suffix}'
history_task_file_path = f'{data_path}/history_tasks{task_suffix}'

db_name = f'{workload}_{data_size}G_{mode}_loftune'
loftune_db_url = f'mysql+mysqldb://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?charset=utf8'

big_gap = '=' * 130
small_gap = '*' * 20

log_base_path = f"log-base-path"
