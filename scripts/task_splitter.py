import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import argparse
import pandas as pd
from util import gen_sql_list
from config.config import mode
from config.common import new_task_file_path, history_task_file_path, all_history_data_file_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--history_task_num', type=int, default=93, help='The number of sampled history tasks')
    opt = parser.parse_args()

    if mode == 'single':
        all_task_list = gen_sql_list()
    else:
        history_data = pd.read_csv(all_history_data_file_path)
        all_task_list = history_data['task_id'].unique().tolist()

    random.seed(opt.seed)
    history_tasks = random.sample(all_task_list, opt.history_task_num)
    new_tasks = list(set(all_task_list) - set(history_tasks))

    with open(new_task_file_path, "w") as file:
        new_task_str = ""
        for task in new_tasks:
            new_task_str += (task + '\n')
        file.write(new_task_str)

    with open(history_task_file_path, "w") as file:
        history_task_str = ""
        for task in history_tasks:
            history_task_str += (task + '\n')
        file.write(history_task_str)
