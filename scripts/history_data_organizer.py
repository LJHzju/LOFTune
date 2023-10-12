import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random

from config.common import *
from config.common import log_base_path
from config.config import *
from util import clear_scale_dict, embedding_columns, add_embedding_to_apps
from config.knobs_list import *
from itertools import combinations
from tqdm import *
import json
import numpy as np
import pandas as pd
import os
import argparse


def get_valid_tasks(task_dict):
    # task_dict: {task_id_1: [config_id_1, config_id_2, ...], task_id_2: [], ...}
    lower_bound = 25
    tuples = {}
    valid_task_dict = {task_id: config_ids for task_id, config_ids in task_dict.items() if len(set(config_ids)) >= lower_bound}
    four_tuples = list(combinations(valid_task_dict.keys(), 4))
    valid_tuples = 0
    for four_tuple in four_tuples:
        common_config_ids = set(valid_task_dict[four_tuple[0]])
        for task_id in four_tuple[1:]:
            common_config_ids = common_config_ids.intersection(set(valid_task_dict[task_id]))
        if len(common_config_ids) >= lower_bound:
            valid_tuples += 1
            tuples[four_tuple] = common_config_ids
            # print(f"Four tuple: {four_tuple}, Common Config ID: {common_config_ids}, Count: {len(common_config_ids)}")
    return tuples


def load_app_events_from_lines(lines):
    app_id = ""
    app_name = ""
    start_time = np.inf
    finish_time = -1
    config = {}

    succeed_job_count = 0
    failed_job_count = 0

    for line in lines:
        event = json.loads(line)
        event_type = event['Event']
        if event_type == 'SparkListenerApplicationStart':
            app_id = event['App ID']
            app_name = event['App Name']
        elif event_type == 'SparkListenerEnvironmentUpdate':
            config = {name: event['Spark Properties'][name] for name in KNOBS}
            config = clear_scale_dict(config)
        elif event_type == 'SparkListenerJobStart':
            if event['Submission Time'] < start_time:
                start_time = event['Submission Time']
        elif event_type == 'SparkListenerJobEnd':
            if event['Completion Time'] > finish_time:
                finish_time = event['Completion Time']
            job_status = event['Job Result']['Result']
            if job_status == 'JobSucceeded':
                succeed_job_count += 1
            else:
                failed_job_count += 1

    app_succeeded = succeed_job_count > 0 and failed_job_count == 0

    duration = finish_time - start_time
    basic_info = {"app_id": app_id, "app_name": app_name, "duration": duration, "status": app_succeeded}

    return basic_info, config


def load_app_events(file_path):
    with open(file_path, "r") as event_log_file:
        lines = event_log_file.read().splitlines()
        app_data, config = load_app_events_from_lines(lines)
        app_data.update(config)
        return app_data


def select_single_sql_history():
    raw_history = pd.read_csv(raw_history_data_file_path, index_col=None)
    raw_history.rename(columns={'app_name': 'task_id'}, inplace=True)
    raw_history['task_id'] = raw_history['task_id'].apply(lambda name: name.split("#")[0])
    sampled_history = raw_history.groupby("task_id", as_index=False).apply(lambda group: group.sample(25))
    sampled_history.to_csv(all_history_data_file_path, index=False)


def select_multi_sql_history():
    raw_history = pd.read_csv(raw_history_data_file_path, index_col='app_id')
    raw_history.rename(columns={'app_name': 'task_id'}, inplace=True)

    split_cols = raw_history['task_id'].str.split('#')
    raw_history['task_id'], raw_history['config_id'] = split_cols.str[0], split_cols.str[2]
    raw_history['config_id'] = raw_history['config_id'].apply(lambda name: name.split("_")[0])

    # 生成task_dist
    task_dict = {}
    for index, row in raw_history.iterrows():
        task_id = row['task_id']
        config_id = row['config_id']
        if task_id not in task_dict.keys():
            task_dict[task_id] = []
        task_dict[task_id].append(config_id)

    tasks = get_valid_tasks(task_dict)

    sampled_tasks = random.sample(tasks.keys(), 30)

    all_records = []
    for task in sampled_tasks:
        sampled_config_ids = random.sample(tasks[task], 25)
        sorted_task = sorted(task)
        task_id_str = '_'.join(sorted_task)
        for config_id in sampled_config_ids:
            values = raw_history.query(f"config_id == '{config_id}'").head(1)[KNOBS].values[0]
            app_idx = []
            total_duration = 0
            for sql in task:
                query_result = raw_history.query(f"task_id == '{sql}' and config_id == '{config_id}'")
                app_idx.append(query_result.index.values[0])
                total_duration += query_result['duration'].values[0]
            app_id_str = '/'.join(sorted(app_idx))
            record = {'app_id': app_id_str, 'task_id': task_id_str, 'duration': total_duration, 'status': True}

            type_calibrated_config = {}
            for name, value in zip(KNOBS, values):
                knob_type = KNOB_DETAILS[name]['type']
                if knob_type == KnobType.INTEGER:
                    type_calibrated_config[name] = int(value)
                elif knob_type == KnobType.NUMERIC:
                    type_calibrated_config[name] = float(value)
                else:
                    type_calibrated_config[name] = int(value)  # CATEGORICAL, 是索引
            record.update(type_calibrated_config)
            all_records.append(record)

    sampled_history = pd.DataFrame(all_records)
    sampled_history.to_csv(all_history_data_file_path, index=False)


def to_our_history_data():
    all_history_data = pd.read_csv(all_history_data_file_path, index_col=None)
    all_history_data['status'] = all_history_data['status'].apply(lambda status: 1 if status else 0)
    apps = []
    for index, row in all_history_data.iterrows():
        apps.append({name: row[name] for name in all_history_data.columns})
    old_embeddings = pd.DataFrame(columns=embedding_columns, index=['task_id'])
    apps, _ = add_embedding_to_apps(apps, old_embeddings)
    all_history_data = pd.DataFrame(apps)
    all_history_data.to_csv(all_history_data_file_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default="",
                        choices=['process-logs', 'select-history', 'convert-history'])
    args = parser.parse_args()

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if args.type == 'process-logs':
        all_apps = []
        for subdir, dirs, files in os.walk(log_base_path):
            for file in tqdm(files):
                file_path = os.path.join(subdir, file)
                app_data = load_app_events(file_path)
                all_apps.append(app_data)

        df = pd.DataFrame(all_apps)
        df.to_csv(raw_history_data_file_path, index=False)
    elif args.type == 'select-history':
        if mode == 'single':
            select_single_sql_history()
        elif mode == 'multi':
            select_multi_sql_history()
    elif args.type == 'convert-history':
        to_our_history_data()
