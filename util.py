import copy
import re
import json
import numpy as np
import pandas as pd
import torch

from hdfs import Client

from config.common import *
from config.knobs_list import *
from sql_encoder.script import encode
from config.encoder_config import sql_embedding_dim
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import create_engine

find_number = lambda x: re.search("\d+(\.\d+)?", x).group()
embedding_columns = [f'task_embedding_{_}' for _ in range(0, sql_embedding_dim)]


def create_session():
    engine = create_engine(loftune_db_url)
    session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
    return session()


def load_event_log_content(app_idx):
    app_id_list = app_idx.split("/")  # app_id_1/app_id_2/app_id_3
    client = Client(hdfs_path)
    lines_of_all_apps = []
    for app_id in app_id_list:
        lines = []
        if EXTRA_KNOBS['spark.submit.deployMode'] == 'client':
            file_name = f"{event_log_hdfs_path}/{app_id}"
        else:
            file_name = f"{event_log_hdfs_path}/{app_id}_1"
        with client.read(file_name, encoding='utf-8', delimiter='\n') as event_log_file:
            for line in event_log_file:
                lines.append(line)
        lines_of_all_apps.append(lines)
    return lines_of_all_apps


def load_info_from_lines(lines_of_apps):
    run_time_of_apps = []
    status_of_apps = []
    for lines in lines_of_apps:
        start_time = np.inf
        finish_time = -1

        succeed_job_count = 0
        failed_job_count = 0

        for line in lines:
            if line == "":
                continue
            event = json.loads(line)
            event_type = event['Event']
            if event_type == 'SparkListenerJobStart':
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
        run_time_of_apps.append(finish_time - start_time)
        status_of_apps.append(app_succeeded)

    return sum(run_time_of_apps), all(status_of_apps)


def gen_task_embedding(task_id):
    if encoding_model == 'bert':
        df = pd.read_csv(f"{cwd}/data/bert-embedding/{workload}_{mode}.csv", index_col='name')
        sql_embedding = df.loc[task_id].values.tolist()
    else:
        sqls = task_id.split('_')
        sql_embeddings = []
        for sql_id in sqls:
            with open(f"{sql_base_path}/{str.lower(workload)}/{sql_id}.sql") as sql_file:
                sql = sql_file.read()
                sql_embeddings.append(encode(sql))
        sql_embeddings = torch.tensor(sql_embeddings)
        sql_embedding = torch.max(sql_embeddings, dim=0)[0].tolist()
        norm = np.linalg.norm(sql_embedding)
        sql_embedding = list(map(lambda x: x / norm, sql_embedding))
    return {f'task_embedding_{i}': sql_embedding[i] for i in range(0, sql_embedding_dim)}


def gen_sql_list():
    all_sql_list = []
    for root, dirs, files in os.walk(f"{sql_base_path}/{str.lower(workload)}"):
        for file in files:
            all_sql_list.append(file[:-4])
    return all_sql_list


def add_embedding_to_apps(apps, old_embeddings, logger=None):
    new_embeddings = []
    new_apps = []
    for app in apps:
        task_id = app['task_id']
        if task_id in old_embeddings.index.values:
            if logger is not None:
                logger.info(f"Fetch old task {task_id} embedding from database.")
            for i in range(0, sql_embedding_dim):
                app[f'task_embedding_{i}'] = old_embeddings.loc[task_id, f'task_embedding_{i}']
        else:
            if logger is not None:
                logger.info(f"New task {task_id} embedding generated.")
            new_embedding = gen_task_embedding(task_id)
            if new_embedding is None:
                continue
            app.update(new_embedding)
            old_embeddings.loc[task_id] = copy.deepcopy(new_embedding)
            new_embedding['task_id'] = task_id
            new_embeddings.append(new_embedding)
        new_apps.append(app)

    return new_apps, new_embeddings


def clear_scale_dict(origin_dict):
    """
        remove scale('k','m','g',...) from dict
    """
    for key, value in origin_dict.items():
        val_type = KNOB_DETAILS[key]['type']
        if val_type == KnobType.INTEGER and isinstance(value, str):
            origin_dict[key] = int(find_number(value))
        elif val_type == KnobType.NUMERIC and isinstance(value, str):
            origin_dict[key] = float(find_number(value))
        elif val_type == KnobType.CATEGORICAL:
            candidates = KNOB_DETAILS[key]['candidates']
            index = candidates.index(value)
            origin_dict[key] = index
    return origin_dict


def add_scale_dict(origin_dict):
    """
        add scale('k','m','g',...) to dict
    """
    new_dict = copy.deepcopy(origin_dict)
    for knob, details in KNOB_DETAILS.items():
        if knob not in new_dict.keys():
            continue
        if details['type'] == KnobType.CATEGORICAL:
            new_dict[knob] = details['candidates'][origin_dict[knob]]
        elif 'unit' in details.keys():
            new_dict[knob] = str(origin_dict[knob]) + details['unit']
    return new_dict


def get_resource_usage_of_config(sample):
    num_executors = sample['spark.executor.instances']
    executor_cores = sample['spark.executor.cores']
    driver_cores = sample['spark.driver.cores']
    executor_memory = sample['spark.executor.memory']
    driver_memory = sample['spark.driver.memory']
    off_heap_size = sample['spark.memory.offHeap.size']

    total_memory = num_executors * (1.1 * executor_memory + off_heap_size) + 1.1 * driver_memory
    total_cores = num_executors * executor_cores + driver_cores
    return total_cores, total_memory


def check_sample(sample, core_thresholds=(CORE_MIN, CORE_MAX), memory_thresholds=(MEMORY_MIN, MEMORY_MAX)):
    total_cores, total_memory = get_resource_usage_of_config(sample)
    if total_cores < core_thresholds[0] or total_cores > core_thresholds[1]:
        return False
    if total_memory < memory_thresholds[0] or total_memory > memory_thresholds[1]:
        return False
    return True
