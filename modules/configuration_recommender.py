import json

import pandas as pd
import pandas.io.sql
from sqlalchemy import text
from config.encoder_config import sql_embedding_dim
from config.common import *
from config.config import encoding_model
from util import create_session
import random


def recommend_config(data, logger):
    sim_str = ""
    for i in range(0, sql_embedding_dim):
        sim_str += f"{data[f'task_embedding_{i}']} * task_embedding_{i} + "
    sim_str = sim_str[0: -3]
    with create_session() as db_session:
        if encoding_model in ['bert', 'tbcnn']:
            results = db_session.execute(text(f"SELECT task_id, {sim_str} AS sim FROM task_embeddings ORDER BY sim DESC LIMIT 1"))
            task_sim = results.mappings().first()
            if task_sim is None:
                print("No similar task is found...")
                return None
            similar_task_id, similarity = task_sim['task_id'], task_sim['sim']
        elif encoding_model == 'tuneful':
            similar_task_id = json.load(open(tuneful_mapping_file_path, "r"))[data['task_id']]
            similarity = 1.0
        elif encoding_model == 'rover':
            similar_task_id = json.load(open(rover_mapping_file_path, "r"))[data['task_id']]
            similarity = 1.0
        elif encoding_model == 'random':
            with open(history_task_file_path, "r") as history_task_file:
                history_tasks = history_task_file.read().splitlines()
            similar_task_id = random.choice(history_tasks)
            similarity = 1.0
        logger.info(f"new task = {data['task_id']}, similar task = {similar_task_id}, similarity = {similarity}")

        results = db_session.execute(text(f"SELECT * FROM task_best_config WHERE task_id = '{similar_task_id}'"))
        config = results.mappings().first()
        if config is None:
            print(f"No configuration for similar task {similar_task_id} is found...")
            return None
        config = dict(config)
        config.pop("duration")

        row = {"history_task_id": config.pop("task_id"), "new_task_id": data['task_id']}
        pd.io.sql.to_sql(pd.DataFrame([row]), 'matched_history_tasks',
                         con=db_session.bind, if_exists='append', index=False)

    return config, similar_task_id

