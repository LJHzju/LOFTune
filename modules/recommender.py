import pandas as pd
import pandas.io.sql
from sqlalchemy import text
from config.encoder_config import sql_embedding_dim
from util import create_session


def recommend_config(data, logger):
    sim_str = ""
    for i in range(0, sql_embedding_dim):
        sim_str += f"{data[f'task_embedding_{i}']} * task_embedding_{i} + "
    sim_str = sim_str[0: -3]
    with create_session() as db_session:
        results = db_session.execute(text(f"SELECT task_id, {sim_str} AS sim FROM task_embeddings ORDER BY sim DESC LIMIT 1"))
        task_sim = results.mappings().first()
        if task_sim is None:
            print("No similar task is found...")
            return None
        similar_task_id = task_sim['task_id']
        logger.info(f"new task = {data['task_id']}, similar task = {similar_task_id}, similarity = {task_sim['sim']}")

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
