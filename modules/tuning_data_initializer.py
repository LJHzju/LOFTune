# -*- coding: utf-8 -*-
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import pandas.io.sql
from sqlalchemy import text
from config.knobs_list import KNOBS
from config.common import history_task_file_path, all_history_data_file_path
from util import create_session, embedding_columns


def load_history_tasks(sqls_in_all_history_data):
    with open(history_task_file_path, "r") as history_task_file:
        history_tasks = history_task_file.read().splitlines()
        history_tasks = set(history_tasks) & set(sqls_in_all_history_data)  # 取交集，万一历史数据里面漏了几条SQL，不会报错

    return list(history_tasks)


def gen_embedding_data(history_data, connection):
    embeddings = history_data[['task_id'] + embedding_columns]
    embeddings = embeddings.drop_duplicates(subset=['task_id'], keep='first')

    pd.io.sql.to_sql(embeddings, 'task_embeddings', con=connection.bind, if_exists='append', index=False)


def gen_history_data(history_data, connection):
    history_data = history_data[['app_id', 'task_id'] + KNOBS + embedding_columns + ['status', 'duration']]
    history_data['duration'] = history_data['duration'].astype('int64')

    pd.io.sql.to_sql(history_data, 'task_history', con=connection.bind, if_exists='append', index=False)
    return history_data


def gen_best_config_data(history_data, connection):
    best_config_data = history_data.groupby('task_id', as_index=False).apply(lambda t: t[(t.duration == t.duration.min()) & (t.duration != 3600000)].sample(1))
    best_config_data = best_config_data[['task_id'] + KNOBS + ['duration']]

    pd.io.sql.to_sql(best_config_data, 'task_best_config', con=connection.bind, if_exists='append', index=False)


def gen_matched_history_tasks(connection):
    matched_history_tasks_columns = ['history_task_id', 'new_task_id']
    matched_history_tasks = pd.DataFrame(columns=matched_history_tasks_columns)

    pd.io.sql.to_sql(matched_history_tasks, 'matched_history_tasks', con=connection.bind, if_exists='append', index=False)


def init_tuning_data(file_path=all_history_data_file_path, clear_content=False):
    with create_session() as db_session:  # with会回收资源
        db_session.execute(text("DROP TABLE IF EXISTS task_embeddings"))
        db_session.execute(text("DROP TABLE IF EXISTS task_history"))
        db_session.execute(text("DROP TABLE IF EXISTS task_best_config"))
        db_session.execute(text("DROP TABLE IF EXISTS matched_history_tasks"))

        all_history_data = pd.read_csv(file_path, index_col='task_id')

        history_tasks = load_history_tasks(all_history_data.index.unique().tolist())
        all_history_data = all_history_data.loc[history_tasks]

        all_history_data.reset_index(inplace=True)
        history_data = all_history_data

        gen_embedding_data(history_data, db_session)
        history_data = gen_history_data(history_data, db_session)
        gen_best_config_data(history_data, db_session)
        gen_matched_history_tasks(db_session)

        if clear_content:
            db_session.execute(text("DELETE FROM task_embeddings"))
            db_session.execute(text("DELETE FROM task_history"))
            db_session.execute(text("DELETE FROM task_best_config"))
            db_session.execute(text("DELETE FROM matched_history_tasks"))
            db_session.commit()
            print("Finish creating tables.")
        else:
            print("Finish initializing tuning data.")

