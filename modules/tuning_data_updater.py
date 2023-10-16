import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pandas.io.sql
from util import *
from sqlalchemy import text


def add_new_embeddings_to_database(db_session, new_embeddings):
    new_embeddings_data = pd.DataFrame(new_embeddings)
    new_embeddings_data.drop_duplicates(subset=['task_id'], inplace=True)
    pd.io.sql.to_sql(new_embeddings_data, 'task_embeddings', con=db_session.bind, if_exists='append', index=False)


def gen_best_config(db_session, apps):
    history_data = pd.DataFrame(apps)
    pd.io.sql.to_sql(history_data, 'task_history', con=db_session.bind, if_exists='append', index=False)

    new_best_config = history_data.groupby('task_id', group_keys=False).apply(lambda t: t[t.duration == t.duration.min()].sample(1))
    new_best_config = new_best_config[['task_id'] + KNOBS + ['duration']]
    new_best_config.set_index('task_id', inplace=True)
    return new_best_config


def update_data(app_id, task_id, config, duration, logger):
    with create_session() as db_session:
        app = {"app_id": app_id, "task_id": task_id, "duration": duration, "status": 1}
        app.update(clear_scale_dict(config))
        apps = [app]

        old_embeddings = pd.read_sql(text(f"SELECT * FROM task_embeddings WHERE task_id = '{task_id}'"),
                                     db_session.connection(), index_col='task_id')
        apps, new_embeddings = add_embedding_to_apps(apps, old_embeddings, logger)
        add_new_embeddings_to_database(db_session, new_embeddings)

        new_best_config = gen_best_config(db_session, apps)

        old_best_config = pd.read_sql(text(f"SELECT * FROM task_best_config WHERE task_id = '{task_id}'"),
                                      db_session.connection(), index_col='task_id')
        del_task_list = []
        old_task_list = old_best_config.index.unique().tolist()
        for task in old_task_list:
            new_time = int(new_best_config.loc[task, 'duration'])
            old_time = int(old_best_config.loc[task, 'duration'])
            if new_time < old_time:
                del_task_list.append(task)
            else:
                new_best_config = new_best_config.drop(task)
        new_best_config = new_best_config[new_best_config['duration'] != 3600000]
        new_best_config.reset_index(inplace=True)

        if len(del_task_list) > 0:
            db_session.execute(text(f"DELETE FROM task_best_config WHERE task_id = '{del_task_list[0]}'"))
            db_session.commit()
        pd.io.sql.to_sql(new_best_config, 'task_best_config', con=db_session.bind, if_exists='append', index=False)

        if task_id not in old_task_list:
            logger.info(
                f"Update data for task {task_id} finishes. "
                f"No previous best time found, set best time = {int(app['duration'])} ms.")
        else:
            best_time = int(old_best_config.loc[task_id, 'duration'])
            if int(app['duration']) < best_time:
                logger.info(f"Update data for task {task_id} finishes. Previous best time = {best_time} ms, "
                            f"update to {int(app['duration'])} ms.")
            else:
                logger.info(f"Update data for task {task_id} finishes. Previous best time = {best_time} ms, keep unchanged.")
