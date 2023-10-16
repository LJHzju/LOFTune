import optuna
import pandas as pd
import numpy as np
import copy
import warnings
from sqlalchemy import text

from config.common import big_gap
from config.config import *
from config.knobs_list import *
from util import check_sample, add_scale_dict, create_session, embedding_columns, get_resource_usage_of_config
from modules.regression_model import PerformanceModel

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


def update_knob_detail(task_best_config):
    core_thresholds = (CORE_MIN, CORE_MAX)
    memory_thresholds = (MEMORY_MIN, MEMORY_MAX)

    updated_knob_details = copy.deepcopy(KNOB_DETAILS)
    for knob, best_value in task_best_config.items():
        knob_details = KNOB_DETAILS[knob]
        knob_type = knob_details['type']
        if knob_details['range_adjustable']:
            min_value, max_value, step_length = knob_details['range'][0: 3]

            new_min_value = best_value * 0.8
            if not knob_details['limit_exceed'][0]:
                new_min_value = max(new_min_value, min_value)
            if knob_type == KnobType.INTEGER:
                new_min_value = int(max(1, new_min_value))
            updated_knob_details[knob]['range'][0] = new_min_value

            expand_ratio = 1.4 if best_value >= min_value else 2
            new_max_value = best_value * expand_ratio + step_length
            if not knob_details['limit_exceed'][1]:
                new_max_value = min(new_max_value, max_value)
            if knob_type == KnobType.INTEGER:
                new_max_value = int(new_max_value)
            updated_knob_details[knob]['range'][1] = new_max_value

    return updated_knob_details, core_thresholds, memory_thresholds


def get_task_embedding(db_session, task_id):
    embeddings = pd.read_sql(text(f"SELECT * FROM task_embeddings WHERE task_id = '{task_id}'"), db_session.connection())
    task_embedding = embeddings[embedding_columns].values.tolist()[0] if not embeddings.empty else None
    return task_embedding


def get_task_best_performance(task_id):
    with create_session() as db_session:
        best_performance = pd.read_sql(text(f"SELECT task_id, duration FROM task_best_config WHERE task_id = '{task_id}'"),
                                       db_session.connection(), index_col='task_id')
    return best_performance.loc[task_id, 'duration']


def update_history(update_task_id, epoch_id, logger, weights=None):
    logger.info(f"Config {epoch_id} for history task {update_task_id} generation starts.")
    with create_session() as db_session:
        task_embedding = get_task_embedding(db_session, update_task_id)
        if task_embedding is None:
            print(f"No history data is found for task {update_task_id}...")
            return None

        best_config = pd.read_sql(text(f"SELECT * FROM task_best_config WHERE task_id = '{update_task_id}'"),
                                  db_session.connection())
        if best_config.empty:
            print(f"No best configuration is found for task {update_task_id}...")
            return None
        task_best_config = {knob: best_config[knob][0] for knob in KNOB_DETAILS.keys()}
        updated_knob_details, core_thresholds, memory_thresholds = update_knob_detail(task_best_config)
        logger.info(f"Updated resource thresholds for task {update_task_id}: "
                    f"cores [{core_thresholds[0]}, {core_thresholds[1]}], "
                    f"memory [{memory_thresholds[0]}m, {memory_thresholds[1]}m].")

        history_data = pd.read_sql(text("SELECT * FROM task_history"), db_session.connection())
        regression_model = PerformanceModel(logger=logger,
                                            core_thresholds=core_thresholds,
                                            memory_thresholds=memory_thresholds,
                                            weights=weights)
        regression_model.train(history_data)

        estimated_running_time = -1
        while True:
            params = regression_model.search_new_config(task_embedding, updated_knob_details)
            data = [params[knob] for knob in KNOBS]
            data.extend(task_embedding)
            predict_data = np.array(data).reshape(1, -1)
            if check_sample(params, core_thresholds, memory_thresholds):
                estimated_running_time = regression_model.predict(predict_data)[0]
                break
            else:
                logger.info(f"Config {epoch_id} for history task {update_task_id} generated failed.")
        config = add_scale_dict(params)
        logger.info(f"Config {epoch_id} for history task {update_task_id} generated successfully: {config}, "
                    f"estimated running time = {estimated_running_time} ms.")

    logger.info(f"Update history of task {update_task_id} finished.")

    if weights is None:
        return config
    else:
        return config, regression_model.probabilities, regression_model.selected_index
