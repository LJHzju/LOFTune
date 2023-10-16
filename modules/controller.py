from modules.tuning_data_initializer import init_tuning_data
from modules.configuration_recommender import recommend_config
from modules.knowledge_base_updater import update_history, get_task_best_performance
from modules.tuning_data_updater import update_data
from modules.config_random_sampler import sample_configs
from util import *
from modules.task_runner import run_task
from config.logging_config import *
import pandas as pd
import math
import logging.config

logging.config.dictConfig(LOGGING_CONFIG)


def gen_result_file_name(task_id, type, num_epochs):
    return f"{cwd}/result/{workload}#{data_size}G#{task_id}#{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}#{type}#{num_epochs}epochs.csv"


def run_task_and_update(task_id, config, logger, update=True):
    app_idx = run_task(task_id, config)
    if app_idx == "":
        logger.error(f"App launch failed. Please refer to the spark output logs.")
        return None
    event_log_content_of_apps = load_event_log_content(app_idx)
    run_time, app_succeeded = load_info_from_lines(event_log_content_of_apps)
    if app_succeeded:
        logger.info(f"Run successfully. Duration = {run_time} ms.")
        if update:
            update_data(app_idx, task_id, config, run_time, logger)
    else:
        logger.error(f"Run failed.")
        run_time = 3600000

    tuning_data = {"app_id": app_idx, "duration": run_time}
    tuning_data.update(config)
    return tuning_data


def recommend_config_alternately(task_id, num_epochs_per_round=1):
    logger = logging.getLogger("recommender")
    logger.info(big_gap)
    logger.info(f"Start tuning for new task {task_id}.")
    data = gen_task_embedding(task_id)
    if data is None:
        return
    data['task_id'] = task_id
    config, similar_task_id = recommend_config(data, logger)
    if config is None:
        return

    config = add_scale_dict(config)
    logger.info(f"First suggested config for {task_id} = {config}.")
    tuning_data = run_task_and_update(task_id, config, logger, update=False)
    all_tuning_data = [tuning_data]

    for epoch in range(0, num_epochs_per_round):
        logger.info(small_gap)
        history_task_config = update_history(similar_task_id, epoch, logger)
        logger.info(small_gap)
        if history_task_config is None:
            return
        logger.info(f"Suggested config for {similar_task_id} in iter {epoch + 1}  = {history_task_config}.")
        run_task_and_update(similar_task_id, history_task_config, logger)

    df = pd.DataFrame(all_tuning_data)
    df.to_csv(gen_result_file_name(task_id, 'new-alternately', num_epochs_per_round), index=False)
    logger.info(f"Finish tuning for new task {task_id}.")
    logger.info(big_gap)


def recommend_config_for_new_task_without_history(task_id, random_sample_epochs=10, model_sample_epochs=30):
    init_tuning_data(all_history_data_file_path, clear_content=True)

    logger = logging.getLogger("recommender")
    logger.info(big_gap)
    logger.info(f"Start tuning for new task {task_id}. Use empty history data, and tune from scratch.")
    data = gen_task_embedding(task_id)
    if data is None:
        return

    all_tuning_data = []

    # randomly sample configurations
    # sampled_configs = sample_configs(random_sample_epochs, tasks=[task_id])
    # logger.info(f"Randomly sample {random_sample_epochs} configurations.")
    # for index, config in enumerate(sampled_configs):
    #     logger.info(f"Running iter {index + 1} in random configurations, config = {config}.")
    #     tuning_data = run_task_and_update(task_id, config, logger)
    #     if tuning_data is not None:
    #         all_tuning_data.append(tuning_data)

    # use existing randomly sampled configurations
    existing_data = pd.read_csv(all_history_data_file_path)
    existing_data = existing_data[existing_data['task_id'] == task_id].head(random_sample_epochs)
    for index, row in existing_data.iterrows():
        config = add_scale_dict({name: row[name] for name in KNOBS})
        update_data(row['app_id'], task_id, config, row['duration'], logger)

        tuning_data = {"app_id": row['app_id'], "duration": row['duration']}
        tuning_data.update(config)
        all_tuning_data.append(tuning_data)

    logger.info(f"Run random configurations finished, start tuning for {model_sample_epochs} iters.")

    epochs = 0
    cur_best_performance = get_task_best_performance(task_id)
    weights = [1.0, 1.0]

    while True:
        logger.info(small_gap)
        config, probabilities, selected_index = update_history(task_id, epochs + 1, logger, weights)
        logger.info(small_gap)
        if config is None:
            return
        logger.info(f"Suggested config for {task_id} in iter {epochs + 1} = {config}.")
        tuning_data = run_task_and_update(task_id, config, logger)
        if tuning_data is None:
            continue

        pi = (cur_best_performance - tuning_data['duration']) / cur_best_performance
        reward = 1 / (1 + math.exp(-5 * pi)) if pi < 0 else 1 / (1 + math.exp(-10 * pi))
        modified_reward = [reward / probabilities[0], 0] if selected_index == 0 else [0, reward / probabilities[1]]
        weights = [weights[0] * math.exp(rate_tradeoff * modified_reward[0] / 2),
                   weights[1] * math.exp(rate_tradeoff * modified_reward[1] / 2)]
        if tuning_data['duration'] < cur_best_performance:
            cur_best_performance = tuning_data['duration']
        logger.info(f"Performance improvement = {pi}, reward = {reward}. Update weights to {weights}.")

        all_tuning_data.append(tuning_data)

        epochs += 1
        if epochs >= model_sample_epochs:
            break

    df = pd.DataFrame(all_tuning_data)
    df.to_csv(gen_result_file_name(task_id, 'new-no-history', random_sample_epochs + model_sample_epochs), index=False)
    logger.info(f"Finish tuning for new task {task_id}.")
    logger.info(big_gap)


def update_history_task(update_task_id, num_epochs):
    weights = [1.0, 1.0]
    logger = logging.getLogger("history_tuner")
    logger.info(big_gap)
    logger.info(f"Start history update for task {update_task_id}.")
    epochs = 0
    all_tuning_data = []
    cur_best_performance = get_task_best_performance(update_task_id)
    while True:
        logger.info(small_gap)
        config, probabilities, selected_index = update_history(update_task_id, epochs + 1, logger, weights)
        logger.info(small_gap)
        if config is None:
            return
        logger.info(f"Suggested config for {update_task_id} in iter {epochs + 1} = {config}.")
        tuning_data = run_task_and_update(update_task_id, config, logger)
        if tuning_data is None:
            continue

        pi = (cur_best_performance - tuning_data['duration']) / cur_best_performance
        reward = 1 / (1 + math.exp(-5 * pi)) if pi < 0 else 1 / (1 + math.exp(-10 * pi))
        modified_reward = [reward / probabilities[0], 0] if selected_index == 0 else [0, reward / probabilities[1]]
        weights = [weights[0] * math.exp(rate_tradeoff * modified_reward[0] / 2),
                   weights[1] * math.exp(rate_tradeoff * modified_reward[1] / 2)]
        if tuning_data['duration'] < cur_best_performance:
            cur_best_performance = tuning_data['duration']
        logger.info(f"Performance improvement = {pi}, reward = {reward}. Update weights to {weights}.")

        all_tuning_data.append(tuning_data)

        epochs += 1
        if epochs >= num_epochs:
            break

    df = pd.DataFrame(all_tuning_data)
    df.to_csv(gen_result_file_name(update_task_id, 'history', num_epochs), index=False)
    logger.info(f"Finish history update for task {update_task_id}.")
    logger.info(big_gap)


def recommend_config_for_new_task(task_id, num_epochs=1):
    init_tuning_data(all_history_data_file_path)

    weights = [1.0, 1.0]

    logger = logging.getLogger("recommender")
    logger.info(big_gap)
    logger.info(f"Start tuning for new task {task_id}.")
    data = gen_task_embedding(task_id)
    if data is None:
        return
    data['task_id'] = task_id
    config, _ = recommend_config(data, logger)
    if config is None:
        return
    config = add_scale_dict(config)
    logger.info(f"Suggested config using similar history task for {task_id} = {config}.")
    tuning_data = run_task_and_update(task_id, config, logger)
    logger.info(f"Start {num_epochs - 1} iters of further tuning for task {task_id}.")
    all_tuning_data = [tuning_data]

    cur_best_performance = tuning_data['duration']

    epochs = 1
    while True:
        epochs += 1
        if epochs > num_epochs:
            break

        logger.info(small_gap)
        config, probabilities, selected_index = update_history(task_id, epochs, logger, weights)
        logger.info(small_gap)
        if config is None:
            return
        logger.info(f"Suggested config for {task_id} in iter {epochs} = {config}.")
        tuning_data = run_task_and_update(task_id, config, logger)
        if tuning_data is None:
            continue

        pi = (cur_best_performance - tuning_data['duration']) / cur_best_performance
        reward = 1 / (1 + math.exp(-5 * pi)) if pi < 0 else 1 / (1 + math.exp(-10 * pi))
        modified_reward = [reward / probabilities[0], 0] if selected_index == 0 else [0, reward / probabilities[1]]
        weights = [weights[0] * math.exp(rate_tradeoff * modified_reward[0] / 2),
                   weights[1] * math.exp(rate_tradeoff * modified_reward[1] / 2)]
        if tuning_data['duration'] < cur_best_performance:
            cur_best_performance = tuning_data['duration']
        logger.info(f"Performance improvement = {pi}, reward = {reward}. Update weights to {weights}.")

        all_tuning_data.append(tuning_data)

    df = pd.DataFrame(all_tuning_data)
    df.to_csv(gen_result_file_name(task_id, 'new', num_epochs), index=False)
    logger.info(f"Finish tuning for new task {task_id}.")
    logger.info(big_gap)
