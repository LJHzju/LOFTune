import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.config_random_sampler import sample_configs
from modules.task_runner import run_task
from util import *
from config.common import cwd
from config.config import *
import pandas as pd


def init_history_data(num_sampled_configs, sqls=None):
    all_apps = []
    generated_configs = sample_configs(num_sampled_configs, tasks=[sqls[0]])
    for config in generated_configs:
        for sql in sqls:
            app_idx = run_task(sql, config)
            event_log_content_of_apps = load_event_log_content(app_idx)
            run_time, app_succeeded = load_info_from_lines(event_log_content_of_apps)
            tuning_data = {"app_id": app_idx, "duration": run_time, "task_id": sql}
            tuning_data.update(config)
            all_apps.append(tuning_data)
    df = pd.DataFrame(all_apps)
    df.to_csv(f"{cwd}/result/{workload}_{data_size}G_similar_SQL_trend.csv", index=False)


if __name__ == '__main__':
    init_history_data(50, sqls=gen_sql_list())
