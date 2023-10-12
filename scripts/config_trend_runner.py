import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.config_random_sampler import sample_configs
from modules.task_runner import run_task
from util import *
from config.knobs_list import *
from config.common import cwd
from config.config import *
import pandas as pd


def init_history_data(num_sampled_configs, sqls=None):
    all_apps = []
    generated_configs = sample_configs(num_sampled_configs, tasks=[sqls[0]])
    for config in generated_configs:
        for sql in sqls:
            app_idx = run_task(sql, config)
            event_log_content_of_apps = load_event_log_content(app_idx)  # 根据app_id从HDFS上拉文件
            run_time, app_succeeded = load_info_from_lines(event_log_content_of_apps)  # 获取运行时间
            tuning_data = {"app_id": app_idx, "duration": run_time, "task_id": sql}
            tuning_data.update(config)
            all_apps.append(tuning_data)
    df = pd.DataFrame(all_apps)
    df.to_csv(f"{cwd}/result/{workload}_{data_size}G_similar_SQL_trend.csv", index=False)


def run_other_queries(sqls):
    df = pd.read_csv("./TPCDS_all.csv")

    all_apps = []
    for index, row in df.iterrows():
        config = {name: row[name] for name in KNOBS}
        config['spark.broadcast.compress'] = 'true' if config['spark.broadcast.compress'] else 'false'
        config['spark.rdd.compress'] = 'true' if config['spark.rdd.compress'] else 'false'
        config['spark.sql.join.preferSortMergeJoin'] = 'true' if config['spark.sql.join.preferSortMergeJoin'] else 'false'
        for sql in sqls:
            app_idx = run_task(sql, config)
            event_log_content_of_apps = load_event_log_content(app_idx)  # 根据app_id从HDFS上拉文件
            run_time, app_succeeded = load_info_from_lines(event_log_content_of_apps)  # 获取运行时间
            tuning_data = {"app_id": app_idx, "duration": run_time, "task_id": sql}
            tuning_data.update(config)
            all_apps.append(tuning_data)

    other_df = pd.DataFrame(all_apps)
    other_df.to_csv(f"{cwd}/result/{workload}_{data_size}G_similar_SQL_trend_new.csv", index=False)


if __name__ == '__main__':
    init_history_data(17, sqls=gen_sql_list())
    # init_history_data(20, sqls=['HIBENCH-JOIN'])