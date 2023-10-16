import os
import sys
import argparse
import config.config
import pymysql
from sqlalchemy_utils import database_exists, create_database


def conf_check():
    from config.common import sql_base_path, loftune_db_url, cwd, db_name, config_path
    from config.encoder_config import tree_sitter_sql_lib_path

    if not os.path.exists(sql_base_path):
        print("The path of SQL statements does not exist, please specify `sql_base_path` in common.py.")
        sys.exit()

    if not os.path.exists(config_path):
        print(f"The path for Spark Configuration files does not exist, creating directory {config_path}.")
        os.makedirs(config_path)

    if not os.path.exists(tree_sitter_sql_lib_path):
        print("sql.so is not found, please specify `tree_sitter_sql_lib_path` in encoder_config.py.")
        sys.exit()

    pymysql.install_as_MySQLdb()
    if not database_exists(loftune_db_url):
        print(f"Database {db_name} does not exist, creating database according to the connection in config.py.")
        create_database(loftune_db_url)

    if not os.path.exists(f'{cwd}/data'):
        os.makedirs(f'{cwd}/data')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, default="", choices=['IMDB', 'TPCDS', 'TPCH', 'JOIN', 'SCAN', 'AGGR'],
                        help="The benchmark name.")
    parser.add_argument('--data_size', type=int, default=0, help="The data size in GB.")
    parser.add_argument('--type', type=str, default='',
                        choices=['init-tuning-data', 'recommend-config',
                                 'recommend-config-no-history', 'update-history',
                                 'recommend-config-alternately'],
                        help='Decide what to do.')
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'multi'])
    parser.add_argument('--task_id', type=str, default='', help='The workload id (e.g., q2, q9, q11) for operation.')
    parser.add_argument('--epochs', type=int, default=2, help='The number of sampled configs for each history task.')
    parser.add_argument('--random_epochs', type=int, default=10, help='The number of sampled configs for each history task.')
    opt = parser.parse_args()

    config.config.workload = opt.workload
    config.config.data_size = opt.data_size
    config.config.mode = opt.mode

    conf_check()

    if opt.type == 'init-tuning-data':
        from modules.tuning_data_initializer import init_tuning_data
        init_tuning_data()

    # python main.py --type recommend-config --task_id {the task id for config recommendation}
    elif opt.type == 'recommend-config':
        if opt.task_id == '':
            print("Please specify the task id.")
            sys.exit()
        from modules.controller import recommend_config_for_new_task
        recommend_config_for_new_task(opt.task_id, opt.epochs)

    elif opt.type == 'recommend-config-alternately':
        if opt.task_id == '':
            print("Please specify the task id.")
            sys.exit()
        from modules.controller import recommend_config_alternately
        recommend_config_alternately(opt.task_id, opt.epochs)

    elif opt.type == 'recommend-config-no-history':
        if opt.task_id == '':
            print("Please specify the task id.")
            sys.exit()
        from modules.controller import recommend_config_for_new_task_without_history
        recommend_config_for_new_task_without_history(opt.task_id, random_sample_epochs=opt.random_epochs,
                                                      model_sample_epochs=opt.epochs)

    # python main.py --type update-history --task_id {the task id for history update}
    #                                      --epochs {the number of sampled configs for each history task}
    elif opt.type == 'update-history':
        if opt.task_id == '':
            print("Please specify the task id.")
            sys.exit()
        from modules.controller import update_history_task
        update_history_task(opt.task_id, opt.epochs)

