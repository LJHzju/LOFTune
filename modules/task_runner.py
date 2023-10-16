import time
from config.config import workload, data_size
from config.common import cwd, config_path
from config.knobs_list import EXTRA_KNOBS
import copy
import os
import subprocess


def write_config_file(config, file_name):
    with open(file_name, "w") as conf_file:
        for knob, value in EXTRA_KNOBS.items():
            if knob in config:
                continue
            config[knob] = value
        for conf in config:
            conf_file.write(f"{conf} {config[conf]}\n")


def run_task(task_id, config):
    cur_time = int(round(time.time() * 1000))
    sqls = task_id.split("_")

    if workload == 'JOIN':
        sqls = ['JOIN']
    elif workload == 'SCAN':
        sqls = ['SCAN']
    elif workload == 'AGGR':
        sqls = ['AGGR']

    os.chdir(os.getenv("SPARK_HOME"))
    cmd = "./benchmarks/scripts/run_benchmark_task.sh"

    app_idx = []
    for index, sql in enumerate(sqls):
        name = f"{cur_time}_{index}"
        config_file_path = f"{config_path}/{name}.conf"
        write_config_file(copy.deepcopy(config), config_file_path)

        print(f"{cmd} --workload={workload} --task={sql} --data_size={data_size} --name={name}")

        result = subprocess.run([cmd, f'--workload={workload}', f'--task={sql}', f'--data_size={data_size}', f'--name={name}'],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        output_lines = result.stdout.decode('utf-8').splitlines()
        app_id = output_lines[1].split(": ")[1]

        if app_id == "":
            os.chdir(cwd)
            return ""

        app_idx.append(app_id)

    os.chdir(cwd)
    app_idx = '/'.join(app_idx)
    return app_idx
