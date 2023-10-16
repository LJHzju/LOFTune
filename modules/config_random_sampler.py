# -*- coding: utf-8 -*-
"""
description: Knob information

"""
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.stats.qmc import Sobol
from config.knobs_list import *
from util import check_sample, gen_sql_list, add_scale_dict, get_resource_usage_of_config
import copy
import math
import time


def sample_and_scale_to_configs(knob_names, num, seed):
    sampler = Sobol(d=len(knob_names), seed=seed)
    samples = sampler.random_base2(num)
    qualified_configs = []
    for sample in samples:
        knobs = {}
        for idx in range(len(knob_names)):
            name = knob_names[idx]
            details = KNOB_DETAILS[name]
            sample_value = sample[idx]
            knob_type = details['type']
            if knob_type == KnobType.INTEGER:
                min_value, max_value, step_length = details['range'][0: 3]
                converted_value = int(round(min_value + sample_value * (max_value - min_value)))
                steps = int(round(float(converted_value - min_value) / step_length))
                converted_value = min_value + step_length * steps
                eval_value = converted_value
            elif knob_type == KnobType.NUMERIC:
                min_value, max_value, step_length = details['range'][0: 3]
                converted_value = round(min_value + sample_value * (max_value - min_value), 2)
                steps = int(round(float(converted_value - min_value) / step_length))
                converted_value = min_value + step_length * steps
                eval_value = round(converted_value, 2)
            elif knob_type == KnobType.CATEGORICAL:
                candidates = details['candidates']
                index = int(sample_value * len(candidates))
                eval_value = index if index < len(candidates) else len(candidates) - 1
            else:
                print('Wrong knob type: ', knob_type)
                continue

            knobs[name] = eval_value
        qualified_configs.append(knobs)

    return qualified_configs


def generate_resource_configs(num_sampled_configs, seed):
    print(f"Generating resource configurations.", flush=True)
    power = int(math.log(num_sampled_configs, 2)) if num_sampled_configs >= 8 else 3
    resource_configs = []
    while len(resource_configs) < num_sampled_configs:
        unchecked_configs = sample_and_scale_to_configs(list(RESOURCE_KNOB_DETAILS.keys()), power, seed)
        configs = []
        for config in unchecked_configs:
            if check_sample(config):
                configs.append(config)
        resource_configs.extend(configs)
        seed += 1
    random.seed(seed)
    resource_configs = random.sample(resource_configs, num_sampled_configs)

    return resource_configs, seed


def generate_non_resource_configs(num_sampled_configs, seed):
    print(f"Generating non-resource configurations.", flush=True)
    power = int(math.log(num_sampled_configs, 2)) if num_sampled_configs > 2 else 1
    non_resource_configs = sample_and_scale_to_configs(list(NON_RESOURCE_KNOB_DETAILS.keys()), power, seed)
    while len(non_resource_configs) < num_sampled_configs:
        seed += 1
        configs = sample_and_scale_to_configs(list(NON_RESOURCE_KNOB_DETAILS.keys()), power, seed)
        random.seed(seed)
        configs = random.sample(configs, num_sampled_configs - len(non_resource_configs))
        non_resource_configs.extend(configs)

    if len(non_resource_configs) > num_sampled_configs:
        non_resource_configs = random.sample(non_resource_configs, num_sampled_configs)

    return non_resource_configs, seed


def set_parallelism(config, seed):
    total_cores, total_memory = get_resource_usage_of_config(config)
    random.seed(seed)
    if 'spark.default.parallelism' in KNOBS:
        config['spark.default.parallelism'] = random.randint(2 * total_cores, 4 * total_cores)
    if 'spark.sql.shuffle.partitions' in KNOBS:
        config['spark.sql.shuffle.partitions'] = random.randint(2 * total_cores, 4 * total_cores)
    return config


def sample_configs(num_sampled_configs, tasks=None):
    if tasks is None:
        tasks = gen_sql_list()

    num_sampled_configs_per_sql = math.ceil(num_sampled_configs / len(tasks))

    configs_for_all_sqls = []
    seed = int(time.time() * 1000)
    for task in tasks:
        print(f"Generating {num_sampled_configs_per_sql} configurations for {task}.", flush=True)
        resource_configs, seed = generate_resource_configs(num_sampled_configs_per_sql, seed)
        non_resource_configs, seed = generate_non_resource_configs(num_sampled_configs_per_sql, seed)
        configs = []
        for config_1, config_2 in zip(resource_configs, non_resource_configs):
            config = copy.deepcopy(config_1)
            config.update(config_2)
            configs.append(config)
        configs_for_all_sqls.append(configs)
        seed += 1

    generated_configs = []
    for group in zip(*configs_for_all_sqls):
        for config in group:
            generated_configs.append(add_scale_dict(set_parallelism(config, seed)))
            seed += 1
    generated_configs = generated_configs[: num_sampled_configs]

    return generated_configs
