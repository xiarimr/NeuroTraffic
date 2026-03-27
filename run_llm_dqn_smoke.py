import argparse
import os
import random
import time

import numpy as np

from utils import config
from utils.utils import merge, pipeline_wrapper

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-memo", type=str, default="exp_llmmode_dqn_smoke")
    parser.add_argument("-traffic_file", type=str, default="anon_3_4_jinan_real.json")
    parser.add_argument("-template", type=str, default="Jinan")
    parser.add_argument("-road_net", type=str, default="3_4")
    parser.add_argument("-num_rounds", type=int, default=3)
    parser.add_argument("-run_counts", type=int, default=900)
    parser.add_argument("-mode_selector_window", type=int, default=300)
    parser.add_argument("-reward_mode", type=str, default="balanced",
                        choices=["balanced", "queue_clearance", "main_road_priority", "congestion_resistance"])
    parser.add_argument("-llm_backend", type=str, default="api", choices=["mock", "local", "api"])
    parser.add_argument("-llm_model", type=str, default="deepseek-chat")
    parser.add_argument("-llm_api_base", type=str, default="https://api.deepseek.com")
    parser.add_argument("-llm_api_key", type=str, default=None)
    parser.add_argument("-seed", type=int, default=0)
    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def main(in_args=None):
    if in_args is None:
        in_args = parse_args()

    set_random_seed(in_args.seed)

    num_col = int(in_args.road_net.split("_")[1])
    num_row = int(in_args.road_net.split("_")[0])
    num_intersections = num_row * num_col

    dic_agent_conf_extra = {
        "CNN_layers": [[32, 32]],
    }
    deploy_dic_agent_conf = merge(getattr(config, "DIC_BASE_AGENT_CONF"), dic_agent_conf_extra)

    dic_traffic_env_conf_extra = {
        "NUM_ROUNDS": in_args.num_rounds,
        "NUM_GENERATORS": 1,
        "NUM_AGENTS": 1,
        "NUM_INTERSECTIONS": num_intersections,
        "RUN_COUNTS": in_args.run_counts,
        "SEED": in_args.seed,
        "MODEL_NAME": "AdvancedDQN",
        "NUM_ROW": num_row,
        "NUM_COL": num_col,
        "TRAFFIC_FILE": in_args.traffic_file,
        "ROADNET_FILE": "roadnet_{0}.json".format(in_args.road_net),
        "LIST_STATE_FEATURE": [
            "cur_phase",
            "traffic_movement_pressure_queue_efficient",
            "lane_enter_running_part",
            "adjacency_matrix",
        ],
        "REWARD_MODE": in_args.reward_mode,
        "MODE_SELECTOR_ENABLED": True,
        "SELECTOR_TYPE": "llm",
        "MODE_SELECTOR_WINDOW": in_args.mode_selector_window,
        "LLM_SELECTOR_BACKEND": in_args.llm_backend,
        "LLM_SELECTOR_MODEL": in_args.llm_model,
        "LLM_SELECTOR_API_BASE": in_args.llm_api_base,
        "LLM_SELECTOR_API_KEY": in_args.llm_api_key,
        "DIC_REWARD_INFO": {
            "queue_length": -0.25,
        },
    }

    timestamp = time.strftime("%m_%d_%H_%M_%S", time.localtime(time.time()))
    dic_path_extra = {
        "PATH_TO_MODEL": os.path.join("model", in_args.memo, in_args.traffic_file + "_" + timestamp),
        "PATH_TO_WORK_DIRECTORY": os.path.join("records", in_args.memo, in_args.traffic_file + "_" + timestamp),
        "PATH_TO_DATA": os.path.join("data", in_args.template, str(in_args.road_net)),
        "PATH_TO_ERROR": os.path.join("errors", in_args.memo),
    }

    deploy_dic_traffic_env_conf = merge(config.dic_traffic_env_conf, dic_traffic_env_conf_extra)
    deploy_dic_path = merge(config.DIC_PATH, dic_path_extra)

    pipeline_wrapper(
        dic_agent_conf=deploy_dic_agent_conf,
        dic_traffic_env_conf=deploy_dic_traffic_env_conf,
        dic_path=deploy_dic_path,
    )

    print("Smoke experiment records saved to:", deploy_dic_path["PATH_TO_WORK_DIRECTORY"])
    print("Smoke experiment model saved to:", deploy_dic_path["PATH_TO_MODEL"])
    return deploy_dic_path["PATH_TO_WORK_DIRECTORY"]


if __name__ == "__main__":
    main()
