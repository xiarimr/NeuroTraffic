import importlib


AGENT_IMPORTS = {
    "Fixedtime": "models.fixedtime_agent:FixedtimeAgent",
    "MaxPressure": "models.maxpressure_agent:MaxPressureAgent",
    "EfficientMaxPressure": "models.efficient_maxpressure_agent:EfficientMaxPressureAgent",
    "AdvancedMaxPressure": "models.advanced_maxpressure_agent:AdvancedMaxPressureAgent",
    "EfficientPressLight": "models.presslight_one:PressLightAgentOne",
    "EfficientColight": "models.colight_agent:CoLightAgent",
    "PPOColight": "models.ppo_colight_agent:PPOCoLightAgent",
    "EfficientMPLight": "models.mplight_agent:MPLightAgent",
    "AdvancedMPLight": "models.advanced_mplight_agent:AdvancedMPLightAgent",
    "AdvancedColight": "models.colight_agent:CoLightAgent",
    "AdvancedDQN": "models.simple_dqn_one:SimpleDQNAgentOne",
    "Attend": "models.attendlight_agent:AttendLightAgent",
}


def get_agent_class(agent_name):
    module_name, class_name = AGENT_IMPORTS[agent_name].split(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


DIC_AGENTS = AGENT_IMPORTS

DIC_PATH = {
    "PATH_TO_MODEL": "model/default",
    "PATH_TO_WORK_DIRECTORY": "records/default",
    "PATH_TO_DATA": "data/template",
    "PATH_TO_PRETRAIN_MODEL": "model/default",
    "PATH_TO_ERROR": "errors/default",
}

dic_traffic_env_conf = {

    "LIST_MODEL": ["Fixedtime", "MaxPressure", "EfficientMaxPressure", "AdvancedMaxPressure",
                   "EfficientPressLight", "EfficientColight", "PPOColight", "EfficientMPLight",
                   "AdvancedMPLight", "AdvancedColight", "AdvancedDQN", "Attend"],
    "LIST_MODEL_NEED_TO_UPDATE": ["EfficientPressLight", "EfficientColight", "EfficientMPLight",
                                  "AdvancedMPLight", "AdvancedColight", "AdvancedDQN", "Attend",
                                  "PPOColight"],

    "NUM_LANE": 12,
    "PHASE_MAP": [[1, 4, 12, 13, 14, 15, 16, 17], [7, 10, 18, 19, 20, 21, 22, 23], [0, 3, 18, 19, 20, 21, 22, 23], [6, 9, 12, 13, 14, 15, 16, 17]],
    "FORGET_ROUND": 20,
    "RUN_COUNTS": 3600,
    "SEED": 0,
    "MODEL_NAME": None,
    "TOP_K_ADJACENCY": 5,

    "ACTION_PATTERN": "set",
    "NUM_INTERSECTIONS": 1,

    "OBS_LENGTH": 167,
    "MIN_ACTION_TIME": 15,
    "MEASURE_TIME": 15,

    "BINARY_PHASE_EXPANSION": True,

    "YELLOW_TIME": 5,
    "ALL_RED_TIME": 0,
    "NUM_PHASES": 4,
    "NUM_LANES": [3, 3, 3, 3],

    "INTERVAL": 1,

    "LIST_STATE_FEATURE": [
        "cur_phase",
        "time_this_phase",
        "lane_num_vehicle",
        "lane_num_vehicle_downstream",
        "traffic_movement_pressure_num",
        "traffic_movement_pressure_queue",
        "traffic_movement_pressure_queue_efficient",
        "pressure",
        "adjacency_matrix"
    ],
    "REWARD_MODE": "balanced",
    "REWARD_LEGACY_COMPAT": True,
    "REWARD_WEIGHTS_BY_MODE": {},
    "REWARD_MAIN_ROAD_APPROACHES": ["W", "E"],
    "SELECTOR_TYPE": "rule",
    "MODE_SELECTOR_ENABLED": True,
    "MODE_SELECTOR_WINDOW": 300,
    "MODE_SELECTOR_THRESHOLDS": {},
    "LLM_SELECTOR_BACKEND": "mock",
    "LLM_SELECTOR_MODEL": "mock-llm-mode-selector",
    "LLM_SELECTOR_API_BASE": None,
    "LLM_SELECTOR_API_KEY": None,
    "LLM_SELECTOR_TIMEOUT": 30,
    "LLM_SELECTOR_TEMPERATURE": 0.0,
    "DIC_REWARD_INFO": {
        "queue_length": 0,
        "pressure": 0,
    },
    "PHASE": {
        1: [0, 1, 0, 1, 0, 0, 0, 0],
        2: [0, 0, 0, 0, 0, 1, 0, 1],
        3: [1, 0, 1, 0, 0, 0, 0, 0],
        4: [0, 0, 0, 0, 1, 0, 1, 0]
    },
    "list_lane_order": ["WL", "WT", "EL", "ET", "NL", "NT", "SL", "ST"],
    "PHASE_LIST": ['WT_ET', 'NT_ST', 'WL_EL', 'NL_SL'],

}

DIC_BASE_AGENT_CONF = {
    "D_DENSE": 20,
    "LEARNING_RATE": 0.001,
    "PATIENCE": 10,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "SAMPLE_SIZE": 3000,
    "MAX_MEMORY_LEN": 12000,

    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,

    "GAMMA": 0.8,
    "NORMAL_FACTOR": 20,

    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
}

DIC_FIXEDTIME_AGENT_CONF = {
    "FIXED_TIME": [15, 15, 15, 15]
}

DIC_MAXPRESSURE_AGENT_CONF = {
    "FIXED_TIME": [15, 15, 15, 15]
}
