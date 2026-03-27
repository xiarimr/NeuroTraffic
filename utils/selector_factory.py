from llm.llm_mode_selector import LLMSelector

from .mode_selector import ModeSelector


def create_mode_selector(dic_traffic_env_conf):
    dic_traffic_env_conf = dic_traffic_env_conf or {}
    selector_type = str(dic_traffic_env_conf.get("SELECTOR_TYPE", "rule")).lower()

    if selector_type == "llm":
        return LLMSelector.from_env_config(dic_traffic_env_conf)
    return ModeSelector.from_env_config(dic_traffic_env_conf)
