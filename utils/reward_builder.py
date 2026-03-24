import copy
from typing import Any, Dict, Optional

import numpy as np


class RewardBuilder:
    """Centralized reward computation for both online env steps and offline sample construction."""

    SUPPORTED_MODES = (
        "balanced",
        "queue_clearance",
        "main_road_priority",
        "congestion_resistance",
    )

    DEFAULT_MODE_WEIGHTS = {
        # 兼顾排队、延迟、吞吐和切相成本
        "balanced": {
            "queue_length": -0.25,
            "delay": -0.05,
            "throughput": 0.1,
            "phase_switch": -0.02,
        },
        # 优先清空排队车辆，适合车流较小的场景
        "queue_clearance": {
            "queue_length": -0.6,
            "delay": -0.1,
            "throughput": 0.15,
            "phase_switch": -0.05,
        },
        # 优先保证主路通行，适合主路车流较大的场景
        "main_road_priority": {
            "main_road_queue_length": -0.6,
            "queue_length": -0.1,
            "main_road_throughput": 0.4,
            "throughput": 0.1,
            "phase_switch": -0.03,
        },
        # 强化抗拥堵能力，适合车流较大的场景
        "congestion_resistance": {
            "queue_length": -0.35,
            "delay": -0.2,
            "pressure": -0.1,
            "throughput": 0.15,
            "phase_switch": -0.05,
        },
    }

    def __init__(self, reward_info=None, default_mode="balanced",
                 mode_weights=None, use_legacy_compat=True):
        self.reward_info = copy.deepcopy(reward_info or {})
        self.default_mode = default_mode if default_mode in self.SUPPORTED_MODES else "balanced"
        self.use_legacy_compat = use_legacy_compat

        self.mode_weights = copy.deepcopy(self.DEFAULT_MODE_WEIGHTS)
        if mode_weights:
            for mode, weights in mode_weights.items():
                if mode not in self.SUPPORTED_MODES or not isinstance(weights, dict):
                    continue
                self.mode_weights[mode].update(weights)

    @classmethod
    def from_env_config(cls, dic_traffic_env_conf):
        dic_traffic_env_conf = dic_traffic_env_conf or {}
        return cls(
            reward_info=dic_traffic_env_conf.get("DIC_REWARD_INFO"),
            default_mode=dic_traffic_env_conf.get("REWARD_MODE", "balanced"),
            mode_weights=dic_traffic_env_conf.get("REWARD_WEIGHTS_BY_MODE"),
            use_legacy_compat=dic_traffic_env_conf.get("REWARD_LEGACY_COMPAT", True),
        )

    @staticmethod
    def _safe_sum(value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (int, float, np.integer, np.floating, bool)):
            return float(value)
        return float(np.sum(value))

    @classmethod
    def normalize_metrics(cls, metrics: Optional[Dict[str, Any]]) -> Dict[str, float]:
        metrics = metrics or {}

        queue_length = cls._safe_sum(
            metrics.get("queue_length", metrics.get("lane_num_waiting_vehicle_in"))
        )
        delay = cls._safe_sum(metrics.get("delay"))
        if delay == 0.0:
            delay = queue_length + cls._safe_sum(metrics.get("lane_num_waiting_vehicle_out"))

        throughput = cls._safe_sum(metrics.get("throughput"))
        phase_switch = cls._safe_sum(metrics.get("phase_switch"))
        pressure = cls._safe_sum(metrics.get("pressure_total", metrics.get("pressure")))

        normalized = {
            "queue_length": queue_length,
            "delay": delay,
            "throughput": throughput,
            "phase_switch": phase_switch,
            "pressure": pressure,
            "main_road_queue_length": cls._safe_sum(metrics.get("main_road_queue_length")),
            "main_road_throughput": cls._safe_sum(metrics.get("main_road_throughput")),
        }

        if normalized["main_road_queue_length"] == 0.0:
            normalized["main_road_queue_length"] = queue_length
        if normalized["main_road_throughput"] == 0.0:
            normalized["main_road_throughput"] = throughput

        return normalized

    def _resolve_weights(self, mode: str) -> Dict[str, float]:
        if mode == "balanced" and self.use_legacy_compat and self.reward_info:
            return {
                component: float(weight)
                for component, weight in self.reward_info.items()
                if weight != 0
            }
        return self.mode_weights[mode]

    def compute(self, metrics: Optional[Dict[str, Any]], mode: Optional[str] = None) -> float:
        reward_mode = mode or self.default_mode
        if reward_mode not in self.SUPPORTED_MODES:
            raise ValueError("Unsupported reward mode: {0}".format(reward_mode))

        normalized_metrics = self.normalize_metrics(metrics)
        reward = 0.0
        for component, weight in self._resolve_weights(reward_mode).items():
            reward += normalized_metrics.get(component, 0.0) * weight
        return float(reward)
