import copy


class ModeSelector:
    SUPPORTED_MODES = (
        "balanced",
        "queue_clearance",
        "main_road_priority",
        "congestion_resistance",
    )

    DEFAULT_THRESHOLDS = {
        "high_average_queue_length": 12.0,
        "very_high_average_queue_length": 20.0,
        "high_trunk_queue_ratio": 0.6,
        "high_spillback_risk": 0.5,
        "negative_throughput_change_rate": -0.1,
    }

    def __init__(self, window_size=300, thresholds=None):
        self.window_size = max(1, int(window_size))
        self.thresholds = copy.deepcopy(self.DEFAULT_THRESHOLDS)
        self.selector_type = "rule"
        self.backend = "rule"
        if thresholds:
            self.thresholds.update(thresholds)

    @classmethod
    def from_env_config(cls, dic_traffic_env_conf):
        dic_traffic_env_conf = dic_traffic_env_conf or {}
        return cls(
            window_size=dic_traffic_env_conf.get("MODE_SELECTOR_WINDOW", 300),
            thresholds=dic_traffic_env_conf.get("MODE_SELECTOR_THRESHOLDS"),
        )

    @staticmethod
    def _safe_div(numerator, denominator):
        if abs(denominator) < 1e-6:
            return 0.0
        return float(numerator) / float(denominator)

    def summarize_window(self, window_snapshots, previous_window_summary=None):
        if not window_snapshots:
            return {
                "average_queue_length": 0.0,
                "trunk_queue_ratio": 0.0,
                "throughput_change_rate": 0.0,
                "spillback_risk": 0.0,
                "average_throughput": 0.0,
            }

        snapshot_count = float(len(window_snapshots))
        queue_sum = sum(snapshot["average_queue_length"] for snapshot in window_snapshots)
        throughput_sum = sum(snapshot["average_throughput"] for snapshot in window_snapshots)
        spillback_sum = sum(snapshot["spillback_risk"] for snapshot in window_snapshots)
        trunk_queue_sum = sum(snapshot["total_trunk_queue"] for snapshot in window_snapshots)
        total_queue_sum = sum(snapshot["total_queue"] for snapshot in window_snapshots)

        average_queue_length = queue_sum / snapshot_count
        average_throughput = throughput_sum / snapshot_count
        trunk_queue_ratio = self._safe_div(trunk_queue_sum, total_queue_sum)
        spillback_risk = spillback_sum / snapshot_count

        previous_average_throughput = 0.0
        if previous_window_summary is not None:
            previous_average_throughput = previous_window_summary.get("average_throughput", 0.0)
        throughput_change_rate = self._safe_div(
            average_throughput - previous_average_throughput,
            abs(previous_average_throughput),
        ) if previous_window_summary is not None else 0.0

        return {
            "average_queue_length": float(average_queue_length),
            "trunk_queue_ratio": float(trunk_queue_ratio),
            "throughput_change_rate": float(throughput_change_rate),
            "spillback_risk": float(spillback_risk),
            "average_throughput": float(average_throughput),
        }

    def select_mode(self, features, current_mode="balanced"):
        average_queue_length = float(features.get("average_queue_length", 0.0))
        trunk_queue_ratio = float(features.get("trunk_queue_ratio", 0.0))
        throughput_change_rate = float(features.get("throughput_change_rate", 0.0))
        spillback_risk = float(features.get("spillback_risk", 0.0))

        if (
            spillback_risk >= self.thresholds["high_spillback_risk"]
            or average_queue_length >= self.thresholds["very_high_average_queue_length"]
        ):
            reason = (
                "spillback_risk={0:.3f}, average_queue_length={1:.3f} exceeded congestion threshold"
                .format(spillback_risk, average_queue_length)
            )
            return "congestion_resistance", reason

        if (
            trunk_queue_ratio >= self.thresholds["high_trunk_queue_ratio"]
            and average_queue_length >= self.thresholds["high_average_queue_length"]
        ):
            reason = (
                "trunk_queue_ratio={0:.3f} with average_queue_length={1:.3f} favored trunk priority"
                .format(trunk_queue_ratio, average_queue_length)
            )
            return "main_road_priority", reason

        if (
            average_queue_length >= self.thresholds["high_average_queue_length"]
            or throughput_change_rate <= self.thresholds["negative_throughput_change_rate"]
        ):
            reason = (
                "average_queue_length={0:.3f}, throughput_change_rate={1:.3f} triggered queue clearance"
                .format(average_queue_length, throughput_change_rate)
            )
            return "queue_clearance", reason

        reason = (
            "average_queue_length={0:.3f}, trunk_queue_ratio={1:.3f}, "
            "throughput_change_rate={2:.3f}, spillback_risk={3:.3f} stayed in stable range"
        ).format(
            average_queue_length,
            trunk_queue_ratio,
            throughput_change_rate,
            spillback_risk,
        )
        return "balanced", reason

    def select_mode_with_reason(self, features, current_mode="balanced"):
        return self.select_mode(features, current_mode=current_mode)

    def select_mode_with_details(self, features, current_mode="balanced"):
        mode, reason = self.select_mode(features, current_mode=current_mode)
        return {
            "mode": mode,
            "reason": reason,
            "selector_type": self.selector_type,
            "backend": self.backend,
            "fallback_triggered": False,
            "raw_output": mode,
            "prompt": "",
        }
