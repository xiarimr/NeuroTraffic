import csv
import json
import os


class ExperimentLogger:
    FIELDNAMES = [
        "stage",
        "episode_name",
        "round",
        "generator",
        "model_name",
        "mode_selector_enabled",
        "selector_type",
        "selector_backend",
        "reward_mode",
        "current_mode",
        "total_reward",
        "average_waiting_time",
        "average_queue_length",
        "throughput",
        "average_travel_time",
        "selector_fallback_count",
        "mode_switch_count",
        "episode_duration",
    ]

    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.csv_path = os.path.join(experiment_dir, "episode_metrics.csv")
        self.jsonl_path = os.path.join(experiment_dir, "episode_metrics.jsonl")
        self._ensure_csv_header()

    def _ensure_csv_header(self):
        if os.path.exists(self.csv_path):
            return
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()

    def log_episode(self, episode_metrics):
        row = {field: episode_metrics.get(field, "") for field in self.FIELDNAMES}
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(row)

        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(row) + "\n")
