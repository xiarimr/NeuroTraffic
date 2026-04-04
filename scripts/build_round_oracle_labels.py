import argparse
import copy
import csv
import json
import os
import pickle
import random
import shutil
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - runtime dependency check
    np = None

try:
    import torch
except ImportError:  # pragma: no cover - runtime dependency check
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.construct_sample import ConstructSample
from utils.model_test import test as model_test
from utils.updater import Updater


VALID_MODES = [
    "balanced",
    "queue_clearance",
    "main_road_priority",
    "congestion_resistance",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build round-level oracle reward-mode labels by replaying one round with four candidate modes."
    )
    parser.add_argument("--experiment_dir", required=True, help="Path under records/<memo>/<experiment>.")
    parser.add_argument("--model_dir", default=None, help="Optional matching model directory.")
    parser.add_argument("--output_dir", default=None, help="Directory for oracle outputs.")
    parser.add_argument(
        "--rounds",
        default=None,
        help="Comma-separated round ids to evaluate, e.g. 0,5,10. If omitted, first rounds are used.",
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=3,
        help="When --rounds is omitted, only evaluate the first N available rounds.",
    )
    parser.add_argument(
        "--modes",
        default=",".join(VALID_MODES),
        help="Comma-separated candidate modes. Defaults to all four modes.",
    )
    parser.add_argument(
        "--eval_run_counts",
        type=int,
        default=None,
        help="Optional override for evaluation RUN_COUNTS. Default uses traffic_env.conf.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Global seed reused for every candidate.")
    parser.add_argument(
        "--keep_temp",
        action="store_true",
        default=False,
        help="Keep temporary replay directories for debugging.",
    )
    return parser.parse_args()


def require_dependencies():
    missing = []
    if np is None:
        missing.append("numpy")
    if torch is None:
        missing.append("torch")
    if missing:
        raise RuntimeError(
            "Missing runtime dependencies: {0}. Install project requirements before running oracle labeling.".format(
                ", ".join(missing)
            )
        )


def load_json(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def normalize_traffic_env_conf(dic_traffic_env_conf):
    normalized = copy.deepcopy(dic_traffic_env_conf)
    phase_dict = normalized.get("PHASE")
    if isinstance(phase_dict, dict):
        normalized["PHASE"] = {
            int(key) if isinstance(key, str) and key.lstrip("-").isdigit() else key: value
            for key, value in phase_dict.items()
        }
    return normalized


def set_global_seed(seed):
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def parse_csv_int(value):
    if value in (None, ""):
        return None
    return int(float(value))


def parse_csv_float(value):
    if value in (None, ""):
        return None
    return float(value)


def parse_rounds_arg(rounds_text):
    if not rounds_text:
        return None
    parsed = []
    for item in rounds_text.split(","):
        item = item.strip()
        if not item:
            continue
        parsed.append(int(item))
    return sorted(set(parsed))


def parse_modes_arg(modes_text):
    modes = []
    for item in modes_text.split(","):
        item = item.strip()
        if not item:
            continue
        if item not in VALID_MODES:
            raise ValueError("Unsupported mode: {0}".format(item))
        modes.append(item)
    if not modes:
        raise ValueError("At least one candidate mode is required.")
    return modes


def discover_model_dir(experiment_dir, explicit_model_dir=None):
    if explicit_model_dir:
        model_dir = Path(explicit_model_dir).resolve()
        if not model_dir.exists():
            raise FileNotFoundError("Model directory does not exist: {0}".format(model_dir))
        return model_dir

    experiment_dir = experiment_dir.resolve()
    parts = list(experiment_dir.parts)
    if "records" not in parts:
        raise ValueError("Unable to infer model directory from non-records path: {0}".format(experiment_dir))
    records_index = parts.index("records")
    inferred_parts = parts[:]
    inferred_parts[records_index] = "model"
    model_dir = Path(*inferred_parts)
    if not model_dir.exists():
        raise FileNotFoundError(
            "Inferred model directory does not exist: {0}. Pass --model_dir explicitly.".format(model_dir)
        )
    return model_dir


def discover_available_rounds(experiment_dir):
    train_round_dir = experiment_dir / "train_round"
    if not train_round_dir.exists():
        raise FileNotFoundError("Missing train_round under {0}".format(experiment_dir))
    rounds = []
    for child in train_round_dir.iterdir():
        if child.is_dir() and child.name.startswith("round_"):
            rounds.append(int(child.name.split("_", 1)[1]))
    if not rounds:
        raise ValueError("No round directories found under {0}".format(train_round_dir))
    return sorted(rounds)


def select_target_rounds(available_rounds, specified_rounds=None, max_rounds=3):
    if specified_rounds is not None:
        missing = [round_id for round_id in specified_rounds if round_id not in available_rounds]
        if missing:
            raise ValueError("Requested rounds are missing from experiment: {0}".format(missing))
        return specified_rounds
    return available_rounds[:max_rounds]


def ensure_clean_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_static_experiment_files(experiment_dir, records_dir, traffic_conf):
    records_dir.mkdir(parents=True, exist_ok=True)
    file_names = [
        "agent.conf",
        "traffic_env.conf",
        "anon_env.conf",
        "cityflow.config",
        traffic_conf["TRAFFIC_FILE"],
        traffic_conf["ROADNET_FILE"],
    ]
    for file_name in file_names:
        src = experiment_dir / file_name
        if src.exists():
            shutil.copy2(src, records_dir / file_name)


def ensure_history_until(experiment_dir, base_train_round_dir, traffic_conf, built_until, target_round):
    if target_round <= built_until + 1:
        return built_until

    for round_id in range(built_until + 1, target_round):
        source_round_dir = experiment_dir / "train_round" / "round_{0}".format(round_id)
        target_round_dir = base_train_round_dir / "round_{0}".format(round_id)
        if not source_round_dir.exists():
            raise FileNotFoundError("Missing source round logs: {0}".format(source_round_dir))
        if not target_round_dir.exists():
            shutil.copytree(source_round_dir, target_round_dir)

        constructor = ConstructSample(
            path_to_samples=str(base_train_round_dir),
            cnt_round=round_id,
            dic_traffic_env_conf=traffic_conf,
        )
        constructor.make_reward_for_system()
        built_until = round_id

    return built_until


def copy_history_samples(base_train_round_dir, candidate_train_round_dir):
    candidate_train_round_dir.mkdir(parents=True, exist_ok=True)
    for sample_file in base_train_round_dir.glob("total_samples_inter_*.pkl"):
        shutil.copy2(sample_file, candidate_train_round_dir / sample_file.name)


def override_round_mode(round_dir, mode):
    for generator_dir in round_dir.iterdir():
        if not generator_dir.is_dir() or not generator_dir.name.startswith("generator"):
            continue
        for inter_file in generator_dir.glob("inter_*.pkl"):
            with open(inter_file, "rb") as file_obj:
                logging_data = pickle.load(file_obj)

            for item in logging_data:
                item["reward_mode"] = mode
                item["mode_reason"] = "oracle_candidate:{0}".format(mode)
                item["selector_type"] = "oracle_candidate"
                item["selector_backend"] = "offline"
                item["selector_fallback_triggered"] = 0

            with open(inter_file, "wb") as file_obj:
                pickle.dump(logging_data, file_obj, -1)


def prepare_candidate_round_logs(experiment_dir, candidate_train_round_dir, cnt_round, mode):
    source_round_dir = experiment_dir / "train_round" / "round_{0}".format(cnt_round)
    target_round_dir = candidate_train_round_dir / "round_{0}".format(cnt_round)
    if not source_round_dir.exists():
        raise FileNotFoundError("Missing round logs for round {0}: {1}".format(cnt_round, source_round_dir))
    shutil.copytree(source_round_dir, target_round_dir)
    override_round_mode(target_round_dir, mode)
    return target_round_dir


def get_required_checkpoint_rounds(cnt_round, dic_agent_conf):
    if cnt_round <= 0:
        return []

    required_rounds = {cnt_round - 1}
    freq = int(dic_agent_conf.get("UPDATE_Q_BAR_FREQ", 0) or 0)
    if freq > 0:
        if dic_agent_conf.get("UPDATE_Q_BAR_EVERY_C_ROUND", False):
            q_bar_round = max(((cnt_round - 1) // freq) * freq, 0)
        else:
            q_bar_round = max(cnt_round - freq, 0)
        required_rounds.add(q_bar_round)
    return sorted(required_rounds)


def copy_required_checkpoints(model_dir, candidate_model_dir, cnt_round, num_agents, dic_agent_conf):
    candidate_model_dir.mkdir(parents=True, exist_ok=True)
    for required_round in get_required_checkpoint_rounds(cnt_round, dic_agent_conf):
        for agent_index in range(num_agents):
            file_name = "round_{0}_inter_{1}.pt".format(required_round, agent_index)
            src = model_dir / file_name
            if not src.exists():
                raise FileNotFoundError("Missing checkpoint required for oracle replay: {0}".format(src))
            shutil.copy2(src, candidate_model_dir / file_name)


def run_candidate_training(candidate_records_dir, candidate_model_dir, cnt_round, dic_agent_conf, dic_traffic_env_conf):
    dic_path = {
        "PATH_TO_MODEL": str(candidate_model_dir),
        "PATH_TO_WORK_DIRECTORY": str(candidate_records_dir),
        "PATH_TO_DATA": str(candidate_records_dir),
        "PATH_TO_ERROR": str(candidate_records_dir / "errors"),
    }
    updater = Updater(
        cnt_round=cnt_round,
        dic_agent_conf=copy.deepcopy(dic_agent_conf),
        dic_traffic_env_conf=copy.deepcopy(dic_traffic_env_conf),
        dic_path=dic_path,
    )
    updater.load_sample_for_agents()
    updater.update_network_for_agents()


def build_evaluation_conf(dic_traffic_env_conf, eval_run_counts, seed):
    eval_conf = copy.deepcopy(dic_traffic_env_conf)
    eval_conf["SEED"] = seed
    eval_conf["MODE_SELECTOR_ENABLED"] = False
    eval_conf["SELECTOR_TYPE"] = "rule"
    eval_conf["LLM_SELECTOR_BACKEND"] = "mock"
    if eval_run_counts is not None:
        eval_conf["RUN_COUNTS"] = eval_run_counts
    return eval_conf


def run_candidate_evaluation(candidate_model_dir, cnt_round, eval_conf):
    model_test(
        str(candidate_model_dir),
        cnt_round,
        eval_conf["RUN_COUNTS"],
        eval_conf,
    )


def read_test_metrics(records_dir, cnt_round):
    metrics_path = records_dir / "episode_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError("Missing episode_metrics.csv after evaluation: {0}".format(metrics_path))

    matched_row = None
    with open(metrics_path, "r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            if row.get("stage") != "test":
                continue
            if parse_csv_int(row.get("round")) != cnt_round:
                continue
            matched_row = row

    if matched_row is None:
        raise ValueError("No test row found for round {0} in {1}".format(cnt_round, metrics_path))

    return {
        "total_reward": parse_csv_float(matched_row.get("total_reward")),
        "average_waiting_time": parse_csv_float(matched_row.get("average_waiting_time")),
        "average_queue_length": parse_csv_float(matched_row.get("average_queue_length")),
        "throughput": parse_csv_float(matched_row.get("throughput")),
        "average_travel_time": parse_csv_float(matched_row.get("average_travel_time")),
        "episode_duration": parse_csv_float(matched_row.get("episode_duration")),
    }


def rank_modes(metrics_by_mode):
    ranked = sorted(
        metrics_by_mode.items(),
        key=lambda item: (
            item[1]["average_travel_time"],
            item[1]["average_waiting_time"],
            item[1]["average_queue_length"],
            -item[1]["throughput"],
            -(item[1]["total_reward"] if item[1]["total_reward"] is not None else float("-inf")),
        ),
    )
    return [mode for mode, _ in ranked]


def write_candidate_metrics_csv(output_dir, rows):
    output_path = output_dir / "candidate_metrics.csv"
    fieldnames = [
        "round",
        "mode",
        "total_reward",
        "average_waiting_time",
        "average_queue_length",
        "throughput",
        "average_travel_time",
        "episode_duration",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def infer_feature_dim(feature_name, dic_traffic_env_conf):
    if "cur_phase" in feature_name:
        if dic_traffic_env_conf.get("BINARY_PHASE_EXPANSION", False):
            phase_encoding = next(iter(dic_traffic_env_conf["PHASE"].values()))
            return len(phase_encoding)
        return 1

    if feature_name == "adjacency_matrix":
        return min(
            dic_traffic_env_conf["TOP_K_ADJACENCY"],
            dic_traffic_env_conf["NUM_INTERSECTIONS"],
        )

    if feature_name in {
        "time_this_phase",
        "queue_length",
        "delay",
        "throughput",
        "phase_switch",
        "pressure_total",
        "main_road_queue_length",
        "main_road_throughput",
    }:
        return 1

    if feature_name in {
        "pressure",
        "num_in_seg_attend",
        "lane_num_vehicle",
        "lane_num_vehicle_downstream",
        "delta_lane_num_vehicle",
        "lane_num_waiting_vehicle_in",
        "lane_num_waiting_vehicle_out",
        "traffic_movement_pressure_queue",
        "traffic_movement_pressure_queue_efficient",
        "traffic_movement_pressure_num",
        "lane_enter_running_part",
    }:
        return dic_traffic_env_conf["NUM_LANE"]

    return None


def get_feature_value_length(value):
    if hasattr(value, "shape"):
        if len(value.shape) == 0:
            return 1
        if len(value.shape) >= 1:
            return int(value.shape[-1])
    if isinstance(value, (list, tuple)):
        return len(value)
    return 1


def validate_sample_record(sample, dic_traffic_env_conf, context):
    state = sample[0]
    next_state = sample[2]
    for container_name, container in [("state", state), ("next_state", next_state)]:
        for feature_name in dic_traffic_env_conf["LIST_STATE_FEATURE"]:
            if feature_name not in container:
                raise ValueError(
                    "{0}: missing feature `{1}` in {2}".format(context, feature_name, container_name)
                )
            expected_dim = infer_feature_dim(feature_name, dic_traffic_env_conf)
            actual_dim = get_feature_value_length(container[feature_name])
            if expected_dim is not None and actual_dim != expected_dim:
                raise ValueError(
                    "{0}: feature `{1}` has dim {2}, expected {3} in {4}".format(
                        context,
                        feature_name,
                        actual_dim,
                        expected_dim,
                        container_name,
                    )
                )


def validate_replay_samples(candidate_train_round_dir, cnt_round, mode, dic_traffic_env_conf):
    sample_files = sorted(candidate_train_round_dir.glob("total_samples_inter_*.pkl"))
    if not sample_files:
        raise ValueError(
            "round {0} mode {1}: no total_samples_inter_*.pkl generated under {2}".format(
                cnt_round, mode, candidate_train_round_dir
            )
        )

    validated = 0
    for sample_file in sample_files:
        with open(sample_file, "rb") as file_obj:
            while True:
                try:
                    samples = pickle.load(file_obj)
                except EOFError:
                    break
                if not samples:
                    continue
                context = "round {0} mode {1} sample_file {2}".format(
                    cnt_round, mode, sample_file.name
                )
                validate_sample_record(samples[0], dic_traffic_env_conf, context)
                validated += 1
                break

    if validated == 0:
        raise ValueError(
            "round {0} mode {1}: generated sample files exist but no non-empty samples were found".format(
                cnt_round, mode
            )
        )


def main():
    args = parse_args()
    require_dependencies()

    experiment_dir = Path(args.experiment_dir).resolve()
    if not experiment_dir.exists():
        raise FileNotFoundError("Experiment directory does not exist: {0}".format(experiment_dir))

    model_dir = discover_model_dir(experiment_dir, args.model_dir)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (
        PROJECT_ROOT / "oracle_round_labels" / experiment_dir.parent.name / experiment_dir.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dic_agent_conf = load_json(experiment_dir / "agent.conf")
    traffic_conf_path = experiment_dir / "traffic_env.conf"
    if not traffic_conf_path.exists():
        raise FileNotFoundError("Missing traffic_env.conf under {0}".format(experiment_dir))
    dic_traffic_env_conf = normalize_traffic_env_conf(load_json(traffic_conf_path))

    available_rounds = discover_available_rounds(experiment_dir)
    target_rounds = select_target_rounds(
        available_rounds,
        specified_rounds=parse_rounds_arg(args.rounds),
        max_rounds=args.max_rounds,
    )
    candidate_modes = parse_modes_arg(args.modes)
    eval_run_counts = args.eval_run_counts or dic_traffic_env_conf["RUN_COUNTS"]

    temp_root = output_dir / "_tmp_round_oracle"
    ensure_clean_dir(temp_root)

    base_history_dir = temp_root / "base_history"
    base_train_round_dir = base_history_dir / "train_round"
    base_train_round_dir.mkdir(parents=True, exist_ok=True)

    built_until = -1
    round_oracle = {}
    candidate_metric_rows = []

    for cnt_round in target_rounds:
        print("=== building oracle for round {0} ===".format(cnt_round))
        built_until = ensure_history_until(
            experiment_dir=experiment_dir,
            base_train_round_dir=base_train_round_dir,
            traffic_conf=copy.deepcopy(dic_traffic_env_conf),
            built_until=built_until,
            target_round=cnt_round,
        )

        metrics_by_mode = {}

        for mode in candidate_modes:
            print("round {0}: evaluating mode {1}".format(cnt_round, mode))
            candidate_root = temp_root / "round_{0}".format(cnt_round) / mode
            candidate_records_dir = candidate_root / "records"
            candidate_model_dir = candidate_root / "model"
            candidate_train_round_dir = candidate_records_dir / "train_round"

            ensure_clean_dir(candidate_root)
            copy_static_experiment_files(experiment_dir, candidate_records_dir, dic_traffic_env_conf)
            copy_history_samples(base_train_round_dir, candidate_train_round_dir)
            prepare_candidate_round_logs(experiment_dir, candidate_train_round_dir, cnt_round, mode)

            round_constructor = ConstructSample(
                path_to_samples=str(candidate_train_round_dir),
                cnt_round=cnt_round,
                dic_traffic_env_conf=copy.deepcopy(dic_traffic_env_conf),
            )
            round_constructor.make_reward_for_system()
            validate_replay_samples(
                candidate_train_round_dir=candidate_train_round_dir,
                cnt_round=cnt_round,
                mode=mode,
                dic_traffic_env_conf=dic_traffic_env_conf,
            )

            copy_required_checkpoints(
                model_dir=model_dir,
                candidate_model_dir=candidate_model_dir,
                cnt_round=cnt_round,
                num_agents=int(dic_traffic_env_conf["NUM_AGENTS"]),
                dic_agent_conf=dic_agent_conf,
            )

            set_global_seed(args.seed)
            run_candidate_training(
                candidate_records_dir=candidate_records_dir,
                candidate_model_dir=candidate_model_dir,
                cnt_round=cnt_round,
                dic_agent_conf=dic_agent_conf,
                dic_traffic_env_conf=dic_traffic_env_conf,
            )

            eval_conf = build_evaluation_conf(dic_traffic_env_conf, eval_run_counts, args.seed)
            set_global_seed(args.seed)
            run_candidate_evaluation(candidate_model_dir, cnt_round, eval_conf)

            metrics = read_test_metrics(candidate_records_dir, cnt_round)
            metrics_by_mode[mode] = metrics
            candidate_metric_rows.append({
                "round": cnt_round,
                "mode": mode,
                **metrics,
            })

        ranking = rank_modes(metrics_by_mode)
        oracle_mode = ranking[0]
        round_oracle[str(cnt_round)] = {
            "oracle_mode": oracle_mode,
            "ranking": ranking,
            "metrics_by_mode": metrics_by_mode,
        }
        print("round {0}: oracle_mode={1}".format(cnt_round, oracle_mode))

    result = {
        "experiment_dir": str(experiment_dir),
        "model_dir": str(model_dir),
        "target_rounds": target_rounds,
        "candidate_modes": candidate_modes,
        "evaluation_run_counts": eval_run_counts,
        "seed": args.seed,
        "round_oracle": round_oracle,
    }

    with open(output_dir / "oracle_labels.json", "w", encoding="utf-8") as file_obj:
        json.dump(result, file_obj, indent=2)
    write_candidate_metrics_csv(output_dir, candidate_metric_rows)

    if not args.keep_temp:
        shutil.rmtree(temp_root)

    print("Oracle labels written to {0}".format(output_dir / "oracle_labels.json"))
    print("Candidate metrics written to {0}".format(output_dir / "candidate_metrics.csv"))


if __name__ == "__main__":
    main()
