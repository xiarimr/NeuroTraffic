import argparse
import csv
import json
import pickle
import random
import re
import shutil
import sys
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.construct_sample import ConstructSample
from utils.updater import Updater


CANDIDATE_MODES = [
    "balanced",
    "queue_clearance",
    "main_road_priority",
    "congestion_resistance",
]

TRAIN_SAMPLE_RE = re.compile(
    r"^train:train_round_(?P<round_id>\d+)_generator_(?P<generator_id>\d+):(?P<time_step>\d+)$"
)


class WindowOracleError(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a single-sample window-level oracle label using the current training pipeline."
    )
    parser.add_argument(
        "--dataset-json",
        default=None,
        help="Optional dataset.json used only to enrich sample metadata.",
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help="Optional sample id, for example train:train_round_11_generator_0:000200",
    )
    parser.add_argument(
        "--source-experiment-dir",
        default=None,
        help="Source records experiment directory. Required when --dataset-json is not provided.",
    )
    parser.add_argument(
        "--round-id",
        type=int,
        default=None,
        help="Target train round id. Used when --sample-id is not provided.",
    )
    parser.add_argument(
        "--generator-id",
        type=int,
        default=None,
        help="Target generator id. Used when --sample-id is not provided.",
    )
    parser.add_argument(
        "--time-step",
        type=int,
        default=None,
        help="Target window start time. Used when --sample-id is not provided.",
    )
    parser.add_argument(
        "--output-root",
        default="window_oracle_labels",
        help="Root directory for oracle outputs.",
    )
    parser.add_argument(
        "--workspace-root",
        default="tmp_window_oracle",
        help="Root directory for temporary candidate workspaces.",
    )
    parser.add_argument(
        "--candidate-modes",
        nargs="+",
        default=CANDIDATE_MODES,
        choices=CANDIDATE_MODES,
        help="Candidate reward modes to evaluate.",
    )
    parser.add_argument(
        "--run-count",
        type=int,
        default=None,
        help="Optional override for test run count. Defaults to traffic_env.conf RUN_COUNTS.",
    )
    parser.add_argument(
        "--keep-workspaces",
        action="store_true",
        help="Keep candidate workspaces after evaluation for inspection.",
    )
    parser.add_argument(
        "--keep-round-logs",
        action="store_true",
        help="Keep copied round_* logs in the candidate workspace.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve inputs and planned paths without running candidate evaluation.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def sanitize_name(value):
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return sanitized.strip("_") or "sample"


def parse_train_sample_id(sample_id):
    match = TRAIN_SAMPLE_RE.match(sample_id)
    if not match:
        raise WindowOracleError(
            "Only train samples are supported. Expected format like "
            "'train:train_round_11_generator_0:000200', got: {0}".format(sample_id)
        )
    parsed = {
        "sample_id": sample_id,
        "round_id": int(match.group("round_id")),
        "generator_id": int(match.group("generator_id")),
        "time_step": int(match.group("time_step")),
    }
    parsed["episode_id"] = "train:train_round_{0}_generator_{1}".format(
        parsed["round_id"], parsed["generator_id"]
    )
    return parsed


def build_train_sample_id(round_id, generator_id, time_step):
    return "train:train_round_{0}_generator_{1}:{2:06d}".format(
        int(round_id), int(generator_id), int(time_step)
    )


def build_parsed_sample(round_id, generator_id, time_step):
    return parse_train_sample_id(build_train_sample_id(round_id, generator_id, time_step))


def resolve_cli_target(args):
    if args.sample_id is not None:
        parsed_sample = parse_train_sample_id(args.sample_id)
        explicit_values = [args.round_id, args.generator_id, args.time_step]
        if any(value is not None for value in explicit_values):
            expected = build_parsed_sample(
                args.round_id if args.round_id is not None else parsed_sample["round_id"],
                args.generator_id if args.generator_id is not None else parsed_sample["generator_id"],
                args.time_step if args.time_step is not None else parsed_sample["time_step"],
            )
            if expected != parsed_sample:
                raise WindowOracleError(
                    "--sample-id and explicit round/generator/time arguments are inconsistent"
                )
        return parsed_sample

    if args.round_id is None or args.generator_id is None or args.time_step is None:
        raise WindowOracleError(
            "Provide either --sample-id or the full set of --round-id --generator-id --time-step"
        )
    return build_parsed_sample(args.round_id, args.generator_id, args.time_step)


def normalize_env_conf(env_conf):
    normalized = deepcopy(env_conf)
    phase_map = normalized.get("PHASE")
    if isinstance(phase_map, dict):
        normalized["PHASE"] = {
            int(key) if isinstance(key, str) and key.isdigit() else key: value
            for key, value in phase_map.items()
        }
    return normalized


def find_sample_by_id(dataset_payload, sample_id):
    for sample in dataset_payload["samples"]:
        if sample.get("sample_id") == sample_id:
            return sample
    raise WindowOracleError("Sample id not found in dataset.json: {0}".format(sample_id))


def resolve_experiment_from_sample(dataset_meta, sample, override=None):
    _ = sample
    if override:
        return Path(override).resolve()
    experiment_dir = dataset_meta.get("source_experiment_dir")
    if not experiment_dir:
        raise WindowOracleError("dataset.json meta.source_experiment_dir is missing")
    return Path(experiment_dir).resolve()


def resolve_experiment_from_args(args, dataset_payload=None, sample=None):
    if args.source_experiment_dir:
        return Path(args.source_experiment_dir).resolve()
    if dataset_payload is None:
        raise WindowOracleError(
            "--source-experiment-dir is required when --dataset-json is not provided"
        )
    return resolve_experiment_from_sample(dataset_payload.get("meta", {}), sample, None)


def infer_source_model_dir(source_records_dir):
    parts = list(source_records_dir.parts)
    if "records" not in parts:
        return None
    records_index = parts.index("records")
    model_parts = parts[:]
    model_parts[records_index] = "model"
    return Path(*model_parts)


def derive_experiment_key(source_records_dir):
    parts = list(source_records_dir.parts)
    if "records" in parts:
        idx = parts.index("records")
        if idx + 2 < len(parts):
            return Path(parts[idx + 1]) / parts[idx + 2]
    return Path(sanitize_name(source_records_dir.name))


def ensure_clean_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def find_support_file(file_name):
    for candidate in ROOT_DIR.rglob(file_name):
        if candidate.is_file():
            return candidate
    return None


def copy_static_experiment_files(source_records_dir, target_records_dir, env_conf):
    target_records_dir.mkdir(parents=True, exist_ok=True)

    agent_conf_src = source_records_dir / "agent.conf"
    traffic_env_conf_src = source_records_dir / "traffic_env.conf"
    if not agent_conf_src.exists():
        raise WindowOracleError("Missing agent.conf in source experiment: {0}".format(agent_conf_src))
    if not traffic_env_conf_src.exists():
        raise WindowOracleError(
            "Missing traffic_env.conf in source experiment: {0}".format(traffic_env_conf_src)
        )

    shutil.copy2(agent_conf_src, target_records_dir / "agent.conf")
    shutil.copy2(traffic_env_conf_src, target_records_dir / "traffic_env.conf")

    for file_key in ["TRAFFIC_FILE", "ROADNET_FILE"]:
        file_name = env_conf[file_key]
        source_candidate = source_records_dir / file_name
        if not source_candidate.exists():
            source_candidate = find_support_file(file_name)
        if source_candidate is None or not source_candidate.exists():
            raise WindowOracleError("Unable to locate support file: {0}".format(file_name))
        shutil.copy2(source_candidate, target_records_dir / file_name)


def copy_required_checkpoints(source_model_dir, target_model_dir, target_round, num_agents):
    if target_round <= 0:
        return False
    if source_model_dir is None or not source_model_dir.exists():
        return False

    required_files = []
    for round_idx in range(target_round):
        for inter_idx in range(num_agents):
            required_files.append(
                source_model_dir / "round_{0}_inter_{1}.pt".format(round_idx, inter_idx)
            )

    if not all(path.exists() for path in required_files):
        return False

    target_model_dir.mkdir(parents=True, exist_ok=True)
    for checkpoint_path in required_files:
        shutil.copy2(checkpoint_path, target_model_dir / checkpoint_path.name)
    return True


def load_selector_snapshot_from_records(source_records_dir, parsed_sample):
    mode_log_path = (
        source_records_dir
        / "train_round"
        / "round_{0}".format(parsed_sample["round_id"])
        / "generator_{0}".format(parsed_sample["generator_id"])
        / "mode_selector_log.csv"
    )
    if not mode_log_path.exists():
        return {}

    with open(mode_log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row_time = int(float(row["time"]))
            except Exception:
                continue
            if row_time == parsed_sample["time_step"]:
                return {
                    "time": row_time,
                    "selector_type": row.get("selector_type"),
                    "backend": row.get("backend"),
                    "previous_mode": row.get("previous_mode"),
                    "selected_mode": row.get("selected_mode"),
                    "current_mode": row.get("current_mode"),
                    "mode_changed": row.get("mode_changed"),
                    "fallback_triggered": row.get("fallback_triggered"),
                    "reason": row.get("reason"),
                    "raw_output": row.get("raw_output"),
                    "average_queue_length": row.get("average_queue_length"),
                    "trunk_queue_ratio": row.get("trunk_queue_ratio"),
                    "throughput_change_rate": row.get("throughput_change_rate"),
                    "spillback_risk": row.get("spillback_risk"),
                    "average_throughput": row.get("average_throughput"),
                }
    return {}


def build_sample_payload_from_records(source_records_dir, parsed_sample):
    selector_snapshot = load_selector_snapshot_from_records(source_records_dir, parsed_sample)
    meta = {
        "episode_id": parsed_sample["episode_id"],
        "time_step": parsed_sample["time_step"],
        "source": "records_direct",
    }
    if selector_snapshot:
        meta.update({
            "selector_type": selector_snapshot.get("selector_type"),
            "selector_backend": selector_snapshot.get("backend"),
            "mode_changed": selector_snapshot.get("mode_changed") == "1",
            "reason": selector_snapshot.get("reason"),
            "raw_output": selector_snapshot.get("raw_output"),
        })
    return {
        "sample_id": parsed_sample["sample_id"],
        "meta": meta,
        "labels": {},
        "outcomes": {},
        "source_record": selector_snapshot,
    }


def merge_sample_payload(base_payload, overlay_payload):
    merged = deepcopy(base_payload)
    for key in ["meta", "labels", "outcomes"]:
        merged.setdefault(key, {})
        merged[key].update(overlay_payload.get(key, {}))
    if overlay_payload.get("source_record"):
        merged["source_record"] = overlay_payload["source_record"]
    return merged


def rewrite_reward_mode_for_window(generator_dir, start_time, end_time, mode):
    patched_records = 0
    patched_files = 0
    for inter_path in sorted(generator_dir.glob("inter_*.pkl")):
        with open(inter_path, "rb") as f:
            rows = pickle.load(f)

        changed = 0
        for row in rows:
            current_time = int(float(row["time"]))
            if start_time <= current_time < end_time:
                row["reward_mode"] = mode
                row["mode_reason"] = "window_oracle_override:{0}@[{1},{2})".format(
                    mode, start_time, end_time
                )
                changed += 1

        with open(inter_path, "wb") as f:
            pickle.dump(rows, f, -1)

        patched_records += changed
        patched_files += 1

    if patched_files == 0:
        raise WindowOracleError("No inter_*.pkl files found under {0}".format(generator_dir))
    if patched_records == 0:
        raise WindowOracleError(
            "No log rows were patched for window [{0}, {1}) in {2}".format(
                start_time, end_time, generator_dir
            )
        )
    return {"patched_files": patched_files, "patched_records": patched_records}


def load_all_samples_from_total_sample(path):
    samples = []
    if not path.exists():
        return samples
    with open(path, "rb") as f:
        while True:
            try:
                samples.extend(pickle.load(f))
            except EOFError:
                break
    return samples


def apply_forget_to_total_samples(train_round_dir, env_conf, agent_conf, cnt_round):
    forget_round = int(env_conf.get("FORGET_ROUND", 0))
    if forget_round <= 0 or cnt_round % forget_round != 0:
        return

    max_memory_len = int(agent_conf["MAX_MEMORY_LEN"])
    for inter_idx in range(env_conf["NUM_INTERSECTIONS"]):
        total_sample_path = train_round_dir / "total_samples_inter_{0}.pkl".format(inter_idx)
        if not total_sample_path.exists():
            continue
        all_samples = load_all_samples_from_total_sample(total_sample_path)
        if not all_samples:
            continue
        trimmed = all_samples[max(0, len(all_samples) - max_memory_len):]
        with open(total_sample_path, "wb") as f:
            pickle.dump(trimmed, f, -1)


def extract_test_metrics(records_dir, target_round):
    metrics_path = records_dir / "episode_metrics.jsonl"
    if not metrics_path.exists():
        raise WindowOracleError("Missing episode metrics file: {0}".format(metrics_path))

    latest_match = None
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("stage") == "test" and int(payload.get("round", -1)) == target_round:
                latest_match = payload

    if latest_match is None:
        raise WindowOracleError(
            "No test metrics found for round {0} in {1}".format(target_round, metrics_path)
        )
    return latest_match


def evaluate_candidate_model(candidate_model_dir, candidate_records_dir, target_round, run_count, env_conf):
    from utils import model_test

    eval_env_conf = deepcopy(env_conf)
    eval_env_conf["RUN_COUNTS"] = int(run_count)
    model_test.test(
        model_dir=str(candidate_model_dir),
        cnt_round=target_round,
        run_cnt=int(run_count),
        _dic_traffic_env_conf=eval_env_conf,
    )
    return extract_test_metrics(candidate_records_dir, target_round)


def candidate_sort_key(metrics):
    return (
        float(metrics["average_travel_time"]),
        float(metrics["average_waiting_time"]),
        float(metrics["average_queue_length"]),
        -float(metrics["throughput"]),
        -float(metrics["total_reward"]),
    )


def rank_candidate_modes(candidate_results):
    successful = [result for result in candidate_results if result["status"] == "ok"]
    successful.sort(key=lambda item: candidate_sort_key(item["metrics"]))
    for rank_idx, item in enumerate(successful, start=1):
        item["rank"] = rank_idx
        item["sort_key"] = list(candidate_sort_key(item["metrics"]))

    failed = [result for result in candidate_results if result["status"] != "ok"]
    for item in failed:
        item["rank"] = None
        item["sort_key"] = None

    return successful + failed


def write_candidate_metrics_csv(path, ranked_results):
    fieldnames = [
        "mode",
        "status",
        "rank",
        "workspace_dir",
        "average_travel_time",
        "average_waiting_time",
        "average_queue_length",
        "throughput",
        "total_reward",
        "mode_switch_count",
        "selector_fallback_count",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in ranked_results:
            metrics = result.get("metrics", {})
            writer.writerow({
                "mode": result["mode"],
                "status": result["status"],
                "rank": result.get("rank"),
                "workspace_dir": result.get("workspace_dir", ""),
                "average_travel_time": metrics.get("average_travel_time", ""),
                "average_waiting_time": metrics.get("average_waiting_time", ""),
                "average_queue_length": metrics.get("average_queue_length", ""),
                "throughput": metrics.get("throughput", ""),
                "total_reward": metrics.get("total_reward", ""),
                "mode_switch_count": metrics.get("mode_switch_count", ""),
                "selector_fallback_count": metrics.get("selector_fallback_count", ""),
                "error": result.get("error", ""),
            })


def stage_round_logs(source_records_dir, candidate_records_dir, round_idx):
    source_round_dir = source_records_dir / "train_round" / "round_{0}".format(round_idx)
    if not source_round_dir.exists():
        raise WindowOracleError("Missing source round directory: {0}".format(source_round_dir))
    target_round_dir = candidate_records_dir / "train_round" / "round_{0}".format(round_idx)
    if target_round_dir.exists():
        shutil.rmtree(target_round_dir)
    shutil.copytree(source_round_dir, target_round_dir)
    return target_round_dir


def rebuild_history_samples_until_round(
    source_records_dir,
    candidate_records_dir,
    target_round,
    parsed_sample,
    env_conf,
    agent_conf,
    candidate_mode,
    bootstrap_from_source_checkpoints,
    keep_round_logs,
):
    train_round_dir = candidate_records_dir / "train_round"
    train_round_dir.mkdir(parents=True, exist_ok=True)

    dic_path = {
        "PATH_TO_MODEL": str(candidate_records_dir.parent / "model"),
        "PATH_TO_WORK_DIRECTORY": str(candidate_records_dir),
    }

    for round_idx in range(target_round + 1):
        staged_round_dir = stage_round_logs(source_records_dir, candidate_records_dir, round_idx)
        if round_idx == target_round:
            generator_dir = staged_round_dir / "generator_{0}".format(parsed_sample["generator_id"])
            window_size = int(env_conf["MODE_SELECTOR_WINDOW"])
            rewrite_reward_mode_for_window(
                generator_dir=generator_dir,
                start_time=parsed_sample["time_step"],
                end_time=parsed_sample["time_step"] + window_size,
                mode=candidate_mode,
            )

        construct_sample = ConstructSample(
            path_to_samples=str(train_round_dir),
            cnt_round=round_idx,
            dic_traffic_env_conf=env_conf,
        )
        construct_sample.make_reward_for_system()

        if bootstrap_from_source_checkpoints and round_idx < target_round:
            apply_forget_to_total_samples(train_round_dir, env_conf, agent_conf, round_idx)
        else:
            updater = Updater(
                cnt_round=round_idx,
                dic_agent_conf=deepcopy(agent_conf),
                dic_traffic_env_conf=env_conf,
                dic_path=dic_path,
            )
            updater.load_sample_for_agents()
            updater.update_network_for_agents()

        if not keep_round_logs and round_idx < target_round:
            shutil.rmtree(staged_round_dir)


def train_candidate_for_window(
    source_records_dir,
    source_model_dir,
    parsed_sample,
    sample_payload,
    agent_conf,
    env_conf,
    candidate_mode,
    workspace_root,
    run_count,
    keep_workspaces,
    keep_round_logs,
):
    safe_sample = sanitize_name(parsed_sample["sample_id"])
    candidate_workspace = workspace_root / safe_sample / candidate_mode
    candidate_records_dir = candidate_workspace / "records"
    candidate_model_dir = candidate_workspace / "model"

    ensure_clean_dir(candidate_workspace)
    candidate_records_dir.mkdir(parents=True, exist_ok=True)
    candidate_model_dir.mkdir(parents=True, exist_ok=True)

    copy_static_experiment_files(source_records_dir, candidate_records_dir, env_conf)
    bootstrap_from_source_checkpoints = copy_required_checkpoints(
        source_model_dir=source_model_dir,
        target_model_dir=candidate_model_dir,
        target_round=parsed_sample["round_id"],
        num_agents=env_conf["NUM_AGENTS"],
    )

    seed_everything(int(env_conf.get("SEED", 0)))
    rebuild_history_samples_until_round(
        source_records_dir=source_records_dir,
        candidate_records_dir=candidate_records_dir,
        target_round=parsed_sample["round_id"],
        parsed_sample=parsed_sample,
        env_conf=env_conf,
        agent_conf=agent_conf,
        candidate_mode=candidate_mode,
        bootstrap_from_source_checkpoints=bootstrap_from_source_checkpoints,
        keep_round_logs=keep_round_logs,
    )

    metrics = evaluate_candidate_model(
        candidate_model_dir=candidate_model_dir,
        candidate_records_dir=candidate_records_dir,
        target_round=parsed_sample["round_id"],
        run_count=run_count,
        env_conf=env_conf,
    )

    result = {
        "mode": candidate_mode,
        "status": "ok",
        "workspace_dir": str(candidate_workspace),
        "bootstrap_from_source_checkpoints": bool(bootstrap_from_source_checkpoints),
        "metrics": metrics,
        "sample_id": sample_payload["sample_id"],
    }

    if not keep_workspaces:
        shutil.rmtree(candidate_workspace)
    return result


def build_output_payload(source_records_dir, sample_payload, parsed_sample, ranked_results, run_count):
    successful = [result for result in ranked_results if result["status"] == "ok"]
    oracle_mode = successful[0]["mode"] if successful else None
    oracle_confidence = None
    if len(successful) >= 2:
        oracle_confidence = (
            float(successful[1]["metrics"]["average_travel_time"])
            - float(successful[0]["metrics"]["average_travel_time"])
        )

    return {
        "meta": {
            "source_experiment_dir": str(source_records_dir),
            "run_count": int(run_count),
            "oracle_definition": "single_window_marginal_oracle",
            "candidate_modes": [result["mode"] for result in ranked_results],
        },
        "sample": {
            "sample_id": sample_payload["sample_id"],
            "meta": sample_payload.get("meta", {}),
            "labels": sample_payload.get("labels", {}),
            "outcomes": sample_payload.get("outcomes", {}),
            "source_record": sample_payload.get("source_record", {}),
            "parsed": parsed_sample,
        },
        "oracle_mode": oracle_mode,
        "oracle_confidence": oracle_confidence,
        "ranking": ranked_results,
    }


def main():
    args = parse_args()
    parsed_sample = resolve_cli_target(args)
    dataset_payload = None
    dataset_json_path = None
    sample_payload = None

    if args.dataset_json:
        dataset_json_path = Path(args.dataset_json).resolve()
        dataset_payload = load_json(dataset_json_path)
        try:
            sample_payload = find_sample_by_id(dataset_payload, parsed_sample["sample_id"])
        except WindowOracleError:
            sample_payload = None

    source_records_dir = resolve_experiment_from_args(args, dataset_payload, sample_payload)
    source_model_dir = infer_source_model_dir(source_records_dir)

    records_sample_payload = build_sample_payload_from_records(source_records_dir, parsed_sample)
    if sample_payload is None:
        sample_payload = records_sample_payload
    else:
        sample_payload = merge_sample_payload(sample_payload, records_sample_payload)

    if sample_payload.get("meta", {}).get("episode_id") != parsed_sample["episode_id"]:
        raise WindowOracleError(
            "Sample metadata episode_id mismatch for {0}".format(parsed_sample["sample_id"])
        )

    env_conf = normalize_env_conf(load_json(source_records_dir / "traffic_env.conf"))
    agent_conf = load_json(source_records_dir / "agent.conf")
    run_count = args.run_count if args.run_count is not None else int(env_conf["RUN_COUNTS"])
    experiment_key = derive_experiment_key(source_records_dir)
    output_dir = Path(args.output_root).resolve() / experiment_key / sanitize_name(parsed_sample["sample_id"])
    workspace_root = Path(args.workspace_root).resolve() / experiment_key

    if args.dry_run:
        dry_run_payload = {
            "dataset_json": str(dataset_json_path) if dataset_json_path is not None else None,
            "sample_id": parsed_sample["sample_id"],
            "parsed_sample": parsed_sample,
            "source_records_dir": str(source_records_dir),
            "source_model_dir": str(source_model_dir) if source_model_dir is not None else None,
            "output_dir": str(output_dir),
            "workspace_root": str(workspace_root),
            "candidate_modes": args.candidate_modes,
            "run_count": run_count,
            "records_source_snapshot": sample_payload.get("source_record", {}),
        }
        print(json.dumps(dry_run_payload, ensure_ascii=False, indent=2))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_results = []
    for candidate_mode in args.candidate_modes:
        print(
            "[window-oracle] sample={0} mode={1} round={2}".format(
                parsed_sample["sample_id"], candidate_mode, parsed_sample["round_id"]
            )
        )
        try:
            result = train_candidate_for_window(
                source_records_dir=source_records_dir,
                source_model_dir=source_model_dir,
                parsed_sample=parsed_sample,
                sample_payload=sample_payload,
                agent_conf=agent_conf,
                env_conf=env_conf,
                candidate_mode=candidate_mode,
                workspace_root=workspace_root,
                run_count=run_count,
                keep_workspaces=args.keep_workspaces,
                keep_round_logs=args.keep_round_logs,
            )
        except Exception as exc:  # pragma: no cover
            result = {
                "mode": candidate_mode,
                "status": "error",
                "workspace_dir": str(workspace_root / sanitize_name(parsed_sample["sample_id"]) / candidate_mode),
                "error": "{0}: {1}".format(type(exc).__name__, exc),
                "traceback": traceback.format_exc(),
            }
            if not args.keep_workspaces:
                candidate_workspace = Path(result["workspace_dir"])
                if candidate_workspace.exists():
                    shutil.rmtree(candidate_workspace)
        candidate_results.append(result)

    ranked_results = rank_candidate_modes(candidate_results)
    payload = build_output_payload(
        source_records_dir=source_records_dir,
        sample_payload=sample_payload,
        parsed_sample=parsed_sample,
        ranked_results=ranked_results,
        run_count=run_count,
    )

    save_json(output_dir / "window_oracle_labels.json", payload)
    write_candidate_metrics_csv(output_dir / "candidate_metrics.csv", ranked_results)
    print(json.dumps({
        "output_dir": str(output_dir),
        "oracle_mode": payload["oracle_mode"],
        "candidate_statuses": {item["mode"]: item["status"] for item in ranked_results},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
