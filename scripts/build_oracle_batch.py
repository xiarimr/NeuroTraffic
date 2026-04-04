import argparse
import csv
import fnmatch
import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKER_SCRIPT = Path(__file__).with_name("build_round_oracle_labels.py")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch scheduler for building round-level oracle labels across multiple experiments."
    )
    parser.add_argument("--records_root", default="records", help="Root directory containing experiment logs.")
    parser.add_argument("--model_root", default="model", help="Root directory containing model checkpoints.")
    parser.add_argument(
        "--output_root",
        default="oracle_round_labels_batch/batch_default",
        help="Root directory for batch oracle outputs.",
    )
    parser.add_argument(
        "--memo_filter",
        default=None,
        help="Optional comma-separated memo names, e.g. exp_llmmode_dqn,exp_llmmode_dqn_100.",
    )
    parser.add_argument(
        "--experiment_pattern",
        default=None,
        help="Optional fnmatch pattern for experiment directory names, e.g. *jinan_real_2000*.",
    )
    parser.add_argument(
        "--model_name_filter",
        default="AdvancedDQN",
        help="Only run experiments whose MODEL_NAME matches this value. Use * to disable filtering.",
    )
    parser.add_argument(
        "--rounds",
        default=None,
        help="Explicit comma-separated rounds for every experiment, e.g. 5,10,20.",
    )
    parser.add_argument(
        "--round_stride",
        type=int,
        default=5,
        help="When --rounds is omitted, sample rounds every N rounds.",
    )
    parser.add_argument(
        "--start_round",
        type=int,
        default=0,
        help="Minimum round id to include when deriving rounds automatically.",
    )
    parser.add_argument(
        "--max_rounds_per_experiment",
        type=int,
        default=None,
        help="Optional cap on sampled rounds per experiment.",
    )
    parser.add_argument(
        "--eval_run_counts",
        type=int,
        default=None,
        help="Optional override passed down to build_round_oracle_labels.py.",
    )
    parser.add_argument(
        "--modes",
        default="balanced,queue_clearance,main_road_priority,congestion_resistance",
        help="Candidate modes forwarded to the worker script.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed forwarded to the worker script.")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=False,
        help="Skip experiments whose oracle_labels.json already exists under output_root.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        default=False,
        help="Continue batch execution after one experiment fails.",
    )
    parser.add_argument(
        "--keep_temp",
        action="store_true",
        default=False,
        help="Forward --keep_temp to the worker script for debugging.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def parse_comma_list(raw_value):
    if not raw_value:
        return None
    values = []
    for item in raw_value.split(","):
        item = item.strip()
        if item:
            values.append(item)
    return values or None


def parse_rounds(raw_rounds):
    parsed = parse_comma_list(raw_rounds)
    if not parsed:
        return None
    return [int(item) for item in parsed]


def discover_available_rounds(experiment_dir):
    train_round_dir = experiment_dir / "train_round"
    rounds = []
    for child in train_round_dir.iterdir():
        if child.is_dir() and child.name.startswith("round_"):
            rounds.append(int(child.name.split("_", 1)[1]))
    return sorted(rounds)


def derive_rounds(available_rounds, explicit_rounds, start_round, round_stride, max_rounds_per_experiment):
    if explicit_rounds is not None:
        selected = [round_id for round_id in explicit_rounds if round_id in available_rounds]
    else:
        selected = [
            round_id for round_id in available_rounds
            if round_id >= start_round and ((round_id - start_round) % round_stride == 0)
        ]
    if max_rounds_per_experiment is not None:
        selected = selected[:max_rounds_per_experiment]
    return selected


def discover_experiments(records_root, memo_filter, experiment_pattern):
    memo_allowlist = set(memo_filter or [])
    experiments = []
    for memo_dir in sorted(records_root.iterdir()):
        if not memo_dir.is_dir():
            continue
        if memo_allowlist and memo_dir.name not in memo_allowlist:
            continue
        for experiment_dir in sorted(memo_dir.iterdir()):
            if not experiment_dir.is_dir():
                continue
            if experiment_pattern and not fnmatch.fnmatch(experiment_dir.name, experiment_pattern):
                continue
            if not (experiment_dir / "traffic_env.conf").exists():
                continue
            if not (experiment_dir / "agent.conf").exists():
                continue
            if not (experiment_dir / "train_round").exists():
                continue
            experiments.append({
                "memo": memo_dir.name,
                "experiment_name": experiment_dir.name,
                "experiment_dir": experiment_dir,
            })
    return experiments


def ensure_output_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def run_worker(experiment, model_dir, output_dir, rounds, args):
    command = [
        sys.executable,
        str(WORKER_SCRIPT),
        "--experiment_dir",
        str(experiment["experiment_dir"]),
        "--model_dir",
        str(model_dir),
        "--output_dir",
        str(output_dir),
        "--rounds",
        ",".join(str(round_id) for round_id in rounds),
        "--modes",
        args.modes,
        "--seed",
        str(args.seed),
    ]
    if args.eval_run_counts is not None:
        command.extend(["--eval_run_counts", str(args.eval_run_counts)])
    if args.keep_temp:
        command.append("--keep_temp")

    started_at = time.time()
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    duration = time.time() - started_at

    ensure_output_dir(output_dir)
    with open(output_dir / "batch_worker_stdout.log", "w", encoding="utf-8") as file_obj:
        file_obj.write(completed.stdout or "")
    with open(output_dir / "batch_worker_stderr.log", "w", encoding="utf-8") as file_obj:
        file_obj.write(completed.stderr or "")

    return completed, duration


def extract_error_excerpt(stdout_text, stderr_text, max_lines=8):
    combined = []
    if stderr_text:
        combined.extend(line.strip() for line in stderr_text.splitlines() if line.strip())
    if stdout_text:
        combined.extend(line.strip() for line in stdout_text.splitlines() if line.strip())
    if not combined:
        return ""
    return " | ".join(combined[-max_lines:])


def write_batch_summary_csv(output_root, rows):
    output_path = output_root / "batch_summary.csv"
    fieldnames = [
        "memo",
        "experiment_name",
        "model_name",
        "status",
        "reason",
        "available_rounds",
        "selected_rounds",
        "returncode",
        "duration_seconds",
        "error_excerpt",
        "output_dir",
        "experiment_dir",
        "model_dir",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()

    records_root = (PROJECT_ROOT / args.records_root).resolve()
    model_root = (PROJECT_ROOT / args.model_root).resolve()
    output_root = (PROJECT_ROOT / args.output_root).resolve()
    ensure_output_dir(output_root)

    if not WORKER_SCRIPT.exists():
        raise FileNotFoundError("Worker script not found: {0}".format(WORKER_SCRIPT))
    if not records_root.exists():
        raise FileNotFoundError("records_root does not exist: {0}".format(records_root))
    if not model_root.exists():
        raise FileNotFoundError("model_root does not exist: {0}".format(model_root))

    memo_filter = parse_comma_list(args.memo_filter)
    explicit_rounds = parse_rounds(args.rounds)
    experiments = discover_experiments(records_root, memo_filter, args.experiment_pattern)

    if not experiments:
        raise ValueError("No experiments matched the current filters.")

    summary_rows = []
    success_count = 0
    skipped_count = 0
    failed_count = 0

    for experiment in experiments:
        experiment_dir = experiment["experiment_dir"]
        model_dir = model_root / experiment["memo"] / experiment["experiment_name"]
        output_dir = output_root / experiment["memo"] / experiment["experiment_name"]

        traffic_conf = load_json(experiment_dir / "traffic_env.conf")
        model_name = traffic_conf.get("MODEL_NAME")
        available_rounds = discover_available_rounds(experiment_dir)
        selected_rounds = derive_rounds(
            available_rounds=available_rounds,
            explicit_rounds=explicit_rounds,
            start_round=args.start_round,
            round_stride=args.round_stride,
            max_rounds_per_experiment=args.max_rounds_per_experiment,
        )

        row = {
            "memo": experiment["memo"],
            "experiment_name": experiment["experiment_name"],
            "model_name": model_name,
            "status": "",
            "reason": "",
            "available_rounds": ",".join(str(round_id) for round_id in available_rounds),
            "selected_rounds": ",".join(str(round_id) for round_id in selected_rounds),
            "returncode": "",
            "duration_seconds": "",
            "error_excerpt": "",
            "output_dir": str(output_dir),
            "experiment_dir": str(experiment_dir),
            "model_dir": str(model_dir),
        }

        if args.model_name_filter != "*" and model_name != args.model_name_filter:
            row["status"] = "skipped"
            row["reason"] = "model_name_mismatch"
            summary_rows.append(row)
            skipped_count += 1
            continue

        if not model_dir.exists():
            row["status"] = "skipped"
            row["reason"] = "missing_model_dir"
            summary_rows.append(row)
            skipped_count += 1
            continue

        if not selected_rounds:
            row["status"] = "skipped"
            row["reason"] = "no_rounds_selected"
            summary_rows.append(row)
            skipped_count += 1
            continue

        if args.skip_existing and (output_dir / "oracle_labels.json").exists():
            row["status"] = "skipped"
            row["reason"] = "existing_output"
            summary_rows.append(row)
            skipped_count += 1
            continue

        print(
            "[batch] running memo={0} experiment={1} rounds={2}".format(
                experiment["memo"],
                experiment["experiment_name"],
                row["selected_rounds"],
            )
        )

        completed, duration = run_worker(
            experiment=experiment,
            model_dir=model_dir,
            output_dir=output_dir,
            rounds=selected_rounds,
            args=args,
        )

        row["returncode"] = completed.returncode
        row["duration_seconds"] = round(duration, 3)
        if completed.returncode == 0:
            row["status"] = "success"
            success_count += 1
        else:
            row["status"] = "failed"
            row["reason"] = "worker_failed"
            row["error_excerpt"] = extract_error_excerpt(completed.stdout, completed.stderr)
            failed_count += 1
            if not args.continue_on_error:
                summary_rows.append(row)
                write_batch_summary_csv(output_root, summary_rows)
                with open(output_root / "batch_summary.json", "w", encoding="utf-8") as file_obj:
                    json.dump(
                        {
                            "success_count": success_count,
                            "skipped_count": skipped_count,
                            "failed_count": failed_count,
                            "rows": summary_rows,
                        },
                        file_obj,
                        indent=2,
                    )
                raise RuntimeError(
                    "Batch stopped because worker failed for {0}/{1}. See {2}".format(
                        experiment["memo"],
                        experiment["experiment_name"],
                        output_dir / "batch_worker_stderr.log",
                    )
                )

        summary_rows.append(row)

    write_batch_summary_csv(output_root, summary_rows)
    with open(output_root / "batch_summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "success_count": success_count,
                "skipped_count": skipped_count,
                "failed_count": failed_count,
                "rows": summary_rows,
            },
            file_obj,
            indent=2,
        )

    print(
        "Batch finished: success={0}, skipped={1}, failed={2}. Summary saved to {3}".format(
            success_count,
            skipped_count,
            failed_count,
            output_root,
        )
    )


if __name__ == "__main__":
    main()
