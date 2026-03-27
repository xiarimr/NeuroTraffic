import argparse
import csv
import json
import math
import os
from pathlib import Path


MODEL_DISPLAY_NAMES = {
    "PPOColight": "PPO",
    "AdvancedDQN": "DQN",
}

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dirs", nargs="+", help="records directory paths")
    parser.add_argument("--output_dir", default="comparison_outputs", help="output directory for markdown and charts")
    return parser.parse_args()


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def load_json(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def build_experiment_label(traffic_conf):
    model_name = traffic_conf.get("MODEL_NAME", "Unknown")
    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    if traffic_conf.get("MODE_SELECTOR_ENABLED", False):
        selector_type = str(traffic_conf.get("SELECTOR_TYPE", "rule")).lower()
        if selector_type == "llm":
            return "LLMMode + {0}".format(display_name)
        return "RuleMode + {0}".format(display_name)
    return "Pure {0}".format(display_name)


def build_selector_display(traffic_conf, final_row=None):
    if not traffic_conf.get("MODE_SELECTOR_ENABLED", False):
        return "off"

    selector_type = str(
        (final_row or {}).get("selector_type", traffic_conf.get("SELECTOR_TYPE", "rule"))
    ).lower()
    selector_backend = str(
        (final_row or {}).get("selector_backend", traffic_conf.get("LLM_SELECTOR_BACKEND", "rule"))
    ).lower()
    if selector_type == "llm":
        return "llm:{0}".format(selector_backend or "mock")
    return "rule"


def load_episode_rows(experiment_dir):
    csv_path = experiment_dir / "episode_metrics.csv"
    rows = []
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                normalized = dict(row)
                normalized["round"] = safe_int(row.get("round"))
                normalized["total_reward"] = safe_float(row.get("total_reward"))
                normalized["average_waiting_time"] = safe_float(row.get("average_waiting_time"))
                normalized["average_queue_length"] = safe_float(row.get("average_queue_length"))
                normalized["throughput"] = safe_float(row.get("throughput"))
                normalized["average_travel_time"] = safe_float(row.get("average_travel_time"))
                normalized["mode_switch_count"] = safe_int(row.get("mode_switch_count"))
                normalized["episode_duration"] = safe_float(row.get("episode_duration"))
                rows.append(normalized)
        return rows

    jsonl_path = experiment_dir / "episode_metrics.jsonl"
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                normalized = dict(row)
                normalized["round"] = safe_int(row.get("round"))
                normalized["total_reward"] = safe_float(row.get("total_reward"))
                normalized["average_waiting_time"] = safe_float(row.get("average_waiting_time"))
                normalized["average_queue_length"] = safe_float(row.get("average_queue_length"))
                normalized["throughput"] = safe_float(row.get("throughput"))
                normalized["average_travel_time"] = safe_float(row.get("average_travel_time"))
                normalized["mode_switch_count"] = safe_int(row.get("mode_switch_count"))
                normalized["episode_duration"] = safe_float(row.get("episode_duration"))
                rows.append(normalized)
        return rows

    nested_matches = []
    if experiment_dir.exists():
        nested_matches = [str(path) for path in experiment_dir.rglob("episode_metrics.csv")]
        if not nested_matches:
            nested_matches = [str(path) for path in experiment_dir.rglob("episode_metrics.jsonl")]
    message = (
        "Missing episode_metrics.csv/jsonl under {0}. "
        "This usually means the experiment was generated before unified experiment logging was added, "
        "or the run did not finish cleanly. "
        "Please confirm utils/generator.py and utils/model_test.py include ExperimentLogger, then rerun the experiment."
    ).format(experiment_dir)
    if nested_matches:
        message += " Found nested metric files: {0}".format(", ".join(nested_matches[:3]))
    raise FileNotFoundError(message)
    return rows


def summarize_experiment(experiment_dir):
    traffic_conf = load_json(experiment_dir / "traffic_env.conf")
    episode_rows = load_episode_rows(experiment_dir)
    label = build_experiment_label(traffic_conf)
    test_rows = [row for row in episode_rows if row.get("stage") == "test"]
    trend_rows = test_rows if test_rows else episode_rows
    final_row = trend_rows[-1] if trend_rows else {}

    return {
        "label": label,
        "experiment_dir": str(experiment_dir),
        "traffic_conf": traffic_conf,
        "rows": episode_rows,
        "trend_rows": trend_rows,
        "final_row": final_row,
    }


def render_markdown_table(experiments):
    headers = [
        "Experiment",
        "Model",
        "Selector",
        "Episodes",
        "Total Reward",
        "Avg Wait",
        "Avg Queue",
        "Throughput",
        "Avg Travel",
        "Current Mode",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for exp in experiments:
        final_row = exp["final_row"]
        traffic_conf = exp["traffic_conf"]
        lines.append(
            "| {0} | {1} | {2} | {3} | {4:.2f} | {5:.2f} | {6:.2f} | {7:.2f} | {8:.2f} | {9} |".format(
                exp["label"],
                traffic_conf.get("MODEL_NAME", "Unknown"),
                build_selector_display(traffic_conf, final_row),
                len(exp["rows"]),
                safe_float(final_row.get("total_reward")),
                safe_float(final_row.get("average_waiting_time")),
                safe_float(final_row.get("average_queue_length")),
                safe_float(final_row.get("throughput")),
                safe_float(final_row.get("average_travel_time")),
                final_row.get("current_mode", ""),
            )
        )
    return "\n".join(lines)


def _scale_point(value, min_value, max_value, low, high):
    if math.isclose(max_value, min_value):
        return (low + high) / 2.0
    return low + (value - min_value) * (high - low) / (max_value - min_value)


def render_line_chart_svg(experiments, metric_name, title, output_path):
    width = 960
    height = 420
    left = 70
    right = 30
    top = 40
    bottom = 60
    plot_width = width - left - right
    plot_height = height - top - bottom

    all_values = []
    max_points = 0
    for exp in experiments:
        values = [safe_float(row.get(metric_name)) for row in exp["trend_rows"]]
        if values:
            all_values.extend(values)
            max_points = max(max_points, len(values))
    if not all_values:
        all_values = [0.0]
        max_points = 1

    min_value = min(all_values)
    max_value = max(all_values)
    if math.isclose(min_value, max_value):
        max_value = min_value + 1.0

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{0}" height="{1}">'.format(width, height),
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="{0}" y="24" font-size="18" font-family="Arial">{1}</text>'.format(left, title),
        '<line x1="{0}" y1="{1}" x2="{2}" y2="{1}" stroke="#333"/>'.format(left, height - bottom, width - right),
        '<line x1="{0}" y1="{1}" x2="{0}" y2="{2}" stroke="#333"/>'.format(left, top, height - bottom),
    ]

    for tick in range(5):
        ratio = tick / 4.0
        y = top + (1.0 - ratio) * plot_height
        value = min_value + ratio * (max_value - min_value)
        lines.append('<line x1="{0}" y1="{1}" x2="{2}" y2="{1}" stroke="#ddd"/>'.format(left, y, width - right))
        lines.append('<text x="8" y="{0}" font-size="12" font-family="Arial">{1:.2f}</text>'.format(y + 4, value))

    for exp_idx, exp in enumerate(experiments):
        values = [safe_float(row.get(metric_name)) for row in exp["trend_rows"]]
        if not values:
            continue
        points = []
        for idx, value in enumerate(values):
            x = left if len(values) == 1 else left + idx * plot_width / float(max(1, max_points - 1))
            y = top + (1.0 - _scale_point(value, min_value, max_value, 0.0, 1.0)) * plot_height
            points.append("{0:.2f},{1:.2f}".format(x, y))
            lines.append('<circle cx="{0:.2f}" cy="{1:.2f}" r="3" fill="{2}"/>'.format(x, y, COLORS[exp_idx % len(COLORS)]))
        lines.append('<polyline fill="none" stroke="{0}" stroke-width="2" points="{1}"/>'.format(
            COLORS[exp_idx % len(COLORS)],
            " ".join(points),
        ))
        lines.append('<text x="{0}" y="{1}" font-size="12" font-family="Arial" fill="{2}">{3}</text>'.format(
            width - right - 220,
            top + 18 * (exp_idx + 1),
            COLORS[exp_idx % len(COLORS)],
            exp["label"],
        ))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines + ["</svg>"]))


def render_bar_chart_svg(experiments, metric_name, title, output_path):
    width = 960
    height = 420
    left = 70
    right = 30
    top = 40
    bottom = 80
    plot_width = width - left - right
    plot_height = height - top - bottom

    values = [safe_float(exp["final_row"].get(metric_name)) for exp in experiments]
    max_value = max(values) if values else 1.0
    if math.isclose(max_value, 0.0):
        max_value = 1.0

    bar_width = plot_width / float(max(1, len(experiments) * 2))
    gap = bar_width

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{0}" height="{1}">'.format(width, height),
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="{0}" y="24" font-size="18" font-family="Arial">{1}</text>'.format(left, title),
        '<line x1="{0}" y1="{1}" x2="{2}" y2="{1}" stroke="#333"/>'.format(left, height - bottom, width - right),
        '<line x1="{0}" y1="{1}" x2="{0}" y2="{2}" stroke="#333"/>'.format(left, top, height - bottom),
    ]

    for idx, exp in enumerate(experiments):
        value = safe_float(exp["final_row"].get(metric_name))
        x = left + gap / 2.0 + idx * (bar_width + gap)
        bar_height = plot_height * value / max_value
        y = height - bottom - bar_height
        color = COLORS[idx % len(COLORS)]
        lines.append('<rect x="{0:.2f}" y="{1:.2f}" width="{2:.2f}" height="{3:.2f}" fill="{4}"/>'.format(
            x, y, bar_width, bar_height, color
        ))
        lines.append('<text x="{0:.2f}" y="{1:.2f}" font-size="12" font-family="Arial">{2:.2f}</text>'.format(
            x, y - 6, value
        ))
        lines.append('<text x="{0:.2f}" y="{1}" font-size="12" font-family="Arial">{2}</text>'.format(
            x,
            height - bottom + 18,
            exp["label"],
        ))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines + ["</svg>"]))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = [summarize_experiment(Path(exp_dir)) for exp_dir in args.experiment_dirs]
    markdown_table = render_markdown_table(experiments)

    reward_chart = output_dir / "total_reward_trend.svg"
    travel_chart = output_dir / "final_average_travel_time.svg"
    render_line_chart_svg(experiments, "total_reward", "Total Reward Trend", reward_chart)
    render_bar_chart_svg(experiments, "average_travel_time", "Final Average Travel Time", travel_chart)

    markdown_output = output_dir / "comparison_summary.md"
    markdown_content = "\n".join([
        "# Experiment Comparison",
        "",
        markdown_table,
        "",
        "![Total Reward Trend](total_reward_trend.svg)",
        "",
        "![Final Average Travel Time](final_average_travel_time.svg)",
        "",
    ])
    markdown_output.write_text(markdown_content, encoding="utf-8")

    print(markdown_table)
    print("")
    print("Saved markdown to:", markdown_output)
    print("Saved charts to:", reward_chart, "and", travel_chart)


if __name__ == "__main__":
    main()
