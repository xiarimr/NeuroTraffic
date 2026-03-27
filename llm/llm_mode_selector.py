import json
import os
import re
import urllib.error
import urllib.request


SUPPORTED_MODES = (
    "balanced",
    "queue_clearance",
    "main_road_priority",
    "congestion_resistance",
)


def prompt_builder(features, current_mode="balanced"):
    features = features or {}
    lines = [
        "You are a traffic signal control mode selector.",
        "Choose exactly one reward mode for the next control window.",
        "Allowed modes: balanced, queue_clearance, main_road_priority, congestion_resistance.",
        "Output only the mode name, no explanation.",
        "",
        "Current mode: {0}".format(current_mode),
        "Structured traffic features:",
    ]

    for key in sorted(features.keys()):
        value = features[key]
        if isinstance(value, float):
            value = round(value, 6)
        lines.append("- {0}: {1}".format(key, value))

    lines.extend([
        "",
        "Decision hints:",
        "- High queue pressure suggests queue_clearance.",
        "- High trunk pressure suggests main_road_priority.",
        "- Severe spillback or congestion suggests congestion_resistance.",
        "- Stable traffic suggests balanced.",
    ])
    return "\n".join(lines)


class LLMSelector:
    SUPPORTED_MODES = SUPPORTED_MODES
    DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"

    def __init__(
        self,
        window_size=300,
        backend="api",
        model_name=DEFAULT_DEEPSEEK_MODEL,
        api_base=DEFAULT_DEEPSEEK_BASE_URL,
        api_key=None,
        timeout=30,
        temperature=0.0,
        default_mode="balanced",
        api_caller=None,
        local_generate_fn=None,
        mock_response_fn=None,
    ):
        self.selector_type = "llm"
        self.window_size = max(1, int(window_size))
        self.backend = backend
        self.model_name = model_name
        self.api_base = api_base or os.environ.get("DEEPSEEK_API_BASE") or self.DEFAULT_DEEPSEEK_BASE_URL
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.timeout = timeout
        self.temperature = temperature
        self.default_mode = default_mode if default_mode in self.SUPPORTED_MODES else "balanced"
        self.api_caller = api_caller
        self.local_generate_fn = local_generate_fn
        self.mock_response_fn = mock_response_fn

    @classmethod
    def from_env_config(cls, dic_traffic_env_conf):
        dic_traffic_env_conf = dic_traffic_env_conf or {}
        return cls(
            window_size=dic_traffic_env_conf.get("MODE_SELECTOR_WINDOW", 300),
            backend=dic_traffic_env_conf.get("LLM_SELECTOR_BACKEND", "api"),
            model_name=dic_traffic_env_conf.get("LLM_SELECTOR_MODEL", cls.DEFAULT_DEEPSEEK_MODEL),
            api_base=dic_traffic_env_conf.get("LLM_SELECTOR_API_BASE") or os.environ.get("DEEPSEEK_API_BASE") or cls.DEFAULT_DEEPSEEK_BASE_URL,
            api_key=dic_traffic_env_conf.get("LLM_SELECTOR_API_KEY") or os.environ.get("DEEPSEEK_API_KEY"),
            timeout=dic_traffic_env_conf.get("LLM_SELECTOR_TIMEOUT", 30),
            temperature=dic_traffic_env_conf.get("LLM_SELECTOR_TEMPERATURE", 0.0),
            default_mode=dic_traffic_env_conf.get("REWARD_MODE", "balanced"),
        )

    @staticmethod
    def summarize_window(window_snapshots, previous_window_summary=None):
        if not window_snapshots:
            return {
                "average_queue_length": 0.0,
                "trunk_queue_ratio": 0.0,
                "throughput_change_rate": 0.0,
                "spillback_risk": 0.0,
                "average_throughput": 0.0,
            }

        snapshot_count = float(len(window_snapshots))
        queue_sum = sum(snapshot.get("average_queue_length", 0.0) for snapshot in window_snapshots)
        throughput_sum = sum(snapshot.get("average_throughput", 0.0) for snapshot in window_snapshots)
        spillback_sum = sum(snapshot.get("spillback_risk", 0.0) for snapshot in window_snapshots)
        trunk_queue_sum = sum(snapshot.get("total_trunk_queue", 0.0) for snapshot in window_snapshots)
        total_queue_sum = sum(snapshot.get("total_queue", 0.0) for snapshot in window_snapshots)

        average_queue_length = queue_sum / snapshot_count
        average_throughput = throughput_sum / snapshot_count
        spillback_risk = spillback_sum / snapshot_count
        trunk_queue_ratio = 0.0 if abs(total_queue_sum) < 1e-6 else trunk_queue_sum / total_queue_sum

        throughput_change_rate = 0.0
        if previous_window_summary is not None:
            previous_average_throughput = float(previous_window_summary.get("average_throughput", 0.0))
            if abs(previous_average_throughput) > 1e-6:
                throughput_change_rate = (
                    average_throughput - previous_average_throughput
                ) / abs(previous_average_throughput)

        return {
            "average_queue_length": float(average_queue_length),
            "trunk_queue_ratio": float(trunk_queue_ratio),
            "throughput_change_rate": float(throughput_change_rate),
            "spillback_risk": float(spillback_risk),
            "average_throughput": float(average_throughput),
        }

    def build_prompt(self, features, current_mode="balanced"):
        return prompt_builder(features, current_mode=current_mode)

    def select_mode(self, features, current_mode="balanced"):
        mode, _ = self.select_mode_with_reason(features, current_mode=current_mode)
        return mode

    def select_mode_with_reason(self, features, current_mode="balanced"):
        details = self.select_mode_with_details(features, current_mode=current_mode)
        return details["mode"], details["reason"]

    def select_mode_with_details(self, features, current_mode="balanced"):
        prompt = self.build_prompt(features, current_mode=current_mode)
        raw_output = self._query_model(prompt, features, current_mode)
        mode = self._extract_mode(raw_output)
        fallback_triggered = mode is None
        final_mode = "balanced" if fallback_triggered else mode
        reason = "llm_output={0}".format(raw_output)
        if fallback_triggered:
            reason = "fallback_to_balanced: invalid_llm_output={0}".format(raw_output)
        return {
            "mode": final_mode,
            "reason": reason,
            "selector_type": self.selector_type,
            "backend": self.backend,
            "fallback_triggered": fallback_triggered,
            "raw_output": raw_output,
            "prompt": prompt,
        }

    def _query_model(self, prompt, features, current_mode):
        try:
            if self.backend == "api":
                return self._query_api(prompt, features, current_mode)
            if self.backend == "local":
                return self._query_local(prompt, features, current_mode)
            return self._query_mock(prompt, features, current_mode)
        except Exception as exc:
            return "selector_error: {0}".format(exc)

    def _query_local(self, prompt, features, current_mode):
        if callable(self.local_generate_fn):
            return str(self.local_generate_fn(prompt=prompt, features=features, current_mode=current_mode))
        return self._query_mock(prompt, features, current_mode)

    def _query_api(self, prompt, features, current_mode):
        if callable(self.api_caller):
            return str(self.api_caller(prompt=prompt, features=features, current_mode=current_mode))

        if not self.api_base:
            return self._query_mock(prompt, features, current_mode)
        if not self.api_key:
            return "selector_error: missing_api_key"

        url = self.api_base.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = url + "/chat/completions"

        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": 16,
            "stream": False,
            "messages": [
                {"role": "system", "content": "Return only one traffic control mode name from: balanced, queue_clearance, main_road_priority, congestion_resistance."},
                {"role": "user", "content": prompt},
            ],
        }
        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": "Bearer {0}".format(self.api_key)} if self.api_key else {}),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            return "selector_error: api_error={0}".format(exc)

        try:
            return str(result["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError):
            return "selector_error: malformed_api_response"

    def _query_mock(self, prompt, features, current_mode):
        if callable(self.mock_response_fn):
            return str(self.mock_response_fn(prompt=prompt, features=features, current_mode=current_mode))

        average_queue_length = float(features.get("average_queue_length", 0.0))
        trunk_queue_ratio = float(features.get("trunk_queue_ratio", 0.0))
        throughput_change_rate = float(features.get("throughput_change_rate", 0.0))
        spillback_risk = float(features.get("spillback_risk", 0.0))

        if spillback_risk >= 0.5 or average_queue_length >= 20.0:
            return "congestion_resistance"
        if trunk_queue_ratio >= 0.6 and average_queue_length >= 12.0:
            return "main_road_priority"
        if average_queue_length >= 12.0 or throughput_change_rate <= -0.1:
            return "queue_clearance"
        return "balanced"

    def _extract_mode(self, raw_output):
        if raw_output is None:
            return None

        text = str(raw_output).strip().lower()
        for mode in self.SUPPORTED_MODES:
            if text == mode:
                return mode

        for mode in self.SUPPORTED_MODES:
            if re.search(r"\b{0}\b".format(re.escape(mode)), text):
                return mode

        return None


if __name__ == "__main__":
    selector = LLMSelector(backend="mock")
    sample_features = {
        "average_queue_length": 15.2,
        "trunk_queue_ratio": 0.67,
        "throughput_change_rate": -0.08,
        "spillback_risk": 0.31,
    }

    print("Prompt:")
    print(selector.build_prompt(sample_features))
    print("")
    print("Selected mode:")
    print(selector.select_mode(sample_features))
