from rulervla.cues.metric_ruler import MetricRulerConfig


def build_scale_prompt(task: str, ruler: MetricRulerConfig, *, echo_history_steps=None, echo_views=None):
    if not isinstance(task, str) or not task.strip():
        raise ValueError("Task instruction must be nonempty")
    lines = [f"Task: {task.strip()}"]
    if ruler.enabled:
        lines.extend([
            "Spatial Reference:",
            "Origin: Current end-effector TCP.",
            "Axes: Robot world XYZ.",
            "The auxiliary cue uses red for X, green for Y, and blue for Z.",
            "Only positive robot world X, Y, and Z axes are displayed as solid lines.",
        ])
        if ruler.mode == "metric":
            lines.extend([
                f"Each axis spans 0 cm to +{ruler.extent_m * 100:g} cm from the origin.",
                f"All ticks are equivalent; each physical interval is {ruler.tick_spacing_m * 100:g} cm.",
                "Crowded or clipped ticks may be omitted; visible ticks retain their true physical positions.",
            ])
    if echo_history_steps is not None:
        if not isinstance(echo_history_steps, int) or echo_history_steps < 1:
            raise ValueError("Echo history must be positive control steps")
        if echo_views is None or set(echo_views) != {"agentview", "wrist"}:
            raise ValueError("Echo prompt requires both rendered view diagnostics")
        lines.append("Motion Echo:")
        if all(not value["history_valid"] for value in echo_views.values()):
            lines.append("No valid Motion Echo is available for this observation.")
        elif all(value.get("stationary", False) for value in echo_views.values()):
            lines.append("Measured displacement is below the display threshold; Motion Echo is blank.")
        else:
            lines.append(f"Measured EE displacement over {echo_history_steps} environment control steps.")
            for view, info in echo_views.items():
                if not info["rendered"]:
                    state = "not visible (" + info["reason"].replace("_", " ") + ")."
                elif info["head_rendered"]:
                    state = "yellow arrow to current EE."
                else:
                    state = "yellow segment; current EE is clipped."
                lines.append(f"{view.capitalize()}: {state}")
            if ruler.enabled and any(value["rendered"] for value in echo_views.values()):
                lines.append("Colors mix where cues overlap.")
    return "\n".join(lines)


class StrictPromptTokenizer:
    def __init__(self, tokenizer):
        self.base = tokenizer

    def tokenize(self, prompt, state=None):
        if state is not None:
            raise ValueError("Metric protocol requires continuous state outside the prompt")
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        length = len(self.base._tokenizer.encode(cleaned, add_bos=True))
        length += len(self.base._tokenizer.encode("\n"))
        if length > self.base._max_len:
            raise ValueError(f"Prompt would truncate: {length} tokens > {self.base._max_len}")
        return self.base.tokenize(prompt, state)
