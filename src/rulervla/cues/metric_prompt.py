"""Language protocol generated from the same physical ruler configuration."""
from rulervla.cues.metric_ruler import MetricCueConfig


def build_scale_prompt(task: str, ruler: MetricCueConfig, *, echo_history_steps: int | None = None,
                       echo_valid: bool | None = None, echo_stationary: bool = False,
                       echo_rendered: bool | None = None, echo_arrow_visible: bool | None = None):
    if not isinstance(task, str) or not task.strip():
        raise ValueError("Task instruction must be nonempty")
    if not ruler.enabled:
        raise ValueError("Scale-aware prompt requires an enabled ruler")
    spacing_cm = ruler.tick_spacing_m * 100
    lines = [
        f"Task: {task.strip()}",
        "Spatial Reference:",
        "Origin: Current end-effector TCP.",
        "Axes: Robot world XYZ.",
        "The auxiliary cue uses red for X, green for Y, and blue for Z.",
        "Solid lines and filled ticks indicate positive directions.",
        "Dashed lines and hollow ticks indicate negative directions.",
        f"One basic tick interval represents {spacing_cm:g} cm in physical space.",
        f"Every {ruler.major_tick_every} tick is a major tick.",
    ]
    if echo_history_steps is not None:
        if not isinstance(echo_history_steps, int) or echo_history_steps < 1:
            raise ValueError("Echo history must be positive control steps")
        lines.append("Motion Echo:")
        if echo_valid is False:
            lines.append("No valid Motion Echo is available for this observation.")
        elif echo_stationary:
            lines.append("Measured motion is below the Motion Echo display threshold; no arrow is drawn.")
        elif echo_rendered is False:
            lines.append("No Motion Echo is visible in the current camera views.")
        elif echo_arrow_visible is False:
            lines.extend(["Visible yellow line segments show the measured end-effector displacement.",
                          "Clipped endpoints do not mark the current end-effector position."])
        else:
            lines.extend(["The yellow arrow shows the measured end-effector displacement",
                          f"from {echo_history_steps} environment control steps ago to the current position."])
    return "\n".join(lines)


class StrictPromptTokenizer:
    """Use the actual PaliGemma tokenizer while refusing silent truncation."""

    def __init__(self, tokenizer):
        self.base = tokenizer

    def tokenize(self, prompt, state=None):
        if state is not None:
            raise ValueError("Metric protocol currently requires continuous state outside the prompt")
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        length = len(self.base._tokenizer.encode(cleaned, add_bos=True))
        length += len(self.base._tokenizer.encode("\n"))
        if length > self.base._max_len:
            raise ValueError(f"Prompt would truncate: {length} tokens > {self.base._max_len}")
        return self.base.tokenize(prompt, state)
