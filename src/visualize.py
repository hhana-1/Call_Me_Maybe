"""Human-readable rendering of the function-call generation process."""

from typing import Any

_WIDTH = 64
_BAR_WIDTH = 26


def _bar(probability: float) -> str:
    """Draw a proportional probability bar using block characters."""
    filled = round(max(0.0, min(1.0, probability)) * _BAR_WIDTH)
    return "█" * filled + "·" * (_BAR_WIDTH - filled)


def render_generation(
    trace: dict[str, Any], parameters: dict[str, Any]
) -> str:
    """Render one prompt's routing pipeline as an annotated text block.

    Args:
        trace: Selection trace recorded by the router during generation.
        parameters: The extracted arguments for the chosen function.

    Returns:
        A multi-line string visualising each generation step.
    """
    lines: list[str] = ["═" * _WIDTH]
    lines.append(f"PROMPT: {trace['prompt']}")
    lines.append("─" * _WIDTH)

    tokens = trace.get("prompt_tokens") or []
    lines.append(f"1. Tokenization  ({len(tokens)} tokens)")
    lines.append(f"   {tokens}")
    lines.append("")

    candidates = trace["candidates"]
    lines.append(f"2. Candidate functions  ({len(candidates)} shortlisted)")
    lines.append("   " + ", ".join(item["name"] for item in candidates))
    lines.append("")

    lines.append("3. Model routing  (restricted logits → softmax)")
    name_width = max((len(item["name"]) for item in candidates), default=0)
    for item in candidates:
        marker = "►" if item["name"] == trace["selected"] else " "
        lines.append(
            f"   {marker} {item['name']:<{name_width}}  "
            f"logit={item['logit']:7.2f}  "
            f"p={item['probability']:6.1%}  "
            f"{_bar(item['probability'])}"
        )
    lines.append("")

    lines.append(f"4. Selected function: {trace['selected']}")
    lines.append("")

    lines.append("5. Extracted arguments")
    if parameters:
        for key, value in parameters.items():
            lines.append(
                f"   {key} = {value!r}  ({type(value).__name__})"
            )
    else:
        lines.append("   (none)")

    lines.append("═" * _WIDTH)
    return "\n".join(lines)
