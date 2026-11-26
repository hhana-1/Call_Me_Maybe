"""Example of the function-call generation process."""

import json
import textwrap
from typing import Any

_WIDTH = 72
_BAR_WIDTH = 24


def _bar(probability: float) -> str:
    """Draw a proportional probability bar using block characters."""
    filled = round(max(0.0, min(1.0, probability)) * _BAR_WIDTH)
    return "█" * filled + "·" * (_BAR_WIDTH - filled)


def _name_lines(
    steps: list[dict[str, Any]], shortlist: list[str], count: int,
    ranker: bool,
) -> list[str]:
    """Show which functions the model chose from, then how it wrote it."""
    if not ranker:
        kept = f"keyword ranker off: the model chooses from all {count}"
    elif len(shortlist) == count:
        kept = f"keyword ranker kept all {count} functions"
    else:
        kept = (
            f"keyword ranker kept {len(shortlist)} of {count}: "
            f"{', '.join(shortlist)}"
        )
    lines = textwrap.wrap(
        f"2. Function name  ({kept})",
        width=_WIDTH,
        subsequent_indent="   ",
    )
    for step in steps:
        if step["fast_forward"]:
            lines.append(f"   {step['text']!r:14} fast-forward")
            continue
        lines.extend(_call_lines(step, "   ", 5))
    return lines


def _call_lines(step: dict[str, Any], indent: str, limit: int) -> list[str]:
    """Show one model call: the best allowed tokens and their probability."""
    lines = [f"{indent}{step['allowed']} allowed tokens, one model call:"]
    for option in step["options"][:limit]:
        marker = "►" if option["chosen"] else " "
        lines.append(
            f"{indent}{marker} {option['text']!r:14} "
            f"{option['label'][:30]:30} p={option['probability']:6.1%}  "
            f"{_bar(option['probability'])}"
        )
    hidden = step["allowed"] - min(limit, len(step["options"]))
    if hidden > 0:
        lines.append(f"{indent}  ... {hidden} more")
    return lines


def _argument_lines(parameters: dict[str, Any]) -> list[str]:
    """List the arguments with their Python types."""
    lines = ["3. Arguments  (read with regular expressions, no model)"]
    if not parameters:
        lines.append("   (no parameters)")
    for key, value in parameters.items():
        lines.append(f"   {key} = {value!r}  ({type(value).__name__})")
    return lines


def render_generation(
    trace: dict[str, Any], parameters: dict[str, Any]
) -> str:
    """Render one prompt's constrained decoding as an annotated text block.

    Args:
        trace: Steps recorded by the router during generation.
        parameters: The decoded arguments for the chosen function.

    Returns:
        A multi-line string visualising each generation step.
    """
    steps = trace["steps"]
    tokens = trace["prompt_tokens"]
    fast_forward = sum(step["fast_forward"] for step in steps)
    return "\n".join([
        "═" * _WIDTH,
        f"PROMPT: {trace['prompt']}",
        "─" * _WIDTH,
        f"1. Tokenization  ({len(tokens)} tokens)",
        *textwrap.wrap(repr(tokens), width=_WIDTH, initial_indent="   ",
                       subsequent_indent="   "),
        "",
        *_name_lines(steps, trace["shortlist"], trace["function_count"],
                     trace["ranker"]),
        "",
        *_argument_lines(parameters),
        "",
        "4. Result",
        f"   {trace['selected']} "
        f"{json.dumps(parameters, ensure_ascii=False)}",
        f"   {trace['model_calls']} model calls, "
        f"{fast_forward} fast-forward tokens",
        "═" * _WIDTH,
    ])
