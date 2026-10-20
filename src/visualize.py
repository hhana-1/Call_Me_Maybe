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


def _name_lines(steps: list[dict[str, Any]], count: int) -> list[str]:
    """Show how the name was written: forced tokens and the model's choice."""
    lines = [f"2. Function name  (only the {count} defined names can be "
             "written)"]
    for step in steps:
        if step["forced"]:
            lines.append(f"   {step['text']!r:14} forced")
            continue
        lines.append(f"   {step['allowed']} allowed tokens, one model call:")
        for option in step["options"]:
            marker = "►" if option["chosen"] else " "
            lines.append(
                f"   {marker} {option['text']!r:14} {option['label'][:30]:30}"
                f" p={option['probability']:6.1%}  "
                f"{_bar(option['probability'])}"
            )
        hidden = step["allowed"] - len(step["options"])
        if hidden > 0:
            lines.append(f"     ... {hidden} more")
    return lines


def _argument_lines(steps: list[dict[str, Any]]) -> list[str]:
    """List each argument's tokens: [forced] or chosen with probability."""
    lines = ["3. Arguments  (every token must fit the parameter's type; "
             "[x] = forced)"]
    fields: dict[str, list[str]] = {}
    for step in steps:
        text = repr(step["text"])
        fields.setdefault(step["field"], []).append(
            f"[{text}]" if step["forced"]
            else f"{text} {step['probability']:.0%} of {step['allowed']:,}"
        )
    if not fields:
        lines.append("   (no parameters)")
    for field, pieces in fields.items():
        lines += textwrap.wrap(
            "  ".join(pieces),
            width=_WIDTH,
            initial_indent=f"   {field}: ",
            subsequent_indent=" " * (5 + len(field)),
            break_long_words=False,
        )
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
    forced = sum(step["forced"] for step in steps)
    return "\n".join([
        "═" * _WIDTH,
        f"PROMPT: {trace['prompt']}",
        "─" * _WIDTH,
        f"1. Tokenization  ({len(tokens)} tokens)",
        *textwrap.wrap(repr(tokens), width=_WIDTH, initial_indent="   ",
                       subsequent_indent="   "),
        "",
        *_name_lines([s for s in steps if s["field"] == "name"],
                     trace["function_count"]),
        "",
        *_argument_lines([s for s in steps if s["field"] != "name"]),
        "",
        "4. Result",
        f"   {trace['selected']} "
        f"{json.dumps(parameters, ensure_ascii=False)}",
        f"   {trace['model_calls']} model calls, {forced} forced tokens",
        "═" * _WIDTH,
    ])
