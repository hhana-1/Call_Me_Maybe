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
    steps: list[dict[str, Any]], shortlist: list[str], count: int
) -> list[str]:
    """Show the keyword shortlist, then how the name was written."""
    if len(shortlist) == count:
        kept = f"kept all {count} functions"
    else:
        kept = f"kept {len(shortlist)} of {count}: {', '.join(shortlist)}"
    lines = textwrap.wrap(
        f"2. Function name  (keyword ranker {kept})",
        width=_WIDTH,
        subsequent_indent="   ",
    )
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


def _argument_lines(parameters: dict[str, Any]) -> list[str]:
    """List the arguments read from the prompt, with their Python types."""
    lines = ["3. Arguments  (read from the prompt with regular expressions)"]
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
                     trace["shortlist"], trace["function_count"]),
        "",
        *_argument_lines(parameters),
        "",
        "4. Result",
        f"   {trace['selected']} "
        f"{json.dumps(parameters, ensure_ascii=False)}",
        f"   {trace['model_calls']} model calls, {forced} forced tokens",
        "═" * _WIDTH,
    ])
