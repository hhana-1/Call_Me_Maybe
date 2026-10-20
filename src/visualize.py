"""Example of the function-call generation process."""

import json
import textwrap
from typing import Any

_WIDTH = 72
_BAR_WIDTH = 24
_SHOWN_OPTIONS = 5


def _bar(probability: float) -> str:
    """Draw a proportional probability bar using block characters."""
    filled = round(max(0.0, min(1.0, probability)) * _BAR_WIDTH)
    return "█" * filled + "·" * (_BAR_WIDTH - filled)


def _name_lines(trace: dict[str, Any]) -> list[str]:
    """Describe how the function name was decoded through the name trie."""
    lines = [
        f"2. Function name  (only the {trace['function_count']} defined "
        "names can be written)"
    ]
    for step in trace["name_steps"]:
        if step["forced"]:
            lines.append(f"   {step['text']!r:14} forced (only one name fits)")
            continue
        lines.append(
            f"   {step['allowed']} allowed tokens, one model call:"
        )
        options = step["options"]
        for option in options[:_SHOWN_OPTIONS]:
            marker = "►" if option["id"] == step["id"] else " "
            names = ", ".join(option["functions"])
            lines.append(
                f"   {marker} {option['text']!r:14} {names[:30]:30} "
                f"p={option['probability']:6.1%}  "
                f"{_bar(option['probability'])}"
            )
        if len(options) > _SHOWN_OPTIONS:
            lines.append(f"     ... {len(options) - _SHOWN_OPTIONS} more")
    return lines


def _argument_lines(trace: dict[str, Any]) -> list[str]:
    """List each argument's tokens: [forced] or chosen with probability."""
    lines = [
        "3. Arguments  (every token must fit the parameter's type; "
        "[x] = forced)"
    ]
    fields: dict[str, list[str]] = {}
    for step in trace["argument_steps"]:
        text = repr(step["text"])
        piece = (
            f"[{text}]" if step["forced"]
            else f"{text} {step['probability']:.0%} of {step['allowed']:,}"
        )
        fields.setdefault(step["field"], []).append(piece)
    if not fields:
        lines.append("   (no parameters)")
    for field, pieces in fields.items():
        lines.extend(textwrap.wrap(
            "  ".join(pieces),
            width=_WIDTH,
            initial_indent=f"   {field}: ",
            subsequent_indent=" " * (5 + len(field)),
            break_long_words=False,
        ))
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
    tokens = trace.get("prompt_tokens") or []
    forced = sum(
        step["forced"]
        for step in trace["name_steps"] + trace["argument_steps"]
    )
    lines = [
        "═" * _WIDTH,
        f"PROMPT: {trace['prompt']}",
        "─" * _WIDTH,
        f"1. Tokenization  ({len(tokens)} tokens)",
        *textwrap.wrap(repr(tokens), width=_WIDTH, initial_indent="   ",
                       subsequent_indent="   "),
        "",
        *_name_lines(trace),
        "",
        *_argument_lines(trace),
        "",
        "4. Result",
        f"   {trace['selected']} "
        f"{json.dumps(parameters, ensure_ascii=False)}",
        f"   {trace['model_calls']} model calls, {forced} forced tokens",
        "═" * _WIDTH,
    ]
    return "\n".join(lines)
