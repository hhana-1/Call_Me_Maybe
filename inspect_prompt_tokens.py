"""Display tokenization and detokenization for a prompt from the input JSON."""

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_INPUT = "data/input/function_calling_tests.json"


def parse_args() -> argparse.Namespace:
    """Parse prompt selection arguments."""
    parser = argparse.ArgumentParser(description="Inspect prompt tokens")
    parser.add_argument("--index", type=int, default=0, help="Prompt index, starting at 0")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Prompt JSON file")
    return parser.parse_args()


def load_prompt(path: str, index: int) -> str:
    """Load one prompt from the JSON input array."""
    with Path(path).open(encoding="utf-8") as handle:
        prompts: Any = json.load(handle)
    if not isinstance(prompts, list):
        raise ValueError("Prompt input must be a JSON array")
    try:
        prompt = prompts[index]["prompt"]
    except (IndexError, KeyError, TypeError) as error:
        raise ValueError(f"No prompt at index {index}") from error
    if not isinstance(prompt, str):
        raise ValueError("Prompt must be a string")
    return prompt


def main() -> int:
    """Encode, display, and decode one prompt."""
    args = parse_args()
    try:
        from llm_sdk.llm_sdk import Small_LLM_Model

        prompt = load_prompt(args.input, args.index)
        model = Small_LLM_Model()
        encoded = model.encode(prompt)
        token_ids = encoded.detach().cpu().tolist()[0]
        decoded = model.decode(token_ids)
        tokens = [model.decode([token_id]) for token_id in token_ids]

        print(f"Original: {prompt}")
        print(f"Token IDs: {token_ids}")
        print(f"Tokens: {tokens}")
        print(f"Decoded: {decoded}")
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
