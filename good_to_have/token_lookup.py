"""Look up token IDs in the model vocabulary."""

import argparse
import json
from pathlib import Path
from typing import Any

from llm_sdk.llm_sdk import Small_LLM_Model


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def parse_args() -> argparse.Namespace:
    """Parse token IDs and vocabulary source."""
    parser = argparse.ArgumentParser(
        description="Look up model vocabulary tokens"
    )
    parser.add_argument(
        "token_ids", nargs="+", type=int, help="Token IDs to look up"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Hugging Face model ID used by the SDK "
            f"(default: {DEFAULT_MODEL})"
        ),
    )
    parser.add_argument(
        "--vocabulary",
        help=(
            "Optional vocabulary JSON file to use instead of the "
            "model vocabulary"
        ),
    )
    return parser.parse_args()


def load_vocabulary(path: str) -> dict[int, str]:
    """Load a token-to-ID JSON mapping and reverse it for lookup."""
    with Path(path).open(encoding="utf-8") as handle:
        raw: Any = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Vocabulary must be a JSON object")
    return {int(token_id): token for token, token_id in raw.items()}


def load_model_vocabulary(model_id: str) -> dict[int, str]:
    """Load the token-to-ID mapping from the SDK model vocabulary file."""
    model = Small_LLM_Model(model_id)
    return load_vocabulary(model.get_path_to_vocab_file())


def main() -> int:
    """Print the vocabulary value for each requested token ID."""
    args = parse_args()
    try:
        vocabulary = (
            load_vocabulary(args.vocabulary)
            if args.vocabulary
            else load_model_vocabulary(args.model)
        )
        for token_id in args.token_ids:
            token = vocabulary.get(token_id)
            if token is None:
                print(f"{token_id}: <not found>")
            else:
                print(f"{token_id}: {token!r}")
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
