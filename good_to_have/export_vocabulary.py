"""Export the current SDK model tokenizer vocabulary to JSON."""

import argparse
import json
from pathlib import Path
from typing import Any

from llm_sdk.llm_sdk import Small_LLM_Model


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data/output/vocabulary.json"


def parse_args() -> argparse.Namespace:
    """Parse model and output paths."""
    parser = argparse.ArgumentParser(description="Export the model vocabulary")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Hugging Face model ID used by the SDK",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="JSON file to create",
    )
    return parser.parse_args()


def export_vocabulary(model_id: str, output_path: str) -> int:
    """Export the model tokenizer's token-to-ID mapping."""
    print(f"Loading tokenizer for {model_id}...")
    model = Small_LLM_Model(model_id)
    tokenizer_path = Path(model.get_path_to_tokenizer_file())
    with tokenizer_path.open(encoding="utf-8") as handle:
        tokenizer_data: dict[str, Any] = json.load(handle)

    vocabulary = tokenizer_data.get("model", {}).get("vocab")
    if not isinstance(vocabulary, dict):
        raise ValueError("Tokenizer file does not contain a model vocabulary")

    sorted_vocabulary = {
        token: token_id
        for token, token_id in sorted(
            vocabulary.items(), key=lambda item: int(item[1])
        )
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(sorted_vocabulary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Saved {len(sorted_vocabulary)} tokens to {destination}")
    return 0


if __name__ == "__main__":
    arguments = parse_args()
    try:
        raise SystemExit(export_vocabulary(arguments.model, arguments.output))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"Error: {error}")
        raise SystemExit(1)
