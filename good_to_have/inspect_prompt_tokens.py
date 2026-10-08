"""Display tokenization and detokenization for prompts from the input JSON."""

import argparse
import json
from pathlib import Path
from typing import List

from pydantic import BaseModel, RootModel

DEFAULT_INPUT = "../data/input/function_calling_tests.json"


# Define the schema for a single prompt object
class PromptItem(BaseModel):
    prompt: str


# Define the schema for the entire JSON file array
class PromptList(RootModel[List[PromptItem]]):
    pass


def parse_args() -> argparse.Namespace:
    """Parse prompt selection arguments."""
    parser = argparse.ArgumentParser(description="Inspect prompt tokens")
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Optional: specific prompt index to display (starts at 0)",
    )
    parser.add_argument(
        "--input", default=DEFAULT_INPUT, help="Prompt JSON file"
    )
    return parser.parse_args()


def load_all_prompts(path: str) -> List[str]:
    """Load and validate all prompts from the JSON input using Pydantic."""
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    # Pydantic validates the structure automatically.
    validated_data = PromptList.model_validate(data)
    # Extract just the raw strings out of the validated models.
    return [item.prompt for item in validated_data.root]


def main() -> int:
    """Encode, display, and decode prompts."""
    args = parse_args()
    try:
        from llm_sdk.llm_sdk import Small_LLM_Model

        all_prompts = load_all_prompts(args.input)
        model = Small_LLM_Model()

        if args.index is not None:
            if 0 <= args.index < len(all_prompts):
                prompts_to_process = [(args.index, all_prompts[args.index])]
            else:
                raise ValueError(
                    f"No prompt found at index {args.index}. "
                    f"Total items available: {len(all_prompts)}"
                )
        else:
            prompts_to_process = list(enumerate(all_prompts))

        for _, prompt in prompts_to_process:
            encoded = model.encode(prompt)
            # Flatten the tensor to 1D before list conversion to avoid
            # the double-bracket bug.
            token_ids = encoded.detach().cpu().flatten().tolist()
            decoded = model.decode(token_ids)
            tokens = [model.decode([token_id]) for token_id in token_ids]

            print(f"Original: {prompt}")
            print(f"Token IDs: {token_ids}")
            print(f"Tokens: {tokens}")
            print(f"Decoded: {decoded}")
            print()

        return 0
    except Exception as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
