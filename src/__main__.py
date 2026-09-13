"""Command-line entry point for the function-calling project."""

import argparse
import json
from pathlib import Path

from client import TokenAwareConstrainedRouter, load_functions
from llm_sdk.llm_sdk import Small_LLM_Model


def parse_args() -> argparse.Namespace:
    """Parse command-line paths."""
    parser = argparse.ArgumentParser(description="Generate constrained function calls")
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument(
        "--output", default="data/output/function_calling_results.json"
    )
    return parser.parse_args()


def main() -> int:
    """Process prompts and write validated function calls."""
    args = parse_args()
    try:
        functions = load_functions(args.functions_definition)
        with open(args.input, encoding="utf-8") as handle:
            prompts = json.load(handle)
        if not isinstance(prompts, list):
            raise ValueError("Input prompts must be a JSON array")

        router = TokenAwareConstrainedRouter(Small_LLM_Model())
        results = []
        for item in prompts:
            prompt = item["prompt"]
            results.append(router.generate_call(prompt, functions).model_dump())

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        return 0
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
    ) as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())