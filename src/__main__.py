"""Command-line entry point for the function-calling project."""

import argparse
import json
import time
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
        total_started = time.perf_counter()
        for item in prompts:
            prompt = item["prompt"]
            response_started = time.perf_counter()
            response_start_time = time.strftime("%H:%M:%S")
            result = router.generate_call(prompt, functions).model_dump()
            response_finished = time.perf_counter()
            response_finish_time = time.strftime("%H:%M:%S")
            response_elapsed_minutes = (response_finished - response_started) / 60
            print(
                json.dumps(
                    {
                        "response": result,
                        "started_at": response_start_time,
                        "finished_at": response_finish_time,
                        "elapsed_minutes": round(response_elapsed_minutes, 3),
                    },
                    indent=2,
                ),
                flush=True,
            )
            results.append(result)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        total_finished = time.perf_counter()
        total_elapsed_seconds = total_finished - total_started
        print(
            f"Total time for {len(results)} response(s): "
            f"{total_elapsed_seconds:.2f} seconds",
            flush=True,
        )
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