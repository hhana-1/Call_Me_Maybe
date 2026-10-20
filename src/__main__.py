"""Command-line entry point for the function-calling project."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

from client import TokenAwareConstrainedRouter, load_functions
from llm_sdk.llm_sdk import Small_LLM_Model
from src.models import DEFAULT_MODEL, MODEL_PRESETS, resolve_model
from src.tokenizer import BPETokenizer
from src.visualize import render_generation


def parse_args() -> argparse.Namespace:
    """Parse command-line paths."""
    parser = argparse.ArgumentParser(
        description="Generate constrained function calls"
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input", default="data/input/function_calling_tests.json"
    )
    parser.add_argument(
        "--output", default="data/output/function_calling_results.json"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Local HF model alias or repo id. Aliases: "
            + ", ".join(sorted(MODEL_PRESETS)) + "."
        ),
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Visualize each prompt's generation pipeline step by step.",
    )
    return parser.parse_args()


def load_prompts(path: str) -> list[Any]:
    """Load the prompt file, which must be a JSON array."""
    with open(path, encoding="utf-8") as handle:
        prompts = json.load(handle)
    if not isinstance(prompts, list):
        raise ValueError("Input prompts must be a JSON array")
    return prompts


def write_results(path: str, results: list[dict[str, Any]]) -> None:
    """Write the results file, creating its folder if needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def process_prompts(
    router: TokenAwareConstrainedRouter,
    prompts: list[Any],
    functions: list[Any],
    trace: bool,
) -> tuple[list[dict[str, Any]], int]:
    """Generate a call for every prompt, skipping the ones that fail.

    One bad item never stops the run: it is reported and the next prompt is
    processed. Ctrl+C stops early but keeps the results produced so far.

    Returns:
        The results, and how many prompts were skipped.
    """
    results: list[dict[str, Any]] = []
    skipped = 0
    for number, item in enumerate(prompts, start=1):
        prompt = item.get("prompt") if isinstance(item, dict) else None
        if not isinstance(prompt, str):
            print(f"Skipped item {number}: expected {{\"prompt\": \"...\"}}")
            skipped += 1
            continue
        started = time.perf_counter()
        started_at = time.strftime("%H:%M:%S")
        try:
            result = router.generate_call(prompt, functions).model_dump()
        except KeyboardInterrupt:
            print("Interrupted: writing the results produced so far.")
            return results, skipped + len(prompts) - number + 1
        except Exception as error:  # recover and continue with the rest
            print(f"Skipped prompt {number} ({prompt!r}): {error}")
            skipped += 1
            continue
        if trace and router.last_trace is not None:
            print(render_generation(router.last_trace, result["parameters"]))
        else:
            elapsed_minutes = (time.perf_counter() - started) / 60
            print(json.dumps({
                "response": result,
                "started_at": started_at,
                "finished_at": time.strftime("%H:%M:%S"),
                "time passed": round(elapsed_minutes, 3),
            }, indent=2))
        results.append(result)
    return results, skipped


def main() -> int:
    """Process prompts and write validated function calls."""
    args = parse_args()
    try:
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)
        model_name = resolve_model(args.model)
        print(f"Using model: {model_name}", flush=True)
        model = Small_LLM_Model(model_name=model_name)
        router = TokenAwareConstrainedRouter(
            llm=model,
            tokenizer=BPETokenizer.from_model(model),
            trace_enabled=args.trace,
        )
        started = time.perf_counter()
        results, skipped = process_prompts(
            router, prompts, functions, args.trace
        )
        write_results(args.output, results)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(f"Error: {error}")
        return 1
    except KeyboardInterrupt:
        print("Interrupted before any prompt was processed.")
        return 130
    print(
        f"Wrote {len(results)} call(s) to {args.output} "
        f"in {time.perf_counter() - started:.2f} seconds"
        + (f"; skipped {skipped}." if skipped else ".")
    )
    return 1 if skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
