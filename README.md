_This project has been created as part of the 42 curriculum by hhulic._

# Function Calling System

## Description

This project converts natural-language requests into structured function calls.
It uses the provided `llm_sdk` package to select one function from a JSON
definition file and extract its arguments. The model defaults to
`Qwen/Qwen3-0.6B` but can be swapped for any of several free local models (see
[Model options](#model-options)).

The generated output contains exactly these fields:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2, "b": 3}
}
```

## Setup

The project uses `uv` for dependency management and execution.

```sh
uv sync
```

The SDK downloads `Qwen/Qwen3-0.6B` from Hugging Face on its first run. Make
sure the machine has enough disk space for the model and its Python packages.

## Usage

Run the default input set:

```sh
make run
```

Equivalent command:

```sh
uv run python -m src
```

The default files are:

- Function definitions: `data/input/functions_definition.json`
- Prompts: `data/input/function_calling_tests.json`
- Results: `data/output/function_calling_results.json`

Custom paths can be supplied with:

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Inspect the result with:

```sh
python3 -m json.tool data/output/function_calling_results.json
```

## Model options

The backend model is selected with `--model`, which accepts either a built-in
alias or any ungated Hugging Face causal-LM repo id (downloaded on first use):

```sh
uv run python -m src --model smollm2-360m
# or via make:
make run-model MODEL=smollm2-360m
```

Built-in aliases:

| Alias            | Hugging Face repo                         |
| ---------------- | ----------------------------------------- |
| `qwen3-0.6b`     | `Qwen/Qwen3-0.6B` (default)               |
| `qwen2.5-0.5b`   | `Qwen/Qwen2.5-0.5B-Instruct`              |
| `smollm2-360m`   | `HuggingFaceTB/SmolLM2-360M-Instruct`     |
| `smollm2-135m`   | `HuggingFaceTB/SmolLM2-135M-Instruct`     |
| `tinyllama-1.1b` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0`      |

All aliases are ungated, so no Hugging Face login or token is required. A full
repo id such as `--model org/model` is also accepted, which lets you use gated
models whose license you have already accepted. Every model runs through the
same logit-routing path, so results stay directly comparable; the smallest
models (135M/360M) are fastest on CPU but may route less accurately.

## Algorithm

`choose_tokenizer.py` loads the live JSON definitions and test prompts. It
tokenizes prompts and schema descriptions, protects quoted strings and numeric
values, and ranks candidate schemas using meaningful token overlap and declared
parameter types.

`client.py` uses that shortlist to build the model context. It calls the SDK's
public `encode` and `get_logits_from_input_ids` methods, restricts selection to
the candidate functions, and validates the selected function and argument types
with Pydantic before writing JSON.

The provided `llm_sdk` directory is treated as an external project component and
is not modified by this project.

## Generation Visualization

Pass `--trace` (or run `make trace`) to visualize the generation process for
every prompt step by step instead of printing the timing JSON:

```sh
uv run python -m src --trace
```

For each prompt the trace shows:

1. **Tokenization** — the prompt split into model tokens.
2. **Candidate functions** — the schemas kept after shortlisting.
3. **Model routing** — the restricted next-token logits for each candidate's
   description token, the softmax probabilities derived from them, and a bar
   chart, with the selected function marked.
4. **Selected function** — the winning function name.
5. **Extracted arguments** — the final arguments and their Python types.

This makes the constrained routing decision (logits → probabilities →
selection) directly observable. The rendering lives in `src/visualize.py`; the
router records the per-step trace in `TokenAwareConstrainedRouter.last_trace`
only when `trace_enabled` is set, so normal runs pay no overhead.

## Project Structure

```text
client.py                         Routing and argument validation
choose_tokenizer.py               JSON loading and tokenizer-based shortlisting
src/__main__.py                   Command-line entry point
src/schema.py                     Pydantic models
src/visualize.py                  Generation-process trace rendering (--trace)
llm_sdk/                          Provided Qwen SDK
data/input/                       Function definitions and test prompts
data/output/                      Generated results, not submitted
```

## Testing

Compile the project:

```sh
uv run python -m py_compile \
  client.py choose_tokenizer.py src/schema.py src/__main__.py
```

Run static checks:

```sh
make lint
```

The sample input covers addition, greetings, string reversal, square roots,
regular-expression replacement, quoted strings, multiple parameters, and
numbers embedded inside quoted text.

## Make Targets

- `make install`: synchronize dependencies with `uv`
- `make run`: run the function-calling pipeline
- `make debug`: run the pipeline under `pdb`
- `make lint`: run `flake8` and `mypy`
- `make clean`: remove Python caches

## Resources And AI Use

Resources include the project specification, Python documentation, Pydantic
documentation, Hugging Face model documentation, and the provided `llm_sdk`
source.

AI was used for code review, implementation suggestions, debugging, and test
planning. All generated code must be understood, tested, and peer-reviewed
before submission.



Load function definitions and prompts.
|
v
Rank candidate functions.
|
v
Select a function using token logits.
|
v
Extract arguments.
|
v
Validate the result.
|
v
Write JSON output.


Tokenizes the prompt.
Finds vocabulary terms by domain.
Tokenizes each function name and description.
Finds vocabulary terms in each function.
Gives points for shared domain terms.
Adds a smaller bonus when both belong to the same domain.
Keeps every JSON-defined function available.
```bash
Prompt:
Substitute the word 'cat' with 'dog' ...

Prompt vocabulary:
prose -> substitute, word

Substitution function vocabulary:
prose -> substitute, word, string, replace
```