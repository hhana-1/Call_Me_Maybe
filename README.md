_This project has been created as part of the 42 curriculum by hhulic._

# Function Calling System

## Description

This project converts natural-language requests into structured function calls.
It uses the provided `llm_sdk` package and the `Qwen/Qwen3-0.6B` model to select
one function from a JSON definition file and extract its arguments.

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

## Project Structure

```text
client.py                         Routing and argument validation
choose_tokenizer.py               JSON loading and tokenizer-based shortlisting
src/__main__.py                   Command-line entry point
src/schema.py                     Pydantic models
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