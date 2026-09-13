# Function Calling System

This project has been created as part of the 42 curriculum by hhulic.

## Description

This program converts natural-language requests into validated function calls.
It uses a local Ollama model through its HTTP API and does not use PyTorch,
Hugging Face, Transformers, DSPy, or similar Python ML packages.

## Setup

Synchronize the project environment with `uv`:

```sh
make install
```

Install and start Ollama separately, then pull the model:

```sh
ollama pull qwen3:0.6b
```

Run the program:

```sh
make run
```

The default output is `data/output/function_calling_results.json`. Set
`OLLAMA_URL` or `CALL_ME_MAYBE_MODEL` to use a different server or model.

## Algorithm

The backend sends the function definitions and user request to Ollama and asks
for a JSON-schema-constrained response. The router then validates the selected
function, required parameter keys, and declared parameter types with Pydantic.
No model-generated prose is accepted in the output.

## Testing

Run:

```sh
uv run python -m py_compile client.py src/schema.py src/model_backend.py src/__main__.py
make lint
```

Test malformed input JSON, missing files, unknown function names, missing
parameters, and invalid parameter types as well as normal prompts.

## Resources And AI Use

The assignment specification, Python standard library documentation, Pydantic
documentation, and Ollama API documentation are the primary resources. AI was
used for code review, implementation suggestions, and test planning; generated
code must be understood and peer-reviewed before submission.
