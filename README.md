_This project has been created as part of the 42 curriculum by hhulic._

# call me maybe: function calling with a small LLM

# Description

The program turns a natural-language request into a function call. Given a
request and a JSON file of function definitions, it writes the function to
call and its arguments with the right types: "What is the sum of 2 and 3?"
becomes `fn_add_numbers` with `{"a": 2, "b": 3}`.

A small model (`Qwen/Qwen3-0.6B`) chooses the function with **constrained
decoding**: at every step the code decides which tokens are allowed, and the
model only chooses among them, so it can only write a real function name.
Regular expressions then read the arguments from the request, by the types in
the function definition. Every output is valid JSON that matches the schema.
The model is reached only through the public methods of `llm_sdk`.

# Instructions

```sh
make install     # uv sync; the model (about 1.5 GB) is downloaded on the first run
make run         # uv run python -m src
make lint        # flake8 and mypy (make lint-strict: mypy --strict)
```

Files are read from `data/input/` and written to `data/output/` by default;
change them with `--functions_definition`, `--input` and `--output`.

| Option | Make target | Effect |
| ------ | ----------- | ------ |
| `--trace` | `make trace` | show every decoding step |
| `--ranker` | `make run-ranker` | shortlist the functions with a keyword ranker first; when it keeps one, the model is not called (off by default) |
| `--model <alias>` | `make run-model MODEL=<alias>` | `qwen3-0.6b` (default), `qwen2.5-0.5b`, `smollm2-360m`, `smollm2-135m`, or a repo id |

## Example usage

```sh
$ uv run python -m src --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
Using model: Qwen/Qwen3-0.6B
{
  "response": {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2,
      "b": 3
    }
  },
  "started_at": "01:29:11",
  "finished_at": "01:29:12",
  "time passed": 0.017
}
...
Wrote 11 call(s) to data/output/function_calling_results.json in 10.23 seconds.
```

A prompt that cannot be processed is reported and skipped, and the results
file is still written: `Skipped prompt 5 ('Sort [3, 1, 2]'): Unsupported parameter type: array`.

## Algorithm explanation

**1. The prompt.** `chat()` writes the request in Qwen's chat format. The
system part lists every function as `- name: description`, and the program
starts the answer with `{"name": "` (prefilling).

**2. One token at a time.** `_pick` asks the model for its scores (logits, one
per token of the 151,936-token vocabulary) and masks them:

| Step | Code | Result |
| ---- | ---- | ------ |
| scores | `get_logits_from_input_ids(context)` | 151,936 numbers |
| mask | `np.full(shape, -inf)`, then `masked[ids] = logits[ids]` | only allowed tokens keep a score |
| choose | `np.argmax(masked)` | the best allowed token |

If only one token is allowed, it is **fast-forwarded**: added without calling
the model, since the result could not be different.

**3. The function name.** Every name, with its closing quote, is encoded into
tokens (`fn` `_add` `_numbers` `"`), and together they form a trie. `_follow`
allows only the next tokens of the names that still match. All names start
with `fn`, so the model makes one choice (`_add`, `_g`...) per name.

**4. The arguments.** `regex_arguments.py` reads each value from the request
with regular expressions, by the parameter's type, with no model call:

| Type | Read as |
| ---- | ------- |
| `number`, `integer` | the numbers of the request, in order (an integer must be whole) |
| `boolean` | a `true` / `false` in the request, or the answer to an even/odd question |
| `string` | a pattern for some parameter names (`regex`, `replacement`, `source_string`, `path`, `encoding`), else the quoted texts in order, else the last word |

The result goes into a Pydantic `FunctionCall` and is written with `json.dump`.

## Design decisions

- **The model chooses, not keywords.** The keyword ranker (`--ranker`) is a bonus part I added for fun as a flag, not a core of the program, since it performs really well in small controlled environment
- **Regular expressions read the arguments.** Letting the model write them
  with constrained decoding also got 10 of 11 calls right, but needed 57
  model calls (about 43 s); the regular expressions need none (about 10 s
  in all).
- **Short prompt.** The function list leaves out the parameters, so every
  model call is cheaper.
- **The SDK's tokenizer**, through `encode` and `decode`; `tokenizer.py` only
  adapts it (lists instead of tensors, the chat markers).
- **Code split:** `client.py` chooses the name, `regex_arguments.py` reads
  the arguments, `tokenizer.py` holds the tokenizer and `chat()`.

## Performance analysis

Measured on the cluster CPU with `Qwen/Qwen3-0.6B`, on the 11 sample prompts
and 5 functions:

| Measure | `make run` | `make run-ranker` |
| ------- | ---------- | ----------------- |
| right function | 11/11 | 11/11 |
| fully right call | **10/11** | **10/11** |
| valid JSON | 11/11 | 11/11 |
| model calls | 11 | 0 |
| time | about 10 s | 0.02 s |

Both modes give the same 11 calls; the miss is the replacement `asterisks`
instead of `*`. The ranker keeps 1 of the 5 functions for every prompt, so
`make run-ranker` needs no model call. Start-up adds about 34 s in both modes.

On the earlier set of 22 prompts and 12 functions the regular expressions get
17/22 right (77%): they read `database` instead of `production` / `system`,
`asterisks` instead of `*`, and only one word of the two templates. The
rules are written for these kinds of requests, so new wordings fail more
easily than the function choice.

- **Speed:** a model call costs about 0.3 s plus 5 ms per prompt token, since
  the SDK keeps no cache. Each prompt needs one call, for the function name.
- **Reliability:** the function name is always a real one. A missing value
  stops only that prompt: "What is the sum of 7?" is skipped with
  `Missing number argument: b`.

## Challenges faced

- **Deciding what is constrained decoding and what is heuristic** Sometimes the fastest executing solution wasn't relying on LLM choice (even though it was correct). Fixed that by adding it as a bonus flag so program doesn't really rely on it but it is there to strike a discussion amongst peers. 

## Testing strategy

- Scoring script
- Trial and error

# Resources

- Andrej Karpathy's channel https://www.youtube.com/@AndrejKarpathy/videos
- Willard, Louf, *Efficient Guided Generation for Large Language Models*,
  2023: <https://arxiv.org/abs/2307.09702>
- Qwen3-0.6B model card: <https://huggingface.co/Qwen/Qwen3-0.6B>
- RFC 8259, *The JSON Data Interchange Format*: <https://www.rfc-editor.org/rfc/rfc8259>
- Pydantic documentation: <https://docs.pydantic.dev>

**AI use.** Claude was used to explain LLM concepts (tokens, logits,
masking), to measure performance, to measure constrained decoding, fix flake8 and mypy errors, help with regex, and to draft this README