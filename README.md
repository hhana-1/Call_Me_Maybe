_This project has been created as part of the 42 curriculum by hhulic._

# call me maybe: function calling with a small LLM

# Description

The program turns a natural-language request into a function call. Given a
request and a JSON file of function definitions, it writes the function to
call and its arguments with the right types: "What is the sum of 2 and 3?"
becomes `fn_add_numbers` with `{"a": 2, "b": 3}`.

A small model (`Qwen/Qwen3-0.6B`) writes the whole call with **constrained
decoding**: at every step the code decides which tokens are allowed, and the
model only chooses among them, so every output is valid JSON that matches the
schema. The model is reached only through the public methods of `llm_sdk`.

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
| `--ranker` | `make run-ranker` | shortlist the functions with a keyword ranker first (off by default) |
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
  "started_at": "02:41:29",
  "finished_at": "02:41:31",
  "time passed": 0.047
}
...
Wrote 22 call(s) to data/output/function_calling_results.json in 111.42 seconds.
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

**4. The arguments.** A second prompt shows only the chosen function with its
parameter types. The code writes the keys and punctuation; the model writes
each value, limited by its type:

| Type | Allowed |
| ---- | ------- |
| `number`, `integer` | one of the numbers in the request; written digit by digit if there is none |
| `boolean` | ` true` or ` false` |
| `string` | a piece of the request (copied, so only its end is a real choice), or free text where only tokens that keep the JSON valid are allowed |

The result goes into a Pydantic `FunctionCall` and is written with `json.dump`.

## Design decisions

- **The model chooses, not keywords.** The keyword ranker (`--ranker`) is a bonus part I added for fun ad a flag, not a core of the program, since it performs really well in small controlled environment
- **The model writes the arguments.** I first did it with regex and it was superfast
  (29 s) but got only 17 of 22 calls right. When the model chooses like now, it gets 21 of 22.
- **Fewer model calls.** Values are copied from the request, so most tokens
  are fast-forwarded (163 s became about 112 s), and both prompts are short:
  no parameters in the name prompt, one function in the argument prompt.
- **The SDK's tokenizer**, through `encode` and `decode`; `tokenizer.py` only
  adapts it (lists instead of tensors, all tokens, the chat markers).
- **Code split:** `client.py` chooses the name, `arguments.py` writes the
  arguments, `tokenizer.py` holds the tokenizer and `chat()`.

## Performance analysis

Measured on the cluster CPU with `Qwen/Qwen3-0.6B`, on the 22 sample prompts:

| Measure | Result |
| ------- | ------ |
| right function | 22/22 |
| fully right call | **21/22** (the miss: regex `\d+\s` instead of `\d+`) |
| valid JSON | 22/22 |
| time | about 112 s for 132 model calls, plus about 34 s start-up |

- **Speed:** a model call costs about 0.3 s plus 5 ms per prompt token, since
  the SDK keeps no cache. "Give me 10 plus 20" needs 3 calls for 15 tokens.
- **Reliability:** the mask makes every output valid, not always correct. The
  model invents values the request lacks ("sum of 7" gives 7 and 7), and a
  number word next to digits fails ("Add ten and 5" gives 5 and 5).

## Challenges faced

- **Time without a cache.** Every call rereads the whole prompt. Solved with
  fast-forwarding, copying from the request and shorter prompts.
- **Tokens, not characters.** When the code wrote the opening quote, the model
  dropped the `/` of a path and the `-` of `-4`. Letting the model choose the
  token (` "/`, ` -`) fixed both.
- **JSON strings.** A free string must never break the JSON: tokens with `"`
  or `\` are checked at every step, all others are sorted out once.
- **Integers.** "4.5" for an integer once became 45. Now an integer is chosen
  among the whole numbers, and a decimal is refused.

## Testing strategy

- A scoring script compares the 22 sample outputs with the expected calls.
- Edge cases: negative and decimal numbers, accents, empty strings, Windows
  paths, missing values, a function file the code has never seen.
- Errors: missing or invalid JSON files, items without `"prompt"`,
  unsupported parameter types, an unknown model name.
- `--trace` to check every step by hand; `make lint` and `make lint-strict`.

# Resources

- Andrej Karpathy's channel https://www.youtube.com/@AndrejKarpathy/videos
- Willard, Louf, *Efficient Guided Generation for Large Language Models*,
  2023: <https://arxiv.org/abs/2307.09702>
- Qwen3-0.6B model card: <https://huggingface.co/Qwen/Qwen3-0.6B>
- RFC 8259, *The JSON Data Interchange Format*: <https://www.rfc-editor.org/rfc/rfc8259>
- Pydantic documentation: <https://docs.pydantic.dev>

**AI use.** Claude was used to explain LLM concepts (tokens, logits,
masking), to help measure constrained decoding, fix flake8 and mypy errors, and to draft this README