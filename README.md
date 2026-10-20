_This project has been created as part of the 42 curriculum by hhulic._

# Function Calling System

## Description

This project turns natural-language requests into structured function calls.
Given a request such as "What is the sum of 2 and 3?" and a JSON file that
defines the available functions, it outputs the function to call and its
arguments with the correct types:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

A small language model (`Qwen/Qwen3-0.6B` by default) writes every call. It
is driven by **constrained decoding**: at each step only the tokens that keep
the output valid JSON and true to the function's schema are allowed, so every
call is parseable and has the right argument types, even from a 0.6B model.
The model is reached only through the provided `llm_sdk`, and text is encoded
by the project's own tokenizer.

## Instructions

### Setup

The project uses `uv`:

```sh
uv sync          # or: make install
```

The SDK downloads the model (about 1.5 GB) from Hugging Face on the first run
and caches it in `~/.cache/huggingface`. Set `HF_HOME` to choose another
location.

### Usage

```sh
make run         # same as: uv run python -m src
```

By default the program reads `data/input/functions_definition.json` and
`data/input/function_calling_tests.json` and writes
`data/output/function_calling_results.json`. Every path can be changed:

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Options:

- `--trace` shows each prompt's decoding step by step (see
  [Generation visualization](#generation-visualization)). Also `make trace`.
- `--model <alias or repo id>` selects another model (see below).

### Example usage

```sh
$ uv run python -m src --input my_prompts.json --output my_results.json
Using model: Qwen/Qwen3-0.6B
{
  "response": {
    "prompt": "Read the file at /home/user/data.json with utf-8 encoding",
    "name": "fn_read_file",
    "parameters": {
      "path": "/home/user/data.json",
      "encoding": "utf-8"
    }
  },
  "started_at": "19:49:40",
  "finished_at": "19:49:52",
  "time passed": 0.206
}
{
  "response": {
    "prompt": "Greet john",
    "name": "fn_greet",
    "parameters": {
      "name": "john"
    }
  },
  "started_at": "19:49:52",
  "finished_at": "19:49:56",
  "time passed": 0.065
}
Wrote 2 call(s) to my_results.json in 16.23 seconds.
```

A prompt that cannot be processed is reported and skipped; the rest of the
file is still processed and the results file is always written:

```text
Skipped prompt 2 (''): Unsupported parameter type 'array' for 'items'
Skipped item 3: expected {"prompt": "..."}
Skipped item 4: expected {"prompt": "..."}
Skipped prompt 5 ('Sort [3, 1, 2]'): Unsupported parameter type 'array' for 'items'
Wrote 2 call(s) to messy_out.json in 8.34 seconds; skipped 4.
```

The exit code is 0 when every prompt produced a call and 1 otherwise.

### Choosing a model

```sh
uv run python -m src --model smollm2-360m
make run-model MODEL=smollm2-360m
```

| Alias            | Hugging Face repo                     |
| ---------------- | ------------------------------------- |
| `qwen3-0.6b`     | `Qwen/Qwen3-0.6B` (default)           |
| `qwen2.5-0.5b`   | `Qwen/Qwen2.5-0.5B-Instruct`          |
| `smollm2-360m`   | `HuggingFaceTB/SmolLM2-360M-Instruct` |
| `smollm2-135m`   | `HuggingFaceTB/SmolLM2-135M-Instruct` |

A full repo id (`--model org/model`) is also accepted. The model must ship a
byte-level BPE tokenizer (`vocab.json` + `merges.txt`); SentencePiece models
such as TinyLlama are rejected with a clear error.

### Make targets

| Target              | What it does                                  |
| ------------------- | --------------------------------------------- |
| `make install`      | install dependencies with `uv sync`           |
| `make run`          | run the pipeline on the default files         |
| `make trace`        | run with the step-by-step visualization       |
| `make run-model`    | run with `MODEL=<alias>`                      |
| `make debug`        | run under `pdb`                               |
| `make lint`         | `flake8 .` and `mypy` with the required flags |
| `make lint-strict`  | `flake8 .` and `mypy . --strict`              |
| `make clean`        | remove `__pycache__` and `.mypy_cache`        |
| `make export-vocabulary` | save the model's vocabulary to `data/output/` |

## Algorithm

### Constrained decoding

The model writes each call as JSON, one token at a time:

```text
{"name": "fn_add_numbers", "arguments": {"a": -5, "b": 3.5}}
```

At every step the decoder builds the list of **allowed** tokens. It then takes
the model's logits, sets every other token to minus infinity, and picks the
highest remaining one. If exactly one token is allowed, it is appended
**without calling the model**. Text that the schema fixes (`", "b":`, `}}`) is
appended the same way.

The token masks are built once at start-up, by decoding all 151,643 tokens of
the vocabulary with the project's tokenizer:

| Mask                  | Tokens                                              |
| --------------------- | --------------------------------------------------- |
| digits, `.` and `-`   | 12 single-character tokens                          |
| plain string content  | 145,442 tokens without quotes or backslashes        |
| quote and escape      | 1,226 tokens containing `"` or `\`, checked one by one |
| string openings       | 131 tokens that start with ` "` (` "/`, ` "[`, ...) |

### Stage 1: the function name

The prompt lists every function as `name(param: type, ...): description` and
ends with `{"name": "`. Each name, with its closing quote, is encoded into a
token sequence, and together they form a trie: at each step only the next
tokens of names that still match are allowed. The closing quote makes every
sequence end differently, even when one name starts another. For the sample functions every name starts with `fn` (forced), the
model makes **one** choice between 12 tokens (`_add`, `_g`, `_reverse`, ...),
and the rest of the name is forced again. The LLM therefore chooses the
function from the full list; no keyword matching is involved.

### Stage 2: the arguments

A second, shorter prompt shows only the chosen function and ends with
`{"name": "fn_...", "arguments": {"<first key>":`. Each value is decoded with
the grammar of its declared type:

- **number**: first ` ` or ` -`, then digits with at most one `.`, no
  leading zeros. Two regular expressions describe a number being written and
  a finished one; a digit or `.` is allowed only if the text still matches
  the first, and the token that closes the value (`,` or `}}`) only once it
  matches the second. Output as a float.
- **integer**: the same without `.`. Output as an int.
- **boolean**: ` true` or ` false`.
- **string**: an opening token starting with ` "`, then any content tokens.
  Backslash escapes are checked as they are written (`\"`, `\\`, `\n`, ...),
  raw control characters are forbidden, and a closing quote may be merged
  with the JSON that follows it (`",`, `"}}`). A string longer than 48
  tokens is closed.

Values are parsed with `json.loads`, then placed in a Pydantic `FunctionCall`
with exactly the keys `prompt`, `name` and `parameters`, and written with
`json.dump`. The output is therefore always valid JSON with the schema's
types.

### Prompt format

Prompts use the model's own chat format when its tokenizer has the special
tokens: `<|im_start|>` turns, an empty `<think>` block (Qwen3's "no thinking"
mode) and Qwen's `<tool_call>` marker. A model without them gets a plain-text
prompt instead.

## Tokenizer

`src/tokenizer.py` is a from-scratch byte-level BPE tokenizer with public
`encode` and `decode` methods. The main code never calls the SDK's `encode` or
`decode`: the SDK only supplies files, through `get_path_to_vocab_file()`,
`get_path_to_merges_file()` and `get_path_to_tokenizer_file()` (for the
special tokens). The model is driven only through `get_logits_from_input_ids()`.

```python
tokenizer = BPETokenizer.from_model(model)
tokenizer.encode("fn_add_numbers")      # [8822, 2891, 32964]: 'fn' '_add' '_numbers'
tokenizer.decode([8822, 2891, 32964])   # 'fn_add_numbers'
```

How `encode` works:

1. **Special tokens** such as `<|im_start|>` are cut out first and become
   their single ids.
2. **Normalize** the rest to Unicode NFC, as Qwen's tokenizer does, so a
   character written in decomposed form (`G` + a combining dot) becomes `Ġ`.
3. **Pre-tokenize** with Qwen's split pattern: letters, single digits,
   punctuation runs and whitespace become separate chunks. Python's `re` has
   no `\p{L}` / `\p{N}`, so the exact Unicode letter and number classes are
   built once from `unicodedata`.
4. **Map bytes to characters**: each chunk's UTF-8 bytes become printable
   characters, which is why a space appears as `Ġ` in the vocabulary.
5. **Merge**: adjacent symbols are merged repeatedly, always applying the
   highest-priority rule from `merges.txt`, until no rule applies.
6. **Look up** each piece in `vocab.json`. Results are cached per chunk.

`decode` reverses the vocabulary, maps the characters back to bytes and decodes
UTF-8. Special tokens are skipped, as the SDK does.

The tokenizer gives exactly the same ids as the SDK's `encode` and the same
text as its `decode` (see [Testing strategy](#testing-strategy)). Qwen2,
Qwen2.5 and Qwen3 share one vocabulary, merge list and split pattern, so it is
exact for all of them. SmolLM2 loads the same way but splits text with GPT-2's
pattern and does not normalize, so its tokens can differ when two or more
spaces come right before a digit or for decomposed Unicode; the tokens are
still valid and decode to the same text.

## Generation visualization

`--trace` (or `make trace`) prints each prompt's decoding instead of the timing
JSON:

```text
PROMPT: Read the file at /home/user/data.json with utf-8 encoding
1. Tokenization  (14 tokens)
   ['Read', ' the', ' file', ' at', ' /', 'home', '/user', '/data', ...]

2. Function name  (only the 12 defined names can be written)
   'fn'           forced
   12 allowed tokens, one model call:
   ► '_read'        fn_read_file                   p=100.0%  ████████████████████████
     '_get'         fn_get_square_root             p=  0.0%  ························
     ... 10 more
   '_file'        forced
   '"'            forced

3. Arguments  (every token must fit the parameter's type; [x] = forced)
   path: ' "/' 100% of 120  'home' 100% of 145,766  '/user' 100% of
         145,766  '/data' 100% of 145,766  '.json' 100% of 145,766  '",'
         100% of 145,766  [' "']  ['encoding']  ['":']
   encoding: ' "' 100% of 112  'utf' 100% of 145,725  ...  '"}}' 83% of 145,725

4. Result
   fn_read_file {"path": "/home/user/data.json", "encoding": "utf-8"}
   12 model calls, 5 forced tokens
```

It shows how encoding and decoding work together with constrained decoding.
`encode` builds the name trie and the forced JSON structure. `decode` turns
every vocabulary id into text, which the masks are built from, and labels each
token in the trace. Each line shows how many tokens were allowed and the
probability of the chosen one. The trace is recorded only when `--trace` is
given.

## Design decisions

- **The model chooses, the grammar constrains.** An earlier version ranked
  functions by keyword overlap and extracted arguments with regular
  expressions. It reached 77% on the sample but failed on reworded prompts and
  on unseen function files, and it contradicted the subject. Both parts were
  replaced by constrained decoding.
- **Two prompts instead of one.** The SDK recomputes the whole prompt for
  every token (about 0.3 s plus 6 ms per prompt token on our CPU). Listing
  all functions costs about 320 tokens but is needed only for the name, so
  the arguments use a 96-token prompt with just the chosen function.
- **Skip the model whenever possible.** Forced tokens and fixed JSON
  structure never call the model. A typical call needs 3 to 25 model calls.
- **Respect the tokenizer's boundaries.** Qwen merges a quote with what
  follows it (` "/`, ` "[`, ` "\\`) and a space with a minus sign (` -`).
  Forcing a lone ` "` made the model drop the `/` of `/home/...` and the
  brackets of `[aeiou]`, so openings and signs are model choices.
- **The model's own chat format.** Measured on the 22 sample prompts, the
  chat format chose the right function 22 times out of 22, against 20 for a
  plain-text prompt. Qwen's full JSON tool listing was better on reworded
  prompts but three times slower, which would break the 5-minute limit.
- **Never stop the whole run.** Input files are validated up front; each
  prompt is then processed on its own, so a malformed item or an unsupported
  parameter type skips only that prompt. Ctrl+C writes the results so far.
- **Pydantic for every class**, `numpy` for the logit masks, and no use of
  private SDK attributes.

## Performance analysis

Measured on the cluster CPU with `Qwen/Qwen3-0.6B`:

| Set                                            | Function  | Full call |
| ---------------------------------------------- | --------- | --------- |
| Sample prompts (22)                            | 22/22     | 20/22     |
| Reworded prompts, edge cases, unseen functions | 21/22     | 21/22     |

- On the sample, the two calls counted as wrong are a vowel pattern written
  as `([aeiouAEIOU])` instead of `[aeiou]` (it matches the same vowels) and
  a template that lost its inner quotes (`Say hello to {name}`).
- In the second set, the only miss is "Is the statement 'the sky is green'
  true?", sent to `fn_is_even` instead of `fn_extract_boolean`.
- **Speed:** 190 to 216 s for the 22 sample prompts (225 model calls),
  depending on the run. Numbers cost one call per digit and strings one call
  per token, so a call takes from 3 s (`Is 7 an even number?`) to 25 s
  (compound interest with three long numbers). Start-up adds about 36 s: 24 s to import torch and
  transformers, 9 s to load the model and 3.6 s to build the tokenizer and
  masks.
- **Reliability:** every output is valid JSON with the schema's types, since
  the grammar never allows anything else. No input crashed the program:
  empty prompts, a 30-digit number, unknown function files, malformed items,
  missing or invalid files and unsupported types all end with a result or a
  clear message.

## Challenges faced

- **A 0.6B model and a rule against heuristics.** Keyword matching made the
  sample look good but did not generalize. The fix was to let the model
  choose under strict constraints, and to measure prompt formats instead of
  guessing.
- **Time without a cache.** Each generated token reruns the whole prompt, so
  the design minimizes model calls: forced tokens, a short second prompt and
  one decision for the function name.
- **Token boundaries.** Values lost characters (`/home` became `home`, `-5`
  lost its sign) until the decoder allowed the merged tokens the model was
  trained on.
- **Re-implementing the tokenizer exactly.** Matching the SDK needed the
  Unicode letter and number classes, NFC normalization (found when two lines
  of the subject text tokenized differently) and special tokens.

## Testing strategy

- **Tokenizer vs SDK:** identical ids to the SDK's `encode` on 229 texts
  (prompts, function names and descriptions, code and documentation lines,
  CJK, emoji, decomposed accents, numerals, whitespace runs), and identical
  text to its `decode` for 3,000 random tokens. Also compared with the
  reference `tokenizer.json` for Qwen2, Qwen2.5, Qwen3 and SmolLM2.
- **Accuracy:** the 22 sample prompts with their expected calls, 18 reworded
  and edge-case prompts (empty prompt, missing numbers, negative and 30-digit
  numbers, accents, empty strings), and a separate function file the code
  has never seen (weather, temperature, email, word count).
- **Error handling:** missing and invalid JSON files, a prompt file with
  non-object items and missing keys, a function with an `array` parameter, a
  broken default function file next to a valid `--functions_definition`, and
  an unknown model name.
- **Static checks:** `make lint` and `make lint-strict` pass.
- **By hand:** `--trace` shows every decision, which is how the boundary
  problems above were found.

## Project structure

```text
src/__main__.py      Command line, per-prompt error recovery
client.py            Constrained decoder (TokenAwareConstrainedRouter)
src/tokenizer.py     Own byte-level BPE tokenizer (encode / decode)
src/schema.py        Pydantic models for definitions and calls
src/visualize.py     --trace rendering
src/models.py        Model aliases for --model
good_to_have/        Helper scripts: export the vocabulary, look up ids
llm_sdk/             Provided SDK (unmodified)
data/input/          Function definitions and test prompts
data/output/         Generated results (not submitted)
```

## Resources

- Sennrich, Haddow, Birch, *Neural Machine Translation of Rare Words with
  Subword Units* (BPE), 2016: <https://arxiv.org/abs/1508.07909>
- Radford et al., *Language Models are Unsupervised Multitask Learners*
  (byte-level BPE in GPT-2), 2019:
  <https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>
- Willard, Louf, *Efficient Guided Generation for Large Language Models*
  (constrained decoding), 2023: <https://arxiv.org/abs/2307.09702>
- Hugging Face Tokenizers documentation (normalizers, pre-tokenizers, BPE):
  <https://huggingface.co/docs/tokenizers>
- RFC 8259, *The JSON Data Interchange Format*:
  <https://www.rfc-editor.org/rfc/rfc8259>
- Pydantic documentation: <https://docs.pydantic.dev>
- The project subject and the provided `llm_sdk` source.

### AI use

AI (Claude) was used for:

- the byte-level BPE tokenizer and its comparison with the SDK,
- the constrained decoder (name trie, type grammars, chat prompts) and the
  measurements that shaped it,
- the `--trace` visualization, per-prompt error recovery and lint fixes,
- reviewing the project against the subject, and drafting this README.

All of it was reviewed, run and tested before submission.
