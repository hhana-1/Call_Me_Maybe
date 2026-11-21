_This project has been created as part of the 42 curriculum by hhulic._

# Function Calling System

## Description


The model sees:

```bash
<|im_start|>system
You call functions. Pick the one function ...
<|im_end|>
<|im_start|>user
What is the sum of 2 and 3?
<|im_end|>
```


This project turns natural-language requests into structured function calls.
Given a request such as "What is the sum of 2 and 3?" and a JSON file that
defines the available functions, it outputs the function to call and its
arguments with the correct types:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2, "b": 3}
}
```

A small language model (`Qwen/Qwen3-0.6B` by default) chooses the function
with **constrained decoding**: it writes the function name one token at a
time, and only tokens that continue a real function name are allowed. An
optional keyword ranker (`--ranker`, off by default) can first shortlist the
functions whose words and domain best match the request, so that the model
chooses among fewer. The arguments are read from the request with regular
expressions, according to each parameter's type. The model is reached only
through the provided `llm_sdk`, and text is encoded by the project's own
tokenizer.

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

- `--trace` shows how each function was chosen, step by step (see
  [Generation visualization](#generation-visualization)). Also `make trace`.
- `--model <alias or repo id>` selects another model (see below).
- `--ranker` turns on the keyword ranker, which shortlists the functions
  before the model chooses (see
  [Shortlisting](#shortlisting-keyword-ranking-optional)). Without it, the
  model chooses among all the functions. Also `make run-ranker` and
  `make trace-ranker`.

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
Skipped prompt 2 (''): Unsupported parameter type: array
Skipped item 3: expected {"prompt": "..."}
Skipped item 4: expected {"prompt": "..."}
Skipped prompt 5 ('Sort [3, 1, 2]'): Unsupported parameter type: array
Wrote 2 call(s) to messy_out.json in 2.81 seconds; skipped 4.
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
| `make run-ranker`   | run with the keyword ranker on                |
| `make trace-ranker` | trace with the keyword ranker on              |
| `make run-model`    | run with `MODEL=<alias>`                      |
| `make debug`        | run under `pdb`                               |
| `make lint`         | `flake8 .` and `mypy` with the required flags |
| `make lint-strict`  | `flake8 .` and `mypy . --strict`              |
| `make clean`        | remove `__pycache__` and `.mypy_cache`        |
| `make export-vocabulary` | save the model's vocabulary to `data/output/` |

## Algorithm

### Shortlisting: keyword ranking (optional)

Off by default; `--ranker` turns it on. Without it, every function is a
candidate and the model makes the choice.

`keyword_ranker.shortlist_functions` scores every function against the
request: shared words count, shared terms from five domain vocabularies
(economy, database, math, boolean, prose) count more, a matching domain adds
a bonus, and the number of numbers in the request is compared with the
function's number parameters. The functions with the best score are kept;
if nothing scores, all of them are kept.

### Choosing the function: constrained decoding

The prompt lists the candidate functions as `name: description` and ends
with `{"name": "`. The model
then writes the function name one token at a time, but it may only write a
candidate's name:

- Each name, with its closing quote, is encoded into a token sequence.
  Together the sequences form a trie.
- At each step only the next tokens of the names that still match are
  allowed. The decoder takes the model's logits, sets every other token to
  minus infinity, and picks the highest remaining one.
- If exactly one token is allowed, it is a **fast-forward token**: it is
  appended **without calling the model**. Calling the model could not change
  the result, because every other token is masked out.

With several candidates, the model makes the choice: for the sample names
every name starts with `fn` (fast-forward), the model chooses once between
the next tokens (`_add`, `_g`, `_reverse`, ...), and the rest of the name is
fast-forwarded again. So without the ranker, choosing among all 12 functions
takes exactly one model call. When the ranker is on and keeps a single
function, every token is fast-forwarded and the model is not called. The
closing quote makes every sequence end differently, even when one name starts
another (`fn_read` and `fn_read_file`).

### Extracting the arguments: regular expressions

The arguments are read from the request in the order the schema lists them:

- **number**: the numbers in the request, in order, ignoring digits inside
  words. Whole numbers are written as ints, others as floats.
- **integer**: the same, but the number must be whole.
- **string**: a few patterns for common parameter names (`regex`,
  `replacement`, `source_string`, `path`, `encoding`), then the quoted parts
  of the request in order, and finally the last word.
- **boolean**: the word `true` or `false`, or the answer to an
  "is N even/odd" question.

If a value cannot be found, the prompt is reported and skipped (see
[Example usage](#example-usage)). The values go into a Pydantic
`FunctionCall` with exactly the keys `prompt`, `name` and `parameters`, which
is written with `json.dump`, so the output is always valid JSON.

### Prompt format

`chat()` in `src/client.py` builds the prompt in the conversation layout Qwen
was trained on. For `Say hi to Alice` (shortened here to 3 of the 12 listed
functions), the model receives exactly this:

```text
<|im_start|>system
You call functions. Pick the one function whose description matches what the user asks for, and reply with the call as JSON.

Functions:
- fn_add_numbers: Add two numbers together and return their sum.
- fn_greet: Generate a greeting message for a person by name.
- fn_reverse_string: Reverse a string and return the reversed result.<|im_end|>
<|im_start|>user
Say hi to Alice<|im_end|>
<|im_start|>assistant
<think>

</think>

<tool_call>
{"name": "
```

- `<|im_start|>system`, `user` and `assistant` mark who is speaking, and
  `<|im_end|>` ends each turn.
- The empty `<think>` block tells Qwen3 to skip its reasoning step.
- `<tool_call>` is the marker Qwen3 learned to write right before a function
  call, so the model is at exactly the place where it is used to writing one.
- `{"name": "` is the start of the answer. The program writes it, not the
  model; the model only continues it. This is called *prefilling*: the
  program controls the start of the answer, not just the question.

The `<think>` block and `<tool_call>` are added only when the tokenizer has
them as special tokens; for a model without `<|im_start|>` and `<|im_end|>`,
those are written as plain text. Only the markers become special-token ids:
the function list, the request and the answer start are encoded with
`allow_special=False`, so a request that contains `<|im_start|>` stays plain
text and cannot open a new turn (prompt injection).

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
   their single ids. With `allow_special=False` they stay plain text, which
   is how user and file text is encoded.
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

`--trace` (or `make trace`) prints how each function was chosen instead of the
timing JSON. By default the model chooses among all the functions:

```text
PROMPT: Say hi to Alice
1. Tokenization  (4 tokens)
   ['Say', ' hi', ' to', ' Alice']

2. Function name  (keyword ranker off: the model chooses from all 12)
   'fn'           fast-forward
   12 allowed tokens, one model call:
   ► '_g'           fn_greet                       p= 99.9%  ████████████████████████
     '_get'         fn_get_square_root             p=  0.1%  ························
     '_add'         fn_add_numbers                 p=  0.0%  ························
     '_reverse'     fn_reverse_string              p=  0.0%  ························
     '_execute'     fn_execute_sql_query           p=  0.0%  ························
     ... 7 more
   'reet'         fast-forward
   '"'            fast-forward

3. Arguments  (read from the prompt with regular expressions)
   name = 'Alice'  (str)

4. Result
   fn_greet {"name": "Alice"}
   1 model calls, 3 fast-forward tokens
```

With `--ranker` (or `make trace-ranker`), when the ranker keeps one function,
every token of the name is fast-forwarded:

```text
PROMPT: Read the file at /home/user/data.json with utf-8 encoding
2. Function name  (keyword ranker kept 1 of 12: fn_read_file)
   'fn'           fast-forward
   '_read'        fast-forward
   '_file'        fast-forward
   '"'            fast-forward

3. Arguments  (read from the prompt with regular expressions)
   path = '/home/user/data.json'  (str)
   encoding = 'utf-8'  (str)

4. Result
   fn_read_file {"path": "/home/user/data.json", "encoding": "utf-8"}
   0 model calls, 4 fast-forward tokens
```

It shows how encoding and decoding work together with constrained decoding:
`encode` turns the prompt and every candidate name into tokens (the trie),
and `decode` turns the allowed tokens back into text for the trace, with the
probability the model gave each one. The trace is recorded only when
`--trace` is given.

## Design decisions

- **The model chooses; the keyword shortlist is optional.** The subject asks
  for the function to be chosen by the LLM, not by heuristics, so by default
  the model chooses among all the functions and can only write a real name.
  The keyword ranker (`--ranker`) can narrow the functions down by words and
  domain first. On the 22 sample prompts both give the same 22 calls: with
  the ranker the model is never called (0.26 s), without it the model is
  called once per prompt (31 s).
- **Short function lines.** The model sees `name: description`, without the
  parameter list: each call costs roughly in proportion to the prompt
  length, and the model only chooses the name. On 32 prompts this was 20%
  faster than `name(a: number, b: number): description`, with the same 32
  right choices. The descriptions stay, because they are what lets the model
  match "plus" to "Add two numbers".
- **Regular expressions for the arguments.** Reading values from the request
  with regex needs no model call. A version that also decoded every argument
  with constrained decoding (see the git history) reached 20 of 22 full calls
  on the sample but took 190 to 216 seconds, against under a second now.
- **The model's own chat format.** Measured on the 22 sample prompts, the
  chat format chose the right function 22 times out of 22, against 20 for a
  plain-text prompt.
- **Never stop the whole run.** Input files are validated up front; each
  prompt is then processed on its own, so a malformed item, a missing value or
  an unsupported parameter type skips only that prompt. Ctrl+C writes the
  results so far.
- **Pydantic for every class**, `numpy` for the logit mask, and no use of
  private SDK attributes.

## Performance analysis

Measured on the cluster CPU with `Qwen/Qwen3-0.6B`:

| Set                                            | Function | Full call |
| ---------------------------------------------- | -------- | --------- |
| Sample prompts (22)                            | 22/22    | 17/22     |
| Reworded prompts, edge cases, unseen functions | -        | 16/22     |

- **Function choice** is right for every sample prompt, with and without
  the ranker. By default the model chooses among all 12 functions, with one
  model call per prompt. With `--ranker`, the ranker keeps a single function
  for all 22, so the model is not called; on the second set it kept several
  functions for 6 prompts and the model chose.
- **Arguments:** 17 of 22 sample calls are fully right (77%), below the
  subject's 90% target. The misses are `database` read as `"database"`
  instead of `production` / `system` (2), `replacement` read as
  `"asterisks"` instead of `"*"`, and the two templates read as `"profile"`
  and `"hello"` instead of the whole template.
- In the second set, the email address is not found (`to` gets the subject
  instead), `José` loses its accent, and three prompts are skipped because a
  value is missing or not a whole number ("What is the sum of 7?", "Is 4.5 an
  even number?", "Is the statement 'the sky is green' true?").
- **Speed:** 31 s for the 22 sample prompts by default (one model call each,
  about 1.4 s per prompt), and 0.26 s with `--ranker` (no model call). Times
  vary with other load on the machine. Start-up
  adds about 34 s:
  24 s to import torch and transformers, 9 s to load the model and 1 s to
  build the tokenizer.
- **Reliability:** every output is valid JSON with exactly the required keys.
  No input crashed the program: missing or invalid files, malformed items,
  missing values and unsupported types all end with a result or a clear
  message.

## Challenges faced

- **Combining keywords and the model.** Keyword ranking is fast and right on
  the sample, but it keeps all functions for requests it has no words for
  ("Say hi to Alice"). The model, limited to real names, handles those.
  Measuring prompt formats instead of guessing made its choice reliable.
- **Time without a cache.** Each model call reruns the whole prompt (about
  0.3 s plus 6 ms per prompt token), so the number of calls decides the run
  time. Choosing the name needs one call; decoding every argument token by
  token needed about ten times as many.
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
  has never seen (weather, temperature, email, word count). The same sets
  were run on the version with full constrained decoding to compare.
- **Error handling:** missing and invalid JSON files, a prompt file with
  non-object items and missing keys, a function with an `array` parameter, a
  broken default function file next to a valid `--functions_definition`, and
  an unknown model name.
- **Static checks:** `make lint` and `make lint-strict` pass.
- **By hand:** `--trace` shows how every function was chosen.

## Project structure

```text
src/__main__.py      Command line, per-prompt error recovery
src/client.py        Function choice (constrained decoding) and regex arguments
src/keyword_ranker.py  Optional keyword and domain shortlist (--ranker)
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
- Fixing mypy and flake8 errors
- the byte-level BPE tokenizer and its comparison with the SDK,
- the constrained decoding of the function name (name trie, chat prompt) and
  the measurements that shaped it,
- connecting the keyword ranker to the name decoding,
- the `--trace` visualization, per-prompt error recovery and lint fixes,
- reviewing the project against the subject, and drafting this README.

All of it was reviewed, run and tested before submission.

looking up regex


<|im_start|>system
You call functions. Pick the one function ...
fn_add_numbers: Add two numbers together ...        ← the function descriptions
fn_multiply_numbers: Multiply two numbers ...
<|im_end|>
<|im_start|>user
What is the sum of 2 and 3?                         ← the prompt
<|im_end|>
<|im_start|>assistant
<think>\n\n</think>\n\n<tool_call>\n{"name": "fn     ← what's written so far


For the context ... What is the sum of 2 and 3? ... {"name": "fn:

Step	Tensor	What happens
1	embed_tokens [151936, 1024]	Look up, no maths: for each token id in the context, take that row. Token 2891 (_add) becomes row 2891, a vector of 1024 numbers. The context is now one vector per token.
2a	input_layernorm [1024]	Rescale each vector so the numbers don't grow too big.
2b	q_proj, k_proj, v_proj	Multiply each vector by these 3 tables to get 3 new vectors: a query ("what am I looking for?"), a key ("what do I contain?") and a value ("what do I pass on?").
2c	(attention itself)	For the last token, multiply its query with every earlier token's key: a high result means "relevant". Softmax turns these into weights (the same exp idea as in your trace), then it takes a weighted sum of the values. This is how information from "sum" reaches the last position.
2d	o_proj	Multiply the result back to 1024 numbers and add it to the vector.
2e	mlp: gate_proj, up_proj, down_proj	Expand each vector to 3072 numbers, apply a nonlinear function (SiLU), and shrink it back to 1024. Each token is processed on its own; much of the "facts" learned in training is stored here.
layers.1 to layers.27	Repeat 2a to 2e with that layer's own tables, 28 times in total.
3	model.norm, then lm_head [151936, 1024]	Rescale the last token's final vector, then multiply it with every row: 151,936 results, which are the logits _pick gets.




About 25 steps in total. You've done 7, so 18 are left. Here's the map, in four parts:

Part A: setup, once per run (steps 1-8)
1-7 ✓ command → main() → args → functions → prompts → model name → model → tokenizer
8. the router object is created (Pydantic checks llm)

Part B: the loop starts (steps 9-10)
9. process_prompts takes item 1 and checks it's {"prompt": "..."}
10. generate_call starts and resets the trace counters

Part C: the heart, one prompt "Give me 10 plus 20" (steps 11-21)
11. the keyword ranker scores all 12 functions ✓
12. the shortlist: which names survive✓
13. candidates, the FunctionSchema objects for those names✓
14. chat() builds the token ids of the context✓
15. _choose_text splits the names into tokens
16. round 1: fn is fast-forwarded✓
17. round 2: the model call: logits → mask → argmax
18. the last rounds: fast-forward, then the name is returned
19. the name is matched back to its FunctionSchema
20. extract_arguments: regex finds a = 10.0, b = 20.0
21. the trace and the FunctionCall are built and returned

Part D: output and finish (steps 22-25)
22. render_generation prints the trace block
23. the result is added to the list, and the other 9 prompts go through the same steps 9-22
24. write_results writes the JSON file
25. the summary line, return 0 → SystemExit(0)

Part C (steps 11-21) is the part most worth taking slowly. It's the actual function calling, and evaluators will mostly ask about it. Parts B and D are short.
