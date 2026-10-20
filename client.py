"""Constrained decoding of function calls, one token at a time.

The model writes each call as JSON::

    {"name": "<function>", "parameters": {"<key>": <value>, ...}}

At every step the decoder lists the tokens that keep this text valid and
faithful to the function schema, sets every other logit to minus infinity
and takes the best remaining token. When only one token is allowed it is
appended without calling the model at all.

Generation runs in two stages to keep each model call short:

1. **Function name**: the prompt lists every function, and the name is
   decoded through a trie of the names' token sequences.
2. **Arguments**: a shorter prompt shows only the chosen function, and each
   value is decoded under the grammar of its type (number, integer,
   boolean or string).

Prompts use the model's own chat format (``<|im_start|>`` turns and an
empty ``<think>`` block for Qwen3) when its tokenizer has those special
tokens, and plain text otherwise.
"""

import json
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator

from src.schema import FunctionCall, FunctionSchema
from src.tokenizer import BPETokenizer

_DIGITS = "0123456789"
_JSON_ESCAPES = '"\\/bfnrt'
_NO_INDEX = -1

TokenIds = npt.NDArray[np.int64]


def _signature(function: FunctionSchema) -> str:
    """Render a function as ``name(a: number, b: string): description``."""
    parameters = ", ".join(
        f"{key}: {spec.get('type', 'unknown')}"
        for key, spec in function.parameters.items()
    )
    return f"{function.name}({parameters}): {function.description}"


def _selection_instructions(functions: Sequence[FunctionSchema]) -> str:
    """System text for stage 1: every function the model may call."""
    listing = "\n".join(f"- {_signature(function)}" for function in functions)
    return (
        "You call functions. Pick the one function whose description matches "
        "what the user asks for, and reply with the call as JSON.\n\n"
        f"Functions:\n{listing}"
    )


def _argument_instructions(function: FunctionSchema) -> str:
    """System text for stage 2: only the function that was chosen."""
    return (
        "You call functions. Fill in the arguments of this function with the "
        "values the user asks for, and reply with the call as JSON.\n\n"
        f"Function:\n- {_signature(function)}"
    )


def _argument_answer(function: FunctionSchema) -> str:
    """Start of the stage-2 answer; it ends where the first value begins."""
    keys = list(function.parameters)
    opening = f"{json.dumps(keys[0])}:" if keys else "}}"
    return (
        f'{{"name": {json.dumps(function.name)}, "arguments": {{{opening}'
    )


def _number_options(text: str, integer: bool, limit: int) -> tuple[str, bool]:
    """Return the characters that may follow ``text`` in a JSON number.

    Args:
        text: The number written so far (without the leading space).
        integer: Whether ``.`` is forbidden.
        limit: Maximum number of characters before the number must end.

    Returns:
        The allowed next characters, and whether ``text`` is already a
        complete number (so the value may end here).
    """
    body = text[1:] if text.startswith("-") else text
    if not body:
        return (_DIGITS + "-" if not text else _DIGITS), False
    if len(text) >= limit:
        return ("", True) if body[-1] != "." else (_DIGITS, False)
    if "." in body:
        fraction = body.split(".", 1)[1]
        return _DIGITS, bool(fraction)
    allowed = "" if body == "0" else _DIGITS  # JSON forbids leading zeros
    return allowed + ("" if integer else "."), True


def _scan_string_token(text: str, escaped: bool) -> tuple[bool, bool, int]:
    """Check one token inside a JSON string.

    Args:
        text: The token's text.
        escaped: Whether the previous character was an unfinished ``\\``.

    Returns:
        ``(valid, escaped_after, close_index)``, where ``close_index`` is the
        position of the closing quote in ``text`` or ``-1`` if it has none.
    """
    for index, char in enumerate(text):
        if escaped:
            if char not in _JSON_ESCAPES:
                return False, False, _NO_INDEX
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return True, False, index
        elif ord(char) < 0x20:
            return False, False, _NO_INDEX
    return True, escaped, _NO_INDEX


class TokenAwareConstrainedRouter(BaseModel):
    """Generate schema-valid function calls with constrained decoding.

    Attributes:
        llm: The SDK model; only ``get_logits_from_input_ids`` is used.
        tokenizer: The project's BPE tokenizer for the same model.
        trace_enabled: Record each decoding step in ``last_trace``.
        last_trace: Steps of the most recent call when tracing is enabled.
        max_string_tokens: Longest string value before it is closed.
        max_number_chars: Longest number value before it is ended.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Any
    tokenizer: BPETokenizer
    trace_enabled: bool = False
    last_trace: dict[str, Any] | None = None
    max_string_tokens: int = 48
    max_number_chars: int = 40

    _texts: list[str] = PrivateAttr(default_factory=list)
    _plain_string_ids: TokenIds = PrivateAttr(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    _special_string_ids: list[int] = PrivateAttr(default_factory=list)
    _opening_ids: list[int] = PrivateAttr(default_factory=list)
    _char_ids: dict[str, int] = PrivateAttr(default_factory=dict)
    _endings: dict[str, list[int]] = PrivateAttr(default_factory=dict)
    _steps: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _calls: int = PrivateAttr(default=0)

    @field_validator("llm")
    @classmethod
    def _check_llm(cls, value: Any) -> Any:
        """Require the SDK method the decoder relies on."""
        if not callable(getattr(value, "get_logits_from_input_ids", None)):
            raise ValueError("Model must provide get_logits_from_input_ids()")
        return value

    def model_post_init(self, __context: Any) -> None:
        """Decode every token once and sort tokens into the masks we need."""
        size = max(self.tokenizer.vocab.values()) + 1
        self._texts = [
            self.tokenizer.token_text(index) for index in range(size)
        ]
        plain: list[int] = []
        for index, text in enumerate(self._texts):
            if not text or "�" in text or any(ord(c) < 0x20 for c in text):
                continue
            if '"' in text or "\\" in text:
                self._special_string_ids.append(index)
            else:
                plain.append(index)
        self._plain_string_ids = np.array(plain, dtype=np.int64)
        self._opening_ids = [
            index for index in self._special_string_ids
            if self._texts[index].startswith(' "')
        ]
        for char in _DIGITS + ".-":
            ids = self.tokenizer.encode(char)
            if len(ids) != 1:
                raise ValueError(f"Tokenizer has no single token for {char!r}")
            self._char_ids[char] = ids[0]

    def generate_call(
        self, prompt: str, functions: Sequence[FunctionSchema]
    ) -> FunctionCall:
        """Generate one validated function call for ``prompt``.

        Raises:
            ValueError: If no functions are given or a parameter type is not
                supported.
        """
        if not functions:
            raise ValueError("No function definitions were provided")
        self._steps, self._calls = [], 0
        function, name_steps = self._choose_function(prompt, functions)
        parameters = self._fill_arguments(prompt, function)
        if self.trace_enabled:
            self.last_trace = {
                "prompt": prompt,
                "prompt_tokens": [
                    self.tokenizer.token_text(index)
                    for index in self.tokenizer.encode(prompt)
                ],
                "function_count": len(functions),
                "name_steps": name_steps,
                "selected": function.name,
                "argument_steps": self._steps,
                "model_calls": self._calls,
            }
        return FunctionCall(
            prompt=prompt, name=function.name, parameters=parameters
        )

    # ----- stage 1: function name -----------------------------------------

    def _choose_function(
        self, prompt: str, functions: Sequence[FunctionSchema]
    ) -> tuple[FunctionSchema, list[dict[str, Any]]]:
        """Decode the function name, allowing only real names (a trie)."""
        context = self._conversation(
            _selection_instructions(functions), prompt, '{"name": "'
        )
        names = [
            (item, self.tokenizer.encode(item.name)) for item in functions
        ]
        close_quote = self.tokenizer.encode('"')[0]
        steps: list[dict[str, Any]] = []
        while True:
            position = len(steps)
            finished = [item for item, ids in names if len(ids) == position]
            longer = [pair for pair in names if len(pair[1]) > position]
            if finished and not longer:
                return finished[0], steps
            branches: dict[int, list[str]] = {}
            for item, ids in longer:
                branches.setdefault(ids[position], []).append(item.name)
            if finished:  # a name that is a prefix of another may end here
                branches[close_quote] = [finished[0].name]
            token, step = self._pick(context, list(branches), options=True)
            step["functions"] = branches[token]
            for option in step.get("options", []):
                option["functions"] = branches[option["id"]]
            steps.append(step)
            if token == close_quote and finished:
                return finished[0], steps
            context.append(token)
            names = [pair for pair in longer if pair[1][position] == token]

    # ----- stage 2: arguments ---------------------------------------------

    def _fill_arguments(
        self, prompt: str, function: FunctionSchema
    ) -> dict[str, Any]:
        """Decode every argument value under the grammar of its type."""
        context = self._conversation(
            _argument_instructions(function),
            prompt,
            _argument_answer(function),
        )
        keys = list(function.parameters)
        values: dict[str, Any] = {}
        for position, key in enumerate(keys):
            ending = (
                f", {json.dumps(keys[position + 1])}:"
                if position + 1 < len(keys)
                else "}}"
            )
            kind = function.parameters[key].get("type")
            if kind in ("number", "integer"):
                values[key] = self._number(context, key, ending, kind)
            elif kind == "boolean":
                values[key] = self._boolean(context, key, ending)
            elif kind == "string":
                values[key] = self._string(context, key, ending)
            else:
                raise ValueError(
                    f"Unsupported parameter type {kind!r} for {key!r} "
                    f"in {function.name}"
                )
        return values

    def _number(
        self, context: list[int], key: str, ending: str, kind: str
    ) -> int | float:
        """Decode a JSON number: digits, an optional sign and decimal point.

        Qwen writes `` -`` (space + minus) as one token, so the value starts
        with a choice between `` `` and `` -`` instead of a forced space.
        """
        space, signed = self.tokenizer.encode(" "), self.tokenizer.encode(" -")
        text = ""
        if len(space) == 1 and len(signed) == 1:
            token, step = self._pick(context, [space[0], signed[0]])
            self._record(step, key)
            context.append(token)
            text = "-" if token == signed[0] else ""
            sign_done = True
        else:
            self._force(context, " ", key)
            sign_done = False
        while True:
            chars, complete = _number_options(
                text, kind == "integer", self.max_number_chars
            )
            if sign_done:
                chars = chars.replace("-", "")
            by_token = {self._char_ids[char]: char for char in chars}
            allowed = list(by_token)
            if complete:
                allowed += self._ending_ids(ending)
            token, step = self._pick(context, allowed)
            self._record(step, key)
            context.append(token)
            if token in by_token:
                text += by_token[token]
                continue
            self._force(context, ending[len(self._texts[token]):], key)
            value = json.loads(text)
            return int(value) if kind == "integer" else float(value)

    def _boolean(self, context: list[int], key: str, ending: str) -> bool:
        """Decode ``true`` or ``false``."""
        words = {
            word: self.tokenizer.encode(" " + word)
            for word in ("true", "false")
        }
        written: list[int] = []
        while True:
            matching = {
                word: ids for word, ids in words.items()
                if ids[:len(written)] == written
            }
            done = [word for word, ids in matching.items() if ids == written]
            if done:
                self._force(context, ending, key)
                return done[0] == "true"
            allowed = sorted({ids[len(written)] for ids in matching.values()})
            token, step = self._pick(context, allowed)
            self._record(step, key)
            context.append(token)
            written.append(token)

    def _string(self, context: list[int], key: str, ending: str) -> str:
        """Decode a JSON string; escapes are checked as they are written.

        The first token must start with `` "``. Qwen merges the quote with
        what follows (`` "/``, `` "[``), so forcing a lone quote would stop
        the model from writing ``/home`` or ``[aeiou]``.
        """
        content, escaped, count = "", False, 0
        while True:
            opening = count == 0
            closing_only = count >= self.max_string_tokens and not escaped
            allowed: list[int] = []
            candidates = (
                self._opening_ids if opening else self._special_string_ids
            )
            for index in candidates:
                text = self._texts[index][2 if opening else 0:]
                valid, _, close = _scan_string_token(text, escaped)
                if not valid or (closing_only and close == _NO_INDEX):
                    continue
                if close == _NO_INDEX or ending.startswith(text[close + 1:]):
                    allowed.append(index)
            plain = (
                self._plain_string_ids
                if not (escaped or closing_only or opening)
                else np.zeros(0, dtype=np.int64)
            )
            token, step = self._pick(context, allowed, plain)
            self._record(step, key)
            context.append(token)
            count += 1
            piece = self._texts[token][2 if opening else 0:]
            _, escaped, close = _scan_string_token(piece, escaped)
            if close == _NO_INDEX:
                content += piece
                continue
            content += piece[:close]
            self._force(context, ending[len(piece) - close - 1:], key)
            value: str = json.loads(f'"{content}"')
            return value

    # ----- shared helpers ---------------------------------------------------

    def _conversation(self, system: str, user: str, answer: str) -> list[int]:
        """Encode a chat prompt that ends inside the assistant's answer.

        Special tokens are inserted by id, so a model without them (or with
        only some of them) still gets a sensible prompt.
        """
        tokenizer = self.tokenizer
        start = tokenizer.special_id("<|im_start|>")
        end = tokenizer.special_id("<|im_end|>")
        if start is None or end is None:
            return tokenizer.encode(
                f"{system}\n\nRequest: {user}\nCall: {answer}"
            )
        ids = [start, *tokenizer.encode(f"system\n{system}"), end]
        ids += [*tokenizer.encode("\n"), start]
        ids += [*tokenizer.encode(f"user\n{user}"), end]
        ids += [*tokenizer.encode("\n"), start]
        ids += tokenizer.encode("assistant\n")
        think = tokenizer.special_id("<think>")
        think_end = tokenizer.special_id("</think>")
        if think is not None and think_end is not None:  # Qwen3: no thinking
            ids += [think, *tokenizer.encode("\n\n"), think_end]
            ids += tokenizer.encode("\n\n")
        tool_call = tokenizer.special_id("<tool_call>")
        if tool_call is not None:
            return ids + [tool_call, *tokenizer.encode("\n" + answer)]
        return ids + tokenizer.encode(answer)

    def _ending_ids(self, ending: str) -> list[int]:
        """Tokens whose text starts the structure that follows a value."""
        if ending not in self._endings:
            self._endings[ending] = [
                index for index, text in enumerate(self._texts)
                if text and ending.startswith(text)
            ]
        return self._endings[ending]

    def _force(self, context: list[int], text: str, key: str) -> None:
        """Append fixed JSON structure without calling the model."""
        for token in self.tokenizer.encode(text):
            context.append(token)
            self._record(
                {"id": token, "text": self._texts[token], "forced": True}, key
            )

    def _record(self, step: dict[str, Any], key: str) -> None:
        """Remember an argument step for the trace."""
        if self.trace_enabled:
            self._steps.append({**step, "field": key})

    def _pick(
        self,
        context: list[int],
        allowed: list[int],
        extra: TokenIds | None = None,
        options: bool = False,
    ) -> tuple[int, dict[str, Any]]:
        """Choose the best allowed token; call the model only if needed.

        Args:
            context: Token ids written so far (prompt plus output).
            allowed: Ids allowed at this step.
            extra: More allowed ids, as an array (the large string mask).
            options: Also report every allowed token's probability.

        Returns:
            The chosen id and a description of the step for the trace.
        """
        ids = np.array(allowed, dtype=np.int64)
        if extra is not None and extra.size:
            ids = np.concatenate([ids, extra])
        if ids.size == 0:
            raise ValueError("No token is allowed at this step")
        if ids.size == 1:
            token = int(ids[0])
            return token, {"id": token, "text": self._texts[token],
                           "forced": True}
        self._calls += 1
        logits = np.asarray(
            self.llm.get_logits_from_input_ids(context), dtype=np.float64
        )
        masked = np.full(logits.shape, -np.inf)
        masked[ids] = logits[ids]  # every other token is now impossible
        token = int(np.argmax(masked))
        weights = np.exp(logits[ids] - logits[token])
        step: dict[str, Any] = {
            "id": token,
            "text": self._texts[token],
            "forced": False,
            "allowed": int(ids.size),
            "logit": float(logits[token]),
            "probability": float(1.0 / weights.sum()),
        }
        if options:
            order = np.argsort(-logits[ids])
            step["options"] = [
                {
                    "id": int(ids[i]),
                    "text": self._texts[int(ids[i])],
                    "logit": float(logits[ids[i]]),
                    "probability": float(weights[i] / weights.sum()),
                }
                for i in order
            ]
        return token, step


def load_functions(path: str) -> list[FunctionSchema]:
    """Load and validate function definitions from JSON."""
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError("Function definitions must be a JSON array")
    return [FunctionSchema.model_validate(item) for item in raw]
