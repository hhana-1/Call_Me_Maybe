"""Constrained decoding of function calls, one token at a time.

The model writes each call as JSON::

    {"name": "fn_add_numbers", "arguments": {"a": 2, "b": 3}}

At every step only the tokens that keep this text valid, and true to the
function's schema, are allowed: every other logit is set to minus infinity.
When just one token is allowed it is added without calling the model.

Two prompts keep each model call short:

1. **Name**: the prompt lists every function; the model writes one name.
2. **Arguments**: the prompt shows only the chosen function; each value is
   written under the rules of its type (number, integer, boolean, string).
"""

import json
import re
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator

from src.schema import FunctionCall, FunctionSchema
from src.tokenizer import BPETokenizer

MAX_STRING_TOKENS = 48
MAX_NUMBER_CHARS = 40
JSON_ESCAPES = '"\\/bfnrt'

# A number written so far ("", "-", "12", "12.", "12.5") and a finished one.
NUMBER_PREFIX = re.compile(r"-?((0|[1-9][0-9]*)(\.[0-9]*)?)?")
NUMBER = re.compile(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?")
INTEGER_PREFIX = re.compile(r"-?(0|[1-9][0-9]*)?")
INTEGER = re.compile(r"-?(0|[1-9][0-9]*)")

PICK_FUNCTION = (
    "You call functions. Pick the one function whose description matches "
    "what the user asks for, and reply with the call as JSON.\n\nFunctions:\n"
)
FILL_ARGUMENTS = (
    "You call functions. Fill in the arguments of this function with the "
    "values the user asks for, and reply with the call as JSON.\n\n"
    "Function:\n"
)


def describe(function: FunctionSchema) -> str:
    """Render a function as ``- name(a: number, b: string): description``."""
    parameters = ", ".join(
        f"{key}: {spec.get('type', 'unknown')}"
        for key, spec in function.parameters.items()
    )
    return f"- {function.name}({parameters}): {function.description}"


def chat(tokenizer: BPETokenizer, system: str, user: str, answer: str
         ) -> list[int]:
    """Encode a prompt in Qwen's chat format, ending inside the answer.

    The markers are written as text; the tokenizer turns each one into its
    single special-token id. Qwen3 also gets an empty ``<think>`` block (no
    thinking) and its ``<tool_call>`` marker when the model has them.
    """
    text = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    if "<think>" in tokenizer.special_tokens:
        text += "<think>\n\n</think>\n\n"
    if "<tool_call>" in tokenizer.special_tokens:
        text += "<tool_call>\n"
    return tokenizer.encode(text + answer)


def scan_string(text: str, escaped: bool) -> tuple[bool, bool, str | None]:
    """Check a token's text written inside a JSON string.

    Args:
        text: The token's text.
        escaped: Whether the string so far ends with an unfinished ``\\``.

    Returns:
        ``(valid, escaped, rest)``: ``rest`` is the text after the closing
        quote, or None if the string is still open.
    """
    for index, char in enumerate(text):
        if escaped:
            if char not in JSON_ESCAPES:
                return False, False, None
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return True, False, text[index + 1:]
        elif ord(char) < 0x20:
            return False, False, None
    return True, escaped, None


class TokenAwareConstrainedRouter(BaseModel):
    """Generate schema-valid function calls with constrained decoding.

    Attributes:
        llm: The SDK model; only ``get_logits_from_input_ids`` is used.
        tokenizer: The project's BPE tokenizer for the same model.
        trace_enabled: Record every decoding step in ``last_trace``.
        last_trace: Steps of the latest call, when tracing is enabled.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Any
    tokenizer: BPETokenizer
    trace_enabled: bool = False
    last_trace: dict[str, Any] | None = None

    _texts: list[str] = PrivateAttr(default_factory=list)
    _plain_ids: npt.NDArray[np.int64] = PrivateAttr(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    _quote_ids: list[int] = PrivateAttr(default_factory=list)
    _opening_ids: list[int] = PrivateAttr(default_factory=list)
    _char_ids: dict[int, str] = PrivateAttr(default_factory=dict)
    _sign_ids: dict[int, str] = PrivateAttr(default_factory=dict)
    _endings: dict[str, list[int]] = PrivateAttr(default_factory=dict)
    _steps: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _field: str = PrivateAttr(default="name")
    _calls: int = PrivateAttr(default=0)

    @field_validator("llm")
    @classmethod
    def _check_llm(cls, value: Any) -> Any:
        """Require the SDK method the decoder relies on."""
        if not callable(getattr(value, "get_logits_from_input_ids", None)):
            raise ValueError("Model must provide get_logits_from_input_ids()")
        return value

    def model_post_init(self, __context: Any) -> None:
        """Decode every token once and sort the tokens into masks."""
        size = max(self.tokenizer.vocab.values()) + 1
        self._texts = [self.tokenizer.token_text(i) for i in range(size)]
        plain = []
        for index, text in enumerate(self._texts):
            if not text or "�" in text or min(map(ord, text)) < 0x20:
                continue  # empty, half a character, or a control character
            if '"' in text or "\\" in text:
                self._quote_ids.append(index)
            else:
                plain.append(index)
        self._plain_ids = np.array(plain, dtype=np.int64)
        self._opening_ids = [
            index for index in self._quote_ids
            if self._texts[index].startswith(' "')
        ]
        self._char_ids = {self._single_id(c): c for c in "0123456789."}
        self._sign_ids = {self._single_id(" "): "", self._single_id(" -"): "-"}

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
        self._steps, self._calls, self._field = [], 0, "name"

        # Stage 1: the name. The closing quote makes every choice end
        # differently, even when one name starts another (fn_read, fn_read_x).
        context = chat(
            self.tokenizer,
            PICK_FUNCTION + "\n".join(map(describe, functions)),
            prompt,
            '{"name": "',
        )
        name = self._choose_text(context, [f'{f.name}"' for f in functions])
        function = next(f for f in functions if f'{f.name}"' == name)

        # Stage 2: the arguments, in the order the schema lists them.
        parameters: dict[str, Any] = {}
        keys = list(function.parameters)
        if keys:
            context = chat(
                self.tokenizer,
                FILL_ARGUMENTS + describe(function),
                prompt,
                f'{{"name": {json.dumps(function.name)}, "arguments": '
                f"{{{json.dumps(keys[0])}:",
            )
        for position, key in enumerate(keys):
            self._field = key
            last = position + 1 == len(keys)
            ending = "}}" if last else f", {json.dumps(keys[position + 1])}:"
            parameters[key] = self._value(
                context, function.parameters[key].get("type"), ending
            )

        if self.trace_enabled:
            self.last_trace = {
                "prompt": prompt,
                "prompt_tokens": [
                    self.tokenizer.token_text(i)
                    for i in self.tokenizer.encode(prompt)
                ],
                "function_count": len(functions),
                "selected": function.name,
                "steps": self._steps,
                "model_calls": self._calls,
            }
        return FunctionCall(
            prompt=prompt, name=function.name, parameters=parameters
        )

    def _value(self, context: list[int], kind: Any, ending: str) -> Any:
        """Decode one argument value, then add the JSON that follows it."""
        if kind in ("number", "integer"):
            return self._number(context, kind == "integer", ending)
        if kind == "string":
            return self._string(context, ending)
        if kind == "boolean":
            word = self._choose_text(context, [" true", " false"])
            self._force(context, ending)
            return word == " true"
        raise ValueError(
            f"Unsupported parameter type {kind!r} for {self._field!r}"
        )

    def _choose_text(self, context: list[int], choices: list[str]) -> str:
        """Let the model write exactly one of ``choices``.

        The choices' token sequences form a trie: at each step only the next
        tokens of the choices that still match are allowed.
        """
        sequences = {text: self.tokenizer.encode(text) for text in choices}
        written: list[int] = []
        while True:
            branches: dict[int, list[str]] = {}
            for text, ids in sequences.items():
                if ids[:len(written)] != written:
                    continue
                if len(ids) == len(written):
                    return text
                branches.setdefault(ids[len(written)], []).append(text)
            labels = {
                token: ", ".join(text.strip(' "') for text in texts)
                for token, texts in branches.items()
            }
            written.append(self._pick(context, list(branches), labels=labels))

    def _number(self, context: list[int], integer: bool, ending: str
                ) -> int | float:
        """Decode a JSON number: a sign, digits and, unless integer, one dot.

        It starts with `` `` or `` -``: Qwen writes space + minus as one
        token, so a forced space would make negative numbers unlikely.
        """
        prefix, finished = (
            (INTEGER_PREFIX, INTEGER) if integer else (NUMBER_PREFIX, NUMBER)
        )
        text = self._sign_ids[self._pick(context, list(self._sign_ids))]
        while True:
            complete = finished.fullmatch(text) is not None
            room = len(text) < MAX_NUMBER_CHARS or not complete
            chars = {
                token: char for token, char in self._char_ids.items()
                if room and prefix.fullmatch(text + char)
            }
            allowed = list(chars) + (self._ending_ids(ending) if complete
                                     else [])
            token = self._pick(context, allowed)
            if token not in chars:  # the value ended with the next JSON
                self._force(context, ending[len(self._texts[token]):])
                return int(text) if integer else float(text)
            text += chars[token]

    def _string(self, context: list[int], ending: str) -> str:
        """Decode a JSON string, checking escapes as they are written.

        The first token must start with `` "``: Qwen merges the quote with
        what follows (`` "/``, `` "[``), so a forced lone quote would keep
        the model from writing ``/home`` or ``[aeiou]``. After
        ``MAX_STRING_TOKENS`` tokens only a closing quote is allowed.
        """
        written, escaped = "", False
        for count in range(MAX_STRING_TOKENS + 1):
            skip = 2 if count == 0 else 0  # leave out the opening ' "'
            must_close = count == MAX_STRING_TOKENS
            allowed = []
            for token in self._opening_ids if count == 0 else self._quote_ids:
                text = self._texts[token][skip:]
                valid, _, rest = scan_string(text, escaped)
                if valid and (
                    ending.startswith(rest) if rest is not None
                    else not must_close
                ):
                    allowed.append(token)
            plain = None if skip or escaped or must_close else self._plain_ids
            token = self._pick(context, allowed, plain)
            piece = self._texts[token][skip:]
            _, escaped, rest = scan_string(piece, escaped)
            if rest is None:
                written += piece
                continue
            written += piece[:len(piece) - len(rest) - 1]
            self._force(context, ending[len(rest):])
            value: str = json.loads(f'"{written}"')
            return value
        raise ValueError("The string value did not end")

    def _ending_ids(self, ending: str) -> list[int]:
        """Tokens whose text starts the JSON that follows a value."""
        if ending not in self._endings:
            self._endings[ending] = [
                token for token, text in enumerate(self._texts)
                if text and ending.startswith(text)
            ]
        return self._endings[ending]

    def _single_id(self, text: str) -> int:
        """The id of ``text``, which must be exactly one token."""
        ids = self.tokenizer.encode(text)
        if len(ids) != 1:
            raise ValueError(f"Tokenizer has no single token for {text!r}")
        return ids[0]

    def _force(self, context: list[int], text: str) -> None:
        """Add fixed JSON structure without calling the model."""
        for token in self.tokenizer.encode(text):
            context.append(token)
            self._record({"text": self._texts[token], "forced": True})

    def _record(self, step: dict[str, Any]) -> None:
        """Remember a step for the trace."""
        if self.trace_enabled:
            self._steps.append({"field": self._field, **step})

    def _pick(
        self,
        context: list[int],
        allowed: list[int],
        extra: npt.NDArray[np.int64] | None = None,
        labels: dict[int, str] | None = None,
    ) -> int:
        """Add the best allowed token to ``context`` and return it.

        Args:
            context: Token ids so far (prompt plus output); extended here.
            allowed: The ids allowed at this step.
            extra: More allowed ids as an array (the big string mask).
            labels: What each allowed token stands for, shown in the trace.
        """
        ids = np.array(allowed, dtype=np.int64)
        if extra is not None:
            ids = np.concatenate([ids, extra])
        if ids.size == 0:
            raise ValueError("No token is allowed at this step")
        if ids.size == 1:  # forced: no need to ask the model
            token = int(ids[0])
            context.append(token)
            self._record({"text": self._texts[token], "forced": True})
            return token
        logits = np.asarray(
            self.llm.get_logits_from_input_ids(context), dtype=np.float64
        )
        masked = np.full(logits.shape, -np.inf)
        masked[ids] = logits[ids]  # every other token becomes impossible
        token = int(np.argmax(masked))
        context.append(token)
        self._calls += 1
        if self.trace_enabled:
            weights = np.exp(logits[ids] - logits[token])
            step: dict[str, Any] = {
                "text": self._texts[token],
                "forced": False,
                "allowed": int(ids.size),
                "probability": float(1 / weights.sum()),
            }
            if labels:
                step["options"] = [
                    {
                        "text": self._texts[int(ids[i])],
                        "label": labels[int(ids[i])],
                        "probability": float(weights[i] / weights.sum()),
                        "chosen": int(ids[i]) == token,
                    }
                    for i in np.argsort(-weights)[:5]
                ]
            self._record(step)
        return token


def load_functions(path: str) -> list[FunctionSchema]:
    """Load and validate function definitions from JSON."""
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError("Function definitions must be a JSON array")
    return [FunctionSchema.model_validate(item) for item in raw]
