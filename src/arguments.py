"""Write the arguments of the chosen function with constrained decoding.

The model gets a second, short prompt: only the chosen function, with its
parameters (``describe_call``), and continues ``{"name": "...",
"parameters": {``. The code writes each key and the punctuation from the
schema; the model writes each value, limited by the parameter's type:

- ``number`` / ``integer``: one of the numbers in the request, or written
  digit by digit when the request has none;
- ``string``: a piece of the request copied, or written freely with only
  the tokens that keep the JSON valid;
- ``boolean``: ``true`` or ``false``.

``ArgumentWriter`` is the base class of ``TokenAwareConstrainedRouter``
(``client.py``), which gives it the token steps ``_pick``, ``_choose_text``
and ``_follow``.
"""

import json
import re
from abc import abstractmethod
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, PrivateAttr

from src.schema import FunctionSchema
from src.tokenizer import SDKTokenizer, chat

FILL_ARGUMENTS = (
    "You call functions. Fill in the arguments of this function with the "
    "values from the user's request, and reply with the call as JSON.\n\n"
    "Function:\n"
)
# A number written in the request, not part of a word such as "utf-8".
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
# Characters allowed after a backslash in a JSON string (no \u escapes).
JSON_ESCAPES = '"\\/bfnrt'
MAX_SPAN_WORDS = 12  # longest piece of the request a string can copy
MAX_VALUE_TOKENS = 48  # longest freely written string
MAX_NUMBER_CHARS = 16  # longest freely written number
SUPPORTED_TYPES = ("number", "integer", "string", "boolean")


def describe_call(function: FunctionSchema) -> str:
    """Render ``- name(a: number, b: string): description``.

    Used once the function is chosen, so the model knows which arguments it
    has to fill in.
    """
    parameters = ", ".join(
        f"{name}: {spec.get('type', 'unknown')}"
        for name, spec in function.parameters.items()
    )
    return f"- {function.name}({parameters}): {function.description}"


def prompt_spans(prompt: str) -> list[str]:
    """Every run of up to ``MAX_SPAN_WORDS`` words of the request.

    Each run is also offered without surrounding quotes and without trailing
    punctuation, so ``'hello'`` can be copied as ``hello``. These are only the
    pieces a string may copy; the model chooses which one, or writes a value
    of its own.
    """
    words = [match.span() for match in re.finditer(r"\S+", prompt)]
    spans: dict[str, None] = {}
    for first, (start, _) in enumerate(words):
        for _, end in words[first:first + MAX_SPAN_WORDS]:
            raw = prompt[start:end]
            for span in (raw, raw.strip("'\"`"),
                         raw.rstrip(".,;:!?").strip("'\"`")):
                if span:
                    spans[span] = None
    return list(spans)


def escape_state(text: str, pending: bool = False) -> bool | None:
    """Scan more of a JSON string body.

    Args:
        text: The next characters of the string.
        pending: Whether the text so far ended with an unfinished escape
            (a lone backslash).

    Returns:
        Whether an escape is unfinished after ``text``, or ``None`` if
        ``text`` cannot continue the string (an unescaped quote, a control
        character or an unknown escape).
    """
    for char in text:
        if pending:
            if char not in JSON_ESCAPES:
                return None
            pending = False
        elif char == "\\":
            pending = True
        elif char == '"' or ord(char) < 0x20:
            return None
    return pending


class ArgumentWriter(BaseModel):
    """Write the arguments of a function, one value at a time.

    Attributes:
        tokenizer: The SDK's tokenizer.
    """

    tokenizer: SDKTokenizer

    # The argument being written, or None while the name is written.
    _argument: str | None = PrivateAttr(default=None)
    # Token ids by kind, computed once from the vocabulary (see _token_sets).
    _plain: npt.NDArray[np.int64] | None = PrivateAttr(default=None)
    _after_escape: npt.NDArray[np.int64] | None = PrivateAttr(default=None)
    _open_tokens: npt.NDArray[np.int64] = PrivateAttr(
        default_factory=lambda: np.array([], dtype=np.int64)
    )
    _tricky: list[tuple[int, str]] = PrivateAttr(default_factory=list)

    # ----- token steps, written in the router (client.py) ------------------

    @abstractmethod
    def _pick(
        self,
        context: list[int],
        allowed: Sequence[int] | npt.NDArray[np.int64],
        labels: dict[int, str],
    ) -> int:
        """Add the best allowed token to ``context`` and return it."""

    @abstractmethod
    def _choose_text(
        self,
        context: list[int],
        choices: list[str],
        labels: dict[str, str] | None = None,
    ) -> str:
        """Let the model write exactly one of ``choices``."""

    @abstractmethod
    def _follow(
        self,
        context: list[int],
        sequences: dict[str, list[int]],
        written: list[int],
        labels: dict[str, str],
    ) -> str:
        """Continue writing one of seq from the tokens ``written``."""

    # ----- the arguments ---------------------------------------------------

    def write_arguments(
        self, prompt: str, function: FunctionSchema
    ) -> dict[str, Any]:
        """Let the model write the arguments of ``function`` for ``prompt``.

        The model sees only the chosen function, with its parameters, and
        continues ``{"name": "...", "parameters": {``. The code writes each
        key and the punctuation (no model call); the model writes each value
        within the rules of its type.

        Raises:
            ValueError: If a parameter has an unsupported type.
        """
        kinds: dict[str, str] = {}
        for name, spec in function.parameters.items():
            kind = spec.get("type")
            if kind not in SUPPORTED_TYPES:
                raise ValueError(f"Unsupported parameter type: {kind}")
            kinds[name] = str(kind)
        if not kinds:
            return {}
        names = list(kinds)

        def key(name: str) -> str:
            """``"name":``, without the space after the colon.

            Every value starts with its own space token (`` "``, `` "/``,
            `` true``, `` -``), so the model writes that space itself, as the
            tokenizer would split the text.
            """
            return f"{json.dumps(name)}:"

        head = f'{{"name": {json.dumps(function.name)}, "parameters": {{'
        context = chat(
            self.tokenizer,
            FILL_ARGUMENTS + describe_call(function),
            prompt,
            head + key(names[0]),
        )
        arguments: dict[str, Any] = {}
        for index, name in enumerate(names):
            # The text right after this value: the next key, or the end.
            after = (
                ", " + key(names[index + 1])
                if index + 1 < len(names) else "}}"
            )
            self._argument = name
            kind = kinds[name]
            if kind == "boolean":
                arguments[name] = self._write_boolean(context, after)
            elif kind == "string":
                arguments[name] = self._write_string(context, prompt, after)
            else:
                arguments[name] = self._write_number(
                    context, prompt, after, integer=kind == "integer"
                )
        self._argument = None
        return arguments

    def _write_boolean(self, context: list[int], after: str) -> bool:
        """The model writes ``true`` or ``false``."""
        choice = self._choose_text(
            context,
            [" true" + after, " false" + after],
            {" true" + after: "true", " false" + after: "false"},
        )
        return choice.startswith(" true")

    def _write_number(
        self, context: list[int], prompt: str, after: str, integer: bool
    ) -> int | float:
        """The model writes one of the numbers in the request.

        Each number is a choice, so the model only chooses where the numbers
        differ and the other digits are fast-forwarded. An ``integer`` is
        chosen among the whole numbers when there are any. If the request has
        no number, the model writes one digit by digit.

        Raises:
            ValueError: If an ``integer`` parameter gets a number that is
                not whole.
        """
        found = list(dict.fromkeys(NUMBER.findall(prompt)))
        whole = [text for text in found if "." not in text]
        numbers = whole if integer and whole else found
        if numbers:
            choice = self._choose_text(
                context,
                [f" {text}{after}" for text in numbers],
                {f" {text}{after}": text for text in numbers},
            )
            text = choice[1:-len(after)]
        else:
            text = self._write_free_number(context, after, integer)
        if integer:
            if "." in text:
                raise ValueError(f"Integer argument must be whole: {text}")
            return int(text)
        value = float(text)
        return int(value) if value.is_integer() else value

    def _write_free_number(
        self, context: list[int], after: str, integer: bool
    ) -> str:
        """The model writes a number one character at a time.

        Allowed: a space or a space with a minus sign first (`` -`` is one
        token), then digits, one decimal point (not for ``integer``), and the
        end once the number is complete, at most ``MAX_NUMBER_CHARS`` long.
        """
        def token(text: str) -> int:
            return self.tokenizer.encode(text)[0]

        digits = [token(char) for char in "0123456789"]
        end = self.tokenizer.encode(after)
        sign = self._pick(context, [token(" "), token(" -")], {})
        text = "-" if sign == token(" -") else ""
        while True:
            complete = text not in ("", "-") and not text.endswith(".")
            if complete and len(text) >= MAX_NUMBER_CHARS:
                allowed = [end[0]]
            else:
                allowed = list(digits)
                if not integer and complete and "." not in text:
                    allowed.append(token("."))
                if complete:
                    allowed.append(end[0])
            chosen = self._pick(context, allowed, {end[0]: "(end)"})
            if chosen == end[0]:
                context.extend(end[1:])
                return text
            text += self.tokenizer.token_text(chosen)

    def _write_string(
        self, context: list[int], prompt: str, after: str
    ) -> str:
        """The model copies a piece of the request or writes a string.

        The value starts with a token that opens the quote (`` "`` or a
        merged one such as `` "/``). If the model's first content token
        starts one of the request's pieces (``prompt_spans``), it keeps
        copying: inside a word only one token is possible, so it is
        fast-forwarded, and the model only decides at which word to stop.
        Otherwise it writes freely until it closes the string.
        """
        closing = '"' + after
        sequences: dict[str, list[int]] = {}
        labels: dict[str, str] = {}
        for span in prompt_spans(prompt):
            body = json.dumps(span, ensure_ascii=False)[1:-1]
            text = ' "' + body + closing
            sequences[text] = self.tokenizer.encode(text)
            labels[text] = span

        # The string ends with a lone '"', or with the token that joins the
        # quote to what follows it (such as '",').
        closing_ids = self.tokenizer.encode(closing)
        quote = self.tokenizer.encode('"')[0]
        ends = [closing_ids[0], quote]
        self._token_sets()
        opening = self.tokenizer.encode(' "')[0]

        written: list[int] = []
        while True:
            # Next tokens of the pieces that still match what is written.
            copy: dict[int, list[str]] = {}
            for text, ids in sequences.items():
                if ids[:len(written)] == written and len(ids) > len(written):
                    copy.setdefault(ids[len(written)], []).append(labels[text])
            if written:  # just the opening quote so far
                free = self._string_tokens(False, ends)
            else:
                free = self._open_tokens
            allowed = np.union1d(free, list(copy))
            labels_now = {token: ", ".join(spans)
                          for token, spans in copy.items()}
            labels_now.update({end: "(end)" for end in ends})
            if not written:
                labels_now[opening] = "(opening quote)"
            token = self._pick(context, allowed, labels_now)
            if written and token in ends:  # an empty string
                break
            written.append(token)
            if token in copy and token != opening:  # copy from the request
                return labels[self._follow(context, sequences, written,
                                           labels)]
            if token != opening:  # the model writes its own value
                token = self._write_free_string(context, written, ends)
                break
        context.extend(
            closing_ids[1:] if token == closing_ids[0]
            else self.tokenizer.encode(after)
        )
        body = self.tokenizer.decode(written)[2:]  # without ' "'
        try:
            return str(json.loads('"' + body + '"'))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid string argument: {error}") from error

    def _write_free_string(
        self, context: list[int], written: list[int], ends: list[int]
    ) -> int:
        """Let the model continue a string until it closes it.

        ``written`` holds the tokens so far, starting with the opening; it is
        extended here. Returns the token that closed the string.
        """
        text = self.tokenizer.decode(written)[2:]
        pending = escape_state(text)
        while True:
            if pending is None:  # cannot happen: only valid tokens allowed
                raise ValueError("Invalid string argument")
            done = len(written) > MAX_VALUE_TOKENS
            allowed = self._string_tokens(pending, ends, only_end=done)
            token = self._pick(context, allowed, {e: "(end)" for e in ends})
            if not pending and token in ends:
                return token
            written.append(token)
            pending = escape_state(self.tokenizer.token_text(token), pending)

    def _string_tokens(
        self, pending: bool, ends: list[int], only_end: bool = False
    ) -> npt.NDArray[np.int64]:
        """The token ids allowed next inside a JSON string.

        Args:
            pending: The string so far ends with a lone backslash.
            ends: The tokens that close the string (not allowed while an
                escape is unfinished).
            only_end: Allow only closing (the value is long enough).
        """
        self._token_sets()
        end_ids = [] if pending else ends
        if only_end and end_ids:
            return np.array(end_ids, dtype=np.int64)
        tricky = [
            token for token, text in self._tricky
            if escape_state(text, pending) is not None
        ]
        base = self._after_escape if pending else self._plain
        if base is None:  # cannot happen: _token_sets filled it
            raise RuntimeError("Token sets are not ready")
        return np.union1d(base, np.array(tricky + end_ids, dtype=np.int64))

    def _token_sets(self) -> None:
        """Sort the vocabulary once by what it can do inside a JSON string.

        ``_plain``: no quote, backslash or control character, so allowed
        anywhere; ``_after_escape``: plain tokens that may follow a
        backslash; ``_tricky``: tokens with a quote or a backslash, checked
        at each step; ``_open_tokens``: tokens that open a string value
        (`` "`` followed by a valid start, such as `` "/``). Tokens with a
        control character are never allowed.
        """
        if self._plain is not None:
            return
        plain: list[int] = []
        after_escape: list[int] = []
        opening: list[int] = []
        for token in sorted(self.tokenizer.vocab.values()):
            text = self.tokenizer.token_text(token)
            if not text or any(ord(char) < 0x20 for char in text):
                continue
            if '"' in text or "\\" in text:
                self._tricky.append((token, text))
                opens = text.startswith(' "')
                if opens and escape_state(text[2:]) is not None:
                    opening.append(token)
                continue
            plain.append(token)
            if text[0] in JSON_ESCAPES:
                after_escape.append(token)
        self._plain = np.array(plain, dtype=np.int64)
        self._after_escape = np.array(after_escape, dtype=np.int64)
        self._open_tokens = np.array(opening, dtype=np.int64)
