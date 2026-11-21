"""Choose the function and write its arguments with constrained decoding.

1. **Shortlist, keyword ranking** (optional, off by default):
   ``keyword_ranker.shortlist_functions`` scores every function by shared
   words and domains and keeps the best. When it is off, every function is a
   candidate.
2. **Function name, constrained decoding**: the prompt lists the candidate
   functions and the model writes the name one token at a time. Only tokens
   that continue a candidate's name are allowed: every other logit is set to
   minus infinity. When just one token is allowed it is fast-forwarded:
   added without calling the model.
3. **Arguments, constrained decoding**: the model continues the same JSON,
   ``"parameters": {...}``. The code writes the keys and punctuation from the
   schema; the model writes each value, limited by the parameter's type:
   a number from the request, a string copied from the request or written
   freely, ``true`` or ``false``. With ``regex_arguments`` the values are read
   from the request with regular expressions instead.
"""

import json
import re
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator

from src.keyword_ranker import shortlist_functions
from src.schema import FunctionCall, FunctionSchema
from src.tokenizer import BPETokenizer

PICK_FUNCTION = (
    "You call functions. Pick the one function whose description matches "
    "what the user asks for, and reply with the call as JSON.\n\nFunctions:\n"
)
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


def describe(function: FunctionSchema) -> str:
    """Render a function as ``- name: description`` for the model.

    The parameter list is left out: the model only chooses the name, and the
    shorter prompt makes every model call faster.
    """
    return f"- {function.name}: {function.description}"


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


def chat(tokenizer: BPETokenizer, system: str, user: str, answer: str
         ) -> list[int]:
    """Encode a prompt in Qwen's chat format, ending inside the answer.

    Each marker is followed by its text. Only the markers become
    special-token ids: the system text, the request and the answer start are
    encoded as plain text, so a request containing ``<|im_start|>`` cannot
    open a new turn. Qwen3 also gets an empty ``<think>`` block (no
    thinking) and its ``<tool_call>`` marker when the model has them.
    """
    parts = [
        ("<|im_start|>", f"system\n{system}"),
        ("<|im_end|>", "\n"),
        ("<|im_start|>", f"user\n{user}"),
        ("<|im_end|>", "\n"),
        ("<|im_start|>", "assistant\n"),
    ]
    if "<think>" in tokenizer.special_tokens:
        parts += [("<think>", "\n\n"), ("</think>", "\n\n")]
    if "<tool_call>" in tokenizer.special_tokens:
        parts.append(("<tool_call>", "\n"))
    marker, text = parts.pop()
    parts.append((marker, text + answer))  # the answer continues that text
    return [
        token_id
        for marker, text in parts
        for token_id in tokenizer.encode(marker)
        + tokenizer.encode(text, allow_special=False)
    ]


class TokenAwareConstrainedRouter(BaseModel):
    """Choose functions with constrained decoding; extract their arguments.

    Attributes:
        llm: The SDK model; only ``get_logits_from_input_ids`` is used.
        tokenizer: The project's BPE tokenizer.
        trace_enabled: Record every decoding step in ``last_trace``.
        use_ranker: Shortlist the functions with the keyword ranker before
            the model chooses. Off by default: the model chooses among all.
        regex_arguments: Read the arguments with regular expressions instead
            of letting the model write them. Off by default.
        last_trace: Steps of the latest call, when tracing is enabled.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Any
    tokenizer: BPETokenizer
    trace_enabled: bool = False
    use_ranker: bool = False
    regex_arguments: bool = False
    last_trace: dict[str, Any] | None = None

    _steps: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _calls: int = PrivateAttr(default=0)
    # The argument being written, or None while the name is written.
    _argument: str | None = PrivateAttr(default=None)
    # Token ids by kind, computed once from the vocabulary (see _token_sets).
    _plain: npt.NDArray[np.int64] | None = PrivateAttr(default=None)
    _after_escape: npt.NDArray[np.int64] | None = PrivateAttr(default=None)
    _open_tokens: npt.NDArray[np.int64] = PrivateAttr(
        default_factory=lambda: np.array([], dtype=np.int64)
    )
    _tricky: list[tuple[int, str]] = PrivateAttr(default_factory=list)

    @field_validator("llm")
    @classmethod
    def _check_llm(cls, value: Any) -> Any:
        """Require the SDK method the router relies on."""
        if not callable(getattr(value, "get_logits_from_input_ids", None)):
            raise ValueError("Model must provide get_logits_from_input_ids()")
        return value

    def generate_call(
        self, prompt: str, functions: Sequence[FunctionSchema]
    ) -> FunctionCall:
        """Generate one validated function call for ``prompt``.

        Raises:
            ValueError: If no functions are given, or an argument cannot be
                found in the prompt.
        """
        if not functions:
            raise ValueError("No function definitions were provided")
        self._steps, self._calls, self._argument = [], 0, None

        if self.use_ranker:
            # The keyword ranker shortlists the functions with the best score.
            # Func defs are cp,exists only while shortlist_functions runs
            shortlist = {
                schema["name"] for schema in shortlist_functions(
                    prompt, [function.model_dump() for function in functions]
                )
            }
            candidates = [f for f in functions if f.name in shortlist]
        else:
            # Without the ranker the model chooses among all the functions.
            candidates = list(functions)

        # The model writes one of the candidates' names. The closing quote
        # makes every name end differently, even when one name starts
        # another (fn_read, fn_read_file).
        context = chat(
            self.tokenizer,
            PICK_FUNCTION + "\n".join(map(describe, candidates)),
            prompt,
            '{"name": "',
        )
        name = self._choose_text(context, [f'{f.name}"' for f in candidates])
        function = next(f for f in candidates if f'{f.name}"' == name)
        if self.regex_arguments:
            parameters = self.extract_arguments(prompt, function)
        else:
            parameters = self.write_arguments(prompt, function)

        if self.trace_enabled:
            self.last_trace = {
                "prompt": prompt,
                "prompt_tokens": [
                    self.tokenizer.token_text(i)
                    for i in self.tokenizer.encode(prompt, allow_special=False)
                ],
                "function_count": len(functions),
                "ranker": self.use_ranker,
                "shortlist": [f.name for f in candidates],
                "selected": function.name,
                "arguments_by": "regex" if self.regex_arguments else "model",
                "steps": self._steps,
                "model_calls": self._calls,
            }
        return FunctionCall(
            prompt=prompt, name=function.name, parameters=parameters
        )

    def _choose_text(
        self,
        context: list[int],
        choices: list[str],
        labels: dict[str, str] | None = None,
    ) -> str:
        """Let the model write exactly one of ``choices``.

        The choices' token sequences form a trie: at each step only the next
        tokens of the choices that still match are allowed.

        Args:
            context: Token ids so far; extended with the written tokens.
            choices: The texts the model may write.
            labels: How each choice is shown in the trace (default: the
                text without spaces and quotes around it).
        """
        sequences = {
            text: self.tokenizer.encode(text, allow_special=False)
            for text in choices
        }
        if labels is None:
            labels = {text: text.strip(' "') for text in choices}
        return self._follow(context, sequences, [], labels)

    def _follow(
        self,
        context: list[int],
        sequences: dict[str, list[int]],
        written: list[int],
        labels: dict[str, str],
    ) -> str:
        """Continue writing one of ``sequences`` from the tokens ``written``.

        Returns the text whose tokens have all been written.
        """
        written = list(written)
        while True:
            branches: dict[int, list[str]] = {}
            for text, ids in sequences.items():
                if ids[:len(written)] != written:
                    continue
                if len(ids) == len(written):
                    return text
                branches.setdefault(ids[len(written)], []).append(text)
            if not branches:  # cannot happen when written came from here
                raise ValueError("No choice continues the written tokens")
            token_labels = {
                token: ", ".join(labels[text] for text in texts)
                for token, texts in branches.items()
            }
            written.append(self._pick(context, list(branches), token_labels))

    def _pick(
        self,
        context: list[int],
        allowed: Sequence[int] | npt.NDArray[np.int64],
        labels: dict[int, str],
    ) -> int:
        """Add the best allowed token to ``context`` and return it.

        Args:
            context: Token ids so far (prompt plus output); extended here.
            allowed: The ids allowed at this step (a list or a NumPy array).
            labels: What allowed tokens stand for, shown in the trace.
        """
        if len(allowed) == 1:  # fast-forward: no need to ask the model
            token = int(allowed[0])
            context.append(token)
            self._record({"text": self.tokenizer.token_text(token),
                          "fast_forward": True})
            return token
        logits = np.asarray(
            self.llm.get_logits_from_input_ids(context), dtype=np.float64
        )
        ids = np.array(allowed, dtype=np.int64)
        masked = np.full(logits.shape, -np.inf)
        masked[ids] = logits[ids]  # every other token becomes impossible
        token = int(np.argmax(masked))
        context.append(token)
        self._calls += 1
        if self.trace_enabled:
            weights = np.exp(logits[ids] - logits[token])
            self._record({
                "text": self.tokenizer.token_text(token),
                "fast_forward": False,
                "allowed": len(allowed),
                "probability": float(1 / weights.sum()),
                "options": [
                    {
                        "text": self.tokenizer.token_text(int(ids[i])),
                        "label": labels.get(int(ids[i]), ""),
                        "probability": float(weights[i] / weights.sum()),
                        "chosen": int(ids[i]) == token,
                    }
                    for i in np.argsort(-weights)[:5]
                ],
            })
        return token

    def _record(self, step: dict[str, Any]) -> None:
        """Remember a step for the trace.

        ``argument`` is the parameter being written, or None for the name.
        """
        if self.trace_enabled:
            self._steps.append({"argument": self._argument, **step})

    # ----- the arguments: constrained decoding -----------------------------

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
            return self.tokenizer.encode(text, allow_special=False)[0]

        digits = [token(char) for char in "0123456789"]
        end = self.tokenizer.encode(after, allow_special=False)
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
            sequences[text] = self.tokenizer.encode(text, allow_special=False)
            labels[text] = span

        # The string ends with a lone '"', or with the token that joins the
        # quote to what follows it (such as '",').
        closing_ids = self.tokenizer.encode(closing, allow_special=False)
        quote = self.tokenizer.encode('"', allow_special=False)[0]
        ends = [closing_ids[0], quote]
        self._token_sets()
        opening = self.tokenizer.encode(' "', allow_special=False)[0]

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
            else self.tokenizer.encode(after, allow_special=False)
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

    # ----- the arguments: regular expressions (--regex-arguments) ----------

    def extract_arguments(
        self, prompt: str, function: FunctionSchema
    ) -> dict[str, Any]:
        """Extract values conservatively according to the function schema."""
        numbers = [float(value) for value in re.findall(
            r"(?<![\w.])-?\d+(?:\.\d+)?", prompt
        )]
        quoted = [match[1] for match in re.findall(
            r"(['\"])(.*?)(?<!\\)\1", prompt
        )]
        arguments: dict[str, Any] = {}
        number_index = 0
        string_index = 0

        for name, specification in function.parameters.items():
            parameter_type = specification.get("type")
            if parameter_type == "number":
                # Numbers in prompt order, ignoring digits inside words.
                if number_index >= len(numbers):
                    raise ValueError(f"Missing number argument: {name}")
                value = numbers[number_index]
                arguments[name] = int(value) if value.is_integer() else value
                number_index += 1
            elif parameter_type == "integer":
                if number_index >= len(numbers):
                    raise ValueError(f"Missing integer argument: {name}")
                value = numbers[number_index]
                if not value.is_integer():
                    raise ValueError(f"Integer argument must be whole: {name}")
                arguments[name] = int(value)
                number_index += 1
            elif parameter_type == "string":
                arguments[name] = self._string_argument(
                    prompt, name, quoted, string_index
                )
                string_index += 1
            elif parameter_type == "boolean":
                match = re.search(r"\b(true|false)\b", prompt, re.IGNORECASE)
                if match is not None:
                    arguments[name] = match.group(1).lower() == "true"
                else:
                    math_result = self._math_boolean_result(prompt)
                    if math_result is None:
                        raise ValueError(f"Missing boolean argument: {name}")
                    arguments[name] = math_result
            else:
                raise ValueError(
                    f"Unsupported parameter type: {parameter_type}"
                )

        return arguments

    @staticmethod
    def _math_boolean_result(prompt: str) -> bool | None:
        """Evaluate simple even/odd questions."""
        match = re.search(
            r"\b(?:is|are)\s+(-?\d+(?:\.\d+)?)\s+(?:a|an)\s+"
            r"(even|odd)(?:\s+number)?\b",
            prompt,
            re.IGNORECASE,
        )
        if match is None:
            return None
        value = float(match.group(1))
        if not value.is_integer():
            raise ValueError("Even/odd checks require an integer")
        is_even = int(value) % 2 == 0
        return is_even if match.group(2).lower() == "even" else not is_even

    @staticmethod
    def _string_argument(
        prompt: str, name: str, quoted: list[str], index: int
    ) -> str:
        """Handle quoted values and the subject's natural-language patterns."""
        if name == "regex":
            if re.search(r"number", prompt, re.IGNORECASE):
                return r"\d+"
            if re.search(r"vowel", prompt, re.IGNORECASE):
                return r"[aeiou]"
            match = re.search(r"word ['\"]([^'\"]+)", prompt, re.IGNORECASE)
            if match:
                return re.escape(match.group(1))
        if name == "replacement":
            match = re.search(
                r"with\s+(.+?)(?:\s+in\s+|$)", prompt, re.IGNORECASE
            )
            if match:
                return match.group(1).strip(" '\"")
            if re.search(r"asterisk", prompt, re.IGNORECASE):
                return "*"
        if name == "source_string":
            match = re.search(r"in\s+(['\"])(.*?)\1", prompt, re.IGNORECASE)
            if match:
                return match.group(2)
        if name == "path":
            path_candidates: list[str] = [
                token.strip(" '\"")
                for token in re.findall(r"[^\s,;]+", prompt)
                if TokenAwareConstrainedRouter._looks_like_path(token)
            ]
            if path_candidates:
                return path_candidates[0]
        if name == "encoding":
            match = re.search(
                r"(?:with|using)\s+([A-Za-z0-9_.-]+)\s+encoding\b|"
                r"\bencoding\s*[:=]\s*([A-Za-z0-9_.-]+)",
                prompt,
                re.IGNORECASE,
            )
            if match:
                return (match.group(1) or match.group(2)).lower()
        if index < len(quoted):
            return quoted[index]
        words: list[str] = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", prompt)
        if not words:
            raise ValueError(f"Missing string argument: {name}")
        return words[-1]

    @staticmethod
    def _looks_like_path(value: str) -> bool:
        """Recognize common path shapes across operating systems."""
        return bool(
            re.fullmatch(
                r"(?:[A-Za-z]:[\\/][^\s,;]+|"
                r"(?:/|\./|\.\./|~/)[^\s,;]+|"
                r"[\w.-]+(?:[\\/][\w .-]+)+|"
                r"[\w.-]+\.[A-Za-z0-9]+)",
                value.strip(" '\""),
            )
        )
