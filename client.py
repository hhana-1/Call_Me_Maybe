"""Choose the function, then extract its arguments with regex.

1. **Shortlist, keyword ranking**: ``choose_tokenizer.rank_schema_candidates``
   scores every function by shared words and domains and keeps the best.
2. **Function name, constrained decoding**: the prompt lists the shortlisted
   functions and the model writes the name one token at a time. Only tokens
   that continue a shortlisted name are allowed: every other logit is set to
   minus infinity. When just one token is allowed it is added without
   calling the model.
3. **Arguments, regular expressions**: numbers, quoted strings, paths and
   booleans are read from the prompt in the order the schema lists them.
"""

import json
import re
from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator

from choose_tokenizer import rank_schema_candidates
from src.schema import FunctionCall, FunctionSchema
from src.tokenizer import BPETokenizer

PICK_FUNCTION = (
    "You call functions. Pick the one function whose description matches "
    "what the user asks for, and reply with the call as JSON.\n\nFunctions:\n"
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


class TokenAwareConstrainedRouter(BaseModel):
    """Choose functions with constrained decoding; extract their arguments.

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

    _steps: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _calls: int = PrivateAttr(default=0)

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
        self._steps, self._calls = [], 0

        # The keyword ranker shortlists the functions with the best score.
        shortlist = {
            schema["name"] for schema in rank_schema_candidates(
                prompt, [function.model_dump() for function in functions]
            )
        }
        candidates = [f for f in functions if f.name in shortlist]

        # The model writes one of the shortlisted names. The closing quote
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
        parameters = self.extract_arguments(prompt, function)

        if self.trace_enabled:
            self.last_trace = {
                "prompt": prompt,
                "prompt_tokens": [
                    self.tokenizer.token_text(i)
                    for i in self.tokenizer.encode(prompt)
                ],
                "function_count": len(functions),
                "shortlist": [f.name for f in candidates],
                "selected": function.name,
                "steps": self._steps,
                "model_calls": self._calls,
            }
        return FunctionCall(
            prompt=prompt, name=function.name, parameters=parameters
        )

    # ----- the function name: constrained decoding ------------------------

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
            written.append(self._pick(context, list(branches), labels))

    def _pick(
        self, context: list[int], allowed: list[int], labels: dict[int, str]
    ) -> int:
        """Add the best allowed token to ``context`` and return it.

        Args:
            context: Token ids so far (prompt plus output); extended here.
            allowed: The ids allowed at this step.
            labels: What each allowed token stands for, shown in the trace.
        """
        if len(allowed) == 1:  # forced: no need to ask the model
            token = allowed[0]
            context.append(token)
            self._record({"text": self.tokenizer.token_text(token),
                          "forced": True})
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
                "forced": False,
                "allowed": len(allowed),
                "probability": float(1 / weights.sum()),
                "options": [
                    {
                        "text": self.tokenizer.token_text(int(ids[i])),
                        "label": labels[int(ids[i])],
                        "probability": float(weights[i] / weights.sum()),
                        "chosen": int(ids[i]) == token,
                    }
                    for i in np.argsort(-weights)[:5]
                ],
            })
        return token

    def _record(self, step: dict[str, Any]) -> None:
        """Remember a step for the trace."""
        if self.trace_enabled:
            self._steps.append({"field": "name", **step})

    # ----- the arguments: regular expressions -----------------------------

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
        """Evaluate simple even/odd questions without using the model."""
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


def load_functions(path: str) -> list[FunctionSchema]:
    """Load and validate function definitions from JSON."""
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError("Function definitions must be a JSON array")
    return [FunctionSchema.model_validate(item) for item in raw]
