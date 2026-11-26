"""Choose the function with constrained decoding; read its arguments with
regular expressions (``regex_arguments.py``).
"""

from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator

from src.keyword_ranker import shortlist_functions
from src.regex_arguments import extract_arguments
from src.schema import FunctionCall, FunctionSchema
from src.tokenizer import SDKTokenizer, chat

PICK_FUNCTION = (
    "You call functions. Pick the one function whose description matches "
    "what the user asks for, and reply with the call as JSON.\n\nFunctions:\n"
)


def describe(function: FunctionSchema) -> str:
    """Render a function as ``- name: description`` for the model.

    The parameter list is left out: the model only chooses the name, and the
    shorter prompt makes every model call faster.
    """
    return f"- {function.name}: {function.description}"


class TokenAwareConstrainedRouter(BaseModel):
    """Choose the function with constrained decoding, then read its arguments.

    Attributes:
        llm: The SDK model; only ``get_logits_from_input_ids`` is used.
        tokenizer: The SDK's tokenizer.
        trace_enabled: Record every decoding step in ``last_trace``.
        use_ranker: Shortlist the functions with the keyword ranker before
            the model chooses; when it keeps one, the model is not called.
            Off by default: the model chooses among all.
        last_trace: Steps of the latest call, when tracing is enabled.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Any
    tokenizer: SDKTokenizer
    trace_enabled: bool = False
    use_ranker: bool = False
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
            ValueError: If no functions are given, a parameter has an
                unsupported type, or an ``integer`` gets a number that is not
                whole.
        """
        if not functions:
            raise ValueError("No function definitions were provided")
        self._steps, self._calls = [], 0

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
        # Regular expressions read the arguments, with no model call.
        parameters = extract_arguments(prompt, function)

        if self.trace_enabled:
            self.last_trace = {
                "prompt": prompt,
                "prompt_tokens": [
                    self.tokenizer.token_text(i)
                    for i in self.tokenizer.encode(prompt)
                ],
                "function_count": len(functions),
                "ranker": self.use_ranker,
                "shortlist": [f.name for f in candidates],
                "selected": function.name,
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
            text: self.tokenizer.encode(text)
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
        allowed: Sequence[int],
        labels: dict[int, str],
    ) -> int:
        """Add the best allowed token to ``context`` and return it.

        Args:
            context: Token ids so far (prompt plus output); extended here.
            allowed: The ids allowed at this step.
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
        masked[ids] = logits[ids]
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
        """Remember a step for the trace."""
        if self.trace_enabled:
            self._steps.append(step)
