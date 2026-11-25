"""The model's own tokenizer, used through the SDK's public methods.

``encode`` and ``decode`` call the SDK's ``Small_LLM_Model.encode`` and
``decode``. The SDK also gives the paths of ``vocab.json`` (every regular
token and its id) and ``tokenizer.json`` (the added tokens such as
``<|im_start|>``): the constrained decoder needs them to know which tokens
exist.

``chat`` writes a prompt in the model's chat format (as Hugging Face's
``apply_chat_template`` does) and encodes it.
"""

import json
from typing import Any

from pydantic import BaseModel, ConfigDict


def _load_special_tokens(path: str) -> dict[str, int]:
    """Read the added tokens (``<|im_start|>``...) from ``tokenizer.json``."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        token["content"]: int(token["id"])
        for token in data.get("added_tokens", [])
    }


class SDKTokenizer(BaseModel):
    """Encode and decode text with the SDK model's tokenizer.

    Attributes:
        llm: The SDK model; only its public ``encode`` and ``decode`` are
            used.
        vocab: Token string to token id, from ``vocab.json``.
        special_tokens: Added tokens such as ``<|im_start|>``, from
            ``tokenizer.json``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Any
    vocab: dict[str, int]
    special_tokens: dict[str, int] = {}

    @classmethod
    def from_model(cls, model: Any) -> "SDKTokenizer":
        """Read the vocabulary and the added tokens the SDK model points to.

        Raises:
            ValueError: If the model has no ``vocab.json`` (for example
                SentencePiece models).
        """
        try:
            path = model.get_path_to_vocab_file()
            with open(path, encoding="utf-8") as handle:
                vocab = json.load(handle)
        except Exception as error:  # SDK download errors are not OSError
            raise ValueError(
                f"Model has no vocab.json vocabulary: {error}"
            ) from error
        if not isinstance(vocab, dict):
            raise ValueError("Vocabulary file must map tokens to ids")
        try:
            special = _load_special_tokens(model.get_path_to_tokenizer_file())
        except Exception:  # optional: without it, no Qwen3 markers are used
            special = {}
        return cls(llm=model, vocab=vocab, special_tokens=special)

    @property
    def vocab_size(self) -> int:
        """Number of regular (non-added) tokens."""
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        """Convert text to token ids with the SDK, adding no extra tokens.

        The SDK turns the text of an added token, such as ``<|im_start|>``,
        into that token's single id, wherever it appears, also inside a
        user's request.
        """
        if not text:
            return []
        return [int(token) for token in self.llm.encode(text)[0].tolist()]

    def decode(self, ids: list[int]) -> str:
        """Convert token ids back to text with the SDK.

        Added tokens are skipped (the SDK decodes with
        ``skip_special_tokens=True``).
        """
        return str(self.llm.decode(list(ids)))

    def token_text(self, token_id: int) -> str:
        """Return the text of a single token (may be a partial character)."""
        return self.decode([token_id])


def chat(tokenizer: SDKTokenizer, system: str, user: str, answer: str
         ) -> list[int]:
    """Encode a prompt in Qwen's chat format, ending inside the answer.

    Each marker (``<|im_start|>``, ``<|im_end|>``...) is followed by its
    text, and both are encoded by the SDK's tokenizer. Qwen3 also gets an
    empty ``<think>`` block (no thinking) and its ``<tool_call>`` marker when
    the model has them.

    The SDK's tokenizer turns a marker's text into its token wherever it
    appears, so a request that contains ``<|im_start|>`` is encoded with the
    real marker too.
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
        + tokenizer.encode(text)
    ]
