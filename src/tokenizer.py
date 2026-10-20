"""Byte-level BPE tokenizer rebuilt from the model's own vocabulary files.

The SDK only hands us file paths (``get_path_to_vocab_file`` and
``get_path_to_merges_file``); everything else -- normalizing, splitting,
merging and mapping to ids -- happens here, so the main code never calls
the SDK's ``encode`` or ``decode``.
"""

import json
import re
import sys
import unicodedata
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, PrivateAttr


def _unicode_classes() -> tuple[str, str]:
    """Build regex class bodies for Unicode letters (L*) and numbers (N*).

    The standard ``re`` module has no ``\\p{L}`` / ``\\p{N}``, so the code
    points are collected from ``unicodedata`` once and compressed to ranges.
    """
    ranges: dict[str, list[list[int]]] = {"L": [], "N": []}
    for code in range(sys.maxunicode + 1):
        group = ranges.get(unicodedata.category(chr(code))[0])
        if group is None:
            continue
        if group and group[-1][1] == code - 1:
            group[-1][1] = code
        else:
            group.append([code, code])

    def body(spans: list[list[int]]) -> str:
        """Join code-point ranges into a regex class body like ``a-z``."""
        return "".join(
            re.escape(chr(start))
            if start == end
            else f"{re.escape(chr(start))}-{re.escape(chr(end))}"
            for start, end in spans
        )

    return body(ranges["L"]), body(ranges["N"])


@lru_cache(maxsize=1)
def _pre_tokenizer() -> "re.Pattern[str]":
    """Compile the Qwen2/Qwen3 pre-tokenizer pattern with plain ``re``.

    Original (from ``tokenizer.json``)::

        (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}
        | ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+

    Numbers match one digit at a time, so ``265`` becomes ``2``, ``6``, ``5``.
    """
    letters, numbers = _unicode_classes()
    return re.compile(
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
        rf"|[^\r\n{letters}{numbers}]?[{letters}]+"
        rf"|[{numbers}]"
        rf"| ?[^\s{letters}{numbers}]+[\r\n]*"
        r"|\s*[\r\n]+"
        r"|\s+(?!\S)"
        r"|\s+"
    )


def _load_special_tokens(path: str) -> dict[str, int]:
    """Read the added tokens (``<|im_start|>``...) from ``tokenizer.json``."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        token["content"]: int(token["id"])
        for token in data.get("added_tokens", [])
    }


def _bytes_to_unicode() -> dict[int, str]:
    """Map every byte to a printable character, as GPT-2 style BPE does.

    Printable Latin-1 bytes map to themselves; the rest (space, control
    characters, ...) are shifted above U+0100, which is why a space shows up
    as ``Ġ`` and a newline as ``Ċ`` in the vocabulary.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    mapping = {byte: chr(byte) for byte in printable}
    shift = 0
    for byte in range(256):
        if byte not in mapping:
            mapping[byte] = chr(256 + shift)
            shift += 1
    return mapping


class BPETokenizer(BaseModel):
    """Encode and decode text with the model's byte-level BPE rules.

    Attributes:
        vocab: Token string to token id, from ``vocab.json``.
        merges: Ordered merge rules from ``merges.txt``; earlier rules win.
        special_tokens: Added tokens such as ``<|im_start|>``, from
            ``tokenizer.json``; each is always encoded as its single id.
    """

    vocab: dict[str, int]
    merges: list[tuple[str, str]]
    special_tokens: dict[str, int] = {}

    _ranks: dict[tuple[str, str], int] = PrivateAttr(default_factory=dict)
    _id_to_token: dict[int, str] = PrivateAttr(default_factory=dict)
    _byte_encoder: dict[int, str] = PrivateAttr(default_factory=dict)
    _byte_decoder: dict[str, int] = PrivateAttr(default_factory=dict)
    _cache: dict[str, list[int]] = PrivateAttr(default_factory=dict)
    _special_pattern: "re.Pattern[str] | None" = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Build the lookup tables once, after pydantic validation."""
        self._ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        if self.special_tokens:
            longest_first = sorted(self.special_tokens, key=len, reverse=True)
            self._special_pattern = re.compile(
                "|".join(re.escape(token) for token in longest_first)
            )
        self._id_to_token = {
            token_id: token for token, token_id in self.vocab.items()
        }
        self._byte_encoder = _bytes_to_unicode()
        self._byte_decoder = {
            char: byte for byte, char in self._byte_encoder.items()
        }

    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str) -> "BPETokenizer":
        """Load ``vocab.json`` and ``merges.txt``.

        Args:
            vocab_path: Path to the JSON token-to-id mapping.
            merges_path: Path to the ranked BPE merge rules.

        Raises:
            ValueError: If the files are not a byte-level BPE vocabulary.
        """
        try:
            with open(vocab_path, encoding="utf-8") as handle:
                vocab = json.load(handle)
        except UnicodeDecodeError as error:
            raise ValueError(
                f"Unsupported tokenizer: {vocab_path} is not a JSON "
                "byte-level BPE vocabulary"
            ) from error
        if not isinstance(vocab, dict):
            raise ValueError("Vocabulary file must map tokens to ids")

        merges: list[tuple[str, str]] = []
        with open(merges_path, encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if line.startswith("#version") or len(parts) != 2:
                    continue
                merges.append((parts[0], parts[1]))
        return cls(vocab=vocab, merges=merges)

    @classmethod
    def from_model(cls, model: Any) -> "BPETokenizer":
        """Build the tokenizer from the files the SDK model points to.

        ``vocab.json`` and ``merges.txt`` are required. The special tokens are
        read from ``tokenizer.json`` when the model has one.

        Raises:
            ValueError: If the model has no byte-level BPE files (for
                example SentencePiece models, which ship no ``merges.txt``).
        """
        try:
            vocab_path = model.get_path_to_vocab_file()
            merges_path = model.get_path_to_merges_file()
        except Exception as error:  # SDK download errors are not OSError
            raise ValueError(
                "Model has no byte-level BPE tokenizer files "
                f"(vocab.json + merges.txt): {error}"
            ) from error
        tokenizer = cls.from_files(vocab_path, merges_path)
        try:
            special = _load_special_tokens(model.get_path_to_tokenizer_file())
        except Exception:  # optional: without it, specials stay plain text
            return tokenizer
        return cls(
            vocab=tokenizer.vocab,
            merges=tokenizer.merges,
            special_tokens=special,
        )

    @property
    def vocab_size(self) -> int:
        """Number of regular (non-special) tokens."""
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        """Convert text to token ids without adding special tokens.

        The text is first normalized to Unicode NFC, as Qwen's tokenizer
        does, so a decomposed character such as ``G`` + U+0307 becomes
        ``Ġ`` before splitting.

        Args:
            text: Any string.

        Returns:
            The token ids the model expects for ``text``.
        """
        ids: list[int] = []
        start = 0
        if self._special_pattern is not None:
            for match in self._special_pattern.finditer(text):
                ids.extend(self._encode_text(text[start:match.start()]))
                ids.append(self.special_tokens[match.group()])
                start = match.end()
        ids.extend(self._encode_text(text[start:]))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Convert token ids back to text.

        Special-token ids are skipped, matching the SDK's
        ``skip_special_tokens=True``.
        """
        pieces = "".join(self._id_to_token.get(index, "") for index in ids)
        raw = bytes(self._byte_decoder[char] for char in pieces)
        return raw.decode("utf-8", errors="replace")

    def token_text(self, token_id: int) -> str:
        """Return the text of a single token (may be a partial character)."""
        return self.decode([token_id])

    def _encode_text(self, text: str) -> list[int]:
        """Encode text that contains no special tokens."""
        ids: list[int] = []
        normalized = unicodedata.normalize("NFC", text)
        for chunk in _pre_tokenizer().findall(normalized):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def _encode_chunk(self, chunk: str) -> list[int]:
        """Encode one pre-tokenized chunk, reusing earlier results."""
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        symbols = [self._byte_encoder[byte] for byte in chunk.encode("utf-8")]
        ids: list[int] = []
        for piece in self._merge(symbols):
            token_id = self.vocab.get(piece)
            if token_id is None:
                raise ValueError(f"Token {piece!r} missing from vocabulary")
            ids.append(token_id)
        self._cache[chunk] = ids
        return ids

    def _merge(self, symbols: list[str]) -> list[str]:
        """Repeatedly merge the best-ranked adjacent pair until none apply."""
        while len(symbols) > 1:
            pairs = list(zip(symbols, symbols[1:]))
            best = min(
                pairs, key=lambda pair: self._ranks.get(pair, len(self._ranks))
            )
            if best not in self._ranks:
                break
            merged: list[str] = []
            index = 0
            while index < len(symbols):
                if (
                    index < len(symbols) - 1
                    and (symbols[index], symbols[index + 1]) == best
                ):
                    merged.append(symbols[index] + symbols[index + 1])
                    index += 2
                else:
                    merged.append(symbols[index])
                    index += 1
            symbols = merged
        return symbols
