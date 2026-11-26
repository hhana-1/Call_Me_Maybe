"""Read the arguments with regular expressions, without the model.

Once the function is chosen (by the model, or by the keyword ranker with
``--ranker``), these rules fill in its arguments, with no model call.

The rules go by parameter type and, for some strings, by parameter name
(``regex``, ``replacement``, ``source_string``, ``path``, ``encoding``):

- ``number`` / ``integer``: the numbers of the request, in order;
- ``boolean``: a ``true`` / ``false`` in the request, or an even/odd question;
- ``string``: a pattern for the parameter's name, else the quoted texts in
  order, else the last word.
"""

import re
from typing import Any

from src.schema import FunctionSchema

# A number written in the request, not part of a word such as "utf-8".
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


def extract_arguments(
    prompt: str, function: FunctionSchema
) -> dict[str, Any]:
    """Read the arguments of ``function`` from ``prompt``.

    Raises:
        ValueError: If a value is missing, an integer is not whole, or a
            parameter has an unsupported type.
    """
    numbers = [float(value) for value in NUMBER.findall(prompt)]
    quoted = [
        str(match[1])
        for match in re.findall(r"(['\"])(.*?)(?<!\\)\1", prompt)
    ]
    arguments: dict[str, Any] = {}
    number_index = 0
    string_index = 0

    for name, specification in function.parameters.items():
        parameter_type = specification.get("type")
        if parameter_type in ("number", "integer"):
            # Numbers in the order of the request.
            if number_index >= len(numbers):
                raise ValueError(f"Missing number argument: {name}")
            value = numbers[number_index]
            number_index += 1
            if parameter_type == "integer" and not value.is_integer():
                raise ValueError(f"Integer argument must be whole: {name}")
            arguments[name] = int(value) if value.is_integer() else value
        elif parameter_type == "string":
            arguments[name] = string_argument(
                prompt, name, quoted, string_index
            )
            string_index += 1
        elif parameter_type == "boolean":
            match = re.search(r"\b(true|false)\b", prompt, re.IGNORECASE)
            if match is not None:
                arguments[name] = match.group(1).lower() == "true"
            else:
                result = even_odd_answer(prompt)
                if result is None:
                    raise ValueError(f"Missing boolean argument: {name}")
                arguments[name] = result
        else:
            raise ValueError(f"Unsupported parameter type: {parameter_type}")
    return arguments


def even_odd_answer(prompt: str) -> bool | None:
    """Answer "Is 4 an even number?"-style questions, or return None."""
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


def string_argument(
    prompt: str, name: str, quoted: list[str], index: int
) -> str:
    """Read one string: by the parameter's name, a quote, or the last word.

    Raises:
        ValueError: If the request has no word to use.
    """
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
        paths = [
            str(token).strip(" '\"")
            for token in re.findall(r"[^\s,;]+", prompt)
            if looks_like_path(token)
        ]
        if paths:
            return paths[0]
    if name == "encoding":
        match = re.search(
            r"(?:with|using)\s+([A-Za-z0-9_.-]+)\s+encoding\b|"
            r"\bencoding\s*[:=]\s*([A-Za-z0-9_.-]+)",
            prompt,
            re.IGNORECASE,
        )
        if match:
            return str(match.group(1) or match.group(2)).lower()
    if index < len(quoted):
        return quoted[index]
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", prompt)
    if not words:
        raise ValueError(f"Missing string argument: {name}")
    return str(words[-1])


def looks_like_path(value: str) -> bool:
    """Recognize common path shapes (Unix, Windows, relative, file.ext)."""
    return bool(
        re.fullmatch(
            r"(?:[A-Za-z]:[\\/][^\s,;]+|"
            r"(?:/|\./|\.\./|~/)[^\s,;]+|"
            r"[\w.-]+(?:[\\/][\w .-]+)+|"
            r"[\w.-]+\.[A-Za-z0-9]+)",
            value.strip(" '\""),
        )
    )
