"""Pydantic models for function definitions and generated calls."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class FunctionSchema(BaseModel):
    """One entry of ``functions_definition.json``.

    Attributes:
        name: Function name the model must write, e.g. ``fn_add_numbers``.
        description: What the function does, shown to the model.
        parameters: Argument name to its spec, e.g. ``{"type": "number"}``.
        returns: Spec of the return value.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, dict[str, Any]]
    returns: dict[str, str]


class FunctionCall(BaseModel):
    """One object of the output file: exactly these three keys.

    Attributes:
        prompt: The original natural-language request.
        name: The function chosen for it.
        parameters: Every argument, with the type its schema requires.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, Any]
