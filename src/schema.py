from typing import Any

from pydantic import BaseModel, ConfigDict

class FunctionSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, dict[str, Any]]
    returns: dict[str, str]


class FunctionCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, Any]
