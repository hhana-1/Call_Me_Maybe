"""Registry of selectable free local Hugging Face models."""

MODEL_PRESETS: dict[str, str] = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",            # current default
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "tinyllama-1.1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

DEFAULT_MODEL = "qwen3-0.6b"


def resolve_model(name: str) -> str:
    """Map an alias to a repo id; pass through any raw HF repo id unchanged."""
    if name in MODEL_PRESETS:
        return MODEL_PRESETS[name]
    if "/" in name:            # looks like an explicit HF repo id
        return name
    raise ValueError(
        f"Unknown model '{name}'. Choose one of {sorted(MODEL_PRESETS)} "
        f"or pass a full Hugging Face repo id (e.g. 'org/model')."
    )
