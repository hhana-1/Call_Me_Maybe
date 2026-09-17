import json
import re
from pathlib import Path
from typing import Any, Sequence

# =====================================================================
# 1. DATA DEFINITIONS
# =====================================================================

_ROOT = Path(__file__).resolve().parent
_DATA_INPUT = _ROOT / "data" / "input"


def _load_json(filename: str) -> Any:
    """Load a project input file relative to the repository root."""
    with (_DATA_INPUT / filename).open(encoding="utf-8") as handle:
        return json.load(handle)


FUNCTION_SCHEMAS: list[dict[str, Any]] = _load_json(
    "functions_definition.json"
)

DOMAIN_VOCABULARIES = {
    "economy": {
        "economy", "economic", "interest", "compound", "principal",
        "rate", "finance", "financial", "investment", "loan",
    },
    "database": {
        "sql", "query", "database", "table", "select", "insert",
        "update", "delete", "production", "logs", "users",
    },
    "math": {
        "add", "sum", "plus", "multiply", "product",
        "square", "root", "even", "odd", "calculate",
    },
    "boolean": {
        "boolean", "bool", "true", "false", "yes", "no", "whether",
        "if", "check", "is", "flag", "enabled", "disabled",
    },
    "prose": {
        "greet", "reverse", "replace", "substitute", "word", "string",
        "format", "template", "read", "file", "vowel", "asterisk",
    },
}

# =====================================================================
# 2. MODEL TOKENIZATION AND DOMAIN MATCHING
# =====================================================================

def tokenize_with_model(model: Any, text: str) -> tuple[list[int], list[str]]:
    """Tokenize text with the model SDK and decode each token for display."""
    encoded = model.encode(text)
    if hasattr(encoded, "detach"):
        encoded = encoded.detach().cpu().tolist()
    while encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    token_ids = [int(token_id) for token_id in encoded]
    tokens = [model.decode([token_id]) for token_id in token_ids]
    return token_ids, tokens

def word_tokens(text: str) -> set[str]:
    """Return normalized words for lightweight candidate scoring."""
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z]+", text)
    }

# =====================================================================
# 3. ROUTER / SELECTOR LOGIC
# =====================================================================

def detect_domain(text: str) -> str | None:
    """Find the first matching domain in the required priority order."""
    words = {word.lower() for word in re.findall(r"[A-Za-z]+", text)}
    for domain in ("economy", "database", "math", "boolean", "prose"):
        if words & DOMAIN_VOCABULARIES[domain]:
            return domain
    return None

def schema_domain(schema: dict[str, Any]) -> str | None:
    """Infer a function domain from its name and description."""
    return detect_domain(f"{schema['name']} {schema['description']}")


def domain_vocabulary_matches(text: str) -> dict[str, set[str]]:
    """Return vocabulary terms found in text for each domain."""
    words = {word.lower() for word in re.findall(r"[A-Za-z]+", text)}
    return {
        domain: words & vocabulary
        for domain, vocabulary in DOMAIN_VOCABULARIES.items()
    }

def build_routing_context(
    prompt: str,
    functions: Sequence[dict[str, Any]] | None = None,
    model: Any | None = None,
) -> str:
    """Build routing context using the SDK tokenizer when a model is given."""
    schemas = list(functions) if functions is not None else FUNCTION_SCHEMAS
    lines = ["Tokenized function definitions:"]
    for schema in schemas:
        if model is not None:
            _, description_tokens = tokenize_with_model(
                model, schema["description"]
            )
        else:
            description_tokens = sorted(word_tokens(schema["description"]))
        lines.append(
            f"{schema['name']}: {' '.join(description_tokens)}"
        )
    if model is not None:
        _, prompt_tokens = tokenize_with_model(model, prompt)
    else:
        prompt_tokens = sorted(word_tokens(prompt))
    lines.append(f"Tokenized user request: {' '.join(prompt_tokens)}")
    return "\n".join(lines)


def rank_schema_candidates(
    prompt: str, functions: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Shortlist schemas using token boundaries before model scoring."""
    stopwords = {"a", "all", "and", "for", "in", "is", "of", "the", "to", "with"}
    prompt_tokens = {
        token for token in word_tokens(prompt)
        if token not in stopwords
    }
    prompt_words = {
        token for token in word_tokens(prompt) if token not in stopwords
    }
    numeric_count = sum(
        1 for token in re.findall(r"-?\d+(?:\.\d+)?", prompt)
    )
    prompt_domain = detect_domain(prompt)
    prompt_domain_matches = domain_vocabulary_matches(prompt)
    scored: list[tuple[int, dict[str, Any]]] = []
    for schema in functions:
        description_tokens = {
            token
            for token in word_tokens(
                schema["name"].replace("_", " ")
                + " "
                + schema["description"]
            )
            if token not in stopwords
        }
        schema_words = {
            token
            for token in word_tokens(
                schema["name"].replace("_", " ")
                + " "
                + schema["description"]
            )
            if token not in stopwords
        }
        score = (
            2 * len(prompt_words & schema_words)
            + len(prompt_tokens & description_tokens)
        )
        schema_domain_matches = domain_vocabulary_matches(
            f"{schema['name']} {schema['description']}"
        )
        shared_domain_terms = sum(
            len(prompt_domain_matches[domain] & schema_domain_matches[domain])
            for domain in DOMAIN_VOCABULARIES
        )
        score += 5 * shared_domain_terms
        if prompt_domain is not None and schema_domain(schema) == prompt_domain:
            score += 5
        numeric_parameters = sum(
            specification.get("type") == "number"
            for specification in schema.get("parameters", {}).values()
        )
        if numeric_count and numeric_count == numeric_parameters:
            score += 2
        elif numeric_count and numeric_parameters == 0:
            score -= 1
        scored.append((score, schema))
    best_score = max(score for score, _ in scored)
    if best_score == 0:
        return [schema for _, schema in scored]
    return [schema for score, schema in scored if score == best_score]

