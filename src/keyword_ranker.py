"""Keyword and domain ranking that shortlists functions for a prompt."""

import re
from typing import Any, Sequence

DOMAIN_KEYWORDS = {
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


def lowercase_words(text: str) -> set[str]:
    """Return the lowercase words of a text, ignoring digits and symbols."""
    return {word.lower() for word in re.findall(r"[A-Za-z]+", text)}


def detect_domain(text: str) -> str | None:
    """Find the first matching domain in the required priority order."""
    words = lowercase_words(text)
    for domain in ("economy", "database", "math", "boolean", "prose"):
        if words & DOMAIN_KEYWORDS[domain]:
            return domain
    return None


def function_domain(schema: dict[str, Any]) -> str | None:
    """Infer a function domain from its name and description."""
    return detect_domain(f"{schema['name']} {schema['description']}")


def domain_keyword_matches(text: str) -> dict[str, set[str]]:
    """Return the keywords found in text for each domain."""
    words = lowercase_words(text)
    return {
        domain: words & keywords
        for domain, keywords in DOMAIN_KEYWORDS.items()
    }


def shortlist_functions(
    prompt: str, functions: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Keep the functions with the best score, or all of them if none scores.

    A function scores 3 per shared word, 5 per shared domain keyword, 5 if
    its domain matches the prompt's, and +2 / -1 depending on whether the
    count of numbers in the prompt fits its number parameters.
    """
    stopwords = {
        "a", "all", "and", "for", "in", "is", "of", "the", "to", "with"
    }
    prompt_words = lowercase_words(prompt) - stopwords
    numbers_in_prompt = len(re.findall(r"-?\d+(?:\.\d+)?", prompt))
    prompt_domain = detect_domain(prompt)
    prompt_matches = domain_keyword_matches(prompt)
    scored: list[tuple[int, dict[str, Any]]] = []
    for schema in functions:
        function_words = lowercase_words(
            schema["name"].replace("_", " ") + " " + schema["description"]
        ) - stopwords
        score = 3 * len(prompt_words & function_words)
        function_matches = domain_keyword_matches(
            f"{schema['name']} {schema['description']}"
        )
        score += 5 * sum(
            len(prompt_matches[domain] & function_matches[domain])
            for domain in DOMAIN_KEYWORDS
        )
        if prompt_domain is not None and (
            function_domain(schema) == prompt_domain
        ):
            score += 5
        number_parameters = sum(
            specification.get("type") == "number"
            for specification in schema.get("parameters", {}).values()
        )
        if numbers_in_prompt and numbers_in_prompt == number_parameters:
            score += 2
        elif numbers_in_prompt and number_parameters == 0:
            score -= 1
        scored.append((score, schema))
    best_score = max(score for score, _ in scored)
    if best_score == 0:
        return [schema for _, schema in scored]
    return [schema for score, schema in scored if score == best_score]
