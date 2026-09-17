import json
import re
from pathlib import Path
from typing import Any, List, Sequence, Tuple

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
TEST_PROMPTS: list[dict[str, str]] = _load_json("function_calling_tests.json")

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
# 2. TOKENIZATION ENGINES (Simulated & Rule-Based)
# =====================================================================

def tokenize_bpe(text: str) -> List[str]:
    """Simulates BPE by breaking subwords and code structures (snake_case)."""
    # Splitting camelCase and snake_case or punctuation
    tokens = re.findall(r'[a-zA-Z]+|_\w+|\d+|[^\w\s]', text)
    processed = []
    for t in tokens:
        if t.startswith('_'):
            processed.extend(['_', t[1:]])
        else:
            processed.append(t)
    return processed

def tokenize_wordpiece(text: str) -> List[str]:
    """Simulates WordPiece text chunking with wordpieces (## tokens)."""
    # Naive split with ## appended to subsequent parts of rare terms
    words = text.split()
    tokens = []
    for w in words:
        clean = re.sub(r'[^\w]', '', w)
        if len(clean) > 6:
            tokens.extend([clean[:4], f"##{clean[4:]}"])
        else:
            tokens.append(w)
    return tokens

def tokenize_unigram(text: str) -> List[str]:
    """Simulates Unigram behavior optimizing for full natural language words."""
    return re.findall(r'\b\w+\b|[^\w\s]', text)

def tokenize_custom_hybrid(text: str) -> List[str]:
    """
    Custom Hybrid Tokenizer.
    Guarantees quoted blocks and multi-digit integers remain atomic.
    """
    pattern = r'(?:"[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\'|\d+|\b\w+(?:\'\w+)?\b|[^\s\w])'
    return re.findall(pattern, text)

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

def choose_and_run_tokenizer(text: str, is_schema: bool = False) -> Tuple[str, List[str]]:
    """
    Inspects input structural traits and executes the absolute best 
    tokenization algorithm for that specific payload.
    """
    if is_schema:
        # Code syntax, raw strings, and regular expressions favor structural BPE
        if "_" in text or "regex" in text.lower() or "\\" in text:
            return "Byte-Pair Encoding (BPE)", tokenize_bpe(text)
        return "WordPiece", tokenize_wordpiece(text)
        
    # Check variables inside user prompts
    has_quotes = '"' in text or "'" in text
    has_large_numbers = any(len(num) > 1 for num in re.findall(r'\d+', text))
    
    # User prompts with exact variable arguments require custom boundary protection
    if has_quotes or has_large_numbers:
        return "Custom Hybrid (Rule-Based)", tokenize_custom_hybrid(text)
        
    # Default clean natural language structures fit Unigram optimization maps
    return "Unigram", tokenize_unigram(text)


def build_routing_context(
    prompt: str, functions: Sequence[dict[str, Any]] | None = None
) -> str:
    """Build a token-aware routing context from the live JSON definitions."""
    schemas = list(functions) if functions is not None else FUNCTION_SCHEMAS
    lines = ["Tokenized function definitions:"]
    for schema in schemas:
        _, description_tokens = choose_and_run_tokenizer(
            schema["description"], is_schema=True
        )
        lines.append(
            f"{schema['name']}: {' '.join(description_tokens)}"
        )
    _, prompt_tokens = choose_and_run_tokenizer(prompt)
    lines.append(f"Tokenized user request: {' '.join(prompt_tokens)}")
    return "\n".join(lines)


def rank_schema_candidates(
    prompt: str, functions: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Shortlist schemas using token boundaries before model scoring."""
    stopwords = {"a", "all", "and", "for", "in", "is", "of", "the", "to", "with"}
    prompt_tokens = {
        token.lower()[:4]
        for token in tokenize_unigram(prompt)
        if token.isalpha() and token.lower() not in stopwords
    }
    prompt_words = {
        token.lower()
        for token in tokenize_unigram(prompt)
        if token.isalpha() and token.lower() not in stopwords
    }
    numeric_count = sum(
        bool(re.fullmatch(r"-?\d+(?:\.\d+)?", token))
        for token in tokenize_custom_hybrid(prompt)
    )
    prompt_domain = detect_domain(prompt)
    prompt_domain_matches = domain_vocabulary_matches(prompt)
    scored: list[tuple[int, dict[str, Any]]] = []
    for schema in functions:
        description_tokens = {
            token.lower()[:4]
            for token in tokenize_unigram(schema["name"].replace("_", " "))
            + tokenize_unigram(schema["description"])
            if token.isalpha() and token.lower() not in stopwords
        }
        schema_words = {
            token.lower()
            for token in tokenize_unigram(
                schema["name"].replace("_", " ")
                + " "
                + schema["description"]
            )
            if token.isalpha() and token.lower() not in stopwords
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

# =====================================================================
# 4. EXECUTION FLOW
# =====================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PHASE 1: PROCESSING SCHEMAS AND FUNCTIONS")
    print("=" * 80)
    for schema in FUNCTION_SCHEMAS:
        func_name = schema["name"]
        algo, tokens = choose_and_run_tokenizer(func_name, is_schema=True)
        print(f"Function : {func_name:<35} -> Algo Selected: {algo}")
        print(f"Tokens   : {tokens}\n")

    print("\n" + "=" * 80)
    print("PHASE 2: RUNNING TEST PROMPTS")
    print("=" * 80)
    for idx, item in enumerate(TEST_PROMPTS, 1):
        prompt_text = item["prompt"]
        algo, tokens = choose_and_run_tokenizer(prompt_text, is_schema=False)
        print(f"Test #{idx}: {prompt_text}")
        print(f"Selected: {algo}")
        print(f"Tokens  : {tokens}\n")
