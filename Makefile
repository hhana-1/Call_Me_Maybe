install:
	uv sync

run:
	uv run python -m src

run-model:
	uv run python -m src --model $(MODEL)

trace:
	uv run python -m src --trace

trace-choices:
	uv run python -m src --trace --input data/input/model_choice_tests.json --output data/output/model_choice_results.json

export-vocabulary:
	uv run python good_to_have/export_vocabulary.py

debug:
	uv run python -m pdb -m src

clean:
	find . -type d \( -name __pycache__ -o -name .mypy_cache \) -prune -exec rm -rf {} +

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict