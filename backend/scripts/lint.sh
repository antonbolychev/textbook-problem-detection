
set -e
set -x

if [[ "$*" == *"--format"* ]]; then
    echo "Formatting code..."
    uv run ruff check app --fix
    uv run ruff check research/detector --fix
    uv run ruff check arq_worker --fix
    uv run ruff format app
    uv run ruff format research/detector
    uv run ruff format arq_worker
fi

uv run mypy app
uv run ruff check app
uv run ruff format app --check

uv run mypy research/detector
uv run ruff check research/detector
uv run ruff format research/detector --check

uv run mypy arq_worker
uv run ruff check arq_worker
uv run ruff format arq_worker --check