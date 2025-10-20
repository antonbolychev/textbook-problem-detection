#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BACKEND_DIR="${PROJECT_ROOT}/backend"
FRONTEND_DIR="${PROJECT_ROOT}/frontend"
OPENAPI_PATH="${FRONTEND_DIR}/openapi.json"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${PROJECT_ROOT}/.uv-cache}"

pushd "${BACKEND_DIR}" >/dev/null
uv run python -c "import json; from app.main import app; print(json.dumps(app.openapi()))" > "${OPENAPI_PATH}"
popd >/dev/null

pushd "${FRONTEND_DIR}" >/dev/null
npm run generate-client
popd >/dev/null
