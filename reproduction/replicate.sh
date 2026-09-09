#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v uv >/dev/null 2>&1; then
  if ! command -v python3 >/dev/null 2>&1; then
    echo 'Install Python 3 or uv, then rerun this script.' >&2
    exit 1
  fi
  if [[ ! -x .bootstrap/bin/uv ]]; then
    python3 -m venv .bootstrap
    .bootstrap/bin/python -m pip install 'uv==0.10.0'
  fi
  UV="$PWD/.bootstrap/bin/uv"
else
  UV="$(command -v uv)"
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.cache}"
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
"$UV" sync --locked --python 3.11.13
exec "$UV" run --locked --python 3.11.13 python replication.py "$@"
