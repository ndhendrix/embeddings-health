#!/bin/bash
# Run Claude Code inside an isolated Apptainer container, safe to use with
# --dangerously-skip-permissions.
#
# Copy (or symlink) this file into any project folder under $HOME. Wherever
# it's placed, it self-scopes to THAT folder plus its $SCRATCH equivalent:
#   $HOME/<project>      -> bound rw at /workspace/home
#   $SCRATCH/<project>   -> bound rw at /workspace/scratch (created if missing)
#
# The container login/config/npm-cache is shared and centralized at
# $HOME/.claude-sandbox (never on $SCRATCH, which purges after 90 days idle).
#
# Nothing else on the filesystem -- other project folders, $GROUP_HOME, $OAK,
# ssh keys, other users' scratch, slurm/munge sockets -- is reachable from
# inside the container.
set -euo pipefail

SANDBOX_STORE="$HOME/.claude-sandbox"
SIF="$SANDBOX_STORE/image/claude-sandbox.sif"
CONTAINER_HOME="$SANDBOX_STORE/home"

# The project folder is wherever THIS script currently lives.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$PROJECT_DIR" == "$HOME" ]]; then
    echo "error: refusing to sandbox your entire \$HOME. Place this script inside a project subfolder (e.g. \$HOME/embeddings-health/claude-sandbox.sh) instead." >&2
    exit 1
fi

case "$PROJECT_DIR" in
    "$SANDBOX_STORE"*)
        echo "error: this script lives inside $SANDBOX_STORE -- copy it into a project folder instead." >&2
        exit 1
        ;;
    "$HOME"/*)
        REL_PATH="${PROJECT_DIR#"$HOME"/}"
        ;;
    *)
        echo "error: this script must be placed inside a subfolder of \$HOME (found: $PROJECT_DIR)." >&2
        exit 1
        ;;
esac

if [[ ! -f "$SIF" ]]; then
    echo "error: sandbox image not found at $SIF -- build it first" >&2
    exit 1
fi

mkdir -p "$CONTAINER_HOME"

SCRATCH_PROJECT_DIR="$SCRATCH/$REL_PATH"
if [[ ! -d "$SCRATCH_PROJECT_DIR" ]]; then
    echo "note: no matching scratch dir at $SCRATCH_PROJECT_DIR -- creating it" >&2
    mkdir -p "$SCRATCH_PROJECT_DIR"
fi
BIND_ARGS=(--bind "$PROJECT_DIR:/workspace/home" --bind "$SCRATCH_PROJECT_DIR:/workspace/scratch")

echo "sandboxed to: $PROJECT_DIR + $SCRATCH_PROJECT_DIR" >&2

exec apptainer exec \
    --containall \
    --cleanenv \
    --pwd /workspace/home \
    --home "$CONTAINER_HOME:/home/nhendrix" \
    "${BIND_ARGS[@]}" \
    --env "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}" \
    "$SIF" \
    claude --dangerously-skip-permissions "$@"
