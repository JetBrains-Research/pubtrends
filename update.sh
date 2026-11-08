#!/usr/bin/env bash
set -eo pipefail

CONFIG_DIR="${HOME}/.pubtrends"
LOG_DIR="${CONFIG_DIR}/logs"
LOCK_FILE="/tmp/pubtrends_update.lock"
STAGE_STATE_FILE="${CONFIG_DIR}/.pipeline_state"
mkdir -p "${LOG_DIR}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_DIR}/pipeline_master.log"
}

# 1. Single-Instance Protection, taken before any work including uv sync
if command -v flock >/dev/null 2>&1; then
    exec 200>"${LOCK_FILE}"
    if ! flock -n 200; then
        log "Another update pipeline instance is currently running. Exiting."
        exit 1
    fi
else
    # No flock (e.g. macOS): mkdir is atomic, the PID file lets us detect stale locks after a crash
    LOCK_DIR="${LOCK_FILE}.d"
    if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
        LOCK_PID=$(cat "${LOCK_DIR}/pid" 2>/dev/null || true)
        if [[ -n "${LOCK_PID}" ]] && kill -0 "${LOCK_PID}" 2>/dev/null; then
            log "Another update pipeline instance (PID ${LOCK_PID}) is currently running. Exiting."
            exit 1
        fi
        log "Removing stale lock left by PID ${LOCK_PID:-unknown}"
        rm -rf "${LOCK_DIR}"
        mkdir "${LOCK_DIR}"
    fi
    echo $$ > "${LOCK_DIR}/pid"
    trap 'rm -rf "${LOCK_DIR}"' EXIT
fi

# Locate pubtrends.jar (local build or distribution)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# Prepare for GPU: torch for embeddings, faiss-gpu for Faiss index training
uv sync --no-install-project --extra gpu
/bin/bash scripts/nlp.sh
# Following uv run calls shouldn't re-sync the environment without gpu extra
export UV_NO_SYNC=1
export PUBK_EMBEDDINGS_BACKEND=torch
export PUBTRENDS_EMBEDDINGS_BACKEND=torch

# Locate pubtrends jar (from build/libs/ or packed by dist.sh in dist/)
JAR=""
for candidate in \
    "${SCRIPT_DIR}"/build/libs/pubtrends*.jar \
    "${SCRIPT_DIR}"/build/libs/*.jar \
    "${SCRIPT_DIR}"/dist/pubtrends*.jar \
    "${SCRIPT_DIR}"/dist/*.jar \
    "${SCRIPT_DIR}"/app/pubtrends*.jar \
    "${SCRIPT_DIR}"/app/*.jar; do
    if [ -f "${candidate}" ]; then
        JAR="${candidate}"
        break
    fi
done

if [ -z "${JAR}" ]; then
    log "Error: pubtrends jar not found in build/libs/ or dist/"
    exit 1
fi
log "Using JAR: ${JAR}"

# 2. Pipeline State Management for Failure Recovery
get_stage() {
    if [[ -f "${STAGE_STATE_FILE}" ]]; then
        cat "${STAGE_STATE_FILE}"
    else
        echo "1"
    fi
}

set_stage() {
    echo "$1" > "${STAGE_STATE_FILE}"
    log ">>> Advanced to Stage $1"
}

retry_command() {
    local max_attempts=$1
    local delay=$2
    shift 2
    local attempt=1
    until "$@"; do
        if (( attempt >= max_attempts )); then
            log "Command '$*' failed after ${attempt} attempts."
            return 1
        fi
        log "Attempt ${attempt} failed. Retrying in ${delay}s..."
        sleep "${delay}"
        (( attempt++ ))
    done
}

CURRENT_STAGE=$(get_stage)
log "Starting Pubtrends Update Pipeline from Stage ${CURRENT_STAGE}"

# -------------------------------------------------------------
# Stage 1: Update PubMed Database
# -------------------------------------------------------------
if (( CURRENT_STAGE <= 1 )); then
    log "[Stage 1/4] Updating PubMed PostgreSQL Tables..."
    retry_command 5 30 java -cp "${JAR}" org.jetbrains.bio.pubtrends.pm.PubmedLoader --fillDatabase
    set_stage 2
fi

# -------------------------------------------------------------
# Stage 2: Sync Lucene Keyword Index
# -------------------------------------------------------------
if (( CURRENT_STAGE <= 2 )); then
    log "[Stage 2/4] Updating Lucene BM25 Search Index..."
    retry_command 3 15 uv run python pysrc/preprocess/update_lucene.py --min-year "$(( $(date +%Y) - 1 ))"
    set_stage 3
fi

# -------------------------------------------------------------
# Stage 3: Update Semantic Embeddings
# -------------------------------------------------------------
if (( CURRENT_STAGE <= 3 )); then
    log "[Stage 3/4] Computing & Storing Semantic Embeddings into Vector DB..."
    retry_command 3 20 uv run python pysrc/preprocess/update_embeddings.py --min-year "$(( $(date +%Y) - 1 ))"
    set_stage 4
fi

# -------------------------------------------------------------
# Stage 4: Sync FAISS Vector Search Index
# -------------------------------------------------------------
if (( CURRENT_STAGE <= 4 )); then
    log "[Stage 4/4] Updating FAISS Search Index from Vector DB..."
    retry_command 3 20 uv run python pysrc/preprocess/update_embeddings.py --to-faiss --min-year "$(( $(date +%Y) - 1 ))"

    # Reset pipeline state upon complete success
    rm -f "${STAGE_STATE_FILE}"
    log "Pubtrends Pipeline update completed successfully across all 4 stages."
fi
