#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║  archive_cursor_state.sh                                         ║
# ║                                                                   ║
# ║  Manually mirror the local Cursor agent state into OSS so it     ║
# ║  survives the next DSW container rebuild.                        ║
# ║                                                                   ║
# ║  Run this whenever you want to "save" the running session:       ║
# ║      bash scripts/dsw/archive_cursor_state.sh                    ║
# ║                                                                   ║
# ║  Idempotent: rsync-based, only touches changed files.            ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# WHAT GETS ARCHIVED
# ──────────────────
#   /root/.cursor/projects/        →  ${ARCHIVE_ROOT}/projects/
#       ├─ <workspace-key>/agent-transcripts/<uuid>/<uuid>.jsonl
#       ├─ <workspace-key>/terminals/*.txt
#       └─ <workspace-key>/mcps/...
#   /root/.cursor/ide_state.json   →  ${ARCHIVE_ROOT}/ide_state.json
#
# Plus a timestamped point-in-time snapshot under
#   ${ARCHIVE_ROOT}/snapshots/<TS>/
# pruned to the last KEEP_SNAPSHOTS (default 20).
#
# CONCURRENCY NOTE
# ────────────────
# Cursor agent holds an open fd on the active jsonl while writing to it.
# rsync reads files block-by-block, so a partially-flushed tail is
# possible. The timestamped snapshot taken AFTER the rsync is the
# "definitive" copy for that point in time; the live mirror under
# ${ARCHIVE_ROOT}/projects/ tracks the latest state for fast restore.

set -euo pipefail

SRC_PROJECTS="${SRC_PROJECTS:-/root/.cursor/projects}"
SRC_IDE_STATE="${SRC_IDE_STATE:-/root/.cursor/ide_state.json}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/mnt/data/dsw-secrets/cursor-archive}"
KEEP_SNAPSHOTS="${KEEP_SNAPSHOTS:-20}"

log()  { printf '[archive] %s\n' "$*"; }
warn() { printf '[archive][WARN] %s\n' "$*" >&2; }
die()  { printf '[archive][ERROR] %s\n' "$*" >&2; exit 1; }

[[ -d "${SRC_PROJECTS}" ]] || die "no source dir at ${SRC_PROJECTS} (cursor agent never ran here?)"

mkdir -p "${ARCHIVE_ROOT}/projects"
mkdir -p "${ARCHIVE_ROOT}/snapshots"

# ────────────────────────────────────────────────────────────────────
# 1) Mirror live state with rsync. --delete keeps OSS in sync with
#    local removals (e.g. you closed an old workspace). Active jsonl
#    files keep growing — that's fine, next archive picks up the rest.
# ────────────────────────────────────────────────────────────────────
log "mirroring ${SRC_PROJECTS}/ → ${ARCHIVE_ROOT}/projects/"
rsync -a --delete --human-readable \
      --exclude='*.tmp' --exclude='*.lock' \
      "${SRC_PROJECTS}/" "${ARCHIVE_ROOT}/projects/"

if [[ -f "${SRC_IDE_STATE}" ]]; then
  cp -f "${SRC_IDE_STATE}" "${ARCHIVE_ROOT}/ide_state.json"
  log "copied ide_state.json"
fi

# ────────────────────────────────────────────────────────────────────
# 2) Timestamped snapshot of the *current* mirror so a future bad
#    archive (e.g. accidental deletion) doesn't lose history.
# ────────────────────────────────────────────────────────────────────
TS="$(date +%Y%m%d-%H%M%S)"
SNAP_DIR="${ARCHIVE_ROOT}/snapshots/${TS}"
log "snapshot → ${SNAP_DIR}"
mkdir -p "${SNAP_DIR}"
# Hardlink-based snapshot would be ideal but ossfs2 doesn't support
# hardlinks across the fuse layer; fall back to plain cp -a.
cp -a "${ARCHIVE_ROOT}/projects" "${SNAP_DIR}/projects"
[[ -f "${ARCHIVE_ROOT}/ide_state.json" ]] && \
  cp -f "${ARCHIVE_ROOT}/ide_state.json" "${SNAP_DIR}/ide_state.json"

# ────────────────────────────────────────────────────────────────────
# 3) Prune old snapshots (keep newest KEEP_SNAPSHOTS).
# ────────────────────────────────────────────────────────────────────
mapfile -t snaps < <(ls -1d "${ARCHIVE_ROOT}/snapshots/"*/ 2>/dev/null | sort)
n_total="${#snaps[@]}"
if (( n_total > KEEP_SNAPSHOTS )); then
  to_prune=$(( n_total - KEEP_SNAPSHOTS ))
  log "pruning ${to_prune} old snapshot(s) (keeping newest ${KEEP_SNAPSHOTS})"
  for (( i=0; i<to_prune; i++ )); do
    rm -rf "${snaps[$i]}"
  done
fi

# ────────────────────────────────────────────────────────────────────
# 4) Summary.
# ────────────────────────────────────────────────────────────────────
log "summary:"
log "  workspaces archived: $(ls -1 "${ARCHIVE_ROOT}/projects/" 2>/dev/null | wc -l)"
log "  total mirror size:   $(du -sh "${ARCHIVE_ROOT}/projects/" 2>/dev/null | cut -f1)"
log "  snapshot count:      $(ls -1d "${ARCHIVE_ROOT}/snapshots/"*/ 2>/dev/null | wc -l)"
log "  this snapshot:       ${SNAP_DIR}"
log "done."
