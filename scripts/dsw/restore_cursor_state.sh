#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║  restore_cursor_state.sh                                         ║
# ║                                                                   ║
# ║  Mirror the OSS-archived Cursor agent state back into the local  ║
# ║  rootfs after a DSW container rebuild.                           ║
# ║                                                                   ║
# ║  Usually invoked automatically by                                ║
# ║      scripts/dsw/bootstrap_after_restart.sh                      ║
# ║  but also runnable on its own:                                   ║
# ║      bash scripts/dsw/restore_cursor_state.sh                    ║
# ║                                                                   ║
# ║  Idempotent.                                                     ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# WHAT GETS RESTORED
# ──────────────────
#   ${ARCHIVE_ROOT}/projects/      →  /root/.cursor/projects/
#   ${ARCHIVE_ROOT}/ide_state.json →  /root/.cursor/ide_state.json
#
# rsync's --update flag is used for a SAFETY-FIRST policy:
#   only files newer in the OSS archive overwrite local ones.
# This means: if the local already has a fresher session jsonl
# (e.g. you ran archive yesterday but the agent ran since), restore
# will *not* clobber it. Manual override:  RESTORE_FORCE=1.

set -euo pipefail

DST_PROJECTS="${DST_PROJECTS:-/root/.cursor/projects}"
DST_IDE_STATE="${DST_IDE_STATE:-/root/.cursor/ide_state.json}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/mnt/data/dsw-secrets/cursor-archive}"
RESTORE_FORCE="${RESTORE_FORCE:-0}"

log()  { printf '[restore] %s\n' "$*"; }
warn() { printf '[restore][WARN] %s\n' "$*" >&2; }

if [[ ! -d "${ARCHIVE_ROOT}/projects" ]]; then
  warn "no archive at ${ARCHIVE_ROOT}/projects — nothing to restore (first ever boot?)"
  exit 0
fi

# Empty source check (sometimes the archive dir exists but holds no workspaces).
n_src="$(ls -1 "${ARCHIVE_ROOT}/projects/" 2>/dev/null | wc -l)"
if (( n_src == 0 )); then
  warn "archive ${ARCHIVE_ROOT}/projects is empty — skip restore"
  exit 0
fi

mkdir -p "${DST_PROJECTS}"

if [[ "${RESTORE_FORCE}" == "1" ]]; then
  log "FORCE restore (will overwrite even newer local files)"
  RSYNC_FLAGS=(-a --human-readable)
else
  log "safe restore (--update; will not overwrite newer local files)"
  RSYNC_FLAGS=(-a --update --human-readable)
fi

log "syncing ${ARCHIVE_ROOT}/projects/ → ${DST_PROJECTS}/"
rsync "${RSYNC_FLAGS[@]}" \
      --exclude='*.tmp' --exclude='*.lock' \
      "${ARCHIVE_ROOT}/projects/" "${DST_PROJECTS}/"

if [[ -f "${ARCHIVE_ROOT}/ide_state.json" ]]; then
  if [[ "${RESTORE_FORCE}" == "1" || ! -f "${DST_IDE_STATE}" \
        || "${ARCHIVE_ROOT}/ide_state.json" -nt "${DST_IDE_STATE}" ]]; then
    cp -f "${ARCHIVE_ROOT}/ide_state.json" "${DST_IDE_STATE}"
    log "restored ide_state.json"
  else
    log "local ide_state.json is newer, kept as-is"
  fi
fi

log "summary:"
log "  workspaces in archive : ${n_src}"
log "  workspaces locally    : $(ls -1 "${DST_PROJECTS}/" 2>/dev/null | wc -l)"
log "  total local size      : $(du -sh "${DST_PROJECTS}/" 2>/dev/null | cut -f1)"
log "done."
