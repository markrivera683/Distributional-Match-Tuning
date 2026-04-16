#!/usr/bin/env bash
set -euo pipefail

log() {
  printf "\n[%s] %s\n" "$(date '+%H:%M:%S')" "$*"
}

warn() {
  printf "warning: %s\n" "$*" >&2
}

die() {
  printf "error: %s\n" "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/dlc/recreate_env_from_freeze.sh

Recreate .venv and .teacherVenv from the exact freeze snapshots written by
prepare_g3_dlc_snapshot.sh.

Environment variables:
  REPO_DIR              Target repo root. Defaults to this repo.
  SNAPSHOT_DIR          Defaults to $REPO_DIR/.dlc_snapshot
  STUDENT_VENV          Defaults to $REPO_DIR/.venv
  TEACHER_VENV          Defaults to $REPO_DIR/.teacherVenv
  INSTALL_APT_DEPS      1 (default) to install build deps via apt
  REBUILD_STUDENT_ENV   1 (default) or 0
  REBUILD_TEACHER_ENV   1 (default) or 0
  FORCE_REBUILD         0 (default) or 1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REPO_DIR="${REPO_DIR:-$DEFAULT_REPO_DIR}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-$REPO_DIR/.dlc_snapshot}"
STUDENT_VENV="${STUDENT_VENV:-$REPO_DIR/.venv}"
TEACHER_VENV="${TEACHER_VENV:-$REPO_DIR/.teacherVenv}"
INSTALL_APT_DEPS="${INSTALL_APT_DEPS:-1}"
REBUILD_STUDENT_ENV="${REBUILD_STUDENT_ENV:-1}"
REBUILD_TEACHER_ENV="${REBUILD_TEACHER_ENV:-1}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

STUDENT_META_PATH="${SNAPSHOT_DIR}/student.packaging.env"
TEACHER_META_PATH="${SNAPSHOT_DIR}/teacher.packaging.env"
STUDENT_REQ_PATH="${SNAPSHOT_DIR}/student.requirements.install.txt"
TEACHER_REQ_PATH="${SNAPSHOT_DIR}/teacher.requirements.install.txt"

ensure_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || die "Missing required command: $cmd"
}

run_apt_install() {
  if [[ "${INSTALL_APT_DEPS}" != "1" ]]; then
    return
  fi

  if ! command -v apt-get >/dev/null 2>&1; then
    return
  fi

  local apt_runner=()
  if command -v sudo >/dev/null 2>&1; then
    apt_runner=(sudo)
  elif [[ "$(id -u)" -eq 0 ]]; then
    apt_runner=()
  else
    warn "Skipping apt dependencies because sudo is unavailable."
    return
  fi

  log "Installing Ubuntu/Debian build dependencies"
  "${apt_runner[@]}" apt-get update
  "${apt_runner[@]}" apt-get install -y \
    build-essential \
    ca-certificates \
    curl \
    git \
    ninja-build \
    pkg-config
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
    return
  fi

  ensure_cmd curl
  log "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  UV_BIN="${HOME}/.local/bin/uv"
  [[ -x "${UV_BIN}" ]] || die "uv installation failed"
}

ensure_snapshot_files() {
  [[ -d "${REPO_DIR}" ]] || die "REPO_DIR not found: ${REPO_DIR}"
  [[ -d "${SNAPSHOT_DIR}" ]] || die "SNAPSHOT_DIR not found: ${SNAPSHOT_DIR}"
  [[ -f "${STUDENT_META_PATH}" ]] || die "Missing snapshot metadata: ${STUDENT_META_PATH}"
  [[ -f "${TEACHER_META_PATH}" ]] || die "Missing snapshot metadata: ${TEACHER_META_PATH}"
  [[ -f "${STUDENT_REQ_PATH}" ]] || die "Missing install freeze: ${STUDENT_REQ_PATH}"
  [[ -f "${TEACHER_REQ_PATH}" ]] || die "Missing install freeze: ${TEACHER_REQ_PATH}"
}

compute_fingerprint() {
  local meta_path="$1"
  local req_path="$2"
  cat "${meta_path}" "${req_path}" | sha256sum | awk '{print $1}'
}

load_packaging_meta() {
  local prefix="$1"
  local meta_path="$2"
  local line
  while IFS='=' read -r key value; do
    [[ -z "${key}" ]] && continue
    printf -v "${prefix}_${key}" '%s' "${value}"
  done < "${meta_path}"
}

validate_requirements_paths() {
  local req_path="$1"
  python - "${req_path}" <<'PY'
import pathlib
import sys
import urllib.parse

req_path = pathlib.Path(sys.argv[1])
missing = []
for raw_line in req_path.read_text().splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    if "@ file://" not in line:
        continue
    _, uri = line.split("@", 1)
    parsed = urllib.parse.urlparse(uri.strip())
    target = pathlib.Path(urllib.parse.unquote(parsed.path))
    if not target.exists():
        missing.append(str(target))

if missing:
    print("missing file-backed requirements:", file=sys.stderr)
    for path in missing:
        print(f"  {path}", file=sys.stderr)
    sys.exit(1)
PY
}

create_venv() {
  local venv_path="$1"
  local python_version="$2"
  local pip_version="$3"
  local setuptools_version="$4"
  local wheel_version="$5"

  mkdir -p "$(dirname "${venv_path}")"
  rm -rf "${venv_path}"
  log "Creating virtualenv at ${venv_path} with Python ${python_version}"
  "${UV_BIN}" venv --seed --python "${python_version}" "${venv_path}"
  local python_bin="${venv_path}/bin/python"
  [[ -x "${python_bin}" ]] || die "Virtualenv creation failed: ${venv_path}"
  "${python_bin}" -m pip install --upgrade \
    "pip==${pip_version}" \
    "setuptools==${setuptools_version}" \
    "wheel==${wheel_version}"
}

sanitize_freeze() {
  local input_path="$1"
  local output_path="$2"
  python - "${input_path}" "${output_path}" <<'PY'
import pathlib
import sys

input_path = pathlib.Path(sys.argv[1])
output_path = pathlib.Path(sys.argv[2])
lines = []
for raw_line in input_path.read_text().splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith("-e ") or line.startswith("--editable"):
        continue
    lines.append(line)
output_path.write_text("\n".join(lines) + ("\n" if lines else ""))
PY
}

split_teacher_requirements() {
  local input_path="$1"
  local base_output_path="$2"
  local override_output_path="$3"
  python - "${input_path}" "${base_output_path}" "${override_output_path}" <<'PY'
import pathlib
import sys

input_path = pathlib.Path(sys.argv[1])
base_output_path = pathlib.Path(sys.argv[2])
override_output_path = pathlib.Path(sys.argv[3])

override_prefixes = (
    "transformers==",
)

base_lines = []
override_lines = []
for raw_line in input_path.read_text().splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith(override_prefixes):
        override_lines.append(line)
    else:
        base_lines.append(line)

base_output_path.write_text("\n".join(base_lines) + ("\n" if base_lines else ""))
override_output_path.write_text("\n".join(override_lines) + ("\n" if override_lines else ""))
PY
}

verify_freeze_matches() {
  local python_bin="$1"
  local expected_req_path="$2"
  local env_name="$3"
  local tmp_dir
  tmp_dir="$(mktemp -d)"

  local raw_path="${tmp_dir}/${env_name}.raw.txt"
  local sanitized_path="${tmp_dir}/${env_name}.sanitized.txt"
  "${python_bin}" -m pip freeze > "${raw_path}"
  sanitize_freeze "${raw_path}" "${sanitized_path}"

  if ! diff -u "${expected_req_path}" "${sanitized_path}"; then
    rm -rf "${tmp_dir}"
    die "${env_name} freeze does not match snapshot"
  fi

  rm -rf "${tmp_dir}"
}

write_stamp() {
  local venv_path="$1"
  local fingerprint="$2"
  printf "%s\n" "${fingerprint}" > "${venv_path}/.dlc_snapshot_fingerprint"
}

stamp_matches() {
  local venv_path="$1"
  local fingerprint="$2"
  [[ -f "${venv_path}/.dlc_snapshot_fingerprint" ]] || return 1
  [[ "$(tr -d '\n' < "${venv_path}/.dlc_snapshot_fingerprint")" == "${fingerprint}" ]]
}

rebuild_student_env() {
  load_packaging_meta "STUDENT" "${STUDENT_META_PATH}"
  local fingerprint
  fingerprint="$(compute_fingerprint "${STUDENT_META_PATH}" "${STUDENT_REQ_PATH}")"

  if [[ "${FORCE_REBUILD}" != "1" ]] && [[ -d "${STUDENT_VENV}" ]] && stamp_matches "${STUDENT_VENV}" "${fingerprint}"; then
    log "Student env already matches snapshot; skipping rebuild"
    return
  fi

  validate_requirements_paths "${STUDENT_REQ_PATH}"
  create_venv "${STUDENT_VENV}" "${STUDENT_PYTHON_VERSION}" "${STUDENT_PIP_VERSION}" "${STUDENT_SETUPTOOLS_VERSION}" "${STUDENT_WHEEL_VERSION}"

  local python_bin="${STUDENT_VENV}/bin/python"
  log "Installing student environment from freeze snapshot"
  "${python_bin}" -m pip install -r "${STUDENT_REQ_PATH}"
  log "Installing repo in editable mode without dependency mutation"
  "${python_bin}" -m pip install --no-deps -e "${REPO_DIR}"
  verify_freeze_matches "${python_bin}" "${STUDENT_REQ_PATH}" "student"
  write_stamp "${STUDENT_VENV}" "${fingerprint}"
}

rebuild_teacher_env() {
  load_packaging_meta "TEACHER" "${TEACHER_META_PATH}"
  local fingerprint
  fingerprint="$(compute_fingerprint "${TEACHER_META_PATH}" "${TEACHER_REQ_PATH}")"

  if [[ "${FORCE_REBUILD}" != "1" ]] && [[ -d "${TEACHER_VENV}" ]] && stamp_matches "${TEACHER_VENV}" "${fingerprint}"; then
    log "Teacher env already matches snapshot; skipping rebuild"
    return
  fi

  validate_requirements_paths "${TEACHER_REQ_PATH}"
  create_venv "${TEACHER_VENV}" "${TEACHER_PYTHON_VERSION}" "${TEACHER_PIP_VERSION}" "${TEACHER_SETUPTOOLS_VERSION}" "${TEACHER_WHEEL_VERSION}"

  local python_bin="${TEACHER_VENV}/bin/python"
  local tmp_dir
  tmp_dir="$(mktemp -d)"
  local teacher_base_req_path="${tmp_dir}/teacher.base.requirements.txt"
  local teacher_override_req_path="${tmp_dir}/teacher.override.requirements.txt"

  split_teacher_requirements "${TEACHER_REQ_PATH}" "${teacher_base_req_path}" "${teacher_override_req_path}"

  log "Installing teacher base environment from freeze snapshot without dependency mutation"
  "${python_bin}" -m pip install --no-deps -r "${teacher_base_req_path}"
  if [[ -s "${teacher_override_req_path}" ]]; then
    local requirement
    while IFS= read -r requirement; do
      [[ -n "${requirement}" ]] || continue
      log "Applying teacher override without dependency mutation: ${requirement}"
      "${python_bin}" -m pip install --no-deps "${requirement}"
    done < "${teacher_override_req_path}"
  fi
  verify_freeze_matches "${python_bin}" "${TEACHER_REQ_PATH}" "teacher"
  write_stamp "${TEACHER_VENV}" "${fingerprint}"
  rm -rf "${tmp_dir}"
}

main() {
  ensure_snapshot_files
  run_apt_install
  ensure_uv

  if [[ "${REBUILD_STUDENT_ENV}" == "1" ]]; then
    rebuild_student_env
  else
    log "Skipping student env rebuild"
  fi

  if [[ "${REBUILD_TEACHER_ENV}" == "1" ]]; then
    rebuild_teacher_env
  else
    log "Skipping teacher env rebuild"
  fi

  log "Done"
  printf "Student env: %s\n" "${STUDENT_VENV}"
  printf "Teacher env: %s\n" "${TEACHER_VENV}"
}

main "$@"
