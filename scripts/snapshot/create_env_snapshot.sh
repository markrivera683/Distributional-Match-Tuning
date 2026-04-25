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
  bash scripts/snapshot/create_env_snapshot.sh

Create a restorable environment snapshot for the current repo.

Default snapshot root:
  /mnt/data/ebft-teacher-distribution/snapshots

Included by default:
  - repo working tree (with .git, without repo outputs/)
  - .venv
  - .teacherVenv
  - /usr/local/lib/python3.12/site-packages for the student env
  - /root/.cache/huggingface
  - /root/.cache/pip
  - /root/outputs/teacher_cache_shared

Important:
  - archives are plain .tar files to favor restore speed
  - repo outputs/ is excluded by default
  - snapshotting large caches can take a while

Environment variables:
  REPO_DIR
  SNAPSHOT_ROOT
  SNAPSHOT_NAME
  SNAPSHOT_DIR
  SOURCE_STUDENT_VENV
  SOURCE_TEACHER_VENV
  SOURCE_STUDENT_BASE_SITE_PACKAGES
  SOURCE_HF_CACHE_DIR
  SOURCE_PIP_CACHE_DIR
  SOURCE_TEACHER_CACHE_DIR
  INCLUDE_HF_CACHE=1|0
  INCLUDE_PIP_CACHE=1|0
  INCLUDE_TEACHER_CACHE=1|0
  COPY_GIT_DIR=1|0
  COPY_REPO_OUTPUTS=1|0
  OVERWRITE=1|0
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR_DEFAULT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REPO_DIR="${REPO_DIR:-${REPO_DIR_DEFAULT}}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-/mnt/data/ebft-teacher-distribution/snapshots}"
SNAPSHOT_NAME="${SNAPSHOT_NAME:-Distributional-Match-Tuning_env_$(date +%Y%m%d_%H%M%S)}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-${SNAPSHOT_ROOT}/${SNAPSHOT_NAME}}"

SOURCE_STUDENT_VENV="${SOURCE_STUDENT_VENV:-${REPO_DIR}/.venv}"
SOURCE_TEACHER_VENV="${SOURCE_TEACHER_VENV:-${REPO_DIR}/.teacherVenv}"
SOURCE_STUDENT_BASE_SITE_PACKAGES="${SOURCE_STUDENT_BASE_SITE_PACKAGES:-/usr/local/lib/python3.12/site-packages}"
SOURCE_HF_CACHE_DIR="${SOURCE_HF_CACHE_DIR:-/root/.cache/huggingface}"
SOURCE_PIP_CACHE_DIR="${SOURCE_PIP_CACHE_DIR:-/root/.cache/pip}"
SOURCE_TEACHER_CACHE_DIR="${SOURCE_TEACHER_CACHE_DIR:-/root/outputs/teacher_cache_shared}"

INCLUDE_HF_CACHE="${INCLUDE_HF_CACHE:-1}"
INCLUDE_PIP_CACHE="${INCLUDE_PIP_CACHE:-1}"
INCLUDE_TEACHER_CACHE="${INCLUDE_TEACHER_CACHE:-1}"
COPY_GIT_DIR="${COPY_GIT_DIR:-1}"
COPY_REPO_OUTPUTS="${COPY_REPO_OUTPUTS:-0}"
OVERWRITE="${OVERWRITE:-0}"

ARCHIVE_DIR="${SNAPSHOT_DIR}/archives"
METADATA_DIR="${SNAPSHOT_DIR}/metadata"
TOOLS_DIR="${SNAPSHOT_DIR}/tools"
WHEELS_DIR="${SNAPSHOT_DIR}/wheels"

ensure_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || die "Missing required command: ${cmd}"
}

ensure_paths() {
  [[ -d "${REPO_DIR}" ]] || die "REPO_DIR not found: ${REPO_DIR}"
  [[ -x "${SOURCE_STUDENT_VENV}/bin/python" ]] || die "Student env missing: ${SOURCE_STUDENT_VENV}"
  [[ -x "${SOURCE_TEACHER_VENV}/bin/python" ]] || die "Teacher env missing: ${SOURCE_TEACHER_VENV}"
  [[ -d "${SOURCE_STUDENT_BASE_SITE_PACKAGES}" ]] || die "Student base site-packages missing: ${SOURCE_STUDENT_BASE_SITE_PACKAGES}"
  if [[ "${INCLUDE_HF_CACHE}" == "1" ]]; then
    [[ -d "${SOURCE_HF_CACHE_DIR}" ]] || die "HF cache dir not found: ${SOURCE_HF_CACHE_DIR}"
  fi
  if [[ "${INCLUDE_PIP_CACHE}" == "1" ]]; then
    [[ -d "${SOURCE_PIP_CACHE_DIR}" ]] || die "pip cache dir not found: ${SOURCE_PIP_CACHE_DIR}"
  fi
  if [[ "${INCLUDE_TEACHER_CACHE}" == "1" ]]; then
    [[ -d "${SOURCE_TEACHER_CACHE_DIR}" ]] || die "teacher cache dir not found: ${SOURCE_TEACHER_CACHE_DIR}"
  fi
}

prepare_snapshot_dir() {
  if [[ -e "${SNAPSHOT_DIR}" ]]; then
    if [[ "${OVERWRITE}" != "1" ]]; then
      die "SNAPSHOT_DIR already exists: ${SNAPSHOT_DIR}. Set OVERWRITE=1 to replace it."
    fi
    log "Removing existing snapshot dir ${SNAPSHOT_DIR}"
    rm -rf "${SNAPSHOT_DIR}"
  fi
  mkdir -p "${ARCHIVE_DIR}" "${METADATA_DIR}" "${TOOLS_DIR}" "${WHEELS_DIR}"
}

archive_directory_contents() {
  local src_dir="$1"
  local archive_path="$2"
  [[ -d "${src_dir}" ]] || die "archive source dir not found: ${src_dir}"
  log "Archiving ${src_dir} -> ${archive_path}"
  mkdir -p "$(dirname "${archive_path}")"
  (
    cd "${src_dir}"
    tar --numeric-owner -cpf "${archive_path}" .
  )
}

archive_repo_snapshot() {
  local archive_path="${ARCHIVE_DIR}/repo.tar"
  local tar_args=(
    --numeric-owner
    --exclude=".venv"
    --exclude=".teacherVenv"
    --exclude=".cursor"
    --exclude="__pycache__"
    --exclude=".pytest_cache"
    --exclude=".mypy_cache"
    --exclude=".ruff_cache"
  )

  if [[ "${COPY_GIT_DIR}" != "1" ]]; then
    tar_args+=(--exclude=".git")
  fi
  if [[ "${COPY_REPO_OUTPUTS}" != "1" ]]; then
    tar_args+=(--exclude="outputs")
  fi

  log "Archiving repo working tree -> ${archive_path}"
  (
    cd "${REPO_DIR}"
    tar "${tar_args[@]}" -cpf "${archive_path}" .
  )
}

write_python_metadata() {
  local label="$1"
  local python_bin="$2"
  local freeze_path="${METADATA_DIR}/${label}.pip_freeze.txt"
  local report_path="${METADATA_DIR}/${label}.python_report.txt"
  local show_path="${METADATA_DIR}/${label}.pip_show.txt"

  log "Writing ${label} environment metadata"
  "${python_bin}" -m pip freeze > "${freeze_path}"
  "${python_bin}" -m pip show openrlhf torch deepspeed ray flash-attn transformers vllm > "${show_path}" 2>&1 || true
  "${python_bin}" - "${label}" > "${report_path}" <<'PY'
import importlib
import json
import site
import sys

label = sys.argv[1]
modules_by_label = {
    "student": ["torch", "deepspeed", "ray", "flash_attn", "transformers", "openrlhf"],
    "teacher": ["torch", "transformers", "vllm"],
}
report = {
    "label": label,
    "sys_executable": sys.executable,
    "sys_prefix": sys.prefix,
    "sys_base_prefix": sys.base_prefix,
    "version": sys.version,
    "site_packages": site.getsitepackages(),
    "modules": {},
}
for module_name in modules_by_label.get(label, []):
    try:
        module = importlib.import_module(module_name)
        report["modules"][module_name] = {
            "version": getattr(module, "__version__", "n/a"),
            "path": getattr(module, "__file__", "<namespace>"),
        }
    except Exception as exc:
        report["modules"][module_name] = {
            "import_error": repr(exc),
        }
print(json.dumps(report, indent=2, sort_keys=True))
PY
}

write_selected_env_vars() {
  local output_path="${METADATA_DIR}/selected_env_vars.json"
  python3 - > "${output_path}" <<'PY'
import json
import os

prefixes = (
    "CUDA",
    "HF_",
    "PIP_",
    "PYTORCH_",
    "RAY_",
    "TOKENIZERS_",
    "TORCH_",
    "VLLM_",
    "WANDB_",
    "NCCL_",
)
selected = {}
for key, value in sorted(os.environ.items()):
    if key.startswith(prefixes) or key in {"MODEL_PATH", "HF_ENDPOINT"}:
        selected[key] = value
print(json.dumps(selected, indent=2, sort_keys=True))
PY
}

write_machine_metadata() {
  log "Writing machine metadata"
  uname -a > "${METADATA_DIR}/uname.txt"
  if [[ -f /etc/os-release ]]; then
    cp /etc/os-release "${METADATA_DIR}/os-release"
  fi
  python3 --version > "${METADATA_DIR}/system_python.txt"
  nvidia-smi > "${METADATA_DIR}/nvidia-smi.txt" 2>&1 || true
  df -h "${REPO_DIR}" "${SNAPSHOT_ROOT}" > "${METADATA_DIR}/df.txt" 2>&1 || true
  write_selected_env_vars
}

write_repo_metadata() {
  if [[ ! -d "${REPO_DIR}/.git" ]]; then
    warn "Skipping git metadata because ${REPO_DIR} has no .git directory."
    return
  fi

  log "Writing git metadata"
  git -C "${REPO_DIR}" rev-parse HEAD > "${METADATA_DIR}/git_head.txt"
  git -C "${REPO_DIR}" branch --show-current > "${METADATA_DIR}/git_branch.txt" || true
  git -C "${REPO_DIR}" status --short > "${METADATA_DIR}/git_status.txt"
  git -C "${REPO_DIR}" diff --binary > "${METADATA_DIR}/git_diff.patch"
  git -C "${REPO_DIR}" diff --binary --staged > "${METADATA_DIR}/git_diff_staged.patch"
  git -C "${REPO_DIR}" ls-files --others --exclude-standard > "${METADATA_DIR}/git_untracked.txt"
}

write_component_sizes() {
  local size_path="${METADATA_DIR}/component_sizes.txt"
  {
    du -sh "${REPO_DIR}"
    du -sh "${SOURCE_STUDENT_VENV}"
    du -sh "${SOURCE_TEACHER_VENV}"
    du -sh "${SOURCE_STUDENT_BASE_SITE_PACKAGES}"
    if [[ "${INCLUDE_HF_CACHE}" == "1" ]]; then
      du -sh "${SOURCE_HF_CACHE_DIR}"
    fi
    if [[ "${INCLUDE_PIP_CACHE}" == "1" ]]; then
      du -sh "${SOURCE_PIP_CACHE_DIR}"
    fi
    if [[ "${INCLUDE_TEACHER_CACHE}" == "1" ]]; then
      du -sh "${SOURCE_TEACHER_CACHE_DIR}"
    fi
  } > "${size_path}" 2>&1 || true
}

write_manifest() {
  local source_student_python_version
  local source_teacher_python_version
  local source_student_python_mm
  local source_teacher_python_mm

  source_student_python_version="$("${SOURCE_STUDENT_VENV}/bin/python" - <<'PY'
import sys
print(sys.version.split()[0])
PY
)"
  source_teacher_python_version="$("${SOURCE_TEACHER_VENV}/bin/python" - <<'PY'
import sys
print(sys.version.split()[0])
PY
)"
  source_student_python_mm="$("${SOURCE_STUDENT_VENV}/bin/python" - <<'PY'
import sys
print(f"{sys.version_info[0]}.{sys.version_info[1]}")
PY
)"
  source_teacher_python_mm="$("${SOURCE_TEACHER_VENV}/bin/python" - <<'PY'
import sys
print(f"{sys.version_info[0]}.{sys.version_info[1]}")
PY
)"

  {
    echo "# Auto-generated snapshot manifest"
    echo "SNAPSHOT_CREATED_UTC=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    printf "SNAPSHOT_NAME=%q\n" "${SNAPSHOT_NAME}"
    printf "SNAPSHOT_DIR=%q\n" "${SNAPSHOT_DIR}"
    printf "SOURCE_REPO_DIR=%q\n" "${REPO_DIR}"
    printf "SOURCE_STUDENT_VENV=%q\n" "${SOURCE_STUDENT_VENV}"
    printf "SOURCE_TEACHER_VENV=%q\n" "${SOURCE_TEACHER_VENV}"
    printf "SOURCE_STUDENT_BASE_SITE_PACKAGES=%q\n" "${SOURCE_STUDENT_BASE_SITE_PACKAGES}"
    printf "SOURCE_HF_CACHE_DIR=%q\n" "${SOURCE_HF_CACHE_DIR}"
    printf "SOURCE_PIP_CACHE_DIR=%q\n" "${SOURCE_PIP_CACHE_DIR}"
    printf "SOURCE_TEACHER_CACHE_DIR=%q\n" "${SOURCE_TEACHER_CACHE_DIR}"
    printf "SOURCE_STUDENT_PYTHON_VERSION=%q\n" "${source_student_python_version}"
    printf "SOURCE_TEACHER_PYTHON_VERSION=%q\n" "${source_teacher_python_version}"
    printf "SOURCE_STUDENT_PYTHON_MM=%q\n" "${source_student_python_mm}"
    printf "SOURCE_TEACHER_PYTHON_MM=%q\n" "${source_teacher_python_mm}"
    printf "INCLUDE_HF_CACHE=%q\n" "${INCLUDE_HF_CACHE}"
    printf "INCLUDE_PIP_CACHE=%q\n" "${INCLUDE_PIP_CACHE}"
    printf "INCLUDE_TEACHER_CACHE=%q\n" "${INCLUDE_TEACHER_CACHE}"
    printf "COPY_GIT_DIR=%q\n" "${COPY_GIT_DIR}"
    printf "COPY_REPO_OUTPUTS=%q\n" "${COPY_REPO_OUTPUTS}"
  } > "${METADATA_DIR}/snapshot_manifest.env"
}

copy_tools() {
  log "Copying snapshot tools into ${TOOLS_DIR}"
  cp "${SCRIPT_DIR}/create_env_snapshot.sh" "${TOOLS_DIR}/create_env_snapshot.sh"
  cp "${SCRIPT_DIR}/restore_env_snapshot.sh" "${TOOLS_DIR}/restore_env_snapshot.sh"
  chmod +x "${TOOLS_DIR}/create_env_snapshot.sh" "${TOOLS_DIR}/restore_env_snapshot.sh"
}

vendor_local_file_dependencies() {
  log "Vendoring local file-backed dependencies"
  python3 - "${METADATA_DIR}/student.pip_freeze.txt" "${METADATA_DIR}/teacher.pip_freeze.txt" "${WHEELS_DIR}" <<'PY'
import hashlib
import json
import shutil
import sys
import urllib.parse
from pathlib import Path

freeze_paths = [Path(sys.argv[1]), Path(sys.argv[2])]
wheel_dir = Path(sys.argv[3])
wheel_dir.mkdir(parents=True, exist_ok=True)

records = []
copied = {}

for freeze_path in freeze_paths:
    if not freeze_path.exists():
        continue
    for raw_line in freeze_path.read_text().splitlines():
        line = raw_line.strip()
        if "@ file://" not in line:
            continue
        _, uri = line.split("@", 1)
        parsed = urllib.parse.urlparse(uri.strip())
        source_path = Path(urllib.parse.unquote(parsed.path))
        if not source_path.exists():
            records.append(
                {
                    "freeze_file": freeze_path.name,
                    "requirement": line,
                    "source_path": str(source_path),
                    "copied_path": None,
                    "sha256": None,
                    "missing": True,
                }
            )
            continue

        if str(source_path) not in copied:
            target_path = wheel_dir / source_path.name
            if target_path.exists():
                stem = source_path.stem
                suffix = source_path.suffix
                idx = 2
                while target_path.exists():
                    target_path = wheel_dir / f"{stem}.{idx}{suffix}"
                    idx += 1
            shutil.copy2(source_path, target_path)
            digest = hashlib.sha256(target_path.read_bytes()).hexdigest()
            copied[str(source_path)] = {
                "copied_path": str(target_path),
                "sha256": digest,
            }

        records.append(
            {
                "freeze_file": freeze_path.name,
                "requirement": line,
                "source_path": str(source_path),
                "copied_path": copied[str(source_path)]["copied_path"],
                "sha256": copied[str(source_path)]["sha256"],
                "missing": False,
            }
        )

(wheel_dir / "local_file_dependencies.json").write_text(
    json.dumps(records, indent=2, sort_keys=True) + "\n"
)
PY
}

write_archive_inventory() {
  log "Writing archive inventory"
  python3 - "${ARCHIVE_DIR}" > "${METADATA_DIR}/archive_inventory.json" <<'PY'
import json
import os
import sys
from pathlib import Path

archive_dir = Path(sys.argv[1])
items = []
for path in sorted(archive_dir.glob("*.tar")):
    stat = path.stat()
    items.append(
        {
            "name": path.name,
            "size_bytes": stat.st_size,
        }
    )
print(json.dumps(items, indent=2, sort_keys=True))
PY
}

write_snapshot_readme() {
  cat > "${SNAPSHOT_DIR}/README.txt" <<EOF
Distributional-Match-Tuning environment snapshot

Created at:
  $(date -u '+%Y-%m-%d %H:%M:%S UTC')

What is included:
  - repo working tree archive
  - student env archive (.venv)
  - student base site-packages archive
  - teacher env archive (.teacherVenv)
  - Hugging Face cache archive: ${INCLUDE_HF_CACHE}
  - pip cache archive: ${INCLUDE_PIP_CACHE}
  - teacher cache archive: ${INCLUDE_TEACHER_CACHE}
  - metadata about git state, imports, versions, and source paths

Quick restore:
  bash "${SNAPSHOT_DIR}/tools/restore_env_snapshot.sh"

Safer restore into an existing machine:
  ALLOW_OVERWRITE=1 bash "${SNAPSHOT_DIR}/tools/restore_env_snapshot.sh"

Compatibility assumptions:
  - Linux x86_64
  - Python ${SOURCE_STUDENT_VENV}/bin/python compatible major.minor
  - NVIDIA driver / CUDA userland compatible with the copied wheels
EOF
}

main() {
  ensure_cmd tar
  ensure_cmd python3
  ensure_paths
  prepare_snapshot_dir

  write_machine_metadata
  write_repo_metadata
  write_python_metadata "student" "${SOURCE_STUDENT_VENV}/bin/python"
  write_python_metadata "teacher" "${SOURCE_TEACHER_VENV}/bin/python"
  write_component_sizes
  write_manifest

  archive_repo_snapshot
  archive_directory_contents "${SOURCE_STUDENT_VENV}" "${ARCHIVE_DIR}/student_venv.tar"
  archive_directory_contents "${SOURCE_TEACHER_VENV}" "${ARCHIVE_DIR}/teacher_venv.tar"
  archive_directory_contents "${SOURCE_STUDENT_BASE_SITE_PACKAGES}" "${ARCHIVE_DIR}/student_base_site_packages.tar"

  if [[ "${INCLUDE_HF_CACHE}" == "1" ]]; then
    archive_directory_contents "${SOURCE_HF_CACHE_DIR}" "${ARCHIVE_DIR}/huggingface_cache.tar"
  fi
  if [[ "${INCLUDE_PIP_CACHE}" == "1" ]]; then
    archive_directory_contents "${SOURCE_PIP_CACHE_DIR}" "${ARCHIVE_DIR}/pip_cache.tar"
  fi
  if [[ "${INCLUDE_TEACHER_CACHE}" == "1" ]]; then
    archive_directory_contents "${SOURCE_TEACHER_CACHE_DIR}" "${ARCHIVE_DIR}/teacher_cache_shared.tar"
  fi

  vendor_local_file_dependencies
  copy_tools
  write_archive_inventory
  write_snapshot_readme

  log "Snapshot ready"
  printf "Snapshot dir:\n  %s\n" "${SNAPSHOT_DIR}"
  printf "Restore command:\n  bash %q\n" "${SNAPSHOT_DIR}/tools/restore_env_snapshot.sh"
}

main "$@"
