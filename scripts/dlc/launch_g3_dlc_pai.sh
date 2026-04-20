#!/usr/bin/env bash
set -euo pipefail

SNAPSHOT_REPO_DIR="${SNAPSHOT_REPO_DIR:-/mnt/data/ebft-teacher-distribution/code/Distributional-Match-Tuning-g3-dlc}"
SOURCE_TEACHER_CACHE_DIR="${SOURCE_TEACHER_CACHE_DIR:-/mnt/data/ebft-teacher-distribution/teacher_cache_shared_g3_dlc}"
TARGET_REPO_DIR="${TARGET_REPO_DIR:-/root/code/Distributional-Match-Tuning}"
TARGET_TEACHER_CACHE_DIR="${TARGET_TEACHER_CACHE_DIR:-/root/outputs/teacher_cache_shared}"
BOOTSTRAP="${BOOTSTRAP:-${SNAPSHOT_REPO_DIR}/scripts/dlc/bootstrap_g3_dlc_runtime.sh}"
SYNC_DIR_BASE="${SYNC_DIR_BASE:-/mnt/data/ebft-teacher-distribution/dlc_g3_sync}"
TRAIN_LAUNCH_SCRIPT="${TRAIN_LAUNCH_SCRIPT:-scripts/run_G3_rebase_2node_once.sh}"

SYNC_REPO="${SYNC_REPO:-1}"
SYNC_TEACHER_CACHE="${SYNC_TEACHER_CACHE:-1}"
REBUILD_ENVS="${REBUILD_ENVS:-1}"
FORCE_REBUILD_ENVS="${FORCE_REBUILD_ENVS:-0}"
INSTALL_APT_DEPS="${INSTALL_APT_DEPS:-1}"

if [[ ! -f "${BOOTSTRAP}" ]]; then
  echo "[ERROR] bootstrap script not found: ${BOOTSTRAP}" >&2
  exit 1
fi

if [[ "${WORLD_SIZE:-2}" != "2" ]]; then
  echo "[ERROR] This launcher expects WORLD_SIZE=2, got ${WORLD_SIZE:-unset}" >&2
  exit 1
fi

if [[ -z "${RANK:-}" ]]; then
  echo "[ERROR] RANK is required in the DLC runtime environment." >&2
  exit 1
fi

write_sync_text() {
  local dest="$1"
  local value="$2"
  local tmp_file
  tmp_file="$(mktemp)"
  printf '%s\n' "${value}" > "${tmp_file}"
  cp -f "${tmp_file}" "${dest}"
  rm -f "${tmp_file}"
}

copy_sync_file() {
  local src="$1"
  local dest="$2"
  cp -f "${src}" "${dest}"
}

touch_sync_file() {
  local dest="$1"
  local tmp_file
  tmp_file="$(mktemp)"
  : > "${tmp_file}"
  cp -f "${tmp_file}" "${dest}"
  rm -f "${tmp_file}"
}

env \
  SNAPSHOT_REPO_DIR="${SNAPSHOT_REPO_DIR}" \
  SOURCE_TEACHER_CACHE_DIR="${SOURCE_TEACHER_CACHE_DIR}" \
  TARGET_REPO_DIR="${TARGET_REPO_DIR}" \
  TARGET_TEACHER_CACHE_DIR="${TARGET_TEACHER_CACHE_DIR}" \
  SYNC_REPO="${SYNC_REPO}" \
  SYNC_TEACHER_CACHE="${SYNC_TEACHER_CACHE}" \
  REBUILD_ENVS="${REBUILD_ENVS}" \
  FORCE_REBUILD_ENVS="${FORCE_REBUILD_ENVS}" \
  INSTALL_APT_DEPS="${INSTALL_APT_DEPS}" \
  bash "${BOOTSTRAP}"

SYNC_DIR="${SYNC_DIR_BASE}/${JOB_NAME:-g3_dlc_job}"
mkdir -p "${SYNC_DIR}" /root/.ssh /run/sshd
chmod 700 /root/.ssh

export DEBIAN_FRONTEND=noninteractive
apt-get update >/dev/null
apt-get install -y openssh-server openssh-client rsync >/dev/null

ssh-keygen -A >/dev/null 2>&1 || true
grep -q "^PermitRootLogin yes" /etc/ssh/sshd_config || echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
grep -q "^PubkeyAuthentication yes" /etc/ssh/sshd_config || echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config
/usr/sbin/sshd

HOST_NAME="$(hostname -f 2>/dev/null || hostname)"
HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
if [[ -z "${HOST_IP}" ]]; then
  HOST_IP="$(getent ahostsv4 "${HOST_NAME}" | awk 'NR==1 {print $1}')"
fi
if [[ -z "${HOST_IP}" ]]; then
  echo "[ERROR] failed to determine local IPv4 for ${HOST_NAME}" >&2
  exit 1
fi

write_sync_text "${SYNC_DIR}/node_${RANK}.host" "${HOST_NAME}"
write_sync_text "${SYNC_DIR}/node_${RANK}.ip" "${HOST_IP}"

if [[ "${RANK}" == "0" ]]; then
  rm -f "${SYNC_DIR}/head.pub" "${SYNC_DIR}/worker.ready" "${SYNC_DIR}/done" "${SYNC_DIR}/final.rc"

  if [[ ! -f /root/.ssh/dlc_id_ed25519 ]]; then
    ssh-keygen -t ed25519 -N "" -f /root/.ssh/dlc_id_ed25519 >/dev/null
  fi
  copy_sync_file /root/.ssh/dlc_id_ed25519.pub "${SYNC_DIR}/head.pub"

  while [[ ! -f "${SYNC_DIR}/node_1.host" || ! -f "${SYNC_DIR}/node_1.ip" ]]; do
    sleep 2
  done
  while [[ ! -f "${SYNC_DIR}/worker.ready" ]]; do
    sleep 2
  done

  export REPO_ROOT="${TARGET_REPO_DIR}"
  export STUDENT_VENV="${TARGET_REPO_DIR}/.venv"
  export TEACHER_VENV="${TARGET_REPO_DIR}/.teacherVenv"
  export TEACHER_CACHE_DIR="${TARGET_TEACHER_CACHE_DIR}"
  export ARCHIVE_OUTPUT_ROOT="${ARCHIVE_OUTPUT_ROOT:-/mnt/data/ebft-teacher-distribution/outputs_g3_0.99}"
  export OUTPUT_ROOT="${OUTPUT_ROOT:-${ARCHIVE_OUTPUT_ROOT}}"
  export HEAD_NODE="$(cat "${SYNC_DIR}/node_0.host")"
  export HEAD_NODE_IP="$(cat "${SYNC_DIR}/node_0.ip")"
  export WORKER_NODE="$(cat "${SYNC_DIR}/node_1.host")"
  export WORKER_NODE_IP="$(cat "${SYNC_DIR}/node_1.ip")"
  export WORKER_SSH_HOST="${WORKER_NODE_IP}"
  export SSH_USER=root
  export SSH_OPTS="-i /root/.ssh/dlc_id_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

  cd "${REPO_ROOT}"
  rc=0
  bash "${TRAIN_LAUNCH_SCRIPT}" || rc=$?

  write_sync_text "${SYNC_DIR}/final.rc" "${rc}"
  touch_sync_file "${SYNC_DIR}/done"
  exit "${rc}"
fi

while [[ ! -f "${SYNC_DIR}/head.pub" ]]; do
  sleep 2
done
cat "${SYNC_DIR}/head.pub" >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
touch_sync_file "${SYNC_DIR}/worker.ready"

while [[ ! -f "${SYNC_DIR}/done" ]]; do
  sleep 30
done
exit "$(cat "${SYNC_DIR}/final.rc" 2>/dev/null || echo 0)"
