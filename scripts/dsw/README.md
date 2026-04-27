# DSW Lifecycle Helpers

Utilities for surviving Alibaba PAI **DSW** (Data-Science-Workshop) container
rebuilds. DLC users do not need any of this — DLC bakes the venv into the
docker image, so there is no rootfs-loss problem.

## The DSW persistence model

```
/                    overlay rootfs       wiped on image pull
/tmp                 ext4 on /dev/vda     wiped on image pull
/mnt/workspace       ext4 on /dev/vda     wiped on image pull   (!)
/home/admin/*        ext4 on /dev/vda     wiped on image pull
/mnt/data            ossfs2  (REMOTE)     PERSISTENT
```

DSW rebuilds the entire local block device (`/dev/vda`) on every image
pull, so anywhere except `/mnt/data` is treated as ephemeral. The git
repo itself lives on OSS at `/mnt/data/ebft-distribution-new/code`, which
is why these helper scripts are checked into the repo — they survive too.

## After every DSW restart

```bash
cd /mnt/data/ebft-distribution-new/code
bash scripts/dsw/bootstrap_after_restart.sh
```

The bootstrap is idempotent and:

1. Copies the GitHub SSH private key from `/mnt/data/dsw-secrets/.ssh/`
   into `~/.ssh/`, chmod `0600` (OSS-fuse cannot honor POSIX bits, so the
   on-disk source is permissive; the local copy is strict).
2. Writes `~/.ssh/config` so `git@github.com` tunnels through
   `ssh.github.com:443`.
3. Pre-trusts `[ssh.github.com]:443` in `~/.ssh/known_hosts`.
4. If `/mnt/data/dsw-secrets/venv-snapshots/{.venv,.teacherVenv}.tar.zst`
   exists, extracts it to `/mnt/workspace/venvs/`; otherwise warns the
   user to run `scripts/setup_env.sh`.
5. Symlinks the per-repo `.venv` / `.teacherVenv` to the local-ext4 venv
   roots.
6. Runs `ssh -T git@github.com` as a health check.

## Persistent secrets layout (on OSS)

```
/mnt/data/dsw-secrets/
├── README.md
├── .ssh/
│   ├── id_ed25519_github          (GitHub auth, ed25519, in use)
│   └── id_ed25519_github.pub
└── venv-snapshots/                (optional; speeds up venv recovery)
    ├── .venv.tar.zst              (tar --zstd of /mnt/workspace/venvs/.venv)
    └── .teacherVenv.tar.zst
```

Provisioning a new DSW (or rotating the key) is just:

```bash
ssh-keygen -t ed25519 -C "<comment>" \
    -f /mnt/data/dsw-secrets/.ssh/id_ed25519_github -N ''
# paste /mnt/data/dsw-secrets/.ssh/id_ed25519_github.pub into
# https://github.com/settings/ssh/new
```

## Snapshotting a venv (optional, for fast cold-boot)

After a `scripts/setup_env.sh` produces a working venv:

```bash
mkdir -p /mnt/data/dsw-secrets/venv-snapshots
cd /mnt/workspace/venvs
tar --zstd -cf /mnt/data/dsw-secrets/venv-snapshots/.venv.tar.zst .venv
tar --zstd -cf /mnt/data/dsw-secrets/venv-snapshots/.teacherVenv.tar.zst .teacherVenv
```

Cold-boot recovery becomes ~30–90 s per venv (vs 15–30 min reinstall).

## Why not put the venv directly on `/mnt/data` / OSS?

OSS is fuse-mounted via `ossfs2`, which makes per-file `stat()` ~1000× slower
than local ext4. A typical Python venv triggers tens of thousands of stats
during `import` resolution; running it directly off OSS makes startup time
unbearable. Tarball-on-OSS + extract-on-boot avoids both this problem and the
"venv is wiped on image pull" problem.

## Why not put `~/.ssh` on OSS directly?

OpenSSH refuses keys whose POSIX permissions are looser than `0600`. OSS-fuse
ignores `chmod` and reports its own default mode, so a key sitting on OSS will
appear `0777` to OpenSSH and get rejected. The bootstrap copies the key onto
local ext4 (`~/.ssh/`) where `chmod 0600` actually sticks.
