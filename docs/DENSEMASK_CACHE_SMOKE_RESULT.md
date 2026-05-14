# DenseMask Cache Training Smoke Result

Date: 2026-05-14

## What Was Tested

This smoke test used the clean delivery package, not the acceleration research
worktree. It enabled only the dense 4D strided-mask cache:

```bash
export EBFT_CACHE_STRIDED_MASKS=1
export EBFT_CACHE_STRIDED_MASKS_MAX_MB=1024
```

The test ran a single-GPU EBFT training smoke with:

- model: local `Qwen3.5-0.8B-Base`
- reward path: `cf_l1oo`
- target mode: `single`
- teacher: disabled
- max samples: `2`
- target: reach `Global step 1`

## Result

The run reached:

```text
Global step 1
```

Strict error scan found no:

- `Traceback`
- `ERROR`
- `OutOfMemory`
- `CUDA out of memory`
- `RuntimeError`

The teacher gate behaved as expected:

```text
cf_target_mode=single
teacher_in_reward = False
```

GPU memory was released after completion.

## Log Location On This Machine

```text
/root/autodl-tmp/outputs/densemask_cache_delivery_smoke_20260514/train.log
```

This path is not required by the package; it records the local validation run.
