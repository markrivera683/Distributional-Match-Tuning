# DenseMask Cache Delivery README

This is the clean delivery version based on the original
`Distributional-Match-Tuning.zip`.

It only adds the EBFT dense 4D strided-mask cache optimization. It does not
include sparse masks, BlockMask, distribution-aware mask changes, teacher
changes, reward changes, or training-setting changes.

## Enable

```bash
export EBFT_CACHE_STRIDED_MASKS=1
export EBFT_CACHE_STRIDED_MASKS_MAX_MB=1024
```

Then run the existing training scripts normally.

## Validate

```bash
python scripts/validate_densemask_cache_equivalence.py --device cpu --include-bf16
python scripts/benchmark_densemask_cache.py --device cpu
```

More details are in `docs/DENSEMASK_CACHE_DELIVERY.md`.

The local delivery smoke result is recorded in
`docs/DENSEMASK_CACHE_SMOKE_RESULT.md`.
