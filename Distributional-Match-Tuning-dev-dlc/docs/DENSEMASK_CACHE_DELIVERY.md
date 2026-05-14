# EBFT Dense 4D Mask Cache Delivery

## Scope

This delivery keeps the original EBFT dense 4D additive strided mask as the
reference implementation. It does not introduce sparse masks, BlockMask,
distribution-aware masks, teacher changes, reward changes, or training-setting
changes.

The only runtime optimization is:

```bash
EBFT_CACHE_STRIDED_MASKS=1
```

When enabled, repeated rollout calls with the same dense-mask shape reuse the
same dense 4D attention mask and `position_ids` instead of rebuilding them every
time.

## Why This Is Semantics-Preserving

For the common `document_masking=False` path, the dense 4D strided mask is fully
determined by:

- batch size
- sequence lengths
- prompt/context/generation lengths
- stride
- number of blocks
- device and dtype

The cache key uses exactly this shape/grid metadata. `doc_ids` are intentionally
not part of the key because the original function fills the document mask with
`True` when `document_masking=False`.

For `document_masking=True`, caching is disabled because the mask depends on
per-sample document IDs.

## Controls

| Environment variable | Default | Meaning |
| --- | ---: | --- |
| `EBFT_CACHE_STRIDED_MASKS` | `0` | Enable dense 4D mask cache when set to `1`. |
| `EBFT_CACHE_STRIDED_MASKS_MAX_MB` | `1024` | LRU cache memory budget in MB. |

The default is intentionally off, so existing runs behave exactly like the
original code unless the optimization is explicitly enabled.

## Touched Files

- `openrlhf/models/utils.py`
- `scripts/validate_densemask_cache_equivalence.py`
- `scripts/benchmark_densemask_cache.py`
- `docs/DENSEMASK_CACHE_DELIVERY.md`

## Validation

Run the CPU equivalence check:

```bash
python scripts/validate_densemask_cache_equivalence.py --device cpu --include-bf16
```

Expected result:

```text
PASS: dense strided-mask cache is equivalent for tested shapes.
```

Run the micro-benchmark:

```bash
python scripts/benchmark_densemask_cache.py --device cpu
```

For GPU:

```bash
python scripts/benchmark_densemask_cache.py --device cuda
```

## Recommended Training Usage

```bash
export EBFT_CACHE_STRIDED_MASKS=1
export EBFT_CACHE_STRIDED_MASKS_MAX_MB=1024
```

Then run the existing EBFT / G1 / G2 / G3 scripts as usual.

## Observed Internal Benchmark

On our single-A800 smoke benchmark, dense mask caching reduced EBFT step time
from about `4.68s/step` to about `1.60s/step` under the same small benchmark
setting, roughly `2.9x` faster. With a larger batch that better uses memory,
sample throughput improved further. Treat these as engineering smoke numbers,
not final training-quality claims.

## Non-Goals

This delivery intentionally does not include:

- sparse / BlockMask backend
- distribution-aware mask design
- FlashAttention forcing for dense 4D masks
- teacher or reward changes
- hyperparameter / setting changes

Those should remain separate research branches because they can change runtime
semantics or introduce higher engineering risk.
