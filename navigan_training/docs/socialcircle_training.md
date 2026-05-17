# Training NaviGAN with SocialCircle pooling

Quick reference for the SocialCircle pooling variant. For the design rationale, see [`integration_into_navigan.md`](integration_into_navigan.md).

## What this is

A `pooling_type` option that swaps NaviGAN's social pooling module from `PoolHiddenNet` / `SocialPooling` to `SocialCircle` (Wong et al., CVPR 2024). Everything else in the model — intention/force branches, attention fusion, GAN losses, hinge `resist_loss` — is unchanged.

**Per target pedestrian, SocialCircle**:
1. Splits the surrounding disk into `circle_partitions` equal angular sectors.
2. Per sector, aggregates two factors from the neighbours in that sector:
   - mean distance to target
   - density (neighbour count)
3. Concatenates the resulting `(partitions × 2)` social feature with the encoder hidden state.
4. Projects to `bottleneck_dim` via an MLP.

The angular extraction is parameter-free; only the post-MLP is learnable.

## Quick start

Single GPU:

```bash
python train.py --config configs/zara1_socialcircle.yaml
```

Multi-GPU via torchrun (UCT HPC):

```bash
torchrun --nproc_per_node=4 train.py --distributed \
    --config configs/zara1_socialcircle.yaml
```

Smoke test (10 iters):

```bash
python train.py --config configs/zara1_socialcircle.yaml --max-iter 10
```

Output goes to `runs/zara1_socialcircle/navigan_checkpoint.pt`.

## Configurable parameters

All parameters in `configs/zara1_socialcircle.yaml`. Only the SocialCircle-specific knobs are listed here — see `configs/zara1_scratch.yaml` for the rest of the (shared) training schedule.

| Param | Default | What it does |
|-------|---------|--------------|
| `pooling_type` | `social_circle` | Selects this module. Other valid values: `spool`, `pool_net`, `null` |
| `circle_partitions` | `8` | Number of angular sectors. More partitions → finer directional resolution, more MLP input dim, more params |
| `bottleneck_dim` | `32` | Output dim of the SocialCircle MLP. Must match the rest of the generator's `bottleneck_dim`. Don't change in isolation |
| `mlp_dim` | `64` | Hidden width of the SocialCircle MLP. Inherits the generator's `mlp_dim` — shared across all MLPs |

CLI overrides (work for any config):

| Flag | Effect |
|------|--------|
| `--max-iter N` | Cap `num_iterations` (smoke tests) |
| `--d_safe X` | Override hinge safety margin in metres |
| `--resist-weight W` | Override `resist_loss_weight` |

## Suggested starting parameters

The supplied `zara1_socialcircle.yaml` is a good starting point — it matches the benchmark zara1 checkpoint's training schedule exactly except for the pooling swap. Train it first as your baseline.

Three sweeps worth running, in order:

### 1. Baseline (start here)

```bash
python train.py --config configs/zara1_socialcircle.yaml
```

`circle_partitions=8`. Paper default. Roughly one neighbour per sector at typical zara1 ped densities — a sensible operating point. **Train this one first, evaluate, only proceed to sweeps if results justify it.**

### 2. Finer angular resolution

Copy the YAML, change one line:

```yaml
circle_partitions: 16
output_dir: runs/zara1_socialcircle_p16
```

Doubles the SocialCircle feature dim (`16 × 2 = 32` features vs `8 × 2 = 16`). MLP head is slightly bigger. Test whether more directional granularity helps; risk is overfitting on small zara1 (~30k train tuples).

### 3. Coarser, smaller model

```yaml
circle_partitions: 4
output_dir: runs/zara1_socialcircle_p4
```

Useful as ablation lower bound — if `p4` already matches `p8`, the angular resolution isn't doing much and the gains (if any) are coming from the MLP head, not the angular encoding.

### Optional: stricter safety

If the baseline shows lower `resist_count` than the benchmark, push further:

```bash
python train.py --config configs/zara1_socialcircle.yaml \
    --d_safe 0.8 --resist-weight 5.0
```

Same combo as `zara1_safe.yaml`. Trains a safer model at the cost of some ADE/FDE.

## Comparing against the benchmark checkpoint

After training completes, three commands:

```bash
# (a) Eval the new run's best_safety variant against zara1 val split
python scripts/eval_checkpoint.py \
    runs/zara1_socialcircle/navigan_checkpoint.pt \
    --dataset zara1 --split val --variant best_safety

# (b) Eval the production benchmark checkpoint on the same split
python scripts/eval_checkpoint.py \
    /path/to/benchmark_zara1_with_model.pt \
    --dataset zara1 --split val --variant final

# (c) Plot training curves + prediction samples for the new run
python scripts/plot_training.py runs/zara1_socialcircle/navigan_checkpoint.pt
python scripts/plot_predictions.py runs/zara1_socialcircle/navigan_checkpoint.pt \
    --variant best_safety -o plots/socialcircle/
```

The new checkpoint stores three best-variant snapshots (`best_ade`, `best_fde`, `best_safety`) thanks to the existing multi-criterion tracking. Compare each against the benchmark's `final` to see which trade-off the new architecture lands at.

**Suggested pass criteria** (loose — adjust to taste):

- `ADE` within 10% of benchmark *or* better
- `FDE` within 10% of benchmark *or* better
- `resist_count` strictly lower at the same `d_safe`
- Orin p95 inference latency under 80 ms (test in `navigan_node.py` with `enable_profiling:=true` after copying the checkpoint over)

## Caveats

1. **Not resume-compatible with `benchmark_zara1_with_model.pt`.** The SocialCircle MLP head has a different parameter shape than `PoolHiddenNet` / `SocialPooling`. Use `resume_from: null` and train from scratch.

2. **Per-step pooling path (`pool_every_timestep: true`) is wired but untested at scale.** Production currently runs `pool_every_timestep: false` (encoder-only pooling) to avoid the CUDA race documented in `navigan/CUDA_RACE_CONDITION_EXPLAINED.md`. SocialCircle's `sector_idx.clamp` defends against NaN positions analogously to `SocialPooling`'s `grid_pos.clamp`, but if you enable per-step pooling, monitor for instabilities and check `force_rel_pos` magnitudes during the first few epochs.

3. **Velocity / move-direction factors omitted.** The original SocialCircle paper supports four factors (distance, density, velocity, move-direction). Only distance + density are wired here so the forward signature stays identical to `PoolHiddenNet` (positions only). Adding velocity-aware variants would require threading `obs_traj_rel` into the pooling call — straightforward but a future enhancement.

4. **First-epoch BatchNorm note.** Current config has `batch_norm: false` — no action needed. If you enable it, the SocialCircle MLP picks it up automatically via `make_mlp`.
