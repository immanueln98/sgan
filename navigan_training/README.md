# navigan_training

Standalone training package for the **NaviGAN late-attention generator** used by the ROS2 social-navigation stack (`navigan_node`) on the Husky / Jetson Orin platform. No ROS dependencies — pure PyTorch. Designed to be split into its own git repo.

The package supports:

1. **From-scratch training** on ETH/UCY trajectory datasets (Gupta et al. SGAN format).
2. **Resuming from existing checkpoints** (e.g. `benchmark_zara1_with_model.pt`) without architecture changes.
3. **Tunable social-repulsion loss** so generated paths maintain configurable clearance from pedestrians.
4. **Multi-GPU training** on UCT HPC via `torchrun` + Slurm.

---

## Checkpoint findings (`benchmark_zara1_with_model.pt`)

These observations come from inspecting the production checkpoint with `scripts/inspect_checkpoint.py`. They drive several design choices in this repo — read before editing.

| Property | Value | Implication |
|---|---|---|
| Model class | `LateAttentionFullGenerator` | Two parallel encoder/decoder branches (intention + social) fused per-timestep via an attention MLP |
| Total parameters | **~101,924** | Tiny — single-GPU is sufficient for dev; DDP only useful for scale-out |
| `noise_dim` | `(0,)` | **Deterministic** — no stochastic variety. Changing this requires retraining from scratch (state-dict shapes diverge). |
| `best_k` | `1` | No variety loss in original training |
| `pooling_type` | `spool` | Encoder-only social pool. The per-timestep pooling path (`pool_every_timestep=True`) is what triggered the CUDA race-condition in production — avoid. |
| `pool_every_timestep` | `False` | Confirms encoder-only path |
| `resist_loss_weight` | `1.0` | Repulsion was active during training but `resist_loss` code was **not in the source repo** — reconstructed here per the NaviGAN paper. |
| `intention_loss_weight` | `0.05` | Same: intention-branch supervision was active, source code missing — reconstructed. |
| `d_safe` | `0.5` m | Hinge threshold below which repulsion kicks in |
| Discriminator | Trained but `d_loss` flat ~1.37 | GAN signal contributed little vs L2 + repulsion. Don't expect adversarial term to dominate. |
| `g_waypointbest_state` | **missing** | Older "best-K" pathway not used — never reference it in loaders. |

**Reconstructed losses.** The source NaviGAN repo ships with `resist_loss_weight=1.0` and `intention_loss_weight=0.05` baked into the checkpoint args but neither loss is implemented in the codebase. We reconstruct them from the paper (Tsai & Oh, ICRA 2020):

- `resist_loss` — pairwise hinge: `max(0, d_safe - ‖x_i(t) − x_j(t)‖)` summed over `i<j` within each sequence, every timestep. Returns both the scalar loss and a *violator count* (matches the checkpoint's `resist_count` series).
- `intention_loss` — direct L2 on the intention-branch rollout, pre-attention-fusion. The model already exposes `intention_trajectories` as an output of `forward()`.

Verify the reconstruction by running a resume-eval against the checkpoint and checking that `val resist_loss ≈ 0.21` and `val resist_count ≈ 3970` (the final values recorded during the original training).

**CUDA fix carried forward.** The model copy in `navigan_training/models/late_attention.py` includes the `grid_pos.clamp(0, N-1)` defence in `SocialPooling` from `navigan/python/models.py:340` — required to keep `scatter_add` from asserting on out-of-bounds indices that can leak from decoder rollout. See `../navigan/CUDA_RACE_CONDITION_EXPLAINED.md` for background.

---

## Repo layout

```
navigan_training/
├── train.py                       # entry point (single-GPU OR torchrun-DDP)
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── zara1_scratch.yaml         # from-scratch, matches checkpoint architecture
│   ├── zara1_resume.yaml          # resume from benchmark_zara1_with_model.pt
│   └── zara1_safe.yaml            # same arch, d_safe=0.8, resist_weight=5.0
├── scripts/
│   ├── inspect_checkpoint.py      # dump args + state_dict inventory
│   ├── eval_checkpoint.py         # eval a .pt on any split, pick state variant
│   ├── plot_training.py           # render loss/metric curves from a .pt
│   ├── plot_predictions.py        # render obs/gt/pred trajectories + d_safe rings
│   ├── download_data.sh           # fetch SGAN ETH/UCY bundle
│   └── slurm_train.sbatch         # UCT HPC template
├── datasets/                      # populated by download_data.sh (gitignored)
│   └── {eth,hotel,univ,zara1,zara2}/{train,val,test}/*.txt
├── models/                        # drop trained .pt files here (gitignored)
└── navigan_training/
    ├── data/
    │   ├── trajectories.py        # TrajectoryDataset + seq_collate (9-tuple, with goals)
    │   └── loader.py              # DataLoader factory (DistributedSampler when DDP)
    ├── models/
    │   └── late_attention.py      # LateAttentionFullGenerator + TrajectoryDiscriminator + deps
    ├── losses.py                  # gan_{g,d}_loss, l2_loss, resist_loss, intention_loss
    ├── train_loop.py              # discriminator_step, generator_step, evaluate, run_training
    └── utils.py                   # AttrDict, ckpt save/load, ctor-kwargs helpers, seeding
```

---

## Quick start

```bash
# 1. Install (uses a venv or conda env of your choice)
pip install -e .

# 2. Fetch ETH/UCY trajectories (~25 MB, Dropbox)
bash scripts/download_data.sh

# 3. Train from scratch on zara1
python train.py --config configs/zara1_scratch.yaml

# Smoke test (100 iters, no save logic affected)
python train.py --config configs/zara1_scratch.yaml --max-iter 100
```

Outputs land in `runs/<config>/navigan_checkpoint.pt` with the same dict layout as the production checkpoint (`g_state`, `d_state`, `g_optim_state`, `d_optim_state`, `counters`, `args`, `metrics_train`, `metrics_val`), plus the multi-criterion best-state slots described below.

---

## Best-checkpoint tracking (GAN-aware)

GAN val metrics oscillate epoch-to-epoch and no single scalar captures both prediction quality **and** social safety, so the trainer keeps three independent best-state snapshots per run (in addition to the rolling `final` state). They are recomputed every time we run a val eval (every `checkpoint_every` iterations) and overwrite their slot when an improvement is seen:

| Slot | Criterion | What it optimises |
|---|---|---|
| `g_best_ade_state` / `d_best_ade_state` | min val ADE | Average displacement accuracy across the prediction horizon |
| `g_best_fde_state` / `d_best_fde_state` | min val FDE | Final-pose accuracy (endpoint error) |
| `g_best_safety_state` / `d_best_safety_state` | min val `resist_count` | Fewest pedestrian-proximity violations under the configured `d_safe` |

Per-slot metadata (the value, iter `t`, and the full metric dict at that moment) is stored under `checkpoint['bests'][slot]`. At deploy time you choose which trade-off to ship — typically `best_safety` for the Husky and `best_ade` for accuracy benchmarks. The `final` state is still saved every checkpoint interval so you can also resume training.

## Eval-only script

`scripts/eval_checkpoint.py` loads any `.pt`, rebuilds the model from `checkpoint['args']` (so architecture always matches), and runs the same `evaluate()` used during training. Pick the state variant via `--variant`:

```bash
# Verify the reconstructed resist_loss against the production checkpoint
python scripts/eval_checkpoint.py models/benchmark_zara1_with_model.pt \
    --dataset zara1 --split val
# Expect: resist_loss ~ 0.21, resist_count ~ 3970

# Compare deploy-time options after a safety-tuned run
python scripts/eval_checkpoint.py runs/zara1_safe/navigan_checkpoint.pt \
    --dataset zara1 --split val --variant best_safety
python scripts/eval_checkpoint.py runs/zara1_safe/navigan_checkpoint.pt \
    --dataset zara1 --split val --variant best_ade

# Final test-split numbers, machine-readable
python scripts/eval_checkpoint.py runs/zara1_scratch/navigan_checkpoint.pt \
    --dataset zara1 --split test --variant best_ade --json
```

Variants: `final` (default), `best_ade`, `best_fde`, `best_safety`. `--d_safe` overrides the threshold used to recompute the resist metrics (useful for sweep analysis without retraining). `--json` emits a single JSON line for piping into result tables.

---

## Plotting

The trainer records full loss + metric series inside the `.pt` (no external logger required). Two scripts render them. Install the `plot` extra first:

```bash
pip install -e '.[plot]'   # adds matplotlib
```

### Training curves (`scripts/plot_training.py`)

2×3 grid: generator loss components (l2 / adv / resist / intention / total), discriminator loss, and val+train curves for ADE, FDE, resist_loss, resist_count. Vertical dotted lines mark the iters where `best_ade`, `best_fde`, and `best_safety` snapshots were taken — useful for picking which variant to deploy.

```bash
# Saves <stem>_training.png next to the checkpoint
python scripts/plot_training.py runs/zara1_scratch/navigan_checkpoint.pt

# Custom output dir, skip the best-iter markers
python scripts/plot_training.py runs/zara1_safe/navigan_checkpoint.pt \
    -o plots/ --no-bests
```

Also prints a per-best summary table to stderr (slot, iter, metric value).

### Trajectory predictions (`scripts/plot_predictions.py`)

Samples scenes from the chosen split, runs the generator, and plots all agents' observed past (solid), ground-truth future (dotted), and predicted future (dashed). The robot is highlighted in red. A `d_safe` ring is drawn around each pedestrian at the first prediction step — any predicted robot trajectory entering a ring is a clearance violation, visible at a glance.

```bash
# Quick qualitative check on the production checkpoint
python scripts/plot_predictions.py models/benchmark_zara1_with_model.pt \
    --dataset zara1 --split val --num-scenes 6

# Compare safety variants on the same scenes
python scripts/plot_predictions.py runs/zara1_safe/navigan_checkpoint.pt \
    --variant best_ade    --num-scenes 9 -o plots/baseline/
python scripts/plot_predictions.py runs/zara1_safe/navigan_checkpoint.pt \
    --variant best_safety --num-scenes 9 -o plots/safe/

# Re-score the same predictions against a tighter clearance ring
python scripts/plot_predictions.py runs/zara1_scratch/navigan_checkpoint.pt \
    --d_safe 1.0 --num-scenes 6
```

Both scripts use the `Agg` matplotlib backend, so they run cleanly on HPC compute nodes / headless Jetsons.

---

## Configs explained

All three configs use the same architecture (so they are state-dict-compatible with `benchmark_zara1_with_model.pt`). They differ only in **training schedule** and **safety tuning**.

| Config | `resume_from` | `d_safe` | `resist_loss_weight` | Intended use |
|---|---|---|---|---|
| `zara1_scratch.yaml` | `null` | 0.5 | 1.0 | Reproduce the production checkpoint from scratch |
| `zara1_resume.yaml` | `models/benchmark_zara1_with_model.pt` | 0.5 | 1.0 | Continue training (extends to `num_epochs=1200`) |
| `zara1_safe.yaml` | `null` | 0.8 | 5.0 | Safer variant — wider clearance, fewer safety-zone violations (slight ADE/FDE cost) |

### Tuning safety

Two levers compose:

- `d_safe` — distance threshold below which the hinge activates. Higher = wider clearance.
- `resist_loss_weight` — multiplier on the hinge loss term in the generator total. Higher = stronger pressure to keep clear.

Sweep grid suggested in `zara1_safe.yaml`:

- `d_safe ∈ {0.6, 0.8, 1.0}`
- `resist_loss_weight ∈ {2.0, 5.0, 10.0}`

CLI overrides without editing the YAML:

```bash
python train.py --config configs/zara1_safe.yaml --d_safe 1.0 --resist-weight 10.0
```

Resume configs ignore the model-architecture fields in the YAML — those come from `checkpoint['args']` at load time so the saved state-dict shapes always match. The training-schedule fields (`num_epochs`, `g_learning_rate`, `d_safe`, `resist_loss_weight`, etc.) take effect.

---

## Multi-GPU on UCT HPC

Template in `scripts/slurm_train.sbatch`. Edit the placeholders before submitting:

```bash
#SBATCH --partition=<your-partition>     # TODO
#SBATCH --account=<your-account>         # TODO
# Plus: source your env (module load / conda activate / venv) below the SBATCH block
```

Submit:

```bash
sbatch scripts/slurm_train.sbatch                              # uses configs/zara1_scratch.yaml
sbatch scripts/slurm_train.sbatch configs/zara1_safe.yaml      # custom config
```

The template requests 1 node × 4 GPUs and runs `torchrun --standalone --nproc_per_node=$SLURM_GPUS_ON_NODE train.py --distributed --config <CONFIG>`. NCCL is the backend. A `DistributedSampler` is wired in automatically when `--distributed` is set.

The model is small (~101k params), so DDP scaling is mostly about pushing through more iterations in wall-clock — single-GPU is fine for dev work and for the Jetson side.

---

## Verification checklist

After install, run these to make sure everything works:

1. **State-dict load** — confirm the copied model loads the production checkpoint cleanly:
   ```bash
   python -c "
   import torch
   from navigan_training.models import LateAttentionFullGenerator
   from navigan_training.utils import model_ctor_kwargs, AttrDict
   ck = torch.load('models/benchmark_zara1_with_model.pt', weights_only=False, map_location='cpu')
   args = AttrDict(ck['args']) if isinstance(ck['args'], dict) else AttrDict(vars(ck['args']))
   g = LateAttentionFullGenerator(**model_ctor_kwargs(args))
   g.load_state_dict(ck['g_state'])
   print('OK:', sum(p.numel() for p in g.parameters()), 'params')
   "
   ```

2. **Loss reconstruction** — eval the checkpoint on `zara1/val` and confirm `resist_loss ≈ 0.21`, `resist_count ≈ 3970` (final training values). Significant deviation (> 2×) suggests the hinge formulation needs revisiting.

3. **From-scratch smoke** — `python train.py --config configs/zara1_scratch.yaml --max-iter 100` should run end-to-end with decreasing losses, no NaN, and a checkpoint write.

4. **Resume** — `python train.py --config configs/zara1_resume.yaml` should restore counters from `t≈29020` and pick up at the saved loss values.

5. **DDP smoke (2 GPUs)** — `torchrun --nproc_per_node=2 train.py --distributed --config configs/zara1_scratch.yaml --max-iter 10` should log from both ranks with no NCCL deadlock.

6. **Safer model** — train `zara1_safe.yaml` (`d_safe=0.8`, weight `5.0`) for a full run and confirm `resist_count` falls faster than the baseline run and stays lower at convergence.

---

## Architecture caveat — determinism

The production checkpoint uses `noise_dim=(0,)`, which makes the generator **deterministic**: same input → same output. Sample-and-pick-best ("variety loss", `best_k > 1`) is therefore a no-op here. To enable stochastic variety you must:

1. Change `noise_dim` to e.g. `(8,)` in the config.
2. Train **from scratch** — resume would fail because the noise-injection layer shapes change.
3. Adopt `best_k > 1` in the generator step (currently not implemented; would need to add the L2 best-of-K reduction back in).

---

## Out of scope

- **Per-scene configs for `eth/hotel/univ/zara2`** — only `zara1` shipped. Copy `zara1_scratch.yaml` and change `dataset_name` + `output_dir`.
- **No TensorBoard / Weights & Biases integration.** Loss curves and val metrics are still recorded — they live inside the `.pt` file under `checkpoint['G_losses']`, `checkpoint['D_losses']`, `checkpoint['metrics_train']`, `checkpoint['metrics_val']` (same nested-dict format as the production checkpoint). Use `scripts/plot_training.py` to render them, or load the `.pt` and route the dicts to your own logger if you want live dashboards.
- **ONNX export / TensorRT for Jetson deployment** — separate concern, handled outside this repo by the ROS2 inference stack.
