# Social Trajectory Prediction — Post-NaviGAN (2020-2025) Survey

**Audience**: NaviGAN training team.

**Scope**: Architecture options published after NaviGAN (Tsai & Oh, ICRA 2020).

**Status**: Research-only. No code changes yet.

> **Read this first if you are deciding what to build**: [`integration_into_navigan.md`](integration_into_navigan.md) reframes the question as "what can we graft INTO NaviGAN while preserving the intention+force fusion?" — which is the actual team decision. The recommendations below treat each candidate as a *replacement* for NaviGAN, which is useful as literature reference but not what the team is doing.

This document synthesises two more detailed surveys:
- [`part1_gan_and_transformer.md`](part1_gan_and_transformer.md) — GAN + Transformer families
- [`part2_diffusion_cvae_graph.md`](part2_diffusion_cvae_graph.md) — Diffusion + CVAE/NF + Graph + Goal-conditioned families

Read the parts for per-paper detail. This master doc focuses on **decision** + **integration sketch**.

---

## Executive summary

NaviGAN sits in a relatively quiet corner of the trajectory-prediction literature. The mainstream has moved on from GAN-based stochastic prediction toward (a) Transformers with explicit motion-mode priors, (b) Equivariant graph networks, and (c) Diffusion with leap-step acceleration. On the standard ETH/UCY benchmark, the strongest published numbers on **zara1 (8 obs / 12 pred, K=20)** are:

| Model | zara1 ADE/FDE | Family |
|-------|---------------|--------|
| NaviGAN (baseline, deterministic) | — (to be measured) | SGAN+attention |
| SGAN (anchor) | 0.34 / 0.69 | GAN |
| STAR | 0.26 / 0.55 | Transformer |
| TUTR | 0.18 / 0.34 | Transformer |
| EqMotion | **0.18 / 0.32** | Equivariant graph |
| AgentFormer | **0.15 / 0.23** | Transformer+CVAE |
| LED | ~0.18 / 0.27 (unverified) | Diffusion |
| Trajectron++ | ~0.15 / 0.33 (unverified) | CVAE |

Three architectures are recommended for first-round benchmarking based on the combination of reported accuracy gain, integration cost against the existing `navigan_training/` repo, and Jetson Orin AGX inference budget (~100 ms target):

1. **EqMotion** (Equivariant graph, CVPR 2023) — strongest verified zara1 numbers, single forward pass, equivariance prior is a principled upgrade over PoolHiddenNet's coordinate-frame-dependent encoding. **Top pick.**
2. **TUTR** (Trajectory Unified Transformer, ICCV 2023) — best latency story for Orin (~19 FPS dense scene on 3090), explicit motion-mode prior restores the multi-modality NaviGAN lost when `noise_dim` was set to `(0,)`. **Strong transformer pick.**
3. **SocialVAE** (ECCV 2022) — closest structural analog to NaviGAN; timewise latents map cleanly onto our per-timestep attention fusion; lowest integration lift inside the existing GAN training loop. **Lowest-risk pick.**

Honourable mention: **Social-Ways** (CVPRW 2019) is the closest published match to the user's "InfoGAN for trajectories" question — it adds an InfoGAN-style mutual-information loss with a Q-network on top of an SGAN backbone. It is the only viable path to "extend NaviGAN with InfoGAN-style disentanglement" without throwing away the existing GAN stack. Note: NaviGAN currently has `noise_dim=(0,)` so the latent codes need re-enabling for the MI loss to mean anything.

Caveats: several ETH/UCY split-level numbers from the surveyed papers could not be re-verified from the PDFs in the research budget — they are flagged `(unverified)` in the part docs and should be confirmed against the original papers before any final report.

---

## Decision criteria (used to rank)

| Criterion | Weight | Rationale |
|-----------|--------|-----------|
| Verified zara1 ADE/FDE delta over SGAN | High | Direct measure of the gain we'd get |
| Repo integration cost | High | We have ~1 person doing the work; new model file + loss is fine, full pipeline rewrite is not |
| Jetson Orin AGX latency (~100 ms target) | High | Must hold the ROS2 10 Hz run loop without starving ZED |
| Hinge social-repulsion (`resist_loss`) compatibility | High | The team explicitly reconstructed this loss; new model must accept it |
| Goal/intention input compatibility | Medium | NaviGAN consumes `goal_state` from `BasicNavigator`; ripping this out costs Nav2 integration |
| Determinism vs K-sample stochasticity | Medium | Production currently picks the single greedy path; K-sample candidates need a deterministic mode-selector |
| Coords-only input (no map, no pose) | Hard constraint | We have only ZED `ObjectsStamped` + AMCL pose. No HD map, no body pose. |
| Open-source code with maintained PyTorch repo | Medium | Cuts re-implementation risk |

---

## Top 3 recommended candidates — with integration sketches

### 1. EqMotion (Xu et al., CVPR 2023)
- **Paper**: arXiv:2303.10876. Code: https://github.com/MediaBrain-SJTU/EqMotion
- **Why**: Verified **0.18/0.32 on zara1** (paper Table), single forward pass (no diffusion sampling), equivariance prior is a meaningful upgrade over PoolHiddenNet's coordinate-frame sensitivity, modular PyTorch repo.
- **Caveat**: K=20 sample output — need a deterministic mode-selector at deploy (pick mode closest to `goal_state`, or argmax probability head).

**What to build inside `navigan_training/`:**
```
navigan_training/
└── navigan_training/
    └── models/
        ├── late_attention.py      # existing — unchanged
        └── eqmotion.py            # NEW — port EqMotion backbone (geometric + interaction + pattern modules)
    └── train_loop.py              # MODIFY — add non-GAN path (no discriminator step) gated by config flag
    └── losses.py                  # MODIFY — adapt resist_loss to operate on K-mode output (apply per-mode, mean over modes)
configs/
    └── zara1_eqmotion.yaml        # NEW — model=eqmotion, K=20, resist_loss_weight tuned for K-mode aggregation
scripts/
    └── eval_checkpoint.py         # MODIFY — accept model_type in checkpoint args, dispatch to EqMotion when set
```

Estimated effort: **3-5 days** (1-2 days port + tests, 1 day train-loop adaptation, 1-2 days config/eval/benchmark).

---

### 2. TUTR (Shi et al., ICCV 2023)
- **Paper**: openaccess.thecvf.com (ICCV 2023). Code: https://github.com/lssiair/TUTR
- **Why**: 0.18/0.34 zara1, **fastest model in the benchmark** (0.0533 s for N=80 on 3090 — likely fits Orin 100 ms easily for N=20), explicit motion-mode prior + classification head means we get multi-modality without CVAE/diffusion overhead. No post-processing tail.
- **Caveat**: Requires an offline build step — k-means over training endpoints to produce L motion-mode anchors per dataset. Easy but new.

**What to build inside `navigan_training/`:**
```
navigan_training/
└── navigan_training/
    └── models/
        └── tutr.py                # NEW — mode-level encoder + social-level decoder
    └── data/
        └── motion_modes.py        # NEW — offline k-means over training endpoint deltas, dumps L anchors to .pt
    └── train_loop.py              # MODIFY — non-GAN path + dual-head loss (regression + classification)
    └── losses.py                  # MODIFY — add mode classification CE; resist_loss applied to argmax mode
configs/
    └── zara1_tutr.yaml            # NEW — model=tutr, L=20 modes, motion_modes_path=...
scripts/
    └── build_motion_modes.py      # NEW — runs the k-means offline, called once per dataset
    └── eval_checkpoint.py         # MODIFY — same dispatch as EqMotion
```

Estimated effort: **4-6 days** (1-2 days port, 1 day motion-modes pipeline, 1 day train-loop, 1-2 days eval/benchmark).

---

### 3. SocialVAE (Xu et al., ECCV 2022)
- **Paper**: arXiv:2203.08207. Code: https://github.com/xupei0610/SocialVAE
- **Why**: Closest structural analog to NaviGAN — timewise latent CVAE with social attention. Lowest risk because the existing per-timestep loss stack (L2 + hinge + intention) maps onto the timewise-latent decoder without redesign. RNN-scale model, comparable latency to current NaviGAN.
- **Caveat**: Adds KL divergence to the loss stack — need to re-balance `resist_loss_weight` against KL weight. Verified zara1 numbers `(unverified)` — confirm from paper Table 1 first.

**What to build inside `navigan_training/`:**
```
navigan_training/
└── navigan_training/
    └── models/
        └── socialvae.py           # NEW — encoder + timewise-CVAE decoder + social attention
    └── train_loop.py              # MODIFY — replace GAN step with CVAE step (reconstruction + KL), gated by config
    └── losses.py                  # MODIFY — add KL term + balance against existing l2 / hinge / intention
configs/
    └── zara1_socialvae.yaml       # NEW — model=socialvae, latent_dim, kl_weight, plus existing resist_loss_weight
```

Estimated effort: **3-4 days** (1 day port, 1 day CVAE train-loop variant, 1-2 days hyperparameter tuning for KL/hinge balance).

---

## Honourable mention: InfoGAN-style extension (Social-Ways)

The user explicitly asked about InfoGAN. The closest published "InfoGAN for pedestrian trajectories" is **Social-Ways** (Amirian et al., CVPRW 2019, arXiv:1904.09507, code at github.com/amiryanj/socialways). It replaces SGAN's L2 variety loss with an InfoGAN-style mutual-information loss between latent codes `c` and generated trajectories, learned via a Q-network. Post-2020 surveys found no clean InfoGAN-trajectory follow-up beyond controllable-trajectory GAN variants (Springer 2024) without reproducible code or strong ETH/UCY numbers.

**This is the only option that preserves NaviGAN's existing GAN + hinge stack.** If the team wants an incremental GAN upgrade rather than a family switch, this is the natural path:

```
navigan_training/
└── navigan_training/
    └── models/
        └── late_attention.py      # MODIFY — add Q-network head for latent-code prediction
    └── losses.py                  # MODIFY — add mutual_info_loss(predicted_c, sampled_c)
    └── train_loop.py              # MODIFY — sample noise z + latent code c during generator step; add MI term
configs/
    └── zara1_infogan.yaml         # NEW — noise_dim=(8,), latent_code_dim=4, mi_loss_weight=1.0
```

**Critical caveat**: NaviGAN's current production checkpoint has `noise_dim=(0,)` so the generator is deterministic. An InfoGAN extension only makes sense if you **train from scratch** (NOT resume) with `noise_dim > 0`. Resume-from-ckpt would not work because the new noise injection changes weight shapes.

Estimated effort: **2-3 days** (smallest change of all options because everything else stays the same).

**Trade-off**: this is the least ambitious option. Expected gain is modest (Social-Ways's main contribution was diversity, not absolute ADE/FDE). Recommended only if the team prioritises minimum implementation risk over benchmark headline numbers.

---

## Do NOT recommend (with reasons)

| Model | Reason |
|-------|--------|
| **HiVT** | AV/HD-map vectorised input only. No ETH/UCY benchmark. |
| **MUSE-VAE** | Requires scene/environment image (pixel-space CVAE cascade). Robot has no map pipeline. |
| **Y-Net** | Requires top-down scene image. Same map-pipeline blocker. |
| **SoPhie** (full version) | Physical attention requires scene image. Stripping it reduces to SGAN+social-attention — marginal gain. |
| **Social-Transmotion** (full multimodal) | Best when 2D/3D body pose is available; we have only object bounding boxes from ZED `ObjectsStamped`. Reduces to a plain xy Transformer in our setup, then there are better alternatives (TUTR, STAR). |
| **MID** | T=100 denoising steps — far beyond 100 ms Jetson budget. LED supersedes it. |
| **BCDiff** | 2x diffusion cost (bidirectional). Niche: optimised for "instantaneous" prediction with very short obs windows; we have obs=8 already. |
| **AgentFormer** | Best absolute numbers (0.15/0.23 zara1) but CVAE + DLow trajectory sampler is a heavy structural shift; awkward hinge-loss fit. Defer unless TUTR underdelivers. |
| **MemoNet** | Memory bank must ship with the model (~11 MB on SDD, smaller on ETH/UCY); 3-stage training pipeline; latency unverified on Orin. |
| **Generic post-2020 WGAN-GP / controllable-GAN variants** | No reproducible SOTA on ETH/UCY; consolidation work, not a meaningful upgrade. |

---

## Latency budgets — Orin AGX target

The ROS2 run loop in `navigan_node.py` ticks at 10 Hz, so the **end-to-end NaviGAN forward pass + post-processing must complete in <100 ms** (with margin for ZED + AMCL + Path publish in the same window). The 20% GPU reservation for ZED means realistically the model gets a ~60-80 ms window.

| Candidate | Reported desktop GPU latency | Orin estimate | Risk |
|-----------|------------------------------|---------------|------|
| NaviGAN baseline | (current) | ~10-30 ms (measured in `latency_results/`) | — |
| EqMotion | (unreported by authors) | ~30-60 ms (single fwd, comparable scale) | Low |
| TUTR | 53 ms / N=80 on RTX 3090 | ~80-150 ms / N=80 on Orin; ~20-40 ms / N=20 | Low (small scenes) |
| SocialVAE | (unreported) | ~20-40 ms (RNN-scale) | Low |
| LED | 46 ms / NBA on RTX 3090 | ~100-250 ms on Orin | Borderline |
| AgentFormer | (unreported) | quadratic in (1+N)*T tokens | Borderline |
| MemoNet | 18 ms / sample on 3090 | ~50-150 ms on Orin + bank queries | Borderline |

Recommendation: instrument all candidates with the existing `utils/instrumentation.py` `TimingCSVWriter` on Orin **before** finalising any benchmark report. Desktop GPU numbers are not a reliable Orin proxy.

---

## Open questions for the team (please answer before implementation starts)

1. **Determinism vs K-sample inference.** NaviGAN currently outputs a single greedy trajectory. EqMotion, TUTR, SocialVAE, LED all output K modes. Are you willing to add a deterministic mode-selector at deploy (e.g. pick the mode closest to `goal_state`, or argmax over the classification head)? If not, the GAN-family options (Social-Ways InfoGAN extension) are the only ones that preserve single-mode output.

2. **Inference budget — strict 100 ms or willing to drop to 5 Hz?** A relaxed budget unlocks LED (diffusion) and AgentFormer. A strict 100 ms budget restricts us to EqMotion / TUTR / SocialVAE / Social-Ways. Affects the candidate slate.

3. **Training data scope.** Train candidates on zara1 only (apples-to-apples vs current checkpoint), or train on full ETH/UCY (eth/hotel/univ/zara1/zara2) for a more general benchmark? Affects compute time and number of YAML configs to write.

4. **Discriminator survival.** For non-GAN candidates (EqMotion / TUTR / SocialVAE), should we keep the discriminator as a separate auxiliary critic, or drop entirely? Dropping is simpler — `train_loop.py` needs a config-gated branch. Keeping is more conservative but doubles the loss-balance tuning surface.

5. **InfoGAN-style extension priority.** Is "extend NaviGAN with InfoGAN-style disentanglement" (Social-Ways path) a goal in its own right, or only of interest if the family-switch options underdeliver? Answer drives whether we run Social-Ways in parallel with the top-3 or hold it as fallback.

---

## Benchmark protocol proposal

If the team approves a candidate, suggest the following benchmark protocol (using the existing `scripts/eval_checkpoint.py` and `scripts/plot_predictions.py`):

1. **Train candidate on zara1** with same obs/pred horizons (8/12), same `d_safe`, same `resist_loss_weight` as the production NaviGAN config (`zara1_scratch.yaml`).
2. **Track all three best-checkpoint variants** (best_ade / best_fde / best_safety) using the existing `_update_bests` machinery in `train_loop.py`.
3. **Evaluate against `benchmark_zara1_with_model.pt`** using `eval_checkpoint.py` on the val split. Report ADE, FDE, resist_loss, resist_count for `final` + all three best variants.
4. **Latency benchmark on Orin** — copy candidate checkpoint to Jetson, run inside `navigan_node.py` with `enable_profiling:=true`, compare timing CSV against current NaviGAN baseline.
5. **Visualise predictions** — `plot_predictions.py` for 6-12 scenes side-by-side with baseline NaviGAN predictions, highlight d_safe ring violations.
6. **Failure-mode comparison** — replay rosbags through both models, compare freeze/oscillation counts from `FoVCSVWriter`.

Pass criteria for replacing the production checkpoint: (a) ADE/FDE within 10% of NaviGAN baseline OR better, (b) resist_count strictly lower at same `d_safe`, (c) p95 inference latency on Orin under 80 ms.

---

## File outputs

- This master synthesis: `navigan_training/docs/social_gan_advances_survey.md`
- GAN+Transformer detail: `navigan_training/docs/part1_gan_and_transformer.md`
- Diffusion+CVAE+Graph detail: `navigan_training/docs/part2_diffusion_cvae_graph.md`

No code changes were made. Next step is a team decision on the open questions above and selection of one (or two parallel) candidate(s) to implement.
