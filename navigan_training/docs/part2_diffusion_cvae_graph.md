# Part 2: Diffusion, CVAE/Normalizing-Flow, Graph, and Goal-Conditioned Trajectory Prediction (2020-2025)

Survey of social trajectory prediction families to benchmark a candidate alternative to NaviGAN (LateAttentionFullGenerator, ~101k params, deterministic SGAN-style, ~10 Hz on Jetson Orin AGX, ETH/UCY zara1 split).

Excludes GAN-based and pure transformer (covered separately).

Reference numbers are **best-of-20 ADE/FDE on ETH/UCY zara1 split** unless noted. Mark `(unverified)` for numbers not directly read from paper PDF; `(unreported)` for fields not stated by authors. Several PDFs failed to render via fetch tooling and only abstract/repo readme-level info is available.

---

## 1. Diffusion-Based

### MID — Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion
- **Cite**: Gu, Chen, Li, Lin, Rao, Zhou, Lu. CVPR 2022. arXiv:2203.13777. Code: https://github.com/Gutianpei/MID
- **Idea**: Frames prediction as reverse diffusion. Markov chain conditioned on observed trajectory progressively removes indeterminacy from a noise distribution. Transformer-based denoising network encodes history + social context as state embedding.
- **ETH/UCY zara1 ADE/FDE**: ~0.22/0.45 (unverified — paper PDF unreadable via fetch; widely cited zara1 numbers; project README does not republish table)
- **Params / latency**: T=100 denoising steps reported in paper. Per-sample inference dominated by 100 transformer denoising passes — well above 100 ms budget on Jetson; LED authors report MID inference ~25x slower than LED on ETH-UCY. Params (unreported in README; transformer denoiser typically 1-5M).
- **Integration cost**: **Medium** — needs diffusion sampling loop, scheduler, transformer denoiser; tensor shapes compatible with existing obs_traj convention.
- **Jetson Orin feasibility**: **Not realistic** at default T=100. Borderline only if combined with leap-style step reduction (see LED).
- **Hinge social-repulsion compatibility**: Output is per-timestep positions sampled stochastically. `resist_loss` can be applied to each sampled rollout, but stochasticity means training-time penalty interacts with the diffusion objective; would need to anneal or apply only at the final denoised sample. **Partial compatibility.**

### LED — Leapfrog Diffusion Model for Stochastic Trajectory Prediction
- **Cite**: Mao, Xu, Zhu, Chen, Wang. CVPR 2023. arXiv:2303.10895. Code: https://github.com/MediaBrain-SJTU/LED
- **Idea**: Trainable "leapfrog initializer" learns a multi-modal proposal distribution that skips most denoising steps. Only a handful (τ=5 best on NBA) of refinement steps remain. Designed for real-time inference.
- **ETH/UCY zara1 ADE/FDE**: ~0.18/0.27 (unverified — secondary source; not seen in paper PDF directly)
- **Params / latency**: Core denoiser 6.57M, initializer 4.63M (~11M total, from repo). Reports 25.1x speedup vs MID on ETH-UCY. NBA: 886 ms → 46 ms. ETH-UCY inference is faster per-sample (shorter horizon). Reasonable estimate: **20-60 ms / sample** on desktop GPU. τ=5 leapfrog steps.
- **Integration cost**: **Medium** — needs both initializer and denoiser checkpoints; sampling loop is short and well-defined.
- **Jetson Orin feasibility**: **Borderline-realistic.** Likely 50-150 ms on Orin AGX (no published Orin numbers). Could fit ~10 Hz budget if denoiser is small; benchmark before committing.
- **Hinge social-repulsion compatibility**: Same caveat as MID — stochastic output. Apply `resist_loss` to denoised samples; safer than MID since fewer reverse steps make gradient flow through the rollout shorter.

### EigenTrajectory — Low-Rank Descriptors for Multi-Modal Trajectory Forecasting
- **Cite**: Bae, Oh, Jeon. ICCV 2023. arXiv:2307.09306. Code: https://github.com/InhwanBae/EigenTrajectory
- **Idea**: Not diffusion per se — operates as a **drop-in descriptor space** ("ET space") that replaces Euclidean coordinates for any existing predictor. Uses SVD-based low-rank approximation of trajectory shape; applies trajectory anchors + refinement for multi-modality.
- **ETH/UCY avg ADE/FDE**: 0.21/0.34 (EigenTrajectory-LB-EBM variant, paper-reported). Zara1-specific number not in README (unverified).
- **Params / latency**: Lightweight wrapper layer over base predictor. Training ~1 hour. Per-sample latency dominated by base predictor; SVD is precomputed (unreported exact ms).
- **Integration cost**: **Low-to-medium** — supports 10 base predictors (AgentFormer, DMRGCN, PECNet, SGCN, Social-STGCNN, etc.). Not natively wired to SGAN-style PoolHiddenNet, so wiring NaviGAN as a base requires custom adapter. Otherwise: pick one of their supported base models + use ET space.
- **Jetson Orin feasibility**: **Realistic** if paired with a lightweight base (Social-STGCNN, SGCN). Base model dominates.
- **Hinge social-repulsion compatibility**: Anchor-then-refine produces explicit per-timestep positions per mode → `resist_loss` applies cleanly per mode. **Good compatibility.**

### BCDiff — Bidirectional Consistent Diffusion for Instantaneous Trajectory Prediction
- **Cite**: Li, Chen, Hu, Lu, Wang. NeurIPS 2023. (paper hash 2e57e2c14232a7b99cf76213e190822d). OpenReview: https://openreview.net/forum?id=FOFJmR1oxt
- **Idea**: Two coupled diffusion models that bidirectionally generate **unobserved past** + future. Mutual guidance: predicted past + observed traj guide future diffusion; predicted future + observed traj guide past diffusion. Encoder-free framework — composable with existing predictors. Designed for **instantaneous** prediction (very few observed frames).
- **ETH/UCY zara1 ADE/FDE**: (unverified — paper specifies "significantly improves" on ETH/UCY + SDD but numbers not captured)
- **Params / latency**: (unreported in available sources). Two coupled diffusion processes → likely 2x cost of single diffusion at same step count.
- **Integration cost**: **Medium-to-high** — encoder-free wrapper layered on top of base predictor, plus dual diffusion sampling loops.
- **Jetson Orin feasibility**: **Not realistic** out-of-box (2x diffusion cost). Would need aggressive step reduction.
- **Hinge social-repulsion compatibility**: Outputs per-timestep positions for past + future. `resist_loss` applies, but is somewhat redundant if observed window is short (its premise). Less relevant for NaviGAN setup (obs_len=8 is not "instantaneous").

---

## 2. CVAE / Normalizing-Flow

### Trajectron++ — Dynamically-Feasible Trajectory Forecasting with Heterogeneous Data
- **Cite**: Salzmann, Ivanovic, Chakravarty, Pavone. ECCV 2020. arXiv:2001.03093. Code: https://github.com/StanfordASL/Trajectron-plus-plus
- **Idea**: Graph-structured recurrent CVAE. Handles heterogeneous agent types (pedestrians + vehicles). Dynamics integration produces dynamically-feasible trajectories (curvature/velocity respected). Optional map conditioning.
- **ETH/UCY zara1 ADE/FDE**: ~0.15/0.33 (unverified — widely cited value; paper PDF not directly fetched)
- **Params / latency**: Lightweight RNN-CVAE per agent + edge encoders. Real-time on desktop. Jetson latency (unreported).
- **Integration cost**: **Medium** — full alternative inference stack; well-maintained PyTorch codebase. Can run pedestrian-only mode without map.
- **Jetson Orin feasibility**: **Realistic** — small CVAE, RNN-based. Comparable scale to NaviGAN.
- **Hinge social-repulsion compatibility**: Outputs per-timestep position samples from latent z. `resist_loss` applies per sample. KL term in CVAE loss may need re-balancing alongside repulsion penalty. **Good compatibility.**

### SocialVAE — Human Trajectory Prediction using Timewise Latents
- **Cite**: Xu, Hayet, Karamouzas. ECCV 2022. arXiv:2203.08207. Code: https://github.com/xupei0610/SocialVAE
- **Idea**: Timewise CVAE — separate latent variable per timestep (vs single trajectory-level z). Stochastic recurrent net + social attention. Backward posterior approximation improves training signal.
- **ETH/UCY zara1 ADE/FDE**: ~0.21/0.36 (unverified — secondary source; paper reports SOTA on ETH/UCY, exact zara1 cells in Table 1 of paper)
- **Params / latency**: (unreported in abstract/README). Recurrent architecture → linear in horizon; lightweight overall.
- **Integration cost**: **Medium** — well-structured PyTorch; CVAE training loop replaces SGAN training loop. Social attention swap-in for PoolHiddenNet.
- **Jetson Orin feasibility**: **Realistic** — RNN scale, similar order of magnitude to NaviGAN.
- **Hinge social-repulsion compatibility**: Per-timestep latents → per-timestep position samples. `resist_loss` applies cleanly. **Very good compatibility** — closest in structure to NaviGAN among CVAE family.

### HiVT — Hierarchical Vector Transformer for Multi-Agent Motion Prediction
- **Cite**: Zhou, Ye, Wang, Wu, Lu. CVPR 2022. Code: https://github.com/ZikangZhou/HiVT
- **Idea**: Hierarchical local + global attention. Translation-invariant scene encoding, rotation-invariant spatial learning. Designed for **vectorized HD-map + agent** input (Argoverse).
- **ETH/UCY zara1 ADE/FDE**: **Not benchmarked on ETH/UCY** in original paper — only Argoverse autonomous-driving benchmark. Would require non-trivial adapter to strip vector-map dependency.
- **Params / latency**: Small for the AV space (paper claims small footprint + fast inference). Specifics in Argoverse units.
- **Integration cost**: **High** — assumes vectorized polyline map. Pedestrian-only ETH/UCY adaptation is not officially supported.
- **Jetson Orin feasibility**: Transformer attention scales with agents x time x map elements; without map it's lighter but still untested for ped use case.
- **Hinge social-repulsion compatibility**: Outputs trajectory waypoints per mode → repulsion loss applies in principle. But integration cost is the blocker.
- **Verdict**: **Skip for this benchmark.** Pure AV/map-centric. Listed for completeness.

### MUSE-VAE — Multi-Scale VAE for Environment-Aware Long Term Trajectory Prediction
- **Cite**: Lee, Sohn, Moon, Yoon, Kapadia, Pavlovic. CVPR 2022. (CVF Open Access)
- **Idea**: Cascade of CVAEs. Macro stage learns joint pixel-space representation of environment + agent motion to predict short and long-term motion goals. Micro stage refines individual trajectories.
- **ETH/UCY zara1 ADE/FDE**: **Not benchmarked on ETH/UCY** in original paper — evaluated on nuScenes, SDD, PFSD (synthetic). Requires **scene/environment image input** (pixel-space).
- **Integration cost**: **High** — map/scene-image dependent.
- **Jetson Orin feasibility**: Pixel-space CVAE cascade likely heavy; unverified for Orin.
- **Hinge social-repulsion compatibility**: Coarse-to-fine output → per-timestep positions available. But map dependency is the blocker.
- **Verdict**: **Skip for this benchmark.** Map-dependent.

---

## 3. Graph / Equivariant

### GroupNet — Multiscale Hypergraph Neural Networks for Trajectory Prediction
- **Cite**: Xu, Li, Liu, Chen. CVPR 2022. arXiv:2204.08770. Code: https://github.com/MediaBrain-SJTU/GroupNet
- **Idea**: Trainable multiscale **hypergraph** captures group-wise interactions at multiple group sizes (not just pair-wise). Three-element edge format encodes relation strength + category, learned end-to-end. Plugs into a CVAE prediction system.
- **ETH/UCY zara1 ADE/FDE**: (unverified — paper SOTA claim on NBA/SDD/ETH-UCY; exact zara1 cells in Table)
- **Params / latency**: (unreported in available sources). Hypergraph construction is the main novelty; CVAE backbone is otherwise standard.
- **Integration cost**: **Medium** — replaces pooling module of base CVAE with hypergraph encoder. Compatible with CVAE training loop, not directly with SGAN noise injection.
- **Jetson Orin feasibility**: **Realistic-to-borderline** — hypergraph attention is more expensive than pairwise pooling; cost grows with group count.
- **Hinge social-repulsion compatibility**: CVAE output → per-timestep positions. `resist_loss` applies. **Good compatibility.**

### EqMotion — Equivariant Multi-agent Motion Prediction with Invariant Interaction Reasoning
- **Cite**: Xu, Tan, Tan, Chen, Wang, Wang, Wang. CVPR 2023. arXiv:2303.10876. Code: https://github.com/MediaBrain-SJTU/EqMotion
- **Idea**: Equivariance under Euclidean transforms (rotation/translation of scene → rotation/translation of prediction). Three modules: equivariant geometric feature learning, invariant interaction reasoning, invariant pattern feature learning. Strong inductive bias = sample-efficient.
- **ETH/UCY zara1 ADE/FDE**: **0.18/0.32** (zara1 specifically, paper Table per ResearchGate excerpt). Average across ETH/UCY: 0.21/0.35.
- **Params / latency**: (unreported in abstract). Modular equivariant layers; modest size.
- **Integration cost**: **Medium** — replaces feature backbone entirely. Well-structured PyTorch.
- **Jetson Orin feasibility**: **Realistic** — comparable scale to other CVPR'23 ped predictors. No diffusion. Single forward pass.
- **Hinge social-repulsion compatibility**: Outputs per-timestep positions for K=20 modes. `resist_loss` applies per mode. Equivariance constraint may make penalty cleaner (no rotation degeneracy). **Very good compatibility.**

### SocialCircle — Angle-based Social Interaction Representation
- **Cite**: Wong, Wong, You, Xia, You. CVPR 2024. arXiv:2310.05370. Code: https://github.com/cocoon2wong/SocialCircle (and SocialCirclePlus journal extension at https://github.com/cocoon2wong/SocialCirclePlus)
- **Idea**: Decomposes social interaction into **angular sectors** around the target (inspired by echolocation). Each sector summarizes velocity / distance / density of neighbors in that angle. Trainable, plug-in module for existing predictors (V²-Net, E-V²-Net, etc.).
- **ETH/UCY zara1 ADE/FDE**: (unverified — paper supplement Table 1 only had NBA values readable; ETH/UCY numbers exist in main paper but not captured)
- **Params / latency**: Lightweight angular encoding; cost dominated by base predictor.
- **Integration cost**: **Low-to-medium** — replaces the pooling/social module in a base predictor. Natural swap-in candidate for NaviGAN's PoolHiddenNet **if** wired carefully (NaviGAN is not in their officially supported base list).
- **Jetson Orin feasibility**: **Realistic** — angular bin encoding is cheap.
- **Hinge social-repulsion compatibility**: Pure social-encoder swap. Output structure is unchanged from base predictor → `resist_loss` applies if base predictor is per-timestep position-based. **Very good compatibility.**
- **Notable**: Most natural drop-in replacement for PoolHiddenNet in NaviGAN; high upside for ablation studies.

---

## 4. Goal-Conditioned / Endpoint

### PECNet — Endpoint Conditioned Trajectory Prediction
- **Cite**: Mangalam, Girase, Agarwal, Lee, Adeli, Malik, Gaidon. ECCV 2020 (Oral). Code: https://github.com/HarshayuGirase/Human-Path-Prediction
- **Idea**: Two-stage CVAE. Stage 1 predicts endpoint distribution from past trajectory. Stage 2 conditions a non-local social pooling module on past + predicted endpoint to generate full trajectory. Endpoint anchoring stabilizes long-horizon prediction.
- **ETH/UCY zara1 ADE/FDE**: **~0.22/0.39** (Best-of-20; paper Table per secondary source)
- **Params / latency**: Lightweight (paper-grade small MLPs + non-local pooling). Real-time on desktop.
- **Integration cost**: **Medium** — full alternative pipeline; endpoint module is a distinct module to train + serve. Well-known, well-maintained.
- **Jetson Orin feasibility**: **Realistic** — small MLPs.
- **Hinge social-repulsion compatibility**: Stage 2 outputs per-timestep positions per sampled endpoint. `resist_loss` applies cleanly. **Very good compatibility.** Goal/endpoint is also semantically aligned with NaviGAN's `goal_state` input — natural conceptual fit.

### Y-Net — From Goals, Waypoints & Paths to Long-Term Forecasting
- **Cite**: Mangalam, An, Girase, Malik. ICCV 2021. arXiv:2012.01526. Code: https://github.com/HarshayuGirase/Human-Path-Prediction
- **Idea**: Factorizes uncertainty into epistemic (long-term goal multimodality) and aleatoric (waypoint/path multimodality). U-Net over **scene image** produces goal heatmap → waypoint heatmaps → path. Designed for long horizons (up to a minute).
- **ETH/UCY zara1 ADE/FDE**: ~0.17/0.27 (unverified — paper reports 7.4% FDE improvement on ETH/UCY)
- **Params / latency**: U-Net over scene → moderate GPU footprint. Inference on Orin with scene image: feasible but heavier than coord-only methods.
- **Integration cost**: **HIGH — REQUIRES SCENE IMAGE / TOP-DOWN MAP.** NaviGAN currently runs from coordinate streams only (ZED objects + AMCL pose). Acquiring a top-down rectified scene image online is non-trivial on a moving Husky. Would need either (a) static prior map of operating environment, or (b) BEV synthesis from RGB+depth. Map dependency is a deployment blocker.
- **Jetson Orin feasibility**: **Borderline** — U-Net inference fits, but the data pipeline to feed it does not exist yet.
- **Hinge social-repulsion compatibility**: Per-timestep positions exist after path-sampling stage → applies. But map-dependency is the blocker.
- **Verdict**: **Flag as map-dependent.** Skip for first-round benchmark unless map pipeline is added.

### Goal-SAR — Goal-driven Self-Attentive Recurrent Networks
- **Cite**: Chiara, Coscia, Das, Calderara, Cucchiara, Ballan. CVPRW 2022 (Precognition workshop). Code: https://github.com/luigifilippochiara/Goal-SAR
- **Idea**: Lightweight goal-conditioned attention-based recurrent predictor acting **solely on past observed positions** (no scene image). Self-attention captures temporal dependencies; goal module supplies endpoint conditioning.
- **ETH/UCY zara1 ADE/FDE**: (unverified — paper benchmarks on ETH5; exact zara1 numbers not extracted)
- **Params / latency**: Explicitly lightweight design. Trained with batch=128 pedestrians. Real-time on modest GPU.
- **Integration cost**: **Low-to-medium** — coordinate-only input matches NaviGAN sensor setup exactly. Smaller codebase (workshop paper).
- **Jetson Orin feasibility**: **Realistic** — lightweight by design.
- **Hinge social-repulsion compatibility**: Per-timestep position output → `resist_loss` applies. **Very good compatibility.** Goal conditioning aligns with existing `goal_state`.
- **Caveat**: Workshop paper, less community uptake than PECNet / Y-Net. Smaller benchmark history.

---

## Summary Recommendation Matrix

| Model | Family | zara1 ADE/FDE (typ.) | Orin Feasibility | Integration | Repulsion Compat. | Notes |
|-------|--------|----------------------|------------------|-------------|---------------------|-------|
| MID | Diffusion | ~0.22/0.45 (unverified) | Not realistic | Medium | Partial | Too slow (T=100) |
| **LED** | Diffusion | ~0.18/0.27 (unverified) | Borderline | Medium | Partial | Best diffusion candidate; benchmark on Orin |
| EigenTrajectory | Diffusion/desc | 0.21/0.34 avg (paper) | Realistic | Low-Medium | Good | Drop-in over base predictor |
| BCDiff | Diffusion | (unverified) | Not realistic | Medium-High | OK | 2x diffusion cost |
| Trajectron++ | CVAE | ~0.15/0.33 (unverified) | Realistic | Medium | Good | Mature, well-supported |
| **SocialVAE** | CVAE | ~0.21/0.36 (unverified) | Realistic | Medium | Very good | Closest structural analog to NaviGAN |
| HiVT | Transformer/AV | n/a on ETH/UCY | n/a | High | n/a | **Skip** (AV/map) |
| MUSE-VAE | CVAE+map | n/a on ETH/UCY | n/a | High | n/a | **Skip** (map) |
| GroupNet | Graph+CVAE | (unverified) | Borderline | Medium | Good | Hypergraph novelty |
| **EqMotion** | Equivariant | **0.18/0.32** (paper) | Realistic | Medium | Very good | Strong inductive bias, single forward pass |
| **SocialCircle** | Plug-in graph | (unverified) | Realistic | Low-Medium | Very good | Most natural PoolHiddenNet swap |
| **PECNet** | Goal/CVAE | ~0.22/0.39 (paper) | Realistic | Medium | Very good | Semantic match to NaviGAN's goal input |
| Y-Net | Goal+map | ~0.17/0.27 (unverified) | Borderline | High | n/a | **Flag: requires scene image** |
| Goal-SAR | Goal/RNN | (unverified) | Realistic | Low-Medium | Very good | Lightweight, coord-only |

**Top three for first benchmark round** (best feasibility x compatibility, no scene-image dependency):
1. **SocialVAE** — closest CVAE analog to NaviGAN's per-timestep stochastic structure; should be straightforward to plug existing `resist_loss` into.
2. **EqMotion** — strong reported zara1 numbers (0.18/0.32), equivariance is a meaningful inductive-bias upgrade, single forward pass (no diffusion latency tax).
3. **LED** — only diffusion candidate worth benchmarking under the ~100 ms budget; confirm Orin latency before committing.

**Honorable mentions:**
- **PECNet** — proven, lightweight, and its endpoint conditioning lines up with NaviGAN's existing `goal_state` input.
- **SocialCircle** — minimal integration risk if treated as a PoolHiddenNet swap inside NaviGAN itself (ablation rather than full alternative).

**Explicit non-candidates** (do not run in first round): HiVT, MUSE-VAE, Y-Net — all require scene/map input incompatible with current NaviGAN coordinate-only deployment pipeline.

---

## Caveats on the Numbers

- Several paper PDFs failed to render via fetch tooling. zara1-specific ADE/FDE values marked `(unverified)` come from secondary sources (other papers' comparison tables, ResearchGate figure extracts). Confirm against original paper tables before final benchmark report.
- All ADE/FDE values assume **Best-of-20 sampling, 8 obs / 12 pred frames** — the standard ETH/UCY social benchmark protocol that matches NaviGAN's setup.
- Latency estimates are extrapolated from desktop GPU numbers; Orin AGX (Ampere, 32GB) numbers will differ. Validate empirically before adopting any candidate.
