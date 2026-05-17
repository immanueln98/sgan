# Integrating Post-2020 Ideas INTO NaviGAN (preserving intention+force fusion)

**Reframing**: rather than replacing NaviGAN with a different architecture, identify which components/ideas from the surveyed papers can be **grafted into the existing `LateAttentionFullGenerator`** while keeping its core design intact.

**Hard requirement**: the intention-branch decoder + force-branch decoder + per-timestep attention-fusion structure MUST be preserved. This is non-negotiable for goal-seeking navigation under Nav2 — `BasicNavigator.followPath` consumes a single deterministic path and the intention branch is what makes the model goal-directed in the first place.

This doc supersedes the top-3 ranking in [`social_gan_advances_survey.md`](social_gan_advances_survey.md) for the actual question being asked. The companion docs ([`part1_gan_and_transformer.md`](part1_gan_and_transformer.md), [`part2_diffusion_cvae_graph.md`](part2_diffusion_cvae_graph.md)) remain valid as literature reference.

---

## What NaviGAN's structure allows us to change

Anatomy of `LateAttentionFullGenerator` (see `navigan_training/navigan_training/models/late_attention.py`):

```
obs_traj ──► Encoder (LSTM, h=64) ──► encoded_h ──┐
                                                   ├─► add_noise (currently dim=0)
                                                   ├─► add_goal (goal_state injected)
                                                   ▼
                                       intention_decoder.step_forward ──► intention_rel_pos
                                       force_decoder.step_forward      ──► force_rel_pos
                                                   │            │
                                                   ▼            ▼
                       attention_mlp([force_rel_pos, intention_rel_pos])
                                                   │
                                                softmax
                                                   │
                              fused_pos = force * α + intention * (1-α)
```

Per timestep `t` in `range(pred_len)`: both branches step, attention MLP picks the mix, fused position becomes the next input. Force branch uses `PoolHiddenNet` (or `SocialPooling`) internally.

**Components we can swap without breaking the fusion**:

| Component | What feeds it | What consumes it | Constraints if swapped |
|-----------|---------------|------------------|------------------------|
| `Encoder` (LSTM) | `obs_traj (8, N, 2)` | Both decoder branches via `add_goal` | Output shape must be `(num_layers, batch, h_dim)` |
| `PoolHiddenNet` / `SocialPooling` | force_decoder hidden + end_pos | `force_decoder.step_forward` only | Must return tensor compatible with force decoder's `mlp_pre_pool` input |
| `attention_mlp` | concat([force, intention]) | softmax → blend weights | Output dim = 2 (one weight per branch) |
| Noise injection in `add_noise` | `aux_input` or random | encoded_h before decoder | Currently dim=0 → can re-enable for latent-code injection |
| `force_decoder` LSTM | encoded_h | per-step position | Must produce `(N, 2)` per step |
| `intention_decoder` LSTM | encoded_h + goal | per-step position | Must produce `(N, 2)` per step |
| `TrajectoryDiscriminator` | full traj | adversarial loss | Can be augmented with auxiliary heads |

**Components we cannot remove**:
- Force/intention branch separation (architectural identity)
- `goal_state` input pathway (Nav2 integration)
- Per-timestep position output (Path msg consumer)
- Hinge `resist_loss` operating on `force_rel_pos` (the team's reconstructed safety signal)

---

## Integration candidates — ranked

### Tier 1: Pure component swap (lowest risk, fastest to ship)

#### 1.1 SocialCircle pooling swap — *Top integration pick*
- **Source**: Wong et al., *SocialCircle: A Lightweight Plug-and-Play Component for Pedestrian Trajectory Prediction*, CVPR 2024. arXiv:2310.05370. Code: github.com/cocoon2wong/SocialCircle
- **Idea**: Replace `PoolHiddenNet` with **angular-sector social encoding**. Each sector around the target pedestrian summarises velocity, distance, and density of neighbours in that angle. Inspired by echolocation. Lightweight, trainable, designed as a drop-in for any predictor's social module.
- **What changes in NaviGAN**: in `late_attention.py`, replace `self.pool_net = PoolHiddenNet(...)` with `self.pool_net = SocialCircle(...)`. Force decoder consumes its output exactly as before. Intention branch, attention fusion, GAN losses, hinge `resist_loss` — all unchanged.
- **What this tests**: "Is angle-based social encoding better than max-pooled-MLP for NaviGAN's force branch?" Direct ablation of one component.
- **Expected gain**: Modest but principled. SocialCircle's paper shows consistent improvements across base predictors (V²-Net, E-V²-Net etc.) on ETH/UCY.
- **Effort**: **2-3 days**.
- **Files touched**:
  ```
  navigan_training/models/late_attention.py    # swap PoolHiddenNet → SocialCircle
  navigan_training/models/social_circle.py     # NEW — port SocialCircle module
  configs/zara1_socialcircle.yaml              # NEW — pooling_type='social_circle', circle_partitions=8
  ```
- **Caveat**: NaviGAN is not in SocialCircle's officially supported base list — must wire carefully against the production CUDA-clamp fix in our `PoolHiddenNet` copy.

#### 1.2 Social-Ways InfoGAN extension — *Top GAN-preserving pick*
- **Source**: Amirian et al., *Social Ways*, CVPRW 2019. arXiv:1904.09507. Code: github.com/amiryanj/socialways
- **Idea**: Add an **InfoGAN-style Q-network** on top of the existing GAN. Sample a latent code `c` (categorical or continuous) alongside noise `z`; the Q-network learns to predict `c` from the generated trajectory; mutual information `I(c; G(z, c))` is maximised. Disentangles latent factors into meaningful style codes.
- **What changes in NaviGAN**: 
  - Re-enable noise injection: `noise_dim=(8,)` (currently `(0,)` → deterministic). Latent code `c` has dim 4 (continuous) or 4-way categorical.
  - Add `Q_network` MLP head sharing the discriminator trunk (standard InfoGAN pattern).
  - Add `mutual_info_loss` term to generator loss.
  - Force/intention/attention/hinge — all untouched.
- **What this tests**: "Can we get disentangled latent control over trajectory style (e.g. cautious vs assertive, left-leaning vs right-leaning) that we can tune at deploy time without retraining?" High value if the codes turn out interpretable — gives a runtime knob the team currently lacks.
- **Expected gain**: Primary win is *controllability* + sample diversity, not raw ADE/FDE. Social-Ways's headline contribution was anti-mode-collapse, not absolute accuracy.
- **Effort**: **2-3 days**.
- **Files touched**:
  ```
  navigan_training/models/late_attention.py    # noise_dim>0; expose latent code in add_noise
  navigan_training/models/late_attention.py    # add Q-head class (shares Discriminator trunk)
  navigan_training/losses.py                   # add mutual_info_loss
  navigan_training/train_loop.py               # sample c during G-step, add MI term to g_total
  configs/zara1_infogan.yaml                   # NEW — noise_dim=8, latent_code_dim=4, mi_loss_weight=1.0
  ```
- **Critical caveat**: Re-enabling noise changes generator weight shapes — **train from scratch, not resume from `benchmark_zara1_with_model.pt`**.

---

### Tier 2: Principled backbone upgrade (modest risk, real gains expected)

#### 2.1 EqMotion equivariant interaction module
- **Source**: Xu et al., *EqMotion: Equivariant Multi-agent Motion Prediction with Invariant Interaction Reasoning*, CVPR 2023. arXiv:2303.10876. Code: github.com/MediaBrain-SJTU/EqMotion
- **Idea**: EqMotion's three modules are (a) equivariant geometric feature learning, (b) **invariant interaction reasoning**, (c) invariant pattern feature learning. Take **(b) only** — the interaction module — and graft it as the social-pooling layer of NaviGAN's force branch. Invariance under SE(2) (rotation+translation) means social encoding is robot-heading-agnostic.
- **What changes in NaviGAN**: replace `PoolHiddenNet` with `EqInteractionModule`. Force decoder consumes invariant interaction features. Intention branch and attention fusion unchanged.
- **What this tests**: "Does rotation-equivariant social encoding fix the implicit coordinate-frame dependency in PoolHiddenNet?" PoolHiddenNet's max-pool over MLP features over `(end_pos_i - end_pos_j)` is translation-equivariant but NOT rotation-equivariant — it learns a heading-dependent encoding. The robot operates at arbitrary heading; this could explain inconsistent behaviour across map orientations.
- **Expected gain**: Modest-to-strong. EqMotion's full model hits 0.18/0.32 zara1; isolating the interaction module captures part of that gain.
- **Effort**: **3-4 days** (port + adapt to NaviGAN's seq_start_end batch format).
- **Files touched**:
  ```
  navigan_training/models/late_attention.py    # swap PoolHiddenNet → EqInteractionModule
  navigan_training/models/eq_interaction.py    # NEW — port just the interaction module from EqMotion
  configs/zara1_eqinteraction.yaml             # NEW
  ```
- **Caveat**: EqMotion's interaction module assumes positions + velocities; NaviGAN passes only positions to `pool_net`. Need to also pass `obs_traj_rel` (displacements) into the new module.

#### 2.2 Transformer encoder swap (STAR-style)
- **Source**: Yu et al., *Spatio-Temporal Graph Transformer Networks*, ECCV 2020. arXiv:2005.08514. Code: github.com/cunjunyu/STAR
- **Idea**: Replace NaviGAN's `Encoder` (LSTM) with a **spatial-temporal Transformer**: interleaved spatial attention (over agents per frame) and temporal attention (over frames per agent). Both decoder branches consume the Transformer encoding instead of LSTM hidden state.
- **What changes in NaviGAN**: `Encoder` class replaced; output reshaped to match decoder `(num_layers, batch, h_dim)` interface. Force/intention decoders unchanged in their step loop. Per-step social pooling in force branch unchanged (still `PoolHiddenNet`).
- **What this tests**: "Does richer history encoding (attention over the full 8 obs frames + agents) feed better features into both branches than an LSTM final state?" LSTM-only encoding compresses to a single hidden vector; Transformer preserves token-level structure.
- **Expected gain**: STAR's full model hits 0.26/0.55 zara1; using just its encoder inside NaviGAN captures a fraction.
- **Effort**: **5-7 days** (Transformer port + interface adapter + retraining).
- **Files touched**:
  ```
  navigan_training/models/late_attention.py    # replace Encoder class; keep decoder + fusion
  navigan_training/models/star_encoder.py      # NEW — port STAR's spatial+temporal Transformer
  configs/zara1_star_encoder.yaml              # NEW
  ```

#### 2.3 AgentFormer-style agent-aware attention (as social pooling alternative)
- **Source**: Yuan et al., *AgentFormer*, ICCV 2021. arXiv:2103.14023. Code: github.com/Khrylx/AgentFormer
- **Idea**: Take only AgentFormer's **agent-aware attention layer** — where attention weights depend on whether tokens come from the same agent or different agents — and use it as the social pooling mechanism in NaviGAN's force branch. Skip the CVAE + DLow wrapper.
- **What changes in NaviGAN**: replace `PoolHiddenNet` with `AgentAwareAttention` layer. Force decoder consumes it.
- **Effort**: **4-5 days**.
- **Risk**: Agent-aware attention is most powerful in the joint socio-temporal setting AgentFormer uses (one Transformer over all (agent, time) tokens); isolating it as a pooling layer may discard much of its value. **Lower priority than EqMotion interaction module** which is designed as a modular component.

---

### Tier 3: Cross-cutting additions (preserves architecture, adds capability)

#### 3.1 PECNet-style endpoint augmentation for intention branch
- **Source**: Mangalam et al., *PECNet*, ECCV 2020. Code: github.com/HarshayuGirase/Human-Path-Prediction
- **Idea**: NaviGAN's intention branch is currently conditioned on the user-supplied `goal_state` (from Nav2's `BasicNavigator`). When the Nav2 goal is far (e.g. 20 m away) the intention branch must extrapolate over a long horizon with little guidance about the *next* 4.8 s. Add a small **endpoint-prediction head** that predicts a distribution over plausible near-term endpoints (e.g. at t+12 frames = 4.8 s out) from the past trajectory. Intention decoder conditions on **both** the long-term Nav2 goal and the near-term predicted endpoint.
- **What changes in NaviGAN**: add `endpoint_head` MLP on top of encoder output; sample/argmax predicted endpoint; modify `add_goal` to inject both `goal_state` and predicted endpoint into intention decoder. Force branch and attention fusion unchanged.
- **What this tests**: "Does separating long-term goal-seeking (Nav2) from near-term endpoint anticipation (learned) help when goals are distant?"
- **Effort**: **3-4 days**.
- **Files touched**:
  ```
  navigan_training/models/late_attention.py    # add endpoint_head; modify add_goal
  navigan_training/losses.py                   # add endpoint_loss (L2 vs ground-truth endpoint)
  configs/zara1_pecnet_intent.yaml             # NEW
  ```

#### 3.2 LED leap-diffusion refinement on fused output (not recommended)
- **Source**: Mao et al., *LED*, CVPR 2023. arXiv:2303.10895
- **Idea**: After NaviGAN computes fused `last_pos_rel` sequence, run τ=3-5 diffusion refinement steps to smooth/de-noise.
- **Why not recommended**: Adds 30-100 ms latency tax on Orin (LED reports 46 ms on NBA, desktop GPU; Orin will be slower). Eats the ROS2 10 Hz budget. The architecture preserves force/intention fusion but the deployment cost is hard to justify versus the gains.

---

### Tier 4: Do NOT graft into NaviGAN (architecturally incompatible)

| Model | Why incompatible |
|-------|------------------|
| **TUTR** | Mode-classification head + L motion-mode priors *replace* the per-step force/intention fusion. Can't preserve both. |
| **AgentFormer (full)** | CVAE + DLow sampler is a whole-trajectory stochastic model; fights deterministic per-step attention fusion. (Only agent-aware-attention layer is graftable — see Tier 2.3.) |
| **SocialVAE** | Timewise latents at each step replace the deterministic fusion output. Could co-exist but the resulting `fused_pos = latent_sample + fusion` would be semantically confused. |
| **MID / BCDiff** | Diffusion replaces the decoder entirely. No place to plug intention/force branches. |
| **EqMotion (full)** | Has its own pattern decoder that produces trajectories directly — no slot for two-branch fusion. |
| **MemoNet** | Non-parametric memory retrieval; force branch has nowhere to go. |
| **HiVT / MUSE-VAE / Y-Net / SoPhie / Social-Transmotion (multimodal)** | All need scene image / HD map / body pose — inputs the robot does not produce. |
| **Trajectron++** | Whole-system CVAE replaces NaviGAN end-to-end. |

---

## Top 3 integration recommendations (ranked)

| Rank | Candidate | What it changes | Effort | Expected gain | Risk |
|------|-----------|-----------------|--------|---------------|------|
| 1 | **SocialCircle pooling swap** | `PoolHiddenNet` → angular-sector encoder | 2-3 days | Modest, principled | Low |
| 2 | **Social-Ways InfoGAN extension** | Add Q-net + MI loss; re-enable noise | 2-3 days | Disentangled latent control (deploy-time tunable) | Low (but must retrain from scratch) |
| 3 | **EqMotion equivariant interaction module** | `PoolHiddenNet` → SE(2)-invariant encoder | 3-4 days | Heading-agnostic social encoding | Medium |

**Suggested first move**: implement **SocialCircle** first because it is the smallest change with the cleanest ablation signal. If it improves both `resist_count` (fewer pedestrian-clearance violations) and `ADE/FDE` on zara1 val, that's a free win — ship it as the new production checkpoint. If only one improves, you have new data on whether NaviGAN's bottleneck is social encoding or somewhere else.

**Then implement Social-Ways InfoGAN** in parallel — it tests a different axis (controllability rather than raw accuracy) and the team has explicitly asked about InfoGAN. Both fit comfortably inside the existing `train_loop.py` + `losses.py` structure.

**Hold EqMotion's equivariant interaction module** as the third bet — biggest principled upgrade, but requires the most adaptation work (positions + velocities into the new pooling interface).

---

## Open questions before implementation

1. **For SocialCircle** — circle_partitions=8 (default) or higher? Higher = finer angular resolution but more compute. On Orin with N=20 peds, 8 should be plenty.

2. **For Social-Ways InfoGAN** — categorical or continuous latent code `c`? Categorical (e.g. 4 styles) is more interpretable; continuous gives finer control. Recommend categorical first to verify the codes are interpretable, then switch to continuous if needed.

3. **For EqMotion interaction module** — the module needs velocities, not just positions. Use `obs_traj_rel` (displacements) as velocity proxy, or compute finite differences explicitly? `obs_traj_rel` already in the data loader 9-tuple — easier.

4. **For all three** — train on zara1 only (apples-to-apples vs current production checkpoint) or full ETH/UCY (more general)? Recommend zara1 first; if the change helps on zara1, broaden later.

5. **Benchmark protocol** — same as in master doc (`scripts/eval_checkpoint.py` + `scripts/plot_predictions.py` + Orin timing CSV + rosbag failure-mode replay). Pass criteria: ADE/FDE within 10% or better, `resist_count` strictly lower, p95 Orin latency under 80 ms.

---

## What this does NOT cover

- **TensorRT/ONNX deploy** of the modified model — separate concern, same as existing NaviGAN.
- **Re-training of the discriminator** — for SocialCircle and EqMotion swaps, the existing discriminator should still work; for InfoGAN, the discriminator needs the Q-head augmentation (trivial PyTorch change).
- **C++ inference path** — assumes the team continues to use the Python `Navigan.py` wrapper in `navigan_node.py`. C++ rewrite is out of scope.

Nothing has been implemented yet. This is a research/decision doc only.
