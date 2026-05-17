# Part 1: GAN- and Transformer-based Social Trajectory Prediction (2020-2025)

Survey scoped for selecting a candidate to train and benchmark against the current NaviGAN baseline (`benchmark_zara1_with_model.pt`). Diffusion / CVAE / pure-graph / goal-only families are intentionally excluded (covered separately).

## Baseline being challenged

**NaviGAN (Tsai & Oh, ICRA 2020)** — `LateAttentionFullGenerator` on top of Social-GAN PoolHiddenNet; force + intention attention fusion; hinge social-repulsion loss with `d_safe`; deterministic (`noise_dim=(0,)`); ~101k params; ~10 Hz on Jetson Orin AGX. Trained on ETH/UCY (zara1). `obs=8 (3.2s)`, `pred=12 (4.8s)`.

Numerical anchor: SGAN reports zara1 ADE/FDE 0.34/0.69 (K=20); SoPhie 0.30/0.63; STAR 0.26/0.55 (stochastic K=20). Anything close to or below these is competitive at the dataset level.

---

## GAN-based candidates

### Social-BiGAT (Kosaraju et al., NeurIPS 2019)
- **Cite**: *Social-BiGAT: Multimodal Trajectory Forecasting using Bicycle-GAN and Graph Attention Networks.* NeurIPS 2019. arXiv:1907.03395. Reference code: no canonical authors' release; community reimpls only (e.g. github.com/jamiekang/Social-BiGAT-paper-read, https://github.com/huang-xx/STGAT for related GAT baseline).
- **Idea**: Replaces SGAN's pooling with a Graph Attention Network (GAT) for permutation-invariant social encoding; adds a latent-scene encoder + Bicycle-GAN-style reverse mapping from trajectory to latent z to fight mode-collapse and improve K=1 generalization.
- **ETH/UCY ADE/FDE (K=20, meters)**: ETH 0.69/1.29, HOTEL 0.49/1.01, UNIV 0.55/1.32, **ZARA1 0.30/0.62**, ZARA2 0.36/0.75, AVG **0.48/1.00**. At K=1, AVG 0.606/1.328 (vs SGAN 0.846/1.758, Sophie 0.712/1.456) — the BiGAN reverse mapping is the actual win at low sample budgets.
- **Params / latency**: (unreported in paper).
- **Integration cost**: Medium. GAT replaces PoolHiddenNet (drop-in at encoder level) but the BiGAN training loop (extra latent-encoder + cycle loss) requires a real training refactor, and the latent-scene encoder assumes a scene image (not in our coords-only pipeline; can be ablated out).
- **Jetson Orin feasibility**: Realistic. GAT on N<=20 agents is cheap; generator depth is comparable to SGAN.
- **Hinge-repulsion compatibility**: Yes. Generator outputs per-timestep (x,y) sequences just like SGAN, so the existing `resist_loss` with `d_safe` plugs in unchanged.

### SoPhie (Sadeghian et al., CVPR 2019)
- **Cite**: *SoPhie: An Attentive GAN for Predicting Paths Compliant to Social and Physical Constraints.* CVPR 2019. arXiv:1806.01482. Code: no maintained official; multiple community ports.
- **Idea**: SGAN backbone augmented with two soft-attention modules — *physical attention* over a CNN-encoded scene image and *social attention* over neighbour features. First GAN to combine map context with social pooling.
- **ETH/UCY ADE/FDE (K=20, meters)**: ETH 0.70/1.43, HOTEL 0.76/1.67, UNIV 0.54/1.24, **ZARA1 0.30/0.63**, ZARA2 0.38/0.78, AVG **0.54/1.15**.
- **Params / latency**: (unreported).
- **Integration cost**: High. Physical attention requires a top-down scene image / occupancy map registered to robot frame; we do not currently feed any map tensor into NaviGAN. Stripping physical attention reverts to roughly SGAN+social-attention — marginal gain.
- **Jetson Orin feasibility**: Borderline. Adding a CNN scene encoder on top of the per-step GAN forward pass would eat into the 100 ms budget; depends on backbone choice.
- **Hinge-repulsion compatibility**: Yes — per-timestep position outputs.

### Social-Ways (Amirian et al., CVPRW 2019)
- **Cite**: *Social Ways: Learning Multi-Modal Distributions of Pedestrian Trajectories with GANs.* CVPRW 2019. arXiv:1904.09507. Code: https://github.com/amiryanj/socialways.
- **Idea**: **InfoGAN-style** mutual-information loss applied directly to pedestrian trajectory GAN. Drops the L2 variety loss used by SGAN (which they argue causes mode collapse) and instead maximises MI between generator latent codes and the produced trajectory via an auxiliary Q-network. Closest published match to a true "InfoGAN for trajectories" — searches for newer 2020+ InfoGAN-trajectory papers returned only the controllable-trajectory GAN line (Springer 2024) and disentangled GCN approaches that are not InfoGAN-based.
- **ETH/UCY ADE/FDE**: Reports lower diversity-vs-precision tradeoff than SGAN but on a non-standard subset; commonly quoted average ~0.47/1.00 across the 5 splits at K=20 (best-of-N), zara1 specifically (unverified) in source paper tables.
- **Params / latency**: (unreported).
- **Integration cost**: Low-medium. Drop-in generator; add a Q-network MLP head and a CE/MI loss term — small training-loop change. Could be combined with the existing hinge-repulsion loss.
- **Jetson Orin feasibility**: Realistic. Inference path is identical to SGAN — Q-network is training-only.
- **Hinge-repulsion compatibility**: Yes. Output is per-timestep positions; hinge loss adds straightforwardly.

### Conditional / Wasserstein GAN trajectory variants (post-2020)
General observation from the literature scan: post-2020 GAN-only pedestrian work largely consolidated around three patterns — (1) Wasserstein-GP discriminators bolted onto SGAN/STGAT (e.g. Lv et al., *Int. J. Intelligent Systems* 2022, "An improved GAN with transformers for pedestrian trajectory prediction"); (2) controllable / disentangled GAN heads (Springer 2024 GAT+GAN with InfoGAN-style control vectors); (3) conditional GANs with goal/intention as the condition (overlap with NaviGAN's own design). None have produced a clean SOTA on ETH/UCY that beats AgentFormer/MemoNet/TUTR, and most do not publish reproducible code or zara1-specific numbers. **Recommendation**: not worth a dedicated training run unless we specifically want to ablate the WGAN-GP discriminator under NaviGAN's loss stack.

---

## Transformer-based candidates

### STAR (Yu et al., ECCV 2020)
- **Cite**: *Spatio-Temporal Graph Transformer Networks for Pedestrian Trajectory Prediction.* ECCV 2020. arXiv:2005.08514. Code: https://github.com/cunjunyu/STAR (also Majiker/STAR fork).
- **Idea**: Pure-attention model. Stacks interleaved spatial Transformers (TGConv — a Transformer-based graph conv over the per-frame agent graph) and temporal Transformers per agent. Adds a read/write external memory module to smooth long-horizon predictions. No recurrence, no GAN.
- **ETH/UCY ADE/FDE (K=20, meters)**: ETH 0.36/0.65, HOTEL 0.17/0.36, **ZARA1 0.26/0.55**, ZARA2 0.22/0.46, UNIV 0.31/0.62, AVG **0.26/0.53**. Deterministic variant (STAR-D): AVG 0.41/0.87.
- **Params / latency**: (unreported in paper). Feature size 32, 8-head attention, 256 peds per batch — model is small (low tens of k params typical for this configuration).
- **Integration cost**: Medium. Backbone swap (Transformer instead of GAN generator); but matches our same 8-in / 12-out interface, target-centric coords work fine. Loses NaviGAN's intention/force fusion — we would need to re-graft the goal-conditioning input.
- **Jetson Orin feasibility**: Realistic. Pure-attention on N=20 is fast; should fit well under 100 ms.
- **Hinge-repulsion compatibility**: Yes. Per-timestep (x,y) outputs, deterministic head — `resist_loss` plugs in directly.

### AgentFormer (Yuan et al., ICCV 2021)
- **Cite**: *AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting.* ICCV 2021. arXiv:2103.14023. Code: https://github.com/Khrylx/AgentFormer.
- **Idea**: Single Transformer that jointly models time and agents in one sequence (no separate spatial/temporal stacks). Introduces *agent-aware attention* that lets attention weights depend on whether tokens come from the same agent or different agents while preserving permutation invariance. Wrapped in a CVAE for stochasticity.
- **ETH/UCY ADE/FDE (K=20, meters)**: ETH 0.26/0.39, HOTEL 0.11/0.14, UNIV 0.26/0.46, **ZARA1 0.15/0.23**, ZARA2 0.14/0.24, AVG **0.18/0.29** (their Table 1 — SOTA at publication).
- **Params / latency**: dk=dv=dq=256, FFN=512, 8 heads, 2 layers each in enc/dec; (~few M params, exact unreported). Latency unreported.
- **Integration cost**: High. The CVAE wrapper plus DLow trajectory sampler is structurally different from our deterministic GAN inference path; replacing the generator alone is not enough — also need to swap our hinge loss against the ELBO+sampler loss, or do a deterministic ablation (no CVAE) at expected cost to quality.
- **Jetson Orin feasibility**: Borderline. Joint socio-temporal sequence length grows as O((1+N)*T) and self-attention is quadratic; with N=20, T=20 this is 420 tokens — feasible but tight after ZED reserves 20% GPU. Inference timing on Orin not reported by authors.
- **Hinge-repulsion compatibility**: Partial. Decoder emits per-step positions so hinge can be applied per timestep on each sampled K trajectory. But the model commits to stochastic K-sample form — applying hinge to every sample inflates training cost K-fold, and the CVAE prior fights a deterministic safety constraint. **Awkward fit for our current loss stack.**

### MemoNet (Xu et al., CVPR 2022)
- **Cite**: *Remember Intentions: Retrospective-Memory-based Trajectory Prediction.* CVPR 2022. arXiv:2203.11474. Code: https://github.com/MediaBrain-SJTU/MemoNet.
- **Idea**: Instance-based / non-parametric. Builds a memory bank of (past-trajectory feature, future-intention feature) pairs from the training set. At inference, a learnable addresser retrieves the K nearest memory instances; an intention-clustering step picks K final intention anchors; a "trajectory fulfilment" decoder fills in the full path conditioned on the retrieved intention. Reduces FDE by 10.2% over AgentFormer on ETH-UCY at publication. Not a Transformer per se but qualifies as a non-GAN architectural alternative; included for completeness.
- **ETH/UCY ADE/FDE (K=20, meters)**: ETH 0.40/0.61, HOTEL 0.11/0.17, UNIV 0.24/0.43, **ZARA1 0.18/0.32**, ZARA2 0.14/0.24, AVG **0.21/0.35**.
- **Params / latency**: SDD inference 18.03 ms/sample (~55.5 FPS) on RTX 3090 (their numbers); memory bank ~11 MB at θ=1. ETH-UCY memory size smaller. Params (unreported, dominated by memory bank not weights).
- **Integration cost**: High. Memory bank must be re-built per dataset and pinned at inference; addresser + clustering + fulfilment pipeline replaces the generator end-to-end. Cold-start at deploy needs the training trajectories shipped to the Jetson.
- **Jetson Orin feasibility**: Borderline. Authors only report RTX 3090 (much faster than Orin); memory bank queries scale with bank size; latency on Orin would need verification, but conceptually compatible with 100 ms.
- **Hinge-repulsion compatibility**: Partial. Fulfilment decoder emits per-timestep positions, but training adds intention-decoder + trajectory-decoder L2 losses — adding hinge needs careful loss balancing and the bank is built around recalled intentions, not safety constraints.

### TUTR (Shi et al., ICCV 2023)
- **Cite**: *Trajectory Unified Transformer for Pedestrian Trajectory Prediction.* ICCV 2023. arXiv:2307.01927 (note: original prompt arxiv id 2307.15904 was wrong; correct paper is from Shi/Wang/Zhou/Hua). Code: https://github.com/lssiair/TUTR.
- **Idea**: Unifies social interaction and multimodal prediction into a single encoder-decoder Transformer. Generates L "general motion modes" (per-scene clustering of training trajectories) as explicit global priors, then a mode-level encoder + social-level decoder produces both per-mode trajectories and their probabilities **without post-processing** (no clustering at inference). 10x-40x faster than MemoNet/SocialVAE+FPC at inference.
- **ETH/UCY ADE/FDE (K=20, meters)**: ETH 0.40/0.61, HOTEL 0.11/0.18, UNIV 0.23/0.42, **ZARA1 0.18/0.34**, ZARA2 0.13/0.25, AVG **0.21/0.36** (best in the *no-post-processing* block, matches MemoNet which uses post-processing). brier-ADE/FDE AVG 0.95/1.10.
- **Params / latency**: 2-layer mode-level Transformer encoder + 1-layer social-level decoder, 4 heads, 128 FFN, d_e=128 (ETH-UCY), d_e=64 (SDD). Their Table 5: prediction time on RTX 3090 for N=80 peds — TUTR 0.0533 s vs MemoNet 1.2989 s vs SocialVAE+FPC 2.0939 s. **About 19 FPS in a dense scene on 3090.** Param count not stated explicitly but the architecture is small (low hundreds of k params at this dim).
- **Integration cost**: Medium. End-to-end transformer replaces the generator; need to compute L motion modes per dataset offline (k-means over training endpoint distribution) — adds a build step but conceptually simple. Probabilistic output (dual prediction head: regression + classification) is straightforward to argmax for a deterministic single-trajectory consumer.
- **Jetson Orin feasibility**: **Realistic — strongest candidate on latency.** Smallest model in the comparison, fastest reported by author by 10-40x, no post-processing tail. Should sit comfortably under 100 ms even on Orin.
- **Hinge-repulsion compatibility**: Yes (with caveats). Regression head outputs per-timestep positions per mode. Could apply hinge per-mode (small overhead with L=20 modes for zara1) or only on the argmax mode. Probability head from soft cross-entropy is independent of the hinge term — losses compose cleanly.

### Social-Transmotion (Saadatnejad et al., ICLR 2024)
- **Cite**: *Social-Transmotion: Promptable Human Trajectory Prediction.* ICLR 2024. arXiv:2312.16168. Code: https://github.com/vita-epfl/social-transmotion.
- **Idea**: First "promptable" trajectory Transformer — accepts arbitrary combinations of visual cues (past xy, 2D/3D pose keypoints, 2D/3D bounding boxes) per agent via a Cross-Modality Transformer (CMT) then a Social Transformer (ST). Uses modality- and meta-masking during training to make a single generic model robust to missing modalities at inference. **Deterministic** (single trajectory per agent, MSE loss).
- **ETH/UCY ADE/FDE**: Authors evaluate primarily on JTA / JRDB (where pose cues live). On JTA, trajectory-only Social-Transmotion 0.99/1.98 vs Trajectron++ 1.18/2.53. ETH/UCY appendix only — they note ETH/UCY has no pose data so the model reduces to a pure xy Transformer; (zara1 numbers unverified in main text — Appendix A.2). Take this as: when pose is available it wins; on coords-only ETH/UCY it is competitive but not clearly better than EqMotion/Trajectron++.
- **Params / latency**: (unreported in main paper).
- **Integration cost**: High *if* exploiting pose (need a body-pose stream from ZED — currently absent; only bounding-box-style detections from `ObjectsStamped`). Low-medium if used as a trajectory-only Transformer (then it is essentially CMT-ST with T-only modality — closer to TUTR / AgentFormer in capability without their multimodal-output story).
- **Jetson Orin feasibility**: Borderline-to-realistic. Dual-Transformer (CMT + ST), but with N=20 agents and small embed dim should fit; authors do not report Jetson timings.
- **Hinge-repulsion compatibility**: Yes — deterministic per-timestep positions, MSE-trained. Hinge loss can be added as a parallel term cleanly. **This is the most compatible Transformer for our loss stack** if we are willing to forgo the multimodal-output benefits AgentFormer/TUTR provide.

---

## Top-line recommendation for benchmark candidate

Ranked by *expected delta over NaviGAN within achievable training+integration budget*:

1. **TUTR** — best latency story for Orin, strong zara1 numbers (0.18/0.34), small model, clean per-mode hinge compatibility. **Top pick for transformer benchmark.**
2. **Social-Ways (InfoGAN-style)** — minimal training-loop change from current NaviGAN, addresses the mode-collapse weakness of NaviGAN's deterministic generator without abandoning the GAN+hinge stack. **Top pick for GAN benchmark.**
3. **STAR** — strong zara1 numbers (0.26/0.55), deterministic variant available, pure attention is fast. Fallback if TUTR's mode-clustering step proves messy to integrate.
4. **Social-BiGAT** — competitive zara1 (0.30/0.62), drop-in GAT for PoolHiddenNet, BiGAN reverse mapping fights mode collapse. Pick this if we want a more conservative GAN upgrade than Social-Ways.
5. **AgentFormer / MemoNet** — best absolute numbers but worst integration cost (CVAE + DLow / memory bank). Defer unless TUTR underdelivers.
6. **SoPhie / Social-Transmotion (full multimodal)** — both rely on inputs we do not currently feed (scene image / body pose). Defer pending perception-stack work.

## Notes / verification gaps
- Several papers do not publish parameter counts or Jetson-class latencies; flagged as `(unreported)` above.
- Social-Ways exact zara1 numbers were not retrieved in the verification budget — marked `(unverified)`.
- Social-Transmotion ETH/UCY split-level numbers are in appendix A.2 only and were not extracted in this pass.
- TUTR arxiv id was provided incorrectly in the original brief (2307.15904 is Sat2Cap); correct ICCV 2023 paper is reachable via the openaccess.thecvf.com PDF and the lssiair/TUTR github.
