"""LateAttentionFullGenerator + dependencies.

Copied from navigan/python/naviGAN_paper/sgan/notes_various_length_models.py.
Differences from source:
  - `_DEVICE_` global removed. Tensors derive device from input (DDP-safe).
  - Dead `mix_coefficient` Parameter inside forward dropped (was computed, unused).
  - SocialPooling keeps `grid_pos.clamp` (matches production CUDA fix in
    navigan/python/models.py:340 — defends against NaN/boundary indices in
    scatter_add during decoder rollout).
  - Only LateAttentionFullGenerator + TrajectoryDiscriminator kept.
  - Added SocialCircle as a third `pooling_type` option (Wong et al., CVPR 2024).

State_dict is shape-compatible with `benchmark_zara1_with_model.pt` when
`pooling_type='spool'` or `pool_net`. SocialCircle yields a different parameter
shape (new MLP head) and is not resume-compatible with the production checkpoint.
"""
import math

import torch
import torch.nn as nn


def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


def get_noise(shape, noise_type, aux_input=None, device=None):
    if noise_type == 'gaussian':
        return torch.randn(*shape, device=device)
    if noise_type == 'uniform':
        return torch.rand(*shape, device=device).sub_(0.5).mul_(2.0)
    if noise_type == 'inject_goal':
        return aux_input.reshape(shape).to(device)
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


class Encoder(nn.Module):
    def __init__(self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1, dropout=0.0):
        super().__init__()
        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self.spatial_embedding = nn.Linear(2, embedding_dim)

    def init_hidden(self, batch, device):
        return (torch.zeros(self.num_layers, batch, self.h_dim, device=device),
                torch.zeros(self.num_layers, batch, self.h_dim, device=device))

    def forward(self, obs_traj):
        batch = obs_traj.size(1)
        emb = self.spatial_embedding(obs_traj.reshape(-1, 2))
        emb = emb.view(-1, batch, self.embedding_dim)
        state = self.init_hidden(batch, obs_traj.device)
        _, state = self.encoder(emb, state)
        return state[0]


class PoolHiddenNet(nn.Module):
    def __init__(self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
                 activation='relu', batch_norm=True, dropout=0.0):
        super().__init__()
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.mlp_pre_pool = make_mlp(
            [embedding_dim + h_dim, 512, bottleneck_dim],
            activation=activation, batch_norm=batch_norm, dropout=dropout,
        )

    @staticmethod
    def _repeat_interleave(tensor, num_reps):
        # R1, R1, R2, R2 (each row repeated num_reps times consecutively)
        col_len = tensor.size(1)
        return tensor.unsqueeze(1).repeat(1, num_reps, 1).view(-1, col_len)

    def forward(self, h_states, seq_start_end, end_pos):
        pool_h = []
        for start, end in seq_start_end:
            start, end = start.item(), end.item()
            num_ped = end - start
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)  # H1,H2,H1,H2
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)  # P1,P2,P1,P2
            curr_end_pos_2 = self._repeat_interleave(curr_end_pos, num_ped)  # P1,P1,P2,P2
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
            mlp_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
            curr_pool_h = self.mlp_pre_pool(mlp_input)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h)
        return torch.cat(pool_h, dim=0)


class SocialPooling(nn.Module):
    """Social-LSTM grid pooling. Includes grid_pos.clamp CUDA-safety fix."""

    def __init__(self, h_dim=64, activation='relu', batch_norm=True, dropout=0.0,
                 neighborhood_size=2.0, grid_size=8, pool_dim=None):
        super().__init__()
        self.h_dim = h_dim
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size
        out_dim = pool_dim if pool_dim else h_dim
        self.mlp_pool = make_mlp(
            [grid_size * grid_size * h_dim, out_dim],
            activation=activation, batch_norm=batch_norm, dropout=dropout,
        )

    def get_bounds(self, ped_pos):
        n2 = self.neighborhood_size / 2
        top_left = torch.stack([ped_pos[:, 0] - n2, ped_pos[:, 1] + n2], dim=1)
        bottom_right = torch.stack([ped_pos[:, 0] + n2, ped_pos[:, 1] - n2], dim=1)
        return top_left, bottom_right

    def get_grid_locations(self, top_left, other_pos):
        cell_x = torch.floor(
            ((other_pos[:, 0] - top_left[:, 0]) / self.neighborhood_size) * self.grid_size)
        cell_y = torch.floor(
            ((top_left[:, 1] - other_pos[:, 1]) / self.neighborhood_size) * self.grid_size)
        return cell_x + cell_y * self.grid_size

    @staticmethod
    def _repeat_interleave(tensor, num_reps):
        col_len = tensor.size(1)
        return tensor.unsqueeze(1).repeat(1, num_reps, 1).view(-1, col_len)

    def forward(self, h_states, seq_start_end, end_pos):
        pool_h = []
        for start, end in seq_start_end:
            start, end = start.item(), end.item()
            num_ped = end - start
            grid_size = self.grid_size * self.grid_size
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_hidden_repeat = curr_hidden.repeat(num_ped, 1)
            curr_end_pos = end_pos[start:end]
            curr_pool_h_size = num_ped * grid_size + 1
            curr_pool_h = curr_hidden.new_zeros((curr_pool_h_size, self.h_dim))
            top_left, bottom_right = self.get_bounds(curr_end_pos)
            curr_end_pos = curr_end_pos.repeat(num_ped, 1)
            top_left = self._repeat_interleave(top_left, num_ped)
            bottom_right = self._repeat_interleave(bottom_right, num_ped)
            grid_pos = self.get_grid_locations(top_left, curr_end_pos).type_as(seq_start_end)

            x_bound = ((curr_end_pos[:, 0] >= bottom_right[:, 0]) +
                       (curr_end_pos[:, 0] <= top_left[:, 0]))
            y_bound = ((curr_end_pos[:, 1] >= top_left[:, 1]) +
                       (curr_end_pos[:, 1] <= bottom_right[:, 1]))
            within_bound = x_bound + y_bound
            within_bound[0::num_ped + 1] = 1  # exclude self
            within_bound = within_bound.view(-1)

            grid_pos += 1
            offset = torch.arange(0, grid_size * num_ped, grid_size).type_as(seq_start_end)
            offset = self._repeat_interleave(offset.view(-1, 1), num_ped).view(-1)
            grid_pos += offset
            grid_pos[within_bound != 0] = 0
            # CUDA-safety: clamp to valid scatter_add range. Decoder rollout can
            # produce NaN/inf positions which floor() to garbage indices; without
            # this clamp scatter_add triggers an unrecoverable device-side assert.
            grid_pos = grid_pos.clamp(0, curr_pool_h_size - 1)
            grid_pos = grid_pos.view(-1, 1).expand_as(curr_hidden_repeat)
            curr_pool_h = curr_pool_h.scatter_add(0, grid_pos, curr_hidden_repeat)
            pool_h.append(curr_pool_h[1:].view(num_ped, -1))
        return self.mlp_pool(torch.cat(pool_h, dim=0))


class SocialCircle(nn.Module):
    """Angular-sector social encoder. Drop-in replacement for PoolHiddenNet.

    Per target pedestrian, partitions the surrounding disk into `partitions`
    equal angular sectors. In each sector, aggregates two factors over the
    neighbours falling in that sector:
      - mean distance (Euclidean, target -> neighbour)
      - density (neighbour count)
    The resulting per-target social feature has shape (partitions * 2) and is
    concatenated with the target's hidden state, then projected to
    bottleneck_dim by an MLP — matching PoolHiddenNet's output contract so the
    rest of the generator is unchanged.

    Reference: Wong et al., "SocialCircle: A Lightweight Plug-and-Play
    Component for Pedestrian Trajectory Prediction", CVPR 2024
    (arXiv:2310.05370). Here we use the distance + density factors only and
    omit velocity / move-direction factors so the forward signature stays
    identical to PoolHiddenNet (positions only).
    """

    def __init__(self, h_dim=64, bottleneck_dim=1024, partitions=8,
                 mlp_dim=512, activation='relu', batch_norm=True, dropout=0.0):
        super().__init__()
        self.h_dim = h_dim
        self.partitions = partitions
        self.bottleneck_dim = bottleneck_dim
        self.num_features = 2  # mean distance + density
        social_feat_dim = partitions * self.num_features
        self.mlp_post = make_mlp(
            [h_dim + social_feat_dim, mlp_dim, bottleneck_dim],
            activation=activation, batch_norm=batch_norm, dropout=dropout,
        )

    def _scene_features(self, end_pos):
        """Per-ped angular-sector features for a single scene.

        end_pos: (n, 2). Returns (n, partitions * 2).
        """
        n = end_pos.size(0)
        if n == 0:
            return end_pos.new_zeros((0, self.partitions * self.num_features))

        # delta[i, j] = pos[j] - pos[i]  ->  vector from target i to neighbour j
        delta = end_pos.unsqueeze(0) - end_pos.unsqueeze(1)  # (n, n, 2)
        dist = delta.norm(dim=-1)  # (n, n)
        angle = torch.atan2(delta[..., 1], delta[..., 0])  # (n, n) in (-pi, pi]
        angle = (angle + 2 * math.pi) % (2 * math.pi)  # -> [0, 2pi)
        sector_idx = (angle / (2 * math.pi / self.partitions)).long()
        # Defence: rollout positions can be NaN/inf. NaN.long() is undefined
        # and would corrupt the one-hot expansion. Mirrors SocialPooling's
        # grid_pos.clamp safety fix.
        sector_idx = sector_idx.clamp(0, self.partitions - 1)

        # Self-exclusion mask (target should not see itself in any sector)
        eye = torch.eye(n, dtype=torch.bool, device=end_pos.device)
        valid = ~eye  # (n, n)

        sector_one_hot = torch.nn.functional.one_hot(
            sector_idx, num_classes=self.partitions).bool()  # (n, n, partitions)
        mask = sector_one_hot & valid.unsqueeze(-1)  # (n, n, partitions)
        mask_f = mask.float()

        count = mask_f.sum(dim=1)  # (n, partitions) - density
        dist_sum = (dist.unsqueeze(-1) * mask_f).sum(dim=1)  # (n, partitions)
        dist_mean = dist_sum / count.clamp(min=1.0)

        feats = torch.stack([dist_mean, count], dim=-1)  # (n, partitions, 2)
        return feats.view(n, -1)

    def forward(self, h_states, seq_start_end, end_pos):
        h_flat = h_states.view(-1, self.h_dim)
        out = []
        for start, end in seq_start_end:
            start, end = start.item(), end.item()
            scene_pos = end_pos[start:end]
            scene_h = h_flat[start:end]
            social_feats = self._scene_features(scene_pos)
            out.append(torch.cat([scene_h, social_feats], dim=1))
        return self.mlp_post(torch.cat(out, dim=0))


class Decoder(nn.Module):
    """Per-step decoder with optional pool-every-timestep. step_forward used by
    LateAttentionFullGenerator's per-timestep attention fusion."""

    def __init__(self, seq_len, embedding_dim=64, h_dim=128, mlp_dim=1024, num_layers=1,
                 pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
                 activation='relu', batch_norm=True, pooling_type='pool_net',
                 neighborhood_size=2.0, grid_size=8, spatial_dim=2,
                 circle_partitions=8):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.pool_every_timestep = pool_every_timestep
        self.decoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)

        if pool_every_timestep:
            if pooling_type == 'pool_net':
                self.pool_net = PoolHiddenNet(
                    embedding_dim=embedding_dim, h_dim=h_dim, mlp_dim=mlp_dim,
                    bottleneck_dim=bottleneck_dim, activation=activation,
                    batch_norm=batch_norm, dropout=dropout,
                )
            elif pooling_type == 'spool':
                self.pool_net = SocialPooling(
                    h_dim=h_dim, activation=activation, batch_norm=batch_norm,
                    dropout=dropout, neighborhood_size=neighborhood_size,
                    grid_size=grid_size,
                )
            elif pooling_type == 'social_circle':
                self.pool_net = SocialCircle(
                    h_dim=h_dim, bottleneck_dim=bottleneck_dim,
                    partitions=circle_partitions, mlp_dim=mlp_dim,
                    activation=activation, batch_norm=batch_norm, dropout=dropout,
                )
            self.mlp = make_mlp(
                [h_dim + bottleneck_dim, mlp_dim, h_dim],
                activation=activation, batch_norm=batch_norm, dropout=dropout,
            )
        self.spatial_embedding = nn.Linear(spatial_dim, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, 2)

    def step_forward(self, last_pos, last_pos_rel, state_tuple, seq_start_end):
        batch = last_pos.size(0)
        decoder_input = self.spatial_embedding(last_pos_rel).view(1, batch, self.embedding_dim)
        output, state_tuple = self.decoder(decoder_input, state_tuple)
        rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
        curr_pos = rel_pos + last_pos

        if self.pool_every_timestep:
            decoder_h = state_tuple[0]
            pool_h = self.pool_net(decoder_h, seq_start_end, curr_pos)
            decoder_h = torch.cat([decoder_h.view(-1, self.h_dim), pool_h], dim=1)
            decoder_h = self.mlp(decoder_h).unsqueeze(0)
            state_tuple = (decoder_h, state_tuple[1])
        return rel_pos, state_tuple


class LateAttentionFullGenerator(nn.Module):
    """Two parallel encoder/decoder branches (intention + force) fused per-step
    via an attention MLP. Intention branch is goal-conditioned; force branch is
    social-pool-conditioned."""

    def __init__(self, obs_len, pred_len, embedding_dim=64, encoder_h_dim=64,
                 decoder_h_dim=128, mlp_dim=1024, num_layers=1, noise_dim=(0,),
                 noise_type='gaussian', noise_mix_type='ped', pooling_type=None,
                 pool_every_timestep=True, dropout=0.0, bottleneck_dim=1024,
                 activation='relu', batch_norm=True, neighborhood_size=2.0,
                 grid_size=8, goal_dim=(2,), spatial_dim=True,
                 circle_partitions=8):
        super().__init__()

        if pooling_type and pooling_type.lower() == 'none':
            pooling_type = None

        self.spatial_dim = spatial_dim
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.pooling_type = pooling_type
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = 1024
        self.goal_dim = goal_dim

        self.intention_encoder = Encoder(embedding_dim, encoder_h_dim, mlp_dim,
                                         num_layers, dropout)
        self.force_encoder = Encoder(embedding_dim, encoder_h_dim, mlp_dim,
                                     num_layers, dropout)

        self.intention_decoder = Decoder(
            pred_len, embedding_dim=embedding_dim, h_dim=decoder_h_dim,
            mlp_dim=mlp_dim, num_layers=num_layers, pool_every_timestep=False,
            dropout=dropout, bottleneck_dim=bottleneck_dim, activation=activation,
        )
        self.force_decoder = Decoder(
            pred_len, embedding_dim=embedding_dim, h_dim=decoder_h_dim,
            mlp_dim=mlp_dim, num_layers=num_layers,
            pool_every_timestep=pool_every_timestep, dropout=dropout,
            bottleneck_dim=bottleneck_dim, activation=activation,
            batch_norm=batch_norm, pooling_type=pooling_type, grid_size=grid_size,
            neighborhood_size=neighborhood_size,
            spatial_dim=2 if spatial_dim else decoder_h_dim,
            circle_partitions=circle_partitions,
        )

        if pooling_type == 'pool_net':
            self.pool_net = PoolHiddenNet(
                embedding_dim=embedding_dim, h_dim=encoder_h_dim, mlp_dim=mlp_dim,
                bottleneck_dim=bottleneck_dim, activation=activation,
                batch_norm=batch_norm,
            )
        elif pooling_type == 'spool':
            self.pool_net = SocialPooling(
                h_dim=encoder_h_dim, activation=activation, batch_norm=batch_norm,
                dropout=dropout, neighborhood_size=neighborhood_size,
                grid_size=grid_size,
            )
        elif pooling_type == 'social_circle':
            self.pool_net = SocialCircle(
                h_dim=encoder_h_dim, bottleneck_dim=bottleneck_dim,
                partitions=circle_partitions, mlp_dim=mlp_dim,
                activation=activation, batch_norm=batch_norm, dropout=dropout,
            )

        if self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = noise_dim[0]

        input_dim = encoder_h_dim + bottleneck_dim if pooling_type else encoder_h_dim

        if self.goal_dim[0] == 0:
            self.goal_dim = None
        else:
            self.goal_first_dim = goal_dim[0]

        if self._force_mlp_needed():
            self.force_mlp_decoder_context = make_mlp(
                [input_dim, mlp_dim, decoder_h_dim - self.noise_first_dim],
                activation=activation, batch_norm=batch_norm, dropout=dropout,
            )

        if self._intention_mlp_needed():
            self.intention_mlp_decoder_context = make_mlp(
                [encoder_h_dim, mlp_dim, decoder_h_dim - self.goal_first_dim],
                activation=activation, batch_norm=batch_norm, dropout=dropout,
            )

        self.attention_mlp = nn.Linear(2 * decoder_h_dim, 2)

    def _force_mlp_needed(self):
        return bool(self.noise_dim or self.pooling_type or
                    self.encoder_h_dim != self.decoder_h_dim)

    def _intention_mlp_needed(self):
        return bool(self.goal_dim or self.encoder_h_dim != self.decoder_h_dim)

    def add_noise(self, _input, seq_start_end, user_noise=None, aux_input=None):
        if not self.noise_dim:
            return _input
        if self.noise_mix_type == 'global':
            noise_shape = (seq_start_end.size(0),) + self.noise_dim
        else:
            noise_shape = (_input.size(0),) + self.noise_dim

        if user_noise is not None:
            z = user_noise
        else:
            z = get_noise(noise_shape, self.noise_type,
                          aux_input=aux_input, device=_input.device)

        if self.noise_mix_type == 'global':
            parts = []
            for idx, (start, end) in enumerate(seq_start_end):
                start, end = start.item(), end.item()
                vec = z[idx].view(1, -1).repeat(end - start, 1)
                parts.append(torch.cat([_input[start:end], vec], dim=1))
            return torch.cat(parts, dim=0)
        return torch.cat([_input, z], dim=1)

    def add_goal(self, _input, seq_start_end, goal_input=None):
        if not self.goal_dim:
            return _input
        goal_shape = (_input.size(0),) + self.goal_dim
        z = get_noise(goal_shape, 'inject_goal',
                      aux_input=goal_input, device=_input.device)
        return torch.cat([_input, z], dim=1)

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, aux_input=None,
                user_noise=None, goal_input=None, seq_len=12, goal_aggr=0.5):
        """
        Inputs:
          obs_traj:     (obs_len, batch, 2)
          obs_traj_rel: (obs_len, batch, 2) — relative to obs_traj[0]
          seq_start_end: (num_seqs, 2)
          goal_input:   (batch, 2) or reshape-compatible — exit-point goal in rel coords
        Returns:
          (pred_traj_rel, [attention, intent, social])
            pred_traj_rel: (seq_len, batch, 2)
            attention: (seq_len, batch, 2)
            intent:    (seq_len, batch, 2) — intention-branch rollout (used by intention_loss)
            social:    (seq_len, batch, 2) — force-branch rollout
        """
        batch_size = obs_traj_rel.size(1)
        device = obs_traj.device

        force_h = self.force_encoder(obs_traj_rel)
        intention_h = self.intention_encoder(obs_traj_rel)

        if self.pooling_type:
            end_pos = obs_traj[-1, :, :]
            pool_h = self.pool_net(force_h, seq_start_end, end_pos)
            force_ctx = torch.cat([force_h.view(-1, self.encoder_h_dim), pool_h], dim=1)
        else:
            force_ctx = force_h.view(-1, self.encoder_h_dim)
        intention_ctx = intention_h.view(-1, self.encoder_h_dim)

        if self._force_mlp_needed():
            force_ctx = self.force_mlp_decoder_context(force_ctx)
        force_decoder_h = self.add_noise(
            force_ctx, seq_start_end, aux_input=aux_input, user_noise=user_noise)
        force_decoder_h = force_decoder_h.unsqueeze(0)
        force_decoder_c = torch.zeros(self.num_layers, batch_size,
                                      self.decoder_h_dim, device=device)

        if self._intention_mlp_needed():
            intention_ctx = self.intention_mlp_decoder_context(intention_ctx)
        intention_decoder_h = self.add_goal(
            intention_ctx, seq_start_end, goal_input=goal_input)
        intention_decoder_h = intention_decoder_h.unsqueeze(0)
        intention_decoder_c = torch.zeros(self.num_layers, batch_size,
                                          self.decoder_h_dim, device=device)

        force_state = (force_decoder_h, force_decoder_c)
        intention_state = (intention_decoder_h, intention_decoder_c)

        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]

        ret, attention, intent, social = [], [], [], []
        for _ in range(seq_len):
            intention_rel_pos, intention_state = self.intention_decoder.step_forward(
                last_pos, last_pos_rel, intention_state, seq_start_end)
            intention_pos = intention_rel_pos + obs_traj[0]

            if self.spatial_dim:
                force_rel_pos, force_state = self.force_decoder.step_forward(
                    intention_pos, intention_rel_pos, force_state, seq_start_end)
            else:
                force_rel_pos, force_state = self.force_decoder.step_forward(
                    intention_pos, intention_state[0], force_state, seq_start_end)

            attention_score = self.attention_mlp(torch.cat([
                force_state[0].view(-1, self.decoder_h_dim),
                intention_state[0].view(-1, self.decoder_h_dim),
            ], dim=1))
            attention_score = torch.nn.functional.softmax(attention_score, dim=1)

            last_pos_rel = (force_rel_pos * attention_score[:, 0].view(-1, 1)
                            + intention_rel_pos * attention_score[:, 1].view(-1, 1))
            ret.append(last_pos_rel)
            attention.append(attention_score)
            intent.append(intention_rel_pos)
            social.append(force_rel_pos)
            last_pos = last_pos_rel + obs_traj[0]

        return (torch.stack(ret), [torch.stack(attention), torch.stack(intent), torch.stack(social)])


class TrajectoryDiscriminator(nn.Module):
    def __init__(self, obs_len, pred_len, embedding_dim=64, h_dim=64, mlp_dim=1024,
                 num_layers=1, activation='relu', batch_norm=True, dropout=0.0,
                 d_type='local'):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.h_dim = h_dim
        self.d_type = d_type

        self.encoder = Encoder(embedding_dim, h_dim, mlp_dim, num_layers, dropout)
        self.real_classifier = make_mlp(
            [h_dim, mlp_dim, 1],
            activation=activation, batch_norm=batch_norm, dropout=dropout,
        )
        if d_type == 'global':
            self.pool_net = PoolHiddenNet(
                embedding_dim=embedding_dim, h_dim=h_dim, mlp_dim=mlp_dim,
                bottleneck_dim=h_dim, activation=activation, batch_norm=batch_norm,
            )

    def forward(self, traj, traj_rel, seq_start_end=None):
        final_h = self.encoder(traj_rel)
        if self.d_type == 'local':
            classifier_input = final_h.squeeze(0)
        else:
            classifier_input = self.pool_net(final_h.squeeze(0), seq_start_end, traj[0])
        return self.real_classifier(classifier_input)
