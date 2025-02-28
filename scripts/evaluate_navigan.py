import numpy as np
import sys

import argparse
import os
import torch
from attrdict import AttrDict
from pathlib import Path

from scripts.evaluate_model import evaluate_helper
from scripts.goal import evaluate_model_trajectories, count_suitable_target_agents_in_dataset, seek_goal_simulated_data, \
    create_goal_state
from scripts.model_loaders import get_combined_generator
from sgan.data.loader import data_loader
from sgan.data.trajectories import read_file
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path, plot_trajectories, plot_losses, save_trajectory_plot

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=1, type=int)
parser.add_argument('--dset_type', default='test', type=str)
_DEVICE_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(args, loader, dset, generator, num_samples, dset_path):
    ade_outer, fde_outer = [], []
    total_traj = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            # batch = [tensor for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch
            # print(obs_traj[::,0].T)
            # print(obs_traj_rel[::,0].T)
            # sys.exit(0)
            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)
            ota = obs_traj.numpy()

            for _ in range(num_samples):
                goal_state = create_goal_state(dpath=dset_path, pred_len=generator.goal.pred_len,
                                               goal_obs_traj=obs_traj[::, [index[0] for index in seq_start_end]])
                pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, goal_state, goal_aggro=0.5)
                pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

                ade.append(displacement_error(pred_traj_fake, pred_traj_gt, mode='raw'))
                fde.append(final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'))
            #     # plot the lowest fde trajectory of the batch
            #     # print(f'len FDE list {len(fde)}, Tensor shape {fde[0].shape}')
            #     fde_unpacked = [torch.argmax(t).item() for t in fde]
            #     # print(*[t[0:10] for t in fde])
            #     min_fde = fde_unpacked.index(min(fde_unpacked))
            #     # print(f'Index of traj with smallest FDE: {min_fde}')
            #
            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)

        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)

        return ade, fde


def write(text, model_name):
    with open(f'/home/david/Pictures/plots/goal_test/{model_name}/eval_stats.txt', 'a') as f:
        f.write(text)


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    for path in paths:
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        generator = get_combined_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])
        dpath = get_dset_path(_args.dataset_name, args.dset_type)
        dset, loader = data_loader(_args, dpath)
        # plot_losses(checkpoint, train=True)
        # plot_losses(checkpoint, train=False)

        # ade, fde = evaluate(_args, loader, dset, generator, args.num_samples, dpath)
        # print(f'Model: {os.path.basename(path)}, Dataset: {_args.dataset_name}, Pred Len: {_args.pred_len},'
        #       f' ADE: {ade:.2f}, FDE: {fde:.2f}')

        # count_suitable_target_agents_in_dataset(dpath, loader, generator)
        goal_aggro = float(input('Goal aggro: '))
        iters = 60
        suc, fail, sbreach, seqs = evaluate_model_trajectories(dpath, loader, generator,
                                                               model_name=f'{os.path.basename(path)}', iters=iters,
                                                               goal_aggro=goal_aggro)
        write(
            f'\nModel: {os.path.basename(path)}, Dataset: {_args.dataset_name}, Pred Len: {_args.pred_len},'
            f' Goal Aggro: {goal_aggro} No. of iters {iters} No. of seqs: '
            f'{seqs} \nSuccesses: {suc} Fails: {fail} Social Breaches: {sbreach}\n\n', model_name=f'{Path(path).with_suffix("").name}')
        # seek_goal_simulated_data(generator, x=11, y=10, arrival_tol=2.2)


if __name__ == '__main__':
    args = parser.parse_args()
    # args.model_path = '/home/david/data/sgan-models/checkpoint_intention_with_model.pt'
    main(args)
