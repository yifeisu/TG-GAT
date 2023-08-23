import os
import warnings
import json
import time
import numpy as np
from collections import defaultdict

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 先加入绝对路径，否则会报错，注意__file__表示的是当前执行文件的路径
from utils.misc import set_random_seed
from utils.logger import write_to_record_file, print_progress, timeSince
from utils.distributed import init_distributed, is_default_gpu, init_distributed_mode

from agent import NavCMTAgent
from env_mapdataset import ANDHNavBatchMap, SimulatorAVDN, simple_cat_collate
from parser import parse_args


def build_dataset(args, rank=0, is_test=False):
    dataset_class = ANDHNavBatchMap

    train_env = dataset_class(
        args.train_anno_dir,
        args.train_dataset_dir,
        ['train'],
        seed=args.seed + rank, )

    train_full_traj_env = None
    val_env_names = ['val_seen', 'val_unseen', ]  # 'test_unseen'
    if args.submit:
        val_env_names.append('test_unseen')

    val_envs = {}
    for split in val_env_names:
        val_env = dataset_class(
            args.val_anno_dir,
            args.val_dataset_dir,
            [split],
            seed=args.seed, )
        val_envs[split] = val_env

    val_full_traj_envs = None

    return train_env, train_full_traj_env, val_envs, val_full_traj_envs


def train(args, train_env, train_full_traj_env, val_envs, val_full_traj_envs, aug_env=None, rank=-1):
    default_gpu = is_default_gpu(args)

    if default_gpu:
        with open(os.path.join(args.log_dir, 'training_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        writer = SummaryWriter(log_dir=args.log_dir)
        record_file = os.path.join(args.log_dir, 'train.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    agent_class = NavCMTAgent
    agent = agent_class(args, rank=rank)

    # -------------------------------------------------------------------------------------- #
    # # resume file
    # -------------------------------------------------------------------------------------- #
    start_iter = 0
    if args.resume_file is not None:
        start_iter = agent.load(os.path.join(args.resume_file))
        if default_gpu:
            write_to_record_file("\nLOAD the model from {}, iteration {}".format(args.resume_file, start_iter), record_file)

    # -------------------------------------------------------------------------------------- #
    # # first evaluation
    # -------------------------------------------------------------------------------------- #

    # -------------------------------------------------------------------------------------- #
    # # Start Training
    # -------------------------------------------------------------------------------------- #
    start = time.time()
    if default_gpu:
        write_to_record_file('\nListener training starts, start iteration: %s' % str(start_iter), record_file)

    best_val = {'val_unseen': {"spl": 0., "state": ""}, 'val_unseen_full_traj': {"spl": 0., "state": ""}}
    interval = int(len(train_env) / args.batch_size) * args.log_every

    # -------------------------------------------------------------------------------------- #
    # build dataloaders;
    # -------------------------------------------------------------------------------------- #
    if args.world_size > 1:
        sampler = DistributedSampler(train_env)
        train_loader = DataLoader(
            train_env, batch_size=args.batch_size, shuffle=False, drop_last=False,
            num_workers=4, collate_fn=simple_cat_collate, sampler=sampler)
    else:
        train_loader = DataLoader(
            train_env, batch_size=args.batch_size,
            shuffle=False, drop_last=False, num_workers=2, collate_fn=simple_cat_collate)

    zero_start_iter = 0
    epoch = -1
    for idx in range(start_iter, start_iter + args.iters, interval):
        agent.logs = defaultdict(list)
        iter = idx + interval
        epoch += 1
        # -------------------------------------------------------------------------------------- #
        # # build dataset and dataloader;
        # -------------------------------------------------------------------------------------- #
        if args.world_size > 1:
            train_loader.sampler.set_epoch(epoch)

        agent.ds = train_env
        # Train for 1 epochs before evaluate again
        agent.train(train_loader, args.log_every, feedback=args.feedback, nss_w_weighting=1)

        IL_loss = sum(agent.logs['IL_loss']) / max(len(agent.logs['IL_loss']), 1)
        if default_gpu:
            writer.add_scalar("loss/IL_loss", IL_loss, iter)
            write_to_record_file("\nIL_loss %.4f" % IL_loss, record_file)

        # -------------------------------------------------------------------------------------- #
        # # Run validation on single gpu;
        # -------------------------------------------------------------------------------------- #
        if default_gpu:
            loss_str = "iter {}".format(iter)
            agent.save(iter, os.path.join(args.ckpt_dir, "latest_dict_" + str(iter)))

            agent_class_eval = NavCMTAgent
            agent_eval = agent_class_eval(args, allow_ngpus=False, rank=rank)
            print("Loaded the listener model at iter %d from %s" %
                  (agent_eval.load(os.path.join(args.ckpt_dir, "latest_dict_" + str(iter))), os.path.join(args.ckpt_dir, "latest_dict_" + str(iter))))
            for env_name, env in val_envs.items():
                agent_eval.ds = env
                # -------------------------------------------------------------------------------------- #
                # build dataloader;
                # -------------------------------------------------------------------------------------- #
                loader = torch.utils.data.DataLoader(env, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=2, collate_fn=simple_cat_collate)

                # Get validation distance from goal under test evaluation conditions
                agent_eval.test(loader, feedback='student')
                pred_results = agent_eval.get_results()

                score_summary, result = env.eval_metrics(pred_results)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                    writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], iter)
                if env_name in best_val:
                    if score_summary['spl'] >= best_val[env_name]['spl']:
                        best_val[env_name]['spl'] = score_summary['spl']
                        best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                        agent_eval.save(iter, os.path.join(args.ckpt_dir, "best_%s" % (env_name)))

            # evaluate human attention
            for env_name, env in val_envs.items():
                env_name += '_human_att'
                agent_eval.ds = env
                # -------------------------------------------------------------------------------------- #
                # build dataloader;
                # -------------------------------------------------------------------------------------- #
                loader = torch.utils.data.DataLoader(env, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=2, collate_fn=simple_cat_collate)

                agent_eval.test(loader, feedback='teacher')  # use teacher mode to evaluate human attention pred
                preds = agent_eval.get_results()

                score_summary, _ = env.eval_metrics(preds, human_att_eval=True)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                    writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], iter)

            write_to_record_file(('%s (%d %d%%) %s' % (timeSince(start, float(iter) / args.iters), iter, float(iter) / args.iters * 100, loss_str)), record_file)
            write_to_record_file("BEST RESULT TILL NOW", record_file)
            for env_name in best_val:
                write_to_record_file(env_name + ' | ' + best_val[env_name]['state'], record_file)
            zero_start_iter += interval


def valid(args, val_envs, val_full_traj_envs, rank=-1):
    # default_gpu = is_default_gpu(args)

    agent_class = NavCMTAgent
    agent = agent_class(args, allow_ngpus=False, rank=rank)
    if args.resume_file is not None:
        print("Loaded the listener model at iter %d from %s" % (agent.load(args.resume_file), args.resume_file))

    with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
        json.dump(vars(args), outf, indent=4)
    record_file = os.path.join(args.log_dir, 'valid.txt')
    write_to_record_file(str(args) + '\n\n', record_file)
    loss_str = "iter {}".format(iter)

    for env_name, env in val_envs.items():
        agent.ds = env
        # -------------------------------------------------------------------------------------- #
        # build dataloader;
        # -------------------------------------------------------------------------------------- #
        loader = torch.utils.data.DataLoader(env, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=2, collate_fn=simple_cat_collate)

        # Get validation distance from goal under test evaluation conditions
        agent.test(loader, feedback='student', env_name=env_name)
        pred_results = agent.get_results()

        if 'test_unseen' in env_name:
            print('inference_result on test is generated.')
            np.save('./output_test_result.npy', pred_results)
        else:
            score_summary, result = env.eval_metrics(pred_results)
            loss_str += "Env name: %s" % env_name
            for metric, val in score_summary.items():
                loss_str += ', %s: %.2f' % (metric, val)
            write_to_record_file(loss_str + '\n', record_file)


def main():
    args = parse_args()

    if args.world_size > 1:
        rank = init_distributed_mode(args)
        # torch.cuda.set_device(args.local_rank)
        # torch.distributed.barrier()
    else:
        rank = 0

    set_random_seed(args.seed + rank)
    train_env, train_full_traj_env, val_envs, val_full_traj_envs = build_dataset(args, rank=rank)

    if not args.inference:
        train(args, train_env, train_full_traj_env, val_envs, val_full_traj_envs, rank=rank)
    else:
        valid(args, val_envs, val_full_traj_envs, rank=rank)


if __name__ == '__main__':
    main()
