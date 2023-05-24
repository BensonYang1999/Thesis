import argparse
import os
import random
from shutil import copyfile

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import signal
import sys

from src.FTR_trainer import ZITS_video, LaMa
from src.config import Config


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    cleanup()
    sys.exit(0)

def main_worker(gpu, args):
    try:
        rank = args.node_rank * args.gpus + gpu
        torch.cuda.set_device(gpu)

        if args.DDP:
            dist.init_process_group(backend='nccl',
                                    init_method='env://',
                                    world_size=args.world_size,
                                    rank=rank,
                                    group_name='mtorch')

        # load config file
        config = Config(args.config_path, args.model_name)
        config.MODE = 1
        config.nodes = args.nodes
        config.gpus = args.gpus
        config.GPU_ids = args.GPU_ids
        config.DDP = args.DDP
        if config.DDP:
            config.world_size = args.world_size
        else:
            config.world_size = 1

        torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
        cv2.setNumThreads(0)

        # initialize random seed
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed_all(config.SEED)
        np.random.seed(config.SEED)
        random.seed(config.SEED)

        # build the model and initialize
        if args.lama:
            model = LaMa(config, gpu, rank)
        else:
            model = ZITS_video(args, config, gpu, rank)

        # model training
        if rank == 0:
            config.print()
            print('\nstart training...\n')
        model.train()
    except Exception as e:
        print('Exception in main_worker:', e)
    finally:
        cleanup()

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default="./ckpt",
                        help='model checkpoints path (default: ./checkpoints)')
    parser.add_argument('--config_file', type=str, default='./config_list/config_ZITS_video.yml',
                        help='The config file of each experiment ')
    parser.add_argument('--nodes', type=int, default=1, help='how many machines')
    parser.add_argument('--gpus', type=int, default=2, help='how many GPUs in one node')
    parser.add_argument('--GPU_ids', type=str, default='0,1')
    parser.add_argument('--node_rank', type=int, default=0, help='the id of this machine')
    parser.add_argument('--DDP', action='store_true', help='DDP')
    parser.add_argument('--lama', action='store_true', help='train the lama first')
    # For Video version
    parser.add_argument('--model_name', type=str, default='ZITS_video', help='the name of this model')
    parser.add_argument('--ref_frame_num', type=int, default=5)
    parser.add_argument('--n_layer', type=int, default=16)
    parser.add_argument('--n_embd', type=int, default=256)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--lr', type=float, default=4.24e-4)
    parser.add_argument('--AMP', action='store_true', help='Automatic Mixed Precision')
    parser.add_argument('--loss_hole_valid_weight', type=float, nargs='+', default=[0.8, 0.2], help='the weight for computing the hole/valid part ')
    parser.add_argument('--loss_edge_line_weight', type=float, nargs='+', default=[1.0, 1.0], help='the weight for computing the edge/line part ')
    parser.add_argument('--loss_choice', type=str, default="bce", help='the choice of loss function: l1, mse, bce')
    parser.add_argument('--edge_gaussian', type=int, default=0, help='the sigma of gaussian kernel for edge')
    parser.add_argument('--input_size', type=tuple, default=(240,432))
    parser.add_argument('--dataset', type=str, default="youtubevos")
    parser.add_argument("--neighbor_stride", type=int, default=1)
    args = parser.parse_args()
    
    args.path = os.path.join(args.path, args.model_name)
    config_path = os.path.join(args.path, 'config.yml')

    # create checkpoints path if does't exist
    os.makedirs(args.path, exist_ok=True)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile(args.config_file, config_path)  ## Training, always copy

    args.config_path = config_path

    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_ids
    if args.DDP:
        args.world_size = args.nodes * args.gpus
        os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '22323'
        os.environ['MASTER_PORT'] = '50580'
    else:
        args.world_size = 1

    mp.spawn(main_worker, nprocs=args.world_size, args=(args,))

