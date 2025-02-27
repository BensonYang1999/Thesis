import argparse
import logging
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets.dataset_TSR import ContinuousEdgeLineDatasetMask, ContinuousEdgeLineDatasetMaskFinetune
from datasets.dataset_TSR import ContinuousEdgeLineDatasetMask_video, EdgeLineDataset_v2
from src.TSR_trainer import TrainerConfig, TrainerForContinuousEdgeLine, TrainerForEdgeLineFinetune
from src.TSR_trainer import TrainerForContinuousEdgeLine_video, TrainerForContinuousEdgeLine_plus, TrainerForContinuousStruct_video
from src.models.TSR_model import EdgeLineGPT256RelBCE, EdgeLineGPTConfig
from src.models.TSR_model import EdgeLineGPT256RelBCE_video, StructGPT256RelBCE_video, EdgeLine_CNN
from src.utils import set_seed

# torch.cuda.set_per_process_memory_fraction(0.8, 1)
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

def main_worker(rank, opts):
    set_seed(42)
    # gpu = torch.device("cuda")
    gpu = torch.device("cuda", rank)
    torch.cuda.set_device(rank)
    if opts.DDP:
        dist.init_process_group(backend='nccl', init_method='env://', 
                                world_size=opts.world_size, rank=rank)
    torch.backends.cudnn.benchmark = True
    if rank == 0:
        os.makedirs(os.path.dirname(opts.ckpt_path), exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(stream=sys.stdout)
    logger.addHandler(sh)
    logger.propagate = False
    fh = logging.FileHandler(os.path.join(opts.ckpt_path, 'log.txt'))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # Define the model
    model_config = EdgeLineGPTConfig(embd_pdrop=0.0, resid_pdrop=0.0, n_embd=opts.n_embd, block_size=32,
                                     attn_pdrop=0.0, n_layer=opts.n_layer, n_head=opts.n_head, ref_frame_num=opts.ref_frame_num)
    if opts.exp:
        IGPT_model = StructGPT256RelBCE_video(model_config, opts, device=gpu)
    elif opts.cnn:
        IGPT_model = EdgeLine_CNN()
    else:
        IGPT_model = EdgeLineGPT256RelBCE_video(model_config, opts, device=gpu)

    if opts.DDP:
        IGPT_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(IGPT_model.cuda())
        IGPT_model = torch.nn.parallel.DistributedDataParallel(IGPT_model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    if rank == 0:
        num_params = sum(p.numel() for p in IGPT_model.parameters() if p.requires_grad)
        print('Number of parameters: %d' % num_params)

    # Define the dataset
    if not opts.MaP:
        # train_dataset = ContinuousEdgeLineDatasetMask_video(opts, sample=opts.ref_frame_num, size=(opts.image_w,opts.image_h), split='train', name=opts.dataset_name, root=opts.dataset_root)
        # test_dataset = ContinuousEdgeLineDatasetMask_video(opts, sample=opts.ref_frame_num, size=(opts.image_w,opts.image_h), split='valid', name=opts.dataset_name, root=opts.dataset_root)
        train_dataset = EdgeLineDataset_v2(opts, sample=opts.ref_frame_num, size=(opts.image_w,opts.image_h), split='train', name=opts.dataset_name, root=opts.dataset_root)
        test_dataset = EdgeLineDataset_v2(opts, sample=opts.ref_frame_num, size=(opts.image_w,opts.image_h), split='valid', name=opts.dataset_name, root=opts.dataset_root)

    else:  # TODO
        train_dataset = ContinuousEdgeLineDatasetMaskFinetune(opts.data_path, mask_path=opts.mask_path, is_train=True,
                                                              mask_rates=opts.mask_rates, image_size=opts.image_size,
                                                              line_path=opts.train_line_path)
        test_dataset = ContinuousEdgeLineDatasetMaskFinetune(opts.validation_path, test_mask_path=opts.valid_mask_path,
                                                             is_train=False, image_size=opts.image_size,
                                                             line_path=opts.val_line_path)

    # iterations_per_epoch = len(train_dataset.image_id_list) // opts.batch_size
    iterations_per_epoch = len(train_dataset) // opts.batch_size   # video
    train_epochs = opts.train_epoch
    train_config = TrainerConfig(max_epochs=train_epochs, batch_size=opts.batch_size,
                                 learning_rate=opts.lr, betas=(0.9, 0.95),
                                 weight_decay=0, lr_decay=True,
                                 warmup_iterations=1500,
                                 final_iterations=train_epochs * iterations_per_epoch / opts.world_size,
                                 ckpt_path=opts.ckpt_path, num_workers=8, GPU_ids=opts.GPU_ids,
                                 world_size=opts.world_size,
                                 AMP=opts.AMP, print_freq=opts.print_freq)

    if not opts.MaP:
        # trainer = TrainerForContinuousEdgeLine(IGPT_model, train_dataset, test_dataset, train_config, gpu, rank,
        #                                        iterations_per_epoch, logger=logger)
        if opts.exp:
            # trainer = TrainerForContinuousEdgeLine_plus(IGPT_model, train_dataset, test_dataset, train_config, gpu, rank,
            #                                     iterations_per_epoch, logger=logger)
            trainer = TrainerForContinuousStruct_video(IGPT_model, train_dataset, test_dataset, train_config, gpu, rank,
                                                iterations_per_epoch, logger=logger)
        else:
            trainer = TrainerForContinuousEdgeLine_video(IGPT_model, train_dataset, test_dataset, train_config, gpu, rank,
                                                         iterations_per_epoch, logger=logger)
    else:  # TODO
        trainer = TrainerForEdgeLineFinetune(IGPT_model, train_dataset, test_dataset, train_config, gpu, rank,
                                             iterations_per_epoch, logger=logger)
    loaded_ckpt = trainer.load_checkpoint(opts.resume_ckpt)
    trainer.train(loaded_ckpt)
    print("Finish the training ...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='places2_continous_edgeline', help='The name of this exp')
    parser.add_argument('--GPU_ids', type=str, default='0, 1')
    parser.add_argument('--ckpt_path', type=str, default='./ckpt')
    parser.add_argument('--dataset_root', type=str, default="./datasets", help='Indicate where is the root of training set folder')
    parser.add_argument('--dataset_name', type=str, default="YouTubeVOS", help='Indicate which training set')
    # parser.add_argument('--data_path', type=str, default=None, help='Indicate where is the training set')
    # parser.add_argument('--train_line_path', type=str, default=None, help='Indicate where is the wireframes of training set')
    # parser.add_argument('--mask_path', type=list, default=['data_list/irregular_mask_list.txt', 'data_list/coco_mask_list.txt'])
    # parser.add_argument('--mask_rates', type=list, default=[1., 0., 0.],
    #                     help='irregular rate, coco rate, addition rate')  # for video
    # parser.add_argument('--mask_rates', type=list, default=[0.4, 0.8, 1.0],
    #                     help='irregular rate, coco rate, addition rate')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--train_epoch', type=int, default=100, help='how many epochs')
    parser.add_argument('--print_freq', type=int, default=100, help='While training, the freq of printing log')
    parser.add_argument('--validation_path', type=str, default=None, help='where is the validation set of ImageNet')
    # parser.add_argument('--val_line_path', type=str, default=None, help='Indicate where is the wireframes of val set')
    # parser.add_argument('--valid_mask_path', type=str, default=None)
    parser.add_argument('--image_w', type=int, default=432, help='input frame width')
    parser.add_argument('--image_h', type=int, default=240, help='input frame height')
    parser.add_argument('--image_size', type=int, default=256, help='input sequence length = image_size*image_size')
    parser.add_argument('--resume_ckpt', type=str, default='latest.pth', help='start from where, the default is latest')
    # Mask and predict finetune
    parser.add_argument('--MaP', action='store_true', help='set True when finetune for mask and predict')
    # Define the size of transformer
    parser.add_argument('--ref_frame_num', type=int, default=5)
    parser.add_argument('--n_layer', type=int, default=16)
    parser.add_argument('--n_embd', type=int, default=256)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--lr', type=float, default=4.24e-4)
    # AMP
    parser.add_argument('--nodes', type=int, default=1, help='how many machines')
    parser.add_argument('--gpus', type=int, default=1, help='how many GPUs in one node')
    parser.add_argument('--AMP', action='store_true', help='Automatic Mixed Precision')
    parser.add_argument('--local_rank', type=int, default=-1, help='the id of this machine')
    parser.add_argument('--DDP', action='store_true', help='DDP')
    # for video
    # parser.add_argument('--loss_item', type=str, nargs='+', default=["hole", "valid"], help='the id of this machine')
    parser.add_argument('--loss_hole_valid_weight', type=float, nargs='+', default=[0.8, 0.2], help='the weight for computing the hole/valid part ')
    parser.add_argument('--loss_edge_line_weight', type=float, nargs='+', default=[1.0, 0.0], help='the weight for computing the edge/line part ')
    # add the choice to decide the loss function with l1 or mse or binary cross entropy with choice
    parser.add_argument('--loss_choice', type=str, default="bce", help='the choice of loss function: l1, mse, bce')
    parser.add_argument('--edge_gaussian', type=int, default=0, help='the sigma of gaussian kernel for edge')

    # experimental model
    parser.add_argument('--exp', action='store_true', help='whether to use the experimental model')
    parser.add_argument('--cnn', action='store_true', help='whether to use the cnn model')

    # show th loss_coice and edge_gaussian and the loss_hole_valid_weight and loss_edge_line_weight
    print("The loss_choice is: ", parser.parse_args().loss_choice)
    print("The edge_gaussian is: ", parser.parse_args().edge_gaussian)
    print("The loss_hole_valid_weight is: ", parser.parse_args().loss_hole_valid_weight)
    print("The loss_edge_line_weight is: ", parser.parse_args().loss_edge_line_weight)

    opts = parser.parse_args()
    opts.ckpt_path = os.path.join(opts.ckpt_path, opts.name)
    opts.resume_ckpt = os.path.join(opts.ckpt_path, opts.resume_ckpt)
    os.makedirs(opts.ckpt_path, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.GPU_ids

    if opts.DDP:
        opts.world_size = opts.nodes * opts.gpus
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12380'
    else:
        opts.world_size = 1
    # rank = 0
    # torch.cuda.set_device(rank)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # main_worker(rank, opts, is_distributed)
    mp.spawn(main_worker, nprocs=opts.world_size, args=(opts,))
