import os

from src.losses.adversarial import NonSaturatingWithR1, NonSaturatingWithR1_video_2D
from src.losses.feature_matching import masked_l1_loss, feature_matching_loss
from src.losses.perceptual import ResNetPL, ResNetPL_video
from src.losses.clip import *
from src.losses.smoothness import *
from src.models.LaMa import *
from src.models.TSR_model import *
from src.models.upsample import StructureUpsampling
from src.utils import get_lr_schedule_with_warmup, torch_init_model


def make_optimizer(parameters, kind='adamw', **kwargs):
    if kind == 'adam':
        optimizer_class = torch.optim.Adam
    elif kind == 'adamw':
        optimizer_class = torch.optim.AdamW
    elif kind == 'sgd':
        optimizer_class = torch.optim.SGD
    else:
        raise ValueError(f'Unknown optimizer kind {kind}')
    return optimizer_class(parameters, **kwargs)


def set_requires_grad(module, value):
    for param in module.parameters():
        param.requires_grad = value


def add_prefix_to_keys(dct, prefix):
    return {prefix + k: v for k, v in dct.items()}


class LaMaBaseInpaintingTrainingModule(nn.Module):
    def __init__(self, config, gpu, name, rank, *args, test=False, **kwargs):
        super().__init__(*args, **kwargs)
        print('BaseInpaintingTrainingModule init called')
        self.global_rank = rank
        self.config = config
        self.iteration = 0
        self.name = name
        self.test = test
        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

        self.generator = LaMa_model().cuda(gpu)
        self.best = None

        if not test:
            self.discriminator = NLayerDiscriminator(**self.config.discriminator).cuda(gpu)
            self.adversarial_loss = NonSaturatingWithR1(**self.config.losses['adversarial'])
            self.generator_average = None
            self.last_generator_averaging_step = -1

            if self.config.losses.get("l1", {"weight_known": 0})['weight_known'] > 0:
                self.loss_l1 = nn.L1Loss(reduction='none')

            if self.config.losses.get("mse", {"weight": 0})['weight'] > 0:
                self.loss_mse = nn.MSELoss(reduction='none')

            assert self.config.losses['perceptual']['weight'] == 0

            if self.config.losses.get("resnet_pl", {"weight": 0})['weight'] > 0:
                self.loss_resnet_pl = ResNetPL(**self.config.losses['resnet_pl'])
            else:
                self.loss_resnet_pl = None
            self.gen_optimizer, self.dis_optimizer = self.configure_optimizers()
        if self.config.AMP:  # use AMP
            self.scaler = torch.cuda.amp.GradScaler()

        self.load()
        if self.config.DDP:
            # import apex
            # self.generator = apex.parallel.convert_syncbn_model(self.generator)
            # self.discriminator = apex.parallel.convert_syncbn_model(self.discriminator)
            # self.generator = apex.parallel.DistributedDataParallel(self.generator)
            # self.discriminator = apex.parallel.DistributedDataParallel(self.discriminator)
            self.generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.generator)
            self.discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.discriminator)
            self.generator = torch.nn.parallel.DistributedDataParallel(self.generator)
            self.discriminator = torch.nn.parallel.DistributedDataParallel(self.discriminator)

    def load(self):
        if self.test:
            self.gen_weights_path = os.path.join(self.config.PATH, self.name + '_best_gen.pth')
            print('Loading %s generator...' % self.name)
            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])

        if not self.test and os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.gen_optimizer.load_state_dict(data['optimizer'])
            self.iteration = data['iteration']
            if self.iteration > 0:
                gen_weights_path = os.path.join(self.config.PATH, self.name + '_best_gen.pth')
                if torch.cuda.is_available():
                    data = torch.load(gen_weights_path)
                else:
                    data = torch.load(gen_weights_path, map_location=lambda storage, loc: storage)
                self.best = data['best_fid']
                print('Loading best psnr...')

        else:
            print('Warnning: There is no previous optimizer found. An initialized optimizer will be used.')

        # load discriminator only when training
        if not self.test and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.dis_weights_path)
            else:
                data = torch.load(self.dis_weights_path, map_location=lambda storage, loc: storage)
            self.dis_optimizer.load_state_dict(data['optimizer'])
            self.discriminator.load_state_dict(data['discriminator'])
        else:
            print('Warnning: There is no previous optimizer found. An initialized optimizer will be used.')

    def save(self):
        print('\nsaving %s...\n' % self.name)
        raw_model = self.generator.module if hasattr(self.generator, "module") else self.generator
        torch.save({
            'iteration': self.iteration,
            'optimizer': self.gen_optimizer.state_dict(),
            'generator': raw_model.state_dict()
        }, self.gen_weights_path)
        raw_model = self.discriminator.module if hasattr(self.discriminator, "module") else self.discriminator
        torch.save({
            'optimizer': self.dis_optimizer.state_dict(),
            'discriminator': raw_model.state_dict()
        }, self.dis_weights_path)

    def configure_optimizers(self):
        discriminator_params = list(self.discriminator.parameters())
        return [
            make_optimizer(self.generator.parameters(), **self.config.optimizers['generator']),
            make_optimizer(discriminator_params, **self.config.optimizers['discriminator'])
        ]


class BaseInpaintingTrainingModule(nn.Module):
    def __init__(self, config, gpu, name, rank, *args, test=False, **kwargs):
        super().__init__(*args, **kwargs)
        print('BaseInpaintingTrainingModule init called')
        self.global_rank = rank
        self.config = config
        self.iteration = 0
        self.name = name
        self.test = test
        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

        self.str_encoder = StructureEncoder(config).cuda(gpu)
        self.generator = ReZeroFFC(config).cuda(gpu)
        self.best = None

        print('Loading %s StructureUpsampling...' % self.name)
        self.structure_upsample = StructureUpsampling()
        data = torch.load(config.structure_upsample_path, map_location='cpu')
        self.structure_upsample.load_state_dict(data['model'])
        self.structure_upsample = self.structure_upsample.cuda(gpu).eval()
        print("Loading trained transformer...")
        model_config = EdgeLineGPTConfig(embd_pdrop=0.0, resid_pdrop=0.0, n_embd=256, block_size=32,
                                         attn_pdrop=0.0, n_layer=16, n_head=8)
        self.transformer = EdgeLineGPT256RelBCE(model_config)
        checkpoint = torch.load(config.transformer_ckpt_path, map_location='cpu')
        if config.transformer_ckpt_path.endswith('.pt'):
            self.transformer.load_state_dict(checkpoint)
        else:
            self.transformer.load_state_dict(checkpoint['model'])
        self.transformer.cuda(gpu).eval()
        self.transformer.half()

        if not test:
            self.discriminator = NLayerDiscriminator(**self.config.discriminator).cuda(gpu)
            self.adversarial_loss = NonSaturatingWithR1(**self.config.losses['adversarial'])
            self.generator_average = None
            self.last_generator_averaging_step = -1

            if self.config.losses.get("l1", {"weight_known": 0})['weight_known'] > 0:
                self.loss_l1 = nn.L1Loss(reduction='none')

            if self.config.losses.get("mse", {"weight": 0})['weight'] > 0:
                self.loss_mse = nn.MSELoss(reduction='none')

            assert self.config.losses['perceptual']['weight'] == 0

            if self.config.losses.get("resnet_pl", {"weight": 0})['weight'] > 0:
                self.loss_resnet_pl = ResNetPL(**self.config.losses['resnet_pl'])
            else:
                self.loss_resnet_pl = None
            self.gen_optimizer, self.dis_optimizer = self.configure_optimizers()
            self.str_optimizer = torch.optim.Adam(self.str_encoder.parameters(), lr=config.optimizers['generator']['lr'])
        if self.config.AMP:  # use AMP
            self.scaler = torch.cuda.amp.GradScaler()
        if not test:
            self.load_rezero()  # load pretrain model
        self.load()  # reload for restore

        # reset lr
        if not test:
            for group in self.gen_optimizer.param_groups:
                group['lr'] = config.optimizers['generator']['lr']
                group['initial_lr'] = config.optimizers['generator']['lr']
            for group in self.dis_optimizer.param_groups:
                group['lr'] = config.optimizers['discriminator']['lr']
                group['initial_lr'] = config.optimizers['discriminator']['lr']

        if self.config.DDP and not test:
            # import apex
            # self.generator = apex.parallel.convert_syncbn_model(self.generator)
            # self.discriminator = apex.parallel.convert_syncbn_model(self.discriminator)
            # self.generator = apex.parallel.DistributedDataParallel(self.generator)
            # self.discriminator = apex.parallel.DistributedDataParallel(self.discriminator)
            self.generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.generator)
            self.discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.discriminator)
            self.generator = torch.nn.parallel.DistributedDataParallel(self.generator)
            self.discriminator = torch.nn.parallel.DistributedDataParallel(self.discriminator)

        if self.config.optimizers['decay_steps'] is not None and self.config.optimizers['decay_steps'] > 0 and not test:
            self.g_scheduler = torch.optim.lr_scheduler.StepLR(self.gen_optimizer, config.optimizers['decay_steps'],
                                                               gamma=config.optimizers['decay_rate'])
            self.d_scheduler = torch.optim.lr_scheduler.StepLR(self.dis_optimizer, config.optimizers['decay_steps'],
                                                               gamma=config.optimizers['decay_rate'])
            self.str_scheduler = get_lr_schedule_with_warmup(self.str_optimizer,
                                                             num_warmup_steps=config.optimizers['warmup_steps'],
                                                             milestone_step=config.optimizers['decay_steps'],
                                                             gamma=config.optimizers['decay_rate'])
            if self.iteration - self.config.START_ITERS > 1:
                for _ in range(self.iteration - self.config.START_ITERS):
                    self.g_scheduler.step()
                    self.d_scheduler.step()
                    self.str_scheduler.step()
        else:
            self.g_scheduler = None
            self.d_scheduler = None
            self.str_scheduler = None

    def load_rezero(self):
        if os.path.exists(self.config.gen_weights_path0):
            print('Loading %s generator...' % self.name)
            data = torch.load(self.config.gen_weights_path0, map_location='cpu')
            torch_init_model(self.generator, data, 'generator', rank=self.global_rank)
            self.gen_optimizer.load_state_dict(data['optimizer'])
            self.iteration = data['iteration']
        else:
            print('Warnning: There is no previous optimizer found. An initialized optimizer will be used.')

        # load discriminator only when training
        if (self.config.MODE == 1 or self.config.score) and os.path.exists(self.config.dis_weights_path0):
            print('Loading %s discriminator...' % self.name)
            data = torch.load(self.config.dis_weights_path0, map_location='cpu')
            torch_init_model(self.discriminator, data, 'discriminator', rank=self.global_rank)
            self.dis_optimizer.load_state_dict(data['optimizer'])
        else:
            print('Warnning: There is no previous optimizer found. An initialized optimizer will be used.')

    def load(self):
        if self.test:
            self.gen_weights_path = os.path.join(self.config.PATH, self.name + '_best_gen.pth')
            print('Loading %s generator...' % self.name)
            data = torch.load(self.gen_weights_path, map_location='cpu')
            self.generator.load_state_dict(data['generator'])
            self.str_encoder.load_state_dict(data['str_encoder'])
        if not self.test and os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)
            data = torch.load(self.gen_weights_path, map_location='cpu')
            self.generator.load_state_dict(data['generator'])
            self.str_encoder.load_state_dict(data['str_encoder'])
            self.gen_optimizer.load_state_dict(data['optimizer'])
            self.str_optimizer.load_state_dict(data['str_opt'])
            self.iteration = data['iteration']
            if self.iteration > 0:
                gen_weights_path = os.path.join(self.config.PATH, self.name + '_best_gen_HR.pth')
                data = torch.load(gen_weights_path, map_location='cpu')
                self.best = data['best_fid']
                print('Loading best fid...')

        else:
            print('Warnning: There is no previous optimizer found. An initialized optimizer will be used.')

        # load discriminator only when training
        if not self.test and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)
            data = torch.load(self.dis_weights_path, map_location='cpu')
            self.dis_optimizer.load_state_dict(data['optimizer'])
            self.discriminator.load_state_dict(data['discriminator'])
        else:
            print('Warnning: There is no previous optimizer found. An initialized optimizer will be used.')

    def save(self):
        print('\nsaving %s...\n' % self.name)
        raw_model = self.generator.module if hasattr(self.generator, "module") else self.generator
        raw_encoder = self.str_encoder.module if hasattr(self.str_encoder, "module") else self.str_encoder
        torch.save({
            'iteration': self.iteration,
            'optimizer': self.gen_optimizer.state_dict(),
            'str_opt': self.str_optimizer.state_dict(),
            'str_encoder': raw_encoder.state_dict(),
            'generator': raw_model.state_dict()
        }, self.gen_weights_path)
        raw_model = self.discriminator.module if hasattr(self.discriminator, "module") else self.discriminator
        torch.save({
            'optimizer': self.dis_optimizer.state_dict(),
            'discriminator': raw_model.state_dict()
        }, self.dis_weights_path)

    def configure_optimizers(self):
        discriminator_params = list(self.discriminator.parameters())
        return [
            make_optimizer(self.generator.parameters(), **self.config.optimizers['generator']),
            make_optimizer(discriminator_params, **self.config.optimizers['discriminator'])
        ]


class BaseInpaintingTrainingModule_video(nn.Module):
    def __init__(self, config, gpu, name, rank, args, test=False, **kwargs):
        super().__init__()
        print('BaseInpaintingTrainingModule init called')
        self.global_rank = rank
        self.config = config
        self.iteration = 0
        self.name = name
        self.test = test
        if test:
            if args.useGT_LE:
                self.gen_weights_path = os.path.join(config.PATH, name + '_best_gen_HR.pth')
                self.dis_weights_path = os.path.join(config.PATH, name + '_best_dis_HR.pth')
            else:
                self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
                self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')
        else:
            self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
            self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

        if config.SFE_structure == "resnet":
            print("Using ResNet as structure encoder")
            self.str_encoder = StructureEncoder_video_2D(config).cuda(gpu) # ResNet version
        else: #  config.SFE_structure == "transformer"
            print("Using Transformer as structure encoder")
            self.str_encoder = StructureEncoder_video_3D(config).cuda(gpu) # Transformer version

        self.generator = ReZeroFFC_video_2D(config).cuda(gpu) # generator
        self.best = None

        print('Loading %s StructureUpsampling...' % self.name)
        self.structure_upsample = StructureUpsampling()
        data = torch.load(config.structure_upsample_path, map_location='cpu')
        self.structure_upsample.load_state_dict(data['model'])
        self.structure_upsample = self.structure_upsample.cuda(gpu).eval()
        print("Loading trained transformer...")
        # model_config = EdgeLineGPTConfig(embd_pdrop=0.0, resid_pdrop=0.0, n_embd=args.n_embd, block_size=32,
        #                              attn_pdrop=0.0, n_layer=args.n_layer, n_head=args.n_head, ref_frame_num=args.ref_frame_num) # video version
        # self.transformer = EdgeLineGPT256RelBCE_video(model_config, args, device=gpu)
        self.transformer = EdgeLine_CNN()
        checkpoint = torch.load(config.transformer_ckpt_path, map_location='cpu')
        if config.transformer_ckpt_path.endswith('.pt'): 
            self.transformer.load_state_dict(checkpoint) # load the line/edge inpainted model
        else:
            self.transformer.load_state_dict(checkpoint['model'])
        self.transformer.cuda(gpu).eval()  # eval mode of line/edge inpainted model
        # self.transformer.half() # half precision  # 9/14 disabled half precision

        if not test:
            # self.discriminator = NLayerDiscriminator_video(**self.config.discriminator).cuda(gpu) 
            self.discriminator = NLayerDiscriminator_video_2D(config).cuda(gpu) 
            # self.discriminator = net.Discriminator(
            #     in_channels=3, use_sigmoid=config['losses']['GAN_LOSS'] != 'hinge').cuda(gpu)  # video version referr to FuseFormer
            self.adversarial_loss = NonSaturatingWithR1_video_2D(**self.config.losses['adversarial'])
            # self.adversarial_loss = AdversarialLoss(type=self.config['losses']['GAN_LOSS']).cuda(gpu) # video version reference to FuseFormer
            self.generator_average = None
            self.last_generator_averaging_step = -1

            if self.config.losses.get("l1", {"weight_known": 0})['weight_known'] > 0:
                self.loss_l1 = nn.L1Loss(reduction='none')

            if self.config.losses.get("mse", {"weight": 0})['weight'] > 0:
                self.loss_mse = nn.MSELoss(reduction='none')

            assert self.config.losses['perceptual']['weight'] == 0

            if self.config.losses.get("resnet_pl", {"weight": 0})['weight'] > 0:
                self.loss_resnet_pl = ResNetPL_video(**self.config.losses['resnet_pl'])
            else:
                self.loss_resnet_pl = None
            
            if self.config.losses.get("clip", {"weight": 0})['weight'] > 0:
                self.loss_clip = clip_loss(**self.config.losses['clip'])
            else:
                self.loss_clip = None
            
            if self.config.losses.get("smooth", {"weight": 0})['weight'] > 0:
                self.loss_smooth = smooth_loss(**self.config.losses['smooth'])
            else:
                self.loss_smooth = None
            
            self.gen_optimizer, self.dis_optimizer = self.configure_optimizers()
            self.str_optimizer = torch.optim.Adam(self.str_encoder.parameters(), lr=config.optimizers['generator']['lr'])
        if self.config.AMP:  # use AMP
            self.scaler = torch.cuda.amp.GradScaler()
        if not test:
            self.load_rezero()  # load pretrain model
        self.load()  # reload for restore

        # reset lr
        if not test:
            for group in self.gen_optimizer.param_groups:
                group['lr'] = config.optimizers['generator']['lr']
                group['initial_lr'] = config.optimizers['generator']['lr']
            for group in self.dis_optimizer.param_groups:
                group['lr'] = config.optimizers['discriminator']['lr']
                group['initial_lr'] = config.optimizers['discriminator']['lr']

        if self.config.DDP and not test:
            # import apex
            # self.generator = apex.parallel.convert_syncbn_model(self.generator)
            # self.discriminator = apex.parallel.convert_syncbn_model(self.discriminator)
            # self.generator = apex.parallel.DistributedDataParallel(self.generator)
            # self.discriminator = apex.parallel.DistributedDataParallel(self.discriminator)
            
            # self.generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.generator)
            # self.discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.discriminator)
            self.generator = torch.nn.parallel.DistributedDataParallel(self.generator, find_unused_parameters=True, broadcast_buffers=False)
            self.discriminator = torch.nn.parallel.DistributedDataParallel(self.discriminator, find_unused_parameters=True, broadcast_buffers=False)

        if self.config.optimizers['decay_steps'] is not None and self.config.optimizers['decay_steps'] > 0 and not test:
            self.g_scheduler = torch.optim.lr_scheduler.StepLR(self.gen_optimizer, config.optimizers['decay_steps'],
                                                               gamma=config.optimizers['decay_rate'])
            self.d_scheduler = torch.optim.lr_scheduler.StepLR(self.dis_optimizer, config.optimizers['decay_steps'],
                                                               gamma=config.optimizers['decay_rate'])
            self.str_scheduler = get_lr_schedule_with_warmup(self.str_optimizer,
                                                             num_warmup_steps=config.optimizers['warmup_steps'],
                                                             milestone_step=config.optimizers['decay_steps'],
                                                             gamma=config.optimizers['decay_rate'])
            if self.iteration - self.config.START_ITERS > 1:
                for _ in range(self.iteration - self.config.START_ITERS):
                    self.g_scheduler.step()
                    self.d_scheduler.step()
                    self.str_scheduler.step()
        else:
            self.g_scheduler = None
            self.d_scheduler = None
            self.str_scheduler = None
        
        # print network parameter number
        self.print_network()

    def load_rezero(self):
        if os.path.exists(self.config.gen_weights_path0):
            print('Loading %s generator...' % self.name)
            data = torch.load(self.config.gen_weights_path0, map_location='cpu')
            torch_init_model(self.generator, data, 'generator', rank=self.global_rank)
            self.gen_optimizer.load_state_dict(data['optimizer'])
            self.iteration = data['iteration']
        else:
            print('Warnning: There is no previous optimizer found. An initialized optimizer will be used.')

        # load discriminator only when training
        if (self.config.MODE == 1 or self.config.score) and os.path.exists(self.config.dis_weights_path0):
            print('Loading %s discriminator...' % self.name)
            data = torch.load(self.config.dis_weights_path0, map_location='cpu')
            torch_init_model(self.discriminator, data, 'discriminator', rank=self.global_rank)
            self.dis_optimizer.load_state_dict(data['optimizer'])
        else:
            print('Warnning: There is no previous optimizer found. An initialized optimizer will be used.')

    def load(self):
        if self.test:
            self.gen_weights_path = os.path.join(self.config.PATH, self.name + '_best_gen_HR.pth')
            print('Loading %s generator...' % self.name)
            data = torch.load(self.gen_weights_path, map_location='cpu')
            self.generator.load_state_dict(data['generator'])
            self.str_encoder.load_state_dict(data['str_encoder'])
        if not self.test and os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)
            data = torch.load(self.gen_weights_path, map_location='cpu')
            self.generator.load_state_dict(data['generator'])
            self.str_encoder.load_state_dict(data['str_encoder'])
            self.gen_optimizer.load_state_dict(data['optimizer'])
            self.str_optimizer.load_state_dict(data['str_opt'])
            self.iteration = data['iteration']
            # if self.iteration > 0:
            #     gen_weights_path = os.path.join(self.config.PATH, self.name + '_best_gen_HR.pth')
            #     data = torch.load(gen_weights_path, map_location='cpu')
            #     self.best = data['best_vfid']
            #     print('Loading best vfid...')

        else:
            print('Warnning: There is no previous optimizer found. An initialized optimizer will be used.')

        # load discriminator only when training
        if not self.test and os.path.exists(self.dis_weights_path):
            print('Loading %s discriminator...' % self.name)
            data = torch.load(self.dis_weights_path, map_location='cpu')
            self.dis_optimizer.load_state_dict(data['optimizer'])
            self.discriminator.load_state_dict(data['discriminator'])
        else:
            print('Warnning: There is no previous optimizer found. An initialized optimizer will be used.')

    def save(self):
        print('saving %s...' % self.name)
        raw_model = self.generator.module if hasattr(self.generator, "module") else self.generator
        raw_encoder = self.str_encoder.module if hasattr(self.str_encoder, "module") else self.str_encoder
        torch.save({
            'iteration': self.iteration,
            'optimizer': self.gen_optimizer.state_dict(),
            'str_opt': self.str_optimizer.state_dict(),
            'str_encoder': raw_encoder.state_dict(),
            'generator': raw_model.state_dict()
        }, self.gen_weights_path)
        raw_model = self.discriminator.module if hasattr(self.discriminator, "module") else self.discriminator
        torch.save({
            'optimizer': self.dis_optimizer.state_dict(),
            'discriminator': raw_model.state_dict()
        }, self.dis_weights_path)

    def configure_optimizers(self):
        discriminator_params = list(self.discriminator.parameters())
        return [
            make_optimizer(self.generator.parameters(), **self.config.optimizers['generator']),
            make_optimizer(discriminator_params, **self.config.optimizers['discriminator'])
        ]

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(
            'Network [%s] was created. Total number of parameters: %.1f million. '
            'To see the architecture, do print(network).' %
            (type(self).__name__, num_params / 1000000))


class LaMaInpaintingTrainingModule(LaMaBaseInpaintingTrainingModule):
    def __init__(self, *args, gpu, rank, image_to_discriminator='predicted_image', test=False, **kwargs):
        super().__init__(*args, gpu=gpu, name='InpaintingModel', rank=rank, test=test, **kwargs)
        self.image_to_discriminator = image_to_discriminator
        self.refine_mask_for_losses = None

    def forward(self, batch):
        img = batch['image']
        mask = batch['mask']
        masked_img = img * (1 - mask)
        masked_img = torch.cat([masked_img, mask], dim=1)
        batch['predicted_image'] = self.generator(masked_img.to(torch.float32))
        batch['inpainted'] = mask * batch['predicted_image'] + (1 - mask) * batch['image']
        batch['mask_for_losses'] = mask
        return batch

    def process(self, batch):
        self.iteration += 1

        self.discriminator.zero_grad()
        # discriminator loss
        self.adversarial_loss.pre_discriminator_step(real_batch=batch['image'], fake_batch=None,
                                                     generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(batch['image'])

        real_loss, _, _ = self.adversarial_loss.discriminator_real_loss(real_batch=batch['image'],
                                                                  discr_real_pred=discr_real_pred)
        batch = self.forward(batch)
        predicted_img = batch[self.image_to_discriminator].detach()

        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img.to(torch.float32))

        fake_loss = self.adversarial_loss.discriminator_fake_loss(discr_fake_pred=discr_fake_pred, mask=batch['mask'])

        dis_loss = fake_loss + real_loss

        dis_metric = {}
        dis_metric['discr_adv'] = dis_loss.item()
        dis_metric.update(add_prefix_to_keys(dis_metric, 'adv_'))

        dis_loss.backward()
        self.dis_optimizer.step()

        # generator loss
        self.generator.zero_grad()
        img = batch['image']
        predicted_img = batch[self.image_to_discriminator]
        original_mask = batch['mask']
        supervised_mask = batch['mask_for_losses']

        # L1
        l1_value = masked_l1_loss(predicted_img, img, supervised_mask,
                                  self.config.losses['l1']['weight_known'],
                                  self.config.losses['l1']['weight_missing'])

        gen_loss = l1_value
        gen_metric = dict(gen_l1=l1_value.item())

        # vgg-based perceptual loss
        if self.config.losses['perceptual']['weight'] > 0:
            pl_value = self.loss_pl(predicted_img, img,
                                    mask=supervised_mask).sum() * self.config.losses['perceptual']['weight']
            gen_loss = gen_loss + pl_value
            gen_metric['gen_pl'] = pl_value.item()

        # discriminator
        # adversarial_loss calls backward by itself
        mask_for_discr = original_mask
        self.adversarial_loss.pre_generator_step(real_batch=img, fake_batch=predicted_img,
                                                 generator=self.generator, discriminator=self.discriminator)
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img.to(torch.float32))
        adv_gen_loss, adv_metrics = self.adversarial_loss.generator_loss(discr_fake_pred=discr_fake_pred,
                                                                         mask=mask_for_discr)
        gen_loss = gen_loss + adv_gen_loss
        gen_metric['gen_adv'] = adv_gen_loss.item()
        gen_metric.update(add_prefix_to_keys(adv_metrics, 'adv_'))

        # feature matching
        if self.config.losses['feature_matching']['weight'] > 0:
            need_mask_in_fm = self.config.losses['feature_matching'].get('pass_mask', False)
            mask_for_fm = supervised_mask if need_mask_in_fm else None
            discr_real_pred, discr_real_features = self.discriminator(img)
            fm_value = feature_matching_loss(discr_fake_features, discr_real_features,
                                             mask=mask_for_fm) * self.config.losses['feature_matching']['weight']
            gen_loss = gen_loss + fm_value
            gen_metric['gen_fm'] = fm_value.item()

        if self.loss_resnet_pl is not None:
            with torch.cuda.amp.autocast():
                resnet_pl_value = self.loss_resnet_pl(predicted_img, img)
            gen_loss = gen_loss + resnet_pl_value
            gen_metric['gen_resnet_pl'] = resnet_pl_value.item()

        if self.config.AMP:
            self.scaler.scale(gen_loss).backward()
            self.scaler.step(self.gen_optimizer)
            self.scaler.update()
            gen_metric['loss_scale'] = self.scaler.get_scale()
        else:
            gen_loss.backward()
            self.gen_optimizer.step()
        # create logs
        logs = [dis_metric, gen_metric]

        return batch['predicted_image'], gen_loss, dis_loss, logs, batch


class DefaultInpaintingTrainingModule(BaseInpaintingTrainingModule):
    def __init__(self, *args, gpu, rank, image_to_discriminator='predicted_image', test=False, **kwargs):
        super().__init__(*args, gpu=gpu, name='InpaintingModel', rank=rank, test=test, **kwargs)
        self.image_to_discriminator = image_to_discriminator
        if self.config.AMP:  # use AMP
            self.scaler = torch.cuda.amp.GradScaler()

    def forward(self, batch):
        img = batch['image']
        mask = batch['mask']
        masked_img = img * (1 - mask)

        masked_img = torch.cat([masked_img, mask], dim=1)
        masked_str = torch.cat([batch['edge'], batch['line'], mask], dim=1)
        if self.config.rezero_for_mpe is not None and self.config.rezero_for_mpe:
            str_feats, rel_pos_emb, direct_emb = self.str_encoder(masked_str, batch['rel_pos'], batch['direct'])
            batch['predicted_image'] = self.generator(masked_img.to(torch.float32), rel_pos_emb, direct_emb, str_feats)
        else:
            str_feats = self.str_encoder(masked_str)
            batch['predicted_image'] = self.generator(masked_img.to(torch.float32), batch['rel_pos'],
                                                      batch['direct'], str_feats)
        batch['inpainted'] = mask * batch['predicted_image'] + (1 - mask) * batch['image']
        batch['mask_for_losses'] = mask
        return batch

    def process(self, batch):
        self.iteration += 1
        self.discriminator.zero_grad()
        # discriminator loss
        dis_loss, batch, dis_metric = self.discriminator_loss(batch)
        self.dis_optimizer.step()
        if self.d_scheduler is not None:
            self.d_scheduler.step()

        # generator loss
        self.generator.zero_grad()
        self.str_optimizer.zero_grad()
        # generator loss
        gen_loss, gen_metric = self.generator_loss(batch)

        if self.config.AMP:
            self.scaler.step(self.gen_optimizer)
            self.scaler.update()
            self.scaler.step(self.str_optimizer)
            self.scaler.update()
        else:
            self.gen_optimizer.step()
            self.str_optimizer.step()

        if self.str_scheduler is not None:
            self.str_scheduler.step()
        if self.g_scheduler is not None:
            self.g_scheduler.step()

        # create logs
        if self.config.AMP:
            gen_metric['loss_scale'] = self.scaler.get_scale()
        logs = [dis_metric, gen_metric]

        return batch['predicted_image'], gen_loss, dis_loss, logs, batch

    def generator_loss(self, batch):
        img = batch['image']
        predicted_img = batch[self.image_to_discriminator]
        original_mask = batch['mask']
        supervised_mask = batch['mask_for_losses']

        # L1
        l1_value = masked_l1_loss(predicted_img, img, supervised_mask,
                                  self.config.losses['l1']['weight_known'],
                                  self.config.losses['l1']['weight_missing'])

        total_loss = l1_value
        metrics = dict(gen_l1=l1_value.item())

        # discriminator
        # adversarial_loss calls backward by itself
        mask_for_discr = original_mask
        self.adversarial_loss.pre_generator_step(real_batch=img, fake_batch=predicted_img,
                                                 generator=self.generator, discriminator=self.discriminator)
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img.to(torch.float32))
        adv_gen_loss, adv_metrics = self.adversarial_loss.generator_loss(discr_fake_pred=discr_fake_pred,
                                                                         mask=mask_for_discr)
        total_loss = total_loss + adv_gen_loss
        metrics['gen_adv'] = adv_gen_loss.item()
        metrics.update(add_prefix_to_keys(adv_metrics, 'adv_'))
        # feature matching
        if self.config.losses['feature_matching']['weight'] > 0:
            discr_real_pred, discr_real_features = self.discriminator(img)
            need_mask_in_fm = self.config.losses['feature_matching'].get('pass_mask', False)
            mask_for_fm = supervised_mask if need_mask_in_fm else None
            fm_value = feature_matching_loss(discr_fake_features, discr_real_features,
                                             mask=mask_for_fm) * self.config.losses['feature_matching']['weight']
            total_loss = total_loss + fm_value
            metrics['gen_fm'] = fm_value.item()

        if self.loss_resnet_pl is not None:
            resnet_pl_value = self.loss_resnet_pl(predicted_img, img)
            total_loss = total_loss + resnet_pl_value
            metrics['gen_resnet_pl'] = resnet_pl_value.item()

        if self.config.AMP:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        return total_loss.item(), metrics

    def discriminator_loss(self, batch):
        self.adversarial_loss.pre_discriminator_step(real_batch=batch['image'], fake_batch=None,
                                                     generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(batch['image'])
        real_loss, dis_real_loss, grad_penalty = self.adversarial_loss.discriminator_real_loss(
            real_batch=batch['image'],
            discr_real_pred=discr_real_pred)
        real_loss.backward()
        if self.config.AMP:
            with torch.cuda.amp.autocast():
                batch = self.forward(batch)
        else:
            batch = self(batch)
        batch[self.image_to_discriminator] = batch[self.image_to_discriminator].to(torch.float32)
        predicted_img = batch[self.image_to_discriminator].detach()
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_img.to(torch.float32))
        fake_loss = self.adversarial_loss.discriminator_fake_loss(discr_fake_pred=discr_fake_pred, mask=batch['mask'])
        fake_loss.backward()
        total_loss = fake_loss + real_loss
        metrics = {}
        metrics['dis_real_loss'] = dis_real_loss.mean().item()
        metrics['dis_fake_loss'] = fake_loss.item()
        metrics['grad_penalty'] = grad_penalty.mean().item()

        return total_loss.item(), batch, metrics


class DefaultInpaintingTrainingModule_video(BaseInpaintingTrainingModule_video):
    def __init__(self, args, config, gpu, rank, video_to_discriminator='predicted_video', test=False, **kwargs):
        super().__init__(args=args, config=config, gpu=gpu, name='InpaintingModel', rank=rank, test=test, **kwargs)
        self.video_to_discriminator = video_to_discriminator
        if self.config.AMP:  # use AMP
            self.scaler = torch.cuda.amp.GradScaler()

    def forward(self, batch):
        img = batch['frames'] # [B, 3, H, W] -> [B, T, 3, H, W] for video
        mask = batch['masks'] # [B, 1, H, W] -> [B, T, 1, H, W] for video
        # masked_img = img * (1 - mask) + mask # [B, T, 3, H, W] for video ======> old version before 0717
        masked_img = img * (1 - mask) # [B, T, 3, H, W] for video
        
        # masked_edge = batch['edges'] * (1 - mask) # add at 8/19 when checking whether the edge/line have been correctly masked
        # masked_line = batch['lines'] * (1 - mask) # add at 8/19 when checking whether the edge/line have been correctly masked

        masked_img = torch.cat([masked_img, mask], dim=2) # [B, T, 4, H, W] for video
        masked_str = torch.cat([batch['edges'], batch['lines'], mask], dim=2) # [B, T, 3, H, W] for video  ======> old version before 0819
        # masked_str = torch.cat([masked_edge, masked_line, mask], dim=2) # [B, T, 3, H, W] for video
        if self.config.rezero_for_mpe is not None and self.config.rezero_for_mpe: # rezero for mpe
            str_feats, rel_pos_emb, direct_emb = self.str_encoder(masked_str, batch['rel_pos'], batch['direct']) # structure encoder for masked str(line/edge) and relative, direct position
            batch['predicted_video'] = self.generator(masked_img.to(torch.float32), rel_pos_emb, direct_emb, str_feats) # [B, T, 3, H, W] for video from the inpainting model
        else:
            str_feats = self.str_encoder(masked_str)  # not using masked positional encoding, so just use structure encoder for masked str(line/edge)
            batch['predicted_video'] = self.generator(masked_img.to(torch.float32), batch['rel_pos'],
                                                      batch['direct'], str_feats)
        
        batch['inpainted'] = mask * batch['predicted_video'] + (1 - mask) * batch['frames'] # [B, T, 3, H, W] for video
        batch['mask_for_losses'] = mask
        return batch

    def process(self, batch):
        self.iteration += 1 # iteration for optimizer
        
        # discriminator loss
        self.discriminator.zero_grad() # clear gradient for discriminator
        dis_loss, batch, dis_metric = self.discriminator_loss(batch) #  [B, T, 3, H, W] for video compute the discriminator loss

        # discriminator update
        self.dis_optimizer.step() # update discriminator
        if self.d_scheduler is not None: 
            self.d_scheduler.step() # update discriminator scheduler (learning rate decay)
        
        # generator loss
        self.generator.zero_grad() # clear gradient for generator
        self.str_optimizer.zero_grad() # clear gradient for structure encoder
        # generator loss
        gen_loss, gen_metric = self.generator_loss(batch) # compute the generator loss

        # generator update
        if self.config.AMP:  # use AMP
            self.scaler.step(self.gen_optimizer) #  update generator with scaler
            self.scaler.update()
            self.scaler.step(self.str_optimizer)
            self.scaler.update()
        else:
            self.gen_optimizer.step()
            self.str_optimizer.step()

        if self.str_scheduler is not None:
            self.str_scheduler.step()
        if self.g_scheduler is not None:
            self.g_scheduler.step()

        # create logs
        if self.config.AMP:
            gen_metric['loss_scale'] = self.scaler.get_scale()
        logs = [dis_metric, gen_metric]

        return batch['predicted_video'], gen_loss, dis_loss, logs, batch

    def generator_loss(self, batch):
        frames = batch['frames'] # Change 'image' to 'video'
        predicted_video = batch[self.video_to_discriminator] 
        original_mask = batch['masks']
        supervised_mask = batch['mask_for_losses']
        metrics = dict()
        total_loss = 0.

        # L1
        l1_value = masked_l1_loss(predicted_video, frames, supervised_mask,
                                self.config.losses['l1']['weight_known'],
                                self.config.losses['l1']['weight_missing'])

        total_loss = l1_value
        metrics[f'gen_l1'] = l1_value.item()

        # discriminator
        # adversarial_loss calls backward by itself
        self.adversarial_loss.pre_generator_step(real_batch=frames, fake_batch=predicted_video,
                                                generator=self.generator, discriminator=self.discriminator)
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_video.to(torch.float32))
        adv_gen_loss, adv_metrics = self.adversarial_loss.generator_loss(discr_fake_pred=discr_fake_pred,
                                                                        mask=original_mask)
        total_loss = total_loss + adv_gen_loss
        metrics[f'gen_adv'] = adv_gen_loss.item()
        metrics.update(add_prefix_to_keys(adv_metrics, f'adv'))

        # feature matching
        if self.config.losses['feature_matching']['weight'] > 0:
            discr_real_pred, discr_real_features = self.discriminator(frames)
            need_mask_in_fm = self.config.losses['feature_matching'].get('pass_mask', False)
            mask_for_fm = supervised_mask if need_mask_in_fm else None
            fm_value = feature_matching_loss(discr_fake_features, discr_real_features,
                                            mask=mask_for_fm) * self.config.losses['feature_matching']['weight']
            total_loss = total_loss + fm_value
            metrics[f'gen_fm'] = fm_value.item()

        if self.loss_resnet_pl is not None:
            resnet_pl_value = self.loss_resnet_pl(predicted_video, frames)
            total_loss = total_loss + resnet_pl_value
            metrics[f'gen_resnet_pl'] = resnet_pl_value.item()

        # clip
        if self.loss_clip is not None:
            clip_value, consistency_value, traj_value = self.loss_clip(predicted_video, frames)
            if clip_value is not None:
                total_loss = total_loss + clip_value
                metrics[f'gen_clip'] = clip_value.item()
            if consistency_value is not None:
                total_loss = total_loss + consistency_value
                metrics[f'gen_clip_consistency'] = consistency_value.item()
            if traj_value is not None:
                total_loss = total_loss + traj_value
                metrics[f'gen_clip_traj'] = traj_value.item()

        # image smoothness loss
        if self.loss_smooth is not None:
            smooth_value = self.loss_smooth(predicted_video)
            total_loss = total_loss + smooth_value
            metrics[f'gen_smooth'] = smooth_value.item()

        if self.config.AMP:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        return total_loss.item(), metrics

    def discriminator_loss(self, batch):
        self.adversarial_loss.pre_discriminator_step(real_batch=batch['frames'], fake_batch=None,
                                                     generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(batch['frames'])
        real_loss, dis_real_loss, grad_penalty = self.adversarial_loss.discriminator_real_loss(
            real_batch=batch['frames'],
            discr_real_pred=discr_real_pred)
        real_loss.backward()
        if self.config.AMP:
            with torch.cuda.amp.autocast():
                batch = self.forward(batch)
        else:
            batch = self(batch)
        batch[self.video_to_discriminator] = batch[self.video_to_discriminator].to(torch.float32)
        predicted_video = batch[self.video_to_discriminator].detach()
        discr_fake_pred, discr_fake_features = self.discriminator(predicted_video.to(torch.float32))
        fake_loss = self.adversarial_loss.discriminator_fake_loss(discr_fake_pred=discr_fake_pred, mask=batch['masks'])
        fake_loss.backward()
        total_loss = fake_loss + real_loss
        # total_loss.backward(retain_graph=True)
        metrics = {}
        metrics['dis_real_loss'] = dis_real_loss.mean().item()
        metrics['dis_fake_loss'] = fake_loss.item()
        metrics['grad_penalty'] = grad_penalty.mean().item()

        return total_loss.item(), batch, metrics