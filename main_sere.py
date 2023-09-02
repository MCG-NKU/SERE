# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# iBOT: https://github.com/bytedance/ibot
# --------------------------------------------------------

import argparse
import os
import sys
import datetime
import time
import math
import json
import numpy as np
import utils
import models
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import transforms_overlap
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tensorboardX import SummaryWriter
from models.head import iBOTHead
from loader import ImageFolderMask


def get_args_parser():
    parser = argparse.ArgumentParser('SERE', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--window_size', default=7, type=int, help="""Size of window - default 7.
        This config is only valid for Swin Transofmer and is ignoired for vanilla ViT architectures.""")
    parser.add_argument('--out_dim', default=8192, type=int, help="""Dimensionality of
        output for [CLS] token.""")
    parser.add_argument('--patch_out_dim', default=8192, type=int, help="""Dimensionality of
        output for patch tokens.""")
    parser.add_argument('--shared_head', default=False, type=utils.bool_flag, help="""Wether to share 
        the same head for [CLS] token output and patch tokens output. When set to false, patch_out_dim
        is ignored and enforced to be same with out_dim. (Default: False)""")
    parser.add_argument('--shared_head_teacher', default=True, type=utils.bool_flag, help="""See above.
        Only works for teacher model. (Defeault: True)""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--norm_in_head', default=None,
        help="Whether to use batch normalizations in projection head (Default: None)")
    parser.add_argument('--act_in_head', default='gelu',
        help="Whether to use batch normalizations in projection head (Default: gelu)")
    parser.add_argument('--use_masked_im_modeling', default=True, type=utils.bool_flag,
        help="Whether to use masked image modeling (mim) in backbone (Default: True)")
    parser.add_argument('--pred_ratio', default=0.3, type=float, nargs='+', help="""Ratio of partial prediction.
        If a list of ratio is specified, one of them will be randomly choosed for each patch.""")
    parser.add_argument('--pred_ratio_var', default=0, type=float, nargs='+', help="""Variance of partial prediction
        ratio. Length should be indentical to the length of pred_ratio. 0 for disabling. """)
    parser.add_argument('--pred_shape', default='block', type=str, help="""Shape of partial prediction.""")
    parser.add_argument('--pred_start_epoch', default=0, type=int, help="""Start epoch to perform masked
        image prediction. We typically set this to 50 for swin transformer. (Default: 0)""")
    parser.add_argument('--lambda1', default=1.0, type=float, help="""loss weight for dino
        loss over [CLS] tokens (Default: 1.0)""")
    parser.add_argument('--lambda2', default=1.0, type=float, help="""loss weight for beit 
        loss over masked patch tokens (Default: 1.0)""")
        
    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_patch_temp', default=0.04, type=float, help="""See 
        `--warmup_teacher_temp`""")
    parser.add_argument('--teacher_patch_temp', default=0.07, type=float, help=""""See 
        `--teacher_temp`""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--load_from', default=None, help="""Path to load checkpoints to resume training.""")
    parser.add_argument('--drop_path', type=float, default=0.1, help="""Drop path rate for student network.""")
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Multi-crop parameters
    parser.add_argument('--global_crops_number', type=int, default=2, help="""Number of global
        views to generate. Default is to use two global crops. """)
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.14, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=0, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=40, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    # SERE
    parser.add_argument("--overlap_size", nargs="+", default=(13, 13), type=int, help="The sampling size of grid sample.")
    parser.add_argument('--alpha', default=1.0, type=float, help='Scaling factor of spatial relf-relation.')
    parser.add_argument('--beta', default=1.0, type=float, help='Scaling factor of channel relf-relation.')
    parser.add_argument('--temperature_pixel', default=0.5, type=float, help='Temperature term of spatial relf-relation.')
    parser.add_argument('--temperature_channel', default=0.1, type=float, help='Temperature term of spatial relf-relation.')

    return parser

def train_sere(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationiBOT(
        args.global_crops_scale,
        args.local_crops_scale,
        args.global_crops_number,
        args.local_crops_number,
        overlap_size=args.overlap_size,
    )
    pred_size = args.patch_size * 8 if 'swin' in args.arch else args.patch_size
    dataset = ImageFolderMask(
        args.data_path, 
        transform=transform,
        patch_size=pred_size,
        pred_ratio=args.pred_ratio,
        pred_ratio_var=args.pred_ratio_var,
        pred_aspect_ratio=(0.3, 1/0.3),
        pred_shape=args.pred_shape,
        pred_start_epoch=args.pred_start_epoch)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a vision transformer (i.e. vit_tiny, vit_small, vit_base, vit_large)
    if args.arch in models.__dict__.keys():
        student = models.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path,
            return_all_tokens=True,
            masked_im_modeling=args.use_masked_im_modeling,
            local_crop=args.local_crops_number > 0
        )
        teacher = models.__dict__[args.arch](
            patch_size=args.patch_size,
            return_all_tokens=True,
            local_crop=args.local_crops_number > 0
        )
        embed_dim = student.embed_dim
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, iBOTHead(
        embed_dim,
        args.out_dim,
        patch_out_dim=args.patch_out_dim,
        norm=args.norm_in_head,
        act=args.act_in_head,
        norm_last_layer=args.norm_last_layer,
        shared_head=args.shared_head,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        iBOTHead(
            embed_dim, 
            args.out_dim,
            patch_out_dim=args.patch_out_dim,
            norm=args.norm_in_head,
            act=args.act_in_head,
            shared_head=args.shared_head_teacher,
        ),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], broadcast_buffers=False) if \
            'swin' in args.arch else nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], broadcast_buffers=False) if \
        'swin' in args.arch else nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    same_dim = args.shared_head or args.shared_head_teacher
    ibot_loss = iBOTLoss(
        args.out_dim,
        args.out_dim if same_dim else args.patch_out_dim,
        args.global_crops_number,
        args.local_crops_number,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_patch_temp,
        args.teacher_patch_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        mim_start_epoch=args.pred_start_epoch,
        accum_iter=args.accum_iter
    ).cuda()

    if utils.is_main_process(): # Tensorboard configuration
        local_runs = os.path.join(args.output_dir, 'tf_logs')
        writer = SummaryWriter(logdir=local_runs)
        
    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    eff_batch_size = args.batch_size_per_gpu * args.accum_iter * utils.get_world_size()
    print("Effective batch size: %d" % eff_batch_size)
    print("Base lr: %.2e" % args.lr)
    print("Actual lr: %.2e" % (args.lr * (eff_batch_size) / 256.))
    lr_schedule = utils.cosine_scheduler(
        args.lr * (eff_batch_size) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                            args.epochs, len(data_loader))
                  
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    if args.load_from:
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, args.load_from),
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            ibot_loss=ibot_loss,
        )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting SERE training!")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        data_loader.dataset.set_epoch(epoch)

        # ============ training one epoch of SERE ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, ibot_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'ibot_loss': ibot_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and (epoch % args.saveckp_freq == 0) and epoch:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
                for k, v in train_stats.items():
                    writer.add_scalar(k, v, epoch)
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, ibot_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    accum_iter = args.accum_iter
    optimizer.zero_grad()
    
    # common params
    names_q, params_q, names_k, params_k = [], [], [], []
    for name_q, param_q in student.module.named_parameters():
        names_q.append(name_q)
        params_q.append(param_q)
    for name_k, param_k in teacher_without_ddp.named_parameters():
        names_k.append(name_k)
        params_k.append(param_k)
    names_common = list(set(names_q) & set(names_k))
    params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
    params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]

    pred_labels, real_labels = [], []
    for it_current_epoch, (images, labels, masks) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it_current_epoch  # global training iteration

        if it_current_epoch % accum_iter == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        masks = [msk.cuda(non_blocking=True) for msk in masks]        

        # grid is used for grid_sampling to align different crops (Section 3.2 and Figure 3)
        if args.local_crops_number > 0:
            grid_local = images[-4 * args.local_crops_number: -2 * args.local_crops_number]
            grid_global = images[-2 * args.local_crops_number:]        
            images = images[:-4 * args.local_crops_number]
            grid1 = images[-4:-2]
            grid2 = images[-2:]
            images = images[:-4]
        else:
            grid_local = grid_global = None
            grid1 = images[-4:-2]
            grid2 = images[-2:]
            images = images[:-4]
        
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # get global views
            teacher_output, teacher_relation_channel, teacher_relation_pixel, _, _, teacher_relation_pixel_global = \
                teacher(images[:args.global_crops_number], grid1, grid2, teacher=True)
            student_output, student_relation_channel, student_relation_pixel, _, _, _ = \
                student(images[:args.global_crops_number], grid1, grid2, mask=masks[:args.global_crops_number], teacher=False)
            
            # get local views
            student.module.backbone.masked_im_modeling = False
            if len(images) <= args.global_crops_number:
                student_local_cls = None
                student_relation_channel_local = None
                student_relation_pixel_local = None
            else:
                student_local_cls, _, _, student_relation_channel_local, student_relation_pixel_local, _ = student(images[args.global_crops_number:], teacher=False)
                student_local_cls = student_local_cls[0]

            student.module.backbone.masked_im_modeling = args.use_masked_im_modeling

            all_loss = ibot_loss(student_output, teacher_output, student_local_cls, masks, epoch, it=it_current_epoch)
            
            # SERE: loss for pixel self-relation
            loss_sere_pixel = sere_loss_pixel(
                args, 
                student_relation_pixel, teacher_relation_pixel, 
                student_relation_pixel_local, teacher_relation_pixel_global, 
                grid_local, grid_global, ncrops=2, ncrops_local=args.local_crops_number)
            # SERE: loss for channel self-relation
            loss_sere_channel = sere_loss_channel(
                args, 
                student_relation_channel, teacher_relation_channel, 
                student_relation_channel_local, 
                ncrops=2, ncrops_local=args.local_crops_number)

            cls_loss = all_loss.pop('loss')
        loss = cls_loss * (4 - args.alpha - args.beta) / 2.0 + loss_sere_pixel * args.alpha + loss_sere_channel * args.beta

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)
        
        loss /= accum_iter

        # log statistics
        probs1 = teacher_output[0].chunk(args.global_crops_number)
        probs2 = student_output[0].chunk(args.global_crops_number)
        pred1 = utils.concat_all_gather(probs1[0].max(dim=1)[1]) 
        pred2 = utils.concat_all_gather(probs2[1].max(dim=1)[1])
        acc = (pred1 == pred2).sum() / pred1.size(0)
        pred_labels.append(pred1)
        real_labels.append(utils.concat_all_gather(labels.to(pred1.device)))

        # student update
        if fp16_scaler is None:
            loss.backward()
        else:
            fp16_scaler.scale(loss).backward()
        if (it_current_epoch + 1) % accum_iter == 0:
            param_norms = None
            if fp16_scaler is None:
                # loss.backward()
                if args.clip_grad:
                    param_norms = utils.clip_gradients(student, args.clip_grad)
                utils.cancel_gradients_last_layer(epoch, student,
                                                args.freeze_last_layer)
                optimizer.step()
            else:
                # fp16_scaler.scale(loss).backward()
                if args.clip_grad:
                    fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                    param_norms = utils.clip_gradients(student, args.clip_grad)
                utils.cancel_gradients_last_layer(epoch, student,
                                                args.freeze_last_layer)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
        param_norms = None
        if (it_current_epoch + 1) % accum_iter == 0:
            optimizer.zero_grad()

        # EMA update for the teacher
        if (it_current_epoch + 1) % accum_iter == 0:
            with torch.no_grad():
                m = momentum_schedule[it]  # momentum parameter
                for param_q, param_k in zip(params_q, params_k):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=cls_loss.item())
        for key, value in all_loss.items():
            metric_logger.update(**{key: value.item()})

        metric_logger.update(loss_attn=loss_sere_pixel.item())
        metric_logger.update(loss_attn_ffn=loss_sere_channel.item())

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(acc=acc)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return return_dict


class iBOTLoss(nn.Module):
    def __init__(self, out_dim, patch_out_dim, ngcrops, nlcrops, warmup_teacher_temp, 
                 teacher_temp, warmup_teacher_temp2, teacher_temp2, 
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, 
                 center_momentum=0.9, center_momentum2=0.9,
                 lambda1=1.0, lambda2=1.0, mim_start_epoch=0, accum_iter=1):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.ngcrops = ngcrops
        self.nlcrops = nlcrops
        self.ncrops = ngcrops + nlcrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.accum_iter = accum_iter

        self.register_buffer("center_accum", torch.zeros(1, out_dim))
        self.register_buffer("center2_accum", torch.zeros(1, 1, patch_out_dim))

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.teacher_temp2_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp2
        )) if mim_start_epoch == 0 else np.concatenate((
            np.ones(mim_start_epoch) * warmup_teacher_temp2,
            np.linspace(warmup_teacher_temp2,
                        teacher_temp2, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs - mim_start_epoch) * teacher_temp2
        ))

    def forward(self, student_output, teacher_output, student_local_cls, student_mask, epoch, it):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_cls, student_patch = student_output
        teacher_cls, teacher_patch = teacher_output
        
        if student_local_cls is not None:
            student_cls = torch.cat([student_cls, student_local_cls])

        # [CLS] and patch for global patches
        student_cls = student_cls / self.student_temp
        student_cls_c = student_cls.chunk(self.ncrops)
        student_patch = student_patch / self.student_temp
        student_patch_c = student_patch.chunk(self.ngcrops)
        
        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]
        teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)
        teacher_cls_c = teacher_cls_c.detach().chunk(self.ngcrops)
        teacher_patch_c = F.softmax((teacher_patch - self.center2) / temp2, dim=-1)
        teacher_patch_c = teacher_patch_c.detach().chunk(self.ngcrops)

        total_loss1, n_loss_terms1 = 0, 0
        total_loss2, n_loss_terms2 = 0, 0
        for q in range(len(teacher_cls_c)):
            for v in range(len(student_cls_c)):
                if v == q:
                    loss2 = torch.sum(-teacher_patch_c[q] * F.log_softmax(student_patch_c[v], dim=-1), dim=-1)
                    mask = student_mask[v].flatten(-2, -1)
                    loss2 = torch.sum(loss2 * mask.float(), dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
                    total_loss2 += loss2.mean()
                    n_loss_terms2 += 1
                else:
                    loss1 = torch.sum(-teacher_cls_c[q] * F.log_softmax(student_cls_c[v], dim=-1), dim=-1)
                    total_loss1 += loss1.mean()
                    n_loss_terms1 += 1
            
        total_loss1 = total_loss1 / n_loss_terms1 * self.lambda1
        total_loss2 = total_loss2 / n_loss_terms2 * self.lambda2
        total_loss = dict(cls=total_loss1, patch=total_loss2, loss=total_loss1 + total_loss2)

        self.update_center_accum(teacher_cls, teacher_patch)
        if (it + 1) % self.accum_iter == 0:
            self.update_center(teacher_cls, teacher_patch)                  

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_patch):
        """
        Update center used for teacher output.
        """
        cls_center = self.center_accum
        cls_center = cls_center / (len(teacher_cls) * dist.get_world_size() * args.accum_iter)
        self.center = self.center * self.center_momentum + cls_center * (1 - self.center_momentum)

        patch_center = self.center2_accum
        patch_center = patch_center / (len(teacher_patch) * dist.get_world_size() * args.accum_iter)
        self.center2 = self.center2 * self.center_momentum2 + patch_center * (1 - self.center_momentum2)

        self.center_accum.fill_(0)
        self.center2_accum.fill_(0)

    @torch.no_grad()
    def update_center_accum(self, teacher_cls, teacher_patch):
        """
        Update center used for teacher output.
        """
        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        dist.all_reduce(cls_center)
        self.center_accum = self.center_accum + cls_center

        patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
        dist.all_reduce(patch_center)
        self.center2_accum = self.center2_accum + patch_center


class DataAugmentationiBOT(object):
    def __init__(self, global_crops_scale, local_crops_scale, global_crops_number, local_crops_number, overlap_size=7):
        flip_and_color_jitter = transforms_overlap.ComposeOverLap([
            transforms_overlap.RandomHorizontalFlipOverLap(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.global_crops_number = global_crops_number
        # transformation for the first global crop
        self.global_transfo1 = transforms_overlap.ComposeOverLap([
            transforms_overlap.RandomResizedCropOverLap(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # transformation for the rest of global crops
        self.global_transfo2 = transforms_overlap.ComposeOverLap([
            transforms_overlap.RandomResizedCropOverLap(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms_overlap.ComposeOverLap([
            transforms_overlap.RandomResizedCropOverLap(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])
        self.overlap_size = overlap_size

    def __call__(self, image):
        crops = []
        mappings_local = []
        img1, mapping1 = self.global_transfo1(image)
        crops.append(img1)
        for _ in range(self.global_crops_number - 1):
            img2, mapping2 = self.global_transfo2(image)
            crops.append(img2)

        for _ in range(self.local_crops_number):
            local_img, local_mapping = self.local_transfo(image)
            crops.append(local_img)
            mappings_local.append(local_mapping)
        
        # prepare grid_sampling to align two global crops (Section 3.2 and Figure 3)
        gridq, gridk = transforms_overlap.get_grid(mapping1, mapping2, self.overlap_size[0])
        crops += [gridq, gridk]
        gridq, gridk = transforms_overlap.get_grid(mapping1, mapping2, self.overlap_size[1])
        crops += [gridq, gridk]

        # prepare grid_sampling to align a global crop and a local crop (Section 3.2 and Figure 3)
        local_crops = []
        global_crops = []
        for i in range(self.local_crops_number):
            grid_local, grid_global = transforms_overlap.get_grid(mappings_local[i], mapping1, 6)
            local_crops.append(grid_local)
            global_crops.append(grid_global)
        
        for i in range(self.local_crops_number):
            grid_local, grid_global = transforms_overlap.get_grid(mappings_local[i], mapping2, 6)
            local_crops.append(grid_local)
            global_crops.append(grid_global)

        return crops + local_crops + global_crops


def sere_loss_pixel(args, student_output, teacher_output, student_output_local, teacher_output_global, grid_local, grid_global, ncrops, ncrops_local):
    """
    Cross-entropy between softmax self-relations of the teacher and student networks.
    """
    student_out = student_output / 1.0
    student_out = student_out.chunk(ncrops)  # [B H N]
    if student_output_local is not None:
        student_out_local = student_output_local.chunk(ncrops_local)

    # sharpening
    teacher_out = (teacher_output / args.temperature_pixel).detach().chunk(ncrops)
    teacher_out_global = (teacher_output_global / args.temperature_pixel).detach().chunk(ncrops)

    total_loss = 0
    n_loss_terms = 0

    for iq, q in enumerate(teacher_out):
        for v in range(len(student_out)):
            if v == iq:
                # we skip cases where student and teacher operate on the same view
                continue
            loss = torch.sum(-F.softmax(q, dim=-1) * F.log_softmax(student_out[v], dim=-1), dim=-1)
            total_loss += loss.mean()
            n_loss_terms += 1

    if student_output_local is not None:
        for iq, q in enumerate(teacher_out_global):
            for v in range(len(student_out_local)):
                local_crop = sampling(student_out_local[v], grid_local[iq * ncrops_local + v])
                global_crop = sampling(q, grid_global[iq * ncrops_local + v])
                loss = torch.sum(-F.softmax(global_crop, dim=-1) * F.log_softmax(local_crop, dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

    total_loss /= n_loss_terms
    return total_loss


def sampling(relation, grid):
    b, h, n, _ = relation.shape  # [B H N N]
    target_n = grid[0].shape[1] * grid[0].shape[1]
    relation = relation.contiguous().view(b, h * n, int(math.sqrt(n)), int(math.sqrt(n)))
    relation = (
        F.grid_sample(
            relation.float(),
            grid,
            mode="bilinear",
            align_corners=False,
        )
        .view(b, h, n, target_n)
        .transpose(3, 2)
    )
    relation = relation.contiguous().view(
        b, h * target_n, int(math.sqrt(n)), int(math.sqrt(n))
    )
    relation = (
        F.grid_sample(
            relation.float(),
            grid,
            mode="bilinear",
            align_corners=False,
        )
        .view(b, h, target_n, target_n)
        .transpose(3, 2)
    )
    return relation


def sere_loss_channel(args, student_output, teacher_output, student_output_local, ncrops, ncrops_local):
    """
    Cross-entropy between softmax self-relations of the teacher and student networks.
    """
    student_out = student_output / 1.0
    student_out = student_out.chunk(ncrops)  # [B H N]
    if student_output_local is not None:
        student_out_local = student_output_local / 1.0
        student_out_local = student_out_local.chunk(ncrops_local)  # [B H N]

    # sharpening
    teacher_out = teacher_output / args.temperature_channel
    teacher_out = F.softmax(teacher_out, dim=-1).detach().chunk(ncrops)
    
    total_loss = 0
    n_loss_terms = 0

    for iq, q in enumerate(teacher_out):
        for v in range(len(student_out)):
            if v == iq:
                # we skip cases where student and teacher operate on the same view
                continue
            loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
            total_loss += loss.mean()
            n_loss_terms += 1

        if student_output_local is not None:
            for v in range(len(student_out_local)):
                loss = torch.sum(-q * F.log_softmax(student_out_local[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

    total_loss /= n_loss_terms
    return total_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser('iBOT', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_sere(args)
