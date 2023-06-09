# Copyright (c) Cyril Zakka.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Copied from:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/MAE
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import wandb
import torch
from timeit import default_timer

import utils.misc as misc
import utils.lr_sched as lr_sched


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int, loss_scaler,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    epoch_start = data_start = default_timer()
    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        data_time = default_timer() - data_start

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        fwd_bwd_start = default_timer()

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        fwd_bwd_time = default_timer() - fwd_bwd_start

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        num_frames = samples.size(0) * samples.size(2)
        fps = num_frames / (fwd_bwd_time + data_time)
        print(
            f"step {data_iter_step} | loss {loss_value:.4f} | fwdbwd_t {fwd_bwd_time:.4f} | "
            f"data_t {data_time:.4f} | fps {fps:.4f}"
        )
        if data_iter_step % 10 == 0 and misc.is_main_process():
            wandb.log(
                {
                    'loss': loss_value,
                    'fwdbwd_t': fwd_bwd_time,
                    'data_t': data_time,
                    'fps': fps,
                },
                step=data_iter_step,
            )

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

        data_start = default_timer()

    epoch_time = default_timer() - epoch_start
    pritn(f"Epoch time: {epoch_time}")
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
