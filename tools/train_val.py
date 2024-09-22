import argparse
import logging
import os
import pprint
import shutil
import time

import torch
import torch.distributed as dist
import torch.optim
import yaml
from datasets.data_builder import build_dataloader
from easydict import EasyDict
from models.model_helper import ModelHelper
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.criterion_helper import build_criterion
from utils.dist_helper import setup_distributed
from utils.eval_helper import dump, log_metrics, merge_together, performances
from utils.lr_helper import get_scheduler
from utils.misc_helper import (
    AverageMeter,
    create_logger,
    get_current_time,
    load_state,
    save_checkpoint,
    set_random_seed,
    update_config,
)
from utils.optimizer_helper import get_optimizer
from utils.vis_helper import visualize_compound, visualize_single

parser = argparse.ArgumentParser(description="OneNIP Framework")
parser.add_argument("--config", default="../configs/onenip_config.yaml")
parser.add_argument("-e", "--evaluate", action="store_true")
parser.add_argument("--local_rank", default=None, help="local rank for dist")
parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")


def convert_string(value):
    try:
        # 尝试将字符串转换为小数
        return float(value)
    except ValueError:
        if value.lower() == "true":
            # 如果字符串为 "true"，则返回 True
            return True
        elif value.lower() == "false":
            # 如果字符串为 "false"，则返回 False
            return False
        else:
            # 如果无法转换，则返回原始字符串
            return value


def merge_from_list(config, cfg_list):
    """
      Merge config (keys, values) in a list (e.g., from command line) into
      this CfgNode. For example, `cfg_list = ['FOO.BAR', 0.5]`.
    """

    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = full_key.split(".")
        d = config
        for subkey in key_list[:-1]:
            if '[' in subkey:
                subsubkey = subkey.split('[')
                d = d[subsubkey[0]]
                d = d[int(subsubkey[-1].strip(']'))]
            else:
                d = d[subkey]
        subkey = key_list[-1]

        d[subkey] = convert_string(v)
    return config


def main():
    global args, config, key_metric, best_metric
    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    
    if args.opts:
        config = merge_from_list(config, args.opts)

    
    # update save dir
    config.saver.load_path =  os.path.join(config.saver.save_dir, 'ckpt.pkl')
    config.saver.log_dir = os.path.join(config.saver.save_dir, 'log')
    config.evaluator.save_dir = os.path.join(config.saver.save_dir, 'result_eval_temp')
    config.evaluator.vis_compound.save_dir = os.path.join(config.saver.save_dir,  'vis_compound')
    config.net[2].kwargs.save_recon.save_dir = os.path.join(config.saver.save_dir,  'result_recon')
       
    config.port = config.get("port", None)
    rank, world_size = setup_distributed(port=config.port)
    config = update_config(config)


    config.exp_path = config.saver.save_dir
    config.save_path = config.saver.save_dir
    config.log_path = config.saver.log_dir
    config.evaluator.eval_dir = config.evaluator.save_dir

    if rank == 0:
        os.makedirs(config.save_path, exist_ok=True)
        os.makedirs(config.log_path, exist_ok=True)

        current_time = get_current_time()
        tb_logger = SummaryWriter(config.log_path + "/events_dec/" + current_time)
        logger = create_logger(
            "global_logger", config.log_path + "/dec_{}.log".format(current_time)
        )
        logger.info("args: {}".format(pprint.pformat(args)))
        logger.info("config: {}".format(pprint.pformat(config)))
    else:
        tb_logger = None

    random_seed = config.get("random_seed", None)
    reproduce = config.get("reproduce", None)
    if random_seed:
        set_random_seed(int(random_seed), reproduce)

    # create model
    model = ModelHelper(config.net)
    model.cuda()
    local_rank = int(os.environ["LOCAL_RANK"])
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    layers = []
    for module in config.net:
        layers.append(module["name"])
    frozen_layers = config.get("frozen_layers", [])
    active_layers = list(set(layers) ^ set(frozen_layers))
    if rank == 0:
        logger.info("layers: {}".format(layers))
        logger.info("active layers: {}".format(active_layers))

    # parameters needed to be updated
    parameters = [
        {"params": getattr(model.module, layer).parameters()} for layer in active_layers
    ]

    for layer in frozen_layers:
        module = getattr(model.module, layer)
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    trainable_pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

 
    optimizer = get_optimizer(parameters, config.trainer.optimizer)
    lr_scheduler = get_scheduler(optimizer, config.trainer.lr_scheduler)

    key_metric = config.evaluator["key_metric"]
    best_metric = 0
    last_epoch = 0

    # load model: auto_resume > resume_model > load_path
    auto_resume = config.saver.get("auto_resume", True)
    resume_model = config.saver.get("resume_model", None)
    load_path = config.saver.get("load_path", None)

    if resume_model and not resume_model.startswith("/"):
        resume_model = os.path.join(config.exp_path, resume_model)
    lastest_model = os.path.join(config.save_path, "ckpt.pth.tar")
    if auto_resume and os.path.exists(lastest_model):
        resume_model = lastest_model
    elif load_path:
        load_state(load_path, model)

    train_loader, val_loader = build_dataloader(config.dataset, distributed=True)

    if args.evaluate:
        validate(val_loader, model)
        return

    criterion = build_criterion(config.criterion)

    for epoch in range(last_epoch, config.trainer.max_epoch):
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)
        last_iter = epoch * len(train_loader)
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            lr_scheduler,
            epoch,
            last_iter,
            tb_logger,
            criterion,
            frozen_layers,
        )
        lr_scheduler.step(epoch)

        if (epoch + 1) % config.trainer.val_freq_epoch == 0:
            ret_metrics = validate(val_loader, model)
            # only ret_metrics on rank0 is not empty
            if rank == 0:
                ret_key_metric = ret_metrics[key_metric]
                is_best = ret_key_metric >= best_metric
                best_metric = max(ret_key_metric, best_metric)
                save_checkpoint(
                    {
                        #"epoch": epoch + 1,
                        #"arch": config.net,
                        "state_dict": model.state_dict(),
                        #"best_metric": best_metric,
                        #"optimizer": optimizer.state_dict(),
                    },
                    is_best,
                    config,
                )


def train_one_epoch(
    train_loader,
    model,
    optimizer,
    lr_scheduler,
    epoch,
    start_iter,
    tb_logger,
    criterion,
    frozen_layers,
):

    batch_time = AverageMeter(config.trainer.print_freq_step)
    data_time = AverageMeter(config.trainer.print_freq_step)
    losses = AverageMeter(config.trainer.print_freq_step)

    model.train()
    # freeze selected layers
    for layer in frozen_layers:
        module = getattr(model.module, layer)
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    logger = logging.getLogger("global_logger")
    end = time.time()

    for i, input in enumerate(train_loader):
        curr_step = start_iter + i
        current_lr = lr_scheduler.get_lr()[0]

        # measure data loading time
        data_time.update(time.time() - end)

        # forward
        outputs = model(input)
        loss = 0
        for name, criterion_loss in criterion.items():
            weight = criterion_loss.weight
            loss += weight * criterion_loss(outputs)
        reduced_loss = loss.clone()
        dist.all_reduce(reduced_loss)
        reduced_loss = reduced_loss / world_size
        losses.update(reduced_loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        # update
        if config.trainer.get("clip_max_norm", None):
            max_norm = config.trainer.clip_max_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)

        if (curr_step + 1) % config.trainer.print_freq_step == 0 and rank == 0:
            tb_logger.add_scalar("loss_train", losses.avg, curr_step + 1)
            tb_logger.add_scalar("lr", current_lr, curr_step + 1)
            tb_logger.flush()

            logger.info(
                "Epoch: [{0}/{1}]\t"
                "Iter: [{2}/{3}]\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                "Loss {loss.val:.5f} ({loss.avg:.5f})\t"
                "LR {lr:.5f}\t".format(
                    epoch + 1,
                    config.trainer.max_epoch,
                    curr_step + 1,
                    len(train_loader) * config.trainer.max_epoch,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=current_lr,
                )
            )

        end = time.time()


def validate(val_loader, model):
    batch_time = AverageMeter(0)
    losses = AverageMeter(0)

    rank = dist.get_rank()

    learn_prameters = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e+6
    fixed_prameters = sum(p.numel() for p in model.parameters() if not p.requires_grad)/1e+6
    all_prameters = sum(p.numel() for p in model.parameters())/1e+6
    
    if rank == 0:
        print(f'learn_prameters:{learn_prameters:.1f}M',
              f'fixed_prameters:{fixed_prameters:.1f}M',
              f'all_prameters:{all_prameters:.1f}M',
            )

    model.eval()
    
    logger = logging.getLogger("global_logger")
    criterion = build_criterion(config.criterion)
    end = time.time()

    if rank == 0:
        os.makedirs(config.evaluator.eval_dir, exist_ok=True)
    # all threads write to config.evaluator.eval_dir, it must be made before every thread begin to write
    dist.barrier()
    
    #nums = 0
    #total_time = 0
    with torch.no_grad():
        for i, input in enumerate(val_loader):
            # forward
        
            #torch.cuda.synchronize()
            #start_time = time.time()

            outputs = model(input)

            #torch.cuda.synchronize()
            #end_time = time.time()
           
            ''' 
            run_time = end_time - start_time
            total_time += run_time
            nums += input['image'].shape[0]
            '''

            dump(config.evaluator.eval_dir, outputs)

            # record loss
            loss = 0
            for name, criterion_loss in criterion.items():
                weight = criterion_loss.weight
                loss += weight * criterion_loss(outputs)
            num = len(outputs["filename"])
            losses.update(loss.item(), num)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % config.trainer.print_freq_step == 0 and rank == 0:
                logger.info(
                    "Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})".format(
                        i + 1, len(val_loader), batch_time=batch_time
                    )
                )
    '''
    print("total_times", total_time)
    print("total_nums", nums)
    print("average times", total_time/nums)
    print("fps", nums/total_time) 
    '''
    # gather final results
    dist.barrier()
    total_num = torch.Tensor([losses.count]).cuda()
    loss_sum = torch.Tensor([losses.avg * losses.count]).cuda()
    dist.all_reduce(total_num, async_op=True)
    dist.all_reduce(loss_sum, async_op=True)
    final_loss = loss_sum.item() / total_num.item()

    ret_metrics = {}  # only ret_metrics on rank0 is not empty
    if rank == 0:
        logger.info("Gathering final results ...")
        # total loss
        logger.info(" * Loss {:.5f}\ttotal_num={}".format(final_loss, total_num.item()))
        #fileinfos, preds, masks = merge_together(config.evaluator.eval_dir)
        fileinfos, rec_preds, ref_preds, masks = merge_together(config.evaluator.eval_dir)
        max_score = rec_preds.max()
        min_score = rec_preds.min() 
        lamda = 0.5  
        # evaluate, log & vis
        if ref_preds.size:
            lamda = 0.50
            fpreds = lamda*rec_preds + (1-lamda)*(max_score+min_score)/2*ref_preds
        else:
            fpreds = rec_preds
            
        ret_metrics = performances(fileinfos, fpreds, masks, config.evaluator.metrics)
        log_metrics(ret_metrics, config.evaluator.metrics)


        shutil.rmtree(config.evaluator.eval_dir)
        
        '''
        if args.evaluate and config.evaluator.get("vis_compound", None):
            visualize_compound(
                fileinfos,
                fpreds,
                masks,
                config.evaluator.vis_compound,
                config.dataset.image_reader,
            )
        if args.evaluate and config.evaluator.get("vis_single", None):
            visualize_single(
                fileinfos,
                fpreds,
                config.evaluator.vis_single,
                config.dataset.image_reader,
            )
        '''
        
    model.train()
    return ret_metrics


if __name__ == "__main__":
    main()
