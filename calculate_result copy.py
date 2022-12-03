import os
import time
import random
import argparse
import lpips
import json
import numpy as np
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

import torch.distributed as dist
from pytorch_msssim import ssim
from tqdm import tqdm
import logger

from utils import MyCustomDataset, get_architecture, Input_diversity, MultiEnsemble


parser = argparse.ArgumentParser(description='PyTorch Unrestricted Attack')
parser.add_argument('--batch_size', type=int, default=5,metavar='N', help='batch size for attack (default: 30)')
parser.add_argument('--attack_method', type=str, default="GA_TDMI_fgsm")
parser.add_argument('--save_path', type=str, default="tmp/images")
parser.add_argument('--threat_model', type=str, default='Linf')
parser.add_argument('--mode', type=str, default="nearest")
parser.add_argument('--loss_fn', type=str, default="ce")#ce
parser.add_argument('--momentum', default=1.0, type=float,help='momentum, (default: 1.0)')
parser.add_argument('--epsilon', default=20, type=float,help='perturbation, (default: 16)')
parser.add_argument('--max_epsilon', default=20, type=float,help='perturbation, (default: 16)')
parser.add_argument('--intervals', default=5, type=int,help='number of intervals')
parser.add_argument('--num_steps', default=10, type=int,help='number of steps in TDMI')
parser.add_argument('--kernel_size', default=5, type=int,help='kernel size of gaussian filter') 
parser.add_argument('--target_id', default=7, type=int,help='InceptionResnetV2 as default')
parser.add_argument('--target_m', default="Salman2020Do_R50", type=str,help='robustbench model to be attacked')
parser.add_argument('--source_list', default='2_3_5', type=str)
parser.add_argument('--auxiliary_list', default='1_4_6', type=str)
parser.add_argument('--prob', default=0.7, type=float,help='input diversity prob')
parser.add_argument('--thres', default=0.2, type=float, help='threshold for continue attack')
parser.add_argument('--scale', default=0.1, type=float,help='input diversity scale')
parser.add_argument('--distributed', action='store_true',help='Use multi-processing distributed training to launch')
parser.add_argument("--local_rank", default=0, type=int, help='local rank for DistributedDataParallel')
# add_dict_to_argparser(parser, defaults)

NUM_CLASSES = 1000
def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html 
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def normalize(item):
    max = item.max()
    min = item.min()
    return (item - min) / (max - min)

def main(args):
    init_seeds(cuda_deterministic=True)
    logger.configure(args.save_path)
    # Target model
    if args.target_m is not None:
        tmp_model = get_architecture(model_name=args.target_m, threat_model=args.threat_model).cuda().eval()
    Target_model = Input_diversity(tmp_model, args=args, num_classes=NUM_CLASSES, 
        prob=args.prob, mode=args.mode, diversity_scale=args.scale)

    Source_model = Target_model

    loader = MyCustomDataset(img_path="data/images")
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(loader)
    else:
        sampler = torch.utils.data.SequentialSampler(loader)

    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_alex = loss_fn_alex.cuda()

    attack_loader = torch.utils.data.DataLoader(dataset=loader,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                sampler=sampler, num_workers=8, pin_memory=True)
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    
    image_path = args.save_path + "/image"
    if not os.path.isdir(image_path):
        os.makedirs(image_path, exist_ok=True)

    natural_err_total, target_err_total = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
    Lp_dis_total = torch.tensor(0.0).cuda()
    lpips_total = torch.tensor(0.0).cuda()
    ssim_total = torch.tensor(0.0).cuda()

    count = 0
    time_start = time.time()
    if not args.distributed or args.local_rank == 0:
        logger.log("Starting time counting: {}\n".format(time_start))


    for (img, label, img_name) in tqdm(attack_loader):
        img, label = img.cuda(), label.cuda()
        batch_szie = img.shape[0]

        with torch.no_grad():
            out = Source_model(img, diversity=False)

        err = (out.data.max(1)[1] != label.data).float().sum()
        natural_err_total += err

        x_adv = torch.zeros_like(img)
        for i in range(batch_szie):
            x_adv[i] = torch.load(image_path+ "/" + img_name[i][:-4]+'.pt').cuda()
        count += batch_szie
        # Attack on Target model
        with torch.no_grad():
            out = Target_model(x_adv, diversity=False)
            err_mask = (out.data.max(1)[1] != label.data)
            err_target = err_mask.float().sum()
            target_err_total += err_target
        Lp_dis = torch.abs(x_adv - img).reshape(len(x_adv), -1).max(dim = -1)[0][err_mask]
        lpips_batch = loss_fn_alex.forward(img, x_adv, normalize=True)[err_mask]
        ssim_batch = ssim(img, x_adv, data_range=1., size_average=False)[err_mask]
        Lp_dis_total += Lp_dis.data.sum()
        lpips_total += lpips_batch.sum()
        ssim_total += ssim_batch.sum()

    if not args.distributed or args.local_rank == 0:
        logger.log("Nature Error total: ", natural_err_total)
        logger.log("Target Success total: ", target_err_total)
        logger.log('Avg Lp_distance: {}'.format((Lp_dis_total / target_err_total)))
        logger.log('Avg LPIPS dis: {}'.format((lpips_total / target_err_total)))
        logger.log('Avg SSIM dis: {}'.format((ssim_total / target_err_total)))

if __name__ == "__main__":
    opt = parser.parse_args()
    OUTPUT_DIR=f"Output_Feature_APR_FGSM/Ensemble_TID_{opt.target_m}"+\
        f"_SIDLIST_{opt.source_list}_Auxiliary_id_list_{opt.auxiliary_list}"+\
        f"_{opt.attack_method}_{opt.loss_fn}_thres_{opt.thres}_intervals_{opt.intervals}"+\
        f"_steps_{opt.num_steps}_max_eps_{opt.max_epsilon}_eps_{opt.epsilon}_mu_{opt.momentum}"+\
        f"_mode_{opt.mode}_kernel_size_{opt.kernel_size}"
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    opt.save_path = OUTPUT_DIR
    main(opt)