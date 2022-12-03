import os
import random
import argparse
import lpips
import json
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

import torch.distributed as dist
from pytorch_msssim import ssim
import logger


# custom attack
from utils import get_architecture, Input_diversity
# from vscodearg import add_dict_to_argparser

parser = argparse.ArgumentParser(description='PyTorch Unrestricted Attack')
parser.add_argument('--batch_size', type=int, default=10,metavar='N', help='batch size for attack (default: 30)')
parser.add_argument('--attack_method', type=str, default="GA_TDMI_fgsm_L2")
parser.add_argument('--save_path', type=str, default="tmp/images")
parser.add_argument('--threat_model', type=str, default='L2')
parser.add_argument('--mode', type=str, default="nearest")
parser.add_argument('--loss_fn', type=str, default="ce")#ce
parser.add_argument('--momentum', default=1.0, type=float,help='momentum, (default: 1.0)')
# parser.add_argument('--epsilon', default=20, type=float,help='perturbation, (default: 16)')
# parser.add_argument('--max_epsilon', default=20, type=float,help='perturbation, (default: 16)')
parser.add_argument('--epsilon', default=160, type=float,help='perturbation, (default: 16)')
parser.add_argument('--max_epsilon', default=160, type=float,help='perturbation, (default: 16)')
parser.add_argument('--intervals', default=5, type=int,help='number of intervals')
parser.add_argument('--num_steps', default=10, type=int,help='number of steps in TDMI')
parser.add_argument('--kernel_size', default=5, type=int,help='kernel size of gaussian filter') 
parser.add_argument('--target_id', default=7, type=int,help='InceptionResnetV2 as default')
parser.add_argument('--target_m', default="Standard", type=str,help='robustbench model to be attacked')
parser.add_argument('--source_list', default='2_3_5', type=str)
parser.add_argument('--auxiliary_list', default='1_4_6', type=str)
parser.add_argument('--prob', default=0.7, type=float,help='input diversity prob')
parser.add_argument('--thres', default=0.2, type=float, help='threshold for continue attack')
parser.add_argument('--scale', default=0.1, type=float,help='input diversity scale')
parser.add_argument('--distributed', action='store_true',help='Use multi-processing distributed training to launch')
parser.add_argument("--local_rank", default=0, type=int, help='local rank for DistributedDataParallel')
# add_dict_to_argparser(parser, defaults)

NUM_CLASSES = 1000

MODEL_NAME_DICT = {
    0: "vit_small_patch16_224",
    1: "Rebuffi2021Fixing_28_10_cutmix_ddpm",
    2: "swin_base_patch4_window7_224",
    3: "swsl_resnext101_32x8d",
    4: "Augustin2020Adversarial_34_10_extra",
    5: "swsl_resnet50",
    6: "Sehwag2021Proxy",
    7: "InceptionResnetV2",
    8: "adv_inception_v3",
    9: "ens_adv_inception_resnet_v2",
    10: "Resnet152-DenoiseAll",
    11: "AdvResnet152",
    12: "Resnext101-DenoiseAll"
}

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

size = 32
def main(args):
    init_seeds(cuda_deterministic=True)
    logger.configure(args.save_path, log_suffix="eval")
    # Target model
    if not args.distributed or args.local_rank == 0:
        logger.log("Using {} as target model!".format(MODEL_NAME_DICT[args.target_id]))
    if args.target_m is not None:
        tmp_model = get_architecture(model_name=args.target_m, threat_model=args.threat_model, cifar10=True).cuda().eval()
    elif args.target_id is not None:
        tmp_model = get_architecture(model_name=MODEL_NAME_DICT[args.target_id]).cuda().eval()
    else:
        raise ValueError("plz assign target_m or target_id")
    Target_model = Input_diversity(tmp_model, args=args, num_classes=10, prob=args.prob, mode=args.mode, diversity_scale=args.scale)

    # Source model
    source_id_list = [int(item) for item in args.source_list.split('_')]
    if not args.distributed or args.local_rank == 0:
        logger.log("Source id list: {}".format(source_id_list))
    
    Source_model = Target_model

    # Auxiliary model
    auxiliary_id_list = [int(item) for item in args.auxiliary_list.split('_')]
    if not args.distributed or args.local_rank == 0:
        logger.log("Auxiliary id list: {}".format(auxiliary_id_list))

    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_alex = loss_fn_alex.cuda()

    dataset = CIFAR10(
        root='/root/hhtpro/123/CIFAR10', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]))
    attack_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, drop_last=True, pin_memory=True)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    
    diff_path = args.save_path + "/diff"
    if not os.path.isdir(diff_path):
        os.makedirs(diff_path, exist_ok=True)
    image_path = args.save_path + "/image"
    if not os.path.isdir(image_path):
        os.makedirs(image_path, exist_ok=True)

    natural_err_total, target_err_total = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
    Lp_dis_total = torch.tensor(0.0).cuda()
    lpips_total = torch.tensor(0.0).cuda()
    ssim_total = torch.tensor(0.0).cuda()
    count = 0
    for (img, label) in tqdm(attack_loader):
        img, label = img.cuda(), label.cuda()
        batch_szie = img.shape[0]

        with torch.no_grad():
            out = Source_model(img, diversity=False)

        err = (out.data.max(1)[1] != label.data).float().sum()
        natural_err_total += err
        ts = torch.load(image_path+ "/" + str(count)+'.pt').cuda()
        x_adv = ts[:int(ts.shape[0]/2)]
        img = ts[int(ts.shape[0]/2):]
        # Attack on Target model
        with torch.no_grad():
            out = Target_model(x_adv, diversity=False)
            err_mask = (out.data.max(1)[1] != label.data)
            err_target = err_mask.float().sum()
            target_err_total += err_target
        Lp_dis = ((x_adv - img) ** 2).sum(dim=(1, 2, 3)).sqrt()
        lpips_batch = loss_fn_alex.forward(img, x_adv, normalize=True)[err_mask]
        ssim_batch = ssim(img, x_adv, data_range=1., size_average=False)[err_mask]
        Lp_dis_total += Lp_dis.data.sum()
        lpips_total += lpips_batch.sum()
        ssim_total += ssim_batch.sum()
        count += batch_szie
    torch.cuda.synchronize()
    if not args.distributed or args.local_rank == 0:
        logger.log("Nature Error total: ", natural_err_total)
        logger.log("Target Success total: ", target_err_total)
        logger.log('Avg L2_distance: {}'.format((Lp_dis_total / target_err_total)))
        logger.log('Avg LPIPS dis: {}'.format((lpips_total / target_err_total)))
        logger.log('Avg SSIM dis: {}'.format((ssim_total / target_err_total)))

if __name__ == "__main__":
    opt = parser.parse_args()

    if opt.distributed:
        print("opt.local_rank:{}".format(opt.local_rank))
        torch.cuda.set_device(opt.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        opt.world_size = dist.get_world_size()
        opt.batch_size = int(opt.batch_size / opt.world_size)
    
    if not opt.distributed or opt.local_rank == 0:
        print(opt)

    OUTPUT_DIR=f"CIFAR10-Output_Feature_APR/Ensemble_TID_{opt.target_m}"+\
        f"_SIDLIST_{opt.source_list}_Auxiliary_id_list_{opt.auxiliary_list}"+\
        f"_{opt.attack_method}_{opt.loss_fn}_thres_{opt.thres}_intervals_{opt.intervals}"+\
        f"_steps_{opt.num_steps}_max_eps_{opt.max_epsilon}_eps_{opt.epsilon}_mu_{opt.momentum}"+\
        f"_mode_{opt.mode}_kernel_size_{opt.kernel_size}"
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    opt.save_path = OUTPUT_DIR
    args_path = os.path.join(OUTPUT_DIR, f"exp.json")
    info_json = json.dumps(vars(opt), sort_keys=False, indent=4, separators=(' ', ':'))
    with open(args_path, 'w') as f:
        f.write(info_json)
    main(opt)