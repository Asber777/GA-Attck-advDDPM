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
import logger

os.environ['TORCH_HOME']='~/.cache/torch/'
def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt

# custom attack
from attacks.linf import DI_fgsm, GA_DI_fgsm, TDI_fgsm, GA_TDI_fgsm, TDMI_fgsm, GA_TDMI_fgsm
from attacks.feature import DI_FSA, DMI_FSA, GA_DMI_FSA, GA_DI_FSA, Feature_Adam_Attack
from perceptual_advex.attacks import ReColorAdvAttack

from utils import MyCustomDataset, get_architecture, Input_diversity, MultiEnsemble
from utils import CrossEntropyLoss, MarginLoss
# from vscodearg import add_dict_to_argparser

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
parser.add_argument('--target_m', default="Standard_R50", type=str,help='robustbench model to be attacked')
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
    1: "vit_base_patch16_224",
    2: "swin_base_patch4_window7_224",
    3: "swsl_resnext101_32x8d",
    4: "ssl_resnext50_32x4d",
    5: "swsl_resnet50",
    6: "InceptionV3",
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

def main(args):

    init_seeds(cuda_deterministic=True)
    logger.configure(args.save_path)
    # Target model
    if not args.distributed or args.local_rank == 0:
        logger.log("Using {} as target model!".format(MODEL_NAME_DICT[args.target_id]))
    if args.target_m is not None:
        tmp_model = get_architecture(model_name=args.target_m, threat_model=args.threat_model).cuda().eval()
    elif args.target_id is not None:
        tmp_model = get_architecture(model_name=MODEL_NAME_DICT[args.target_id]).cuda().eval()
    else:
        raise ValueError("plz assign target_m or target_id")
    Target_model = Input_diversity(tmp_model, args=args, num_classes=NUM_CLASSES, prob=args.prob, mode=args.mode, diversity_scale=args.scale)

    # Source model
    source_id_list = [int(item) for item in args.source_list.split('_')]
    if not args.distributed or args.local_rank == 0:
        logger.log("Source id list: {}".format(source_id_list))
    
    # Source_model_list = []
    # for idx in source_id_list:
    #     temp_model = get_architecture(model_name=MODEL_NAME_DICT[idx]).cuda().eval()
    #     Source_model_list.append(temp_model)

    # Source_model = MultiEnsemble(Source_model_list, args=args, num_classes=NUM_CLASSES, prob=args.prob, mode=args.mode, diversity_scale=args.scale)
    Source_model = Target_model

    # Auxiliary model
    auxiliary_id_list = [int(item) for item in args.auxiliary_list.split('_')]
    if not args.distributed or args.local_rank == 0:
        logger.log("Auxiliary id list: {}".format(auxiliary_id_list))

    Auxiliary_model_list = []
    for idx in auxiliary_id_list:
        temp_model = get_architecture(model_name=MODEL_NAME_DICT[idx]).cuda().eval()
        Auxiliary_model_list.append(temp_model)

    Auxiliary_model = MultiEnsemble(Auxiliary_model_list, args=args, num_classes=NUM_CLASSES, prob=args.prob, mode=args.mode, diversity_scale=args.scale)

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
    
    diff_path = args.save_path + "/diff"
    if not os.path.isdir(diff_path):
        os.makedirs(diff_path, exist_ok=True)
    image_path = args.save_path + "/image"
    if not os.path.isdir(image_path):
        os.makedirs(image_path, exist_ok=True)

    # save budgrt with a dict
    img_budget = {}

    natural_err_total, target_err_total = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
    Lp_dis_total = torch.tensor(0.0).cuda()
    eps_total = torch.tensor(0.0).cuda()
    quality_level = torch.tensor(0.0).cuda()
    lpips_total = torch.tensor(0.0).cuda()
    ssim_total = torch.tensor(0.0).cuda()

    count = 0
    time_start = time.time()
    if not args.distributed or args.local_rank == 0:
        logger.log("Starting time counting: {}\n".format(time_start))

    # loss_fn
    if args.loss_fn == "ce":
        loss_fn = CrossEntropyLoss()
    elif args.loss_fn == "margin":
        loss_fn = MarginLoss()
    else:
        raise Exception("invalid loss function!")

    # distance metric
    if "fgsm" in args.attack_method:
        reward_fn = lambda x: 1.0 / (x * 255.0)
    else:
        reward_fn = lambda x: 1.0 / (x)
        
    for (img, label, img_name) in attack_loader:
        img, label = img.cuda(), label.cuda()
        batch_szie = img.shape[0]

        with torch.no_grad():
            out = Source_model(img, diversity=False)

        err = (out.data.max(1)[1] != label.data).float().sum()
        natural_err_total += err

        if args.attack_method == "TDI_fgsm":
            x_adv, budget = TDI_fgsm(Source_model, img.clone(), label.clone(), args, loss_fn)
        elif args.attack_method == "GA_TDI_fgsm":
            x_adv, budget = GA_TDI_fgsm(Source_model, Auxiliary_model, img.clone(), label.clone(), args, loss_fn)
        elif args.attack_method == "DI_fgsm":
            x_adv, budget = DI_fgsm(Source_model, img.clone(), label.clone(), args, loss_fn)
        elif args.attack_method == "GA_DI_fgsm":
            x_adv, budget = GA_DI_fgsm(Source_model, Auxiliary_model, img.clone(), label.clone(), args, loss_fn)
        elif args.attack_method == "TDMI_fgsm":
            x_adv, budget = TDMI_fgsm(Source_model, img.clone(), label.clone(), args, loss_fn)
        elif args.attack_method == "GA_TDMI_fgsm":
            x_adv, budget = GA_TDMI_fgsm(Source_model, Auxiliary_model, img.clone(), label.clone(), args, loss_fn)


        elif args.attack_method == "FSA":
            x_adv = Feature_Adam_Attack(Source_model, img.clone(), label.clone(), args, loss_fn, diversity=False)
            budget = args.epsilon * torch.ones([x_adv.shape[0]])
        elif args.attack_method == "DI_FSA":
            x_adv, budget= DI_FSA(Source_model, img.clone(), label.clone(), args, loss_fn)
        elif args.attack_method == "GA_DI_FSA":
            x_adv, budget = GA_DI_FSA(Source_model, Auxiliary_model, img.clone(), label.clone(), args, loss_fn)
        elif args.attack_method == "DMI_FSA":
            x_adv, budget = DMI_FSA(Source_model, img.clone(), label.clone(), args, loss_fn)
        elif args.attack_method == "GA_DMI_FSA":
            x_adv, budget = GA_DMI_FSA(Source_model, Auxiliary_model, img.clone(), label.clone(), args, loss_fn)
        elif args.attack_method == "recolor":
            attack = ReColorAdvAttack(model=Source_model, bound=args.epsilon, num_iterations=args.num_steps)
            x_adv = attack(img.clone(), label.clone()).detach()
            budget = args.epsilon * torch.ones([x_adv.shape[0]])
        else:
            raise Exception("invalid attack method !")

        count += batch_szie

        # Attack on Target model
        with torch.no_grad():
            out = Target_model(x_adv, diversity=False)
            err_mask = (out.data.max(1)[1] != label.data)
            err_target = err_mask.float().sum()
            target_err_total += err_target
        Lp_dis = torch.abs(x_adv - img).reshape(len(x_adv), -1).max(dim = -1)[0][err_mask]
        distance_batch = budget[err_mask]
        lpips_batch = loss_fn_alex.forward(img, x_adv, normalize=True)[err_mask]
        ssim_batch = ssim(img, x_adv, data_range=1., size_average=False)[err_mask]
        
        eps_total += distance_batch.sum()
        Lp_dis_total += Lp_dis.data.sum()
        batch_score = reward_fn(distance_batch)
        quality_level += batch_score.sum() 
        lpips_total += lpips_batch.sum()
        ssim_total += ssim_batch.sum()
        logger.log("Attacked: {}, Batch size: {},\
                Image name: {}, Nature error: {}, \
                Target error: {}, Target error mask: {}, \
                budgets: {}, Batch distance Max: {}, \
                Avg: {}, Avg reward: {}, Lp: {}".format(count, batch_szie,
                img_name, err.item(), err_target.item(), err_mask,
                budget, distance_batch.max().item(), 
                distance_batch.mean().item(), batch_score.mean().item(), 
                Lp_dis.data.sum()))

        budget_cpu = budget.detach()
        budget_cpu[~err_mask] *= -1
        budget_cpu = budget_cpu.cpu().numpy()

        for i in range(batch_szie):
            x_adv_cpu = x_adv[i, :, :, :].cpu()
            torch.save(x_adv_cpu, image_path+ "/" + img_name[i][:-4]+'.pt')
            img_adv = transforms.ToPILImage()(x_adv_cpu).convert('RGB')
            img_adv.save(image_path + "/" + img_name[i])

            x_cpu_numpy = 255.0 * img[i, :, :, :].cpu().numpy().transpose(1,2,0)
            x_adv_numpy = 255.0 * x_adv_cpu.numpy().transpose(1,2,0)
            save_diff = np.concatenate((x_cpu_numpy, x_adv_numpy, 255.0 * normalize(x_adv_numpy - x_cpu_numpy)))
            save_diff = np.reshape(save_diff, newshape=[299*3, 299, 3])
            save_diff = save_diff.astype(np.uint8)
            Image.fromarray(save_diff, mode='RGB').save(diff_path + "/" + img_name[i])

            # budget
            img_budget[img_name[i]] = budget_cpu[i]


    if args.distributed:
        eps_total = reduce_tensor(eps_total.data, args.world_size)
        Lp_dis_total = reduce_tensor(Lp_dis_total.data, args.world_size)
        quality_level = reduce_tensor(quality_level.data, args.world_size)
        natural_err_total = reduce_tensor(natural_err_total.data, args.world_size)
        pgd_err_total = reduce_tensor(pgd_err_total.data, args.world_size)
        target_err_total = reduce_tensor(target_err_total.data, args.world_size)

        # save budget localrank-wise
        budget_path = os.path.join(args.save_path, str(args.local_rank) + "_rank_budget.npy")
        np.save(budget_path, img_budget)
    else:
        budget_path = os.path.join(args.save_path, "budget.npy")
        np.save(budget_path, img_budget)

    torch.cuda.synchronize()
    
    time_end = time.time()
    if not args.distributed or args.local_rank == 0:
        logger.log('time cost', time_end-time_start, 's')
        logger.log("Nature Error total: ", natural_err_total)
        logger.log("Target Success total: ", target_err_total)
        if "fgsm" in args.attack_method:
            logger.log('Avg distance of successfully transferred: {}'.format((eps_total / target_err_total) * 255.0))
        else:
            logger.log('Avg distance of successfully transferred: {}'.format((eps_total / target_err_total)))   
        logger.log('Avg Lp_distance: {}'.format((Lp_dis_total / target_err_total)))
        logger.log('Avg perturbation reward: {}'.format((quality_level / target_err_total)))
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

    OUTPUT_DIR=f"Output_Feature_APR_FGSM/Ensemble_TID_{opt.target_m}"+\
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