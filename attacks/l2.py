import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

from .utils import get_kernel

def clip_L2(x, eps):
    norm = ((x ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
    factor = torch.min(eps / norm, torch.ones_like(norm))
    return factor * x

def PGD20_L2(model, x_nature, y, args, loss_fn):
    eps = args.epsilon / 255.0
    alpha = 1.25 * eps / args.num_steps
    delta = torch.zeros_like(x_nature).cuda()
    delta = Variable(delta.detach(), requires_grad=True)

    for _ in range(args.num_steps):
        delta.requires_grad_()
        adv = x_nature + delta
        adv = torch.clamp(adv, 0, 1)

        with torch.enable_grad():
            logits = model(adv, diversity=False)
            loss = loss_fn(logits, y)

        grad = torch.autograd.grad(loss.sum(), [delta])[0].detach()
        grad = grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
        delta = Variable(delta.data + alpha * grad, requires_grad=True)
        # delta = Variable(torch.clamp(delta.data, -eps, eps),requires_grad=True)
        delta = clip_L2(delta, eps)

    X_pgd = Variable(x_nature + delta, requires_grad=False)
    X_pgd = Variable(torch.clamp(X_pgd, 0, 1), requires_grad=False)
    return X_pgd



def GA_TDMI_fgsm_L2(model, eval_model, x_nature, y, args, loss_fn):
    batch_size = x_nature.shape[0]
    g = torch.zeros_like(x_nature)
    delta = torch.zeros_like(x_nature).cuda()
    eps_list = [(idx + 1) * (args.max_epsilon / args.intervals) for idx in range(args.intervals)]
    mask = torch.ones((batch_size, )).bool()
    num_steps = args.num_steps
    for eps in eps_list:
        eps /= 255.0
        step_size = 1.25 * eps / num_steps
        delta = Variable(delta.data, requires_grad=True)

        for _ in range(num_steps):
            delta.requires_grad_()
            adv = x_nature + delta
            adv = torch.clamp(adv, 0, 1)
            with torch.enable_grad():
                ensem_logits = model(adv, diversity=True)
                loss = loss_fn(ensem_logits, y)
            PGD_grad = torch.autograd.grad(loss.sum(), [delta])[0].detach()
            PGD_grad = F.conv2d(PGD_grad, weight=get_kernel(args.kernel_size), stride=(
                1, 1), groups=3, padding=(args.kernel_size - 1) // 2)
            PGD_noise = PGD_grad / \
                torch.abs(PGD_grad).mean(dim=(1, 2, 3), keepdim=True)
            g[mask] = g[mask] * args.momentum + PGD_noise[mask]
            g_ =  g / ((g ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
            delta = Variable(delta.data + step_size * g_, requires_grad=True)
            delta = clip_L2(delta, eps)

        g.zero_()
        with torch.no_grad():
            tmp = x_nature + delta
            tmp = torch.clamp(tmp, 0, 1)
            output = eval_model(tmp, diversity=False).detach()
        prob = F.softmax(output, dim=1)
        conf = prob[np.arange(batch_size), y.long()]
        mask = (conf >= args.thres)

        # early stopping
        if mask.sum() == 0:
            break

    X_pgd = Variable(x_nature + delta, requires_grad=False)
    X_pgd = Variable(torch.clamp(X_pgd, 0, 1), requires_grad=False)
    budget = torch.abs(X_pgd - x_nature).reshape(batch_size, -1).max(dim = -1)[0]
    return X_pgd, budget