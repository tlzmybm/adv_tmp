import torch
from itertools import product
from core.batch_norm import bound_batch_norm
from core.test import test_utils

def test_bound_batch_norm_case(B, C, H, W, backward_input, backward_weight, training, zero_init=False):
    r_mean = torch.randn(C).cuda()
    r_var = (0.5 + torch.randn(C).abs()).cuda()
    momentum = 0.1
    eps = 1e-4
    worst_res = None
    for _ in range(5):
        if zero_init:
            inputL = torch.zeros(B, C, H * W).cuda()
            inputU = torch.zeros(B, C, H * W).cuda()
            weight = torch.zeros(C).cuda()
            bias = torch.zeros(C).cuda()
        else:
            inputL = (torch.randn(B, C, H * W) * torch.randn(1, C, 1).clamp(min=0.25, max=4.0) + torch.randn(1, C, 1)).cuda()
            inputU = (torch.randn(B, C, H * W) * torch.randn(1, C, 1).clamp(min=0.25, max=4.0) + torch.randn(1, C, 1)).cuda()
            inputL, inputU = torch.minimum(inputL, inputU), torch.maximum(inputL, inputU)
            weight = torch.randn(C).cuda()
            bias = torch.randn(C).cuda()
        if backward_input:
            inputL = torch.nn.Parameter(inputL)
            inputU = torch.nn.Parameter(inputU)
        if backward_weight:
            weight = torch.nn.Parameter(weight)
            bias = torch.nn.Parameter(bias)
        res = test_utils.test(bound_batch_norm, inputL, inputU, weight, bias, r_mean, r_var,
                              momentum=momentum, eps=eps, training=training)
        if not test_utils.check(res, 5e-3):
            print_res = test_utils.truncate(res, 7)
            print('Warning!\n',
                  'B=%d, C=%d, H=%d, W=%d'%(B, C, H, W), '\n',
                  'diff_outputL', print_res[0], '\n',
                  'diff_outputU', print_res[1], '\n',
                  'diff_grad_inputL', print_res[2], '\n',
                  'diff_grad_inputU', print_res[3], '\n',
                  'diff_grad_weight', print_res[4], '\n',
                  'diff_grad_bias', print_res[5], '\n',
                  'var', ((inputL + inputU) * 0.5).double().var(dim=[0, 2]).min().item())
        worst_res = test_utils.merge(worst_res, res)
    return worst_res

def test_bound_batch_norm_all(**params):
    Bs = (1, 2, 3, 8, 12, 16, 20, 32, 48, 64, 82, 100, 120)
    Cs = (1, 2, 3, 8, 16, 20, 28, 32, 40, 48, 64, 82, 100, 120)
    HWs = [(1, 1), (2, 1), (3, 3), (4, 4), (3, 6), (4, 8), (16, 16), (18, 18), (28, 28)]
    print('Start testing: ', str(params))
    worst_res = None
    for B, C, (H, W) in product(Bs, Cs, HWs):
        if B * H * W <= 3:
            continue
        res = test_bound_batch_norm_case(B, C, H, W, **params)
        worst_res = test_utils.merge(worst_res, res)
    return worst_res

def test_bound_batch_norm_speed(B, C, H, W, backward, training):
    r_mean = torch.zeros(C).cuda()
    r_var = torch.ones(C).cuda()
    inputL = torch.randn(B, C, H * W).cuda() * torch.randn(1, C, 1).cuda() + torch.randn(1, C, 1).cuda()
    inputU = torch.randn(B, C, H * W).cuda() * torch.randn(1, C, 1).cuda() + torch.randn(1, C, 1).cuda()
    inputL, inputU = torch.minimum(inputL, inputU), torch.maximum(inputL, inputU)
    weight = torch.randn(C).cuda()
    bias = torch.randn(C).cuda()
    momentum = 0.1
    eps = 1e-5
    if backward:
        inputL = torch.nn.Parameter(inputL)
        inputU = torch.nn.Parameter(inputU)
        weight = torch.nn.Parameter(weight)
        bias = torch.nn.Parameter(bias)
    print('Start testing: B=%d, C=%d, H=%d, W=%d'%(B, C, H, W))
    return test_utils.test_speed(bound_batch_norm, inputL, inputU, weight, bias, r_mean, r_var,
                                 momentum=momentum, eps=eps, training=training)

def test_bound_batch_norm():
    torch.manual_seed(2021)
    print(test_utils.truncate(test_bound_batch_norm_all(backward_input=False, backward_weight=False, training=False, zero_init=False), 8))
    print(test_utils.truncate(test_bound_batch_norm_all(backward_input=True, backward_weight=False, training=False, zero_init=False), 8))
    print(test_utils.truncate(test_bound_batch_norm_all(backward_input=True, backward_weight=True, training=False, zero_init=False), 8))
    print(test_utils.truncate(test_bound_batch_norm_all(backward_input=True, backward_weight=False, training=True, zero_init=False), 8))
    print(test_utils.truncate(test_bound_batch_norm_all(backward_input=True, backward_weight=True, training=True, zero_init=False), 8))
    print(test_utils.truncate(test_bound_batch_norm_all(backward_input=True, backward_weight=True, training=True, zero_init=True), 8))
    print('Testing speed:')
    print(test_bound_batch_norm_speed(512, 64, 32, 32, True, True))
    print(test_bound_batch_norm_speed(512, 128, 16, 16, True, True))
    print(test_bound_batch_norm_speed(512, 4096, 1, 1, True, True))