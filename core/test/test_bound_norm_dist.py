import torch
from itertools import product
from core.norm_dist import bound_norm_dist, ext_bound_norm_dist
from core.test import test_utils
from torch.nn.functional import unfold

def test_bound_norm_dist_case(B, CI, CO, H, W, K, G, p, backward_input, backward_weight, ext, zero_init=False):
    if zero_init:
        inputL = torch.zeros(B, CI, H, W).cuda()
        inputU = torch.zeros(B, CI, H, W).cuda()
        weight = torch.zeros(CO, CI // G, K, K).cuda()
    else:
        inputL = torch.randn(B, CI, H, W).cuda()
        inputU = torch.randn(B, CI, H, W).cuda()
        weight = torch.randn(CO, CI // G, K, K).cuda()
        inputL, inputU = torch.minimum(inputL, inputU), torch.maximum(inputL, inputU)
    inputL = unfold(inputL, K, padding=K // 2)
    inputU = unfold(inputU, K, padding=K // 2)
    weight = weight.view(weight.size(0), -1)
    if backward_input:
        inputL = torch.nn.Parameter(inputL)
        inputU = torch.nn.Parameter(inputU)
    if backward_weight:
        weight = torch.nn.Parameter(weight)
    if ext:
        res = test_utils.test(ext_bound_norm_dist, inputL, inputU, weight, p=p, groups=G)
    else:
        res = test_utils.test(bound_norm_dist, inputL, inputU, weight, p=p, groups=G)
    if not test_utils.check(res, 5e-3):
        print_res = test_utils.truncate(res, 7)
        print('Warning!\n',
              'B=%d, CI=%d, CO=%d, H=%d, W=%d, K=%d, G=%d, p=%f'%(B, CI, CO, H, W, K, G, p), '\n',
              'diff_outputL', print_res[0], '\n',
              'diff_outputU', print_res[1], '\n',
              'diff_grad_inputL', print_res[2], '\n',
              'diff_grad_inputU', print_res[3], '\n',
              'diff_grad_weight', print_res[4], '\n')
        exit()
    return res

def test_bound_norm_dist_all(**params):
    Bs = (1, 2, 3, 8, 12, 16, 20, 32, 48, 64, 82, 100, 120)
    CIs = (1, 2, 3, 8, 16, 20, 28, 32, 40, 48, 64, 82, 100, 120)
    COs = (1, 2, 3, 8, 16, 20, 28, 32, 40, 48, 64, 82, 100, 120)
    HWs = [(1, 1), (2, 1), (3, 3), (4, 4), (3, 6), (4, 8), (16, 16), (18, 18), (28, 28)]
    Ks = (1, 2, 3)
    Gs = (1, 2, 3, 4)
    print('Start testing: ', str(params))
    worst_res = None
    for B, CI, CO, (H, W), K, G in product(Bs, CIs, COs, HWs, Ks, Gs):
        if B * H * W > 192: continue
        if (CO >= 64 and CI * K * K >= 64) or (CO >= 64 and B * H * W >= 64) or (CI * K * K >= 64 and B * H * W >= 64):
            continue
        if CI % G != 0 or CO % G != 0:
            continue
        res = test_bound_norm_dist_case(B, CI, CO, H, W, K, G, **params)
        worst_res = test_utils.merge(worst_res, res)
    return worst_res

def test_bound_norm_dist_speed(B, CI, CO, H, W, K, G, p, backward, ext):
    inputL = torch.randn(B, CI, H, W).cuda()
    inputU = torch.randn(B, CI, H, W).cuda()
    weight = torch.randn(CO, CI // G, K, K).cuda()
    inputL = unfold(inputL, K, padding=K // 2)
    inputU = unfold(inputU, K, padding=K // 2)
    weight = weight.view(weight.size(0), -1)
    if backward:
        inputL = torch.nn.Parameter(inputL)
        inputU = torch.nn.Parameter(inputU)
    print('Start testing: B=%d, CI=%d, CO=%d, H=%d, W=%d, K=%d, G=%d, p=%f, backward='%(B, CI, CO, H, W, K, G, p), backward)
    if ext:
        return test_utils.test_speed(ext_bound_norm_dist, inputL, inputU, weight, p=p, groups=G)
    else:
        return test_utils.test_speed(bound_norm_dist, inputL, inputU, weight, p=p, groups=G)

def test_bound_norm_dist():
    torch.manual_seed(2021)
    # print(test_utils.truncate(test_bound_norm_dist_all(ext=False, p=7, backward_input=True, backward_weight=True, zero_init=False), 8))
    # print(test_utils.truncate(test_bound_norm_dist_all(ext=False, p=8, backward_input=True, backward_weight=True, zero_init=False), 8))
    # print(test_utils.truncate(test_bound_norm_dist_all(ext=False, p=20, backward_input=True, backward_weight=True, zero_init=False), 8))
    # print(test_utils.truncate(test_bound_norm_dist_all(ext=False, p=20, backward_input=True, backward_weight=False, zero_init=False), 8))
    # print(test_utils.truncate(test_bound_norm_dist_all(ext=False, p=8, backward_input=True, backward_weight=True, zero_init=True), 8))
    # print(test_utils.truncate(test_bound_norm_dist_all(ext=False, p=20, backward_input=True, backward_weight=True, zero_init=True), 8))
    # print(test_utils.truncate(test_bound_norm_dist_all(ext=False, p=float('inf'), backward_input=False, backward_weight=False, zero_init=False), 8))
    # print(test_utils.truncate(test_bound_norm_dist_all(ext=False, p=float('inf'), backward_input=False, backward_weight=False, zero_init=True), 8))
    # print(test_utils.truncate(test_bound_norm_dist_all(ext=False, p=float('inf'), backward_input=True, backward_weight=True, zero_init=False), 8))
    # print(test_utils.truncate(test_bound_norm_dist_all(ext=False, p=float('inf'), backward_input=True, backward_weight=False, zero_init=False), 8))

    # print(test_utils.truncate(test_bound_norm_dist_all(ext=True, p=7, backward_input=True, backward_weight=True, zero_init=False), 8))
    # print(test_utils.truncate(test_bound_norm_dist_all(ext=True, p=8, backward_input=True, backward_weight=True, zero_init=False), 8))
    print(test_utils.truncate(test_bound_norm_dist_all(ext=True, p=20, backward_input=True, backward_weight=True, zero_init=False), 8))

    print('Testing speed:')
    for p in (8, 20, float('inf')):
        for backward in (False, True):
            print(test_bound_norm_dist_speed(512, 64, 64, 32, 32, 3, 1, p, backward, ext=False))
            print(test_bound_norm_dist_speed(512, 128, 128, 16, 16, 3, 1, p, backward, ext=False))
            print(test_bound_norm_dist_speed(512, 4096, 4096, 1, 1, 1, 1, p, backward, ext=False))
    for p in (8, 20):
        for backward in (False, True):
            print(test_bound_norm_dist_speed(512, 64, 64, 32, 32, 3, 1, p, backward, ext=True))
            print(test_bound_norm_dist_speed(512, 128, 128, 16, 16, 3, 1, p, backward, ext=True))
            print(test_bound_norm_dist_speed(512, 4096, 4096, 1, 1, 1, 1, p, backward, ext=True))