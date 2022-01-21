import torch
from itertools import product
from core.norm_dist import norm_dist_dropout
from core.test import test_utils
from torch.nn.functional import unfold
import random


def sample(batch, group, channel, q):
    while True:
        count = (torch.rand(channel) < q).sum()
        if count > 0:
            break
    indices = torch.topk(torch.rand(batch, group, channel), dim=-1, k=count, sorted=False)[1].int()
    return torch.sort(indices)[0]

def test_norm_dist_case(B, CI, CO, H, W, K, G, p, backward_input, backward_weight, drop_ci, drop_co):
    input = torch.randn(B, CI, H, W).cuda()
    weight = torch.randn(CO, CI // G, K, K).cuda()
    input = unfold(input, K, padding=K // 2)
    weight = weight.view(weight.size(0), -1)

    # print('B=%d, CI=%d, CO=%d, H=%d, W=%d, K=%d, G=%d, p=%f' % (B, CI, CO, H, W, K, G, p), '\n')

    if drop_ci:
        prob = random.random() * 0.75 + 0.25
        w_ci_index = sample((B - 1) // 32 + 1, G, CI // G * K * K, prob).cuda()
        batch_index = torch.arange(B, dtype=torch.long).view(-1, 1, 1).cuda()
        group_index = torch.arange(G, dtype=torch.long).view(1, -1, 1).cuda()
        input = input.view(B, G, -1, input.size(-1))[batch_index, group_index, w_ci_index[0].long()]
        input = input.view(B, -1, input.size(-1))
    else:
        w_ci_index = None
    if drop_co:
        prob = random.random() * 0.75 + 0.25
        w_co_index = sample((B - 1) // 32 + 1, G, CO // G, prob).cuda()
    else:
        w_co_index = None

    if backward_input:
        input = torch.nn.Parameter(input)
    if backward_weight:
        weight = torch.nn.Parameter(weight)
    res = test_utils.test(norm_dist_dropout, input, weight, w_index_ci=w_ci_index, w_index_co=w_co_index, p=p, groups=G)

    if not test_utils.check(res, 5e-3):
        print_res = test_utils.truncate(res, 7)
        print('Warning!\n',
              'B=%d, CI=%d, CO=%d, H=%d, W=%d, K=%d, G=%d, p=%f'%(B, CI, CO, H, W, K, G, p), '\n',
              'diff_output', print_res[0], '\n',
              'diff_grad_input', print_res[1], '\n',
              'diff_grad_weight', print_res[2])
        print('input', input)
        print('weight', weight)
        print('w_ci', w_ci_index)
        print('w_co', w_co_index)
        exit()
    return res

def test_norm_dist_all(**params):
    Bs = (1, 2, 3, 8, 12, 16, 20, 32, 48, 64, 82)
    CIs = (1, 2, 3, 8, 16, 20, 28, 32, 40, 48, 64, 82)
    COs = (1, 2, 3, 8, 16, 20, 28, 32, 40, 48, 64, 82)
    HWs = [(1, 1), (2, 1), (3, 3), (4, 4), (3, 6), (4, 8), (16, 16), (18, 18), (28, 28)]
    Ks = (1, 2, 3)
    Gs = (1, 2, 3)
    print('Start testing: ', str(params))
    worst_res = None
    for B, CI, CO, (H, W), K, G in product(Bs, CIs, COs, HWs, Ks, Gs):
        if B * H * W > 192: continue
        if (CO >= 64 and CI * K * K >= 64) or (CO >= 64 and B * H * W >= 64) or (CI * K * K >= 64 and B * H * W >= 64):
            continue
        if CI % G != 0 or CO % G != 0:
            continue
        res = test_norm_dist_case(B, CI, CO, H, W, K, G, **params)
        worst_res = test_utils.merge(worst_res, res)
    return worst_res

def test_norm_dist_speed(B, CI, CO, H, W, K, G, p, backward, ext):
    input = torch.randn(B, CI, H, W).cuda()
    weight = torch.randn(CO, CI // G, K, K).cuda()
    input = unfold(input, K, padding=K // 2)
    weight = weight.view(weight.size(0), -1)
    if backward:
        input = torch.nn.Parameter(input)
        weight = torch.nn.Parameter(weight)
    print('Start testing: B=%d, CI=%d, CO=%d, H=%d, W=%d, K=%d, G=%d, p=%f, backward='%(B, CI, CO, H, W, K, G, p), backward)
    return test_utils.test_speed(norm_dist, input, weight, p=p, groups=G)

def test_norm_dist_dropout():
    torch.manual_seed(2021)
    print(test_utils.truncate(test_norm_dist_all(p=7, backward_input=True, backward_weight=True, drop_ci=True, drop_co=True), 8))
    print(test_utils.truncate(test_norm_dist_all(p=8, backward_input=True, backward_weight=True, drop_ci=False, drop_co=False), 8))
    print(test_utils.truncate(test_norm_dist_all(p=8, backward_input=True, backward_weight=True, drop_ci=True, drop_co=False), 8))
    print(test_utils.truncate(test_norm_dist_all(p=8, backward_input=True, backward_weight=True, drop_ci=False, drop_co=True), 8))
    print(test_utils.truncate(test_norm_dist_all(p=20, backward_input=True, backward_weight=True, drop_ci=True, drop_co=True), 8))
    print(test_utils.truncate(test_norm_dist_all(p=20, backward_input=True, backward_weight=False, drop_ci=True, drop_co=True), 8))
    print(test_utils.truncate(test_norm_dist_all(p=float('inf'), backward_input=True, backward_weight=True, drop_ci=True, drop_co=True), 8))
    print(test_utils.truncate(test_norm_dist_all(p=float('inf'), backward_input=True, backward_weight=False, drop_ci=True, drop_co=True), 8))
    print(test_utils.truncate(test_norm_dist_all(p=float('inf'), backward_input=True, backward_weight=True, drop_ci=True, drop_co=False), 8))
    print(test_utils.truncate(test_norm_dist_all(p=float('inf'), backward_input=True, backward_weight=False, drop_ci=False, drop_co=True), 8))
    print(test_utils.truncate(test_norm_dist_all(p=float('inf'), backward_input=True, backward_weight=True, drop_ci=False, drop_co=True), 8))
    print(test_utils.truncate(test_norm_dist_all(p=float('inf'), backward_input=True, backward_weight=False, drop_ci=True, drop_co=False), 8))

    # print('Testing speed:')
    # for p in (8, 20, float('inf')):
    #     for backward in (False, True):
    #         print(test_norm_dist_speed(512, 64, 64, 32, 32, 3, 1, p, backward, ext=False))
    #         print(test_norm_dist_speed(512, 128, 128, 16, 16, 3, 1, p, backward, ext=False))
    #         print(test_norm_dist_speed(512, 4096, 4096, 1, 1, 1, 1, p, backward, ext=False))
    #
    # for p in (8, 20):
    #     for backward in (False, True):
    #         print(test_norm_dist_speed(512, 64, 64, 32, 32, 3, 1, p, backward, ext=True))
    #         print(test_norm_dist_speed(512, 128, 128, 16, 16, 3, 1, p, backward, ext=True))
    #         print(test_norm_dist_speed(512, 4096, 4096, 1, 1, 1, 1, p, backward, ext=True))
