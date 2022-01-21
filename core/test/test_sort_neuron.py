import torch
from itertools import product
from core.norm_dist import sort_neuron
from core.test import test_utils
from torch.nn.functional import unfold

def test_sort_case(B, CI, CO, H, W, K, G, q, truncate, backward_input, backward_weight):
    input = torch.randn(B, CI, H, W).cuda()
    weight = torch.randn(CO, CI // G, K, K).cuda()
    input = unfold(input, K, padding=K // 2)
    weight = weight.view(weight.size(0), -1)
    if backward_input:
        input = torch.nn.Parameter(input)
    if backward_weight:
        weight = torch.nn.Parameter(weight)
    res = test_utils.test(sort_neuron, input, weight, groups=G, q=q, truncate=truncate)
    # print('B=%d, CI=%d, CO=%d, H=%d, W=%d, K=%d, G=%d, q=%f, truncate=%d' % (B, CI, CO, H, W, K, G, q, truncate),
    #       'diff_output', test_utils.truncate(res, 7)[0], '\n')
    if not test_utils.check(res, 1e-2):
        print_res = test_utils.truncate(res, 7)
        print('Warning!\n',
              'B=%d, CI=%d, CO=%d, H=%d, W=%d, K=%d, G=%d, q=%f, truncate=%d'%(B, CI, CO, H, W, K, G, q, truncate), '\n',
              'diff_output', print_res[0], '\n',
              'diff_grad_input', print_res[1], '\n',
              'diff_grad_weight', print_res[2])
        # print('input', input)
        # print('weight', weight)
        # exit()
    return res

def test_sort_all(**params):
    Bs = (1, 2, 3, 8, 12, 16, 20)
    CIs = (1, 2, 3, 8, 16, 20)
    COs = (1, 2, 3, 8, 16, 20)
    HWs = [(1, 1), (2, 1), (3, 3), (4, 4), (3, 6), (4, 8)]
    Ks = (1, 2, 3)
    Gs = (1, 2, 3, 4)
    truncates = (1, 2, 3, 4, 5, 6, 8, 10, 15, 20)
    qs = (1.0, 0.8, 0.5)
    print('Start testing: ', str(params))
    worst_res = None
    for B, CI, CO, (H, W), K, G, q, truncate in product(Bs, CIs, COs, HWs, Ks, Gs, qs, truncates):
        if B * H * W > 192: continue
        if (CO >= 64 and CI * K * K >= 64) or (CO >= 64 and B * H * W >= 64) or (CI * K * K >= 64 and B * H * W >= 64):
            continue
        if CI % G != 0 or CO % G != 0:
            continue
        res = test_sort_case(B, CI, CO, H, W, K, G, q, truncate, **params)
        worst_res = test_utils.merge(worst_res, res)
    return worst_res

# def test_norm_dist_speed(B, CI, CO, H, W, K, G, p, backward, ext):
#     input = torch.randn(B, CI, H, W).cuda()
#     weight = torch.randn(CO, CI // G, K, K).cuda()
#     input = unfold(input, K, padding=K // 2)
#     weight = weight.view(weight.size(0), -1)
#     if backward:
#         input = torch.nn.Parameter(input)
#         weight = torch.nn.Parameter(weight)
#     print('Start testing: B=%d, CI=%d, CO=%d, H=%d, W=%d, K=%d, G=%d, p=%f, backward='%(B, CI, CO, H, W, K, G, p), backward)
#     return test_utils.test_speed(norm_dist, input, weight, p=p, groups=G)

def test_sort():
    torch.manual_seed(2021)
    # print(test_utils.truncate(test_sort_all(backward_input=False, backward_weight=False), 8))
    print(test_utils.truncate(test_sort_all(backward_input=True, backward_weight=True), 8))
    print(test_utils.truncate(test_sort_all(backward_input=True, backward_weight=False), 8))

    # print('Testing speed:')
    # for p in (8, 20, float('inf')):
    #     for backward in (False, True):
    #         print(test_norm_dist_speed(512, 64, 64, 32, 32, 3, 1, p, backward, ext=False))
    #         print(test_norm_dist_speed(512, 128, 128, 16, 16, 3, 1, p, backward, ext=False))
    #         print(test_norm_dist_speed(512, 4096, 4096, 1, 1, 1, 1, p, backward, ext=False))
