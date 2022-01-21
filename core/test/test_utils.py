import torch
import math
import time

@torch.no_grad()
def get_diffs(x, y):
    diff = ((x - y).abs() / (y.abs() + 1)).view(-1)
    val, id = diff.max(dim=0)
    return val.item(), x.view(-1)[id].item(), y.view(-1)[id].item()

def test(func, *tensors, **kwargs):
    requires_grad = False
    copy_tensors = []
    for t in tensors:
        copy = t.data.clone()
        if t.requires_grad:
            copy = torch.nn.Parameter(copy)
            requires_grad = True
        copy_tensors.append(copy)
    output = func(*tensors, **kwargs, use_custom_cuda_func=True)
    output2 = func(*copy_tensors, **kwargs, use_custom_cuda_func=False)
    if isinstance(output, torch.Tensor):
        if requires_grad:
            (torch.sin(output) + torch.cos(output)).sum().backward()
            (torch.sin(output2) + torch.cos(output2)).sum().backward()
        ret = [get_diffs(output, output2)]
    else:
        if requires_grad:
            (torch.sin(output[0]) + torch.cos(output[0]) + torch.sin(output[1]) + torch.cos(output[1])).sum().backward()
            (torch.sin(output2[0]) + torch.cos(output2[0]) + torch.sin(output2[1]) + torch.cos(output2[1])).sum().backward()
        ret = [get_diffs(output[0], output2[0]), get_diffs(output[1], output2[1])]
    for t, t2 in zip(tensors, copy_tensors):
        if t.requires_grad:
            ret.append(get_diffs(t.grad, t2.grad))
        else:
            ret.append(None)
    return ret

def check(res, threshold):
    for item in res:
        if item is not None and (item[0] >= threshold or math.isnan(item[0])):
            return False
    return True

def truncate(res, precision):
    ret = []
    for item in res:
        if item is not None:
            ret.append(tuple([round(i, precision) for i in item]))
        else:
            ret.append(None)
    return ret

def merge(res1, res2):
    if res1 is None:
        return res2
    ret = []
    for item1, item2 in zip(res1, res2):
        if item1 is not None and item2 is not None:
            ret.append(max(item1, item2))
        else:
            ret.append(None)
    return ret

def test_speed(func, *tensors, **kwargs):
    func(*tensors, **kwargs, use_custom_cuda_func=True)
    torch.cuda.synchronize()
    start = time.process_time()
    for i in range(20):
        output = func(*tensors, **kwargs, use_custom_cuda_func=True)
        if not isinstance(output, torch.Tensor):
            output = output[0] + output[1]
        if output.requires_grad:
            output.sum().backward()
    torch.cuda.synchronize()
    return time.process_time() - start