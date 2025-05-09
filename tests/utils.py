import time
import pytest
import numpy as np
import torch


def check_op(func, ref_func, coords, *args, check_grad=True, atol=1e-8, rtol=1e-5):

    args_with_grad = []
    if check_grad:
        assert coords.requires_grad, "Coordinates does not require gradient"
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                args_with_grad.append(i)
    
    prb_grads = []
    ref_grads = []

    prb = func(coords, *args)
    
    if check_grad:
        prb.backward()
        prb_grads.append(coords.grad.clone().detach().cpu())
        coords.grad.zero_()
        for i in args_with_grad:
            prb_grads.append(args[i].grad.clone().detach().cpu())
            args[i].grad.zero_()
    
    ref = ref_func(coords, *args)

    if check_grad:
        ref.backward()
        ref_grads.append(coords.grad.clone().detach().cpu())
        coords.grad.zero_()
        for i in args_with_grad:
            ref_grads.append(args[i].grad.clone().detach().cpu())
            args[i].grad.zero_()
    
    assert torch.allclose(ref, prb, atol=atol, rtol=rtol), f"Energy not the same: {ref.cpu().item():.5f} != {prb.cpu().item()}"

    if check_grad:
        for i, (ref_grad, prb_grad) in enumerate(zip(ref_grads, prb_grads)):
            label = 'Coords ' if i == 0 else f'Argument {args_with_grad[i-1]} '
            assert torch.allclose(ref_grad, prb_grad, atol=atol, rtol=rtol), label + f'gradient not the same, max deviation {torch.max(torch.abs(ref_grad - prb_grad))}'


def perf_op(func, *args, desc='perf_op', warmup=10, repeat=1000, run_backward=False, use_cuda_graph=False, explicit_sync=True):
    
    assert torch.cuda.is_available(), "CUDA does not supported"

    perf = []
    if not use_cuda_graph:
        for _ in range(warmup):
            r = func(*args)
            if run_backward:
                r.backward()

        torch.cuda.synchronize() # sync to clean running kernels

        for _ in range(repeat):
            start = time.perf_counter()
            r = func(*args)
            if run_backward:
                r.backward()
            if explicit_sync:
                torch.cuda.synchronize()
            end = time.perf_counter()
            perf.append(end-start)
    else:
        # if use cuda graph, the warm-up has to be in another cuda stream
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                for arg in args:
                    if hasattr(arg, 'grad'):
                        arg.grad = None
                r = func(*args)
                if run_backward:
                    r.backward()
        torch.cuda.current_stream().wait_stream(s)

        if run_backward:
            g = torch.cuda.CUDAGraph()
            for arg in args:
                if hasattr(arg, 'grad'):
                    arg.grad = None
            with torch.cuda.graph(g):
                r = func(*args)
                r.backward()
        else:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                r = func(*args)
        
        for arg in args:
            if hasattr(arg, 'grad'):
                arg.grad = None

        for _ in range(repeat):
            start = time.perf_counter()
            g.replay()
            if explicit_sync:
                torch.cuda.synchronize()
            end = time.perf_counter()
            perf.append(end-start)
    
    perf = np.array(perf) * 1000  # in ms
    print(f"{desc} - Time: {np.mean(perf):.4f}+/-{np.std(perf):.4f} ms (use_cuda_graph = {use_cuda_graph}, run_backward = {run_backward}, explicit_sync = {explicit_sync})")


@pytest.mark.parametrize("run_backward", [True, False])
@pytest.mark.parametrize("use_cuda_graph", [True, False])
@pytest.mark.parametrize("explicit_sync", [True, False])
def test_perf(run_backward, use_cuda_graph, explicit_sync):

    def func(a):
        # return torch.sum(a ** 2)
        return torch.sum(torch.linalg.inv_ex(a, check_errors=False)[0])
    
    x = torch.rand((3, 3), device='cuda', requires_grad=run_backward)
    perf_op(
        func,
        x,
        desc='sum_squared',
        run_backward=run_backward,
        use_cuda_graph=use_cuda_graph,
        explicit_sync=explicit_sync
    )