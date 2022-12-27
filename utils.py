import torch
import time

def clone_param(param):
    return [p.detach().clone(memory_format=torch.contiguous_format) for p in param]

def set_param(param_original, params_new):
    for p, pdata in zip(param_original, params_new):
        p.data = pdata.data.clone()

def restarted_method(optimizer, closure, JacTJac, lb, rel_distance_in_range, nb_iters = 10, nb_restarts = 1, tol=1e-10):


    f_min = closure().item()
    p_min = clone_param(optimizer._params)
    start_param = clone_param(optimizer._params)

    f_min_hist = []
    rel_distance_in_range_hist = []
    time_hist = []

    # start the timer
    start = time.time()

    for i in range(nb_restarts):
        for j in range(nb_iters):
            f_min_hist.append(f_min)
            rel_distance_in_range_hist.append(rel_distance_in_range())
            time_hist.append(time.time() - start)
            shifted_closure = lambda: closure() - lb
            JacTJac(reset=True)
            optimizer.step(shifted_closure, JacTJac)
            loss = closure().item()
            if f_min > loss:
                f_min = min(f_min, closure().item())
                p_min = clone_param(optimizer._params)
            if tol is not None:
                if closure().item() - lb < tol:
                    set_param(optimizer._params, p_min)
                    return f_min_hist, time_hist, rel_distance_in_range_hist

        # Update lower bound
        lb = (f_min + lb)/2.
        print("lb", lb)
        # This is a safeguard; consider commenting out.
        set_param(optimizer._params, start_param)

    set_param(optimizer._params, p_min)

    return f_min_hist, time_hist, rel_distance_in_range_hist

