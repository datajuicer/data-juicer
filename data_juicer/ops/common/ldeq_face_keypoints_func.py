# Adapted from https://github.com/polo5/LDEQ_RwR.git

import random

import numpy as np

from data_juicer.utils.lazy_loader import LazyLoader

torch = LazyLoader("torch")
nn = LazyLoader("torch.nn")
torchinfo = LazyLoader("torchinfo")


class Normalize(nn.Module):
    """normalize to [0,1]"""

    def __init__(self, n_channels, mode, beta=1.0, learn_beta=False):
        super().__init__()
        self.mode = mode
        assert mode in ["softargmax", "linear"], f"norm {mode} not recognized"

        if mode == "softargmax":
            if learn_beta:
                self.nonlinearity = nn.Softplus()
                self.beta = (
                    torch.nn.Parameter(torch.ones(n_channels) * beta).view(1, -1, 1, 1).cuda()
                )  # one beta per heatmap. TODO: why isn't this put on gpu when we put whole model?
            else:
                self.nonlinearity = lambda x: x
                self.beta = beta

    def forward(self, heatmaps):

        if self.mode == "softargmax":
            heatmaps = heatmaps - torch.amax(heatmaps, dim=(2, 3), keepdim=True)
            heatmaps = torch.exp(
                self.nonlinearity(self.beta) * heatmaps
            )  # nonlinearity makes sure beta is positive so that exp input still in [-inf,0] so that output is in [0,1]

        elif self.mode == "linear":
            heatmaps_max, heatmaps_min = torch.amax(heatmaps, dim=(2, 3), keepdim=True), torch.amin(
                heatmaps, dim=(2, 3), keepdim=True
            )  # shape (B,n_kpts,1,1)
            heatmaps = (heatmaps - heatmaps_min) / (heatmaps_max - heatmaps_min + 1e-5)

        return heatmaps


class HeatmapsToKeypoints(nn.Module):
    """converts 2D heatmaps into (x,y) coordinates in range [0,1] that our loss can use"""

    def __init__(self):
        super().__init__()
        self.first_run = True

    def forward(self, heatmaps):
        """heatmap values must all be between 0 and 1. This is achieved with the Normalize class above"""
        B, n_keypoints, H, W = heatmaps.shape
        heatmaps = heatmaps / (1e-4 + torch.sum(heatmaps, dim=[2, 3], keepdim=True))  # now heatmap values all sum to 1

        if self.first_run:
            col_vals = torch.arange(0, W)
            self.col_grid = (
                col_vals.repeat(H, 1).view(1, 1, H, W).to(heatmaps.device)
            )  # each column is a single repeated number
            row_vals = torch.arange(0, H).view(H, -1)
            self.row_grid = (
                row_vals.repeat(1, W).view(1, 1, H, W).float().to(heatmaps.device)
            )  # each row is a single repeated number
            self.first_run = False

        weighted_x = heatmaps * self.col_grid
        x_vals = weighted_x.sum(dim=[2, 3]) / H  # in range [0,1], shape (B,98)
        weighted_y = heatmaps * self.row_grid
        y_vals = weighted_y.sum(dim=[2, 3]) / H  # in range [0,1], shape (B,98)
        out = torch.stack((x_vals, y_vals), dim=2)

        # TODO: not sure this is still correct if using linear normalization for heatmaps:
        var_x = ((self.col_grid - x_vals.unsqueeze(2).unsqueeze(3)).pow(2) * heatmaps).sum(
            dim=[2, 3]
        )  # this is like a variance term and can take on large values (~600 for heatmap size 64)
        var_y = ((self.row_grid - y_vals.unsqueeze(2).unsqueeze(3)).pow(2) * heatmaps).sum(dim=[2, 3])
        # NB: if x_vals=mean=5, then (col_grid - x_vals) will be a grid, with 0 at location of the mean and polynomially increasing values around the mean
        # then heatmaps weighs this grid with the spread of predictions. If heatmaps is non zero only in location of mean, then sigma_x = 0

        stds = torch.sqrt(0.5 * var_x + 0.5 * var_y) / H  # shape (B,98), i.e. one std value per heatmap.

        return out, stds  # out is (B, 98, 2)


class HeatmapsToKeypointsNoSum(nn.Module):
    """converts 2D heatmaps into (x,y) coordinates in range [0,1] that our loss can use,
    This can be used if input heatmaps are already divided by their sum
    """

    def __init__(self):
        super().__init__()
        self.first_run = True

    def forward(self, heatmaps):
        """heatmap values must all be between 0 and 1. This is achieved with the Normalize class above"""
        B, n_keypoints, H, W = heatmaps.shape
        # heatmaps = heatmaps/(1e-4+torch.sum(heatmaps, dim=[2,3], keepdim=True))

        if self.first_run:
            col_vals = torch.arange(0, W)
            self.col_grid = (
                col_vals.repeat(H, 1).view(1, 1, H, W).to(heatmaps.device)
            )  # each column is a single repeated number
            row_vals = torch.arange(0, H).view(H, -1)
            self.row_grid = (
                row_vals.repeat(1, W).view(1, 1, H, W).float().to(heatmaps.device)
            )  # each row is a single repeated number
            self.first_run = False

        weighted_x = heatmaps * self.col_grid
        x_vals = weighted_x.sum(dim=[2, 3]) / H  # in range [0,1], shape (B,98)
        weighted_y = heatmaps * self.row_grid
        y_vals = weighted_y.sum(dim=[2, 3]) / H  # in range [0,1], shape (B,98)
        out = torch.stack((x_vals, y_vals), dim=2)

        # TODO: not sure this is still correct if using linear normalization for heatmaps:
        var_x = ((self.col_grid - x_vals.unsqueeze(2).unsqueeze(3)).pow(2) * heatmaps).sum(
            dim=[2, 3]
        )  # this is like a variance term and can take on large values (~600 for heatmap size 64)
        var_y = ((self.row_grid - y_vals.unsqueeze(2).unsqueeze(3)).pow(2) * heatmaps).sum(dim=[2, 3])
        # NB: if x_vals=mean=5, then (col_grid - x_vals) will be a grid, with 0 at location of the mean and polynomially increasing values around the mean
        # then heatmaps weighs this grid with the spread of predictions. If heatmaps is non zero only in location of mean, then sigma_x = 0

        stds = torch.sqrt(0.5 * var_x + 0.5 * var_y) / H  # shape (B,98), i.e. one std value per heatmap.

        return out, stds


def _safe_norm(v):
    if not torch.isfinite(v).all():
        return np.inf
    return torch.norm(v)


def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0):
    """
    see https://github.com/scipy/scipy/blob/main/scipy/optimize/_linesearch.py
    Minimize over alpha, the function phi(alpha). Uses the interpolation algorithm (Armijo backtracking) as suggested by
    Wright and Nocedal in 'Numerical Optimization', 1999, pp. 56-57. alpha > 0 is assumed to be a descent direction.

    phi = callable function phi(alpha)
    phi0 = value of phi(alpha) for original estimate
    derphi = callable function phi'(alpha).

    In our case phi(alpha) = torch.norm(g(x0 + alpha * update))**2 ?

    """
    ite = 0
    phi_a0 = phi(alpha0)  # First do an update with step size 1
    if phi_a0 <= phi0 + c1 * alpha0 * derphi0:
        return alpha0, phi_a0, ite

    # Otherwise, compute the minimizer of a quadratic interpolant
    alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    # Otherwise loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.
    while alpha1 > amin:  # we are assuming alpha>0 is a descent direction
        factor = alpha0**2 * alpha1**2 * (alpha1 - alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0 * alpha1) - alpha1**2 * (phi_a0 - phi0 - derphi0 * alpha0)
        a = a / factor
        b = -(alpha0**3) * (phi_a1 - phi0 - derphi0 * alpha1) + alpha1**3 * (phi_a0 - phi0 - derphi0 * alpha0)
        b = b / factor

        alpha2 = (-b + torch.sqrt(torch.abs(b**2 - 3 * a * derphi0))) / (3.0 * a)
        phi_a2 = phi(alpha2)
        ite += 1

        if phi_a2 <= phi0 + c1 * alpha2 * derphi0:
            return alpha2, phi_a2, ite

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2 / alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    # Failed to find a suitable step length
    return None, phi_a1, ite


def line_search(update, x0, g0, g, on=True):
    """
    Instead of solving for the best step size to use exactly, we use a fast line search algorithm
    to find an okay step size, so that compute can be spent on computing the update itself rather
    than the step size.

    `update` is the propsoed direction of update.

    Code adapted from scipy.
    """
    tmp_s = [0]
    tmp_g0 = [g0]
    tmp_phi = [torch.norm(g0) ** 2]
    # s_norm = torch.norm(x0) / torch.norm(update) #for wolfe search only

    def phi(s, store=True):
        """takes in step size alpha being tried, produces the next x_est with it,
        and returns what we want to minimize, i.e. norm of g(x_est)"""
        if s == tmp_s[0]:
            return tmp_phi[0]  # If the step size is so small... just return something
        x_est = x0 + s * update
        g0_new = g(x_est)
        phi_new = _safe_norm(g0_new) ** 2
        if store:
            tmp_s[0] = s
            tmp_g0[0] = g0_new
            tmp_phi[0] = phi_new
        return phi_new

    if on:
        s, phi1, ite = scalar_search_armijo(phi, tmp_phi[0], -tmp_phi[0], amin=1e-2)
    if (not on) or s is None:
        s = 1.0
        ite = 0

    x_est = x0 + s * update
    if s == tmp_s[0]:
        g0_new = tmp_g0[0]
    else:
        g0_new = g(x_est)

    return x_est, g0_new, x_est - x0, g0_new - g0, ite


def rmatvec(part_Us, part_VTs, x):
    # Compute x^T(-I + UV^T)
    # x: (N, 2d, L')
    # part_Us: (N, 2d, L', threshold)
    # part_VTs: (N, threshold, 2d, L')
    if part_Us.nelement() == 0:
        return -x
    xTU = torch.einsum("bij, bijd -> bd", x, part_Us)  # (N, threshold)
    return -x + torch.einsum("bd, bdij -> bij", xTU, part_VTs)  # (N, 2d, L'), but should really be (N, 1, (2d*L'))


def matvec(part_Us, part_VTs, x):
    # Compute (-I + UV^T)x
    # x: (N, 2d, L')
    # part_Us: (N, 2d, L', threshold)
    # part_VTs: (N, threshold, 2d, L')
    if part_Us.nelement() == 0:
        return -x
    VTx = torch.einsum("bdij, bij -> bd", part_VTs, x)  # (N, threshold)
    return -x + torch.einsum("bijd, bd -> bij", part_Us, VTx)  # (N, 2d, L'), but should really be (N, (2d*L'), 1)


def broyden(f, x0, max_iters, eps=1e-3, stop_mode="rel", ls=False, verbose=False, save_trajectory=False):
    # print(f'broyden input size: {x0.size()}')
    bsz, total_hsize, seq_len = x0.size()

    # g = lambda y: f(y) - y
    def g(y):
        return f(y) - y

    dev = x0.device
    alternative_mode = "rel" if stop_mode == "abs" else "abs"
    trajectory = []

    x_est = x0  # (bsz, 2d, L')
    gx = g(x_est)  # (bsz, 2d, L')
    nstep = 0
    tnstep = 0

    # For fast calculation of inv_jacobian (approximately)
    Us = torch.zeros(bsz, total_hsize, seq_len, max_iters).to(
        dev
    )  # One can also use an L-BFGS scheme to further reduce memory
    VTs = torch.zeros(bsz, max_iters, total_hsize, seq_len).to(dev)
    update = -matvec(
        Us[:, :, :, :nstep], VTs[:, :nstep], gx
    )  # Formally should be -torch.matmul(inv_jacobian, (-I), gx)
    prot_break = False

    # To be used in protective breaks
    protect_thres = (1e6 if stop_mode == "abs" else 1e3) * seq_len

    trace_dict = {"abs": [], "rel": []}
    lowest_dict = {"abs": 1e8, "rel": 1e8}
    lowest_step_dict = {"abs": 0, "rel": 0}
    nstep, lowest_xest, _ = 0, x_est, gx

    while nstep < max_iters:
        x_est, gx, delta_x, delta_gx, ite = line_search(
            update, x_est, gx, g, on=ls
        )  # returns x_est, gx_new, x_est_new - x_est_prev, gx_new - gx_prev, ite
        nstep += 1
        tnstep += ite + 1

        abs_diffs = gx.norm(dim=1)
        rel_diffs = abs_diffs / (1e-5 + (gx + x_est).norm(dim=1))
        abs_diff, rel_diff = (
            abs_diffs.mean(),
            rel_diffs.mean(),
        )  # rel diff correctly calculated is ~5% different from official implementation

        if verbose:
            print(
                f"abs diff {abs_diff:.2E} \t rel diff: {rel_diff:.2E} \t z scale: {torch.mean(x_est):.0E} +/- {torch.std(x_est):.0E}"
            )
        diff_dict = {"abs": abs_diff, "rel": rel_diff}
        trace_dict["abs"].append(abs_diff)
        trace_dict["rel"].append(rel_diff)
        # print(f'broyden step {nstep} --- abs diff {abs_diff} --- rel diff {rel_diff}')
        for mode in ["rel", "abs"]:
            if diff_dict[mode] < lowest_dict[mode] or nstep == 1:
                if mode == stop_mode:
                    lowest_xest, _ = x_est.clone().detach(), gx.clone().detach()
                lowest_dict[mode] = diff_dict[mode]
                lowest_step_dict[mode] = nstep

        if save_trajectory:
            trajectory.append(x_est.view_as(x0).clone().detach())

        # Added by Paul to measure stability of solver
        if nstep == 1:
            stability = 1
            prev_rel_diff = rel_diff
        else:
            if rel_diff > prev_rel_diff:  # error is jumping around
                stability = 0
            prev_rel_diff = rel_diff

        new_objective = diff_dict[stop_mode]
        if new_objective < eps:  # stop even if haven't reached max_iters steps
            if verbose:
                print(f"STOPPING BROYDEN SPECIAL CASE: met tolerance")
            break
        if (
            new_objective < 3 * eps
            and nstep > 30
            and np.max(trace_dict[stop_mode][-30:]) / np.min(trace_dict[stop_mode][-30:]) < 1.3
        ):
            # if there's hardly been any progress in the last 30 steps
            if verbose:
                print("STOPPING BROYDEN SPECIAL CASE: no progress in last 30 steps")
            break
        if new_objective > trace_dict[stop_mode][0] * protect_thres:
            if verbose:
                print("STOPPING BROYDEN SPECIAL CASE: protect thresh")
            prot_break = True
            break

        part_Us, part_VTs = Us[:, :, :, : nstep - 1], VTs[:, : nstep - 1]
        vT = rmatvec(part_Us, part_VTs, delta_x)
        u = (delta_x - matvec(part_Us, part_VTs, delta_gx)) / torch.einsum("bij, bij -> b", vT, delta_gx)[:, None, None]
        vT[vT != vT] = 0  # replace nans with zeros
        u[u != u] = 0
        VTs[:, nstep - 1] = vT
        Us[:, :, :, nstep - 1] = u
        update = -matvec(Us[:, :, :, :nstep], VTs[:, :nstep], gx)
        # print(update.device)

    # Fill everything up to the max_iters length
    for _ in range(max_iters + 1 - len(trace_dict[stop_mode])):
        trace_dict[stop_mode].append(lowest_dict[stop_mode])
        trace_dict[alternative_mode].append(lowest_dict[alternative_mode])

    # print(f'{name} total broyden steps: {nstep} --- rel diff {rel_diff:02.5f}')

    out = {
        "result": lowest_xest,
        "lowest_abs_diff": lowest_dict["abs"].item(),
        "lowest_rel_diff": lowest_dict["rel"].item(),
        "nstep_best": lowest_step_dict[stop_mode],  # which step was the best in hindsight
        "nstep": nstep,
        "prot_break": prot_break,
        "abs_trace": trace_dict["abs"],
        "rel_trace": trace_dict["rel"],
        "eps": eps,
        "trajectory": trajectory,
        "stability": stability,
    }

    return out


def anderson(
    f, x0, m=6, lam=1e-4, max_iters=50, eps=1e-3, stop_mode="rel", beta=1.0, verbose=False, save_trajectory=False
):
    """Anderson acceleration for fixed point iteration."""
    # print('stop mode ', stop_mode)
    bsz, d, L = x0.shape
    m = int(m)
    alternative_mode = "rel" if stop_mode == "abs" else "abs"
    X = torch.zeros(bsz, m, int(d * L), dtype=x0.dtype, device=x0.device)  # keep track of all previous estimates x_i s
    F = torch.zeros(bsz, m, int(d * L), dtype=x0.dtype, device=x0.device)  # keep track of all previous f(x_i) s
    X[:, 0], F[:, 0] = x0.reshape(bsz, -1), f(x0).reshape(bsz, -1)  # first estimate x0 is given as input
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].reshape_as(x0)).reshape(
        bsz, -1
    )  # second estimate in X is just f(x0) as in fpi because we don't have any previous estimates to lookback to

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    trace_dict = {"abs": [], "rel": []}
    lowest_dict = {"abs": 1e8, "rel": 1e8}
    lowest_step_dict = {"abs": 0, "rel": 0}
    trajectory = []

    # if verbose: print('Original tensors ')
    # if verbose: debug_print([X, F, H, y])

    for k in range(2, max_iters + 2):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1 : n + 1, 1 : n + 1] = (
            torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[None]
        )

        # alpha = torch.solve(y[:,:n+1], H[:,:n+1,:n+1])[0][:, 1:n+1, 0]   # (bsz x n)
        alpha = torch.linalg.solve(H[:, : n + 1, : n + 1], y[:, : n + 1])[:, 1 : n + 1, 0]  # (bsz x n)

        X[:, k % m] = (
            beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        )  # beta=1.0 in normal anderson formulation. beta<1 is damped anderson acceleration, while beta>1 is overprojected
        F[:, k % m] = f(X[:, k % m].reshape_as(x0)).reshape(bsz, -1)
        gx = F[:, k % m] - X[:, k % m]  # .view_as(x0)

        abs_diffs = gx.norm(dim=1)
        rel_diffs = abs_diffs / (1e-5 + F[:, k % m]).norm(dim=1)
        abs_diff, rel_diff = (
            abs_diffs.mean(),
            rel_diffs.mean(),
        )  # rel diff correctly calculated is ~5% different from official implementation
        if verbose:
            print(
                f"abs diff {abs_diff:.2E} \t rel diff: {rel_diff:.2E} \t z scale: {torch.mean(X[:, k % m]):.0E} +/- {torch.std(X[:,k % m]):.0E}"
            )

        diff_dict = {"abs": abs_diff, "rel": rel_diff}
        trace_dict["abs"].append(abs_diff)
        trace_dict["rel"].append(rel_diff)

        for mode in ["rel", "abs"]:
            # print(diff_dict[mode], lowest_dict[mode])
            if (diff_dict[mode] < lowest_dict[mode]) or k == 2:
                if mode == stop_mode:
                    lowest_xest, _ = X[:, k % m].view_as(x0).clone().detach(), gx.clone().detach()
                lowest_dict[mode] = diff_dict[mode]
                lowest_step_dict[mode] = k

        if save_trajectory:
            trajectory.append(X[:, k % m].view_as(x0).clone().detach())
            # print('------ ', float(torch.sum(X[:,k%m].view_as(x0).clone().detach())))

        # --------------- Added by Paul to measure stability of solver
        if k == 2:
            stability = 1
            abs_error_prev = abs_diff
        else:
            if abs_diff > abs_error_prev:  # error is jumping around
                stability = 0
            abs_error_prev = abs_diff
        # ---------------

        if trace_dict[stop_mode][-1] < eps:
            for _ in range(max_iters + 1 - k):  # paul changed -1 to +1
                trace_dict[stop_mode].append(lowest_dict[stop_mode])
                trace_dict[alternative_mode].append(lowest_dict[alternative_mode])
            break

    out = {
        "result": lowest_xest,  # not necessarily the last z of trajectory. It's the z with lowest error
        "lowest_abs_diff": lowest_dict["abs"].item(),
        "lowest_rel_diff": lowest_dict["rel"].item(),
        "nstep_best": lowest_step_dict[stop_mode],  # which step was the best in hindsight
        "nstep": k - 1,
        "prot_break": False,
        "abs_trace": trace_dict["abs"],
        "rel_trace": trace_dict["rel"],
        "eps": eps,
        "trajectory": trajectory,
        "stability": stability,
    }

    return out


def fpi(f, x0, max_iters, eps=1e-3, stop_mode="rel", verbose=False, save_trajectory=False):
    """fast and cheap in memory but no guarantees to return stable FP, contrary to other solvers"""
    trajectory = []
    bsz = x0.shape[0]
    x_prev = x0
    iter_idx = 0

    while iter_idx < max_iters:
        x_new = f(x_prev)
        abs_diffs = (x_new - x_prev).view(bsz, -1).norm(dim=1)
        rel_diffs = abs_diffs / (1e-5 + x_new.view(bsz, -1).norm(dim=1))
        abs_diff, rel_diff = abs_diffs.mean(), rel_diffs.mean()
        if verbose:
            print(
                f"abs diff {abs_diff:.3E} \t rel diff: {rel_diff:.3E} \t z scale: {torch.mean(x_new):.0E} +/- {torch.std(x_new):.0E}"
            )

        if save_trajectory:
            trajectory.append(x_new.clone().detach())

        # --------------- Added by Paul to measure stability of solver
        if iter_idx == 0:
            stability = 1
            abs_error_prev = abs_diff
        else:
            if abs_diff > abs_error_prev:  # error is jumping around
                stability = 0
            abs_error_prev = abs_diff
        # ---------------

        iter_idx += 1

        if (stop_mode == "abs" and abs_diff < eps) or (stop_mode == "rel" and rel_diff < eps):
            break

        x_prev = x_new

    return x_new, iter_idx, abs_diff.item(), rel_diff.item(), stability, trajectory


def root_solver(f, x0, max_iters, solver_args, stochastic_max_iters=False, save_trajectory=False, name="forward"):
    """
    There are many solvers that all return different metrics and take different arguments.
    This is a wrapping function that evaluates each solver.
    solver_args must contain the solver specific arguments like:
    solver_args.anderson_m = 6 etc.

    returns: n_iters, final_rel_error
    """

    max_iters = random.randint(1, max_iters) if stochastic_max_iters else max_iters
    if solver_args.verbose_solver:
        print(f"----- SOLVER: {solver_args.solver} {name} mi={max_iters}")

    if solver_args.solver == "broyden":
        results_dict = broyden(
            f=f,
            x0=x0,
            max_iters=max_iters,
            eps=solver_args.abs_diff_target if solver_args.stop_mode == "abs" else solver_args.rel_diff_target,
            stop_mode=solver_args.stop_mode,
            ls=False,
            verbose=solver_args.verbose_solver,
            save_trajectory=save_trajectory,
        )
        solution, n_iters, final_abs_diff, final_rel_diff, stability, trajectory = (
            results_dict["result"],
            results_dict["nstep"],
            results_dict["lowest_abs_diff"],
            results_dict["lowest_rel_diff"],
            results_dict["stability"],
            results_dict["trajectory"],
        )

    elif solver_args.solver == "anderson":
        results_dict = anderson(
            f=f,
            x0=x0,
            m=solver_args.anderson_m,
            lam=solver_args.anderson_lam,
            max_iters=max_iters,
            eps=solver_args.abs_diff_target if solver_args.stop_mode == "abs" else solver_args.rel_diff_target,
            stop_mode=solver_args.stop_mode,
            beta=solver_args.anderson_beta,
            verbose=solver_args.verbose_solver,
            save_trajectory=save_trajectory,
        )
        solution, n_iters, final_abs_diff, final_rel_diff, stability, trajectory = (
            results_dict["result"],
            results_dict["nstep"],
            results_dict["lowest_abs_diff"],
            results_dict["lowest_rel_diff"],
            results_dict["stability"],
            results_dict["trajectory"],
        )

    elif solver_args.solver == "fpi":
        solution, n_iters, final_abs_diff, final_rel_diff, stability, trajectory = fpi(
            f,
            x0=x0,
            max_iters=max_iters,
            eps=solver_args.abs_diff_target if solver_args.stop_mode == "abs" else solver_args.rel_diff_target,
            stop_mode=solver_args.stop_mode,
            verbose=solver_args.verbose_solver,
            save_trajectory=save_trajectory,
        )

    else:
        raise NotImplementedError(f"solver {solver_args.solver} unknown")

    # print('stability ', stability)

    solver_logs = {
        "n_iters": n_iters,
        "final_abs_diff": final_abs_diff,
        "final_rel_diff": final_rel_diff,
        "stability": stability,
        "trajectory": trajectory,
        "max_iters": max_iters,
    }

    return solution, solver_logs


def make_cell(args):
    return eval(args.cell_name)(args)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal(m.weight.data)
        torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.GroupNorm):
        if m.weight is not None:
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)


####################################################################################


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, norm="BN", GN_groups=1, no_relu=False):
        super(Conv, self).__init__()
        assert norm in ["BN", "GN", "None"], f"norm given {norm} unrecognized"
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.relu = (lambda x: x) if no_relu else nn.LeakyReLU()
        self.norm = (
            (lambda x: x)
            if norm == "None"
            else (nn.BatchNorm2d(out_dim) if norm == "BN" else nn.GroupNorm(GN_groups, out_dim))
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.norm(out)
        return out


class Hourglass(nn.Module):
    """
    Write out explicitly since nested formulation is too messy for multi-resolution.
    Same as default Hourglass for hg_dpeth=4 and n<10 and double_output=True
    Downside is that this now only supports hg_depth=4

    It was verified to have the same number of parameters (228572, if base_width=16 and increment=8)
    """

    def __init__(self, base_width, width_increment, norm="GN", GN_groups=8):
        super().__init__()
        self.downres = nn.AvgPool2d(2, 2)
        self.upres = nn.Upsample(scale_factor=2)
        w, i = base_width, width_increment
        w1, w2, w3, w4 = w + i, w + 2 * i, w + 3 * i, w + 4 * i

        self.same1 = Conv(w, w, 3, norm=norm, GN_groups=GN_groups)
        self.upchan1 = Conv(w, w1, 3, norm=norm, GN_groups=GN_groups)
        self.same2 = Conv(w1, w1, 3, norm=norm, GN_groups=GN_groups)
        self.upchan2 = Conv(w1, w2, 3, norm=norm, GN_groups=GN_groups)
        self.same3 = Conv(w2, w2, 3, norm=norm, GN_groups=GN_groups)
        self.upchan3 = Conv(w2, w3, 3, norm=norm, GN_groups=GN_groups)
        self.same4 = Conv(w3, w3, 3, norm=norm, GN_groups=GN_groups)
        self.upchan4 = Conv(w3, w4, 3, norm=norm, GN_groups=GN_groups)

        self.bottlneck = Conv(w4, w4, 3, norm=norm, GN_groups=GN_groups)

        self.downchan1 = Conv(w4, w3, 3, norm=norm, GN_groups=GN_groups)
        self.downchan2 = Conv(2 * w3, w3, 1, norm=norm, GN_groups=GN_groups)
        self.same5 = Conv(w3, w3, 5, norm=norm, GN_groups=GN_groups)
        self.downchan3 = Conv(w3, w2, 3, norm=norm, GN_groups=GN_groups)
        self.downchan4 = Conv(2 * w2, w2, 1, norm=norm, GN_groups=GN_groups)
        self.same6 = Conv(w2, w2, 5, norm=norm, GN_groups=GN_groups)
        self.downchan5 = Conv(w2, w1, 3, norm=norm, GN_groups=GN_groups)
        self.downchan6 = Conv(2 * w1, w1, 1, norm=norm, GN_groups=GN_groups)
        self.same7 = Conv(w1, w1, 5, norm=norm, GN_groups=GN_groups)
        self.downchan7 = Conv(w1, w, 3, norm=norm, GN_groups=GN_groups)
        self.downchan8 = Conv(2 * w, w, 1, norm=norm, GN_groups=GN_groups)
        self.same8 = Conv(w, w, 5, norm=norm, GN_groups=GN_groups)

    def forward(self, x):
        same1 = self.same1(x)
        downres1 = self.downres(same1)
        upchan1 = self.upchan1(downres1)

        same2 = self.same2(upchan1)
        downres2 = self.downres(same2)
        upchan2 = self.upchan2(downres2)

        same3 = self.same3(upchan2)
        downres3 = self.downres(same3)
        upchan3 = self.upchan3(downres3)

        same4 = self.same4(upchan3)
        downres4 = self.downres(same4)
        upchan4 = self.upchan4(downres4)

        # -----------------------------
        bottleneck = self.bottlneck(upchan4)
        # -----------------------------

        downchan1 = self.downchan1(bottleneck)
        upres1 = self.upres(downchan1)
        stack = torch.cat((same4, upres1), 1)
        downchan2 = self.downchan2(stack)
        same5 = self.same5(downchan2) + downchan2

        downchan3 = self.downchan3(same5)
        upres2 = self.upres(downchan3)
        stack = torch.cat((same3, upres2), 1)
        downchan4 = self.downchan4(stack)
        same6 = self.same6(downchan4) + downchan4

        downchan5 = self.downchan5(same6)
        upres3 = self.upres(downchan5)
        stack = torch.cat((same2, upres3), 1)
        downchan6 = self.downchan6(stack)
        same7 = self.same7(downchan6) + downchan6

        downchan7 = self.downchan7(same7)
        upres4 = self.upres(downchan7)
        stack = torch.cat((same1, upres4), 1)
        downchan8 = self.downchan8(stack)
        same8 = self.same8(downchan8) + downchan8

        return same8


####################################################################################


class Cell0(nn.Module):
    """same as Cell0 but always outputs data in [0,1]. We try various normalization techniques"""

    def __init__(self, args):
        super().__init__()
        norm_layer = "BN" if args.cell_use_bn_for_explicit and args.model_mode == "explicit" else "GN"
        self.tail = Conv(
            args.z_width + args.injection_width, args.cell_base_width, 1, norm=norm_layer, GN_groups=args.cell_gn_groups
        )
        self.hourglass = Hourglass(
            args.cell_base_width, args.cell_width_increment, norm=norm_layer, GN_groups=args.cell_gn_groups
        )
        self.head = Conv(args.cell_base_width, args.z_width, 1, stride=1, norm="None", no_relu=True)
        self.features_to_heatmaps = (
            (lambda x: x)
            if args.cell_norm == "None"
            else Normalize(
                args.z_width,
                mode=args.cell_norm,
                beta=args.cell_softargmax_beta,
                learn_beta=args.cell_learn_softargmax_beta,
            )
        )

    def forward(self, z, injection):
        # print(z.shape, injection.shape)
        out = self.tail(torch.cat([z, injection], dim=1))
        out = self.hourglass(out)
        out = self.head(out)  # heatmap size
        out = self.features_to_heatmaps(out)  # heatmap = normalized features

        return out


####################################################################################


class DEQLayer(nn.Module):
    """
    A DEQ layer applies the same cell with weight-sharing for several iterations.
    It can do so explicitly (track operations in autograd) or implicitly (only track very last iteration)
    """

    def __init__(self, cell, args):
        super().__init__()
        self.cell, self.heatmap_size, self.z_width = cell, args.heatmap_size, args.z_width

    def _forward_explicit(self, x, args, z0, save_trajectory=False):
        fwd_logs = None
        trajectory = [z0] if save_trajectory else []
        out = z0
        depth = 2 if args is None else args.explicit_depth  # torchinfo debug
        for _ in range(depth):
            out = self.cell(out, injection=x)
            if save_trajectory:
                trajectory.append(out.detach())
        # no need to do one more tracked forward pass here because they're all tracked already

        return out, fwd_logs, trajectory

    def _forward_implicit(self, x, args, z0, save_trajectory=False):
        trajectory = []
        z_shape = (x.shape[0], self.z_width, self.heatmap_size, self.heatmap_size)  # agnostic to x dimensions
        z_shape_solver = (
            x.shape[0],
            self.z_width * self.heatmap_size * self.heatmap_size,
            1,
        )  # agnostic to x dimensions
        # func = lambda z: self.cell(z.view(z_shape), injection=x).view(
        #     z_shape_solver
        # )  # inputs/outputs vector of shape z_shape_solver

        def func(z):
            return self.cell(z.view(z_shape), injection=x).view(z_shape_solver)

        stochastic_max_iters = args.stochastic_max_iters if self.training else False
        max_iters = (
            max(1, round(args.max_iters / 2)) if (not self.training and args.stochastic_max_iters) else args.max_iters
        )

        with torch.no_grad():
            z_star, fwd_logs = root_solver(
                f=func,
                x0=z0,
                max_iters=max_iters,
                solver_args=args,
                stochastic_max_iters=stochastic_max_iters,
                save_trajectory=save_trajectory,
                name="forward",
            )

        if self.training:
            z_star_new = func(z_star.requires_grad_())  # extra tracked step so we create a computational graph

            if args.solver == "fpi":
                with torch.no_grad():
                    fwd_logs["final_solver_error"] = float(
                        torch.norm(z_star_new - z_star) / (torch.norm(z_star_new) + 1e-9)
                    )  # same as the one in solver_logs if using tracing. But fpi doesn't do tracing.

            if not args.JFB:

                def backward_hook(grad):
                    if self.hook is not None:
                        self.hook.remove()
                        torch.cuda.synchronize()

                    # func = lambda y: torch.autograd.grad(z_star_new, z_star, y, retain_graph=True)[0] + grad
                    def func(y):
                        return torch.autograd.grad(z_star_new, z_star, y, retain_graph=True)[0] + grad

                    solution, solver_logs_bwd = root_solver(
                        f=func,
                        x0=torch.zeros_like(grad),
                        max_iters=max(1, round(args.max_iters / 2)) if args.stochastic_max_iters else args.max_iters,
                        solver_args=args,
                        stochastic_max_iters=False,
                        save_trajectory=False,
                        name="backward",
                    )
                    # solution, solver_logs_bwd = root_solver(f=func, x0=torch.rand_like(grad), solver_args=args, stochastic_max_iters=False, save_trajectory=False, name="backward") #not good
                    if args.verbose_solver:
                        print(
                            f"original grad: scale = {torch.mean(torch.abs(grad)):01.1e}, pos sign frac = {100*torch.mean((torch.sign(grad)+1)/2):02.0f}%"
                        )
                        print(
                            f"  new    grad: scale = {torch.mean(torch.abs(solution)):01.1e}, pos sign frac = {100*torch.mean((torch.sign(solution)+1)/2):02.0f}%"
                        )
                        print(
                            f" ---   change: value = {100*torch.mean(torch.abs((solution-grad)))/torch.mean(torch.abs(grad)):02.0f}%, sign: {100*torch.mean((torch.sign(grad)-torch.sign(solution))/2):02.0f}%"
                        )
                    return solution

                self.hook = z_star_new.register_hook(
                    backward_hook
                )  # WARNING: leads to memory leak if not cleared with .backward() at each batch

        else:
            if args.take_one_less_inference_step:
                z_star_new = z_star
            else:
                with torch.no_grad():
                    z_star_new = func(
                        z_star
                    )  # usually don't need to take this step at inference if close enough to solution already
                    if args.solver == "fpi":
                        fwd_logs["final_solver_error"] = float(
                            torch.norm(z_star_new - z_star) / (torch.norm(z_star_new) + 1e-9)
                        )
                    # Note that when this extra step is performed we are actually taking max_iters+1 iterations

        if save_trajectory:  # change shape and add z0
            trajectory = [z.view(z_shape) for z in fwd_logs["trajectory"]]
            trajectory.insert(0, z0.view(z_shape))
            del fwd_logs["trajectory"]

        return z_star_new.view(z_shape), fwd_logs, trajectory

    def forward(self, x, mode, args, z0=None, save_trajectory=False):
        z_shape = (x.shape[0], self.z_width, self.heatmap_size, self.heatmap_size)
        z_shape_solver = (x.shape[0], self.z_width * self.heatmap_size * self.heatmap_size, 1)

        if mode == "explicit":
            z0 = z0.view(*z_shape)
            out, fwd_logs, trajectory = self._forward_explicit(x, args, z0, save_trajectory)
        elif mode == "implicit":
            z0 = z0.view(*z_shape_solver)
            out, fwd_logs, trajectory = self._forward_implicit(x, args, z0, save_trajectory)
        else:
            raise NotImplementedError

        z_star_copy = out.detach().view(*z_shape)
        return out, z_star_copy, fwd_logs, trajectory


####################################################################################


class LDEQ(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        out_width = args.injection_width

        self.tail = nn.Sequential(
            Conv(3, out_width // 4, 7, 2),
            Conv(out_width // 4, out_width // 2, 3, 1),
            nn.MaxPool2d(2, 2),
            Conv(out_width // 2, out_width, 3, 1),
        )

        cell = make_cell(args)
        self.deq_layer = DEQLayer(cell, args)  # outputs potentially already [0,1] normalized heatmaps
        self.final_features_to_heatmaps = (
            Normalize(args.z_width, "softargmax", args.cell_softargmax_beta, False)
            if args.cell_norm == "None"
            else lambda x: x
        )  # output of cell possibly already normalized
        self.heatmaps_to_keypoints = HeatmapsToKeypoints()

    def forward(self, x, mode="implicit", args=None, z0=None, save_trajectory=False):
        """
        mode = 'implicit' or 'explicit'. Explicit is done with weight sharing
        zc only added for mode==implicit_broyden_strategy1_forward_only
        """

        x = self.tail(x)
        z_star, z_star_copy, fwd_logs, trajectory = self.deq_layer(
            x, mode, args, z0, save_trajectory
        )  # z0 and z_star can be tensors or lists of tensors.
        out = self.final_features_to_heatmaps(z_star)
        preds, uncertainty = self.heatmaps_to_keypoints(out[:, : self.args.n_keypoints, :, :])
        results = {
            "keypoints": preds,
            "uncertainty": uncertainty,
            "fwd_logs": fwd_logs,
            "z_star": z_star_copy,
            "trajectory": trajectory,
        }

        return results
