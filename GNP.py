import torch
import torch.autograd.functional
from torch.optim.optimizer import Optimizer
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import lsmr, lsqr
from typing import Callable, Iterable, Optional
import linops
from enum import Enum


class LinearSystemSolver(Enum):
    """Contains the available linear system solvers for PolyakBundle.

    LSMR: The SciPy implementation of the LSMR algorithm.
    MINRES: A torch implementation of MINRES on JAC^TJAC
    """

    LSMR = 1
    MINRES = 2


class GNP(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.Tensor],
            LSMR_tol=1e-5,
            linsys_solver: LinearSystemSolver = LinearSystemSolver.LSMR,
    ):
        super(GNP, self).__init__(params, dict())

        self._params = self.param_groups[0]["params"]
        self._numel = sum([p.numel() for p in self._params])
        self.tol = LSMR_tol
        self.linsys_solver = linsys_solver
        self.MINRES_warm_start = None
        self.LSMR_warm_start = None

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel

    def _gather_flat_grad(self) -> np.ndarray:
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    @torch.no_grad()
    def step(self, closure: Callable, JacTJac: Callable = None):

        # allow closure to call autograd
        with torch.enable_grad():
            # evaluate closure
            loss = closure().item()

        g = self._gather_flat_grad()

        if JacTJac is not None:
            sz = g.numel()
            if self.linsys_solver == LinearSystemSolver.MINRES:

                JacTJac_linop = linops.LinearOperator()
                JacTJac_linop._shape = (sz, sz)
                JacTJac_linop._matmul_impl = JacTJac
                JacTJac_linop._adjoint = JacTJac_linop
                JacTJac_linop.supports_operator_matrix = True

                # g = g.view(-1)
                # Solve for (JacJacT)step = g.
                if self.MINRES_warm_start is None:
                    step = JacTJac_linop.solve_A_x_eq_b(b=g)
                else:
                    step = JacTJac_linop.solve_A_x_eq_b(b=g, x0=self.MINRES_warm_start)
                self.MINRES_warm_start = step

            elif self.linsys_solver == LinearSystemSolver.LSMR:
                def JacTJac_np(x):
                    return JacTJac(torch.from_numpy(x)).numpy()

                g_np = g.view(-1).detach().numpy()
                JacTJac_lo = LinearOperator(
                    (sz, sz), matvec=JacTJac_np, rmatvec=JacTJac_np
                )
                if self.LSMR_warm_start is None:
                    step = lsmr(
                        JacTJac_lo,
                        g_np,
                        atol=self.tol,  # max(1e-15, 1e-15),
                        btol=self.tol,
                        conlim=0.0,
                    )[0]
                else:
                    step = lsmr(
                        JacTJac_lo,
                        g_np,
                        atol=self.tol,  # max(1e-15, 1e-15),
                        btol=self.tol,
                        conlim=0.0,
                        x0=self.LSMR_warm_start,
                    )[0]
                self.LSMR_warm_start = step
                step = torch.from_numpy(step)

            else:
                raise ValueError("Need to choose a valid linear system solver")
            # compute the denominator
            ## norm(Jac(JacJacT)^{-1}Jac^T grad)^2
            ## dotp(JacTJac((JacJacT)^{-1}Jac^T grad), (JacJacT)^{-1}Jac^T grad)
            denom = (JacTJac(step) @ step).item()
            lr = (loss / denom)
        else:
            step = g
            lr = loss / (g.norm() ** 2).item()

        self._add_grad(-lr, step)
