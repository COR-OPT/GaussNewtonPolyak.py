import torch
import numpy as np
class TensorSensingProblem:
    def __init__(self, d, r, m, n, pfail=0, noise_std = 0.001, cond_num=1.0):
        self.m = m
        self.d = d
        self.r = r
        self.n = n
        self.nfact = np.math.factorial(self.n)
        self.pfail = pfail
        self.A = torch.randn(m, d, dtype=torch.double)
        self.X_star = torch.randn(d, r, dtype=torch.double)
        Q, _ = torch.linalg.qr(self.X_star)
        self.X_star = Q @ torch.diag(torch.linspace(1, cond_num, r, dtype=torch.double))
        self.X_star = self.X_star / torch.linalg.norm(self.X_star.view(-1))
        self.X_star = self.X_star / torch.linalg.matrix_norm(self.X_star, 2)
        self.fail = torch.bernoulli(torch.ones(m//2, dtype=torch.double) * pfail)
        self.b = self.measure(self.X_star) + noise_std*self.fail * torch.randn(m//2, dtype=torch.double)
        self.norm_CX = torch.sum(torch.pow((self.X_star).T @ self.X_star, n).view(-1))
        self.gram = None
        self.gramnm1 = None
        self.gramnm2 = None

    def JacTJac(self, method, X):
        # convert method to lower case
        method = method.lower()
        if method in ["polyak", "rpolyak"]:
            return self.JacTJac_Polyak(X)
        elif method in ["scaledsm", "rscaledsm"]:
            return self.JacTJac_ScaledSM(X)
        elif method in ["gnp", "rgnp"]:
            return self.JacTJac_GNP(X)
        else:
            raise ValueError("method must be one of Polyak, ScaledSM, or GNP")

    def JacTJac_Polyak(self, X):
        def JacTJac_func(Y=X, reset=False):
            if reset:
                self.reset_gram(X)
            Y = torch.reshape(Y, self.X_star.shape)
            if self.n >= 2:
                return Y.view(-1)
            else:
                raise ValueError("n must be at least 2")
        return JacTJac_func

    def JacTJac_ScaledSM(self, X):
        def JacTJac_func(Y=X, reset=False):
            if reset:
                self.reset_gram(X)
            Y = torch.reshape(Y, self.X_star.shape)
            # gram = X.T @ X
            if self.n == 2:
                return (Y @ self.gram).view(-1)
            else:
                raise ValueError("n must be exactly 2")
        return JacTJac_func

    def JacTJac_GNP(self, X):
        def JacTJac_func(Y=X, reset=False):
            if reset:
                self.reset_gram(X)
            Y = torch.reshape(Y, self.X_star.shape)
            # gram = X.T @ X
            YTX = Y.T @ X
            if self.n > 2:
                return (self.n * (self.n - 1) * X @ (self.gramnm2 * YTX) + self.n * Y @ self.gramnm1).view(-1)
                # return (self.n * (self.n - 1) * X @ (torch.pow(gram, self.n - 2) * YTX) + self.n * Y @ torch.pow(gram, self.n - 1)).view(-1)
            elif self.n == 2:
                return 2 * (X @ (YTX) + Y @ self.gram).view(-1)
            else:
                raise ValueError("n must be at least 2")
        return JacTJac_func

    def reset_gram(self, X):
        Y = X.detach().clone()
        self.gram = Y.T @ Y
        self.gramnm2 = torch.pow(self.gram, self.n - 2)
        self.gramnm1 = self.gram * self.gramnm2

    def measure(self, Y):
        return torch.sum(torch.pow(self.A[:self.m//2,:] @ Y, self.n) - torch.pow(self.A[self.m//2:,:] @ Y, self.n), dim=1)

    def rel_dist_in_range(self, X):
        gram_term = torch.sum(torch.pow(X.T @ X, self.n).view(-1))
        cross_term = torch.sum(torch.pow(self.X_star.T @ X, self.n).view(-1))
        return torch.sqrt(((gram_term - 2 * cross_term + self.norm_CX)/ self.norm_CX)).item()

    def loss(self):
        def f(z):
            return torch.linalg.norm((self.measure(z) - self.b).view(-1), ord=1)
        return f

    def initializer(self, delta):
        G = torch.randn(self.d, self.r)
        X = self.X_star + delta * G / torch.linalg.norm(G.view(-1))
        X = X.requires_grad_(True)
        return X

