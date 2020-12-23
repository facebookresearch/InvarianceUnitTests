# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
import torch
import math


class Example1:
    """
    Cause and effect of a target with heteroskedastic noise
    """

    def __init__(self, dim_inv, dim_spu, n_envs):
        self.scramble = torch.eye(dim_inv + dim_spu)
        self.dim_inv = dim_inv
        self.dim_spu = dim_spu
        self.dim = dim_inv + dim_spu

        self.task = "regression"
        self.envs = {}

        if n_envs >= 2:
            self.envs = {'E0': 0.1, 'E1': 1.5}
        if n_envs >= 3:
            self.envs["E2"] = 2
        if n_envs > 3:
            for env in range(3, n_envs):
                var = 10 ** torch.zeros(1).uniform_(-2, 1).item()
                self.envs["E" + str(env)] = var
        print("Environments variables:", self.envs)

        self.wxy = torch.randn(self.dim_inv, self.dim_inv) / self.dim_inv
        self.wyz = torch.randn(self.dim_inv, self.dim_spu) / self.dim_spu

    def sample(self, n=1000, env="E0", split="train"):
        sdv = self.envs[env]
        x = torch.randn(n, self.dim_inv) * sdv
        y = x @ self.wxy + torch.randn(n, self.dim_inv) * sdv
        z = y @ self.wyz + torch.randn(n, self.dim_spu)

        if split == "test":
            z = z[torch.randperm(len(z))]

        inputs = torch.cat((x, z), -1) @ self.scramble
        outputs = y.sum(1, keepdim=True)

        return inputs, outputs


class Example2:
    """
    Cows and camels
    """

    def __init__(self, dim_inv, dim_spu, n_envs):
        self.scramble = torch.eye(dim_inv + dim_spu)
        self.dim_inv = dim_inv
        self.dim_spu = dim_spu
        self.dim = dim_inv + dim_spu

        self.task = "classification"
        self.envs = {}

        if n_envs >= 2:
            self.envs = {
                'E0': {"p": 0.95, "s": 0.3},
                'E1': {"p": 0.97, "s": 0.5}
            }
        if n_envs >= 3:
            self.envs["E2"] = {"p": 0.99, "s": 0.7}
        if n_envs > 3:
            for env in range(3, n_envs):
                self.envs["E" + str(env)] = {
                    "p": torch.zeros(1).uniform_(0.9, 1).item(),
                    "s": torch.zeros(1).uniform_(0.3, 0.7).item()
                }
        print("Environments variables:", self.envs)

        # foreground is 100x noisier than background
        self.snr_fg = 1e-2
        self.snr_bg = 1

        # foreground (fg) denotes animal (cow / camel)
        cow = torch.ones(1, self.dim_inv)
        self.avg_fg = torch.cat((cow, cow, -cow, -cow))

        # background (bg) denotes context (grass / sand)
        grass = torch.ones(1, self.dim_spu)
        self.avg_bg = torch.cat((grass, -grass, -grass, grass))

    def sample(self, n=1000, env="E0", split="train"):
        p = self.envs[env]["p"]
        s = self.envs[env]["s"]
        w = torch.Tensor([p, 1 - p] * 2) * torch.Tensor([s] * 2 + [1 - s] * 2)
        i = torch.multinomial(w, n, True)
        x = torch.cat((
            (torch.randn(n, self.dim_inv) /
                math.sqrt(10) + self.avg_fg[i]) * self.snr_fg,
            (torch.randn(n, self.dim_spu) /
                math.sqrt(10) + self.avg_bg[i]) * self.snr_bg), -1)

        if split == "test":
            x[:, self.dim_spu:] = x[torch.randperm(len(x)), self.dim_spu:]

        inputs = x @ self.scramble
        outputs = x[:, :self.dim_inv].sum(1, keepdim=True).gt(0).float()

        return inputs, outputs


class Example3:
    """
    Small invariant margin versus large spurious margin
    """

    def __init__(self, dim_inv, dim_spu, n_envs):
        self.scramble = torch.eye(dim_inv + dim_spu)
        self.dim_inv = dim_inv
        self.dim_spu = dim_spu
        self.dim = dim_inv + dim_spu

        self.task = "classification"
        self.envs = {}

        for env in range(n_envs):
            self.envs["E" + str(env)] = torch.randn(1, dim_spu)

    def sample(self, n=1000, env="E0", split="train"):
        m = n // 2
        sep = .1

        invariant_0 = torch.randn(m, self.dim_inv) * .1 + \
            torch.Tensor([[sep] * self.dim_inv])
        invariant_1 = torch.randn(m, self.dim_inv) * .1 - \
            torch.Tensor([[sep] * self.dim_inv])

        shortcuts_0 = torch.randn(m, self.dim_spu) * .1 + self.envs[env]
        shortcuts_1 = torch.randn(m, self.dim_spu) * .1 - self.envs[env]

        x = torch.cat((torch.cat((invariant_0, shortcuts_0), -1),
                       torch.cat((invariant_1, shortcuts_1), -1)))

        if split == "test":
            x[:, self.dim_inv:] = x[torch.randperm(len(x)), self.dim_inv:]

        inputs = x @ self.scramble
        outputs = torch.cat((torch.zeros(m, 1), torch.ones(m, 1)))

        return inputs, outputs


class Example1s(Example1):
    def __init__(self, dim_inv, dim_spu, n_envs):
        super().__init__(dim_inv, dim_spu, n_envs)

        self.scramble, _ = torch.qr(torch.randn(self.dim, self.dim))


class Example2s(Example2):
    def __init__(self, dim_inv, dim_spu, n_envs):
        super().__init__(dim_inv, dim_spu, n_envs)

        self.scramble, _ = torch.qr(torch.randn(self.dim, self.dim))


class Example3s(Example3):
    def __init__(self, dim_inv, dim_spu, n_envs):
        super().__init__(dim_inv, dim_spu, n_envs)

        self.scramble, _ = torch.qr(torch.randn(self.dim, self.dim))


DATASETS = {
    "Example1": Example1,
    "Example2": Example2,
    "Example3": Example3,
    "Example1s": Example1s,
    "Example2s": Example2s,
    "Example3s": Example3s
}
