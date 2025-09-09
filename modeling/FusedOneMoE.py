# coding = utf-8
# Copyright 2025 Rikka Botan. All rights reserved
# Licensed under "MIT License"
# Commercial use is of course permitted
# Fused One MoE official PyTorch implementation

import torch
from torch import nn
import torch.nn.functional as F
from typing import Any
import math
import torch


def moe_fwd(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight_list: nn.Parameter
) -> torch.Tensor:
    ## fused dense moe
    y = torch.einsum("bse, bsg, geo -> bso", x, F.sigmoid(gate), weight_list)
    return y


def topk_sigmoid(
    x: torch.Tensor,
    topk: int
) -> torch.Tensor:
    _, idx = torch.topk(x, topk, dim=-1)
    mask = torch.zeros_like(x, device=x.device, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return F.sigmoid(x)*mask


def smoe_fwd(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight_list: nn.Parameter,
    topk: int
) -> torch.Tensor:
    ## fused sparse moe
    y = torch.einsum("bse, bsg, geo -> bso", x, topk_sigmoid(gate, topk), weight_list)
    return y


"""
IMoE: Indefinacy Mixture of Experts architecture dynamically selects neurons.
This implementation replicates the neurons
which is dynamically changing synapses
and enables efficient and diverse inference.

インディフィナシー・ミクスチャー・オブ・エキスパート：不定形性専門家混合機構

Appendix A: IMoE theory

Realization of the neural network based on random graph theory.
The growth process of brain neural circuits can be explained by the theory
that integrates Erdős-Rényi-Gilbert model
and fitness model (Bianconi-Barabási model).

During childhood, the brain neural circuits and synapses increase rapidly
and then synapses are pruned as the brain grows.
This growth process is equivalent to applying Erdős-Rényi-Gilbert model
then fitness model (Bianconi-Barabási model).

Step 1: Erdős-Rényi-Gilbert model like growth process

Individual synapses are not affected by the state of other synapses
and are probabilistically formed.
This phenomenon is represented by Erdős-Rényi-Gilbert model
and is achieved in the algorithm by a parallel definition of the modules.

Step 2: fitness model (Bianconi-Barabási model) like routing

Individual neurons have link coefficients
which affect connected synapses architecture.
These link coefficients change dynamically in response to the environment.
These link coefficients are a random distribution in childhood,
but converge to a constant distribution as they grow older.
This mechanism is realized by a dynamic, multi-level branching process
using softmax functions and non linear projections.

> gate = Softmax(Linear(x))

> x = Top-p(gate) * x
"""

def get_top_p(
    x: torch.Tensor,
    top_p: float = 0.3,
    temperature: float = 1.0,
    dim: int = -1,
    noise: float = 0.1,
    training: bool = False
) -> torch.Tensor:
    """
    ## The function of getting Top-p

    get vals and indices according to temperature

    outputs:
        top_p_vals: coefficients of each experts (experts score)
        top_p_indices: indices of experts
    """
    bsz, seql, _ = x.size()
    x = F.softmax(x, dim=dim).reshape(bsz*seql, -1)
    if training:
        if noise != 0:
            x = x + noise * top_p * torch.randn_like(x)
    if temperature != 1.0:
        x = x / temperature

    if top_p >= 1.0:
        ValueError('top_p should be less than 1.0. default value is 0.3.')

    top_p = (x > top_p)
    if top_p.any() is True:
        idx = top_p.indices()
        return x, idx
    else:
        return x, torch.zeros(0, device=x.device)


def topp_sigmoid(
    x: torch.Tensor,
    top_p: float = 0.3,
    temperature: float = 1.0,
    dim: int = -1,
    noise: float = 0.1,
    training: bool = False
) -> torch.Tensor:
    _, idx = get_top_p(x, top_p, temperature, dim, noise, training)
    mask = torch.zeros_like(x, device=x.device, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return F.sigmoid(x)*mask


def imoe_fwd(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight_list: nn.Parameter,
    top_p: float = 0.3,
    temperature: float = 1.0,
    dim: int = -1,
    noise: float = 0.1,
    training: bool = False
) -> torch.Tensor:
    ## fused sparse moe
    y = torch.einsum(
        "bse, bsg, geo -> bso",
        x,
        topp_sigmoid(gate, top_p, temperature, dim, noise, training),
        weight_list
    )
    return y


class FusedOneMoE(nn.Module):
    def __init__(
        self,
        hidden_size: int = 512,
        intermediate_size: int = 3072,
        groups: int = 12,
        is_sparse: bool = False,
        topk: int = 1,
        is_indefinacy: bool = False,
        topp: float = 0.3,
        temperature: float = 1.0,
        noise: float = 0.1,
        training: bool = False,
        bias: bool = False,
        device: Any | None = None,
        dtype: Any | None = None
    ):
        """
        ## Kernel Fused One Line MoE class 
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.bias = bias
        self.is_sparse = is_sparse
        self.topk = topk
        self.is_indefinacy = is_indefinacy
        self.topp = topp
        self.temperature = temperature
        self.noise = noise
        self.training = training
        self.WG = nn.Parameter(
            torch.randn(
                (hidden_size, groups),
                device=device,
                dtype=dtype
            )
        )
        self.Wi = nn.Parameter(
            torch.randn(
                (groups, hidden_size, intermediate_size),
                device=device,
                dtype=dtype
            )
        )
        nn.init.kaiming_normal_(self.WG, a=math.sqrt(0.5))
        nn.init.kaiming_normal_(self.Wi, a=math.sqrt(0.5))
        if bias:
            self.up_bias = nn.Parameter(
                torch.randn(
                    (intermediate_size),
                    device=device,
                    dtype=dtype
                )
            )
            nn.init.kaiming_normal_(self.up_bias, a=math.sqrt(0.5))
        self.down = nn.Parameter(
            torch.randn(
                (intermediate_size, hidden_size),
                device=device,
                dtype=dtype
            )
        )
        nn.init.kaiming_normal_(self.down, a=math.sqrt(0.5))
        if bias:
            self.down_bias = nn.Parameter(
                    torch.randn(
                        (hidden_size),
                        device=device,
                    dtype=dtype
                )
            )
            nn.init.kaiming_normal_(self.down_bias, a=math.sqrt(0.5))

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        gate = torch.matmul(hidden_states, self.WG)

        if self.is_sparse:
            if self.is_indefinacy:
                hidden_states = imoe_fwd(
                    hidden_states, gate, self.Wi, self.topp,
                    self.temperature, -1, self.noise, self.training
                )
            else:
                hidden_states = smoe_fwd(hidden_states, gate, self.Wi, self.topk)
        else:
            hidden_states = moe_fwd(hidden_states, gate, self.Wi)
        if self.bias:
            hidden_states = hidden_states + self.up_bias
        hidden_states = F.relu(hidden_states).square()
        hidden_states = torch.matmul(hidden_states, self.down)
        if self.bias:
            hidden_states = hidden_states + self.down_bias

        return hidden_states
