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


def smoe_fwd(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight_list: nn.Parameter,
    topk: int
) -> torch.Tensor:
    ## fused sparse moe
    _, idx = torch.topk(gate, topk, dim=-1)
    mask = torch.zeros_like(gate, device=gate.device)
    mask.scatter_(-1, idx, True)
    y = torch.einsum("bse, bsg, geo -> bso", x, F.sigmoid(gate*mask), weight_list)
    return y


class FusedOneMoE(nn.Module):
    def __init__(
        self,
        hidden_size: int = 512,
        intermediate_size: int = 3072,
        groups: int = 12,
        is_sparse: bool = False,
        topk: int = 1,
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
