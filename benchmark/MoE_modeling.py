# coding = utf-8
# Copyright 2025 Rikka Botan. All rights reserved
# Licensed under "MIT License"
# Mixture of Experts

import torch
from torch import nn
import torch.nn.functional as F
from typing import Any
import math


class MoE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        inter_dim: int,
        gate_num: int,
        top_k: int = 4,
        bias: bool = False,
        device: Any | None = None,
        dtype: Any | None = None
    ):
        super().__init__()
        self.inter_dim = inter_dim
        self.gate_num = gate_num
        self.top_k = top_k
        self.experts = nn.ModuleList([
            nn.Linear(
                in_features=input_dim,
                out_features=inter_dim,
                bias=bias,
                device=device,
                dtype=dtype
            )
            for _ in range(gate_num)
        ])
        self.gate = nn.Linear(
            in_features=input_dim,
            out_features=gate_num,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.out_linear = nn.Linear(
            in_features=inter_dim,
            out_features=input_dim,
            bias=bias,
            device=device,
            dtype=dtype
        )
        nn.init.kaiming_normal_(self.gate.weight, a=math.sqrt(0.5))
        nn.init.kaiming_normal_(self.out_linear.weight, a=math.sqrt(0.5))
        for exp in self.experts:
            nn.init.kaiming_normal_(exp.weight, a=math.sqrt(0.5))


    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        bsz, seql, embs = x.size()
        x = x.reshape(-1, embs)
        gate_scores = self.gate(x)

        probs, indices = torch.topk(
            x = gate_scores,
            k=self.top_k,
            dim=-1
        )

        probs = probs.to(x.dtype)

        final_hidden_states = torch.zeros(
            (bsz*seql, self.inter_dim), dtype=x.dtype, device=x.device
        )

        for expert_idx in range(self.gate_num):
            mask = (indices == expert_idx).nonzero(as_tuple=True)[0]
            if mask.numel() == 0:
                continue

            current_state = x.index_select(0, mask)

            current_hidden_states = (
                self.experts[expert_idx](current_state)
                * probs[mask, 0].unsqueeze(-1)
            )
            
            final_hidden_states.index_add_(
                0,
                mask,
                current_hidden_states.to(x.dtype))

        outputs = final_hidden_states.reshape(bsz, seql, -1)
        outputs = self.out_linear(F.relu(outputs).square())

        return outputs

