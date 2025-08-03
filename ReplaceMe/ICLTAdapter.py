
# ICLTAdapter.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class ICLTAdapter(nn.Module):
    def __init__(self, U_dir, d_model, K, rank, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.K = K
        self.rank = rank
        self.d_model = d_model
        self.device = device
        self.dtype = dtype

        self.U = nn.ParameterList()
        self.V = nn.ParameterList()

        for k in range(K):
            U_k = torch.load(os.path.join(U_dir, f"U_cluster_{k}.pt"), map_location=device).to(dtype)
            V_k = torch.load(os.path.join(U_dir, f"V_cluster_{k}.pt"), map_location=device).to(dtype)
            self.U.append(nn.Parameter(U_k))
            self.V.append(nn.Parameter(V_k))

        self.alpha_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, K)
        ).to(dtype)

    def forward(self, x):
        # x: [B, S, D]
        x = x.to(self.dtype)
        bsz, seqlen, d = x.shape

        # compute soft assignment
        alpha_logits = self.alpha_mlp(x)  # [B, S, K]
        alpha = F.softmax(alpha_logits, dim=-1)

        # build dynamic T_eff
        T_eff = torch.zeros(bsz, seqlen, d, d, dtype=self.dtype, device=x.device)
        for k in range(self.K):
            T_k = self.U[k] @ self.V[k]  # [d, d]
            T_k = T_k.unsqueeze(0).unsqueeze(0)  # [1, 1, d, d]
            T_eff += alpha[..., k].unsqueeze(-1).unsqueeze(-1) * T_k

        # apply T_eff to x
        x_out = torch.matmul(T_eff, x.unsqueeze(-1)).squeeze(-1)  # [B, S, D]
        return x_out
