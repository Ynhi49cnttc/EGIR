"""
model/egir_stage3.py — Model Stage 3: bọc lại MEFARGStage1 (từ SymStage1.py,
KHÔNG sửa file gốc) để lấy trực tiếp feature F^t (thứ SymStage1.forward()
không expose ra ngoài), cộng với Stage2GraphModule đã đóng băng, thực hiện
vòng lặp T bước Query -> Readout -> Energy -> Update.
"""
import math
import torch
import torch.nn as nn

from model.kg_encoder import Stage2GraphModule


class EGIR_Stage3(nn.Module):
    def __init__(self, net_stage1: nn.Module, emb_dim: int):
        """net_stage1: một instance MEFARGStage1 đã load checkpoint Stage 1
                        (backbone + au_head + expr_head sẽ tiếp tục fine-tune)."""
        super().__init__()
        self.net_stage1 = net_stage1
        self.graph_module = Stage2GraphModule(net_stage1.mid_channels, emb_dim)
        self.emb_dim = emb_dim

    def readout(self, F_t):
        """Gọi lại đúng au_head/expr_head của MEFARGStage1 trên F_t bất kỳ
           (không chỉ F^0) — đây chính là cơ chế 'Readout dùng chung head'."""
        V_a, p_a = self.net_stage1.au_head(F_t)
        V_e, p_e = self.net_stage1.expr_head(F_t)
        return p_a, p_e

    def forward(self, x, K_fixed, T, eta):
        """K_fixed: (N_node, emb_dim) — hằng số, tính 1 lần từ Stage 2 (đã đóng băng)."""
        feat = self.net_stage1.backbone(x)
        feat = self.net_stage1.global_linear(feat)          # F^0
        F_t = feat.requires_grad_(True)

        p_a0, p_e0 = self.readout(F_t)
        preds_au, preds_ex, energies = [p_a0], [p_e0], []

        for t in range(T):
            p_a_t, p_e_t = self.readout(F_t)

            q = self.graph_module.W_Q(F_t.mean(dim=1))
            attn_logits = torch.einsum('bd,nd->bn', q, K_fixed) / math.sqrt(self.emb_dim)
            alpha = torch.softmax(attn_logits, dim=-1)

            E_t = self.graph_module.compute_energy(p_a_t, p_e_t, alpha)
            energies.append(E_t)

            grad_F = torch.autograd.grad(E_t.sum(), F_t, create_graph=self.training,
                                          retain_graph=True)[0]
            F_t = F_t - eta * grad_F
            preds_au.append(p_a_t); preds_ex.append(p_e_t)

        p_a_f, p_e_f = self.readout(F_t)
        preds_au.append(p_a_f); preds_ex.append(p_e_f)

        return {'preds_au': preds_au, 'preds_ex': preds_ex, 'energies': energies,
                'final_au': p_a_f, 'final_ex': p_e_f}
