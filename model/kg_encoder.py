"""
model/kg_encoder.py — R-GCN encoder cho đồ thị tri thức FACS đa quan hệ (Stage 2 EGIR-v2).

Khác với model/graph.py + model/graph_edge_model.py gốc (dùng cho ANFL/MEFL,
đồ thị k-NN học từ dữ liệu), module này lan truyền thông tin trên đồ thị
CỐ ĐỊNH (implies/excludes/co_occurs/component_of) xây từ graph_rules.py.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import graph_rules as G


class RGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_rel):
        super().__init__()
        self.W_rel = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_rel)])
        self.W_self = nn.Linear(in_dim, out_dim)

    def forward(self, H, A_list):
        """H: (B, N, in_dim); A_list: list adjacency (N,N) đã chuẩn hoá theo degree, 1/quan hệ."""
        out = self.W_self(H)
        for r, A in enumerate(A_list):
            agg = torch.einsum('nm,bmd->bnd', A, H)
            out = out + self.W_rel[r](agg)
        return F.relu(out)


class KnowledgeGraphEncoder(nn.Module):
    def __init__(self, n_node, n_rel, emb_dim):
        super().__init__()
        self.n_node = n_node
        self.layer1 = RGCNLayer(emb_dim, emb_dim, n_rel)
        self.layer2 = RGCNLayer(emb_dim, emb_dim, n_rel)
        self.register_buffer('A_list', self._build_adjacency())

    def _build_adjacency(self):
        A = torch.zeros(len(G.REL_TYPES), self.n_node, self.n_node)
        for src, dst, rel in G.STRUCT_EDGES:
            A[rel, dst, src] = 1.0
        for r in range(len(G.REL_TYPES)):
            deg = A[r].sum(dim=1, keepdim=True).clamp(min=1.0)
            A[r] = A[r] / deg
        return A

    def forward(self, node_init):
        """node_init: (B, N_node, emb_dim) hoặc (1, N_node, emb_dim)."""
        A_list = [self.A_list[r] for r in range(len(G.REL_TYPES))]
        h = self.layer1(node_init, A_list)
        h = self.layer2(h, A_list)
        return h


class Stage2GraphModule(nn.Module):
    """Gồm: R-GCN + trọng số w_r (theo loại quan hệ) + c_r (theo từng cạnh,
       khởi tạo từ M_AE) + Cross-Attention query (W_Q). Được train ở Stage 2,
       ĐÓNG BĂNG hoàn toàn khi sang Stage 3 (giống cách SymGraphAU đóng băng
       GCN/logic embedder ở Phase 3)."""
    def __init__(self, mid_dim, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.kg_encoder = KnowledgeGraphEncoder(G.N_NODE, len(G.REL_TYPES), emb_dim)
        self.W_Q = nn.Linear(mid_dim, emb_dim)
        self.w_r = nn.Parameter(torch.ones(len(G.REL_TYPES)))
        c_init_t = torch.tensor(G.C_INIT, dtype=torch.float32).clamp(1e-3, 1 - 1e-3)
        self.c_r = nn.Parameter(torch.log(c_init_t / (1 - c_init_t)))

    def compute_energy(self, p_a, p_e, alpha):
        B = p_a.shape[0]
        E = torch.zeros(B, device=p_a.device)
        for e_id, edge in enumerate(G.ENERGY_EDGES):
            i, j, rel = edge['i'], edge['j'], edge['rel']
            a = G.get_node_value(i, p_a, p_e)
            b = G.get_node_value(j, p_a, p_e)
            viol = G.VIOL_FN[rel](a, b)
            w = F.softplus(self.w_r[G.REL2ID[rel]])
            c = torch.sigmoid(self.c_r[e_id])
            alpha_edge = 0.5 * (alpha[:, i] + alpha[:, j])
            E = E + alpha_edge * w * c * viol
        return E

    def forward(self, feat, node_init, p_a, p_e):
        """feat: (B, D, mid_dim) — dùng để tính query attention.
           node_init: (N_node, emb_dim) — Class Center cố định (không batch)."""
        B = feat.shape[0]
        K = self.kg_encoder(node_init.unsqueeze(0).expand(B, -1, -1))
        q = self.W_Q(feat.mean(dim=1))
        attn_logits = torch.einsum('bd,bnd->bn', q, K) / math.sqrt(self.emb_dim)
        alpha = torch.softmax(attn_logits, dim=-1)
        E = self.compute_energy(p_a, p_e, alpha)
        return E, K
