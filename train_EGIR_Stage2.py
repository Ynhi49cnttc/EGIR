"""
train_EGIR_Stage2.py — Graph Construction of Knowledge Embedding (bản EGIR-v2).

Khác với train_Sym_Stage_2.py gốc (CNF + PySAT + triplet loss), file này:
  - Đóng băng TOÀN BỘ Stage 1 (giống repo gốc).
  - Tính Class Center MỘT LẦN duy nhất (compute_class_centers, không lặp mỗi batch).
  - Train Stage2GraphModule (R-GCN + w_r + c_r + Cross-Attention) trên
    ẢNH THẬT (không sinh synthetic proposition) — đây là điểm khác biệt
    chủ đích so với SymGraphAU gốc.

CHẠY SAU train_EGIR_Stage1.py:
  python train_EGIR_Stage2.py --dataset DISFA --fold 1 -e 5 --resume <path best_model_fold1.pth>
"""
import os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging

from model.SymStage1 import MEFARGStage1
from model.kg_encoder import Stage2GraphModule
import graph_rules as G
from dataset import *
from utils import *
from conf import get_config, set_logger, set_outdir, set_env


def get_dataloader(conf):
    if conf.dataset == 'DISFA':
        trainset = DISFA(conf.dataset_path, train=True, fold=conf.fold,
                          transform=image_train(crop_size=conf.crop_size),
                          crop_size=conf.crop_size, stage=1)
    else:
        trainset = BP4D(conf.dataset_path, train=True, fold=conf.fold,
                         transform=image_train(crop_size=conf.crop_size),
                         crop_size=conf.crop_size, stage=1)
    train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True,
                               num_workers=conf.num_workers)
    return train_loader


@torch.no_grad()
def compute_class_centers(net_stage1, loader, num_aus, num_expr, emb_dim, M_AE, neutral_index):
    """MỘT LƯỢT duy nhất qua train set — không tính lại ở epoch sau."""
    net_stage1.eval()
    sum_a = torch.zeros(num_aus, emb_dim); cnt_a = torch.zeros(num_aus)
    sum_e = torch.zeros(num_expr, emb_dim); cnt_e = torch.zeros(num_expr)

    for inputs, targets in tqdm(loader, desc="Tính Class Center (1 lượt)"):
        targets = targets.float()
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        V_a, V_e, p_a, p_e = net_stage1(inputs)
        V_a, V_e = V_a.cpu(), V_e.cpu()

        for i in range(num_aus):
            mask = targets[:, i] == 1
            if mask.sum() > 0:
                sum_a[i] += V_a[mask, i, :].sum(dim=0); cnt_a[i] += mask.sum()

        targets_Emo = G.au_to_expr_pseudo(targets, M_AE.cpu(), neutral_index)
        for m in range(num_expr):
            mask = targets_Emo[:, m] == 1
            if mask.sum() > 0:
                sum_e[m] += V_e[mask, m, :].sum(dim=0); cnt_e[m] += mask.sum()

    centers_au = sum_a / cnt_a.clamp(min=1).unsqueeze(1)
    centers_expr = sum_e / cnt_e.clamp(min=1).unsqueeze(1)
    return centers_au, centers_expr


def build_node_init(centers_au, centers_expr, emb_dim, device):
    node_init = torch.zeros(G.N_NODE, emb_dim, device=device)
    node_init[:G.N_AU, :] = centers_au.to(device)
    node_init[G.N_AU:G.N_AU + G.N_EXPR, :] = centers_expr.to(device)
    for c_id, comp in enumerate(G.COMPONENTS):
        idxs = [G.AU_IDX[au] for au in comp['members']]
        node_init[G.COMP_OFFSET + c_id, :] = centers_au[idxs].mean(dim=0).to(device)
    return node_init


def main(conf):
    EMB_DIM = 256
    M_AE_np = np.load(os.path.join("matrixMAE", "M_AE_DISFA.npy"))
    M_AE = torch.from_numpy(M_AE_np).float()

    train_loader = get_dataloader(conf)

    # ---- Load & đóng băng TOÀN BỘ Stage 1 (đúng tinh thần repo gốc) ----
    net_stage1 = MEFARGStage1(num_aus=conf.num_classes, backbone=conf.arc, num_expr=7)
    assert conf.resume != '', "Cần --resume trỏ tới checkpoint Stage 1 (best_model_foldN.pth)"
    net_stage1 = load_state_dict(net_stage1, conf.resume)
    if torch.cuda.is_available():
        net_stage1 = net_stage1.cuda()
        M_AE = M_AE.cuda()
    for p in net_stage1.parameters():
        p.requires_grad = False
    net_stage1.eval()
    logging.info(f"[Stage1] Đã load & đóng băng: {conf.resume}")

    # ---- Class Center: MỘT LẦN duy nhất ----
    centers_au, centers_expr = compute_class_centers(
        net_stage1, train_loader, conf.num_classes, 7, EMB_DIM, M_AE, neutral_index=6)
    device = next(net_stage1.parameters()).device
    node_init_fixed = build_node_init(centers_au, centers_expr, EMB_DIM, device)

    mid_dim = net_stage1.module.mid_channels if hasattr(net_stage1, 'module') else net_stage1.mid_channels
    graph_module = Stage2GraphModule(mid_dim, EMB_DIM).to(device)
    optimizer = optim.AdamW(graph_module.parameters(), lr=conf.learning_rate,
                             weight_decay=conf.weight_decay)

    c_init_t = torch.tensor(G.C_INIT, device=device)

    for epoch in range(conf.epochs):
        graph_module.train()
        pbar = tqdm(train_loader, desc=f"Stage2 Fold{conf.fold} Epoch{epoch+1}")
        for inputs, targets in pbar:
            targets = targets.float()
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            with torch.no_grad():
                feat = net_stage1.backbone(inputs) if not hasattr(net_stage1, 'module') \
                    else net_stage1.module.backbone(inputs)
                gl = net_stage1.global_linear if not hasattr(net_stage1, 'module') else net_stage1.module.global_linear
                feat = gl(feat)
                au_head = net_stage1.au_head if not hasattr(net_stage1, 'module') else net_stage1.module.au_head
                expr_head = net_stage1.expr_head if not hasattr(net_stage1, 'module') else net_stage1.module.expr_head
                _, p_a = au_head(feat)
                _, p_e = expr_head(feat)

            optimizer.zero_grad()
            E, _ = graph_module(feat, node_init_fixed, p_a, p_e)
            L_energy = E.mean()
            w_all = F.softplus(graph_module.w_r)
            c_all = torch.sigmoid(graph_module.c_r)
            L_reg = ((w_all - 1) ** 2).mean() + ((c_all - c_init_t) ** 2).mean()
            loss = L_energy + 0.05 * L_reg
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'L_energy': f"{L_energy.item():.4f}", 'L_reg': f"{L_reg.item():.4f}"})

        infostr = f"Stage2 Fold{conf.fold} Epoch {epoch+1}: L_energy_last_batch={L_energy.item():.4f}"
        print(infostr); logging.info(infostr)

    ckpt_path = os.path.join(conf['outdir'], f"stage2_fold{conf.fold}.pth")
    torch.save({
        'graph_module': graph_module.state_dict(),
        'node_init_fixed': node_init_fixed.cpu(),
        'centers_au': centers_au, 'centers_expr': centers_expr,
        'mid_dim': mid_dim, 'emb_dim': EMB_DIM,
    }, ckpt_path)
    logging.info(f"[LƯU] {ckpt_path}")
    print(f"[LƯU] {ckpt_path}")


if __name__ == "__main__":
    conf = get_config()
    set_env(conf)
    set_outdir(conf)
    set_logger(conf)
    main(conf)
