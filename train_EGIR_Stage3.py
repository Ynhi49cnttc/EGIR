"""
train_EGIR_Stage3.py — Iterative Neuro-Symbolic Reasoning (bản EGIR-v2).

Load Stage 1 (tiếp tục fine-tune) + load & ĐÓNG BĂNG Stage 2 (R-GCN, w_r, c_r,
attention). Vì Stage 2 đóng băng và Class Center cố định, K là MỘT TENSOR
HẰNG SỐ tính một lần trước khi train — không tính lại mỗi batch/mỗi bước T.

CHẠY SAU train_EGIR_Stage2.py:
  python train_EGIR_Stage3.py --dataset DISFA --fold 1 -e 10 \
      --resume <best_model_fold1.pth Stage1> --resume-phase2 <stage2_fold1.pth>
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
from model.egir_stage3 import EGIR_Stage3
import graph_rules as G
from dataset import *
from utils import *
from conf import get_config, set_logger, set_outdir, set_env


def get_dataloader(conf):
    if conf.dataset == 'DISFA':
        trainset = DISFA(conf.dataset_path, train=True, fold=conf.fold,
                          transform=image_train(crop_size=conf.crop_size),
                          crop_size=conf.crop_size, stage=1)
        valset = DISFA(conf.dataset_path, train=False, fold=conf.fold,
                        transform=image_test(crop_size=conf.crop_size), stage=1)
    else:
        trainset = BP4D(conf.dataset_path, train=True, fold=conf.fold,
                         transform=image_train(crop_size=conf.crop_size),
                         crop_size=conf.crop_size, stage=1)
        valset = BP4D(conf.dataset_path, train=False, fold=conf.fold,
                       transform=image_test(crop_size=conf.crop_size), stage=1)
    train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True,
                               num_workers=conf.num_workers)
    val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False,
                             num_workers=conf.num_workers)
    return train_loader, val_loader


def compute_total_loss(out, y_au, y_ex, conf, au_criterion, ex_criterion,
                        lambda1, lambda2, lambda3, mono_eps, lambda_e):
    L_wa = au_criterion(out['final_au'], y_au)
    L_we = ex_criterion(out['final_ex'], y_ex)
    L_energy = torch.stack(out['energies']).mean() if out['energies'] else torch.tensor(0.0, device=y_au.device)
    L_mono = torch.tensor(0.0, device=y_au.device)
    for t in range(1, len(out['energies'])):
        L_mono = L_mono + F.relu(out['energies'][t] - out['energies'][t-1] + mono_eps).mean()
    L_deep = torch.tensor(0.0, device=y_au.device)
    n_steps = len(out['preds_au'])
    for t, (p_a_t, p_e_t) in enumerate(zip(out['preds_au'], out['preds_ex'])):
        beta_t = (t + 1) / n_steps
        L_deep = L_deep + beta_t * (au_criterion(p_a_t, y_au) + lambda_e * ex_criterion(p_e_t, y_ex))
    total = L_wa + conf.lam * L_we + lambda1 * L_energy + lambda2 * L_mono + lambda3 * L_deep
    return total, {'L_wa': L_wa.item(), 'L_we': L_we.item(), 'L_energy': L_energy.item(),
                    'L_mono': L_mono.detach().item(), 'L_deep': L_deep.detach().item()}


def main(conf):
    T_STEPS = 2
    ETA = 0.05
    LAMBDA1, LAMBDA2, LAMBDA3 = 0.1, 0.1, 0.5
    MONO_EPS, LAMBDA_E = 0.01, 0.3

    M_AE_np = np.load(os.path.join("matrixMAE", "M_AE_DISFA.npy"))
    M_AE = torch.from_numpy(M_AE_np).float()

    train_loader, val_loader = get_dataloader(conf)

    net_stage1 = MEFARGStage1(num_aus=conf.num_classes, backbone=conf.arc, num_expr=7)
    assert conf.resume != '', "Cần --resume trỏ tới checkpoint Stage 1"
    net_stage1 = load_state_dict(net_stage1, conf.resume)

    assert conf.resume_phase2 != '', "Cần --resume-phase2 trỏ tới checkpoint Stage 2 (stage2_foldN.pth)"
    stage2_ckpt = torch.load(conf.resume_phase2, map_location='cpu')
    emb_dim = stage2_ckpt['emb_dim']

    model = EGIR_Stage3(net_stage1, emb_dim)
    model.graph_module.load_state_dict(stage2_ckpt['graph_module'])
    for p in model.graph_module.parameters():
        p.requires_grad = False

    if torch.cuda.is_available():
        model = model.cuda()
        M_AE = M_AE.cuda()

    node_init_fixed = stage2_ckpt['node_init_fixed'].to(next(model.parameters()).device)
    with torch.no_grad():
        K_fixed = model.graph_module.kg_encoder(node_init_fixed.unsqueeze(0)).squeeze(0)
    logging.info(f"[Stage2] Đã load & đóng băng graph_module. K_fixed shape={tuple(K_fixed.shape)} "
                 f"(hằng số dùng chung mọi ảnh/mọi batch/mọi bước T).")

    train_weight_path = os.path.join(conf.dataset_path, 'list',
                                      f'{conf.dataset}_train_weight_fold{conf.fold}.txt')
    train_weight = torch.from_numpy(np.loadtxt(train_weight_path)).float()
    if torch.cuda.is_available():
        train_weight = train_weight.cuda()
    au_criterion = WeightedAsymmetricLoss(weight=train_weight)
    ex_criterion = ExpressionBCELoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in trainable_params)
    logging.info(f"[Optimizer] Huấn luyện {n_train:,}/{n_total:,} tham số ({100*n_train/n_total:.1f}%).")
    optimizer = optim.AdamW(trainable_params, betas=(0.9, 0.999), lr=conf.learning_rate,
                             weight_decay=conf.weight_decay)

    best_f1 = 0.0
    for epoch in range(conf.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Stage3 Fold{conf.fold} Epoch{epoch+1}")
        for inputs, targets in pbar:
            targets = targets.float()
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            targets_Emo = G.au_to_expr_pseudo(targets, M_AE, neutral_index=6)

            optimizer.zero_grad()
            out = model(inputs, K_fixed, T=T_STEPS, eta=ETA)
            loss, parts = compute_total_loss(out, targets, targets_Emo, conf,
                                              au_criterion, ex_criterion,
                                              LAMBDA1, LAMBDA2, LAMBDA3, MONO_EPS, LAMBDA_E)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({k: f"{v:.4f}" for k, v in parts.items()})

        model.eval()
        statistics_list = None
        for inputs, targets in tqdm(val_loader, desc="Eval"):
            targets = targets.float()
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            out = model(inputs, K_fixed, T=T_STEPS, eta=ETA)
            update_list = statistics(out['final_au'].detach(), targets.detach(), 0.5)
            statistics_list = update_statistics_list(statistics_list, update_list)

        mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
        infostr = f"Stage3 Fold{conf.fold} Epoch {epoch+1}: mean F1 = {100.*mean_f1_score:.2f}%"
        print(infostr); logging.info(infostr)

        if mean_f1_score > best_f1:
            best_f1 = mean_f1_score
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'best_f1': best_f1},
                       os.path.join(conf['outdir'], f'final_model_fold{conf.fold}.pth'))
            logging.info(f"[BEST] Epoch {epoch+1}: F1={100.*best_f1:.2f}%")

    print(f"===== Fold {conf.fold} xong. Best F1 = {100.*best_f1:.2f}% =====")


if __name__ == "__main__":
    conf = get_config()
    set_env(conf)
    set_outdir(conf)
    set_logger(conf)
    main(conf)
