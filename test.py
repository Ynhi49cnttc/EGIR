"""
  python test.py --dataset DISFA --fold 2 --arc resnet50 -b 32 --resume results/egir_v2/bs_64_seed_0_lr_1e-06/final_model_fold2.pth --resume-phase2 results/egir_v2/bs_64_seed_0_lr_0.001/stage2_fold2.pth
Add --t-steps 0 to also/only run the reasoning-off control for comparison.
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.SymStage1 import MEFARGStage1
from model.egir_stage3 import EGIR_Stage3
import graph_rules as G
from dataset import *
from utils import *
from conf import get_config, set_env


def get_test_loader(conf):
    if conf.dataset == 'DISFA':
        testset = DISFA(conf.dataset_path, train=False, fold=conf.fold,
                         transform=image_test(crop_size=conf.crop_size), stage=1)
    else:
        testset = BP4D(conf.dataset_path, train=False, fold=conf.fold,
                        transform=image_test(crop_size=conf.crop_size), stage=1)
    return DataLoader(testset, batch_size=conf.batch_size, shuffle=False,
                       num_workers=conf.num_workers)


def run_eval(model, test_loader, K_fixed, T, eta, device, au_list):
    model.eval()
    statistics_list = None

    for inputs, targets in tqdm(test_loader, desc=f"Testing (T={T})"):
        targets = targets.float()
        if device.type == 'cuda':
            inputs, targets = inputs.cuda(), targets.cuda()
        out = model(inputs, K_fixed, T=T, eta=eta)
        pred = out['final_au'].detach()

        update_list = statistics(pred, targets.detach(), 0.5)
        statistics_list = update_statistics_list(statistics_list, update_list)

    mean_f1, f1_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)

    print(f"\n{'='*68}")
    print(f"TEST RESULTS — T={T} (eta={eta})")
    print(f"{'='*68}")
    print(f"{'AU':<10}{'F1 (%)':>12}{'Acc (%)':>12}")
    print(f"{'-'*68}")
    for i, au in enumerate(au_list):
        print(f"AU{au:<8}{100.*f1_list[i]:>12.2f}{100.*acc_list[i]:>12.2f}")
    print(f"{'-'*68}")
    print(f"{'Mean':<10}{100.*mean_f1:>12.2f}{100.*mean_acc:>12.2f}")
    print(f"{'='*68}\n")

    return mean_f1, f1_list, mean_acc, acc_list


def main(conf):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    assert conf.resume != '', "Need --resume pointing to final_model_fold{N}.pth (Stage 3 checkpoint)"
    assert conf.resume_phase2 != '', "Need --resume-phase2 pointing to stage2_fold{N}.pth (for node_init_fixed)"

    test_loader = get_test_loader(conf)

    stage2_ckpt = torch.load(conf.resume_phase2, map_location='cpu')
    emb_dim = stage2_ckpt['emb_dim']
    node_init_fixed = stage2_ckpt['node_init_fixed'].to(device)

    # Build an empty architecture skeleton — real weights are loaded from final_model below
    net_stage1 = MEFARGStage1(num_aus=conf.num_classes, backbone=conf.arc, num_expr=7)
    model = EGIR_Stage3(net_stage1, emb_dim).to(device)

    final_ckpt = torch.load(conf.resume, map_location=device)
    model.load_state_dict(final_ckpt['model'])
    print(f"[Loaded] {conf.resume}")
    print(f"[Checkpoint info] saved at epoch: {final_ckpt.get('epoch', '?')}, "
          f"best_f1 during training: {100.*final_ckpt.get('best_f1', 0):.2f}%")

    model.eval()
    with torch.no_grad():
        K_fixed = model.graph_module.kg_encoder(node_init_fixed.unsqueeze(0)).squeeze(0)

    au_list = G.AU_LIST

    T_STEPS = conf.t_steps
    ETA = conf.eta
    mean_f1, f1_list, mean_acc, acc_list = run_eval(
        model, test_loader, K_fixed, T=T_STEPS, eta=ETA, device=device, au_list=au_list)

    # ---- Auto control: if you didn't explicitly ask for T=0, run it too for direct comparison ----
    if T_STEPS != 0:
        print("Running additional T=0 control (reasoning disabled) for direct comparison...")
        mean_f1_t0, f1_list_t0, mean_acc_t0, acc_list_t0 = run_eval(
            model, test_loader, K_fixed, T=0, eta=ETA, device=device, au_list=au_list)

        print(f"{'='*68}")
        print(f"COMPARISON — Fold {conf.fold}")
        print(f"{'='*68}")
        print(f"  With reasoning (T={T_STEPS}): Mean F1 = {100.*mean_f1:.2f}%")
        print(f"  Without reasoning (T=0)   : Mean F1 = {100.*mean_f1_t0:.2f}%")
        delta = 100. * (mean_f1 - mean_f1_t0)
        print(f"  Delta from reasoning      : {delta:+.2f} F1 points")
        print(f"{'='*68}\n")

    # Save detailed results next to the checkpoint for easy cross-fold comparison later
    out_path = os.path.join(os.path.dirname(conf.resume), f"test_result_fold{conf.fold}.txt")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"Fold {conf.fold} — T={T_STEPS}, eta={ETA}\n")
        f.write(f"Mean F1: {100.*mean_f1:.2f}%\n")
        f.write(f"Mean Acc: {100.*mean_acc:.2f}%\n")
        for i, au in enumerate(au_list):
            f.write(f"AU{au}: F1={100.*f1_list[i]:.2f}%  Acc={100.*acc_list[i]:.2f}%\n")
        if T_STEPS != 0:
            f.write(f"\n[Control T=0] Mean F1: {100.*mean_f1_t0:.2f}%\n")
            f.write(f"Delta from reasoning: {100.*(mean_f1 - mean_f1_t0):+.2f} F1 points\n")
    print(f"[Saved] Detailed results at: {out_path}")


if __name__ == "__main__":
    conf = get_config()
    set_env(conf)
    main(conf)
