"""
train_EGIR_Stage1.py — GẦN NHƯ NGUYÊN VĂN train_Sym_Stage_1.py của SymGraphAU.

Đây là bước 1/3 của EGIR-v2: train Backbone + AU head + Expression head
(model.SymStage1.MEFARGStage1). Checkpoint kết quả sẽ được Stage 2 load lại
và đóng băng — không sửa gì so với quy trình gốc.

Thay đổi duy nhất so với bản gốc: đường dẫn M_AE dùng os.path.join thay vì
chuỗi Windows cứng (r"matrixMAE\\M_AE_DISFA.npy") để chạy được cả Linux/Mac.

CHUẨN BỊ TRƯỚC KHI CHẠY:
  1. Đổ dữ liệu thô vào data/DISFA/img/ và giữ ActionUnit_Labels gốc ở đâu đó.
  2. Chạy tool/DISFA_image_label_process.py để sinh list file (list/DISFA_*_fold*.txt).
  3. Chạy tool/DISFA_calculate_AU_class_weights.py để sinh weight file.
  4. python train_EGIR_Stage1.py --dataset DISFA --fold 1 -e 8
     (lặp lại với --fold 2, --fold 3)
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging
from datetime import datetime

from model.SymStage1 import MEFARGStage1
from dataset import *
from utils import *
from conf import get_config, set_logger, set_outdir, set_env


def get_dataloader(conf):
    print('==> Preparing data...')
    if conf.dataset == 'BP4D':
        trainset = BP4D(conf.dataset_path, train=True, fold=conf.fold,
                         transform=image_train(crop_size=conf.crop_size),
                         crop_size=conf.crop_size, stage=1)
        train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True,
                                   num_workers=conf.num_workers)
        valset = BP4D(conf.dataset_path, train=False, fold=conf.fold,
                       transform=image_test(crop_size=conf.crop_size), stage=1)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False,
                                 num_workers=conf.num_workers)

    elif conf.dataset == 'DISFA':
        trainset = DISFA(conf.dataset_path, train=True, fold=conf.fold,
                          transform=image_train(crop_size=conf.crop_size),
                          crop_size=conf.crop_size, stage=1)
        train_loader = DataLoader(trainset, batch_size=conf.batch_size, shuffle=True,
                                   num_workers=conf.num_workers)
        valset = DISFA(conf.dataset_path, train=False, fold=conf.fold,
                        transform=image_test(crop_size=conf.crop_size), stage=1)
        val_loader = DataLoader(valset, batch_size=conf.batch_size, shuffle=False,
                                 num_workers=conf.num_workers)

    return train_loader, val_loader, len(trainset), len(valset)


def au_to_expr_pseudo(Y_a: torch.Tensor, M_AE: torch.Tensor, neutral_index: int) -> torch.Tensor:
    """Y_a: (B, N_a) nhãn AU. M_AE: (N_a, N_e). Trả về Y_e (B, N_e) one-hot."""
    B, N_a = Y_a.shape
    N_e = M_AE.shape[1]
    Y_a_float = Y_a.float()
    scores = Y_a_float @ M_AE
    ke = scores.argmax(dim=1)
    neutral_mask = (Y_a_float.sum(dim=1) == 0)
    ke[neutral_mask] = neutral_index
    Y_e = torch.zeros(B, N_e, device=Y_a.device, dtype=Y_a_float.dtype)
    Y_e.scatter_(1, ke.unsqueeze(1), 1.0)
    return Y_e


M_AE_np = np.load(os.path.join("matrixMAE", "M_AE_DISFA.npy"))
M_AE = torch.from_numpy(M_AE_np).float()


def train(conf, net, train_loader, optimizer, epoch, criterion, criterion_Em):
    losses = AverageMeter()
    net.train()
    train_loader_len = len(train_loader)
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for batch_idx, (inputs, targets) in enumerate(pbar):
        adjust_learning_rate(optimizer, epoch, conf.epochs, conf.learning_rate,
                              batch_idx, train_loader_len)
        targets = targets.float()
        if torch.cuda.is_available():
            inputs = inputs.cuda(); targets = targets.cuda()

        targets_Emo = au_to_expr_pseudo(targets, M_AE, neutral_index=6)
        if torch.cuda.is_available():
            targets_Emo = targets_Emo.cuda()

        optimizer.zero_grad()
        V_a, V_e, outputs_AU, outputs_Emo = net(inputs)

        L_wa = criterion(outputs_AU, targets)
        L_we = criterion_Em(outputs_Emo, targets_Emo)
        gamma = conf.lam
        loss = L_wa + gamma * L_we

        loss.backward()
        optimizer.step()
        losses.update(loss.item(), inputs.size(0))
        pbar.set_postfix({'L_wa': f"{L_wa.item():.4f}", 'L_we': f"{L_we.item():.4f}",
                           'L_jf': f"{loss.item():.4f}"})

    return losses.avg


def val(net, val_loader, criterion):
    losses = AverageMeter()
    net.eval()
    statistics_list = None
    emo_statistics_list = None
    emo_correct = 0
    emo_total = 0
    pbar = tqdm(val_loader, desc="Val")

    for batch_idx, (inputs, targets) in enumerate(pbar):
        with torch.no_grad():
            targets = targets.float()
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            V_a, V_e, outputs_AU, outputs_Emo = net(inputs)
            loss = criterion(outputs_AU, targets)
            losses.update(loss.item(), inputs.size(0))
            update_list = statistics(outputs_AU, targets.detach(), 0.5)
            statistics_list = update_statistics_list(statistics_list, update_list)

            targets_Emo = au_to_expr_pseudo(targets, M_AE, neutral_index=6)
            emo_update = statistics(outputs_Emo, targets_Emo, 0.5)
            emo_statistics_list = update_statistics_list(emo_statistics_list, emo_update)
            emo_correct += (outputs_Emo.argmax(dim=1) == targets_Emo.argmax(dim=1)).sum().item()
            emo_total += targets.size(0)
            pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})

    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)
    emo_mean_f1, emo_f1_list = calc_f1_score(emo_statistics_list)
    emo_acc = emo_correct / emo_total
    return losses.avg, mean_f1_score, f1_score_list, mean_acc, acc_list, emo_mean_f1, emo_acc


def main(conf):
    if conf.dataset == 'BP4D':
        dataset_info = BP4D_infolist
    elif conf.dataset == 'DISFA':
        dataset_info = DISFA_infolist

    start_epoch = 0
    train_loader, val_loader, train_data_num, val_data_num = get_dataloader(conf)

    weight_path = os.path.join(conf.dataset_path, 'list',
                                f'{conf.dataset}_train_weight_fold{conf.fold}.txt')
    train_weight = np.loadtxt(weight_path)
    train_weight = torch.from_numpy(train_weight).float()
    print(f"[WAL] weight_path = {weight_path}")
    print(f"[WAL] w = {train_weight.tolist()}, sum={float(train_weight.sum()):.6f}")

    logging.info("Fold: [{} | {}  val_data_num: {} ]".format(conf.fold, conf.N_fold, val_data_num))

    net = MEFARGStage1(num_aus=conf.num_classes, backbone=conf.arc, num_expr=7)
    if conf.resume != '':
        logging.info("Resume form | {} ]".format(conf.resume))
        net = load_state_dict(net, conf.resume)

    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()
        train_weight = train_weight.cuda()
        global M_AE
        M_AE = M_AE.cuda()

    criterion = WeightedAsymmetricLoss(weight=train_weight)
    criterion_Em = ExpressionBCELoss()
    optimizer = optim.AdamW(net.parameters(), betas=(0.9, 0.999), lr=conf.learning_rate,
                             weight_decay=conf.weight_decay)
    print('the init learning rate is ', conf.learning_rate)

    start_time = datetime.now()
    print("Start time:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    best_f1 = 0.0

    for epoch in range(start_epoch, conf.epochs):
        print("--Weight will be Saved At",
              os.path.join(conf['outdir'], 'epoch' + str(epoch + 1) + '_model_fold' + str(conf.fold) + '.pth'))
        logging.info(f"[CFG] arc={conf.arc} bs={conf.batch_size} lr0={conf.learning_rate} wd={conf.weight_decay}")

        lr = optimizer.param_groups[0]['lr']
        logging.info("Epoch: [{} | {} LR: {} ]".format(epoch + 1, conf.epochs, lr))
        train_loss = train(conf, net, train_loader, optimizer, epoch, criterion, criterion_Em)
        val_loss, val_mean_f1_score, val_f1_score, val_mean_acc, val_acc, val_emo_f1, val_emo_acc = \
            val(net, val_loader, criterion)

        infostr = {'Epoch:  {}   train_loss: {:.5f}  val_loss: {:.5f}  AU_F1 {:.2f}  AU_Acc {:.2f}  '
                   'Emo_F1 {:.2f}  Emo_Acc {:.2f}'.format(
                       epoch + 1, train_loss, val_loss, 100. * val_mean_f1_score,
                       100. * val_mean_acc, 100. * val_emo_f1, 100. * val_emo_acc)}
        print(infostr)
        logging.info(infostr)

        if val_mean_f1_score > best_f1:
            best_f1 = val_mean_f1_score
            checkpoint = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, os.path.join(conf['outdir'], 'best_model_fold' + str(conf.fold) + '.pth'))
            logging.info(f"[BEST] Epoch {epoch+1}: F1={100.*best_f1:.2f}% -> best_model_fold{conf.fold}.pth")

        if (epoch + 1) % 4 == 0:
            checkpoint = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint,
                       os.path.join(conf['outdir'], 'epoch' + str(epoch + 1) + '_model_fold' + str(conf.fold) + '.pth'))

    end_time = datetime.now()
    print("End time:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Duration:", end_time - start_time)


if __name__ == "__main__":
    conf = get_config()
    set_env(conf)
    set_outdir(conf)
    set_logger(conf)
    main(conf)
