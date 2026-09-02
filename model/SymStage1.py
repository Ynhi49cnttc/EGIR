import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .swin_transformer import (
    swin_transformer_tiny,
    swin_transformer_small,
    swin_transformer_base,
)
from .resnet import resnet18, resnet50, resnet101
from .basic_block import *   # dùng LinearBlock cho global_linear


EMB_DIM = 256  # kích thước embedding cố định cho cả AU và Expression


# ------------------------------------------------------------
#  Conv1D extractor cho 1 AU / 1 Emotion (phương án A)
#  Input : x (B, D, C_in)
#  Output: emb (B, C_emb=EMB_DIM)
# ------------------------------------------------------------
class Conv1DExtractor(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hid_channels: int,
                 emb_channels: int = EMB_DIM):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hid_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(hid_channels)
        self.conv2 = nn.Conv1d(hid_channels, emb_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(emb_channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        x: (B, D, C_in)
        Return:
            emb: (B, C_emb)
        """
        # (B, D, C_in) -> (B, C_in, D) cho Conv1d
        x = x.transpose(1, 2)
        # Conv1
        x = self.conv1(x)          # (B, C_hid, D)
        x = self.bn1(x)
        x = self.relu(x)
        # Conv2
        x = self.conv2(x)          # (B, C_emb, D)
        x = self.bn2(x)
        x = self.relu(x)
        # GAP theo chiều D
        emb = x.mean(dim=-1)       # (B, C_emb)
        return emb


# ------------------------------------------------------------
#  AU Head: N_a nhánh, mỗi nhánh = extractor riêng cho 1 AU
# ------------------------------------------------------------
class AUHead(nn.Module):
    """
    Joint AU head cho Stage-1 (JFL).

    Input:
        x: (B, D, C_in)  # feature map từ backbone (đã qua global_linear)
    Output:
        V_a: (B, N_a, EMB_DIM)   # AU embeddings V^a_i
        p_a: (B, N_a)            # AU probabilities (dùng cho L_wa)
    """
    def __init__(self,
                 in_channels: int,
                 num_aus: int,
                 hid_channels: int):
        super().__init__()
        self.in_channels  = in_channels
        self.num_aus      = num_aus
        self.hid_channels = hid_channels
        self.emb_channels = EMB_DIM

        # Mỗi AU có 1 Conv1DExtractor riêng
        self.extractors = nn.ModuleList([
            Conv1DExtractor(
                in_channels=self.in_channels,
                hid_channels=self.hid_channels,
                emb_channels=self.emb_channels,
            )
            for _ in range(self.num_aus)
        ])

        # Classifier cho từng AU: EMB_DIM -> 1
        self.classifiers = nn.ModuleList([
            nn.Linear(self.emb_channels, 1)
            for _ in range(self.num_aus)
        ])

    def forward(self, x):
        """
        x: (B, D, C_in)
        """
        emb_list = []
        logit_list = []

        # Lặp từng AU extractor
        for i in range(self.num_aus):
            emb_i = self.extractors[i](x)          # (B, EMB_DIM)
            emb_list.append(emb_i.unsqueeze(1))    # (B, 1, EMB_DIM)
            logit_i = self.classifiers[i](emb_i)   # (B, 1)
            logit_list.append(logit_i)

        # Ghép các AU lại
        V_a    = torch.cat(emb_list, dim=1)        # (B, N_a, EMB_DIM)
        logits = torch.cat(logit_list, dim=1)      # (B, N_a)

        # Chuyển logits -> probabilities p^a (dùng đúng trong (3))
        p_a = torch.sigmoid(logits)

        return V_a, p_a


# ------------------------------------------------------------
#  Expression Head: N_e nhánh, giống AUHead
# ------------------------------------------------------------
class ExprHead(nn.Module):
    """
    Joint Expression head cho Stage-1 (JFL).

    Input:
        x: (B, D, C_in)
    Output:
        V_e: (B, N_e, EMB_DIM)   # Expression embeddings V^e_j
        p_e: (B, N_e)            # Expression probabilities (dùng cho L_we)
    """
    def __init__(self,
                 in_channels: int,
                 num_expr: int,
                 hid_channels: int):
        super().__init__()
        self.in_channels  = in_channels
        self.num_expr     = num_expr
        self.hid_channels = hid_channels
        self.emb_channels = EMB_DIM

        self.extractors = nn.ModuleList([
            Conv1DExtractor(
                in_channels=self.in_channels,
                hid_channels=self.hid_channels,
                emb_channels=self.emb_channels,
            )
            for _ in range(self.num_expr)
        ])

        self.classifiers = nn.ModuleList([
            nn.Linear(self.emb_channels, 1)
            for _ in range(self.num_expr)
        ])

    def forward(self, x):
        """
        x: (B, D, C_in)
        """
        emb_list = []
        logit_list = []

        for i in range(self.num_expr):
            emb_i = self.extractors[i](x)          # (B, EMB_DIM)
            emb_list.append(emb_i.unsqueeze(1))    # (B, 1, EMB_DIM)
            logit_i = self.classifiers[i](emb_i)   # (B, 1)
            logit_list.append(logit_i)

        V_e    = torch.cat(emb_list, dim=1)        # (B, N_e, EMB_DIM)
        logits = torch.cat(logit_list, dim=1)      # (B, N_e)

        # probabilities p^e (dùng đúng trong (4))
        p_e = torch.sigmoid(logits)

        return V_e, p_e


# ------------------------------------------------------------
#  Stage-1 model: Backbone + AUHead + ExprHead
# ------------------------------------------------------------
class MEFARGStage1(nn.Module):
    """
    Stage-1: Joint Tasks for Node Feature Learning (JFL)
    - Backbone (Swin/ResNet) -> feature map (B, D, C_in)
    - global_linear: C_in -> C_mid
    - AUHead:    tạo V^a và AU probabilities p^a
    - ExprHead:  tạo V^e và Expression probabilities p^e

    num_aus  : N_a  (ví dụ DISFA: 8)
    num_expr : N_e  (ví dụ: 7 emotion: Angry, Fear, Happy, Sad, Surp, Disg, Neutral)
    """
    def __init__(self,
                 num_aus: int = 8,
                 num_expr: int = 7,
                 backbone: str = 'swin_transformer_base'):
        super().__init__()

        # ---------------- Backbone ----------------
        if 'transformer' in backbone:
            if backbone == 'swin_transformer_tiny':
                self.backbone = swin_transformer_tiny()
            elif backbone == 'swin_transformer_small':
                self.backbone = swin_transformer_small()
            else:
                self.backbone = swin_transformer_base()

            # Swin trả về (B, D, C_in)
            self.in_channels = self.backbone.num_features
            # giảm kênh một chút
            self.mid_channels = self.in_channels // 2
            # bỏ head classification mặc định
            self.backbone.head = None

        elif 'resnet' in backbone:
            if backbone == 'resnet18':
                self.backbone = resnet18()
            elif backbone == 'resnet101':
                self.backbone = resnet101()
            else:
                self.backbone = resnet50()

            # Giả định resnet đã sửa để trả (B, D, C_in)
            self.in_channels = self.backbone.fc.weight.shape[1]
            self.mid_channels = self.in_channels // 4
            self.backbone.fc = None
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Map C_in -> C_mid, giữ nguyên D
        self.global_linear = LinearBlock(self.in_channels, self.mid_channels)

        # Hai head: dùng mid_channels làm C_in cho Conv1d,
        # hid_channels có thể để = mid_channels
        self.au_head = AUHead(
            in_channels=self.mid_channels,
            num_aus=num_aus,
            hid_channels=self.mid_channels
        )
        self.expr_head = ExprHead(
            in_channels=self.mid_channels,
            num_expr=num_expr,
            hid_channels=self.mid_channels
        )

    def forward(self, x):
        """
        x: input images, shape tùy backbone (ví dụ (B, 3, 224, 224))

        Returns:
            V_a: (B, N_a, EMB_DIM)   # AU node features
            V_e: (B, N_e, EMB_DIM)   # Expression node features
            p_a: (B, N_a)            # AU probabilities  (dùng trong L_wa)
            p_e: (B, N_e)            # Expr probabilities (dùng trong L_we)
        """
        # Backbone output: (B, D, C_in) – ví dụ (64, 49, 2048)
        feat = self.backbone(x)

        # LinearBlock trên kênh, giữ D → (B, D, mid_channels)
        feat = self.global_linear(feat)

        # AU branch
        V_a, p_a = self.au_head(feat)

        # Expression branch
        V_e, p_e = self.expr_head(feat)

        return V_a, V_e, p_a, p_e
