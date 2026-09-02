# EGIR-v2: Energy-Guided Iterative Feature Refinement với Knowledge Query

Cải tiến từ SymGraphAU cho bài toán Facial Action Unit Recognition — tri thức FACS
tham gia trực tiếp vào suy luận (cả train lẫn inference) thay vì chỉ là regularizer.

## Cấu trúc thư mục

```
EGIR-v2/
├── conf.py                    # argparse + config, TÁI SỬ DỤNG từ SymGraphAU gốc
├── config/
│   ├── DISFA_config.yaml
│   └── BP4D_config.yaml
├── dataset.py                 # BP4D / DISFA Dataset, TÁI SỬ DỤNG nguyên
├── utils.py                   # Loss, statistics, transform, TÁI SỬ DỤNG nguyên
├── matricMAE.py                # script sinh ma trận AU-Expression gốc
├── matrixMAE/
│   └── M_AE_DISFA.npy          # ma trận M_AE tính sẵn (dùng cho pseudo-label ở cả 3 stage)
├── graph_rules.py              # MỚI — luật đồ thị FACS đa quan hệ cho EGIR-v2 (Stage 2/3)
├── model/
│   ├── SymStage1.py            # TÁI SỬ DỤNG NGUYÊN — backbone + AU/Expr head
│   ├── resnet.py / swin_transformer.py / basic_block.py   # TÁI SỬ DỤNG nguyên
│   ├── kg_encoder.py           # MỚI — R-GCN + Stage2GraphModule (w_r, c_r, attention)
│   └── egir_stage3.py          # MỚI — model Stage 3, vòng lặp T bước reasoning
├── tool/
│   ├── DISFA_image_label_process.py       # sinh list file train/test từ ảnh + nhãn thô
│   └── DISFA_calculate_AU_class_weights.py # sinh file trọng số AU
├── data/DISFA/
│   ├── img/                    # đổ ảnh thô vào đây (theo cấu trúc SNxxx/frame.png)
│   └── list/                   # tool script sẽ sinh list .txt vào đây
├── train_EGIR_Stage1.py        # GẦN NHƯ NGUYÊN VĂN train_Sym_Stage_1.py gốc
├── train_EGIR_Stage2.py        # MỚI — đóng băng Stage1, Class Center 1 lần, train R-GCN
└── train_EGIR_Stage3.py        # MỚI — đóng băng Stage2, K cố định, fine-tune backbone qua T bước
```

## 1. Cài đặt môi trường

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Tải checkpoint pretrained backbone (bắt buộc, tải tay)

`model/resnet.py` và `model/swin_transformer.py` load checkpoint từ thư mục
`checkpoints/`, không tự tải qua mạng. Khuyến nghị dùng ResNet50 (chỉ cần 1 file):

```bash
mkdir -p checkpoints
wget -O checkpoints/resnet50-19c8e357.pth \
  https://download.pytorch.org/models/resnet50-19c8e357.pth
```

Nếu muốn dùng Swin thay vì ResNet, cần tự tìm checkpoint tương ứng
(`swin_base_patch4_window7_224.pth`) đặt cùng thư mục `checkpoints/`.
Toàn bộ lệnh dưới đây dùng `--arc resnet50` để đơn giản.

## 3. Chuẩn bị dữ liệu DISFA

Đổ dữ liệu thô vào:
- `data/DISFA/img/SNxxx/{frame}.png`
- Thư mục `ActionUnit_Labels` gốc (đặt ở đâu cũng được, chỉ cần biết đường dẫn)

Sinh list file (train/test theo 3-fold cố định) + nhãn:

```bash
python tool/DISFA_image_label_process.py \
  --image-dir data/DISFA/img \
  --label-dir /đường/dẫn/tới/ActionUnit_Labels \
  --out-dir data/DISFA/list
```

Sinh file trọng số AU (dùng cho `WeightedAsymmetricLoss`):

```bash
python tool/DISFA_calculate_AU_class_weights.py --dataset-path data/DISFA
```

> Xem chi tiết tham số chính xác trong `tool/README.md` (copy từ repo gốc) —
> có thể khác đôi chút tuỳ phiên bản, kiểm tra `--help` của từng script trước khi chạy.

## 4. Chạy tuần tự 3 stage (lặp lại `--fold 1/2/3`)

### Stage 1 — Joint Feature Learning
```bash
python train_EGIR_Stage1.py --dataset DISFA --fold 1 --arc resnet50 \
    -e 8 -b 16 -lr 0.00001 --exp-name egir_v2
```
→ lưu `results/egir_v2/.../best_model_fold1.pth`

### Stage 2 — Graph Construction of Knowledge Embedding
```bash
python train_EGIR_Stage2.py --dataset DISFA --fold 1 --arc resnet50 \
    -e 5 -b 32 -lr 0.001 --exp-name egir_v2 \
    --resume results/egir_v2/.../best_model_fold1.pth
```
→ lưu `results/egir_v2/.../stage2_fold1.pth`

### Stage 3 — Iterative Neuro-Symbolic Reasoning
```bash
python train_EGIR_Stage3.py --dataset DISFA --fold 1 --arc resnet50 \
    -e 10 -b 16 -lr 0.00001 --exp-name egir_v2 \
    --resume results/egir_v2/.../best_model_fold1.pth \
    --resume-phase2 results/egir_v2/.../stage2_fold1.pth
```
→ lưu `results/egir_v2/.../final_model_fold1.pth`

Lặp lại cả 3 lệnh trên với `--fold 2` và `--fold 3`, rồi lấy trung bình F1 3 fold.

## Điểm khác biệt chính so với SymGraphAU gốc (để ghi vào phần Implementation Details)

| | SymGraphAU gốc | EGIR-v2 |
|---|---|---|
| Stage 2 | CNF + PySAT + synthetic proposition + triplet loss | R-GCN đa quan hệ + Energy trên dữ liệu thật, không synthetic |
| Tri thức lúc inference | Không dùng (chỉ regularize lúc train) | Vẫn chạy đủ T bước reasoning |
| Đóng băng | Stage2: đóng băng Stage1. Stage3: đóng băng GCN | Giống hệt — Stage2 đóng băng Stage1, Stage3 đóng băng graph_module |
| Class Center | Tính 1 lần (`compute_class_centers`) | Giống hệt |
| K (knowledge embedding) | N/A | Hằng số cố định sau Stage 2, không tính lại mỗi batch ở Stage 3 |

## Ghi chú

- `graph_rules.py` định nghĩa đồ thị FACS rút gọn cho 8 AU của DISFA
  (AU1,2,4,6,9,12,25,26). Muốn dùng BP4D (nhiều AU hơn) cần mở rộng
  `AU_AA_COOCCUR`, `CNF_AA_EXCLUSION`, `CNF_AE_COMBO` cho đúng tập AU của BP4D.
- Toàn bộ code đã được kiểm tra chạy được (cú pháp + forward/backward tích hợp
  qua dữ liệu giả), nhưng CHƯA train trên dữ liệu DISFA thật — cần bạn tự
  chạy và kiểm tra số liệu trước khi đưa vào báo cáo/paper.
