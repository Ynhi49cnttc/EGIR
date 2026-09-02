"""
Xây dựng ma trận M_{A-E} cho DISFA theo cơ chế SymGraphAU:

- beta = 0.1
- primary relevance   -> 1 - beta = 0.9
- secondary relevance -> 0.5
- no relevance        -> beta = 0.1

AUs (DISFA): 1, 2, 4, 6, 9, 12, 25, 26
Expressions: Angry, Fear, Happy, Sad, Surprise, Disgust, Neutral
"""

import numpy as np
import pandas as pd


def build_M_AE(
    au_list,
    expr_list,
    primary_pairs,
    secondary_pairs,
    beta: float = 0.1,
):
    """
    Xây ma trận M_{A-E}.

    Parameters
    ----------
    au_list : list[int] | list[str]
        Danh sách AU theo thứ tự cố định.
    expr_list : list[str]
        Danh sách biểu cảm.
    primary_pairs : list[tuple[au, expr]]
        Các cặp (AU, Expr) primary relevance.
    secondary_pairs : list[tuple[au, expr]]
        Các cặp (AU, Expr) secondary relevance.
    beta : float
        Tham số β (no relevance).

    Returns
    -------
    M_AE : np.ndarray, shape = (N_a, N_e)
    au_to_idx : dict
    expr_to_idx : dict
    df_M_AE : pd.DataFrame
    """

    au_to_idx = {au: i for i, au in enumerate(au_list)}
    expr_to_idx = {expr: j for j, expr in enumerate(expr_list)}

    N_a = len(au_list)
    N_e = len(expr_list)

    # Khởi tạo toàn bộ là "no relevance" = beta
    M_AE = np.full((N_a, N_e), fill_value=beta, dtype=np.float32)

    primary_value = 1.0 - beta    # ví dụ 0.9 nếu beta = 0.1
    secondary_value = 0.5

    # Gán primary: M(i,j) = 1 - beta
    for au, expr in primary_pairs:
        if au not in au_to_idx:
            raise ValueError(f"AU {au} không có trong au_list.")
        if expr not in expr_to_idx:
            raise ValueError(f"Biểu cảm {expr} không có trong expr_list.")
        i = au_to_idx[au]
        j = expr_to_idx[expr]
        M_AE[i, j] = primary_value

    # Gán secondary: M(i,j) = 0.5 nhưng KHÔNG đè lên primary
    for au, expr in secondary_pairs:
        if au not in au_to_idx:
            raise ValueError(f"AU {au} không có trong au_list.")
        if expr not in expr_to_idx:
            raise ValueError(f"Biểu cảm {expr} không có trong expr_list.")
        i = au_to_idx[au]
        j = expr_to_idx[expr]

        # Nếu ô này chưa phải primary thì mới gán secondary
        if M_AE[i, j] != primary_value:
            M_AE[i, j] = secondary_value
        # Nếu đã là primary (0.9) thì giữ nguyên, bỏ qua secondary

    # DataFrame để dễ nhìn / debug
    df_M_AE = pd.DataFrame(
        M_AE,
        index=[f"AU{au}" for au in au_list],
        columns=expr_list,
    )

    return M_AE, au_to_idx, expr_to_idx, df_M_AE


def build_M_AE_DISFA(beta: float = 0.1):
    """
    Xây M_{A-E} cho DISFA với 8 AU & 7 expressions.
    Mapping AU–Expression được thiết kế theo FACS thường dùng.
    Bạn có thể chỉnh lại PRIMARY / SECONDARY cho phù hợp paper gốc.
    """

    # 8 AU thường dùng trong DISFA
    disfa_aus = [1, 2, 4, 6, 9, 12, 25, 26]

    # 7 biểu cảm (6 basic + Neutral)
    expr_list = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Disgust", "Neutral"]

    # PRIMARY relevance
    PRIMARY = {
        "Happy":    [6, 12, 25],
        "Sad":      [1, 4],
        "Angry":    [4],
        "Fear":     [1, 2, 25],
        "Surprise": [1, 2, 25, 26],
        "Disgust":  [9],
        "Neutral":  [],  # neutral không gán AU ưu tiên
    }

    # SECONDARY relevance
    SECONDARY = {
        "Happy":    [26],
        "Sad":      [6, 25],
        "Angry":    [1, 2, 9, 25],
        "Fear":     [4, 6, 26],
        "Surprise": [6, 12],
        "Disgust":  [4, 12, 25],
        "Neutral":  [],
    }

    primary_pairs = [
        (au, expr)
        for expr, aus in PRIMARY.items()
        for au in aus
    ]

    secondary_pairs = [
        (au, expr)
        for expr, aus in SECONDARY.items()
        for au in aus
    ]

    return build_M_AE(
        au_list=disfa_aus,
        expr_list=expr_list,
        primary_pairs=primary_pairs,
        secondary_pairs=secondary_pairs,
        beta=beta,
    )


if __name__ == "__main__":
    beta = 0.1
    M_AE, au_to_idx, expr_to_idx, df_M_AE = build_M_AE_DISFA(beta=beta)

    print("=== AU list (DISFA) ===")
    print(au_to_idx)
    print("\n=== Expression list ===")
    print(expr_to_idx)

    print("\n=== M_{A-E} (DataFrame) ===")
    print(df_M_AE)

    # Nếu bạn muốn lưu ra file dùng cho training:
    np.save("matrixMAE/M_AE_DISFA.npy", M_AE)
    df_M_AE.to_csv("matrixMAE/M_AE_DISFA.csv", float_format="%.2f")
    print("\nĐã lưu M_AE_DISFA.npy và M_AE_DISFA.csv")
