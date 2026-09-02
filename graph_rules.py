"""
graph_rules.py — Định nghĩa đồ thị tri thức FACS cho EGIR-v2.

Khác với rules.py / matricMAE.py gốc của SymGraphAU (dùng cho CNF + PySAT +
triplet loss ở Stage 2 gốc), file này định nghĩa đồ thị ĐA QUAN HỆ dùng cho
cơ chế Energy-based reasoning của EGIR-v2 (Stage 2/3).

AU_AA_COOCCUR, CNF_AA_EXCLUSION, CNF_AE_COMBO lấy tinh thần từ rules.py gốc,
rút gọn cho đúng 8 AU có nhãn trong DISFA (AU1,2,4,6,9,12,25,26).
"""
import numpy as np
import torch

AU_LIST = [1, 2, 4, 6, 9, 12, 25, 26]
EXPR_LIST = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Disgust', 'Neutral']
NEUTRAL_INDEX = 6

AU_IDX = {au: i for i, au in enumerate(AU_LIST)}
EXPR_IDX = {e: i for i, e in enumerate(EXPR_LIST)}
N_AU, N_EXPR = len(AU_LIST), len(EXPR_LIST)

# ---- Luật AU-AU (rút gọn từ Table 1 FACS, giới hạn 8 AU của DISFA) ----
AU_AA_COOCCUR = [
    ["AU1", "AU2"], ["AU2", "AU4"], ["AU6", "AU12"], ["AU6", "AU25"],
    ["AU12", "AU25"], ["AU4", "AU9"], ["AU4", "AU25"], ["AU9", "AU25"],
    ["AU6", "AU26"], ["AU25", "AU26"],
]
CNF_AA_EXCLUSION = [
    ["¬AU2", "¬AU6"], ["¬AU2", "¬AU9"], ["¬AU1", "¬AU9"],
    ["¬AU9", "¬AU12"], ["¬AU4", "¬AU12"],
]

# ---- Luật AU(-tổ hợp) -> Emotion (rút gọn từ EMFACS, Table 2/3) ----
CNF_AE_COMBO = [
    (["AU6", "AU12"], "Happy"),
    (["AU1", "AU4"], "Sad"),
    (["AU1", "AU2", "AU26"], "Surprise"),
    (["AU1", "AU2", "AU4", "AU26"], "Fear"),
    (["AU4"], "Angry"),
    (["AU9"], "Disgust"),
]

COMPONENTS = [{'members': [int(a[2:]) for a in aus], 'emotion': emo}
              for aus, emo in CNF_AE_COMBO if len(aus) > 1]
N_COMP = len(COMPONENTS)
COMP_OFFSET = N_AU + N_EXPR
EXPR_OFFSET = N_AU
N_NODE = N_AU + N_EXPR + N_COMP

REL_TYPES = ['implies', 'excludes', 'co_occurs', 'component_of']
REL2ID = {r: i for i, r in enumerate(REL_TYPES)}


def _au_num(s):
    return int(s.replace('¬', '').replace('AU', ''))


def build_M_AE(beta=0.1):
    """Ma trận AU->Expression, thang beta/0.5/(1-beta) — dùng cho pseudo-label
       Stage 1 VÀ khởi tạo c_r cho các cạnh implies ở Stage 2."""
    PRIMARY = {"Happy": [6, 12, 25], "Sad": [1, 4], "Angry": [4], "Fear": [1, 2, 25],
               "Surprise": [1, 2, 25, 26], "Disgust": [9], "Neutral": []}
    SECONDARY = {"Happy": [26], "Sad": [6, 25], "Angry": [1, 2, 9, 25], "Fear": [4, 6, 26],
                 "Surprise": [6, 12], "Disgust": [4, 12, 25], "Neutral": []}
    M = np.full((N_AU, N_EXPR), fill_value=beta, dtype=np.float32)
    for expr, aus in PRIMARY.items():
        for au in aus:
            if au in AU_IDX:
                M[AU_IDX[au], EXPR_IDX[expr]] = 1.0 - beta
    for expr, aus in SECONDARY.items():
        for au in aus:
            if au in AU_IDX and M[AU_IDX[au], EXPR_IDX[expr]] != 1.0 - beta:
                M[AU_IDX[au], EXPR_IDX[expr]] = 0.5
    return M


def build_graph(M_AE_mat):
    """Trả về:
       energy_edges — list cạnh dùng tính E_phi (không gồm component_of)
       struct_edges — list (src, dst, rel_id) dùng xây adjacency cho R-GCN
       c_init       — giá trị khởi tạo c_r cho từng cạnh trong energy_edges
    """
    energy_edges, struct_edges, c_init = [], [], []

    def au_node(au): return AU_IDX[au]
    def expr_node(e): return EXPR_OFFSET + EXPR_IDX[e]

    for a, b in AU_AA_COOCCUR:
        i, j = au_node(_au_num(a)), au_node(_au_num(b))
        energy_edges.append({'i': i, 'j': j, 'rel': 'co_occurs'})
        c_init.append(0.7)
        struct_edges += [(i, j, REL2ID['co_occurs']), (j, i, REL2ID['co_occurs'])]

    for a, b in CNF_AA_EXCLUSION:
        i, j = au_node(_au_num(a)), au_node(_au_num(b))
        energy_edges.append({'i': i, 'j': j, 'rel': 'excludes'})
        c_init.append(0.7)
        struct_edges += [(i, j, REL2ID['excludes']), (j, i, REL2ID['excludes'])]

    comp_id = 0
    for aus, emo in CNF_AE_COMBO:
        j = expr_node(emo)
        if len(aus) == 1:
            au = _au_num(aus[0])
            i = au_node(au)
            energy_edges.append({'i': i, 'j': j, 'rel': 'implies'})
            c_init.append(float(M_AE_mat[AU_IDX[au], EXPR_IDX[emo]]))
            struct_edges.append((i, j, REL2ID['implies']))
        else:
            c_node = COMP_OFFSET + comp_id
            for a in aus:
                struct_edges.append((au_node(_au_num(a)), c_node, REL2ID['component_of']))
            energy_edges.append({'i': c_node, 'j': j, 'rel': 'implies'})
            c_init.append(0.9)
            struct_edges.append((c_node, j, REL2ID['implies']))
            comp_id += 1

    return energy_edges, struct_edges, c_init


M_AE_NP = build_M_AE(beta=0.1)
M_AE = torch.from_numpy(M_AE_NP).float()
ENERGY_EDGES, STRUCT_EDGES, C_INIT = build_graph(M_AE_NP)


def au_to_expr_pseudo(Y_a, M_AE_mat=None, neutral_index=NEUTRAL_INDEX):
    """Sinh pseudo-label Expression từ nhãn AU (đúng cơ chế train_Sym_Stage_1.py gốc)."""
    if M_AE_mat is None:
        M_AE_mat = M_AE.to(Y_a.device)
    scores = Y_a.float() @ M_AE_mat
    ke = scores.argmax(dim=1)
    neutral_mask = (Y_a.float().sum(dim=1) == 0)
    ke[neutral_mask] = neutral_index
    Y_e = torch.zeros(Y_a.shape[0], M_AE_mat.shape[1], device=Y_a.device)
    Y_e.scatter_(1, ke.unsqueeze(1), 1.0)
    return Y_e


def get_node_value(idx, p_a, p_e):
    """Giá trị runtime của 1 node (AU/Expression/Component) từ prediction hiện tại."""
    if idx < N_AU:
        return p_a[:, idx]
    elif idx < N_AU + N_EXPR:
        return p_e[:, idx - N_AU]
    else:
        comp_id = idx - COMP_OFFSET
        members = COMPONENTS[comp_id]['members']
        v = torch.ones_like(p_a[:, 0])
        for au in members:
            v = v * p_a[:, AU_IDX[au]]
        return v


def viol_implies(a, b): return torch.clamp(a - b, min=0.0)
def viol_excludes(a, b): return a * b
def viol_co_occurs(a, b): return torch.abs(a - b)

VIOL_FN = {'implies': viol_implies, 'excludes': viol_excludes, 'co_occurs': viol_co_occurs}
